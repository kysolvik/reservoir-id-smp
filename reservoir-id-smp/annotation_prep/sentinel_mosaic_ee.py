"""
SentinelExport_AllBands_AllBrazil
kylen.solvik@colorado.edu

Python port: build_sentinel_mosaic(year) returns the stacked image with band
order s2 10m (B2, B3, B4, B8), s2 20m (B11, B12), s1 (VV, VH)
Requires ee.Initialize() to have been called before use.
"""
import ee

####### Initial Setup ///////
s2exportBands10m = ['B2', 'B3', 'B4', 'B8']
s2exportBands20m = ['B11', 'B12']
s1exportBands = ['VV', 'VH']
cloudBand = 'cs_cdf'


####### Simple Composite function ///////
def simpleComposite(collection,
                    percentile,
                    cloudScoreRange,
                    maxDepth=None):

    # Select a sufficient set of images, and compute TOA and cloudScore.

    # Determine the per-pixel cloud score threshold.
    cloudThreshold = (collection.reduce(ee.Reducer.max())
                      .select(cloudBand + '_max')
                      .subtract(cloudScoreRange))

    # Mask out pixels below the cloud score threshold, and update the mask of
    # the remaining pixels to be no higher than the cloud score mask.
    def updateMask(image):
        cloud = image.select(cloudBand)
        cloudMask = cloud.mask().min(cloud.gte(cloudThreshold))
        return image.mask(image.mask().min(cloudMask))
    masked = collection.map(updateMask)

    # Take the (mask-weighted) median (or other percentile)
    # of the good pixels.
    result = masked.reduce(ee.Reducer.percentile([percentile]))
    count = masked.select('B2').count().rename('s2Count')
    # Clean up the band names by removing the suffix that reduce() added.
    badNames = result.bandNames()
    goodNames = badNames.map(
        lambda x: ee.String(x).replace('_[^_]*$', ''))
    result = result.select(badNames, goodNames).addBands(count)
    return result


def build_sentinel_mosaic(year):
    """Build the stacked Sentinel mosaic for a single year.

    Band order: s2 10m (B2, B3, B4, B8), s2 20m (B11, B12),
    s1 (VV_p90, VH_p90). All bands uint16, clipped to brazilBuffer.
    """
    s1Col = ee.ImageCollection("COPERNICUS/S1_GRD")
    s2Clouds = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2Col = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    brazilBuffer = ee.FeatureCollection("projects/ksolvik-misc/assets/Brazil_10kmbuffer_noremoteislands_noholes")

    ####### Prep Sentinel 2 Collection //////

    # Area of Interest Only
    s2AOI = s2Col.filterBounds(brazilBuffer)
    # Filter to year
    s2Dry = s2AOI.filterDate(str(year) + '-01-01', str(year) + '-12-31')

    # Adding a filter for less than 50 percent cloudy
    s2Dry = s2Dry.filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 50)

    # Make composite.
    s2MedianOG = simpleComposite(
        s2Dry.linkCollection(s2Clouds, [cloudBand]),  # Collection with clouds linked
        50,  # Percentile (50 is median)
        0.1  # CloudScoreRange (0.0-1.0)
        )

    # Take convert to Uint16 and clip
    s2MedianInt = s2MedianOG.toUint16().clip(brazilBuffer)

    # Get export bands
    s2MedianInt20m = s2MedianInt.select(s2exportBands20m)
    s2MedianInt10m = s2MedianInt.select(s2exportBands10m)

    ####### Prep Sentinel 1 Collection //////

    # Create radar image
    s1Desc = (s1Col.filter(ee.Filter.eq('instrumentMode', 'IW'))
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))

    # Area of interest only
    s1AOI = s1Desc.filterBounds(brazilBuffer).select(s1exportBands)

    # Filter to year
    s1Dry = s1AOI.filterDate(str(year) + '-01-01', str(year) + '-12-31')

    # Take 90th percentile
    s1Med = s1Dry.reduce(ee.Reducer.percentile([90]), 2).clip(brazilBuffer)

    # Scale and convert to int16
    s1Scaled = ee.Image.cat(
        s1Med.select('VV_p90').unitScale(-20, 0),
        s1Med.select('VH_p90').unitScale(-25, 0)
        )
    s1ScaledInt = s1Scaled.multiply(65534).uint16()

    ####### Stack: s2 10m, s2 20m, s1 //////
    return ee.Image.cat(s2MedianInt10m, s2MedianInt20m, s1ScaledInt)
