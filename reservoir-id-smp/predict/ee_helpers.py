"""Create EE image object with processed landsat imagery for year and LS name

Can then be handed off to computePixels() to get the data"""
import io
import logging
import numpy as np

import ee

ee.Initialize(project='ksolvik-misc')

brazilBuffer = ee.FeatureCollection("users/kyso1389/Brazil_aea_10kmbuffer_noremoteislands_noholes")

RESOLUTION = 0.000089831528412
# Name conversion lookup table
LS_NAME_DICT = {
     'ls5': 'LT05',
     'ls7': 'LE07',
     'ls8': 'LC08',
     'ls9': 'LC09'
}

EXPORT_BANDS_DICT = {
    'ls5': ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7'],
    'ls7': ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7'],
    'ls8':  ['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7'],
    'ls9': ['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7']
}

### Prep function for SR ##/
def prepSR(image):
    # Develop masks for unwanted pixels (fill, cloud, cloud shadow).
    saturationMask = image.select('QA_RADSAT').eq(0)

    # Apply the scaling factors to the appropriate bands.

    def func_mnu(factorNames):
        factorList = image.toDictionary().select(factorNames).values()
        return ee.Image.constant(factorList)

    getFactorImg = func_mnu
    scaleImg = getFactorImg([
    'REFLECTANCE_MULT_BAND_.*'])
    offsetImg = getFactorImg([
    'REFLECTANCE_ADD_BAND_.*'])
    scaled = image.select('SR_B.*').multiply(scaleImg).add(offsetImg)

    # Replace original bands with scaled bands and apply masks.
    return image.addBands(scaled, None, True).updateMask(saturationMask)

### Create Collection with SR and CloudScore ###/
def filteredCloudScoreCol(SR,TOA,centerYear):
    srCloudScore = SR.linkCollection(TOA.map(ee.Algorithms.Landsat.simpleCloudScore),
    ['cloud'])
    srAOI = srCloudScore.filterBounds(brazilBuffer)
    # 3-year composite
    srFilt = srAOI.filterDate(str(centerYear-1) + '-01-01', str(centerYear+1) + '-12-31')
    # Apply scaling and such
    srFiltPrepped = srFilt.map(prepSR)
    return srFiltPrepped

def clean_name(band_name):
    return ee.String(band_name).replace('_[^_]*$', '')

### Simple Composite function ###/
def SRComposite(collection,asFloat,percentile,cloudScoreRange,maxDepth):

    # Select a sufficient set of images, and compute TOA and cloudScore.
    prepared =  ee.Algorithms.Landsat.pathRowLimit(
    collection, maxDepth, 4 * maxDepth)

    # Determine the per-pixel cloud score threshold.
    cloudThreshold = prepared.reduce(ee.Reducer.min()) \
    .select('cloud_min') \
    .add(cloudScoreRange)

    # Mask out pixels above the cloud score threshold, and update the mask of
    # the remaining pixels to be no higher than the cloud score mask.
    def updateMask(image):
        cloud = image.select('cloud')
        cloudMask = cloud.mask().min(cloud.lte(cloudThreshold))
        # Drop the cloud band and QA bands.
        image = image.select(['SR_B.*', 'cloud'])
        return image.mask(image.mask().min(cloudMask))

    masked = prepared.map(updateMask)

    # Take the (mask-weighted) median (or other percentile)
    # of the good pixels.
    result = masked.reduce(ee.Reducer.percentile([percentile]))

    # Force the mask up to 1 if it's non-zero, to hide L7 SLC artifacts.
    result = result.mask(result.mask().gt(0))

    # Clean up the band names by removing the suffix that reduce() added.
    badNames = result.bandNames()
    goodNames = badNames.map(clean_name)
    result = result.select(badNames, goodNames)

    return result

### Function that combines it all ###
def createAnnualMosaic(ls_name, centerYear):
    ls_code = LS_NAME_DICT[ls_name]
    SR = ee.ImageCollection(f"LANDSAT/{ls_code}/C02/T1_L2")
    TOA = ee.ImageCollection(f"LANDSAT/{ls_code}/C02/T1_TOA")
    lsExportBands = EXPORT_BANDS_DICT[ls_name]
    lsCol = filteredCloudScoreCol(SR, TOA, centerYear)
    lsComposite = SRComposite(lsCol, True, 50, 20, 100)
    print(lsExportBands)
    print(ls_code)
    print(ls_name)
    print(centerYear)
    lsClipComposite = lsComposite.select(lsExportBands).multiply(65534).round().toUint16()
    return lsClipComposite.resample('bicubic')


def get_patch(im, lon, lat, w, h, ls_name):
    # Make a request object.
    print('lat', lat)
    print('lon', lon)
    request = {
        'expression': im,
        'fileFormat': 'NPY',
        'grid': {
            'dimensions': {
                'width': w,
                'height': h
            },
            'affineTransform': {
                'scaleX': RESOLUTION,
                'shearX': 0,
                'translateX': lon,
                'shearY': 0,
                'scaleY': RESOLUTION,
                'translateY': lat
            },
            'crsCode': 'EPSG:4326',
        },
    }

    raw = ee.data.computePixels(request)
    logging.warning(f"EE bytes: {len(raw)}")
    if not raw:
            raise RuntimeError("Empty EE response")

    try:
        x = np.load(io.BytesIO(raw))
        arr = np.array([x[band] for band in EXPORT_BANDS_DICT[ls_name]])
        # arr = arr.view(np.float64).reshape(arr.shape + (-1,))
        print(arr)
        with open('output_file_test.bin', 'wb') as f:
            f.write(raw)
    except Exception as e:
        raise RuntimeError("Failed to load NPY for coords") from e

    logging.warning(
        f"EE end, bytes={len(raw)}"
    )

    return arr

def get_scales(proj, resolution):
    proj = ee.Projection(proj).atScale(resolution)
    proj_dict = proj.getInfo()

    scale_x = proj_dict['transform'][0]
    scale_y = -proj_dict['transform'][4]

    return scale_x, scale_y
