"""Basic script for getting LULC within buffer of reservoir polygons"""

import os

import numpy as np
import pandas as pd
import geopandas as gpd
import dask_geopandas as dgpd
import exactextract

BUFFER_SIZE = 1000
if BUFFER_SIZE >=500:
    BUFFER_RESOLUTION=4
else:
    # Higher precision for smaller buffers
    BUFFER_RESOLUTION=16

IN_GPKG = '../clean_summarize/out_polys/sentinel_2021_v7_aea_cleaned_polygons.gpkg'
IN_MB_TIFF = './in/brazil/brazil_coverage_2021_c10_aea.tif'
OUT_GPKG = './out/poly_buffer_{}m.gpkg'.format(BUFFER_SIZE)
OUT_DIFF_GPKG = './out/poly_buffer_{}m_diff.gpkg'.format(BUFFER_SIZE)
OUT_ZSTATS_PARQUET = './out/poly_extract_{}m_frac.parquet'.format(BUFFER_SIZE)
OUT_SUMMARY_CSV = './out/poly_lulc_fracs_summary_{}m.csv'.format(BUFFER_SIZE)

# Input
in_gdf = gpd.read_file(IN_GPKG)

if not os.path.isfile(OUT_GPKG):
    # Buffer in parallel using dask
    dgdf = dgpd.from_geopandas(in_gdf, npartitions=32)
    buffered_dask = dgdf.buffer(BUFFER_SIZE, resolution=BUFFER_RESOLUTION)
    buffered_gdf = buffered_dask.compute()
    #  Write out
    buffered_gdf.to_file(OUT_GPKG)
else:
    buffered_gdf = gpd.read_file(OUT_GPKG)


if not os.path.isfile(OUT_DIFF_GPKG):
    # Element-wise difference (this works element-wise in parallel)
    rings_gdf = buffered_gdf.geometry.difference(in_gdf.geometry)

    # Write out
    rings_gdf.to_file(OUT_DIFF_GPKG)
else:
    rings_gdf = gpd.read_file(OUT_DIFF_GPKG)

if not os.path.isfile(OUT_ZSTATS_PARQUET):
    # Get zonal stats (takes about 15 min)
    zstats_df = exactextract.exact_extract(IN_MB_TIFF, OUT_DIFF_GPKG,
                                           ['values','unique','frac'],
                                           output='pandas', progress=True)

    zstats_df.to_parquet(OUT_ZSTATS_PARQUET)
else:
    zstats_df = pd.read_parquet(OUT_ZSTATS_PARQUET)


# Summarize results
MB_KEYS_DICT = {                                                                               
    'natural': np.array([1, 3, 4, 5, 6, 49, 10, 11, 12, 32, 29, 50, 23]),
    'water': np.array([26, 31, 33]),
}

def get_lulc_dict(row):
    return dict(zip(row['unique'], row['frac']))

def get_lulc_fractions(row):
    out_dict = {}
    lulc_dict = get_lulc_dict(row)
    sum_all = 0
    for lulc_class in MB_KEYS_DICT.keys():                                                     
        sum_of_class = 0
        for key in lulc_dict.keys():
            if key in MB_KEYS_DICT[lulc_class]:
                sum_of_class += lulc_dict[key]
        out_dict[lulc_class] = sum_of_class
        sum_all += sum_of_class
    out_dict['other'] = 1 - sum_all
    return out_dict
if not os.path.isfile(OUT_SUMMARY_CSV):
    frac_dicts = zstats_df.apply(get_lulc_fractions, axis=1)
    frac_df = pd.DataFrame(list(frac_dicts.values))

    frac_df['og_area'] = in_gdf.geometry.area
    frac_df['ring_area'] = rings_gdf.geometry.area
    frac_df.to_csv(OUT_SUMMARY_CSV, index=False)

else:
    frac_df = pd.read_csv(OUT_SUMMARY_CSV)

# Print out some basic stats
frac_df['og_area_ha'] = frac_df['og_area']*0.0001
frac_df['natural_no_water'] = frac_df['natural']/(1 - frac_df['water'])
frac_df['other_no_water'] = frac_df['other']/(1 - frac_df['water'])

for key in ['natural','water','other']:
    frac_df[key+'_area'] = frac_df[key]*frac_df['ring_area']

print('Totals')
print(frac_df[['natural_area', 'water_area','other_area']].sum(axis=0))
print('Area (km2) totals')
print(frac_df[['natural_area', 'water_area','other_area']].sum(axis=0)/(1000*1000))

print('Mean totals, all reservoirs')
print(frac_df[['natural', 'water','other']].mean(axis=0))
print('Mean totals, >1 ha reservoirs')
print(frac_df.loc[frac_df['og_area_ha']>1,['natural', 'water','other']].mean(axis=0))

print('Meeting 0.5 requirement:')
print((frac_df.loc[frac_df['og_area_ha']>1, 'natural_no_water'] > 0.5).mean())
print('Meeting 0.9 requirement:')
print((frac_df.loc[frac_df['og_area_ha']>1, 'natural_no_water'] > 0.9).mean())
print('Total available for reforestation')
print(frac_df.loc[frac_df['og_area_ha']>1, ['other_area']].sum()/(1000*1000))
