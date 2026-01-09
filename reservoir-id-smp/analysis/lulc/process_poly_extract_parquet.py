import pandas as pd
import numpy as np
import geopandas as gpd

in_parquet = './out/poly_extract_50m_frac.parquet'
in_og_shp = '../clean_summarize/out_polys/sentinel_2021_v7_aea_cleaned_polygons.gpkg'
in_diff_shp = './out/poly_buffer_50m_diff.gpkg'
out_csv = './out/poly_lulc_fracs_50m.csv'

mb_keys_dict = {                                                                               
    'natural': np.array([1, 3, 4, 5, 6, 49, 10, 11, 12, 32, 29, 50, 23]),
    'water': np.array([26, 31, 33]),
}

def get_lulc_dict(row):
    return dict(zip(row['unique'], row['frac']))

def get_lulc_fractions(row):
    out_dict = {}
    lulc_dict = get_lulc_dict(row)
    sum_all = 0
    for lulc_class in mb_keys_dict.keys():                                                     
        sum_of_class = 0
        for key in lulc_dict.keys():
            if key in mb_keys_dict[lulc_class]:
                sum_of_class += lulc_dict[key]
        out_dict[lulc_class] = sum_of_class
        sum_all += sum_of_class
    out_dict['other'] = 1 - sum_all
    return out_dict

df = pd.read_parquet(in_parquet)

frac_dicts = df.apply(get_lulc_fractions, axis=1)
frac_df = pd.DataFrame(list(frac_dicts.values))

print('done with lulc')

frac_df['og_area'] = gpd.read_file(in_og_shp).geometry.area
frac_df['ring_area'] = gpd.read_file(in_diff_shp).geometry.area
frac_df.to_csv(out_csv, index=False)
