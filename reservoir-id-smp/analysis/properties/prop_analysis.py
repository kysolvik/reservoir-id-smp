import geopandas as gpd
import rasterio as rio
import pandas as pd
import glob
import os
import rasterstats
import re

# Constants
PROPERTY_SHP = './data/pa_br_landtenure_studyarea_only_aea.shp'
RES_CSV_PATTERN = '../remove_bad_water/out/ls*merged.csv'
LULC_TIF_PATTERN = '../lulc/in/mato_grosso/*.tif'

def year_from_string(string):
    """Regex helper function"""
    match = re.match(r'.*([1-2][0-9]{3})', string)
    if match is None:
        raise ValueError('No year found in string')
    return match.group(1)

def read_process_csv_to_gdf(csv):
    temp_df = pd.read_csv(csv)
    temp_df['satellite'] = os.path.basename(csv)[:3]
    temp_df['year'] = int(os.path.basename(csv)[4:8])
    temp_df = temp_df.loc[temp_df['hydropoly_max']<100]
    temp_df['area_ha'] = temp_df['area']*100/10000 # HA
    temp_df['area_km'] = temp_df['area']*100/(1000*1000) # km2
    temp_df = temp_df.loc[temp_df['area_ha']<100] # Remove greater than 100 ha
    temp_gdf = gpd.GeoDataFrame(
        temp_df, geometry=gpd.points_from_xy(temp_df.center_lon, temp_df.center_lat),
        crs='ESRI:102033'
    )
    return temp_gdf

def sjoin_summarize(points_gdf, poly_gdf, poly_field):
    
    joined_gdf = gpd.sjoin(points_gdf, poly_gdf, predicate='within', how='inner')
    return joined_gdf[['area_ha', poly_field]].groupby(poly_field).agg(['sum', 'count', 'median'])['area_ha']
    
def main():
    prop_gdf = gpd.read_file(PROPERTY_SHP)
    prop_gdf = prop_gdf.rename(columns={'area_ha':'prop_area_ha'})

    # Reservoir analysis
    print('Starting Reservoir Analysis')
    all_csvs = glob.glob(RES_CSV_PATTERN)
    for csv in all_csvs:
        year = year_from_string(os.path.basename(csv))
        print(year, ' Start')
        out_path = './out/res_stats/prop_res_stats_{}.csv'.format(year)
        res_gdf = read_process_csv_to_gdf(csv)
        prop_res_joined = sjoin_summarize(res_gdf, prop_gdf, 'fid')
        prop_res_joined.index = prop_res_joined.index.astype(int)
        prop_res_joined.to_csv(out_path)
        print(year, ' End')

    print('Starting MapBiomas LULC Analysis')
    all_mb_tifs = glob.glob(LULC_TIF_PATTERN)
    for tif_path in all_mb_tifs:
        year = year_from_string(os.path.basename(tif_path))
        print(year, ' Start')
        out_path = './out/mb_stats/prop_mb_stats{}.csv'.format(year)
        mb_lulc = rasterstats.zonal_stats(prop_gdf, tif_path, stats=None, categorical=True)
        mb_lulc_df = pd.DataFrame(mb_lulc).fillna(0).astype(int)
        mb_lulc_df.loc[:,'fid'] = prop_gdf['fid']
        mb_lulc_df.set_index('fid').to_csv(out_path)
        print(year, ' End')

if __name__=='__main__':
    main()
