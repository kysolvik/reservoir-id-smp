import geopandas as gpd
import pandas as pd
import rioxarray
import numpy as np

def zonal_stats_clip_max(df, rioar):
    ar_clip = rioar.rio.clip(df.geometry, df.crs, from_disk=True, all_touched=True)
    return ar_clip.max().values

def poly_flow_max(gdf, log_every=0):
    flow_acc = rioxarray.open_rasterio("data/sa_acc_3s.tif", masked=True)
    out_dict = {}
    i = 0
    for id in gdf.index:
        gdf_small = gdf.loc[[id]]
        out_dict[id] = zonal_stats_clip_max(gdf_small, flow_acc)
        if log_every > 0 and i % log_every == 0:
            print(i, id)
        i+=1
    max_flow_df = pd.DataFrame.from_dict(out_dict, orient='index', columns=['max_flow_acc'])
    return gdf.join(max_flow_df)

def prep_run(gdf, out_file):
    gdf = gdf.to_crs('EPSG:4326')
    gdf['geometry'] = gdf.make_valid()
    res_flow_max = poly_flow_max(gdf, log_every=1000)
    res_flow_max.to_csv(out_file)

# Our reservoirs
res_gdf_polies = gpd.read_file('../clean_summarize/out_polys/sentinel_2021_v7_aea_cleaned_no0.gpkg')
prep_run(res_gdf_polies, out_file='./out/flow_acc/res_flow_max.csv')

# ANA
ana_gdf = gpd.read_file('../compare_previous_results/data/ana/Massas_d_Agua.shp')
ana_gdf = ana_gdf.loc[ana_gdf['detipomass']=='Artificial']
prep_run(ana_gdf, './out/flow_acc/ana_flow_max.csv')

# CAR
car_gdf = gpd.read_file('../compare_previous_results/data/car/full_reservoirs_dissolve_explode.shp')
car_gdf = car_gdf.loc[~car_gdf.clip_by_rect(-100, -60, -30, 20).is_empty]
prep_run(car_gdf, out_file='./out/flow_acc/car_flow_max.csv')

# MapBiomas
mb_gdf = gpd.read_file('../compare_previous_results/data/mapbiomas/collection4_v1.gpkg')
mb_gdf = mb_gdf.loc[mb_gdf['DN']==3]
prep_run(car_gdf, out_file='./out/flow_acc/mb_flow_max.csv')

