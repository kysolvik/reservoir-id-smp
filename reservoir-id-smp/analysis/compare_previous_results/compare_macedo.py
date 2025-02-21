import geopandas as gpd

aster_gdf = gpd.read_file('./data/macedo_2013/aster_reservoirs_masked_aug062011.shp').to_crs('ESRI:102033')
aster_gdf = gpd.read_file('./data/macedo_2013/aster_reservoirs_masked_aug062011.shp').to_crs('ESRI:102033')
print('Area: ', aster_gdf.loc[aster_gdf['area_km2']<1, 'area_km2'].sum())
print('Count: ',aster_gdf.loc[aster_gdf['area_km2']<1].shape)