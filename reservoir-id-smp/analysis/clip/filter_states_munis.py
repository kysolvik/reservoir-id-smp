import geopandas as gpd
import numpy as np

# Munis
muni_cd_mun_list = ['5101803', '5106257', '5100201', '5102702', '5107065']

muni_gdf = gpd.read_file('/home/ksolvik/research/reservoirs/analysis/data/misc/municipal_units/BR_Municipios_2021.shp')

muni_gdf_filtered = muni_gdf.loc[muni_gdf['CD_MUN'].isin(muni_cd_mun_list)]
muni_gdf_filtered['ID'] = np.arange(muni_gdf_filtered.shape[0])
muni_gdf_filtered = muni_gdf_filtered.to_crs('EPSG:4326')
muni_gdf_filtered.to_file('./brazil_munis_interviews.geojson', driver='GeoJSON')

muni_gdf_aea = muni_gdf_filtered.to_crs('ESRI:102033')
muni_gdf_aea.to_file('./brazil_munis_interviews_aea.shp')

# States
states_gdf = gpd.read_file('/home/ksolvik/research/reservoirs/analysis/data/misc/general_borders/Brazilian_States.shp')
states_gdf_filtered = states_gdf.loc[states_gdf['UF_05']=='MT']
states_gdf_filtered.to_crs('EPSG:4326').to_file('/home/ksolvik/research/reservoirs/analysis/data/misc/general_borders/MatoGrosso.shp')
states_gdf_filtered.to_crs('ESRI:102033').to_file('/home/ksolvik/research/reservoirs/analysis/data/misc/general_borders/MatoGrosso_aea.shp')