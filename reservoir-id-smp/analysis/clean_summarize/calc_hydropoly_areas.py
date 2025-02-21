"""
Calculate hydropoly areas in ha and round to int before rasterizing
"""

import numpy as np
import geopandas as gpd


brazil_shp = '/home/ksolvik/research/reservoirs/analysis/data/misc/general_borders/Brazil_10kmbuffer_noremoteislands_noholes.shp'
hydropolies_gdb = './data/hydropolys.gdb'


brazil = gpd.read_file('../../../../../analysis/data/misc/general_borders/Brazil_10kmbuffer_noremoteislands_noholes.shp')
hp_gdf = gpd.read_file(hydropolies_gdb, mask=brazil).to_crs('ESRI:102033')
hp_gdf['area_ha'] = hp_gdf.area * 0.0001
hp_gdf.loc[hp_gdf['area_ha'] > 254, 'area_ha'] = 254
hp_gdf['area_ha'] = hp_gdf['area_ha'].astype(np.uint8)
hp_gdf.to_file('./data/hydropolys_aea_area_ha.shp')
