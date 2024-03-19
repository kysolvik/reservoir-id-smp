"""
Simple script for removing hydropolies smaller than 10ha before rasterize
"""

import geopandas as gpd

brazil_shp = '/home/ksolvik/research/reservoirs/analysis/data/misc/general_borders/Brazil_10kmbuffer_noremoteislands_noholes.shp'
hydropolies_gdb = './data/hydropolys.gdb'

brazil = gpd.read_file('../../../../../analysis/data/misc/general_borders/Brazil_10kmbuffer_noremoteislands_noholes.shp')

hydropolys = gpd.read_file('./data/hydropolys.gdb/', mask=brazil).to_crs('ESRI:102033')

hydropolys.loc[hydropolys.area > 100000].to_crs('EPSG:4326').to_file(
        './data/hydropolys_gt10ha.shp')
