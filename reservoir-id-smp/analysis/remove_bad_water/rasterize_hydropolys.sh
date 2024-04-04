# gdal_rasterize -a 'area_ha' -ot Byte -tr 0.000089831528412 0.000089831528412 -te -74.0539679 -33.7869853 -34.7077585 5.3175773 -co "COMPRESS=LZW" ./data/hydropolys_area_ha.shp ./data/hydropolys_raster_byte.tif

gdal_rasterize -a 'area_ha' -ot Byte -tr 10 10 -te -1658960.622 4214708.733 2975949.378 -394331.267 -co "COMPRESS=LZW" ./data/hydropolys_aea_area_ha.shp ./data/hydropolys_raster_byte_aea.tif
