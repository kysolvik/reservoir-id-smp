gdal_rasterize -burn 1 -ot Uint16 -tr 0.000089831528412 0.000089831528412 -te -74.0539679 -33.7869853 -34.7077585 5.3175773 -co "COMPRESS=LZW" ./data/hydropolys_gt10ha.shp ./data/hydropolys_raster.tif

