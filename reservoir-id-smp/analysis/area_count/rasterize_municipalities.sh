gdal_rasterize -ot Byte -tr 10 10 -te 656477.922 1847011.958 915697.922 2396361.958 -a "ID" -co "COMPRESS=LZW" -a_nodata 255 ../clip/brazil_munis_interviews_aea.shp ./data/munis_raster_aea_v2.tif

