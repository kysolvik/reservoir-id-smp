gdalbuildvrt data/mapbiomas/v3.vrt ./data/mapbiomas/v3/*.tif
gdalwarp -co "COMPRESS=LZW" -t_srs "ESRI:102033" data/mapbiomas/v3.vrt data/mapbiomas/v3_aea.tif
