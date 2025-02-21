gdalbuildvrt data/souza/souza_full.vrt ./data/souza/2017*.tif
gdalwarp -co "COMPRESS=LZW" -t_srs "ESRI:102033" data/souza/souza_full.vrt data/souza/souza_full_aea.tif
