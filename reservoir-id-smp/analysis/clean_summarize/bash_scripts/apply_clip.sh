# Warp to aea
fp_aea=$1
fp_aea_clip=$2
gdalwarp $fp_aea $fp_aea_clip -co "COMPRESS=LZW" \
    -t_srs "ESRI:102033" \
    -tr 10 10 \
    -te -1658960.622 -394331.267 2975949.378 4214708.733 \
    -wm 2000 -multi -wo NUM_THREADS=2 -co "TILED=YES" \
    -cutline ../regions/data/lm_bioma_250_DISSOLVED_AEA.shp

