in=$1
out=$2
gdalwarp $in $out -co "COMPRESS=LZW" \
    -t_srs "+proj=aea +lat_0=-32 +lon_0=-60 +lat_1=-5 +lat_2=-42 +x_0=0 +y_0=0 +ellps=aust_SA +units=m +no_defs +type=crs" \
    -tr 10 10 \
    -wm 2000 -multi -wo NUM_THREADS=4 -co "TILED=YES"
