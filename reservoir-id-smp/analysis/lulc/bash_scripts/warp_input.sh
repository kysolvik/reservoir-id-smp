for f in ./in/mato_grosso/wgs84/*.tif;
do
    gdalwarp $f ${f/.tif/_aea_30m.tif} -tr 30 30 -t_srs ESRI:102033 -co "COMPRESS=LZW" -srcnodata 0 -dstnodata 0 -wm 4096 --config GDAL_CACHEMAX 4096
done

# for f in ./in/wgs84/*.tif;
# do
#     gdalwarp $f ${f/.tif/_aea_10m.tif} -tr 10 10 -t_srs ESRI:102033 -te 656477.922 1847011.958 915697.922 2396361.958 -co "COMPRESS=LZW" -srcnodata 0 -dstnodata 0 -wm 4096 --config GDAL_CACHEMAX 4096
# done
