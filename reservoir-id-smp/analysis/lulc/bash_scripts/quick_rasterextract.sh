for y in {1985..2022..1};
do
    res_shp=./in/shps/*${y}*.shp
    mb_tif=./in/*${y}*.tif
    python3 raster-buffer-extract/raster-buffer-extract/fraster_extract_wrapper.py $res_shp $mb_tif out/test_${y}_counts_10000sample.csv 1000 --stat count_dict --not_latlon --nsample 10000
done
