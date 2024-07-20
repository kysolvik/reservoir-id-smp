for y_res in {1985..2022..1};
do
    for y_mb in $(seq 1985 2022);
    do
        res_shp=./in/shps_all_mt/*${y_res}*.shp
        mb_tif=./in/mato_grosso/aea/*${y_mb}*.tif
        python3 raster-buffer-extract/raster-buffer-extract/fraster_extract_wrapper.py $res_shp $mb_tif out/lulc_stats_res_${y_res}_mb_${y_mb}_counts.csv 1000 --stat count_dict --not_latlon --nsample 10000
    done
done
