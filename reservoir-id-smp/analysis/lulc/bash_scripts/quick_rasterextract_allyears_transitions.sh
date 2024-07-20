for y_res in {1985..2022..1};
do
    for y_mb in $(seq 1986 2022);
    do
        y_mb_last=$((y_mb-1))
        res_shp=./in/shps_all_mt/*${y_res}*.shp
        mb_tif=./out/lulc_transition_maps/mb_transition_${y_mb_last}_${y_mb}.tif
        python3 raster-buffer-extract/raster-buffer-extract/fraster_extract_wrapper.py $res_shp $mb_tif out/lulc_stats_res_${y_res}_mb_${y_mb}_counts_transitions.csv 1000 --stat count_dict --not_latlon --nsample 10000
    done
done
