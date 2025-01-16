for y_res in {1984..2023..1};
do
    for sat in ls5 ls7 ls8;
    do
        res_shp=./in/shps_all_mt/${sat}*${y_res}*.shp
        if [ -f $res_shp ]; 
        then
            for y_mb in $(seq 1985 2023);
            do
            echo $sat $y_res $y_mb
                mb_tif=./in/mato_grosso/aea/*${y_mb}*.tif
                time python3 raster-buffer-extract/raster-buffer-extract/fraster_extract_wrapper.py $res_shp $mb_tif out/full/lulc_stats_${sat}_res_${y_res}_mb_${y_mb}_counts.csv 1000 --stat count_dict --not_latlon --nsample 10000
            done
        fi
    done
done
