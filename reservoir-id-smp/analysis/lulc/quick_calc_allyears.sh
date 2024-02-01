for y in {2000..2020..5};
do
    for landsat in ls5 ls7 ls8
    do 
        echo "Starting"
        echo $landsat
        echo $y
        res_tif="../clip/out/${landsat}_${y}_v2_clip_aea.tif"
        muni_tif="../area_count/data/munis_raster_aea.tif"
        out_csv="./csvs/${landsat}_${y}_mins.csv"
        if test -f $res_tif; then
            python3 ./calc_areas_count_regions_lulc.py $res_tif $muni_tif $y $out_csv
        fi
    done
done
