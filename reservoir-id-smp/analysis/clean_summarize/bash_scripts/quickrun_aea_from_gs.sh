for f in $(gsutil ls gs://res-id/cnn/predict/landsat_v3/full_outputs_cloudfilt_aea/);
do 
    fn_aea=$(basename $f)
    csv_out=out/${fn_aea/tif/csv}
    merge_csv_out=${csv_out/.csv/_merged.csv}
    landsat_name=${fn_aea:0:3}
    year=${fn_aea:4:4}

    if [ -f "$csv_out" ]; then
        echo "Skipping ${csv_out}, already exists"
    else
        if [ $landsat_name = 'ls5' ]; then
            pred_cutoff=9
        fi
        if [ $landsat_name = 'ls7' ]; then
            pred_cutoff=1
        fi
        if [ $landsat_name = 'ls8' ]; then
            pred_cutoff=179
        fi
        if [ $landsat_name = 'ls9' ]; then
            pred_cutoff=179
        fi

        mkdir -p out

        echo $fn_aea
        echo $csv_out
        echo $landsat_name
        echo $year

        gsutil cp $f $fn_aea

        time python3 -u find_overlaps_badwater.py $fn_aea $pred_cutoff data/hydropolys_raster_byte_aea.tif $csv_out
        rm $fn_aea

        time python3 merge_borders.py $csv_out $merge_csv_out
    fi
done

