for f in $(gsutil ls gs://res-id/cnn/predict/landsat_v2/full_outputs_aea/);
do 
    fn_aea=$(basename $f)
    csv_out=out/${fn_aea/tif/csv}
    landsat_name=${fn_aea:0:3}
    year=${fn_aea:4:4}

    mkdir -p out

    echo $fn_aea
    echo $csv_out
    echo $landsat_name
    echo $year

    # Skip 5 year intervals (already run previously)
    if ((year%5 == 0));
    then
        continue
    fi

    gsutil cp $f $fn_aea

    time python3 find_overlaps_badwater.py $fn_aea data/hydropolys_raster_byte_aea.tif $csv_out

    rm $fn_aea
done

