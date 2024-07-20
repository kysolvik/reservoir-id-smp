for f in $(gsutil ls gs://res-id/cnn/predict/landsat_v2/full_outputs_aea/);
do 
    fn_aea=$(basename $f)
    merge_csv=out/${fn_aea/.tif/_merged.csv}
    out_dir=out_tifs/${fn_aea/.tif/}
    out_zip=${fn_aea/.tif/_merged.zip}
    landsat_name=${fn_aea:0:3}
    year=${fn_aea:4:4}

    mkdir -p $out_dir

    # Skip 5 year intervals (already run previously)
    if ((year%5 == 0));
    then
        continue
    fi
    
    # Copy input tif
    gsutil cp $f $fn_aea

    # Write out new rasters
    time python3 write_out_new_rasters.py $fn_aea $merge_csv $out_dir

    # Remove input tif
    rm $fn_aea

    # Zip up results and copy to cloud storage
    zip -r $out_zip $out_dir
    gsutil cp $out_zip gs://res-id/cnn/predict/landsat_v2/zips_merged/
    rm $out_zip
    rm -r $out_dir


done

