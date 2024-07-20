for f in ../../predict/full_outputs/v2/full_aea/*.tif;
do 
    basename_f=$(basename $f)
    csv_in=out/${basename_f/.tif/_merged.csv}
    out_dir=out_tifs/${basename_f/.tif/}
    mkdir -p $out_dir
    time python3 write_out_new_rasters.py $f $csv_in $out_dir
done
