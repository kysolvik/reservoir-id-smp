for f in ../../predict/full_outputs/v3/ls8*.tif;
do 
    basename_f=$(basename $f)
    csv_out=out/${basename_f/tif/csv}
    time python3 find_overlaps_badwater.py $f 178 data/hydropolys_raster_byte_aea.tif $csv_out
done
for f in ../../predict/full_outputs/v3/ls7*.tif;
do 
    basename_f=$(basename $f)
    csv_out=out/${basename_f/tif/csv}
    time python3 find_overlaps_badwater.py $f 1 data/hydropolys_raster_byte_aea.tif $csv_out
done
for f in ../../predict/full_outputs/v3/ls5*.tif;
do 
    basename_f=$(basename $f)
    csv_out=out/${basename_f/tif/csv}
    time python3 find_overlaps_badwater.py $f 9 data/hydropolys_raster_byte_aea.tif $csv_out
done
for f in ../../predict/full_outputs/v3/ls9*.tif;
do 
    basename_f=$(basename $f)
    csv_out=out/${basename_f/tif/csv}
    time python3 find_overlaps_badwater.py $f 178 data/hydropolys_raster_byte_aea.tif $csv_out
done