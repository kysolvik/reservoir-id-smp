for f in ../../predict/full_outputs/v2/full_aea/*.tif;
do 
    basename_f=$(basename $f)
    csv_out=out/${basename_f/tif/csv}
    time python3 find_overlaps_badwater.py $f data/hydropolys_raster_byte_aea.tif $csv_out
done

# for f in ../../predict/full_outputs/v2/full/ls7*;
# do 
#     basename_f=$(basename $f)
#     csv_out=out/${basename_f/tif/csv}
#     time python3 find_overlaps_badwater.py $f data/hydropolys_raster_byte.tif $csv_out
# done
