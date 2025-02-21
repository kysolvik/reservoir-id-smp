f=$1
basename_f=$(basename $f)
csv_out=out/${basename_f/tif/csv}
time python3 find_overlaps_badwater.py $f data/hydropolys_raster_byte_aea.tif $csv_out
