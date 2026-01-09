f=$1
cutoff=62
basename_f=$(basename $f)
csv_out=out/${basename_f/tif/csv}
merge_csv_out=${csv_out/.csv/_bordermerged.csv}
out_tif_dir=out_tifs/${basename_f/.tif/_cleaned}
mkdir $out_tif_dir

python3 find_overlaps_badwater.py $f $cutoff data/hydropolys_raster_byte_aea.tif $csv_out

python3 merge_borders.py $csv_out $merge_csv_out

python3 write_out_new_rasters_10m.py $f $merge_csv_out $out_tif_dir
