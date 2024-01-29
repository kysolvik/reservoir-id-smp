input_dir='/home/ksolvik/research/reservoirs/code/reservoir-id-smp/reservoir-id-smp/predict/full_outputs/v2/'

for f in $input_dir/*.tif;
do 
    f_base=${f##*/}
    output_file="out/${f_base/.tif/_clip_wgs84.tif}"
    gdalwarp $f $output_file -co "COMPRESS=LZW" \
        -cutline './brazil_munis_interviews.geojson' -crop_to_cutline \
        -wm 2000 -multi -wo NUM_THREADS=4 -co "TILED=YES"
done
