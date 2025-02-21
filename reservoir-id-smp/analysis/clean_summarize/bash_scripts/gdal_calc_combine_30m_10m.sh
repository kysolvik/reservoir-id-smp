for f in ./out_tifs/sentinel_2021_v6/*.tif;
do
    basename="$(basename -- $f)"
    in_file_30m=./out_tifs/sentinel_2021_v6_30m/$basename
    out_file=./out_tifs/sentinel_2021_v6_combined/$basename
    gdal_calc.py --calc="numpy.max([A,B], axis=0)"\
        --outfile=$out_file \
        --co='COMPRESS=LZW' \
        -A $f -B $in_file_30m
done
