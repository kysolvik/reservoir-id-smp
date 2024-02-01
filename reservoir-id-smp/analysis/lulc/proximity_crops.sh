for f in ./in/*aea_10m.tif;
do
    year=$(echo $f | sed 's/.*\([0-9]\{4\}\).*/\1/g')
    out_tif="./out/mb_c8_crops_prox_${year}.tif"
    gdal_proximity.py $f $out_tif -values 18,19,39,20,40,62,41,36,46,47,35,48 \
        -distunits PIXEL -use_input_nodata YES -co "COMPRESS=LZW" -ot Byte -maxdist 250
done

