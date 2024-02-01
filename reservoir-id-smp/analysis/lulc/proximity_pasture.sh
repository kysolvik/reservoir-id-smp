# in_tif=$1
# out_tif=$2
# gdal_proximity.py $in_tif $out_tif -values 15 -distunits PIXEL -use_input_nodata YES -co "COMPRESS=LZW" -ot Byte -maxdist 200
for f in ./in/*aea_10m.tif;
do
    year=$(echo $f | sed 's/.*\([0-9]\{4\}\).*/\1/g')
    out_tif="./out/mb_c8_pasture_prox_${year}.tif"
    gdal_proximity.py $f $out_tif -values 15\
        -distunits PIXEL -use_input_nodata YES -co "COMPRESS=LZW" -ot Byte -maxdist 250
done
