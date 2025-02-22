mkdir -p fullout/wgs84
mkdir -p fullout/aea

for f in $(gsutil ls gs://res-id/cnn/predict/sentinel_v6/30m/zips/*2021*);
do
    # Get info about year and satellite
    zip_name=$(basename $f)
    fp_zip="./fullout/zips/$zip_name"
    sat_name="sentinel"
    year=${zip_name:9:4}
    filename_base=${sat_name}_${year}_v6
    echo $year
    echo $zip_name
    echo $filename_base
    dir_unzipped="./fullout/unzipped/"
    fp_vrt="temp_vrts/full_temp.vrt"
    fp_wgs84="./fullout/wgs84/${filename_base}_30m.tif"
    fp_aea="./fullout/aea/${filename_base}_aea_30m.tif"

    # Copy and unzip data
    gsutil cp $f $fp_zip

    mkdir -p $dir_unzipped
    unzip $fp_zip -d $dir_unzipped

    # Build vrts
    mkdir -p temp_vrts
    find fullout/unzipped/ -type f > out_filelist.txt
    gdalbuildvrt -srcnodata 255 -vrtnodata 255 $fp_vrt -input_file_list out_filelist.txt


    # Create wgs84 tif
    gdal_translate -co "COMPRESS=LZW" $fp_vrt $fp_wgs84

    # Warp to aea
	gdalwarp $fp_wgs84 $fp_aea -co "COMPRESS=LZW" \
	    -t_srs "ESRI:102033" \
	    -tr 10 10 \
	    -te -1658960.622 -394331.267 2975949.378 4214708.733 \
	    -wm 2000 -multi -wo NUM_THREADS=2 -co "TILED=YES"


    # Copy outputs to google cloud
    gsutil cp $fp_wgs84 gs://res-id/cnn/predict/sentinel_v6/30m/full_outputs/
    gsutil cp $fp_aea gs://res-id/cnn/predict/sentinel_v6/30m/full_outputs_aea/


    # Remove inputs
#    rm -r temp_vrts
#    rm $fp_zip
#    rm -r $dir_unzipped
#    rm $fp_wgs84
#    rm $fp_aea

done
