for f in $(gsutil ls gs://res-id/cnn/predict/landsat_v3/zips/);
do
    # Get info about year and satellite
	zip_name=$(basename $f)
    fp_zip="./fullout/zips/$zip_name"
    landsat_name=${zip_name:0:3}
    ls_number=${landsat_name:2:1}
    year=${zip_name:4:4}
    filename_base=${landsat_name}_${year}_v3
    dir_unzipped="./fullout/unzipped/"
    fp_vrt="temp_vrts/full_temp.vrt"
    fp_wgs84="./fullout/wgs84/${filename_base}.tif"
    fp_aea="./fullout/aea/${filename_base}_aea.tif"

    # Copy and unzip data
    gsutil cp $f $fp_zip

    mkdir -p $dir_unzipped
    unzip $fp_zip -d $dir_unzipped

    # Build vrts
    mkdir -p temp_vrts
    find ${dir_unzipped} -type f > out_filelist.txt

    gdalbuildvrt -a_srs "EPSG:4326" -srcnodata 255 -vrtnodata 255 $fp_vrt -input_file_list out_filelist.txt

    # Create wgs84 tif
    gdal_translate -a_srs "EPSG:4326" -co "COMPRESS=LZW" $fp_vrt $fp_wgs84

    # Warp to aea
	gdalwarp $fp_wgs84 $fp_aea -co "COMPRESS=LZW" \
	    -t_srs "ESRI:102033" \
	    -tr 10 10 \
	    -te -1658960.622 -394331.267 2975949.378 4214708.733 \
	    -wm 2000 -multi -wo NUM_THREADS=2 -co "TILED=YES"


    # Copy outputs to google cloud
    gsutil cp $fp_wgs84 gs://res-id/cnn/predict/landsat_v3/full_outputs/
    gsutil cp $fp_aea gs://res-id/cnn/predict/landsat_v3/full_outputs_aea/


    # Remove inputs
    rm -r temp_vrts
    rm $fp_zip
    rm -r $dir_unzipped
    rm $fp_wgs84
    rm $fp_aea

done
