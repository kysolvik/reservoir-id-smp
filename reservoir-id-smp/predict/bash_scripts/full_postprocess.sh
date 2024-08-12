for f in $(gsutil ls gs://res-id/cnn/predict/landsat_v2/zips/);
do
    # Get info about year and satellite
	zip_name=$(basename $f)
    fp_zip="./fullout/zips/$zip_name"
    landsat_name=${zip_name:0:3}
    ls_number=${landsat_name:2:1}
    year=${zip_name:4:4}
    filename_base=${landsat_name}_${year}_v2
    dir_unzipped="./fullout/unzipped/"
    fp_vrt="temp_vrts/full_temp.vrt"
    fp_wgs84="./fullout/wgs84/${filename_base}.tif"
    fp_aea="./fullout/aea/${filename_base}_aea.tif"

    # Skip 5 year intervals (already run previously)
    if ((year%5 == 0));
    then
        continue
    fi
    
    # Also skip 1984 and 2017
    if ((year==1984)) || ((year==2017))
    then
        continue
    fi

    # Copy and unzip data
    gsutil cp $f $fp_zip

    mkdir -p $dir_unzipped
    unzip $fp_zip -d $dir_unzipped

    # Build vrts
    mkdir -p temp_vrts
    gdalbuildvrt -srcnodata 255 -vrtnodata 255 temp_vrts/temp_0.vrt ${dir_unzipped}/pred_0*.tif

    for i in {10..99};do
        gdalbuildvrt -srcnodata 255 -vrtnodata 255 temp_vrts/temp_${i}.vrt ${dir_unzipped}/pred_${i}*.tif
    done

    gdalbuildvrt -srcnodata 255 -vrtnodata 255 $fp_vrt temp_vrts/temp_*.vrt


    # Create wgs84 tif
    gdal_translate -co "COMPRESS=LZW" $fp_vrt $fp_wgs84


    # Warp to aea
	gdalwarp $fp_wgs84 $fp_aea -co "COMPRESS=LZW" \
	    -t_srs "+proj=aea +lat_0=-32 +lon_0=-60 +lat_1=-5 +lat_2=-42 +x_0=0 +y_0=0 +ellps=aust_SA +units=m +no_defs +type=crs" \
	    -tr 10 10 \
	    -wm 2000 -multi -wo NUM_THREADS=2 -co "TILED=YES"


    # Copy outputs to google cloud
    gsutil cp $fp_wgs84 gs://res-id/cnn/predict/landsat_v2/full_outputs/
    gsutil cp $fp_aea gs://res-id/cnn/predict/landsat_v2/full_outputs_aea/


    # Remove inputs
    rm -r temp_vrts
    rm $fp_zip
    rm -r $dir_unzipped
    rm $fp_wgs84
    rm $fp_aea

done
