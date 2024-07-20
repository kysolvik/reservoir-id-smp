mkdir -p out_tifs
mkdir -p fullout/zips/
mkdir -p fullout/aea/
mkdir -p temp_vrts
for f in $(gsutil ls gs://res-id/cnn/predict/landsat_v2/zips_merged/);
do
    # Get info about year and satellite
	zip_name=$(basename $f)
    fp_zip="./fullout/zips/$zip_name"
    landsat_name=${zip_name:0:3}
    year=${zip_name:4:4}
    filename_base=${landsat_name}_${year}_v2
    dir_unzipped="./out_tifs/${filename_base}_aea"
    fp_vrt="temp_vrts/full_temp.vrt"
    fp_aea="./fullout/aea/${filename_base}_aea_cleaned.tif"

    echo $f
    echo $zip_name
    echo $landsat_name
    echo $year
    echo $filename_base
    echo $dir_unzipped
    echo $fp_vrt
    echo $fp_aea

    # Skip 5 year intervals (already run previously)
    if ((year%5 == 0));
    then
        continue
    fi
    
    # Copy and unzip data
    gsutil cp $f $fp_zip

    unzip $fp_zip

    # Build vrt
    gdalbuildvrt -srcnodata 255 -vrtnodata 255 $fp_vrt $dir_unzipped/*.tif


    # Create full tif
    gdal_translate -co "COMPRESS=LZW" $fp_vrt $fp_aea

    # Copy outputs to google cloud
    gsutil cp $fp_aea gs://res-id/cnn/predict/landsat_v2/full_outputs_aea_cleaned/


    # Remove inputs
    rm $fp_vrt
    rm $fp_zip
    rm -r $dir_unzipped
    rm $fp_aea

done
