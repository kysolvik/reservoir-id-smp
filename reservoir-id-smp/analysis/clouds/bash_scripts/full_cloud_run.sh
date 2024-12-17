y=$1
sat_num=$2

# Copy inputs
gsutil cp gs://res-id/cnn/predict/landsat_v3/full_outputs/ls${sat_num}_${y}_v3.tif ./data/preds/
gsutil -m cp -r gs://res-id/ee_exports/all_brazil/landsat${sat_num}_30m/clouds/${y} ./data/clouds/


# Build vrt
gdalbuildvrt data/clouds/ls${sat_num}_${y}_clouds.vrt data/clouds/${y}/*.tif \
    -te -74.0467814 -33.7797988 -34.7149450 5.3103908 

# Run cloud filt
python3 cloud_filter.py data/preds/ls${sat_num}_${y}_v3.tif data/clouds/ls${sat_num}_${y}_clouds.vrt ${y}

# Build output tif and copy to cloud
gdalbuildvrt ls${sat_num}_${y}_cloudfilt_v3.vrt out/*$y*
gdal_translate -co COMPRESS=LZW -co TILED=YES ls${sat_num}_${y}_cloudfilt_v3.vrt ls${sat_num}_${y}_cloudfilt_v3.tif
gsutil cp ls${sat_num}_${y}_cloudfilt_v3.tif gs://res-id/cnn/predict/landsat_v3/full_outputs_cloudfilt/
rm ./ls${sat_num}_${y}_cloudfilt_v3.tif
rm ./ls${sat_num}_${y}_cloudfilt_v3.vrt
rm out/*
rm data/preds/*
rm -r data/clouds/*


