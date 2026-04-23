ls5_ls7_list=$(seq 2000 2011)
ls7_ls8_list=$(seq 2014 2025)
ls8_ls9_list=$(seq 2022 2025)

# Ls5 ls7 first
for y in $ls5_ls7_list;
do
    gsutil cp gs://res-id/cnn/predict/landsat_v3/full_outputs_cloudfilt_aea/ls5*${y}* ./data/
    gsutil cp gs://res-id/cnn/predict/landsat_v3/full_outputs_cloudfilt_aea/ls7*${y}* ./data/
    python3 -u compare_full_maps.py ./data/ls5*$y*.tif ./data/ls7*${y}*.tif out/ls5ls7_${y}_compare.csv
    rm ./data/*$y*
done

# Ls7 ls8 
for y in $ls7_ls8_list;
do
    gsutil cp gs://res-id/cnn/predict/landsat_v3/full_outputs_cloudfilt_aea/ls7*${y}* ./data/
    gsutil cp gs://res-id/cnn/predict/landsat_v3/full_outputs_cloudfilt_aea/ls8*${y}* ./data/
    python3 -u compare_full_maps.py ./data/ls7*$y*.tif ./data/ls8*${y}*.tif out/ls7ls8_${y}_compare.csv
    rm ./data/*$y*
done

# Ls8 ls9 
for y in $ls8_ls9_list;
do
    gsutil cp gs://res-id/cnn/predict/landsat_v3/full_outputs_cloudfilt_aea/ls8*${y}* ./data/
    gsutil cp gs://res-id/cnn/predict/landsat_v3/full_outputs_cloudfilt_aea/ls9*${y}* ./data/
    python3 -u compare_full_maps.py ./data/ls8*$y*.tif ./data/ls9*${y}*.tif out/ls8ls9_${y}_compare.csv
    rm ./data/*$y*
done
