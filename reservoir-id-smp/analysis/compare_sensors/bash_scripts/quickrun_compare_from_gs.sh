ls5_ls7_list=$(seq 2000 2011)
ls7_ls8_list=$(seq 2014 2023)

# Ls5 ls7 first
for y in $ls5_ls7_list;
do
    gsutil -m cp gs://res-id/cnn/predict/landsat_v3/full_outputs_aea/*${y}* ./data/
    python3 -u compare_full_maps.py ./data/ls5*$y*.tif ./data/ls7*${y}*.tif out/${y}_compare.csv
    rm ./data/*$y*
done

# Ls5 ls8 
for y in $ls7_ls8_list;
do
    gsutil cp gs://res-id/cnn/predict/landsat_v3/full_outputs_aea/*${y}* ./data/
    python3 -u compare_full_maps.py ./data/ls7*$y*.tif ./data/ls8*${y}*.tif out/${y}_compare.csv
    rm ./data/*$y*
done
