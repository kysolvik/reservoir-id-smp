# For landsat
satellite=$1
sat_num=${satellite:(-1)}
sat_short="ls${sat_num}"
sudo apt install -y zip
for d in out_allbrazil/$satellite/*;
do 
    y=$(basename $d)
    zip -rj "${sat_short}_${y}_preds_v3.zip" $d
done
gsutil -m cp *.zip gs://res-id/cnn/predict/landsat_v3/zips/
