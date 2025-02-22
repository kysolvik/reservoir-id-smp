# For sentinel
sudo apt install -y zip
for d in out_allbrazil/sentinel/*;
do 
    y=$(basename $d)
    zip -rj "sentinel_2021_preds_v6.zip" $d
done
gsutil -m cp *.zip gs://res-id/cnn/predict/sentinel_v6/zips/
