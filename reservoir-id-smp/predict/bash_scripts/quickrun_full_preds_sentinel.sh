y=2023
satellite="sentinel"

# Args:
out_base_path=./out_allbrazil/${satellite}
model_gs_path=gs://res-id/cnn/models/best_smp/manet_resnet_model25.ckpt

mean_std_path=gs://res-id/cnn/training/prepped_gaip/mean_std_sentinel_v7.npy
bands_minmax_path=gs://res-id/cnn/training/prepped_gaip/all_imgs_bands_min_max_sentinel_v6.npy

# Careful! Cannot include trailing '/' in data_path
data_path_s1_10m=gs://res-id/ee_exports/sentinel1_10m_allbrazil
data_path_s2_10m=gs://res-id/ee_exports/sentinel2_10m_allbrazil_v2
data_path_s2_20m=gs://res-id/ee_exports/sentinel2_20m_allbrazil
region_shp_path='gs://res-id/cnn/aux_data/Brazil_10kmbuffer_noremoteislands_noholes.*'

# Terminal settings
export TERM=xterm
echo 'export TERM=xterm' >> .bashrc

# Get code
git clone https://github.com/kysolvik/reservoir-id-smp.git
cd reservoir-id-smp/reservoir-id-smp/predict/
git checkout quantized-prediction


# Get data files
gsutil cp $model_gs_path ./model.ckpt
gsutil cp $mean_std_path ./mean_std.npy
gsutil cp $bands_minmax_path ./bands_minmax.npy
mkdir -p shp
gsutil cp $region_shp_path ./shp/

# Prep vrt
out_path=${out_base_path}/$y
echo $out_path
mkdir -p $out_path
gsutil ls "${data_path_s1_10m}/${y}/*.tif" > filelist_s1_10m.txt
sed -i 's!gs://!/vsigs/!' filelist_s1_10m.txt
gsutil ls "${data_path_s2_10m}/${y}/*.tif" > filelist_s2_10m.txt
sed -i 's!gs://!/vsigs/!' filelist_s2_10m.txt
gsutil ls "${data_path_s2_20m}/${y}/*.tif" > filelist_s2_20m.txt
sed -i 's!gs://!/vsigs/!' filelist_s2_20m.txt

mkdir -p vrts
gdalbuildvrt vrts/${satellite}_${y}_s1_10m.vrt -r nearest -te -74.0539679 -33.7869853 -34.7077585 5.3175773 -tap -tr 0.000089831528412 0.000089831528412 -input_file_list filelist_s1_10m.txt 
gdalbuildvrt vrts/${satellite}_${y}_s2_10m.vrt -r nearest -te -74.0539679 -33.7869853 -34.7077585 5.3175773 -tap -tr 0.000089831528412 0.000089831528412 -input_file_list filelist_s2_10m.txt 
gdalbuildvrt vrts/${satellite}_${y}_s2_20m.vrt -r cubic -te -74.0539679 -33.7869853 -34.7077585 5.3175773 -tap -tr 0.000089831528412 0.000089831528412 -input_file_list filelist_s2_20m.txt 
# Small area test
# gdalbuildvrt vrts/${satellite}_${y}_s1_10m.vrt -r nearest -tap -te -50.0601663 -10.7932735 -49.7898416 -9.3238655 -tr 0.000089831528412 0.000089831528412 -input_file_list filelist_s1_10m.txt; 
# 
# gdalbuildvrt vrts/${satellite}_${y}_s2_10m.vrt -r nearest -tap -te -50.0601663 -10.7932735 -49.7898416 -9.3238655 -tr 0.000089831528412 0.000089831528412 -input_file_list filelist_s2_10m.txt; 
# 
# gdalbuildvrt vrts/${satellite}_${y}_s2_20m.vrt -r cubic -tap -te -50.0601663 -10.7932735 -49.7898416 -9.3238655 -tr 0.000089831528412 0.000089831528412 -input_file_list filelist_s2_20m.txt; 


    # Run
tsp python3 -u predict_smp_sentinel.py vrts/${satellite}_${y}_s2_10m.vrt model.ckpt mean_std.npy bands_minmax.npy $out_path --region_shp shp/Brazil_10kmbuffer_noremoteislands_noholes.shp --calc_nds

# Prep logs/shutdown
mkdir -p logs
tsp bash tsp_wrapup.sh
