year_list=(2020)
satellite="landsat8"
pred_threshold=0.703

# Args:
out_base_path=./out_allbrazil/${satellite}
model_gs_path=gs://res-id/cnn/models/best_smp/ls8_sr_v21_quantized.pt

mean_std_path=gs://res-id/cnn/training/prepped_gaip_landsat/mean_std_ls8_v9.npy
bands_minmax_path=gs://res-id/cnn/training/prepped_gaip_landsat/landsat_all_imgs_bands_min_max_ls8_10m_v9.npy

# Careful! Cannot include trailing '/' in data_path
data_path_base=gs://res-id/ee_exports/all_brazil/${satellite}_30m
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
for y in ${year_list[@]};
do
    out_path=${out_base_path}/$y
    echo $out_path
    mkdir -p $out_path
    gsutil ls "${data_path_base}/${y}/*.tif" > filelist.txt
    sed -i 's!gs://!/vsigs/!' filelist.txt

    mkdir -p vrts
    # For Landsat8:
    gdalbuildvrt vrts/ls_10m_${satellite}_${y}.vrt -r cubic -tap -tr 0.000089831528412 0.000089831528412 -input_file_list filelist.txt
    rm filelist.txt

    # Run
    tsp python3 -u predict_smp_landsat.py vrts/ls_10m_${satellite}_${y}.vrt model.ckpt mean_std.npy bands_minmax.npy \
        $out_path --quantized --region_shp shp/Brazil_10kmbuffer_noremoteislands_noholes.shp \
        --threshold $pred_threshold
done

# Prep logs/shutdown
# mkdir -p logs
# tsp bash tsp_wrapup.sh
