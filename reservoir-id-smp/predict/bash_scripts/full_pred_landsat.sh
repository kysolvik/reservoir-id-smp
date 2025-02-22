# Note: Used spot vms instead of this script
# See "spot_vm_startupscript.sh"
year_start=1984
year_end=1984
satellite="landsat5"

# Args:
out_base_path=./out_allbrazil/${satellite}
model_gs_path=gs://res-id/cnn/models/best_smp/ls8_model31_6band_quantized.ckpt

# mean_std_path=gs://res-id/cnn/training/prepped_gaip_landsat/mean_std_ls8_v3.npy
# bands_minmax_path=gs://res-id/cnn/training/prepped_gaip_landsat/landsat_all_imgs_bands_min_max_ls8_10m_v3.npy

# mean_std_path=gs://res-id/cnn/training/prepped_gaip_landsat/mean_std_ls7_v7.npy
# bands_minmax_path=gs://res-id/cnn/training/prepped_gaip_landsat/landsat_all_imgs_bands_min_max_ls10m_ls7_v6.npy

mean_std_path=gs://res-id/cnn/training/prepped_gaip_landsat/mean_std_ls5_v2.npy
bands_minmax_path=gs://res-id/cnn/training/prepped_gaip_landsat/landsat_all_imgs_bands_min_max_ls10m_ls5_v2.npy

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
for y in $(seq $year_start $year_end);
do

    out_path=${out_base_path}/$y
    echo $out_path
    mkdir -p $out_path
    gsutil ls "${data_path_base}/${y}/*.tif" > filelist.txt
    sed -i 's!gs://!/vsigs/!' filelist.txt

    mkdir -p vrts
    # For Landsat8:
    # gdalbuildvrt vrts/ls_10m_${satellite}_${y}.vrt -r cubic -tap -tr 0.000089831528412 0.000089831528412 -input_file_list filelist.txt -b 2 -b 3 -b 4 -b 5 -b 6 -b 7
    # For Landsat7 and 5:
    gdalbuildvrt vrts/ls_10m_${satellite}_${y}.vrt -r cubic -tap -tr 0.000089831528412 0.000089831528412 -input_file_list filelist.txt -b 1 -b 2 -b 3 -b 4 -b 5 -b 6
    rm filelist.txt

    # Run
    tsp python3 -u predict_smp_landsat.py vrts/ls_10m_${satellite}_${y}.vrt model.ckpt mean_std.npy bands_minmax.npy $out_path --quantized --region_shp shp/Brazil_10kmbuffer_noremoteislands_noholes.shp
done

# Prep logs/shutdown
mkdir -p logs
tsp bash bash_scripts/tsp_wrapup.sh
