year_list=(2020)
satellite="landsat8"

# Landsat 8
if [ $satellite = 'landsat8' ]; then
    mean_std_path=gs://res-id/cnn/training/prepped_gaip_landsat/mean_std_ls8_v9.npy
    bands_minmax_path=gs://res-id/cnn/training/prepped_gaip_landsat/landsat_all_imgs_bands_min_max_ls8_10m_v9.npy
fi

# Landsat 7
if [ $satellite = 'landsat7' ]; then
    mean_std_path=gs://res-id/cnn/training/prepped_gaip_landsat/mean_std_ls7_v9.npy
    bands_minmax_path=gs://res-id/cnn/training/prepped_gaip_landsat/landsat_all_imgs_bands_min_max_ls7_10m_v9.npy
fi

# Landsat 5
if [ $satellite = 'landsat5' ]; then
    mean_std_path=gs://res-id/cnn/training/prepped_gaip_landsat/mean_std_ls7_v9.npy
    bands_minmax_path=gs://res-id/cnn/training/prepped_gaip_landsat/landsat_all_imgs_bands_min_max_ls5_10m_v9.npy
fi

# Args:
out_base_path=./out_allbrazil/${satellite}
model_gs_path=gs://res-id/cnn/models/best_smp/l8_sr_v21_quantized.pt

# Careful! Cannot include trailing '/' in data_path
data_path_base=gs://res-id/ee_exports/all_brazil/${satellite}_30m
region_shp_path='gs://res-id/cnn/aux_data/Brazil_10kmbuffer_noremoteislands_noholes.*'

# Terminal settings
export TERM=xterm
echo 'export TERM=xterm' >> .bashrc

# GDAL proj info locations
export GDAL_DATA=/home/ksolvik/miniconda3/share/gdal/
export PROJ_LIB=/home/ksolvik/miniconda3/share/proj/

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
    gdalbuildvrt vrts/ls_10m_${satellite}_${y}.vrt -r cubic -tap -te -74.0539679 -33.7869853 -34.7077585 5.3175773 -tr 0.000089831528412 0.000089831528412 -input_file_list filelist.txt
    # Small area test
#     gdalbuildvrt vrts/ls_10m_${satellite}_${y}.vrt -r cubic -tap -te -50.0601663 -10.7932735 -49.7898416 -9.3238655 -tr 0.000089831528412 0.000089831528412 -input_file_list filelist.txt
    rm filelist.txt

    # Run
    tsp python3 -u predict_smp_landsat.py vrts/ls_10m_${satellite}_${y}.vrt model.ckpt mean_std.npy bands_minmax.npy \
        $out_path --quantized --region_shp shp/Brazil_10kmbuffer_noremoteislands_noholes.shp
done

# Prep logs/shutdown
mkdir -p logs
tsp bash tsp_wrapup.sh
