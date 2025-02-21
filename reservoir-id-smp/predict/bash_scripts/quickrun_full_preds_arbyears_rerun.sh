year_list=(2000)
satellite="landsat7"

# Landsat 8
if [ $satellite = 'landsat8' ]; then
    pred_threshold=0.703
    mean_std_path=gs://res-id/cnn/training/prepped_gaip_landsat/mean_std_ls8_v9.npy
    bands_minmax_path=gs://res-id/cnn/training/prepped_gaip_landsat/landsat_all_imgs_bands_min_max_ls8_10m_v9.npy
fi

# Landsat 7
if [ $satellite = 'landsat7' ]; then
    pred_threshold=0.0032
    mean_std_path=gs://res-id/cnn/training/prepped_gaip_landsat/mean_std_ls7_v9.npy
    bands_minmax_path=gs://res-id/cnn/training/prepped_gaip_landsat/landsat_all_imgs_bands_min_max_ls7_10m_v9.npy
fi

# Landsat 5
if [ $satellite = 'landsat5' ]; then
    pred_threshold=0.035
    mean_std_path=gs://res-id/cnn/training/prepped_gaip_landsat/mean_std_ls7_v9.npy
    bands_minmax_path=gs://res-id/cnn/training/prepped_gaip_landsat/landsat_all_imgs_bands_min_max_ls5_10m_v9.npy
fi

# Args:
out_base_path=./out_allbrazil/${satellite}
model_gs_path=gs://res-id/cnn/models/best_smp/l8_sr_v21_quantized.pt

# Careful! Cannot include trailing '/' in data_path
# data_path_base=gs://res-id/ee_exports/all_brazil/${satellite}_30m
data_path_base=gs://res-id/ee_exports/all_brazil_v1/${satellite}_30m
region_shp_path='gs://res-id/cnn/aux_data/Brazil_10kmbuffer_noremoteislands_noholes.*'

# Get code
cd reservoir-id-smp/reservoir-id-smp/predict/

for y in ${year_list[@]};
do
    out_path=${out_base_path}/$y
    echo $out_path

    # Run
    tsp python3 -u predict_smp_landsat.py vrts/ls_10m_${satellite}_${y}.vrt model.ckpt mean_std.npy bands_minmax.npy \
        $out_path --quantized --region_shp shp/Brazil_10kmbuffer_noremoteislands_noholes.shp \
        --threshold $pred_threshold
done

# Prep logs/shutdown
mkdir -p logs
tsp bash tsp_wrapup.sh
