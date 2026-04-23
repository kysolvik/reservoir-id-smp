# Set some necessary env variables
export PATH="/home/ksolvik/miniconda3/bin/:$PATH"
# GDAL settings
export GDAL_DATA=/home/ksolvik/miniconda3/share/gdal/
export PROJ_LIB=/home/ksolvik/miniconda3/share/proj/
satellite=$1
# Landsat 9
if [ $satellite = "landsat9" ]; then
    mean_std_path=gs://res-id/cnn/training/prepped_gaip_landsat/mean_std_ls9_v9.npy
    bands_minmax_path=gs://res-id/cnn/training/prepped_gaip_landsat/landsat_all_imgs_bands_min_max_ls9_10m_v9.npy
fi
# Landsat 8
if [ $satellite = "landsat8" ]; then
    mean_std_path=gs://res-id/cnn/training/prepped_gaip_landsat/mean_std_ls8_v9.npy
    bands_minmax_path=gs://res-id/cnn/training/prepped_gaip_landsat/landsat_all_imgs_bands_min_max_ls8_10m_v9.npy
fi

# Landsat 7
if [ $satellite = "landsat7" ]; then
    mean_std_path=gs://res-id/cnn/training/prepped_gaip_landsat/mean_std_ls7_v9.npy
    bands_minmax_path=gs://res-id/cnn/training/prepped_gaip_landsat/landsat_all_imgs_bands_min_max_ls7_10m_v9.npy
fi

# Landsat 5
if [ $satellite = "landsat5" ]; then
    mean_std_path=gs://res-id/cnn/training/prepped_gaip_landsat/mean_std_ls7_v9.npy
    bands_minmax_path=gs://res-id/cnn/training/prepped_gaip_landsat/landsat_all_imgs_bands_min_max_ls5_10m_v9.npy
fi
echo $mean_std_path

model_gs_path=gs://res-id/cnn/models/best_smp/l8_sr_v21_quantized.pt

# Get data files
gsutil cp $model_gs_path ./model.ckpt
gsutil cp $mean_std_path ./mean_std.npy
gsutil cp $bands_minmax_path ./bands_minmax.npy
