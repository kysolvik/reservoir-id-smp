# Args:
model_gs_path=gs://res-id/cnn/models/best_smp/ls8_model31_6band_quantized.ckpt

# mean_std_path=gs://res-id/cnn/training/prepped_gaip_landsat/mean_std_ls8_v3.npy
# bands_minmax_path=gs://res-id/cnn/training/prepped_gaip_landsat/landsat_all_imgs_bands_min_max_ls8_10m_v3.npy

mean_std_path=gs://res-id/cnn/training/prepped_gaip_landsat/mean_std_ls7_v7.npy
bands_minmax_path=gs://res-id/cnn/training/prepped_gaip_landsat/landsat_all_imgs_bands_min_max_ls10m_ls7_v6.npy

# mean_std_path=gs://res-id/cnn/training/prepped_gaip_landsat/mean_std_ls5_v2.npy
# bands_minmax_path=gs://res-id/cnn/training/prepped_gaip_landsat/landsat_all_imgs_bands_min_max_ls10m_ls5_v2.npy

# Careful! Cannot include trailing '/' in data_path
data_path=gs://res-id/ee_exports/landsat7_30m_v2/2005
region_shp_path='gs://res-id/cnn/aux_data/Brazil_10kmbuffer_noremoteislands_noholes.*'


# Terminal settings
export TERM=xterm
echo 'export TERM=xterm' >> .bashrc

# # Machine setup: Can comment out if machine already has these packages
# mkdir -p ~/miniconda3
# 
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
# 
# bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
# rm -rf ~/miniconda3/miniconda.sh
# export PATH="$HOME/miniconda3/bin:$PATH"
# echo export PATH="$HOME/miniconda3/bin:$PATH" >> .bashrc
# conda install -y libmamba
# conda config --set solver libmamba
# 
# conda install -y -c conda-forge pytorch-lightning pytorch rasterio segmentation-models-pytorch geopandas albumentations rasterio gdal
# pip install neural-compressor
# sudo apt install -y gdal-bin
# sudo apt install -y task-spooler
# sudo apt install -y git

# Get code
git clone https://github.com/kysolvik/reservoir-id-smp.git
cd reservoir-id-smp/reservoir-id-smp/predict/
git checkout quantized-prediction

# Get data files
gsutil cp $model_gs_path ./model.ckpt
gsutil cp $mean_std_path ./mean_std.npy
gsutil cp $bands_minmax_path ./bands_minmax.npy
mkdir shp
gsutil cp $region_shp_path ./shp/

# Prep vrt
gsutil ls ${data_path}/*.tif > filelist.txt
sed -i 's!gs://!/vsigs/!' filelist.txt
# For Landsat8:
# gdalbuildvrt ls_10m.vrt -r cubic -tap -tr 0.000089831528412 0.000089831528412 -te -74.0601663 -33.7932735 -29.7898416 5.3238655 -input_file_list filelist.txt -b 2 -b 3 -b 4 -b 5 -b 6 -b 7
# For Landsat7 and 5:
gdalbuildvrt ls_10m.vrt -r cubic -tap -tr 0.000089831528412 0.000089831528412 -te -74.0601663 -33.7932735 -29.7898416 5.3238655 -input_file_list filelist.txt -b 1 -b 2 -b 3 -b 4 -b 5 -b 6


# Run
mkdir out/
tsp python3 -u predict_smp_landsat.py ls_10m.vrt model.ckpt mean_std.npy bands_minmax.npy out/ --quantized --region_shp shp/Brazil_10kmbuffer_noremoteislands_noholes.shp

# Prep logs/shutdown
mkdir logs
tsp bash tsp_wrapup.sh
