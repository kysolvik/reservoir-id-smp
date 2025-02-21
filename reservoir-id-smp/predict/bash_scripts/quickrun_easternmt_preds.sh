year_start=1984
year_end=1985
satellite="landsat5"

# Args:
out_base_path="./out/${satellite}/"
model_gs_path=gs://res-id/cnn/models/best_smp/ls8_model31_6band_quantized.ckpt

# mean_std_path=gs://res-id/cnn/training/prepped_gaip_landsat/mean_std_ls8_v3.npy
# bands_minmax_path=gs://res-id/cnn/training/prepped_gaip_landsat/landsat_all_imgs_bands_min_max_ls8_10m_v3.npy

# mean_std_path=gs://res-id/cnn/training/prepped_gaip_landsat/mean_std_ls7_v7.npy
# bands_minmax_path=gs://res-id/cnn/training/prepped_gaip_landsat/landsat_all_imgs_bands_min_max_ls10m_ls7_v6.npy

mean_std_path=gs://res-id/cnn/training/prepped_gaip_landsat/mean_std_ls5_v2.npy
bands_minmax_path=gs://res-id/cnn/training/prepped_gaip_landsat/landsat_all_imgs_bands_min_max_ls10m_ls5_v2.npy

# Careful! Cannot include trailing '/' in data_path
# data_path=gs://res-id/ee_exports/landsat8_30m_v2/2020
# data_path=gs://res-id/ee_exports/landsat7_30m_v2/2000
data_path_base="gs://res-id/ee_exports/interview_area_only/${satellite}_30m"
region_shp_path='gs://res-id/cnn/aux_data/brazil_munis_interviews_aea_10kmbuffer.*'


# Get data files
# gsutil cp $model_gs_path ./model.ckpt
# gsutil cp $mean_std_path ./mean_std.npy
# gsutil cp $bands_minmax_path ./bands_minmax.npy
# mkdir -p shp
# gsutil cp $region_shp_path ./shp/

# Prep vrt
for y in {$year_start..$year_end};
do

    out_path=${out_base_path}/$y/
    mkdir -p $out_path
    echo "${data_path_base}/${y}/*.tif"
    gsutil ls "${data_path_base}/${y}/*.tif" > filelist.txt
    sed -i 's!gs://!/vsigs/!' filelist.txt

    # For Landsat8:
    # gdalbuildvrt ls_10m.vrt -r cubic -tap -tr 0.000089831528412 0.000089831528412 -input_file_list filelist.txt -b 2 -b 3 -b 4 -b 5 -b 6 -b 7
    # For Landsat7 and 5:
    gdalbuildvrt ls_10m.vrt -r cubic -tap -tr 0.000089831528412 0.000089831528412 -input_file_list filelist.txt -b 1 -b 2 -b 3 -b 4 -b 5 -b 6

    # Run
    echo tsp python3 -u predict_smp_landsat.py ls_10m.vrt model.ckpt mean_std.npy bands_minmax.npy $out_path --quantized --region_shp shp/brazil_munis_interviews_aea_10kmbuffer.shp
done

# Prep logs/shutdown
# mkdir -p logs
# tsp bash tsp_wrapup.sh
