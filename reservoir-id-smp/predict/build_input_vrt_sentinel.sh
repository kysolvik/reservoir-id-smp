output_file=$1

gsutil ls gs://res-id/ee_exports/sentinel2_10m/*.tif > temp_filelist_s2_10m.txt
gsutil ls gs://res-id/ee_exports/sentinel2_20m/*.tif > temp_filelist_s2_20m.txt
gsutil ls gs://res-id/ee_exports/sentinel1_10m_v2/*.tif > temp_filelist_s1_10m.txt
sed -i 's!gs://!/vsigs/!' temp_filelist_s2_10m.txt
sed -i 's!gs://!/vsigs/!' temp_filelist_s2_20m.txt
sed -i 's!gs://!/vsigs/!' temp_filelist_s1_10m.txt
gdalbuildvrt s2_10m.vrt -r nearest -tap -tr 0.000089831528412 0.000089831528412 -input_file_list temp_filelist_s2_10m.txt
gdalbuildvrt s2_20m.vrt -r nearest -tap -tr 0.000089831528412 0.000089831528412 -input_file_list temp_filelist_s2_20m.txt
gdalbuildvrt s1_10m.vrt -r cubic -tap -tr 0.000089831528412 0.000089831528412 -input_file_list temp_filelist_s1_10m.txt
