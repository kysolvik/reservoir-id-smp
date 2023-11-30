input_dir=$1
output_file=$2
gsutil ls $input_dir/*.tif > temp_filelist.txt
sed -i 's!gs://!/vsigs/!' temp_filelist.txt
gdalbuildvrt $output_file -tap -tr 0.000089831528412 0.000089831528412 -input_file_list temp_filelist.txt

