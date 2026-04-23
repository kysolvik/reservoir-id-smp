ls5_ls7_list=$(seq 2000 2011)
ls7_ls8_list=$(seq 2014 2025)
ls8_ls9_list=$(seq 2022 2025)

in_dir="../../predict/full_outputs/v3/"

# Ls5 ls7 first
for y in $ls5_ls7_list;
do
    python3 -u compare_full_maps.py ${in_dir}/ls5*$y*.tif ${in_dir}/ls7*${y}*.tif out/ls5ls7_${y}_compare.csv
done

# Ls7 ls8 
for y in $ls7_ls8_list;
do
    python3 -u compare_full_maps.py ${in_dir}/ls7*$y*.tif ${in_dir}/ls8*${y}*.tif out/ls7ls8_${y}_compare.csv
done

# Ls8 ls9 
for y in $ls8_ls9_list;
do
    python3 -u compare_full_maps.py ${in_dir}/ls8*$y*.tif ${in_dir}/ls9*${y}*.tif out/ls8ls9_${y}_compare.csv
done
