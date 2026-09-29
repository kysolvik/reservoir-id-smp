echo "---1985 ls5 extract---"
# python3 make_shps.py ../clean_summarize/out/v3_cloudfilt_cleaned/ls5_1985_reservoirs.csv \
#     x_aea y_aea in/shps_all_brazil/
# python3 raster-buffer-extract/raster-buffer-extract/fraster_extract_wrapper.py \
#     ./in/shps_all_brazil/ls5_1985_reservoirs.shp \
#     ./in/brazil/brazil_coverage-col11_1985_aea_30m.tif out/lulc_stats_res_ls5_1985.csv 1000 \
#     --stat count_dict --not_latlon --nsample 10000
# python3 process_single_lulc_csv.py out/lulc_stats_res_ls5_1985.csv out/lulc_stats_res_ls5_1985_processed.csv

echo "---2025 ls8 extract---"
# python3 make_shps.py ../clean_summarize/out/v3_cloudfilt_cleaned/ls8_2025_reservoirs.csv \
#     x_aea y_aea in/shps_all_brazil/
python3 raster-buffer-extract/raster-buffer-extract/fraster_extract_wrapper.py \
    ./in/shps_all_brazil/ls8_2025_reservoirs.shp \
    ./in/brazil/brazil_coverage-col11_2025_aea_30m.tif out/lulc_stats_res_ls8_2025.csv 1000 \
    --stat count_dict --not_latlon --nsample 10000
python3 process_single_lulc_csv.py out/lulc_stats_res_ls8_2025.csv out/lulc_stats_res_ls8_2025_processed.csv
