# Sentinel Workflow

Full workflow for recreating modeling and analysis from Solvik et al., "Uncovering a million small reservoirs in Brazil using deep learning".

If not mentioned, not relevant to Sentinel analysis.

## Code

For each stage, see the README inside the directory for more information.

### 1 - ee_export/

Readme includes publicly accessible link to export script on Google Earth Engine

### 2 - annotation_prep/

Code for extracting tiles from satellite mosaics for LabelBox annotation

- A. build_input_vrt.sh
Build vrt from exported tifs to extract random tiles.

- B. extract_tiles.py
Extracts random tiles from mosaic. Can control size and count of tiles.

- C. match_tiles.py
Given directory of tiles from "extract_tiles.py", extract matching tiles from other mosaics.

- D. prep_labelbox_csv.sh
Prepare labelbox upload for annotation.

### 3 - train/

- A. download_ims_masks.py
Download annotated masks from LabelBox and matching images from Google cloud storage.

- B. sentinel_prep.ipynb
Prepare Sentinel dataset for training, including train/valid/test/split

- C. sentinel_train.ipynb
Notebook for training. For running on Google Colab.

- D. sentinel_eval.ipynb
Eval model performance on validation or test set.

### 4 - predict/

Scripts for running prediction along with some helper modules

- bash_scripts/full_pred_sentinel*
- bash_scripts/postprocess_sentinel*
Wrapper scripts for predicting and post-processing.

- predict_smp_sentinel.py
Script for running prediction

- model.py
Helper module that contains information about model structure

- dataset.py
Helper module that contains information about data for prediction.

### 5 - analysis/

Lots of different analyses notebooks. Most are self-explanatory based on dir and file names, but clean_summarize needs more explanation:

- clean_summarize/
To clean reservoir outputs and get basic reservoir area statistics, run the following:

    - A. find_overlaps_badwater.py
    Run to identify overlapping reservoirs with ESRI/Garmin World Water Bodies dataset
    - B. merge_borders.py
    Combine reservoirs across borders of analys
    - C. write_out_new_rasters*
    Write out rasters with a size threshold and removing ones that overlap with ESRI dataset.
    - D. bash_scripts/buildvrt_tif.sh
    Combine tiles output by (C) into full tif.
    - E. bash_scripts/gdal_calc_combine_30m_10m.sh

To create the combined, 10m/30m map requires running these multiple times:
1. Run A-C separately on on both 30m and 10m outputs.
2. Run E to produce maximum value raster tiles.
3. Run D on maximum value raster tiles.
4. Run A-D again on the full maximum value raster.