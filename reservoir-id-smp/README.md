# reservoir-id-smp: Code

For each stage, see the README inside the directory for more information.

## 1. ./ee_export

Readme includes a link to export script on Google Earth Engine


## 2 ./annotation_prep

Code for extracting tiles from satellite mosaics for LabelBox annotation

- A. build_input_vrt.sh

- B. extract_tiles.py
Extracts random tiles from mosaic. Can control size and count of tiles.

- C. match_tiles.py
Given directory of tiles from "extract_tiles.py", extract matching tiles from other mosaics.

- D. prep_labelbox_csv.sh


## 3 ./preprocessing

Code for preparing annotated images for training

- prep_smp_dataset.ipynb
Prepare dataset for training


## 4 ./train

Includes notebook for training on Google Colab.

- train_smp_segmentation_unet.ipynb
Notebook for training. For running on Google Colab.


## 5 ./predict

Scripts for running prediction along with some helper modules

- build_input_vrts_sentinel.sh
Build full raster vrts for running predictions

- predict_smp_sentinel.py
Script for running prediction

- models.py
Helper module that contains information about model structure

- dataset.py
Helper module that contains information about data for prediction.
