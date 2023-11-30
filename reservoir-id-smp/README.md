# reservoir-id-smp: Code

For each stage, see the README inside the directory for more information.

## 1 ./annotation_prep

Code for extracting tiles from larger mosaic

- extract_tiles.py
Extracts random tiles from mosaic. Can control size and count of tiles.


- match_tiles.py
Given directory of tiles from "extract_tiles.py", extract matching tiles from other mosaics.


## 2 ./preprocessing

- prep_smp_dataset.ipynb
Prepare dataset for training

## 3 ./train

- train_smp_segmentation_unet.ipynb
Notebook for training. For running on Google Colab.

## 4 ./predict

- predict_smp.py
Script for running prediction

- models.py
Helper module that contains information about model structure

- dataset.py
Helper module that contains information about data for prediction.
