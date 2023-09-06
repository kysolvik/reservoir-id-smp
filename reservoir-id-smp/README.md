# reservoir-id-smp: Code

## 1. ./annotation_prep

Code for extracting tiles from larger mosaic

- extract_tiles.py
Extracts random tiles from mosaic. Can control size and count of tiles.


- match_tiles.py
Given directory of tiles, extract matching tiles from other mosaics.


## 2. ./preprocessing

- prep_smp_dataset.ipynb

## 3. ./train

- train_smp_segmentation_unet.ipynb

Notebook for training. For running on Google Colab.

## 4. ./predict

- predict_smp.py

- models.py
Helper module that contains information about model structure

- dataset.py
Helper module that contains information about data
