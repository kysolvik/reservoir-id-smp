# Preprocess raw images and annotated masks for training

### A. Download images and masks for labelbox json

First, download JSON from labelbox (under "Export" in LabelBox project. Use v1)

python3 download_ims_masks.py labelbox.json

### B. Prep dataset

Open jupyter notebook:
```
jupyter notebook
```

And run prep_smp_dataset.ipynb

### C. Copy to cloud storage for training on Google Colab

```
bash zip_and_cp_to_gstorage.sh
```

