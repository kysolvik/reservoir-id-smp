# Prediction

### A. Build vrts

```
bash build_input_vrts_sentinel.sh
```

### B. Run predictions

```
python3 predict_smp_sentinel.py ./s2_10m.vrt ./models/manet_resnet_model24.ckpt ./mean_stds/mean_std_sentinel_v7.npy ./bands_minmax/all_imgs_bands_min_max_sentinel_v6.npy out_test/
```
