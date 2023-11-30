# Example Usage:

### A. Create a vrt from a Google Cloud Storage directory:

Note: This requires setting up a Google access key and putting it in ~/.boto

See: https://cloud.google.com/storage/docs/boto-gsutil
https://stackoverflow.com/questions/60540854/how-to-get-gs-secret-access-key-and-gs-access-key-id-in-google-cloud-storage

```
bash build_input_vrt.sh gs://res-id/ee_exports/sentinel2_10m/ s2_10m.vrt nearest
```

The last argument ('nearest' in this case) is the resampling method. It doesn't matter much for rasters that are already at the resolution we're going to be doing our analysis in, but for rasters with courser rresolution (e.g. Landsat or 20m Sentinel bands) cubic can be used (see below)

### B. Extract 10 tiles that are 500x500 with 70 padding (640x640 input images):

*Note: Tiles that fall outside the boundaries of brazil (null-value) will not be saved, so the actual number may be smaller than the specified number*

```
mkdir -p out/s2_10m/
python3 extract_tiles.py s2_10m.vrt 10 500 500 70 70 ./out/s2_10m/ --out_prefix='eg_tile_'
```

Be sure to create the ./out directory first

### C. Match those tiles, extracting from a different image (e.g. Sentinel 1)

```
# Sentinel1 10m
bash build_input_vrt.sh gs://res-id/ee_exports/sentinel1_10m_v2/ s1_10m.vrt nearest
mkdir -p out/s1_10m/
python3 match_tiles.py ./out/s2_10m/grid_indices.csv s1_10m.vrt ./out/s1_10m/ s1_10m 

# Sentinel2 20m bands
bash build_input_vrt.sh gs://res-id/ee_exports/sentinel2_20m/ s2_20m.vrt cubic
mkdir -p out/s2_20m/
python3 match_tiles.py ./out/s2_10m/grid_indices.csv s2_20m.vrt ./out/s2_20m/ s2_20m 
```

### D. Upload files to google cloud storage for labelbox
```
bash prep_labelbox_csv.sh gs://res-id/labelbox_tiles/sentinel_batch1/ ./out/s2_10m/
```
