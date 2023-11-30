# Example Usage:

### Create a vrt from a Google Cloud Storage directory:

```
bash build_input_vrt.sh gs://res-id/ee_exports/sentinel2_10m/ s2_10m.vrt
```


### Extract 100 tiles that are 500x500 with 70 padding (640x640 input images):

```
mkdir -p out/s2_10m/
python3 extract_tiles.py s2_10m.vrt 100 500 500 70 70 ./out/s2_10m/ --out_prefix='eg_tile_'
```

Be sure to create the ./out directory first

### Match those 100 tiles, extracting from a different image

```
bash build_input_vrt.sh gs://res-id/ee_exports/sentinel1_10m/ s1_10m.vrt
mkdir -p out/s1_10m/
python3 match_tiles.py s1_10m.vrt ./out/s1_10m/
```

### Upload files to google cloud storage for labelbox
```
bash prep_labelbox_csv.sh gs://res-id/labelbox_tiles/sentinel_batch1/ ./out/s2_10m/
```
