#!/bin/bash
###
# Script for creating a csv for labelbox upload
###

# Set google cloud storage path for putting files
gs_dir=$1

# Directory containing tiles
tile_dir=$2  

# Copy files to gs
gsutil -m cp "${tile_dir}/*ndwi.png" $gs_dir

# Set permissions
gsutil iam ch -r allUsers:legacyObjectReader $gs_dir
gsutil acl -r ch -u AllUsers:R $gs_dir

# Create csv
echo 'Image URL' > ./labelbox_urls.csv
gsutil ls $gs_dir >> ./labelbox_urls.csv
sed -i -e "s@gs://@https://storage.googleapis.com/@g" ./labelbox_urls.csv
