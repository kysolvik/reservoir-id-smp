#!/usr/bjn/env python3
# -*- coding: utf-8 -*-
"""Download images and masks from labelbox and GCS"""


from google.cloud import storage
import pandas as pd
import subprocess as sp
import os
import urllib.request
import numpy as np
from PIL import Image
from skimage import io
from skimage import morphology
import random
import argparse
import augment_data as augment
import glob
import requests
from io import BytesIO

INPUT_SIZE = 640

def argparse_init():
    """Prepare ArgumentParser for inputs."""

    p = argparse.ArgumentParser(
            description='Prepare images and masks for training and testing.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('labelbox_json',
                   help='Path to LabelBox exported JSON.',
                   type=str)
    return p


def numpy_array_from_mask_url(url):
    """Given LB Mask URL, load into numpy array"""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.array(img)

def merge_object_masks(image_urls, dim_x, dim_y):
    """Merge Labelbox's separate masks for same object type, same image"""
    if len(image_urls)>0:
        all_masks = np.stack([numpy_array_from_mask_url(url) for url in image_urls], axis=-1)
        print(all_masks.shape)
        mask_array = np.max(all_masks, axis=-1)
        mask_array = mask_array[:,:,0]
        print(mask_array.shape)
    else:
        mask_array = np.zeros([dim_x,dim_y], dtype=np.uint8)


    return mask_array

def find_ims_masks(labelbox_json):
    """Finds locations of images and masks from labelbox-generated json."""

    label_df = pd.read_json(labelbox_json, orient='records')

    # Get masks lists for each label
    label_df = label_df.loc[~label_df['Skipped']]
    label_df['Masks'] = [[j['instanceURI'] for j in i['objects']] if
                         (len(i) >0 and len(i['objects'])>0) else [] for i in label_df['Label'].values]
    label_df = label_df.loc[label_df['Masks'].notna()]

    # URLs for original images
    og_urls = label_df['Labeled Data'].replace('ndwi.png', 'og.tif', regex=True)


    # URLS for
    s1_urls = label_df['Labeled Data'].replace(
        'ndwi.png', 's1_v2_og.tif', regex=True)
    s2_20m_urls = label_df['Labeled Data'].replace(
        'ndwi.png', 's2_20m_og.tif', regex=True)

    # URLs a for image masks
    mask_urls = label_df['Masks'].values

    og_mask_tuples = zip(og_urls, s1_urls, s2_20m_urls, mask_urls)

    # Find the bucket name
    sample_og_url = og_urls[0]
    og_gs_path = sample_og_url.replace('https://storage.googleapis.com/', '')
    gs_bucket_name = og_gs_path.split('/')[0]

    return og_mask_tuples, gs_bucket_name


def download_ims_mask_pair(og_urls, mask_urls, gs_bucket,
                          destination_dir='./data/ab/', dim_x=500, dim_y=500):
    """Downloads original image and mask, renaming mask to match image."""

    name_mask = True
    for im_url in og_urls:
        # Download og file from google cloud storage using gsutil
        og_dest_file = '{}/{}'.format(destination_dir, os.path.basename(im_url))
        og_gs_path = im_url.replace('https://storage.googleapis.com/{}/'
                                 .format(gs_bucket.name), '')
        if INPUT_SIZE !=500:
            og_gs_path = og_gs_path.replace(
                '500x500', '{}x{}'.format(INPUT_SIZE, INPUT_SIZE))
            print(og_gs_path)
        og_dest_file = og_dest_file.replace('im_', 'im_ab_')
        print(og_dest_file)
        blob = gs_bucket.blob(og_gs_path)
        blob.download_to_filename(og_dest_file)

        # Using first og url (sentinel 2 10m bands) name local mask file
        if name_mask:
            mask_dest_file = og_dest_file.replace('og.tif', 'mask.png')
            print(mask_dest_file)
            name_mask = False

    mask_array = merge_object_masks(mask_urls, dim_x, dim_y)
    io.imsave(mask_dest_file, mask_array)

    return None



def main():
    # Get command line args
    parser = argparse_init()
    args = parser.parse_args()

    og_mask_tuples, gs_bucket_name = find_ims_masks(args.labelbox_json)

    # Download imgs using Google Cloud Storage client
    storage_client = storage.Client()
    gs_bucket = storage_client.get_bucket(gs_bucket_name)
    for og_mask_pair in og_mask_tuples:
        download_ims_mask_pair(og_mask_pair[0:3], og_mask_pair[3],
                                gs_bucket)

    return


if __name__=='__main__':
    main()
