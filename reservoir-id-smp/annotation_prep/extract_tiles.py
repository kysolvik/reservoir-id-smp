#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract subset images for annotation

This script extracts subset images from a large geoTiff. These images can then
be annotated to create training/test data for the CNN.

Example:
    Create 5 10x10 sub-images of raster 'eg.tif' with 1 padding in each dir
    $ python3 extract_tiles.py eg.tif 5 10 10 1 1 out/ --out_prefix='eg_sub_'

Notes:
    In order to work with Labelbox, the images must be exported as png or jpg.
"""


import os.path
import argparse
import numpy as np
import pandas as pd
from skimage import io
import rasterio


# Set seed
np.random.seed(50)


def argparse_init():
    """Prepare ArgumentParser for inputs"""

    p = argparse.ArgumentParser(
            description='Extract subest images from larger raster/image.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('source_path',
                   help='Path to raw input image',
                   type=str)
    p.add_argument('num_subsets',
                   help='Number of subsets to create',
                   type=int)
    p.add_argument('subset_dim_x',
                   help='Subset image X dimension in # pixels',
                   type=int)
    p.add_argument('subset_dim_y',
                   help='Subset image Y dimension in # pixels',
                   type=int)
    p.add_argument('padding_x',
                   help='X direction Padding for image (if bigger than mask',
                   type=int)
    p.add_argument('padding_y',
                   help='Y direction Padding for image (if bigger than mask',
                   type=int)
    p.add_argument('out_dir',
                   help='Output directory for subset images',
                   type=str)
    p.add_argument('--out_prefix',
                   help='Prefix for output tiffs',
                   default='image_',
                   type=str)

    return p


def write_append_csv(df, csv_path):
    """Check if csv already exists. Append if it does, write w/ header if not"""

    if not os.path.isfile(csv_path):
        df.to_csv(csv_path, header=True, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)


def scale_image_tobyte(ar):
    """Scale larger data type array to byte"""
    minVals = np.amin(np.amin(ar, 1), 0)
    maxVals = np.amax(np.amax(ar, 1), 0)
    byte_ar = np.round(255.0 * (ar - minVals) / (maxVals - minVals)) \
        .astype(np.uint8)
    byte_ar[ar == 0] = 0

    return byte_ar


def normalized_diff(ar1, ar2):
    """Returns normalized difference of two arrays."""

    # Convert arrays to float32
    ar1 = ar1.astype('float32')
    ar2 = ar2.astype('float32')

    return (ar1 - ar2) / (ar1 + ar2)


def create_gmaps_link(xmin_pix, ymin_pix, xmax_pix, ymax_pix, gt):
    """Create a Google Maps link to include in csv to help with annotation
    Link will zoom to center of subset image in Google Maps"""

    xmean_pix = (xmin_pix + xmax_pix)/2
    ymean_pix = (ymin_pix + ymax_pix)/2

    # Longitude, latitude of center
    center_coords = np.stack((gt[2] + xmean_pix*gt[1]+(ymean_pix*gt[0]),
                             gt[5] + xmean_pix*gt[4]+(ymean_pix*gt[3])),
                             axis=1)

    gmaps_links = ["https://www.google.com/maps/@{},{},5000m/data=!3m1!1e3"\
                   .format(coord[1], coord[0]) for coord in center_coords]

    return gmaps_links


def subset_image(fh, num_subsets, dim_x, dim_y, pad_x, pad_y,
                 out_dir, source_path, out_prefix, nodata=0):
    """Create num_subsets images of (dim_x, dim_y) size from og_im."""

    # Randomly select locations for sub-arrays
    sub_xmins = np.random.random_integers(0, fh.shape[0] - (dim_x + 2*pad_x + 1),
                                          num_subsets)
    sub_ymins = np.random.random_integers(0, fh.shape[1] - (dim_y + 2*pad_y + 1),
                                          num_subsets)

    # Get xmaxs and ymaxs
    sub_xmaxs = sub_xmins + dim_x + 2*pad_x
    sub_ymaxs = sub_ymins + dim_y + 2*pad_y

    # Geotransformation
    source_geotrans = fh.transform

    # Get Google maps link
    sub_gmaps_links = create_gmaps_link(sub_xmins, sub_ymins, sub_xmaxs,
                                        sub_ymaxs, source_geotrans)

    # Create and save csv containing grid coordinates for images
    grid_indices_df = pd.DataFrame({
        'name': ['{}{}_ndwi'.format(out_prefix, snum)
                 for snum in range(0, num_subsets)],
        'source': os.path.basename(source_path),
        'lon_min': sub_xmins,
        'lon_max': sub_xmaxs,
        'lat_min': sub_ymins,
        'lat_max': sub_ymaxs,
        'gmaps_link': sub_gmaps_links
        })

    # Save sub-arrays
    null_im_mask = np.ones(num_subsets, dtype=bool)
    for snum in range(0, num_subsets):
        # NDWI image, for annotating
        subset_ndwi_path = '{}/{}{}_ndwi.png'.format(out_dir, out_prefix, snum)

        read_window = ((sub_xmins[snum], sub_xmaxs[snum]),
                       (sub_ymins[snum], sub_ymaxs[snum]))
        sub_ndwi_im = np.moveaxis(fh.read(window=read_window), 0, -1)

        # Check image for no data
        if np.any(sub_ndwi_im == nodata):
            null_im_mask[snum] = False
            continue

        # Save ndwi image for labelbox
        sub_ndwi_im = normalized_diff(
                sub_ndwi_im[pad_x:-(pad_x), pad_y:-(pad_y), 1],
                sub_ndwi_im[pad_x:-(pad_x), pad_y:-(pad_y):, 3])
        sub_ndwi_im_byte = scale_image_tobyte(sub_ndwi_im)
        io.imsave(subset_ndwi_path, sub_ndwi_im_byte, plugin='pil')

        # Original image, for training
        subset_og_path = '{}/{}{}_og.tif'.format(out_dir, out_prefix, snum)
        sub_og_im = np.moveaxis(fh.read(window=read_window), 0, -1)

        io.imsave(subset_og_path, sub_og_im, plugin='tifffile')

    # Write grid indices to csv
    grid_indices_df = grid_indices_df.iloc[null_im_mask]
    write_append_csv(grid_indices_df, '{}/grid_indices.csv'.format(out_dir))

    return


def main():
    # Get command line args
    parser = argparse_init()
    args = parser.parse_args()

    # Read image
    fh = rasterio.open(args.source_path)

    # Get subsets
    subset_image(fh, args.num_subsets,
                 args.subset_dim_x, args.subset_dim_y,
                 args.padding_x, args.padding_y,
                 args.out_dir, args.source_path, args.out_prefix)

    return


if __name__ == '__main__':
    main()
