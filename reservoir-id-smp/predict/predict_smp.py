#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predict reservoirs on Sentinel 10m data using Segmentation Models Pytorch

This script creates VRT, subsets a it, and predicts on the subsets
then optional merges them back into a full image.

Example:
    $ python3 predict_smp.py ./input_dir/ ./model.ckpt mean_std.npy ./temp_tile_dir/
"""

import argparse
import gc
import glob
import os
import re

import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
import pytorch_lightning as pl
import albumentations as albu
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import affine
import torch
import segmentation_models_pytorch as smp
import torchvision

from dataset import ResDataset
from model import ResModel

TILE_ROWS = 640
TILE_COLS = 640

OUT_ROWS = 500
OUT_COLS = 500

OVERLAP = 140

NBANDS_ORIGINAL = 12

BATCH_SIZE = 16

BAND_SELECTION = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15]


def argparse_init():
    """Prepare ArgumentParser for inputs"""

    p = argparse.ArgumentParser(
            description='Predict reservoirs from directory of sentinel tiles',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('source_path',
                   help='Path to main s2 vrt',
                   type=str)
    p.add_argument('model_checkpoint',
                   help='Pytorch Lightning model checkpoint file',
                   type=str)
    p.add_argument('mean_std_file',
                   help='.npy file containing train mean and std for scaling',
                   type=str)
    p.add_argument('bands_minmax_file',
                   help='.npy file containing band min and maxes scaling',
                   type=str)
    p.add_argument('--region_shp',
                   help='Shapefile to predict within. Skips tiles outside',
                   type=str,
                   default=None)
    p.add_argument('out_dir',
                   help='Output directory for predicted subsets',
                   type=str)

    return p


def normalized_diff(ar1, ar2):
    """Returns normalized difference of two arrays."""

    return np.nan_to_num(((ar1 - ar2) / (ar1 + ar2)), 0)


def calc_nd(imgs, band1, band2):
    """Add band containing NDWI."""

    nd = normalized_diff(
            imgs[:, :, :, band1].astype('float64'),
            imgs[:, :, :, band2].astype('float64'))

    # Rescale to uint8
    nd = np.round(255.*(nd - (-1))/(1 - (-1)))
    if nd.max() > 255:
        print(nd.max())
        print('Error: overflow')

    return nd.astype(np.uint8)


def load_single_image(start_inds, src_list):
    """Load single tile from list of src"""
    row, col = start_inds[0], start_inds[1]
    og_img_list = []
    # First check if valid against for image
    img_src = src_list[0]

    base_img = img_src.read(window=((row, row + TILE_ROWS),
                                    (col, col + TILE_COLS)))
    if np.min(base_img) > 0:
        og_img_list += [base_img]
        for img_src in src_list[1:]:
            og_img_list += [img_src.read(window=((row, row + TILE_ROWS),
                                                 (col, col + TILE_COLS)))]
        out_tuple = (np.vstack(og_img_list), True)
    else:
        out_tuple = (None, False)

    return out_tuple


def load_image_batch(start_inds, batch_start_point, src_list):
    """LOad full batch of images based on BATmCH_SIZE"""
    imgs = []
    img_count = 0
    i = batch_start_point
    invalid_list = []
    valid_list = []
    while img_count < BATCH_SIZE and i < start_inds.shape[0]:
        new_img, valid_flag = load_single_image(start_inds[i], src_list)
        if valid_flag:
            imgs.append(new_img)
            valid_list.append(start_inds[i])
            img_count += 1
        else:
            invalid_list.append(start_inds[i])
        i += 1

    # Append invalid list to file
    with open('invalid_indices.txt', 'a') as f:
        for ind in invalid_list:
            f.write("{},{}\n".format(ind[0], ind[1]))
    print('Bypassed {} invalid images'.format(len(invalid_list)))
    print('Predicting batch of {}'.format(img_count))

    # Moveaxis is used to move bands to last axis
    return np.moveaxis(np.stack(imgs), 1, 3), np.asarray(valid_list), i


def write_imgs(preds, out_dims, out_dir, batch_indices, src_img):
    """Write a batch of predictions to tiffs"""
    preds[preds >= 0.5] = 255
    preds[preds < 0.5] = 0
    for i in range(batch_indices.shape[0]):
        outfile = '{}/pred_{}-{}.tif'.format(
            out_dir, batch_indices[i, 0], batch_indices[i, 1])

        new_dataset = rasterio.open(
            outfile, 'w', driver='GTiff',
            height=out_dims[1], width=out_dims[0],
            count=1, dtype='uint8', compress='lzw',
            crs=src_img.crs, nodata=0,
            transform=get_geotransform((batch_indices[i, 1],
                                        batch_indices[i, 0]),
                                       src_img.transform,
                                       overlap=OVERLAP)
        )
        pred = preds[i]
        new_dataset.write(pred.astype('uint8'), 1)


def get_geotransform(indice_pair, src_transform, overlap):
    """Calculate geotransform of a tile.

    Notes:
        Using .affine instead of .transform because it should work with all
        rasterio > 0.9. See https://github.com/mapbox/rasterio/issues/86.

    Args:
        indice_pair (tuple): Row, Col indices of upper left corner of tile.
        src_transform (tuple): Geo transform/affine of src image

    """
    if overlap > 0:
        indice_pair = (indice_pair[0]+(overlap/2),
                       indice_pair[1]+(overlap/2))
    new_ul = [src_transform[2] + indice_pair[0]*src_transform[0] + indice_pair[1]*src_transform[1],
              src_transform[5] + indice_pair[0]*src_transform[3] + indice_pair[1]*src_transform[4]]

    new_affine = affine.Affine(src_transform[0], src_transform[1], new_ul[0],
                               src_transform[3], src_transform[4], new_ul[1])

    return new_affine


def calc_all_nds(imgs):
    """Calc normalized diff indices"""

    nd_list = []

#     # Add  Gao NDWI
#     nd_list += [calc_nd(imgs, 2, 3)]
#     # Add  MNDWI
#     nd_list += [calc_nd(imgs, 0, 3)]
#     # Add McFeeters NDWI band
#     nd_list += [calc_nd(imgs, 0, 2)]
#     # Add NDVI band
#     nd_list += [calc_nd(imgs, 2, 1)]
    
    # Add  Gao NDWI
    nd_list += [calc_nd(imgs, 3, 11)]
    # Add  MNDWI
    nd_list += [calc_nd(imgs, 1, 11)]
    # Add McFeeters NDWI band
    nd_list += [calc_nd(imgs, 1, 3)]
    # Add NDVI band
    nd_list += [calc_nd(imgs, 3, 2)]
    return np.stack(nd_list, axis=3)


def rescale_to_minmax_uint8(imgs, bands_minmax):
    """Rescales images to 0-255 based on (precalculated) min/maxes of bands"""
    imgs = np.where(imgs > bands_minmax[1], bands_minmax[1], imgs)
    imgs = (255. * (imgs.astype('float64') - bands_minmax[0]) / (bands_minmax[1] - bands_minmax[0]))
    imgs = np.round(imgs)
    if imgs.max() > 255:
        print(imgs.max())
        print('Error: overflow')
    return imgs.astype(np.uint8)


def preprocess(imgs, bands_minmax, mean_std, band_selection):
    """Prep the input images"""
    imgs = rescale_to_minmax_uint8(imgs, bands_minmax)
    nds = calc_all_nds(imgs)
    imgs = np.concatenate([imgs, nds], axis=3)[:, :, :, band_selection]
    imgs = (imgs - mean_std[0])/mean_std[1]

    return imgs


def predict(model, data_loader):
    """Apply model to prediction, convert to numpy array"""
    trainer = pl.Trainer()
    with torch.no_grad():
        preds = np.vstack(trainer.predict(model, data_loader))[:, 0, :, :]
    return preds


def load_model(checkpoint_path):
    """Load the model weights from checkpoint"""
    model = ResModel.load_from_checkpoint(
        checkpoint_path, in_channels=len(BAND_SELECTION), out_classes=1, arch='MAnet',
        encoder_name='resnet34', map_location=torch.device('cpu'))
    return model


def open_sources(source_path):
    """Open rasterio objects for all the source images"""
    s1_10m_path = source_path.replace('s2_10m', 's1_10m')
    s2_20m_path = source_path.replace('s2_10m', 's2_20m')
    src_list = [
        rasterio.open(source_path),
        rasterio.open(s1_10m_path),
        rasterio.open(s2_20m_path)]

    return src_list


def get_done_list(out_dir):
    file_list = glob.glob(os.path.join(out_dir, 'pred_*.tif'))
    ind_strs = [re.findall(r'[0-9]+', f) for f in file_list]
    done_indices = np.asarray(ind_strs).astype(int)

    # Check for invalid ones (not in bounds)
    if os.path.isfile('invalid_indices.txt'):

        invalid_df = pd.read_csv('./invalid_indices.txt',
                                 header=None, names=['x', 'y'])
        invalid_list = np.array([invalid_df['x'].values,
                                 invalid_df['y'].values]
                                ).T
        if done_indices.shape[0] > 0:
            done_indices = np.vstack([done_indices, invalid_list])
        else:
            done_indices = np.vstack([invalid_list])

    return np.ascontiguousarray(done_indices)


def geofilter_indices(src_transform, start_ind, region_gpd):
    """Eliminate indices outside the prediction region"""
    center_points = [(src_transform[2]
                      + (start_ind[:, 0]+(TILE_ROWS/2))*src_transform[0]
                      + (start_ind[:, 1]+(TILE_COLS/2))*src_transform[1]),
                     (src_transform[5]
                      + (start_ind[:, 0]+(TILE_ROWS/2))*src_transform[3]
                      + (start_ind[:, 1]+(TILE_COLS/2))*src_transform[4])
                     ]
    cp_df = pd.DataFrame({
        'x': center_points[0],
        'y': center_points[1],
        })
    cp_gdf = gpd.GeoDataFrame(
            cp_df, geometry=gpd.points_from_xy(cp_df.x, cp_df.y),
            crs="EPSG:4326"
            )
    cp_gdf['i'] = np.arange(cp_gdf.shape[0])

    cp_gdf_filt = cp_gdf.sjoin(region_gpd)

    # Invalid indices:
    invalid_inds_geofilt = start_ind[~cp_gdf['i'].isin(cp_gdf_filt['i'])]
    with open('invalid_indices.txt', 'a') as f:
        for ind in invalid_inds_geofilt:
            f.write("{},{}\n".format(ind[0], ind[1]))

    # Write out geopandas shapes
    cp_gdf_invalid = cp_gdf[~cp_gdf['i'].isin(cp_gdf_filt['i'])]
    cp_gdf_invalid.to_file('./gpd_invalid.geojson', driver='GeoJSON')
    cp_gdf_filt.to_file('./gpd_valid.geojson', driver='GeoJSON')

    return start_ind[cp_gdf_filt['i']]


def get_indices(src, done_ind, region_gpd=None):
    """Get the indices for the tiles in the larger vrt"""

    total_rows, total_cols = src.height, src.width

    row_starts = np.arange(0, total_rows - TILE_ROWS, TILE_ROWS - OVERLAP)
    col_starts = np.arange(0, total_cols - TILE_COLS, TILE_COLS - OVERLAP)

    # Add final indices to row_starts and col_starts
    row_starts = np.append(row_starts, total_rows - TILE_ROWS)
    col_starts = np.append(col_starts, total_cols - TILE_COLS)

    # Create Nx2 array with row/col start indices
    start_ind = np.array(np.meshgrid(row_starts, col_starts)).T.reshape(-1, 2)
    print('Num of tiles before any filter:', start_ind.shape[0])

    # Eliminate already predicted indices
    if done_ind.shape[0] > 0:
        print(start_ind.flags['F_CONTIGUOUS'])
        print(done_ind.flags['F_CONTIGUOUS'])
        start_in_done = np.in1d(start_ind.astype('int64').view('int64, int64'),
                               done_ind.astype('int64').view('int64, int64'))
        start_ind = start_ind[~start_in_done]
    print('Num of tiles after done_ind removed:', start_ind.shape[0])

    # Filter to region
    if region_gpd is not None:
        start_ind = geofilter_indices(src.transform, start_ind, region_gpd)
        print('Num of tiles after geofilter:', start_ind.shape[0])

    return start_ind


def to_tensor(x, **kwargs):
    """Tensor needs reordered indices"""
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing():
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def write_pred_tiles(input_imgs, out_dims, out_dir, batch_indices, src_img):
    """Write a batch of input tiles to tiffs"""
    print(input_imgs.shape)
    for i in range(batch_indices.shape[0]):
        outfile = '{}/in_{}-{}.tif'.format(
            out_dir, batch_indices[i, 0], batch_indices[i, 1])

        new_dataset = rasterio.open(
            outfile, 'w', driver='GTiff',
            height=out_dims[1], width=out_dims[0],
            count=input_imgs.shape[-1], dtype='float64', compress='lzw',
            crs=src_img.crs, nodata=0,
            transform=get_geotransform((batch_indices[i, 1],
                                        batch_indices[i, 0]),
                                       src_img.transform,
                                       overlap=OVERLAP)
        )
        img = input_imgs[i]
        for i in range(img.shape[-1]):
            new_dataset.write(img[:,:,i], i+1)


def main():
    """Main function that runs the script"""

    parser = argparse_init()
    args = parser.parse_args()

    # Open filehandles of rasters
    src_list = open_sources(args.source_path)

    # Open region shapefile to predict within
    if args.region_shp is not None:
        region_gpd = gpd.read_file(args.region_shp)
    else:
        region_gpd = None

    # Get lists of indices to run
    done_ind = get_done_list(args.out_dir)
    start_ind = get_indices(src_list[0], done_ind, region_gpd)

    # Calculate mins and maxes for scaling bands
    bands_minmax_all = np.load(args.bands_minmax_file)
    bands_minmax = np.array([np.min(bands_minmax_all[0], axis=0),
                             np.percentile(bands_minmax_all[1], 80, axis=0)])
    mean_std = np.load(args.mean_std_file)

    # Load model
    model = load_model(args.model_checkpoint)

    batch_start_point = 0
    print(start_ind.shape[0])
    while batch_start_point < start_ind.shape[0]:
        print('Starting loading batch {}'.format(batch_start_point))
        imgs, batch_indices, batch_start_point = load_image_batch(
                start_ind, batch_start_point, src_list)
        print('Loaded')
        imgs = preprocess(imgs, bands_minmax, mean_std, BAND_SELECTION)
#        write_pred_tiles(imgs, [640, 640], './pred_tiles/', batch_indices, src_list[0])
        print('Prepped')
        ds = ResDataset(imgs, preprocessing=get_preprocessing(), mean_std=mean_std)
        dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=1)
        out_imgs = predict(model, dl)
        print(out_imgs.max())
        print('Predicted')
        write_imgs(out_imgs, [OUT_ROWS, OUT_COLS], args.out_dir,
                   batch_indices, src_list[0])
        print('Written')
        gc.collect()

    print('Done with batch, starting new from {}'.format(batch_start_point))


if __name__ == '__main__':
    main()

