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
from neural_compressor.utils.pytorch import load as nc_load
from dataset import ResDataset
from model import ResModel

TILE_ROWS = 640
TILE_COLS = 640

OUT_ROWS = 500
OUT_COLS = 500

OVERLAP = 140

BAND_SELECTION = [0, 1, 2, 3, 4, 5]

# NOTE: For LS8 should be 1-6. For LS7 and 5 should be 0-5
# MEAN_STD_BAND_SELECTION = [0, 1, 2, 3, 4, 5]
MEAN_STD_BAND_SELECTION = [1, 2, 3, 4, 5, 6]

def argparse_init():
    """Prepare ArgumentParser for inputs"""

    p = argparse.ArgumentParser(
            description='Predict reservoirs from vrt of landsat',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('source_path',
                   help='Path to main ls8 vrt',
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
    p.add_argument('--quantized',
                   help='Provide if providing quantized model .pt file.',
                   action='store_true',
                   default=False)
    p.add_argument('--calc_nds',
                   help='Calculate 4 normalized difference bands.',
                   action='store_true',
                   default=False)
    p.add_argument('out_dir',
                   help='Output directory for predicted subsets',
                   type=str)

    return p


def load_model(checkpoint_path, crs):
    """Load the model weights from checkpoint"""
    model = ResModel.load_from_checkpoint(
        checkpoint_path, in_channels=6, out_classes=1, arch='MAnet',
        encoder_name='resnet34', map_location=torch.device('cpu'), crs=crs)
    return model

def load_model_quantized(checkpoint_path, crs):
    """Load the model weights from checkpoint"""
    model = ResModel(arch='MAnet', encoder_name="resnet34",
                     in_channels=6, out_classes=1, weights=None, crs=crs)
    model.model = nc_load(checkpoint_path, model.model)
    return model


def open_sources(source_path):
    """Open rasterio objects for all the source images.

    Note:
        For Landsat, only one source image
    """
    src_list = [
        rasterio.open(source_path)]

    return src_list


def get_done_list(out_dir):
    file_list = glob.glob(os.path.join(out_dir, 'pred_*.tif'))
    ind_strs = [re.findall(r'[0-9]+', os.path.basename(f)) for f in file_list]
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
#    cp_gdf_invalid = cp_gdf[~cp_gdf['i'].isin(cp_gdf_filt['i'])]
#    cp_gdf_invalid.to_file('./gpd_invalid.geojson', driver='GeoJSON')
#    cp_gdf_filt.to_file('./gpd_valid.geojson', driver='GeoJSON')

    return start_ind[cp_gdf_filt['i']]


def get_indices(src, done_ind, region_gpd=None):
    """Get the indices for the tiles in the larger vrt"""

    total_cols, total_rows = src.height, src.width

    row_starts = np.arange(0, total_rows - TILE_ROWS, TILE_ROWS - OVERLAP)
    col_starts = np.arange(0, total_cols - TILE_COLS, TILE_COLS - OVERLAP)

    # Add final indices to row_starts and col_starts
    row_starts = np.append(row_starts, total_rows - TILE_ROWS)
    col_starts = np.append(col_starts, total_cols - TILE_COLS)

    # Create Nx2 array with row/col start indices
    start_ind = np.array(np.meshgrid(row_starts, col_starts)).T.reshape(-1, 2)
    print('Num of tiles before any filter:', start_ind.shape[0])

    # Filter to region
    if region_gpd is not None:
        start_ind = geofilter_indices(src.transform, start_ind, region_gpd)
        print('Num of tiles after geofilter:', start_ind.shape[0])

    # Eliminate already predicted indices
    if done_ind.shape[0] > 0:
        print(start_ind.flags['F_CONTIGUOUS'])
        print(done_ind.flags['F_CONTIGUOUS'])
        start_in_done = np.in1d(start_ind.astype('int64').view('int64, int64'),
                                done_ind.astype('int64').view('int64, int64'))
        start_ind = start_ind[~start_in_done]
    print('Num of tiles after done_ind removed:', start_ind.shape[0])

    return start_ind


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
    start_inds = get_indices(src_list[0], done_ind, region_gpd)

    # Calculate mins and maxes for scaling bands
    bands_minmax_all = np.load(args.bands_minmax_file)
    bands_minmax = np.array([np.min(bands_minmax_all[0], axis=0),
                             np.percentile(bands_minmax_all[1], 80, axis=0)])
    bands_minmax = bands_minmax[:, MEAN_STD_BAND_SELECTION]
    print(bands_minmax.shape)
    mean_std = np.load(args.mean_std_file)[:, MEAN_STD_BAND_SELECTION]
    print(mean_std.shape)

    # Load model
    if args.quantized:
        model = load_model_quantized(args.model_checkpoint, crs=src_list[0].crs)
    else:
        model = load_model(args.model_checkpoint, crs=src_list[0].crs)
    model.model.eval()

    # Create dataset and loader
    ds = ResDataset(start_inds, fh=src_list[0], mean_std=mean_std, bands_minmax=bands_minmax,
                    band_selection=BAND_SELECTION, add_nds=args.calc_nds, tile_rows=TILE_ROWS, tile_cols=TILE_COLS,
                    overlap=OVERLAP, out_dir=args.out_dir)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=1, collate_fn=ds.collate)

    trainer = pl.Trainer()
    with torch.no_grad():
        trainer.predict(model, dl)

if __name__ == '__main__':
    main()

