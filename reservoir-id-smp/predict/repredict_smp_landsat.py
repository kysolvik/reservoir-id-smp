#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repredict specified tiles using ee.data.computePixels() instead of full EE export

Example:
    $ python3 repredict_smp_landsat.py bad_tiles.csv 5 1985 ./model.ckpt mean_std.npy ./temp_tile_dir/

Note:
    bad_tiles.csv must contain columns: row_start, column_start
"""

import argparse
import glob
import os
import re

import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import affine
import torch
from neural_compressor.utils.pytorch import load as nc_load
from predict.dataset_ee import ResDatasetEE
from model import ResModel

TILE_ROWS = 640
TILE_COLS = 640

OUT_ROWS = 480
OUT_COLS = 480

OVERLAP = 160

BAND_SELECTION = [0, 1, 2, 3, 4, 5]

OUT_CRS = rasterio.CRS.from_wkt('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]')
SRC_TRANSFORM = affine.Affine(8.9831528412e-05, 0.0, -74.05396791935839, 0.0, -8.9831528412e-05, 5.31757732434834)

def argparse_init():
    """Prepare ArgumentParser for inputs"""

    p = argparse.ArgumentParser(
            description='Predict reservoirs from vrt of landsat',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('bad_tiles_csv',
                   help='Path to bad tiles csv',
                   type=str)
    p.add_argument('ls_number',
                   help='Landsat number (5, 7, 8, or 9)',
                   type=int)
    p.add_argument('year',
                   help='Year',
                   type=int)
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
    p.add_argument('--threshold',
                   help='Prediction threshold for binary class (0.0 - 1.0)',
                   default=None,
                   type=float)
    p.add_argument('out_dir',
                   help='Output directory for predicted subsets',
                   type=str)

    return p


def load_model(checkpoint_path, crs, threshold):
    """Load the model weights from checkpoint"""
    model = ResModel.load_from_checkpoint(
        checkpoint_path, in_channels=6, out_classes=1, arch='MAnet',
        encoder_name='resnet34', map_location=torch.device('cpu'), crs=crs,
        center_crop=OUT_ROWS, threshold=threshold, prob_scale=254)
    return model


def load_model_quantized(checkpoint_path, crs, threshold):
    """Load the model weights from checkpoint"""
    model = ResModel(arch='MAnet', encoder_name="resnet34",
                     in_channels=6, out_classes=1, weights=None, crs=crs,
                     center_crop=OUT_ROWS, threshold=threshold, prob_scale=254)
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
            # Remove last batch of predictions in case of shutdown
            sorted_ind = np.lexsort((done_indices[:,1],
                                     done_indices[:,0]))
            done_indices = done_indices[sorted_ind]

            done_indices = done_indices[:-4]
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

    return start_ind[cp_gdf_filt['i']]


def get_indices_from_csv(csv, done_ind, region_gpd=None):
    """Get the indices for the tiles in the larger vrt"""

    tile_df = pd.read_csv(csv)

    start_ind = np.array([tile_df['row'].values,tile_df['col'].values]).T

    # Create Nx2 array with row/col start indices
    print('Num of tiles before any filter:', start_ind.shape[0])

    # Filter to region
    if region_gpd is not None:
        start_ind = geofilter_indices(SRC_TRANSFORM, start_ind, region_gpd)
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

    ls_name = f'ls{args.ls_number}'

    # Open region shapefile to predict within
    if args.region_shp is not None:
        region_gpd = gpd.read_file(args.region_shp)
    else:
        region_gpd = None

    # Get lists of indices to run
    done_ind = get_done_list(args.out_dir)
    start_inds = get_indices_from_csv(args.bad_tiles_csv, done_ind, region_gpd)

    # Calculate mins and maxes for scaling bands
    bands_minmax_all = np.load(args.bands_minmax_file)
    bands_minmax = np.array([np.min(bands_minmax_all[0], axis=0),
                             np.percentile(bands_minmax_all[1], 80, axis=0)])
    bands_minmax = bands_minmax[:, BAND_SELECTION]
    mean_std = np.load(args.mean_std_file)[:, BAND_SELECTION]

    # Load model
    if args.quantized:
        model = load_model_quantized(args.model_checkpoint, crs=OUT_CRS,
                    threshold=args.threshold)
    else:
        model = load_model(args.model_checkpoint, crs=OUT_CRS,
                    threshold=args.threshold)
    model.model.eval()

    # Create dataset and loader
    ds = ResDatasetEE(start_inds, ls_name=ls_name, year=args.year, mean_std=mean_std, bands_minmax=bands_minmax,
                    band_selection=BAND_SELECTION, add_nds=args.calc_nds, tile_rows=TILE_ROWS, tile_cols=TILE_COLS,
                    offset=OVERLAP, out_dir=args.out_dir)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=1, collate_fn=ds.collate)

    trainer = pl.Trainer()
    with torch.no_grad():
        trainer.predict(model, dl)

if __name__ == '__main__':
    main()

