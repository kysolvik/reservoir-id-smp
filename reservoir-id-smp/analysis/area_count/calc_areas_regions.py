#!/usr/bin/env python3
""" Quick script to get counts and distribution of reservoir areas'"""

import numpy as np
import sys
from scipy import ndimage, stats
import os
from PIL import Image
import rasterio as rio
from rasterio.windows import Window
import pandas as pd


tif = sys.argv[1]
region_tif = sys.argv[2]
properties_tif = sys.argv[3]
out_csv = sys.argv[4]
labeled_tif = tif.replace('.tif', '_labeled.tif')
COMPARE_TO_PREVIOUS = True
y = os.path.basename(tif)[9:13]
last_labeled_tif = labeled_tif.replace(y, str(int(y)-1))

box_size = 200000

fh = rio.open(tif)
region_fh = rio.open(region_tif)
property_fh = rio.open(properties_tif)


def convert_row_col_to_xy(affine_transformer, centers,
                          start_ind_row, start_ind_col):
    x, y = affine_transformer.xy(start_ind_row + centers[:, 0],
                                 start_ind_col + centers[:, 1])
    return x, y

def create_com_func(box_width):

    def calc_com(ar, pos):
        pos_row, pos_col = np.divmod(pos, box_width)

        return np.array([np.mean(pos_row), np.mean(pos_col)])

    return calc_com

def calc_mode(vals):
    mode_result = stats.mode(vals)
    return mode_result.mode

def get_labels_count(ar):
    mask = ar == 1
    # Get count
    label_im, nb_labels = ndimage.label(
            mask,
            structure=[[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])
    label_values = np.arange(1, nb_labels + 1)
    print('Count done')

    # Get sizes
    sizes = ndimage.labeled_comprehension(mask, label_im, label_values, np.sum, int, 0)
    return sizes, label_im, label_values

def get_region_stats(label_im, label_values, region_ar):
    # Attribute to region
    regions = ndimage.labeled_comprehension(region_ar, label_im, label_values, calc_mode, int, 0)

    return regions

def get_property_stats(label_im, label_values, property_ar):
    # Attribute to property
    property_mode = ndimage.labeled_comprehension(
            property_ar,
            label_im,
            label_values,
            calc_mode,
            int,
            0)
    property_all = ndimage.labeled_comprehension(
            property_ar,
            label_im,
            label_values,
            np.unique,
            np.ndarray,
            [0])
    return property_mode, property_all

def get_centers(label_im, label_values):
    box_width = label_im.shape[1]
    calc_com_func = create_com_func(box_width)
    centers = ndimage.labeled_comprehension(
            label_im>0,
            label_im,
            label_values,
            calc_com_func,
            np.ndarray,
            [0],
            pass_positions=True)
    return np.vstack(centers)


def compare_previous_year(label_im, label_values, last_label_im):
    previous_all = ndimage.labeled_comprehension(
            last_label_im,
            label_im,
            label_values,
            np.unique,
            np.ndarray,
            [0])
    return previous_all

total_rows, total_cols = fh.RasterYSize, fh.RasterXSize
current_row = 0
current_col = 0
row_starts = np.arange(0, total_rows, box_size)
col_starts = np.arange(0, total_cols, box_size)

# Create Nx2 array with row/col start indices
start_ind = np.array(np.meshgrid(row_starts, col_starts)).T.reshape(-1, 2)

# Get some affine information
cur_transform = fh.transform
affine_transformer = rio.transform.AffineTransformer(cur_transform)

# Set numpy print options so it doesn't abbreviate anything
np.set_printoptions(threshold=4*box_size)

for i in range(start_ind.shape[0]):
    # For the indices near edge we need to use a smaller box size
    box_size_rows = min(total_rows - start_ind[i, 0], box_size)
    box_size_cols = min(total_cols - start_ind[i, 1], box_size)
    start_ind_row, start_ind_col = (start_ind[i,0], start_ind[i,1])

    # Get base counts and labeled image
    read_window = Window(int(start_ind_col), int(start_ind_row),
                         int(box_size_cols), int(box_size_rows))
    ar = fh.read(1, window=read_window)
    if (ar == 1).sum() > 0:
        sizes, label_im, label_values = get_labels_count(ar)

        if COMPARE_TO_PREVIOUS:
            # Write label_image
            im = Image.fromarray(label_im.astype('uint16'))
            im.save(labeled_tif, compression='tiff_lzw')
            # If last labeled image exist, 
            if os.path.isfile(last_labeled_tif):
                prev_fh = rio.open(last_labeled_tif)
                prev_label_ar = prev_fh.read(1)
                previous_labels = compare_previous_year(
                        label_im, label_values, prev_label_ar)
            else:
                previous_labels = np.zeros_like(label_values)

        # Get region stats
        reg_ar = region_fh.read(1, window=read_window)
        if (reg_ar != 255).sum() > 0:
            regions = get_region_stats(label_im, label_values, reg_ar)

            # Get property stats
            prop_ar = property_fh.read(1, window=read_window)
            property_mode, property_all = get_property_stats(
                    label_im, label_values, prop_ar)

            # Centers of mass
            centers_of_mass = get_centers(label_im, label_values)
            centers_x, centers_y = convert_row_col_to_xy(
                    affine_transformer, centers_of_mass,
                    start_ind_row, start_ind_col)

            print('Stats done')

        if COMPARE_TO_PREVIOUS and write_labels:
            out_dict = {
                    'id': label_values,
                    'label': label_values,
                    'row_start': start_ind_row,
                    'col_start': start_ind_col,
                    'box_rows': box_size_rows,
                    'box_cols': box_size_cols,
                    'area': sizes,
                    'reg': regions,
                    'property_mode': property_mode,
                    'property_all': property_all,
                    'previous_labels': previous_labels
                    'center_col': centers_of_mass[:, 1],
                    'center_row': centers_of_mass[:, 0],
                    'center_y': centers_y,
                    'center_x': centers_y,
                    }
        else:
            out_dict = {
                    'id': label_values,
                    'label': label_values,
                    'row_start': start_ind_row,
                    'col_start': start_ind_col,
                    'box_rows': box_size_rows,
                    'box_cols': box_size_cols,
                    'area': sizes,
                    'reg': regions,
                    'property_mode': property_mode,
                    'property_all': property_all,
                    'center_col': centers_of_mass[:, 1],
                    'center_row': centers_of_mass[:, 0],
                    'center_y': centers_y,
                    'center_x': centers_y,
                    }
            out_df = pd.DataFrame(out_dict)

            out_df.to_csv(out_csv, mode='a',
                          header=(not os.path.isfile(out_csv)),
                          index=False)
