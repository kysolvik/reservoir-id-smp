#!/usr/bin/env python3
""" Quick script to get counts and distribution of reservoir areas'"""

from osgeo import gdal
import numpy as np
import sys
from scipy import ndimage, stats

tif = sys.argv[1]
region_tif = sys.argv[2]
properties_tif = sys.argv[3]
out_txt = sys.argv[4]
box_size = 200000

fh = gdal.Open(tif)
region_fh = gdal.Open(region_tif)
property_fh = gdal.Open(properties_tif)

def calc_mode(vals):
    mode_result = stats.mode(vals)
    return mode_result.mode

def get_labels_count(ar):
    mask = ar == 1
    # Get count
    label_im, nb_labels = ndimage.label(mask,
                                    structure = [[1,1,1],[1,1,1],[1,1,1]])
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
    property_mode = ndimage.labeled_comprehension(property_ar, label_im, label_values, calc_mode, int, 0)
    property_all = ndimage.labeled_comprehension(property_ar, label_im, label_values, np.unique, np.ndarray, [0])
    return property_mode, property_all


total_rows, total_cols = fh.RasterYSize, fh.RasterXSize
current_row = 0
current_col = 0
row_starts = np.arange(0, total_rows, box_size)
col_starts = np.arange(0, total_cols, box_size)

# Create Nx2 array with row/col start indices
start_ind = np.array(np.meshgrid(row_starts, col_starts)).T.reshape(-1, 2)

with open(out_txt, 'w') as f:
    f.write('area,reg,property_mode,property_all\n')
for i in range(start_ind.shape[0]):
    # For the indices near edge we need to use a smaller box size
    box_size_rows = min(total_rows - start_ind[i,0], box_size)
    box_size_cols = min(total_cols - start_ind[i,1], box_size)

    # Get base counts and labeled image
    ar = fh.GetRasterBand(1).ReadAsArray(
        int(start_ind[i, 1]), int(start_ind[i, 0]),
        int(box_size_cols), int(box_size_rows))
    if (ar == 1).sum() > 0:
        sizes, label_im, label_values = get_labels_count(ar)

        # Get region stats
        reg_ar = region_fh.GetRasterBand(1).ReadAsArray(
            int(start_ind[i, 1]), int(start_ind[i, 0]),
            int(box_size_cols), int(box_size_rows))
        if (reg_ar != 255).sum() > 0:
            regions = get_region_stats(label_im, label_values, reg_ar)

            # Get property stats
            prop_ar = property_fh.GetRasterBand(1).ReadAsArray(
                int(start_ind[i, 1]), int(start_ind[i, 0]),
                int(box_size_cols), int(box_size_rows))
            property_mode, property_all = get_property_stats(label_im, label_values, prop_ar)

            print('Stats done')

            with open(out_txt, 'a') as f:
                for i in range(len(sizes)):
                    f.write("{},{},{},{}\n".format(int(sizes[i]), regions[i], property_mode[i], property_all[i]))
