#!/usr/bin/env python3
""" Quick script to get counts and distribution of reservoir areas'"""

from osgeo import gdal
import numpy as np
import sys
from scipy import ndimage, stats
import cv2
# Contains a simple array for calculating overlapping borders
from _border_ar import calc_border_ar

tif = sys.argv[1]
hydropoly_tif = sys.argv[2]
out_txt = sys.argv[3]
box_size = 25000

fh = gdal.Open(tif)
hydropoly_fh = gdal.Open(hydropoly_tif)

def create_com_func(box_width):

    def calc_com(ar, pos):
        pos_row, pos_col = np.divmod(pos, box_width)

        return [np.mean(pos_row), np.mean(pos_col)]

    return calc_com

def get_labels_count(start_ind_col, start_ind_row,
                     box_size_cols, box_size_rows):
    mask = (fh.GetRasterBand(1).ReadAsArray(
        int(start_ind_col), int(start_ind_row),
        int(box_size_cols), int(box_size_rows)) == 1)

    if mask.sum() > 0:
        # Get count
        label_im, nb_labels = ndimage.label(mask,
                                        structure = [[1,1,1],[1,1,1],[1,1,1]])
        label_values = np.arange(1, nb_labels + 1)
        print('Count done')

        # Get sizes
        sizes = ndimage.labeled_comprehension(mask, label_im, label_values, np.sum, int, 0)
        return sizes, label_im, label_values
    else:
        return [0], [0], [0]

def get_border_minmax(label_im, label_values, border_ar):
    border_vals = ndimage.labeled_comprehension(
            border_ar,
            label_im,
            label_values,
            np.unique,
            np.ndarray,
            [0])

    return border_vals

def get_hydropoly_val(label_im, label_values, hydropoly_ar):
    hydropoly_max = ndimage.labeled_comprehension(hydropoly_ar, label_im, label_values, np.max, int, 0)
    return hydropoly_max

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
    return centers


total_rows, total_cols = fh.RasterYSize, fh.RasterXSize
current_row = 0
current_col = 0
row_starts = np.arange(0, total_rows, box_size)
col_starts = np.arange(0, total_cols, box_size)

# Create Nx2 array with row/col start indices
start_ind = np.array(np.meshgrid(row_starts, col_starts)).T.reshape(-1, 2)

with open(out_txt, 'w') as f:
    f.write('id,row_start,col_start,box_rows,box_cols,area,hydropoly,center_x,center_y,border_vals\n')
for i in range(start_ind.shape[0]):
    # For the indices near edge we need to use a smaller box size
    box_size_rows = min(total_rows - start_ind[i,0], box_size)
    box_size_cols = min(total_cols - start_ind[i,1], box_size)

    # Get base counts and labeled image
    start_ind_col, start_ind_row = (start_ind[i,1], start_ind[i,0])
    sizes, label_im, label_values = get_labels_count(
            start_ind_col, start_ind_row, box_size_cols, box_size_rows)
    if np.max(label_values) > 0:
        centers_of_mass = get_centers(label_im, label_values)
        border_ar = calc_border_ar(box_size_rows, box_size_cols)
        border_vals = get_border_minmax(label_im, label_values, border_ar)

        hydropoly_ar = hydropoly_fh.GetRasterBand(1).ReadAsArray(
            int(start_ind[i, 1]), int(start_ind[i, 0]),
            int(box_size_cols), int(box_size_rows))
        hydropoly_vals = get_hydropoly_val(label_im, label_values, hydropoly_ar)
        print('Stats done')

        with open(out_txt, 'a') as f:
            for i in range(len(sizes)):
                f.write("{},{},{},{},{},{},{},{},{},'{}'\n".format(
                    label_values[i],
                    start_ind_row,
                    start_ind_col,
                    box_size_rows,
                    box_size_cols,
                    int(sizes[i]),
                    hydropoly_vals[i],
                    centers_of_mass[i][0],
                    centers_of_mass[i][1],
                    border_vals[i],
                    )
                    )
