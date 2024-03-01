#!/usr/bin/env python3
""" Quick script to get counts and distribution of reservoir areas'"""

from osgeo import gdal
import numpy as np
import sys
from scipy import ndimage, stats

tif = sys.argv[1]
region_tif = sys.argv[2]
out_txt = sys.argv[3]
box_size = 200000

fh = gdal.Open(tif)
region_fh = gdal.Open(region_tif) 

def calc_mode(vals):
    mode_result = stats.mode(vals)
    return mode_result.mode

def get_count(ar, region_ar):
    mask = ar == 1
    # Get count
    label_im, nb_labels = ndimage.label(mask,
                                    structure = [[1,1,1],[1,1,1],[1,1,1]])
    label_values = np.arange(1, nb_labels + 1)
    print('Count done')

    # Get sizes
    sizes = ndimage.labeled_comprehension(mask, label_im, label_values, np.sum, int, 0)

    # Attribute to region
    regions = ndimage.labeled_comprehension(region_ar, label_im, label_values, calc_mode, int, 0)
    print('Object Props done')
    return sizes, regions

total_rows, total_cols = fh.RasterYSize, fh.RasterXSize
current_row = 0
current_col = 0
row_starts = np.arange(0, total_rows, box_size)
col_starts = np.arange(0, total_cols, box_size)

# Create Nx2 array with row/col start indices
start_ind = np.array(np.meshgrid(row_starts, col_starts)).T.reshape(-1, 2)

with open(out_txt, 'w') as f:
    f.write('area,reg\n')
for i in range(start_ind.shape[0]):
    # For the indices near edge we need to use a smaller box size
    box_size_rows = min(total_rows - start_ind[i,0], box_size)
    box_size_cols = min(total_cols - start_ind[i,1], box_size)
    ar = fh.GetRasterBand(1).ReadAsArray(
        int(start_ind[i, 1]), int(start_ind[i,0]),
        int(box_size_cols),int(box_size_rows))
    reg_ar = region_fh.GetRasterBand(1).ReadAsArray(
        int(start_ind[i, 1]), int(start_ind[i,0]),
        int(box_size_cols),int(box_size_rows))
    if (ar==1).sum() > 0 and (reg_ar!=255).sum() > 0:
        sizes, regions = get_count(ar, reg_ar)
        with open(out_txt, 'a') as f:
            for i in range(len(sizes)):
                f.write("{},{}\n".format(int(sizes[i]), regions[i]))
