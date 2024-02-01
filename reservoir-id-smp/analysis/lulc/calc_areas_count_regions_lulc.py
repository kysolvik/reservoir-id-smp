#!/usr/bin/env python3
""" Quick script to get counts and distribution of reservoir areas'"""

from osgeo import gdal
import numpy as np
import sys
from scipy import ndimage

tif = sys.argv[1]
region_tif = sys.argv[2]
year = sys.argv[3]
out_txt = sys.argv[4]
box_size = 20000

fh = gdal.Open(tif)
region_fh = gdal.Open(region_tif) 
crop_fh = gdal.Open('./out/mb_c8_crops_prox_{}.tif'.format(year))
pasture_fh = gdal.Open('./out/mb_c8_pasture_prox_{}.tif'.format(year))
forest_fh = gdal.Open('./out/mb_c8_forest_prox_{}.tif'.format(year))


def get_count_dist(ar, crop_ar, pasture_ar, forest_ar):
    mask = ar == 1
    # Get count
    label_im, nb_labels = ndimage.label(mask,
                                        structure=[[1,1,1],[1,1,1],[1,1,1]])
    print('Count done')

    # Region props
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    forest_dists = ndimage.minimum(forest_ar, label_im, range(nb_labels + 1))
    crop_dists = ndimage.minimum(crop_ar, label_im, range(nb_labels + 1))
    pasture_dists = ndimage.minimum(pasture_ar, label_im, range(nb_labels + 1))
    print('Reg Props done')
    return sizes, crop_dists, pasture_dists, forest_dists

total_rows, total_cols = fh.RasterYSize, fh.RasterXSize
current_row = 0
current_col = 0
row_starts = np.arange(0, total_rows, box_size)
col_starts = np.arange(0, total_cols, box_size)

# Create Nx2 array with row/col start indices
start_ind = np.array(np.meshgrid(row_starts, col_starts)).T.reshape(-1, 2)

with open(out_txt, 'w') as f:
    f.write('reg,area,crop,pasture,forest\n')
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
    crop_ar = crop_fh.GetRasterBand(1).ReadAsArray(
        int(start_ind[i, 1]), int(start_ind[i,0]),
        int(box_size_cols),int(box_size_rows))
    pasture_ar = pasture_fh.GetRasterBand(1).ReadAsArray(
        int(start_ind[i, 1]), int(start_ind[i,0]),
        int(box_size_cols),int(box_size_rows))
    forest_ar = forest_fh.GetRasterBand(1).ReadAsArray(
        int(start_ind[i, 1]), int(start_ind[i,0]),
        int(box_size_cols),int(box_size_rows))
    for reg_num in np.unique(reg_ar):
        if reg_num==255:
            continue
        ar_cur_reg = ar.copy() 
        ar_cur_reg[reg_ar!=reg_num] = 0
        print(reg_num)
        sizes, crop_dists, pasture_dists, forest_dists = get_count_dist(ar_cur_reg, crop_ar, pasture_ar, forest_ar)
        with open(out_txt, 'a') as f:
            for i in range(len(sizes)):
                row = "{},{},{},{},{}\n".format(
                        int(reg_num), int(sizes[i]),
                        crop_dists[i], pasture_dists[i], forest_dists[i]
                        )
                f.write(row)
