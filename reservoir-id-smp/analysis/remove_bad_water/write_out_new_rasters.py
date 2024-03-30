#!/usr/bin/env python3
"""Using merged border dataframe, writes out new reservoir raster with hydropolies removed.
    Note: box_size MUST be the same as ./find_overlaps_badwater.py
"""

from osgeo import gdal
import rasterio as rio
from rasterio.windows import Window
import numpy as np
import sys
from scipy import ndimage, stats
import cv2
import pandas as pd
import os
import sys
# Contains a simple array for calculating overlapping borders
from _border_ar import calc_border_ar

tif = sys.argv[1]
in_csv = sys.argv[2]
out_dir = sys.argv[3]
box_size = 25000
hydropoly_max_size = 50

df = pd.read_csv(in_csv)
fh = rio.open(tif)


# Set numpy print options so it doesn't abbreviate border array
np.set_printoptions(threshold=4*box_size)


def get_labels(start_ind_col, start_ind_row,
               box_size_cols, box_size_rows):
    mask = (fh.read(1,  
                    window=Window(
                        int(start_ind_col), int(start_ind_row),
                        int(box_size_cols), int(box_size_rows))
                    ) == 1)

    if mask.sum() > 0:
        # Get count
        label_im, nb_labels = ndimage.label(mask,
                                        structure = [[1,1,1],[1,1,1],[1,1,1]])
        label_values = np.arange(1, nb_labels + 1)
        print('Count done')

        return label_im, label_values
    else:
        return [0], [0]

total_rows, total_cols = fh.height, fh.width
current_row = 0
current_col = 0
row_starts = np.arange(0, total_rows, box_size)
col_starts = np.arange(0, total_cols, box_size)

# Create Nx2 array with row/col start indices
start_ind = np.array(np.meshgrid(row_starts, col_starts)).T.reshape(-1, 2)
# start_ind = np.array([[75000, 375000], [100000, 375000]])

old_transform = fh.transform
affine_transformer = rio.transform.AffineTransformer(old_transform)

for i in range(start_ind.shape[0]):
    # For the indices near edge we need to use a smaller box size
    box_size_rows = min(total_rows - start_ind[i,0], box_size)
    box_size_cols = min(total_cols - start_ind[i,1], box_size)

    # Get base counts and labeled image
    start_ind_col, start_ind_row = (start_ind[i,1], start_ind[i,0])
    cur_df = df.loc[(df['row_start']==start_ind_row) &
                    (df['col_start']==start_ind_col) &
                    (df['hydropoly_max']<hydropoly_max_size)
                    ]
    label_im, label_values = get_labels(
            start_ind_col, start_ind_row, box_size_cols, box_size_rows)

    if np.max(label_values) > 0:
        label_im[~np.isin(label_im, cur_df['id_in_tile'].values)] = 0

        new_x, new_y = affine_transformer.xy(start_ind_row, start_ind_col)
        new_transform = rio.transform.Affine(
            a=old_transform.a,
            b=old_transform.b,
            c=new_x,
            d=old_transform.d,
            e=old_transform.e,
            f=new_y)
        out_tif = os.path.join(out_dir, '{}_{}_out.tif'.format(
            start_ind_col, start_ind_row))
        with rio.open(out_tif, 'w',
                      driver='GTiff', width=box_size, height=box_size, count=1,
                      dtype=rio.uint16, compress='lzw',
                      crs=fh.crs, transform=new_transform) as dst:
            dst.write(label_im, indexes=1)


