#!/usr/bin/env python3

import rasterio as rio
from rasterio.windows import Window
import numpy as np
import sys
import pandas as pd
import os
from scipy import ndimage

base_tif = sys.argv[1]
compare_tif = sys.argv[2]
out_csv = sys.argv[3]
box_size = 25000

out_csv_base_overlaps = base_tif.replace('.tif', '_overlaps.csv')
out_csv_comp_overlaps = compare_tif.replace('.tif', '_overlaps.csv')
fh_base = rio.open(base_tif)
fh_comp = rio.open(compare_tif)


def calc_stats(ar1, ar2):
    tp = np.sum((ar1 == 1) * (ar2 == 1))
    fp = np.sum((ar1 == 0) * (ar2 == 1))
    fn = np.sum((ar1 == 1) * (ar2 == 0))
    return tp, fp, fn

def unique_with_counts(ar):
    return np.unique(ar, return_counts=True)


total_rows, total_cols = fh_base.height, fh_base.width
current_row = 0
current_col = 0
row_starts = np.arange(0, total_rows, box_size)
col_starts = np.arange(0, total_cols, box_size)

# Create Nx2 array with row/col start indices
start_ind = np.array(np.meshgrid(row_starts, col_starts)).T.reshape(-1, 2)

# Get some affine information
cur_transform = fh_base.transform
affine_transformer = rio.transform.AffineTransformer(cur_transform)

out_dict_list = []
for i in range(start_ind.shape[0]):
    # For the indices near edge we need to use a smaller box size
    box_size_rows = min(total_rows - start_ind[i, 0], box_size)
    box_size_cols = min(total_cols - start_ind[i, 1], box_size)

    # Start indices
    start_ind_row, start_ind_col = (start_ind[i, 0], start_ind[i, 1])

    # Read in arrays
    base_ar = (fh_base.read(1,
                            window=Window(
                                int(start_ind_col), int(start_ind_row),
                                int(box_size_cols), int(box_size_rows))
                            ) == 1)
    comp_ar = (fh_comp.read(1,
                            window=Window(
                                int(start_ind_col), int(start_ind_row),
                                int(box_size_cols), int(box_size_rows))
                            ) == 1)

    tp, fp, fn = calc_stats(base_ar, comp_ar)
    out_dict = {
            'tile_id': i,
            'row_start': start_ind_row,
            'col_start': start_ind_col,
            'box_rows': box_size_rows,
            'box_cols': box_size_cols,
            'tp': tp,
            'fp': fp,
            'fn': fn
            }
    out_dict_list.append(out_dict)

    # Labeling in both directions
    base_labeled, base_ar_nb_labeled = ndimage.label(
            base_ar,
            structure=[[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])
    base_label_values = np.arange(1, base_ar_nb_labeled + 1)

    comp_labeled, comp_ar_nb_labeled = ndimage.label(
            base_ar,
            structure=[[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])
    comp_label_values = np.arange(1, comp_ar_nb_labeled + 1)

    base_sizes = ndimage.labeled_comprehension(
            base_ar,
            base_labeled,
            base_label_values,
            np.sum,
            int,
            )
    comp_sizes = ndimage.labeled_comprehension(
            comp_ar,
            comp_labeled,
            comp_label_values,
            np.sum,
            int,
            )
    base_containing_comp = ndimage.labeled_comprehension(
            comp_labeled,
            base_labeled,
            base_label_values,
            np.unique,
            np.ndarray,
            [0])

    comp_containing_base = ndimage.labeled_comprehension(
            base_labeled,
            comp_labeled,
            base_label_values,
            np.unique,
            np.ndarray,
            [0])

    out_dict_base = {
            'id': base_label_values,
            'size': base_sizes,
            'row_start': start_ind_row,
            'col_start': start_ind_col,
            'box_rows': box_size_rows,
            'box_cols': box_size_cols,
            'overlaps': base_containing_comp
            }

    out_dict_comp = {
            'id': comp_label_values,
            'size': comp_sizes,
            'row_start': start_ind_row,
            'col_start': start_ind_col,
            'box_rows': box_size_rows,
            'box_cols': box_size_cols,
            'overlaps': comp_containing_base
            }
    pd.DataFrame(out_dict_base).to_csv(out_csv_base_overlaps,
                                       mode='a', header=(not os.path.isfile(out_csv_base_overlaps)),
                                       index=False)

    pd.DataFrame(out_dict_comp).to_csv(out_csv_comp_overlaps,
                                       mode='a', header=(not os.path.isfile(out_csv_comp_overlaps)),
                                       index=False)

print(out_dict_list)
out_df = pd.DataFrame(out_dict_list)
out_df.to_csv(out_csv, mode='w', header=True,
              index=False)
