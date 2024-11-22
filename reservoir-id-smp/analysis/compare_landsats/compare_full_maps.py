#!/usr/bin/env python3

import rasterio as rio
from rasterio.windows import Window
import numpy as np
import sys
import pandas as pd
import os

base_tif = sys.argv[1]
compare_tif = sys.argv[2]
out_csv = sys.argv[3]
box_size = 50000

fh_base = rio.open(base_tif)
fh_comp = rio.open(compare_tif)


def calc_stats(ar1, ar2):
    tp = np.sum((ar1 == 1) * (ar2 == 1))
    fp = np.sum((ar1 == 0) * (ar2 == 1))
    fn = np.sum((ar1 == 1) * (ar2 == 0))
    return tp, fp, fn


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
    print(out_dict)

print(out_dict_list)
out_df = pd.DataFrame(out_dict_list)
out_df.to_csv(out_csv, mode='w', header=True,
              index=False)
