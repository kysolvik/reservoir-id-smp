#!/usr/bin/env python3

import rasterio as rio
from rasterio.windows import Window
import numpy as np
import sys
import pandas as pd
import os
from scipy import ndimage

ls_tif = sys.argv[1]
sent_tif = sys.argv[2]
out_csv = sys.argv[3]
ls_name = os.path.basename(ls_tif)[:3]
sent_name = 'sentinel'


cutoff_dict = {
        'ls5': 9,
        'ls7': 1,
        'ls8': 179
        }
ls_cutoff = cutoff_dict[ls_name]
sent_cutoff = 0.5 #cutoff_dict[comp_ls_name] 
box_size = 10000

out_csv_ls_overlaps = os.path.join('./out/landsat_sentinel/', os.path.basename(ls_tif).replace('.tif', '_overlaps.csv'))
out_csv_sent_overlaps = os.path.join('./out/landsat_sentinel/', os.path.basename(sent_tif).replace('.tif', '_overlaps.csv'))
fh_ls = rio.open(ls_tif)
fh_sent = rio.open(sent_tif)


def calc_stats(ar1, ar2):
    tp = np.sum((ar1 == 1) * (ar2 == 1))
    fp = np.sum((ar1 == 0) * (ar2 == 1))
    fn = np.sum((ar1 == 1) * (ar2 == 0))
    return tp, fp, fn

def unique_with_counts(ar):
    return np.unique(ar, return_counts=True)


total_rows, total_cols = fh_ls.height, fh_ls.width
current_row = 0
current_col = 0
row_starts = np.arange(0, total_rows, box_size)
col_starts = np.arange(0, total_cols, box_size)

# Create Nx2 array with row/col start indices
start_ind = np.array(np.meshgrid(row_starts, col_starts)).T.reshape(-1, 2)

# Get some affine information
cur_transform = fh_ls.transform
affine_transformer = rio.transform.AffineTransformer(cur_transform)

out_dict_list = []
for i in range(start_ind.shape[0]):
    # For the indices near edge we need to use a smaller box size
    box_size_rows = min(total_rows - start_ind[i, 0], box_size)
    box_size_cols = min(total_cols - start_ind[i, 1], box_size)

    # Start indices
    start_ind_row, start_ind_col = (start_ind[i, 0], start_ind[i, 1])

    # Read in arrays
    ls_ar = (fh_ls.read(1,
                            window=Window(
                                int(start_ind_col), int(start_ind_row),
                                int(box_size_cols), int(box_size_rows))
                            ))
    sent_ar = (fh_sent.read(1,
                            window=Window(
                                int(start_ind_col), int(start_ind_row),
                                int(box_size_cols), int(box_size_rows))
                            ))
    ls_ar = (ls_ar >= ls_cutoff)*(ls_ar!=255)
    sent_ar = (sent_ar >= sent_cutoff)*(sent_ar!=255)

    tp, fp, fn = calc_stats(ls_ar, sent_ar)
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
    ls_labeled, ls_ar_nb_labeled = ndimage.label(
            ls_ar,
            structure=[[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])
    ls_label_values = np.arange(1, ls_ar_nb_labeled + 1)

    sent_labeled, sent_ar_nb_labeled = ndimage.label(
            sent_ar,
            structure=[[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])
    sent_label_values = np.arange(1, sent_ar_nb_labeled + 1)

    if ls_ar_nb_labeled > 0:
        ls_sizes = ndimage.labeled_comprehension(
                ls_ar,
                ls_labeled,
                ls_label_values,
                np.sum,
                int,
                0
                )
        ls_containing_sent = ndimage.labeled_comprehension(
                sent_labeled,
                ls_labeled,
                ls_label_values,
                np.unique,
                np.ndarray,
                [0])


        out_dict_ls = {
                'id': ls_label_values,
                'size': ls_sizes,
                'row_start': start_ind_row,
                'col_start': start_ind_col,
                'box_rows': box_size_rows,
                'box_cols': box_size_cols,
                'overlaps': ls_containing_sent
                }
        pd.DataFrame(out_dict_ls).to_csv(
                out_csv_ls_overlaps,
                mode='a',
                header=(not os.path.isfile(out_csv_ls_overlaps)),
                index=False)


    if sent_ar_nb_labeled > 0:
        sent_sizes = ndimage.labeled_comprehension(
                sent_ar,
                sent_labeled,
                sent_label_values,
                np.sum,
                int,
                0
                )
        sent_containing_ls = ndimage.labeled_comprehension(
                ls_labeled,
                sent_labeled,
                sent_label_values,
                np.unique,
                np.ndarray,
                [0])
        out_dict_sent = {
                'id': sent_label_values,
                'size': sent_sizes,
                'row_start': start_ind_row,
                'col_start': start_ind_col,
                'box_rows': box_size_rows,
                'box_cols': box_size_cols,
                'overlaps': sent_containing_ls
                }
        pd.DataFrame(out_dict_sent).to_csv(
                out_csv_sent_overlaps,
                mode='a',
                header=(not os.path.isfile(out_csv_sent_overlaps)),
                index=False)

out_df = pd.DataFrame(out_dict_list)
out_df.to_csv(out_csv, mode='w', header=True,
              index=False)
