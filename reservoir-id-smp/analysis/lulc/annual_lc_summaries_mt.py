#!/usr/bin/env python3
"""Get area of landcover classes."""

import rasterio as rio
import numpy as np
import sys
import glob
import os
import pandas as pd
import numba as nb

mb_in_dir = sys.argv[1] # in/mato_grosso/aea/
out_csv = sys.argv[2] # csvs/lulc_summary_mt.csv

@nb.njit(parallel=True)
def nb_isin_listcomp(matrix, index_to_remove):
    #matrix and index_to_remove have to be numpy arrays
    #if index_to_remove is a list with different dtypes this 
    #function will fail
    og_shape = matrix.shape
    matrix = matrix.flatten()
    out=np.empty(matrix.shape[0],dtype=nb.boolean)
    index_to_remove_set=set(index_to_remove)

    for i in nb.prange(matrix.shape[0]):
        if matrix[i] in index_to_remove_set:
            out[i]=True
        else:
            out[i]=False

    return out.reshape(og_shape)

mb_keys_dict = {
    'crop': np.array([18,19,39,20,40,62,41,36,46,47,35,48]),
    'forest': np.array([3]),
    'savanna': np.array([4]),
    'grassland':np.array([12]),
    'pasture': np.array([15])
}

year_range = np.arange(1985, 2023)

areas_per_year = []
for y in year_range:
    fp = glob.glob(os.path.join(mb_in_dir, '*-{}_aea_30m.tif'.format(y)))[0]
    ar = rio.open(fp).read()
    out_dict = {
        'pasture': (ar==15).sum(),
        'forest': (ar==3).sum(),
        'crop': nb_isin_listcomp(ar, mb_keys_dict['crop']).sum()
    }
    areas_per_year.append(out_dict)

area_df = pd.DataFrame(areas_per_year)
area_df.loc[:,'year'] = np.arange(1985,2023)
area_df = area_df.set_index('year')

area_df.to_csv(out_csv)
