#!/usr/bin/env python3
"""Get area of landcover classes. EXCLUDES Agua Boa"""

import rasterio as rio
import numpy as np
import sys
import glob
import os
import pandas as pd

mb_in_dir = sys.argv[1]
out_csv = sys.argv[2]

muni_tif = '../area_count/data/munis_raster_aea_v2.tif'
muni_ar = rio.open(muni_tif).read(1)

year_range = np.arange(1985, 2023)

crop_values = np.array([18,19,39,20,40,62,41,36,46,47,35,48])
dict_list = []
for y in year_range:
    tif = glob.glob(os.path.join(mb_in_dir, '*-{}_aea_10m.tif'.format(y)))[0]
    fh = rio.open(tif)
    ar = fh.read(1)
    ar[muni_ar == 0] = 0
    area_pasture = np.sum(ar == 15)
    area_crop = np.sum(np.isin(ar, crop_values))
    area_forest = np.sum(ar == 3)

    out_dict = {
            'year':y,
            'area_pasture': area_pasture,
            'area_crop': area_crop,
            'area_forest': area_forest
            }
    print(out_dict)
    dict_list.append(out_dict)

out_df = pd.DataFrame(dict_list)

out_df.to_csv(out_csv, index=False)
