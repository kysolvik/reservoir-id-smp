import pandas as pd
import os

invalid_df = pd.read_csv('./invalid_indices.txt', header=None, names=['x','y'])
invalid_df['fullname'] = 'pred_' + invalid_df['x'].astype(str) + '-' + invalid_df['y'].astype(str) + '.tif'

for f in invalid_df['fullname']:
    if os.path.isfile(os.path.join('out',f)):
            os.remove(os.path.join('out',f))
