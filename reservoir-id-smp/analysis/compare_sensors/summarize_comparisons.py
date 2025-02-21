import pandas as pd
import numpy as np
def convert_to_array(string):
    ar = np.fromstring(string.strip('[]'),sep=' ').astype(int)
    return ar[ar!=0]

ls7 = pd.read_csv('./ls7_2020_v3_aea_overlaps.csv', converters={'overlaps':convert_to_array})
ls8 = pd.read_csv('./ls8_2020_v3_aea_overlaps.csv', converters={'overlaps':convert_to_array})
ls7['overlap_count'] = ls7['overlaps'].apply(len)
ls8['overlap_count'] = ls8['overlaps'].apply(len)
ls7_overlap = ls7.loc[ls7.overlap_count>0]
ls8_overlap = ls8.loc[ls8.overlap_count>0]

ls7_nooverlap = ls7.loc[ls7.overlap_count==0]
ls8_nooverlap = ls8.loc[ls8.overlap_count==0]

print('LS7 total area:', ls7['size'].sum())
print('LS8 total area:', ls8['size'].sum())
print('LS7 mean size:' ,ls7['size'].mean())
print('LS8 mean size:' ,ls8['size'].mean())
print('LS7 count:' ,ls7['size'].shape[0])
print('LS8 count:' ,ls8['size'].shape[0])

print('~'*10)

print('LS7_overlap total area:', ls7_overlap['size'].sum())
print('LS8_overlap total area:', ls8_overlap['size'].sum())
print('LS7_overlap mean size:' ,ls7_overlap['size'].mean())
print('LS8_overlap mean size:' ,ls8_overlap['size'].mean())
print('LS7_overlap count:' ,ls7_overlap['size'].shape[0])
print('LS8_overlap count:' ,ls8_overlap['size'].shape[0])

print('~'*10)

print('LS7_nooverlap total area:', ls7_nooverlap['size'].sum())
print('LS8_nooverlap total area:', ls8_nooverlap['size'].sum())
print('LS7_nooverlap mean size:' ,ls7_nooverlap['size'].mean())
print('LS8_nooverlap mean size:' ,ls8_nooverlap['size'].mean())
print('LS7_nooverlap count:' ,ls7_nooverlap['size'].shape[0])
print('LS8_nooverlap count:' ,ls8_nooverlap['size'].shape[0])

