import pandas as pd
import numpy as np
import glob
import re
import sys

in_csv = sys.argv[1]
out_csv = sys.argv[2]

mb_keys_dict = {
    'forest':np.array([3,4,5,6,49]),
    'herb':np.array([11,12,32,29,50]),
    'pasture': np.array([15]),
    'crop': np.array([18,19,39,20,40,62,41,36,46,47,35,48]),
    'mosaic': np.array([21]),
    'other': np.array([23, 24, 30, 75, 25, 21, 9, 26, 31, 33])
}

def year_from_string(string):
    """Regex helper function"""
    match = re.match(r'.*([1-2][0-9]{3}).*', string)
    if match is None:
        raise ValueError('No year found in string')
    return match.group(1)

def read_process_lulc_csv(csv):
    df = pd.read_csv(csv, index_col=0)
    df = df.loc[:, df.columns[-1]]
    lulc_y = year_from_string(df.name) #int(df.name[44:48]) # For all of mato grosso

    df = pd.DataFrame(list(df.apply(eval).values), index=df.index)
    df.columns = pd.MultiIndex.from_product([[lulc_y], df.columns])
    return df

def summarize_lulc(year_df):
    out_df = pd.DataFrame()
    for lulc_class in mb_keys_dict.keys():
        sum_of_class = year_df.loc[:, np.in1d(year_df.columns, mb_keys_dict[lulc_class])].sum(axis=1)
        out_df[lulc_class] = sum_of_class
    out_df['other'] = year_df.sum(axis=1) - out_df.sum(axis=1)
    return out_df

def classify_lulc(lulc_counts_df):
    lulc_counts_df['class'] = lulc_counts_df.idxmax(axis=1)
    return lulc_counts_df

def calc_lulc_full(df):

    for y in df.columns.levels[0]:
        year_df = summarize_lulc(df[y])
        year_df.index = df.index
        classes = classify_lulc(year_df)

    return classes

def process_lulc_csv(in_csv):
    stats_to_save = pd.read_csv(in_csv)[['area', 'id_in_tile','row_start','col_start']].set_index('id_in_tile')
    print(stats_to_save)
    full_df = read_process_lulc_csv(in_csv)
    lulc_df = calc_lulc_full(full_df)
    print(lulc_df)
    lulc_df = pd.concat([stats_to_save, lulc_df], axis=1)
    return lulc_df

def main():
    lulc_y_df = process_lulc_csv(in_csv)
    lulc_y_df.to_csv(out_csv)

if __name__ == '__main__':
    main()