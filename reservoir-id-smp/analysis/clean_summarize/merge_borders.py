
import pandas as pd
import numpy as np
import re
import ast
import sys

in_csv = sys.argv[1]
out_csv = sys.argv[2]

def convert_to_array(string):
    ar = np.fromstring(string.strip('[]'),sep=' ').astype(int)
    return ar[ar!=0]

def get_unique_row_col_starts(df):
    unique_vals = pd.Series(pd.unique(df['row_start'].astype(str) + '_' + df['col_start'].astype(str)))
    row_col_starts = unique_vals.str.split('_')
    return np.vstack(row_col_starts.values).astype(int)

def get_border_dfs(df, row_start, col_start):
    cur_df = df.loc[(df['on_border']) &
                    (df['row_start']==row_start) &
                    (df['col_start']==col_start)]
    if cur_df.shape[0] > 0:
        # Right filter
        right_df = df.loc[(df['on_border']) &
                          (df['row_start'] == row_start) &
                          (df['col_start'] == col_start + cur_df['box_cols'].values[0])]

        # Bottom filter
        bottom_df = df.loc[(df['on_border']) &
                           (df['row_start'] == row_start + cur_df['box_rows'].values[0]) &
                           (df['col_start'] == col_start)]
    else:
        cur_df, right_df, bottom_df = None, None, None

    return cur_df, right_df, bottom_df


def find_overlaps(border_vals, target_series):
    list_of_overlaps = [np.any(np.isin(border_vals*-1, t)) for t in target_series]
    return list_of_overlaps

def run_full_overlaps(df, row_start, col_start):
    cur, right, bottom = get_border_dfs(df, row_start, col_start)
    if bottom is not None:
        bottom_overlaps = np.vstack(
            cur['border_vals'].apply(find_overlaps, target_series=bottom['border_vals'].values).values
        )
    else:
        bottom_overlaps = None
    if right is not None:
        right_overlaps = np.vstack(
            cur['border_vals'].apply(find_overlaps, target_series=right['border_vals'].values).values
        )
    else:
        right_overlaps = None
    return cur, right, bottom, right_overlaps, bottom_overlaps


def merge_ids(cur_df, target_df, overlaps):
    cur_df_ilocs, target_df_ilocs = np.where(overlaps)
    target_df.loc[target_df.index[target_df_ilocs], 'id'] = cur_df.iloc[cur_df_ilocs]['id'].values.astype(int)
    return target_df


df = pd.read_csv(in_csv, converters={'border_vals': convert_to_array})
df = df.rename(columns = {'id':'id_in_tile'})
df['id'] = np.arange(df.shape[0]) + 1
df['on_border'] = df['border_vals'].apply(len) != 0
row_col_starts = get_unique_row_col_starts(df)

for i in range(len(row_col_starts)):
    cur_df, right_df, bot_df, right_matches, bot_matches = run_full_overlaps(
        df, row_col_starts[i, 0], row_col_starts[i, 1])

    if bot_df is not None and np.sum(bot_matches) > 0:
        bot_updates = merge_ids(cur_df, bot_df, bot_matches)
        df.loc[bot_updates.index, 'id'] = bot_updates['id']

    if right_df is not None and np.sum(right_matches) > 0:
        right_updates = merge_ids(cur_df, right_df, right_matches)
        df.loc[right_updates.index, 'id'] = right_updates['id']


hydropoly_max = df[['id','hydropoly']].groupby('id').max().rename(columns={'hydropoly':'hydropoly_max'})

df = df.merge(hydropoly_max, left_on='id', right_index=True)

df.to_csv(out_csv, index=False)
