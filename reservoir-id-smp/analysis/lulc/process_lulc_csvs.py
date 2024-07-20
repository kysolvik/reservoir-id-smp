import pandas as pd
import numpy as np
import glob

mb_keys_dict = {
    'crop': np.array([18,19,39,20,40,62,41,36,46,47,35,48]),
    'forest': np.array([3]),
    'savanna': np.array([4]),
    'grassland':np.array([12]),
    'pasture': np.array([15])
}

def read_process_lulc_csv(csv):
    df = pd.read_csv(csv, index_col=0)
    df = df.loc[:, df.columns[-1]]
    lulc_y = int(df.name[42:46]) # For all of mato grosso

    df = pd.DataFrame(list(df.apply(eval).values), index=df.index)
    df.columns = pd.MultiIndex.from_product([[lulc_y], df.columns])
    return df

def summarize_lulc(year_df):
    out_df = pd.DataFrame()
    for lulc_class in mb_keys_dict.keys():
        sum_of_class = year_df.loc[:, np.in1d(year_df.columns, mb_keys_dict[lulc_class])].sum(axis=1)
        out_df[lulc_class] = sum_of_class
    return out_df

def classify_lulc(lulc_counts_df):

    natural_covers = ['forest', 'grassland', 'savanna']
    ag_covers = ['crop','pasture']
    natural_df = lulc_counts_df[natural_covers]
    ag_df = lulc_counts_df[ag_covers]
    ag_sum = ag_df.sum(axis=1)
    natural_sum = natural_df.sum(axis=1)
    is_natural = ((ag_sum < 1000) & (natural_sum> 1000))
    natural_cover = natural_df.loc[is_natural].idxmax(axis=1)
    is_ag = ((~is_natural) & (ag_sum > 1000))
    ag_cover = ag_df.loc[is_ag].idxmax(axis=1)
    lulc_counts_df['class'] = 'other'
    lulc_counts_df.loc[is_ag, 'class'] = ag_cover
    lulc_counts_df.loc[is_natural, 'class'] = natural_cover
    return lulc_counts_df['class']

def calc_lulc_full(df):
    new_df = pd.DataFrame(df.index).set_index('id_in_tile')

    for y in df.columns.levels[0]:
        year_df = summarize_lulc(df[y])
        year_df.index = df.index
        classes = classify_lulc(year_df)
        new_df.loc[:, y] = classes
        
    return new_df

def process_year_lulc(y):
    all_csvs = glob.glob('./out/lulc_res_counts/lulc_stats_res_{}_mb_*_counts.csv'.format(y))
    all_csvs.sort()
    full_df = pd.concat([read_process_lulc_csv(csv) for csv in all_csvs], axis=1)
    lulc_df = calc_lulc_full(full_df)
    return lulc_df


def main():
    year_range = np.arange(1985, 2022)

    for y in year_range:
        lulc_y_df = process_year_lulc(y)
        lulc_y_df.to_csv('./out/res_lulc_processed/res_lulc_{}.csv'.format(y))


if __name__=='__main__':
    main()