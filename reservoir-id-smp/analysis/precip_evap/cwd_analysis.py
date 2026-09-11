import geopandas as gpd
import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np



def read_process_region_csv(csv):
    temp_df = pd.read_csv(csv)
    temp_df['satellite'] = os.path.basename(csv)[:3]
    temp_df['year'] = int(os.path.basename(csv)[4:8])

    return temp_df

def get_biome_data():
    # Read and process
    biome_csvs = glob.glob('../regions/out/ls*cloudfilt*biome*.csv')
    biome_csvs.sort()
    biome_list = [read_process_region_csv(csv) for csv in biome_csvs]
    biome_df = pd.concat(biome_list).set_index('year')
    # Some filtering
    # biome_df = biome_df.loc[~((biome_df.index==2002)&(biome_df.satellite=='ls5'))]
    # biome_df = biome_df.loc[~((biome_df.index<2001)&(biome_df.satellite=='ls7'))]
    biome_df = biome_df.loc[~((biome_df.index>2019)&(biome_df.satellite=='ls7'))]
    biome_df = biome_df.drop(columns='satellite').groupby(['year', 'Bioma']).mean().reset_index().set_index('year')
    # biome_df = biome_df.loc[:2019]
    biome_df = biome_df.sort_index()
    biome_df['biome'] = biome_df['Bioma']# .map(biome_shortname_dict)
    biome_df_columns = biome_df.reset_index().set_index(['year','biome']).unstack(level=1).drop(
        columns=['Bioma'])
    biome_columns_sorted = ['Pampa','Pantanal', 'Amazônia', 'Cerrado',  'Mata Atlântica', 'Caatinga']
    biome_df_columns.loc[1984, 'count'].sort_values().index
    biome_df_columns = biome_df_columns.reindex(biome_columns_sorted, axis=1, level=1)
    biome_df_columns['count'] = biome_df_columns['count']/1000
    biome_df_columns['sum'] = biome_df_columns['sum']/100

    return biome_df_columns


biome_df = get_biome_data()
cwd_df = pd.read_csv('./data/mcwd_ppt_biomas.csv')
rename_dict = dict(zip(cwd_df['bioma'].unique(),
                       sorted(biome_df.columns.get_level_values(1).unique())))
cwd_df['bioma'] = cwd_df['bioma'].map(rename_dict)
cwd_df = cwd_df.pivot(index='hydro_year', columns='bioma', values=['mcwd_mm','ppt_mm'])


# Plot
full_df = cwd_df.join(biome_df).swaplevel(axis=1)
def line_plot(reg, full_df):
    full_df[reg][['mcwd_mm', 'ppt_mm', 'sum', 'count']].plot()
    plt.show()

def line_plot_all_biomes(full_df):
    biomes = [
        'Pampa',
        'Pantanal',
        'Amazônia',
        'Cerrado',
        'Mata Atlântica',
        'Caatinga',
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    axes = axes.ravel()

    for ax, biome in zip(axes, biomes):
        full_df[biome][['mcwd_mm', 'ppt_mm', 'sum', 'count']].plot(ax=ax)
        ax.set_title(biome)
        ax.set_xlabel('Hydrological year')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def detrend_series(series):
    valid = series.notna()
    x = series.index.to_numpy(dtype=float)
    coefficients = np.polyfit(x[valid], series[valid], 1)
    trend = np.polyval(coefficients, x)

    return series - trend


def scatter_plot_all_biomes(full_df, x_variable, y_variable,
                            detrend_x=False, detrend_y=False):
    biomes = [
        'Pampa',
        'Pantanal',
        'Amazônia',
        'Cerrado',
        'Mata Atlântica',
        'Caatinga',
    ]

    plot_df = full_df.copy()

    for biome in biomes:
        if detrend_x:
            plot_df.loc[:, (biome, x_variable)] = detrend_series(
                plot_df[biome][x_variable]
            )
        if detrend_y:
            plot_df.loc[:, (biome, y_variable)] = detrend_series(
                plot_df[biome][y_variable]
            )

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for ax, biome in zip(axes, biomes):
        data = plot_df[biome][[x_variable, y_variable]].dropna()
        x = data[x_variable].to_numpy()
        y = data[y_variable].to_numpy()

        ax.scatter(x, y, alpha=0.7)

        if len(data) >= 2:
            coefficients = np.polyfit(x, y, 1)
            fit_y = np.polyval(coefficients, x)

            ss_res = np.sum((y - fit_y) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

            order = np.argsort(x)
            ax.plot(
                x[order],
                fit_y[order],
                color='black',
                linestyle='--',
                linewidth=2,
            )
            ax.text(
                0.05,
                0.95,
                f'$R^2 = {r2:.2f}$',
                transform=ax.transAxes,
                verticalalignment='top',
            )

        ax.set_title(biome)
        ax.set_xlabel(x_variable)
        ax.set_ylabel(y_variable)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{y_variable} vs. {x_variable}', fontsize=16)
    fig.tight_layout()
    plt.show()

scatter_plot_all_biomes(full_df, 'mcwd_mm', 'sum', detrend_x=True, detrend_y=True)