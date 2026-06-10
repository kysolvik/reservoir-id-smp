import pandas as pd
import geopandas as gpd
import rasterio as rio
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


def distance_to_nearest(gdf_target, gdf_training, k=3):

    assert gdf_training.crs == gdf_target.crs

    coords_train = np.stack([gdf_training.geometry.x, gdf_training.geometry.y], axis=1)
    coords_target = np.stack([gdf_target.geometry.x, gdf_target.geometry.y], axis=1)

    tree = cKDTree(coords_train)
    distances, indices = tree.query(coords_target, k=k)
    if k > 1:
        gdf_target["dist_to_nearest_training"] = distances.mean(axis=1)
    else:
        gdf_target["dist_to_nearest_training"] = distances

    return gdf_target


def prep_sample_stats_csv(csv):

    full_df = pd.read_csv(csv)
    full_df['precision'] = full_df['true_positive_pixels'] / (full_df['true_positive_pixels'] + full_df['false_positive_pixels'] + 1e-6)
    full_df['recall'] = full_df['true_positive_pixels'] / (full_df['true_positive_pixels'] + full_df['false_negative_pixels'] + 1e-6)
    full_df['f1'] = 2 * (full_df['precision'] * full_df['recall']) / (full_df['precision'] + full_df['recall'] + 1e-6)
    full_gdf = gpd.GeoDataFrame(full_df,
                                geometry=gpd.points_from_xy(full_df['center_longitude'], full_df['center_latitude']),
                                crs='EPSG:4326').to_crs('ESRI:102033')
    train_df = full_gdf[full_gdf['set'] == 'train']
    full_df
    val_df = full_gdf[full_gdf['set'] == 'val']
    test_df = full_gdf[full_gdf['set'] == 'test']
    val_df = distance_to_nearest(val_df, train_df)
    test_df = distance_to_nearest(test_df, train_df)
    test_val_df = distance_to_nearest(test_df.copy(), val_df)
    return val_df, test_df, test_val_df

def plot_distance_curve(df, title):
    df["dist_bin"] = pd.cut(df["dist_to_nearest_training"], bins=10)
    binned = df.groupby("dist_bin")["f1"].agg(["mean", "std", "count"])
    bin_centers = binned.index.map(lambda x: x.mid)

    fig, axs = plt.subplots(2, 1)
    ax = axs[0]
    ax.plot(bin_centers.values, binned["mean"], marker="o")
    ax.fill_between(
        bin_centers.values,
        binned["mean"] - binned["std"],
        binned["mean"] + binned["std"],
        alpha=0.2,
        label="±1 std"
    )
    ax.set_xlabel("Distance to nearest training point (m)")
    ax.set_ylabel("F1 score")
    ax.set_title(title)
    ax2 = axs[1]
    ax2.bar(x=np.array(bin_centers.values, dtype=np.float32),
            height=binned["count"].values,
            width=(bin_centers.values[1] - bin_centers.values[0]) * 0.8)
    ax2.set_ylabel("Tile count")
    plt.tight_layout()
    fig.show()

val_df, test_df, test_val_df = prep_sample_stats_csv('../annotation_prep/csvs/annotation_locations_stats.csv')

plot_distance_curve(val_df.loc[val_df[['true_positive_pixels', 'false_negative_pixels']].sum(axis=1) > 0], "Validation Set")
plot_distance_curve(test_df.loc[test_df[['true_positive_pixels', 'false_negative_pixels']].sum(axis=1) > 0], "Test Set")
plot_distance_curve(test_val_df.loc[test_val_df[['true_positive_pixels', 'false_negative_pixels']].sum(axis=1) > 0], "Test (Validation Points)")