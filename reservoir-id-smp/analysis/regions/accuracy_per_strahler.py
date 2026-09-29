"""Model accuracy broken up by Strahler order of level8 sub-basins

Two sources of accuracy:
    1. Sentinel test set chips: pixel-wise precision, recall, f1 (pooled per order)
       and true vs predicted reservoir area. 95% CIs from bootstrapping chips.
    2. Manually checked detections: precision only (strict and lenient), with
       Wilson 95% CIs, and median area of detections by label.

Orders >= MAX_ORDER are merged into one group (e.g. '6+'), since few samples
fall in the largest basins.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import binomtest

BASINS_PATH = './data/level8_basins_strahler_brazil.gpkg'
STRAHLER_COL = 'nustrahler'
GRID_CSV = '../../annotation_prep/csvs/grid_indices_with_centers.csv'
TEST_STATS_CSV = '../../train/data/preds/test_per_image.csv'
MANUAL_PATH = '../manual_checks/manually_checked_detections.gpkg'

GROUP_COL = 'strahler_group'
MAX_ORDER = 6
PIXEL_HA = 0.01  # Sentinel 10m pixels
N_BOOT = 2000
SEED = 12
PIXEL_METRICS = ['area_ratio', 'precision', 'recall', 'f1']
MANUAL_METRICS = ['precision_strict', 'precision_lenient']


def load_basins():
    basins = gpd.read_file(BASINS_PATH)
    return basins[[STRAHLER_COL, 'geometry']]


def join_strahler(points, basins, name):
    """Point-in-polygon join to basins, dropping (and reporting) unmatched points"""
    joined = gpd.sjoin(points.to_crs(basins.crs), basins, how='left', predicate='within')
    unmatched = joined[STRAHLER_COL].isna()
    print('{}: {} of {} not within any basin, dropped'.format(
        name, unmatched.sum(), joined.shape[0]))
    joined = joined.loc[~unmatched].copy()
    joined[GROUP_COL] = strahler_group(joined[STRAHLER_COL])
    return joined


def strahler_group(order):
    """Order as string label, with orders >= MAX_ORDER merged into e.g. '6+'"""
    group = order.clip(upper=MAX_ORDER).astype(int).astype(str)
    return group.replace(str(MAX_ORDER), '{}+'.format(MAX_ORDER))


def load_chip_points():
    grid = pd.read_csv(GRID_CSV)
    stats = pd.read_csv(TEST_STATS_CSV)
    chips = stats.merge(grid[['basename', 'xmean', 'ymean']], on='basename', how='inner')
    print('Test chips with centers: {} of {}'.format(chips.shape[0], stats.shape[0]))
    return gpd.GeoDataFrame(
        chips,
        geometry=gpd.points_from_xy(chips['xmean'], chips['ymean']),
        crs='EPSG:4326')


def pixel_metrics(tp, fp, fn):
    """Metrics from tp/fp/fn totals (scalars or bootstrap arrays), NaN where undefined"""
    tp, fp, fn = (np.asarray(x, dtype=float) for x in (tp, fp, fn))
    with np.errstate(divide='ignore', invalid='ignore'):
        metrics = {
            'area_ratio': (tp + fp) / (tp + fn),
            'precision': tp / (tp + fp),
            'recall': tp / (tp + fn),
            'f1': 2*tp / (2*tp + fp + fn),
        }
    return {k: np.where(np.isfinite(v), v, np.nan) for k, v in metrics.items()}


def percentile_ci(samples):
    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return np.nan, np.nan
    return tuple(np.percentile(samples, [2.5, 97.5]))


def pooled_metrics(df):
    """Pool pixels across chips, with 95% CIs from bootstrap resampling of chips"""
    counts = {c: df[c].to_numpy() for c in ['tp', 'fp', 'fn']}
    point = pixel_metrics(*(counts[c].sum() for c in ['tp', 'fp', 'fn']))

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, df.shape[0], size=(N_BOOT, df.shape[0]))
    boot = pixel_metrics(*(counts[c][idx].sum(axis=1) for c in ['tp', 'fp', 'fn']))

    out = {
        'n_chips': df.shape[0],
        # Chips with no true or predicted pixels add nothing to the pooled metrics
        'n_chips_with_res': ((counts['tp'] + counts['fp'] + counts['fn']) > 0).sum(),
        'true_area_ha': (counts['tp'].sum() + counts['fn'].sum()) * PIXEL_HA,
        'pred_area_ha': (counts['tp'].sum() + counts['fp'].sum()) * PIXEL_HA,
    }
    for m in PIXEL_METRICS:
        out[m] = float(point[m])
        out[m + '_lo'], out[m + '_hi'] = percentile_ci(boot[m])
    return pd.Series(out)


def manual_precision(df):
    out = {'n_detections': df.shape[0]}
    correct = {
        'precision_strict': df['label'] == 'True Positive',
        'precision_lenient': df['label'].isin(['True Positive', 'Questionable']),
    }
    for m, is_correct in correct.items():
        ci = binomtest(int(is_correct.sum()), df.shape[0]).proportion_ci(method='wilson')
        out[m] = is_correct.mean()
        out[m + '_lo'], out[m + '_hi'] = ci.low, ci.high
    for label, short in [('True Positive', 'tp'), ('Questionable', 'questionable'),
                         ('False Positive', 'fp')]:
        out['median_area_ha_' + short] = df.loc[df['label'] == label, 'area_ha'].median()
    return pd.Series(out)


def with_all_row(df, group_func):
    by_order = df.groupby(GROUP_COL)[df.columns].apply(group_func)
    by_order.loc['All'] = group_func(df)
    return by_order


def format_ci(df, metrics):
    """Collapse metric/_lo/_hi columns into 'est [lo, hi]' strings for printing"""
    out = df.copy()
    for m in metrics:
        out[m] = ['{:.3f} [{:.3f}, {:.3f}]'.format(e, lo, hi)
                  for e, lo, hi in zip(df[m], df[m + '_lo'], df[m + '_hi'])]
        out = out.drop(columns=[m + '_lo', m + '_hi'])
    return out


def main():
    basins = load_basins()

    # Test set chips
    chips = join_strahler(load_chip_points(), basins, 'Test chips')
    chip_results = with_all_row(chips, pooled_metrics)
    chip_results[['n_chips', 'n_chips_with_res']] = chip_results[['n_chips', 'n_chips_with_res']].astype(int)
    print('\nTest set pixel-wise accuracy by Strahler order '
          '(95% CI from {} chip bootstraps; area_ratio = pred/true area):'.format(N_BOOT))
    print(format_ci(chip_results, PIXEL_METRICS).to_string(float_format='{:.1f}'.format))

    # Manually checked detections, using centroids
    manual = gpd.read_file(MANUAL_PATH)
    manual['geometry'] = manual.geometry.centroid
    manual = join_strahler(manual, basins, 'Manual detections')
    manual_results = with_all_row(manual, manual_precision)
    manual_results['n_detections'] = manual_results['n_detections'].astype(int)
    print('\nManually checked detection precision by Strahler order '
          '(Wilson 95% CI; median areas in ha by label):')
    print(format_ci(manual_results, MANUAL_METRICS).to_string(float_format='{:.3f}'.format))


if __name__ == '__main__':
    main()
