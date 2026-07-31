import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from sklearn.metrics import precision_recall_curve
import torch
from scipy import ndimage
import os


# Object-size classes (pixels) used for size-binned precision/recall reporting.
# Shared by full_evaluation (Sentinel) and the Landsat threshold workflow.
# Sentinel
SIZE_DICT = {
    'remove_xsmall': [0, 4],
    'very_small': [4, 10],
    'small': [10, 100],
    'medium': [100, 1000],
    'large': [1000, 5000],
    'remove_xlarge': [5000, 1000000],
}
# Landsat
# SIZE_DICT = {
#     'remove_xsmall': [0, 10],
#     'small': [10, 100],
#     'medium': [100, 1000],
#     'large': [1000, 10000],
#     'remove_xlarge': [10000, 1000000],
# }


def distance_to_nearest(gdf_target, gdf_training, k=1):

    assert gdf_training.crs == gdf_target.crs

    coords_train = np.stack([gdf_training.geometry.x, gdf_training.geometry.y], axis=1)
    coords_target = np.stack([gdf_target.geometry.x, gdf_target.geometry.y], axis=1)

    tree = cKDTree(coords_train)
    distances, indices = tree.query(coords_target, k=k)
    if k > 1:
        gdf_target["dist_to_nearest_training"] = np.median(distances, axis=1)
    else:
        gdf_target["dist_to_nearest_training"] = distances

    return gdf_target


# Annotation tile locations (name/split/center coords) shared by both sensors.
# Path is relative to the train/ dir the eval scripts run from.
LOCATIONS_CSV = '../annotation_prep/csvs/annotation_locations.csv'
# Equal-area CRS (South America Albers) used for distance computations.
DIST_PROJ_CRS = 'ESRI:102033'


def build_distance_dfs(per_image_df, locations_csv=LOCATIONS_CSV, proj_crs=DIST_PROJ_CRS):
    """Merge per-image tp/fp/fn stats onto annotation locations and measure,
    for each val/test tile, the distance to the nearest *training* tile.

    The tile coordinates come from ``locations_csv`` (so every training tile has
    a position even when only val/test predictions are available), and the
    per-image prediction stats are joined on ``basename`` == annotation ``name``
    minus its ``.tif`` extension.

    Args:
        per_image_df: DataFrame as written by ``per_image_stats`` with columns
            ['basename', 'tp', 'fp', 'fn', ...]. May cover any mix of splits.
        locations_csv: annotation_locations.csv with 'name', 'split',
            'center_longitude', 'center_latitude'.
        proj_crs: equal-area CRS used for the nearest-neighbor distances.

    Returns:
        (val_df, test_df, test_val_df) GeoDataFrames with 'precision'/'recall'/
        'f1' and 'dist_to_nearest_training' columns. ``test_val_df`` measures
        test-tile distance to the nearest *validation* tile instead of training.
    """
    loc = pd.read_csv(locations_csv)
    loc['basename'] = loc['name'].str.replace('.tif', '', regex=False)
    loc_gdf = gpd.GeoDataFrame(
        loc,
        geometry=gpd.points_from_xy(loc['center_longitude'], loc['center_latitude']),
        crs='EPSG:4326').to_crs(proj_crs)

    stats = per_image_df.copy()
    stats['precision'] = stats['tp'] / (stats['tp'] + stats['fp'] + 1e-6)
    stats['recall'] = stats['tp'] / (stats['tp'] + stats['fn'] + 1e-6)
    stats['f1'] = 2 * (stats['precision'] * stats['recall']) / (stats['precision'] + stats['recall'] + 1e-6)
    gdf = loc_gdf.merge(
        stats[['basename', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1']],
        on='basename', how='left')

    train_df = gdf[gdf['split'] == 'train']
    val_df = gdf[(gdf['split'] == 'val') & gdf['f1'].notna()].copy()
    test_df = gdf[(gdf['split'] == 'test') & gdf['f1'].notna()].copy()
    val_df = distance_to_nearest(val_df, train_df)
    test_df = distance_to_nearest(test_df, train_df)
    test_val_df = distance_to_nearest(test_df.copy(), val_df)
    return val_df, test_df, test_val_df


SPLIT_NAMES = {'val': 'Validation', 'test': 'Test', 'train': 'Training'}


def pretty_sensor_label(label):
    """Turn a sensor code into a readable name for figure titles.

    'ls8_2017' -> 'Landsat 8 (2017)'; 'sentinel' -> 'Sentinel-2'; anything else
    is returned unchanged (with underscores turned into spaces).
    """
    low = label.lower()
    if low.startswith('ls') and '_' in low:
        sensor, year = low.split('_', 1)
        return f'Landsat {sensor[2:]} ({year})'
    if low == 'sentinel':
        return 'Sentinel-2'
    return label.replace('_', ' ')


def plot_distance_curves(per_image_df, label, locations_csv=LOCATIONS_CSV,
                         fig_dir=None, proj_crs=DIST_PROJ_CRS):
    """Plot F1-vs-distance-to-nearest-training-tile for the val and test splits.

    Builds the distance DataFrames from ``per_image_df`` + ``locations_csv``
    (see ``build_distance_dfs``) and plots one micro-averaged F1 curve per split.
    All tiles are kept (including those with no reservoirs) so their false
    positives count toward each bin's pooled precision. If ``fig_dir`` is given
    the figures are saved as ``distance_curve_{label}_{split}.jpg``, otherwise
    they are shown interactively.
    """
    val_df, test_df, test_val_df = build_distance_dfs(per_image_df, locations_csv, proj_crs)
    print('Median distance to nearest training tile:', test_df['dist_to_nearest_training'].median())
    print('Median distance to nearest val tile:', test_val_df['dist_to_nearest_training'].median())
    slug = label.lower().replace(' ', '_')
    sensor_name = pretty_sensor_label(label)
    for split_name, df in [('val', val_df), ('test_train', test_df), ('test_val', test_val_df)]:
        save_path = None
        if fig_dir is not None:
            os.makedirs(fig_dir, exist_ok=True)
            save_path = os.path.join(fig_dir, f'distance_curve_{slug}_{split_name}.jpg')
        title = f'{sensor_name} — {SPLIT_NAMES.get(split_name, split_name)} set'
        if split_name == 'test_val':
            cap_distance = 200000
        else:
            cap_distance = 100000
        plot_distance_curve(df, title, save_path, cap=cap_distance)
        if save_path is not None:
            print(f'  wrote {save_path}')


def _draw_distance_panels(df, ax_f1, ax_count, xlabel, bin_width=20000,
                          min_count=1, cap=200000):
    """Draw a micro-averaged F1 curve (``ax_f1``) over a per-bin tile-count bar
    panel (``ax_count``) for one distance DataFrame.

    Tiles are grouped into fixed-width distance bins (``bin_width`` meters) and a
    single F1 is computed per bin by pooling each tile's tp/fp/fn, rather than
    averaging per-tile F1 (which is noisy for tiles with few reservoir pixels).
    Everything beyond ``cap`` meters is pooled into one catch-all bin so the
    sparse long tail doesn't get its own noisy bins; its right edge is labeled
    with the actual maximum distance. Bins with fewer than ``min_count`` tiles
    are dropped. The two axes are expected to share an x-axis.
    """
    df = df.copy()
    bin_km = bin_width / 1000
    cap_km = cap / 1000
    dist_km = df["dist_to_nearest_training"] / 1000

    # Fixed-width bins up to the cap, then one catch-all bin (cap, inf).
    reg_edges = np.arange(0, cap_km + bin_km, bin_km)
    df["dist_bin"] = pd.cut(dist_km, bins=np.append(reg_edges, np.inf))
    grouped = df.groupby("dist_bin", observed=False)
    pooled = grouped[["tp", "fp", "fn"]].sum()
    precision = pooled["tp"] / (pooled["tp"] + pooled["fp"] + 1e-6)
    recall = pooled["tp"] / (pooled["tp"] + pooled["fn"] + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    # Bins with no ground-truth reservoirs (no tp and no fn) have an undefined
    # F1 (rather than ~0); a bin with fn but no tp is a real recall-0 case.
    f1 = f1.where((pooled["tp"] + pooled["fn"]) > 0, np.nan)
    count = grouped.size()
    # Drop sparse bins (too few tiles to be meaningful) from both panels.
    enough = count >= min_count
    f1 = f1.where(enough, np.nan)
    count = count.where(enough, np.nan)

    # Visual bin extents: the catch-all bin is drawn one bin_km wide past the cap.
    left_edges = np.append(reg_edges[:-1], cap_km)
    right_edges = np.append(reg_edges[1:], cap_km + bin_km)
    centers_km = (left_edges + right_edges) / 2

    ax_f1.plot(centers_km, f1.values, marker="o", color="tab:blue", zorder=3)
    ax_f1.set_ylabel("F1 score")
    ax_f1.set_ylim(0, 1.05)
    ax_f1.grid(axis="y", alpha=0.3)

    ax_count.bar(centers_km, count.values, width=bin_km * 0.9,
                 color="0.6", edgecolor="white", zorder=3)
    ax_count.set_ylabel("Tile count")
    ax_count.set_xlabel(xlabel)
    ax_count.grid(axis="y", alpha=0.3)

    # Ticks at the regular bin edges only; clip to the range of populated bins.
    kept = np.flatnonzero(enough.values)
    left, right = left_edges[kept[0]], right_edges[kept[-1]]
    ticks = [e for e in reg_edges if left <= e <= right]
    labels = [f'{int(e)}' for e in ticks]
    # Label the right edge of the catch-all bin with the actual maximum
    # distance present in the data, rather than a ">cap" bin.
    if right > cap_km:
        ticks.append(right_edges[-1])
        labels.append(f'{int(round(dist_km.max()))}\n(max)')
    ax_count.set_xticks(ticks)
    ax_count.set_xticklabels(labels)
    ax_count.set_xlim(left, right)


def plot_distance_curve(df, title, save_path=None, bin_width=20000, min_count=1,
                        cap=200000):
    """Plot a micro-averaged F1 vs distance-to-nearest-training-tile curve.

    See ``_draw_distance_panels`` for the binning/pooling details. The lower
    panel shows the per-bin tile count.
    """
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(7, 5),
                            gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(title, fontsize=13, fontweight="bold")
    _draw_distance_panels(df, axs[0], axs[1],
                          "Distance to nearest training tile (km)",
                          bin_width=bin_width, min_count=min_count, cap=cap)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        fig.show()
    return fig


def plot_distance_figure(per_image_df, label, locations_csv=LOCATIONS_CSV,
                         fig_dir=None, proj_crs=DIST_PROJ_CRS,
                         bin_width=20000, min_count=1):
    """One publication figure per sensor: test-tile F1 vs distance to the nearest
    *training* tile (left column) and to the nearest *validation* tile (right
    column). Each column is an F1 curve over a tile-count panel.

    Panels carry no titles; they are tagged ``(a)``-``(d)`` in their top-left
    corners, in column-major order (top-left, bottom-left, top-right, bottom-
    right). Saved as ``distance_figure_{label}.jpg`` when ``fig_dir`` is given.
    """
    _, test_df, test_val_df = build_distance_dfs(per_image_df, locations_csv, proj_crs)

    fig, axs = plt.subplots(2, 2, sharex="col", figsize=(12, 5),
                            gridspec_kw={"height_ratios": [3, 1]})
    columns = [
        (test_df, "Distance to nearest training tile (km)", 100000),
        (test_val_df, "Distance to nearest validation tile (km)", 200000),
    ]
    for col, (df, xlabel, cap) in enumerate(columns):
        _draw_distance_panels(df, axs[0, col], axs[1, col], xlabel,
                              bin_width=bin_width, min_count=min_count, cap=cap)

    # Tag panels (a)-(d) in column-major order: top-left, bottom-left, top-right,
    # bottom-right.
    panel_axes = [axs[0, 0], axs[1, 0], axs[0, 1], axs[1, 1]]
    for ax, tag in zip(panel_axes, 'abcd'):
        ax.text(0.02, 0.95, f'({tag})', transform=ax.transAxes,
                ha='left', va='top', fontweight='bold', fontsize=12)

    fig.tight_layout()
    if fig_dir is not None:
        os.makedirs(fig_dir, exist_ok=True)
        slug = label.lower().replace(' ', '_')
        save_path = os.path.join(fig_dir, f'distance_figure_{slug}.jpg')
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  wrote {save_path}')
    else:
        fig.show()
    return fig

def compute_stats(true, preds, cutoff):
    preds_binary = torch.Tensor(preds>cutoff).long()
    tp, fp, fn, tn = smp.metrics.get_stats(
        preds_binary,
        true,
        mode="binary")

    iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    prec = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")

    return np.array([iou, f1, prec, recall])

def find_best_cutoff(preds, masks, cutoffs, max_iou_mode=False):
    """Search ``cutoffs`` for the best pixel-wise threshold against ``masks``.

    If ``max_iou_mode`` is True the cutoff maximizing IoU is chosen, otherwise
    the cutoff that best balances precision and recall (minimizes their
    absolute difference). Returns the median cutoff among all that tie the
    selected cutoff's IoU (matches the original threshold-calc behavior).

    Args:
        preds: float prediction array (n, h, w).
        masks: ground-truth/baseline mask array or tensor (n, h, w).
        cutoffs: 1-D array of candidate thresholds.
        max_iou_mode: select by max IoU (True) or balanced P/R (False).
    """
    cutoffs = np.asarray(cutoffs)
    masks = torch.as_tensor(np.asarray(masks)).long()
    all_stats = np.vstack([compute_stats(masks, preds, c) for c in cutoffs])
    if max_iou_mode:
        best_index = np.argmax(all_stats[:, 0])
    else:
        best_index = np.argmin(np.abs(all_stats[:, 2] - all_stats[:, 3]))
    return np.median(cutoffs[np.where(all_stats[:, 0] == all_stats[best_index, 0])[0]])

def get_objects(ar, pred_thresh=0.5):
    labeled_ar, num_objects = ndimage.label(ar>pred_thresh)
    return labeled_ar, num_objects

def calculate_iou(maska, maskb):
  # Calculates the Intersection over Union (IoU) of two bounding boxes.
  top = np.sum(maska * maskb)
  bottom = np.sum(np.max([maska, maskb], axis=0))
  return top/bottom

def per_image_stats(preds, truth, img_dir, best_cutoff=0.5):
    preds_binary = torch.Tensor(preds>best_cutoff).long()
    tp, fp, fn, tn = smp.metrics.get_stats(
        preds_binary,
        truth,
        mode="binary")
    ids = sorted(os.listdir(img_dir))
    out_df = pd.DataFrame(
        {'basename': [n[:-4] for n in ids],
         'tp': tp.sum(axis=1),
         'fp': fp.sum(axis=1),
         'fn': fn.sum(axis=1),
         'tn': tn.sum(axis=1)}
    )
    return out_df


def object_stats(truth_masks, pred_masks, pred_thresh=0.5):
    total_truth_objects = 0
    total_pred_objects = 0
    total_truth_area = 0
    total_pred_area = 0
    max_pred_size = 0
    max_truth_size = 0
    min_truth_size = 1000

    for i in range(len(truth_masks)):
        labeled_truth, max_truth = get_objects(truth_masks[i])
        labeled_pred, max_pred = get_objects(pred_masks[i], pred_thresh)

        if max_truth > 0:
            # First, filter only to truth objects within threshold
            for j in range(1, max_truth+1):
                total_truth_objects += 1
                mask_truth = (labeled_truth==j)
                truth_size = mask_truth.sum()
                total_truth_area += truth_size
                if truth_size > max_truth_size:
                    max_truth_size = truth_size
                if truth_size < min_truth_size:
                    min_truth_size = truth_size
        if max_pred > 0:
            for k in range(1, max_pred+1):
                total_pred_objects += 1
                mask_pred = (labeled_pred==k)
                pred_size = mask_pred.sum()
                total_pred_area += pred_size
                if pred_size > max_pred_size:
                    max_pred_size = pred_size
                if mask_pred.sum() > max_pred_size:
                    max_pred_size = mask_pred.sum()
    out_dict = {
        'total_truth_objects': total_truth_objects,
        'total_truth_area': total_truth_area,
        'min_truth_size': min_truth_size,
        'max_truth_size': max_truth_size,
        'total_pred_objects': total_pred_objects,
        'total_pred_area': total_pred_area,
        'max_pred_size': max_pred_size,
    }
    return out_dict

def get_size_stats_dict(truth_masks, pred_masks, pred_thresh):
    all_sizes_preds = []
    pred_assessment = []
    all_sizes_truth = []
    truth_assessment = []
    for i in range(len(truth_masks)):
        labeled_truth, max_truth = get_objects(truth_masks[i])
        labeled_pred, max_pred = get_objects(pred_masks[i], pred_thresh)
        for j in range(1, max_truth+1):
            mask_truth = (labeled_truth==j)
            all_sizes_truth.append(mask_truth.sum())
            if np.max(labeled_pred * mask_truth) > 0:
                truth_assessment.append(1) # TP
                pred_assessment.append(1) # TP
                # Find the max overlap
                max_val = 0
                max_overlap = 0
                for val in np.unique(labeled_pred[(labeled_pred>0)&(mask_truth>0)]):
                    overlap = np.sum((labeled_pred==val)*(mask_truth>0))
                    if overlap > max_overlap:
                        max_val = val
                        max_overlap = overlap
                all_sizes_preds.append(np.sum(labeled_pred==max_val))
                # Remove that pred from the pool
                labeled_pred[labeled_pred==max_val] = 0
            else:
                truth_assessment.append(0) # FN

        for k in np.unique(labeled_pred):
            if k != 0:
                pred_assessment.append(0) # FPrange(1, max_pred+1):
                mask_pred = (labeled_pred==k)
                all_sizes_preds.append(mask_pred.sum())

    return {
        'truth_sizes': all_sizes_truth,
        'truth_assessment': truth_assessment,
        'pred_sizes': all_sizes_preds,
        'pred_assessment': pred_assessment
    }


def process_size_stats(true_df, pred_df, size_dict):
    all_dicts = []
    for size_class, sizes in size_dict.items():
        temp_true = true_df.loc[(true_df['size']>sizes[0])&(true_df['size']<=sizes[1])]
        temp_pred = pred_df.loc[(pred_df['size']>sizes[0])&(pred_df['size']<=sizes[1])]
        out_dict = {
            'size_class': size_class,
            'total_true':temp_true.shape[0],
            'total_pred': temp_pred.shape[0],
            'tp_true': temp_true['tp'].sum().item(),
            'tp_pred': temp_pred['tp'].sum().item()
        }
        out_dict['precision'] = min(out_dict['tp_true'], out_dict['tp_pred']) / temp_pred.shape[0] if temp_pred.shape[0]>0 else 0
        out_dict['recall'] = min(out_dict['tp_true'], out_dict['tp_pred']) / temp_true.shape[0] if temp_true.shape[0]>0 else 0
        print(out_dict)
        all_dicts.append(out_dict)
    # Overall precision and recall
    smallest_size = list(size_dict.keys())[0]
    largest_size = list(size_dict.keys())[-1]


    temp_true = true_df.loc[(true_df['size']>size_dict[smallest_size][1])
                            &(true_df['size']<=size_dict[largest_size][0])]
    temp_pred = pred_df.loc[(pred_df['size']>size_dict[smallest_size][1])
                            &(pred_df['size']<=size_dict[largest_size][0])]
    out_dict = {
        'size_class': 'all',
        'total_true':temp_true.shape[0],
        'total_pred': temp_pred.shape[0],
        'tp_true': temp_true['tp'].sum().item(),
        'tp_pred': temp_pred['tp'].sum().item()
        }
    out_dict['precision'] = min(out_dict['tp_true'], out_dict['tp_pred']) / temp_pred.shape[0] if temp_pred.shape[0]>0 else 0
    out_dict['recall'] = min(out_dict['tp_true'], out_dict['tp_pred']) / temp_true.shape[0] if temp_true.shape[0]>0 else 0
    print(out_dict)
    all_dicts.append(out_dict)
    full_df = pd.DataFrame(all_dicts)
    return full_df


def size_stats(truth_masks, pred_masks, size_min, size_max, pred_thresh=0.5):
    total_truth_objects = 0
    truth_objects_detected = 0
    total_pred_objects = 0
    pred_objects_false = 0
    pred_objects_true = 0

    for i in range(len(truth_masks)):
        labeled_truth, max_truth = get_objects(truth_masks[i])
        labeled_pred, max_pred = get_objects(pred_masks[i], pred_thresh)

        if max_pred > 0:
            labeled_pred_temp = labeled_pred.copy()
            # First, filter only to truth objects within threshold
            for j in range(1, max_truth+1):
                mask_truth = (labeled_truth==j)
                if size_min < mask_truth.sum() <= size_max:
                    total_truth_objects += 1
            for k in range(1, max_pred+1):
                mask_pred = (labeled_pred==k)
                if size_min < mask_pred.sum() <= size_max:
                    total_pred_objects += 1
                    if np.max(labeled_truth * mask_pred) == 0:
                        pred_objects_false += 1
                    else:
                        pred_objects_true += 1
                        # Find the max overlap
                        max_val = 0
                        max_overlap = 0
                        for val in np.unique(labeled_truth[(labeled_truth>0)&(mask_pred>0)]):
                            overlap = np.sum((labeled_truth==val)*(mask_pred>0))
                            if overlap > max_overlap:
                                max_val = val
                                max_overlap = overlap
                        labeled_truth[labeled_truth==max_val] = 0
        else:
            for j in range(1, max_truth+1):
                mask_truth = (labeled_truth==j)
                if size_min < mask_truth.sum() <= size_max:
                    total_truth_objects += 1


    out_dict = {
        'Total Truth': total_truth_objects,
        'Total Pred': total_pred_objects,
        'TP': pred_objects_true,
        'FP': pred_objects_false
    }
    out_dict['recall'] = out_dict['TP'] / (out_dict['Total Truth'])#out_dict['TP'] + out_dict['FN'])
    out_dict['precision'] = out_dict['TP'] / (out_dict['Total Pred'])
    return out_dict


def pr_curve(preds_path, truth_path):
    preds = np.load(preds_path).flatten()
    truth = np.load(truth_path).flatten()
    return precision_recall_curve(truth, preds)


def report_object_stats(truth, preds, best_cutoff):
    """Print object-level and size-binned precision/recall for one prediction set.

    Args:
        truth: ground-truth mask array/tensor (n, h, w).
        preds: float prediction array (n, h, w).
        best_cutoff: probability threshold used to binarize ``preds``.
    """
    print('Object stats', object_stats(truth, preds, pred_thresh=best_cutoff))

    size_stats_dict = get_size_stats_dict(truth, preds, pred_thresh=best_cutoff)
    pred_df = pd.DataFrame({'size': np.array(size_stats_dict['pred_sizes']),
                            'tp': np.array(size_stats_dict['pred_assessment'])})
    true_df = pd.DataFrame({'size': np.array(size_stats_dict['truth_sizes']),
                            'tp': np.array(size_stats_dict['truth_assessment'])})
    full_df = process_size_stats(true_df, pred_df, SIZE_DICT)
    return full_df


def full_evaluation(preds_path, truth_path, img_dir, per_image_csv, find_cutoff=True, best_cutoff=0.5, crop_to_400=False):
    preds = np.load(preds_path)
    truth = torch.Tensor(np.load(truth_path)).long()
    if crop_to_400:
        preds = preds[:, 50:450, 50:450]
        truth = truth[:, 50:450, 50:450]

    # Basic pixel-wise stats
    if find_cutoff:
        best_cutoff = find_best_cutoff(preds, truth, np.arange(0, 1.0, 0.01), max_iou_mode=True)
    print('Using cutoff:', best_cutoff)
    print('Final pixel-wise stats:', compute_stats(truth, preds, cutoff=best_cutoff))

    # Object-level and size-binned stats
    object_df = report_object_stats(truth, preds, best_cutoff)
    object_df.to_csv(per_image_csv.replace('per_image.csv', 'object_stats.csv'), index=False)

    per_image_stats(preds, truth, img_dir, best_cutoff=best_cutoff).to_csv(per_image_csv, index=False)
