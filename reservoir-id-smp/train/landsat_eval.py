#!/usr/bin/env python
"""Evaluate Landsat reservoir detections using the saved per-sensor cutoffs.

Loads the calibrated cutoffs written by ``landsat_threshold_calcs.py``
(``data/preds/landsat_cutoffs.json``) and the quantized prediction arrays from
``predict_train_val_test.py`` (``{sensor}_preds_{split}_quant.npy`` plus the
shared ``{split}_masks.npy``), then reports pixel-wise and object-level /
size-binned precision/recall for the val and test splits. Per-image stat CSVs
are written alongside the predictions.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')  # script context: render figures to files, no display
import pandas as pd
import torch

import _eval_helpers as eval_helpers
from landsat_threshold_calcs import (
    CUTOFFS_JSON,
    PRED_DIR,
    SENSOR_CUTOFF,
    SENSORS,
    load_npy,
    load_preds,
)


SPLITS = ['val', 'test']

# Distance-vs-F1 curves are saved here, one figure per sensor and split.
FIG_DIR = os.path.join(PRED_DIR, 'distance_curves')

# Image dirs used only to recover per-image basenames for the per-image CSVs.
DATA_DIRS = {
    'ls5_2010': './data/landsat5_2010_v9_sr',
    'ls7_2010': './data/landsat7_2010_v9_sr',
    'ls7_2017': './data/landsat7_2017_v9_sr',
    'ls8_2017': './data/landsat8_2017_v9_sr',
    'ls8_2022': './data/landsat8_2022_v9_sr',
    'ls9_2022': './data/landsat9_2022_v9_sr',
    'ls8_2017_bilinear': './data/landsat8_2017_v9_bilinear',
    'ls8_2017_30m': './data/landsat8_2017_v9_30m',
}

# Most sensors share the 500x500 masks; the native 30m grid has its own
# 166x166 masks written by predict_train_val_test.py as {split}_masks_30m.npy.
SENSOR_MASKS = {'ls8_2017_30m': '{split}_masks_30m.npy'}


def masks_path(sensor, split):
    template = SENSOR_MASKS.get(sensor, '{split}_masks.npy')
    return os.path.join(PRED_DIR, template.format(split=split))


def load_cutoffs(path=CUTOFFS_JSON):
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found; run landsat_threshold_calcs.py first')
    with open(path) as f:
        return json.load(f)


def report_pixel_stats(preds, cutoffs, split, get_masks):
    """Pixel-wise [IoU, F1, precision, recall] for each sensor vs its masks."""
    print(f'\n--- {split}: pixel-wise stats vs masks ---')
    for s in SENSORS:
        p, c = preds[s], cutoffs.get(SENSOR_CUTOFF[s])
        masks_t = get_masks(s)
        if p is None or c is None or masks_t is None:
            print(f'{s}: skipped (missing preds, cutoff, or masks)')
            continue
        print(f'{s}: {eval_helpers.compute_stats(masks_t, p, c)}')


def report_object_stats(preds, cutoffs, split, get_masks):
    """Per-object + size-binned stats and per-image CSVs for each sensor."""
    print(f'\n=== {split}: object-level stats ===')
    for s in SENSORS:
        p, c = preds[s], cutoffs.get(SENSOR_CUTOFF[s])
        masks_t = get_masks(s)
        if p is None or c is None or masks_t is None:
            continue
        print(f'\n--- {s} ({split}) @ cutoff {c} ---')
        object_df = eval_helpers.report_object_stats(masks_t, p, c)
        object_csv = os.path.join(PRED_DIR, f'{s}_{split}_object_stats.csv')
        object_df.to_csv(object_csv, index=False)
        print(f'  wrote {object_csv}')

        img_dir = os.path.join(DATA_DIRS[s], f'img_dir/{split}')
        if os.path.isdir(img_dir):
            out_csv = os.path.join(PRED_DIR, f'{s}_{split}_per_image.csv')
            eval_helpers.per_image_stats(p, masks_t, img_dir, best_cutoff=c).to_csv(out_csv, index=False)
            print(f'  wrote {out_csv}')


def evaluate_split(split, cutoffs):
    preds = load_preds(split)

    # Cache masks per file so sensors sharing the 500x500 masks load them once,
    # while the 30m variant transparently picks up its own {split}_masks_30m.npy.
    masks_cache = {}

    def get_masks(sensor):
        path = masks_path(sensor, split)
        if path not in masks_cache:
            arr = load_npy(path)
            masks_cache[path] = None if arr is None else torch.Tensor(arr).long()
        return masks_cache[path]

    report_pixel_stats(preds, cutoffs, split, get_masks)
    report_object_stats(preds, cutoffs, split, get_masks)


def plot_distance_curves():
    """Per-sensor F1 vs distance-to-nearest-training-tile for val and test.

    Uses the per-image stat CSVs written by ``report_object_stats`` (one per
    sensor and split) merged onto the shared annotation locations. Tile
    positions for the training split come from the locations file, so no
    training predictions are required.
    """
    print('\n=== distance curves ===')
    for s in SENSORS:
        paths = [os.path.join(PRED_DIR, f'{s}_{split}_per_image.csv') for split in SPLITS]
        existing = [p for p in paths if os.path.exists(p)]
        if not existing:
            continue
        per_image = pd.concat([pd.read_csv(p) for p in existing], ignore_index=True)
        eval_helpers.plot_distance_curves(per_image, label=s, fig_dir=FIG_DIR)
        eval_helpers.plot_distance_figure(per_image, label=s, fig_dir=FIG_DIR)


def main():
    cutoffs = load_cutoffs()
    print('Loaded cutoffs:', cutoffs)
    for split in SPLITS:
        evaluate_split(split, cutoffs)
    plot_distance_curves()


if __name__ == '__main__':
    main()
