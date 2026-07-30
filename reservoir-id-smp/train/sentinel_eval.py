"""Evaluate the Sentinel reservoir-segmentation predictions.

Runs the full pixel-wise + object-level + size-binned evaluation (via
``_eval_helpers.full_evaluation``) on the Sentinel train/val/test predictions
written by ``predict_train_val_test.py``, writes the per-image stat CSVs, and
saves the training-progress / precision-recall figure.

Landsat evaluation now lives in ``landsat_eval.py`` (with cutoff calibration in
``landsat_threshold_calcs.py``).

Pass ``--tune-threshold`` to calibrate the cutoff on the val split (instead of
the hardcoded ``BEST_CUTOFF``) before evaluating both splits
``--max-iou`` picks the cutoff that maximizes IoU instead of the
balancing precision/recall (default).
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')  # script context: render figures to files, no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import _eval_helpers as eval_helpers


PRED_DIR = './data/preds'
IMG_ROOT = './data/reservoirs_10band'
PRED_PREFIX = 'reservoirs_10band_manet_datav12_modelv6'
MASK_PREFIX = 'reservoirs_10band_masks'
BEST_CUTOFF = 0.62

TRAIN_METRICS_CSV = './data/train_csvs/manet_resnet_datav12_modelv6_metrics.csv'
FIG_DIR = '/home/ksolvik/research/reservoirs/figs/ch0'

SPLITS = ['val', 'test']


def tune_threshold(max_iou_mode=False):
    """Calibrate the cutoff on the val split's preds vs masks."""
    preds_path = os.path.join(PRED_DIR, f'{PRED_PREFIX}_val.npy')
    masks_path = os.path.join(PRED_DIR, f'{MASK_PREFIX}_val.npy')
    preds = np.load(preds_path)
    masks = torch.Tensor(np.load(masks_path)).long()
    cutoff = eval_helpers.find_best_cutoff(preds, masks, np.arange(0, 1.0, 0.01), max_iou_mode)
    print(f'Tuned cutoff (vs val masks): {cutoff}')
    return cutoff


def evaluate_split(split, best_cutoff=BEST_CUTOFF):
    """Full evaluation for one split; writes {split}_per_image.csv."""
    preds_path = os.path.join(PRED_DIR, f'{PRED_PREFIX}_{split}.npy')
    masks_path = os.path.join(PRED_DIR, f'{MASK_PREFIX}_{split}.npy')
    img_dir = os.path.join(IMG_ROOT, f'img_dir/{split}')
    per_image_csv = os.path.join(PRED_DIR, f'{split}_per_image.csv')

    if not (os.path.exists(preds_path) and os.path.exists(masks_path)):
        print(f'[skip] {split}: missing preds or masks')
        return

    print(f'\n===== {split} =====')
    eval_helpers.full_evaluation(
        preds_path, masks_path, img_dir, per_image_csv,
        find_cutoff=False, best_cutoff=best_cutoff, crop_to_400=False,
    )


def plot_training_and_pr_curves():
    """Save the combined training-progress + precision-recall figure."""
    val_preds = os.path.join(PRED_DIR, f'{PRED_PREFIX}_val.npy')
    val_masks = os.path.join(PRED_DIR, f'{MASK_PREFIX}_val.npy')
    test_preds = os.path.join(PRED_DIR, f'{PRED_PREFIX}_test.npy')
    test_masks = os.path.join(PRED_DIR, f'{MASK_PREFIX}_test.npy')
    needed = [TRAIN_METRICS_CSV, val_preds, val_masks, test_preds, test_masks]
    missing = [p for p in needed if not os.path.exists(p)]
    if missing:
        print(f'[skip figure] missing {missing}')
        return

    training_stats = pd.read_csv(TRAIN_METRICS_CSV)
    per_epoch = training_stats.groupby('epoch').mean()
    to_plot = per_epoch[['train_loss', 'valid_loss']].rename(
        columns={
            'train_loss': 'Training',
            'valid_loss': 'Validation',
        }
    )

    prec_recall_val = eval_helpers.pr_curve(val_preds, val_masks)
    prec_recall_test = eval_helpers.pr_curve(test_preds, test_masks)

    fig, axs = plt.subplots(1, 2, figsize=(7.35, 3), constrained_layout=True)
    colors = ['tab:blue', 'tab:red']
    linestyles = ['-', '--', '-', '--']

    for col, color, ls in zip(to_plot.columns, colors, linestyles):
        axs[0].plot(to_plot.index, to_plot[col], color=color, linestyle=ls, label=col, lw=1.2)
    axs[0].annotate('A', xy=(1, 1), xycoords='axes fraction',
                    xytext=(-1.3, -1.3), textcoords='offset fontsize',
                    fontsize=12, verticalalignment='bottom', fontfamily='serif')
    axs[0].legend(loc='upper right', bbox_to_anchor=(0.94, 1.01))
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_ylim(0, 1.05)

    axs[1].plot(prec_recall_val[0], prec_recall_val[1], label='Validation', color='tab:red')
    axs[1].plot(prec_recall_test[0], prec_recall_test[1], label='Test', color='tab:orange')
    axs[1].annotate('B', xy=(1, 1), xycoords='axes fraction',
                    xytext=(-1.3, -1.3), textcoords='offset fontsize',
                    fontsize=12, verticalalignment='bottom', fontfamily='serif')
    axs[1].legend()
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')

    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, 'training_progress_curve.svg'), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, 'training_progress_curve.jpg'), dpi=300,
                pil_kwargs={'quality': 95}, bbox_inches='tight')
    print(f'wrote figures to {FIG_DIR}')


def plot_distance_curves():
    """F1 vs distance-to-nearest-training-tile for the val and test splits.

    Concatenates the per-image stat CSVs written by ``evaluate_split`` (across
    all splits, so training tiles contribute their positions) and merges them
    onto the annotation locations before plotting. Figures are saved to FIG_DIR.
    """
    per_image_paths = [os.path.join(PRED_DIR, f'{split}_per_image.csv') for split in SPLITS]
    existing = [p for p in per_image_paths if os.path.exists(p)]
    if not existing:
        print('[skip distance curves] no per-image CSVs found')
        return
    per_image = pd.concat([pd.read_csv(p) for p in existing], ignore_index=True)
    print('\n===== distance curves =====')
    eval_helpers.plot_distance_curves(per_image, label='Sentinel', fig_dir=FIG_DIR)
    eval_helpers.plot_distance_figure(per_image, label='Sentinel', fig_dir=FIG_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tune-threshold', action='store_true',
                        help='Calibrate the cutoff on the val split instead of using BEST_CUTOFF.')
    parser.add_argument('--max-iou', action='store_true',
                        help='With --tune-threshold, pick the cutoff that maximizes IoU '
                             'instead of the default balanced precision/recall criterion.')
    return parser.parse_args()


def main():
    args = parse_args()
    best_cutoff = tune_threshold(args.max_iou) if args.tune_threshold else BEST_CUTOFF

    for split in SPLITS:
        evaluate_split(split, best_cutoff=best_cutoff)
    plot_training_and_pr_curves()
    plot_distance_curves()


if __name__ == '__main__':
    main()
