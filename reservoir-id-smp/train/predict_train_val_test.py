#!/usr/bin/env python
"""Run and save post-training predictions for the reservoir segmentation models.

Unified prediction step for both the Sentinel model and every Landsat
sensor/year model. For each entry in ``CONFIGS`` and each split in ``SPLITS``
it loads the checkpoint, predicts on the split, and writes the prediction
arrays (and, once per split, the ground-truth masks) to ``PRED_DIR`` as ``.npy``.

Landsat entries additionally produce quantized (neural_compressor) predictions,
which are what ``landsat_threshold_calcs.py`` and ``landsat_eval.py`` consume.

Edit the flags and ``CONFIGS`` table below, then run:

    python predict_train_val_test.py

Configs whose checkpoint / data_dir / mean_std are missing on disk are skipped
with a warning, so the loop is safe to run even when only some sensors' data
is available locally.
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from neural_compressor.utils.pytorch import load
from neural_compressor.config import (
    PostTrainingQuantConfig,
    TuningCriterion,
    AccuracyCriterion,
)
from neural_compressor.quantization import fit

from _helper_datasets import Dataset, DatasetImageOnly, get_preprocessing
from _helper_model import ResModel


# ---------------------------------------------------------------------------
# Run settings
# ---------------------------------------------------------------------------
RUN_LIST = ['ls8_2017_30m']
SPLITS = ['val', 'test']      # splits to predict on (e.g. add 'train')
SAVE_MASKS = True             # write ground-truth masks once per split
FIT_QUANTIZE = False          # re-fit the quantized Landsat model (slow); else load
PRED_DIR = './data/preds'
QUANT_MODEL_DIR = './models/best/quantized_model_l8/'  # shared Landsat quant model
QUANT_FIT_SPLIT = 'val'       # split used to calibrate/eval when FIT_QUANTIZE

BATCH_SIZE = 4
NUM_WORKERS = 2
ENCODER_NAME = 'resnet34'

# Shared Landsat checkpoint, applied to each sensor's imagery with a
# sensor-specific mean_std for normalization.
SENTINEL_CKPT = './models/best/sentinel_datav12_modelv6.ckpt'
LANDSAT_CKPT = './models/best/l8_sr_v21.ckpt'

# ---------------------------------------------------------------------------
# Per-model configuration. Output templates use {split}. Paths are relative to
# the train/ directory; adjust data_dir / mean_std to match your local layout.
# ---------------------------------------------------------------------------
CONFIGS = {
    'sentinel': {
        'checkpoint': SENTINEL_CKPT,
        'data_dir': './data/reservoirs_10band',
        'mean_std': './data/mean_stds/mean_std_sentinel_v12.npy',
        'in_channels': 10,
        'center_crop': 500,
        'quantize': False,
        'og_out': 'reservoirs_10band_manet_datav12_modelv6_{split}.npy',
        'quant_out': None,
        'masks_out': 'reservoirs_10band_masks_{split}.npy',
    },
    'ls8_2017': {
        'checkpoint': LANDSAT_CKPT,
        'data_dir': './data/landsat8_2017_v9_sr',
        'mean_std': './data/mean_stds/mean_std_ls8_v9.npy',
        'in_channels': 6,
        'center_crop': 500,
        'quantize': True,
        'og_out': 'ls8_2017_preds_{split}_og.npy',
        'quant_out': 'ls8_2017_preds_{split}_quant.npy',
        'masks_out': '{split}_masks.npy',
    },
    # Resampling-method comparisons against the ls8_2017 baseline (nearest-neighbor
    # 'sr'). These reuse the LS8 cutoff, so they are not calibrated in
    # landsat_threshold_calcs.py. 'bilinear' shares the 500x500 masks; the native
    # 30m grid has a smaller 166x166 mask, written to its own {split}_masks_30m.npy.
    'ls8_2017_bilinear': {
        'checkpoint': LANDSAT_CKPT,
        'data_dir': './data/landsat8_2017_v9_bilinear',
        'mean_std': './data/mean_stds/mean_std_ls8_2017_v9_bilinear.npy',
        'in_channels': 6,
        'center_crop': 500,
        'quantize': True,
        'og_out': 'ls8_2017_bilinear_preds_{split}_og.npy',
        'quant_out': 'ls8_2017_bilinear_preds_{split}_quant.npy',
        'masks_out': '{split}_masks.npy',
    },
    'ls8_2017_30m': {
        'checkpoint': './models/best/ls8_30m_v2.ckpt',
        'data_dir': './data/landsat8_2017_v9_30m',
        'mean_std': './data/mean_stds/mean_std_ls8_2017_v9_30m.npy',
        'in_channels': 6,
        'center_crop': 166,
        'quantize': False,
        'og_out': 'ls8_2017_30m_preds_{split}_og.npy',
        'quant_out': 'ls8_2017_30m_preds_{split}_quant.npy',
        'masks_out': '{split}_masks_30m.npy',
    },
    'ls8_2022': {
        'checkpoint': LANDSAT_CKPT,
        'data_dir': './data/landsat8_2022_v9_sr',
        'mean_std': './data/mean_stds/mean_std_ls8_v9.npy',
        'in_channels': 6,
        'center_crop': 500,
        'quantize': True,
        'og_out': 'ls8_2022_preds_{split}_og.npy',
        'quant_out': 'ls8_2022_preds_{split}_quant.npy',
        'masks_out': '{split}_masks.npy',
    },
    'ls9_2022': {
        'checkpoint': LANDSAT_CKPT,
        'data_dir': './data/landsat9_2022_v9_sr',
        'mean_std': './data/mean_stds/mean_std_ls9_v9.npy',
        'in_channels': 6,
        'center_crop': 500,
        'quantize': True,
        'og_out': 'ls9_2022_preds_{split}_og.npy',
        'quant_out': 'ls9_2022_preds_{split}_quant.npy',
        'masks_out': '{split}_masks.npy',
    },
    'ls7_2017': {
        'checkpoint': LANDSAT_CKPT,
        'data_dir': './data/landsat7_2017_v9_sr',
        'mean_std': './data/mean_stds/mean_std_ls7_v9.npy',
        'in_channels': 6,
        'center_crop': 500,
        'quantize': True,
        'og_out': 'ls7_2017_preds_{split}_og.npy',
        'quant_out': 'ls7_2017_preds_{split}_quant.npy',
        'masks_out': '{split}_masks.npy',
    },
    'ls7_2010': {
        'checkpoint': LANDSAT_CKPT,
        'data_dir': './data/landsat7_2010_v9_sr',
        'mean_std': './data/mean_stds/mean_std_ls7_v9.npy',
        'in_channels': 6,
        'center_crop': 500,
        'quantize': True,
        'og_out': 'ls7_2010_preds_{split}_og.npy',
        'quant_out': 'ls7_2010_preds_{split}_quant.npy',
        'masks_out': '{split}_masks.npy',
    },
    'ls5_2010': {
        'checkpoint': LANDSAT_CKPT,
        'data_dir': './data/landsat5_2010_v9_sr',
        'mean_std': './data/mean_stds/mean_std_ls5_v9.npy',
        'in_channels': 6,
        'center_crop': 500,
        'quantize': True,
        'og_out': 'ls5_2010_preds_{split}_og.npy',
        'quant_out': 'ls5_2010_preds_{split}_quant.npy',
        'masks_out': '{split}_masks.npy',
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_model(cfg):
    """Load a ResModel from a checkpoint for prediction (weights on CPU)."""
    return ResModel.load_from_checkpoint(
        cfg['checkpoint'],
        in_channels=cfg['in_channels'],
        out_classes=1,
        arch='',
        center_crop=cfg['center_crop'],
        encoder_name=ENCODER_NAME,
        map_location=torch.device('cpu'),
    )


def make_dataset(cfg, mean_std, img_dir, ann_dir, image_only=False):
    cls = DatasetImageOnly if image_only else Dataset
    return cls(
        img_dir,
        ann_dir,
        preprocessing=get_preprocessing(),
        classes=['water'],
        mean_std=mean_std,
    )


def predict(model, loader, trainer):
    """Return stacked sigmoid predictions of shape (n, h, w)."""
    return np.vstack(trainer.predict(model, loader))[:, 0, :, :]


def compute_masks(dataset):
    """Stack ground-truth masks from a (with-mask) Dataset into (n, h, w)."""
    return np.vstack([dataset[i]['mask'] for i in range(len(dataset))])


def make_quant_model(cfg, mean_std, trainer, fit_quant=False):
    """Build a Landsat model with quantized weights loaded from QUANT_MODEL_DIR.

    When ``fit_quant`` is True the quantized model is first re-fit with
    neural_compressor using the QUANT_FIT_SPLIT data, then saved. A separate
    model instance is returned so the original (float) model is never mutated.
    """
    qmodel = build_model(cfg)

    if fit_quant:
        img_dir = os.path.join(cfg['data_dir'], f'img_dir/{QUANT_FIT_SPLIT}')
        ann_dir = os.path.join(cfg['data_dir'], f'ann_dir/{QUANT_FIT_SPLIT}')
        calib_ds = make_dataset(cfg, mean_std, img_dir, ann_dir, image_only=True)
        calib_loader = DataLoader(calib_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        mask_ds = make_dataset(cfg, mean_std, img_dir, ann_dir)
        mask_loader = DataLoader(mask_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        masks = torch.Tensor(compute_masks(mask_ds)).long()

        def eval_func(candidate_model):
            qmodel.model = candidate_model
            preds = predict(qmodel, mask_loader, trainer)
            tp, fp, fn, tn = smp.metrics.get_stats(
                torch.Tensor(preds > 0.5).long(), masks, mode="binary")
            return float(smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro"))

        conf = PostTrainingQuantConfig(
            approach="auto",
            backend="default",
            tuning_criterion=TuningCriterion(max_trials=5),
            accuracy_criterion=AccuracyCriterion(tolerable_loss=0.01),
        )
        q_model = fit(model=build_model(cfg).model, conf=conf,
                      calib_dataloader=calib_loader, eval_func=eval_func)
        q_model.save(QUANT_MODEL_DIR)

    qmodel.model = load(QUANT_MODEL_DIR, qmodel.model)
    return qmodel


def missing_paths(cfg):
    return [p for p in (cfg['checkpoint'], cfg['data_dir'], cfg['mean_std'])
            if not os.path.exists(p)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(PRED_DIR, exist_ok=True)
    saved_masks = set()  # mask output paths already written this run

    for name in RUN_LIST:
        cfg = CONFIGS[name]
        missing = missing_paths(cfg)
        if missing:
            print(f'[skip] {name}: missing {missing}')
            continue

        print(f'[run]  {name}')
        mean_std = np.load(cfg['mean_std'])
        model = build_model(cfg)
        trainer = pl.Trainer()

        qmodel = None
        if cfg['quantize']:
            qmodel = make_quant_model(cfg, mean_std, trainer, fit_quant=FIT_QUANTIZE)

        for split in SPLITS:
            img_dir = os.path.join(cfg['data_dir'], f'img_dir/{split}')
            ann_dir = os.path.join(cfg['data_dir'], f'ann_dir/{split}')
            if not os.path.isdir(img_dir):
                print(f'  [skip] {name}/{split}: no {img_dir}')
                continue

            ds = make_dataset(cfg, mean_std, img_dir, ann_dir)
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

            og_path = os.path.join(PRED_DIR, cfg['og_out'].format(split=split))
            np.save(og_path, predict(model, loader, trainer))
            print(f'  saved {og_path}')

            if SAVE_MASKS:
                masks_path = os.path.join(PRED_DIR, cfg['masks_out'].format(split=split))
                if masks_path not in saved_masks:
                    np.save(masks_path, compute_masks(ds))
                    saved_masks.add(masks_path)
                    print(f'  saved {masks_path}')

            if cfg['quantize'] and cfg['quant_out'] is not None:
                quant_path = os.path.join(PRED_DIR, cfg['quant_out'].format(split=split))
                np.save(quant_path, predict(qmodel, loader, trainer))
                print(f'  saved {quant_path}')


if __name__ == '__main__':
    main()
