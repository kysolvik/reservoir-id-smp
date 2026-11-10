#!/usr/bin/env python3
"""Georeference Labelbox mask pngs using the geebeam tiffs they were drawn on

Masks from train/download_ims_masks.py --lb-only are named <key>_mask.png, where
<key> is the geebeam tile name (e.g. mata_atlantica_00012). The NDWI png shown in
Labelbox was made pixel-for-pixel from <tif_dir>/<strata>/full/<key>.tif, so each
mask gets that tiff's crs and transform. Output masks are uint8, 1 = reservoir.

Example:
    $ python3 georeference_masks.py ../train/data/masks_2021 ../train/data/masks_2021_geo

View all in QGIS with:
    $ gdalbuildvrt masks_2021.vrt ../train/data/masks_2021_geo/*_mask.tif
"""

import os
import glob
import argparse
import rasterio
from skimage import io

TIF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out/sentinel_2021')


def main():
    p = argparse.ArgumentParser(description='Georeference mask pngs from geebeam tiffs')
    p.add_argument('mask_dir', help='Directory of <key>_mask.png files', type=str)
    p.add_argument('out_dir', help='Output directory for <key>_mask.tif files', type=str)
    p.add_argument('--tif-dir', help='geebeam tiff root (<strata>/full/<key>.tif)',
                   type=str, default=TIF_DIR)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    n_written, n_skipped = 0, 0
    for mask_path in sorted(glob.glob(os.path.join(args.mask_dir, '*_mask.png'))):
        key = os.path.basename(mask_path)[:-len('_mask.png')]
        strata = key.rsplit('_', 1)[0]
        tif_path = os.path.join(args.tif_dir, strata, 'full', key + '.tif')
        if not os.path.exists(tif_path):
            print('Skipping {}: no tiff at {}'.format(mask_path, tif_path))
            n_skipped += 1
            continue

        mask = io.imread(mask_path)
        with rasterio.open(tif_path) as src:
            if mask.shape != (src.height, src.width):
                raise ValueError('{} shape {} != tiff shape {}'.format(
                    mask_path, mask.shape, (src.height, src.width)))
            crs, transform = src.crs, src.transform

        with rasterio.open(os.path.join(args.out_dir, key + '_mask.tif'), 'w',
                           driver='GTiff', height=mask.shape[0], width=mask.shape[1],
                           count=1, dtype='uint8', crs=crs, transform=transform,
                           compress='lzw') as dst:
            dst.write((mask > 0).astype('uint8'), 1)
        n_written += 1

    print('Wrote {}, skipped {}'.format(n_written, n_skipped))


if __name__ == '__main__':
    main()
