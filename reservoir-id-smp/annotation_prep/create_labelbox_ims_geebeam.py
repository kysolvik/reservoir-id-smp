#!/usr/bin/env python3
"""Prep tiles downloaded by download_tiles_geebeam.py for annotation

Writes an NDWI png of each tiff (for Labelbox) and a grid_indices.csv with
each tile's bounds and a Google Maps link, taken from the tiff georeference.

Example:
    $ python3 extract_tiles_geebeam.py out/sentinel_2021/pantanal/full/ out/labelbox/pantanal/
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import rasterio
from skimage import io


def ndwi_byte(green, nir):
    """NDWI scaled to 0-255"""
    green = green.astype('float32')
    nir = nir.astype('float32')
    ndwi = (green - nir) / (green + nir)
    return np.round(255 * (ndwi - ndwi.min()) / (ndwi.max() - ndwi.min())).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description='Make NDWI pngs and csv from geebeam tiffs')
    p.add_argument('tif_dir', help='Directory of geebeam tiffs', type=str)
    p.add_argument('out_dir', help='Output directory for pngs and csv', type=str)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Region name from geebeam layout: <region>/full/*.tif
    strata = os.path.basename(os.path.dirname(os.path.normpath(args.tif_dir)))

    rows = []
    for tif_path in sorted(glob.glob(os.path.join(args.tif_dir, '*.tif'))):
        name = os.path.splitext(os.path.basename(tif_path))[0] + '_ndwi'
        with rasterio.open(tif_path) as src:
            ar = src.read()
            bounds = src.bounds

        # Skip tiles with no data (outside mosaic). S1 bands can be 0 over
        # water, so only count pixels where all bands are 0
        if np.any(np.all(ar == 0, axis=0)) or np.any(np.isnan(ar)):
            print('Skipping {}: contains no data'.format(tif_path))
            continue

        # Bands: B2, B3, B4, B8, ...
        io.imsave(os.path.join(args.out_dir, name + '.png'),
                  ndwi_byte(ar[1], ar[3]), check_contrast=False)

        lon_c = (bounds.left + bounds.right) / 2
        lat_c = (bounds.bottom + bounds.top) / 2
        rows.append({
            'name': name,
            'source': os.path.basename(tif_path),
            'strata': strata,
            'lon_min': bounds.left,
            'lon_max': bounds.right,
            'lat_min': bounds.bottom,
            'lat_max': bounds.top,
            'gmaps_link': 'https://www.google.com/maps/@{},{},5000m/data=!3m1!1e3'.format(lat_c, lon_c),
        })

    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, 'grid_indices.csv'), index=False)


if __name__ == '__main__':
    main()
