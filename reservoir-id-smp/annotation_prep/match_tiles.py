#!/usr/bin/env python3
"""
Extract arrays from new rasters that match previously extracted training tiles

Example:


"""

import argparse
import pandas as pd
import subprocess as sp
from osgeo import gdal
import glob


# Set target resolution
TARGET_RES = 8.9831528412e-05


def argparse_init():
    """Prepare ArgumentParser for inputs"""

    p = argparse.ArgumentParser(
            description='Extract matching tiles from new rasters',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('grid_indices_latlon',
                   help='grid indices csv output by extract_tiles.py',
                   type=str)
    p.add_argument('target_raster',
                   help='Target raster file to extract from (VRT or TIF)',
                   type=str)
    p.add_argument('output_dir',
                   help='Output directory.',
                   type=str)
    p.add_argument('output_suffix',
                   help='Output suffix, e.g. sent 2_20m',
                   type=str)

    return p

def subset_target(target_raster, output_file, subset_df_row):
    xmin = str(subset_df_row['lon_min'])
    xmax = str(subset_df_row['lon_max'])
    ymin = str(subset_df_row['lat_min'])
    ymax = str(subset_df_row['lat_max'])
    sp.call(['gdalwarp', '-tr', str(TARGET_RES), str(TARGET_RES),
             '-te', xmin, ymin, xmax, ymax, '-overwrite', '-co', 'COMPRESS=LZW',
             target_raster, output_file])

    return 


def main():

    # Get command line args
    parser = argparse_init()
    args = parser.parse_args()

    # Make output dir
    sp.call(['mkdir', '-p', args.output_dir])

    # Read in input dataframe
    grid_df = pd.read_csv(args.grid_indices_latlon)

    # Create matching arrays
    for row_i in range(grid_df.shape[0]):
        cur_row = grid_df.loc[row_i]
        output_file = '{}/{}.tif'.format(args.output_dir, cur_row['name'].replace(
            'ndwi', args.output_suffix))
        subset_target(args.target_raster, output_file, cur_row)

    return


if __name__ == '__main__':
    main()
