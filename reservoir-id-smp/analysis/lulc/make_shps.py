"""Convert csv with reservoir points to shpfile"""

import geopandas as gpd
import pandas as pd
import argparse
import os
import glob


def argparse_init():
    """Prepare ArgumentParser for inputs"""

    p = argparse.ArgumentParser(
            description='Convert csvs to shapefiles',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        'csv_pattern',
        help='Glob pattern for csvs, Make sure it is in quotes in command line',
        type=str)
    p.add_argument(
        'x_column',
        help='Name of x-coordinate column in csv.',
        type=str)
    p.add_argument(
        'y_column',
        help='Name of y-coordinate column in csv.',
        type=str)
    p.add_argument(
        'out_dir',
        help='Path to output directory.',
        type=str)
    p.add_argument(
        '--clip_shp',
        default=None,
        help='Optional polygon shpfile to clip with.',
        type=str)
    return p


def read_csv_to_gpd(csv_path, x_column, y_column):
    df = pd.read_csv(csv_path)
    df['geometry'] = gpd.points_from_xy(df[x_column], df[y_column], crs='ESRI:102033')
    gdf = gpd.GeoDataFrame(df, geometry=df['geometry'])

    return gdf


def write_to_shp(gdf, csv_path, out_dir):
    out_path = os.path.join(out_dir, os.path.basename(csv_path).replace('.csv', '.shp'))
    gdf.to_file(out_path)

    return


def clip_points(gdf, clip_gdf):
    return 


def main():
    # Get Command Line Args
    parser = argparse_init()
    args = parser.parse_args()

    # Read in shapefile for clipping, if provided
    if args.clip_shp is not None:
        clip_gdf = gpd.read_file(args.clip_shp)
     
    # Get list of csvs and run for each
    csv_list = glob.glob(args.csv_pattern)
    for csv_path in csv_list:
        gdf = read_csv_to_gpd(csv_path, args.x_column, args.y_column)
        if args.clip_shp is not None:
            gdf = gpd.clip(gdf, clip_gdf)
        write_to_shp(gdf, csv_path, args.out_dir)


if __name__=='__main__':
    main()
