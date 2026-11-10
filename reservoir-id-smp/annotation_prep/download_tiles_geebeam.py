"""Sample and download images tiles for S2 mosaic using geebeam"""

import argparse
import ee
import geebeam
import sentinel_mosaic_ee
import google

EE_PROJECT_ID = 'tropics-woodwell'
ee.Initialize(project=EE_PROJECT_ID)


def main():
    p = argparse.ArgumentParser(
            description='Extract subest images from ee mosaic',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('year',
                   help='Mosaic year',
                   type=int)
    p.add_argument('num_subsets',
                   help='Number of subsets to create',
                   type=int)
    p.add_argument('input_shpfile',
                   help='Shapefile for random sampling',
                   type=str)
    p.add_argument('out_dir',
                   help='Output directory for subset images',
                   type=str)
    p.add_argument('--side_length',
                   help='Side length of each image',
                   type=int,
                   required=False,
                   default=500)
    args = p.parse_args()

    ee_mosaic = sentinel_mosaic_ee.build_sentinel_mosaic(args.year)
    geebeam.sample_and_run_pipeline(
        image_list = [ee_mosaic],
        n_sample=args.num_subsets,
        align_transform=[8.9831528412e-05, 0.0, -74.04767971236954, 0.0, -8.9831528412e-05, 5.3112891173595],
        project=EE_PROJECT_ID,
        crs="EPSG:4326",
        patch_size=args.side_length, # Pixel dimensions in each direction
        validation_ratio=0.0, # Fraction to select as validation data
        output_type='tiff',
        output_path=args.out_dir,
        sampling_region=args.input_shpfile,
    )

if __name__ == '__main__':
    main()
