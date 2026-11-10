#!/usr/bin/env python3
"""Create a Labelbox dataset from NDWI pngs hosted on GCS

Each data row gets the tile name as its global key, strata as a 'tag'
metadata field, and the Google Maps link as a RAW_TEXT attachment (shows as a
clickable link under Attachments in the labeling editor).

Requires LABELBOX_API_KEY env var.

Example:
    $ python3 upload_labelbox_geebeam.py out/sentinel_2021/labelbox/all/grid_indices_all.csv \
        https://storage.googleapis.com/res-id/labelbox/s2_2021_stratified/ s2_2021_stratified
"""

import argparse
import pandas as pd


def main():
    p = argparse.ArgumentParser(description='Upload pngs on GCS to a Labelbox dataset')
    p.add_argument('grid_csv', help='Merged grid_indices csv', type=str)
    p.add_argument('url_prefix', help='Public https prefix for the pngs', type=str)
    p.add_argument('dataset_name', help='Name of new Labelbox dataset', type=str)
    p.add_argument('--limit', help='Only upload first N rows (for testing)', type=int, default=None)
    p.add_argument('--dry_run', help='Print rows without uploading', action='store_true')
    args = p.parse_args()

    df = pd.read_csv(args.grid_csv)
    if args.limit:
        df = df.head(args.limit)

    rows = [{
        'row_data': args.url_prefix.rstrip('/') + '/' + r['name'] + '.png',
        'global_key': r['name'],
        'media_type': 'IMAGE',
        'metadata_fields': [{'name': 'tag', 'value': r['strata']}],
        'attachments': [{'type': 'RAW_TEXT', 'name': 'Google Maps', 'value': r['gmaps_link']}],
    } for _, r in df.iterrows()]

    if args.dry_run:
        print(rows[0])
        print('{} rows'.format(len(rows)))
        return

    import labelbox as lb
    client = lb.Client()
    dataset = client.create_dataset(name=args.dataset_name)
    task = dataset.create_data_rows(rows)
    task.wait_till_done()
    print('Errors: {}'.format(task.errors))
    print('Dataset {} ({}) has {} data rows'.format(
        dataset.name, dataset.uid, dataset.row_count))


if __name__ == '__main__':
    main()
