import pandas as pd
import glob
import geopandas as gpd
import argparse
import os

def process_df(df, csv, clip_polygon=None):
    df['satellite'] = os.path.basename(csv)[:3]
    df['year'] = int(os.path.basename(csv)[4:8])
    df = df.loc[df['hydropoly_max']<100]
    df['area_ha'] = df['area']*100/10000 # HA
    df['area_km2'] = df['area']*100/(1000*1000) # km2
    df = df.loc[df['area_ha']<=100] # Remove gt 100 ha
    df = df.loc[df['area_ha']>0.1] # Remove lt 0.1 ha
    df = df.reset_index()


    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.center_lon, df.center_lat),
        crs='ESRI:102033'
    ).to_crs('EPSG:4326')

    if clip_polygon is not None:
        gdf = gpd.clip(gdf, clip_polygon)

    df = pd.DataFrame(gdf.drop(columns='geometry'))
    df['x_aea'] = df['center_lon']
    df['y_aea'] = df['center_lat']
    df['longitude'] = gdf.geometry.x
    df['latitude'] = gdf.geometry.y

    # Filter out columns
    df['id_for_year_sat'] = df.index
    df = df[['id_for_year_sat', 'year', 'satellite','longitude', 'latitude', 'x_aea', 'y_aea', 'area_ha', 'area_km2']]

    return df

def write_to_csv(df, out_dir):
    sat = df['satellite'].values[0]
    year = df['year'].values[0]
    out_path = os.path.join(out_dir, f'{sat}_{year}_reservoirs.csv')
    df.to_csv(out_path, index=False)
    return

def append_to_parquet(df, out_path_parquet):
    if not os.path.exists(out_path_parquet):
        df.to_parquet(out_path_parquet, engine='fastparquet', compression='gzip')
    else:
        df.to_parquet(out_path_parquet, engine='fastparquet', append=True)
    return



def main():
    parser = argparse.ArgumentParser(
        description='Clean merged csvs and write as csvs and parquet'
        )
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()

    all_csvs = sorted(glob.glob(os.path.join(args.input_dir, '*v3_merged.csv')))

    clip_polygon = gpd.read_file('../regions/data/lm_bioma_250_DISSOLVED.shp').to_crs('EPSG:4326')

    out_path_parquet = os.path.join(args.output_dir, 'ls_all_reservoirs.parquet.gzip')

    for csv in all_csvs:
        df = pd.read_csv(csv)
        df = process_df(df, csv, clip_polygon)
        write_to_csv(df, args.output_dir)
#         if not (df['satellite'].values[0] == 'ls7' and df['year'].values[0] > 2019):
#             append_to_parquet(df, out_path_parquet)


if __name__ == '__main__':
    main()
