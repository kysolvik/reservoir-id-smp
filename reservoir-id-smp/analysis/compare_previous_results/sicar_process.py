"""Filter and merge SICAR polygons"""
import glob
import os
import geopandas as gpd
import pandas as pd

in_dir = './SICAR/'

zip_list = glob.glob(os.path.join(in_dir, '*/*.zip'))
final_df_list = []
for z in zip_list:
    state = os.path.basename(os.path.dirname(zip_list[0]))
    print(state)
    df = gpd.read_file('zip:{}'.format(z))
    print(df.shape)
    df = df.loc[df['cod_tema']=='RESERVATORIO_ARTIFICIAL_DECORRENTE_BARRAMENTO']
    df['state'] = state
    final_df_list.append(df)

full_df = pd.concat(final_df_list)

full_df.to_file('./SICAR/full_reservoirs.shp')
print(full_df.shape)
print(full_df.geometry.area.sum())

# First dissolve everything
full_df = full_df.dissolve()


# Then explode into individual polys
full_df = full_df.explode()
print(full_df.shape)


full_df.to_file('./SICAR/full_reservoirs_dissolve_explode.shp')
