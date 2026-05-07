#!/usr/bin/env python3
"""Quick script for preprocessing and cleaning municipality"""

import pandas as pd
import geopandas as  gpd
import os

### Cattle
cattle = pd.read_excel('./data/tabela3939.xlsx')
cattle.columns =  cattle.iloc[2]
year_list = cattle.columns[2:]
cattle.columns = ['cd_mun', 'municipal'] + list(year_list)
cattle = cattle.drop(columns=['municipal'])
cattle = cattle.iloc[4:-1]
cattle['cd_mun'] = cattle['cd_mun'].astype(int)
cattle = cattle.set_index('cd_mun')
cattle = cattle.replace(['-', '...'], 'NaN')
cattle = cattle.astype(float)
cattle.to_csv('./data/cattle.csv')
# Example read:
# pd.read_csv('./data/cattle.csv', index_col=0)

## Aquiculture
aqua = pd.read_excel('./data/tabela3940.xlsx')
aqua.columns = aqua.iloc[3]
prod_list = aqua.columns[2:]
aqua.columns = ['cd_mun', 'municipal'] + list(prod_list)
aqua = aqua.drop(columns=['municipal'])
aqua = aqua.iloc[4:-1]
aqua['cd_mun'] = aqua['cd_mun'].astype(int)
aqua = aqua.set_index('cd_mun')
# Things in quilos only
aqua = aqua[[c for c in aqua.columns if 'Quilogramas' in c]]
aqua = aqua.replace(['-', '...'], 0)
aqua = aqua.astype(float)
aqua['total_aqua_kilos'] = aqua.sum(axis=1)
aqua.to_csv('./data/aqua.csv')


### River and road lengths, plus counts of crossings
river_lengths = gpd.read_file(
    './data/rivers_in_munis.gpkg'
    ).set_index(
        'cd_mun'
        )[['segment_length']].rename(
            columns={'segment_length':'river_length'}
            )
road_lengths = gpd.read_file(
    './data/roads_in_munis.gpkg'
    )
crossings = gpd.read_file('./data/river_road_intersections.gpkg')
road_crossings = gpd.sjoin(crossings, road_lengths.to_crs('EPSG:4674'), predicate='within', how='left')
road_crossings_count = road_crossings.groupby('cd_mun').size().rename('crossing_count').astype(int)
muni_river_roads = road_lengths.set_index('cd_mun').join(road_crossings_count).fillna(0)
muni_river_roads = muni_river_roads[['crossing_count', 'segment_length']].rename(
    columns={'segment_length': 'road_length'}).join(
        river_lengths
    )
muni_river_roads.to_csv('./data/muni_river_roads.csv')


### Crops ###
def process_crops(excel_file, variable):
    crops_rename_dict = {
        'Algodão arbóreo (em caroço)':'cotton_tree_' + variable,
        'Algodão herbáceo (em caroço)': 'cotton_upland_' + variable,
        'Arroz (em casca)': 'rice_' + variable,
        'Café (em grão) Total': 'coffee_' + variable,
        'Cana-de-açúcar': 'sugarcane_' + variable,
        'Milho (em grão)': 'corn_' + variable,
        'Soja (em grão)': 'soy_' + variable,
        'Trigo (em grão)': 'wheat_' + variable,
        'Cacau (em amêndoa)': 'cocoa_' + variable
    }

    crops = pd.read_excel(excel_file)
    crop_names = crops.iloc[3]
    years = crops.iloc[2].astype(str)
    if variable == 'value':
        # Fix for value
        years = years.str.split(' ').str[0]

    # Assign years
    years_unique = years.unique()
    years_unique = years_unique[(years_unique!='nan') & (~years_unique.isna())].astype(int)
    crops.columns = crops.iloc[1]
    crops = crops.rename(columns={'Município':'cd_mun'}).set_index('cd_mun')
    crops.columns = pd.MultiIndex.from_product([years_unique, crop_names.unique()[1:]])
    crops = crops.iloc[4:-1]
    crops.index = crops.index.astype(int)
    crops = crops.replace(['-', '...'], 'NaN')
    crops = crops.astype(float)
    crops = crops.rename(columns=crops_rename_dict)
    return crops
crops_area = process_crops('./data/area_colhida.xlsx', 'area')
crops_quantity = process_crops('./data/quantity_produced.xlsx', 'quantity')
crops_value = process_crops('./data/value_produced.xlsx', 'value')
crops_all = crops_area.join(crops_quantity).join(crops_value)
crops_all.to_csv('./data/crops.csv')
# Example read:
# pd.read_csv('./data/crops.csv', header=[0,1], index_col=0)

### Economic data ###
pib_rename_dict = {
       'Valor adicionado bruto da Agropecuária, \na preços correntes\n(R$ 1.000)': 'agriculture',
        'Valor adicionado bruto da Indústria,\na preços correntes\n(R$ 1.000)': 'industry',
       'Valor adicionado bruto dos Serviços,\na preços correntes \n- exceto Administração, defesa, educação e saúde públicas e seguridade social\n(R$ 1.000)': 'services',
       'Valor adicionado bruto da Administração, defesa, educação e saúde públicas e seguridade social, \na preços correntes\n(R$ 1.000)': 'administration',
       'Valor adicionado bruto total, \na preços correntes\n(R$ 1.000)': 'total',
       'Impostos, líquidos de subsídios, sobre produtos, \na preços correntes\n(R$ 1.000)': 'taxes_subsidies',
       'Produto Interno Bruto, \na preços correntes\n(R$ 1.000)': 'gdp',
       'Produto Interno Bruto per capita, \na preços correntes\n(R$ 1,00)': 'gdp_per_capita',

}
pib_remove_list = [
    'Atividade com maior valor adicionado bruto',
    'Atividade com segundo maior valor adicionado bruto',
    'Atividade com terceiro maior valor adicionado bruto'
    ]

def process_pib(excel_file):
    pib = pd.read_excel(excel_file)
    pib = pib.rename(columns={'Código do Município':'cd_mun'}
                    ).set_index(['cd_mun', 'Ano'])
    pib = pib.iloc[:,30:].reset_index()
    pib = pib.rename(columns=pib_rename_dict).drop(columns=pib_remove_list, errors='ignore')
    pib = pib.pivot(index='cd_mun', columns='Ano')
    pib.columns = pib.columns.swaplevel(0,1)
    return pib
pib_new = process_pib('./data/PIB dos Munic¡pios - base de dados 2010-2021.xlsx')
pib_old = process_pib('./data/PIB dos Munic¡pios - base de dados 2002-2009.xls')
pib_full = pib_old.join(pib_new)
pib_full.to_csv('./data/pib.csv')
# Read example:
# pd.read_csv('./data/pib.csv', header=[0,1], index_col=0)

### Population data ###
pop = pd.read_excel('./data/tabela6579.xlsx') 
years = pop.iloc[2, 1:].astype(int)
pop.columns = ['cd_mun'] + list(years)
pop = pop.iloc[3:-1]
pop = pop.set_index('cd_mun')
pop.index = pop.index.astype(int)
pop = pop.replace(['...', '-'], 'NaN').astype(float)
pop.to_csv('./data/pop.csv')
# Example read
# pd.read_csv('./data/pop.csv',index_col=0)


### Deforestation data ###
mb_df = pd.read_excel('./data/MAPBIOMAS_BRAZIL-COL.10-BIOME_STATE_MUNICIPALITY.xlsx',
                      sheet_name='COVERAGE_10')
biome_df = mb_df.rename(columns={'geocode':'cd_muni'})[['cd_muni', 'biome']].groupby('cd_muni').first()
biome_df.to_csv('./data/mb_biome.csv')
mb_df_clean = mb_df.drop(
    columns=['ID','country','state','biome','municipality','municipality - state',
             'feature_id','class','class_level_0','class_level_2',
             'class_level_3','class_level_4']
).rename(columns={'geocode':'cd_muni'})

mb_df_grouped = mb_df_clean.groupby(['cd_muni','class_level_1']).sum()
mb_df_grouped.to_csv('./data/mb_lulc_cleaned.csv')



### Bring it all together into full dataframe
def read_process_csv_to_gdf(csv):
    temp_df = pd.read_csv(csv)
    temp_df['satellite'] = os.path.basename(csv)[:8]
    temp_df['year'] = int(os.path.basename(csv)[9:13])
    temp_df = temp_df.loc[temp_df['hydropoly_max']<=50]
    temp_df['area_ha'] = temp_df['area']*100/10000 # HA
    temp_df['area_km'] = temp_df['area']*100/(1000*1000) # km2
    temp_df = temp_df.loc[temp_df['area_ha']<=50] # Remove greater than 50 ha
    temp_df = temp_df.loc[temp_df['area_ha']>=0.05] # Remove less than 0.05 ha
    temp_gdf = gpd.GeoDataFrame(
        temp_df, geometry=gpd.points_from_xy(temp_df.longitude, temp_df.latitude),
        crs='EPSG:4326'
    )
    return temp_gdf

def sjoin_summarize(points_gdf, poly_gdf, poly_field):
    joined_gdf = gpd.sjoin(points_gdf, poly_gdf, predicate='within', how='inner')
    return joined_gdf[['res_area_ha', poly_field]].groupby(poly_field).agg(['sum', 'count', 'median'])['res_area_ha']

def sjoin_summarize_nogroup(points_gdf, poly_gdf):
    joined_gdf = gpd.sjoin(points_gdf, poly_gdf, predicate='within', how='inner')
    return joined_gdf[['area_ha']].agg(['sum', 'count', 'median'])

# Muni stats
in_csv = '../clean_summarize/out/sentinel_v7/sentinel_2021_v7_wgs84_bordermerged.csv'
muni_gdf = gpd.read_file('./data/municipios.shp')
muni_gdf['center_lon'] = muni_gdf.geometry.centroid.x
muni_gdf['center_lat'] = muni_gdf.geometry.centroid.y
print(muni_gdf.shape)
res_gdf = read_process_csv_to_gdf(in_csv)
res_gdf['res_area_ha'] = res_gdf['area_ha']
joined_gdf = gpd.sjoin(res_gdf, muni_gdf, predicate='within', how='inner')
muni_res_stats = sjoin_summarize(res_gdf, muni_gdf, 'cd_mun')
muni_res_stats = muni_res_stats.join(muni_gdf.set_index('cd_mun')[['area_ha', 'center_lon','center_lat']], how='outer')
print(muni_res_stats.shape)

# Most predictors
cattle = pd.read_csv('./data/cattle.csv', index_col=0)
aqua = pd.read_csv('./data/aqua.csv', index_col=0)[['total_aqua_kilos']]
aqua.columns = (pd.MultiIndex.from_product([['2021'], aqua.columns]))
muni_river_roads = pd.read_csv('./data/muni_river_roads.csv', index_col=0)
muni_river_roads.columns = (pd.MultiIndex.from_product([['2021'], muni_river_roads.columns]))
crops = pd.read_csv('./data/crops.csv', header=[0,1], index_col=0)
pib = pd.read_csv('./data/pib.csv', header=[0,1], index_col=0)
pop = pd.read_csv('./data/pop.csv',index_col=0)
cattle.columns = (pd.MultiIndex.from_product([cattle.columns, ['cattle']]))
pop.columns = (pd.MultiIndex.from_product([pop.columns, ['pop']]))
biome_df = pd.read_csv('./data/biome.csv', index_col=0)
state_df = pd.read_csv('./data/mb_state.csv', index_col=0)

# Irrigation
irrigation = pd.read_csv('../irrigation/data/center_pivot_area_1985_2022.csv')
irrigation = irrigation.rename(columns={'mun_código':'cd_mun'})
irrigation = irrigation.iloc[:-9]
irrigation['cd_mun'] = irrigation['cd_mun'].astype(int)
irrigation = irrigation.set_index(
    'cd_mun'
    ).drop(
        columns=['mun_nome','UF_nome','UF_sigla','Região']
        )
irrigation.columns = 'irrigation_' + irrigation.columns
irrigation = irrigation.replace(',','', regex=True).astype(float)

# Precip just needs a little processing
precip = pd.read_parquet('./data/pr_indi.parquet')
precip = precip.groupby(['code_muni', 'month']).mean().reset_index()
precip = (precip.set_index('code_muni')*30).reset_index()
# Select columns
precip = precip[['code_muni', 'month','mean','p10','p90']]
precip['low_rain'] = precip['mean']<100
precip = precip.rename(columns={'code_muni': 'cd_muni'})
precip_std = precip[['cd_muni', 'mean']].groupby('cd_muni').std().rename(
    columns={'mean':'std'})
precip_min = precip.groupby('cd_muni').min()[['mean']].rename(columns={'mean': 'min'})
precip_minmax = precip.groupby('cd_muni').max()['mean'] - precip.groupby('cd_muni').min()['mean']
precip_minmax.name = 'range'
precip = precip.groupby('cd_muni').mean().join(precip_std).join(precip_minmax).join(precip_min)
precip['std_div_mean'] = precip['std']/precip['mean']
precip.columns = 'precip_' + precip.columns
precip = precip.drop(columns='precip_month')

# If using from 1960-1990 (historical normals), uncomment
# precip = pd.read_parquet('./data/pr_normal.parquet')
# precip = (precip.set_index('code_muni')*30).reset_index()
# precip['low_rain'] = precip['normal_mean']<100
# precip = precip.rename(columns={'code_muni': 'cd_muni'})
# precip_std = precip[['cd_muni', 'normal_mean']].groupby('cd_muni').std().rename(
#     columns={'normal_mean':'normal_std'})
# precip_min = precip.groupby('cd_muni').min()[['normal_mean']].rename(columns={'normal_mean': 'normal_min'})
# precip_minmax = precip.groupby('cd_muni').max()['normal_mean'] - precip.groupby('cd_muni').min()['normal_mean']
# precip_minmax.name = 'range'
# precip = precip.groupby('cd_muni').mean().join(precip_std).join(precip_minmax).join(precip_min)
# precip['std_div_mean'] = precip['normal_std']/precip['normal_mean']
# precip.columns = 'precip_' + precip.columns
# precip = precip.drop(columns='precip_month')

# Mapbiomas
def l1_processing(df):
    df = df.reset_index().set_index('cd_muni').pivot(columns='class_level_1').fillna(0)['2021'].rename(
        columns={
            '1. Forest': 'forest',
            '2. Non Forest Natural Formation': 'natural_nonforest',
            '3. Farming': 'farming',
            '4. Non vegetated area': 'non_veg',
            '5. Water and Marine Environment': 'water',
            '6. Not Observed': 'na',
        }
    )
    df['natural_total'] = df['forest'] + df['natural_nonforest']
    return df

# Mapbiomas data
# Level 1
mb_df = pd.read_csv('./data/mb_lulc_cleaned.csv', index_col=[0,1])
mb_df = mb_df.fillna(0)
# Convert to percentages

mb_df = mb_df.div(mb_df.reset_index().groupby('cd_muni').sum()['1985'], axis=0) * 100

mb_diffs = mb_df[['2021']].copy()
mb_diffs['2021'] = mb_df['2021'] - mb_df['1985']
mb_diffs = l1_processing(mb_diffs)
mb_cur = l1_processing(mb_df['2021'])

# # Level 4
mb_l4_df = pd.read_csv('./data/mb_lulc_cleaned_level4.csv', index_col=[0,1])
mb_l4_df = mb_l4_df.fillna(0)
# Convert to percentages
mb_l4_df = mb_l4_df.div(mb_l4_df.reset_index().groupby('cd_muni').sum()['1985'], axis=0) * 100
mb_l4_cur = mb_l4_df['2021']
mb_l4_diffs = mb_l4_df['2021'] - mb_l4_df['1985']
mb_l4_diffs = mb_l4_diffs.reset_index().set_index('cd_muni').pivot(columns='class_level_4').fillna(0)[0]
mb_l4_cur = mb_l4_cur.reset_index().set_index('cd_muni').pivot(columns='class_level_4').fillna(0)['2021']

mb_diffs['soy'] = mb_l4_diffs['3.2.1.1. Soybean']
mb_diffs['rice'] = mb_l4_diffs['3.2.1.3. Rice']
mb_diffs['pasture'] = mb_l4_diffs['3.1. Pasture']
mb_cur['soy'] = mb_l4_cur['3.2.1.1. Soybean']
mb_cur['rice'] = mb_l4_diffs['3.2.1.3. Rice']
mb_cur['pasture'] = mb_l4_cur['3.1. Pasture']

mb_cur.columns = mb_cur.columns + '_current'
mb_diffs.columns = mb_diffs.columns + '_diff'


# Merge everything together
preds_2021 = cattle.join(
    crops, how='outer'
    ).join(
        aqua, how='outer'
        ).join(
            muni_river_roads, how='outer'
            ).join(pib, how='outer').join(
                pop, how='outer'
                )['2021'].join(
                    precip, how='outer').join(
                        mb_diffs, how='outer').join(
                            mb_cur, how='outer'
                            ).join(
                                irrigation[['irrigation_2022']], how='outer'
                                )

print(preds_2021.shape)
print(muni_res_stats.shape)
full_df = preds_2021.join(muni_res_stats)
full_df_density = full_df.div((full_df['area_ha']/100), axis=0)
full_df_density.columns = full_df_density.columns + '_density'
full_df_percapita = full_df.div((full_df['pop']), axis=0).drop(columns='gdp') # Duplicate
full_df_percapita.columns = full_df_percapita.columns + '_per_capita'

full_df = full_df.join(full_df_density).join(full_df_percapita).join(biome_df).join(state_df)
# Remove a bad muni
full_df = full_df.loc[~full_df['biome'].isna()]

# Drop response vars
pred_df = full_df.copy()
for response_var in ['count','median','sum']:
    pred_df = pred_df.loc[:,~pred_df.columns.str.startswith(response_var)]

full_df.to_csv('./data/full_df.csv')