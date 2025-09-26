#!/usr/bin/env python3
"""Quick script for preprocessing and cleaning municipality data downloade from IBGE"""

import pandas as pd


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
    years_unique = years_unique[years_unique!='nan'].astype(int)
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

### Precip data ###
# precip = pd.read_parquet('./data/pr_normal.parquet')
# precip = precip.rename(columns={'code_muni': 'cd_muni'}).set_index('cd_muni')
# precip.columns = 'precip_' + precip.columns
