"""Calculate sub-basin Strahler order based on ANA data"""
import geopandas as gpd

drainage = gpd.read_file('./data/geoft_bho_area_drenagem.gpkg')
drainage['nunivotto8'] = drainage['cobacia'].str[:8]
level8 = drainage.dissolve(by='nunivotto8')
level8.to_file('./data/geoft_bho_area_drenagem_level8dissolved.gpkg')

rivers = gpd.read_file('../municipalities/data/geoft_bho_trecho_drenagem.gpkg')
basins = level8
rivers['nunivotto8'] = rivers['cobacia'].str[:8]
rivers['nustrahler'] = rivers['nustrahler'].fillna(0)
rivers_max = rivers.groupby('nunivotto8')['nustrahler'].max()
joined = basins.join(rivers_max)
joined.to_file('./data/level8_basins_strahler.gpkg')