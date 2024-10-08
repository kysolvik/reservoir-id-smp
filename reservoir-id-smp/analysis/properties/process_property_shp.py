import geopandas as gpd
import sys
in_file=sys.argv[1]
out_gpkg=sys.argv[2]
muni_df = gpd.read_file('../clip/brazil_munis_interviews_aea.shp')
prop_df = gpd.read_file(in_file, bbox=muni_df)#.geometry.unary_union)
prop_df.to_file(out_gpkg)
# prop_df = gpd.read_file('./data/pa_br_landtenure_imaflora_2021.gpkg', bbox=muni_df)#.geometry.unary_union)
# prop_df.to_file('./data/pa_br_landtenure_studyarea_only.gpkg')
