# import argparse
import geopandas as gpd
import dask_geopandas as dgpd

buffer_size = 50

in_shp = '../clean_summarize/out_polys/sentinel_2021_v7_aea_cleaned_polygons.gpkg'
out_shp = './out/poly_buffer_{}m.gpkg'.format(buffer_size)
out_diff_shp = './out/poly_buffer_{}m_diff.gpkg'.format(buffer_size)

gdf = gpd.read_file(in_shp)
gdf['area_m'] = gdf.geometry.area

# Buffer in parallel using dask
dgdf = dgpd.from_geopandas(gdf, npartitions=32)
buffered_dask = dgdf.buffer(buffer_size)
buffered_gdf = buffered_dask.compute()

#  Write out
buffered_gdf.to_file(out_shp)

# Element-wise difference (this works element-wise in parallel)
rings_gdf = buffered_gdf.geometry.difference(gdf.geometry)

# Write out
rings_gdf.to_file(out_diff_shp)