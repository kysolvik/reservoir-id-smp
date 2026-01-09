import exactextract
import geopandas as gpd
import rasterio
import rasterio.plot

buffer_size = 50

mb_lulc = './in/brazil/brazil_coverage_2021_c10_aea.tif'
reservoirs = './out/poly_buffer_{}m_diff.gpkg'.format(buffer_size)


foo = exactextract.exact_extract(mb_lulc, reservoirs,
                                 ['values','unique','frac'], output='pandas', progress=True)

foo.to_parquet('./out/poly_extract_{}m_frac.parquet'.format(buffer_size))
