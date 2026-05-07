import geopandas as gpd

ROADS_FILE = './data/ibge_bc250_2026_roads.gpkg'
RIVER_FILE = './data/geoft_bho_trecho_drenagem.gpkg'

roads_gdf = gpd.read_file(ROADS_FILE)
river_gdf = gpd.read_file(RIVER_FILE)

def find_intersections(roads_gdf, river_gdf, output_path):
    # 1. CRS Alignment
    # Intersections will fail or be wildly inaccurate if projections don't match.
    if roads_gdf.crs != river_gdf.crs:
        print("Aligning CRS...")
        river_gdf = river_gdf.to_crs(roads_gdf.crs)

    # 2. Spatial Join (The performance booster)
    # This uses a spatial index to find only the lines that actually touch.
    # 'inner' ensures we only keep pairs that intersect.
    print("Finding intersecting pairs...")
    joined = gpd.sjoin(roads_gdf, river_gdf, how='inner', predicate='intersects')

    # 3. Vectorized Intersection
    # We calculate the actual geometry of the intersection for the joined pairs.
    print("Calculating intersection points...")
    # We use .loc to align the river geometries with the joined road geometries
    intersection_geoms = joined.geometry.intersection(river_gdf.loc[joined.index_right].geometry, align=False)

    # 4. Cleanup and Format
    # Intersections can return MultiPoints; 'explode' turns them into individual Point rows.
    intersections_gdf = gpd.GeoDataFrame(geometry=intersection_geoms, crs=roads_gdf.crs)
    intersections_gdf = intersections_gdf.explode(index_parts=False)

    # 5. Filter for Points
    # Occasionally, lines overlap (parallel), resulting in LineStrings. 
    # This keeps only the specific crossing points.
    intersections_gdf = intersections_gdf[intersections_gdf.geometry.type == 'Point']

    # 6. Write to File
    # GeoPackage (.gpkg) is generally faster and more robust than Shapefiles.
    print(f"Saving {len(intersections_gdf)} points to {output_path}...")
    intersections_gdf.to_file(output_path, driver="GPKG")
    
    return intersections_gdf

def get_line_length_in_polygons(lines_gdf, polygons_gdf, poly_id_col):
    """
    Calculates the total length of lines within each polygon.
    
    Parameters:
    - lines_gdf: The GeoDataFrame containing LineStrings.
    - polygons_gdf: The GeoDataFrame containing Polygons.
    - poly_id_col: A unique identifier column name in polygons_gdf.
    """
    
    # 1. Ensure CRS match and are in a PROJECTED system (e.g., UTM)
    # This is CRITICAL for accurate length measurements (meters vs degrees).
    if lines_gdf.crs != polygons_gdf.crs:
        lines_gdf = lines_gdf.to_crs(polygons_gdf.crs)

    if polygons_gdf.crs.is_geographic:
        print("Warning: CRS is geographic (degrees). Results will be inaccurate.")
        print("Consider projecting to a UTM zone or local CRS first.")

    # 2. Perform the Overlay (Intersection)
    # This 'clips' the lines to the polygon boundaries.
    print("Clipping lines to polygons...")
    clipped_lines = gpd.overlay(lines_gdf, polygons_gdf, how='intersection')

    # 3. Calculate length of the clipped segments
    clipped_lines['segment_length'] = clipped_lines.geometry.length

    # 4. Sum lengths by polygon ID
    length_stats = clipped_lines.groupby(poly_id_col)['segment_length'].sum().reset_index()

    # 5. Merge back to the original polygons to keep all features
    # (Including those that might have had 0 line length inside)
    final_gdf = polygons_gdf.merge(length_stats, on=poly_id_col, how='left')
    final_gdf['segment_length'] = final_gdf['segment_length'].fillna(0)

    print("Calculation complete.")
    return final_gdf

find_intersections(roads_gdf, river_gdf, './data/river_road_intersections.gpkg')


# Get length of roads and length of streams per muni
muni_gdf = gpd.read_file('./data/municipios.shp')

roads_in_muni = get_line_length_in_polygons(roads_gdf.to_crs('ESRI:102033'),
                                            muni_gdf.to_crs('ESRI:102033'),
                                            'cd_mun')
rivers_in_muni = get_line_length_in_polygons(river_gdf.to_crs('ESRI:102033'),
                                             muni_gdf.to_crs('ESRI:102033'),
                                             'cd_mun')

roads_in_muni.to_file('./data/roads_in_munis.gpkg')
rivers_in_muni.to_file('./data/rivers_in_munis.gpkg')