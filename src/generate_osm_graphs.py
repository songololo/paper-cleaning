"""
Prepares OSM graphs for downstream comparisons
"""

# %%
from __future__ import annotations

import pathlib
import sys

sys.path.append(str(pathlib.Path.cwd().parent.absolute()))

import geopandas as gpd
from cityseer.tools import graphs, io

from src import util

# %%
# Load Madrid boundary
bounds_gdf = gpd.read_file("../temp/buffered_bounds.gpkg")
bounds_gdf.to_crs(3035, inplace=True)
# includes a 20km buffer so reverse by 10km
# this still includes enough for 10km centralities
bounds = bounds_gdf.geometry.iloc[0].buffer(-10000)

# %%
# raw network
mad_nx_raw = util.osm_raw_graph_from_poly(poly_geom=bounds, poly_crs_code=3035, to_crs_code=3035)
mad_raw_gdf = io.geopandas_from_nx(mad_nx_raw, crs=3035)
mad_raw_gdf.to_file("../temp/madrid_network_raw.gpkg")

# %%
# basic cleaning
mad_nx_basic_clean = graphs.nx_remove_filler_nodes(mad_nx_raw)
mad_nx_basic_clean = graphs.nx_remove_dangling_nodes(
    mad_nx_basic_clean, despine=15, remove_disconnected=100
)
# save
mad_basic_clean_gdf = io.geopandas_from_nx(mad_nx_basic_clean, crs=3035)
mad_basic_clean_gdf.to_file("../temp/madrid_network_basic_clean.gpkg")

# %%
# full clean
for hwy_keys, matched_only, split_dist, consol_dist, cent_by_itx in [
    (["motorway"], True, 60, 30, False),
    (["trunk"], True, 40, 20, False),
    (["primary"], True, 40, 20, False),
    (["secondary"], True, 30, 15, False),
    (["tertiary"], True, 30, 15, False),
    (["residential"], True, 20, 12, True),
    (None, True, 12, 10, True),
]:
    contains_buffer_dist = max(split_dist, 25)
    mad_clean_matches = graphs.nx_split_opposing_geoms(
        mad_nx_basic_clean,
        buffer_dist=split_dist,
        prioritise_by_hwy_tag=True,
        osm_hwy_target_tags=hwy_keys,
        osm_matched_tags_only=matched_only,
        contains_buffer_dist=contains_buffer_dist,
    )
    mad_clean_matches = graphs.nx_consolidate_nodes(
        mad_clean_matches,
        buffer_dist=consol_dist,
        crawl=True,
        centroid_by_itx=cent_by_itx,
        prioritise_by_hwy_tag=True,
        contains_buffer_dist=contains_buffer_dist,
        osm_hwy_target_tags=hwy_keys,
        osm_matched_tags_only=matched_only,
    )
    mad_clean_matches = graphs.nx_remove_filler_nodes(mad_clean_matches)
mad_clean_matches = graphs.nx_iron_edges(mad_clean_matches)
# save
mad_clean_matches_gdf = io.geopandas_from_nx(mad_clean_matches, crs=3035)
mad_clean_matches_gdf.to_file("../temp/madrid_network_matched_clean.gpkg")

# %%
for hwy_keys, matched_only, split_dist, consol_dist, cent_by_itx in [
    (["motorway"], False, 30, 20, False),
    (["trunk", "primary"], False, 20, 15, False),
    (["secondary", "tertiary"], False, 15, 12, False),
    (["residential"], False, 12, 10),
    (None, False, 10, 5, True),
]:
    contains_buffer_dist = max(split_dist, 25)
    mad_clean_full_gdf = graphs.nx_split_opposing_geoms(
        mad_clean_matches,
        buffer_dist=split_dist,
        prioritise_by_hwy_tag=True,
        osm_hwy_target_tags=hwy_keys,
        osm_matched_tags_only=matched_only,
        contains_buffer_dist=contains_buffer_dist,
    )
    mad_clean_full_gdf = graphs.nx_consolidate_nodes(
        mad_clean_full_gdf,
        buffer_dist=consol_dist,
        crawl=True,
        centroid_by_itx=cent_by_itx,
        prioritise_by_hwy_tag=True,
        contains_buffer_dist=contains_buffer_dist,
        osm_hwy_target_tags=hwy_keys,
        osm_matched_tags_only=matched_only,
    )
    mad_clean_full_gdf = graphs.nx_remove_filler_nodes(mad_clean_full_gdf)
mad_clean_full_gdf = graphs.nx_iron_edges(mad_clean_full_gdf)
# save
mad_clean_full_gdf = io.geopandas_from_nx(mad_clean_full_gdf, crs=3035)
mad_clean_full_gdf.to_file("../temp/madrid_network_full_clean.gpkg")
