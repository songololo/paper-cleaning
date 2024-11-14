"""
Prepare datasets by running madrid-ua-dataset repo workflow and place in temp folder.
"""

# %%
from __future__ import annotations

import logging

import geopandas as gpd
import osmnx as ox
import pandas as pd
import seaborn as sns
from cityseer.tools import graphs, io
from shapely.strtree import STRtree
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")

# %%
# bounds
# network_edges_gdf = gpd.read_file("temp/neighbourhoods.gpkg")
# assert network_edges_gdf.crs.is_projected
# bounds = network_edges_gdf.union_all()
# bounds_wgs = network_edges_gdf.to_crs(4326).union_all()
# bounds_buff = bounds.buffer(10000)

WORKING_CRS = 3035
bound_buff = 500

# %%
# relation_fid = "R2417889"
# map_key = "barcelona"
relation_fid = "R5326784"
map_key = "madrid"

# %%
bounds = ox.geocode_to_gdf(
    relation_fid,
    by_osmid=True,
)
bounds = bounds.to_crs(WORKING_CRS)
bounds_buff = bounds.buffer(bound_buff).union_all()
bounds_wgs = bounds.buffer(bound_buff).to_crs(4326).union_all()
bounds_buff

# %%
# spaces for selecting / discarding footpaths
feats_gdf = ox.features_from_polygon(
    bounds_wgs,
    tags={
        "landuse": ["cemetery", "forest"],
        "leisure": ["park", "garden", "sports_centre"],
        # "highway": ["pedestrian"],
    },
)
feats_gdf.to_crs(WORKING_CRS, inplace=True)
# extract ways and convert to polys
# not interested in segments - which are captured separately from network query
# but do want squares etc. described as ways - hence buffer and reverse buffer
ways_gdf = feats_gdf.xs("way", level="element_type", drop_level=True)
ways_gdf = ways_gdf.explode(index_parts=False).reset_index(drop=True)
ways_gdf = ways_gdf[ways_gdf.geometry.type == "Polygon"]
# extract relations
relations_gdf = feats_gdf.xs("relation", level="element_type", drop_level=True)
relations_gdf = relations_gdf.explode(index_parts=False).reset_index(drop=True)
relations_gdf = relations_gdf[relations_gdf.geometry.type == "Polygon"]
# combine
combined_gdf = pd.concat([ways_gdf, relations_gdf])
combined_geom = combined_gdf.union_all()
# extract geoms and explode
combined_gdf = gpd.GeoDataFrame({"geometry": [combined_geom]}, crs=combined_gdf.crs)
park_area_gdf = combined_gdf.explode(index_parts=False).reset_index(drop=True)
park_area_gdf = park_area_gdf[park_area_gdf.geometry.type == "Polygon"]
park_area_gdf = park_area_gdf[~park_area_gdf.geometry.is_empty]
park_area_gdf = park_area_gdf[park_area_gdf.geometry.is_valid]
# exploded_gdf = exploded_gdf[exploded_gdf.area >= 20000]
park_area_gdf.to_file(f"temp/{map_key}_parks_poly.gpkg")


# %%
"""
way["highway"]
        ["highway"~"trunk|motorway"]
        ["tunnel"!="yes"](poly:"{geom_osm}");
"""
request = """
    /* https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL */
    [out:json];
    (
    way["highway"]
        ["highway"!="footway"]
        ["footway"="sidewalk"](poly:"{geom_osm}");
    way["highway"]
        ["highway"!~"bus_guideway|busway|escape|raceway|proposed|planned|abandoned|platform|emergency_bay|rest_area|disused|path|corridor|ladder|bus_stop|track|elevator|services"]
        ["area"!="yes"]
        ["footway"!="sidewalk"]
        ["amenity"!~"charging_station|parking|fuel|motorcycle_parking|parking_entrance|parking_space"]
        ["indoor"!="yes"]
        ["level"!="-2"]
        ["level"!="-3"]
        ["level"!="-4"]
        ["level"!="-5"](poly:"{geom_osm}");
    );
    out body;
    >;
    out qt;
"""
# test_geom, _ = io.buffered_point_poly(-3.7010254, 40.4327720, 150)
nx_raw = io.osm_graph_from_poly(
    bounds_buff.simplify(500),
    poly_crs_code=WORKING_CRS,
    to_crs_code=WORKING_CRS,
    simplify=False,
    iron_edges=False,
    custom_request=request,
)
_edges_gdf = io.geopandas_from_nx(nx_raw, crs=WORKING_CRS)
_edges_gdf.to_file(f"temp/{map_key}_osm_network_pre.gpkg")

# %%
nx_copy = nx_raw.copy()
# discard edges intersecting green spaces
uniq_hwy_tags = set()
remove_edges = []
# use STR Tree for performance
parks_str_tree = STRtree(park_area_gdf.geometry.to_list())
parks_buff_str_tree = STRtree(park_area_gdf.buffer(5).geometry.to_list())
# iter edges to find edges for removal
for start_node_key, end_node_key, edge_data in tqdm(
    nx_copy.edges(data=True), total=nx_copy.number_of_edges()
):
    edge_geom = edge_data["geom"]
    # remove if footway if not itx with pedestrian poly
    if "footway" in edge_data["highways"]:  # and "pedestrian" not in edge_data["highways"]
        itx = parks_str_tree.query(edge_geom, predicate="within")
        if len(itx):
            remove_edges.append((start_node_key, end_node_key))
            continue
    # e.g. cemetries
    if "service" in edge_data["highways"]:
        itx = parks_buff_str_tree.query(edge_geom, predicate="intersects")
        if len(itx):
            remove_edges.append((start_node_key, end_node_key))
            continue
    uniq_hwy_tags.update(edge_data["highways"])
# report and remove
print(f"Hwy tags: {uniq_hwy_tags}")
print(f"Edges before removing green itx: {nx_copy.number_of_edges()}")
# remove all edges regardless of index between start & end nodes
# otherwise index ordering becomes unpredictable
nx_copy.remove_edges_from(remove_edges)
print(f"Edges after removing green itx: {nx_copy.number_of_edges()}")
_edges_gdf = io.geopandas_from_nx(nx_copy, crs=WORKING_CRS)
_edges_gdf.to_file(f"temp/{map_key}_osm_network_filt.gpkg")

# %%
# remove short danglers and disconnected components
G1 = graphs.nx_remove_dangling_nodes(nx_copy, despine=10, remove_disconnected=100)
_edges_gdf = io.geopandas_from_nx(G1, crs=WORKING_CRS)
_edges_gdf.to_file(f"temp/{map_key}_osm_network_1.gpkg")

# %%
# start by bridging gaps between dead-ended segments
G2 = graphs.nx_remove_filler_nodes(G1)
G2 = graphs.nx_snap_gapped_endings(
    G2,
    buffer_dist=20,
    # not main roads
    osm_hwy_target_tags=[
        "residential",
        "footway",
        "living_street",
        "pedestrian",
    ],
)
_edges_gdf = io.geopandas_from_nx(G2, crs=WORKING_CRS)
_edges_gdf.to_file(f"temp/{map_key}_osm_network_2.gpkg")

# %%
# look for degree 1 dead-ends and link to nearby edges
G3 = graphs.nx_split_opposing_geoms(
    G2,
    buffer_dist=25,
    min_node_degree=1,
    max_node_degree=1,
    squash_nodes=False,
)
_edges_gdf = io.geopandas_from_nx(G3, crs=WORKING_CRS)
_edges_gdf.to_file(f"temp/{map_key}_osm_network_3.gpkg")

# %%
# remove fillers
nx_basic_clean = graphs.nx_remove_filler_nodes(G3)
# remove longer danglers
nx_basic_clean = graphs.nx_remove_dangling_nodes(nx_basic_clean, despine=20)
# clean by highway types
for dist, tags, simplify_line_angles in (
    # (40, ["motorway"], False),  # does this matter in reality?  , "motorway_link"
    # (24, ["trunk"], 45, 24),  # , "trunk_link"
    (18, ["primary"], 45),  # , "primary_link"
    (16, ["secondary"], 55),  # , "secondary_link"
    (14, ["tertiary"], 65),  # , "tertiary_link"
    (12, ["residential"], 95),
):
    nx_basic_clean = graphs.nx_consolidate_nodes(
        nx_basic_clean,
        buffer_dist=dist,
        crawl=False,
        centroid_by_itx=False,
        osm_hwy_target_tags=tags,
        prioritise_by_hwy_tag=True,
        simplify_line_angles=simplify_line_angles,
    )
for dist, tags, simplify_line_angles in (
    # (40, ["motorway"], False),  # does this matter in reality?  , "motorway_link"
    # (24, ["trunk"], 45, 24),  # , "trunk_link"
    (18, ["primary"], 45),  # , "primary_link"
    (16, ["secondary"], 55),  # , "secondary_link"
    (14, ["tertiary"], 65),  # , "tertiary_link"
    (12, ["residential"], 95),
):
    nx_basic_clean = graphs.nx_split_opposing_geoms(
        nx_basic_clean,
        buffer_dist=dist,
        squash_nodes=True,
        osm_hwy_target_tags=tags,
        centroid_by_itx=False,
        simplify_line_angles=simplify_line_angles,
    )
    nx_basic_clean = graphs.nx_remove_filler_nodes(nx_basic_clean)
    _edges_gdf = io.geopandas_from_nx(nx_basic_clean, crs=WORKING_CRS)
    _edges_gdf.to_file(f"temp/{map_key}_osm_network_4-{tags[0]}.gpkg")

# %%
tags = ["residential", "service", "cycleway", "busway", "footway"]
dist = 10
simplify_angles = 95
#
nx_final_clean = graphs.nx_consolidate_nodes(
    nx_basic_clean,
    buffer_dist=dist,
    crawl=True,
    osm_hwy_target_tags=tags,
    centroid_by_itx=True,
    prioritise_by_hwy_tag=True,
    simplify_line_angles=simplify_angles,
)
nx_final_clean = graphs.nx_split_opposing_geoms(
    nx_final_clean,
    buffer_dist=dist,
    squash_nodes=True,
    osm_hwy_target_tags=tags,
    centroid_by_itx=False,
    simplify_line_angles=simplify_angles,
)
nx_final_clean = graphs.nx_consolidate_nodes(
    nx_final_clean,
    buffer_dist=dist,
    crawl=False,
    osm_hwy_target_tags=tags,
    centroid_by_itx=False,
    prioritise_by_hwy_tag=True,
    simplify_line_angles=45,
)
nx_final_clean = graphs.nx_remove_filler_nodes(nx_final_clean)
_edges_gdf = io.geopandas_from_nx(nx_final_clean, crs=WORKING_CRS)
_edges_gdf.to_file(f"temp/{map_key}_osm_network_5.gpkg")
