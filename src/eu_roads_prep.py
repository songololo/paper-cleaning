"""
Prepare datasets by running madrid-ua-dataset repo workflow and place in temp folder.
"""

# %%
from __future__ import annotations

import pathlib
import sys

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import seaborn as sns
from cityseer.metrics import layers, networks
from cityseer.tools import graphs, io
from scipy.stats import spearmanr
from shapely import geometry
from shapely.strtree import STRtree
from tqdm import tqdm

# %%
sys.path.append(str(pathlib.Path.cwd().parent.absolute()))

sns.set_theme(style="whitegrid")

# %%
# bounds
network_edges_gdf = gpd.read_file("../temp/neighbourhoods.gpkg")
assert network_edges_gdf.crs.is_projected
bounds = network_edges_gdf.union_all()
bounds_wgs = network_edges_gdf.to_crs(4326).union_all()
bounds_buff = bounds.buffer(10000)

# %%
# green spaces for discarding footpaths
green_gdf = ox.features_from_polygon(
    bounds_wgs, tags={"landuse": ["cemetery", "forest"], "leisure": ["park", "garden"]}
)
ways_gdf = green_gdf.xs("way", level="element_type", drop_level=True)
relations_gdf = green_gdf.xs("relation", level="element_type", drop_level=True)
combined_gdf = pd.concat([ways_gdf, relations_gdf])
combined_gdf = combined_gdf.to_crs(network_edges_gdf.crs.to_epsg())
combined_geom = combined_gdf.union_all()
combined_gdf = gpd.GeoDataFrame({"geometry": [combined_geom]}, crs=combined_gdf.crs)
exploded_gdf = combined_gdf.explode(index_parts=False).reset_index(drop=True)
exploded_gdf = exploded_gdf[exploded_gdf.area >= 20000]
exploded_gdf.to_file("../temp/discard_green_space.gpkg")
green_bounds = exploded_gdf.union_all().simplify(2)

# %%
request = """
    /* https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL */
    [out:json];
    (
    way["highway"]
        ["highway"~"trunk|motorway"]
        ["tunnel"!="yes"](poly:"{geom_osm}");
    way["highway"]
        ["highway"!~"trunk|motorway|bus_guideway|busway|escape|raceway|proposed|planned|abandoned|platform|construction|emergency_bay|rest_area|disused|path|corridor|ladder|bus_stop|track|elevator|service|services"]
        ["area"!="yes"]
        ["footway"!="sidewalk"]
        ["amenity"!~"charging_station|parking|fuel|motorcycle_parking|parking_entrance|parking_space"]
        ["indoor"!="yes"]
        ["tunnel"!="yes"]
        ["level"!="-2"]
        ["level"!="-3"]
        ["level"!="-4"]
        ["level"!="-5"](poly:"{geom_osm}");
    );
    out body;
    >;
    out qt;
"""
nx_raw = io.osm_graph_from_poly(
    bounds_buff.simplify(100),
    poly_crs_code=network_edges_gdf.crs.to_epsg(),
    to_crs_code=network_edges_gdf.crs.to_epsg(),
    simplify=False,
    iron_edges=False,
    custom_request=request,
)

# %%
# discard edges intersecting green spaces
uniq_hwy_tags = set()
remove_edges = []
# use STR Tree for performance
green_geom_index = STRtree(exploded_gdf.geometry.to_list())
# iter edges to find edges for removal
for start_node_key, end_node_key, edge_data in tqdm(
    nx_raw.edges(data=True), total=nx_raw.number_of_edges()
):
    edge_geom = edge_data["geom"]
    # if intersecting boundings box
    itx = green_geom_index.query(edge_geom)
    if len(itx):
        # if a footway
        if "footway" in edge_data["highways"]:
            # check for true contains
            if green_bounds.intersects(edge_geom):
                # remove if so
                remove_edges.append((start_node_key, end_node_key))
                continue
    if "highways" in edge_data:
        uniq_hwy_tags.update(edge_data["highways"])
# report and remove
print(f"Hwy tags: {uniq_hwy_tags}")
print(f"Edges before removing green itx: {nx_raw.number_of_edges()}")
# remove all edges regardless of index between start & end nodes
# otherwise index ordering becomes unpredictable
nx_raw.remove_edges_from(remove_edges)
print(f"Edges after removing green itx: {nx_raw.number_of_edges()}")

# %%
nx_basic_clean = graphs.nx_remove_filler_nodes(nx_raw)
nx_basic_clean = graphs.nx_remove_dangling_nodes(
    nx_basic_clean, despine=15, remove_disconnected=100
)
_, _edges_gdf, _ = io.network_structure_from_nx(
    nx_basic_clean, crs=network_edges_gdf.crs.to_epsg()
)
_edges_gdf.to_file("../temp/osm_network.gpkg")

# %%
cent_distances = [500, 1000, 2000, 5000]
lu_distances = [100, 200, 500]
hwy_target_tags = [
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "residential",
]
dfs = {}
clean_dists = [(10, 5)]
# clean_dists = [0, 2, 4, 6, 8, 10, 12, 16, 20, 24]

# %%
for dist_par, dist_cons in clean_dists:
    # load a fresh copy of prems
    # assignments will change across iterations
    prems = gpd.read_file("../temp/premises_clean.gpkg")
    prems.index = prems.index.astype(str)
    # simplification by current distance
    nx_simpl = graphs.nx_split_opposing_geoms(
        nx_basic_clean,
        buffer_dist=dist_par,
        prioritise_by_hwy_tag=True,
        osm_hwy_target_tags=hwy_target_tags,
        osm_matched_tags_only=True,
    )
    nx_simpl = graphs.nx_consolidate_nodes(
        nx_simpl,
        buffer_dist=dist_cons,
        crawl=True,
        prioritise_by_hwy_tag=True,
        osm_hwy_target_tags=hwy_target_tags,
        osm_matched_tags_only=True,
    )
    # to dual
    G_nx_dual = graphs.nx_to_dual(nx_simpl)
    # set live nodes for nodes within boundary
    for nd_key, nd_data in tqdm(
        G_nx_dual.nodes(data=True), total=G_nx_dual.number_of_nodes()
    ):
        point = geometry.Point(nd_data["x"], nd_data["y"])
        G_nx_dual.nodes[nd_key]["live"] = point.intersects(bounds)
    # network structure
    nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(
        G_nx_dual, crs=network_edges_gdf.crs.to_epsg()
    )
    # create separate network structure with length weighted data
    for nd_key, nd_data in tqdm(
        G_nx_dual.nodes(data=True), total=G_nx_dual.number_of_nodes()
    ):
        G_nx_dual.nodes[nd_key]["weight"] = nd_data["primal_edge"].length
    # extract length weighted structure
    nodes_gdf, edges_gdf, network_structure_len_wt = io.network_structure_from_nx(
        G_nx_dual, crs=edges_gdf.crs
    )
    # compute length weighted centralities
    nodes_gdf = networks.node_centrality_shortest(
        network_structure_len_wt, nodes_gdf, distances=cent_distances
    )
    # rename length weighted columns for saving (prevents overwriting)
    for col_extract in [
        "cc_density",
        "cc_beta",
        "cc_farness",
        "cc_harmonic",
        "cc_hillier",
        "cc_betweenness",
        "cc_betweenness_beta",
    ]:
        new_col_extract = col_extract.replace("cc_", "cc_lw_")
        nodes_gdf.columns = [
            col.replace(col_extract, new_col_extract) for col in nodes_gdf.columns
        ]
    # compute unweighted centralities
    nodes_gdf = networks.node_centrality_shortest(
        network_structure, nodes_gdf, distances=cent_distances
    )
    # compute accessibilities
    nodes_gdf, prems = layers.compute_accessibilities(
        prems,
        landuse_column_label="division_desc",
        accessibility_keys=[
            "food_bev",
            "creat_entert",
            "retail",
            "services",
            "education",
            "accommod",
            "sports_rec",
            "health",
        ],
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=lu_distances,
    )
    # compute mixed uses
    nodes_gdf, prems = layers.compute_mixed_uses(
        prems,
        landuse_column_label="division_desc",
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=lu_distances,
    )
    # filter by live nodes
    nodes_gdf = nodes_gdf[nodes_gdf.live]
    # save and track in dict
    nodes_gdf.to_file(f"../temp/df_{dist}_clean.gpkg")
    dfs[dist] = nodes_gdf

# %%
# load if necessary - very slow
dfs_load = {}
clean_dists = [0, 2, 4, 6, 8, 10, 12, 16, 20, 24]
if False:
    for dist in clean_dists:
        dfs_load[clean_dists] = gpd.read_file(f"../temp/df_{dist}_clean.gpkg")
# manually run dfs = dfs_load to prevent accidental loss of loaded data

# %%
mad_gpd = gpd.read_file("../temp/dataset.gpkg")
dfs["mn"] = mad_gpd

# %%
stats = pd.DataFrame()
for dist, df in dfs.items():
    stats.loc[dist, "node_count"] = len(df)
    for cent_dist in cent_distances:
        stats.loc[dist, f"node_density_{cent_dist}"] = np.nanmean(
            df[[f"cc_density_{cent_dist}"]]
        )
        stats.loc[dist, f"street_density_{cent_dist}"] = np.nanmean(
            df[[f"cc_lw_density_{cent_dist}"]]
        )
        corr_coef, p_value = spearmanr(
            df[[f"cc_harmonic_{cent_dist}"]], df[[f"cc_food_bev_100_wt"]]
        )
        stats.loc[dist, f"lu_corr_{cent_dist}"] = corr_coef
stats
