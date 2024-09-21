"""
Prepare datasets by running madrid-ua-dataset repo workflow and place in temp folder.
"""

# %%
from __future__ import annotations

import logging
import pathlib
import sys
from typing import cast

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import seaborn as sns
from cityseer import config
from cityseer.tools import graphs, io, util
from cityseer.tools.util import EdgeData, MultiDiGraph, NodeKey
from pyproj import Transformer
from scipy.stats import spearmanr
from shapely import geometry, ops
from shapely.strtree import STRtree
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
# spaces for selecting / discarding footpaths
feats_gdf = ox.features_from_polygon(
    bounds_wgs,
    tags={
        # "landuse": ["cemetery", "forest"],
        # "leisure": ["park", "garden"],
        "highway": ["pedestrian"],
    },
)
ways_gdf = feats_gdf.xs("way", level="element_type", drop_level=True)
relations_gdf = feats_gdf.xs("relation", level="element_type", drop_level=True)
combined_gdf = pd.concat([ways_gdf, relations_gdf])
combined_gdf = relations_gdf.to_crs(network_edges_gdf.crs.to_epsg())
combined_geom = combined_gdf.union_all()
combined_gdf = gpd.GeoDataFrame({"geometry": [combined_geom]}, crs=combined_gdf.crs)
exploded_gdf = combined_gdf.explode(index_parts=False).reset_index(drop=True)
# exploded_gdf = exploded_gdf[exploded_gdf.area >= 20000]
exploded_gdf.to_file("../temp/hwy_poly.gpkg")
feats_bounds = exploded_gdf.union_all().simplify(2)


# %%
def nx_from_osm_nx(
    nx_multidigraph: MultiDiGraph,
    node_attributes: list[str] | None = None,
    edge_attributes: list[str] | None = None,
    tolerance: float = config.ATOL,
) -> nx.MultiGraph:
    if not isinstance(nx_multidigraph, nx.MultiDiGraph):
        raise TypeError(
            "This method requires a directed networkX MultiDiGraph as derived from `OSMnx`."
        )
    if node_attributes is not None and not isinstance(node_attributes, (list, tuple)):
        raise TypeError(
            "Node attributes to be copied should be provided as either a list or tuple of attribute keys."
        )
    if edge_attributes is not None and not isinstance(edge_attributes, (list, tuple)):
        raise TypeError(
            "Edge attributes to be copied should be provided as either a list or tuple of attribute keys."
        )
    logger.info("Converting OSMnx MultiDiGraph to cityseer MultiGraph.")
    # target MultiGraph
    g_multi = nx.MultiGraph()

    def _process_node(nd_key: NodeKey) -> tuple[float, float]:
        # x
        if "x" not in nx_multidigraph.nodes[nd_key]:
            raise KeyError(f'Encountered node missing "x" coordinate attribute for node {nd_key}.')
        x: float = nx_multidigraph.nodes[nd_key]["x"]
        # y
        if "y" not in nx_multidigraph.nodes[nd_key]:
            raise KeyError(f'Encountered node missing "y" coordinate attribute for node {nd_key}.')
        y: float = nx_multidigraph.nodes[nd_key]["y"]
        # add attributes if necessary
        if nd_key not in g_multi:
            g_multi.add_node(nd_key, x=x, y=y)
            if node_attributes is not None:
                for node_att in node_attributes:
                    if node_att not in nx_multidigraph.nodes[nd_key]:
                        raise ValueError(
                            f"Specified attribute {node_att} is not available for node {nd_key}."
                        )
                    g_multi.nodes[nd_key][node_att] = nx_multidigraph.nodes[nd_key][node_att]

        return x, y

    # copy nodes and edges
    start_nd_key: NodeKey
    end_nd_key: NodeKey
    edge_idx: int
    edge_data: EdgeData
    for start_nd_key, end_nd_key, edge_idx, edge_data in tqdm(  # type: ignore
        nx_multidigraph.edges(data=True, keys=True),
        disable=config.QUIET_MODE,  # type: ignore
    ):
        edge_data = cast(EdgeData, edge_data)  # type: ignore
        s_x, s_y = _process_node(start_nd_key)
        e_x, e_y = _process_node(end_nd_key)
        # copy edge if present
        if "geometry" in edge_data:
            line_geom: geometry.LineString = edge_data["geometry"]
        # otherwise create
        else:
            line_geom = geometry.LineString([[s_x, s_y], [e_x, e_y]])
        # check for LineString validity
        if line_geom.geom_type != "LineString":
            raise TypeError(
                f"Expecting LineString geometry but found {line_geom.geom_type} geometry for "
                f"edge {start_nd_key}-{end_nd_key}."
            )
        # orient LineString
        geom_coords = line_geom.coords
        if not np.allclose((s_x, s_y), geom_coords[0][:2], atol=tolerance, rtol=0):
            geom_coords = util.align_linestring_coords(geom_coords, (s_x, s_y))
        # check starting and ending tolerances
        if not np.allclose((s_x, s_y), geom_coords[0][:2], atol=tolerance, rtol=0):
            raise ValueError(
                "Starting node coordinates don't match LineString geometry starting coordinates."
            )
        if not np.allclose((e_x, e_y), geom_coords[-1][:2], atol=tolerance, rtol=0):
            raise ValueError(
                "Ending node coordinates don't match LineString geometry ending coordinates."
            )
        # snap starting and ending coords to avoid rounding error issues
        geom_coords = util.snap_linestring_startpoint(geom_coords, (s_x, s_y))
        geom_coords = util.snap_linestring_endpoint(geom_coords, (e_x, e_y))
        g_multi.add_edge(
            start_nd_key,
            end_nd_key,
            key=edge_idx,
            geom=geometry.LineString(geom_coords),
        )
        if edge_attributes is not None:
            for edge_att in edge_attributes:
                if edge_att not in edge_data:
                    raise ValueError(
                        f"Attribute {edge_att} is not available for edge {start_nd_key}-{end_nd_key}."
                    )
                g_multi[start_nd_key][end_nd_key][edge_idx][edge_att] = edge_data[edge_att]

    return g_multi


# %%
def nx_split_opposing_geoms(
    nx_multigraph: nx.MultiGraph,
    buffer_dist: float = 12,
    merge_edges_by_midline: bool = True,
    contains_buffer_dist: int = 25,
    prioritise_by_hwy_tag: bool = False,
    osm_hwy_target_tags: list[str] | None = None,
    osm_matched_tags_only: bool = False,
    min_node_degree: int = 2,
    max_node_degree: int | None = None,
    squash_nodes: bool = True,
) -> nx.MultiGraph:
    def make_edge_key(start_nd_key: NodeKey, end_nd_key: NodeKey, edge_idx: int) -> str:
        return "-".join(sorted([str(start_nd_key), str(end_nd_key)])) + f"-k{edge_idx}"

    # where edges are deleted, keep track of new children edges
    edge_children: dict[str, list] = {}

    # recursive function for retrieving nested layers of successively replaced edges
    def recurse_child_keys(
        _start_nd_key: NodeKey,
        _end_nd_key: NodeKey,
        _edge_idx: int,
        _edge_data: dict,
        current_edges: list,
    ):
        """
        Recursively checks if an edge has been replaced by children, if so, use children instead.
        """
        edge_key = make_edge_key(_start_nd_key, _end_nd_key, _edge_idx)
        # if an edge does not have children, add to current_edges and return
        if edge_key not in edge_children:
            current_edges.append((_start_nd_key, _end_nd_key, _edge_idx, _edge_data))
        # otherwise recursively drill-down until newest edges are found
        else:
            for child_s, child_e, child_k, child_data in edge_children[edge_key]:
                recurse_child_keys(child_s, child_e, child_k, child_data, current_edges)

    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph.")
    _multi_graph = nx_multigraph.copy()
    # if using OSM tags heuristic
    hwy_tags = graphs._extract_tags_to_set(osm_hwy_target_tags)
    # create an edges STRtree (nodes and edges)
    edges_tree, edge_lookups = util.create_edges_strtree(_multi_graph)
    # node groups
    node_groups: list[set] = []
    # iter
    logger.info("Splitting opposing edges.")
    # iterate origin graph (else node structure changes in place)
    nd_key: NodeKey
    for nd_key, nd_data in tqdm(nx_multigraph.nodes(data=True), disable=config.QUIET_MODE):
        # don't split opposing geoms from nodes of degree 1
        nd_degree = nx.degree(_multi_graph, nd_key)
        if nd_degree < min_node_degree or nd_degree > max_node_degree:
            continue
        # check tags
        if osm_hwy_target_tags:
            nb_hwy_tags = graphs._gather_nb_tags(nx_multigraph, nd_key, "highways")
            if not hwy_tags.intersection(nb_hwy_tags):
                continue
        # get name tags for matching against potential gapped edges
        nb_name_tags = graphs._gather_nb_name_tags(nx_multigraph, nd_key)
        # get all other edges within the buffer distance
        # the spatial index using bounding boxes, so further filtering is required (see further down)
        # furthermore, successive iterations may remove old edges, so keep track of removed parent vs new child edges
        n_point = geometry.Point(nd_data["x"], nd_data["y"])
        # spatial query from point returns all buffers with buffer_dist
        edge_hits: list[int] = edges_tree.query(n_point.buffer(buffer_dist))  # type: ignore
        # extract the start node, end node, geom
        edges: list = []
        for edge_hit_idx in edge_hits:
            edge_lookup = edge_lookups[edge_hit_idx]
            start_nd_key = edge_lookup["start_nd_key"]
            end_nd_key = edge_lookup["end_nd_key"]
            edge_idx = edge_lookup["edge_idx"]
            edge_data: dict = nx_multigraph[start_nd_key][end_nd_key][edge_idx]
            # don't add neighbouring edges
            if nd_key in (start_nd_key, end_nd_key):
                continue
            edges.append((start_nd_key, end_nd_key, edge_idx, edge_data))
        # review gapped edges
        # if already removed, get the new child edges
        current_edges: list = []
        for start_nd_key, end_nd_key, edge_idx, edge_data in edges:
            recurse_child_keys(start_nd_key, end_nd_key, edge_idx, edge_data, current_edges)
        # check that edges are within buffer
        gapped_edges: list = []
        for start_nd_key, end_nd_key, edge_idx, edge_data in current_edges:
            edge_geom = edge_data["geom"]
            # check whether the geom is truly within the buffer distance
            if edge_geom.distance(n_point) > buffer_dist:
                continue
            gapped_edges.append((start_nd_key, end_nd_key, edge_idx, edge_data))
        # abort if no gapped edges
        if not gapped_edges:
            continue
        # prepare the root node's point geom
        n_geom = geometry.Point(nd_data["x"], nd_data["y"])
        # nodes for squashing
        node_group = [nd_key]
        # iter gapped edges
        for start_nd_key, end_nd_key, edge_idx, edge_data in gapped_edges:
            edge_geom = edge_data["geom"]
            # hwy tags
            if osm_hwy_target_tags:
                edge_hwy_tags = graphs._tags_from_edge_key(edge_data, "highways")
                if not hwy_tags.intersection(edge_hwy_tags):
                    continue
            # names tags
            if osm_matched_tags_only is True:
                edge_name_tags = graphs._gather_name_tags(edge_data)
                if not nb_name_tags.intersection(edge_name_tags):
                    continue
            # project a point and split the opposing geom
            # ops.nearest_points returns tuple of nearest from respective input geoms
            # want the nearest point on the line at index 1
            nearest_point: geometry.Point = ops.nearest_points(n_geom, edge_geom)[-1]
            # if a valid nearest point has been found, go ahead and split the geom
            # use a snap because rounding precision errors will otherwise cause issues
            split_geoms: geometry.GeometryCollection = ops.split(
                ops.snap(edge_geom, nearest_point, 0.01), nearest_point
            )
            # in some cases the line will be pointing away, but is still near enough to be within max
            # in these cases a single geom will be returned
            if len(split_geoms.geoms) < 2:
                continue
            new_edge_geom_a: geometry.LineString
            new_edge_geom_b: geometry.LineString
            new_edge_geom_a, new_edge_geom_b = split_geoms.geoms  # type: ignore
            # add the new node and edges to _multi_graph (don't modify nx_multigraph because of iter in place)
            new_nd_name, is_dupe = util.add_node(
                _multi_graph,
                [start_nd_key, nd_key, end_nd_key],
                x=nearest_point.x,
                y=nearest_point.y,
            )
            # continue if a node already exists at this location
            if is_dupe:
                continue
            node_group.append(new_nd_name)
            # copy edge data
            edge_data_copy = {k: v for k, v in edge_data.items() if k != "geom"}
            _multi_graph.add_edge(start_nd_key, new_nd_name, **edge_data_copy)
            _multi_graph.add_edge(end_nd_key, new_nd_name, **edge_data_copy)
            # get starting geom for orientation
            s_nd_data = _multi_graph.nodes[start_nd_key]
            s_nd_geom = geometry.Point(s_nd_data["x"], s_nd_data["y"])
            if np.allclose(
                s_nd_geom.coords,
                new_edge_geom_a.coords[0][:2],
                atol=config.ATOL,
                rtol=0,
            ) or np.allclose(
                s_nd_geom.coords,
                new_edge_geom_a.coords[-1][:2],
                atol=config.ATOL,
                rtol=0,
            ):
                s_new_geom = new_edge_geom_a
                e_new_geom = new_edge_geom_b
            else:
                # double check matching geoms
                if not np.allclose(
                    s_nd_geom.coords,
                    new_edge_geom_b.coords[0][:2],
                    atol=config.ATOL,
                    rtol=0,
                ) and not np.allclose(
                    s_nd_geom.coords,
                    new_edge_geom_b.coords[-1][:2],
                    atol=config.ATOL,
                    rtol=0,
                ):
                    raise ValueError("Unable to match split geoms to existing nodes")
                s_new_geom = new_edge_geom_b
                e_new_geom = new_edge_geom_a
            # if splitting a looped component, then both new edges will have the same starting and ending nodes
            # in these cases, there will be multiple edges
            if start_nd_key == end_nd_key:
                if _multi_graph.number_of_edges(start_nd_key, new_nd_name) != 2:
                    raise ValueError(
                        f"Number of edges between {start_nd_key} and {new_nd_name} does not equal 2"
                    )
                s_k = 0
                e_k = 1
            else:
                if _multi_graph.number_of_edges(start_nd_key, new_nd_name) != 1:
                    raise ValueError(
                        f"Number of edges between {start_nd_key} and {new_nd_name} does not equal 1."
                    )
                if _multi_graph.number_of_edges(end_nd_key, new_nd_name) != 1:
                    raise ValueError(
                        f"Number of edges between {end_nd_key} and {new_nd_name} does not equal 1."
                    )
                s_k = e_k = 0
            # write the new edges
            _multi_graph[start_nd_key][new_nd_name][s_k]["geom"] = s_new_geom
            _multi_graph[end_nd_key][new_nd_name][e_k]["geom"] = e_new_geom
            # add the new edges to the edge_children dictionary
            edge_key = make_edge_key(start_nd_key, end_nd_key, edge_idx)
            edge_children[edge_key] = [
                (
                    start_nd_key,
                    new_nd_name,
                    s_k,
                    _multi_graph[start_nd_key][new_nd_name][s_k],
                ),
                (
                    end_nd_key,
                    new_nd_name,
                    e_k,
                    _multi_graph[end_nd_key][new_nd_name][e_k],
                ),
            ]
            # drop the old edge from _multi_graph
            if _multi_graph.has_edge(start_nd_key, end_nd_key, edge_idx):
                _multi_graph.remove_edge(start_nd_key, end_nd_key, edge_idx)
        node_groups.append(list(node_group))
    # iter and squash
    if squash_nodes is True:
        logger.info("Squashing opposing nodes")
        for node_group in node_groups:
            _multi_graph = graphs._squash_adjacent(
                _multi_graph,
                node_group,
                centroid_by_itx=True,
                prioritise_by_hwy_tag=prioritise_by_hwy_tag,
            )
    else:
        for node_group in node_groups:
            origin_nd_key = node_group.pop(0)
            for new_nd_key in node_group:
                origin_nd_data = _multi_graph.nodes[origin_nd_key]
                new_nd_data = _multi_graph.nodes[new_nd_key]
                _multi_graph.add_edge(
                    origin_nd_key,
                    new_nd_key,
                    names=[],
                    routes=[],
                    highways=[],
                    geom=geometry.LineString(
                        [
                            [origin_nd_data["x"], origin_nd_data["y"]],
                            [new_nd_data["x"], new_nd_data["y"]],
                        ]
                    ),
                )
    # squashing nodes can result in edge duplicates
    deduped_graph = graphs.nx_merge_parallel_edges(
        _multi_graph,
        merge_edges_by_midline,
        contains_buffer_dist,
        osm_hwy_target_tags,
        osm_matched_tags_only,
    )
    deduped_graph = graphs.nx_remove_filler_nodes(deduped_graph)

    return deduped_graph


# %%
def osm_graph_from_poly(
    poly_geom,
    poly_crs_code: int | str = 4326,
    to_crs_code: int | str = 4326,
):
    in_transformer = Transformer.from_crs(poly_crs_code, 4326, always_xy=True)
    coords = [in_transformer.transform(lng, lat) for lng, lat in poly_geom.exterior.coords]
    geom_osm = str.join(" ", [f"{lat} {lng}" for lng, lat in coords])
    osm_response = io.fetch_osm_network(request)
    graph_wgs = io.nx_from_osm(osm_json=osm_response.text)  # type: ignore
    # cast to UTM
    graph_crs = io.nx_epsg_conversion(graph_wgs, 4326, to_crs_code)
    graph_crs = graphs.nx_simple_geoms(graph_crs)
    graph_crs = graphs.nx_remove_filler_nodes(graph_crs)
    return graph_crs


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
# test_geom, _ = io.buffered_point_poly(-3.7010254, 40.4327720, 150)
nx_raw = io.osm_graph_from_poly(
    bounds_buff.simplify(100),
    # test_geom,
    poly_crs_code=network_edges_gdf.crs.to_epsg(),
    # poly_crs_code=4326,
    to_crs_code=network_edges_gdf.crs.to_epsg(),
    simplify=False,
    iron_edges=False,
    custom_request=request,
)

# %%
nx_copy = nx_raw.copy()
# discard edges intersecting green spaces
uniq_hwy_tags = set()
remove_edges = []
# use STR Tree for performance
geom_index = STRtree(exploded_gdf.geometry.to_list())
# iter edges to find edges for removal
for start_node_key, end_node_key, edge_data in tqdm(
    nx_copy.edges(data=True), total=nx_copy.number_of_edges()
):
    edge_geom = edge_data["geom"]
    # remove if footway if not itx with pedestrian poly
    if "footway" in edge_data["highways"]:
        itx = geom_index.query(edge_geom)
        if not len(itx) or not feats_bounds.intersects(edge_geom):
            remove_edges.append((start_node_key, end_node_key))
            continue
    if "highways" in edge_data:
        uniq_hwy_tags.update(edge_data["highways"])
# report and remove
print(f"Hwy tags: {uniq_hwy_tags}")
print(f"Edges before removing green itx: {nx_copy.number_of_edges()}")
# remove all edges regardless of index between start & end nodes
# otherwise index ordering becomes unpredictable
nx_copy.remove_edges_from(remove_edges)
print(f"Edges after removing green itx: {nx_copy.number_of_edges()}")

# %%
_, _edges_gdf, _ = io.network_structure_from_nx(nx_copy, crs=network_edges_gdf.crs.to_epsg())
_edges_gdf.to_file("../temp/osm_network_raw.gpkg")

# %%
nx_basic_clean = graphs.nx_remove_filler_nodes(nx_copy)
nx_basic_clean = graphs.nx_remove_dangling_nodes(nx_basic_clean, remove_disconnected=0)
nx_basic_clean = graphs.nx_remove_filler_nodes(nx_basic_clean)
nx_basic_clean = nx_split_opposing_geoms(
    nx_basic_clean,
    buffer_dist=15,
    min_node_degree=1,
    max_node_degree=1,
    squash_nodes=False,
)
# nx_basic_clean = graphs.nx_consolidate_nodes(
#     nx_basic_clean,
#     buffer_dist=5,
#     neighbour_policy="indirect",
#     crawl=False,
#     centroid_by_itx=True,
#     prioritise_by_hwy_tag=True,
# )
_, _edges_gdf, _ = io.network_structure_from_nx(nx_basic_clean, crs=network_edges_gdf.crs.to_epsg())
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
clean_dists = [(15, 8)]
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
    for nd_key, nd_data in tqdm(G_nx_dual.nodes(data=True), total=G_nx_dual.number_of_nodes()):
        point = geometry.Point(nd_data["x"], nd_data["y"])
        G_nx_dual.nodes[nd_key]["live"] = point.intersects(bounds)
    # network structure
    nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(
        G_nx_dual, crs=network_edges_gdf.crs.to_epsg()
    )
    # # create separate network structure with length weighted data
    # for nd_key, nd_data in tqdm(
    #     G_nx_dual.nodes(data=True), total=G_nx_dual.number_of_nodes()
    # ):
    #     G_nx_dual.nodes[nd_key]["weight"] = nd_data["primal_edge"].length
    # # extract length weighted structure
    # nodes_gdf, edges_gdf, network_structure_len_wt = io.network_structure_from_nx(
    #     G_nx_dual, crs=edges_gdf.crs
    # )
    # # compute length weighted centralities
    # nodes_gdf = networks.node_centrality_shortest(
    #     network_structure_len_wt, nodes_gdf, distances=cent_distances
    # )
    # # rename length weighted columns for saving (prevents overwriting)
    # for col_extract in [
    #     "cc_density",
    #     "cc_beta",
    #     "cc_farness",
    #     "cc_harmonic",
    #     "cc_hillier",
    #     "cc_betweenness",
    #     "cc_betweenness_beta",
    # ]:
    #     new_col_extract = col_extract.replace("cc_", "cc_lw_")
    #     nodes_gdf.columns = [
    #         col.replace(col_extract, new_col_extract) for col in nodes_gdf.columns
    #     ]
    # # compute unweighted centralities
    # nodes_gdf = networks.node_centrality_shortest(
    #     network_structure, nodes_gdf, distances=cent_distances
    # )
    # # compute accessibilities
    # nodes_gdf, prems = layers.compute_accessibilities(
    #     prems,
    #     landuse_column_label="division_desc",
    #     accessibility_keys=[
    #         "food_bev",
    #         "creat_entert",
    #         "retail",
    #         "services",
    #         "education",
    #         "accommod",
    #         "sports_rec",
    #         "health",
    #     ],
    #     nodes_gdf=nodes_gdf,
    #     network_structure=network_structure,
    #     distances=lu_distances,
    # )
    # # compute mixed uses
    # nodes_gdf, prems = layers.compute_mixed_uses(
    #     prems,
    #     landuse_column_label="division_desc",
    #     nodes_gdf=nodes_gdf,
    #     network_structure=network_structure,
    #     distances=lu_distances,
    # )
    # # filter by live nodes
    nodes_gdf = nodes_gdf[nodes_gdf.live]
    # save and track in dict
    nodes_gdf.to_file(f"../temp/df_{dist_par}_{dist_cons}_clean.gpkg")
    dfs[dist_par] = nodes_gdf

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
        stats.loc[dist, f"node_density_{cent_dist}"] = np.nanmean(df[[f"cc_density_{cent_dist}"]])
        stats.loc[dist, f"street_density_{cent_dist}"] = np.nanmean(
            df[[f"cc_lw_density_{cent_dist}"]]
        )
        corr_coef, p_value = spearmanr(df[[f"cc_harmonic_{cent_dist}"]], df[["cc_food_bev_100_wt"]])
        stats.loc[dist, f"lu_corr_{cent_dist}"] = corr_coef
stats
