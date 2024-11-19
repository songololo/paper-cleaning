"""
Prepare datasets by running madrid-ua-dataset repo workflow and place in temp folder.
"""

# %%
from __future__ import annotations

import logging

import osmnx as ox
import seaborn as sns
from cityseer.tools import io

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
bounds_buff

# test_geom, _ = io.buffered_point_poly(-3.7010254, 40.4327720, 150)
nx_raw = io.osm_graph_from_poly(
    bounds_buff.simplify(500),
    poly_crs_code=WORKING_CRS,
    to_crs_code=WORKING_CRS,
    simplify=True,
)
_edges_gdf = io.geopandas_from_nx(nx_raw, crs=WORKING_CRS)
_edges_gdf.to_file(f"temp/{map_key}_auto_clean.gpkg")
