""" """

# %%
from __future__ import annotations

import logging

import geopandas as gpd
import seaborn as sns
from cityseer.tools import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")

WORKING_CRS = 3035

# %%
# set city key for naming output files
map_key = "nicosia"  # or "madrid"
# set input path for boundary to corresponding file in data directory
bounds_path = "data/nicosia_buffered_bounds.gpkg"  # or "data/madrid_buffered_bounds.gpkg

# %%
bounds_buff = gpd.read_file(bounds_path)
bounds_wgs = bounds_buff.to_crs(4326).union_all()
bounds_buff = bounds_buff.to_crs(WORKING_CRS).union_all()
bounds_buff

# %%
nx_raw = io.osm_graph_from_poly(
    bounds_buff.simplify(500),
    poly_crs_code=WORKING_CRS,
    to_crs_code=WORKING_CRS,
    simplify=True,
)
_edges_gdf = io.geopandas_from_nx(nx_raw, crs=WORKING_CRS)
_edges_gdf.to_file(f"temp/{map_key}_auto_clean.gpkg")
