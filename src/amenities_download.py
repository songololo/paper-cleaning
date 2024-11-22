""" """

# %%
from __future__ import annotations

import json

import geopandas as gpd
import pandas as pd
from osmnx import _errors as ox_errs
from osmnx import features

WORKING_CRS = 3035

# %%
# set city key for naming output files
map_key = "nicosia"  # or "madrid"
# set input path for boundary to corresponding file in data directory
bounds_path = "data/nicosia_buffered_bounds.gpkg"  # or "data/madrid_buffered_bounds.gpkg

# %%
bounds_buff = gpd.read_file(bounds_path)
# create WGS 84 geographic coordinate boundary for passing to osmnx
bounds_wgs = bounds_buff.to_crs(4326).union_all()
# create projected version as well
bounds_buff = bounds_buff.to_crs(WORKING_CRS).union_all()
bounds_buff

# %%
# set OSM tags
SCHEMA = {
    # category name for output as "cat_key" column
    "bus_stop": {
        # OSM key: OSM values for OSM query
        "highway": ["bus_stop"],
    },
    # Educational facilities (nursery, preschools, primary, secondary school, etc..)
    "education": {
        # OSM key: OSM values for OSM query
        "amenity": [
            "school",
            "college",
            "language_school",
            "research_institute",
            "training",
            "university",
            "music_school",
        ]
    },
    # Services centres or post offices (Governmental services)
    "service": {
        # OSM key: OSM values for OSM query
        "amenity": [
            "library",
            "courthouse",
            "fire_station",
            "police",
            "post_office",
            "townhall",
            "community_centre",
            "social_centre",
        ],
    },
}

# %%
dfs = []
for cat_key, val in SCHEMA.items():
    print(f"Fetching {cat_key}")
    for osm_key, osm_vals in val.items():
        try:
            data_gdf = features.features_from_polygon(bounds_wgs, tags={osm_key: osm_vals})
        except ox_errs.InsufficientResponseError as e:
            print(e)
            continue
        data_gdf = data_gdf.to_crs(WORKING_CRS)
        data_gdf.loc[:, "centroids"] = data_gdf.geometry.centroid
        data_gdf.drop(columns="geometry", inplace=True)
        data_gdf.rename(columns={"centroids": "geom"}, inplace=True)
        data_gdf.set_geometry("geom", inplace=True)
        data_gdf.set_crs(WORKING_CRS, inplace=True)
        data_gdf = data_gdf.reset_index(level=0, drop=True)
        data_gdf.index = data_gdf.index.astype(str)
        data_gdf["cat_key"] = cat_key
        data_gdf["osm_key"] = osm_key
        data_gdf["osm_vals"] = json.dumps(osm_vals)
        data_gdf = data_gdf[["cat_key", "osm_key", "osm_vals", "geom"]]
        dfs.append(data_gdf)
landuses_gdf = pd.concat(dfs)
landuses_gdf = landuses_gdf.reset_index()

# %%
# save
landuses_gdf.to_file(f"temp/{map_key}_pois.gpkg")
