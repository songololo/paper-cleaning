# %%
from __future__ import annotations

import json
import pathlib
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from shapely import geometry

sys.path.append(str(pathlib.Path.cwd().parent.absolute()))

from src import util

sns.set_theme(style="whitegrid")

# %%
# plot
hwy_types = ["motorway", "trunk", "primary", "secondary", "tertiary", "residential"]
for hwy_type in hwy_types:
    hwy_metrics = gpd.read_file(f"../temp/buff_stats_{hwy_type}.gpkg")
    for metric in ["poly_count", "poly_area", "free_edges_count", "free_edges_len"]:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=hwy_metrics, x="distance", y=metric, marker="o")
        plt.title(f"{hwy_type} - {metric} vs distance")
        plt.xlabel("buffer distance")
        plt.ylabel(metric)
        plt.show()

# %%
# visualise buffers
for hwy_type in hwy_types:
    hwy_metrics = gpd.read_file(f"../temp/buff_stats_{hwy_type}.gpkg")
    x_base = 3157800
    y_base = 2027800
    span = 1000
    bounding_box = geometry.box(x_base, y_base, x_base + span, y_base + span)
    filtered_edges = hwy_metrics[hwy_metrics.intersects(bounding_box)]
    for buff_dist in hwy_metrics["distance"]:
        half_d = buff_dist / 2
        buffered = filtered_edges.buffer(half_d)
        merged_buffer = buffered.union_all()
        buff = merged_buffer.buffer(-half_d)

        plt.figure(figsize=(10, 6))
        # ax = filtered_edges.plot(edgecolor="#ddd", linewidth=0.5, figsize=(10, 10))
        gdf = gpd.GeoDataFrame({"geom": [buff]}, crs=3035, geometry="geom")
        ax = gdf.plot(facecolor="red", linewidth=0.01)

        ax.set_title(f"{hwy_type} - buffer: {buff_dist}")
        ax.set_xlim(x_base, x_base + span)
        ax.set_ylim(y_base, y_base + span)
        ax.set_axis_off()

        plt.show()
