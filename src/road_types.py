# %%
from __future__ import annotations

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
util.db_execute(
    """
    CREATE INDEX IF NOT EXISTS overture_network_edges_raw_hwys_idx
        ON overture.network_edges_raw (highways);    
    """
)


# %%
def fetch_network_by_hwy_type(
    hwy_type: str,
):
    """
    Roughly 2 minute query on trunk roads
    """
    engine = util.get_sqlalchemy_engine()
    edges_gdf = gpd.read_postgis(
        f"""
        WITH bounds AS (
            SELECT geom
            FROM eu.bounds
            WHERE fid = 623
        )
        SELECT ne.*
        FROM overture.network_edges_raw ne
        JOIN bounds b ON ST_Intersects(b.geom, ne.geom)
        WHERE ne.highways = '[''{hwy_type}'']'
        """,
        engine,
        index_col="fid",
        geom_col="geom",
    )
    return edges_gdf


# %%
for buff_range, hwy_type in [
    (list(range(5, 85, 5)), "motorway"),
    (list(range(5, 65, 5)), "trunk"),
    (list(range(5, 55, 5)), "primary"),
    (list(range(5, 45, 5)), "secondary"),
    (list(range(5, 35, 5)), "tertiary"),
    (list(range(2, 22, 2)), "residential"),
]:
    print(f"Processing {hwy_type}:")
    # fetch network
    edges_gdf = fetch_network_by_hwy_type(hwy_type)
    # gather buffering stats
    buffs = []
    poly_counts = []
    poly_areas = []
    for buff_dist in buff_range:
        print(f"...buffer distance: {buff_dist}")
        half_dist = buff_dist / 2
        edges_buff = edges_gdf.buffer(half_dist).union_all()
        edges_rev_buff = edges_buff.buffer(-half_dist)
        buffs.append(edges_rev_buff)
        poly_counts.append(len(edges_buff.geoms))
        poly_areas.append(edges_buff.area)
    # gather into gdf
    metrics = pd.DataFrame(
        {
            "distances": buff_range,
            "poly_counts": poly_counts,
            "poly_areas": poly_areas,
        }
    )
    # plot
    for metric in [
        "poly_counts",
        "poly_areas",
    ]:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=metrics, x="distances", y=metric, marker="o")
        plt.title(f"{metric} vs Buffer Distance")
        plt.xlabel("Buffer Distance")
        plt.ylabel(metric)
        plt.show()

    # visualise buffers
    x_base = 3157800
    y_base = 2027800
    span = 1000
    bounding_box = geometry.box(x_base, y_base, x_base + span, y_base + span)
    filtered_edges = edges_gdf[edges_gdf.intersects(bounding_box)]

    for buff_dist in buff_range:
        half_d = buff_dist / 2
        buffered = filtered_edges.buffer(half_d)
        merged_buffer = buffered.union_all()
        buff = merged_buffer.buffer(-half_d)

        ax = edges_gdf.plot(edgecolor="#ddd", linewidth=0.5, figsize=(10, 10))
        gdf = gpd.GeoDataFrame({"geom": [buff]}, crs=3035, geometry="geom")
        gdf.plot(ax=ax, facecolor="red", linewidth=0.01)

        ax.set_title(f"{hwy_type} - buffer: {buff_dist}")
        ax.set_xlim(x_base, x_base + span)
        ax.set_ylim(y_base, y_base + span)
        ax.set_axis_off()

        plt.show()
