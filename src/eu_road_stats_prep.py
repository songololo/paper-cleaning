# %%
from __future__ import annotations

import pathlib
import sys

import geopandas as gpd
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
            ORDER BY ST_Area(geom) DESC
            LIMIT 10
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


def filter_geoms_by_size(
    multi_geom: geometry.MultiPolygon, min_area: int
) -> geometry.MultiPolygon:
    """ """
    filtered_geoms = [poly for poly in multi_geom.geoms if poly.area >= min_area]
    return geometry.MultiPolygon(filtered_geoms)


# %%
hwy_types = ["motorway", "trunk", "primary", "secondary", "tertiary", "residential"]
buff_ranges = [
    [20, 22, 24, 26, 28, 30],
    [18, 20, 22, 24, 26, 28],
    [16, 18, 20, 22, 24, 26],
    [14, 16, 18, 20, 22, 24],
    [10, 12, 14, 16, 18, 20],
    [4, 8, 12, 16, 20, 24],
]
for buff_range, hwy_type in zip(buff_ranges, hwy_types):
    print(f"Processing {hwy_type}:")
    # fetch network
    edges_gdf = fetch_network_by_hwy_type(hwy_type)
    edges_gdf = edges_gdf.simplify(2)
    # gather buffering stats
    buffs = []
    rev_buffs = []
    poly_counts = []
    poly_areas = []
    free_edges_count = []
    free_edges_len = []
    for buff_dist in buff_range:
        print(f"...buffer distance: {buff_dist}")
        half_dist = buff_dist / 2
        edges_buff = edges_gdf.buffer(half_dist).union_all()
        edges_rev_buff = edges_buff.buffer(-half_dist)
        # clean up small artefacts
        edges_rev_buff = filter_geoms_by_size(edges_rev_buff, 50)
        edges_rev_buff = edges_rev_buff.simplify(1)
        rev_buffs.append(edges_rev_buff)
        poly_counts.append(len(edges_rev_buff.geoms))
        poly_areas.append(edges_rev_buff.area)
        edges_diff = edges_gdf.difference(edges_rev_buff)
        filtered_gdf = edges_diff[~(edges_diff.geometry.is_empty)]
        free_edges_count.append(len(filtered_gdf) / len(edges_gdf))
        free_edges_len.append(filtered_gdf.length.sum() / edges_gdf.length.sum())
    # gather into gdf
    metrics = gpd.GeoDataFrame(
        {
            "distance": buff_range,
            "poly_count": poly_counts,
            "poly_area": poly_areas,
            "free_edges_count": free_edges_count,
            "free_edges_len": free_edges_len,
            "rev_buff": rev_buffs,
        },
        geometry="rev_buff",
        crs=3035,
    )
    metrics.to_file(f"../temp/buff_stats_{hwy_type}.gpkg")
