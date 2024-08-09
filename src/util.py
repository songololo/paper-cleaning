"""

"""

from __future__ import annotations

import json
import os
from typing import Any

import networkx as nx
import psycopg
import sqlalchemy
from cityseer.tools import graphs, io
from pyproj import Transformer
from shapely import geometry


def get_db_config() -> dict[str, str | None]:
    """
    from: https://github.com/UCL/ba-ebdp-toolkit
    """
    db_config_json = os.getenv("DB_CONFIG")
    if db_config_json is None:
        raise ValueError("Unable to retrieve DB_CONFIG environment variable.")
    db_config = json.loads(db_config_json)
    for key in ["host", "port", "user", "dbname", "password"]:
        if key not in db_config:
            raise ValueError(f"Unable to find expected key: {key} in DB_CONFIG")
    return db_config


def get_sqlalchemy_engine() -> sqlalchemy.engine.Engine:
    """
    from: https://github.com/UCL/ba-ebdp-toolkit
    """
    db_config = get_db_config()
    db_con_str = (
        f"postgresql+psycopg://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
    )
    return sqlalchemy.create_engine(db_con_str, pool_pre_ping=True)


def db_execute(query: str, params: tuple[Any] | None = None) -> None:
    """
    from: https://github.com/UCL/ba-ebdp-toolkit
    """
    with psycopg.connect(**get_db_config()) as db_con:  # type: ignore
        with db_con.cursor() as cursor:
            cursor.execute(query, params)
            db_con.commit()


def osm_raw_graph_from_poly(
    poly_geom: geometry.Polygon,
    poly_crs_code: int | str,
    to_crs_code: int | str | None,
) -> nx.MultiGraph:
    """
    from: https://github.com/benchmark-urbanism/cityseer-api/blob/master/pysrc/cityseer/tools/io.py
    repeated here for manual control - generates raw graph
    """
    in_transformer = Transformer.from_crs(poly_crs_code, 4326, always_xy=True)
    coords = [
        in_transformer.transform(lng, lat) for lng, lat in poly_geom.exterior.coords
    ]
    geom_osm = str.join(" ", [f"{lat} {lng}" for lng, lat in coords])
    request = f"""
    /* https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL */
    [out:json];
    (
    way["highway"]
    ["area"!="yes"]
    ["highway"!~"bus_guideway|busway|escape|raceway|proposed|planned|abandoned|platform|construction|emergency_bay|rest_area"]
    ["footway"!="sidewalk"]
    ["service"!~"parking_aisle|driveway|drive-through|slipway"]
    ["amenity"!~"charging_station|parking|fuel|motorcycle_parking|parking_entrance|parking_space"]
    ["indoor"!="yes"]
    ["level"!="-2"]
    ["level"!="-3"]
    ["level"!="-4"]
    ["level"!="-5"]
    (poly:"{geom_osm}");
    );
    out body;
    >;
    out qt;
    """
    osm_response = io.fetch_osm_network(request, timeout=300, max_tries=3)
    graph_wgs = io.nx_from_osm(osm_json=osm_response.text)  # type: ignore
    graph_crs = io.nx_epsg_conversion(graph_wgs, 4326, to_crs_code)
    graph_crs = io.graphs.nx_simple_geoms(graph_crs)
    return graph_crs
