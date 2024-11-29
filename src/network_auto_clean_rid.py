# Import necessary libraries
import osmnx as ox  # For working with OpenStreetMap (OSM) data
from cityseer.tools import io  # For additional I/O tools specific to Cityseer library

# The coordinate reference system (CRS) to use (EPSG:3035 is commonly used for Europe)
WORKING_CRS = 3035
# Buffer distance in meters (10 km) for expanding boundary
# Set this to match the largest distance used for network measures to prevent edge roll-off
buffer_dist = 10000

# Iterate over a list of OSM relation IDs (osm_fid) and corresponding location keys
for osm_fid, location_key in [
    # The OSM relation IDs must match actual OSM boundary codes
    # These can be anything you need
    # Note the prepended "r"
    # The location keys are for naming the output files
    ("r11931566", "delfland"),
    ("r47811", "amsterdam"),
    ("r162378", "birmingham"),
    ("r65606", "london"),
]:
    # Fetch the bounding geometry for the given OSM feature ID
    bounds = ox.geocode_to_gdf(
        osm_fid,  # The OSM relation ID
        by_osmid=True,  # Indicates that geocoding is based on OSM ID
    )
    # Reproject the geometry to the working CRS - this must be projected CRS
    bounds = bounds.to_crs(WORKING_CRS)
    # Create a buffered version of the geometry as a shapely geometry
    bounds_buff = bounds.buffer(buffer_dist).union_all()

    # Fetch the road network using Cityseer's I/O tools
    # Recommend version 4.16.27 (or newer) for latest cleaning workflow
    # The idea is then to further edit the network manually as required
    # The workflow can be customised to target specific OSM highway tags and distances, see for a starting point:
    # https://benchmark-urbanism.github.io/cityseer-examples/examples/graph_cleaning.html#manual-cleaning
    # (Happy to help based on what you're trying to achieve?)
    nx_graph = io.osm_graph_from_poly(
        bounds_buff.simplify(500),
        poly_crs_code=WORKING_CRS,  # CRS of the input polygon
        to_crs_code=WORKING_CRS,  # CRS to which the output graph should be reprojected
        simplify=True,  # Simplify the graph
        final_clean_distances=(  # distances to use for final consolidation steps
            6,
            12,
        ),  # don't go too aggressive - i.e. be careful with more than 12m
        remove_disconnected=100,  # disconnected clusters with fewer than 100 nodes removed
        cycleways=True,  # include cycleways - more convoluted but necessary otherwise important connections missed
        busways=False,  # don't include busways
        green_footways=True,  # can be removed manually from QGIS with associated tag
        green_service_roads=False,  # drop service roads in green spaces (and cemetries etc.)
    )
    # Convert the fetched road network into a GeoDataFrame
    edges_gdf = io.geopandas_from_nx(nx_graph, crs=WORKING_CRS)
    # Save the road network to a GeoPackage file, named after the location key
    edges_gdf.to_file(f"{location_key}_auto_clean_v3.gpkg")
    # By default, Motorways are left alone by cleaning
    # The output network can be filtered in QGIS by highway type based on your needs
    # e.g. remove footway and cycleway types for motorised network
    # e.g. remove motorways and trunk roads for pedestrian
    # Footways intersecting green areas are specially tagged and left alone when cleaning
    # But you may want to remove these before running centrality measures by dropping footway_green
