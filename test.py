import osmnx as ox
import time
from shapely.geometry import Polygon
import os
import numpy as np


def stringify_nonnumeric_cols(gdf):
    import pandas as pd
    for col in (c for c in gdf.columns if not c == "geometry"):
        if not pd.api.types.is_numeric_dtype(gdf[col]):
            gdf[col] = gdf[col].fillna("").astype(str)
    return gdf


def save_graph_shapefile_directional(G, filepath=None, encoding="utf-8"):
    # default filepath if none was provided
    if filepath is None:
        filepath = os.path.join(ox.settings.data_folder, "graph_shapefile")

    # if save folder does not already exist, create it (shapefiles
    # get saved as set of files)
    if not filepath == "" and not os.path.exists(filepath):
        os.makedirs(filepath)
    filepath_nodes = os.path.join(filepath, "nodes.shp")
    filepath_edges = os.path.join(filepath, "edges.shp")

    # convert undirected graph to gdfs and stringify non-numeric columns
    gdf_nodes, gdf_edges = ox.utils_graph.graph_to_gdfs(G)
    gdf_nodes = stringify_nonnumeric_cols(gdf_nodes)
    gdf_edges = stringify_nonnumeric_cols(gdf_edges)
    # We need an unique ID for each edge
    gdf_edges["fid"] = np.arange(0, gdf_edges.shape[0], dtype='int')
    # save the nodes and edges as separate ESRI shapefiles
    gdf_nodes.to_file(filepath_nodes, encoding=encoding)
    gdf_edges.to_file(filepath_edges, encoding=encoding)


if __name__ == "__main__":
    from args import *

    print("osmnx version", ox.__version__)

    # Download by a bounding box
    bounds = (min_lat, max_lat, min_lon, max_lon)
    x1, x2, y1, y2 = bounds
    boundary_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    G = ox.graph_from_polygon(boundary_polygon, network_type='drive')
    start_time = time.time()
    save_graph_shapefile_directional(G, filepath='./network-new')
    print("--- %s seconds ---" % (time.time() - start_time))
