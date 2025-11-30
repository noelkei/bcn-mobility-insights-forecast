import pandas as pd
import streamlit as st
import os
import numpy as np
import pydeck as pdk
from math import radians, cos, sin, asin, sqrt


@st.cache_data
def get_geo_data():
    """
    Loads basic geo coordinates for municipalities.
    Must contain columns: municipio, lat, lon.
    """
    path = os.path.join("data", "processed", "municipios_with_lat_alt.csv")
    
    df = pd.read_csv(path)
    return df[["municipio", "lat", "lon"]]


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance in kilometers between two points on earth.
    
    Parameters:
    -----------
    lat1, lon1 : float
        Latitude and longitude of first point
    lat2, lon2 : float
        Latitude and longitude of second point
    
    Returns:
    --------
    float
        Distance in kilometers
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r


def compute_od_distances(heatmap_data: pd.DataFrame, geo_data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds distance (in km) to OD pairs by looking up coordinates.
    
    Parameters:
    -----------
    heatmap_data : pd.DataFrame
        OD data with columns: origen, destino, total_viajes
    geo_data : pd.DataFrame
        Geographic data with columns: municipio, lat, lon
    
    Returns:
    --------
    pd.DataFrame
        OD data with added column: distancia_km
    """
    result = heatmap_data.copy()
    result["distancia_km"] = np.nan
    
    # Create lookup dictionaries for coordinates
    coord_map = {}
    for _, row in geo_data.iterrows():
        coord_map[row["municipio"]] = (row["lat"], row["lon"])
    
    # Calculate distances
    for idx, row in result.iterrows():
        origen = row["origen"]
        destino = row["destino"]
        
        if origen in coord_map and destino in coord_map:
            lat1, lon1 = coord_map[origen]
            lat2, lon2 = coord_map[destino]
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            result.at[idx, "distancia_km"] = dist
    
    return result


def create_od_flow_layer(
    heatmap_data: pd.DataFrame,
    geo_data: pd.DataFrame,
    min_trips: int = 0
) -> pdk.Layer:
    """
    Creates a PyDeck layer for visualizing OD flows as arrows on a map.
    
    Parameters:
    -----------
    heatmap_data : pd.DataFrame
        OD data with columns: origen, destino, total_viajes
    geo_data : pd.DataFrame
        Geographic data with columns: municipio, lat, lon
    min_trips : int
        Minimum trips to include in visualization
    
    Returns:
    --------
    pdk.Layer
        PyDeck layer object for rendering
    """
    # Filter by minimum trips
    filtered = heatmap_data[heatmap_data["total_viajes"] >= min_trips].copy()
    
    # Create lookup
    coord_map = {}
    for _, row in geo_data.iterrows():
        coord_map[row["municipio"]] = (row["lat"], row["lon"])
    
    # Prepare data for arc layer
    flow_data = []
    for _, row in filtered.iterrows():
        origen = row["origen"]
        destino = row["destino"]
        
        if origen in coord_map and destino in coord_map:
            lat1, lon1 = coord_map[origen]
            lat2, lon2 = coord_map[destino]
            
            # Normalize width based on trips
            width = 1 + (row["total_viajes"] / filtered["total_viajes"].max() * 5)
            
            flow_data.append({
                "source": [lon1, lat1],
                "target": [lon2, lat2],
                "viajes": row["total_viajes"],
                "width": width
            })
    
    if not flow_data:
        return None
    
    flow_df = pd.DataFrame(flow_data)
    
    layer = pdk.Layer(
        "ArcLayer",
        data=flow_df,
        get_source_position="source",
        get_target_position="target",
        get_source_color=[255, 0, 0, 100],
        get_target_color=[0, 0, 255, 100],
        get_width="width",
        pickable=True,
        auto_highlight=True,
    )
    
    return layer


def create_municipio_heatmap_layer(
    heatmap_data: pd.DataFrame,
    geo_data: pd.DataFrame,
    aggregation: str = "destino"
) -> pdk.Layer:
    """
    Creates a PyDeck heatmap layer showing trip concentrations by municipality.
    
    Parameters:
    -----------
    heatmap_data : pd.DataFrame
        OD data with columns: origen, destino, total_viajes
    geo_data : pd.DataFrame
        Geographic data with columns: municipio, lat, lon
    aggregation : str
        "origen" or "destino" - which to aggregate by
    
    Returns:
    --------
    pdk.Layer
        PyDeck heatmap layer
    """
    # Aggregate by municipio
    if aggregation == "destino":
        agg_data = heatmap_data.groupby("destino", as_index=False)["total_viajes"].sum()
        agg_data = agg_data.rename(columns={"destino": "municipio"})
    else:
        agg_data = heatmap_data.groupby("origen", as_index=False)["total_viajes"].sum()
        agg_data = agg_data.rename(columns={"origen": "municipio"})
    
    # Join with geo data
    agg_data = agg_data.merge(geo_data, on="municipio", how="left")
    agg_data = agg_data.dropna(subset=["lat", "lon"])
    
    layer = pdk.Layer(
        "HeatmapLayer",
        data=agg_data,
        get_position=["lon", "lat"],
        get_weight="total_viajes",
        pickable=True,
        auto_highlight=True,
    )
    
    return layer


def aggregate_by_distance_bins(
    heatmap_data: pd.DataFrame,
    geo_data: pd.DataFrame,
    bins: list = None
) -> pd.DataFrame:
    """
    Aggregates OD flows by distance bins (short, medium, long trips).
    
    Parameters:
    -----------
    heatmap_data : pd.DataFrame
        OD data with columns: origen, destino, total_viajes
    geo_data : pd.DataFrame
        Geographic data with columns: municipio, lat, lon
    bins : list
        Distance thresholds in km. Default: [0, 10, 25, 50, 100, 1000]
    
    Returns:
    --------
    pd.DataFrame
        Aggregated data by distance bin with columns: distancia_bin, total_viajes, num_enlaces
    """
    if bins is None:
        bins = [0, 10, 25, 50, 100, 1000]
    
    # Add distances
    data_with_dist = compute_od_distances(heatmap_data, geo_data)
    data_with_dist = data_with_dist.dropna(subset=["distancia_km"])
    
    # Create bins
    data_with_dist["distancia_bin"] = pd.cut(
        data_with_dist["distancia_km"],
        bins=bins,
        labels=[f"{bins[i]}-{bins[i+1]}km" for i in range(len(bins)-1)]
    )
    
    # Aggregate
    result = (
        data_with_dist
        .groupby("distancia_bin", as_index=False)
        .agg({
            "total_viajes": "sum",
            "origen": "count"
        })
        .rename(columns={"origen": "num_enlaces"})
    )
    
    return result


def get_municipio_centroids(geo_data: pd.DataFrame) -> pd.DataFrame:
    """
    Returns centroids (average coordinates) of municipalities.
    Useful as starting point for OD visualizations.
    
    Parameters:
    -----------
    geo_data : pd.DataFrame
        Geographic data with columns: municipio, lat, lon
    
    Returns:
    --------
    pd.DataFrame
        Centroid data with columns: municipio, lat, lon
    """
    return geo_data[["municipio", "lat", "lon"]].copy()


def filter_geo_bounds(
    geo_data: pd.DataFrame,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float
) -> pd.DataFrame:
    """
    Filters geographic data to a bounding box.
    
    Parameters:
    -----------
    geo_data : pd.DataFrame
        Geographic data with columns: municipio, lat, lon
    lat_min, lat_max : float
        Latitude bounds
    lon_min, lon_max : float
        Longitude bounds
    
    Returns:
    --------
    pd.DataFrame
        Filtered geographic data
    """
    return geo_data[
        (geo_data["lat"] >= lat_min) &
        (geo_data["lat"] <= lat_max) &
        (geo_data["lon"] >= lon_min) &
        (geo_data["lon"] <= lon_max)
    ].copy()
