"""
Plot utilities for OPTIMET-BCN visualizations.
Includes functions for creating heatmaps, aggregations, and geographic visualizations.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


# ============================================================
# HEATMAP UTILITIES
# ============================================================

@st.cache_data
def aggregate_heatmap_data(df: pd.DataFrame, date_from: pd.Timestamp, date_to: pd.Timestamp) -> pd.DataFrame:
    """
    Aggregates trip data into an origin-destination (OD) matrix for a date range.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Main dataset with columns: day, municipio_origen_name, municipio_destino_name, viajes
    date_from : pd.Timestamp
        Start date (inclusive)
    date_to : pd.Timestamp
        End date (inclusive)
    
    Returns:
    --------
    pd.DataFrame
        Aggregated OD matrix with columns: origen, destino, total_viajes
    """
    mask = (df["day"] >= date_from) & (df["day"] <= date_to)
    df_filtered = df[mask].copy()
    
    heatmap = (
        df_filtered
        .groupby(["municipio_origen_name", "municipio_destino_name"], as_index=False)
        ["viajes"]
        .sum()
        .rename(columns={
            "municipio_origen_name": "origen",
            "municipio_destino_name": "destino",
            "viajes": "total_viajes"
        })
    )
    
    return heatmap.sort_values("total_viajes", ascending=False)


def create_od_matrix(heatmap_data: pd.DataFrame) -> pd.DataFrame:
    """
    Converts long-format OD data into a pivot table (matrix) format.
    
    Parameters:
    -----------
    heatmap_data : pd.DataFrame
        OD data with columns: origen, destino, total_viajes
    
    Returns:
    --------
    pd.DataFrame
        Pivot table where rows=origen, columns=destino, values=total_viajes
    """
    matrix = heatmap_data.pivot_table(
        index="origen",
        columns="destino",
        values="total_viajes",
        fill_value=0
    )
    
    return matrix


def filter_heatmap_by_municipio(heatmap_data: pd.DataFrame, municipio: str, direction: str = "both") -> pd.DataFrame:
    """
    Filters heatmap data to show flows from/to a specific municipality.
    
    Parameters:
    -----------
    heatmap_data : pd.DataFrame
        OD data with columns: origen, destino, total_viajes
    municipio : str
        Municipality name to filter
    direction : str
        "from" (only outgoing), "to" (only incoming), or "both"
    
    Returns:
    --------
    pd.DataFrame
        Filtered heatmap data
    """
    if direction == "from":
        return heatmap_data[heatmap_data["origen"] == municipio].copy()
    elif direction == "to":
        return heatmap_data[heatmap_data["destino"] == municipio].copy()
    else:  # both
        return heatmap_data[
            (heatmap_data["origen"] == municipio) | 
            (heatmap_data["destino"] == municipio)
        ].copy()


def compute_heatmap_statistics(heatmap_data: pd.DataFrame) -> dict:
    """
    Computes summary statistics for OD flows.
    
    Parameters:
    -----------
    heatmap_data : pd.DataFrame
        OD data with columns: origen, destino, total_viajes
    
    Returns:
    --------
    dict
        Dictionary with keys: total_trips, avg_trips_per_link, max_trips_link, min_trips_link, 
        num_links, num_origins, num_destinations
    """
    if heatmap_data.empty:
        return {
            "total_trips": 0,
            "avg_trips_per_link": 0,
            "max_trips_link": 0,
            "min_trips_link": 0,
            "num_links": 0,
            "num_origins": 0,
            "num_destinations": 0,
        }
    
    return {
        "total_trips": int(heatmap_data["total_viajes"].sum()),
        "avg_trips_per_link": float(heatmap_data["total_viajes"].mean()),
        "max_trips_link": int(heatmap_data["total_viajes"].max()),
        "min_trips_link": int(heatmap_data["total_viajes"].min()),
        "num_links": len(heatmap_data),
        "num_origins": heatmap_data["origen"].nunique(),
        "num_destinations": heatmap_data["destino"].nunique(),
    }


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_heatmap_matrix(matrix: pd.DataFrame, title: str = "OD Heatmap Matrix") -> go.Figure:
    """
    Creates an interactive heatmap visualization of OD flows.
    
    Parameters:
    -----------
    matrix : pd.DataFrame
        OD matrix with origins as rows and destinations as columns
    title : str
        Title for the heatmap
    
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            colorscale="YlOrRd",
            colorbar=dict(title="Viajes")
        )
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Destino",
        yaxis_title="Origen",
        height=600,
        width=1000,
        hovermode="closest"
    )
    
    return fig


def plot_top_od_flows(heatmap_data: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Creates a bar chart of the top OD flows.
    
    Parameters:
    -----------
    heatmap_data : pd.DataFrame
        OD data with columns: origen, destino, total_viajes
    top_n : int
        Number of top flows to display
    
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    top_flows = heatmap_data.nlargest(top_n, "total_viajes").copy()
    top_flows["od_pair"] = top_flows["origen"] + " → " + top_flows["destino"]
    
    fig = px.bar(
        top_flows,
        y="od_pair",
        x="total_viajes",
        orientation="h",
        title=f"Top {top_n} mayores flujos OD",
        labels={"od_pair": "Enlace OD", "total_viajes": "Número de viajes"},
        color="total_viajes",
        color_continuous_scale="Blues"
    )
    
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=500, showlegend=False)
    
    return fig


def plot_origin_distribution(heatmap_data: pd.DataFrame) -> go.Figure:
    """
    Creates a bar chart showing trip distribution by origin municipality.
    
    Parameters:
    -----------
    heatmap_data : pd.DataFrame
        OD data with columns: origen, destino, total_viajes
    
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    origin_dist = heatmap_data.groupby("origen", as_index=False)["total_viajes"].sum()
    origin_dist = origin_dist.sort_values("total_viajes", ascending=False)
    
    fig = px.bar(
        origin_dist,
        x="origen",
        y="total_viajes",
        title="Distribución de viajes por municipio origen",
        labels={"origen": "Municipio", "total_viajes": "Total viajes"},
        color="total_viajes",
        color_continuous_scale="Greens"
    )
    
    fig.update_layout(height=400, xaxis_tickangle=-45)
    
    return fig


def plot_destination_distribution(heatmap_data: pd.DataFrame) -> go.Figure:
    """
    Creates a bar chart showing trip distribution by destination municipality.
    
    Parameters:
    -----------
    heatmap_data : pd.DataFrame
        OD data with columns: origen, destino, total_viajes
    
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    dest_dist = heatmap_data.groupby("destino", as_index=False)["total_viajes"].sum()
    dest_dist = dest_dist.sort_values("total_viajes", ascending=False)
    
    fig = px.bar(
        dest_dist,
        x="destino",
        y="total_viajes",
        title="Distribución de viajes por municipio destino",
        labels={"destino": "Municipio", "total_viajes": "Total viajes"},
        color="total_viajes",
        color_continuous_scale="Oranges"
    )
    
    fig.update_layout(height=400, xaxis_tickangle=-45)
    
    return fig


def plot_flow_by_municipio(heatmap_data: pd.DataFrame, municipio: str) -> go.Figure:
    """
    Creates a visualization showing flows connected to a specific municipality.
    
    Parameters:
    -----------
    heatmap_data : pd.DataFrame
        OD data with columns: origen, destino, total_viajes
    municipio : str
        Municipality to focus on
    
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    # Incoming flows
    incoming = heatmap_data[heatmap_data["destino"] == municipio].copy()
    incoming["tipo"] = "Entrantes"
    incoming["flujo"] = incoming["origen"] + " → " + municipio
    
    # Outgoing flows
    outgoing = heatmap_data[heatmap_data["origen"] == municipio].copy()
    outgoing["tipo"] = "Salientes"
    outgoing["flujo"] = municipio + " → " + outgoing["destino"]
    
    combined = pd.concat([incoming, outgoing], ignore_index=True)
    combined = combined.nlargest(15, "total_viajes")
    
    fig = px.bar(
        combined,
        y="flujo",
        x="total_viajes",
        color="tipo",
        orientation="h",
        title=f"Principales flujos hacia y desde {municipio}",
        labels={"flujo": "Enlace", "total_viajes": "Viajes"},
        color_discrete_map={"Entrantes": "#1f77b4", "Salientes": "#ff7f0e"}
    )
    
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=500)
    
    return fig


def plot_cumulative_distribution(heatmap_data: pd.DataFrame) -> go.Figure:
    """
    Creates a cumulative distribution plot showing concentration of OD flows.
    
    Parameters:
    -----------
    heatmap_data : pd.DataFrame
        OD data with columns: origen, destino, total_viajes
    
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    sorted_data = heatmap_data.sort_values("total_viajes", ascending=False).reset_index(drop=True)
    sorted_data["cumulative_pct"] = (
        sorted_data["total_viajes"].cumsum() / sorted_data["total_viajes"].sum() * 100
    )
    sorted_data["link_rank"] = range(1, len(sorted_data) + 1)
    
    fig = px.line(
        sorted_data,
        x="link_rank",
        y="cumulative_pct",
        title="Curva de concentración de flujos OD (Pareto)",
        labels={"link_rank": "Número de enlaces (ordenados por volumen)", "cumulative_pct": "% acumulado de viajes"},
        markers=True
    )
    
    # Add 80/20 reference line
    fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="80%")
    
    fig.update_layout(height=400, hovermode="x unified")
    
    return fig
