import pandas as pd
import streamlit as st
import os

@st.cache_data
def get_geo_data():
    """
    Loads basic geo coordinates for municipalities.
    Must contain columns: municipio, lat, lon.
    """
    path = os.path.join("data", "processed", "municipios_with_lat_alt.csv")
    
    df = pd.read_csv(path)
    return df[["municipio", "lat", "lon"]]
    
    #path = os.path.join("data", "path")
    #return pd.read_csv(path)
