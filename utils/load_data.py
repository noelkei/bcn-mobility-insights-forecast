import pandas as pd
import streamlit as st
import os

@st.cache_data
def load_data(filename: str):
    """
    Loads a dataset from the /data folder.
    Automatically detects CSV or Parquet.
    """
    data_path = os.path.join("data", filename)

    if filename.endswith(".parquet"):
        return pd.read_parquet(data_path)
    else:
        return pd.read_csv(data_path)
