import os
import json
import joblib
import streamlit as st


@st.cache_resource
def load_lgb_model(model_dir: str = "models/lgb_model_final"):
    """
    Carga el modelo LightGBM final desde disco.
    """
    model_path = os.path.join(model_dir, "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

    model = joblib.load(model_path)
    return model


@st.cache_resource
def load_feature_cols(model_dir: str = "models/lgb_model_final"):
    """
    Carga la lista de columnas de features desde JSON.
    """
    feature_cols_path = os.path.join(model_dir, "feature_cols.json")
    if not os.path.exists(feature_cols_path):
        raise FileNotFoundError(f"No se encontró feature_cols.json en: {feature_cols_path}")

    with open(feature_cols_path, "r") as f:
        feature_cols = json.load(f)

    if not isinstance(feature_cols, list):
        raise ValueError("feature_cols.json debe contener una lista de nombres de columnas.")

    return feature_cols
