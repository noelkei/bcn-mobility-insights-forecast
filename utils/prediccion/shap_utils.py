import streamlit as st
import shap
import pandas as pd


@st.cache_resource
def get_shap_explainer(_model):
    """
    Devuelve un TreeExplainer cacheado para el modelo dado.
    Nota: el argumento se llama _model para evitar que Streamlit intente hashearlo.
    """
    return shap.TreeExplainer(_model)


def compute_global_shap(explainer, X_sample: pd.DataFrame):
    """
    shap_values para un subconjunto de X (global).
    """
    return explainer.shap_values(X_sample)


def compute_local_shap(explainer, x_row: pd.DataFrame):
    """
    shap_values para una sola fila.
    """
    return explainer.shap_values(x_row)
