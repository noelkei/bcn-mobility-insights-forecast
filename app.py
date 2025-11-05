# CODIGO PROVISIONAL PARA HACER PRUEBAS
import streamlit as st
from src import data_preprocessing, visualization, heatmap, clima_eventos, forecasting, simulation

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="OPTIMET-BCN", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("OPTIMET-BCN")
st.sidebar.markdown("### Mobility Data Intelligence Dashboard")

menu = st.sidebar.radio(
    "Selecciona un módulo:",
    [
        "Exploración y calidad de datos",
        "Visualizaciones generales",
        "Heatmap de movilidad",
        "Clima y eventos",
        "Predicción de movilidad",
        "Simulación y optimización"
    ]
)

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    import pandas as pd
    df = pd.read_csv("data/processed/movilidad_clean.csv")
    return df

data = load_data()

# --- CONTENIDO PRINCIPAL ---
if menu == "Exploración y calidad de datos":
    data_preprocessing.show(data)

elif menu == "Visualizaciones generales":
    visualization.show(data)

elif menu == "Heatmap de movilidad":
    heatmap.show(data)

elif menu == "Clima y eventos":
    clima_eventos.show(data)

elif menu == "Predicción de movilidad":
    forecasting.show(data)

elif menu == "Simulación y optimización":
    simulation.show(data)

