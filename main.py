import streamlit as st
import pandas as pd

# Tabs
from tabs.prediccion_viajes import render_prediccion_viajes
from tabs.explicabilidad_global import render_explicabilidad_global
from tabs.explicabilidad_local import render_explicabilidad_local

# Utils
from utils.state_manager import StateManager
from utils.load_data import load_data
from utils.geo_utils import get_geo_data


# -------------------------------------------------------------
# Page setup
# -------------------------------------------------------------
st.set_page_config(
    page_title="BCN Flow Intelligence",
    page_icon="🌊",   # New logo for the project
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------------------------
# Global state & data loading
# -------------------------------------------------------------
global_state = StateManager("global")
global_state.init({
    "df_geo": None,
    "df_model_training": None,
})

# --- Load df_model_training ---
df_model_training = global_state.get("df_model_training")
if df_model_training is None:
    try:
        df_model_training = load_data("processed/df_model_training.csv")
        df_model_training["date"] = pd.to_datetime(df_model_training["date"], errors="coerce")

        # Ensure categorical
        if df_model_training["municipio_origen_name"].dtype != "category":
            df_model_training["municipio_origen_name"] = (
                df_model_training["municipio_origen_name"].astype("category")
            )

        global_state.set("df_model_training", df_model_training)

    except Exception as e:
        st.error(f"Error loading df_model_training.csv: {e}")

# --- Load geo data ---
df_geo = global_state.get("df_geo")
if df_geo is None:
    try:
        df_geo = get_geo_data()
        global_state.set("df_geo", df_geo)
    except Exception as e:
        st.error(f"Error loading municipios_with_lat_alt.csv: {e}")

# -------------------------------------------------------------
# Header
# -------------------------------------------------------------
st.title("🌊 BCN Flow Intelligence")
st.markdown("### Predictive, Explainable & Insight-Driven Inflow Mobility for Barcelona")

# -------------------------------------------------------------
# Tabs
# -------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Data Exploration",
    "📈 Visualizations",
    "🌍 Heatmap",
    "🌦️ Weather & Events",
    "🔮 Prediction",
    "🧠 Global Explainability",
    "🔬 Local Explainability",
])

# -------------------------------------------------------------
# Tab 1 - Data Exploration (placeholder)
# -------------------------------------------------------------
with tab1:
    st.header("📊 Data Exploration")
    st.info("🚧 This module is under construction.")

# -------------------------------------------------------------
# Tab 2 - Visualizations (placeholder)
# -------------------------------------------------------------
with tab2:
    st.header("📈 Visualizations")
    st.info("🚧 This module is under construction.")

# -------------------------------------------------------------
# Tab 3 - Heatmap (placeholder)
# -------------------------------------------------------------
with tab3:
    st.header("🌍 Heatmap")
    st.info("🚧 This module is under construction.")

# -------------------------------------------------------------
# Tab 4 - Weather & Events (placeholder)
# -------------------------------------------------------------
with tab4:
    st.header("🌦️ Weather & Events")
    st.info("🚧 This module is under construction.")

# -------------------------------------------------------------
# Tab 5 — Prediction
# -------------------------------------------------------------
with tab5:
    render_prediccion_viajes()

# -------------------------------------------------------------
# Tab 6 — Global Explainability
# -------------------------------------------------------------
with tab6:
    render_explicabilidad_global()

# -------------------------------------------------------------
# Tab 7 — Local Explainability
# -------------------------------------------------------------
with tab7:
    render_explicabilidad_local()

# -------------------------------------------------------------
# Footer
# -------------------------------------------------------------
st.markdown("---")
st.caption("© 2025 BCN Flow Intelligence")
