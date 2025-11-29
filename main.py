import streamlit as st
from utils.state_manager import StateManager
from utils.load_data import load_data
from utils.geo_utils import get_geo_data
from tabs.visual_plots import render_visualizations  # or show(), depending on implementation

st.set_page_config(
    page_title="OPTIMET-BCN",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------
# 1) Global state initialization
# ----------------------------
global_state = StateManager("global")

global_state.init({
    "df_main": None,   # Main mobility dataset
    "df_geo": None,    # Municipios coordinates
})

if global_state.get("df_main") is None:
    df_main = load_data("processed/final_combined_with_events_2024.csv")
    global_state.set("df_main", df_main)

if global_state.get("df_geo") is None:
    df_geo = get_geo_data()
    global_state.set("df_geo", df_geo)

# (Optional) If you want to restore previous saved state at startup:
# StateManager.load_all()

# ----------------------------
# 2) App header and tabs
# ----------------------------
st.title("🌐 OPTIMET-BCN")
st.markdown("### Digital Twin of Barcelona Metropolitan Mobility")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Exploración de Datos",
    "📈 Visualizaciones",
    "🌍 Heatmap",
    "🌦️ Clima y Eventos",
    "🔮 Predicción",
    "⚙️ Optimización",
])

with tab1:
    st.header("Exploración de Datos")
    st.warning("⚠️ Módulo en desarrollo.")

with tab2:
    # Tab 2 calls the visualization function,
    # which will read df_main and df_geo from global state
    render_visualizations()

with tab3:
    st.header("Heatmap")
    st.warning("⚠️ Módulo en desarrollo.")

with tab4:
    st.header("Clima y Eventos")
    st.warning("⚠️ Módulo en desarrollo.")

with tab5:
    st.header("Predicción")
    st.warning("⚠️ Módulo en desarrollo.")

with tab6:
    st.header("Optimización")
    st.warning("⚠️ Módulo en desarrollo.")

st.markdown("---")
st.caption("© 2025 OPTIMET-BCN | Telefónica Tech")
