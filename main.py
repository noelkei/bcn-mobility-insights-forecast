import streamlit as st
from src.visualization import render_visualizations
# --- Page setup ---
st.set_page_config(
    page_title="OPTIMET-BCN",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",  # hides the sidebar
)

# --- Header ---
st.title("🌐 OPTIMET-BCN")
st.markdown("### Digital Twin of Barcelona Metropolitan Mobility")

# --- Tabs (pestañas) ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Exploración de Datos",
    "📈 Visualizaciones",
    "🌍 Heatmap",
    "🌦️ Clima y Eventos",
    "🔮 Predicción",
    "⚙️ Optimización",
])

# --- Tab 1: Exploración de Datos ---
with tab1:
    st.header("Exploración de Datos")
    st.warning("⚠️ Módulo en desarrollo.")
    st.write("""
    En esta sección se mostrará un resumen inicial de los datasets de movilidad, incluyendo:
    - Número de registros
    - Rango temporal de observaciones
    - Municipios con mayor volumen de viajes
    - Indicadores básicos de calidad de datos

    El objetivo de este módulo es proporcionar una vista general y limpia de los datos disponibles 
    antes de realizar visualizaciones o predicciones.
    """)

# --- Tab 2: Visualizaciones ---
with tab2:
    render_visualizations()

# --- Tab 3: Heatmap ---
with tab3:
    st.header("Heatmap de Movilidad")
    st.warning("⚠️ Módulo en desarrollo.")
    st.write("""
    Esta vista mostrará un mapa dinámico con los flujos de movilidad entre municipios
    y detectará zonas de alta densidad de desplazamientos en diferentes momentos del tiempo.
    """)

# --- Tab 4: Clima y Eventos ---
with tab4:
    st.header("Clima y Eventos")
    st.warning("⚠️ Módulo en desarrollo.")
    st.write("""
    Analizará cómo la meteorología (temperatura, lluvia, viento) y los eventos externos
    (deportivos, culturales, etc.) afectan la movilidad metropolitana.
    """)

# --- Tab 5: Predicción ---
with tab5:
    st.header("Predicción de Movilidad")
    st.warning("⚠️ Módulo en desarrollo.")
    st.write("""
    Utilizará modelos de aprendizaje automático y series temporales (como Prophet)
    para estimar la demanda futura de movilidad en función de los datos históricos.
    """)

# --- Tab 6: Optimización ---
with tab6:
    st.header("Simulación y Optimización")
    st.warning("⚠️ Módulo en desarrollo.")
    st.write("""
    Permitirá simular políticas de mejora y escenarios alternativos para reducir la
    congestión de movilidad mediante ajustes de oferta y demanda.
    """)

# --- Footer ---
st.markdown("---")
st.caption("© 2025 OPTIMET-BCN | Telefónica Tech")
