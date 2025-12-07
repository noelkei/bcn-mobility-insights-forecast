import streamlit as st
import pandas as pd
import gdown
import os
import plotly.express as px
from tabs import eda_tab

# ==============================
# CONFIGURACIÓN BÁSICA
# ==============================
st.set_page_config(
    page_title="OPTIMET-BCN",
    page_icon="📊",
    layout="wide"
)

st.title("📊 OPTIMET-BCN – Data Explorer")
st.markdown("### Explorador inicial del dataset combinado (movilidad + clima + eventos)")

# ==============================
# FUNCIÓN PARA CARGAR LOS DATOS
# ==============================
@st.cache_data
def load_data():
    file_id = "14bu2pLuT3oF9E9UG1X2I3lqWu4xMmYbA"  # ENLACE REAL (NO CHAT)
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = "data/processed/movilidad_combinada.csv"

    # Crear carpeta si no existe
    os.makedirs("data", exist_ok=True)

    # Descargar solo si no existe localmente
    if not os.path.exists(output_path):
        st.info("📥 Descargando dataset desde Google Drive (solo la primera vez)...")
        gdown.download(url, output_path, quiet=False)

    # Cargar el dataset
    df = pd.read_csv(output_path)
    return df

# ==============================
# CARGA DE DATOS
# ==============================
try:
    df = load_data()
    st.success("✅ Dataset cargado correctamente desde Google Drive")
except Exception as e:
    st.error(f"❌ Error al cargar el dataset: {e}")
    st.stop()

# ==============================
# NAVEGACIÓN ENTRE PESTAÑAS
# ==============================
tab1, tab2 = st.tabs(["📋 Explorador básico", "📊 Análisis EDA"])

with tab1:
    # ==============================
    # EXPLORACIÓN BÁSICA
    # ==============================
    st.subheader("📋 Vista previa de los datos")
    st.dataframe(df.head(10))

    # Información básica
    st.subheader("📈 Información general")
    col1, col2, col3 = st.columns(3)
    col1.metric("Número de registros", f"{len(df):,}")
    col2.metric("Número de columnas", len(df.columns))
    if "day" in df.columns:
        col3.metric("Rango temporal", f"{df['day'].min()} → {df['day'].max()}")

    # ==============================
    # VISUALIZACIÓN DE EJEMPLO
    # ==============================
    st.subheader("📅 Evolución de los viajes diarios")

    if "day" in df.columns and "viajes" in df.columns:
        df['day'] = pd.to_datetime(df['day'])
        daily = df.groupby("day")["viajes"].sum().reset_index()

        fig = px.line(
            daily,
            x="day",
            y="viajes",
            title="Tendencia diaria de movilidad",
            labels={"day": "Fecha", "viajes": "Número de viajes"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No se encontraron las columnas 'day' o 'viajes' en el dataset.")

with tab2:
    eda_tab.render_eda(df)

# ==============================
# PIE DE PÁGINA
# ==============================
st.markdown("---")
st.caption("OPTIMET-BCN © 2025 – Telefónica Tech | Streamlit + Python")