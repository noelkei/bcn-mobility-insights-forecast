import streamlit as st
import pandas as pd
import altair as alt

from src.prediccion_od import (
    cargar_o_entrenar,
    predecir_od
)


@st.cache_resource(show_spinner="Cargando modelo OD...")
def cargar_modelo_y_df():
    df, model, le_origen, le_destino, min_date = cargar_o_entrenar()
    return df, model, le_origen, le_destino, min_date



def main():
    st.title("🔮 Predicción Origen → Destino (Random Forest)")

    df, model, le_origen, le_destino, min_date = cargar_modelo_y_df()

    municipios = sorted(df["municipio_origen_name"].unique())

    origen = st.selectbox("Municipio origen", municipios)
    destino = st.selectbox("Municipio destino", municipios)

    fecha = st.date_input("Selecciona fecha futura")

    # -----------------------------
    # HISTÓRICO DEL OD SELECCIONADO
    # -----------------------------
    df_hist = df[
        (df["municipio_origen_name"] == origen) &
        (df["municipio_destino_name"] == destino)
    ].sort_values("day")

    if len(df_hist) == 0:
        st.warning("❗ Selecciona un par Origen → Destino.")
        return

    st.markdown("### 📉 Histórico de viajes para este OD")

    chart = alt.Chart(df_hist).mark_line(
        color="#4DA6FF"
    ).encode(
        x="day:T",
        y="viajes:Q",
        tooltip=["day", "viajes"]
    ).properties(height=250)

    st.altair_chart(chart, use_container_width=True)

    # Resumen estadístico
    st.markdown("#### 📊 Resumen histórico")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mínimo", int(df_hist["viajes"].min()))
    col2.metric("Media", int(df_hist["viajes"].mean()))
    col3.metric("Máximo", int(df_hist["viajes"].max()))

    # -----------------------------
    # PREDICCIÓN
    # -----------------------------
    if st.button("Predecir viajes"):
        pred = predecir_od(model, le_origen, le_destino, origen, destino, fecha, min_date)

        st.success(f"Predicción estimada: **{int(pred):,} viajes**")

        # Frase automática según el histórico
        media = df_hist["viajes"].mean()

        if pred > media * 1.3:
            st.info("📈 La predicción está **por encima de la media histórica**.")
        elif pred < media * 0.7:
            st.info("📉 La predicción está **por debajo de la media histórica**.")
        else:
            st.info("🔎 La predicción está **dentro de un rango normal histórico**.")
