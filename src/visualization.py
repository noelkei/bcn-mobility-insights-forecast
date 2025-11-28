
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
from utils.load_data import load_data
from utils.geo_utils import get_geo_data


def render_visualizations():
    st.header("Visualizaciones Generales")

    # ============================
    # 1. Carga de datos
    # ============================
    df = load_data(r"C:\Users\berna\OneDrive\windowsAntiguo\Document\data\final_combined_with_events_2024.csv")

     # Aseguramos tipos básicos
    if not pd.api.types.is_datetime64_any_dtype(df["day"]):
        df["day"] = pd.to_datetime(df["day"], errors="coerce")

    df["viajes"] = pd.to_numeric(df["viajes"], errors="coerce").fillna(0)

    # ============================
    # 2. Controles (selectores)
    # ============================
    col_left, col_right = st.columns([2, 1])

    with col_left:
        municipios_origen = sorted(df["municipio_origen_name"].dropna().unique())
        municipio_sel = st.selectbox(
            "Selecciona un municipio origen",
            municipios_origen,
            index=municipios_origen.index("Barcelona") if "Barcelona" in municipios_origen else 0,
        )

    with col_right:
        tipos_origen = sorted(df["origen"].dropna().unique())
        tipos_sel = st.multiselect(
            "Tipo de origen",
            tipos_origen,
            default=tipos_origen,
        )

    # Filtrado principal
    df_filtrado = df[
        (df["municipio_origen_name"] == municipio_sel)
        & (df["origen"].isin(tipos_sel))
    ].copy()

    st.markdown(f"**Filas filtradas:** {len(df_filtrado):,}")

    # ============================
    # 3. MINIMAPA con Pydeck
    # ============================
    st.subheader("🗺️ Minimapa de destinos desde el municipio origen")

    # Cargamos coordenadas de municipios desde tu CSV de geodata
    geo = get_geo_data()  # debe devolver columnas: municipio, lat, lon

    # Agregamos viajes totales desde el municipio origen hacia cada destino
    df_map = (
        df[df["municipio_origen_name"] == municipio_sel]
        .groupby("municipio_destino_name", as_index=False)["viajes"]
        .sum()
        .rename(columns={"municipio_destino_name": "municipio"})
    )

    # Join con coordenadas
    df_map = df_map.merge(geo, on="municipio", how="left")

    # Eliminamos destinos sin coordenadas
    df_map = df_map.dropna(subset=["lat", "lon"])

    if df_map.empty:
        st.info("No hay datos geográficos para los destinos de este municipio.")
    else:
        # -------------------------------
        # RADIUS SCALING (log + min/max)
        # -------------------------------
        vmax = df_map["viajes"].max()

        # Radios mínimo y máximo (ajusta si quieres)
        MIN_RADIUS = 120     # radio para los más pequeños
        MAX_RADIUS = 1500    # radio para los más grandes

        if vmax == 0:
            df_map["radius"] = MIN_RADIUS
        else:
            # Escalado logarítmico: diferencias visibles pero controladas
            df_map["radius"] = MIN_RADIUS + (
                (MAX_RADIUS - MIN_RADIUS)
                * (np.log1p(df_map["viajes"]) / np.log1p(vmax))
            )

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_map,
            get_position="[lon, lat]",
            get_radius="radius",   # usamos la columna ya escalada
            radius_scale=1,
            pickable=True,
            get_fill_color=[0, 0, 255, 160],
        )

        view_state = pdk.ViewState(
            latitude=41.39,   # centro aprox. Barcelona
            longitude=2.17,
            zoom=9,
            pitch=0,
        )

        tooltip = {
            "html": "<b>{municipio}</b><br/>Viajes: {viajes}",
            "style": {"color": "white"},
        }

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip,
            )
        )

    st.markdown("---")

    # ============================
    # 4. Serie temporal de viajes diarios
    # ============================
    st.subheader("📈 Serie temporal de viajes diarios")

    if df_filtrado.empty:
        st.info("No hay datos para la selección actual.")
    else:
        serie_diaria = (
            df_filtrado.groupby("day", as_index=False)["viajes"]
            .sum()
            .sort_values("day")
        )
        fig_ts = px.line(
            serie_diaria,
            x="day",
            y="viajes",
            title=f"Viajes diarios desde {municipio_sel}",
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    # ============================
    # 5. Promedio semanal (día de la semana)
    # ============================
    st.subheader("📅 Promedio por día de la semana")

    if not df_filtrado.empty:
        # Ya tienes 'day_of_week' en el CSV (Domingo, Lunes, etc.)
        df_week = (
            df_filtrado.groupby("day_of_week", as_index=False)["viajes"]
            .mean()
            .sort_values("viajes", ascending=False)
        )
        fig_week = px.bar(
            df_week,
            x="day_of_week",
            y="viajes",
            title="Promedio de viajes por día de la semana",
        )
        st.plotly_chart(fig_week, use_container_width=True)
    else:
        st.info("Sin datos suficientes para calcular el promedio semanal.")

    # ============================
    # 6. Comparativa por tipo de origen
    # ============================
    st.subheader("🏷️ Comparativa por tipo de origen")

    df_tipo = (
        df[df["municipio_origen_name"] == municipio_sel]
        .groupby(["day", "origen"], as_index=False)["viajes"]
        .sum()
    )

    fig_tipo = px.area(
        df_tipo,
        x="day",
        y="viajes",
        color="origen",
        title="Evolución de viajes por tipo de origen",
    )
    st.plotly_chart(fig_tipo, use_container_width=True)

    # ============================
    # 7. Top N municipios destino
    # ============================
    st.subheader("🏆 Top municipios destino desde el origen seleccionado")

    topN = (
        df[df["municipio_origen_name"] == municipio_sel]
        .groupby("municipio_destino_name", as_index=False)["viajes"]
        .sum()
        .sort_values("viajes", ascending=False)
        .head(10)
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Tabla Top 10 destinos")
        st.dataframe(topN, use_container_width=True)

    with col2:
        fig_top = px.bar(
            topN,
            x="municipio_destino_name",
            y="viajes",
            title="Top 10 destinos",
        )
        st.plotly_chart(fig_top, use_container_width=True)

    # ============================
    # 8. Proporción intra vs intermunicipal
    # ============================
    st.subheader("🔄 Proporción de movilidad intra vs intermunicipal")

    df_mov = df[df["municipio_origen_name"] == municipio_sel].copy()
    df_mov["tipo_mov"] = df_mov.apply(
        lambda row: "Intra-municipal"
        if row["municipio_origen_name"] == row["municipio_destino_name"]
        else "Inter-municipal",
        axis=1,
    )

    prop = (
        df_mov.groupby("tipo_mov", as_index=False)["viajes"]
        .sum()
    )
    total_viajes = prop["viajes"].sum()
    if total_viajes > 0:
        prop["porcentaje"] = (prop["viajes"] / total_viajes * 100).round(2)

    fig_pie = px.pie(
        prop,
        names="tipo_mov",
        values="viajes",
        title="Distribución de viajes intra vs intermunicipales",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # ============================
    # 9. Exportar datos filtrados
    # ============================
    st.subheader("📤 Exportar datos filtrados")

    csv = df_filtrado.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar CSV",
        data=csv,
        file_name=f"movilidad_{municipio_sel}.csv",
        mime="text/csv",
    )
