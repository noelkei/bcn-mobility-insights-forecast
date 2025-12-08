import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk

from utils.state_manager import StateManager


def render_visualizations():
    st.header("Visualizaciones Generales")

    # ==========================================================
    # 1. → Cargamos el DATASET Y GEODATA desde el STATE GLOBAL
    # ==========================================================
    global_state = StateManager("global")

    df = global_state.get("df_main")
    geo = global_state.get("df_geo")

    if df is None or geo is None:
        st.error("❌ Error: Los datos globales no están cargados en el StateManager.")
        st.info("Asegúrate de que old_main.py ejecuta StateManager y carga los datasets.")
        return

    # ==========================================================
    # 2. Asegurar tipos básicos (solo se hace una vez)
    # ==========================================================
    if not pd.api.types.is_datetime64_any_dtype(df["day"]):
        df["day"] = pd.to_datetime(df["day"], errors="coerce")

    df["viajes"] = pd.to_numeric(df["viajes"], errors="coerce").fillna(0)

    # ==========================================================
    # 3. Controles (selectores)
    # ==========================================================
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

    # ==========================================================
    # 4. MINIMAPA Pydeck
    # ==========================================================
    st.subheader("🗺️ Minimapa de destinos desde el municipio origen")

    # Agregamos viajes por municipio destino
    df_map = (
        df[df["municipio_origen_name"] == municipio_sel]
        .groupby("municipio_destino_name", as_index=False)["viajes"]
        .sum()
        .rename(columns={"municipio_destino_name": "municipio"})
    )

    # Join con coordenadas
    df_map = df_map.merge(geo, on="municipio", how="left")
    df_map = df_map.dropna(subset=["lat", "lon"])

    if df_map.empty:
        st.info("No hay datos geográficos para los destinos de este municipio.")
    else:
        vmax = df_map["viajes"].max()
        MIN_RADIUS = 120
        MAX_RADIUS = 1500

        if vmax == 0:
            df_map["radius"] = MIN_RADIUS
        else:
            df_map["radius"] = MIN_RADIUS + (
                (MAX_RADIUS - MIN_RADIUS)
                * (np.log1p(df_map["viajes"]) / np.log1p(vmax))
            )

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_map,
            get_position="[lon, lat]",
            get_radius="radius",
            pickable=True,
            get_fill_color=[0, 0, 255, 160],
        )

        view_state = pdk.ViewState(
            latitude=41.39,
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

    # ==========================================================
    # 5. Serie temporal viajes diarios
    # ==========================================================
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

    # ==========================================================
    # 6. Promedio semanal
    # ==========================================================
    st.subheader("📅 Promedio por día de la semana")

    if not df_filtrado.empty:
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

    # ==========================================================
    # 7. Comparativa por tipo de origen
    # ==========================================================
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

    # ==========================================================
    # 8. Top N destinos
    # ==========================================================
    st.subheader("🏆 Top municipios destino")

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

    # ==========================================================
    # 9. Proporción intra vs intermunicipal
    # ==========================================================
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

    # ==========================================================
    # 10. Exportar CSV
    # ==========================================================
    st.subheader("📤 Exportar datos filtrados")

    csv = df_filtrado.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar CSV",
        data=csv,
        file_name=f"movilidad_{municipio_sel}.csv",
        mime="text/csv",
    )
