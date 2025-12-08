"""
OPTIMET-BCN: Heatmap de Movilidad
Visualización interactiva de flujos origen-destino (OD) en el área metropolitana de Barcelona
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from utils.state_manager import StateManager
from utils.plot_utils import (
    aggregate_heatmap_data,
    create_od_matrix,
    filter_heatmap_by_municipio,
    compute_heatmap_statistics,
    plot_heatmap_matrix,
    plot_top_od_flows,
    plot_origin_distribution,
    plot_destination_distribution,
    plot_flow_by_municipio,
    plot_cumulative_distribution,
)


def render_heatmap_mobility():
    """
    Main rendering function for the Heatmap de Movilidad tab.
    Displays OD flows, aggregations, and interactive visualizations.
    """
    
    st.header("🔥 Heatmap de Movilidad")
    st.markdown(
        """
        **Analiza los flujos de viajes origen-destino (OD)** en el área metropolitana de Barcelona.
        
        - 📊 **Matriz OD**: Visualiza todos los flujos entre municipios
        - 🎯 **Top flujos**: Identifica los enlaces con mayor volumen de viajes
        - 🌍 **Distribuciones**: Análisis por municipio origen y destino
        - 📈 **Concentración**: Curva de Pareto para detectar flujos críticos
        """
    )
    
    # ================================================================
    # 1. CARGAR DATOS GLOBALES
    # ================================================================
    global_state = StateManager("global")
    df_main = global_state.get("df_main")
    
    if df_main is None or df_main.empty:
        st.error("❌ Error: Los datos principales no están cargados.")
        st.info("Asegúrate de que old_main.py carga el dataset en el StateManager.")
        return
    
    # Asegurar tipos de datos
    if not pd.api.types.is_datetime64_any_dtype(df_main["day"]):
        df_main = df_main.copy()
        df_main["day"] = pd.to_datetime(df_main["day"], errors="coerce")
    
    df_main["viajes"] = pd.to_numeric(df_main["viajes"], errors="coerce").fillna(0)
    
    # Rango de fechas disponibles
    min_day = df_main["day"].min().date()
    max_day = df_main["day"].max().date()
    
    # ================================================================
    # 2. STATE ESPECÍFICO DEL TAB HEATMAP
    # ================================================================
    heatmap_state = StateManager("heatmap_mobility")
    heatmap_state.init({
        "date_from": min_day,  # Último mes por defecto
        "date_to": max_day,
        "selected_municipio": "Barcelona",
        "view_mode": "matrix",
        "last_heatmap_data": None,
    })
    
    # ================================================================
    # 3. SELECTORES (CONTROLES)
    # ================================================================
    st.subheader("⚙️ Parámetros de análisis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_from = st.date_input(
            "Fecha inicial",
            value=heatmap_state.get("date_from", max_day - timedelta(days=30)),
            min_value=min_day,
            max_value=max_day,
            key="hm_date_from"
        )
        heatmap_state.set("date_from", date_from)
    
    with col2:
        date_to = st.date_input(
            "Fecha final",
            value=heatmap_state.get("date_to", max_day),
            min_value=min_day,
            max_value=max_day,
            key="hm_date_to"
        )
        heatmap_state.set("date_to", date_to)
    
    with col3:
        municipios = sorted(set(
            list(df_main["municipio_origen_name"].dropna().unique()) +
            list(df_main["municipio_destino_name"].dropna().unique())
        ))
        
        municipio_sel = st.selectbox(
            "Municipio focus (para análisis detallado)",
            municipios,
            index=municipios.index("Barcelona") if "Barcelona" in municipios else 0,
            key="hm_municipio"
        )
        heatmap_state.set("selected_municipio", municipio_sel)
    
    # Validar rango de fechas
    if date_from > date_to:
        st.error("❌ La fecha inicial no puede ser posterior a la fecha final.")
        return
    
    date_from_ts = pd.to_datetime(date_from)
    date_to_ts = pd.to_datetime(date_to)
    
    # ================================================================
    # 4. AGREGAR DATOS HEATMAP
    # ================================================================
    with st.spinner("📊 Agregando datos de flujos OD..."):
        heatmap_data = aggregate_heatmap_data(df_main, date_from_ts, date_to_ts)
    
    if heatmap_data.empty:
        st.warning("⚠️ No hay datos para el rango de fechas seleccionado.")
        return
    
    # Guardar en state para reutilizar
    heatmap_state.set("last_heatmap_data", heatmap_data)
    
    # ================================================================
    # 5. ESTADÍSTICAS GENERALES
    # ================================================================
    st.subheader("📈 Estadísticas del período")
    
    stats = compute_heatmap_statistics(heatmap_data)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total de viajes", f"{stats['total_trips']:,}")
    
    with col2:
        st.metric("Enlaces OD", f"{stats['num_links']:,}")
    
    with col3:
        st.metric("Municipios origen", f"{stats['num_origins']}")
    
    with col4:
        st.metric("Municipios destino", f"{stats['num_destinations']}")
    
    with col5:
        avg_val = stats['avg_trips_per_link']
        st.metric("Viajes/enlace (promedio)", f"{avg_val:,.0f}")
    
    st.markdown(f"**Rango de período**: {date_from} a {date_to} ({(date_to_ts - date_from_ts).days} días)")
    
    # ================================================================
    # 6. SELECTOR DE VISTA
    # ================================================================
    st.subheader("🎯 Modo de visualización")
    
    view_options = {
        "Matriz OD": "matrix",
        "Top flujos": "top_flows",
        "Origen-Destino separados": "od_separate",
        "Focus en municipio": "municipio_focus",
        "Curva de concentración": "pareto",
    }
    
    view_mode = st.radio(
        "Selecciona la visualización:",
        options=list(view_options.keys()),
        horizontal=True,
        key="hm_view_mode"
    )
    heatmap_state.set("view_mode", view_options[view_mode])
    
    # ================================================================
    # 7. VISUALIZACIONES POR MODO
    # ================================================================
    
    if view_options[view_mode] == "matrix":
        st.subheader("📊 Matriz OD (Heatmap)")
        st.markdown(
            "Filas = municipios origen | Columnas = municipios destino | "
            "Color = volumen de viajes"
        )
        
        matrix = create_od_matrix(heatmap_data)
        
        # Limitar tamaño si es muy grande
        if len(matrix) > 30 or len(matrix.columns) > 30:
            st.info(
                f"⚠️ Matriz muy grande ({len(matrix)} orígenes × {len(matrix.columns)} destinos). "
                "Mostrando solo los 30 principales..."
            )
            top_origins = heatmap_data.groupby("origen")["total_viajes"].sum().nlargest(30).index
            top_dests = heatmap_data.groupby("destino")["total_viajes"].sum().nlargest(30).index
            matrix = matrix.loc[top_origins, top_dests]
        
        fig_matrix = plot_heatmap_matrix(matrix, title="Matriz OD de flujos de viajes")
        st.plotly_chart(fig_matrix, use_container_width=True)
        
        # Tabla con datos
        with st.expander("📋 Ver tabla de datos agregados"):
            st.dataframe(
                heatmap_data.sort_values("total_viajes", ascending=False),
                use_container_width=True,
                height=400
            )
    
    elif view_options[view_mode] == "top_flows":
        st.subheader("🏆 Top flujos OD (mayores volúmenes)")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            top_n = st.number_input("Mostrar top N", min_value=5, max_value=50, value=15)
        
        fig_top = plot_top_od_flows(heatmap_data, top_n=top_n)
        st.plotly_chart(fig_top, use_container_width=True)
        
        # Tabla de top flows
        with st.expander("📋 Detalles de top flujos"):
            top_table = heatmap_data.nlargest(top_n, "total_viajes")[
                ["origen", "destino", "total_viajes"]
            ].reset_index(drop=True)
            top_table.index = top_table.index + 1
            st.dataframe(top_table, use_container_width=True)
    
    elif view_options[view_mode] == "od_separate":
        st.subheader("📊 Distribuciones separadas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_origin = plot_origin_distribution(heatmap_data)
            st.plotly_chart(fig_origin, use_container_width=True)
        
        with col2:
            fig_dest = plot_destination_distribution(heatmap_data)
            st.plotly_chart(fig_dest, use_container_width=True)
    
    elif view_options[view_mode] == "municipio_focus":
        st.subheader(f"🎯 Análisis de flujos: {municipio_sel}")
        st.markdown(
            f"Visualización de todos los flujos **hacia y desde** {municipio_sel}"
        )
        
        fig_focus = plot_flow_by_municipio(heatmap_data, municipio_sel)
        st.plotly_chart(fig_focus, use_container_width=True)
        
        # Estadísticas específicas del municipio
        incoming = heatmap_data[heatmap_data["destino"] == municipio_sel].copy()
        outgoing = heatmap_data[heatmap_data["origen"] == municipio_sel].copy()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                f"Viajes hacia {municipio_sel}",
                f"{incoming['total_viajes'].sum():,.0f}"
            )
        
        with col2:
            st.metric(
                f"Viajes desde {municipio_sel}",
                f"{outgoing['total_viajes'].sum():,.0f}"
            )
        
        with col3:
            st.metric(
                "Municipios origen (entrantes)",
                incoming["origen"].nunique()
            )
        
        with col4:
            st.metric(
                "Municipios destino (salientes)",
                outgoing["destino"].nunique()
            )
    
    elif view_options[view_mode] == "pareto":
        st.subheader("📈 Curva de Pareto (Concentración de flujos)")
        st.markdown(
            "Identifica qué porcentaje de enlaces genera el 80% de los viajes. "
            "Útil para priorizar intervenciones en movilidad."
        )
        
        fig_pareto = plot_cumulative_distribution(heatmap_data)
        st.plotly_chart(fig_pareto, use_container_width=True)
        
        # Estadística de concentración
        sorted_data = heatmap_data.sort_values("total_viajes", ascending=False).reset_index(drop=True)
        total_trips = sorted_data["total_viajes"].sum()
        cumsum = sorted_data["total_viajes"].cumsum()
        
        # Encontrar cuántos enlaces generan 80%
        idx_80 = (cumsum <= total_trips * 0.8).sum()
        pct_links_80 = (idx_80 / len(sorted_data) * 100) if len(sorted_data) > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Enlaces para 80% de viajes",
                idx_80
            )
        
        with col2:
            st.metric(
                "% de total de enlaces",
                f"{pct_links_80:.1f}%"
            )
        
        with col3:
            st.metric(
                "Ratio concentración",
                f"{len(sorted_data) / max(idx_80, 1):.1f}x"
            )
    
    # ================================================================
    # 8. EXPORTAR DATOS
    # ================================================================
    st.subheader("💾 Exportar datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = heatmap_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar datos OD (CSV)",
            data=csv_data,
            file_name=f"heatmap_od_{date_from}_{date_to}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Crear matriz para export
        matrix = create_od_matrix(heatmap_data)
        excel_buffer = pd.ExcelWriter("buffer", engine="openpyxl")
        matrix.to_excel(excel_buffer, sheet_name="Matriz OD")
        
        excel_data = pd.DataFrame(heatmap_data).to_excel(
            excel_buffer, sheet_name="Datos detallados"
        )
        
        st.info("💡 Para descargar como Excel, usa Python localmente o herramientas como pandas.")


# ================================================================
# POINT DE ENTRADA (usado en old_main.py)
# ================================================================
def show():
    """Entry point for the heatmap tab in old_main.py"""
    render_heatmap_mobility()


if __name__ == "__main__":
    # Para testing local
    render_heatmap_mobility()
