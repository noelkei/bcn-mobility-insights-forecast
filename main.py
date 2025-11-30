import streamlit as st
import pandas as pd
import numpy as np
from tabs.prediccion_od import main as prediccion_od_main

from utils.state_manager import StateManager
from utils.load_data import load_data
from utils.geo_utils import get_geo_data
from utils.optimizer_utils import (
    compute_optimizer_basics,
    optimize_resources_for_scenario,
    summarize_optimization,
    build_links_table_for_date,
)

from tabs.visual_plots import render_visualizations
from tabs.heatmap_mobility import render_heatmap_mobility
# from tabs.data_explorer import show as show_data_explorer
# ...


# -------------------------------------------------------------
# Page setup
# -------------------------------------------------------------
st.set_page_config(
    page_title="OPTIMET-BCN",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------------------------
# Global state & data loading (solo una vez)
# -------------------------------------------------------------
global_state = StateManager("global")
global_state.init({
    "df_main": None,
    "df_geo": None,
    "optimizer_R_max": None,
    "optimizer_df_share": None,
    "optimizer_df_hist_od": None,
    "optimizer_daily_totals": None,
})

# --- Load main dataset ---
df_main = global_state.get("df_main")
if df_main is None:
    df_main = load_data("processed/final_combined_2023_2024.csv")
    global_state.set("df_main", df_main)

if df_main is not None and not df_main.empty:
    if not pd.api.types.is_datetime64_any_dtype(df_main["day"]):
        df_main = df_main.copy()
        df_main["day"] = pd.to_datetime(df_main["day"], errors="coerce")
        global_state.set("df_main", df_main)

# --- Load geo data ---
df_geo = global_state.get("df_geo")
if df_geo is None:
    df_geo = get_geo_data()
    global_state.set("df_geo", df_geo)

# --- Precompute optimizer basics once (SOLO enlaces con Barcelona) ---
if df_main is not None and not df_main.empty:
    if global_state.get("optimizer_R_max") is None:
        with st.spinner("Inicializando modelo de optimización (foco: Barcelona)..."):
            R_max, df_share, df_hist_od, df_daily_totals = compute_optimizer_basics(
                df_main,
                focus_city="Barcelona",
            )
        global_state.set("optimizer_R_max", R_max)
        global_state.set("optimizer_df_share", df_share)
        global_state.set("optimizer_df_hist_od", df_hist_od)
        global_state.set("optimizer_daily_totals", df_daily_totals)
    else:
        R_max = global_state.get("optimizer_R_max")
        df_share = global_state.get("optimizer_df_share")
        df_hist_od = global_state.get("optimizer_df_hist_od")
else:
    R_max, df_share, df_hist_od = None, None, None

# --- Date bounds for convenience ---
if df_main is not None and not df_main.empty:
    min_day = df_main["day"].min().date()
    max_day = df_main["day"].max().date()
else:
    min_day = max_day = None

# -------------------------------------------------------------
# Header
# -------------------------------------------------------------
st.title("🌐 OPTIMET-BCN")
st.markdown("### Digital Twin of Barcelona Metropolitan Mobility")

# -------------------------------------------------------------
# Tabs
# -------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Exploración de Datos",
    "📈 Visualizaciones",
    "🌍 Heatmap",
    "🌦️ Clima y Eventos",
    "🔮 Predicción",
    "⚙️ Optimización",
])

# -------------------------------------------------------------
# Tab 1
# -------------------------------------------------------------
with tab1:
    st.header("Exploración de Datos")
    st.warning("⚠️ Módulo en desarrollo.")


# -------------------------------------------------------------
# Tab 2
# -------------------------------------------------------------
with tab2:
    render_visualizations()


# -------------------------------------------------------------
# Tab 3
# -------------------------------------------------------------
with tab3:
    render_heatmap_mobility()


# -------------------------------------------------------------
# Tab 4
# -------------------------------------------------------------
with tab4:
    st.header("Clima y Eventos")
    st.warning("⚠️ Módulo en desarrollo.")


# -------------------------------------------------------------
# Tab 5
# -------------------------------------------------------------
with tab5:
    prediccion_od_main()


# -------------------------------------------------------------
# Tab 6 – Optimización (solo enlaces con Barcelona)
# -------------------------------------------------------------
with tab6:
    st.header("⚙️ Simulación y Optimización de Movilidad (OD) — Foco: Barcelona")

    st.markdown(
        """
        En este módulo:
        - Cada fila representa un enlace **origen–destino** donde interviene *Barcelona*.
        - **Demanda** = nº de viajes para el día seleccionado (o estimado si no hay histórico).
        - **Recursos** ≈ capacidad asignada (vehículos, servicio, etc.).
        - **Temperatura** = `demanda / recursos`: ~1 → equilibrio, >1 → enlace caliente, <1 → enlace frío.

        El optimizador intenta **bajar la temperatura de los enlaces calientes**
        usando primero recursos ociosos (slack) y, opcionalmente, reasignando algo de capacidad
        desde enlaces fríos.
        """
    )

    # Recuperamos objetos globales
    df_main = global_state.get("df_main")
    R_max = global_state.get("optimizer_R_max")
    df_share = global_state.get("optimizer_df_share")
    df_hist_od = global_state.get("optimizer_df_hist_od")

    if (
        df_main is None or df_main.empty or
        R_max is None or df_share is None or df_hist_od is None
    ):
        st.error("❌ Faltan datos globales para la optimización.")
        st.info(
            "Asegúrate de que en main.py se llama a compute_optimizer_basics(df_main, focus_city='Barcelona') "
            "y se guarda en el StateManager."
        )
    else:
        # --- State específico de la pestaña ---
        tab_state = StateManager("simulation_optimizer")
        tab_state.init({
            "scenario_date": df_main["day"].max().date(),
            "scenario_date_for_df": None,
            "scenario_is_historical": None,
            "scenario_df_records": None,
            "allow_reallocation": False,
            "last_df_opt": None,
            "last_summary": None,
        })

        st.markdown(
            f"- **Capacidad global diaria considerada (R_max, sólo enlaces con Barcelona)**: "
            f"`{R_max:,.0f}` unidades de recurso"
        )

        # ================================
        # 1) Elegir fecha
        # ================================
        st.subheader("1️⃣ Fecha del escenario")

        scenario_date = st.date_input(
            "Fecha a optimizar (puede ser pasada o futura)",
            value=tab_state.get("scenario_date", max_day),
            min_value=min_day,
            key="opt_any_date",
            help="Si la fecha no existe en el histórico, se usan valores estimados como punto de partida.",
        )
        tab_state.set("scenario_date", scenario_date)
        scenario_ts = pd.to_datetime(scenario_date)

        # ================================
        # 2) Construir tabla df_links (solo cuando cambia la fecha)
        # ================================
        prev_date_for_df = tab_state.get("scenario_date_for_df")
        need_rebuild = (
            prev_date_for_df is None
            or pd.to_datetime(prev_date_for_df).date() != scenario_ts.date()
        )

        if need_rebuild:
            with st.spinner(
                "Construyendo tabla de enlaces (Barcelona como origen o destino) para la fecha seleccionada..."
            ):
                df_links_default, is_historical = build_links_table_for_date(
                    df_hist_od=df_hist_od,
                    df_share=df_share,
                    scenario_ts=scenario_ts,
                    focus_city="Barcelona",
                )

            tab_state.set("scenario_df_records", df_links_default.to_dict("records"))
            tab_state.set("scenario_date_for_df", str(scenario_ts.date()))
            tab_state.set("scenario_is_historical", is_historical)

        # Cargamos siempre la tabla desde el state
        records = tab_state.get("scenario_df_records") or []
        if records:
            scenario_df = pd.DataFrame(records)
        else:
            scenario_df = pd.DataFrame(columns=["origen", "destino", "demanda"])

        is_historical = bool(tab_state.get("scenario_is_historical", False))

        # Info al usuario
        if is_historical:
            st.info(
                "📅 Para esta fecha hay datos históricos en el dataset **(filtrados a enlaces con Barcelona)**.\n\n"
                "- Para cada enlace OD, la demanda es la observada ese día.\n"
                "- Si un enlace no aparece ese día, su demanda es 0.\n"
                "- En este caso, la columna **Demanda** no es editable."
            )
        else:
            st.info(
                "📅 Para esta fecha **no hay datos históricos directos** (para los enlaces con Barcelona).\n\n"
                "- La demanda por enlace se ha estimado con una media ponderada del histórico.\n"
                "- Puedes editar libremente la columna **Demanda** si quieres ajustar el escenario."
            )

        # ================================
        # 3) Editor del df_links
        # ================================
        st.subheader("2️⃣ Demanda por enlace OD (df_links) — Sólo enlaces con Barcelona")

        disabled_cols = ["origen", "destino"]
        if is_historical:
            disabled_cols.append("demanda")  # no editable si hay datos para ese día

        edited_df = st.data_editor(
            scenario_df.sort_values(["origen", "destino"]).reset_index(drop=True),
            num_rows="fixed",
            disabled=disabled_cols,
            key="df_links_editor",
        )

        # Normalizamos tipo numérico de demanda
        edited_df["demanda"] = pd.to_numeric(
            edited_df["demanda"], errors="coerce"
        ).fillna(0.0)

        # Guardamos la versión editada en el state
        tab_state.set("scenario_df_records", edited_df.to_dict("records"))
        scenario_df = edited_df

        total_demand = float(scenario_df["demanda"].sum()) if not scenario_df.empty else 0.0
        st.markdown(
            f"- Enlaces OD en el escenario (con Barcelona): **{len(scenario_df):,}**  \n"
            f"- Demanda total del escenario (∑ viajes OD): **{total_demand:,.0f}**"
        )

        # ================================
        # 4) Parámetros del optimizador
        # ================================
        st.subheader("3️⃣ Parámetros de la optimización")

        allow_reallocation = st.checkbox(
            "Permitir reasignar recursos desde enlaces fríos como **último recurso** "
            "(puede calentar ligeramente enlaces que antes estaban fríos).",
            value=tab_state.get("allow_reallocation", False),
            help="Si no marcas esta opción, solo se usarán recursos claramente ociosos (slack ≥ 0).",
        )
        tab_state.set("allow_reallocation", allow_reallocation)

        # ================================
        # 5) Optimizar
        # ================================
        st.subheader("4️⃣ Ejecutar optimización")

        df_opt = None
        summary = None

        if st.button("🚀 Optimizar recursos para este escenario", key="btn_optimize"):
            if scenario_df.empty:
                st.warning("No hay enlaces en el escenario para optimizar.")
            else:
                with st.spinner("Ejecutando optimización..."):
                    df_links_for_opt = scenario_df[["origen", "destino", "demanda"]].copy()
                    df_opt = optimize_resources_for_scenario(
                        df_links=df_links_for_opt,
                        df_share=df_share,
                        R_max=R_max,
                        allow_deep_reallocation=allow_reallocation,
                    )
                    summary = summarize_optimization(df_opt, R_max=R_max)

                if df_opt is not None and not df_opt.empty:
                    tab_state.set("last_df_opt", df_opt.to_dict("records"))
                    tab_state.set("last_summary", summary)
        else:
            # reutilizar último resultado si existe
            last_df_opt = tab_state.get("last_df_opt")
            last_summary = tab_state.get("last_summary")
            if last_df_opt is not None and last_summary is not None:
                df_opt = pd.DataFrame(last_df_opt)
                summary = last_summary

        # ================================
        # 6) Resultados
        # ================================
        if df_opt is not None and summary is not None and not df_opt.empty:
            st.subheader("📈 Resumen de la optimización")

            colA, colB, colC = st.columns(3)
            with colA:
                st.metric(
                    "Demanda total (viajes)",
                    f"{summary['total_demand']:,.0f}",
                )
                st.metric(
                    "Uso de recursos antes",
                    f"{summary['effective_before_pct']:.1f} % de R_max",
                )
                st.metric(
                    "Uso de recursos después",
                    f"{summary['effective_after_pct']:.1f} % de R_max",
                )
            with colB:
                st.metric(
                    "Temp. media antes",
                    f"{summary['avg_temp_before']:.2f}",
                )
                st.metric(
                    "Temp. media después",
                    f"{summary['avg_temp_after']:.2f}",
                )
                st.metric(
                    "Índice de calor (antes → después)",
                    f"{summary['heat_index_before']:.2f} → {summary['heat_index_after']:.2f}",
                )
            with colC:
                st.metric(
                    "Enlaces calientes antes",
                    str(summary["num_hot_before"]),
                )
                st.metric(
                    "Enlaces calientes después",
                    str(summary["num_hot_after"]),
                )
                st.metric(
                    "Slack restante después",
                    f"{summary['slack_after_pct']:.1f} % de R_max",
                )

            st.markdown(
                f"""
- Slack antes: **{summary['slack_before']:,.0f}** recursos ({summary['slack_before_pct']:.1f}% de R_max)  
- Slack después: **{summary['slack_after']:,.0f}** recursos ({summary['slack_after_pct']:.1f}% de R_max)
"""
            )

            # --- Top enlaces calientes antes de optimizar ---
            st.markdown("### 🔥 Enlaces más calientes **antes** de la optimización")

            top_before = df_opt.sort_values("temp_before", ascending=False).head(10).copy()
            top_before["delta_R"] = top_before["R_opt"] - top_before["R_base"]
            top_before["delta_R_pct"] = np.where(
                top_before["R_base"] > 0,
                100.0 * top_before["delta_R"] / top_before["R_base"],
                np.nan,
            )

            df_before_display = top_before[
                [
                    "origen",
                    "destino",
                    "demanda",
                    "R_base",
                    "R_opt",
                    "temp_before",
                    "temp_after",
                    "delta_R",
                    "delta_R_pct",
                ]
            ].rename(
                columns={
                    "origen": "Origen",
                    "destino": "Destino",
                    "demanda": "Demanda",
                    "R_base": "Recursos base",
                    "R_opt": "Recursos optimizados",
                    "temp_before": "Temp. antes",
                    "temp_after": "Temp. después",
                    "delta_R": "Δ recursos",
                    "delta_R_pct": "Δ recursos (%)",
                }
            )

            st.dataframe(df_before_display, use_container_width=True)

            # --- Top enlaces calientes después de optimizar ---
            st.markdown("### ❄️ Enlaces más calientes **después** de la optimización")

            top_after = df_opt.sort_values("temp_after", ascending=False).head(10).copy()
            df_after_display = top_after[
                [
                    "origen",
                    "destino",
                    "demanda",
                    "R_base",
                    "R_opt",
                    "temp_before",
                    "temp_after",
                ]
            ].rename(
                columns={
                    "origen": "Origen",
                    "destino": "Destino",
                    "demanda": "Demanda",
                    "R_base": "Recursos base",
                    "R_opt": "Recursos optimizados",
                    "temp_before": "Temp. antes",
                    "temp_after": "Temp. después",
                }
            )

            st.dataframe(df_after_display, use_container_width=True)

            st.markdown(
                """
            **Notas rápidas:**

            - La **temperatura** de un enlace es `demanda / recursos`.
            - `Temp ≈ 1` → enlace equilibrado; `Temp > 1` → saturado/caliente; `Temp < 1` → holgado/frío.
            - El **índice de calor** resume cómo de fuertes son las sobrecargas: da más peso a los enlaces muy calientes
              (cuadrado de la sobrecarga relativa). Cuanto más bajo, mejor repartida está la carga.
            - Solo se consideran enlaces donde el origen o el destino es **Barcelona** y que tienen recursos base asignados.
            - La optimización solo mueve recursos (capacidad), nunca cambia la demanda; la idea es **aplanar el calor**:
              menos enlaces muy calientes y menos recursos muertos en enlaces muy fríos.
            """
            )

        else:
            st.info("Cuando ejecutes una optimización, los resultados aparecerán aquí.")

# -------------------------------------------------------------
# Footer
# -------------------------------------------------------------
st.markdown("---")
st.caption("© 2025 OPTIMET-BCN | Telefónica Tech")
