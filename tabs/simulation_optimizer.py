import streamlit as st
import pandas as pd

from utils.state_manager import StateManager
from utils.optimizer_utils import (
    build_historical_scenario,
    build_future_scenario,
    optimize_resources_for_scenario,
    summarize_optimization,
)


def _show_results(df_opt: pd.DataFrame, summary: dict):
    """
    Muestra métricas y tablas de resultados de la optimización.
    """
    if df_opt.empty:
        st.warning("No hay resultados de optimización para mostrar.")
        return

    st.subheader("📈 Resumen de la optimización")

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric(
            "Demanda total (viajes)",
            f"{summary['total_demand']:,.0f}",
        )
        st.metric(
            "Recursos base (∑ R_base)",
            f"{summary['total_R_base']:,.0f}",
        )
    with colB:
        st.metric(
            "Temperatura media (antes)",
            f"{summary['avg_temp_before']:.2f}",
        )
        st.metric(
            "Temperatura media (después)",
            f"{summary['avg_temp_after']:.2f}",
        )
    with colC:
        st.metric(
            "Enlaces calientes (antes)",
            f"{summary['num_hot_before']}",
        )
        st.metric(
            "Enlaces calientes (después)",
            f"{summary['num_hot_after']}",
        )

    st.markdown("### 🔥 Enlaces más calientes **antes** de la optimización")
    top_before = df_opt.sort_values("temp_before", ascending=False).head(20)
    st.dataframe(
        top_before[["origen", "destino", "demanda", "R_base", "temp_before"]],
        use_container_width=True,
    )

    st.markdown("### ❄️ Enlaces más calientes **después** de la optimización")
    top_after = df_opt.sort_values("temp_after", ascending=False).head(20)
    st.dataframe(
        top_after[["origen", "destino", "demanda", "R_opt", "temp_after"]],
        use_container_width=True,
    )

    st.markdown(
        """
**Notas:**

- La *temperatura* de un enlace es `demanda / recursos`.
- `Temp ≈ 1` → equilibrio; `Temp > 1` → saturación/caliente.
- La optimización primero usa el slack global disponible (recursos ociosos),
  y sólo si se permite, usa una reasignación más agresiva desde enlaces fríos
  hacia enlaces muy calientes.
- Los viajes (demanda) no cambian; sólo cambia la capacidad (recursos) asignada.
"""
    )


def show():
    """
    Pestaña de Streamlit para la simulación y optimización de recursos
    entre enlaces origen-destino.
    """
    st.header("⚙️ Simulación y Optimización de Movilidad (OD)")

    # -------------------------------------------------------------
    # 1. State global y local
    # -------------------------------------------------------------
    global_state = StateManager("global")
    tab_state = StateManager("simulation_optimizer")

    df_main = global_state.get("df_main")
    R_max = global_state.get("optimizer_R_max")
    df_share = global_state.get("optimizer_df_share")
    df_hist_od = global_state.get("optimizer_df_hist_od")

    if df_main is None or R_max is None or df_share is None or df_hist_od is None:
        st.error("❌ Faltan datos globales para la optimización (df_main / R_max / df_share / df_hist_od).")
        st.info("Revisa que main.py inicialice correctamente el StateManager global.")
        return

    if not pd.api.types.is_datetime64_any_dtype(df_main["day"]):
        df_main = df_main.copy()
        df_main["day"] = pd.to_datetime(df_main["day"], errors="coerce")
        global_state.set("df_main", df_main)

    min_day = df_main["day"].min().date()
    max_day = df_main["day"].max().date()

    # Inicializamos valores por defecto de la pestaña
    tab_state.init(
        {
            "mode": "historical",  # "historical" o "future"
            "scenario_date_past": max_day,
            "scenario_date_future": max_day,
            "w_same_day_prev": 0.4,
            "w_same_dow_month": 0.3,
            "w_same_dow": 0.2,
            "w_all": 0.1,
            "allow_reallocation": False,
            "last_df_opt": None,
            "last_summary": None,
        }
    )

    st.markdown(
        f"- **Capacidad global diaria (R_max)**: `{R_max:,.0f}` unidades de recurso (día con máx. viajes)"
    )

    # -------------------------------------------------------------
    # 2. Tipo de escenario: pasado vs futuro
    # -------------------------------------------------------------
    st.subheader("1️⃣ Tipo de escenario")

    current_mode = tab_state.get("mode", "historical")
    mode_labels = ["Escenario histórico", "Escenario futuro"]
    mode_key_map = {"Escenario histórico": "historical", "Escenario futuro": "future"}
    index_default = 0 if current_mode == "historical" else 1

    mode_label = st.radio(
        "¿Qué quieres optimizar?",
        mode_labels,
        index=index_default,
        horizontal=True,
    )
    mode = mode_key_map[mode_label]
    tab_state.set("mode", mode)

    scenario_df = None
    scenario_kind = None
    scenario_ts = None

    # -------------------------------------------------------------
    # 3A. Configuración escenario HISTÓRICO
    # -------------------------------------------------------------
    if mode == "historical":
        st.subheader("2️⃣ Configuración del escenario histórico")

        default_date_past = tab_state.get("scenario_date_past", max_day)
        scenario_date_past = st.date_input(
            "Día histórico a optimizar",
            value=default_date_past,
            min_value=min_day,
            max_value=max_day,
            key="opt_hist_date",
        )
        tab_state.set("scenario_date_past", scenario_date_past)
        scenario_ts = pd.to_datetime(scenario_date_past)

        df_scenario = build_historical_scenario(df_hist_od, scenario_ts)
        scenario_df = df_scenario
        scenario_kind = "Histórico"

        if df_scenario.empty:
            st.warning("No hay datos de movilidad para el día seleccionado.")
        else:
            total_demand = float(df_scenario["demanda"].sum())
            st.markdown(
                f"""
- Tipo de escenario: `Histórico`
- Fecha seleccionada: `{scenario_ts.date()}`
- Demanda total del escenario (∑ viajes OD): **{total_demand:,.0f}**
"""
            )
            st.dataframe(
                df_scenario.sort_values("demanda", ascending=False).head(30),
                use_container_width=True,
            )

    # -------------------------------------------------------------
    # 3B. Configuración escenario FUTURO
    # -------------------------------------------------------------
    else:
        st.subheader("2️⃣ Configuración del escenario futuro")

        default_future_date = tab_state.get("scenario_date_future", max_day)
        scenario_date_future = st.date_input(
            "Día futuro a optimizar",
            value=default_future_date,
            min_value=max_day,
            key="opt_future_date",
            help="Puedes elegir una fecha igual o posterior a la última del dataset.",
        )
        tab_state.set("scenario_date_future", scenario_date_future)
        scenario_ts = pd.to_datetime(scenario_date_future)

        st.markdown("### Pesos para la estimación de demanda por enlace (futuro)")

        colw1, colw2 = st.columns(2)
        with colw1:
            w_same_day_prev = st.number_input(
                "Peso: mismo día & mes en años anteriores",
                min_value=0.0,
                max_value=1.0,
                value=float(tab_state.get("w_same_day_prev", 0.4)),
                step=0.05,
            )
            w_same_dow_month = st.number_input(
                "Peso: mismo día de la semana y mes",
                min_value=0.0,
                max_value=1.0,
                value=float(tab_state.get("w_same_dow_month", 0.3)),
                step=0.05,
            )
        with colw2:
            w_same_dow = st.number_input(
                "Peso: mismo día de la semana (cualquier mes)",
                min_value=0.0,
                max_value=1.0,
                value=float(tab_state.get("w_same_dow", 0.2)),
                step=0.05,
            )
            w_all = st.number_input(
                "Peso: todo el histórico del enlace",
                min_value=0.0,
                max_value=1.0,
                value=float(tab_state.get("w_all", 0.1)),
                step=0.05,
            )

        weights = {
            "same_day_prev": w_same_day_prev,
            "same_dow_month": w_same_dow_month,
            "same_dow": w_same_dow,
            "all_history": w_all,
        }

        # Guardamos pesos en el state
        tab_state.set("w_same_day_prev", w_same_day_prev)
        tab_state.set("w_same_dow_month", w_same_dow_month)
        tab_state.set("w_same_dow", w_same_dow)
        tab_state.set("w_all", w_all)
        tab_state.set("weights", weights)

        st.markdown("### Demanda estimada por enlace (editable)")

        df_future_default = build_future_scenario(
            df_hist_od=df_hist_od,
            df_share=df_share,
            scenario_date=scenario_ts,
            weights=weights,
        )

        if df_future_default.empty:
            st.warning("No se ha podido construir un escenario futuro (no hay enlaces).")
        else:
            df_future_default = df_future_default.sort_values(
                "demanda", ascending=False
            ).reset_index(drop=True)

            st.caption(
                "Edita la columna **demanda** si quieres ajustar manualmente algún enlace. "
                "Si no tocas nada, se usarán los valores estimados por defecto."
            )

            edited_df = st.data_editor(
                df_future_default,
                num_rows="fixed",
                key="opt_future_editor",
            )

            edited_df["demanda"] = pd.to_numeric(
                edited_df["demanda"], errors="coerce"
            ).fillna(0.0)

            tab_state.set("future_table", edited_df.to_dict("records"))

            total_demand = float(edited_df["demanda"].sum())
            st.markdown(
                f"""
- Tipo de escenario: `Futuro`
- Fecha seleccionada: `{scenario_ts.date()}`
- Demanda total del escenario (∑ viajes OD): **{total_demand:,.0f}**
"""
            )

            st.dataframe(
                edited_df.head(30),
                use_container_width=True,
            )

            scenario_df = edited_df
            scenario_kind = "Futuro"

    # -------------------------------------------------------------
    # 4. Parámetros de optimización
    # -------------------------------------------------------------
    st.subheader("3️⃣ Parámetros de la optimización")

    allow_reallocation = st.checkbox(
        "Permitir reasignar recursos desde enlaces fríos como **último recurso** "
        "(puede calentar ligeramente enlaces que antes estaban fríos)",
        value=tab_state.get("allow_reallocation", False),
    )
    tab_state.set("allow_reallocation", allow_reallocation)

    # -------------------------------------------------------------
    # 5. Lanzar optimización (solo al pulsar el botón)
    # -------------------------------------------------------------
    st.subheader("4️⃣ Ejecutar optimización")

    df_opt = None
    summary = None

    if scenario_df is None or scenario_ts is None or scenario_kind is None:
        st.info("Configura primero el escenario antes de optimizar.")
        return

    if scenario_df.empty:
        st.warning("El escenario actual no tiene enlaces con demanda.")
        return

    if st.button("🚀 Optimizar recursos para este escenario"):
        with st.spinner("Ejecutando optimización..."):
            df_opt = optimize_resources_for_scenario(
                df_links=scenario_df,
                df_share=df_share,
                R_max=R_max,
                allow_deep_reallocation=allow_reallocation,
            )
            summary = summarize_optimization(df_opt)

        # Guardamos último resultado en el state
        if df_opt is not None and not df_opt.empty:
            tab_state.set("last_df_opt", df_opt.to_dict("records"))
            tab_state.set("last_summary", summary)

    else:
        # Si no se ha pulsado el botón, intentamos mostrar el último resultado (si existe)
        last_df_opt = tab_state.get("last_df_opt")
        last_summary = tab_state.get("last_summary")
        if last_df_opt is not None and last_summary is not None:
            df_opt = pd.DataFrame(last_df_opt)
            summary = last_summary

    # -------------------------------------------------------------
    # 6. Mostrar resultados (si hay)
    # -------------------------------------------------------------
    if df_opt is not None and summary is not None:
        _show_results(df_opt, summary)
    else:
        st.info("Pulsa el botón de optimización para ver resultados.")
