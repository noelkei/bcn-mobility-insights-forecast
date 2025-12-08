import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils.state_manager import StateManager
from utils.prediccion.model_loader import load_lgb_model, load_feature_cols
from utils.prediccion.feature_engineering import build_feature_row
from utils.prediccion.lag_utils import compute_auto_lags, compute_fallback_means


# ============================================================
# 🔁 Helper to force clean Streamlit refresh
# ============================================================

def _force_refresh():
    st.session_state["_refresh_flag"] = np.random.rand()


# ============================================================
# OPTIONS & LABELS
# ============================================================

def _get_origen_options():
    return ["Internacional", "Nacional", "Regional", "Residente"]


def _get_event_category_options():
    """
    Human-readable label → model feature column.
    """
    return {
        "City festival / holiday": "eventcat_city_festival",
        "Concert": "eventcat_concert",
        "Music festival": "eventcat_festival",
        "Football match": "eventcat_football",
        "Motorsport (F1 / MotoGP)": "eventcat_motorsport",
        "Other sport": "eventcat_other_sport",
        "Trade fair / congress": "eventcat_trade_fair",
    }


# ============================================================
# BEAUTIFUL HISTOGRAM (Plotly Dark)
# ============================================================

def _build_hist_plot(df_model, municipio, origen, real_val, pred_val):
    """
    Histogram (Plotly) of historical trips for this municipality+origin.
    Styled similar to the car-price recommender app.
    """
    mask = (
        (df_model["municipio_origen_name"] == municipio) &
        (df_model["origen"] == origen)
    )
    sub = df_model.loc[mask, "viajes"].dropna()

    if sub.empty:
        st.info("Not enough historical data for this municipality + origin to draw the histogram.")
        return

    values_df = pd.DataFrame({"viajes": sub.values.astype(float)})

    fig = px.histogram(
        values_df,
        x="viajes",
        nbins=80,
        template="plotly_dark",
    )
    fig.update_layout(
        xaxis_title="Daily trips",
        yaxis_title="Count",
        bargap=0.05,
        title=f"Distribution of historical trips — {municipio} ({origen})"
    )

    # Vertical line: REAL
    if real_val is not None:
        fig.add_vline(
            x=real_val,
            line_color="deepskyblue",
            line_dash="dash",
            annotation_text=f"Real ({real_val:.0f})",
            annotation_position="top left",
        )

    # Vertical line: PREDICTED
    if pred_val is not None:
        fig.add_vline(
            x=pred_val,
            line_color="red",
            line_dash="solid",
            annotation_text=f"Predicted ({pred_val:.0f})",
            annotation_position="top right",
        )

    st.plotly_chart(fig, use_container_width=True)

    # Percentile metrics
    def _percentile_rank(arr, v):
        if v is None:
            return None
        return float((arr <= v).mean() * 100)

    p_real = _percentile_rank(values_df["viajes"].values, real_val)
    p_pred = _percentile_rank(values_df["viajes"].values, pred_val)

    cols = st.columns(2)
    with cols[0]:
        st.metric(
            "REAL value percentile",
            f"{p_real:.1f} %" if p_real is not None else "N/A",
        )
    with cols[1]:
        st.metric(
            "PREDICTED value percentile",
            f"{p_pred:.1f} %" if p_pred is not None else "N/A",
        )


# ============================================================
# MAIN TAB
# ============================================================

def render_prediccion_viajes():
    """Prediction tab — LightGBM model."""
    st.header("🔮 Trip Prediction to Barcelona")

    # ----------------------------------------------------------
    # GLOBAL & TAB STATE
    # ----------------------------------------------------------
    global_state = StateManager("global")
    tab_state = StateManager("prediction")

    tab_state.init({
        "municipio_sel": None,
        "origen_sel": "Residente",
        "selected_date": None,
        "latest_prediction": None,

        # Weather state (outside dataset)
        "tavg": 20.0,
        "tmin": 15.0,
        "tmax": 25.0,
        "prcp": 0.0,

        # Event state (outside dataset)
        "event_attendance": 0.0,
        "event_categories_labels": [],
    })

    # ----------------------------------------------------------
    # LOAD DF_MODEL_TRAINING
    # ----------------------------------------------------------
    df_model = global_state.get("df_model_training")
    if df_model is None:
        st.error("❌ df_model_training is missing in StateManager('global').")
        st.info("Ensure main.py loads data/processed/df_model_training.csv into global state.")
        st.stop()

    if not pd.api.types.is_datetime64_any_dtype(df_model["date"]):
        df_model["date"] = pd.to_datetime(df_model["date"], errors="coerce")

    df_model = df_model.sort_values("date")

    min_date = df_model["date"].min().date()
    max_date = df_model["date"].max().date()
    max_future_date = max_date.replace(year=max_date.year + 1)

    # ----------------------------------------------------------
    # LOAD MODEL + FEATURE COLS
    # ----------------------------------------------------------
    try:
        model = load_lgb_model()
        feature_cols = load_feature_cols()
    except Exception as e:
        st.error(f"Error loading model or feature_cols: {e}")
        return

    # ----------------------------------------------------------
    # UI: MUNICIPALITY & ORIGIN
    # ----------------------------------------------------------
    municipios = sorted(df_model["municipio_origen_name"].dropna().unique())
    origen_options = _get_origen_options()

    default_muni = tab_state.get("municipio_sel")
    if default_muni not in municipios:
        default_muni = municipios[0]

    col1, col2 = st.columns(2)
    with col1:
        municipio_sel = st.selectbox(
            "Origin municipality",
            municipios,
            index=municipios.index(default_muni),
            key="municipio_sel_key",
            on_change=_force_refresh,
        )
    with col2:
        origen_sel = st.selectbox(
            "Origin type",
            origen_options,
            index=origen_options.index(tab_state.get("origen_sel", "Residente")),
            key="origen_sel_key",
            on_change=_force_refresh,
        )

    tab_state.set("municipio_sel", municipio_sel)
    tab_state.set("origen_sel", origen_sel)

    # ----------------------------------------------------------
    # UI: DATE
    # ----------------------------------------------------------
    st.subheader("🗓️ Prediction date")

    default_date = tab_state.get("selected_date") or max_date
    target_date = st.date_input(
        "Choose date (from historical range up to +1 year)",
        value=default_date,
        min_value=min_date,
        max_value=max_future_date,
        key="date_sel_key",
        on_change=_force_refresh,
    )
    tab_state.set("selected_date", target_date)
    target_ts = pd.to_datetime(target_date)

    # ----------------------------------------------------------
    # LAGS & FALLBACKS
    # ----------------------------------------------------------
    auto_lags = compute_auto_lags(
        df=df_model,
        municipio=municipio_sel,
        origen=origen_sel,
        target_date=target_ts,
        max_lag=7,
    )
    means = compute_fallback_means(df_model, municipio_sel, origen_sel)

    st.markdown(f"Historical data available: **{min_date} → {max_date}**.")

    if target_date <= max_date:
        st.info(
            "📅 The selected date is **inside the historical dataset**. Lags, weather and events "
            "are retrieved directly if available."
        )
    else:
        st.warning(
            "📅 The selected date is **outside the historical dataset**.\n"
            "- Up to 7 days after the last available date, some lags still exist.\n"
            "- Beyond that, fallback means are used."
        )

    # ----------------------------------------------------------
    # WEATHER (AUTO FROM DATASET IF AVAILABLE)
    # ----------------------------------------------------------
    st.subheader("🌦️ Weather conditions")

    df_day = df_model[df_model["date"] == target_ts]

    if not df_day.empty and all(col in df_day.columns for col in ["tavg", "tmin", "tmax", "prcp"]):
        weather_from_dataset = True
        tavg_val = float(df_day["tavg"].iloc[0])
        tmin_val = float(df_day["tmin"].iloc[0])
        tmax_val = float(df_day["tmax"].iloc[0])
        prcp_val = float(df_day["prcp"].iloc[0])
        weather_help = "Weather automatically loaded from dataset."
    else:
        weather_from_dataset = False
        tavg_val = tab_state.get("tavg", 20.0)
        tmin_val = tab_state.get("tmin", 15.0)
        tmax_val = tab_state.get("tmax", 25.0)
        prcp_val = tab_state.get("prcp", 0.0)
        weather_help = "Enter estimated weather for this date."

    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        tavg = st.number_input("tavg", value=tavg_val, disabled=weather_from_dataset, key="tavg_input", help=weather_help)
    with col_c2:
        tmin = st.number_input("tmin", value=tmin_val, disabled=weather_from_dataset, key="tmin_input")
    with col_c3:
        tmax = st.number_input("tmax", value=tmax_val, disabled=weather_from_dataset, key="tmax_input")
    with col_c4:
        prcp = st.number_input("prcp", value=prcp_val, min_value=0.0, disabled=weather_from_dataset, key="prcp_input")

    if not weather_from_dataset:
        tab_state.set("tavg", tavg)
        tab_state.set("tmin", tmin)
        tab_state.set("tmax", tmax)
        tab_state.set("prcp", prcp)

    # ----------------------------------------------------------
    # EVENTS (AUTO FROM DATASET IF AVAILABLE)
    # ----------------------------------------------------------
    st.subheader("🎉 Events")

    event_cat_options = _get_event_category_options()
    all_event_cols = list(event_cat_options.values())

    if not df_day.empty and "event_attendance" in df_day.columns:
        attendance_from_dataset = True
        event_attendance_val = float(df_day["event_attendance"].iloc[0])

        active_cols = [col for col in all_event_cols if col in df_day.columns and df_day[col].max() > 0]
        reverse_map = {v: k for k, v in event_cat_options.items()}
        selected_labels = [reverse_map[c] for c in active_cols if c in reverse_map]

        if event_attendance_val == 0 and not selected_labels:
            st.info("No events recorded in the dataset for this date.")
        else:
            st.info("Events are automatically loaded from the dataset (not editable).")
    else:
        attendance_from_dataset = False
        event_attendance_val = tab_state.get("event_attendance", 0.0)
        selected_labels = tab_state.get("event_categories_labels", [])

    col_e1, col_e2 = st.columns([1, 2])

    with col_e2:
        selected_labels = st.multiselect(
            "Event categories",
            list(event_cat_options.keys()),
            default=selected_labels,
            disabled=attendance_from_dataset,
            key="event_categories_multiselect",
        )

    with col_e1:
        event_attendance = st.number_input(
            "Total expected attendance",
            value=event_attendance_val,
            min_value=0.0,
            step=1000.0,
            disabled=attendance_from_dataset,
            key="event_attendance_input",
        )

    if not attendance_from_dataset:
        tab_state.set("event_attendance", event_attendance)
        tab_state.set("event_categories_labels", selected_labels)

    event_categories_columns = [event_cat_options[lbl] for lbl in selected_labels]

    # ----------------------------------------------------------
    # LAGS
    # ----------------------------------------------------------
    st.subheader("📐 Demand lags")

    st.markdown(
        """
        We display:
        - **Total trips to Barcelona** (`total_viajes_dia_lag*`)
        - **Trips from this municipality & origin** (`viajes_lag*`)

        If historical data exists for a lag → **locked value**.  
        Otherwise → **fallback mean**, editable.
        """
    )

    manual_lags = {}

    # GLOBAL LAGS
    with st.expander("🌍 Global lags (total trips to Barcelona)"):
        cols = st.columns(7)
        for i in range(1, 8):
            col = cols[i - 1]
            col_name = f"total_viajes_dia_lag{i}"
            auto_val = auto_lags.get(col_name, np.nan)

            if not np.isnan(auto_val):
                col.number_input(
                    f"Lag {i}",
                    value=float(auto_val),
                    disabled=True,
                    key=f"input_{col_name}",
                )
            else:
                v = col.number_input(
                    f"Lag {i}",
                    value=float(means["global_mean"]),
                    key=f"input_{col_name}",
                )
                manual_lags[col_name] = v

    # MUNICIPALITY LAGS
    with st.expander("🏙️ Municipality→Barcelona lags for this origin"):
        cols = st.columns(7)
        for i in range(1, 8):
            col = cols[i - 1]
            col_name = f"viajes_lag{i}"
            auto_val = auto_lags.get(col_name, np.nan)

            if not np.isnan(auto_val):
                col.number_input(
                    f"Lag {i}",
                    value=float(auto_val),
                    disabled=True,
                    key=f"input_{col_name}",
                )
            else:
                v = col.number_input(
                    f"Lag {i}",
                    value=float(means["muni_mean"]),
                    key=f"input_{col_name}",
                )
                manual_lags[col_name] = v

    # ----------------------------------------------------------
    # PREDICT
    # ----------------------------------------------------------
    st.markdown("---")

    if st.button("🚀 Predict trips", type="primary"):
        with st.spinner("Running prediction..."):

            row_X, lag_values, lag_sources = build_feature_row(
                df_model=df_model,
                feature_cols=feature_cols,
                municipio=municipio_sel,
                origen_label=origen_sel,
                target_date=target_ts,
                tavg=tavg,
                tmin=tmin,
                tmax=tmax,
                prcp=prcp,
                event_categories=event_categories_columns,
                event_attendance=event_attendance,
                manual_lags=manual_lags,
            )

            y_pred_raw = float(model.predict(row_X)[0])
            y_pred = max(y_pred_raw, 0.0)

            mask_real = (
                (df_model["date"] == target_ts) &
                (df_model["municipio_origen_name"] == municipio_sel) &
                (df_model["origen"] == origen_sel)
            )
            real_vals = df_model.loc[mask_real, "viajes"]
            y_real = float(real_vals.iloc[0]) if len(real_vals) else None

        # ------------------------------------------------------
        # RESULTS
        # ------------------------------------------------------
        st.subheader("📌 Prediction result")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Predicted trips", f"{y_pred:,.0f}")
        with c2:
            st.metric("Real trips (historical)", f"{y_real:,.0f}" if y_real is not None else "N/A")

        if y_real is not None:
            abs_err = abs(y_real - y_pred)
            rel_err = abs_err / y_real * 100 if y_real != 0 else None
            st.write(f"Absolute error: **{abs_err:,.0f}**")
            if rel_err is not None:
                st.write(f"Relative error: **{rel_err:.2f}%**")

        # HISTOGRAM (NEW, BEAUTIFUL)
        st.subheader("📊 Historical distribution of trips for this OD link")
        _build_hist_plot(df_model, municipio_sel, origen_sel, y_real, y_pred)

        # SAVE FOR SHAP LOCAL
        latest_pred = {
            "date": str(target_date),
            "municipio": municipio_sel,
            "origen": origen_sel,
            "y_pred": y_pred,
            "y_real": y_real,
            "features": row_X.to_dict(orient="records")[0],
        }
        tab_state.set("latest_prediction", latest_pred)

        st.success("Prediction stored. You can now visit the 🧠 Explainability tab for SHAP interpretation.")
