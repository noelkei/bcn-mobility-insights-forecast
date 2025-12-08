import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render_event_eda(df):
    st.header("🎟️ Events & Mobility Analysis")

    # ---------------------------------------------------------
    # 1. ROBUST DATA PREPARATION
    # ---------------------------------------------------------
    df_working = df.copy()

    # A) FECHA SEGURA
    if "day" in df_working.columns:
        source_col = "day"
    elif "date" in df_working.columns:
        source_col = "date"
    else:
        st.error("⚠️ Critical Error: No 'day' or 'date' column found.")
        return

    # Extraer y convertir fecha
    raw_dates = df_working[source_col]
    if isinstance(raw_dates, pd.DataFrame):
        raw_dates = raw_dates.iloc[:, 0]
    
    date_series = pd.to_datetime(raw_dates, errors='coerce')

    if "date" in df_working.columns:
        df_working = df_working.drop(columns=["date"])

    df_working["date"] = date_series
    df_working = df_working.dropna(subset=["date"])

    # B) RENOMBRADO INTELIGENTE (Buscamos las columnas clave)
    
    # 1. Asistencia
    att_col = None
    if "event_attendance" in df_working.columns:
        att_col = "event_attendance"
    elif "attendance" in df_working.columns:
        att_col = "attendance"
    
    # 2. Flag de Evento
    flag_col = None
    if "event_flag" in df_working.columns:
        flag_col = "event_flag"
    elif "is_event" in df_working.columns:
        flag_col = "is_event"
    elif "event(y/n)" in df_working.columns:
        flag_col = "event(y/n)"

    # Mapeo
    col_map = {
        "event_names": "event",
        "event_category": "event_type",
        "viajes": "trips"
    }
    
    if att_col: col_map[att_col] = "attendance_clean"
    if flag_col: col_map[flag_col] = "is_event"

    rename_dict = {k: v for k, v in col_map.items() if k in df_working.columns}
    df_working = df_working.rename(columns=rename_dict)

    # C) LIMPIEZA Y RELLENO (CRÍTICO)
    
    # Si no existe attendance, creamos 0
    if "attendance_clean" not in df_working.columns:
        df_working["attendance_clean"] = 0
    else:
        df_working["attendance_clean"] = pd.to_numeric(df_working["attendance_clean"], errors='coerce').fillna(0)

    # Si no existe trips, creamos 0
    if "trips" not in df_working.columns:
        df_working["trips"] = 0
    else:
        df_working["trips"] = pd.to_numeric(df_working["trips"], errors='coerce').fillna(0)

    # D) LÓGICA MAESTRA PARA 'is_event' (Arreglo del vacío)
    # Si tenemos columna is_event, la limpiamos. Si no, la deducimos.
    if "is_event" in df_working.columns:
        # Convertimos 'y'/'n' a 1/0 si fuera necesario
        if df_working["is_event"].dtype == 'object':
             df_working["is_event"] = df_working["is_event"].apply(lambda x: 1 if str(x).lower() in ['y', 'yes', '1', 'true'] else 0)
        else:
             df_working["is_event"] = pd.to_numeric(df_working["is_event"], errors='coerce').fillna(0)
    else:
        # Deducir: Es evento si hay asistencia > 0
        df_working["is_event"] = np.where(df_working["attendance_clean"] > 0, 1, 0)

    # E) LIMPIEZA DE NOMBRES DE EVENTO ("none", "nan")
    if "event" in df_working.columns:
        df_working["event"] = df_working["event"].astype(str).replace({'none': None, 'None': None, 'nan': None, 'NaN': None})
        # Si el evento es nulo pero is_event=1, ponemos "Unknown Event"
        df_working.loc[(df_working["is_event"] == 1) & (df_working["event"].isnull()), "event"] = "Unknown Event"
    else:
        df_working["event"] = "Unknown"

    # ---------------------------------------------------------
    # 2. DATE FILTER
    # ---------------------------------------------------------
    unique_dates = df_working["date"].unique()
    if len(unique_dates) == 0:
        st.warning("No valid dates found.")
        return

    min_date, max_date = unique_dates.min(), unique_dates.max()
    date_range = st.date_input("Filter date range:", [min_date, max_date], key="event_date_range")

    if len(date_range) != 2: return
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    
    df_filtered = df_working[(df_working["date"] >= start) & (df_working["date"] <= end)].copy()

    if df_filtered.empty:
        st.warning("No data in range.")
        return

    # ---------------------------------------------------------
    # 3. AGGREGATION & METRICS
    # ---------------------------------------------------------
    # Agrupamos por día
    daily_stats = df_filtered.groupby("date").agg({
        "attendance_clean": "max", # Usamos max porque la asistencia se repite por fila
        "trips": "sum",            
        "is_event": "max"
    }).reset_index()

    # Feature Engineering
    daily_stats["is_weekend"] = daily_stats["date"].dt.dayofweek >= 5

    event_days = daily_stats[daily_stats["is_event"] == 1]
    no_event_days = daily_stats[daily_stats["is_event"] == 0]

    # Metrics
    n_event = len(event_days)
    avg_att = event_days["attendance_clean"].mean() if n_event > 0 else 0
    avg_trips_event = event_days["trips"].mean() if n_event > 0 else 0

    n_no_event = len(no_event_days)
    avg_trips_no_event = no_event_days["trips"].mean() if n_no_event > 0 else 0
    
    # Delta
    delta_trips = 0
    if avg_trips_no_event > 0:
        delta_trips = ((avg_trips_event - avg_trips_no_event) / avg_trips_no_event) * 100

    # ---------------------------------------------------------
    # 4. KPI DISPLAY
    # ---------------------------------------------------------
    st.subheader("1️⃣ Average Metrics Comparison")
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### 🏟️ Event Days")
        st.metric("Number of Days", n_event)
        st.metric("Avg. Attendance", f"{avg_att:,.0f}")
        st.metric("Avg. Mobility (Trips)", f"{avg_trips_event:,.0f}", delta=f"{delta_trips:.1f}% Impact")
    with c2:
        st.markdown("### 🏙️ Non-Event Days")
        st.metric("Number of Days", n_no_event)
        st.metric("Avg. Attendance", "0")
        st.metric("Avg. Mobility (Trips)", f"{avg_trips_no_event:,.0f}")

    # INSIGHT 1
    st.info(
        "**Observation:** Large events create a measurable shift in city dynamics. "
        f"On average, event days see a **{delta_trips:.1f}%** variation in total trips. "
        "However, since many major events occur on **weekends** (when traffic is naturally lower), the global number often stays below a typical busy workday peak."
    )

    st.markdown("---")

    # ---------------------------------------------------------
    # 5. IMPACT ANALYSIS (SECCIÓN RENOMBRADA Y MOVIDA)
    # ---------------------------------------------------------
    st.subheader("2️⃣ Impact Analysis: Events vs. Mobility")

    c_sc, c_box = st.columns(2)

    with c_sc:
        st.caption("Correlation: Attendance vs. Trips")
        # Scatter
        evt_scatter_data = daily_stats[daily_stats["is_event"]==1]
        
        if not evt_scatter_data.empty:
            fig3 = px.scatter(
                evt_scatter_data, 
                x="attendance_clean", 
                y="trips", 
                size="attendance_clean",
                color="attendance_clean",
                trendline="ols",
                labels={"attendance_clean": "Event Attendance", "trips": "City Trips"},
                color_continuous_scale="Oranges"
            )
            fig3.update_layout(height=400, coloraxis_showscale=False, title_text="Correlation: Attendance vs. Trips")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Not enough event days for correlation.")

    with c_box:
        st.caption("Variability: Normal vs Event Days")
        daily_stats["Day Type"] = daily_stats["is_event"].map({0: "Normal Day", 1: "Event Day"})
        
        fig4 = px.box(
            daily_stats, 
            x="Day Type", 
            y="trips", 
            color="Day Type",
            color_discrete_map={"Normal Day": "lightblue", "Event Day": "orange"},
            labels={"trips": "Total Trips", "Day Type": ""}
        )
        fig4.update_layout(showlegend=False, height=400, title_text="Variability: Normal vs Event Days")
        st.plotly_chart(fig4, use_container_width=True)

    # INSIGHT 2 (CORREGIDO)
    st.info(
        "**Analysis:** The correlation between attendance and total trips is **weak**. "
        "A key factor is the **audience origin**: events attracting primarily local residents (already in the city) generate fewer *new* inter-city trips. "
        "Also, the **Weekend Effect** (huge Sunday events vs low traffic baseline) masks the global impact."
    )

    # ---------------------------------------------------------
    # 6. DUAL AXIS TIMELINE (SECCIÓN RENOMBRADA)
    # ---------------------------------------------------------
    st.subheader("3️⃣ Timeline: Mobility & Events Spikes")
    
    fig_dual = make_subplots(specs=[[{"secondary_y": True}]])

    fig_dual.add_trace(
        go.Scatter(
            x=daily_stats["date"], 
            y=daily_stats["trips"], 
            name="Total Trips", 
            line=dict(color="#1f77b4", width=2)
        ),
        secondary_y=False,
    )

    fig_dual.add_trace(
        go.Bar(
            x=daily_stats["date"], 
            y=daily_stats["attendance_clean"], 
            name="Event Attendance", 
            marker=dict(color="red", opacity=0.4)
        ),
        secondary_y=True,
    )

    fig_dual.update_layout(
        title_text="Daily Mobility vs. Event Attendance",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40)
    )
    fig_dual.update_yaxes(title_text="Total City Trips", secondary_y=False)
    fig_dual.update_yaxes(title_text="Event Attendance", secondary_y=True)

    st.plotly_chart(fig_dual, use_container_width=True)
    
    # INSIGHT 3
    st.info(
        "**Timeline Observation:** Massive events (red spikes) often coincide with dips in mobility (blue line). "
        "This visualizes how scheduling major events on weekends leverages the city's spare capacity."
    )

    st.markdown("---")

    # ---------------------------------------------------------
    # 7. FINAL CONCLUSION
    # ---------------------------------------------------------
    st.subheader("📝 Final Conclusion")
    st.success(
        """
        Events trigger localized demand shocks, but their city-wide impact is often buffered by the calendar.
        
        Since major events frequently align with weekends (low-mobility periods), the network often has spare capacity to absorb the extra volume. The real operational challenge isn't just volume, but **variance**: event days are far less predictable than standard days.
        """
    )