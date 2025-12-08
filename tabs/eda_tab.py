import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def render_time_eda(df):
    st.header("⏳ Temporal & Seasonality Analysis")
    # (Sub-caption removed)

    # ---------------------------------------------------------
    # 1. ROBUST DATA PREPARATION
    # ---------------------------------------------------------
    df_working = df.copy()

    # A) Secure Date
    if "day" in df_working.columns:
        source_col = "day"
    elif "date" in df_working.columns:
        source_col = "date"
    else:
        st.error("⚠️ Critical Error: No 'day' or 'date' column found.")
        return

    # Extract date
    raw_dates = df_working[source_col]
    if isinstance(raw_dates, pd.DataFrame):
        raw_dates = raw_dates.iloc[:, 0]
    
    date_series = pd.to_datetime(raw_dates, errors='coerce')

    # Clean
    if "date" in df_working.columns:
        df_working = df_working.drop(columns=["date"])

    df_working["date"] = date_series
    df_working = df_working.dropna(subset=["date"])

    # B) Normalize trips column
    if "viajes" not in df_working.columns:
        possible_names = ["trips", "total_viajes", "demand"]
        found = False
        for name in possible_names:
            if name in df_working.columns:
                df_working["viajes"] = df_working[name]
                found = True
                break
        if not found:
            st.error("Trips column ('viajes') not found.")
            return

    # Ensure numeric
    df_working["viajes"] = pd.to_numeric(df_working["viajes"], errors='coerce').fillna(0)

    # ---------------------------------------------------------
    # 2. DATE FILTER
    # ---------------------------------------------------------
    unique_dates = df_working["date"].unique()
    if len(unique_dates) == 0:
        st.warning("No valid dates found.")
        return

    min_date, max_date = unique_dates.min(), unique_dates.max()
    
    date_range = st.date_input("Filter date range:", [min_date, max_date], key="time_date_range")
    
    if len(date_range) != 2: return
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    
    # FILTERED DATAFRAME (Row Level)
    df_filtered_raw = df_working[(df_working["date"] >= start) & (df_working["date"] <= end)].copy()

    if df_filtered_raw.empty:
        st.warning("No data in selected range.")
        return

    # AGGREGATED DATAFRAME (Day Level)
    daily_df = df_filtered_raw.groupby("date")["viajes"].sum().reset_index()
    
    # Feature Engineering
    daily_df["weekday_name"] = daily_df["date"].dt.day_name()
    daily_df["month_name"] = daily_df["date"].dt.month_name()
    daily_df["month_num"] = daily_df["date"].dt.month
    daily_df["is_weekend"] = daily_df["date"].dt.dayofweek >= 5

    # ---------------------------------------------------------
    # 3. KPI: WEEKEND VS WEEKDAY
    # ---------------------------------------------------------
    st.subheader("1️⃣ Workdays vs. Weekends")

    avg_workday = daily_df[~daily_df["is_weekend"]]["viajes"].mean()
    avg_weekend = daily_df[daily_df["is_weekend"]]["viajes"].mean()
    
    drop_pct = 0
    if avg_workday > 0:
        drop_pct = ((avg_weekend - avg_workday) / avg_workday) * 100

    col1, col2 = st.columns(2)
    col1.metric("Avg. Workday (M-F)", f"{avg_workday:,.0f}")
    col2.metric("Avg. Weekend (S-S)", f"{avg_weekend:,.0f}", delta=f"{drop_pct:.1f}% vs Workday")

    # INSIGHT 1
    st.info(
        "**Observation:** The significant drop in mobility on weekends quantifies the city's reliance on obligatory commuting. "
        "Weekdays represent the system's full capacity load, while weekends reflect the 'base load' of leisure and non-essential travel."
    )

    st.markdown("---")

    # ---------------------------------------------------------
    # 4. SEASONALITY CHARTS
    # ---------------------------------------------------------
    st.subheader("2️⃣ Cyclic Patterns (Seasonality)")

    c_chart1, c_chart2 = st.columns(2)

    # A) Weekly Profile
    with c_chart1:
        # (Sub-caption removed)
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekly_profile = daily_df.groupby("weekday_name")["viajes"].mean().reindex(days_order)
        
        fig1 = px.bar(
            x=weekly_profile.index, 
            y=weekly_profile.values,
            labels={'x': '', 'y': 'Avg. Trips'},
            color_discrete_sequence=['#1f77b4']
        )
        fig1.update_layout(showlegend=False, height=350, margin=dict(t=30, l=50), title_text="Typical Weekly Profile")
        st.plotly_chart(fig1, use_container_width=True)

    # B) Monthly Profile
    with c_chart2:
        # (Sub-caption removed)
        monthly_profile = daily_df.groupby(["month_num", "month_name"])["viajes"].mean().reset_index()
        monthly_profile = monthly_profile.sort_values("month_num")

        fig2 = px.line(
            monthly_profile, 
            x="month_name", 
            y="viajes",
            markers=True,
            labels={'month_name': '', 'viajes': 'Avg. Trips'},
            color_discrete_sequence=['green']
        )
        fig2.update_yaxes(rangemode="tozero") 
        fig2.update_layout(height=350, margin=dict(t=30, l=50), title_text="Annual Evolution (Monthly)")
        st.plotly_chart(fig2, use_container_width=True)

    # INSIGHT 2
    st.info(
        "**Analysis:** The weekly profile clearly identifies the working week peak (usually Tuesday-Thursday) versus the weekend trough. "
        "Monthly variations reveal broader seasonal effects, such as reduced mobility during holiday periods (e.g., August) or peaks during active academic/business months."
    )

    # ---------------------------------------------------------
    # 5. TREND & SMOOTHING
    # ---------------------------------------------------------
    st.subheader("3️⃣ Evolution & Trend")
    
    daily_df["Trend (7-day avg)"] = daily_df["viajes"].rolling(window=7, center=True).mean()

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=daily_df["date"], y=daily_df["viajes"],
        mode='lines', name='Daily (Actual)',
        line=dict(color='gray', width=1), opacity=0.4
    ))
    fig3.add_trace(go.Scatter(
        x=daily_df["date"], y=daily_df["Trend (7-day avg)"],
        mode='lines', name='Trend (7-day avg)',
        line=dict(color='#e74c3c', width=3)
    ))
    
    fig3.update_layout(
        title="Total Daily Trips (Actual vs. Trend)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40)
    )
    st.plotly_chart(fig3, use_container_width=True)

    # INSIGHT 3
    st.info(
        "**Trend Analysis:** The 7-day moving average (red line) filters out the weekly noise (the weekend dips), revealing the true underlying trajectory of city mobility demand over time, showing structural growth or decline."
    )

    st.markdown("---")

    # ---------------------------------------------------------
    # 6. TOP ORIGIN MUNICIPALITIES
    # ---------------------------------------------------------
    st.subheader("4️⃣ Top Origin Municipalities")
    # (Sub-caption removed)

    if "municipio_origen_name" in df_filtered_raw.columns:
        top_origins = df_filtered_raw.groupby("municipio_origen_name")["viajes"].sum().sort_values(ascending=True).tail(15)
        
        fig_ori = px.bar(
            x=top_origins.values,
            y=top_origins.index,
            orientation='h',
            text_auto='.2s',
            labels={'x': 'Total Accumulated Trips', 'y': 'Origin Municipality'},
            color=top_origins.values,
            color_continuous_scale="Blues"
        )
        # FIX: Added margin-left (l=180) to prevent overlap of long names
        fig_ori.update_layout(
            height=500, 
            showlegend=False, 
            coloraxis_showscale=False,
            margin=dict(l=180, t=30) 
        )
        st.plotly_chart(fig_ori, use_container_width=True)
        
        top_1 = top_origins.index[-1]
        # INSIGHT 4
        st.info(
            f"**Key Feeder:** The municipality with the highest volume of trips towards the destination in this period is **{top_1}**. "
            "This chart identifies the primary commuter corridors connecting the metropolitan area."
        )
    
    else:
        st.warning("Column 'municipio_origen_name' not found in the dataset.")

    st.markdown("---")

    # ---------------------------------------------------------
    # 7. CORRELATION MATRIX
    # ---------------------------------------------------------
    st.subheader("5️⃣ Correlation Matrix (Key Drivers)")
    # (Sub-caption removed)

    target_cols = [
        "total_viajes_dia", # Global City Mobility
        "tavg",             # Temperature
        "prcp",             # Rain
        "is_weekend",       # Weekend Factor
        "event_attendance"        # Event Attendance
    ]
    
    cols_to_corr = [c for c in target_cols if c in df_filtered_raw.columns]

    if len(cols_to_corr) > 1:
        corr_matrix = df_filtered_raw[cols_to_corr].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            labels=dict(color="Correlation")
        )
        fig_corr.update_layout(height=500, title="Heatmap of Statistical Relationships")
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # INSIGHT 5
        st.info(
            """
            **Interpretation:** This matrix reveals the strength of relationships between macro-factors. 
            * **Strong Negative (Blue):** Typically seen between `total_viajes_dia` and `is_weekend`, confirming that the calendar is the strongest driver of mobility.
            * **Weak/Neutral (White):** Often observed with weather variables at a daily aggregation level, indicating lower systemic impact compared to calendar factors.
            """
        )
    else:
        st.warning("Not enough numeric columns available for correlation analysis.")

    st.markdown("---")

    # ---------------------------------------------------------
    # 8. FINAL CONCLUSION
    # ---------------------------------------------------------
    st.subheader("📝 Final Conclusion")
    st.success(
        """
        Time determines demand. The analysis confirms that the primary driver of Barcelona's mobility is the **calendar routine (workday vs. weekend)**, which dictates the system's base load.

        While external factors like weather or specific events cause fluctuations, the structural patterns—weekly cycles and key commuter corridors from metropolitan municipalities—remain the dominant forces shaping city dynamics.
        """
    )