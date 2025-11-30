import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk

from utils.state_manager import StateManager


def render_weather_events() -> None:
    """
    Tab 4 – Weather & Events ("Contextual view")

    Analyses how weather (tavg, prcp, …) and events (event(y/n), name, attendance)
    are related to mobility (viajes).
    """
    st.header("🌦️ Weather & Events Impact on Mobility")

    # -------------------------------------------------------------------------
    # 1. Load global data from StateManager
    # -------------------------------------------------------------------------
    global_state = StateManager("global")
    df = global_state.get("df_main")
    geo = global_state.get("df_geo")  # municipalities with lat / lon

    if df is None:
        st.error("Global dataset (df_main) is not loaded in StateManager('global').")
        st.info("Check that main.py loads the combined dataset into df_main.")
        return

    # -------------------------------------------------------------------------
    # 2. Basic type cleaning (only once, dataframe is cached globally)
    # -------------------------------------------------------------------------
    if not pd.api.types.is_datetime64_any_dtype(df["day"]):
        df["day"] = pd.to_datetime(df["day"], errors="coerce")

    numeric_cols = ["viajes", "tavg", "tmin", "tmax", "prcp", "attendance"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Flag days with at least one event
    if "event(y/n)" in df.columns:
        df["has_event"] = df["event(y/n)"].astype(str).str.lower().eq("y")
    else:
        df["has_event"] = False

    # -------------------------------------------------------------------------
    # 3. Daily aggregation: mobility + weather + events
    # -------------------------------------------------------------------------
    daily = (
        df.groupby("day", as_index=False)
        .agg(
            viajes=("viajes", "sum"),
            tavg=("tavg", "mean"),
            tmin=("tmin", "mean"),
            tmax=("tmax", "mean"),
            prcp=("prcp", "mean"),
            has_event=("has_event", "any"),
            total_attendance=("attendance", "sum"),
        )
        .sort_values("day")
    )

    if daily.empty:
        st.warning("No daily data available after aggregation.")
        return

    min_day = daily["day"].min().date()
    max_day = daily["day"].max().date()

    # -------------------------------------------------------------------------
    # 4. Controls – date range
    # -------------------------------------------------------------------------
    st.subheader("📅 Date range")

    date_range = st.date_input(
        "Select date range",
        value=(min_day, max_day),
        min_value=min_day,
        max_value=max_day,
        help="This range will be used for all charts and statistics below.",
    )

    # date_input can return a single date or a tuple
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    mask = (daily["day"].dt.date >= start_date) & (daily["day"].dt.date <= end_date)
    daily_sel = daily.loc[mask].copy()

    if daily_sel.empty:
        st.info("No data in the selected date range.")
        return

    # -------------------------------------------------------------------------
    # 5. KPI mini-dashboard
    # -------------------------------------------------------------------------
    st.subheader("📊 Summary for selected period")

    total_trips = int(daily_sel["viajes"].sum())
    avg_temp = float(daily_sel["tavg"].mean()) if "tavg" in daily_sel else np.nan
    total_rain = float(daily_sel["prcp"].sum()) if "prcp" in daily_sel else np.nan
    days_with_events = int(daily_sel["has_event"].sum())
    total_att = int(daily_sel["total_attendance"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total trips", f"{total_trips:,}")
    c2.metric("Avg. temperature (°C)", f"{avg_temp:.1f}" if not np.isnan(avg_temp) else "—")
    c3.metric("Total rain (mm)", f"{total_rain:.1f}" if not np.isnan(total_rain) else "—")
    c4.metric("Days with events", days_with_events)
    c5.metric("Total attendance", f"{total_att:,}")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 6. Mobility vs Weather
    # -------------------------------------------------------------------------
    st.subheader("🌧️ Mobility vs precipitation")

    if "prcp" in daily_sel.columns:
        fig_prcp = px.scatter(
            daily_sel,
            x="prcp",
            y="viajes",
            color="has_event",
            size="total_attendance",
            labels={"prcp": "Daily precipitation (mm)", "viajes": "Trips"},
            title="Trips vs precipitation (event days in color)",
        )
        st.plotly_chart(fig_prcp, use_container_width=True)
    else:
        st.info("Column 'prcp' is not available in the dataset.")

    st.subheader("🌡️ Mobility vs temperature")

    if "tavg" in daily_sel.columns:
        fig_temp = px.scatter(
            daily_sel,
            x="tavg",
            y="viajes",
            color="has_event",
            size="total_attendance",
            labels={"tavg": "Average temperature (°C)", "viajes": "Trips"},
            title="Trips vs average temperature (event days in color)",
        )
        st.plotly_chart(fig_temp, use_container_width=True)
    else:
        st.info("Column 'tavg' is not available in the dataset.")

    # -------------------------------------------------------------------------
    # 7. Simple correlations
    # -------------------------------------------------------------------------
    st.subheader("📐 Correlations (selected period)")

    corr_cols = [c for c in ["viajes", "prcp", "tavg"] if c in daily_sel.columns]
    if len(corr_cols) >= 2:
        corr_matrix = daily_sel[corr_cols].corr().round(2)
        st.dataframe(corr_matrix)
    else:
        st.info("Not enough numeric columns to compute correlations.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 8. Events over time – highlighting event days
    # -------------------------------------------------------------------------
    st.subheader("🎟️ Events over time")

    fig_ts = px.line(
        daily_sel,
        x="day",
        y="viajes",
        title="Daily trips with event days highlighted",
        labels={"day": "Date", "viajes": "Trips"},
    )

    # Add event markers
    event_days = daily_sel[daily_sel["has_event"]].copy()
    if not event_days.empty:
        fig_ts.add_scatter(
            x=event_days["day"],
            y=event_days["viajes"],
            mode="markers",
            name="Event day",
            marker=dict(size=9),
        )

    st.plotly_chart(fig_ts, use_container_width=True)

    # Show table of days with events in the selected range
    st.write("#### Event days in selected period")
    if not event_days.empty:
        st.dataframe(
            event_days[["day", "viajes", "total_attendance"]],
            use_container_width=True,
        )
    else:
        st.info("No event days in the selected period.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 9. Ranking of impact: Δ trips on event days
    #    Δ = trips_day_event − rolling mean of previous 7 days
    # -------------------------------------------------------------------------
    st.subheader("🏆 Impact ranking of event days")

    daily_sorted = daily.sort_values("day").copy()
    daily_sorted["baseline"] = (
        daily_sorted["viajes"]
        .rolling(window=7, min_periods=3)
        .mean()
        .shift(1)
    )
    daily_sorted["delta"] = daily_sorted["viajes"] - daily_sorted["baseline"]

    ranking = (
        daily_sorted[
            (daily_sorted["has_event"])
            & daily_sorted["baseline"].notna()
        ]
        .sort_values("delta", ascending=False)
        .head(10)
    )

    if ranking.empty:
        st.info("Not enough history to compute impact ranking for event days.")
    else:
        ranking_display = ranking[["day", "viajes", "baseline", "delta", "total_attendance"]]
        ranking_display["baseline"] = ranking_display["baseline"].round(1)
        ranking_display["delta"] = ranking_display["delta"].round(1)

        st.dataframe(ranking_display, use_container_width=True)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 10. Optional map: heatmap of trips on an event day (if geo is available)
    # -------------------------------------------------------------------------
    if geo is not None and not geo.empty:
        st.subheader("🗺️ Mobility heatmap on a selected event day")

        # Only days with events and inside the selected range
        event_days_full = daily_sorted[daily_sorted["has_event"]]["day"].dt.date.unique()
        event_days_range = [
            d for d in event_days_full
            if start_date <= d <= end_date
        ]

        if event_days_range:
            selected_event_day = st.selectbox(
                "Select an event day for the map",
                options=sorted(event_days_range),
            )

            df_day = df[df["day"].dt.date == selected_event_day].copy()
            if df_day.empty:
                st.info("No trip records for the selected event day.")
            else:
                # Aggregate by destination municipality
                df_map = (
                    df_day.groupby("municipio_destino_name", as_index=False)["viajes"]
                    .sum()
                    .rename(columns={"municipio_destino_name": "municipio"})
                )

                df_map = df_map.merge(geo, on="municipio", how="left")
                df_map = df_map.dropna(subset=["lat", "lon"])

                if df_map.empty:
                    st.info("No geospatial data available for this event day.")
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
                        get_fill_color=[255, 140, 0, 160],
                    )

                    view_state = pdk.ViewState(
                        latitude=41.39,
                        longitude=2.17,
                        zoom=9,
                        pitch=0,
                    )

                    tooltip = {
                        "html": "<b>{municipio}</b><br/>Trips: {viajes}",
                        "style": {"color": "white"},
                    }

                    st.pydeck_chart(
                        pdk.Deck(
                            layers=[layer],
                            initial_view_state=view_state,
                            tooltip=tooltip,
                        )
                    )
        else:
            st.info("No event days available for the map in the selected period.")
    else:
        st.info("Geospatial data (df_geo) is not available; skipping map.")
