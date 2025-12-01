import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk

from utils.state_manager import StateManager


def render_weather_events() -> None:
    st.header("🌦️ Clima y Eventos: Impacto en la Movilidad")

    # Cargar datos globales
    global_state = StateManager("global")
    df = global_state.get("df_main")
    geo = global_state.get("df_geo")

    if df is None:
        st.error("No se ha cargado el dataset combinado en StateManager('global').")
        return

    # Conversión de tipos
    if not pd.api.types.is_datetime64_any_dtype(df["day"]):
        df["day"] = pd.to_datetime(df["day"], errors="coerce")

    numeric_cols = ["viajes", "tavg", "tmin", "tmax", "prcp", "attendance"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["has_event"] = df["event(y/n)"].astype(str).str.lower().eq("y") if "event(y/n)" in df.columns else False

    # Agregación diaria
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
        st.warning("No hay datos después de la agregación diaria.")
        return

    min_day = daily["day"].min().date()
    max_day = daily["day"].max().date()

    st.subheader("📅 Rango de fechas")
    date_range = st.date_input(
        "Selecciona el rango de fechas",
        value=(min_day, max_day),
        min_value=min_day,
        max_value=max_day,
        help="Este rango se aplicará a todos los gráficos y KPIs."
    )

    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    mask = (daily["day"].dt.date >= start_date) & (daily["day"].dt.date <= end_date)
    daily_sel = daily.loc[mask].copy()

    if daily_sel.empty:
        st.info("No hay datos en el rango seleccionado.")
        return

    # KPIs
    st.subheader("📊 Resumen del periodo seleccionado")
    total_trips = int(daily_sel["viajes"].sum())
    avg_temp = float(daily_sel["tavg"].mean()) if "tavg" in daily_sel else np.nan
    total_rain = float(daily_sel["prcp"].sum()) if "prcp" in daily_sel else np.nan
    days_with_events = int(daily_sel["has_event"].sum())
    total_att = int(daily_sel["total_attendance"].sum())

    def format_eu(n):
        return f"{n:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")

    def format_kpi(n):
        if n >= 1_000_000:
            return f"{n/1_000_000:.1f} millones".replace(".", ",")
        return format_eu(n)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Viajes totales", format_eu(total_trips))
    c2.metric("Temperatura media (°C)", format_eu(avg_temp) if not np.isnan(avg_temp) else "—")
    c3.metric("Lluvia total (mm)", format_eu(total_rain) if not np.isnan(total_rain) else "—")
    c4.metric("Días con eventos", days_with_events)
    c5.metric("Asistencia total", format_kpi(total_att))

    st.markdown("---")

    # Gráfico: viajes vs precipitación
    st.subheader("🌧️ Viajes vs precipitación diaria")
    if "prcp" in daily_sel.columns:
        fig_prcp = px.scatter(
            daily_sel,
            x="prcp",
            y="viajes",
            size="total_attendance",
            labels={"prcp": "Precipitación diaria (mm)", "viajes": "Viajes"},
        )
        st.plotly_chart(fig_prcp, use_container_width=True)

    st.subheader("🌡️ Viajes vs temperatura media")
    if "tavg" in daily_sel.columns:
        fig_temp = px.scatter(
            daily_sel,
            x="tavg",
            y="viajes",
            size="total_attendance",
            labels={"tavg": "Temperatura media (°C)", "viajes": "Viajes"},
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    st.markdown("---")

    # Gráfico de series temporales con eventos
    st.subheader("🎟️ Eventos en el tiempo")
    fig_ts = px.line(
        daily_sel,
        x="day",
        y="viajes",
        labels={"day": "Fecha", "viajes": "Viajes"},
    )
    event_days = daily_sel[daily_sel["has_event"]].copy()
    if not event_days.empty:
        fig_ts.add_scatter(
            x=event_days["day"],
            y=event_days["viajes"],
            mode="markers",
            name="Día con evento",
            marker=dict(size=9),
        )
    st.plotly_chart(fig_ts, use_container_width=True)

    st.write("#### Días con evento en el periodo seleccionado")
    if not event_days.empty:
        df_disp = event_days[["day", "viajes", "total_attendance"]].copy()
        df_disp["total_attendance"] = df_disp["total_attendance"].apply(format_kpi)
        st.dataframe(df_disp, use_container_width=True)
    else:
        st.info("No hay días con evento en este periodo.")

    st.markdown("---")

    # Ranking impacto eventos
    st.subheader("🏆 Ranking de impacto de eventos")
    daily_sorted = daily.sort_values("day").copy()
    daily_sorted["baseline"] = daily_sorted["viajes"].rolling(window=7, min_periods=3).mean().shift(1)
    daily_sorted["delta"] = daily_sorted["viajes"] - daily_sorted["baseline"]

    ranking = daily_sorted[(daily_sorted["has_event"]) & daily_sorted["baseline"].notna()].sort_values("delta", ascending=False).head(10)
    if ranking.empty:
        st.info("No hay suficientes datos para calcular el ranking.")
    else:
        ranking_display = ranking[["day", "viajes", "baseline", "delta", "total_attendance"]].copy()
        ranking_display["baseline"] = ranking_display["baseline"].round(1)
        ranking_display["delta"] = ranking_display["delta"].round(1)
        ranking_display.columns = ["Fecha", "Viajes", "Promedio anterior", "Variación frente al promedio", "Asistencia total"]
        ranking_display["Asistencia total"] = ranking_display["Asistencia total"].apply(format_kpi)
        st.dataframe(ranking_display, use_container_width=True)

    st.markdown("---")

    # Mapa de calor
    if geo is not None and not geo.empty:
        st.subheader("🗺️ Mapa de calor de movilidad")

        available_days = df["day"].dt.date.unique()
        selected_day = st.selectbox(
            "Selecciona un día para visualizar el mapa",
            options=sorted(d for d in available_days if start_date <= d <= end_date),
        )

        df_day = df[df["day"].dt.date == selected_day].copy()
        if df_day.empty:
            st.info("No hay registros para el día seleccionado.")
        else:
            df_map = (
                df_day.groupby("municipio_destino_name", as_index=False)["viajes"]
                .sum()
                .rename(columns={"municipio_destino_name": "municipio"})
            )
            df_map = df_map.merge(geo, on="municipio", how="left").dropna(subset=["lat", "lon"])

            if df_map.empty:
                st.info("No hay datos geoespaciales para este día.")
            else:
                vmax = df_map["viajes"].max()
                MIN_RADIUS = 120
                MAX_RADIUS = 1500
                df_map["radius"] = MIN_RADIUS + (
                    (MAX_RADIUS - MIN_RADIUS)
                    * (np.log1p(df_map["viajes"]) / np.log1p(vmax))
                    if vmax > 0 else MIN_RADIUS
                )

                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=df_map,
                    get_position="[lon, lat]",
                    get_radius="radius",
                    pickable=True,
                    get_fill_color=[255, 140, 0, 160],
                )

                view_state = pdk.ViewState(latitude=41.39, longitude=2.17, zoom=9, pitch=0)
                tooltip = {"html": "<b>{municipio}</b><br/>Viajes: {viajes}", "style": {"color": "white"}}
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))
    else:
        st.info("No hay datos geoespaciales disponibles para mostrar el mapa.")