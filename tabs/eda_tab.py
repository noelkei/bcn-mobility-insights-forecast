import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
from statsmodels.tsa.stattools import adfuller
from utils.state_manager import StateManager

def render_eda(df: pd.DataFrame) -> None:
    def format_number(value, decimals=2):
        try:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return ""
            format_str = "{:,.%df}" % decimals
            formatted = format_str.format(value)
            formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
            if decimals == 0 and formatted.endswith(","):
                formatted = formatted[:-1]
            return formatted
        except Exception:
            return str(value)

    df = df.copy()

    if isinstance(df.index, pd.DatetimeIndex):
        pass
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.set_index('date', inplace=True)
    elif 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        df.set_index('fecha', inplace=True)

    if not df.index.is_monotonic_increasing:
        df.sort_index(inplace=True)

    min_date = df.index.min()
    max_date = df.index.max()
    date_range = st.date_input("Select date range:", [min_date, max_date])
    start_date, end_date = date_range if len(date_range) == 2 else (date_range, date_range)
    start_ts, end_ts = pd.to_datetime(start_date), pd.to_datetime(end_date)
    if start_ts > end_ts:
        start_ts, end_ts = end_ts, start_ts
    df_filtered = df.loc[start_ts:end_ts].copy()

    st.title("Exploratory Data Analysis")

    st.subheader("Distribution Analysis")
    dist_vars = [c for c in df_filtered.columns if c in ["total_viajes_dia", "tavg", "prcp", "attendance"]]
    cols_per_row = 4
    for i in range(0, len(dist_vars), cols_per_row):
        row_vars = dist_vars[i:i+cols_per_row]
        cols = st.columns(len(row_vars))
        for j, var in enumerate(row_vars):
            with cols[j]:
                st.markdown(f"**{var}**")
                fig, ax = plt.subplots(figsize=(5, 3.5))
                data = df_filtered[var]
                if var == 'prcp':
                    data = data[data < 30]  # reduce skew
                sns.histplot(data.dropna(), ax=ax, kde=True, color='#69b3a2')
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: format_number(x, 1)))
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: format_number(x, 0)))
                ax.set_xlabel(var)
                ax.set_ylabel("Count")
                sns.despine(ax=ax)
                st.pyplot(fig)
                plt.close(fig)

    st.subheader("Temporal Analysis")
    if 'total_viajes_dia' in df_filtered.columns and len(df_filtered) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df_filtered.index, df_filtered['total_viajes_dia'], label="Total viajes dia", color='#377eb8')
        ax.set_title("Daily Trips Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Total Trips per Day")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: format_number(x, 0)))
        fig.autofmt_xdate()
        ax.legend()
        sns.despine(ax=ax)
        st.pyplot(fig)
        plt.close(fig)

        df_filtered['weekday'] = df_filtered.index.weekday
        df_filtered['month'] = df_filtered.index.month

        weekday_means = df_filtered.groupby('weekday')['total_viajes_dia'].mean().reindex(range(7))
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        month_means = df_filtered.groupby('month')['total_viajes_dia'].mean()
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        colA, colB = st.columns(2)
        with colA:
            fig_d, ax_d = plt.subplots(figsize=(5, 3))
            ax_d.bar(day_names, weekday_means.values, color='#70ad47')
            ax_d.set_ylabel("Average daily trips")
            ax_d.set_title("By Day of Week")
            sns.despine(ax=ax_d)
            st.pyplot(fig_d)
            plt.close(fig_d)
        with colB:
            fig_m, ax_m = plt.subplots(figsize=(5, 3))
            ax_m.bar(month_names, month_means.values, color='#f4b183')
            ax_m.set_ylabel("Average daily trips")
            ax_m.set_title("By Month of Year")
            sns.despine(ax=ax_m)
            st.pyplot(fig_m)
            plt.close(fig_m)

    st.subheader("Stationarity Checks")
    if 'total_viajes_dia' in df_filtered.columns and len(df_filtered) > 0:
        ts = df_filtered['total_viajes_dia'].dropna()
        rolling_mean = ts.rolling(30).mean()
        rolling_std = ts.rolling(30).std()

        fig_mean, ax_mean = plt.subplots(figsize=(8, 4))
        ax_mean.plot(ts, label="Original", color='#377eb8')
        ax_mean.plot(rolling_mean, label="30-day Rolling Mean", color='#e41a1c')
        ax_mean.set_title("Rolling Mean vs Original")
        ax_mean.legend()
        st.pyplot(fig_mean)
        plt.close(fig_mean)

        fig_std, ax_std = plt.subplots(figsize=(8, 4))
        ax_std.plot(rolling_std, label="30-day Rolling Std", color='#ff7f00')
        ax_std.set_title("Rolling Standard Deviation")
        ax_std.legend()
        st.pyplot(fig_std)
        plt.close(fig_std)

        adf_stat, p_value, *_ = adfuller(ts, autolag='AIC')
        st.markdown(f"**ADF Statistic:** {format_number(adf_stat, 2)}  ")
        st.markdown(f"**p-value:** {format_number(p_value, 3)}")

    st.subheader("Covariance and Correlation")
    corr_vars = ["total_viajes_dia", "tavg", "prcp", "attendance"]
    corr_data = df_filtered[corr_vars].dropna()
    if not corr_data.empty:
        corr_matrix = corr_data.corr()
        cov_matrix = corr_data.cov()

        fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
        ax_corr.set_title("Correlation Matrix")

        fig_cov, ax_cov = plt.subplots(figsize=(5, 4))
        sns.heatmap(cov_matrix, annot=True, cmap="coolwarm", fmt=".0f", ax=ax_cov)
        ax_cov.set_title("Covariance Matrix")

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig_corr)
        with col2:
            st.pyplot(fig_cov)
        plt.close(fig_corr)
        plt.close(fig_cov)

    if "municipio_origen_name" in df.columns and "viajes" in df.columns:
        st.subheader("Trips by Origin Municipality")
        muni_trips = df.groupby("municipio_origen_name")['viajes'].sum().sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(10, 4))
        muni_trips.plot(kind='bar', ax=ax, color='#4682b4')
        ax.set_title("Top 15 Origin Municipalities by Total Trips")
        ax.set_ylabel("Total Trips")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: format_number(x, 0)))
        st.pyplot(fig)
        plt.close(fig)