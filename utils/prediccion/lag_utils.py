import pandas as pd
import numpy as np


def _compute_daily_totals(df: pd.DataFrame) -> pd.Series:
    """
    Serie: date -> total_viajes_dia
    """
    return df.groupby("date")["viajes"].sum().sort_index()


def _compute_muni_origen_series(df: pd.DataFrame, municipio: str, origen: str) -> pd.Series:
    """
    Serie: date -> viajes para (municipio, origen)
    """
    mask = (df["municipio_origen_name"] == municipio) & (df["origen"] == origen)
    sub = df.loc[mask, ["date", "viajes"]].copy()
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby("date")["viajes"].sum().sort_index()


def compute_auto_lags(
    df: pd.DataFrame,
    municipio: str,
    origen: str,
    target_date: pd.Timestamp,
    max_lag: int = 7,
):
    """
    Calcula lags automáticos (si es posible) desde el dataset para:
    - total_viajes_dia_lag{k}
    - viajes_lag{k}

    Devuelve:
        auto_lags: dict[col_name -> float or np.nan]
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    daily_totals = _compute_daily_totals(df)
    muni_series = _compute_muni_origen_series(df, municipio, origen)

    auto_lags = {}

    for k in range(1, max_lag + 1):
        d = target_date - pd.Timedelta(days=k)

        # Global
        global_val = daily_totals.get(d, np.nan)
        auto_lags[f"total_viajes_dia_lag{k}"] = float(global_val) if pd.notna(global_val) else np.nan

        # Municipal
        muni_val = muni_series.get(d, np.nan)
        auto_lags[f"viajes_lag{k}"] = float(muni_val) if pd.notna(muni_val) else np.nan

    return auto_lags


def compute_fallback_means(
    df: pd.DataFrame,
    municipio: str,
    origen: str,
):
    """
    Medias históricas para usar como fallback cuando no hay datos
    para un cierto lag.
    """
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # media global de total_viajes_dia
    daily_totals = _compute_daily_totals(df)
    global_mean = float(daily_totals.mean()) if len(daily_totals) > 0 else 0.0

    # media por municipio+origen
    muni_series = _compute_muni_origen_series(df, municipio, origen)
    muni_mean = float(muni_series.mean()) if len(muni_series) > 0 else 0.0

    return {
        "global_mean": global_mean,
        "muni_mean": muni_mean,
    }


def build_lag_values(
    df: pd.DataFrame,
    municipio: str,
    origen: str,
    target_date: pd.Timestamp,
    manual_lags: dict | None = None,
    max_lag: int = 7,
):
    """
    Devuelve:
        lag_values: dict[col_name -> float]
        lag_sources: dict[col_name -> 'dataset' | 'manual' | 'fallback']
    """
    manual_lags = manual_lags or {}

    auto = compute_auto_lags(df, municipio, origen, target_date, max_lag=max_lag)
    means = compute_fallback_means(df, municipio, origen)

    lag_values = {}
    lag_sources = {}

    for k in range(1, max_lag + 1):
        # Global
        col_g = f"total_viajes_dia_lag{k}"
        # Municipal
        col_m = f"viajes_lag{k}"

        # GLOBAL
        if not np.isnan(auto[col_g]):
            lag_values[col_g] = auto[col_g]
            lag_sources[col_g] = "dataset"
        elif col_g in manual_lags:
            lag_values[col_g] = float(manual_lags[col_g])
            lag_sources[col_g] = "manual"
        else:
            lag_values[col_g] = means["global_mean"]
            lag_sources[col_g] = "fallback"

        # MUNICIPAL
        if not np.isnan(auto[col_m]):
            lag_values[col_m] = auto[col_m]
            lag_sources[col_m] = "dataset"
        elif col_m in manual_lags:
            lag_values[col_m] = float(manual_lags[col_m])
            lag_sources[col_m] = "manual"
        else:
            lag_values[col_m] = means["muni_mean"]
            lag_sources[col_m] = "fallback"

    return lag_values, lag_sources
