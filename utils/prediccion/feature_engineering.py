import numpy as np
import pandas as pd

from .lag_utils import build_lag_values


def compute_time_features(date: pd.Timestamp) -> dict:
    """
    Desde una fecha -> month, dow, is_weekend, dow_sin, dow_cos, month_sin, month_cos
    """
    month = date.month
    dow = date.weekday()  # 0 lunes
    is_weekend = 1 if dow in (5, 6) else 0

    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)
    month_sin = np.sin(2 * np.pi * (month - 1) / 12)
    month_cos = np.cos(2 * np.pi * (month - 1) / 12)

    return {
        "month": month,
        "dow": dow,
        "is_weekend": is_weekend,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
    }


def build_event_features(
    categories_selected: list[str],
    attendance: float,
) -> dict:
    """
    Convierte selección de categorías de evento en one-hot + attendance.
    """
    cats_all = [
        "eventcat_city_festival",
        "eventcat_concert",
        "eventcat_festival",
        "eventcat_football",
        "eventcat_motorsport",
        "eventcat_other_sport",
        "eventcat_trade_fair",
    ]

    data = {c: 0 for c in cats_all}

    # categories_selected se espera que contenga los nombres EXACTOS de esas columnas
    for c in categories_selected:
        if c in data:
            data[c] = 1

    data["event_attendance"] = float(attendance) if attendance is not None else 0.0
    return data


def build_origen_onehot(origen_label: str) -> dict:
    """
    origen_label: 'Internacional', 'Nacional', 'Regional' o 'Residente'
    """
    mapping = {
        "Internacional": "origen_Internacional",
        "Nacional": "origen_Nacional",
        "Regional": "origen_Regional",
        "Residente": "origen_Residente",
    }

    cols = [
        "origen_Internacional",
        "origen_Nacional",
        "origen_Regional",
        "origen_Residente",
    ]

    out = {c: 0 for c in cols}
    col = mapping.get(origen_label)
    if col is not None:
        out[col] = 1

    return out


def build_feature_row(
    df_model: pd.DataFrame,
    feature_cols: list[str],
    municipio: str,
    origen_label: str,
    target_date: pd.Timestamp,
    tavg: float,
    tmin: float,
    tmax: float,
    prcp: float,
    event_categories: list[str],
    event_attendance: float,
    manual_lags: dict | None = None,
) -> pd.DataFrame:
    """
    Construye una fila de features con el MISMO esquema que feature_cols.
    Usa df_model para obtener lags desde el histórico cuando sea posible.
    """
    manual_lags = manual_lags or {}

    # Time features
    time_feats = compute_time_features(target_date)

    # Clima
    clima = {
        "tavg": float(tavg),
        "tmin": float(tmin),
        "tmax": float(tmax),
        "prcp": float(prcp),
    }

    # Eventos
    event_feats = build_event_features(event_categories, event_attendance)

    # Origen
    origen_feats = build_origen_onehot(origen_label)

    # Lags
    lag_values, lag_sources = build_lag_values(
        df=df_model,
        municipio=municipio,
        origen=origen_label,
        target_date=target_date,
        manual_lags=manual_lags,
        max_lag=7,
    )

    # Municipio origen (categorical)
    muni_feat = {"municipio_origen_name": municipio}

    # Unimos todo
    base_dict = {}
    base_dict.update(time_feats)
    base_dict.update(clima)
    base_dict.update(event_feats)
    base_dict.update(lag_values)
    base_dict.update(origen_feats)
    base_dict.update(muni_feat)

    # Creamos DataFrame y reordenamos columnas
    row = pd.DataFrame([base_dict])

    # Aseguramos que todas las columnas de feature_cols existan; si falta alguna, la rellenamos con 0
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0

    row = row[feature_cols]

    # municipio_origen_name como category, con las mismas categorías que df_model
    if "municipio_origen_name" in row.columns:
        cat = df_model["municipio_origen_name"].astype("category").cat.categories
        row["municipio_origen_name"] = pd.Categorical([municipio], categories=cat)

    return row, lag_values, lag_sources
