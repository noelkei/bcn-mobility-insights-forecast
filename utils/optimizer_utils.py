import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, Tuple, Optional


# ================================================================
# 1. PREPARACIÓN DE DATOS PARA OPTIMIZACIÓN (FOCO: UNA CIUDAD)
# ================================================================

@st.cache_data
def compute_optimizer_basics(
    df_raw: pd.DataFrame,
    focus_city: Optional[str] = "Barcelona",
) -> Tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepara todos los objetos base para la optimización, filtrando
    únicamente los enlaces donde aparece `focus_city` como origen o destino.

    Returns
    -------
    R_max : float
        Capacidad total diaria del sistema (máximo de los viajes diarios
        dentro del subgrafo filtrado por focus_city).
    df_share : DataFrame
        Recursos base por enlace OD según proporción histórica.
        Columnas: [origen, destino, share_median, share_norm, R_base].
    df_hist_od : DataFrame
        Histórico diario por enlace OD con features temporales.
        Columnas: [day, year, month, dow, day_of_month, origen, destino, demanda].
    df_daily_totals : DataFrame
        Totales diarios de demanda (sólo enlaces con focus_city).
        Columnas: [day, demanda_total].
    """
    if df_raw is None or df_raw.empty:
        return 0.0, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = df_raw.copy()
    df["day"] = pd.to_datetime(df["day"], errors="coerce")

    # Filtramos a enlaces donde aparece focus_city en origen o destino
    if focus_city is not None:
        mask = (
            (df["municipio_origen_name"] == focus_city)
            | (df["municipio_destino_name"] == focus_city)
        )
        df = df[mask]

    if df.empty:
        return 0.0, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Totales diarios de viajes dentro del subgrafo filtrado
    df_daily_totals = (
        df.groupby("day", as_index=False)["viajes"]
        .sum()
        .rename(columns={"viajes": "demanda_total"})
    )

    if df_daily_totals.empty:
        return 0.0, pd.DataFrame(), pd.DataFrame(), df_daily_totals

    # Capacidad total diaria = máximo de los viajes diarios
    R_max = float(df_daily_totals["demanda_total"].max())

    # ------------------------------------------------------------------
    # Histórico OD diario con proporciones por día
    # ------------------------------------------------------------------
    df_od_daily = (
        df.groupby(
            ["day", "municipio_origen_name", "municipio_destino_name"],
            as_index=False,
        )["viajes"]
        .sum()
        .rename(
            columns={
                "municipio_origen_name": "origen",
                "municipio_destino_name": "destino",
                "viajes": "demanda",
            }
        )
    )

    # Proporción de cada OD dentro del día
    df_od_daily["total_day"] = df_od_daily.groupby("day")["demanda"].transform("sum")
    df_od_daily["p_od"] = df_od_daily["demanda"] / df_od_daily["total_day"]

    # Share histórico por enlace OD: mediana de la proporción diaria
    df_share = (
        df_od_daily.groupby(["origen", "destino"], as_index=False)["p_od"]
        .median()
        .rename(columns={"p_od": "share_median"})
    )

    sum_shares = float(df_share["share_median"].sum())
    if sum_shares > 0.0:
        df_share["share_norm"] = df_share["share_median"] / sum_shares
    else:
        df_share["share_norm"] = 0.0

    # Recursos base cuando el sistema está al máximo
    df_share["R_base"] = df_share["share_norm"] * R_max

    # Quitamos enlaces que nunca reciben recursos (R_base == 0)
    df_share = df_share[df_share["R_base"] > 0].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Histórico OD con features temporales para estimar demanda
    # ------------------------------------------------------------------
    df_hist_od = df_od_daily[["day", "origen", "destino", "demanda"]].copy()
    df_hist_od["day"] = pd.to_datetime(df_hist_od["day"], errors="coerce")
    df_hist_od["year"] = df_hist_od["day"].dt.year
    df_hist_od["month"] = df_hist_od["day"].dt.month
    df_hist_od["day_of_month"] = df_hist_od["day"].dt.day
    df_hist_od["dow"] = df_hist_od["day"].dt.weekday  # 0=lunes,...,6=domingo

    return R_max, df_share, df_hist_od, df_daily_totals


# ================================================================
# 2. ESTIMACIÓN DE DEMANDA (MEDIA PONDERADA)
# ================================================================

_DEFAULT_WEIGHTS = {
    "same_day_prev": 0.4,
    "same_dow_month": 0.3,
    "same_dow": 0.2,
    "all_history": 0.1,
}


def _normalize_weights(weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    """
    Normaliza pesos a suma 1, ignorando valores <= 0.
    Si todos son <= 0, devuelve pesos por defecto.
    """
    if weights is None:
        return _DEFAULT_WEIGHTS.copy()

    pos = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(pos.values())
    if total <= 0:
        return _DEFAULT_WEIGHTS.copy()

    return {k: v / total for k, v in pos.items()}


def _weighted_median_estimate(
    values: pd.Series,
    weight: float,
) -> Optional[Tuple[float, float]]:
    """
    Helper: devuelve (mediana, peso) si hay datos y el peso > 0, si no None.
    """
    if values.empty or weight <= 0:
        return None
    return float(values.median()), float(weight)


def estimate_demand_for_od(
    df_hist_od: pd.DataFrame,
    origen: str,
    destino: str,
    scenario_date: pd.Timestamp,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Estima la demanda (viajes) para un enlace origen-destino concreto en una fecha
    usando una combinación ponderada de medianas históricas.
    """
    if df_hist_od.empty:
        return 0.0

    df_od = df_hist_od[
        (df_hist_od["origen"] == origen) & (df_hist_od["destino"] == destino)
    ]
    if df_od.empty:
        return 0.0

    d = pd.to_datetime(scenario_date)
    year = d.year
    month = d.month
    dom = d.day
    dow = d.weekday()

    w = _normalize_weights(weights or _DEFAULT_WEIGHTS)

    candidates = []

    # 1) Mismo día y mes en años anteriores
    prev_same_day = df_od[
        (df_od["month"] == month)
        & (df_od["day_of_month"] == dom)
        & (df_od["year"] < year)
    ]
    c = _weighted_median_estimate(prev_same_day["demanda"], w["same_day_prev"])
    if c is not None:
        candidates.append(c)

    # 2) Mismo día de la semana y mes
    same_dow_month = df_od[
        (df_od["dow"] == dow) & (df_od["month"] == month)
    ]
    c = _weighted_median_estimate(same_dow_month["demanda"], w["same_dow_month"])
    if c is not None:
        candidates.append(c)

    # 3) Mismo día de la semana (cualquier mes)
    same_dow = df_od[df_od["dow"] == dow]
    c = _weighted_median_estimate(same_dow["demanda"], w["same_dow"])
    if c is not None:
        candidates.append(c)

    # 4) Todo el histórico
    c = _weighted_median_estimate(df_od["demanda"], w["all_history"])
    if c is not None:
        candidates.append(c)

    if not candidates:
        return 0.0

    numer = sum(m * w_ for (m, w_) in candidates)
    denom = sum(w_ for (_, w_) in candidates)
    return float(numer / denom) if denom > 0 else 0.0


# ================================================================
# 3. ESCENARIOS (HISTÓRICO / FUTURO GENÉRICO)
# ================================================================

@st.cache_data
def build_historical_scenario(
    df_hist_od: pd.DataFrame,
    scenario_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Construye la matriz OD para una fecha histórica concreta.

    Devuelve DataFrame con columnas:
        [origen, destino, demanda]
    """
    if df_hist_od.empty:
        return pd.DataFrame(columns=["origen", "destino", "demanda"])

    d = pd.to_datetime(scenario_date).normalize()
    df_day = df_hist_od[df_hist_od["day"] == d]

    if df_day.empty:
        return pd.DataFrame(columns=["origen", "destino", "demanda"])

    return df_day[["origen", "destino", "demanda"]].copy()


@st.cache_data
def build_future_scenario(
    df_hist_od: pd.DataFrame,
    df_share: pd.DataFrame,
    scenario_date: pd.Timestamp,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Construye la matriz OD (demanda por enlace) para una fecha futura:

    - Usa todos los enlaces que existen en df_share (ya filtrado por focus_city).
    - Estima la demanda para cada enlace con estimate_demand_for_od().

    Devuelve DataFrame con columnas:
        [origen, destino, demanda]
    """
    if df_share.empty:
        return pd.DataFrame(columns=["origen", "destino", "demanda"])

    d = pd.to_datetime(scenario_date).normalize()

    df_links = df_share[["origen", "destino"]].drop_duplicates().copy()
    demandas = []

    for row in df_links.itertuples(index=False):
        est = estimate_demand_for_od(
            df_hist_od=df_hist_od,
            origen=row.origen,
            destino=row.destino,
            scenario_date=d,
            weights=weights,
        )
        demandas.append(est)

    df_links["demanda"] = demandas
    return df_links


# ================================================================
# 4. TABLA df_links PARA UNA FECHA (SÓLO ENLACES CON focus_city)
# ================================================================

def build_links_table_for_date(
    df_hist_od: pd.DataFrame,
    df_share: pd.DataFrame,
    scenario_ts: pd.Timestamp,
    focus_city: Optional[str] = "Barcelona",
) -> Tuple[pd.DataFrame, bool]:
    """
    Construye la tabla df_links para una fecha concreta, PERO
    solo con los enlaces donde aparece focus_city como origen o destino.

    - Siempre devuelve los mismos enlaces (mismo orden) para todas las fechas:
      todos los OD del histórico que incluyan a focus_city (según df_share).
    - Si el día tiene datos históricos:
        demanda = viajes reales de ese día (o 0 si ese enlace no aparece ese día).
        is_historical = True
    - Si el día NO tiene datos:
        demanda = estimación por media ponderada (estimate_demand_for_od).
        is_historical = False
    """
    if df_share.empty:
        return pd.DataFrame(columns=["origen", "destino", "demanda"]), False

    # 1) Lista de todos los enlaces del dataset que contienen a focus_city
    all_links = df_share[["origen", "destino", "R_base"]].drop_duplicates()

    if focus_city is not None:
        mask = (all_links["origen"] == focus_city) | (all_links["destino"] == focus_city)
        all_links = all_links[mask]

    # ya vienen sin R_base==0 gracias a compute_optimizer_basics
    all_links = (
        all_links[all_links["R_base"] > 0]
        .sort_values(["origen", "destino"])
        .reset_index(drop=True)
    )

    if all_links.empty:
        return pd.DataFrame(columns=["origen", "destino", "demanda"]), False

    # 2) Histórico solo para ese día
    d = pd.to_datetime(scenario_ts).normalize()
    df_day = df_hist_od[df_hist_od["day"] == d]

    if focus_city is not None:
        df_day = df_day[
            (df_day["origen"] == focus_city) | (df_day["destino"] == focus_city)
        ]

    has_hist = not df_day.empty

    if has_hist:
        df_day_links = df_day[["origen", "destino", "demanda"]]
        df_links = all_links.merge(
            df_day_links,
            on=["origen", "destino"],
            how="left",
        )
        df_links["demanda"] = df_links["demanda"].fillna(0.0)
    else:
        demandas = []
        for row in all_links.itertuples(index=False):
            est = estimate_demand_for_od(
                df_hist_od=df_hist_od,
                origen=row.origen,
                destino=row.destino,
                scenario_date=d,
                weights=None,
            )
            demandas.append(est)

        df_links = all_links[["origen", "destino"]].copy()
        df_links["demanda"] = demandas

    return df_links, has_hist


# ================================================================
# 5. OPTIMIZACIÓN DE RECURSOS ENTRE ENLACES
# ================================================================

def optimize_resources_for_scenario(
    df_links: pd.DataFrame,
    df_share: pd.DataFrame,
    R_max: float,
    allow_deep_reallocation: bool = False,
    donor_temp_limit: float = 1.1,
) -> pd.DataFrame:
    """
    Reasigna recursos entre enlaces OD manteniendo la capacidad global R_max.

    Devuelve df con columnas:
        [origen, destino, demanda, R_base, R_after_slack, R_opt,
         temp_before, temp_after, overload_before, overload_after]
    """
    if df_links.empty or df_share.empty or R_max <= 0:
        return pd.DataFrame()

    df = df_links.merge(
        df_share[["origen", "destino", "R_base"]],
        on=["origen", "destino"],
        how="left",
    ).copy()

    df["R_base"] = df["R_base"].fillna(0.0)

    # Filtramos enlaces sin recursos base (no hay nada que optimizar)
    df = df[df["R_base"] > 0].copy()
    if df.empty:
        return df

    D = df["demanda"].to_numpy(dtype=float)
    R_base = df["R_base"].to_numpy(dtype=float)
    eps = 1e-6

    # Temperatura con recursos base
    temp_base = np.where(R_base > 0, D / (R_base + eps), np.inf)

    # ------------------------------------------------------------
    # Paso 1: usar SOLO slack seguro (R_base - D >= 0) sin calentar
    # ------------------------------------------------------------
    donation_safe = np.maximum(R_base - D, 0.0)
    slack_pool = float(donation_safe.sum())

    R_after_slack = R_base - donation_safe

    shortage1 = np.maximum(D - R_after_slack, 0.0)
    total_shortage1 = float(shortage1.sum())

    if slack_pool > 0 and total_shortage1 > 0:
        if slack_pool >= total_shortage1:
            extra1 = shortage1
            used_slack = total_shortage1
        else:
            factor = slack_pool / total_shortage1
            extra1 = shortage1 * factor
            used_slack = slack_pool

        R_after_slack = R_after_slack + extra1
        slack_pool -= used_slack

    # ------------------------------------------------------------
    # Paso 2 (opcional): reasignación profunda desde enlaces fríos
    # ------------------------------------------------------------
    R_opt = R_after_slack.copy()

    if allow_deep_reallocation:
        shortage2 = np.maximum(D - R_opt, 0.0)
        total_shortage2 = float(shortage2.sum())

        if total_shortage2 > 0:
            donor_min_capacity = np.where(
                D > 0, D / max(donor_temp_limit, 1.0), 0.0
            )
            donor_max_donation = np.maximum(R_opt - donor_min_capacity, 0.0)
            total_donor = float(donor_max_donation.sum())

            if total_donor > 0:
                moved = min(total_donor, total_shortage2)

                recv_weights = np.where(shortage2 > 0, shortage2 / total_shortage2, 0.0)
                extra2 = recv_weights * moved

                donor_weights = np.where(
                    donor_max_donation > 0, donor_max_donation / total_donor, 0.0
                )
                reduction2 = donor_weights * moved

                R_opt = R_opt + extra2 - reduction2
                R_opt = np.clip(R_opt, 0.0, None)

    # ------------------------------------------------------------
    # Cálculo de temperaturas y sobrecarga
    # ------------------------------------------------------------
    temp_after = np.where(R_opt > 0, D / (R_opt + eps), np.inf)
    overload_before = np.maximum(temp_base - 1.0, 0.0)
    overload_after = np.maximum(temp_after - 1.0, 0.0)

    df["R_after_slack"] = R_after_slack
    df["R_opt"] = R_opt
    df["temp_before"] = temp_base
    df["temp_after"] = temp_after
    df["overload_before"] = overload_before
    df["overload_after"] = overload_after

    return df


# ================================================================
# 6. RESÚMENES AUXILIARES
# ================================================================

def summarize_optimization(
    df_opt: pd.DataFrame,
    R_max: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calcula indicadores globales de la optimización, incluyendo:
    - uso efectivo de recursos (antes/después) respecto a R_max
    - slack global (antes/después) = R_max - recursos necesarios
    - índice de calor (da más peso a enlaces muy calientes)
    """
    if df_opt is None or df_opt.empty:
        return {
            "total_demand": 0.0,
            "total_R_base": 0.0,
            "total_R_opt": 0.0,
            "avg_temp_before": 0.0,
            "avg_temp_after": 0.0,
            "total_overload_before": 0.0,
            "total_overload_after": 0.0,
            "num_hot_before": 0,
            "num_hot_after": 0,
            "slack_before": 0.0,
            "slack_after": 0.0,
            "slack_before_pct": 0.0,
            "slack_after_pct": 0.0,
            "effective_resources_before": 0.0,
            "effective_resources_after": 0.0,
            "effective_before_pct": 0.0,
            "effective_after_pct": 0.0,
            "heat_index_before": 0.0,
            "heat_index_after": 0.0,
        }

    df = df_opt.copy()

    D = df["demanda"].to_numpy(dtype=float)
    R_base = df["R_base"].to_numpy(dtype=float)
    R_opt = df["R_opt"].to_numpy(dtype=float)

    eps = 1e-6

    total_demand = float(D.sum())
    total_R_base = float(R_base.sum())
    total_R_opt = float(R_opt.sum())

    # Temperaturas (evitamos infinitos en el promedio)
    temp_before = np.where(R_base > 0, D / (R_base + eps), np.nan)
    temp_after = np.where(R_opt > 0, D / (R_opt + eps), np.nan)

    finite_before = temp_before[np.isfinite(temp_before)]
    finite_after = temp_after[np.isfinite(temp_after)]

    avg_temp_before = float(finite_before.mean()) if finite_before.size > 0 else 0.0
    avg_temp_after = float(finite_after.mean()) if finite_after.size > 0 else 0.0

    # Enlaces calientes (demanda > recursos)
    num_hot_before = int((D > R_base + eps).sum())
    num_hot_after = int((D > R_opt + eps).sum())

    # Sobrecarga relativa (temp - 1): cuánto "por encima" del equilibrio está cada enlace
    overload_before = np.maximum(temp_before - 1.0, 0.0)
    overload_after = np.maximum(temp_after - 1.0, 0.0)

    total_overload_before = float(
        np.nan_to_num(overload_before, nan=0.0, posinf=0.0, neginf=0.0).sum()
    )
    total_overload_after = float(
        np.nan_to_num(overload_after, nan=0.0, posinf=0.0, neginf=0.0).sum()
    )

    # Uso efectivo de recursos = min(demanda, recursos)
    effective_before = float(np.minimum(D, R_base).sum())
    effective_after = float(np.minimum(D, R_opt).sum())

    # Base para porcentajes (R_max del sistema, o algo razonable si no viene)
    if R_max is None or R_max <= 0:
        R_max_eff = max(effective_before, effective_after, 1.0)
    else:
        R_max_eff = float(R_max)

    # Slack global = capacidad teórica - recursos realmente necesarios
    slack_before = max(R_max_eff - effective_before, 0.0)
    slack_after = max(R_max_eff - effective_after, 0.0)

    slack_before_pct = 100.0 * slack_before / R_max_eff
    slack_after_pct = 100.0 * slack_after / R_max_eff

    effective_before_pct = 100.0 * effective_before / R_max_eff
    effective_after_pct = 100.0 * effective_after / R_max_eff

    # Índice de calor: media de (overload^2) sobre enlaces calientes,
    # recortando sobrecargas extremas para evitar infinitos
    max_overload = 5.0  # por encima de +500% ya lo consideramos "muy caliente"

    ob = overload_before[overload_before > 0]
    ob = ob[np.isfinite(ob)]
    ob = np.clip(ob, 0.0, max_overload)
    if ob.size > 0:
        heat_index_before = float((ob ** 2).mean())
    else:
        heat_index_before = 0.0

    oa = overload_after[overload_after > 0]
    oa = oa[np.isfinite(oa)]
    oa = np.clip(oa, 0.0, max_overload)
    if oa.size > 0:
        heat_index_after = float((oa ** 2).mean())
    else:
        heat_index_after = 0.0

    return {
        "total_demand": total_demand,
        "total_R_base": total_R_base,
        "total_R_opt": total_R_opt,
        "avg_temp_before": avg_temp_before,
        "avg_temp_after": avg_temp_after,
        "total_overload_before": total_overload_before,
        "total_overload_after": total_overload_after,
        "num_hot_before": num_hot_before,
        "num_hot_after": num_hot_after,
        "slack_before": slack_before,
        "slack_after": slack_after,
        "slack_before_pct": slack_before_pct,
        "slack_after_pct": slack_after_pct,
        "effective_resources_before": effective_before,
        "effective_resources_after": effective_after,
        "effective_before_pct": effective_before_pct,
        "effective_after_pct": effective_after_pct,
        "heat_index_before": heat_index_before,
        "heat_index_after": heat_index_after,
    }

    """
    Calcula indicadores globales de la optimización, incluyendo:
    - uso efectivo de recursos (antes/después)
    - slack global (antes/después)
    - índice de calor (da más peso a enlaces muy calientes)
    """
    if df_opt is None or df_opt.empty:
        return {
            "total_demand": 0.0,
            "total_R_base": 0.0,
            "total_R_opt": 0.0,
            "avg_temp_before": 0.0,
            "avg_temp_after": 0.0,
            "total_overload_before": 0.0,
            "total_overload_after": 0.0,
            "num_hot_before": 0,
            "num_hot_after": 0,
            "slack_before": 0.0,
            "slack_after": 0.0,
            "slack_before_pct": 0.0,
            "slack_after_pct": 0.0,
            "effective_resources_before": 0.0,
            "effective_resources_after": 0.0,
            "effective_before_pct": 0.0,
            "effective_after_pct": 0.0,
            "heat_index_before": 0.0,
            "heat_index_after": 0.0,
        }

    total_demand = float(df_opt["demanda"].sum())
    total_R_base = float(df_opt["R_base"].sum())
    total_R_opt = float(df_opt["R_opt"].sum())

    finite_before = df_opt[np.isfinite(df_opt["temp_before"])]
    finite_after = df_opt[np.isfinite(df_opt["temp_after"])]

    avg_temp_before = float(finite_before["temp_before"].mean()) if not finite_before.empty else 0.0
    avg_temp_after = float(finite_after["temp_after"].mean()) if not finite_after.empty else 0.0

    total_overload_before = float(df_opt["overload_before"].sum())
    total_overload_after = float(df_opt["overload_after"].sum())

    num_hot_before = int((df_opt["temp_before"] > 1.0).sum())
    num_hot_after = int((df_opt["temp_after"] > 1.0).sum())

    # Slack global: recursos ociosos (R - demanda, si R > demanda)
    slack_before = float(np.maximum(df_opt["R_base"] - df_opt["demanda"], 0.0).sum())
    slack_after = float(np.maximum(df_opt["R_opt"] - df_opt["demanda"], 0.0).sum())

    # R_max efectivo para porcentajes
    if R_max is None or R_max <= 0:
        R_max_eff = total_R_base if total_R_base > 0 else 1.0
    else:
        R_max_eff = float(R_max)

    slack_before_pct = 100.0 * slack_before / R_max_eff
    slack_after_pct = 100.0 * slack_after / R_max_eff

    effective_before = total_R_base - slack_before
    effective_after = total_R_opt - slack_after

    effective_before_pct = 100.0 * effective_before / R_max_eff
    effective_after_pct = 100.0 * effective_after / R_max_eff

    # Índice de calor: media de overload^2 sobre enlaces calientes
    hot_before = df_opt[df_opt["overload_before"] > 0]
    if not hot_before.empty:
        heat_index_before = float((hot_before["overload_before"] ** 2).mean())
    else:
        heat_index_before = 0.0

    hot_after = df_opt[df_opt["overload_after"] > 0]
    if not hot_after.empty:
        heat_index_after = float((hot_after["overload_after"] ** 2).mean())
    else:
        heat_index_after = 0.0

    return {
        "total_demand": total_demand,
        "total_R_base": total_R_base,
        "total_R_opt": total_R_opt,
        "avg_temp_before": avg_temp_before,
        "avg_temp_after": avg_temp_after,
        "total_overload_before": total_overload_before,
        "total_overload_after": total_overload_after,
        "num_hot_before": num_hot_before,
        "num_hot_after": num_hot_after,
        "slack_before": slack_before,
        "slack_after": slack_after,
        "slack_before_pct": slack_before_pct,
        "slack_after_pct": slack_after_pct,
        "effective_resources_before": effective_before,
        "effective_resources_after": effective_after,
        "effective_before_pct": effective_before_pct,
        "effective_after_pct": effective_after_pct,
        "heat_index_before": heat_index_before,
        "heat_index_after": heat_index_after,
    }
