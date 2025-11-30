import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import os

# Rutas a tus datasets
RUTA_2023 = "data/processed/final_combined_with_events_2023.csv"
RUTA_2024 = "data/processed/final_combined_with_events_2024.csv"

# CSV ya concatenado 2023+2024 (ajusta el nombre si es distinto)
RUTA_CONCAT = "data/processed/final_combined_2023_2024.csv"

# Carpeta donde guardamos el modelo
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "od_model.joblib"
ORIGEN_PATH = MODELS_DIR / "le_origen.joblib"
DESTINO_PATH = MODELS_DIR / "le_destino.joblib"
META_PATH = MODELS_DIR / "metadata.json"


# -----------------------------------------------------
#   CARGA OPTIMIZADA (Agrupación Chunk-by-chunk)
# -----------------------------------------------------
def cargar_chunk_od(path_csv, chunksize=300_000):
    lista_chunks = []

    for chunk in pd.read_csv(
        path_csv,
        usecols=["day", "municipio_origen_name", "municipio_destino_name", "viajes"],
        parse_dates=["day"],
        chunksize=chunksize
    ):
        chunk = chunk.groupby(
            ["day", "municipio_origen_name", "municipio_destino_name"],
            as_index=False
        )["viajes"].sum()

        lista_chunks.append(chunk)

    df = pd.concat(lista_chunks, ignore_index=True)

    df = df.groupby(
        ["day", "municipio_origen_name", "municipio_destino_name"],
        as_index=False
    )["viajes"].sum()

    return df


def preparar_df_od():
    df23 = cargar_chunk_od(RUTA_2023)
    df24 = cargar_chunk_od(RUTA_2024)
    df = pd.concat([df23, df24], ignore_index=True)

    df = df.sort_values("day").reset_index(drop=True)

    df["day_of_week"] = df["day"].dt.weekday
    df["day_of_year"] = df["day"].dt.day_of_year
    df["month"] = df["day"].dt.month
    df["timestep"] = (df["day"] - df["day"].min()).dt.days

    return df

def cargar_df_concat():
    """
    Carga el CSV ya concatenado (2023+2024) y prepara el df
    con las mismas columnas que preparar_df_od().
    """
    if not os.path.exists(RUTA_CONCAT):
        # fallback: si no existe el CSV concatenado, usamos el flujo original
        return preparar_df_od()

    df = pd.read_csv(
        RUTA_CONCAT,
        usecols=["day", "municipio_origen_name", "municipio_destino_name", "viajes"],
        parse_dates=["day"],
    )

    # Nos aseguramos de que está agregado por día / OD
    df = df.groupby(
        ["day", "municipio_origen_name", "municipio_destino_name"],
        as_index=False
    )["viajes"].sum()

    df = df.sort_values("day").reset_index(drop=True)

    # Features temporales (igual que en preparar_df_od)
    df["day_of_week"] = df["day"].dt.weekday
    df["day_of_year"] = df["day"].dt.day_of_year
    df["month"] = df["day"].dt.month
    df["timestep"] = (df["day"] - df["day"].min()).dt.days

    return df



# -----------------------------------------------------
#   ENTRENAR MODELO Y GUARDARLO
# -----------------------------------------------------
def entrenar_y_guardar(df):
    le_origen = LabelEncoder()
    le_destino = LabelEncoder()

    df["origen_id"] = le_origen.fit_transform(df["municipio_origen_name"])
    df["destino_id"] = le_destino.fit_transform(df["municipio_destino_name"])

    X = df[[
        "day_of_week",
        "day_of_year",
        "month",
        "timestep",
        "origen_id",
        "destino_id"
    ]]

    y = df["viajes"]

    model = RandomForestRegressor(
        n_estimators=20,        # <<< reducido para que el modelo sea pequeño
        n_jobs=-1,
        random_state=42,
        max_depth=18,
        max_features="sqrt"
    )

    model.fit(X, y)

    # ---- GUARDAMOS TODO (con compresión) ----
    joblib.dump(model, MODEL_PATH, compress=3)
    joblib.dump(le_origen, ORIGEN_PATH, compress=3)
    joblib.dump(le_destino, DESTINO_PATH, compress=3)

    metadata = {"min_date": str(df["day"].min().date())}
    with open(META_PATH, "w") as f:
        json.dump(metadata, f)

    return model, le_origen, le_destino, df["day"].min()


# -----------------------------------------------------
#   CARGAR SI EXISTE — SI NO ENTRENAR
# -----------------------------------------------------
def cargar_o_entrenar():
    if MODEL_PATH.exists() and ORIGEN_PATH.exists() and DESTINO_PATH.exists() and META_PATH.exists():
        # ---- CARGA ----
        model = joblib.load(MODEL_PATH)
        le_origen = joblib.load(ORIGEN_PATH)
        le_destino = joblib.load(DESTINO_PATH)

        with open(META_PATH, "r") as f:
            meta = json.load(f)

        min_date = pd.to_datetime(meta["min_date"])

        # Para el histórico de la app leemos SOLO el CSV concatenado
        df = cargar_df_concat()

        return df, model, le_origen, le_destino, min_date

    else:
        # ---- ENTRENAR + GUARDAR ----
        df = preparar_df_od()
        model, le_origen, le_destino, min_date = entrenar_y_guardar(df)
        return df, model, le_origen, le_destino, min_date


# -----------------------------------------------------
#   PREDICCIÓN
# -----------------------------------------------------
def predecir_od(model, le_origen, le_destino, origen, destino, fecha, min_date):
    fecha = pd.to_datetime(fecha)

    origen_id = le_origen.transform([origen])[0]
    destino_id = le_destino.transform([destino])[0]

    row = {
        "day_of_week": fecha.weekday(),
        "day_of_year": fecha.day_of_year,
        "month": fecha.month,
        "timestep": (fecha - min_date).days,
        "origen_id": origen_id,
        "destino_id": destino_id,
    }

    return model.predict(pd.DataFrame([row]))[0]
