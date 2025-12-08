import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap

from utils.state_manager import StateManager
from utils.prediccion.model_loader import load_lgb_model, load_feature_cols
from utils.prediccion.shap_utils import get_shap_explainer, compute_global_shap, compute_local_shap


def render_explicabilidad_modelo():
    st.header("🧠 Explicabilidad del Modelo de Predicción")

    global_state = StateManager("global")
    tab_state = StateManager("explicabilidad")

    tab_state.init({
        "local_mode": "Última predicción",
    })

    # --- Cargar df_model_training ---
    df_model = global_state.get("df_model_training")
    if df_model is None:
        st.error("❌ No se ha encontrado df_model_training en el StateManager global.")
        st.info("Asegúrate de que old_main.py cargue 'data/processed/df_model_training.csv' en StateManager('global').")
        return

    if not pd.api.types.is_datetime64_any_dtype(df_model["date"]):
        df_model["date"] = pd.to_datetime(df_model["date"], errors="coerce")

    # --- Cargar modelo + feature_cols ---
    try:
        model = load_lgb_model()
        feature_cols = load_feature_cols()
    except Exception as e:
        st.error(f"Error cargando el modelo o feature_cols: {e}")
        return

    # Aseguramos que municipio_origen_name es category
    if "municipio_origen_name" in df_model.columns:
        if df_model["municipio_origen_name"].dtype != "category":
            df_model["municipio_origen_name"] = df_model["municipio_origen_name"].astype("category")

    # Dataset de features completo
    df_sorted = df_model.sort_values("date")
    X_full = df_sorted[feature_cols].copy()

    # ============================================================
    # 1. SHAP GLOBAL
    # ============================================================
    st.subheader("🌍 SHAP Global — Importancia general de las variables")

    with st.spinner("Calculando SHAP global (muestra de hasta 4000 filas)..."):
        explainer = get_shap_explainer(model)

        sample_size = min(4000, len(X_full))
        X_sample = X_full.sample(sample_size, random_state=42) if sample_size < len(X_full) else X_full

        shap_values = compute_global_shap(explainer, X_sample)

    col_global1, col_global2 = st.columns(2)

    with col_global1:
        st.markdown("#### Ranking de importancia (media |SHAP|)")
        fig1 = plt.figure(figsize=(6, 6))
        shap.summary_plot(
            shap_values,
            X_sample,
            plot_type="bar",
            show=False,
        )
        st.pyplot(fig1)
        plt.close(fig1)

    with col_global2:
        st.markdown("#### SHAP Beeswarm (distribución por feature)")
        fig2 = plt.figure(figsize=(6, 6))
        shap.summary_plot(
            shap_values,
            X_sample,
            plot_type="dot",
            show=False,
        )
        st.pyplot(fig2)
        plt.close(fig2)

    st.markdown(
        """
        - Cada barra representa la contribución media absoluta de una variable a la predicción (MAF de SHAP).  
        - En el beeswarm, cada punto es una fila del dataset:
            - Color = valor de la variable (bajo → azul, alto → rojo).  
            - Eje X = impacto en la predicción (SHAP value).
        """
    )

    # ============================================================
    # 2. SHAP LOCAL
    # ============================================================
    st.subheader("🔍 SHAP Local — Explicación de una predicción concreta")

    local_mode = st.radio(
        "Selecciona el tipo de ejemplo a explicar:",
        ["Última predicción", "Ejemplo aleatorio (test)"],
        index=0 if tab_state.get("local_mode") == "Última predicción" else 1,
        horizontal=True,
    )
    tab_state.set("local_mode", local_mode)

    # --- Construimos train/test split igual que en entrenamiento ---
    test_size = int(len(df_sorted) * 0.20)
    df_train = df_sorted.iloc[:-test_size]
    df_test = df_sorted.iloc[-test_size:]

    X_test = df_test[feature_cols].copy()

    x_to_explain = None
    meta_info = {}

    if local_mode == "Última predicción":
        # Recuperar de pestaña prediction
        pred_state = StateManager("prediction")
        latest = pred_state.get("latest_prediction", None)

        if latest is None:
            st.warning(
                "No hay ninguna 'última predicción' guardada.\n\n"
                "Ve primero a la pestaña 🔮 Predicción, configura un escenario y lanza una predicción."
            )
            return

        st.info(
            f"Usando la última predicción guardada:\n\n"
            f"- Fecha: **{latest['date']}**\n"
            f"- Municipio: **{latest['municipio']}**\n"
            f"- Origen: **{latest['origen']}**\n"
            f"- Viajes predichos: **{latest['y_pred']:.0f}**"
            + (f"\n- Viajes reales: **{latest['y_real']:.0f}**" if latest["y_real"] is not None else "")
        )

        # reconstruimos DataFrame de una fila a partir de latest["features"]
        feat_dict = latest["features"]
        x_to_explain = pd.DataFrame([feat_dict])[feature_cols]

        # aseguramos categoria municipio
        if "municipio_origen_name" in x_to_explain.columns:
            cats = df_model["municipio_origen_name"].astype("category").cat.categories
            x_to_explain["municipio_origen_name"] = pd.Categorical(
                x_to_explain["municipio_origen_name"],
                categories=cats,
            )

        meta_info = latest

    else:  # Ejemplo aleatorio (test)
        if X_test.empty:
            st.error("El conjunto de test está vacío; no se puede muestrear un ejemplo aleatorio.")
            return

        x_to_explain = X_test.sample(1, random_state=123)
        idx = x_to_explain.index[0]

        row_full = df_test.loc[[idx]]
        municipio = row_full["municipio_origen_name"].iloc[0]
        origen = row_full["origen"].iloc[0]
        date = row_full["date"].iloc[0]
        y_true = row_full["viajes"].iloc[0]
        y_pred_raw = float(model.predict(x_to_explain)[0])
        y_pred = max(y_pred_raw, 0.0)

        st.info(
            f"Ejemplo aleatorio tomado del conjunto de TEST:\n\n"
            f"- Fecha: **{date.date()}**\n"
            f"- Municipio: **{municipio}**\n"
            f"- Origen: **{origen}**\n"
            f"- Viajes reales: **{y_true:.0f}**\n"
            f"- Viajes predichos (clip 0): **{y_pred:.0f}**"
        )

        meta_info = {
            "date": str(date.date()),
            "municipio": municipio,
            "origen": origen,
            "y_real": float(y_true),
            "y_pred": y_pred,
        }

    if x_to_explain is None:
        st.error("No se pudo construir un ejemplo para explicar.")
        return

    # --- SHAP local ---
    with st.spinner("Calculando SHAP local para la observación seleccionada..."):
        shap_local = compute_local_shap(explainer, x_to_explain)

    st.markdown("#### Inputs de la observación")
    st.dataframe(x_to_explain, use_container_width=True)

    st.markdown("#### Descomposición SHAP de la predicción")

    # shap.Local values -> construimos Explanation para waterfall
    # shap_local tiene shape (1, n_features)
    values = shap_local[0]
    base_value = explainer.expected_value
    features = x_to_explain.iloc[0]

    # Construimos objeto Explanation
    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=features.values,
        feature_names=list(features.index),
    )

    fig = plt.figure(figsize=(8, 6))
    shap.waterfall_plot(explanation, show=False)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        """
        - Las barras rojas empujan la predicción **hacia arriba** (más viajes).  
        - Las barras azules empujan la predicción **hacia abajo** (menos viajes).  
        - La línea de la izquierda es el valor base (media del modelo), y la de la derecha la predicción final.
        """
    )
