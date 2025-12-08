import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

from utils.state_manager import StateManager
from utils.prediccion.model_loader import load_lgb_model, load_feature_cols
from utils.prediccion.shap_utils import get_shap_explainer, compute_local_shap


# ============================================================
# Global dark theme
# ============================================================
plt.style.use("dark_background")


def render_explicabilidad_local():
    st.header("🔬 Local Explainability")

    global_state = StateManager("global")
    pred_state = StateManager("prediction")
    tab_state = StateManager("explain_local")

    tab_state.init({
        "mode": "Last prediction"
    })

    # Load dataset
    df_model = global_state.get("df_model_training")
    if df_model is None:
        st.error("❌ df_model_training missing in global state.")
        return

    df_sorted = df_model.sort_values("date")

    # Load model
    try:
        model = load_lgb_model()
        feature_cols = load_feature_cols()
    except Exception as e:
        st.error(f"Error loading model or feature columns: {e}")
        return

    # SHAP explainer
    explainer = get_shap_explainer(model)

    # ============================================================
    # MODE SELECTION
    # ============================================================
    mode = st.radio(
        "Select example to explain:",
        ["Last prediction", "Random test example"],
        horizontal=True
    )

    tab_state.set("mode", mode)

    # ============================================================
    # Build train-test split
    # ============================================================
    test_size = int(len(df_sorted) * 0.20)
    df_test = df_sorted.iloc[-test_size:]
    X_test = df_test[feature_cols].copy()

    x_row = None

    # ============================================================
    # MODE 1 — LAST PREDICTION
    # ============================================================
    if mode == "Last prediction":
        last_pred = pred_state.get("latest_prediction", None)

        if last_pred is None:
            st.warning("No stored prediction. Go to the Prediction tab first.")
            return

        st.info(
            f"📌 Using last prediction\n"
            f"- Date: **{last_pred['date']}**\n"
            f"- Municipality: **{last_pred['municipio']}**\n"
            f"- Origin: **{last_pred['origen']}**\n"
            f"- Predicted trips: **{last_pred['y_pred']:.0f}**"
        )

        feat_dict = last_pred["features"]
        x_row = pd.DataFrame([feat_dict])[feature_cols]

        # fix categorical
        if "municipio_origen_name" in x_row.columns:
            cats = df_model["municipio_origen_name"].astype("category").cat.categories
            x_row["municipio_origen_name"] = pd.Categorical(
                x_row["municipio_origen_name"],
                categories=cats
            )

    # ============================================================
    # MODE 2 — RANDOM TEST SAMPLE
    # ============================================================
    else:
        row = X_test.sample(1)
        idx = row.index[0]

        row_full = df_test.loc[idx]
        y_real = row_full["viajes"]
        y_pred = max(float(model.predict(row)[0]), 0.0)

        st.info(
            f"📌 Random test sample\n"
            f"- Date: **{row_full['date'].date()}**\n"
            f"- Municipality: **{row_full['municipio_origen_name']}**\n"
            f"- Origin: **{row_full['origen']}**\n"
            f"- Real trips: **{y_real:.0f}**\n"
            f"- Predicted trips: **{y_pred:.0f}**"
        )

        x_row = row

    # ============================================================
    # SHAP LOCAL PLOT
    # ============================================================
    with st.spinner("Computing local SHAP values..."):
        shap_local = compute_local_shap(explainer, x_row)

    st.markdown("### Input features")
    st.dataframe(x_row, use_container_width=True)

    st.markdown("### SHAP Waterfall Plot")

    values = shap_local[0]
    base = explainer.expected_value
    features = x_row.iloc[0]

    explanation = shap.Explanation(
        values=values,
        base_values=base,
        data=features.values,
        feature_names=list(features.index)
    )

    fig = plt.figure(figsize=(14, 7))
    shap.waterfall_plot(explanation, show=False)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
### Interpretation
- **Red bars** increase the predicted value.
- **Blue bars** decrease it.
- Left = base value (model average), right = final prediction.
""")
