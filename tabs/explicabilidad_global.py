import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

from utils.state_manager import StateManager
from utils.prediccion.model_loader import load_lgb_model, load_feature_cols
from utils.prediccion.shap_utils import get_shap_explainer, compute_global_shap


# ============================================================
# Global matplotlib dark theme
# ============================================================
plt.style.use("dark_background")


def render_explicabilidad_global():
    st.header("🌍 Global Explainability")

    global_state = StateManager("global")

    # Load df_model_training
    df_model = global_state.get("df_model_training")
    if df_model is None:
        st.error("❌ df_model_training not found in global state.")
        return

    if not pd.api.types.is_datetime64_any_dtype(df_model["date"]):
        df_model["date"] = pd.to_datetime(df_model["date"], errors="coerce")

    # Load model & features
    try:
        model = load_lgb_model()
        feature_cols = load_feature_cols()
    except Exception as e:
        st.error(f"Error loading model or feature columns: {e}")
        return

    df_sorted = df_model.sort_values("date")
    X_full = df_sorted[feature_cols].copy()

    # ============================================================
    # SHAP GLOBAL
    # ============================================================
    st.subheader("Feature Importance (SHAP Global)")

    with st.spinner("Computing global SHAP values..."):
        explainer = get_shap_explainer(model)

        sample_size = min(4000, len(X_full))
        X_sample = X_full.sample(sample_size, random_state=42)

        shap_values = compute_global_shap(explainer, X_sample)

    # ------------------------------------------------------------
    # PLOT 1 — Mean SHAP Importance
    # ------------------------------------------------------------
    st.markdown("### 🔝 Ranking by Mean |SHAP|")

    fig1 = plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    st.pyplot(fig1)
    plt.close(fig1)

    # ------------------------------------------------------------
    # PLOT 2 — Beeswarm
    # ------------------------------------------------------------
    st.markdown("### 🐝 SHAP Beeswarm Plot")

    fig2 = plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, plot_type="dot", show=False)
    st.pyplot(fig2)
    plt.close(fig2)

    st.markdown("""
### Interpretation
- **Higher SHAP values** push the prediction **upward** (more trips).
- **Lower SHAP values** push the prediction **downward** (fewer trips).
- The beeswarm plot lets you see distribution + direction of impact.
""")
