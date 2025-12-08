import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np  # <-- FIX

from utils.state_manager import StateManager
from utils.prediccion.model_loader import load_lgb_model, load_feature_cols
from utils.prediccion.shap_utils import get_shap_explainer, compute_global_shap


# ============================================================
# Custom DARK Theme (NOT pure black)
# ============================================================

DARK_BG = "#1A1A1A"         # softer dark grey
DARKER_BG = "#111111"       # for figure background
LIGHT_TEXT = "#EEEEEE"      # near-white for labels
MID_TEXT = "#CCCCCC"        # medium contrast for axis

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": DARK_BG,
    "savefig.facecolor": DARK_BG,
    "axes.labelcolor": LIGHT_TEXT,
    "xtick.color": MID_TEXT,
    "ytick.color": MID_TEXT,
    "text.color": LIGHT_TEXT,
})


# ============================================================
# Fix SHAP plots to be readable in dark mode
# ============================================================

def fix_shap_plot_dark(ax):
    """
    Adjusts axis labels, ticks, spines and text color AFTER SHAP draws.
    """
    if ax is None:
        return

    ax.set_facecolor(DARK_BG)

    ax.tick_params(colors=MID_TEXT, labelcolor=MID_TEXT)

    if ax.xaxis.label:
        ax.xaxis.label.set_color(LIGHT_TEXT)
    if ax.yaxis.label:
        ax.yaxis.label.set_color(LIGHT_TEXT)

    if ax.title:
        ax.title.set_color(LIGHT_TEXT)

    for spine in ax.spines.values():
        spine.set_color(MID_TEXT)

    for txt in ax.get_yticklabels():
        txt.set_color(LIGHT_TEXT)
        txt.set_fontsize(11)

    for txt in ax.get_xticklabels():
        txt.set_color(MID_TEXT)
        txt.set_fontsize(10)


# ============================================================
# MAIN RENDER FUNCTION
# ============================================================

def render_explicabilidad_global():
    st.header("🌍 Global Explainability")

    global_state = StateManager("global")

    df_model = global_state.get("df_model_training")
    if df_model is None:
        st.error("❌ df_model_training not found in global state.")
        return

    if not pd.api.types.is_datetime64_any_dtype(df_model["date"]):
        df_model["date"] = pd.to_datetime(df_model["date"], errors="coerce")

    try:
        model = load_lgb_model()
        feature_cols = load_feature_cols()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    df_sorted = df_model.sort_values("date")
    X_full = df_sorted[feature_cols].copy()

    # ============================================================
    # SHAP GLOBAL
    # ============================================================
    st.subheader("Feature Importance (SHAP Global)")

    with st.spinner("Computing global SHAP values…"):
        explainer = get_shap_explainer(model)
        sample_size = min(4000, len(X_full))
        X_sample = X_full.sample(sample_size, random_state=42)
        shap_values = compute_global_shap(explainer, X_sample)

    # ------------------------------------------------------------
    # PLOT 1 — Mean |SHAP|
    # ------------------------------------------------------------
    st.markdown("### 🔝 Ranking by Mean |SHAP|")

    fig1 = plt.figure(figsize=(7, 6), facecolor=DARK_BG)
    ax1 = plt.gca()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    fix_shap_plot_dark(ax1)
    st.pyplot(fig1, use_container_width=False)
    plt.close(fig1)

    # ------------------------------------------------------------
    # PLOT 2 — Beeswarm
    # ------------------------------------------------------------
    st.markdown("### 🐝 SHAP Beeswarm Plot")

    fig2 = plt.figure(figsize=(7, 6), facecolor=DARK_BG)
    ax2 = plt.gca()
    shap.summary_plot(shap_values, X_sample, plot_type="dot", show=False)
    fix_shap_plot_dark(ax2)
    st.pyplot(fig2, use_container_width=False)
    plt.close(fig2)

    # ------------------------------------------------------------
    # AUTO-GENERATED ANALYSIS (FIXED np.abs)
    # ------------------------------------------------------------
    st.markdown("## 📘 Automated Interpretation")

    mean_abs_shap = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": np.mean(np.abs(shap_values), axis=0)   # <-- FIXED
    }).sort_values("mean_abs_shap", ascending=False)

    top_feature = mean_abs_shap.iloc[0]["feature"]
    top_value = mean_abs_shap.iloc[0]["mean_abs_shap"]

    last_feature = mean_abs_shap.iloc[-1]["feature"]
    last_value = mean_abs_shap.iloc[-1]["mean_abs_shap"]

    avg_shap = mean_abs_shap["mean_abs_shap"].mean()

    st.markdown(f"""
### 🔎 Key Insights

- **Most influential feature:**  
  **`{top_feature}`** with an average SHAP impact of **{top_value:.2f}**.  
  This feature has the strongest influence on predictions.

- **Least influential feature:**  
  **`{last_feature}`** with an average impact of only **{last_value:.4f}**,  
  meaning it barely shifts predictions.

- **Overall model sensitivity:**  
  Across all features, the average SHAP value is **{avg_shap:.2f}**.

---

### 📊 Influence Direction
- **Higher SHAP values → increase predicted trips.**  
- **Lower SHAP values → decrease predicted trips.**

In the beeswarm plot:  
- **Red = high feature values**,  
- **Blue = low feature values**.

This gives a complete picture of model behavior and sensitivity.
""")
