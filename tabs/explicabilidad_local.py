import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from utils.state_manager import StateManager
from utils.prediccion.model_loader import load_lgb_model, load_feature_cols
from utils.prediccion.shap_utils import get_shap_explainer, compute_local_shap


# ============================================================
# Custom DARK Theme (NOT pure black)
# ============================================================

DARK_BG = "#1A1A1A"         # softer dark grey
LIGHT_TEXT = "#EEEEEE"      # near-white
MID_TEXT = "#CCCCCC"        # medium gray

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
# Fix SHAP plots for dark mode
# ============================================================

def fix_shap_plot_dark(ax):
    if ax is None:
        return
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=MID_TEXT, labelcolor=MID_TEXT)
    if ax.xaxis.label: ax.xaxis.label.set_color(LIGHT_TEXT)
    if ax.yaxis.label: ax.yaxis.label.set_color(LIGHT_TEXT)
    if ax.title: ax.title.set_color(LIGHT_TEXT)

    for spine in ax.spines.values():
        spine.set_color(MID_TEXT)

    for txt in ax.get_yticklabels():
        txt.set_color(LIGHT_TEXT)
        txt.set_fontsize(11)
    for txt in ax.get_xticklabels():
        txt.set_color(MID_TEXT)
        txt.set_fontsize(10)


# ============================================================
# MAIN LOCAL EXPLAINABILITY TAB
# ============================================================

def render_explicabilidad_local():
    st.header("🧠 Local Explainability")

    # -------------------------
    # Load dataset
    # -------------------------
    global_state = StateManager("global")
    df_model = global_state.get("df_model_training")

    if df_model is None:
        st.error("❌ df_model_training not found in global state.")
        return

    if not pd.api.types.is_datetime64_any_dtype(df_model["date"]):
        df_model["date"] = pd.to_datetime(df_model["date"], errors="coerce")

    if "municipio_origen_name" in df_model.columns:
        if df_model["municipio_origen_name"].dtype != "category":
            df_model["municipio_origen_name"] = df_model["municipio_origen_name"].astype("category")

    # -------------------------
    # Load model + features
    # -------------------------
    try:
        model = load_lgb_model()
        feature_cols = load_feature_cols()
    except Exception as e:
        st.error(f"Unable to load model or feature columns: {e}")
        return

    df_sorted = df_model.sort_values("date")

    # Same train/test split as training
    test_size = int(len(df_sorted) * 0.20)
    df_test = df_sorted.iloc[-test_size:]
    X_test = df_test[feature_cols].copy()

    # -------------------------
    # Choose observation
    # -------------------------
    tab_state = StateManager("explicabilidad_local")
    tab_state.init({"local_mode": "Last prediction"})

    st.subheader("🔍 Select the observation to explain")

    local_mode = st.radio(
        "Example to explain:",
        ["Last prediction", "Random test example"],
        index=0 if tab_state.get("local_mode") == "Last prediction" else 1,
        horizontal=True,
    )
    tab_state.set("local_mode", local_mode)

    x_to_explain = None
    meta = {}

    if local_mode == "Last prediction":
        pred_state = StateManager("prediction")
        latest = pred_state.get("latest_prediction")

        if latest is None:
            st.warning("No last prediction found — please run a prediction first.")
            return

        st.info(
            f"Using last prediction:\n"
            f"- Date: **{latest['date']}**\n"
            f"- Municipality: **{latest['municipio']}**\n"
            f"- Origin: **{latest['origen']}**\n"
            f"- Predicted trips: **{latest['y_pred']:.0f}**"
        )

        x_to_explain = pd.DataFrame([latest["features"]])[feature_cols]

        if "municipio_origen_name" in x_to_explain:
            cats = df_model["municipio_origen_name"].cat.categories
            x_to_explain["municipio_origen_name"] = pd.Categorical(
                x_to_explain["municipio_origen_name"], categories=cats
            )

        meta = latest

    else:
        if X_test.empty:
            st.error("Test set empty — cannot sample.")
            return

        x_to_explain = X_test.sample(1, random_state=123)
        row = df_test.loc[x_to_explain.index[0]]

        y_pred_raw = float(model.predict(x_to_explain)[0])
        y_pred = max(y_pred_raw, 0)

        st.info(
            f"Random test example:\n\n"
            f"- Date: **{row['date'].date()}**\n"
            f"- Municipality: **{row['municipio_origen_name']}**\n"
            f"- Origin: **{row['origen']}**\n"
            f"- Actual trips: **{row['viajes']:.0f}**\n"
            f"- Predicted trips: **{y_pred:.0f}**"
        )

        meta = {
            "date": str(row["date"].date()),
            "municipio": row["municipio_origen_name"],
            "origen": row["origen"],
            "y_real": float(row["viajes"]),
            "y_pred": y_pred,
        }

    # -------------------------
    # Compute SHAP values
    # -------------------------
    with st.spinner("Computing SHAP values…"):
        explainer = get_shap_explainer(model)
        shap_local = compute_local_shap(explainer, x_to_explain)

    shap_values = shap_local[0]
    features = x_to_explain.columns.tolist()

    base_value = float(np.array(explainer.expected_value).reshape(-1)[0])
    total_contrib = float(shap_values.sum())
    raw_prediction = base_value + total_contrib
    clipped_prediction = max(raw_prediction, 0.0)

    # -------------------------
    # Show input features
    # -------------------------
    st.subheader("📥 Feature values for this observation")
    st.dataframe(x_to_explain, use_container_width=True)

    # -------------------------
    # Waterfall plot (Matplotlib)
    # -------------------------
    st.subheader("📉 SHAP Waterfall Plot")

    exp = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=x_to_explain.iloc[0].values,
        feature_names=features,
    )

    fig = plt.figure(figsize=(7, 5), facecolor=DARK_BG)
    shap.waterfall_plot(exp, max_display=15, show=False)
    ax = plt.gca()
    fix_shap_plot_dark(ax)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    # -------------------------
    # Automated Interpretation
    # -------------------------
    st.markdown("## 📘 Automated Local Interpretation")

    df_feat = pd.DataFrame({
        "feature": features,
        "shap": shap_values,
        "abs_shap": np.abs(shap_values)
    }).sort_values("abs_shap", ascending=False)

    total_abs = df_feat["abs_shap"].sum() + 1e-8
    top5 = df_feat.head(5)
    perc_top5 = top5["abs_shap"].sum() / total_abs * 100

    pos = df_feat[df_feat["shap"] > 0].head(3)
    neg = df_feat[df_feat["shap"] < 0].head(3)

    def _fmt(df):
        if df.empty:
            return "_None_"
        out = []
        for _, r in df.iterrows():
            out.append(f"- **`{r['feature']}`** → SHAP = **{r['shap']:.2f}**")
        return "\n".join(out)

    st.markdown(f"""
### 🔍 Prediction breakdown

- **Base value (average model output):** `{base_value:.2f}`  
- **Sum of SHAP effects:** `{total_contrib:.2f}`  
- **Raw prediction:** `{raw_prediction:.2f}` trips  
- **Clipped prediction:** `{clipped_prediction:.2f}` trips  

### 🎯 Which features matter most?

- The **top 5 features** account for **{perc_top5:.1f}%** of the total explanatory power.

### 🔺 Features increasing the prediction
{_fmt(pos)}

### 🔻 Features decreasing the prediction
{_fmt(neg)}

---

### 🧩 How to read the waterfall plot

- The prediction starts from the **base value**.
- **Red bars** push the prediction **down**.
- **Blue bars** push the prediction **up**.
- The final point is the model’s output for this specific OD + date.

This plot tells you precisely **why the model predicted that number of trips**.
""")
