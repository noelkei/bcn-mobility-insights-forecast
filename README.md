
# 🌊 BCN Flow Intelligence

**BCN Flow Intelligence** is a Streamlit web application for analyzing and forecasting daily inflow mobility to Barcelona from surrounding municipalities.

The app combines:

- 🧮 A trained **LightGBM model**  
- 📊 Rich **visual analytics (EDA)** on time, weather, and events  
- 🧠 **Explainable AI (SHAP)**, both global and local  

It is designed as an analyst-friendly “mobility cockpit” to understand:
- How mobility evolves over time  
- How weather and events impact demand  
- Which features drive the model’s predictions  
- Why specific days behave unusually  

---

## 1. Project Structure

The relevant structure (simplified) is:

```text
.
├── data/
│   └── processed/
│       ├── df_model.csv
│       ├── df_model_training.csv        # Main modelling & app dataset
│       ├── events
│       ├── movilidad_combinada.csv
│       └── municipios_with_lat_alt.csv
├── docs/
│   └── state_manager_guide/
├── models/
│   └── lgb_model_final/
│       ├── model.pkl                    # Trained LightGBM model
│       └── feature_cols.json            # Ordered feature list used by the model
├── tabs/
│   ├── __init__.py
│   ├── eda_tab.py                       # ⏳ Temporal Analysis
│   ├── eda_weather_tab.py               # 🌦️ Weather Analysis
│   ├── eda_events_tab.py                # 🎟️ Events Analysis
│   ├── prediccion_viajes.py             # 🔮 Prediction
│   ├── explicabilidad_global.py         # 🧠 Global Explainability
│   └── explicabilidad_local.py          # 🔬 Local Explainability
├── utils/
│   ├── prediccion/
│   │   ├── feature_engineering.py
│   │   ├── lag_utils.py
│   │   ├── model_loader.py
│   │   └── shap_utils.py
│   ├── geo_utils.py
│   ├── load_data.py
│   └── state_manager.py
├── main.py                              # Streamlit entrypoint
├── requirements.txt
└── eda.ipynb                            # Notebook(s) for exploration / training
````

---

## 2. Installation

### 2.1. Requirements

* Python **3.9+** (3.10 recommended)
* `pip` or `conda`
* Basic terminal / command line

### 2.2. Create and activate a virtual environment (recommended)

```bash
# From the project root
python -m venv .venv
source .venv/bin/activate      # On macOS / Linux
# or
.\.venv\Scripts\activate       # On Windows
```

### 2.3. Install dependencies

```bash
pip install -r requirements.txt
```

Key libraries include:

* `streamlit`
* `pandas`, `numpy`
* `plotly`
* `lightgbm`
* `scikit-learn`
* `shap`

---

## 3. Running the App

From the project root, with the virtual environment activated:

```bash
streamlit run main.py
```

Then open the local URL printed in the terminal (usually `http://localhost:8501`) in your browser.

> ⚠️ The app assumes that `data/processed/df_model_training.csv` and
> `models/lgb_model_final/model.pkl` exist and are consistent with each other.
> These are loaded automatically at startup.

---

## 4. Data Expectations

The central dataset is:

`data/processed/df_model_training.csv`

It must contain at least:

* **Time & ID columns**

  * `date` (or `day`) – daily resolution
  * `municipio_origen_name` – origin municipality (categorical)
  * `origen` encoded as dummies:

    * `origen_Internacional`, `origen_Nacional`,
    * `origen_Regional`, `origen_Residente`

* **Target**

  * `viajes` – trips from origin municipality to Barcelona for that day

* **Calendar features**

  * `month`, `dow`, `is_weekend`, `dow_sin`, `dow_cos`, `month_sin`, `month_cos`

* **Weather**

  * `tavg`, `tmin`, `tmax`, `prcp`

* **Events**

  * `event_attendance`
  * Category flags:

    * `eventcat_city_festival`, `eventcat_concert`, `eventcat_festival`,
    * `eventcat_football`, `eventcat_motorsport`,
    * `eventcat_other_sport`, `eventcat_trade_fair`

* **Lags**

  * Global:  `total_viajes_dia_lag1` … `total_viajes_dia_lag7`
  * Local:   `viajes_lag1` … `viajes_lag7`

Other CSVs in `data/processed/` are used for exploration and geographic enrichment.

---

## 5. How the App Works

### 5.1. Global State

The app uses a simple state manager (`utils/state_manager.py`) to share data and objects between tabs:

* Global state (`StateManager("global")`) stores:

  * `df_model_training` – main modelling dataset
  * `df_geo` – municipalities with lat/long/altitude (when needed)
* Tab-specific state is used to remember user choices (e.g., last prediction).

You do not need to configure this manually; it is wired in `main.py`.

---

## 6. Tabs Overview & Usage

### 6.1. ⏳ Temporal Analysis (`tabs/eda_tab.py`)

**Goal:** Understand structural temporal patterns and seasonality in total mobility.

**Main components:**

1. **Date range filter**

   * Widget: `Filter date range`
   * Filters the analysis to a subset of days.

2. **Workdays vs Weekends**

   * KPIs: average trips on workdays vs weekends.
   * Interpretation text explaining weekday “capacity load” vs weekend “base load”.

3. **Seasonality charts**

   * Weekly average profile: bar chart by weekday.
   * Monthly average profile: line chart over month names.
   * Explains commute-driven peaks and seasonal holiday effects.

4. **Trend & Smoothing**

   * Daily trips + 7-day rolling average.
   * Shows underlying trend beyond weekly noise.

5. **Top Origin Municipalities**

   * Horizontal bar chart of top municipalities by total trips (filtered range).
   * Insight text about main feeder corridors.

6. **Correlation Matrix**

   * Heatmap of correlations between:

     * `total_viajes_dia`, `tavg`, `prcp`, `is_weekend`, `event_attendance`
   * Helps quantify relationships between macro-drivers.

7. **Final conclusion**

   * Narrative summarizing temporal/seasonal drivers.

---

### 6.2. 🌦️ Weather Analysis (`tabs/eda_weather_tab.py`)

**Goal:** Quantify how weather (temperature, rain) interacts with mobility demand.

**Workflow:**

1. **Date range filter**

   * Same logic: restrict analysis period.

2. **Rain impact KPIs**

   * Average trips on dry vs rainy days.
   * Delta (%) showing how much mobility changes with rain.

3. **Scatter + Boxplots**

   * Scatter: `Temp vs Trips` with trendline; color by “Dry Day” vs “Rainy Day”.
   * Boxplot: distribution of trips for dry vs rainy days.
   * Insight about structural vs weather-sensitive demand.

4. **Temperature & Rain timeline**

   * Dual-axis time series:

     * Line: temperature
     * Bars: rain
   * Shows meteorological context over time.

5. **Resilience Matrix**

   * Grouped bar chart:

     * Axis: Workday vs Weekend
     * Colors: Dry vs Rainy
   * Captures interaction effect between calendar and rain.

6. **Rain intensity thresholds**

   * Categories: No Rain, Drizzle, Moderate, Heavy.
   * Bars: average trips at each intensity, with `n` days.
   * Identifies the precipitation level where mobility really drops.

7. **Final conclusion**

   * Narrative on system resilience and operational thresholds.

---

### 6.3. 🎟️ Events Analysis (`tabs/eda_events_tab.py`)

**Goal:** Analyze how large events influence daily demand.

**Key steps:**

1. **Robust event cleaning**

   * Normalizes columns:

     * `attendance_clean`, `is_event`, `event`, `trips`
   * If `is_event` is missing, it is inferred from attendance > 0.
   * Handles missing event names with “Unknown Event”.

2. **Date range filter**

   * Widget: `Filter date range`.
   * Filters for analysis and aggregation.

3. **Average metrics comparison**

   * KPIs for event days vs non-event days:

     * Number of days
     * Average attendance
     * Average trips
   * Delta (%) showing uplift or reduction on event days.

4. **Impact analysis**

   * Scatter: attendance vs trips for event days with trendline.
   * Boxplot: distribution of trips for event vs normal days.
   * Insight about weak/strong coupling between attendance and total city trips.

5. **Timeline: Mobility & Event spikes**

   * Dual-axis chart:

     * Line: total trips
     * Bars: event attendance
   * Visually shows how events are scheduled relative to baseline demand.

6. **Final conclusion**

   * Narrative on event-driven shocks and variance vs volume.

---

### 6.4. 🔮 Prediction (`tabs/prediccion_viajes.py`)

**Goal:** Forecast daily trips from a chosen origin municipality and origin type to Barcelona, with scenario control for date, weather, events and lags.

**Usage:**

1. **Select municipality and origin type**

   * `Municipio de origen` (origin municipality)
   * `Tipo de origen` (Internacional, Nacional, Regional, Residente)

2. **Choose prediction date**

   * `Fecha` between:

     * First date in dataset
     * Up to 1 year after the last historical date
   * The app automatically:

     * Computes lags using `compute_auto_lags`
     * Uses historical means when lags are not available (e.g., far future)

3. **Weather configuration**

   * If the date is inside the dataset:

     * `tavg`, `tmin`, `tmax`, `prcp` are **auto-loaded** from history and locked (read-only).
   * If outside:

     * Fields become editable, with stored defaults in tab state.

4. **Events configuration**

   * If inside dataset:

     * Event categories and `event_attendance` are read from history and locked.
   * If outside:

     * You choose:

       * Categories (multiselect)
       * Total attendance (numeric input, default 0)
   * Internally converted to feature columns (`eventcat_*` + `event_attendance`).

5. **Lags**

   * Two expandable sections:

     * 🌍 Global lags `total_viajes_dia_lag1–7`
     * 🏙️ Municipality→Barcelona lags `viajes_lag1–7`
   * For each lag:

     * If historical value exists → shown as **disabled** input (not editable).
     * Otherwise → filled with a fallback mean:

       * Global mean trips or municipality-specific mean
       * You may edit it manually (what-if scenario).

6. **Prediction output**

   * Button: **“Calcular predicción”** / “Calculate prediction”.
   * Shows:

     * Predicted trips (clipped at 0)
     * Historical trips for that link & date, if available
     * Absolute and relative error when real value exists
   * Distribution panel:

     * Plotly histogram of historical trips for that municipality+origin
     * Vertical lines for real vs predicted values
     * Percentile metrics for both.

7. **Connection to Local Explainability**

   * After prediction:

     * The scenario (inputs + outputs) is stored in `StateManager("prediction")` as `latest_prediction`.
     * The local XAI tab can use this as “Last prediction” to generate SHAP explanations.

---

### 6.5. 🧠 Global Explainability (`tabs/explicabilidad_global.py`)

**Goal:** Understand which features generally drive the LightGBM model.

**What it does:**

1. Loads:

   * Global dataset (`df_model_training`)
   * Trained model (`models/lgb_model_final/model.pkl`)
   * Feature list (`feature_cols.json`)

2. Samples up to 4,000 rows for SHAP to keep plots responsive.

3. Computes global SHAP values with `TreeExplainer`.

4. Plots (matplotlib, dark theme):

   * **Bar summary plot**:

     * Mean |SHAP| per feature → global importance ranking
   * **Beeswarm summary plot**:

     * Distribution of SHAP values for each feature
     * Color encodes feature value (low → blue, high → red)

5. Automated interpretation:

   * Computes mean absolute SHAP values.
   * Dynamically identifies:

     * Most influential feature
     * Least influential feature
     * Average sensitivity
   * Writes a narrative explaining:

     * Which feature the model relies on most
     * Which ones are nearly irrelevant
     * How to read SHAP magnitudes and directions.

You get both **visual** and **textual** global explanations without manual analysis.

---

### 6.6. 🔬 Local Explainability (`tabs/explicabilidad_local.py`)

**Goal:** Explain a single prediction in detail — why the model predicted that value.

**Options to choose the observation:**

1. **Last prediction**

   * Uses the scenario stored from the Prediction tab.
   * Shows:

     * Date, municipality, origin type
     * Predicted and (if available) real value

2. **Random test example**

   * Splits data into train/test (time-based).
   * Samples one row from the test set.
   * Computes prediction and shows real value.

**Outputs:**

* Full feature row shown as a table.
* **SHAP waterfall plot**:

  * Starting from the model’s base value (average prediction)
  * Adds positive (red) and negative (blue) contributions feature by feature
  * Ends at the final prediction.
* Detailed **dynamic text explanation**:

  * Which features push the prediction up or down the most
  * How large those contributions are
  * How the sum of contributions matches the final prediction.

This tab answers: **“Why did the model predict this number for this day and origin?”**

---

## 7. Retraining / Experimentation (Optional)

Outside the app, notebooks like `eda.ipynb` and utilities in `utils/prediccion/` can be used to:

* Perform new random searches over LightGBM hyperparameters.
* Re-train models using the 75/15/10 train/val/test split.
* Save new models to `models/` and update `feature_cols.json`.

If you train a new model and want the app to use it:

1. Overwrite `models/lgb_model_final/model.pkl` with the new model.
2. Overwrite `models/lgb_model_final/feature_cols.json` with the corresponding feature list.
3. Restart Streamlit.

---

## 8. Troubleshooting

* **App says “Data not loaded”**

  * Check that `data/processed/df_model_training.csv` exists and has the correct columns.
  * Ensure `load_data.py` path (`processed/df_model_training.csv`) matches your folder.

* **Model loading error**

  * Check that `models/lgb_model_final/model.pkl` and `feature_cols.json` exist.
  * They must belong to the same training run (same feature ordering).

* **SHAP plots look broken / too slow**

  * Reduce the sample size in `explicabilidad_global.py`.
  * Ensure `shap` and `matplotlib` versions are consistent with `requirements.txt`.

---

## 9. License / Credits

This project was built as a final project for a **Visual Analytics** course, integrating:

* ✅ Streamlit web app
* ✅ Machine Learning model (LightGBM)
* ✅ Explainable AI with SHAP

**BCN Flow Intelligence** is a student project and not an official product.
Use it as a learning tool or starting point for more advanced mobility analytics.

