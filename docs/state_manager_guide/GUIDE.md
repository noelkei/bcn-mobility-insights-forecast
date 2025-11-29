
````markdown
# 🧭 OPTIMET-BCN — State Manager & Project GUIDE

This GUIDE explains **how the OPTIMET-BCN Streamlit project is structured** and **how to correctly use the `StateManager` utility**.

It is written so that:

- Any teammate can understand how to plug in new code.
- Any ChatGPT/LLM that receives this file as context will know **how to generate code that fits the project** and **uses the state system correctly**.

---

## 0️⃣ High-level idea

- The app is a **Streamlit dashboard** with **tabs** (no sidebar navigation).
- Data is loaded from the `data/` folder, never from absolute paths.
- We use a custom utility `StateManager` to manage all state:
  - **Global state**: shared things like the main DataFrame and geodata.
  - **Per-tab state**: each tab’s own UI selections (filters, parameters, etc).
- Tabs are implemented as simple Python modules with a `show()` or `render_*()` function.

---

## 1️⃣ Project structure (expected)

A simplified view of the directory layout:

```text
OPTIMET-BCN/
│
├── main.py                      # Streamlit entrypoint (run this)
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── processed/
│   │   ├── final_combined_with_events_2024.csv
│   │   └── municipios_with_lat_alt.csv
│   └── raw/                     # (optional, not always committed)
│
├── tabs/                        # Each tab/page logic
│   ├── visual_plots.py          # Tab 2: visualizaciones generales
│   ├── ...                      # Other tabs: heatmap, prediction, etc.
│
├── utils/
│   ├── state_manager.py         # StateManager definition
│   ├── load_data.py             # load_data("processed/xxx.csv")
│   ├── geo_utils.py             # get_geo_data() -> municipios + lat/lon
│   └── ...                      # other shared utilities
│
└── docs/
    └── GUIDE.md                 # this guide
````

> 🔴 **IMPORTANT RULE:**
> All file paths in code must be **relative** to the project root, never absolute like `C:\Users\...`.

---

## 2️⃣ The `StateManager` concept

`StateManager` is a helper class defined in `utils/state_manager.py` that wraps `st.session_state` and organizes it like this:

```python
st.session_state["tabs_state"] = {
    "global": {
        "df_main": <DataFrame>,
        "df_geo": <DataFrame>,
        ...
    },
    "data_explorer": {
        "selected_dataset": "...",
        "selected_city": "...",
        ...
    },
    "visual_plots": {
        "municipio_sel": "...",
        "tipos_sel": [...],
        ...
    },
    ...
}
```

Each tab (and also a special `"global"` namespace) gets its own sub-dictionary.

You **do not** edit this structure manually; it is managed by `StateManager`.

---

## 3️⃣ `StateManager` API (how to use it)

In `utils/state_manager.py` there is a class roughly like:

```python
class StateManager:
    def __init__(self, tab_name: str = "global"): ...
    def init(self, defaults: Dict[str, Any]): ...
    def get(self, key: str, default: Any = None) -> Any: ...
    def set(self, key: str, value: Any): ...
    def reset(self, keys: list[str] | None = None): ...
    def get_all(self) -> Dict[str, Any]: ...
    def get_from_tab(self, tab_name: str, key: str, default: Any = None) -> Any: ...
    def copy_from_tab(self, source_tab: str, keys: list[str] | None = None): ...
    @classmethod
    def save_all(cls, filename: str | None = None): ...
    @classmethod
    def load_all(cls, filename: str | None = None): ...
    @staticmethod
    def debug_view(): ...
```

As a tab developer, you **only consume this API**, you don’t change the implementation.

---

## 4️⃣ Basic per-tab setup

For any tab file, e.g. `tabs/1_data_explorer.py`:

```python
import streamlit as st
from utils.state_manager import StateManager

def show():
    # 1) Create a manager for this tab
    state = StateManager("data_explorer")

    # 2) Initialize defaults (only set if not already present)
    state.init({
        "selected_dataset": "movilidad_municipios",
        "selected_city": "Barcelona",
    })

    # 3) Use the state values when building UI
    city = st.selectbox(
        "Municipio",
        ["Barcelona", "Badalona", "Hospitalet"],
        index=["Barcelona", "Badalona", "Hospitalet"].index(state.get("selected_city"))
    )

    # 4) Update state if user changes something
    state.set("selected_city", city)

    st.write(f"Ciudad seleccionada: {state.get('selected_city')}")
```

**Key rules for tabs:**

* Call `StateManager("<tab_name>")` at the beginning of `show()` or `render_*()`.
* Call `state.init({...})` once to declare default keys for this tab.
* Use `state.get()` and `state.set()` to read/write values.
* **Never** modify `st.session_state` directly.

---

## 5️⃣ Global state: loading data once in `main.py`

We want to:

* Load the big CSV `final_combined_with_events_2024.csv` only **once**.
* Load geodata `municipios_with_lat_alt.csv` only **once**.
* Store them in the **global state** so every tab can re-use them.

This is done in `main.py` with `"global"` as the tab name:

```python
import streamlit as st
from utils.state_manager import StateManager
from utils.load_data import load_data
from utils.geo_utils import get_geo_data
from tabs.visual_plots import render_visualizations  # or show(), depending on implementation

st.set_page_config(
    page_title="OPTIMET-BCN",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------
# 1) Global state initialization
# ----------------------------
global_state = StateManager("global")

global_state.init({
    "df_main": None,   # Main mobility dataset
    "df_geo": None,    # Municipios coordinates
})

if global_state.get("df_main") is None:
    df_main = load_data("processed/final_combined_with_events_2024.csv")
    global_state.set("df_main", df_main)

if global_state.get("df_geo") is None:
    df_geo = get_geo_data()
    global_state.set("df_geo", df_geo)

# (Optional) If you want to restore previous saved state at startup:
# StateManager.load_all()

# ----------------------------
# 2) App header and tabs
# ----------------------------
st.title("🌐 OPTIMET-BCN")
st.markdown("### Digital Twin of Barcelona Metropolitan Mobility")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Exploración de Datos",
    "📈 Visualizaciones",
    "🌍 Heatmap",
    "🌦️ Clima y Eventos",
    "🔮 Predicción",
    "⚙️ Optimización",
])

with tab1:
    st.header("Exploración de Datos")
    st.warning("⚠️ Módulo en desarrollo.")

with tab2:
    # Tab 2 calls the visualization function,
    # which will read df_main and df_geo from global state
    render_visualizations()

with tab3:
    st.header("Heatmap")
    st.warning("⚠️ Módulo en desarrollo.")

with tab4:
    st.header("Clima y Eventos")
    st.warning("⚠️ Módulo en desarrollo.")

with tab5:
    st.header("Predicción")
    st.warning("⚠️ Módulo en desarrollo.")

with tab6:
    st.header("Optimización")
    st.warning("⚠️ Módulo en desarrollo.")

st.markdown("---")
st.caption("© 2025 OPTIMET-BCN | Telefónica Tech")
```

> 🧠 **For humans & LLMs:**
> When generating or modifying `main.py`, always:
>
> * Use `StateManager("global")` to store shared dataframes.
> * Avoid calling `load_data()` inside tabs; only in `main.py` (or in a dedicated global loader).

---

## 6️⃣ How tabs should use global data

Inside a tab (for example the visualizations tab, `tabs/visual_plots.py`), we **do not** reload the CSV.
Instead, we read it from the global state:

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk

from utils.state_manager import StateManager

def render_visualizations():
    st.header("Visualizaciones Generales")

    # 1) Access GLOBAL state
    global_state = StateManager("global")
    df = global_state.get("df_main")
    geo = global_state.get("df_geo")

    if df is None or geo is None:
        st.error("❌ Global dataframes not found in StateManager.")
        st.info("Check that main.py loads df_main and df_geo into StateManager('global').")
        return

    # 2) Type safety (only once, df is cached globally)
    if not pd.api.types.is_datetime64_any_dtype(df["day"]):
        df["day"] = pd.to_datetime(df["day"], errors="coerce")

    df["viajes"] = pd.to_numeric(df["viajes"], errors="coerce").fillna(0)

    # 3) UI controls, filters, plots...
    col_left, col_right = st.columns([2, 1])

    with col_left:
        municipios_origen = sorted(df["municipio_origen_name"].dropna().unique())
        municipio_sel = st.selectbox(
            "Selecciona un municipio origen",
            municipios_origen,
            index=municipios_origen.index("Barcelona") if "Barcelona" in municipios_origen else 0,
        )

    with col_right:
        tipos_origen = sorted(df["origen"].dropna().unique())
        tipos_sel = st.multiselect(
            "Tipo de origen",
            tipos_origen,
            default=tipos_origen,
        )

    # Then use df, df_filtrado, geo, etc. as usual.
```

If you want **per-tab UI state** (to persist selections across reruns), you can create a second `StateManager` instance for that tab:

```python
def render_visualizations():
    st.header("Visualizaciones Generales")

    global_state = StateManager("global")
    tab_state = StateManager("visual_plots")

    tab_state.init({
        "municipio_sel": "Barcelona",
        "tipos_sel": None,
    })

    df = global_state.get("df_main")
    geo = global_state.get("df_geo")

    municipios_origen = sorted(df["municipio_origen_name"].dropna().unique())
    tipos_origen = sorted(df["origen"].dropna().unique())

    municipio_default = tab_state.get("municipio_sel")
    tipos_default = tab_state.get("tipos_sel") or tipos_origen

    municipio_sel = st.selectbox(
        "Selecciona un municipio origen",
        municipios_origen,
        index=municipios_origen.index(municipio_default) if municipio_default in municipios_origen else 0,
    )
    tipos_sel = st.multiselect(
        "Tipo de origen",
        tipos_origen,
        default=tipos_default,
    )

    tab_state.set("municipio_sel", municipio_sel)
    tab_state.set("tipos_sel", tipos_sel)

    # seguir con filtrados y gráficas...
```

---

## 7️⃣ Cross-tab communication

`StateManager` also allows tabs to read values from other tabs.

For example, in a prediction tab that wants to reuse the selected city from `visual_plots`:

```python
from utils.state_manager import StateManager
import streamlit as st

def show():
    state = StateManager("prediction")
    global_state = StateManager("global")

    df = global_state.get("df_main")

    selected_city = state.get_from_tab("visual_plots", "municipio_sel", default="Barcelona")

    st.write(f"Predicción usando ciudad: {selected_city}")
    # Use df and selected_city for models, plots, etc.
```

> 🧠 For LLMs:
> When a feature in one tab depends on a selection from another tab, use:
> `state.get_from_tab("<other_tab_name>", "<key>", default=...)`.

---

## 8️⃣ Data loading utilities (must be used by all code)

### `utils/load_data.py`

```python
import os
import pandas as pd
import streamlit as st

@st.cache_data
def load_data(filename: str):
    """
    Loads a dataset from the /data folder.
    Supports CSV and Parquet.

    Example:
        df = load_data("processed/final_combined_with_events_2024.csv")
    """
    data_path = os.path.join("data", filename)

    if filename.endswith(".parquet"):
        return pd.read_parquet(data_path)

    return pd.read_csv(data_path)
```

### `utils/geo_utils.py`

```python
import os
import pandas as pd
import streamlit as st

@st.cache_data
def get_geo_data():
    """
    Loads municipalities geodata from:
        data/processed/municipios_with_lat_alt.csv

    Must contain columns: municipio, lat, lon
    """
    path = os.path.join("data", "processed", "municipios_with_lat_alt.csv")
    df = pd.read_csv(path)

    expected_cols = {"municipio", "lat", "lon"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in municipios file: {missing}")

    return df[["municipio", "lat", "lon"]]
```

> 🔴 **NEVER** hardcode absolute paths like `C:\Users\...`
> Always use `load_data("processed/...")` or `get_geo_data()`.

---

## 9️⃣ Persistence (optional)

If you want your app to **remember state after a restart**, you can use:

In `main.py`, at the top:

```python
from utils.state_manager import StateManager
StateManager.load_all()   # Load previously saved session state (optional)
```

At the bottom of `main.py`:

```python
StateManager.save_all()   # Save current session state to disk
```

This writes a JSON file (for example `session_state.json`) that holds all tab states.

You can ignore this feature if you don’t need persistence across restarts.

---

## 🔟 Rules for collaborators & for ChatGPT/LLMs

When generating or reviewing code for this project:

1. **Use StateManager everywhere for app state.**

   * Global data: `StateManager("global")`.
   * Tab-specific UI state: `StateManager("<tab_name>")`.

2. **Load datasets only via helpers**:

   * `load_data("processed/xxx.csv")` for mobility data.
   * `get_geo_data()` for municipios geodata.

3. **Do NOT use absolute file paths** (`C:\...`, `/Users/...`).
   Only relative ones via the helpers.

4. **Never call `load_data()` inside tabs if the dataset is already in global state.**
   Tabs should read from `StateManager("global").get("df_main")`.

5. **Name tabs and keys clearly**:

   * Tab examples: `"data_explorer"`, `"visual_plots"`, `"heatmap"`, `"prediction"`, `"simulation"`.
   * Key examples: `"municipio_sel"`, `"tipos_sel"`, `"date_range"`, `"model_type"`.

6. **Use `state.init()` to declare your tab’s variables** at the beginning of `show()`:

   ```python
   state.init({"municipio_sel": "Barcelona"})
   ```

7. **Avoid storing large DataFrames inside tab state.**
   Use global state for shared DataFrames, not per-tab.

8. **Use `StateManager.debug_view()` when debugging**:

   ```python
   from utils.state_manager import StateManager
   StateManager.debug_view()
   ```

9. **When asking an LLM (ChatGPT) for code**, always paste:

   * The relevant tab file (or `main.py`),
   * This `GUIDE.md`,
   * And explicitly say:

     > “Use `StateManager` and loaders exactly as described in GUIDE.md”.

---

## ✅ Summary

* `main.py` is responsible for:

  * setting up Streamlit,
  * loading global data (`df_main`, `df_geo`) **once**,
  * storing them in `StateManager("global")`,
  * creating tabs and calling each tab’s `show()`/`render_*()`.

* Each tab file:

  * creates its own `StateManager("<tab_name>")`,
  * defines its defaults with `init()`,
  * reads global data via `StateManager("global").get(...)`,
  * uses `get()` / `set()` to manage its UI state.

Following this GUIDE ensures that:

* All teammates (and any helping ChatGPT/LLM) generate code that plugs into the same architecture.
* The app is fast, modular, and consistent.
* There are no broken imports or hardcoded paths.

**Use this GUIDE as the single source of truth when extending OPTIMET-BCN.**
