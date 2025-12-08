# 🌐 OPTIMET-BCN — Digital Twin of Barcelona Metropolitan Mobility

OPTIMET-BCN es una aplicación interactiva en **Streamlit** para explorar, predecir y optimizar
los flujos diarios de movilidad entre municipios del área metropolitana de Barcelona.

La app combina:

- 🚍 **Datos de movilidad** (viajes diarios origen–destino entre municipios)  
- 🌦️ **Meteorología** y 🎟️ **eventos**  
- 🤖 Un modelo sencillo de **predicción OD**  
- 🧮 Un módulo de **optimización de recursos** centrado en Barcelona

El objetivo es ofrecer una **herramienta de apoyo al análisis y a la decisión**, no un
planificador operativo en producción.

---

## 1. Estructura del proyecto (resumen)

En la raíz del proyecto encontrarás:

- `main.py` – **punto de entrada** de la app de Streamlit
- Carpeta `tabs/` – una pestaña de la app por archivo  
  - `data_explorer.py` – exploración básica del dataset  
  - `visual_plots.py` – visualizaciones generales  
  - `heatmap_mobility.py` – heatmap / concentración de flujos OD  
  - `weather_events.py` – impacto del clima y los eventos  
  - `prediccion_od.py` + `prediccion_od_src.py` – módulo de predicción OD  
- Carpeta `utils/` – funciones compartidas  
  - `state_manager.py` – envoltorio ligero sobre `st.session_state`  
  - `load_data.py`, `geo_utils.py`, `optimizer_utils.py`, `plot_utils.py`, etc.  
- Carpeta `data/processed/`
  - `final_combined_2023_2024.csv` – dataset combinado de movilidad + clima + eventos  
  - `municipios_with_lat_alt.csv` – coordenadas de municipios  
- Carpeta `models/` – artefactos de ML cacheados (se crean automáticamente al entrenar el modelo)
- `requirements.txt` – dependencias de Python

Para **usar** la app no es necesario modificar el código; todo se maneja desde la interfaz.

---

## 2. Instalación

### 2.1. Requisitos previos

- Python **3.10 o superior**  
- `git` (opcional, sólo si clonas el repositorio)  
- Conexión a internet únicamente para instalar los paquetes

### 2.2. Crear entorno virtual

```bash
# 1) Crear entorno virtual
python -m venv .venv

# 2) Activarlo
#   En macOS / Linux:
source .venv/bin/activate
#   En Windows (PowerShell):
# .venv\Scripts\Activate.ps1
#   En Windows (cmd):
# .venv\Scripts\activate.bat

# 3) Instalar dependencias
pip install -r requirements.txt
```

---

## 3. Cómo ejecutar la app

Con el entorno virtual **activado** y desde la raíz del proyecto:

```bash
streamlit run old_main.py
```

El navegador se abrirá automáticamente en una URL tipo
`http://localhost:8501`.

### 3.1. Tiempo de carga inicial

En el primer arranque es normal ver *spinners* de carga durante varios segundos:

* Se carga en memoria el dataset `final_combined_2023_2024.csv`.
* Se carga (o entrena, si aún no existe) un pequeño modelo de **Random Forest** para la predicción OD.
* El módulo de optimización calcula estadísticas agregadas para los enlaces que implican a **Barcelona**.

Las ejecuciones posteriores son más rápidas gracias al cacheo de datos y modelos.

---

## 4. Datos utilizados

La aplicación trabaja con flujos diarios de movilidad enriquecidos con contexto:

* `day` y atributos de calendario (mes, día de la semana, etc.)
* `municipio_origen_name`, `municipio_destino_name`
* `viajes` – número de viajes para ese par OD y día
* Variables meteorológicas (`tavg`, `tmin`, `tmax`, `prcp`)
* Indicadores de eventos y asistencia (`event(y/n)`, `name`, `attendance`, etc.)

Para los mapas se usa `data/processed/municipios_with_lat_alt.csv`,
que contiene las coordenadas de los municipios.

Mientras mantengas la estructura original de carpetas, no necesitas tocar estos archivos.

---

## 5. Uso de la app — pestaña a pestaña

La interfaz principal está organizada en **seis pestañas**.

### 5.1. Pestaña 1 – 📊 Exploración de Datos

**Objetivo:** revisar rápidamente el estado del dataset combinado.

Qué permite hacer:

* Ver una **vista previa** de las primeras filas del dataset.
* Consultar **KPIs básicos**:

  * número de registros y columnas
  * rango temporal cubierto (`day` mínimo y máximo)
* Explorar **histogramas**:

  * viajes totales por día
  * viajes totales por día de la semana
* Ver una lista automática de **días atípicos** (muy alta o muy baja movilidad).
* Inspeccionar una muestra de registros con `viajes = 0`.

Es la pestaña ideal para entender “qué hay en los datos” antes de entrar en vistas más específicas.

---

### 5.2. Pestaña 2 – 📈 Visualizaciones

**Objetivo:** explorar patrones de movilidad desde el punto de vista de un
**municipio origen**.

Controles principales:

* Selector de **municipio origen** (por defecto: Barcelona).
* Selector de **tipo de origen** (agrupación según la fuente / tipo de dato).

Salidas visuales:

* 🗺️ **Mini-mapa de destinos** (mapa de burbujas):
  cada círculo es un municipio destino, con tamaño proporcional al número de viajes.
* 📈 **Serie temporal diaria** de viajes totales desde el origen seleccionado.
* 📅 **Promedio de viajes por día de la semana** (gráfico de barras).
* 🏷️ **Desglose por tipo de origen** a lo largo del tiempo (área apilada).
* 🏆 **Top 10 municipios destino** (tabla + gráfico de barras).
* 🔄 **Movilidad intra vs inter-municipal** (gráfico de tarta).

Ejemplos de preguntas que ayuda a responder:

> “Desde Barcelona, ¿a qué municipios se viaja más y cómo cambia a lo largo de la semana?”

---

### 5.3. Pestaña 3 – 🌍 Heatmap

**Objetivo:** analizar la matriz OD completa y estudiar la concentración de los flujos.

Elementos clave:

* Selector de **rango de fechas** y **municipio foco** (por defecto Barcelona).
* **Heatmap** de la matriz OD agregada en el periodo seleccionado
  (intensidad de color = volumen de viajes).
* **Top flujos OD** ordenados por número de viajes.
* Distribuciones de **origen** y **destino**:
  cuánta movilidad genera y recibe cada municipio.
* Vista de **concentración / Pareto**:

  * cuántos enlaces explican el 80 % de los viajes
  * qué porcentaje del total de enlaces representan.

También puedes **exportar a CSV** los agregados OD de esta pestaña para usarlos fuera de la app.

---

### 5.4. Pestaña 4 – 🌦️ Clima y Eventos

**Objetivo:** entender cómo se relacionan la meteorología y los eventos con la movilidad.

Funciones:

* Seleccionar un **rango de fechas** que se aplica a todas las visualizaciones de la pestaña.
* Ver indicadores diarios agregados:

  * viajes totales
  * temperatura media, mínima y máxima
  * precipitación
  * si hubo al menos un evento
  * asistencia total a eventos
* Visualizar:

  * series temporales de movilidad frente a temperatura o lluvia
  * comparativas entre **días con evento** y **días sin evento**
  * dispersión (*scatterplots*) entre movilidad y variables meteorológicas

Es una vista de **contexto** para interpretar picos o caídas de movilidad
como posibles efectos de lluvia, olas de calor o grandes eventos.

---

### 5.5. Pestaña 5 – 🔮 Predicción

**Objetivo:** obtener una predicción sencilla de viajes entre un par
**origen–destino** para una fecha futura.

Flujo de uso:

1. Selecciona **municipio origen** y **municipio destino**.
2. Indica una **fecha futura**.
3. La app muestra la **serie histórica** de viajes para ese par OD y un resumen
   estadístico (mínimo, media, máximo).
4. Pulsa **“Predecir viajes”** para obtener una estimación del número de viajes
   en la fecha elegida (modelo Random Forest entrenado con 2023–2024).
5. La app interpreta automáticamente si la predicción está por debajo,
   en línea o por encima de la media histórica.

Este módulo está pensado como herramienta de **exploración de escenarios**,
no como un sistema de predicción operativo.

---

### 5.6. Pestaña 6 – ⚙️ Optimización (foco: Barcelona)

**Objetivo:** redistribuir una cantidad fija de **recursos** entre enlaces OD que
implican a **Barcelona** (como origen o destino) para reducir la saturación
en los enlaces más calientes.

Conceptos básicos:

* **Demanda**: número de viajes en un día concreto para un enlace OD.
* **Recursos**: capacidad asignada al enlace (vehículos, oferta, etc.),
  proporcional al peso histórico del enlace.
* **Temperatura** = `demanda / recursos`

  * ≈ 1 → enlace equilibrado
  * > 1 → enlace caliente / saturado
  * < 1 → enlace frío / infrautilizado
* **R_max**: capacidad total correspondiente al día de máxima demanda
  en el histórico (solo enlaces relacionados con Barcelona).

Flujo de trabajo:

1. **Elegir fecha del escenario**

   * Si la fecha existe en el dataset:
     la demanda por enlace es la **observada** ese día y **no es editable**.
   * Si la fecha no existe:
     la app calcula, para cada enlace, una **media ponderada histórica**
     y **permite editar** la demanda manualmente.

2. **Revisar / editar la tabla de enlaces**

   * Cada fila representa un enlace OD con Barcelona como origen o destino.
   * Puedes ajustar la columna **Demanda** (cuando es editable) para construir
     escenarios hipotéticos.

3. **Configurar la optimización**

   * Decidir si el optimizador puede tomar capacidad extra de enlaces muy fríos
     como **último recurso** (lo que puede calentar ligeramente esos enlaces).

4. **Lanzar la optimización**

   * Pulsa **“🚀 Optimizar recursos para este escenario”**.
   * El algoritmo:

     * Usa todo el **slack seguro** (capacidad ociosa que no hace falta para
       mantener temperatura ≤ 1) para enfriar los enlaces más calientes.
     * Si se permite, realiza una segunda redistribución más agresiva obteniendo
       capacidad adicional de enlaces muy fríos.

5. **Interpretar resultados**

   * Demanda total del escenario.
   * Uso de recursos antes y después como % de `R_max`.
   * Temperatura media antes / después.
   * Índice de calor que penaliza especialmente los enlaces muy calientes.
   * Número de enlaces calientes (temperatura > 1) antes y después.
   * Slack (capacidad no utilizada) antes y después, en unidades y en %.
   * Para los 10 enlaces más calientes antes de optimizar:
     cuánto han aumentado sus recursos en valor absoluto y en porcentaje.

   Debajo se muestran tablas con el detalle enlace a enlace:
   **Demanda**, **Recursos base**, **Recursos optimizados**,
   **Temperatura antes**, **Temperatura después**, etc.

Esta pestaña funciona como un laboratorio de **“what-if”** para simular cómo
cambiaría la saturación de los enlaces OD que involucran a Barcelona
si se redistribuyera la oferta.

---

## 6. Rendimiento y tiempos de espera

* La carga del dataset combinado y el cálculo de agregados es relativamente pesado,
  por lo que es normal un **pequeño retraso** al iniciar la app.
* El modelo de predicción OD se guarda en disco y se reutiliza; la primera
  predicción puede tardar algo más, las siguientes son rápidas.
* El módulo de optimización trabaja únicamente con enlaces que incluyen
  Barcelona para mantener la interfaz fluida y utiliza el `StateManager`
  para reutilizar resultados intermedios.

---

## 7. Problemas frecuentes

* **La app no arranca o aparece un error de importación**
  Asegúrate de ejecutar `streamlit run main.py` desde la **carpeta raíz del proyecto**
  y con el entorno virtual **activado**.

* **Mensaje tipo “Global dataset is not loaded in StateManager('global')”**
  Comprueba que el archivo `data/processed/final_combined_2023_2024.csv`
  existe y no ha sido movido o renombrado.

* **La pestaña de Predicción tarda mucho la primera vez**
  Es normal: se prepara el dataset OD y se entrena (o carga) el modelo.
  Las ejecuciones posteriores son más rápidas.

* **La pestaña de Optimización muestra un error sobre datos faltantes**
  Reinicia la app desde la terminal y espera a que desaparezca el mensaje
  *“Inicializando modelo de optimización (foco: Barcelona)”*.
