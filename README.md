# Predicción de la Demanda de Potencia Eléctrica en el SIN Colombiano mediante Machine Learning

Este proyecto desarrolla una solución integral y reproducible para la predicción de la demanda de potencia activa horaria del Sistema Interconectado Nacional (SIN) de Colombia, evaluando tres modelos de diferente naturaleza matemática: **Facebook Prophet**, **Random Forest** y **XGBoost**.

Proyecto desarrollado bajo el marco académico de la **Cátedra Pedro Nel Gómez: *La energía en el desarrollo económico, social y tecnológico de Colombia*** en la **Universidad Nacional de Colombia - Sede Medellín**.

---

## ⚡ Características del Proyecto

1. **Pipeline de ETL Automatizado & Auditoría de Datos:**
   * Ingesta directa desde la API oficial de **XM S.A. E.S.P.** (operador del SIN) con fallback a **SIMEM**.
   * Limpieza de nulos, tratamiento de duplicados y validación física de series de tiempo.
   * **Auditoría de Lineaje:** Corrección del error de la *Hora 24* (23:00) mediante mapeo 0-indexed (`hour_number - 1`), recuperando **1,441 horas** de datos históricos que hacían falta en el dataset original y mejorando la precisión de todos los modelos.
2. **Modelamiento Avanzado:**
   * **Prophet:** Modelo aditivo estructural basado en series de Fourier para tendencias macro a largo plazo.
   * **Random Forest:** Ensamble por Bagging para capturar interacciones no lineales finas.
   * **XGBoost:** Algoritmo de Gradient Boosting regularizado para máxima precisión de corto plazo.
3. **Prevención de Fuga de Datos (Data Leakage):**
   * Creación de variables autorregresivas (`lag_1`, `lag_24`, `lag_168`) y medias móviles mediante desplazamientos temporales estrictamente causales (`.shift(1)`).
   * Validación mediante partición temporal cronológica (*Holdout 80/20*).
4. **Dashboard Interactivo Premium (Streamlit):**
   * Visualización interactiva con gráficos de series de tiempo, perfiles horarios/diarios de carga, heatmaps e inferencia futura (2026-2027).
   * **Simulador de Entrenamiento en Vivo:** Página donde se puede ver en tiempo real cómo convergen las curvas de pérdida del modelo y cómo se va reconstruyendo la señal de demanda paso a paso.
---

## 📊 Resultados y Métricas de Validación (Holdout 2025)

Métricas finales obtenidas sobre el conjunto de prueba cronológico tras la auditoría y saneamiento del dataset:

| Modelo | MAE (kW) | RMSE (kW) | MAPE (%) | $R^2$ (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Prophet** | 867,795.93 | 980,318.59 | 9.04% | 20.67% |
| **Random Forest** | 63,212.37 | 90,545.86 | 0.66% | 99.32% |
| **XGBoost** | **55,193.70** | **75,944.84** | **0.58%** | **99.52%** |

* **XGBoost** y **Random Forest** son idóneos para el despacho intradiario y la operación diaria del SIN (error < 0.6%).
* **Prophet** se consagra como el modelo recomendado para la proyección macro plurianual (2026-2027) debido a su inmunidad frente al **Efecto Cascada** (atenuación de amplitud por realimentación recursiva de lags).

---

## 📂 Estructura del Proyecto

```text
Proyecto-Datos-Energia/
├── config/                   # Parámetros, logs y configuración de ejecución
├── dashboard/                # Aplicación Streamlit (Páginas y Assets)
│   ├── pages/                # Hojas del Dashboard (EDA, Inferencia, Conclusiones, Simulador)
│   └── app.py                # Entrada principal del Dashboard
├── data/                     # CSVs resumidos de validación (Parquet ignorados por Git)
├── models/                   # CSVs de predicciones, métricas e inferencias a futuro
├── reports/                  # Figuras y gráficos del análisis
│   └── figures/              # Gráficos de series de tiempo y modelos (.png)
├── scripts/                  # Ejecutables del pipeline y análisis exploratorio (EDA)
│   └── run_pipeline.py       # Pipeline E2E (ETL + Entrenamiento de Modelos)
├── src/                      # Código fuente estructurado (etl, eda, modeling)
└── requirements.txt          # Dependencias de Python
```

---

## 🛠️ Instalación y Uso Local

### 1. Clonar y Configurar Entorno Virtual
```powershell
# Clonar repositorio
git clone https://github.com/SantiagoMaciasRuiz/Proyecto-Datos-Energia.git
cd Proyecto-Datos-Energia

# Crear y activar entorno virtual (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Ejecutar el Pipeline de Modelos (ETL + Train + Inferencia)
```powershell
python scripts/run_pipeline.py --start-date 2022-01-01 --end-date 2025-12-31 --horizon-hours 17520
```

### 3. Iniciar el Dashboard Interactivo de Streamlit
```powershell
streamlit run dashboard/app.py
```
*Accede en tu navegador a [http://localhost:8501](http://localhost:8501).*

---

## 🎓 Autor y Referencias
* **Autor:** Santiago Macias Ruiz
* **Asignatura:** Cátedra Pedro Nel Gómez - Universidad Nacional de Colombia (Sede Medellín)
* **Datos:** XM S.A. E.S.P. - API de Demanda de Energía Horaria del SIN.
