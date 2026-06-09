# Investigación de Fuentes Oficiales de Información y Arquitectura del Proyecto

Este documento detalla el análisis de las fuentes de información oficiales del sector energético colombiano y la arquitectura del repositorio para la asignatura *Cátedra Pedro Nel Gómez*.

---

## 1. Fuentes Oficiales de Información

### 1.1. XM (Operador del Mercado Eléctrico Colombiano)
XM es la entidad encargada de la planeación y operación del Sistema Interconectado Nacional (SIN) y de la administración del Mercado de Energía Mayorista (MEM) en Colombia.

*   **Portal Principal:** [https://www.xm.com.co](https://www.xm.com.co)
*   **Servicio Web/API:** API de Business Intelligence de XM para extracción de datos en formato JSON/CSV.
*   **Endpoint para Demanda Horaria:** `https://servapibi.xm.com.co/hourly`
*   **Método HTTP:** `POST`
*   **Encabezados Requeridos (Headers):**
    ```json
    {
      "Accept": "application/json, text/csv, */*",
      "User-Agent": "Mozilla/5.0 (compatible; demand-pipeline/1.0)"
    }
    ```
*   **Estructura del Payload (Cuerpo del POST):**
    ```json
    {
      "MetricId": "DemaReal",
      "StartDate": "YYYY-MM-DD",
      "EndDate": "YYYY-MM-DD",
      "Entity": "Sistema"
    }
    ```
    *   `MetricId`: `"DemaReal"` representa la Demanda Real de Energía.
    *   `Entity`: `"Sistema"` obtiene la demanda total del país (SIN) en lugar de desagregarla por comercializadores o subáreas.
*   **Autenticación:** Pública (no requiere tokens ni llaves privadas para este endpoint).
*   **Limitaciones:** La API restringe las solicitudes a un máximo de **31 días por petición**. Para consultas de largo plazo (como de 2022 a 2025), el código realiza solicitudes paginadas en ventanas mensuales automáticamente.

---

### 1.2. Datos Abiertos Colombia & SIMEM (Sistema de Información de Mercado)
Como respaldo cuando la API BI de XM está fuera de servicio o presenta intermitencias de red, el sistema consume los datos a través del portal de Datos Abiertos y el API de SIMEM.

*   **Portal Principal:** [https://www.simem.co](https://www.simem.co)
*   **API Pública de Datos:** `https://www.simem.co/backend-files/api/datos-publicos` (con fallback a `https://www.simem.co/backend-files/api/PublicData`).
*   **Método HTTP:** `GET`
*   **Parámetros de Consulta (Query Params):**
    *   `datasetId`: `"c1b851"` (Identificador único para la Demanda Real Horaria por Sistema en SIMEM).
    *   `startDate`: `"YYYY-MM-DD"`
    *   `endDate`: `"YYYY-MM-DD"`
*   **Autenticación:** Pública.

---

## 2. Arquitectura de Carpetas y Mapeo del Proyecto

El repositorio está organizado bajo un estándar profesional que separa el código de procesamiento, los datos, los entregables académicos y las visualizaciones. A continuación se detalla cómo se mapea la estructura física actual a las necesidades del proyecto:

```
proyecto_graduacion/
│
├── config/                     # Configuraciones del proyecto (parámetros y logs)
│   ├── logging/
│   └── parameters/
│
├── dashboard/                  # Interfaz gráfica interactiva
│   ├── app.py                  # Streamlit Dashboard (Vista unificada con pestañas)
│   ├── assets/                 # Estilos y recursos visuales adicionales
│   └── pages/                  # Páginas complementarias si se decide usar multipágina
│
├── data/                       # Almacenamiento local de datos en fases
│   ├── external/               # Datos externos de soporte (ej. calendarios de festivos)
│   ├── interim/                # Archivos intermedios durante el desarrollo
│   ├── processed/              # Parquet y CSV limpios listos para modelamiento
│   └── raw/                    # Datos crudos tal como se descargaron de XM
│
├── docs/                       # Documentación del proyecto y entregables
│   ├── referencias/            # Papers y literatura académica
│   ├── reportes/               # Reportes parciales
│   ├── fuentes_y_arquitectura.md # (Este archivo)
│   └── etl_documentation.md    # Documentación del flujo ETL
│
├── models/                     # Almacenamiento de modelos y métricas
│   ├── artifacts/              # Modelos Prophet, RF y XGBoost serializados (.joblib)
│   ├── forecast.csv            # Predicciones futuras para 2026-2027
│   ├── model_metrics.csv       # Métricas consolidadas (R², MAPE, MAE, RMSE)
│   └── predicciones.csv        # Predicciones en conjunto de prueba
│
├── notebooks/                  # Jupyter Notebooks para Análisis Exploratorio (EDA)
│
├── reports/                    # Figuras y tablas exportadas para reportes finales
│   ├── figures/
│   ├── tables/
│   └── final/
│
├── scripts/                    # Scripts ejecutables desde consola
│   ├── run_pipeline.py         # Orquestador E2E (Extracción, Limpieza, Entrenamiento)
│   ├── train.py                # Script independiente para entrenamiento y tuning
│   └── generate_report.py      # Generador automatizado de reportes PDF/Markdown
│
├── src/                        # Código fuente modular (Package)
│   ├── etl/                    # Tubería principal de datos
│   │   ├── extractor.py        # Descarga robusta con reintentos XM -> SIMEM
│   │   └── transformer.py      # Transformación, limpieza y feature engineering
│   └── prediccion_demanda_sin/ # Módulos especializados de Machine Learning
│       ├── features/           # Generación de variables y rezagos
│       ├── modeling/           # Configuración y entrenamiento de Prophet, RF, XGB
│       │   └── pipeline.py     # Orquestador interno del entrenamiento
│       └── utils/              # Funciones auxiliares y formateadores
│
├── tests/                      # Pruebas unitarias de calidad de datos y código
│
├── requirements.txt            # Dependencias del entorno de Python
└── README.md                   # Descripción general y guía de instalación
```
