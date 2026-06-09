# prediccion_demanda_sin

Proyecto Python para la predicción de demanda de energía eléctrica con un flujo profesional de trabajo orientado a datos: ingesta y limpieza de información, análisis exploratorio, modelado con Prophet, Random Forest y XGBoost, visualización en Streamlit y generación de reportes.

## Objetivo

Construir una base escalable para:

- ETL de datos eléctricos desde fuentes operativas o históricas.
- EDA para entender patrones de consumo, estacionalidad, anomalías y calidad de datos.
- Entrenamiento y evaluación de modelos de pronóstico.
- Despliegue de un dashboard interactivo con Streamlit.
- Automatización de reportes y artefactos de salida.

## Estructura del proyecto

```text
prediccion_demanda_sin/
├── config/                   # Parámetros, logs y configuración de ejecución
├── dashboard/                # Aplicación Streamlit
├── data/                     # Datos crudos, intermedios y procesados
├── docs/                     # Documentación de apoyo y referencias
├── models/                   # Artefactos de modelos entrenados
├── notebooks/                # Cuadernos para exploración y modelado
├── reports/                  # Figuras, tablas y reportes finales
├── scripts/                  # Entradas ejecutables para entrenamiento y reportes
├── src/prediccion_demanda_sin/
│   ├── etl/                  # Extracción, limpieza y transformación
│   ├── eda/                  # Análisis exploratorio
│   ├── features/             # Ingeniería de variables
│   ├── modeling/             # Entrenamiento y predicción
│   ├── evaluation/           # Métricas y validación
│   ├── reporting/            # Generación de reportes
│   └── utils/                # Utilidades compartidas
└── tests/                    # Pruebas unitarias y de integración
```

## Componentes principales

- **ETL eléctrico**: carga de archivos, validación de esquema, normalización temporal, tratamiento de faltantes y detección de outliers.
- **EDA**: series de tiempo, descomposición, correlaciones, perfiles horarios/diarios/semanales y análisis de estacionalidad.
- **Modelos**: Prophet para forecasting temporal, Random Forest para relaciones no lineales y XGBoost para rendimiento predictivo.
- **Dashboard**: interfaz en Streamlit para explorar datos, comparar modelos y revisar pronósticos.
- **Reportes**: exportación de resultados, gráficos y métricas para entrega ejecutiva o académica.

## Instalación rápida

1. Crear y activar un entorno virtual.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Instalar dependencias.

```powershell
pip install -r requirements.txt
```

3. Ejecutar el dashboard.

```powershell
streamlit run dashboard/app.py
```

## Convenciones recomendadas

- Guardar datos originales en `data/raw/` y no modificarlos manualmente.
- Escribir los datasets limpios en `data/processed/`.
- Versionar los modelos exportados en `models/` cuando sea necesario.
- Generar reportes finales en `reports/final/`.
- Mantener la lógica reutilizable dentro de `src/prediccion_demanda_sin/`.

## Siguientes pasos sugeridos

- Definir el esquema esperado de los datos eléctricos.
- Crear el pipeline ETL principal.
- Implementar el entrenamiento de los tres modelos y su comparación.
- Conectar el dashboard con los artefactos del pipeline.
