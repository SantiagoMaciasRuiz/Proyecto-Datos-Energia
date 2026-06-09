# Informe de Auditoría de Datos y Modelamiento (Fase de Validación Regresiva)

**Proyecto:** Predicción de la demanda eléctrica del Sistema Interconectado Nacional (SIN) mediante técnicas de Machine Learning  
**Asignatura:** Cátedra Pedro Nel Gómez  
**Fecha:** Junio 2026  
**Auditor:** Agente AI (Antigravity)

---

## 1. Introducción y Propósito

Este informe documenta el proceso de auditoría de lineaje regresivo (*backwards validation*) realizado sobre el pipeline completo del proyecto. El propósito fundamental es certificar que los datos consumidos por los modelos predictivos, los indicadores del Dashboard Streamlit y las cifras expuestas en el Informe Final PDF sean matemáticamente exactos, físicamente coherentes y estén libres de anomalías técnicas o fuga de datos (*data leakage*).

---

## 2. Proceso de Auditoría de Lineaje de Datos

La auditoría se realizó siguiendo la trayectoria del flujo de datos en sentido inverso, desde la visualización hasta el origen:

1.  **Capa de Presentación:** Verificación de las métricas visualizadas en el Dashboard y compiladas en el PDF final contra las bases de datos de salida del modelo.
2.  **Capa de Inferencia y Entrenamiento:** Inspección del script de entrenamiento (`pipeline.py`) y del bucle de inferencia recursiva para los años 2026 y 2027.
3.  **Capa de Ingeniería de Variables:** Verificación de la causalidad de los rezagos (`lag_1`, `lag_24`, `lag_168`) y estadísticas móviles (`rolling_mean_24`).
4.  **Capa de ETL (Extracción y Transformación):** Análisis detallado de los scripts `extractor.py` y `transformer.py` encargados de la obtención del dataset primario de demanda del SIN.

---

## 3. Hallazgo Crítico: El Error de la Hora 24

Durante la auditoría del script de transformación de datos (`src/etl/transformer.py`), se identificó una anomalía crítica relacionada con la estructura horaria de la serie de tiempo cruda:

### A. Descripción del Problema
Los datos extraídos de la API oficial de XM reportan las horas de cada día utilizando un formato indexado de 1 a 24 (donde `Values_Hour1` corresponde al periodo 00:00 - 01:00 y `Values_Hour24` al periodo 23:00 - 00:00). 
En la versión inicial del pipeline, la función `validate_demand_data` realizaba un filtrado rígido sobre el número de hora para descartar registros fuera del rango convencional, utilizando la condición de que la hora estuviera estrictamente entre `0` y `23`. Como consecuencia, **todos los registros correspondientes a la Hora 24 (23:00) eran silenciosamente descartados** durante el paso de limpieza, eliminando exactamente 1 hora por cada día del dataset histórico.

### B. Impacto del Error
*   **Pérdida de Datos:** Representaba una pérdida del **4.16%** de la serie de tiempo total (1,441 horas faltantes de un total de 1,441 días).
*   **Distorsión Física:** Al eliminar la Hora 24, el perfil de carga diario perdía continuidad física, y las diferencias temporales entre la última hora de un día y la primera hora del día siguiente se calculaban erróneamente sobre un intervalo de 23 horas en lugar de 24.
*   **Desgaste del Modelo:** Los rezagos autorregresivos de 24 horas (`lag_24`) y semanales (`lag_168`) se desalineaban temporalmente, lo que inyectaba un sesgo sistemático en los modelos.

### C. Solución Aplicada
1.  Se modificó la expansión de la tabla en `src/etl/transformer.py` para mapear correctamente la hora cruda indexada de 1-24 a un índice 0-indexed de 0-23 (`hour_number - 1`).
2.  Se corrigió el cálculo de la fecha y hora final utilizando el desfase exacto:
    $$\text{FechaHora} = \text{Fecha} + \text{timedelta}(\text{hours} = \text{hour\_number} - 1)$$
3.  Se reejecutó el pipeline completo (`run_pipeline.py`) desde el primero de enero de 2022 hasta el 31 de diciembre de 2025. El número total de registros horarios procesados aumentó con precisión de **33,143** a **34,584** horas, recuperando las 1,441 observaciones faltantes.

---

## 4. Verificación de Ingeniería de Variables y Prevención de Fugas

Se revisó la función `_build_feature_frame` del pipeline para garantizar que no existiera fuga de información (*data leakage*):
*   **Rezagos Temporales:** Los rezagos se generan mediante la función `.shift(1)` de pandas, lo que asegura que para predecir el instante $t$ se utilicen estrictamente datos de $t-1$, $t-24$ y $t-168$.
*   **Estadísticas Móviles:** Las medias y desviaciones móviles (`rolling_mean_24`) se calculan aplicando primero el desplazamiento causal `.shift(1)` sobre la demanda, garantizando que el valor real de la demanda en $t$ no se filtre en el cálculo de la ventana móvil.
*   **Holdout Temporal:** La partición de entrenamiento y validación se realiza mediante una división cronológica estricta en el tiempo (80% para entrenamiento, 20% para validación) en lugar de un muestreo aleatorio. Esto preserva la estructura de dependencia secuencial de la serie de tiempo.

---

## 5. Validación Numérica de las Métricas

La reejecución del pipeline tras la corrección de la Hora 24 arrojó las siguientes métricas definitivas de validación sobre el conjunto de test:

| Modelo | MAE (kW) | RMSE (kW) | MAPE (%) | $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| **Prophet** | 867,795.93 | 980,318.59 | 9.04% | 0.2067 |
| **Random Forest** | 63,212.37 | 90,545.86 | 0.66% | 0.9932 |
| **XGBoost** | **55,193.70** | **75,944.84** | **0.58%** | **0.9952** |

### Análisis de la Mejora
*   La inclusión de la Hora 24 estabilizó el comportamiento del modelo de boosting. El MAPE de **XGBoost** disminuyó de **0.59%** a **0.58%**, consolidándose como el modelo con mayor precisión horaria.
*   El MAPE de **Prophet** mejoró de **9.33%** a **9.04%** y su $R^2$ aumentó del **18.38%** al **20.67%**, beneficiándose de una serie temporal continua y sin discontinuidades artificiales a las 23:00 de cada día.

---

## 6. Certificación del Dashboard y Consistencia General

Se validó la alineación de todas las fuentes de información:
1.  **Archivos de Salida del Modelo:** Las métricas de `models/model_metrics.csv` coinciden de manera exacta con las tablas del informe PDF y con las cargadas en el Dashboard.
2.  **Dashboard Streamlit (`app.py`):** Los KPIs principales de la página de inicio cargan dinámicamente el MAPE y $R^2$ directo de `model_metrics.csv`. Se actualizaron las referencias de texto estático en `03_conclusiones.py` para reflejar el $R^2 = 99.32\%$ del Random Forest corregido.
3.  **Diapositivas y PDF:** Se actualizaron los valores numéricos y tablas en `docs/presentacion_diapositivas.md` e `docs/informe_final.md`. El PDF fue recompilado exitosamente.

---

## 7. Conclusión de la Auditoría

El pipeline del proyecto se declara **certificado y coherente**. La corrección del error horaria de la Hora 24 recuperó el lineaje físico e histórico de la demanda del SIN, aumentando la precisión general del modelado sin alterar las directrices académicas ni de diseño propuestas inicialmente.
