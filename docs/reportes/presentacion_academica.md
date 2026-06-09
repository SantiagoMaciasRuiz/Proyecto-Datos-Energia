# Predicción de demanda eléctrica del SIN colombiano con Machine Learning

## Diapositiva 1. Portada
- **Título:** Predicción de demanda eléctrica del Sistema Interconectado Nacional colombiano con Machine Learning
- **Subtítulo:** Integración de ETL, análisis exploratorio, modelado predictivo y visualización interactiva
- **Autor:** Santiago Macias Ruiz
- **Programa:** Ingeniería
- **Institución:** Universidad Nacional de Colombia - Sede Medellín
- **Fecha:** Junio 2026
- **Mensaje clave:** El proyecto propone una solución reproducible para anticipar la demanda horaria del SIN colombiano y apoyar la toma de decisiones operativas y analíticas.

## Diapositiva 2. Problema
- La demanda eléctrica presenta variabilidad horaria, diaria y estacional que dificulta la planeación de generación, compra de energía y operación del sistema.
- Errores de pronóstico pueden traducirse en mayores costos, mayor riesgo operativo y menor eficiencia en la asignación de recursos.
- Los datos energéticos suelen venir de fuentes heterogéneas, con posibles faltantes, duplicados y anomalías que afectan la calidad del análisis.
- **Pregunta de investigación:** ¿Es posible construir un pipeline robusto que integre datos históricos y modelos de ML para predecir la demanda eléctrica horaria del SIN colombiano con precisión y trazabilidad?

## Diapositiva 3. Contexto Colombiano
- El Sistema Interconectado Nacional (SIN) conecta la mayor parte del territorio colombiano y soporta la operación del mercado eléctrico mayorista.
- XM administra y publica información relevante del sector, incluyendo variables horarias de demanda y generación.
- La demanda eléctrica en Colombia está influenciada por patrones laborales, festivos nacionales, clima, actividad económica y comportamiento regional.
- El contexto colombiano exige modelos que capturen estacionalidad, fines de semana, festivos y cambios estructurales del consumo.
- **Relevancia:** una predicción confiable contribuye a la planeación energética, a la operación del sistema y a la analítica del mercado.

## Diapositiva 4. Fuentes de Datos
- **Fuente 1: API de XM**
  - Consumida desde la librería o el servicio público de XM.
  - Permite acceder a variables horarias del SIN, incluyendo demanda real.
- **Fuente 2: Datos Abiertos de Colombia**
  - Dataset “Demanda real” disponible en datos.gov.co.
  - Se usa como respaldo o fuente alternativa si el servicio principal no está disponible.
- **Datos crudos locales**
  - Archivos ubicados en `data/raw/`.
  - Se procesan y consolidan en `data/processed/`.
- **Variables base esperadas**
  - Fecha u hora de observación.
  - Demanda horaria.
  - Variables temporales derivadas.
- **Criterio de calidad:** se prioriza trazabilidad, consistencia temporal y validación de rangos.

## Diapositiva 5. Metodología
- **1. Ingesta de datos:** extracción desde XM y/o datos abiertos.
- **2. Limpieza:** normalización de columnas, eliminación de duplicados, imputación de nulos y validación de rangos.
- **3. Ingeniería temporal:** hora, día, mes, trimestre, año, día de semana, festivo y fin de semana.
- **4. EDA:** análisis de tendencia, estacionalidad, distribución y outliers.
- **5. Modelado:** entrenamiento y comparación de Prophet, Random Forest y XGBoost.
- **6. Evaluación:** métricas como MAE, RMSE, MAPE y comparación visual real vs predicho.
- **7. Visualización:** dashboard en Streamlit para consumo analítico y ejecutivo.
- **Principio rector:** separar claramente extracción, transformación, análisis y presentación.

## Diapositiva 6. Arquitectura del Proyecto
- **`src/etl/`**: extracción y transformación de datos.
- **`src/eda/`**: análisis exploratorio y gráficas automatizadas.
- **`src/prediccion_demanda_sin/modeling/`**: entrenamiento y predicción de modelos.
- **`src/dashboard/`**: interfaz Streamlit para visualización y consulta.
- **`reports/figures/`**: gráficas HTML generadas automáticamente.
- **`reports/final/`**: entregables ejecutivos o académicos.
- **`models/`**: artefactos entrenados y resultados comparativos.
- **Flujo de trabajo:** raw data → processed data → EDA → modelado → evaluación → dashboard/reportes.

## Diapositiva 7. EDA Highlights
- La serie histórica permite identificar crecimiento, estacionalidad y episodios atípicos en la demanda.
- La tendencia anual resume la evolución estructural del consumo eléctrico.
- La tendencia mensual captura variaciones estacionales y posibles efectos de clima o actividad económica.
- El perfil por hora del día revela picos de carga y valles de menor consumo.
- El perfil por día de la semana muestra diferencias entre días laborales, sábados y domingos.
- El heatmap hora vs día de semana facilita detectar patrones operativos recurrentes.
- La distribución de la demanda ayuda a entender asimetría, dispersión y comportamiento extremo.
- La detección de outliers identifica observaciones que pueden ser errores o eventos extraordinarios.

## Diapositiva 8. Resultados de Modelos
- **Modelos evaluados:** Prophet, Random Forest y XGBoost.
- **Objetivo comparativo:** balancear precisión predictiva, robustez temporal e interpretabilidad.
- **Métricas recomendadas:** MAE, RMSE y MAPE.

| Modelo | MAE (kW) | RMSE (kW) | MAPE (%) | Observación |
|---|---:|---:|---:|---|
| Prophet | 867,795.93 | 980,318.59 | 9.04% | Bueno para capturar tendencia y estacionalidad a largo plazo |
| Random Forest | 63,212.37 | 90,545.86 | 0.66% | Útil para relaciones no lineales con variables temporales |
| XGBoost | 55,193.70 | 75,944.84 | 0.58% | Ofrece el más alto desempeño con error promedio de 0.58% |

- **Comparación visual esperada:** real vs predicho, residuos y horizonte futuro por modelo.
- **Conclusión técnica de la auditoría:** el mejor modelo operativo a corto plazo es XGBoost ($R^2 = 99.52\%$). Para planificación macro y largo plazo se recomienda Prophet ($R^2 = 20.67\%$).
- **Nota académica:** valores finales auditados y validados tras la corrección del lineaje de la Hora 24.

## Diapositiva 9. Demo del Dashboard
- El dashboard está construido en Streamlit con tema oscuro y navegación por secciones.
- **Sección 1:** resumen ejecutivo con KPIs de demanda promedio, máxima, mínima y crecimiento anual.
- **Sección 2:** análisis histórico con gráficas interactivas Plotly.
- **Sección 3:** predicciones con comparación real vs predicho y horizonte futuro.
- **Sección 4:** conclusiones ejecutivas.
- **Valor agregado:** permite llevar la analítica del notebook a un entorno consumible para usuarios no técnicos.
- **Resultado esperado en demo:** interacción fluida, lectura rápida y acceso centralizado a insights y pronósticos.

## Diapositiva 10. Conclusiones
- El proyecto demuestra que un pipeline bien estructurado mejora la confiabilidad del análisis de demanda eléctrica.
- Las variables temporales y el calendario colombiano aportan señal explicativa relevante en series horarias.
- Prophet aporta interpretabilidad; Random Forest y XGBoost capturan no linealidades y pueden mejorar precisión.
- El dashboard facilita la comunicación de resultados a audiencias técnicas y no técnicas.
- La calidad del dato y la validación temporal son tan importantes como la elección del modelo.
- **Conclusión principal:** la combinación de ETL robusto, EDA riguroso y modelado comparativo permite construir una solución útil para pronóstico operativo.

## Diapositiva 11. Recomendaciones y trabajo futuro
- Incorporar variables exógenas como temperatura, lluvia, actividad económica o eventos especiales.
- Automatizar retraining con ventanas temporales definidas.
- Evaluar modelos adicionales como SARIMA, LightGBM o enfoques híbridos.
- Añadir validación por bloques temporales y análisis de errores por temporada.
- Publicar el dashboard como servicio interno para seguimiento periódico.
- Integrar monitoreo de drift para detectar cambios estructurales en el consumo.

## Diapositiva 12. Bibliografía
- XM. (s. f.). API XM y documentación de variables horarias del SIN. https://www.xm.com.co
- Equipo Analítica XM. (s. f.). API_XM. https://github.com/EquipoAnaliticaXM/API_XM
- Datos Abiertos de Colombia. (s. f.). Demanda real. https://www.datos.gov.co/en/en/dataset/Demanda-real/bei6-hfpu
- Taylor, S. J., & Letham, B. (2018). Forecasting at scale. The American Statistician.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
- Plotly Technologies Inc. (s. f.). Plotly Python Open Source Graphing Library. https://plotly.com/python/
- Streamlit Inc. (s. f.). Streamlit Documentation. https://docs.streamlit.io/
