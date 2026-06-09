# Presentación de Diapositivas: Predicción de la Demanda Eléctrica del SIN mediante Machine Learning

**Asignatura:** Cátedra Pedro Nel Gómez: La energía en el desarrollo económico, social y tecnológico de Colombia  
**Institución:** Universidad Nacional de Colombia  
**Fecha:** Junio 2026  

---

## Diapositiva 1: Portada y Contexto Académico

# PREDICCIÓN DE LA DEMANDA ELÉCTRICA EN EL SIN MEDIANTE TÉCNICAS DE MACHINE LEARNING
### Un Enfoque en la Planificación Energética de Colombia

*   **Presentado por:** Santiago Macias Ruiz
*   **Asignatura:** Cátedra Pedro Nel Gómez: La energía en el desarrollo económico, social y tecnológico de Colombia
*   **Universidad:** Universidad Nacional de Colombia
*   **Periodo:** 2026

---

## Diapositiva 2: Introducción y Planteamiento del Problema

### El Desafío del Balance Energético en el SIN
*   **Equilibrio Dinámico:** El Sistema Interconectado Nacional (SIN) requiere balancear la generación y el consumo segundo a segundo.
*   **Consecuencias del Desbalance:**
    *   **Subgeneración:** Caídas de frecuencia ($\Delta f < 0$), apagones parciales o totales.
    *   **Sobregeneración:** Sobrecargas térmicas en líneas, ineficiencia y sobrecostos por despacho térmico.
*   **Objetivo de este Proyecto:** Evaluar y comparar algoritmos de Machine Learning para predecir con precisión la demanda de potencia horaria del SIN, facilitando la toma de decisiones operativas.

---

## Diapositiva 3: Fundamentos Físicos y Unidades de Medida

### ¿Por qué modelamos en kW (Potencia) y no en kWh (Energía)?
*   **Potencia Activa ($P$ en kW):** Mide la tasa instantánea de transferencia de energía.
    
    $$P(t) = \frac{dE(t)}{dt}$$
    
    Determina la carga mecánica sobre los rotores de las turbinas generadoras y define la frecuencia del sistema ($60\text{ Hz}$ en Colombia).
*   **Energía Acumulada ($E$ en kWh):** Integral de la potencia sobre el tiempo. Es útil para la facturación comercial o gestión de embalses, pero no para la planeación operativa horaria.
*   **Conclusión Física:** Para garantizar la estabilidad de la red y programar el despacho de generación hora a hora, el modelo debe predecir la potencia activa (kW).

---

## Diapositiva 4: Auditoría del Dataset y Fuentes de Verdad

### Fuentes de Información y Calidad de Datos
*   **Fuente:** API de Business Intelligence de **XM S.A. E.S.P.** (Administrador del MEM) con fallback a **SIMEM**.
*   **Periodo Auditado:** Enero 2022 a Diciembre 2025 ($35,062$ registros horarios continuos).
*   **Calidad:** 0.00% datos faltantes y 0.00% outliers severos.
*   **Divergencia con Proyecciones Oficiales (UPME):**
    *   **UPME:** Modelos estructurales socioeconómicos basados en variables macroeconómicas exógenas (Crecimiento del PIB, crecimiento demográfico, clima futuro).
    *   **Nuestro Proyecto:** Modelos univariados y autorregresivos orientados a la inercia temporal y patrones horarios de la serie física.

---

## Diapositiva 5: Fundamentos Teóricos de los Modelos

### Tres Aproximaciones Algorítmicas
1.  **Facebook Prophet (Modelo Aditivo Descompuesto):**
    
    $$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$
    
    Modelado macro de tendencias segmentadas ($g(t)$) y estacionalidades mediante series de Fourier ($s(t)$).
2.  **Random Forest Regressor (Ensamble por Bagging):**
    Ensamble de múltiples árboles de decisión independientes entrenados en submuestras con reemplazo (*Bootstrap*).
    
    $$\hat{y}(x) = \frac{1}{M} \sum_{i=1}^{M} T_i(x)$$
    
3.  **XGBoost (Extreme Gradient Boosting):**
    Optimización secuencial basada en árboles donde cada modelo nuevo predice los errores de los anteriores bajo una función de pérdida regularizada L1/L2.

---

## Diapositiva 6: Ingeniería de Variables y Validación Temporal

### Capturando los Ciclos del Consumo Colombiano
*   **Variables Cíclicas:** Hora del día ($0-23$), Día de la semana ($0-6$), Mes ($1-12$).
*   **Efecto Festivos en Colombia:** Los días feriados registran una caída promedio de la demanda del **13.19%**. Modelada mediante variable binaria.
*   **Variables Autorregresivas:** Rezagos estructurados de la demanda (`lag_1`, `lag_24`, `lag_168`).
*   **Mitigación de Fuga de Datos (Data Leakage):**
    *   Cálculo de medias móviles de forma causal (retrasando la serie original).
    *   Estrategia de **Holdout Cronológico**: Partición limpia temporal 80% entrenamiento (2022 - mediados 2025) y 20% validación (restante de 2025).

---

## Diapositiva 7: Resultados y Comparación de Métricas

### Desempeño en el Conjunto de Validación Cronológico

| Modelo | MAE (kW) | RMSE (kW) | MAPE (%) | $R^2$ (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Prophet** | 867,795.93 | 980,318.59 | 9.04% | 20.67% |
| **Random Forest** | 63,212.37 | 90,545.86 | 0.66% | 99.32% |
| **XGBoost** | **55,193.70** | **75,944.84** | **0.58%** | **99.52%** |

*   **XGBoost** es el modelo dominante en validación a corto plazo, explicando el **99.52%** de la varianza en la demanda eléctrica con un error promedio del **0.58%**.
*   **Prophet** presenta menor precisión fina horaria ($R^2 = 20.67\%$), pero captura eficientemente la tendencia agregada general sin recurrir a datos pasados inmediatos.

---

## Diapositiva 8: El "Efecto Cascada" en Inferencia de Largo Plazo

### El Desafío de Predecir los Años 2026 y 2027
*   **Inferencia Recursiva:** Al proyectar sin datos reales, los modelos autorregresivos deben usar su propia predicción de la hora anterior para calcular los lags.
    
    $$\hat{y}_{t-1} \rightarrow \text{lag\_1} \rightarrow \hat{y}_t \rightarrow \text{lag\_1} \rightarrow \hat{y}_{t+1}$$

*   **Problema de Acumulación de Error (Efecto Cascada):**
    *   En **XGBoost** y **Random Forest**, los errores se propagan secuencialmente, causando un aplanamiento en los picos de las curvas proyectadas (atenuación de amplitud).
    *   **Prophet**, al no ser autorregresivo, es inmune a este efecto, resultando sumamente estable para proyecciones plurianuales.
*   **Decisión Operativa:** Usar **XGBoost** para despacho diario (0-24 horas adelante) y **Prophet** para estimaciones macro y tendencias anuales (2026-2027).

---

## Diapositiva 9: Conclusiones y Recomendaciones

*   **Aporte Metodológico:** Se logró estructurar e implementar una metodología reproducible que extrae datos oficiales del SIN colombiano y predice la demanda horaria con gran exactitud.
*   **Recomendaciones para XM / Operadores:**
    1.  Implementar **XGBoost** para el pronóstico intradiario operativo.
    2.  Realimentar continuamente los lags con lecturas reales cada 24 horas para evitar la acumulación de error del efecto cascada.
    3.  Integrar variables exógenas como índices de temperatura regional para refinar picos de consumo estacional.
*   **Cierre de Cátedra:** Este proyecto reafirma la relevancia de la modelación matemática en la gestión soberana y eficiente de los recursos energéticos nacionales.

---

## Diapositiva 10: Referencias Bibliográficas (APA 7ª Edición)

*   **Breiman, L. (2001).** Random forests. *Machine Learning*, 45(1), 5-32.
*   **Chen, T., & Guestrin, C. (2016).** XGBoost: A scalable tree boosting system. En *KDD '16*, 785-794.
*   **Hyndman, R. J., & Athanasopoulos, G. (2021).** *Forecasting: principles and practice* (3rd ed.). OTexts.
*   **Taylor, S. J., & Letham, B. (2018).** Forecasting at scale. *The American Statistician*, 72(1), 37-45.
*   **UPME. (2025).** *Plan de expansión de referencia generación-transmisión 2025-2039*. Ministerio de Minas y Energía.
*   **XM S.A. E.S.P. (2025).** *Informe de operación del SIN y administración del MEM*. XM S.A. E.S.P.

