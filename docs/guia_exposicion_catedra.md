# Guía de Exposición y Estudio Académico para Sustentación del Proyecto

**Asignatura:** Cátedra Pedro Nel Gómez: La energía en el desarrollo económico, social y tecnológico de Colombia  
**Institución:** Universidad Nacional de Colombia - Sede Medellín  
**Proyecto:** Predicción de la Demanda de Potencia Eléctrica en el Sistema Interconectado Nacional (SIN) mediante Machine Learning  
**Autor:** Santiago Macias Ruiz

---

## 1. Contexto Académico y Planteamiento

Este documento sirve como guía de preparación para la sustentación y entrega final del proyecto. Condensa los pilares físicos, teóricos, matemáticos y metodológicos que fundamentan la investigación.

### El Reto Operativo del SIN
El Sistema Interconectado Nacional (SIN) requiere un equilibrio en tiempo real entre la energía inyectada por los generadores y la potencia consumida por los usuarios finales. Dado que la energía eléctrica a gran escala no se almacena de forma masiva y económicamente viable (salvo en centrales de bombeo o sistemas de baterías pequeños en comparación con la red), la estabilidad del sistema depende de una planificación predictiva del despacho de generación.

---

## 2. Fundamentos Físicos de la Red Eléctrica

Una pregunta clásica de sustentación es la justificación física de las variables de modelamiento:

### A. Potencia Activa (kW / MW) frente a Energía (kWh / MWh)
*   **Potencia Activa ($P$):** Es la tasa instantánea de transferencia de energía, medida en kilovatios (kW) o megavatios (MW). Físicamente:
    $$P(t) = \frac{dE(t)}{dt}$$
*   **Energía Acumulada ($E$):** Es la integral de la potencia activa consumida a lo largo de un intervalo temporal:
    $$E = \int_{t_1}^{t_2} P(t) dt$$
    Se mide en kilovatios-hora (kWh) o megavatios-hora (MWh).

### B. ¿Por qué predecimos Potencia (kW) y no Energía (kWh)?
1.  **Estabilidad del Sistema:** La estabilidad de la frecuencia de la red (que en Colombia está regulada a $60\text{ Hz}$ con una desviación máxima permitida de $\pm 0.2\text{ Hz}$) está directamente acoplada al balance instantáneo de potencia activa. Si la demanda de potencia ($P_{dem}$) supera a la generación activa ($P_{gen}$), los generadores síncronos desaceleran por el aumento del torque electromagnético y la frecuencia cae ($\Delta f < 0$).
2.  **Operación del Despacho:** Los operadores de la red (XM S.A. E.S.P. en Colombia) programan la operación de las centrales eléctricas en bloques horarios de potencia media activa (kW) para asegurar la reserva rodante y el control de frecuencia de la red. Por tanto, predecir la potencia horaria (kW) es la necesidad física y regulatoria primordial para el despacho económico.

---

## 3. Origen, Lineaje y Calidad de Datos

### A. Fuentes de Información
*   **Origen Primario:** API de Business Intelligence de **XM S.A. E.S.P.**, administrador del Mercado de Energía Mayorista (MEM) y operador del SIN.
*   **Origen de Contingencia:** **SIMEM** (Sistema de Información de la Carga de Energía y del SIN), operado también por XM.
*   **Periodo Evaluado:** 1 de enero de 2022 al 31 de diciembre de 2025 (4 años completos).

### B. Auditoría y la Corrección del Error de la "Hora 24"
Durante el proceso de auditoría regresiva del código del pipeline, se identificó un error en la capa de transformación (`transformer.py`):
*   **El Problema:** La API de XM reporta la demanda horaria indexada de la hora 1 a la hora 24. El script de transformación contenía una regla de validación que filtraba la columna horaria exigiendo que estuviese entre `0` y `23`. Esto descartaba silenciosamente el registro de la Hora 24 (23:00) para todos los días.
*   **Impacto Físico y Estadístico:**
    *   Representaba la pérdida de **1,441 registros horarios** (4.16% del dataset completo).
    *   Introducía discontinuidades físicas en la transición entre la última hora de un día y la primera del día siguiente.
    *   Desalineaba temporalmente los rezagos de periodicidad diaria (`lag_24`) y semanal (`lag_168`).
*   **La Corrección:** Se modificó la regla de transformación para mapear la hora cruda indexada 1-24 a un índice horario 0-23 (`hour_number - 1`) y calcular adecuadamente las marcas temporales. Tras la reejecución, el dataset creció exactamente de **33,143** a **34,584** horas, eliminando los vacíos y mejorando la precisión de todos los modelos.

---

## 4. Fundamentos Teóricos de los Modelos de Machine Learning

Se compararon tres aproximaciones algorítmicas de diferente naturaleza matemática:

### A. Facebook Prophet (Aproximación Estructural Aditiva)
Prophet descompone la serie de tiempo como una suma de componentes de tendencia, estacionalidad y efectos de calendario:
$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$
Donde:
*   $g(t)$ es la tendencia lineal segmentada de crecimiento macro del consumo.
*   $s(t)$ representa las estacionalidades periódicas (diaria, semanal, anual) modeladas mediante series de Fourier:
    $$s(t) = \sum_{n=1}^{N} \left( a_n \cos\left(\frac{2\pi n t}{P}\right) + b_n \sin\left(\frac{2\pi n t}{P}\right) \right)$$
*   $h(t)$ modela el efecto de los días festivos colombianos (caída del consumo).

### B. Random Forest Regressor (Ensamble por Bagging)
Algoritmo de aprendizaje supervisado tabular basado en el promedio de múltiples árboles de decisión independientes entrenados en submuestras con reemplazo (muestreo Bootstrap):
$$\hat{y}(x) = \frac{1}{M} \sum_{i=1}^{M} T_i(x)$$
*   **Ventaja:** Alta robustez al sobreajuste gracias a la aleatorización de variables en cada nodo (Feature Subspacing). Captura interacciones no lineales complejas entre la hora, el día de la semana y la demanda previa.

### C. XGBoost (Extreme Gradient Boosting)
Ensamble secuencial de árboles de decisión optimizados bajo el esquema de Boosting. Cada árbol sucesivo se entrena para corregir los residuos (errores) acumulados por los árboles anteriores, minimizando una función de pérdida regularizada:
$$\mathcal{L}^{(t)} = \sum_{i=1}^{n} l\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)$$
Donde:
*   $\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum w_j^2$ es el término de regularización (L1/L2) que castiga la complejidad del modelo (número de hojas $T$ y pesos de las hojas $w$).

---

## 5. Prevención de Fuga de Datos (Data Leakage)

En series de tiempo, la fuga de datos ocurre si el modelo utiliza información del futuro para predecir el presente. Para garantizar la validez científica del proyecto, se aplicaron dos técnicas:
1.  **Rezagos y Medias Móviles Causales:** Los rezagos se calculan estrictamente con la función `.shift(1)` de pandas. Asimismo, las medias y desviaciones móviles (`rolling_mean_24`) se aplican sobre la serie desplazada. Esto garantiza que la variable de predicción en el instante $t$ se alimente únicamente de datos medidos hasta el instante $t-1$:
    $$\text{rolling\_mean\_24}_t = \frac{1}{24} \sum_{i=1}^{24} y_{t-i}$$
2.  **Holdout Temporal Estricto:** En lugar de una partición aleatoria tradicional (que rompería el lineaje temporal y filtraría información futura), se dividió la serie cronológicamente: el primer 80% (Ene-2022 a mediados de 2025) se reservó para entrenamiento, y el restante 20% para la validación de test.

---

## 6. Resultados Comparativos de Desempeño

Métricas obtenidas tras la corrección del dataset en el conjunto de validación cronológico (Holdout 2025):

| Modelo | MAE (kW) | RMSE (kW) | MAPE (%) | $R^2$ (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Prophet** | 867,795.93 | 980,318.59 | 9.04% | 20.67% |
| **Random Forest** | 63,212.37 | 90,545.86 | 0.66% | 99.32% |
| **XGBoost** | **55,193.70** | **75,944.84** | **0.58%** | **99.52%** |

### Análisis de los Resultados
*   **XGBoost** y **Random Forest** dominan ampliamente en la precisión de corto plazo ($R^2 > 99\%$). Esto se debe a que la demanda de potencia eléctrica tiene una altísima inercia y autocorrelación de corto plazo. Al tener acceso directo a variables autorregresivas (`lag_1`), los modelos tabulares pueden predecir con un error promedio menor al **0.6%**.
*   **Prophet** tiene un menor desempeño en la validación punto a punto ($R^2 = 20.67\%$, MAPE = 9.04%). Dado que no es un modelo autorregresivo instantáneo, no puede reaccionar a la inercia inmediata de la última hora, prediciendo únicamente sobre la base de la tendencia macro y estacionalidades de Fourier precalculadas.

---

## 7. El "Efecto Cascada" en Inferencia Recursiva (2026-2027)

Un concepto de alto nivel técnico crucial para la exposición es el comportamiento de los modelos al proyectar dos años completos en el futuro (2026-2027), donde no existen datos reales de demanda:

### A. La Inferencia Recursiva Autoconsistente
Para proyectar el instante $t$, los modelos tabulares requieren `lag_1` (la demanda del instante anterior). Al no haber datos reales, se debe inyectar la predicción anterior como rezago de entrada:
$$\hat{y}_{t-1} \rightarrow \text{lag\_1} \rightarrow \hat{y}_t \rightarrow \text{lag\_1} \rightarrow \hat{y}_{t+1}$$

### B. El Problema de la Atenuación de Amplitud (Efecto Cascada)
*   **En XGBoost / Random Forest:** El error residual de cada predicción se acumula paso a paso. A medida que avanza el horizonte temporal (horas, semanas, meses), las predicciones recursivas tienden a converger hacia el promedio histórico ponderado de la hora respectiva. Esto se visualiza en los gráficos como una **atenuación de amplitud (aplanamiento de picos y valles)**. Los modelos tabulares pierden la dinámica de picos extremos a largo plazo.
*   **En Prophet:** Al no ser un modelo autorregresivo recursivo, Prophet predice la demanda en $t$ evaluando directamente las funciones deterministas del tiempo. Por ello, es **totalmente inmune al efecto cascada**, conservando intacta la amplitud estacional y la tendencia lineal a lo largo de los dos años completos de forecast.

---

## 8. Conclusiones y Recomendaciones Académicas

1.  **Doble Enfoque de Modelado:** Se concluye que no existe un único modelo óptimo para todos los horizontes. 
    *   Para la **operación diaria y despacho económico inmediato (próximas 24 horas)**, el modelo recomendado es **XGBoost**, dada su altísima precisión intradiaria.
    *   Para la **planificación de expansión a largo plazo (2026-2027)**, el modelo estructural **Prophet** es preferible debido a su estabilidad ante la acumulación de errores y consistencia macro.
2.  **Influencia del Calendario en Colombia:** Los días feriados presentan un descenso del **13.19%** en el consumo eléctrico promedio, mientras que los fines de semana disminuyen un **8.16%**, lo cual resalta la fuerte influencia de la actividad industrial y comercial en la curva de potencia activa.
3.  **Líneas de Trabajo Futuro:** Integrar variables exógenas de climatología regional (fenómenos de El Niño/La Niña y temperatura promedio de los principales nodos del SIN) para predecir anomalías causadas por el uso de aire acondicionado y calefacción.
