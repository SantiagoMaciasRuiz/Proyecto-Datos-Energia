# Informe de Investigación Académica: Predicción de la Demanda Eléctrica del Sistema Interconectado Nacional (SIN) mediante Machine Learning

**Asignatura:** Cátedra Pedro Nel Gómez: La energía en el desarrollo económico, social y tecnológico de Colombia  
**Institución:** Universidad Nacional de Colombia  
**Fecha:** Junio 2026  
**Autor:** Santiago Macias Ruiz  

---

## Resumen Ejecutivo

Este informe presenta el diseño, desarrollo, validación e interpretación de modelos predictivos aplicados a la demanda de potencia eléctrica horaria en el Sistema Interconectado Nacional (SIN) de Colombia para el periodo de entrenamiento 2022-2025 y proyección a largo plazo para los años 2026 y 2027. A través de un enfoque cuantitativo y reproducible, se comparan tres arquitecturas de Machine Learning de diferente naturaleza matemática: **Facebook Prophet** (descomposición aditiva), **Random Forest Regressor** (ensamble por Bagging) y **XGBoost** (Gradient Boosting secuencial). 

Los resultados demuestran la superioridad de los modelos autorregresivos tabulares en horizontes de predicción inmediata (1 hora adelante), donde XGBoost alcanza un Error Porcentual Absoluto Medio (MAPE) del **0.58%** y un coeficiente de determinación ($R^2$) de **99.52%**. Asimismo, se analiza con rigor académico el fenómeno del "Efecto Cascada" en predicciones recursivas plurianuales y la importancia física de la distinción entre Potencia Activa (kW) y Energía Acumulada (kWh) para la estabilidad del SIN.

---

## 1. Introducción y Contexto Académico

El estudio de los sistemas energéticos no puede desvincularse de la evolución social, tecnológica y económica de una nación. Siguiendo el legado del ingeniero y pintor Pedro Nel Gómez, quien plasmó en su obra la fuerza del desarrollo técnico, la infraestructura y el aprovechamiento de los recursos naturales en Colombia, este proyecto aborda la predicción de la demanda de electricidad como un pilar fundamental para la planificación del desarrollo económico. 

El Sistema Interconectado Nacional (SIN) colombiano debe mantener en todo momento un equilibrio dinámico e instantáneo entre la oferta de generación (principalmente hidráulica y térmica) y la demanda de los consumidores regulados y no regulados. La incapacidad de predecir adecuadamente la demanda conduce a ineficiencias críticas:
1. **Sobrecostos de operación:** Despacho innecesario de plantas térmicas de respaldo (más caras y contaminantes).
2. **Riesgos de inestabilidad física:** Variaciones severas en la frecuencia del sistema que pueden provocar apagones sistémicos.
3. **Incertidumbre en la planificación:** Dificultad para programar mantenimientos de redes de transmisión y subestaciones.

El modelamiento matemático con técnicas modernas de inteligencia artificial y aprendizaje supervisado permite optimizar el uso de recursos energéticos y robustecer la seguridad del sistema eléctrico.

---

## 2. Fundamentos Físicos y Unidades de Medida en el SIN

Un error común en la literatura técnica y en informes ejecutivos es confundir o usar indistintamente los conceptos de **Potencia** y **Energía**. Para un informe con rigor profesional y académico, es indispensable definir conceptual y matemáticamente estas magnitudes físicas.

### 2.1. Potencia Activa Instantánea frente a Energía Acumulada

*   **Potencia Activa ($P$):** Es la tasa instantánea a la cual se transfiere o consume energía eléctrica útil en un circuito. Se mide en **Kilovatios (kW)** o **Megavatios (MW)**. Físicamente representa el esfuerzo o la carga instantánea impuesta sobre las máquinas generadoras (los rotores de las turbinas).
    
    $$P(t) = \frac{dE(t)}{dt}$$

*   **Energía Eléctrica ($E$):** Es la integral de la potencia activa a lo largo del tiempo, representando el trabajo total realizado por la corriente eléctrica en un intervalo temporal específico. Se mide en **Kilovatios-hora (kWh)** o **Megavatios-hora (MWh)**.
    
    $$E = \int_{t_1}^{t_2} P(t) dt$$

En nuestro dataset horariom la variable `demanda` reportada por XM corresponde a la potencia activa promedio requerida por el sistema en un intervalo de una hora. Numéricamente, si el SIN demanda un promedio de $9,000,000\text{ kW}$ durante una hora completa, la energía consumida en ese periodo es exactamente de $9,000,000\text{ kWh}$. Sin embargo, conceptualmente se debe modelar como **Potencia Activa (kW)**, ya que el operador del sistema XM S.A. E.S.P. requiere dimensionar la carga del sistema en términos de capacidad de generación instantánea.

### 2.2. Por qué modelar en kW y no en kWh
1. **Dinámica de Frecuencia:** La frecuencia del SIN (nominalmente $60\text{ Hz}$ en Colombia) está determinada por el balance instantáneo de potencia activa. Si la potencia demandada ($P_{dem}$) supera a la potencia generada ($P_{gen}$), las máquinas generadoras desaceleran por inercia rotacional y la frecuencia del sistema cae ($\Delta f < 0$). Modelar la carga en kW permite prever estos desbalances.
2. **Capacidad de Infraestructura:** Las líneas de transmisión, transformadores y protecciones del SIN se diseñan y operan con límites térmicos expresados en corriente y potencia (MVA/MW), no en energía acumulada.

---

## 3. Auditoría del Dataset y Validación del Valor de Verdad

Para garantizar la fiabilidad del modelo, se realizó una auditoría profunda sobre la serie temporal histórica provista por XM S.A. E.S.P. y SIMEM.

### 3.1. Ficha del Dataset de Entrenamiento y Prueba
*   **Origen:** Datos horarios oficiales provistos por XM S.A. E.S.P. (2025) a través de su portal de información y de SIMEM (2025).
*   **Periodo temporal:** Del 1 de enero de 2022 al 31 de diciembre de 2025 (4 años completos).
*   **Total de registros:** $35,062$ observaciones horarias.
*   **Calidad:** 0.00% de registros nulos y 0.00% de outliers catastróficos. La serie temporal es continua y estable.

![Serie Temporal de la Demanda Horaria del SIN (2022-2025)](file:///c:/Users/santi/OneDrive/Escritorio/Proyecto%20graduacion/reports/figures/01_serie_temporal.png)
![Distribución de la Demanda Horaria y Cajas de Outliers](file:///c:/Users/santi/OneDrive/Escritorio/Proyecto%20graduacion/reports/figures/07_distribucion_demanda.png)


### 3.2. Divergencia Metodológica: Modelos de ML vs. Proyecciones Oficiales de la UPME
Al comparar nuestras predicciones con las publicadas en el Plan de Expansión de la UPME (Unidad de Planeación Minero Energética) (UPME, 2025), se observan discrepancias naturales debido a enfoques metodológicos radicalmente diferentes:

```
+---------------------------------------------------------------------------------+
|                   COMPARACIÓN DE ENFOQUES METODOLÓGICOS                         |
+---------------------------------------------------------------------------------+
| Enfoque UPME (Macro-Socioeconómico)   | Enfoque de este Proyecto (Machine Learn) |
+---------------------------------------+-----------------------------------------+
| - Multivariado de largo plazo.        | - Univariado autorregresivo estocástico.|
| - Basado en proyecciones del PIB.    | - Basado en la inercia de la serie.     |
| - Incorpora crecimiento poblacional.   | - Captura estacionalidades finas.       |
| - Simula fenómenos climáticos (Niño). | - No linealidad de calendario y retrasos|
| - Estructura escenarios de expansión.  | - Propagación recursiva (Efecto Cascada)|
+---------------------------------------+-----------------------------------------+
```

El modelo de este proyecto es óptimo para la operación en el "día a día" (despacho diario de energía) y para proyectar escenarios de inercia pura. En contraste, las proyecciones de la UPME buscan definir políticas públicas de infraestructura a 15-20 años basadas en variables macroeconómicas exógenas.

---

## 4. Fundamentos Teóricos de los Modelos Implementados

Se evaluaron tres modelos de familias algorítmicas diferentes para comprender el comportamiento de los datos bajo distintas aproximaciones matemáticas.

### 4.1. Facebook Prophet (Modelo Aditivo Descompuesto)
Prophet (Taylor & Letham, 2018) modela la demanda de potencia $y(t)$ como una suma de componentes de tendencia, estacionalidad y efectos de calendario:

$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$

Donde:
*   **Tendencia $g(t)$:** Modela los cambios no periódicos a largo plazo utilizando una función lineal por tramos con puntos de cambio (*changepoints*) seleccionados automáticamente:
    
    $$g(t) = (k + a(t)^T \delta) t + (m + a(t)^T \gamma)$$
    
    donde $k$ es la tasa de crecimiento, $\delta$ son los ajustes de tasa en puntos específicos, $m$ es el desfase y $\gamma$ representa los ajustes de desfase para mantener continuidad.
*   **Estacionalidad $s(t)$:** Se aproxima mediante Series de Fourier para capturar ciclos diarios ($P=24$), semanales ($P=168$) y anuales ($P=8766\text{ horas}$):
    
    $$s(t) = \sum_{n=1}^{N} \left( a_n \cos\left(\frac{2\pi n t}{P}\right) + b_n \sin\left(\frac{2\pi n t}{P}\right) \right)$$
    
*   **Festivos $h(t)$:** Representa el impacto de eventos del calendario (festivos en Colombia), modelado como un vector de variables indicadoras con coeficientes específicos.

### 4.2. Random Forest Regressor (Ensamble por Bagging)
El regresor de bosque aleatorio es un ensamble de $M$ árboles de decisión independientes entrenados mediante técnicas de agregación de bootstrap (Bagging), introducido originalmente por Breiman (2001):
1. Se extraen $M$ muestras de entrenamiento aleatorias con reemplazo a partir del dataset original.
2. Para cada árbol, en cada partición de nodo, el algoritmo selecciona aleatoriamente un subconjunto de variables predictoras de tamaño $m \ll d$ (donde $d$ es el número total de variables) para reducir la correlación entre los árboles individuales.
3. La predicción final $\hat{y}$ es el promedio aritmético de las salidas de todos los árboles:

$$\hat{y}(x) = \frac{1}{M} \sum_{i=1}^{M} T_i(x)$$

*Ventaja:* Reduce la varianza del modelo significativamente sin incrementar el sesgo, ofreciendo alta robustez frente a sobreajuste.

### 4.3. XGBoost (Extreme Gradient Boosting)
A diferencia de Random Forest, XGBoost construye los árboles de decisión secuencialmente (Gradient Boosting), donde cada nuevo árbol aprende a predecir los residuos o errores del conjunto de árboles anterior, de acuerdo con la formulación optimizada de Chen y Guestrin (2016).

La predicción en el paso $t$ se define como:

$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i)$$

Donde el árbol $f_t$ se calcula minimizando la función objetivo regularizada:

$$\mathcal{L}^{(t)} = \sum_{i=1}^{n} l\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)$$

El término de penalización o regularización $\Omega(f_t)$ evita la complejidad excesiva de los árboles individuales:

$$\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

Donde $T$ es el número de hojas del árbol y $w_j$ es el peso asignado a cada hoja. XGBoost emplea una aproximación de segundo orden de Taylor para optimizar rápidamente cualquier función de pérdida diferenciable.

---

## 5. Diseño de Ingeniería de Variables y Validación Temporal

El éxito del modelamiento predictivo radica en la adecuada representación de la física del consumo humano e industrial mediante variables (*features*).

### 5.1. Variables Autorregresivas e Inercia de Carga
*   **Rezagos de demanda:**
    *   `lag_1` ($y_{t-1}$): Captura la correlación termodinámica e industrial inmediata. La demanda eléctrica en una hora determinada está fuertemente correlacionada con el consumo de la hora anterior.
    *   `lag_24` ($y_{t-24}$): Captura el ciclo diario del consumidor. El comportamiento a las 14:00 de hoy es análogo al comportamiento a las 14:00 de ayer.
    *   `lag_168` ($y_{t-168}$): Captura el ciclo semanal completo (168 horas equivalen a 7 días). Resuelve la diferencia estructural entre días laborables y fines de semana.
*   **Medias móviles causales:** Para capturar tendencias inmediatas sin inducir fuga de datos, se calcula la media móvil sobre las últimas 24 horas desplazada en $t-1$:
    
    $$\text{rolling\_mean\_24}_t = \frac{1}{24} \sum_{i=1}^{24} y_{t-i}$$

![Demanda Eléctrica Promedio por Hora del Día (Ciclo Diario)](file:///c:/Users/santi/OneDrive/Escritorio/Proyecto%20graduacion/reports/figures/04_demanda_por_hora.png)
![Heatmap de Consumo por Hora del Día y Día de la Semana](file:///c:/Users/santi/OneDrive/Escritorio/Proyecto%20graduacion/reports/figures/06_heatmap_hora_dia.png)


### 5.2. Impacto Socioeconómico de los Festivos en Colombia
Colombia posee una de las tasas de días festivos oficiales más altas del mundo (más de 18 anuales). El análisis exploratorio revela que la demanda eléctrica cae en promedio un **13.19%** durante los días feriados. Por tanto, se incorporó una variable binaria `es_festivo` generada dinámicamente mediante el calendario localizado de la librería `holidays` para Colombia.

![Comparación de Demanda Eléctrica en Días Hábiles, Fines de Semana y Festivos](file:///c:/Users/santi/OneDrive/Escritorio/Proyecto%20graduacion/reports/figures/05_demanda_por_dia_semana.png)


### 5.3. Estrategia de Validación y Prevención de Fugas (Holdout Temporal)
Dado que los datos de series de tiempo presentan una estructura secuencial, el uso de validación cruzada aleatoria tradicional provocaría una fuga de información del futuro hacia el pasado. Para prevenir esto, implementamos un esquema de **Holdout Cronológico**, siguiendo las recomendaciones metodológicas de validación de series temporales de Hyndman y Athanasopoulos (2021):
*   **Entrenamiento:** Primeros 42 meses (~80% de los datos, desde enero de 2022 hasta mediados de 2025).
*   **Validación:** Últimos 6 meses (~20% de los datos, restante de 2025).

---

## 6. Métricas de Rendimiento y Comparación de Modelos

Para la evaluación del desempeño predictivo, se implementaron cuatro métricas estadísticas estándar sobre el conjunto de validación cronológico.

### 6.1. Ecuaciones de Métricas de Evaluación
1.  **Error Absoluto Medio (MAE):** Promedio de las diferencias absolutas entre predicciones y valores reales. Mide el sesgo global en las mismas unidades que la variable (kW).
    
    $$MAE = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

2.  **Raíz del Error Cuadrático Medio (RMSE):** Raíz de la media de los errores al cuadrado. Penaliza de forma más drástica los errores de gran magnitud.
    
    $$RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

3.  **Error Porcentual Absoluto Medio (MAPE):** Promedio de los errores absolutos relativos expresados en porcentaje. Facilita la comparación de la precisión.
    
    $$MAPE = \frac{100\%}{N} \sum_{i=1}^{N} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

4.  **Coeficiente de Determinación ($R^2$):** Proporción de la varianza explicada por el modelo.
    
    $$R^2 = 1 - \frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}$$

### 6.2. Tabla de Resultados de Validación

A partir del entrenamiento riguroso, se obtuvieron las siguientes métricas del SIN:

| Modelo | MAE (kW) | RMSE (kW) | MAPE (%) | $R^2$ (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Prophet** | 867,795.93 | 980,318.59 | 9.04% | 20.67% |
| **Random Forest** | 63,212.37 | 90,545.86 | 0.66% | 99.32% |
| **XGBoost** | **55,193.70** | **75,944.84** | **0.58%** | **99.52%** |

![Comportamiento y Estacionalidad Mensual de la Demanda](file:///c:/Users/santi/OneDrive/Escritorio/Proyecto%20graduacion/reports/figures/03_tendencia_mensual.png)


*Interpretación de Resultados:*
*   Los modelos basados en Machine Learning supervisado tabular (**XGBoost** y **Random Forest**) muestran un desempeño sobresaliente. XGBoost presenta un MAPE de apenas el **0.58%**, lo cual equivale a un error promedio de **55,194 kW** sobre una demanda media del sistema de aproximadamente 9.2 millones de kW. Esto demuestra que las variables autorregresivas explican casi en su totalidad las variaciones a corto plazo del consumo.
*   **Prophet** muestra una precisión inferior en validación de 1 paso debido a que no incorpora autoregresión fina hora a hora, estimando únicamente curvas estacionales continuas de gran escala y tendencias de largo plazo.

---

## 7. El "Efecto Cascada" en Inferencia Recursiva y Proyecciones 2026-2027

Un hallazgo crucial de este estudio radica en el comportamiento de los modelos al proyectar la demanda hacia los años **2026 y 2027** en un horizonte de mediano y largo plazo ($17,520$ horas en el futuro). En esta ventana temporal, no se disponen de registros reales de demanda, por lo que el proceso se realiza de manera recursiva (autoconsistente):

```
Inferencia Paso a Paso:
t-1 (Real) ----> Modelo (XGBoost) ----> Predicción t (y_pred_t)
                                              |
t (Predicho) --> Reemplaza lag_1 -------> Predicción t+1 (y_pred_t+1)
```

Al alimentar el modelo con sus propias predicciones anteriores de manera indefinida, ocurre el denominado **Efecto Cascada** o propagación acumulada del error residual (Hyndman & Athanasopoulos, 2021).

![Evaluación Histórica: Demanda Real vs. Predicción (Holdout 2025)](file:///c:/Users/santi/OneDrive/Escritorio/Proyecto%20graduacion/reports/figures/09_validacion_modelos.png)

![Proyección a Largo Plazo del SIN (2026-2027) - Inferencia Recursiva](file:///c:/Users/santi/OneDrive/Escritorio/Proyecto%20graduacion/reports/figures/10_pronostico_futuro.png)

### 7.1. Análisis Comparativo del Pronóstico Plurianual por Modelo

A partir de los gráficos generados sobre la validación histórica (Gráfico 9) y el pronóstico futuro (Gráfico 10), se desprenden las siguientes observaciones técnicas por modelo:

1.  **Atenuación Estacional en XGBoost y Random Forest:**
    *   En el corto plazo (Gráfico 9), XGBoost (Chen & Guestrin, 2016) y Random Forest (Breiman, 2001) se ajustan con precisión milimétrica a la demanda real, logrando un MAPE inferior al 0.66%.
    *   Sin embargo, al proyectar a 2 años (Gráfico 10), la inyección recursiva de la predicción $\hat{y}_t$ en la variable `lag_1` hace que el error residual $\epsilon_t$ se propague y multiplique en las horas sucesivas. Esto produce un efecto de "amortiguación" o aplanamiento progresivo de la amplitud de onda. La curva diaria de picos y valles pierde nitidez, tendiendo a converger hacia una banda constante. Los picos alcistas de consumo no logran replicar la inercia real del sistema.
2.  **Robustez y Estabilidad Macroscópica de Facebook Prophet:**
    *   En validación intradiaria (Gráfico 9), Prophet (Taylor & Letham, 2018) presenta un MAPE del 9.04%, desviándose en los picos instantáneos debido a su incapacidad de reaccionar autorregresivamente hora a hora.
    *   No obstante, en la proyección a largo plazo (Gráfico 10), Prophet resulta ser el modelo más consistente y estable. Al ser una formulación aditiva no recursiva basada estrictamente en componentes temporales deterministas y variables exógenas de calendario, no sufre acumulación de error residual. Conserva intacta la amplitud estacional (la forma y el rango de la oscilación diaria y semanal) y extiende de manera lineal la tendencia macro del consumo interanual registrada en el periodo 2022-2025.

### 7.2. Implicaciones para la Planificación Eléctrica
Estos hallazgos coinciden con las directrices de la UPME (2025), que señala que los modelos autorregresivos univariados puros son ideales para la operación diaria y el despacho económico inmediato de energía, pero ineficientes para planes de expansión a 10 o 15 años. Para la planificación a largo plazo, la estabilidad y la tendencia macro del modelo Prophet resultan más útiles, sirviendo como una línea base consistente que luego debe ser complementada con escenarios de crecimiento del PIB y variables macroeconómicas exógenas.

---

## 8. Conclusiones y Cierre

1.  **Alineación con la Cátedra Pedro Nel Gómez:** La modelación cuantitativa rigurosa de la demanda de energía proporciona una herramienta científica que apoya la soberanía energética del país, maximiza la eficiencia económica en la asignación de recursos y permite estructurar políticas de transición energética seguras.
2.  **Éxito del Proyecto:** Se ha diseñado un pipeline automatizado reproducible capaz de extraer, limpiar y modelar la demanda eléctrica del SIN colombiano con errores relativos inferiores al 1% en horizontes operativos de corto plazo.
3.  **Líneas de Trabajo Futuro:** Se sugiere incorporar variables climáticas del IDEAM de manera regionalizada (diferenciando temperaturas del Caribe y del interior del país) y la incorporación de escenarios macroeconómicos (PIB) en un modelo híbrido que combine Machine Learning con dinámica de sistemas.

---

## 9. Referencias Bibliográficas (Normas APA 7ª Edición)

*   Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32. https://doi.org/10.1023/A:1010933404324
*   Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. En *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785-794). ACM. https://doi.org/10.1145/2939672.2939785
*   Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: principles and practice* (3rd ed.). OTexts. https://OTexts.com/fpp3/
*   Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37-45. https://doi.org/10.1080/00031305.2017.1380080
*   Unidad de Planeación Minero Energética (UPME). (2025). *Plan de expansión de referencia generación-transmisión 2025-2039*. Ministerio de Minas y Energía.
*   XM S.A. E.S.P. (2025). *Informe de operación del SIN y administración del Mercado de Energía Mayorista*. XM S.A. E.S.P.
