# Metodología de Modelamiento y Estrategia de Validación

Este documento detalla la formulación matemática, la arquitectura de validación y la prevención de fugas de información (*data leakage*) empleadas en el proyecto académico **"PREDICCIÓN DE LA DEMANDA ELÉCTRICA DEL SISTEMA INTERCONECTADO NACIONAL MEDIANTE TÉCNICAS DE MACHINE LEARNING"**.

---

## 1. Modelos de Predicción Implementados

Se seleccionaron tres algoritmos de distinta naturaleza matemática para comparar su rendimiento en la predicción del consumo eléctrico horario de Colombia:

### A. Facebook Prophet
Es un modelo aditivo de descomposición de series temporales. Estructura la demanda $y(t)$ como:

$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$

Donde:
- $g(t)$: Curva de tendencia no periódica (lineal o logística).
- $s(t)$: Componentes periódicos (estacionalidad diaria, semanal y anual).
- $h(t)$: Efecto de días festivos oficiales en Colombia.
- $\epsilon_t$: Término de error no modelado.

*Ventaja:* Maneja de forma robusta las irregularidades por días feriados y cambios estructurales de tendencia a largo plazo.

### B. Random Forest Regressor
Algoritmo de aprendizaje supervisado basado en el ensamble de múltiples árboles de decisión independientes (*Bagging*).
- Entrena $N$ árboles de decisión sobre submuestras aleatorias con reemplazo (*Bootstrap*).
- La predicción final es el promedio simple de las predicciones de los $N$ árboles.
- Reduce significativamente la varianza del modelo y evita el sobreajuste.

*Ventaja:* Captura relaciones no lineales complejas entre la hora del día, el día de la semana y el consumo eléctrico sin necesidad de escalar variables.

### C. XGBoost (Extreme Gradient Boosting)
Algoritmo optimizado de potenciación de gradiente (*Gradient Boosting*) secuencial.
- Construye árboles de decisión de forma aditiva: cada nuevo árbol intenta corregir los errores residuales del conjunto acumulado de árboles anteriores.
- Incorpora penalización L1 (Lasso) y L2 (Ridge) en la función objetivo para controlar la complejidad del modelo y evitar el sobreajuste.

*Ventaja:* Es extremadamente eficiente en velocidad y suele entregar las métricas de error más bajas en datos estructurados y series de tiempo tabuladas.

---

## 2. Estrategia de Validación Temporal (Holdout Cronológico)

Para evaluar el rendimiento real de los modelos y simular su comportamiento en producción, no se pueden usar validaciones aleatorias cruzadas tradicionales (*K-Fold* clásico). Esto violaría el principio de causalidad temporal (predecir el pasado con información del futuro).

En su lugar, implementamos una partición **Holdout Cronológico**:

1.  **Conjunto de Entrenamiento (80%):** Registros más antiguos ordenados cronológicamente (desde el 1 de enero de 2022 hasta aproximadamente mediados de 2025).
2.  **Conjunto de Validación (20%):** Registros más recientes ordenados cronológicamente (restante del 2025).

```
[---------------- Entrenamiento (80%) ----------------][---- Validación (20%) ----]
Ene 2022                                              Jun 2025             Dic 2025
```

---

## 3. Mitigación de Fuga de Datos (*Data Leakage*)

La fuga de datos ocurre cuando el modelo de entrenamiento tiene acceso inadvertido a información del futuro que no estaría disponible en un escenario real de inferencia. En series temporales, esto suele ocurrir al calcular variables retrasadas (*lags*) o estadísticas móviles de forma global sobre todo el dataset antes del split.

Para mitigar esto, nuestro pipeline implementa las siguientes salvaguardas:

1.  **Cálculo de Rezagos estrictamente pasados:**
    - `lag_1`: Demanda registrada en la hora anterior ($t-1$).
    - `lag_24`: Demanda registrada en la misma hora del día anterior ($t-24$).
    - `lag_168`: Demanda registrada en la misma hora de la semana anterior ($t-168$).
2.  **Estadísticas móviles causales:** Las medias móviles (`rolling_mean_24`, `rolling_mean_168`) y la desviación estándar móvil (`rolling_std_24`) se calculan desplazando la serie original en $t-1$. Esto garantiza que la ventana de cálculo solo contenga valores pasados y nunca el valor en el instante $t$ que se desea predecir.
3.  **Inferencia recursiva paso a paso:** Para proyectar la demanda hacia los años futuros (2026 y 2027), no utilizamos valores reales de demanda (ya que no existen). En su lugar, el pipeline realiza un bucle recursivo hora por hora, donde la predicción $\hat{y}_t$ generada por el modelo se alimenta de forma iterativa como la entrada de rezago `lag_1` para calcular $\hat{y}_{t+1}$.

---

## 4. Métricas de Evaluación del Rendimiento

Para cuantificar el error de los modelos sobre el conjunto de validación, se computan cuatro métricas estándar:

### 1. Error Absoluto Medio (MAE)
Mide la magnitud promedio de los errores en las predicciones, sin importar su dirección.

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### 2. Raíz del Error Cuadrático Medio (RMSE)
Mide la desviación promedio de las predicciones frente a los valores reales. Penaliza con mayor severidad los errores grandes (debido al término cuadrático).

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

### 3. Error Porcentual Absoluto Medio (MAPE)
Expresa el error como un porcentaje promedio, facilitando la comprensión del margen de error relativo de la predicción.

$$MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

### 4. Coeficiente de Determinación ($R^2$)
Mide la proporción de la varianza en la demanda eléctrica que es predecible a partir de las variables de entrada. Un valor cercano a $1.0$ (o $100\%$) representa un ajuste óptimo.

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

Donde $\bar{y}$ es la media de los valores reales.

---

## 5. Resultados y Comparación del Rendimiento

Tras entrenar los modelos con datos del periodo **2022 - 2025** y evaluarlos sobre la ventana de validación temporal (último 20% del histórico), se obtuvieron las siguientes métricas de rendimiento:

| Modelo | MAE (kW) | RMSE (kW) | MAPE (%) | $R^2$ (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Prophet** | 896,652.48 | 1,009,762.04 | 9.33% | 18.38% |
| **Random Forest** | 58,698.99 | 87,830.02 | 0.61% | 99.38% |
| **XGBoost** | **56,366.53** | **77,671.25** | **0.59%** | **99.52%** |

### Análisis de Resultados Académicos

1. **Modelos Autorregresivos Tabulares (XGBoost y Random Forest):**
   - Ambos modelos logran un ajuste casi perfecto ($R^2 > 99\%$) y un error relativo inferior al $0.62\%$.
   - Este rendimiento excepcional se explica por la inclusión de variables autorregresivas de alta correlación: `lag_1` (demanda de la hora previa) y `lag_24` (demanda del día anterior). Al conocer el nivel inmediato anterior de carga del SIN, el modelo ajusta con precisión milimétrica la predicción de la siguiente hora.
   - **XGBoost** supera ligeramente a **Random Forest**, logrando reducir el error promedio absoluto en aproximadamente **2,332 kW**.

2. **Prophet (Modelo de Tendencia y Estacionalidad):**
   - Muestra un ajuste significativamente inferior ($R^2 = 18.38\%$) y un error de **9.33%**.
   - **Explicación técnica:** Prophet no utiliza variables rezagadas a nivel horario. Se enfoca estrictamente en modelar componentes macro (tendencia global y curvas continuas de estacionalidad diaria, semanal y anual). Por lo tanto, no es reactivo a perturbaciones imprevistas de la demanda hora a hora (como apagones o variaciones industriales rápidas), pero resulta sumamente estable.
   - En escenarios de predicción recursiva a largo plazo (2026-2027), los modelos tabulares sufren del problema de acumulación de error (ya que usan predicciones para calcular los futuros `lag_1`), mientras que Prophet provee una línea base de tendencia sumamente consistente y libre de divergencias numéricas.

