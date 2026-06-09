# Análisis de Datos de Entrenamiento y Validación de Fuente de Verdad

Este documento presenta una auditoría técnica del conjunto de datos utilizado para entrenar los modelos predictivos del proyecto **"PREDICCIÓN DE LA DEMANDA ELÉCTRICA DEL SISTEMA INTERCONECTADO NACIONAL MEDIANTE TÉCNICAS DE MACHINE LEARNING"** y explica por qué existen discrepancias naturales entre las predicciones autoregresivas y las proyecciones macro del sector eléctrico.

---

## 1. Caracterización del Dataset de Entrenamiento

Los datos de entrenamiento fueron extraídos del portal de datos abiertos de **XM S.A. E.S.P.** (administrador del mercado de energía mayorista en Colombia).

### Ficha Técnica de los Datos
- **Periodo Temporal:** 1 de enero de 2022 a 31 de diciembre de 2025.
- **Granularidad:** Horaria (24 lecturas diarias).
- **Número Total de Registros:** 35,062 registros horarios.
- **Variable Objetivo:** Demanda de energía real medida en kilovatios (kW).
- **Calidad de Datos:** 
  - **Valores Faltantes:** 0.00% (No se presentan nulos en la serie principal).
  - **Registros Atípicos (Outliers):** 0.00% detectados bajo criterio IQR estricto, indicando una serie limpia y libre de ruidos atípicos severos o fallas catastróficas de medición.

---

## 2. Diferencias entre Predicciones de Machine Learning y Proyecciones Oficiales (UPME / XM)

Es normal no encontrar una similitud exacta entre nuestras predicciones a largo plazo (2026-2027) y las proyecciones publicadas por la **Unidad de Planeación Minero Energética (UPME)** o **XM**. Esto se debe a la diferencia fundamental en la **metodología de formulación**:

### A. Metodología Oficial (UPME / XM)
Las proyecciones oficiales de la UPME se basan en modelos estructurados **socioeconómicos y multivariados**:
- **Crecimiento del PIB:** Correlaciona la demanda con el crecimiento económico del país.
- **Crecimiento Demográfico:** Proyecciones de población del DANE.
- **Proyectos Industriales:** Incorpora la entrada de grandes consumidores industriales planeados (minería, transporte masivo, etc.).
- **Escenarios Climáticos:** Fenómenos de variabilidad climática como El Niño o La Niña.

### B. Metodología de Nuestro Proyecto (Machine Learning Autoregresivo)
Nuestros modelos (Random Forest y XGBoost) son **univariados autoregresivos**:
- **Sin Exógenas Macroeconómicas:** El modelo no conoce el PIB, el clima ni el crecimiento demográfico futuro.
- **Dependencia Histórica (Lags):** Predecimos la hora $t$ basándonos en la hora $t-1$, $t-24$ y $t-168$.
- **Propagación de Tendencias Pasadas:** Proyecta la inercia del consumo eléctrico del periodo 2022-2025 hacia el 2026 y 2027.
- **Efecto Cascada:** Al proyectar de forma recursiva a largo plazo, el error residual de la predicción previa se acumula en los siguientes pasos, lo que tiende a "estabilizar" o "atenuar" las curvas estacionales en lugar de mostrar picos alcistas agresivos como los escenarios de crecimiento del PIB de la UPME.

---

## 3. Variables Empleadas en el Entrenamiento (Features)

Para validar el valor de verdad del modelo, se detalla qué información consume para predecir:

| Variable | Tipo | Descripción | Justificación Física |
| :--- | :--- | :--- | :--- |
| `hora` | Cíclica | Hora del día ($0 - 23$) | Representa la curva de carga diaria (valle a las 4:00, pico a las 20:00). |
| `dia_semana` | Cíclica | Día de la semana ($0 - 6$) | Captura la caída de consumo los sábados y domingos por inactividad comercial. |
| `es_festivo` | Binaria | Si el día es festivo oficial | Colombia tiene 18+ festivos al año; causan caídas de demanda de hasta el 13.19%. |
| `lag_1` | Numérica | Consumo en la hora anterior ($t-1$) | Captura la correlación inmediata (el consumo no cambia drásticamente en 1 hora). |
| `lag_24` | Numérica | Consumo hace 24 horas ($t-24$) | Captura la correlación diaria (el lunes a las 10:00 se parece al martes a las 10:00). |
| `lag_168` | Numérica | Consumo hace 1 semana ($t-168$) | Captura la correlación semanal (este domingo a las 15:00 se parece al domingo anterior). |
| `rolling_mean_24` | Numérica | Media móvil de las últimas 24 horas | Suaviza el ruido y da el nivel base de carga diaria. |
