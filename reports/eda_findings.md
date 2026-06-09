# Informe de Hallazgos y Conclusiones del Análisis Exploratorio (EDA)

Este reporte recopila los hallazgos cuantitativos del análisis del Sistema Interconectado Nacional (SIN) obtenidos directamente del dataset procesado para el periodo **2022 - 2025**.

---

## 1. Comportamiento y Estadísticas Generales

*   **Punto de Demanda Máxima Histórica:** Se registró un valor de **11,977,759.16 kW** el **2025-12-11** a las **20:00**.
*   **Punto de Demanda Mínima Histórica:** Se registró un valor de **5,788,415.17 kW** el **2022-01-01** a las **7:00**.
*   **Media General de Consumo:** El consumo promedio del SIN es de **9,187,981.97 kW**.

---

## 2. Estacionalidad y Patrones de Consumo

### 2.1. Estacionalidad Horaria (Ciclo Diario)
*   **Hora Pico Promedio:** La demanda máxima diaria ocurre típicamente a las **20:00**, impulsada por el encendido del alumbrado público y el regreso al hogar de los usuarios (demanda residencial).
*   **Hora Valle Promedio:** El mínimo de demanda ocurre a las **4:00**, correspondiente al periodo de inactividad de la madrugada.

### 2.2. Efecto de Calendario (Días Hábiles vs Festivos)
*   **Efecto Fin de Semana:** El promedio en días de fin de semana (**8,640,399.32 kW**) es un **8.16% menor** en comparación con los días hábiles (**9,407,973.83 kW**). Esto refleja la parálisis de la demanda de carga industrial y comercial pesada.
*   **Efecto Días Festivos:** Los días festivos oficiales en Colombia experimentan una caída del **13.19%** en promedio frente a un día laborable normal, demostrando que los asuetos tienen un comportamiento similar al de un domingo.

### 2.3. Tendencia Interanual e Intermensual
- **2022 vs 2023:** Crecimiento promedio del **5.47%**
- **2023 vs 2024:** Crecimiento promedio del **2.49%**
- **2024 vs 2025:** Crecimiento promedio del **1.63%**

*   **Meses con Mayor Demanda:** El consumo tiende a incrementarse en los periodos de verano y meses festivos debido al aumento de temperaturas en la región Caribe (aire acondicionado) y alumbrados navideños en diciembre.

---

## 3. Calidad de Datos y Datos Atípicos (Outliers)

*   **Presencia de Outliers:** Utilizando el método del Rango Intercuartílico (IQR), identificamos **0 registros atípicos**, que equivalen al **0.00%** del total del dataset.
*   **Interpretación:** La ausencia de valores atípicos (0.00%) confirma la alta consistencia y calidad del dataset de demanda. No se registran lecturas nulas ni caídas catastróficas fuera del rango de operación normal del sistema (el cual oscila de manera sumamente estable entre los 5.8M kW y los 12M kW).
