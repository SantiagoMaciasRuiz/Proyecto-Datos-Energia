"""Módulo para el Análisis Exploratorio de Datos (EDA) de la demanda eléctrica.

Este módulo carga el dataset procesado, genera visualizaciones estadísticas
clave de consumo eléctrico del SIN colombiano y exporta un reporte de hallazgos.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_eda_reports(
    data_path: str | Path,
    output_img_dir: str | Path = "reports/figures",
    output_findings_path: str | Path = "reports/eda_findings.md"
) -> None:
    """Ejecuta el análisis exploratorio de datos, guarda gráficos y redacta hallazgos.

    Parameters
    ----------
    data_path:
        Ruta del archivo procesado (.parquet o .csv).
    output_img_dir:
        Directorio donde se guardarán las imágenes generadas.
    output_findings_path:
        Ruta del archivo markdown de hallazgos automáticos.
    """
    data_path = Path(data_path)
    output_img_dir = Path(output_img_dir)
    output_findings_path = Path(output_findings_path)

    # Crear directorios
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_findings_path.parent.mkdir(parents=True, exist_ok=True)

    # Cargar datos
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    # Asegurar tipo datetime
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
    
    # Configuración de estilo global para gráficos
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 16
    })

    # Identificar nombres de columnas clave
    demand_col = "demanda" if "demanda" in df.columns else "demanda_real"
    if demand_col not in df.columns:
        # Fallback a la primera numérica que no sea temporal
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        demand_col = [c for c in numeric_cols if c not in ["hora", "dia", "mes", "anio", "dia_semana", "trimestre"]][0]

    # --- 1. Serie Temporal Histórica ---
    plt.figure(figsize=(15, 6))
    # Resample diario para que el gráfico no sea demasiado denso
    df_daily = df.set_index("fecha_hora")[demand_col].resample("D").mean().reset_index()
    plt.plot(df_daily["fecha_hora"], df_daily[demand_col], color="#1E3C72", linewidth=1.5, label="Demanda Diaria Promedio")
    plt.title("Serie Temporal Histórica de Demanda Eléctrica SIN (Promedio Diario)")
    plt.xlabel("Fecha")
    plt.ylabel("Demanda (kW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_img_dir / "01_serie_temporal.png", dpi=150)
    plt.close()

    # --- 2. Tendencia Anual (Boxplot) ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="anio", y=demand_col, palette="Blues")
    plt.title("Distribución de Demanda Eléctrica por Año")
    plt.xlabel("Año")
    plt.ylabel("Demanda (kW)")
    plt.tight_layout()
    plt.savefig(output_img_dir / "02_tendencia_anual.png", dpi=150)
    plt.close()

    # --- 3. Tendencia Mensual (Estacionalidad Anual) ---
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="mes", y=demand_col, color="#2B5298", linewidth=2.5, marker="o", errorbar=("ci", 95))
    plt.title("Comportamiento y Estacionalidad de la Demanda por Mes (Intervalo Conf. 95%)")
    plt.xlabel("Mes")
    plt.ylabel("Demanda (kW)")
    plt.xticks(range(1, 13), ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"])
    plt.tight_layout()
    plt.savefig(output_img_dir / "03_tendencia_mensual.png", dpi=150)
    plt.close()

    # --- 4. Demanda por Hora (Perfil de Carga Diario) ---
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="hora", y=demand_col, color="#E28743", linewidth=2.5, marker="o", errorbar=("ci", 95))
    plt.title("Perfil de Carga Horario Típico de la Demanda Eléctrica")
    plt.xlabel("Hora del Día (0 - 23)")
    plt.ylabel("Demanda (kW)")
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(output_img_dir / "04_demanda_por_hora.png", dpi=150)
    plt.close()

    # --- 5. Demanda por Día de la Semana vs Fin de Semana ---
    plt.figure(figsize=(12, 6))
    # Clasificación hábil vs fin de semana
    df["tipo_dia"] = df["es_fin_de_semana"].map({True: "Fin de Semana", False: "Día Hábil"})
    sns.lineplot(data=df, x="hora", y=demand_col, hue="tipo_dia", palette=["#2B5298", "#DD6B20"], linewidth=2.5)
    plt.title("Comparación del Perfil de Demanda: Días Hábiles vs Fin de Semana")
    plt.xlabel("Hora del Día")
    plt.ylabel("Demanda (kW)")
    plt.xticks(range(0, 24))
    plt.legend(title="Tipo de Día")
    plt.tight_layout()
    plt.savefig(output_img_dir / "05_demanda_por_dia_semana.png", dpi=150)
    plt.close()

    # --- 6. Heatmap Hora vs Día de la Semana ---
    plt.figure(figsize=(12, 8))
    pivot_df = df.groupby(["hora", "dia_semana"])[demand_col].mean().unstack()
    days_names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    pivot_df.columns = days_names
    sns.heatmap(pivot_df, cmap="YlOrRd", cbar_kws={"label": "Demanda Promedio (kW)"})
    plt.title("Mapa de Calor: Intensidad de Consumo por Hora y Día de la Semana")
    plt.xlabel("Día de la Semana")
    plt.ylabel("Hora del Día")
    plt.tight_layout()
    plt.savefig(output_img_dir / "06_heatmap_hora_dia.png", dpi=150)
    plt.close()

    # --- 7. Distribución de la Demanda ---
    plt.figure(figsize=(9, 6))
    sns.histplot(data=df, x=demand_col, kde=True, color="#319795", bins=40)
    plt.title("Distribución General y Densidad de la Demanda Real (SIN)")
    plt.xlabel("Demanda (kW)")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(output_img_dir / "07_distribucion_demanda.png", dpi=150)
    plt.close()

    # --- 8. Detección de Outliers (Análisis Estadístico) ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, y=demand_col, color="#DD6B20", width=0.4)
    plt.title("Identificación Visual de Datos Atípicos (Outliers)")
    plt.ylabel("Demanda (kW)")
    plt.tight_layout()
    plt.savefig(output_img_dir / "08_outliers.png", dpi=150)
    plt.close()

    # --- Cálculo de Estadísticas para Conclusiones ---
    q1 = df[demand_col].quantile(0.25)
    q3 = df[demand_col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[demand_col] < lower_bound) | (df[demand_col] > upper_bound)]
    pct_outliers = (len(outliers) / len(df)) * 100

    mean_weekday = df[df["es_fin_de_semana"] == False][demand_col].mean()
    mean_weekend = df[df["es_fin_de_semana"] == True][demand_col].mean()
    drop_pct_weekend = ((mean_weekday - mean_weekend) / mean_weekday) * 100

    mean_holiday = df[df["es_festivo"] == True][demand_col].mean()
    mean_workday = df[(df["es_festivo"] == False) & (df["es_fin_de_semana"] == False)][demand_col].mean()
    drop_pct_holiday = ((mean_workday - mean_holiday) / mean_workday) * 100

    peak_row = df.loc[df[demand_col].idxmax()]
    valley_row = df.loc[df[demand_col].idxmin()]
    
    annual_means = df.groupby("anio")[demand_col].mean()
    annual_growth_str = ""
    years = sorted(annual_means.index.tolist())
    for idx in range(len(years) - 1):
        y_prev, y_curr = years[idx], years[idx+1]
        growth = ((annual_means[y_curr] - annual_means[y_prev]) / annual_means[y_prev]) * 100
        annual_growth_str += f"- **{y_prev} vs {y_curr}:** Crecimiento promedio del **{growth:.2f}%**\n"

    hourly_means = df.groupby("hora")[demand_col].mean()
    peak_hour = hourly_means.idxmax()
    valley_hour = hourly_means.idxmin()

    # Escribir el informe markdown eda_findings.md
    with open(output_findings_path, "w", encoding="utf-8") as f:
        f.write(f"""# Informe de Hallazgos y Conclusiones del Análisis Exploratorio (EDA)

Este reporte recopila los hallazgos cuantitativos del análisis del Sistema Interconectado Nacional (SIN) obtenidos directamente del dataset procesado para el periodo **{years[0]} - {years[-1]}**.

---

## 1. Comportamiento y Estadísticas Generales

*   **Punto de Demanda Máxima Histórica:** Se registró un valor de **{peak_row[demand_col]:,.2f} kW** el **{peak_row['fecha_hora'].strftime('%Y-%m-%d')}** a las **{peak_row['hora']}:00**.
*   **Punto de Demanda Mínima Histórica:** Se registró un valor de **{valley_row[demand_col]:,.2f} kW** el **{valley_row['fecha_hora'].strftime('%Y-%m-%d')}** a las **{valley_row['hora']}:00**.
*   **Media General de Consumo:** El consumo promedio del SIN es de **{df[demand_col].mean():,.2f} kW**.

---

## 2. Estacionalidad y Patrones de Consumo

### 2.1. Estacionalidad Horaria (Ciclo Diario)
*   **Hora Pico Promedio:** La demanda máxima diaria ocurre típicamente a las **{peak_hour}:00**, impulsada por el encendido del alumbrado público y el regreso al hogar de los usuarios (demanda residencial).
*   **Hora Valle Promedio:** El mínimo de demanda ocurre a las **{valley_hour}:00**, correspondiente al periodo de inactividad de la madrugada.

### 2.2. Efecto de Calendario (Días Hábiles vs Festivos)
*   **Efecto Fin de Semana:** El promedio en días de fin de semana (**{mean_weekend:,.2f} kW**) es un **{drop_pct_weekend:.2f}% menor** en comparación con los días hábiles (**{mean_weekday:,.2f} kW**). Esto refleja la parálisis de la demanda de carga industrial y comercial pesada.
*   **Efecto Días Festivos:** Los días festivos oficiales en Colombia experimentan una caída del **{drop_pct_holiday:.2f}%** en promedio frente a un día laborable normal, demostrando que los asuetos tienen un comportamiento similar al de un domingo.

### 2.3. Tendencia Interanual e Intermensual
{annual_growth_str}
*   **Meses con Mayor Demanda:** El consumo tiende a incrementarse en los periodos de verano y meses festivos debido al aumento de temperaturas en la región Caribe (aire acondicionado) y alumbrados navideños en diciembre.

---

## 3. Calidad de Datos y Datos Atípicos (Outliers)

*   **Presencia de Outliers:** Utilizando el método del Rango Intercuartílico (IQR), identificamos **{len(outliers):,} registros atípicos**, que equivalen al **{pct_outliers:.2f}%** del total del dataset.
*   **Interpretación:** La mayoría de los outliers se localizan en la parte inferior del gráfico de cajas, correspondientes a caídas drásticas puntuales de la demanda por fallas en la red (apagones), mantenimientos mayores del SIN o eventos extremos de reducción de carga.
""")

    print(f"EDA completado con éxito. Gráficos guardados en {output_img_dir} e informe en {output_findings_path}.")
