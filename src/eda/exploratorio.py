"""Análisis exploratorio automático para demanda eléctrica horaria del SIN.

El módulo construye y guarda un paquete de figuras en Plotly para facilitar la
exploración de una serie horaria ya procesada. Las visualizaciones generadas son:

- Serie temporal histórica.
- Tendencia anual.
- Tendencia mensual.
- Demanda promedio por hora del día.
- Demanda promedio por día de la semana.
- Heatmap hora vs día de semana.
- Distribución de la demanda.
- Detección de outliers con regla IQR.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


FIGURES_DIRNAME = "reports/figures"
DEFAULT_OUTPUT_PREFIX = "eda_demanda"

DATE_CANDIDATES = (
    "fecha_hora",
    "fecha",
    "timestamp",
    "datetime",
    "date",
)
DEMAND_CANDIDATES = (
    "demanda",
    "demanda_mw",
    "demanda_kw",
    "demanda_kwh",
    "valor",
    "consumo",
    "energia",
    "value",
)

DAY_ORDER = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
DAY_MAP = {
    0: "Lunes",
    1: "Martes",
    2: "Miércoles",
    3: "Jueves",
    4: "Viernes",
    5: "Sábado",
    6: "Domingo",
}


@dataclass(slots=True)
class EDAResult:
    """Resultado del análisis exploratorio y rutas de las figuras generadas."""

    figures: dict[str, go.Figure]
    saved_files: dict[str, Path]
    outliers: pd.DataFrame


def generate_exploratory_analysis(
    data: pd.DataFrame,
    output_dir: str | Path | None = None,
    output_prefix: str = DEFAULT_OUTPUT_PREFIX,
) -> EDAResult:
    """Genera automáticamente el paquete de análisis exploratorio.

    Parameters
    ----------
    data:
        ``DataFrame`` con la demanda eléctrica ya procesada y variables
        temporales disponibles.
    output_dir:
        Carpeta donde se guardarán las figuras. Si no se especifica, se usa
        ``reports/figures/``.
    output_prefix:
        Prefijo para nombrar los archivos de salida.

    Returns
    -------
    EDAResult
        Diccionario de figuras, rutas de archivos guardados y observaciones de
        outliers detectados.
    """

    working = _prepare_frame(data)
    datetime_column = _infer_datetime_column(working)
    demand_column = _infer_demand_column(working)

    if datetime_column is None:
        raise ValueError("No se encontró una columna temporal para el análisis exploratorio.")
    if demand_column is None:
        raise ValueError("No se encontró una columna de demanda para el análisis exploratorio.")

    transformed = working.copy()
    transformed[datetime_column] = pd.to_datetime(transformed[datetime_column], errors="coerce")
    transformed[demand_column] = pd.to_numeric(transformed[demand_column], errors="coerce")
    transformed = transformed.dropna(subset=[datetime_column, demand_column]).sort_values(datetime_column).reset_index(drop=True)

    figures = {
        "serie_temporal": build_historical_time_series(transformed, datetime_column, demand_column),
        "tendencia_anual": build_yearly_trend(transformed, datetime_column, demand_column),
        "tendencia_mensual": build_monthly_trend(transformed, datetime_column, demand_column),
        "promedio_hora": build_hourly_average(transformed, datetime_column, demand_column),
        "promedio_dia_semana": build_weekday_average(transformed, datetime_column, demand_column),
        "heatmap_hora_dia": build_hour_weekday_heatmap(transformed, datetime_column, demand_column),
        "distribucion": build_demand_distribution(transformed, demand_column),
    }

    outliers, outlier_figure = detect_outliers(transformed, datetime_column, demand_column)
    figures["outliers"] = outlier_figure

    target_dir = Path(output_dir) if output_dir is not None else _project_root() / FIGURES_DIRNAME
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_files = _save_figures(figures, target_dir, output_prefix)
    outliers_path = target_dir / f"{output_prefix}_outliers.csv"
    outliers.to_csv(outliers_path, index=False)
    saved_files["outliers_csv"] = outliers_path

    return EDAResult(figures=figures, saved_files=saved_files, outliers=outliers)


def build_historical_time_series(
    frame: pd.DataFrame,
    datetime_column: str,
    demand_column: str,
) -> go.Figure:
    """Construye la serie temporal histórica de la demanda."""

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame[datetime_column],
            y=frame[demand_column],
            mode="lines",
            name="Demanda",
            line=dict(color="#1f77b4", width=1.8),
        )
    )
    fig.update_layout(
        title="Serie temporal histórica de la demanda",
        xaxis_title="Fecha y hora",
        yaxis_title="Demanda",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def build_yearly_trend(
    frame: pd.DataFrame,
    datetime_column: str,
    demand_column: str,
) -> go.Figure:
    """Calcula y grafica la tendencia anual promedio de la demanda."""

    working = frame.copy()
    working["anio"] = pd.to_datetime(working[datetime_column], errors="coerce").dt.year
    annual = working.groupby("anio", as_index=False)[demand_column].mean()

    fig = px.line(
        annual,
        x="anio",
        y=demand_column,
        markers=True,
        title="Tendencia anual de la demanda promedio",
    )
    fig.update_traces(line=dict(color="#2ca02c", width=2))
    fig.update_layout(template="plotly_white", xaxis_title="Año", yaxis_title="Demanda promedio")
    return fig


def build_monthly_trend(
    frame: pd.DataFrame,
    datetime_column: str,
    demand_column: str,
) -> go.Figure:
    """Calcula y grafica la tendencia mensual promedio de la demanda."""

    working = frame.copy()
    monthly = (
        working.assign(periodo=pd.to_datetime(working[datetime_column], errors="coerce").dt.to_period("M"))
        .groupby("periodo", as_index=False)[demand_column]
        .mean()
    )
    monthly["periodo"] = monthly["periodo"].astype(str)

    fig = px.line(
        monthly,
        x="periodo",
        y=demand_column,
        markers=True,
        title="Tendencia mensual de la demanda promedio",
    )
    fig.update_traces(line=dict(color="#ff7f0e", width=2))
    fig.update_layout(template="plotly_white", xaxis_title="Mes", yaxis_title="Demanda promedio")
    return fig


def build_hourly_average(
    frame: pd.DataFrame,
    datetime_column: str,
    demand_column: str,
) -> go.Figure:
    """Grafica la demanda promedio por hora del día."""

    working = frame.copy()
    working["hora"] = pd.to_datetime(working[datetime_column], errors="coerce").dt.hour
    hourly = working.groupby("hora", as_index=False)[demand_column].mean().sort_values("hora")

    fig = px.bar(
        hourly,
        x="hora",
        y=demand_column,
        title="Demanda promedio por hora del día",
        labels={"hora": "Hora", demand_column: "Demanda promedio"},
    )
    fig.update_traces(marker_color="#9467bd")
    fig.update_layout(template="plotly_white")
    return fig


def build_weekday_average(
    frame: pd.DataFrame,
    datetime_column: str,
    demand_column: str,
) -> go.Figure:
    """Grafica la demanda promedio por día de la semana."""

    working = frame.copy()
    working["dia_semana"] = pd.to_datetime(working[datetime_column], errors="coerce").dt.dayofweek
    weekday = (
        working.groupby("dia_semana", as_index=False)[demand_column]
        .mean()
        .sort_values("dia_semana")
    )
    weekday["dia_semana_nombre"] = weekday["dia_semana"].map(DAY_MAP)

    fig = px.bar(
        weekday,
        x="dia_semana_nombre",
        y=demand_column,
        title="Demanda promedio por día de la semana",
        labels={"dia_semana_nombre": "Día de la semana", demand_column: "Demanda promedio"},
    )
    fig.update_traces(marker_color="#d62728")
    fig.update_layout(template="plotly_white")
    return fig


def build_hour_weekday_heatmap(
    frame: pd.DataFrame,
    datetime_column: str,
    demand_column: str,
) -> go.Figure:
    """Construye un heatmap de demanda promedio por hora y día de semana."""

    working = frame.copy()
    working["hora"] = pd.to_datetime(working[datetime_column], errors="coerce").dt.hour
    working["dia_semana"] = pd.to_datetime(working[datetime_column], errors="coerce").dt.dayofweek

    pivot = (
        working.pivot_table(
            index="dia_semana",
            columns="hora",
            values=demand_column,
            aggfunc="mean",
        )
        .reindex(index=range(7), columns=range(24))
    )
    pivot.index = [DAY_MAP.get(index, str(index)) for index in pivot.index]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(hour) for hour in pivot.columns],
            y=pivot.index.tolist(),
            colorscale="Viridis",
            colorbar=dict(title="Demanda promedio"),
        )
    )
    fig.update_layout(
        title="Heatmap de demanda promedio: hora vs día de semana",
        xaxis_title="Hora del día",
        yaxis_title="Día de la semana",
        template="plotly_white",
    )
    return fig


def build_demand_distribution(frame: pd.DataFrame, demand_column: str) -> go.Figure:
    """Genera la distribución de la demanda con histograma y boxplot."""

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("Distribución de la demanda", "Resumen tipo boxplot"),
    )

    fig.add_trace(
        go.Histogram(
            x=frame[demand_column],
            nbinsx=50,
            name="Histograma",
            marker_color="#17becf",
            opacity=0.85,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Box(
            x=frame[demand_column],
            name="Boxplot",
            marker_color="#17becf",
            boxmean=True,
            orientation="h",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_layout(template="plotly_white", bargap=0.05, title="Distribución de la demanda")
    return fig


def detect_outliers(
    frame: pd.DataFrame,
    datetime_column: str,
    demand_column: str,
) -> tuple[pd.DataFrame, go.Figure]:
    """Detecta outliers usando la regla del rango intercuartílico (IQR).

    Returns
    -------
    tuple[pd.DataFrame, go.Figure]
        Un ``DataFrame`` con los puntos atípicos y una figura Plotly con la serie
        histórica resaltando dichos puntos.
    """

    q1 = frame[demand_column].quantile(0.25)
    q3 = frame[demand_column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (frame[demand_column] < lower_bound) | (frame[demand_column] > upper_bound)
    outliers = frame.loc[outlier_mask, [datetime_column, demand_column]].copy()
    outliers["limite_inferior"] = lower_bound
    outliers["limite_superior"] = upper_bound

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame[datetime_column],
            y=frame[demand_column],
            mode="lines",
            name="Demanda",
            line=dict(color="#7f7f7f", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=outliers[datetime_column],
            y=outliers[demand_column],
            mode="markers",
            name="Outliers",
            marker=dict(color="#ff0000", size=7, symbol="circle-open"),
        )
    )
    fig.add_hrect(y0=lower_bound, y1=upper_bound, fillcolor="rgba(44,160,44,0.12)", line_width=0)
    fig.update_layout(
        title="Detección de outliers por regla IQR",
        xaxis_title="Fecha y hora",
        yaxis_title="Demanda",
        template="plotly_white",
        hovermode="x unified",
    )
    return outliers.reset_index(drop=True), fig


def _save_figures(figures: dict[str, go.Figure], output_dir: Path, output_prefix: str) -> dict[str, Path]:
    """Guarda las figuras en disco y devuelve el mapa de rutas creadas."""

    saved_files: dict[str, Path] = {}
    for name, figure in figures.items():
        file_path = output_dir / f"{output_prefix}_{name}.html"
        figure.write_html(file_path, include_plotlyjs="cdn")
        saved_files[name] = file_path
    return saved_files


def _prepare_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Prepara el ``DataFrame`` para inferir columnas temporales y de demanda."""

    if data.empty:
        raise ValueError("El DataFrame de entrada está vacío.")
    working = data.copy()
    working.columns = [_normalize_column_name(column) for column in working.columns]
    return working


def _infer_datetime_column(frame: pd.DataFrame) -> str | None:
    """Infiera la columna temporal más probable."""

    for candidate in DATE_CANDIDATES:
        for column in frame.columns:
            if candidate in column:
                return column
    return None


def _infer_demand_column(frame: pd.DataFrame) -> str | None:
    """Infiera la columna de demanda más probable."""

    for candidate in DEMAND_CANDIDATES:
        for column in frame.columns:
            if candidate in column:
                return column

    numeric_columns = list(frame.select_dtypes(include=["number"]).columns)
    if numeric_columns:
        return numeric_columns[-1]
    return None


def _normalize_column_name(column: str) -> str:
    """Normaliza nombres de columna para facilitar el reconocimiento automático."""

    normalized = (
        column.strip()
        .lower()
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ñ", "n")
    )
    for separator in (" ", "-", ".", "/", "(", ")", "[", "]"):
        normalized = normalized.replace(separator, "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _project_root() -> Path:
    """Resuelve la raíz del proyecto a partir de la ubicación del módulo."""

    return Path(__file__).resolve().parents[2]
