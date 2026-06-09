"""Dashboard Streamlit para demanda eléctrica del SIN colombiano.

La aplicación se organiza en 4 páginas accesibles desde ``st.sidebar``:

1. Resumen ejecutivo con KPIs.
2. Análisis histórico con gráficas interactivas Plotly.
3. Predicciones con comparación real vs. predicho y horizonte futuro.
4. Conclusiones.

El diseño usa una paleta oscura, layout amplio y componentes visuales pensados
para lectura ejecutiva y exploración analítica.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_TITLE = "Predicción de Demanda SIN"
APP_ICON = "⚡"
DEFAULT_DATA_FILENAMES = (
    "demanda_electrica_transformada.parquet",
    "demanda_electrica_transformada.csv",
)
DEFAULT_PREDICTION_FILENAMES = (
    "predicciones.csv",
    "comparison_predictions.csv",
    "forecast.csv",
    "forecast_comparison.csv",
)
DEFAULT_MODEL_RESULTS_FILENAMES = (
    "model_results.csv",
    "metrics.csv",
    "model_metrics.csv",
)

MODEL_DISPLAY_NAMES = {
    "prophet": "Prophet",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def main() -> None:
    """Inicializa la app y despacha la vista seleccionada en la barra lateral."""

    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
    _inject_dark_theme()

    st.sidebar.title("Navegación")
    page = st.sidebar.radio(
        "Selecciona una vista",
        [
            "Resumen ejecutivo",
            "Análisis histórico",
            "Predicciones",
            "Conclusiones",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Proyecto: demanda eléctrica del SIN colombiano")
    st.sidebar.caption(f"Directorio raíz: {_project_root()}")

    if page == "Resumen ejecutivo":
        render_executive_summary()
    elif page == "Análisis histórico":
        render_historical_analysis()
    elif page == "Predicciones":
        render_predictions()
    else:
        render_conclusions()


def render_executive_summary() -> None:
    """Renderiza la página de resumen ejecutivo con KPIs clave."""

    st.title("Resumen ejecutivo")
    st.write(
        "Vista de alto nivel para evaluar el comportamiento histórico de la demanda y el desempeño general del sistema de predicción."
    )

    data = load_processed_data()
    if data.empty:
        st.info("No se encontró un dataset procesado en data/processed/. La vista mostrará KPIs de ejemplo hasta que exista el archivo final.")
        data = _demo_frame()

    datetime_column = infer_datetime_column(data)
    demand_column = infer_demand_column(data)

    if datetime_column is None or demand_column is None:
        st.error("No fue posible identificar las columnas temporales o de demanda.")
        return

    working = _prepare_dataset(data, datetime_column, demand_column)
    kpi_values = calculate_kpis(working, datetime_column, demand_column)

    cols = st.columns(4)
    cols[0].metric("Demanda promedio", f"{kpi_values['avg']:.2f}")
    cols[1].metric("Demanda máxima", f"{kpi_values['max']:.2f}")
    cols[2].metric("Demanda mínima", f"{kpi_values['min']:.2f}")
    cols[3].metric("Crecimiento anual", f"{kpi_values['growth']:.2f}%")

    left, right = st.columns((1.2, 1))
    with left:
        st.subheader("Tendencia reciente")
        trend = build_trend_figure(working, datetime_column, demand_column)
        st.plotly_chart(trend, use_container_width=True)

    with right:
        st.subheader("Distribución operativa")
        distribution = build_distribution_figure(working, demand_column)
        st.plotly_chart(distribution, use_container_width=True)

    st.subheader("Resumen contextual")
    context_cols = st.columns(3)
    context_cols[0].info(f"Registros disponibles: {len(working):,}")
    context_cols[1].info(f"Cobertura temporal: {working[datetime_column].min()} a {working[datetime_column].max()}")
    context_cols[2].info(f"Años analizados: {working[datetime_column].dt.year.nunique()}")


def render_historical_analysis() -> None:
    """Renderiza el análisis histórico con múltiples visualizaciones Plotly."""

    st.title("Análisis histórico")
    st.write("Explora patrones de demanda, estacionalidad y comportamiento operativo del SIN colombiano.")

    data = load_processed_data()
    if data.empty:
        data = _demo_frame()

    datetime_column = infer_datetime_column(data)
    demand_column = infer_demand_column(data)
    if datetime_column is None or demand_column is None:
        st.error("No se encontraron columnas válidas para fecha y demanda.")
        return

    working = _prepare_dataset(data, datetime_column, demand_column)

    fig_time = build_trend_figure(working, datetime_column, demand_column)
    fig_hour = build_hour_profile_figure(working, datetime_column, demand_column)
    fig_weekday = build_weekday_profile_figure(working, datetime_column, demand_column)
    fig_heatmap = build_heatmap_figure(working, datetime_column, demand_column)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_time, use_container_width=True)
        st.plotly_chart(fig_hour, use_container_width=True)
    with c2:
        st.plotly_chart(fig_weekday, use_container_width=True)
        st.plotly_chart(fig_heatmap, use_container_width=True)


def render_predictions() -> None:
    """Renderiza la página de comparación real vs predicho y horizonte futuro."""

    st.title("Predicciones")
    st.write("Compara el valor real con las predicciones de los modelos y revisa el horizonte futuro disponible.")

    data = load_processed_data()
    if data.empty:
        data = _demo_frame()

    predictions = load_prediction_data()
    model_results = load_model_results()

    datetime_column = infer_datetime_column(data)
    demand_column = infer_demand_column(data)
    if datetime_column is None or demand_column is None:
        st.error("No se encontraron columnas válidas para fecha y demanda.")
        return

    working = _prepare_dataset(data, datetime_column, demand_column)

    st.subheader("Comparación real vs predicho")
    if predictions.empty:
        st.warning("No se encontraron archivos de predicciones en models/ o reports/. Se mostrará una comparación de ejemplo.")
        prediction_frame = build_demo_predictions(working, datetime_column, demand_column)
    else:
        prediction_frame = normalize_prediction_frame(predictions, datetime_column, demand_column)

    compare_fig = build_real_vs_predicted_figure(prediction_frame, datetime_column, demand_column)
    st.plotly_chart(compare_fig, use_container_width=True)

    st.subheader("Comparación entre modelos")
    if model_results.empty:
        model_results = build_demo_model_results()
    model_bar = build_model_comparison_figure(model_results)
    st.plotly_chart(model_bar, use_container_width=True)

    st.subheader("Horizonte futuro")
    horizon_fig = build_future_horizon_figure(prediction_frame, datetime_column)
    st.plotly_chart(horizon_fig, use_container_width=True)

    cols = st.columns(3)
    cols[0].metric("Modelos detectados", f"{prediction_frame['modelo'].nunique():,}")
    cols[1].metric("Observaciones comparadas", f"{len(prediction_frame):,}")
    cols[2].metric("Cobertura futura", f"{count_future_rows(prediction_frame):,}")


def render_conclusions() -> None:
    """Renderiza conclusiones ejecutivas y recomendaciones de operación."""

    st.title("Conclusiones")
    st.write("Síntesis ejecutiva del comportamiento de la demanda y del valor agregado del sistema de predicción.")

    st.markdown(
        """
        ### Hallazgos esperados
        - La demanda eléctrica suele mostrar estacionalidad diaria y semanal marcada.
        - Los picos horarios permiten dimensionar mejor la operación y la planeación.
        - La comparación entre Prophet, Random Forest y XGBoost ayuda a balancear interpretabilidad y precisión.
        - La detección de outliers es clave para separar eventos reales de errores de captura o cambios operativos.

        ### Recomendaciones
        - Mantener el pipeline de ETL y recalibrar modelos con frecuencia definida.
        - Revisar resultados por temporada, festivos y fines de semana.
        - Consolidar métricas de error por modelo para elegir el campeón operacional.
        - Automatizar la publicación de reportes y gráficas para seguimiento ejecutivo.
        """
    )

    st.success("La app está lista para consumir datasets procesados y artefactos de predicción cuando estén disponibles en las carpetas estándar del proyecto.")


def load_processed_data() -> pd.DataFrame:
    """Carga el dataset procesado más reciente disponible en el proyecto."""

    processed_dir = _project_root() / "data/processed"
    for filename in DEFAULT_DATA_FILENAMES:
        file_path = processed_dir / filename
        if file_path.exists():
            return _read_tabular_file(file_path)

    candidates = sorted(
        [file_path for file_path in processed_dir.glob("*.*") if file_path.suffix.lower() in {".csv", ".parquet", ".pq"}],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return _read_tabular_file(candidates[0])
    return pd.DataFrame()


def load_prediction_data() -> pd.DataFrame:
    """Carga predicciones históricas o futuras si existen artefactos en disco."""

    search_dirs = [
        _project_root() / "models",
        _project_root() / "reports",
        _project_root() / "data/processed",
    ]
    for base_dir in search_dirs:
        for filename in DEFAULT_PREDICTION_FILENAMES:
            file_path = base_dir / filename
            if file_path.exists():
                return _read_tabular_file(file_path)

        for file_path in sorted(base_dir.glob("*.*"), key=lambda item: item.stat().st_mtime, reverse=True):
            if file_path.suffix.lower() in {".csv", ".parquet", ".pq"} and _looks_like_predictions(file_path.name):
                return _read_tabular_file(file_path)

    return pd.DataFrame()


def load_model_results() -> pd.DataFrame:
    """Carga métricas agregadas de modelos si existen en los directorios del proyecto."""

    for base_dir in [_project_root() / "models", _project_root() / "reports"]:
        for filename in DEFAULT_MODEL_RESULTS_FILENAMES:
            file_path = base_dir / filename
            if file_path.exists():
                return _read_tabular_file(file_path)

    return pd.DataFrame()


def calculate_kpis(frame: pd.DataFrame, datetime_column: str, demand_column: str) -> dict[str, float]:
    """Calcula KPIs ejecutivos de demanda y crecimiento anual."""

    avg_demand = float(frame[demand_column].mean())
    max_demand = float(frame[demand_column].max())
    min_demand = float(frame[demand_column].min())
    growth = calculate_annual_growth(frame, datetime_column, demand_column)
    return {"avg": avg_demand, "max": max_demand, "min": min_demand, "growth": growth}


def calculate_annual_growth(frame: pd.DataFrame, datetime_column: str, demand_column: str) -> float:
    """Calcula el crecimiento anual porcentual entre el primer y último año disponibles."""

    annual = frame.assign(anio=frame[datetime_column].dt.year).groupby("anio", as_index=False)[demand_column].mean().sort_values("anio")
    if len(annual) < 2:
        return 0.0
    first_value = float(annual.iloc[0][demand_column])
    last_value = float(annual.iloc[-1][demand_column])
    if first_value == 0:
        return 0.0
    return ((last_value - first_value) / first_value) * 100.0


def build_trend_figure(frame: pd.DataFrame, datetime_column: str, demand_column: str) -> go.Figure:
    """Crea una gráfica de línea con suavizado semanal sobre la serie histórica."""

    daily = frame.set_index(datetime_column)[demand_column].resample("D").mean().reset_index()
    daily["promedio_7d"] = daily[demand_column].rolling(7, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily[datetime_column], y=daily[demand_column], mode="lines", name="Diaria", line=dict(color="#5dade2", width=1.5)))
    fig.add_trace(go.Scatter(x=daily[datetime_column], y=daily["promedio_7d"], mode="lines", name="Media móvil 7 días", line=dict(color="#f5b041", width=2.5)))
    fig.update_layout(
        title="Tendencia histórica de la demanda",
        xaxis_title="Fecha",
        yaxis_title="Demanda",
        template="plotly_dark",
        hovermode="x unified",
    )
    return fig


def build_distribution_figure(frame: pd.DataFrame, demand_column: str) -> go.Figure:
    """Crea la distribución de la demanda con histograma y boxplot."""

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.08, subplot_titles=("Histograma", "Boxplot"))
    fig.add_trace(go.Histogram(x=frame[demand_column], nbinsx=50, marker_color="#8e44ad", opacity=0.9), row=1, col=1)
    fig.add_trace(go.Box(x=frame[demand_column], orientation="h", marker_color="#58d68d", boxmean=True, showlegend=False), row=2, col=1)
    fig.update_layout(template="plotly_dark", title="Distribución de la demanda")
    return fig


def build_hour_profile_figure(frame: pd.DataFrame, datetime_column: str, demand_column: str) -> go.Figure:
    """Construye la demanda promedio por hora del día."""

    working = frame.copy()
    working["hora"] = working[datetime_column].dt.hour
    hourly = working.groupby("hora", as_index=False)[demand_column].mean().sort_values("hora")

    fig = px.bar(hourly, x="hora", y=demand_column, title="Demanda promedio por hora", labels={"hora": "Hora", demand_column: "Demanda promedio"})
    fig.update_traces(marker_color="#00bcd4")
    fig.update_layout(template="plotly_dark")
    return fig


def build_weekday_profile_figure(frame: pd.DataFrame, datetime_column: str, demand_column: str) -> go.Figure:
    """Construye la demanda promedio por día de la semana."""

    working = frame.copy()
    working["dia_semana"] = working[datetime_column].dt.dayofweek
    weekday = working.groupby("dia_semana", as_index=False)[demand_column].mean().sort_values("dia_semana")
    weekday["nombre_dia"] = weekday["dia_semana"].map(DAY_MAP)

    fig = px.bar(weekday, x="nombre_dia", y=demand_column, title="Demanda promedio por día de la semana", labels={"nombre_dia": "Día", demand_column: "Demanda promedio"})
    fig.update_traces(marker_color="#e67e22")
    fig.update_layout(template="plotly_dark")
    return fig


def build_heatmap_figure(frame: pd.DataFrame, datetime_column: str, demand_column: str) -> go.Figure:
    """Construye un heatmap hora vs día de semana."""

    working = frame.copy()
    working["hora"] = working[datetime_column].dt.hour
    working["dia_semana"] = working[datetime_column].dt.dayofweek
    pivot = working.pivot_table(index="dia_semana", columns="hora", values=demand_column, aggfunc="mean").reindex(index=range(7), columns=range(24))
    pivot.index = [DAY_MAP[index] for index in pivot.index]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(hour) for hour in pivot.columns],
            y=pivot.index.tolist(),
            colorscale="Cividis",
            colorbar=dict(title="Demanda"),
        )
    )
    fig.update_layout(template="plotly_dark", title="Heatmap hora vs día de semana", xaxis_title="Hora", yaxis_title="Día de la semana")
    return fig


def build_real_vs_predicted_figure(frame: pd.DataFrame, datetime_column: str, demand_column: str) -> go.Figure:
    """Grafica la comparación entre el valor real y las predicciones de cada modelo."""

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame[datetime_column], y=frame[demand_column], mode="lines", name="Real", line=dict(color="#ffffff", width=2)))

    model_columns = [column for column in frame.columns if column not in {datetime_column, demand_column, "modelo", "fecha"} and "pred" in column.lower()]
    if not model_columns and "prediccion" in frame.columns:
        model_columns = ["prediccion"]

    if "modelo" in frame.columns and model_columns:
        for model_name, subset in frame.groupby("modelo"):
            prediction_column = _select_prediction_column(subset, model_columns)
            if prediction_column is None:
                continue
            fig.add_trace(go.Scatter(x=subset[datetime_column], y=subset[prediction_column], mode="lines", name=str(model_name), line=dict(width=1.7)))
    elif model_columns:
        prediction_column = model_columns[0]
        fig.add_trace(go.Scatter(x=frame[datetime_column], y=frame[prediction_column], mode="lines", name="Predicho", line=dict(width=1.7, dash="dash")))

    fig.update_layout(template="plotly_dark", title="Real vs predicho", xaxis_title="Fecha", yaxis_title="Demanda")
    return fig


def build_model_comparison_figure(frame: pd.DataFrame) -> go.Figure:
    """Grafica la comparación de desempeño entre modelos."""

    metric_candidates = [column for column in frame.columns if column.lower() in {"mae", "rmse", "mape", "r2", "error", "score"}]
    metric_column = metric_candidates[0] if metric_candidates else frame.select_dtypes(include=["number"]).columns[-1]
    model_column = next((column for column in frame.columns if column.lower() in {"modelo", "model", "name", "algoritmo"}), frame.columns[0])

    fig = px.bar(frame, x=model_column, y=metric_column, color=model_column, title="Comparación de modelos", labels={model_column: "Modelo", metric_column: "Métrica"})
    fig.update_layout(template="plotly_dark", showlegend=False)
    return fig


def build_future_horizon_figure(frame: pd.DataFrame, datetime_column: str) -> go.Figure:
    """Grafica el horizonte futuro por modelo si existe en el archivo de predicciones."""

    future_frame = frame.copy()
    if "escenario" in future_frame.columns:
        future_mask = future_frame["escenario"].astype(str).str.lower().isin({"future", "futuro", "forecast", "proyeccion"})
    elif "tipo" in future_frame.columns:
        future_mask = future_frame["tipo"].astype(str).str.lower().isin({"future", "futuro", "forecast", "proyeccion"})
    else:
        future_mask = future_frame[datetime_column].isna() | (future_frame[datetime_column] > future_frame[datetime_column].max())

    horizon = future_frame.loc[future_mask].copy()
    if horizon.empty:
        horizon = future_frame.tail(min(96, len(future_frame))).copy()

    prediction_column = _select_prediction_column(horizon, [column for column in horizon.columns if "pred" in column.lower()])
    if prediction_column is None:
        prediction_column = _select_prediction_column(future_frame, [column for column in future_frame.columns if "pred" in column.lower()])

    fig = go.Figure()
    if prediction_column is not None and datetime_column in horizon.columns:
        fig.add_trace(go.Scatter(x=horizon[datetime_column], y=horizon[prediction_column], mode="lines+markers", name="Horizonte futuro", line=dict(color="#ff66cc", width=2)))
    else:
        fig.add_annotation(text="No se identificó un horizonte futuro explícito en los artefactos cargados.", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_layout(template="plotly_dark", title="Horizonte futuro", xaxis_title="Fecha", yaxis_title="Demanda")
    return fig


def build_demo_predictions(frame: pd.DataFrame, datetime_column: str, demand_column: str) -> pd.DataFrame:
    """Crea predicciones de ejemplo para mantener la app operativa sin artefactos reales."""

    demo = frame[[datetime_column, demand_column]].copy().tail(min(240, len(frame)))
    demo = demo.rename(columns={demand_column: "real"})
    demo["modelo"] = "Prophet"
    demo["prediccion_prophet"] = demo["real"] * (1 + np.sin(np.linspace(0, 2 * np.pi, len(demo))) * 0.01)
    demo["prediccion_random_forest"] = demo["real"] * (1 + np.cos(np.linspace(0, 3 * np.pi, len(demo))) * 0.008)
    demo["prediccion_xgboost"] = demo["real"] * (1 + np.sin(np.linspace(0, 4 * np.pi, len(demo))) * 0.006)
    demo["escenario"] = np.where(np.arange(len(demo)) >= int(len(demo) * 0.8), "future", "historical")
    return demo.rename(columns={"real": demand_column})


def build_demo_model_results() -> pd.DataFrame:
    """Crea una tabla de métricas de ejemplo para comparación de modelos."""

    return pd.DataFrame(
        {
            "modelo": ["Prophet", "Random Forest", "XGBoost"],
            "rmse": [4.9, 4.2, 3.8],
        }
    )


def normalize_prediction_frame(frame: pd.DataFrame, datetime_column: str, demand_column: str) -> pd.DataFrame:
    """Normaliza un archivo de predicciones a un formato estándar para la app."""

    normalized = frame.copy()
    normalized.columns = [_normalize_column_name(column) for column in normalized.columns]
    if datetime_column not in normalized.columns:
        datetime_candidates = [column for column in normalized.columns if any(candidate in column for candidate in {"fecha", "timestamp", "datetime", "date"})]
        if datetime_candidates:
            normalized = normalized.rename(columns={datetime_candidates[0]: datetime_column})
    if demand_column not in normalized.columns:
        demand_candidates = [column for column in normalized.columns if "real" in column or "observ" in column or "target" in column]
        if demand_candidates:
            normalized = normalized.rename(columns={demand_candidates[0]: demand_column})

    if "modelo" not in normalized.columns:
        normalized["modelo"] = _infer_model_name(normalized.columns)

    normalized[datetime_column] = pd.to_datetime(normalized[datetime_column], errors="coerce")
    return normalized.dropna(subset=[datetime_column]).sort_values(datetime_column).reset_index(drop=True)


def count_future_rows(frame: pd.DataFrame) -> int:
    """Cuenta cuántas filas pertenecen al horizonte futuro si existe esa etiqueta."""

    if "escenario" in frame.columns:
        return int(frame["escenario"].astype(str).str.lower().isin({"future", "futuro", "forecast", "proyeccion"}).sum())
    if "tipo" in frame.columns:
        return int(frame["tipo"].astype(str).str.lower().isin({"future", "futuro", "forecast", "proyeccion"}).sum())
    return 0


def infer_datetime_column(frame: pd.DataFrame) -> str | None:
    """Infiera la columna temporal más probable del dataset."""

    for candidate in ("fecha_hora", "timestamp", "datetime", "date", "fecha"):
        for column in frame.columns:
            if candidate in column:
                return column
    return None


def infer_demand_column(frame: pd.DataFrame) -> str | None:
    """Infiera la columna de demanda más probable del dataset."""

    for candidate in ("demanda", "consumo", "energia", "value", "valor"):
        for column in frame.columns:
            if candidate in column:
                return column
    numeric_columns = list(frame.select_dtypes(include=["number"]).columns)
    return numeric_columns[-1] if numeric_columns else None


def _prepare_dataset(frame: pd.DataFrame, datetime_column: str, demand_column: str) -> pd.DataFrame:
    """Normaliza la serie principal para visualización."""

    working = frame.copy()
    working[datetime_column] = pd.to_datetime(working[datetime_column], errors="coerce")
    working[demand_column] = pd.to_numeric(working[demand_column], errors="coerce")
    working = working.dropna(subset=[datetime_column, demand_column]).sort_values(datetime_column).reset_index(drop=True)
    return working


def _read_tabular_file(file_path: Path) -> pd.DataFrame:
    """Lee CSV o Parquet desde disco."""

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)
    raise ValueError(f"Formato no soportado: {file_path.suffix}")


def _looks_like_predictions(filename: str) -> bool:
    """Detecta nombres de archivo que parecen contener predicciones."""

    normalized = filename.lower()
    return any(keyword in normalized for keyword in ("pred", "forecast", "pronost", "model"))


def _select_prediction_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Selecciona la columna de predicción más probable a partir de candidatos."""

    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    for column in frame.columns:
        if "pred" in column.lower() or "forecast" in column.lower():
            return column
    return None


def _infer_model_name(columns: Iterable[str]) -> str:
    """Infiere un nombre de modelo a partir de las columnas disponibles."""

    joined = " ".join(columns).lower()
    for key, value in MODEL_DISPLAY_NAMES.items():
        if key in joined:
            return value
    return "Modelo"


def _project_root() -> Path:
    """Resuelve la raíz del proyecto a partir de la ubicación del módulo."""

    return Path(__file__).resolve().parents[2]


def _demo_frame() -> pd.DataFrame:
    """Genera una serie temporal sintética para que la app siempre sea visible."""

    idx = pd.date_range("2024-01-01", periods=24 * 90, freq="H")
    base = 8500 + 700 * np.sin(np.linspace(0, 20 * np.pi, len(idx)))
    noise = np.random.default_rng(42).normal(0, 120, len(idx))
    return pd.DataFrame({"fecha_hora": idx, "demanda": base + noise})


def _inject_dark_theme() -> None:
    """Inyecta estilos CSS para una interfaz oscura y consistente."""

    st.markdown(
        """
        <style>
            :root {
                color-scheme: dark;
            }
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(63, 81, 181, 0.18), transparent 30%),
                    radial-gradient(circle at top right, rgba(0, 188, 212, 0.12), transparent 28%),
                    linear-gradient(180deg, #0b1020 0%, #111827 45%, #0f172a 100%);
                color: #e5e7eb;
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
                border-right: 1px solid rgba(255, 255, 255, 0.08);
            }
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
            }
            h1, h2, h3, h4 {
                letter-spacing: -0.02em;
            }
            .stMetric {
                background: rgba(17, 24, 39, 0.72);
                padding: 0.8rem;
                border-radius: 14px;
                border: 1px solid rgba(255, 255, 255, 0.08);
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
            }
            div[data-testid="metric-container"] {
                background: rgba(17, 24, 39, 0.72);
                border-radius: 14px;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
