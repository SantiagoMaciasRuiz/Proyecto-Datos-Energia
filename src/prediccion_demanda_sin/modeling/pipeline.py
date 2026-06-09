"""Pipeline de entrenamiento para predicción de demanda eléctrica.

Este módulo construye y evalúa tres modelos:

- Prophet
- Random Forest
- XGBoost

La salida principal incluye métricas comparativas, predicciones históricas,
horizonte futuro y artefactos serializados para reutilización en el dashboard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


MODEL_NAMES = ("prophet", "random_forest", "xgboost")
DEFAULT_OUTPUT_DIR = "models"
DEFAULT_ARTIFACTS_DIR = "models/artifacts"


@dataclass(slots=True)
class TrainingResult:
    """Resultado completo del entrenamiento y la predicción."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    future_forecast: pd.DataFrame
    model_paths: dict[str, Path]
    metrics_path: Path
    predictions_path: Path
    future_path: Path


def run_model_training_pipeline(
    data: pd.DataFrame,
    output_dir: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    datetime_column: str | None = None,
    target_column: str | None = None,
    test_fraction: float = 0.2,
    horizon_hours: int = 168,
) -> TrainingResult:
    """Entrena Prophet, Random Forest y XGBoost sobre una serie horaria.

    Parameters
    ----------
    data:
        DataFrame procesado con una columna temporal y una columna de demanda.
    output_dir:
        Directorio para archivos tabulares de salida. Por defecto ``models/``.
    artifacts_dir:
        Directorio para serializar modelos individuales. Por defecto
        ``models/artifacts/``.
    datetime_column:
        Nombre explícito de la columna temporal. Si no se proporciona se infiere.
    target_column:
        Nombre explícito de la columna objetivo. Si no se proporciona se infiere.
    test_fraction:
        Porción final de la serie usada como conjunto de validación.
    horizon_hours:
        Número de horas a proyectar hacia el futuro.
    """

    working = _prepare_series(data, datetime_column=datetime_column, target_column=target_column)
    datetime_column = working.attrs["datetime_column"]
    target_column = working.attrs["target_column"]

    feature_frame = _build_feature_frame(working, datetime_column, target_column)
    train_df, test_df = _time_split(feature_frame, test_fraction=test_fraction)
    train_df.attrs["datetime_column"] = datetime_column
    train_df.attrs["target_column"] = target_column
    test_df.attrs["datetime_column"] = datetime_column
    test_df.attrs["target_column"] = target_column

    prophet_outputs = _train_prophet(train_df, test_df, datetime_column, target_column)
    rf_model, rf_test_pred = _train_random_forest(train_df, test_df)
    xgb_model, xgb_test_pred = _train_xgboost(train_df, test_df)

    model_metrics = pd.DataFrame(
        [
            _build_metrics_row("Prophet", test_df[target_column].values, prophet_outputs["test_pred"]),
            _build_metrics_row("Random Forest", test_df[target_column].values, rf_test_pred),
            _build_metrics_row("XGBoost", test_df[target_column].values, xgb_test_pred),
        ]
    )

    predictions = _build_prediction_frame(
        test_df=test_df,
        datetime_column=datetime_column,
        target_column=target_column,
        prophet_pred=prophet_outputs["test_pred"],
        rf_pred=rf_test_pred,
        xgb_pred=xgb_test_pred,
    )

    future_forecast = _build_future_forecast(
        feature_frame=feature_frame,
        datetime_column=datetime_column,
        target_column=target_column,
        prophet_model=prophet_outputs["model"],
        rf_model=rf_model,
        xgb_model=xgb_model,
        horizon_hours=horizon_hours,
    )

    target_output_dir = Path(output_dir) if output_dir is not None else _project_root() / DEFAULT_OUTPUT_DIR
    target_output_dir.mkdir(parents=True, exist_ok=True)
    target_artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else _project_root() / DEFAULT_ARTIFACTS_DIR
    target_artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = target_output_dir / "model_metrics.csv"
    predictions_path = target_output_dir / "predicciones.csv"
    future_path = target_output_dir / "forecast.csv"

    model_metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    future_forecast.to_csv(future_path, index=False)

    model_paths = {
        "prophet": target_artifacts_dir / "prophet_model.joblib",
        "random_forest": target_artifacts_dir / "random_forest_model.joblib",
        "xgboost": target_artifacts_dir / "xgboost_model.joblib",
    }
    joblib.dump(prophet_outputs["model"], model_paths["prophet"])
    joblib.dump(rf_model, model_paths["random_forest"])
    joblib.dump(xgb_model, model_paths["xgboost"])

    metadata = {
        "datetime_column": datetime_column,
        "target_column": target_column,
        "feature_columns": _feature_columns(),
    }
    (target_artifacts_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return TrainingResult(
        metrics=model_metrics,
        predictions=predictions,
        future_forecast=future_forecast,
        model_paths=model_paths,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        future_path=future_path,
    )


def _prepare_series(
    data: pd.DataFrame,
    datetime_column: str | None,
    target_column: str | None,
) -> pd.DataFrame:
    """Prepara la serie temporal base e infiere columnas si no se especifican."""

    if data.empty:
        raise ValueError("El DataFrame de entrada está vacío.")

    working = data.copy()
    working.columns = [_normalize_column_name(column) for column in working.columns]

    datetime_column = datetime_column or _infer_datetime_column(working)
    target_column = target_column or _infer_target_column(working)

    if datetime_column is None:
        raise ValueError("No se encontró una columna temporal en el dataset.")
    if target_column is None:
        raise ValueError("No se encontró una columna objetivo de demanda en el dataset.")

    working[datetime_column] = pd.to_datetime(working[datetime_column], errors="coerce")
    working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
    working = working.dropna(subset=[datetime_column, target_column]).sort_values(datetime_column).reset_index(drop=True)
    working.attrs["datetime_column"] = datetime_column
    working.attrs["target_column"] = target_column
    return working


def _build_feature_frame(frame: pd.DataFrame, datetime_column: str, target_column: str) -> pd.DataFrame:
    """Construye variables temporales y rezagos para modelos tabulares."""

    working = frame.copy()
    timestamp = pd.to_datetime(working[datetime_column], errors="coerce")
    working["hora"] = timestamp.dt.hour
    working["dia"] = timestamp.dt.day
    working["mes"] = timestamp.dt.month
    working["trimestre"] = timestamp.dt.quarter
    working["anio"] = timestamp.dt.year
    working["dia_semana"] = timestamp.dt.dayofweek
    working["es_fin_de_semana"] = working["dia_semana"].isin([5, 6]).astype(int)
    working["es_festivo"] = _holiday_flags(timestamp)

    target = working[target_column].astype(float)
    working["lag_1"] = target.shift(1)
    working["lag_24"] = target.shift(24)
    working["lag_168"] = target.shift(168)
    working["rolling_mean_24"] = target.shift(1).rolling(24, min_periods=1).mean()
    working["rolling_mean_168"] = target.shift(1).rolling(168, min_periods=1).mean()
    working["rolling_std_24"] = target.shift(1).rolling(24, min_periods=2).std().fillna(0.0)

    working = working.dropna(subset=["lag_1", "lag_24", "lag_168"]).reset_index(drop=True)
    working.attrs["datetime_column"] = datetime_column
    working.attrs["target_column"] = target_column
    return working


def _time_split(frame: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa la serie en entrenamiento y prueba respetando el orden temporal."""

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction debe estar entre 0 y 1.")
    split_index = max(int(len(frame) * (1 - test_fraction)), 1)
    if split_index >= len(frame):
        split_index = len(frame) - 1
    train_df = frame.iloc[:split_index].reset_index(drop=True)
    test_df = frame.iloc[split_index:].reset_index(drop=True)
    if train_df.empty or test_df.empty:
        raise ValueError("La partición temporal dejó uno de los conjuntos vacío.")
    return train_df, test_df


def _train_prophet(train_df: pd.DataFrame, test_df: pd.DataFrame, datetime_column: str, target_column: str) -> dict[str, Any]:
    """Entrena Prophet y genera predicción sobre el conjunto de prueba."""

    from prophet import Prophet

    prophet_train = train_df[[datetime_column, target_column]].rename(columns={datetime_column: "ds", target_column: "y"})
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(prophet_train)

    future_test = test_df[[datetime_column]].rename(columns={datetime_column: "ds"})
    forecast_test = model.predict(future_test)
    return {"model": model, "test_pred": forecast_test["yhat"].values}


def _train_random_forest(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[Any, np.ndarray]:
    """Entrena un Random Forest sobre variables tabulares."""

    feature_columns = _feature_columns()
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1, min_samples_leaf=2)
    model.fit(train_df[feature_columns], train_df[_target_name(train_df)])
    predictions = model.predict(test_df[feature_columns])
    return model, predictions


def _train_xgboost(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[Any, np.ndarray]:
    """Entrena XGBoost sobre variables tabulares."""

    from xgboost import XGBRegressor

    feature_columns = _feature_columns()
    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    )
    model.fit(train_df[feature_columns], train_df[_target_name(train_df)])
    predictions = model.predict(test_df[feature_columns])
    return model, predictions


def _build_metrics_row(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Calcula métricas comparativas por modelo."""

    return {
        "modelo": model_name,
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": _mean_absolute_percentage_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def _build_prediction_frame(
    test_df: pd.DataFrame,
    datetime_column: str,
    target_column: str,
    prophet_pred: np.ndarray,
    rf_pred: np.ndarray,
    xgb_pred: np.ndarray,
) -> pd.DataFrame:
    """Convierte las predicciones a formato largo para dashboard y reportes."""

    frames = []
    for model_name, predictions in (
        ("Prophet", prophet_pred),
        ("Random Forest", rf_pred),
        ("XGBoost", xgb_pred),
    ):
        model_frame = pd.DataFrame(
            {
                datetime_column: test_df[datetime_column].values,
                "real": test_df[target_column].values,
                "modelo": model_name,
                "prediccion": predictions,
                "escenario": "historical",
            }
        )
        frames.append(model_frame)
    return pd.concat(frames, ignore_index=True)


def _build_future_forecast(
    feature_frame: pd.DataFrame,
    datetime_column: str,
    target_column: str,
    prophet_model: Any,
    rf_model: Any,
    xgb_model: Any,
    horizon_hours: int,
) -> pd.DataFrame:
    """Genera un horizonte futuro para los tres modelos."""

    last_timestamp = pd.to_datetime(feature_frame[datetime_column].max())
    future_rows = []

    future_dates = pd.date_range(last_timestamp + pd.Timedelta(hours=1), periods=horizon_hours, freq="h")
    prophet_future = prophet_model.predict(pd.DataFrame({"ds": future_dates}))[["ds", "yhat"]].rename(columns={"ds": datetime_column, "yhat": "prediccion"})
    prophet_future["modelo"] = "Prophet"
    prophet_future["escenario"] = "future"

    rf_future = _recursive_tabular_forecast(rf_model, feature_frame, datetime_column, target_column, future_dates, model_name="Random Forest")
    xgb_future = _recursive_tabular_forecast(xgb_model, feature_frame, datetime_column, target_column, future_dates, model_name="XGBoost")

    future_rows.extend([prophet_future, rf_future, xgb_future])
    combined = pd.concat(future_rows, ignore_index=True)
    return combined


def _recursive_tabular_forecast(
    model: Any,
    history_frame: pd.DataFrame,
    datetime_column: str,
    target_column: str,
    future_dates: pd.DatetimeIndex,
    model_name: str,
) -> pd.DataFrame:
    """Genera pronóstico futuro recursivo para modelos tabulares."""

    # Desactivar paralelismo de predicción para evitar overhead extremo e hilos colisionando en Windows
    if hasattr(model, "n_jobs"):
        try:
            model.n_jobs = 1
        except Exception:
            pass
    if hasattr(model, "set_params"):
        try:
            model.set_params(n_jobs=1)
        except Exception:
            pass

    # Cargar calendario de festivos una vez para todos los años del horizonte futuro
    future_years = sorted(list(set(future_dates.year)))
    try:
        holiday_calendar = __import__("holidays").CountryHoliday("CO", years=future_years)
    except Exception:
        holiday_calendar = None

    feature_cols = _feature_columns()
    import warnings

    history_values = history_frame[target_column].astype(float).tolist()
    rows = []
    for current_date in future_dates:
        feature_row = _feature_row_from_history(current_date, history_values, holiday_calendar)
        feature_vector = [feature_row[col] for col in feature_cols]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            pred = float(model.predict([feature_vector])[0])

        rows.append(
            {
                datetime_column: current_date,
                "prediccion": pred,
                "modelo": model_name,
                "escenario": "future",
            }
        )
        history_values.append(pred)
    return pd.DataFrame(rows)


def _feature_row_from_history(
    timestamp: pd.Timestamp,
    history_values: list[float],
    holiday_calendar: Any = None
) -> dict[str, float | int]:
    """Construye una fila de características a partir del historial disponible."""

    last_value = history_values[-1] if history_values else 0.0
    lag_1 = _safe_lag(history_values, 1, last_value)
    lag_24 = _safe_lag(history_values, 24, last_value)
    lag_168 = _safe_lag(history_values, 168, last_value)
    
    # Evaluar festivo optimizado usando el calendario precargado
    if holiday_calendar is not None:
        es_festivo = int(timestamp.date() in holiday_calendar)
    else:
        es_festivo = int(_is_colombian_holiday(timestamp))

    return {
        "hora": int(timestamp.hour),
        "dia": int(timestamp.day),
        "mes": int(timestamp.month),
        "trimestre": int(timestamp.quarter),
        "anio": int(timestamp.year),
        "dia_semana": int(timestamp.dayofweek),
        "es_fin_de_semana": int(timestamp.dayofweek in (5, 6)),
        "es_festivo": es_festivo,
        "lag_1": float(lag_1),
        "lag_24": float(lag_24),
        "lag_168": float(lag_168),
        "rolling_mean_24": float(np.mean(history_values[-24:]) if history_values else last_value),
        "rolling_mean_168": float(np.mean(history_values[-168:]) if history_values else last_value),
        "rolling_std_24": float(np.std(history_values[-24:], ddof=1) if len(history_values[-24:]) > 1 else 0.0),
    }


def _safe_lag(history_values: list[float], steps_back: int, fallback: float) -> float:
    """Devuelve un rezago si existe o un valor de respaldo conservador."""

    if len(history_values) >= steps_back:
        return history_values[-steps_back]
    return fallback


def _holiday_flags(timestamp: pd.Series) -> pd.Series:
    """Marca festivos colombianos para una serie temporal."""

    years = timestamp.dt.year.dropna().astype(int).unique().tolist()
    holiday_calendar = __import__("holidays").CountryHoliday("CO", years=years or [pd.Timestamp.today().year])
    return timestamp.dt.date.map(lambda value: int(value in holiday_calendar))


def _is_colombian_holiday(timestamp: pd.Timestamp) -> bool:
    """Indica si una fecha puntual corresponde a festivo colombiano."""

    holiday_calendar = __import__("holidays").CountryHoliday("CO", years=[timestamp.year])
    return timestamp.date() in holiday_calendar


def _mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula MAPE evitando divisiones por cero."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denominator = np.where(y_true == 0, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denominator)) * 100)


def _feature_columns() -> list[str]:
    """Lista fija de variables usadas por los modelos tabulares."""

    return [
        "hora",
        "dia",
        "mes",
        "trimestre",
        "anio",
        "dia_semana",
        "es_fin_de_semana",
        "es_festivo",
        "lag_1",
        "lag_24",
        "lag_168",
        "rolling_mean_24",
        "rolling_mean_168",
        "rolling_std_24",
    ]


def _normalize_column_name(column: str) -> str:
    """Normaliza nombres de columnas para facilitar inferencia automática."""

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


def _infer_datetime_column(frame: pd.DataFrame) -> str | None:
    """Detecta la columna temporal más probable."""

    for candidate in ("fecha_hora", "timestamp", "datetime", "date", "fecha"):
        for column in frame.columns:
            if candidate in column:
                return column
    return None


def _infer_target_column(frame: pd.DataFrame) -> str | None:
    """Detecta la columna objetivo más probable."""

    for candidate in ("demanda", "consumo", "energia", "value", "valor"):
        for column in frame.columns:
            if candidate in column:
                return column
    numeric_columns = list(frame.select_dtypes(include=["number"]).columns)
    return numeric_columns[-1] if numeric_columns else None


def _target_name(frame: pd.DataFrame) -> str:
    """Recupera la columna objetivo almacenada en los atributos del frame."""

    target_column = frame.attrs.get("target_column")
    if target_column is None:
        raise ValueError("No se encontró el nombre de la columna objetivo en el dataset preparado.")
    return target_column


def _project_root() -> Path:
    """Resuelve la raíz del proyecto desde el módulo actual."""

    return Path(__file__).resolve().parents[3]