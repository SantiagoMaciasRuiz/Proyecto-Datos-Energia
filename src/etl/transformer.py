"""Transformación de datos crudos de demanda eléctrica horaria.

El módulo aplica una secuencia estándar de preparación de datos para el Sistema
Interconectado Nacional colombiano:

1. Carga de archivos desde ``data/raw/`` o desde un ``DataFrame`` en memoria.
2. Normalización de columnas y tipos.
3. Eliminación de duplicados.
4. Imputación de valores nulos.
5. Validaciones de rango para variables de fecha, hora y demanda.
6. Generación de variables temporales y calendario colombiano.
7. Persistencia del resultado en ``data/processed/`` en formato CSV y Parquet.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import ast
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import holidays as holiday_library
except ImportError:  # pragma: no cover - fallback when dependency is unavailable
    holiday_library = None


RAW_DIRNAME = "data/raw"
PROCESSED_DIRNAME = "data/processed"
DEFAULT_OUTPUT_BASENAME = "demanda_electrica_transformada"

DATE_CANDIDATES = (
    "fecha",
    "fecha_hora",
    "fechahora",
    "timestamp",
    "datetime",
    "date",
)
HOUR_CANDIDATES = ("hora", "hour", "hr", "periodo_hora")
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


@dataclass(slots=True)
class TransformResult:
    """Resultado del proceso de transformación y sus rutas de salida."""

    data: pd.DataFrame
    csv_path: Path
    parquet_path: Path


def transform_demand_dataset(
    raw_input: str | Path | pd.DataFrame,
    output_dir: str | Path | None = None,
    output_basename: str = DEFAULT_OUTPUT_BASENAME,
) -> TransformResult:
    """Transforma un conjunto de datos crudos de demanda eléctrica.

    Parameters
    ----------
    raw_input:
        Ruta a un archivo crudo, carpeta con archivos o un ``DataFrame`` en
        memoria.
    output_dir:
        Directorio donde se guardarán los artefactos procesados. Si no se
        especifica, se usa ``data/processed/``.
    output_basename:
        Nombre base para los archivos de salida.

    Returns
    -------
    TransformResult
        Datos transformados junto con las rutas de los archivos generados.
    """

    frame = _load_raw_input(raw_input)
    frame = _expand_hourly_structure(frame)
    cleaned = clean_demand_data(frame)
    transformed = add_temporal_features(cleaned)
    validated = validate_demand_data(transformed)

    target_dir = Path(output_dir) if output_dir is not None else _project_root() / PROCESSED_DIRNAME
    target_dir.mkdir(parents=True, exist_ok=True)

    csv_path = target_dir / f"{output_basename}.csv"
    parquet_path = target_dir / f"{output_basename}.parquet"

    validated.to_csv(csv_path, index=False)
    validated.to_parquet(parquet_path, index=False)

    return TransformResult(data=validated, csv_path=csv_path, parquet_path=parquet_path)


def clean_demand_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Limpia el conjunto de datos crudos.

    La función elimina duplicados, normaliza nombres de columnas, estandariza
    cadenas vacías a nulos y aplica imputación básica para columnas numéricas y
    temporales.
    """

    cleaned = frame.copy()
    cleaned.columns = [_normalize_column_name(column) for column in cleaned.columns]
    cleaned = cleaned.replace({"": np.nan, " ": np.nan, "nan": np.nan, "None": np.nan})
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    datetime_column = _infer_datetime_column(cleaned)
    if datetime_column:
        cleaned[datetime_column] = pd.to_datetime(cleaned[datetime_column], errors="coerce")

    hour_column = _infer_hour_column(cleaned)
    if hour_column:
        cleaned[hour_column] = pd.to_numeric(cleaned[hour_column], errors="coerce")

    demand_column = _infer_demand_column(cleaned)
    if demand_column:
        cleaned[demand_column] = pd.to_numeric(cleaned[demand_column], errors="coerce")

    cleaned = _impute_missing_values(cleaned, datetime_column=datetime_column, hour_column=hour_column, demand_column=demand_column)
    return cleaned


def validate_demand_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Valida y corrige rangos esperados en los datos de demanda.

    Se corrigen valores fuera de rango de hora y demanda, y se descartan filas
    con fechas no válidas.
    """

    validated = frame.copy()

    datetime_column = _infer_datetime_column(validated)
    if datetime_column:
        validated = validated.loc[validated[datetime_column].notna()].copy()

    hour_column = _infer_hour_column(validated)
    if hour_column:
        validated[hour_column] = validated[hour_column].round().astype("Int64")
        validated = validated.loc[validated[hour_column].between(0, 23, inclusive="both")].copy()

    demand_column = _infer_demand_column(validated)
    if demand_column:
        validated = validated.loc[validated[demand_column].notna()].copy()
        validated = validated.loc[validated[demand_column] >= 0].copy()

    return validated.reset_index(drop=True)


def add_temporal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Genera variables temporales y de calendario colombiano.

    Agrega las columnas:
    ``hora``, ``dia``, ``mes``, ``trimestre``, ``anio``, ``dia_semana``,
    ``es_festivo`` y ``es_fin_de_semana``.
    """

    transformed = frame.copy()
    datetime_column = _infer_datetime_column(transformed)

    if datetime_column is None:
        raise ValueError("No se encontró una columna temporal para generar variables de fecha.")

    timestamp = pd.to_datetime(transformed[datetime_column], errors="coerce")
    transformed[datetime_column] = timestamp

    transformed["hora"] = _extract_hour(transformed, timestamp)
    transformed["dia"] = timestamp.dt.day
    transformed["mes"] = timestamp.dt.month
    transformed["trimestre"] = timestamp.dt.quarter
    transformed["anio"] = timestamp.dt.year
    transformed["dia_semana"] = timestamp.dt.dayofweek

    colombian_holidays = _colombian_holidays(_years_from_timestamp(timestamp))
    transformed["es_festivo"] = timestamp.dt.date.astype("object").map(lambda value: value in colombian_holidays)
    transformed["es_fin_de_semana"] = transformed["dia_semana"].isin([5, 6])

    return transformed


def process_raw_files(
    raw_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    output_basename: str = DEFAULT_OUTPUT_BASENAME,
) -> list[TransformResult]:
    """Procesa todos los archivos soportados dentro de ``data/raw/``.

    Si el directorio contiene varios archivos, cada uno se transforma de manera
    independiente y se escribe un par CSV/Parquet por archivo.
    """

    source_dir = Path(raw_dir) if raw_dir is not None else _project_root() / RAW_DIRNAME
    if not source_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de entrada: {source_dir}")

    results: list[TransformResult] = []
    for raw_file in _iter_supported_files(source_dir):
        input_frame = _load_file(raw_file)
        file_basename = raw_file.stem if output_basename == DEFAULT_OUTPUT_BASENAME else f"{output_basename}_{raw_file.stem}"
        results.append(
            transform_demand_dataset(
                input_frame,
                output_dir=output_dir,
                output_basename=file_basename,
            )
        )

    return results


def _load_raw_input(raw_input: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Carga una entrada cruda desde ruta, carpeta o DataFrame."""

    if isinstance(raw_input, pd.DataFrame):
        return raw_input.copy()

    path = Path(raw_input)
    if path.is_dir():
        frames = [_load_file(file_path) for file_path in _iter_supported_files(path)]
        if not frames:
            raise FileNotFoundError(f"No se encontraron archivos soportados en {path}")
        return pd.concat(frames, ignore_index=True)

    return _load_file(path)


def _load_file(file_path: Path) -> pd.DataFrame:
    """Carga un archivo crudo soportado en memoria como DataFrame."""

    suffix = file_path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(file_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    raise ValueError(f"Formato no soportado: {file_path.suffix}")


def _expand_hourly_structure(frame: pd.DataFrame) -> pd.DataFrame:
    """Expande estructuras anidadas de XM/SIMEM a una tabla horaria plana.

    Algunos extractores entregan una fila por día con una columna ``items`` que
    contiene una lista de objetos con la estructura ``Date -> HourlyEntities ->
    Values``. Esta función convierte esa jerarquía a una fila por hora, con las
    columnas principales ``fecha_hora`` y ``demanda``.
    """

    candidate_columns = [column for column in frame.columns if _normalize_column_name(column) in {"items", "item", "data", "result", "results"}]
    if not candidate_columns:
        return frame

    source_column = candidate_columns[0]
    expanded_rows: list[dict[str, object]] = []

    for _, row in frame.iterrows():
        payload = row[source_column]
        parsed_payload = _parse_nested_payload(payload)
        if not isinstance(parsed_payload, list):
            parsed_payload = [parsed_payload]

        row_dict = row.to_dict()
        row_dict.pop(source_column, None)

        for day_record in parsed_payload:
            if not isinstance(day_record, dict):
                continue

            day_value = day_record.get("Date") or day_record.get("date") or day_record.get("Fecha")
            hourly_entities = day_record.get("HourlyEntities") or day_record.get("hourly_entities") or day_record.get("HourlyEntities ")
            if hourly_entities is None and isinstance(day_record.get("Values"), dict):
                hourly_entities = [day_record]

            if not isinstance(hourly_entities, list):
                hourly_entities = [hourly_entities]

            for entity in hourly_entities:
                if not isinstance(entity, dict):
                    continue

                values = entity.get("Values") or entity.get("values") or entity.get("Valor") or {}
                entity_id = entity.get("Id") or entity.get("id") or entity.get("code") or entity.get("Codigo")

                if not isinstance(values, dict):
                    continue

                for hour_key, demand_value in values.items():
                    hour_number = _hour_key_to_int(hour_key)
                    if hour_number is None:
                        continue
                    if hour_key.lower() == "code":
                        continue

                    expanded_row = row_dict.copy()
                    expanded_row["fecha"] = day_value
                    expanded_row["hora"] = hour_number - 1
                    expanded_row["demanda"] = demand_value
                    if entity_id is not None:
                        expanded_row["entidad"] = entity_id
                    expanded_rows.append(expanded_row)

    if not expanded_rows:
        return frame

    expanded = pd.DataFrame(expanded_rows)
    expanded["fecha_hora"] = pd.to_datetime(expanded["fecha"], errors="coerce") + pd.to_timedelta(expanded["hora"], unit="h")
    expanded = expanded.drop(columns=[column for column in ["fecha"] if column in expanded.columns])
    return expanded


def _parse_nested_payload(payload: object) -> object:
    """Convierte texto serializado en estructuras Python cuando aplica."""

    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return payload
        try:
            return ast.literal_eval(text)
        except Exception:
            return payload
    return payload


def _hour_key_to_int(hour_key: object) -> int | None:
    """Convierte claves tipo ``Hour01`` en enteros de hora."""

    if not isinstance(hour_key, str):
        return None
    cleaned = hour_key.strip().lower().replace("hour", "")
    if not cleaned.isdigit():
        return None
    hour_value = int(cleaned)
    return hour_value if 1 <= hour_value <= 24 else None


def _iter_supported_files(directory: Path) -> Iterable[Path]:
    """Itera sobre los archivos soportados dentro de un directorio."""

    supported_suffixes = {".csv", ".txt", ".parquet", ".pq", ".xlsx", ".xls"}
    for file_path in sorted(directory.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in supported_suffixes:
            yield file_path


def _impute_missing_values(
    frame: pd.DataFrame,
    datetime_column: str | None,
    hour_column: str | None,
    demand_column: str | None,
) -> pd.DataFrame:
    """Imputa valores nulos usando reglas simples y conservadoras."""

    imputed = frame.copy()

    if datetime_column and imputed[datetime_column].isna().any():
        imputed[datetime_column] = imputed[datetime_column].ffill().bfill()

    if hour_column and imputed[hour_column].isna().any():
        imputed[hour_column] = imputed[hour_column].interpolate(limit_direction="both")
        imputed[hour_column] = imputed[hour_column].fillna(imputed[hour_column].mode(dropna=True).iloc[0] if not imputed[hour_column].mode(dropna=True).empty else 0)

    if demand_column and imputed[demand_column].isna().any():
        imputed[demand_column] = imputed[demand_column].interpolate(limit_direction="both")
        imputed[demand_column] = imputed[demand_column].fillna(imputed[demand_column].median())

    numeric_columns = imputed.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        if column not in {hour_column, demand_column} and imputed[column].isna().any():
            imputed[column] = imputed[column].fillna(imputed[column].median())

    categorical_columns = imputed.select_dtypes(include=["object", "string"]).columns
    for column in categorical_columns:
        if imputed[column].isna().any():
            imputed[column] = imputed[column].fillna("desconocido")

    return imputed


def _extract_hour(frame: pd.DataFrame, timestamp: pd.Series) -> pd.Series:
    """Obtiene la hora desde la columna temporal o desde una columna explícita."""

    hour_column = _infer_hour_column(frame)
    if hour_column is not None:
        return pd.to_numeric(frame[hour_column], errors="coerce").fillna(timestamp.dt.hour).astype("Int64")
    return timestamp.dt.hour.astype("Int64")


def _infer_datetime_column(frame: pd.DataFrame) -> str | None:
    """Detecta la columna temporal más probable."""

    normalized = {column: _normalize_column_name(column) for column in frame.columns}
    for candidate in DATE_CANDIDATES:
        for original, clean_name in normalized.items():
            if candidate in clean_name:
                return original
    return None


def _infer_hour_column(frame: pd.DataFrame) -> str | None:
    """Detecta la columna que representa la hora del registro."""

    normalized = {column: _normalize_column_name(column) for column in frame.columns}
    for candidate in HOUR_CANDIDATES:
        for original, clean_name in normalized.items():
            if candidate == clean_name or candidate in clean_name:
                return original
    return None


def _infer_demand_column(frame: pd.DataFrame) -> str | None:
    """Detecta la columna que contiene el valor de demanda o consumo."""

    normalized = {column: _normalize_column_name(column) for column in frame.columns}
    for candidate in DEMAND_CANDIDATES:
        for original, clean_name in normalized.items():
            if candidate in clean_name:
                return original

    numeric_candidates = list(frame.select_dtypes(include=["number"]).columns)
    if numeric_candidates:
        return numeric_candidates[-1]
    return None


def _years_from_timestamp(timestamp: pd.Series) -> list[int]:
    """Extrae el conjunto de años presentes en una serie temporal."""

    years = timestamp.dt.year.dropna().astype(int).unique().tolist()
    return years or [pd.Timestamp.today().year]


def _colombian_holidays(years: Iterable[int]) -> set[date]:
    """Devuelve el calendario colombiano para los años solicitados.

    Si la librería ``holidays`` está disponible, se usa su implementación.
    En caso contrario, se calcula una aproximación robusta con las reglas más
    importantes del calendario colombiano (festivos fijos, festivos religiosos
    y traslados a lunes por ley Emiliani).
    """

    years_list = [int(year) for year in years]
    if holiday_library is not None:
        holiday_calendar = holiday_library.CountryHoliday("CO", years=years_list)
        return {holiday_date for holiday_date in holiday_calendar.keys()}

    holidays_set: set[date] = set()
    for year in years_list:
        easter = _easter_sunday(year)
        fixed_holidays = [
            date(year, 1, 1),
            date(year, 5, 1),
            date(year, 7, 20),
            date(year, 8, 7),
            date(year, 12, 8),
            date(year, 12, 25),
        ]
        holidays_set.update(fixed_holidays)

        emiliani_holidays = [
            date(year, 1, 6),
            date(year, 3, 19),
            date(year, 6, 29),
            date(year, 8, 15),
            date(year, 10, 12),
            date(year, 11, 1),
            date(year, 11, 11),
        ]
        holidays_set.update(_move_to_monday(holiday_date) for holiday_date in emiliani_holidays)

        holidays_set.update(
            {
                easter - timedelta(days=3),  # Jueves Santo
                easter - timedelta(days=2),  # Viernes Santo
                _move_to_monday(easter + timedelta(days=43)),  # Ascensión
                _move_to_monday(easter + timedelta(days=64)),  # Corpus Christi
                _move_to_monday(easter + timedelta(days=71)),  # Sagrado Corazón
            }
        )

    return holidays_set


def _move_to_monday(holiday_date: date) -> date:
    """Traslada una fecha al lunes siguiente si no cae en lunes."""

    weekday = holiday_date.weekday()
    if weekday == 0:
        return holiday_date
    days_until_monday = 7 - weekday
    return holiday_date + timedelta(days=days_until_monday)


def _easter_sunday(year: int) -> date:
    """Calcula la fecha de Pascua usando el algoritmo de Meeus/Jones/Butcher."""

    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _normalize_column_name(column: str) -> str:
    """Normaliza nombres de columnas para facilitar la detección automática."""

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
