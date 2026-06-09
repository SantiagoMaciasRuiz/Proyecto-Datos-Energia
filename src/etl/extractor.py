"""Extracción de datos históricos de demanda eléctrica.

Este módulo intenta primero consumir una fuente de XM configurable y, si la
llamada falla por conectividad, autenticación o formato no soportado, recurre al
dataset público de datos abiertos de Colombia.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import StringIO
from typing import Any, Iterable
from urllib.parse import urlparse
import json

import pandas as pd
import requests


XM_DEFAULT_URL = os.getenv("XM_DEMANDA_API_URL", "")
SIMEM_PUBLIC_DATA_URL = os.getenv("SIMEM_PUBLIC_DATA_URL", "https://www.simem.co/backend-files/api/datos-publicos")
SIMEM_DEMANDA_DATASET_ID = os.getenv("SIMEM_DEMANDA_DATASET_ID", "c1b851")
XM_DEFAULT_ENDPOINT = os.getenv("XM_DEMANDA_ENDPOINT", "https://servapibi.xm.com.co/hourly")
XM_MAX_DAYS_PER_REQUEST = int(os.getenv("XM_MAX_DAYS_PER_REQUEST", "31"))
REQUEST_HEADERS = {
    "Accept": "application/json, text/csv, */*",
    "User-Agent": "Mozilla/5.0 (compatible; demand-pipeline/1.0)",
}


@dataclass(slots=True)
class ExtractionResult:
    """Resultado de la extracción con trazabilidad de la fuente utilizada."""

    data: pd.DataFrame
    source: str
    used_fallback: bool
    details: str


def extract_historical_demand(
    start_date: str | None = None,
    end_date: str | None = None,
    xm_url: str | None = None,
    timeout: int = 30,
    page_size: int = 50000,
) -> ExtractionResult:
    """Extrae demanda histórica intentando primero XM y luego el dataset público.

    Parameters
    ----------
    start_date:
        Fecha inicial opcional para acotar la consulta. Debe ser un texto en un
        formato compatible con el API objetivo, por ejemplo ``YYYY-MM-DD``.
    end_date:
        Fecha final opcional para acotar la consulta.
    xm_url:
        URL completa del endpoint de XM. Si no se envía, se usa la variable de
        entorno ``XM_DEMANDA_API_URL``.
    timeout:
        Tiempo máximo de espera para cada petición HTTP, en segundos.
    page_size:
        Tamaño de página para la descarga paginada desde datos abiertos.

    Returns
    -------
    ExtractionResult
        Datos extraídos, fuente usada y un texto descriptivo del resultado.
    """

    xm_candidate = xm_url or XM_DEFAULT_URL or XM_DEFAULT_ENDPOINT
    if xm_candidate:
        try:
            xm_frame = _fetch_from_xm(
                xm_candidate,
                start_date=start_date,
                end_date=end_date,
                timeout=timeout,
            )
            return ExtractionResult(
                data=xm_frame,
                source="xm",
                used_fallback=False,
                details="Extracción exitosa desde XM.",
            )
        except Exception as exc:  # pragma: no cover - el fallback es el objetivo
            fallback_frame = _fetch_from_simem_public_data(
                start_date=start_date,
                end_date=end_date,
                timeout=timeout,
                page_size=page_size,
            )
            return ExtractionResult(
                data=fallback_frame,
                source="datos_abiertos",
                used_fallback=True,
                details=f"XM no estuvo disponible o no respondió con un formato usable: {exc}",
            )

    fallback_frame = _fetch_from_simem_public_data(
        start_date=start_date,
        end_date=end_date,
        timeout=timeout,
        page_size=page_size,
    )
    return ExtractionResult(
        data=fallback_frame,
        source="datos_abiertos",
        used_fallback=True,
        details="No se configuró un endpoint de XM; se usó el dataset público.",
    )


def _fetch_from_xm(
    xm_url: str,
    start_date: str | None = None,
    end_date: str | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Consulta la API de XM por ventanas horarias y normaliza la respuesta.

    La API pública de XM documenta el uso de POST sobre ``/hourly`` con un body
    que incluye ``MetricId``, ``StartDate``, ``EndDate``, ``Entity`` y, cuando
    aplica, ``Filter``. Para demanda horaria del SIN se usa por defecto
    ``MetricId='DemaReal'`` y ``Entity='Sistema'``.
    """

    start = pd.to_datetime(start_date or _default_start_date(), errors="raise")
    end = pd.to_datetime(end_date or _default_end_date(), errors="raise")
    if start > end:
        raise ValueError("start_date no puede ser mayor que end_date.")

    frames: list[pd.DataFrame] = []
    for chunk_start, chunk_end in _date_windows(start, end, max_days=XM_MAX_DAYS_PER_REQUEST):
        payload = {
            "MetricId": "DemaReal",
            "StartDate": chunk_start.strftime("%Y-%m-%d"),
            "EndDate": chunk_end.strftime("%Y-%m-%d"),
            "Entity": "Sistema",
        }
        response = requests.post(xm_url, json=payload, timeout=timeout, headers=REQUEST_HEADERS)
        response.raise_for_status()
        frames.append(_parse_xm_response(response))

    if not frames:
        return pd.DataFrame()

    return _deduplicate_frame(pd.concat(frames, ignore_index=True))


def _fetch_from_simem_public_data(
    start_date: str | None = None,
    end_date: str | None = None,
    timeout: int = 30,
    page_size: int = 50000,
) -> pd.DataFrame:
    """Descarga el dataset público de SIMEM como respaldo de XM.

    La documentación de XM indica que para la variable Demanda real existen
    dataset IDs públicos en SIMEM; se usa por defecto ``c1b851`` porque es el
    identificador documentado para esta variable en el repositorio.
    """

    attempts = [
        (
            SIMEM_PUBLIC_DATA_URL,
            {
                "startDate": start_date,
                "endDate": end_date,
                "datasetId": SIMEM_DEMANDA_DATASET_ID,
            },
        ),
        (
            SIMEM_PUBLIC_DATA_URL,
            {
                "startdate": start_date,
                "enddate": end_date,
                "datasetid": SIMEM_DEMANDA_DATASET_ID,
            },
        ),
        (
            "https://www.simem.co/backend-files/api/PublicData",
            {
                "startDate": start_date,
                "endDate": end_date,
                "datasetId": SIMEM_DEMANDA_DATASET_ID,
            },
        ),
    ]

    last_error: Exception | None = None
    for url, params in attempts:
        clean_params = {key: value for key, value in params.items() if value is not None}
        try:
            response = requests.get(url, params=clean_params, timeout=timeout, headers=REQUEST_HEADERS)
            response.raise_for_status()
            frame = _json_to_frame(response.json())
            frame = _deduplicate_frame(frame)
            if start_date or end_date:
                frame = _filter_simem_by_dates(frame, start_date=start_date, end_date=end_date)
            return frame.reset_index(drop=True)
        except Exception as exc:  # pragma: no cover - fallbacks are part of the strategy
            last_error = exc

    if last_error is not None:
        raise last_error
    return pd.DataFrame()


def _build_date_params(
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, str]:
    """Construye parámetros de consulta relacionados con fechas."""

    params: dict[str, str] = {}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    return params


def _build_open_data_query(
    start_date: str | None,
    end_date: str | None,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    """Arma parámetros para la API de datos abiertos.

    La consulta intenta filtrar de manera genérica usando una columna temporal
    configurable por entorno. Si el dataset no coincide con ese nombre, el
    recurso se descargará completo.
    """

    params: dict[str, Any] = {"$limit": limit, "$offset": offset}
    date_column = _guess_datetime_column()
    where_clause: str | None = None
    if start_date and end_date:
        where_clause = f"{date_column} >= '{start_date}' AND {date_column} <= '{end_date}'"
    elif start_date:
        where_clause = f"{date_column} >= '{start_date}'"
    elif end_date:
        where_clause = f"{date_column} <= '{end_date}'"

    if where_clause:
        params["$where"] = where_clause

    return params


def _guess_datetime_column() -> str:
    """Devuelve el nombre de columna temporal más probable para el dataset."""

    return os.getenv("OPEN_DATA_DATE_COLUMN", "fecha")


def _json_to_frame(payload: Any) -> pd.DataFrame:
    """Convierte una respuesta JSON variada en DataFrame.

    Acepta listas de registros, diccionarios con claves ``data`` o ``results`` y
    estructuras con metadatos al estilo Socrata.
    """

    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            return pd.json_normalize(payload)
        return pd.DataFrame(payload)

    if isinstance(payload, dict):
        for key in ("data", "results", "records"):
            if key in payload and isinstance(payload[key], list):
                if payload[key] and isinstance(payload[key][0], dict):
                    return pd.json_normalize(payload[key])
                return pd.DataFrame(payload[key])
        if "meta" in payload and "data" in payload:
            data_value = payload["data"]
            if isinstance(data_value, list) and data_value and isinstance(data_value[0], dict):
                return pd.json_normalize(data_value)
            return pd.DataFrame(data_value)
        if any(isinstance(value, dict) for value in payload.values()):
            return pd.json_normalize(payload)
        return pd.DataFrame([payload])

    raise ValueError("La respuesta JSON no tiene una estructura reconocible.")


def _deduplicate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados incluso cuando existen columnas con listas o diccionarios."""

    if frame.empty:
        return frame

    normalized = frame.copy()
    for column in normalized.columns:
        if normalized[column].map(lambda value: isinstance(value, (dict, list, tuple, set))).any():
            normalized[column] = normalized[column].map(_stringify_value)
    return normalized.drop_duplicates().reset_index(drop=True)


def _stringify_value(value: Any) -> Any:
    """Convierte valores anidados en texto estable para poder deduplicar."""

    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return value


def _parse_xm_response(response: requests.Response) -> pd.DataFrame:
    """Convierte una respuesta de XM en ``DataFrame``.

    XM puede responder con JSON anidado o con listas planas. Esta función intenta
    ambas rutas y conserva la salida aunque cambie mínimamente el envoltorio.
    """

    content_type = response.headers.get("Content-Type", "").lower()
    if "json" in content_type:
        return _json_to_frame(response.json())

    try:
        return _json_to_frame(response.json())
    except Exception:
        return pd.read_csv(StringIO(response.text))


def _filter_simem_by_dates(
    frame: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    """Filtra localmente el dataset de SIMEM usando una columna temporal inferida."""

    date_column = _guess_datetime_column()
    normalized_columns = {_normalize_column_name(column): column for column in frame.columns}
    candidate_column = normalized_columns.get(_normalize_column_name(date_column))

    if candidate_column is None:
        return frame

    filtered = frame.copy()
    filtered[candidate_column] = pd.to_datetime(filtered[candidate_column], errors="coerce")
    if start_date:
        filtered = filtered.loc[filtered[candidate_column] >= pd.to_datetime(start_date)]
    if end_date:
        filtered = filtered.loc[filtered[candidate_column] <= pd.to_datetime(end_date)]
    return filtered


def _normalize_column_name(column: str) -> str:
    """Normaliza nombres de columna para hacer coincidir columnas equivalentes."""

    normalized = (
        str(column)
        .strip()
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


def _date_windows(start: pd.Timestamp, end: pd.Timestamp, max_days: int) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    """Divide un rango temporal en ventanas compatibles con la API de XM."""

    current = start.normalize()
    end = end.normalize()
    while current <= end:
        window_end = min(current + timedelta(days=max_days - 1), end)
        yield current, window_end
        current = window_end + timedelta(days=1)


def _default_start_date() -> pd.Timestamp:
    """Devuelve una fecha de inicio conservadora para consultas automáticas."""

    return pd.Timestamp.today().normalize() - pd.Timedelta(days=90)


def _default_end_date() -> pd.Timestamp:
    """Devuelve la fecha final por defecto para consultas automáticas."""

    return pd.Timestamp.today().normalize()


def normalize_source_name(url: str) -> str:
    """Genera un nombre corto de fuente a partir de una URL.

    Esta función puede usarse para registrar la procedencia del dato cuando el
    pipeline expone múltiples conectores.
    """

    return urlparse(url).netloc or url
