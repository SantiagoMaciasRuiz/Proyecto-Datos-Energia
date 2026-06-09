"""Orquestador end-to-end para demanda eléctrica del SIN colombiano.

El script realiza, en orden:
1. Extracción desde XM o datos abiertos.
2. Persistencia del dato crudo en ``data/raw/``.
3. Transformación y generación de variables temporales.
4. Entrenamiento y evaluación de Prophet, Random Forest y XGBoost.

El objetivo es tener una única entrada reproducible para ejecutar todo el flujo.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from etl.extractor import ExtractionResult, extract_historical_demand
from etl.transformer import TransformResult, transform_demand_dataset
from prediccion_demanda_sin.modeling import TrainingResult, run_model_training_pipeline


def main() -> None:
    """Ejecuta el pipeline completo y deja trazabilidad en disco."""

    parser = argparse.ArgumentParser(description="Pipeline end-to-end de demanda eléctrica del SIN colombiano.")
    parser.add_argument("--start-date", type=str, default=None, help="Fecha inicial YYYY-MM-DD.")
    parser.add_argument("--end-date", type=str, default=None, help="Fecha final YYYY-MM-DD.")
    parser.add_argument("--xm-url", type=str, default=None, help="Endpoint de XM si deseas sobreescribir el valor por defecto.")
    parser.add_argument("--output-name", type=str, default="historical_demand", help="Nombre base para archivos crudos y procesados.")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Fracción final reservada para validación temporal.")
    parser.add_argument("--horizon-hours", type=int, default=168, help="Horizonte futuro en horas para el forecast.")
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / "data/raw"
    processed_dir = PROJECT_ROOT / "data/processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    extraction = extract_historical_demand(
        start_date=args.start_date,
        end_date=args.end_date,
        xm_url=args.xm_url,
    )
    raw_path = raw_dir / f"{args.output_name}_raw.csv"
    extraction.data.to_csv(raw_path, index=False)
    print(f"Extracción completada desde: {extraction.source}")
    print(f"Dataset crudo guardado en: {raw_path}")

    transform_result = transform_demand_dataset(
        extraction.data,
        output_dir=processed_dir,
        output_basename=f"{args.output_name}_processed",
    )
    print(f"Transformación completada. Archivo procesado: {transform_result.csv_path}")

    training_result = run_model_training_pipeline(
        transform_result.data,
        output_dir=PROJECT_ROOT / "models",
        artifacts_dir=PROJECT_ROOT / "models/artifacts",
        test_fraction=args.test_fraction,
        horizon_hours=args.horizon_hours,
    )

    _print_summary(extraction, transform_result, training_result)


def _print_summary(
    extraction: ExtractionResult,
    transform_result: TransformResult,
    training_result: TrainingResult,
) -> None:
    """Imprime un resumen compacto de los artefactos generados."""

    print("Pipeline finalizado correctamente.")
    print(f"Registros procesados: {len(transform_result.data):,}")
    print(f"Métricas guardadas en: {training_result.metrics_path}")
    print(f"Predicciones guardadas en: {training_result.predictions_path}")
    print(f"Horizonte futuro guardado en: {training_result.future_path}")
    print("Resumen de métricas:")
    print(training_result.metrics.to_string(index=False))
    if extraction.used_fallback:
        print(f"Fallback usado: {extraction.details}")


if __name__ == "__main__":
    main()
