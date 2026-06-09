"""Punto de entrada para entrenar modelos de demanda eléctrica.

Este script ejecuta el flujo completo:
1. Carga el dataset procesado más reciente desde ``data/processed/``.
2. Si no existe, intenta transformar el crudo disponible.
3. Entrena Prophet, Random Forest y XGBoost.
4. Guarda métricas, predicciones y artefactos en ``models/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from etl.transformer import transform_demand_dataset
from prediccion_demanda_sin.modeling import run_model_training_pipeline


def main() -> None:
	"""Ejecuta el entrenamiento end-to-end desde línea de comandos."""

	parser = argparse.ArgumentParser(description="Entrena modelos de demanda eléctrica del SIN colombiano.")
	parser.add_argument("--input", type=str, default=None, help="Ruta a archivo, carpeta o DataFrame procesado en disco.")
	parser.add_argument("--output-dir", type=str, default=None, help="Directorio para métricas y predicciones.")
	parser.add_argument("--artifacts-dir", type=str, default=None, help="Directorio para modelos serializados.")
	parser.add_argument("--test-fraction", type=float, default=0.2, help="Fracción final reservada para validación.")
	parser.add_argument("--horizon-hours", type=int, default=168, help="Horizonte futuro en horas.")
	args = parser.parse_args()

	input_path = Path(args.input) if args.input else PROJECT_ROOT / "data/processed"
	if not input_path.exists():
		raw_dir = PROJECT_ROOT / "data/raw"
		if not raw_dir.exists():
			raise FileNotFoundError("No existen datos procesados ni archivos crudos para entrenar.")
		transform_result = transform_demand_dataset(raw_dir)
		dataset = transform_result.data
	else:
		from pandas import DataFrame

		if input_path.is_dir():
			transform_result = transform_demand_dataset(input_path)
			dataset = transform_result.data
		else:
			transform_result = transform_demand_dataset(input_path)
			dataset = transform_result.data

	result = run_model_training_pipeline(
		dataset,
		output_dir=args.output_dir,
		artifacts_dir=args.artifacts_dir,
		test_fraction=args.test_fraction,
		horizon_hours=args.horizon_hours,
	)

	print("Entrenamiento completado correctamente.")
	print(f"Métricas guardadas en: {result.metrics_path}")
	print(f"Predicciones guardadas en: {result.predictions_path}")
	print(f"Horizonte futuro guardado en: {result.future_path}")


if __name__ == "__main__":
	main()