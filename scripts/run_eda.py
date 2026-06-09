#!/usr/bin/env python
"""Script para ejecutar el pipeline de Análisis Exploratorio de Datos (EDA).
"""

import os
import sys
from pathlib import Path

# Agregar src/ al PYTHONPATH para poder importar el paquete
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from prediccion_demanda_sin.eda.analysis import generate_eda_reports

def main():
    # Ruta del archivo de datos procesados predeterminado
    data_path = project_root / "data" / "processed" / "historical_demand_processed.parquet"
    
    if not data_path.exists():
        # Fallback a CSV
        data_path = project_root / "data" / "processed" / "historical_demand_processed.csv"
        
    if not data_path.exists():
        print(f"Error: No se encontro el dataset procesado en {data_path.parent}.")
        print("Por favor, ejecuta primero el pipeline para generar los datos correspondientes:")
        print("python scripts/run_pipeline.py --start-date 2022-01-01 --end-date 2025-12-31")
        sys.exit(1)
        
    print(f"Iniciando Analisis Exploratorio de Datos (EDA) sobre: {data_path.name}...")
    
    # Rutas de salida
    output_img_dir = project_root / "reports" / "figures"
    output_findings_path = project_root / "reports" / "eda_findings.md"
    
    try:
        generate_eda_reports(
            data_path=data_path,
            output_img_dir=output_img_dir,
            output_findings_path=output_findings_path
        )
        print("Exito: Analisis Exploratorio de Datos (EDA) finalizado con exito!")
    except Exception as e:
        print(f"Error durante la generacion del EDA: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
