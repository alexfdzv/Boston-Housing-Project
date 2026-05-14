"""
Carga el dataset crudo y lo guarda con encabezado
para que los pasos posteriores del pipeline puedan leerlo.
"""

import pandas as pd
import yaml

# Leer configuracion
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

RAW_PATH = params["data"]["raw_path"]
COLUMNS  = params["data"]["columns"]

# Cargar.
# Soporta dos formatos comunes del dataset Boston Housing:
# 1. Archivo original separado por espacios y sin encabezado.
# 2. CSV con encabezados, como el usado para restaurar la trazabilidad del repo.
df = pd.read_csv(RAW_PATH)
normalized_columns = [col.upper() for col in df.columns]

if normalized_columns == COLUMNS:
    df.columns = COLUMNS
else:
    df = pd.read_csv(RAW_PATH, header=None, sep=r"\s+", names=COLUMNS)

# Guardar con encabezado
df.to_csv("data/processed/_loaded.csv", index=False)
print(f"{df.shape[0]} filas x {df.shape[1]} columnas guardadas en data/processed/_loaded.csv")
