"""
Carga el dataset crudo y lo guarda con encabezado
para que los pasos posteriores del pipeline puedan leerlo.
"""

import pandas as pd
import yaml

# Leer configuración
with open("../params.yaml", "r") as f:
    params = yaml.safe_load(f)

RAW_PATH = params["data"]["raw_path"]
COLUMNS  = params["data"]["columns"]

# Cargar
df = pd.read_csv(RAW_PATH, header=None, sep=r"\s+", names=COLUMNS)

# Guardar con encabezado
df.to_csv("../data/processed/_loaded.csv", index=False)
print(f"{df.shape[0]} filas x {df.shape[1]} columnas guardadas en data/processed/_loaded.csv")