import pandas as pd
import numpy as np
import os
import yaml

# ─────────────────────────────────────────────
# 1. Cargar parámetros desde params.yaml
# ─────────────────────────────────────────────
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

RAW_PATH       = params["data"]["raw_path"]
PROCESSED_PATH = params["data"]["processed_path"]
COLUMNS        = params["data"]["columns"]

IQR_FACTOR     = params["cleaning"]["iqr_factor"]
CAP_OUTLIERS   = params["cleaning"]["cap_outliers"]

# Columnas excluidas del tratamiento de outliers:
# CHAS → variable dummy binaria (0 o 1), no aplica IQR
# MEDV → variable objetivo, no se debe modificar
EXCLUDE_COLS = params["normalization"]["exclude_columns"]

# ─────────────────────────────────────────────
# 2. Cargar dataset raw
# ─────────────────────────────────────────────
print("[1/4] Cargando dataset raw...")

df = pd.read_csv("data/processed/_loaded.csv")

print(f"      Shape original: {df.shape}")

# ─────────────────────────────────────────────
# 3. Verificación de valores faltantes
# ─────────────────────────────────────────────
print("[2/4] Verificando valores faltantes...")

missing = df.isnull().sum()
total_missing = missing.sum()

if total_missing == 0:
    print("      No se encontraron valores faltantes. Dataset íntegro.")
else:
    print(f"      Se encontraron {total_missing} valores faltantes:")
    print(missing[missing > 0])
    # Estrategia: imputar con mediana (robusta ante outliers)
    for col in df.columns[df.isnull().any()]:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"      → {col}: imputado con mediana ({median_val:.4f})")

# ─────────────────────────────────────────────
# 4. Tratamiento de outliers con IQR (capping)
# ─────────────────────────────────────────────
print("[3/4] Tratando outliers con método IQR (capping)...")

cols_to_treat = [c for c in df.columns if c not in EXCLUDE_COLS]
outlier_report = {}

for col in cols_to_treat:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - IQR_FACTOR * IQR
    upper = Q3 + IQR_FACTOR * IQR

    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_report[col] = int(n_outliers)

    # Capping: reemplazar por límite inferior/superior
    if CAP_OUTLIERS:
        df[col] = df[col].clip(lower=lower, upper=upper)

    if n_outliers > 0:
        print(f"      {col}: {n_outliers} outliers → capping [{lower:.3f}, {upper:.3f}]")

cols_sin_outliers = [c for c, n in outlier_report.items() if n == 0]
if cols_sin_outliers:
    print(f"      Sin outliers: {', '.join(cols_sin_outliers)}")

print(f"      Columnas excluidas del tratamiento: {EXCLUDE_COLS}")

# ─────────────────────────────────────────────
# 5. Guardar dataset procesado
# ─────────────────────────────────────────────
print("[4/4] Guardando dataset procesado...")

os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
df.to_csv(PROCESSED_PATH, index=False)

print(f"      Guardado en: {PROCESSED_PATH}")
print(f"      Shape final: {df.shape}")
print()
print("Limpieza completada exitosamente.")