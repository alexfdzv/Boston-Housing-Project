"""
Entrena el modelo de regresión lineal (OLS) sobre el dataset
normalizado y guarda el modelo entrenado para su reutilización.
 
Decisión documentada en notebooks/algorithm2.ipynb:
- Se compararon OLS, Ridge y Lasso
- Las tres variantes obtuvieron métricas equivalentes (R² ≈ 0.685)
- Se seleccionó OLS por simplicidad e interpretabilidad
"""
 
import os
import pandas as pd
import numpy as np
import yaml
import joblib
 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
 
# ─────────────────────────────────────────────
# 1. Cargar parámetros desde params.yaml
# ─────────────────────────────────────────────
print("[1/4] Cargando parámetros...")
 
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)
 
SCALED_PATH  = params["data"]["scaled_path"]
MODEL_PATH   = params["data"]["model_path"]
TEST_SIZE    = params["modeling"]["test_size"]
RANDOM_STATE = params["modeling"]["random_state"]
TARGET       = "MEDV" 
 
print(f"      scaled_path  : {SCALED_PATH}")
print(f"      model_path   : {MODEL_PATH}")
print(f"      target       : {TARGET}")
print(f"      test_size    : {TEST_SIZE}")
print(f"      random_state : {RANDOM_STATE}")
 
# ─────────────────────────────────────────────
# 2. Cargar datos y hacer split
# ─────────────────────────────────────────────
print("[2/4] Cargando datos y separando train/test...")
 
df = pd.read_csv(SCALED_PATH)
 
X = df.drop(columns=[TARGET])
y = df[TARGET]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)
 
print(f"      Shape total : {df.shape}")
print(f"      Train       : {X_train.shape[0]} filas")
print(f"      Test        : {X_test.shape[0]} filas")
print(f"      Features    : {X.columns.tolist()}")
 
# ─────────────────────────────────────────────
# 3. Entrenar modelo
# ─────────────────────────────────────────────
print("[3/4] Entrenando modelo OLS (LinearRegression)...")
 
modelo = LinearRegression()
modelo.fit(X_train, y_train)
 
# Evaluar
y_pred = modelo.predict(X_test)
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
 
print(f"      R²   (test) : {r2:.4f}")
print(f"      RMSE (test) : {rmse:.4f} k USD")
print(f"      MAE  (test) : {mae:.4f} k USD")
 
# ─────────────────────────────────────────────
# 4. Guardar modelo
# ─────────────────────────────────────────────
print("[4/4] Guardando modelo entrenado...")
 
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(modelo, MODEL_PATH)
 
print(f"      Guardado en : {MODEL_PATH}")
print()
print("Entrenamiento completado exitosamente.")