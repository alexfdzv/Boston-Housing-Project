"""
Train a baseline OLS linear regression model on the normalized dataset.
"""

import os

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


print("[1/4] Loading parameters...")

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

SCALED_PATH = params["data"]["model_ready"]["linear_scaled_path"]
MODEL_PATH = params["data"]["model_path"]
TEST_SIZE = params["modeling"]["test_size"]
RANDOM_STATE = params["modeling"]["random_state"]
TARGET = "MEDV"

print(f"      scaled_path  : {SCALED_PATH}")
print(f"      model_path   : {MODEL_PATH}")
print(f"      target       : {TARGET}")
print(f"      test_size    : {TEST_SIZE}")
print(f"      random_state : {RANDOM_STATE}")

print("[2/4] Loading data and splitting train/test...")

df = pd.read_csv(SCALED_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

print(f"      Total shape : {df.shape}")
print(f"      Train rows  : {X_train.shape[0]}")
print(f"      Test rows   : {X_test.shape[0]}")
print(f"      Features    : {X.columns.tolist()}")

print("[3/4] Training OLS model...")

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"      R2   (test) : {r2:.4f}")
print(f"      RMSE (test) : {rmse:.4f} k USD")
print(f"      MAE  (test) : {mae:.4f} k USD")

print("[4/4] Saving trained model...")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"      Saved to: {MODEL_PATH}")
print()
print("Training completed successfully.")
