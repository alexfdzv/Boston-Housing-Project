import os

import pandas as pd
import yaml


print("[1/4] Loading parameters...")

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

PROCESSED_PATH = params["data"]["processed_path"]
IQR_FACTOR = params["cleaning"]["iqr_factor"]
CAP_OUTLIERS = params["cleaning"]["cap_outliers"]
EXCLUDE_COLS = params["normalization"]["exclude_columns"]

print("[2/4] Loading staged dataset...")

df = pd.read_csv("data/processed/_loaded.csv")
print(f"      Original shape: {df.shape}")

print("[3/4] Checking missing values...")

missing = df.isnull().sum()
total_missing = int(missing.sum())

if total_missing == 0:
    print("      No missing values found.")
else:
    print(f"      Missing values found: {total_missing}")
    print(missing[missing > 0])
    for col in df.columns[df.isnull().any()]:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"      {col}: imputed with median ({median_val:.4f})")

print("[4/4] Treating outliers with IQR capping...")

cols_to_treat = [col for col in df.columns if col not in EXCLUDE_COLS]
outlier_report = {}

for col in cols_to_treat:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - IQR_FACTOR * iqr
    upper = q3 + IQR_FACTOR * iqr

    n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
    outlier_report[col] = n_outliers

    if CAP_OUTLIERS:
        df[col] = df[col].clip(lower=lower, upper=upper)

    if n_outliers > 0:
        print(
            f"      {col}: {n_outliers} outliers capped "
            f"[{lower:.3f}, {upper:.3f}]"
        )

cols_without_outliers = [
    col for col, count in outlier_report.items() if count == 0
]
if cols_without_outliers:
    print(f"      No outliers: {', '.join(cols_without_outliers)}")

print(f"      Excluded columns: {EXCLUDE_COLS}")

os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
df.to_csv(PROCESSED_PATH, index=False)

print(f"      Saved to: {PROCESSED_PATH}")
print(f"      Final shape: {df.shape}")
print()
print("Cleaning completed successfully.")
