"""
Train a K-Means clustering model for neighborhood segmentation.

The model uses the K-Means model-ready normalized dataset produced by
src/03_normalize.py.
MEDV is excluded from training because it is the target variable used for
supervised models, but it is kept in the output file for interpretation.
"""

import json
import os

import joblib
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


print("[1/5] Loading parameters...")

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

SCALED_PATH = params["data"]["model_ready"]["kmeans_scaled_path"]
CLEAN_PATH = params["data"]["processed_path"]
OUTPUT_PATH = params["data"]["kmeans_path"]
PROFILE_PATH = params["data"]["kmeans_profile_path"]
METRICS_PATH = params["data"]["kmeans_metrics_path"]
MODEL_PATH = params["data"]["kmeans_model_path"]

KMEANS_PARAMS = params["modeling"]["kmeans"]
FEATURE_EXCLUDE_COLUMNS = KMEANS_PARAMS["feature_exclude_columns"]

print(f"      scaled_path : {SCALED_PATH}")
print(f"      clean_path  : {CLEAN_PATH}")
print(f"      n_clusters  : {KMEANS_PARAMS['n_clusters']}")
print(f"      excluded    : {FEATURE_EXCLUDE_COLUMNS}")

print("[2/5] Loading data...")

df_scaled = pd.read_csv(SCALED_PATH)
df_clean = pd.read_csv(CLEAN_PATH)

missing_exclusions = [
    col for col in FEATURE_EXCLUDE_COLUMNS if col not in df_scaled.columns
]
if missing_exclusions:
    raise ValueError(f"Excluded columns not found: {missing_exclusions}")

feature_cols = [
    col for col in df_scaled.columns if col not in FEATURE_EXCLUDE_COLUMNS
]
X = df_scaled[feature_cols]

print(f"      Rows       : {X.shape[0]}")
print(f"      Features   : {feature_cols}")

print("[3/5] Comparing random states and training K-Means...")

random_state_comparison = []
for state in KMEANS_PARAMS.get("random_states_to_compare", []):
    comparison_model = KMeans(
        n_clusters=KMEANS_PARAMS["n_clusters"],
        random_state=state,
        init=KMEANS_PARAMS["init"],
        n_init=KMEANS_PARAMS["n_init"],
    )
    comparison_clusters = comparison_model.fit_predict(X)
    comparison_counts = pd.Series(comparison_clusters).value_counts().sort_index()
    random_state_comparison.append(
        {
            "random_state": int(state),
            "inertia": float(comparison_model.inertia_),
            "silhouette_score": float(silhouette_score(X, comparison_clusters)),
            "cluster_counts": {
                str(cluster): int(count)
                for cluster, count in comparison_counts.items()
            },
        }
    )

model = KMeans(
    n_clusters=KMEANS_PARAMS["n_clusters"],
    random_state=KMEANS_PARAMS["random_state"],
    init=KMEANS_PARAMS["init"],
    n_init=KMEANS_PARAMS["n_init"],
)
clusters = model.fit_predict(X)

df_output = df_clean.copy()
df_output["cluster"] = clusters

print("[4/5] Building metrics and cluster profile...")

cluster_counts = df_output["cluster"].value_counts().sort_index()
profile = df_output.groupby("cluster").mean(numeric_only=True).round(4)
profile["count"] = cluster_counts

metrics = {
    "algorithm": "KMeans",
    "n_clusters": int(KMEANS_PARAMS["n_clusters"]),
    "features": feature_cols,
    "inertia": float(model.inertia_),
    "cluster_counts": {
        str(cluster): int(count) for cluster, count in cluster_counts.items()
    },
    "random_state_comparison": random_state_comparison,
}

if 1 < KMEANS_PARAMS["n_clusters"] < len(X):
    metrics["silhouette_score"] = float(silhouette_score(X, clusters))

print(f"      Inertia          : {metrics['inertia']:.4f}")
if "silhouette_score" in metrics:
    print(f"      Silhouette score : {metrics['silhouette_score']:.4f}")
print(f"      Cluster counts   : {metrics['cluster_counts']}")

print("[5/5] Saving outputs...")

for path in [OUTPUT_PATH, PROFILE_PATH, METRICS_PATH, MODEL_PATH]:
    os.makedirs(os.path.dirname(path), exist_ok=True)

df_output.to_csv(OUTPUT_PATH, index=False)
profile.to_csv(PROFILE_PATH)
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
joblib.dump(model, MODEL_PATH)

print(f"      Clustered data : {OUTPUT_PATH}")
print(f"      Profile        : {PROFILE_PATH}")
print(f"      Metrics        : {METRICS_PATH}")
print(f"      Model          : {MODEL_PATH}")
print()
print("K-Means training completed successfully.")
