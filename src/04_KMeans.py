import json
import os
import joblib
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Cargar datos
df_scaled = pd.read_csv(params['data']['scaled_path'])
df_clean = pd.read_csv(params['data']['processed_path'])

# Features para clustering (excluir MEDV)
feature_cols = [c for c in df_scaled.columns if c not in ['MEDV']]
X = df_scaled[feature_cols]

print(f"Filas: {X.shape[0]}")
print(f"Variables usadas: {feature_cols}")

# Entrenar K-Means
kmeans = KMeans(
    n_clusters=3,
    random_state=params['modeling']['random_state'],
    init='k-means++',
    n_init=10
)

clusters = kmeans.fit_predict(X)
df_clean['cluster'] = clusters

print(f"Inercia: {kmeans.inertia_:.4f}")
print(f"Silhouette score: {silhouette_score(X, clusters):.4f}")
print(df_clean['cluster'].value_counts().sort_index())

# Guardar modelo
os.makedirs('models', exist_ok=True)
joblib.dump(kmeans, 'models/kmeans.joblib')
print("Modelo guardado en models/kmeans.joblib")