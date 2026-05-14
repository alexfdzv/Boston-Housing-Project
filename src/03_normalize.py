import pandas as pd
import yaml
import os
from sklearn.preprocessing import StandardScaler

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Cargar datos limpios
df = pd.read_csv(params['data']['processed_path'])

# Normalizacion
exclude = params['normalization']['exclude_columns']
cols_to_scale = [c for c in df.columns if c not in exclude]

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# Guardar
output_path = params['data']['scaled_path']
model_ready_paths = params['data']['model_ready']

for path in [
    output_path,
    model_ready_paths['kmeans_scaled_path'],
    model_ready_paths['linear_scaled_path'],
]:
    os.makedirs(os.path.dirname(path), exist_ok=True)

df_scaled.to_csv(output_path, index=False)
df_scaled.to_csv(model_ready_paths['kmeans_scaled_path'], index=False)
df_scaled.to_csv(model_ready_paths['linear_scaled_path'], index=False)

print(f"Dataset normalizado guardado en {output_path}")
print(f"Dataset para K-Means guardado en {model_ready_paths['kmeans_scaled_path']}")
print(f"Dataset para regresion lineal guardado en {model_ready_paths['linear_scaled_path']}")
print(f"Filas: {df_scaled.shape[0]}, Columnas: {df_scaled.shape[1]}")
