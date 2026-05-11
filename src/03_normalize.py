import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Cargar datos limpios
df = pd.read_csv(params['data']['processed_path'])

# Normalización
exclude = params['normalization']['exclude_columns']
cols_to_scale = [c for c in df.columns if c not in exclude]

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# Guardar
output_path = params['data']['scaled_path']
df_scaled.to_csv(output_path, index=False)

print(f"Dataset normalizado guardado en {output_path}")
print(f"Filas: {df_scaled.shape[0]}, Columnas: {df_scaled.shape[1]}")