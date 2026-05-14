import pandas as pd
import numpy as np
import yaml
import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Cargar datos
df = pd.read_csv(params['data']['scaled_path'])

# Separar features y target
X = df.drop(columns=['MEDV'])
y = df['MEDV']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params['modeling']['test_size'],
    random_state=params['modeling']['random_state']
)

# Encontrar mejor k
k_range = range(1, 21)
rmse_scores = []

for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse_scores.append(np.sqrt(-scores.mean()))

mejor_k = k_range[np.argmin(rmse_scores)]
print(f"Mejor k: {mejor_k}")

# Entrenar con mejor k
knn = KNeighborsRegressor(n_neighbors=mejor_k)
knn.fit(X_train, y_train)

# Evaluar
y_pred = knn.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")

# Guardar modelo
joblib.dump(knn, 'models/knn.pkl')
print("Modelo guardado en models/knn.pkl")