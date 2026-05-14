import pandas as pd
import numpy as np
import yaml
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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

# Entrenar
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=params['modeling']['random_state']
)
rf.fit(X_train, y_train)

# Evaluar
y_pred = rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")

# Guardar modelo
joblib.dump(rf, 'models/random_forest.pkl')
print("Modelo guardado en models/random_forest.pkl")