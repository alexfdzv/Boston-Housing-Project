# Boston Housing Project

Proyecto integrador de analisis de datos usando Git y DVC sobre el dataset
Boston Housing.

## Alcance actual

Este repositorio contiene el pipeline base para:

1. Cargar el dataset crudo.
2. Limpiar valores faltantes y outliers.
3. Normalizar variables numericas.
4. Entrenar un modelo no supervisado K-Means.
5. Entrenar un modelo lineal supervisado.

## Estructura

- `data/raw/`: datos originales controlados con DVC.
- `data/processed/`: datos generados por el pipeline.
- `data/model_ready/`: datos normalizados especificos para cada modelo.
- `src/`: scripts reproducibles del pipeline.
- `notebooks/`: evidencia exploratoria, visualizaciones y justificaciones.
- `models/`: modelos entrenados generados por DVC.
- `metrics/`: metricas generadas por los modelos.

## Pipeline

```powershell
py -3.12 -m dvc repro
```

Stages principales:

- `load`: lee `data/raw/housing.csv` y asigna nombres de columnas.
- `clean`: revisa valores faltantes y aplica capping de outliers con IQR.
- `normalize`: aplica `StandardScaler` a variables numericas relevantes.
- `kmeans`: entrena K-Means con los datos normalizados y genera clusters.
- `linear_model`: entrena una regresion lineal sobre los datos normalizados.

## Limpieza y preparacion

El dataset se revisa para detectar valores faltantes. Si aparecieran nulos, se
imputan con la mediana porque es robusta ante valores extremos.

Los outliers se tratan con IQR capping. Se eligio capping en lugar de eliminar
filas para conservar el tamano del dataset y reducir el impacto de valores
extremos en modelos sensibles.

`CHAS` se excluye del tratamiento de outliers y normalizacion porque es una
variable binaria. `MEDV` se excluye porque es la variable objetivo.

## Normalizacion

Se usa `StandardScaler` porque las variables tienen escalas muy distintas.
Esto es especialmente importante para algoritmos basados en distancia como
K-Means.

## K-Means

K-Means segmenta vecindarios usando las variables normalizadas, excluyendo
`MEDV` para evitar usar la variable objetivo como insumo del clustering.

El stage `kmeans` genera:

- `data/processed/housing_kmeans.csv`
- `data/processed/kmeans_cluster_profile.csv`
- `metrics/kmeans_metrics.json`
- `models/kmeans_model.joblib`

El numero de clusters se configura en `params.yaml`.

## Datos Normalizados Por Modelo

El stage `normalize` genera datasets listos para cada algoritmo:

- `data/model_ready/kmeans/housing_kmeans_scaled.csv`
- `data/model_ready/linear_model/housing_linear_scaled.csv`

Esto permite documentar que cada modelo recibe una version preparada
explicitamente para su entrenamiento. K-Means usa su archivo model-ready y
excluye `MEDV` durante el entrenamiento, aunque conserva `MEDV` en las salidas
para interpretar los clusters.

## Validacion De K-Means

El notebook `notebooks/algorithm1.ipynb` incluye:

- comparacion de diferentes `random_state`
- inertia por configuracion
- silhouette score por configuracion
- diagrama de silueta del modelo final
- grafica `LSTAT` vs `RM` coloreada por cluster

## Instalacion

```powershell
py -3.12 -m pip install -r requirements.txt
```

Si los datos no estan presentes localmente, primero hay que recuperar los
archivos controlados por DVC:

```powershell
py -3.12 -m dvc pull
```

Nota: el remoto DVC debe ser accesible desde la computadora donde se ejecute el
proyecto.
