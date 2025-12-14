# 1. Carga y exploración del dataset (1 punto)
# • Descarga y carga el archivo usuarios_musica.csv.
# • Muestra las primeras 5 filas y el resumen estadístico de los datos
# Carga y exploración del dataset

import pandas as pd

# Cargar datos
df = pd.read_csv("usuarios_musica.csv")

# Mostrar primeras filas y resumen estadístico
print("Primeras 5 filas:\n", df.head())
print("\nResumen estadístico:\n", df.describe())
print("\nValores faltantes:\n", df.isnull().sum())

# Limpieza y preprocesamiento (1 punto)
# • Revisa valores faltantes o atípicos y aplica imputación o eliminación según sea conveniente.
# • Normaliza los datos numéricos con StandardScaler.
# Paso 2: Limpieza y preprocesamiento

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Imputación de valores faltantes con la media
imputer = SimpleImputer(strategy="mean")
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
print("\n",df_imputed.head(),"\n")
print("\nValores faltantes:\n", df_imputed.isnull().sum())
# Escalado de los datos
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df.columns)
print("\n",df_scaled.head(),"\n")
# 3. Aplicación de K-Means para clusterización (3 puntos)
# • Aplica K-Means con un valor k=3.
# • Agrega la columna "cluster" al DataFrame con el resultado.
# • Visualiza los clusters usando una gráfica 2D (usa PCA para reducir dimensiones si es necesario).

# Aplicación de K-Means
from sklearn.cluster import KMeans

# Aplicar K-Means con k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_scaled["cluster"] = kmeans.fit_predict(df_scaled)

# Agregar resultado al DataFrame original
df_imputed["cluster"] = df_scaled["cluster"]

# Mostrar distribución
print("\nUsuarios por cluster:\n", df_imputed["cluster"].value_counts())

# 4. Reducción de dimensionalidad con PCA (2 puntos)
# • Aplica PCA para reducir las dimensiones a 2 componentes principales.
# • Interpreta brevemente qué representa cada componente principal.
# • Grafica los puntos proyectados en 2D, coloreados por su cluster.

# Reducción de dimensionalidad con PCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# PCA a 2 componentes
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled.drop("cluster", axis=1))

# Agregar componentes al DataFrame
df_scaled["PCA1"] = pca_result[:, 0]
df_scaled["PCA2"] = pca_result[:, 1]

# Gráfico 2D
plt.figure(figsize=(10, 6))
sns.scatterplot(x="PCA1", y="PCA2", hue="cluster", data=df_scaled, palette="Set2", s=60)
plt.title("Visualización de Clusters con PCA")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(True)
plt.show()

# Varianza explicada
print("Varianza explicada por cada componente:", pca.explained_variance_ratio_)

# 5. Detección de anomalías (2 puntos)
# • Calcula la distancia euclidiana entre cada punto y el centro de su cluster.
# • Marca como anomalías a los 5 usuarios más alejados de sus respectivos centroides.
# • Agrega una columna "anomalía" con valores True/False.

# Paso 5: Detección de anomalías
import numpy as np

# Calcular distancia euclidiana al centroide
centroids = kmeans.cluster_centers_
distances = []

for i, row in df_scaled.iterrows():
    cluster_id = int(row["cluster"])
    centroid = centroids[cluster_id]
    point = row.drop(["cluster", "PCA1", "PCA2"]).values
    distance = np.linalg.norm(point - centroid)
    distances.append(distance)

# Agregar distancia y detectar anomalías
df_imputed["distancia_al_centroide"] = distances
top5 = df_imputed["distancia_al_centroide"].nlargest(5)
df_imputed["anomalía"] = df_imputed["distancia_al_centroide"].isin(top5)

# Mostrar usuarios anómalos
print("\nUsuarios anómalos:\n", df_imputed[df_imputed["anomalía"] == True])

# 6. Reflexión final (1 punto)
# • ¿Qué hallazgos consideras más útiles para el equipo de marketing?
# • ¿Cómo podrían usar esta segmentación en campañas personalizadas?

# Paso 6: Reflexión final (comentario)

print("\nREFLEXIÓN FINAL:")
print("""
Hallazgos útiles para el equipo de marketing:
- Se identificaron 3 tipos de usuarios con patrones distintos de comportamiento.
- Algunos escuchan muchos artistas distintos (curiosos), otros repiten canciones o usan 
múltiples dispositivos.

Aplicación en campañas:
- Recomendaciones más acertadas (nuevos artistas para exploradores).
- Promociones premium a usuarios activos.
- Alertas tempranas para usuarios con comportamiento anómalo (riesgo de abandono).
""")
