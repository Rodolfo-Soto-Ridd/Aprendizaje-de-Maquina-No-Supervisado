# Importar librerías

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn import datasets
 
# Cargar el dataset Iris

iris = datasets.load_iris()

X = iris.data[:, :2]  # Usamos solo las dos primeras características para visualizar mejor
 
# Aplicar el método del codo

wcss = []  # Lista para almacenar el WCSS de cada K
 
# Probar valores de K desde 1 hasta 10

for k in range(1, 11):

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)  # inertia_ es el WCSS en scikit-learn
 
# Graficar K vs WCSS

plt.figure(figsize=(8, 5))

plt.plot(range(1, 11), wcss, marker='o', linestyle='-')

plt.title("Método del Codo para determinar K")

plt.xlabel("Número de Clusters (K)")

plt.ylabel("WCSS (Within-Cluster Sum of Squares)")

plt.xticks(range(1, 11))

plt.grid(True)

plt.show()

# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
 
# Cargar el dataset Iris
iris = datasets.load_iris()
X = iris.data[:, :2]  # Usamos solo las dos primeras características para visualizar mejor
 
# Aplicar K-Means con K=3 (porque sabemos que hay 3 clases en Iris)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)         # Asigna cada punto a un cluster
centroids = kmeans.cluster_centers_    # Obtiene los centroides
 
# Graficar los puntos y los clusters
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', label="Datos")
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label="Centroides")
 
plt.xlabel("Longitud del Sépalo")
plt.ylabel("Ancho del Sépalo")
plt.title("Clustering K-Means en Iris (K=3)")
plt.legend()
plt.show()

# Importar librerías

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn import datasets

from sklearn.metrics import silhouette_score
 
# Cargar el dataset Iris

iris = datasets.load_iris()

X = iris.data[:, :2]  # Usamos solo las dos primeras características para facilitar visualización
 
# Aplicar el método de la silueta

silhouette_scores = []  # Lista para guardar los coeficientes promedio
 
# Evaluar K desde 2 hasta 10 (K=1 no tiene sentido para silueta)

for k in range(2, 11):

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

    labels = kmeans.fit_predict(X)

    sil_score = silhouette_score(X, labels)

    silhouette_scores.append(sil_score)
 
# Graficar K vs Silhouette Score

plt.figure(figsize=(8, 5))

plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='-')

plt.title("Método del Coeficiente de Silueta para determinar K")

plt.xlabel("Número de Clusters (K)")

plt.ylabel("Coeficiente de Silueta Promedio")

plt.xticks(range(2, 11))

plt.grid(True)

plt.show()

# Importar librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn import datasets
 
# 1. Cargar el dataset Iris
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
 
# Usamos solo las dos primeras columnas para facilitar la visualización
data = df.iloc[:, [0, 1]].values  # 'sepal length' y 'sepal width'
 
# 2. Construcción del dendrograma para analizar agrupaciones
plt.figure(figsize=(10, 5))
linkage_matrix = sch.linkage(data, method='ward')  # 'ward' minimiza la varianza dentro de los clusters
sch.dendrogram(linkage_matrix)
plt.title("Dendrograma del Agrupamiento Jerárquico")
plt.xlabel("Puntos de datos")
plt.ylabel("Distancia")
plt.show()
 
# 3. Elección del número de clusters basado en el dendrograma
num_clusters = 3  # Se elige observando el dendrograma
 
# 4. Aplicación del clustering jerárquico
hc = AgglomerativeClustering(n_clusters=num_clusters, metric='euclidean', linkage='ward')
y_clusters = hc.fit_predict(data)
 
# 5. Visualización de los clusters resultantes
plt.figure(figsize=(8, 5))
for i in range(num_clusters):
    plt.scatter(data[y_clusters == i, 0], data[y_clusters == i, 1], label=f'Cluster {i+1}')
 
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Agrupamiento Jerárquico - Clusters Encontrados")
plt.legend()
plt.show()
 
# 6. Mostrar los resultados en la terminal
print("Asignación de clusters para los primeros 10 datos:")
print(y_clusters[:10])  # Mostramos solo los primeros 10 resultados   

# Importar librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
 
# 1. Generar un conjunto de datos con forma de luna creciente
X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
X = StandardScaler().fit_transform(X)  # Normalizar datos para mejorar resultados
 
# 2. Aplicar DBSCAN con eps=0.3 y min_samples=5
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels_dbscan = dbscan.fit_predict(X)
 
# 3. Aplicar K-Means con K=2 para comparar resultados
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X)
 
# 4. Graficar los resultados
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
 
# Gráfico DBSCAN
axs[0].scatter(X[:, 0], X[:, 1], c=labels_dbscan, cmap='viridis', marker='o', edgecolor='k')
axs[0].set_title("Clustering con DBSCAN")
 
# Gráfico K-Means
axs[1].scatter(X[:, 0], X[:, 1], c=labels_kmeans, cmap='viridis', marker='o', edgecolor='k')
axs[1].set_title("Clustering con K-Means")
 
plt.show()