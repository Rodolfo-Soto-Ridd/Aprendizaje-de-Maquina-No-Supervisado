## Ejemplo K-Means
 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
 
# Generamos datos aleatorios
np.random.seed(0)
X = np.random.rand(100, 2) * 10  # 100 puntos en un espacio 2D
 
# Aplicamos K-Means con 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
labels = kmeans.labels_
centroides = kmeans.cluster_centers_
 
# Graficamos los clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centroides[:, 0], centroides[:, 1], c='red', marker='X', s=200, label="Centroides")
plt.title("Clusterización con K-Means")
plt.legend()
plt.show()

## Ejemplo Cluster Jerárquico
 
import scipy.cluster.hierarchy as sch

from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt
 
# Creamos el dendrograma

plt.figure(figsize=(8, 5))

dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))

plt.title("Dendrograma - Clustering Jerárquico")

plt.show()
 
# Aplicamos el clustering jerárquico con 3 clusters

hc = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')

labels_hc = hc.fit_predict(X)
 
# Graficamos los clusters

plt.scatter(X[:, 0], X[:, 1], c=labels_hc, cmap='plasma', marker='o')

plt.title("Clusterización Jerárquica")

plt.show()

## Ejemplo DBScan
 
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
 
# Aplicamos DBSCAN
dbscan = DBSCAN(eps=1.35, min_samples=5)
labels_dbscan = dbscan.fit_predict(X)
 
# Graficamos los clusters
plt.scatter(X[:, 0], X[:, 1], c=labels_dbscan, cmap='coolwarm', marker='o')
plt.title("Clusterización con DBSCAN")
plt.show() 

## Ejemplo GMM
 
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
 
# Aplicamos el modelo GMM con 3 clusters

gmm = GaussianMixture(n_components=3, random_state=0)

gmm.fit(X)

labels_gmm = gmm.predict(X)
 
# Graficamos los clusters

plt.scatter(X[:, 0], X[:, 1], c=labels_gmm, cmap='coolwarm', marker='o')

plt.title("Clusterización con Mezcla Gaussiana (GMM)")

plt.show()

 