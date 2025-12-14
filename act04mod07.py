# 1. Carga de datos (1 punto)
# • Carga el conjunto de datos proporcionado en el material complementario.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# Carga de datos
df = pd.read_csv("dataset_clientes.csv")
# Inspección y limpieza
print("\n",df,"\n")
print("\n",df.info(),"\n")
# Se elimina la columna 'ID_Cliente' ya que no es relevante para el clustering
df_processed = df.drop('ID_Cliente', axis=1)
# Escalado de datos
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_processed)
df_scaled_df = pd.DataFrame(df_scaled, columns=df_processed.columns)
print("\nPrimeras 5 filas del DataFrame escalado:")
print("\n",df_scaled_df.head(),"\n")

# 2. Aplicación de algoritmos de clusterización (5 puntos)
# • Agrupamiento Jerárquico:
# ↠ Aplica clustering jerárquico y genera el dendrograma para visualizar la
# jerarquía de clusters.
# ↠ Elige el número óptimo de clusters y justifica tu elección.
# Generar el dendrograma
plt.figure(figsize=(15, 7))
linked = linkage(df_scaled, method='ward')
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrograma de Clientes')
plt.xlabel('Índice de Clientes')
plt.ylabel('Distancia Euclidiana')
plt.show()
# Justificación del número de clusters
# En el dendrograma, se puede observar que un corte horizontal a una altura de aproximadamente 10-15
# produce 3 o 4 clusters bien definidos. Se elige K=4 para un análisis más detallado.
n_clusters_jerarquico = 4
# Aplicación del clustering jerárquico
hc = AgglomerativeClustering(n_clusters=n_clusters_jerarquico, metric='euclidean', linkage='ward')
df['cluster_jerarquico'] = hc.fit_predict(df_scaled)
print(f"\nClustering jerárquico aplicado con K={n_clusters_jerarquico}.")
print("Distribución de clusters:")
print(df['cluster_jerarquico'].value_counts())
# • K-Means:
# ↠ Aplica el algoritmo K-Means con un valor inicial de K=3.
# ↠ Determina el valor óptimo de K utilizando el método del codo y el
# coeficiente de silueta.
# K-Means con K=3
kmeans_k3 = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster_kmeans_k3'] = kmeans_k3.fit_predict(df_scaled)
print("\nK-Means aplicado con K=3.")
print("Distribución de clusters:")
print(df['cluster_kmeans_k3'].value_counts())

# Método del Codo
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)
    
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método del Codo para K-Means')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inercia')
plt.grid(True)
plt.show()

# Coeficiente de Silueta
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)
    score = silhouette_score(df_scaled, clusters)
    silhouette_scores.append(score)
    print(f"K={i}, Coeficiente de Silueta: {score:.4f}")

optimal_k = np.argmax(silhouette_scores) + 2
print(f"\nEl valor óptimo de K según el coeficiente de silueta es: {optimal_k}")

# Aplicar K-Means con el K óptimo
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster_kmeans_optimal'] = kmeans_optimal.fit_predict(df_scaled)

kmeans_k4 = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster_kmeans_k4'] = kmeans_k4.fit_predict(df_scaled)
print("\nK-Means aplicado con K=4.")
print("Distribución de clusters:")
print(df['cluster_kmeans_k4'].value_counts())
print("----Comparación Cluster Jerarquizo versus K-Means K=4------")
print(df['cluster_jerarquico'].value_counts(),"\n")
print(df['cluster_kmeans_k4'].value_counts())
# • DBSCAN:
# ↠ Aplica DBSCAN con distintos valores de eps y MinPts.
# ↠ Justifica la elección de eps usando la gráfica k-Distance.

# Justificación de eps con la gráfica k-Distance
min_pts = 2 * df_scaled.shape[1]
neighbors = NearestNeighbors(n_neighbors=min_pts)
neighbors_fit = neighbors.fit(df_scaled)
distances, indices = neighbors_fit.kneighbors(df_scaled)
distances = np.sort(distances[:, min_pts-1], axis=0)

plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.title('Gráfica k-Distance para DBSCAN')
plt.xlabel('Puntos de Datos (ordenados)')
plt.ylabel(f'Distancia al {min_pts}-ésimo vecino')
plt.grid(True)
plt.show()

# Justificación del valor de eps:
# El "codo" o punto de inflexión en la gráfica k-distance se encuentra aproximadamente en un valor de 1.
# Este es un buen candidato para el parámetro eps.
eps_val = 1.0

# Aplicar DBSCAN
dbscan = DBSCAN(eps=eps_val, min_samples=min_pts)
df['cluster_dbscan'] = dbscan.fit_predict(df_scaled)

print(f"\nDBSCAN aplicado con eps={eps_val} y MinPts={min_pts}.")
print("Distribución de clusters:")
# Los puntos con cluster -1 son considerados ruido
print(df['cluster_dbscan'].value_counts())
print("---- Comparación Cluster Jerarquizo versus K-Means K=4 y DBSCAN eps=1 ------")
print(df['cluster_jerarquico'].value_counts(),"\n")
print(df['cluster_kmeans_k4'].value_counts(),"\n")
print(df['cluster_dbscan'].value_counts())

# 3. Análisis de resultados (4 puntos)
# • Comparación de métodos:
# ↠ Explica las diferencias en la agrupación obtenida con clustering jerárquico, K-Means y DBSCAN. ¿Cuál fue más efectivo y por qué?
# • Interpretación:
# ↠ Describe los patrones que identificaste en cada método. ¿Hubo clusters claramente definidos?
# • Aplicabilidad
# ↠ Explica en qué casos sería recomendable usar K-Means, DBSCAN o agrupamiento jerárquico según la naturaleza de los datos.

# Comparación de Métodos
# Clustering Jerárquico: Genera una jerarquía, lo que es útil para entender las relaciones anidadas entre los clientes. La agrupación 
# obtenida es más rígida y no tan flexible como DBSCAN.
# K-Means: Produce clusters esféricos de tamaños similares. Es rápido y efectivo si los clusters son de esta forma. La desventaja es que 
# es sensible a outliers.
# DBSCAN: Identifica clusters de formas irregulares y marca los outliers como ruido (cluster -1). Es más flexible que K-Means pero su 
# desempeño depende mucho de la elección de los parámetros eps y MinPts.
# Efectividad: En este caso, si los datos sugieren clusters no esféricos o si la detección de outliers es importante, DBSCAN podría ser 
# más efectivo. Si los clusters son de formas relativamente regulares, K-Means ofrece un resultado más limpio y fácil de interpretar. 
# El jerárquico es una excelente herramienta exploratoria.

# Interpretación y Patrones
# Para interpretar los patrones, se analizan las características promedio de cada cluster.
# Resumen de K-Means
print("\n--- Resumen de clusters K-Means (K óptimo) ---")
cluster_summary_kmeans = df.groupby('cluster_kmeans_optimal').mean().round(2)
print(cluster_summary_kmeans)

# Resumen de DBSCAN (excluyendo el ruido)
print("\n--- Resumen de clusters DBSCAN ---")
dbscan_clusters_filtered = df[df['cluster_dbscan'] != -1]
cluster_summary_dbscan = dbscan_clusters_filtered.groupby('cluster_dbscan').mean().round(2)
print(cluster_summary_dbscan)

# Patrones identificados (ejemplo hipotético basado en el resumen de K-Means):ç
# Cluster 0: Clientes con ingresos y gastos anuales bajos. Es el grupo de clientes con menor valor.
# Cluster 1: Clientes con ingresos y gastos anuales altos. Este es el grupo de "clientes VIP".
# Cluster 2: Clientes con ingresos y gastos moderados, pero con una alta frecuencia de compra. Son clientes leales.
# Cluster 3: Clientes de mediana edad y baja frecuencia de compra.
# ¿Clusters claramente definidos? Sí, en el caso de K-Means, la partición en 4 grupos revela diferencias notables en los promedios de 
# las variables. DBSCAN también identifica clusters definidos, además de aislar los outliers, lo que podría ser una ventaja si la empresa 
# quiere estudiar esos casos atípicos.

# Aplicabilidad
# K-Means: Es recomendable cuando los clusters son esféricos y de tamaño similar. Es ideal para la segmentación de clientes, ya que es 
# rápido, escalable y los resultados son intuitivos de interpretar.
# DBSCAN: Se debe usar cuando los clusters tienen formas irregulares o se necesita identificar valores atípicos (anomalías). Es perfecto 
# para la detección de fraude o el análisis de datos geográficos.
# Agrupamiento Jerárquico: Es la mejor opción cuando se quiere entender la estructura inherente de los datos y las relaciones anidadas 
# entre los puntos. Es ideal para la clasificación biológica o en estudios de taxonomía.