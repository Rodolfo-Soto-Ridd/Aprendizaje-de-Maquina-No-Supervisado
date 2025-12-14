# 1. Carga y exploración de datos (1 punto)
# • Carga el dataset proporcionado, que contiene información sobre la popularidad de distintos
# géneros musicales en países como Chile, EE.UU., México, Corea, Japón, Alemania, Rusia e Italia.
# • Analiza las características del dataset, identificando distribuciones y tendencias iniciales.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Cargar el dataset
df = pd.read_csv('dataset_generos_musicales.csv')

# Guardar los nombres de los países para referencia
paises = df['País']
df_numerico = df.drop('País', axis=1)
print(df.info(),"\n")
print("Datos originales:")
print(df)
print("\nEstadísticas descriptivas:")
print(df_numerico.describe(),"\n")

# Normalizar los datos para que todos los géneros tengan la misma importancia
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numerico)

# Crear un DataFrame con los datos normalizados para facilitar la visualización
df_scaled_df = pd.DataFrame(df_scaled, columns=df_numerico.columns, index=paises)
print("\nDatos normalizados:")
print(df_scaled_df,"\n")

# 2. Aplicación de algoritmos de clusterización (5 puntos)
# K-Means:
# • Aplica el algoritmo K-Means con un valor inicial de K=3.
# • Determina el valor óptimo de K utilizando el método del codo y el coeficiente de silueta.
# K-Means con K=3
kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters_kmeans_3 = kmeans_3.fit_predict(df_scaled)
df_kmeans = df.copy()
df_kmeans['Cluster_K3'] = clusters_kmeans_3
print("Resultados de K-Means con K=3:")
print(df_kmeans[['País', 'Cluster_K3']])

# Método del codo para determinar el K óptimo
inercia = []
for i in range(1, len(paises)):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inercia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(paises)), inercia, marker='o')
plt.title('Método del Codo')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inercia')
plt.xticks(range(1, len(paises)))
plt.show()

# Coeficiente de silueta para determinar el K óptimo
silueta = []
for i in range(2, len(paises)):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)
    silueta.append(silhouette_score(df_scaled, clusters))

plt.figure(figsize=(10, 6))
plt.plot(range(2, len(paises)), silueta, marker='o')
plt.title('Coeficiente de Silueta')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Coeficiente de Silueta')
plt.xticks(range(2, len(paises)))
plt.show()

# K-Means con el K óptimo (basado en los gráficos)
k_optimo = 2 # El gráfico del codo muestra una curva en K=2, y el silueta tiene un pico en K=4 --> ESTABA EN LO CORRECTO.
kmeans_optimo = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
clusters_kmeans_optimo = kmeans_optimo.fit_predict(df_scaled)
df_kmeans_optimo = df.copy()
df_kmeans_optimo['Cluster_K_Optimo'] = clusters_kmeans_optimo
print(f"\nResultados de K-Means con K óptimo ({k_optimo}):")
print(df_kmeans_optimo[['País', 'Cluster_K_Optimo']])

# Clustering jerárquico:
# • Genera un dendrograma y determina el número óptimo de clusters.
# • Aplica clustering jerárquico y compara con los resultados de K-Means.

# Generar el dendrograma
plt.figure(figsize=(15, 8))
dendrograma = dendrogram(linkage(df_scaled, method='ward'), labels=paises.tolist()) # ERAN 3 GRUPOS, EL DENDOGRAMA DA MUY FINO
plt.title('Dendrograma de Clustering Jerárquico')
plt.xlabel('Países')
plt.ylabel('Distancia Euclidiana')
plt.show()

# Aplicar Clustering Jerárquico con 3 clusters
hc = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
clusters_hc = hc.fit_predict(df_scaled)
df_hc = df.copy() # --> SI, ESTO ES PARA PRESERVAR EL DATAFRAME ORIGINAL
df_hc['Cluster_HC'] = clusters_hc
print("\nResultados del Clustering Jerárquico:")
print(df_hc[['País', 'Cluster_HC']])

# DBSCAN:
# • Aplica DBSCAN con diferentes valores de eps y MinPts.
# • Justifica la elección de los parámetros y analiza si DBSCAN identifica agrupaciones 
# significativas.

from sklearn.neighbors import NearestNeighbors
neighbors = NearestNeighbors(n_neighbors=2).fit(df_scaled)
distances, indices = neighbors.kneighbors(df_scaled)
distances = np.sort(distances[:, 1], axis=0)

plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.title('Gráfico de Distancias para DBSCAN')
plt.xlabel('Puntos')
plt.ylabel('Distancia al 2do vecino más cercano')
plt.show()

# En base al gráfico, elegimos eps=1.5
dbscan = DBSCAN(eps=3.382, min_samples=2) 
clusters_dbscan = dbscan.fit_predict(df_scaled)
df_dbscan = df.copy()
df_dbscan['Cluster_DBSCAN'] = clusters_dbscan
print(f"\nResultados de DBSCAN (eps=3.382 , min_samples=2):")
print(df_dbscan[['País', 'Cluster_DBSCAN']])

# 3. Aplicación de reducción de dimensionalidad (3 puntos)
# PCA:
# • Aplica PCA y determina cuántos componentes principales explican al menos el 90% de la varianza.
# • Visualiza los países en un gráfico bidimensional con las primeras dos componentes principales.

pca = PCA()
df_pca = pca.fit_transform(df_scaled)
explicacion_varianza = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explicacion_varianza) + 1), explicacion_varianza, marker='o') # ESTO ES PARA INCLUIR EL ULTIMO DATO (ES COMO FUNCIONA EL COMANDO "RANGE")
plt.title('Varianza Acumulada Explicada por Componentes Principales')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza Acumulada Explicada')
plt.axhline(y=0.90, color='r', linestyle='--')
plt.text(1, 0.92, '90% de Varianza', color='red')
plt.show()

# Visualizar en 2D con las 2 primeras componentes
pca_2d = PCA(n_components=2)
df_pca_2d = pca_2d.fit_transform(df_scaled)
df_pca_visual = pd.DataFrame(df_pca_2d, columns=['PC1', 'PC2'], index=paises)
df_pca_visual['Cluster_K_Optimo'] = clusters_kmeans_optimo

plt.figure(figsize=(12, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster_K_Optimo', data=df_pca_visual, s=100, palette='viridis')
for i, txt in enumerate(paises):
    plt.annotate(txt, (df_pca_visual['PC1'][i], df_pca_visual['PC2'][i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.title('PCA 2D con Clusters K-Means')
plt.xlabel(f'Componente Principal 1 ({pca_2d.explained_variance_ratio_[0]:.2f}% de varianza)')
plt.ylabel(f'Componente Principal 2 ({pca_2d.explained_variance_ratio_[1]:.2f}% de varianza)')
plt.show()

print(f"\nNúmero de componentes principales que explican al menos el 90% de la varianza: {np.argmax(explicacion_varianza >= 0.90) + 1}")

# t-SNE:
# • Aplica t-SNE para visualizar la relación entre los países en un espacio de 2D.
# • Experimenta con diferentes valores de perplexity y analiza cómo afectan la representación.

# Aplicar t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=3, n_iter=1000)
df_tsne = tsne.fit_transform(df_scaled)
df_tsne_visual = pd.DataFrame(df_tsne, columns=['Dim1', 'Dim2'], index=paises)
df_tsne_visual['Cluster_K_Optimo'] = clusters_kmeans_optimo

plt.figure(figsize=(12, 8))
sns.scatterplot(x='Dim1', y='Dim2', hue='Cluster_K_Optimo', data=df_tsne_visual, s=100, palette='viridis')
for i, txt in enumerate(paises):
    plt.annotate(txt, (df_tsne_visual['Dim1'][i], df_tsne_visual['Dim2'][i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.title('t-SNE 2D con Clusters K-Means (Perplexity=3)')
plt.show()

# Experimentar con diferentes perplexity (por ejemplo, 1 y 7)
tsne_1 = TSNE(n_components=2, random_state=42, perplexity=1)
df_tsne_1 = tsne_1.fit_transform(df_scaled)

tsne_7 = TSNE(n_components=2, random_state=42, perplexity=7)
df_tsne_7 = tsne_7.fit_transform(df_scaled)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
df_tsne_visual_1 = pd.DataFrame(df_tsne_1, columns=['Dim1', 'Dim2'], index=paises)
sns.scatterplot(ax=axes[0], x='Dim1', y='Dim2', hue=clusters_kmeans_optimo, data=df_tsne_visual_1, s=100, palette='viridis')
for i, txt in enumerate(paises):
    axes[0].annotate(txt, (df_tsne_visual_1['Dim1'][i], df_tsne_visual_1['Dim2'][i]), textcoords="offset points", xytext=(0,10), ha='center')
axes[0].set_title('t-SNE 2D (Perplexity=1)')

df_tsne_visual_7 = pd.DataFrame(df_tsne_7, columns=['Dim1', 'Dim2'], index=paises)
sns.scatterplot(ax=axes[1], x='Dim1', y='Dim2', hue=clusters_kmeans_optimo, data=df_tsne_visual_7, s=100, palette='viridis')
for i, txt in enumerate(paises):
    axes[1].annotate(txt, (df_tsne_visual_7['Dim1'][i], df_tsne_visual_7['Dim2'][i]), textcoords="offset points", xytext=(0,10), ha='center')
axes[1].set_title('t-SNE 2D (Perplexity=7)')
plt.show()

# 4. Análisis de resultados y conclusiones (1 punto)
# Comparación de métodos:

# • Explica las diferencias entre K-Means, clustering jerárquico y DBSCAN. ¿Cuál funcionó mejor en este caso y por qué?
# K-Means vs. Clustering Jerárquico vs. DBSCAN:
# K-Means y el Clustering Jerárquico funcionaron de manera excelente en este dataset. Ambos algoritmos identificaron los mismos tres 
# grupos de países, lo que indica que la estructura de los datos es robusta y los clusters están bien definidos. La principal ventaja 
# de K-Means es su eficiencia computacional en datasets grandes, mientras que el clustering jerárquico ofrece una visualización intuitiva 
# a través del dendrograma que muestra las relaciones de anidamiento entre los países.
# DBSCAN no fue adecuado para este problema. Su principal desventaja es su sensibilidad a los parámetros y su suposición de que los 
# clusters son áreas de alta densidad separadas por áreas de baja densidad, una condición que no se cumplió en este dataset. 
# Como resultado, la mayoría de los países fueron clasificados como ruido, lo que hace que sus resultados sean poco útiles.

# • Compara los resultados obtenidos con PCA y t-SNE. ¿Cuál técnica permitió visualizar mejor la relación entre los países?
# PCA vs. t-SNE:
# PCA es útil para encontrar las direcciones de mayor varianza en los datos y para entender cuánta información se pierde al reducir la 
# dimensionalidad. Sin embargo, su visualización 2D puede no separar los clusters de manera tan clara como t-SNE, ya que PCA busca 
# preservar la varianza global, no las relaciones locales.
# t-SNE fue superior para la visualización. Su capacidad para preservar las relaciones locales entre los puntos en el espacio de alta 
# dimensión hace que la separación de los clusters sea más evidente en el gráfico 2D. La visualización de t-SNE con un perplexity de 
# 3 o 7 muestra una clara separación de los clusters, lo que valida la agrupación.  

paises_cluster_1=["Chile","Corea","Rusia"]
df_cluster_1=df[df['País'].isin(paises_cluster_1)]
print("\n",df_cluster_1,"\n")
paises_cluster_2=["EEUU","México","Alemania"]
df_cluster_2=df[df['País'].isin(paises_cluster_2)]
print("\n",df_cluster_2,"\n")
paises_cluster_3=["Japón","Italia"]
df_cluster_3=df[df['País'].isin(paises_cluster_3)]
print("\n",df_cluster_3,"\n")

# Interpretación:
# • ¿Los clusters obtenidos reflejan similitudes culturales o geográficas en la música?
# Los clusters obtenidos reflejan similitudes culturales y geográficas en la música.
# Respuesta: Los cluster considerador fueron K-means (K=3) y cluster jerarquico que indicaron las mismas agrupaciones
# DBSCAN no agrupo de la misma manera, debe ser por los pocos datos proporcionados
# Cluster 0 (Chile, Corea, Rusia): Se observa una clara preferencia a los generos Pop y Hip - Hop, y una baja preferencia en el 
# genero Rock. Lo que puede ser indico que fuertes culturas urbanas e influenciadas por medios masivos de musica.
# Cluster 1 (EEUU, Mexico, Alemania): Se observa una clara preferencia por la musica electronica, y una baja preferencia por 
# el genero metal. Esto se puede deber a la fuerza cultural de la musica occidental.
# Cluster 2 (Japón e Italia): Se observa amplias afinidas musicales entre estos dos paises, excepto en el genero Rock. 
# Esto se puede deber a que ambos paises poseen un larga historia de multiculturalidad que los a acercado a diversos generos
# musicales durante muchos años.

# • Relaciona los resultados con tendencias globales en consumo musical.
# Estos resultados sugieren que, si bien la geografía puede jugar un papel (como en EEUU/Mexico), las tendencias 
# culturales y sociales de consumo musical pueden trascender las fronteras geográficas, como se ve en el cluster de Chile, 
# Corea y Rusia, o en Japón con Italia.