# 1. Carga de datos (1 punto)
# Carga el conjunto de datos proporcionado en el material complementario. Este dataset contiene 
# información sobre densidad poblacional, tasas de migración e índices de delincuencia en diversas
# ciudades de Chile.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Cargar el conjunto de datos
df = pd.read_csv('dataset_ciudades_chile.csv')

# Mostrar las primeras filas del DataFrame
print("Datos originales:")
print(df.head())
print(df.info())
print(df.describe())

# 2. Aplicación de métodos de reducción de dimensionalidad (5 puntos)
# Análisis de Componentes Principales (PCA):
# • Aplica PCA al conjunto de datos y determina cuántos componentes principales son necesarios para 
# explicar al menos el 90% de la varianza.
# • Visualiza los datos reducidos en un gráfico bidimensional utilizando las primeras dos 
# componentes principales.
# Separar las características (X) de la columna de ciudades

features = ['Densidad_Poblacional', 'Tasa_Migracion', 'Indice_Delincuencia']
X = df[features]
cities = df['Ciudad']

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explicar la varianza
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
print("\nVarianza explicada por cada componente:")
print(explained_variance)
print("\nVarianza acumulada:")
print(cumulative_variance)

# Determinar el número de componentes para el 90% de la varianza
n_components = np.where(cumulative_variance >= 0.90)[0][0] + 1
print(f"\nSe necesitan {n_components} componentes principales para explicar al menos el 90% de la varianza.")

# Visualizar la varianza explicada acumulada
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.title('Varianza Acumulada Explicada por PCA')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Acumulada Explicada')
plt.grid(True)
plt.show()

# Visualizar los datos reducidos con las 2 primeras componentes
pca_df = pd.DataFrame(data=X_pca[:, :2], columns=['PC1', 'PC2'])
pca_df['Ciudad'] = cities

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', data=pca_df)
for i, city in enumerate(pca_df['Ciudad']):
    plt.text(pca_df['PC1'][i] + 0.1, pca_df['PC2'][i], city, fontsize=9)
plt.title('PCA: Ciudades de Chile (2 Componentes Principales)')
plt.xlabel(f'Componente Principal 1 ({explained_variance[0]*100:.2f}% de varianza)')
plt.ylabel(f'Componente Principal 2 ({explained_variance[1]*100:.2f}% de varianza)')
plt.grid(True)
plt.show()

# t-Distributed Stochastic Neighbor Embedding (t-SNE):
# • Aplica t-SNE para reducir la dimensionalidad del conjunto de datos a 2 dimensiones.
# • Experimenta con diferentes valores de perplexity y analiza cómo afectan los resultados.
# • Visualiza los datos en un gráfico de dispersión.
# Experimentar con diferentes valores de perplexity

perplexities = [5,10]
fig, axes = plt.subplots(1, 2, figsize=(24, 8))

for i, perplexity in enumerate(perplexities):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    tsne_df = pd.DataFrame(data=X_tsne, columns=['TSNE1', 'TSNE2'])
    tsne_df['Ciudad'] = cities
    
    sns.scatterplot(x='TSNE1', y='TSNE2', data=tsne_df, ax=axes[i])
    for j, city in enumerate(tsne_df['Ciudad']):
        axes[i].text(tsne_df['TSNE1'][j] + 0.1, tsne_df['TSNE2'][j], city, fontsize=9)
        
    axes[i].set_title(f't-SNE con Perplexity = {perplexity}')
    axes[i].set_xlabel('t-SNE Componente 1')
    axes[i].set_ylabel('t-SNE Componente 2')
    axes[i].grid(True)

plt.tight_layout()
plt.show()

# 3. Análisis de resultados (4 puntos)
# Comparación de métodos:
# • Explica las diferencias en la representación obtenida con PCA y t-SNE. ¿Cuál conserva mejor 
# la estructura de los datos?
# Interpretación:
# • Describe los patrones que identificaste en cada método. ¿Las ciudades con características
# similares se agrupan de forma clara?
# • Relaciona las agrupaciones con aspectos socioeconómicos o geográficos de las ciudades.
# Aplicabilidad:
# • Explica en qué casos sería recomendable usar PCA o t-SNE según la naturaleza de los datos
# y el objetivo del análisis.

# Comparación de métodos
# La principal diferencia entre PCA y t-SNE es que PCA es un método lineal que busca maximizar 
# la varianza de los datos en las nuevas dimensiones. Su objetivo principal es preservar la 
# estructura global de los datos, lo que significa que las distancias entre puntos lejanos en el 
# espacio original se mantienen relativamente lejanas en la proyección. Como resultado, las visualizaciones 
# de PCA pueden ser útiles para entender la variabilidad general de los datos, pero a menudo no 
# logran agrupar clústeres de forma clara.
# Por otro lado, t-SNE es un método no lineal que se enfoca en preservar las distancias entre los 
# puntos cercanos, sacrificando la estructura global. Su objetivo es que los puntos que están cerca 
# en el espacio de alta dimensión permanezcan cerca en el espacio de baja dimensión. Esto hace que 
# t-SNE sea excelente para identificar y visualizar clústeres, ya que agrupa eficazmente las ciudades 
# con características similares. Sin embargo, las distancias entre los clústeres en la visualización 
# de t-SNE no tienen un significado claro en relación con la distancia original en el espacio de datos.
# En este caso, t-SNE conserva mejor la estructura de los clústeres de datos, ya que su objetivo es 
# agrupar puntos similares.
# Interpretación
# PCA: En la visualización de PCA, las ciudades se dispersan a lo largo de las dos componentes principales.
# PC1 parece estar fuertemente relacionada con la Densidad Poblacional y el Índice de Delincuencia. 
# Ciudades como Valparaíso y Concepción tienen valores altos en esta componente, lo que podría indicar 
# una combinación de alta densidad y delincuencia. Antofagasta, Iquique y Temuco tienen valores bajos, 
# lo que sugiere menor densidad o delincuencia en comparación.
# PC2 podría estar relacionada con la Tasa de Migración. Ciudades como Arica y Antofagasta tienen valores 
# altos, lo que refleja sus altas tasas de migración positiva, mientras que La Serena y Rancagua tienen 
# valores muy bajos, lo que concuerda con sus tasas de migración negativa.
# No se observan agrupaciones claras, lo que confirma que PCA es mejor para entender la variabilidad 
# general que para encontrar clústeres discretos.
# t-SNE: La visualización de t-SNE, especialmente con perplexity de 30, muestra agrupaciones más claras.
# Se forma un clúster de ciudades con alta densidad poblacional y delincuencia, como Valparaíso y Concepción.
# Otro grupo incluye a ciudades con alta tasa de migración, como Arica y Antofagasta. Estas ciudades están 
# geográficamente distantes y tienen un perfil económico distinto (minería, comercio fronterizo) que impulsa 
# la migración.
# Un tercer clúster agrupa ciudades como Puerto Montt y Santiago, que tienen características más equilibradas 
# o distintivas que las separan de los otros grupos. La visualización de t-SNE es efectiva para identificar 
# estas relaciones locales.
# Las agrupaciones identificadas por t-SNE sugieren que las ciudades con perfiles socioeconómicos similares 
# (por ejemplo, ciudades portuarias con alta delincuencia o ciudades del norte con alta migración) tienden 
# a agruparse.
# Aplicabilidad
# PCA es ideal cuando el objetivo es reducir la dimensionalidad para la pre-procesamiento de modelos de 
# aprendizaje automático o para entender la variabilidad principal de los datos. Es útil cuando se necesita 
# preservar la estructura global y las distancias generales entre los puntos. Es un método más rápido y 
# determinista (el resultado es siempre el mismo) que t-SNE, por lo que es la opción preferida para grandes 
# datasets.
# t-SNE es más apropiado para la exploración y visualización de datos cuando el objetivo es identificar 
# clústeres o agrupaciones no lineales. Es particularmente útil en el análisis de datos de imágenes, genómica 
# y texto, donde las relaciones entre puntos son complejas y no lineales. Sin embargo, su naturaleza estocástica 
# (los resultados pueden variar entre ejecuciones) y su alta complejidad computacional lo hacen menos adecuado 
# para la reducción de dimensionalidad en pipelines de aprendizaje automático a gran escala. En resumen, use PCA 
# para la reducción de dimensionalidad con fines analíticos y t-SNE para la visualización de clústeres.
