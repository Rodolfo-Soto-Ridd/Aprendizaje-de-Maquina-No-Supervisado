# 1. Carga de datos (1 punto)
# • Carga el conjunto de datos proporcionado en el material complementario.
import pandas as pd
import numpy as np

# Cargar el conjunto de datos
df = pd.read_csv('datos_clientes.csv')

# Mostrar las primeras filas y la información del dataset para verificar la carga
print("Primeras filas del dataset:")
print("\n",df.head(),"\n")
print("\nInformación del dataset:")
print(df.info(),"\n")

# Preparar los datos: Seleccionar solo las columnas numéricas para el análisis
datos_numericos = df.select_dtypes(include=np.number)
print("\n",datos_numericos,"\n")

#2. Aplicación de técnicas de reducción de dimensionalidad (5 puntos)
# • PCA:
# o Aplica PCA para reducir la dimensionalidad del dataset a 2 componentes principales.
# o Grafica los datos transformados en el nuevo espacio bidimensional.
# • t-SNE:
# o Aplica t-SNE para visualizar los datos en un espacio 2D.
# o Genera un gráfico que represente la distribución de los datos transformados.

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Estandarizar los datos antes de aplicar PCA
scaler = StandardScaler()
datos_escalados = scaler.fit_transform(datos_numericos)

# Aplicar PCA para reducir a 2 componentes principales
pca = PCA(n_components=2)
datos_pca = pca.fit_transform(datos_escalados)

# Crear un DataFrame con los resultados de PCA para facilitar la visualización
df_pca = pd.DataFrame(data=datos_pca, columns=['Componente Principal 1', 'Componente Principal 2'])
df_pca['Preferencia_Producto'] = df['Preferencia_Producto']
print("\n",df_pca,"\n")

# Graficar los datos transformados con PCA
plt.figure(figsize=(10, 7))
productos_unicos = df_pca['Preferencia_Producto'].unique()
colores = plt.cm.get_cmap('tab10', len(productos_unicos))

for i, producto in enumerate(productos_unicos):
    subset = df_pca[df_pca['Preferencia_Producto'] == producto]
    plt.scatter(subset['Componente Principal 1'], subset['Componente Principal 2'],
                c=[colores(i)], label=producto, alpha=0.7)

plt.title('PCA: Reducción de dimensionalidad a 2 componentes')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.grid()
plt.show()

# Explicar la varianza de los componentes principales
print(f"\nVarianza explicada por cada componente principal: {pca.explained_variance_ratio_}")
print(f"Varianza explicada acumulada: {sum(pca.explained_variance_ratio_)}")

# Aplicar t-SNE para visualizar los datos en 2D. Se recomienda usar los datos escalados.
# A menudo, PCA se usa como un paso previo para t-SNE en datasets grandes para reducir el tiempo de cálculo.
# Aquí, aplicaremos t-SNE directamente sobre los datos escalados para una comparación directa.

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
datos_tsne = tsne.fit_transform(datos_escalados)

# Crear un DataFrame con los resultados de t-SNE
df_tsne = pd.DataFrame(data=datos_tsne, columns=['t-SNE Componente 1', 't-SNE Componente 2'])
df_tsne['Preferencia_Producto'] = df['Preferencia_Producto']
print("\n",df_tsne,"\n")

# Generar un gráfico que represente la distribución de los datos transformados con t-SNE
plt.figure(figsize=(10, 7))
for i, producto in enumerate(productos_unicos):
    subset = df_tsne[df_tsne['Preferencia_Producto'] == producto]
    plt.scatter(subset['t-SNE Componente 1'], subset['t-SNE Componente 2'],
                c=[colores(i)], label=producto, alpha=0.7)

plt.title('t-SNE: Visualización de datos en 2D')
plt.xlabel('t-SNE Componente 1')
plt.ylabel('t-SNE Componente 2')
plt.legend()
plt.grid()
plt.show()

# 3. Análisis de resultados (4 puntos)
# • Comparación de métodos: Explica las diferencias entre la representación obtenida con PCA y t-SNE.
# • Interpretación: Describe los patrones que identificaste en la visualización de los datos.
# • Aplicabilidad: Explica en qué casos sería recomendable utilizar PCA y en cuáles sería más útil t-SNE.

# Comparación de métodos: PCA vs. t-SNE
# PCA (Análisis de Componentes Principales): Es un algoritmo lineal que busca las direcciones 
# (componentes principales) en las que la varianza de los datos es máxima. Su objetivo es preservar la 
# estructura global de los datos. En el gráfico de PCA, los puntos están dispuestos de una manera que 
# maximiza la varianza y, en general, se pueden observar tendencias o distribuciones amplias. Sin embargo, 
# puede que no separen bien los grupos de datos si la separación es no lineal.

# t-SNE (t-Distributed Stochastic Neighbor Embedding): Es un algoritmo no lineal diseñado específicamente 
# para la visualización. Su objetivo es preservar las distancias locales entre los puntos, es decir, 
# asegura que los puntos que están cerca en el espacio de alta dimensión también lo estén en el espacio de 
# baja dimensión. En el gráfico de t-SNE, los clústeres de puntos son generalmente más compactos y mejor 
# definidos, revelando la estructura de los datos a nivel local. A menudo, los clústeres de puntos que se 
# ven en t-SNE pueden no ser verdaderos clústeres en el espacio original si la distancia entre ellos no es 
# significativa.

# En resumen, PCA se centra en la estructura global y la varianza, mientras que t-SNE se centra en la 
# estructura local y la proximidad entre los puntos.

# Interpretación de los patrones
# Basándonos en las visualizaciones generadas, se pueden observar los siguientes patrones:
# Con PCA: Los datos muestran una distribución bastante dispersa, sin una clara separación de clústeres 
# basada en la Preferencia_Producto. Esto sugiere que las características originales (Edad, Ingresos, 
# Frecuencia_Compra) no están linealmente correlacionadas de manera fuerte con la preferencia de producto de 
# los clientes. Los puntos de diferentes categorías de productos se superponen en gran medida.

# Con t-SNE: El gráfico de t-SNE muestra clústeres más definidos y separados. Se pueden identificar grupos de 
# clientes con preferencias de productos similares, como "Ropa" y "Alimentos", que tienden a formar sus propios 
# clústeres. Esto indica que hay una estructura no lineal subyacente en los datos que agrupa a los clientes por 
# su preferencia de producto, algo que PCA no pudo capturar de manera efectiva.

# Aplicabilidad: ¿Cuándo usar PCA y t-SNE?
# PCA es recomendable para:
# Reducción de dimensionalidad para el preprocesamiento de modelos de aprendizaje automático: PCA es rápido y 
# se puede usar para reducir el número de características de entrada, lo que puede ayudar a acelerar los 
# algoritmos y prevenir el sobreajuste.
# Identificación de las características más importantes: Los componentes principales pueden ser interpretados 
# para entender qué variables originales contribuyen más a la varianza de los datos.
# Preservar la estructura global: Es útil cuando se busca mantener la varianza total de los datos y se desea una 
# representación lineal de ellos.
# Análisis Exploratorio de Datos (EDA) inicial: Puede dar una idea general de la distribución de los datos, 
# especialmente si las relaciones son lineales.
# t-SNE es más útil para:
# Visualización de datos complejos de alta dimensión: Es la herramienta de elección para la visualización cuando 
# se quiere explorar y entender la estructura de los clústeres de los datos.
# Identificación de clústeres de datos: t-SNE es excelente para revelar la estructura de clústeres no lineales que 
# los algoritmos lineales como PCA no pueden detectar.
# Presentación de resultados a una audiencia no técnica: Los gráficos de t-SNE a menudo son intuitivos y permiten 
# a los usuarios ver de forma clara si existen grupos naturales en los datos.
# Análisis cualitativo: Se usa para obtener una percepción de las relaciones y similitudes entre los puntos de datos, 
# en lugar de para una reducción de características cuantitativa.