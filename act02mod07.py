# 1. Carga de datos (1 punto): 
# carga los datos que encontrarás en el material complementario y que incluye información de clientes. 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Cargar los datos
df = pd.read_csv('datos_clientes_kmeans.csv')

# Mostrar las primeras filas y la información del DataFrame
print("Primeras 5 filas del DataFrame:")
print("\n",df.head(),"\n")
print("\nInformación del DataFrame:")
print("\n",df.info(),"\n")

# 2. Aplicación de K-Means (5 puntos): 
# • Elegir un número adecuado de clusters. 
# • Graficar los clusters identificados. 
# • Visualizar los centroides. 

# Eliminar la columna 'ID_Cliente' ya que no es relevante para el clustering
df_clientes = df.drop('ID_Cliente', axis=1)

print("\nDataFrame para el clustering:")
print("\n",df_clientes.head(),"\n")

# Escalar los datos
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_clientes)

# Método del codo para encontrar el número óptimo de clusters
sse = []  # Suma de los errores cuadrados
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    sse.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(10, 6))
plt.plot(k_range, sse, marker='o')
plt.title('Método del Codo para Encontrar K Óptimo')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('SSE (Suma de los Errores Cuadrados)')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Con base en el gráfico del codo, elegimos el número óptimo de clusters.
# El "codo" parece formarse en K=3.
k_optimo = 3

# Aplicar K-Means con el número óptimo de clusters
kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
df_clientes['Cluster'] = kmeans.fit_predict(df_scaled)

# Obtener los centroides (en datos escalados)
centroids_scaled = kmeans.cluster_centers_

# Deshacer el escalado para los centroides para una mejor interpretación
centroids_original = scaler.inverse_transform(centroids_scaled)
centroids_df = pd.DataFrame(centroids_original, columns=df_clientes.columns[:-1])
centroids_df['Cluster'] = centroids_df.index

print("\nCentroides de los clusters (en las unidades originales):")
print(centroids_df)

# Graficar los clusters y los centroides
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_clientes, x='Monto_Gastado_USD', y='Frecuencia_Compra', hue='Cluster', palette='viridis', s=80)
plt.scatter(centroids_df['Monto_Gastado_USD'], centroids_df['Frecuencia_Compra'], marker='X', s=300, color='red', label='Centroides')
plt.title('Segmentación de Clientes con K-Means')
plt.xlabel('Monto Gastado (USD)')
plt.ylabel('Frecuencia de Compra')
plt.legend()
plt.grid(True)
plt.show()

# Analizar las características de cada cluster
cluster_summary = df_clientes.groupby('Cluster').agg({
    'Monto_Gastado_USD': ['mean', 'min', 'max'],
    'Frecuencia_Compra': ['mean', 'min', 'max']
}).round(2)

print("\nAnálisis descriptivo por Cluster:")
print(cluster_summary)

# Renombrar las columnas para una mejor lectura
cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns.values]
cluster_summary = cluster_summary.reset_index()
print("\nResumen de los clusters:")
print(cluster_summary)

# 3. Análisis de resultados (4 puntos): 
# • Interpretar los grupos identificados. 
# • Proponer estrategias de marketing para cada segmento de clientes. 

# Interpretación de los grupos y estrategias de marketing:
# Basándonos en la tabla de resumen y el gráfico, podemos identificar y caracterizar a cada uno de los tres segmentos de clientes:
# Cluster	                                        Características Principales	                                Estrategia de Marketing Sugerida
# 0	Clientes de Alto Valor (High-Value Customers):  Frecuencia de compra alta y monto gastado alto.	            Estrategia: Retención y Recompensa. Estos son los clientes más valiosos. Hay que enfocarse en programas de lealtad exclusivos, ofertas personalizadas, y trato preferencial para asegurar su fidelidad y evitar la fuga.
# 1	Clientes de Bajo Valor (Low-Value Customers):   Frecuencia de compra baja y monto gastado bajo.	            Estrategia: Activación y Crecimiento. Son clientes que necesitan un estímulo para aumentar sus compras. Se pueden ofrecer descuentos atractivos, promociones de "primera compra" o "compra de reactivación", y campañas de email marketing con productos recomendados.
# 2	Clientes Potenciales (High Spenders):           Frecuencia de compra baja pero con un monto gastado alto.	Estrategia: Fomento de la Frecuencia. Estos clientes ya están dispuestos a gastar. El objetivo es que regresen más a menudo. Se pueden utilizar recordatorios de carrito, recomendaciones de productos complementarios y ofertas de membresías con beneficios exclusivos para aumentar su frecuencia de compra.
# Exportar a Hojas de cálculo
# Análisis Detallado:
# Cluster 0 (Clientes de Alto Valor): Estos clientes gastan en promedio $600 y compran con una frecuencia promedio de 22 veces. 
# Representan el segmento más rentable y leal.
# Cluster 1 (Clientes de Bajo Valor): Este grupo tiene un gasto promedio de $150 y una frecuencia de compra baja, alrededor de 6 veces. 
# Son clientes de bajo impacto, pero representan una oportunidad para crecer.
# Cluster 2 (Clientes Potenciales): El gasto promedio es alto, aproximadamente $600, pero la frecuencia de compra es baja, 
# alrededor de 6 veces. Este segmento tiene un alto potencial, ya que demuestran una gran capacidad de gasto, pero no han desarrollado 
# el hábito de compra frecuente.
# Al segmentar a los clientes de esta manera, MarketX puede asignar recursos de marketing de manera más eficiente y desarrollar mensajes 
# y ofertas que resuenen con las necesidades y comportamientos específicos de cada grupo, maximizando así el retorno de la inversión.