from sklearn.cluster import KMeans
import numpy as np
# Datos ficticios: compras de clientes (ingreso, gasto promedio mensual)
datos = np.array([[30, 300], [40, 400], [25, 250], [35, 350], [50, 500]])
# Aplicar K-Means con 2 clusters
modelo = KMeans(n_clusters=2, random_state=42)
modelo.fit(datos)
# Ver los clusters asignados
print("Etiquetas de cluster asignadas:", modelo.labels_)

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
# Cargar dataset iris
iris = load_iris()
X = iris.data  # Usamos todas las características: [sepal length, sepal width, petal length, petal width]
# Aplicar K-Means con 3 clusters (porque hay 3 especies)
modelo = KMeans(n_clusters=3, random_state=42)
modelo.fit(X)
# Ver etiquetas asignadas
etiquetas = modelo.labels_

etiquetas
# Crear un DataFrame con los datos y etiquetas
df = pd.DataFrame(X, columns=iris.feature_names)
df['Cluster'] = etiquetas

# Visualizar usando las dos características más informativas (petal length y petal width)
plt.figure(figsize=(8, 6))
plt.scatter(df.iloc[:, 2], df.iloc[:, 3], c=df['Cluster'], cmap='viridis', s=50)
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.title('Clustering de Iris con K-Means')
plt.grid(True)
plt.legend()
plt.show()
df['Specie'] = iris.target

df