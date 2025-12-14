import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
 
# 1. Cargar el dataset Digits
digits = load_digits()
X = digits.data         # Matriz de características (imágenes en forma de vectores)
y = digits.target       # Etiquetas (números reales de las imágenes)
 
# 2. Visualizar algunas imágenes originales
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f'Dígito: {digits.target[i]}')
    ax.axis('off')
plt.show()
 
# 3. Normalizar los datos antes de PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# 4. Aplicar PCA para reducir la dimensionalidad a 2 componentes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
 
# 5. Visualizar los datos en el nuevo espacio de 2D
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Dígito real')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Visualización de PCA en el conjunto de datos Digits')
plt.show()
 
# 6. Evaluar la varianza explicada por cada componente
explained_variance = pca.explained_variance_ratio_
print(f'Varianza explicada por la primera componente: {explained_variance[0]:.2%}')
print(f'Varianza explicada por la segunda componente: {explained_variance[1]:.2%}')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
 
# 1. Cargar el dataset Digits
digits = load_digits()
X = digits.data      # Matriz de características
y = digits.target    # Etiquetas (dígitos reales)
 
# 2. Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# 3. Aplicar t-SNE para reducir la dimensionalidad a 2D
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
 
# 4. Visualizar los datos en el nuevo espacio 2D
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='jet', alpha=0.7)
plt.colorbar(scatter, label='Dígito real')
plt.xlabel('Dimensión 1')
plt.ylabel('Dimensión 2')
plt.title('Visualización con t-SNE del conjunto de datos Digits')
plt.show()