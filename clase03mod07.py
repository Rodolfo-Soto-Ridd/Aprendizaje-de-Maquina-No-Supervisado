import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
 
# Crear un conjunto de datos simulado con altura, peso e IMC

np.random.seed(42)

n_muestras = 100
 
altura = np.random.normal(170, 10, n_muestras)   # cm

peso = np.random.normal(70, 15, n_muestras)      # kg

imc = peso / (altura / 100) ** 2                 # kg/m²
 
# Construcción del DataFrame

df = pd.DataFrame({'Altura': altura, 'Peso': peso, 'IMC': imc})
 
# Normalizar los datos antes de aplicar PCA

scaler = StandardScaler()

X_scaled = scaler.fit_transform(df)
 
# Aplicar PCA para reducir de 3 a 2 dimensiones

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)
 
# Convertir los componentes en un DataFrame

df_pca = pd.DataFrame(X_pca, columns=['Componente 1', 'Componente 2'])
 
# Visualizar los resultados

plt.figure(figsize=(8, 6))

plt.scatter(df_pca['Componente 1'], df_pca['Componente 2'], alpha=0.7, color='blue')

plt.xlabel('Componente Principal 1')

plt.ylabel('Componente Principal 2')

plt.title('PCA aplicado a datos de Altura, Peso e IMC')

plt.grid()

plt.show()
 
# Explicación de la varianza explicada por cada componente

print("Varianza explicada por cada componente:", pca.explained_variance_ratio_)

# Modelo t-SNE
## t-SNE
 
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler
 
# Crear un conjunto de datos simulado con características musicales

np.random.seed(42)

n_samples = 200
 
generos = np.random.choice(['Rock', 'Jazz', 'Clásica'], n_samples)

tempo = np.random.normal(120, 20, n_samples)        # BPM

intensidad = np.random.normal(0.7, 0.1, n_samples)  # Volumen normalizado

tonalidad = np.random.randint(0, 12, n_samples)     # Do=0, Re=1, ..., Si=11
 
# Construcción del DataFrame

df = pd.DataFrame({

    'Género': generos,

    'Tempo': tempo,

    'Intensidad': intensidad,

    'Tonalidad': tonalidad

})
 
# Convertir variables categóricas a numéricas

df['Género'] = df['Género'].astype('category').cat.codes
 
# Normalizar los datos

scaler = StandardScaler()

X_scaled = scaler.fit_transform(df[['Género', 'Tempo', 'Intensidad', 'Tonalidad']])
 
# Aplicar t-SNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42)

X_tsne = tsne.fit_transform(X_scaled)
 
# Convertir los resultados en un DataFrame

df_tsne = pd.DataFrame(X_tsne, columns=['Componente t-SNE 1', 'Componente t-SNE 2'])

df_tsne['Género'] = generos
 
# Graficar resultados

plt.figure(figsize=(8, 6))

for genero in ['Rock', 'Jazz', 'Clásica']:

    subset = df_tsne[df_tsne['Género'] == genero]

    plt.scatter(subset['Componente t-SNE 1'], subset['Componente t-SNE 2'],

                label=genero, alpha=0.7)
 
plt.xlabel('Componente t-SNE 1')

plt.ylabel('Componente t-SNE 2')

plt.title('t-SNE aplicado a características de música')

plt.legend()

plt.grid()

plt.show()
 
# Modelo Autoencoder

