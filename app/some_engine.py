# app/some_engine.py
# importamos numpy para manejar operaciones numéricas
# importamos numba para acelerar las operaciones con JIT compilation y paralelización
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def compute_distances(weights, sample):
    dists = np.empty(weights.shape[0], dtype=np.float32)
    for i in prange(weights.shape[0]):
        dists[i] = np.linalg.norm(weights[i] - sample)
    return dists

@njit(parallel=True)
def update_weights(weights, sample, influence, lr_t):
    for i in prange(weights.shape[0]):
        weights[i] += lr_t * influence[i] * (sample - weights[i])

# Definimos la clase SOM (Self-Organizing Map)
# Esta clase implementa una red neuronal competitiva para clustering no supervisado
class SOM:
    """    
    Clase para implementar una Red Neuronal Competitiva (SOM) para clustering no supervisado
    Esta clase permite entrenar una red SOM con un conjunto de datos y calcular la U-Matrix.
    Parámetros:
    - x: Número de filas en la red SOM.
    - y: Número de columnas en la red SOM.
    - input_len: Dimensionalidad de los datos de entrada.
    - learning_rate: Tasa de aprendizaje inicial.
    - sigma: Ancho de la vecindad inicial (opcional, por defecto es la mitad del máximo de x o y).
    - max_iter: Número máximo de iteraciones para el entrenamiento.
    Métodos:
    - train_step: Realiza un paso de entrenamiento con un único ejemplo.
    - get_u_matrix: Calcula la U-Matrix, que muestra la distancia entre los nodos vecinos.
    - _get_neighbors: Obtiene los índices de los nodos vecinos de un nodo dado
    - _euclidean: Calcula la distancia euclidiana entre dos vectores.
    - _get_bmu_index: Encuentra el índice del Best Matching Unit (BMU) para un ejemplo dado.
    - _decay: Aplica una función de decaimiento exponencial a un valor inicial.
    - _neighborhood: Calcula la influencia de la vecindad del BMU en el entrenamiento.
    - _get_neighbors: Obtiene los índices de los nodos vecinos de un nodo dado.
    Atributos:
    - weights: Pesos de la red SOM, inicializados aleatoriamente.
    - locations: Coordenadas de cada nodo en la red SOM.
    - errors: Lista para almacenar el error cuadrático medio (RMSE) en cada iteración.
    """
    # Definimos el constructor de la clase SOM
    # Este método se ejecuta al crear una instancia de la clase SOM
    def __init__(self, x, y, input_len, learning_rate=0.5, sigma=None, max_iter=1000):
        self.x = x
        self.y = y
        self.input_len = input_len
        self.lr = learning_rate
        self.sigma = sigma if sigma else max(x, y) / 2.0
        self.max_iter = max_iter

        self.weights = np.random.rand(x * y, input_len).astype(np.float32)
        self.locations = np.array([[i, j] for i in range(x) for j in range(y)], dtype=np.float32)
        self.errors = []

    # Metodos privados de la clase SOM
    # Estos métodos no son accesibles desde fuera de la clase y se utilizan internamente

    def _get_bmu_index(self, sample):
        dists = compute_distances(self.weights, sample)
        return np.argmin(dists)

    def _decay(self, initial, i):
        return initial * np.exp(-i / self.max_iter)

    def _neighborhood(self, bmu_idx, sigma_t):
        bmu_location = self.locations[bmu_idx]
        dists = np.linalg.norm(self.locations - bmu_location, axis=1)
        return np.exp(-dists ** 2 / (2 * sigma_t ** 2)).astype(np.float32)

    def train_step(self, sample, i):
        bmu_idx = self._get_bmu_index(sample)
        sigma_t = self._decay(self.sigma, i)
        lr_t = self._decay(self.lr, i)
        influence = self._neighborhood(bmu_idx, sigma_t)
        update_weights(self.weights, sample, influence, lr_t)

        # Error cuantización
        error = np.mean(np.min(np.linalg.norm(self.weights - sample, axis=1)))
        self.errors.append(error)

    def get_u_matrix(self):
        u_matrix = np.zeros((self.x, self.y))
        for i in range(self.x):
            for j in range(self.y):
                idx = i * self.y + j
                neighbors = self._get_neighbors(i, j)
                d = [np.linalg.norm(self.weights[idx] - self.weights[n]) for n in neighbors]
                u_matrix[i, j] = np.mean(d) if d else 0
        return u_matrix

    def _get_neighbors(self, i, j):
        neighbors = []
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i + dx, j + dy
            if 0 <= ni < self.x and 0 <= nj < self.y:
                neighbors.append(ni * self.y + nj)
        return neighbors