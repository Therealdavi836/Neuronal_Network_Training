# app/main.py
import tkinter as tk # Importamos tkinter para la interfaz gráfica
from tkinter import filedialog, messagebox # Importamos filedialog y messagebox para manejar archivos y mostrar mensajes
import pandas as pd # Importamos pandas para manejar datos en formato CSV
import numpy as np # Importamos numpy para manejar operaciones numéricas"
import matplotlib.pyplot as plt # Importamos matplotlib para graficar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # Importamos FigureCanvasTkAgg para integrar matplotlib con tkinter
from matplotlib.animation import FuncAnimation # Importamos FuncAnimation para crear animaciones en matplotlib
# Importamos la clase SOM desde el archivo some_engine.py
from some_engine import SOM

# Definimos la clase RNACApp que maneja la interfaz gráfica y la lógica de la aplicación
class RNACApp:
    # Constructor de la clase RNACApp
    # Este método se ejecuta al crear una instancia de la clase RNACApp
    def __init__(self, root):
        self.root = root
        self.root.title("Clustering con SOM - Visualización Dinámica - Red Neuronal Auto-Organizada Competitiva Proyecto SIC")
        self.data = None
        self.som = None
        self.iteration = 0

        # Botones
        self.load_btn = tk.Button(root, text="Cargar CSV", command=self.load_data)
        self.load_btn.pack(pady=5)

        self.train_btn = tk.Button(root, text="Entrenar SOM (Animación)", command=self.start_training)
        self.train_btn.pack(pady=5)

        # Gráfico SOM
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

    # Método para cargar datos desde un archivo CSV
    # Este método abre un diálogo para seleccionar un archivo CSV y carga los datos en un DataFrame de pandas
    # Si ocurre un error al cargar el archivo, muestra un mensaje de error
    def load_data(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filepath:
            try:
                self.data = pd.read_csv(filepath)
                messagebox.showinfo("Éxito", "Datos cargados correctamente.")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    # Método para iniciar el entrenamiento de la red SOM
    # Este método normaliza los datos, crea una instancia de la clase SOM y comienza la animación del entrenamiento
    # Si no hay datos cargados, muestra una advertencia
    def start_training(self):
        if self.data is None:
            messagebox.showwarning("Advertencia", "Carga un dataset primero.")
            return

        X = self.data.select_dtypes(include=['float64', 'int64']).values
        self.X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))  # normalización
        self.som = SOM(x=100, y=100, input_len=self.X.shape[1], learning_rate=0.5, max_iter=300)
        self.iteration = 0

        self.ani = FuncAnimation(self.fig, self.update_plot, frames=self.som.max_iter,
                                 interval=50, repeat=False)
        self.canvas.draw()

    # Método para actualizar el gráfico en cada iteración del entrenamiento
    # Este método toma una muestra aleatoria de los datos, entrena la SOM con esa
    # muestra, actualiza la U-Matrix y el gráfico de error cuadrático medio (RMSE)
    # Luego redibuja el canvas para mostrar los cambios en la interfaz gráfica
    def update_plot(self, frame):
        sample = self.X[np.random.randint(0, len(self.X))]
        self.som.train_step(sample, frame)

        # Actualiza U-Matrix
        self.ax1.clear()
        self.ax1.set_title(f"U-Matrix - Iteración {frame}")
        self.ax1.imshow(self.som.get_u_matrix().T, cmap='bone_r')

        # Error vs Iteración
        self.ax2.clear()
        self.ax2.set_title("Error de cuantización (RMSE)")
        self.ax2.plot(self.som.errors, color='red')
        self.ax2.set_xlabel("Iteración")
        self.ax2.set_ylabel("Error")

        self.canvas.draw()

# Método principal para ejecutar la aplicación
# Este método crea una instancia de la clase RNACApp y ejecuta el bucle principal
if __name__ == "__main__":
    root = tk.Tk()
    app = RNACApp(root)
    root.mainloop()