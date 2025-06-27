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
from sklearn.preprocessing import MinMaxScaler

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

        self.train_btn = tk.Button(root, text="Entrenar SOM (Animación)", command=self.train_som)
        self.train_btn.pack(pady=5)

        self.error_btn = tk.Button(root, text="Ver Error RMSE", command=self.plot_errors)
        self.error_btn.pack(pady=5)

        self.save_umatrix_btn = tk.Button(root, text="Guardar U-Matrix (.png/.csv)", command=self.save_umatrix)
        self.save_umatrix_btn.pack(pady=5)

        self.save_error_btn = tk.Button(root, text="Guardar Error RMSE (.png)", command=self.save_error)
        self.save_error_btn.pack(pady=5)

        # Botón para salir del programa
        self.exit_btn = tk.Button(root, text="Salir del programa", command=self.root.quit, bg="red", fg="white")
        self.exit_btn.pack(pady=10)

        # Gráfico SOM
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()
        
        # Conectar evento de clic en el gráfico
        self.canvas.mpl_connect("button_press_event", self.on_click)


    # Método para cargar datos desde un archivo CSV
    # Este método abre un diálogo para seleccionar un archivo CSV y carga los datos en un DataFrame de pandas
    # Si ocurre un error al cargar el archivo, muestra un mensaje de error
    def load_data(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filepath:
            try:
                df = pd.read_csv(filepath)
                df = df.select_dtypes(include=['float64', 'int64'])  # solo numéricos
                df = df.dropna(axis=0)  # elimina filas con NaN

                # Normalizamos si es necesario
                if df.shape[0] > 0:
                    scaler = MinMaxScaler()
                    df[df.columns] = scaler.fit_transform(df)

                self.data = df
                messagebox.showinfo("Éxito", "Datos cargados correctamente.")
            except Exception as e:
                messagebox.showerror("Error", str(e))


    # Método para iniciar el entrenamiento de la red SOM
    # Este método normaliza los datos, crea una instancia de la clase SOM y comienza la animación del entrenamiento
    # Si no hay datos cargados, muestra una advertencia
    def train_som(self):
        if self.data is None:
            messagebox.showwarning("Advertencia", "Primero carga un dataset.")
            return

        try:
            X = self.data.select_dtypes(include=['float64', 'int64']).values
            som = SOM(x=10, y=10, input_len=X.shape[1], max_iter=100)  # Puedes subir a 10000 neuronas después

            for i in range(som.max_iter):
                sample = X[np.random.randint(0, X.shape[0])]
                som.train_step(sample, i)

                if i % 10 == 0 or i == som.max_iter - 1:
                    self.ax1.clear()
                    self.ax1.set_title(f"Iteración {i}")
                    self.ax1.imshow(som.get_u_matrix().T, cmap='bone_r', origin='lower')
                    self.canvas.draw()
                    self.root.update_idletasks()

            messagebox.showinfo("Entrenamiento completo", "La red SOM fue entrenada correctamente.")

            # Guardamos errores para graficarlos después
            self.last_som = som

        except Exception as e:
            messagebox.showerror("Error", str(e))

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

    def plot_errors(self):
        if not hasattr(self, "last_som"):
            messagebox.showwarning("Advertencia", "Entrena primero la red.")
            return

        self.ax1.clear()
        self.ax1.set_title("Evolución del error RMSE")
        self.ax1.plot(self.last_som.errors, label="RMSE")
        self.ax1.set_xlabel("Iteración")
        self.ax1.set_ylabel("Error")
        self.ax1.legend()
        self.canvas.draw()

    def save_umatrix(self):
        if not hasattr(self, "last_som"):
            messagebox.showwarning("Advertencia", "Primero entrena la red.")
            return

        # Guardar imagen PNG
        fig, ax = plt.subplots()
        ax.set_title("U-Matrix")
        im = ax.imshow(self.last_som.get_u_matrix().T, cmap='bone_r', origin='lower')
        plt.colorbar(im, ax=ax)
        fig.savefig("umatrix.png")
        plt.close(fig)

        # Guardar datos CSV
        u_matrix = self.last_som.get_u_matrix()
        np.savetxt("umatrix.csv", u_matrix, delimiter=",")

        messagebox.showinfo("Éxito", "U-Matrix guardada como umatrix.png y umatrix.csv")

    def save_error(self):
        if not hasattr(self, "last_som"):
            messagebox.showwarning("Advertencia", "Primero entrena la red.")
            return

        fig, ax = plt.subplots()
        ax.set_title("Error RMSE")
        ax.plot(self.last_som.errors, label="RMSE")
        ax.set_xlabel("Iteración")
        ax.set_ylabel("Error")
        ax.legend()
        fig.savefig("rmse_error.png")
        plt.close(fig)

        messagebox.showinfo("Éxito", "Gráfico de error guardado como rmse_error.png")
    
        # Método para manejar clics en la U-Matrix
    # Este método se activa cuando el usuario hace clic sobre el mapa SOM
    # Identifica la neurona (i, j) más cercana al clic y muestra sus pesos o información relevante
    def on_click(self, event):
        if not hasattr(self, "last_som"):
            return

        if event.inaxes != self.ax1:
            return  # Ignoramos clics fuera de la U-Matrix

        x_click, y_click = int(event.xdata), int(event.ydata)
        if not (0 <= x_click < self.last_som.x and 0 <= y_click < self.last_som.y):
            return

        index = y_click * self.last_som.x + x_click  # Transponemos por .T
        weights = self.last_som.weights[index]

        # Mostrar en la segunda gráfica
        self.ax2.clear()
        self.ax2.set_title(f"Pesos de la neurona ({x_click}, {y_click})")
        self.ax2.plot(weights, marker='o', linestyle='-')
        self.ax2.set_xlabel("Dimensión")
        self.ax2.set_ylabel("Valor del peso")
        self.canvas.draw()

# Método principal para ejecutar la aplicación
# Este método crea una instancia de la clase RNACApp y ejecuta el bucle principal
if __name__ == "__main__":
    root = tk.Tk()
    app = RNACApp(root)
    root.mainloop()