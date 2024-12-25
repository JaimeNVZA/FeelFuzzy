import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import csv
import numpy as np
import skfuzzy as fuzz
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Inicializar stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Definir variables para lógica difusa
x_positivo = np.arange(0, 1.1, 0.1)
x_negativo = np.arange(0, 1.1, 0.1)
x_salida = np.arange(0, 10.1, 0.1)

# Funciones de pertenencia ajustadas
positivo_bajo = fuzz.trimf(x_positivo, [0, 0, 0.4])
positivo_medio = fuzz.trimf(x_positivo, [0.2, 0.5, 0.8])
positivo_alto = fuzz.trimf(x_positivo, [0.6, 1, 1])

negativo_bajo = fuzz.trimf(x_negativo, [0, 0, 0.4])
negativo_medio = fuzz.trimf(x_negativo, [0.2, 0.5, 0.8])
negativo_alto = fuzz.trimf(x_negativo, [0.6, 1, 1])

salida_negativa = fuzz.trimf(x_salida, [0, 0, 4])
salida_neutral = fuzz.trimf(x_salida, [3, 5, 7])
salida_positiva = fuzz.trimf(x_salida, [6, 10, 10])

# Función para limpiar texto
def clean_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and len(word) > 1 and word.lower() not in stop_words]
    return ' '.join(tokens)

# Función de análisis de sentimientos con lógica difusa
def fuzzy_sentiment_analysis(filepath):
    resultados = []
    try:
        with open(filepath, 'r') as archivo:
            lector_csv = csv.reader(archivo)
            next(lector_csv)  # Saltar encabezado
            for linea in lector_csv:
                frase = linea[0]
                texto_limpio = clean_text(frase)

                analyzer = SentimentIntensityAnalyzer()
                scores = analyzer.polarity_scores(texto_limpio)
                positivo = scores['pos']
                negativo = scores['neg']

                # Niveles de pertenencia
                nivel_positivo_bajo = fuzz.interp_membership(x_positivo, positivo_bajo, positivo)
                nivel_positivo_medio = fuzz.interp_membership(x_positivo, positivo_medio, positivo)
                nivel_positivo_alto = fuzz.interp_membership(x_positivo, positivo_alto, positivo)

                nivel_negativo_bajo = fuzz.interp_membership(x_negativo, negativo_bajo, negativo)
                nivel_negativo_medio = fuzz.interp_membership(x_negativo, negativo_medio, negativo)
                nivel_negativo_alto = fuzz.interp_membership(x_negativo, negativo_alto, negativo)

                # Reglas difusas
                regla_1 = np.fmin(nivel_positivo_alto, nivel_negativo_bajo)  # Muy positivo
                regla_2 = np.fmin(nivel_positivo_medio, nivel_negativo_medio)  # Neutral
                regla_3 = np.fmin(nivel_positivo_bajo, nivel_negativo_alto)  # Muy negativo

                # Activación de las reglas
                activacion_negativa = np.fmin(regla_3, salida_negativa)
                activacion_neutral = np.fmin(regla_2, salida_neutral)
                activacion_positiva = np.fmin(regla_1, salida_positiva)

                # Agregar las salidas
                salida_agregada = np.fmax(activacion_negativa, np.fmax(activacion_neutral, activacion_positiva))

                # Defuzzificación
                if np.any(salida_agregada):
                    salida_defuzz = fuzz.defuzz(x_salida, salida_agregada, "centroid")
                else:
                    salida_defuzz = 5  # Valor neutral predeterminado

                # Clasificación
                if salida_defuzz < 3.33:
                    sentimiento = "Negativo"
                elif salida_defuzz < 6.66:
                    sentimiento = "Neutro"
                else:
                    sentimiento = "Positivo"

                resultados.append((frase, positivo, negativo, round(salida_defuzz, 2), sentimiento))
    except Exception as e:
        messagebox.showerror("Error", f"Error procesando el archivo: {e}")
    return resultados

# Función para mostrar la interfaz gráfica
def mostrar_gui():
    def cargar_archivo():
        filepath = filedialog.askopenfilename(filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")])
        if filepath:
            resultados = fuzzy_sentiment_analysis(filepath)
            for row in treeview.get_children():
                treeview.delete(row)
            for resultado in resultados:
                treeview.insert("", "end", values=resultado)

    root = tk.Tk()
    root.title("Análisis de Sentimientos con Lógica Difusa")
    root.geometry("800x600")

    label = tk.Label(root, text="Seleccione un archivo CSV para procesar:", font=("Arial", 14))
    label.pack(pady=10)

    button_cargar = tk.Button(root, text="Cargar Archivo", command=cargar_archivo, font=("Arial", 12))
    button_cargar.pack(pady=10)

    treeview = ttk.Treeview(root, columns=("Frase", "Positivo", "Negativo", "Salida", "Sentimiento"), show="headings", height=20)
    treeview.heading("Frase", text="Frase")
    treeview.heading("Positivo", text="Positivo")
    treeview.heading("Negativo", text="Negativo")
    treeview.heading("Salida", text="Salida")
    treeview.heading("Sentimiento", text="Sentimiento")
    treeview.column("Frase", width=300)
    treeview.column("Positivo", width=80, anchor="center")
    treeview.column("Negativo", width=80, anchor="center")
    treeview.column("Salida", width=80, anchor="center")
    treeview.column("Sentimiento", width=120, anchor="center")
    treeview.pack(pady=10, fill=tk.BOTH, expand=True)

    button_salir = tk.Button(root, text="Salir", command=root.quit, font=("Arial", 12))
    button_salir.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    mostrar_gui()
