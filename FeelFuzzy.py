from unicodedata import numeric
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
import numpy as np
import re
import time
import skfuzzy as fuzz

# Descargar las stopwords de NLTK (solo la primera vez)
import os
os.system('python nltk_requirements.py')

# Inicializar la lista de stopwords
stop_words = set(stopwords.words('english'))

# Variables para realizar el benchmark
import csv  # Asegúrate de importar csv en la parte superior
total_time = 0  # Para registrar el tiempo total de procesamiento
cant_positivo = 0
cant_negativo = 0
cant_neutro = 0
resultados = []  # Lista para almacenar los resultados de cada línea



def prepate_text(text,option):

    #para preparar el texto para su limpiaza...
    linea_nueva = ""
    for palabra in linea:
        if(option):
            if(palabra == "," or palabra == "\n"):
                linea_nueva += " "
            else:
                linea_nueva += palabra
        else:
            if(palabra == "0" or palabra == "1"):
                return palabra 
    return linea_nueva


# Función para limpiar texto (remover letras sueltas, símbolos y stopwords)
def clean_text(text):
    # Tokenizar el texto
    tokens = word_tokenize(text)

    # Filtrar tokens alfabéticos, eliminar letras sueltas y stopwords
    tokens = [word for word in tokens if word.isalpha() and len(word) > 1 and word.lower() not in stop_words]
    
    # Unir los tokens filtrados en una oración
    return ' '.join(tokens)

# --------------------------------------------------------------------
# Generar variables del universo

# Rangos para positivo y negativo: [0, 1]
# Rango para salida: [0, 10] en puntos porcentuales
x_positivo = np.arange(0, 1, 0.1)
x_negativo = np.arange(0, 1, 0.1)
x_salida = np.arange(0, 10, 1)

# --------------------------------------------------------------------
# Generar las funciones de pertenencia difusa.

# Funciones de pertenencia difusa para positivo
positivo_bajo = fuzz.trimf(x_positivo, [0, 0, 0.5])
positivo_medio = fuzz.trimf(x_positivo, [0, 0.5, 1])
positivo_alto = fuzz.trimf(x_positivo, [0.5, 1, 1])

# Funciones de pertenencia difusa para negativo
negativo_bajo = fuzz.trimf(x_negativo, [0, 0, 0.5])
negativo_medio = fuzz.trimf(x_negativo, [0, 0.5, 1])
negativo_alto = fuzz.trimf(x_negativo, [0.5, 1, 1])

# Funciones de pertenencia difusa para salida (fijo para todos los lexicons)
salida_negativa = fuzz.trimf(x_salida, [0, 0, 5])  # Escala: Neg Neu Pos
salida_neutral = fuzz.trimf(x_salida, [0, 5, 10])
salida_positiva = fuzz.trimf(x_salida, [5, 10, 10])



with open ("archivo_nuevo.csv","w") as archivo:
    archivo.write("texto_original"+ "\n")

#Limpiaza del texto(Pre-procesado....)
with open("test_data.csv") as archivo:

    archivo_limpio = open("archivo_nuevo.csv","a")
    cant_positivo = 0
    cant_negativo = 0
    cant_neutro = 0
    cantidad = 0
    #Leemos linea por linea el archivo
    for linea in archivo:

        #inicio del proceso
        tiempo_inicio = time.time()

        if(linea == 'sentence,sentiment\n'):
            continue

        #limpiamos el texto...    
        texto_limpio = clean_text(prepate_text(linea,1))
        print("\n" + texto_limpio)

        # Inicializar el analizador
        analyzer = SentimentIntensityAnalyzer()

        # Analizamos los sentimientos de la oracion...
        sentiment_scores = analyzer.polarity_scores(linea)
        positivo = sentiment_scores['pos']
        negativo = sentiment_scores['neg']

        # Calculo de niveles de pertenencia.
        print("valor del sentimiento : " + str(positivo) + "  " + str(negativo))

        # Calcular los niveles de pertenencia positiva (bajo, medio, alto) del tweet
        nivel_positivo_bajo = fuzz.interp_membership(
            x_positivo, positivo_bajo, positivo
        )
        nivel_positivo_medio = fuzz.interp_membership(
            x_positivo, positivo_medio, positivo
        )
        nivel_positivo_alto = fuzz.interp_membership(
            x_positivo, positivo_alto, positivo
        )

        # Calcular los niveles de pertenencia negativa (bajo, medio, alto) del tweet
        nivel_negativo_bajo = fuzz.interp_membership(
            x_negativo, negativo_bajo, negativo
        )
        nivel_negativo_medio = fuzz.interp_membership(
            x_negativo, negativo_medio, negativo
        )
        nivel_negativo_alto = fuzz.interp_membership(
            x_negativo, negativo_alto, negativo
        )

        # ---------------------------------------------------------------------
        # Aplicacion de las reglas de Mamdani utilizando los niveles de pert.

        # El operador OR significa que tomamos el máximo de estas dos.
        regla_activa_1 = np.fmin(nivel_positivo_bajo, nivel_negativo_bajo)
        regla_activa_2 = np.fmin(nivel_positivo_medio, nivel_negativo_bajo)
        regla_activa_3 = np.fmin(nivel_positivo_alto, nivel_negativo_bajo)
        regla_activa_4 = np.fmin(nivel_positivo_bajo, nivel_negativo_medio)
        regla_activa_5 = np.fmin(nivel_positivo_medio, nivel_negativo_medio)
        regla_activa_6 = np.fmin(nivel_positivo_alto, nivel_negativo_medio)
        regla_activa_7 = np.fmin(nivel_positivo_bajo, nivel_negativo_alto)
        regla_activa_8 = np.fmin(nivel_positivo_medio, nivel_negativo_alto)
        regla_activa_9 = np.fmin(nivel_positivo_alto, nivel_negativo_alto)

        # Aplicacion de las reglas de Mamdani
        n1 = np.fmax(regla_activa_4, regla_activa_7)
        n2 = np.fmax(n1, regla_activa_8)
        activacion_salida_bajo = np.fmin(n2, salida_negativa)

        neu1 = np.fmax(regla_activa_1, regla_activa_5)
        neu2 = np.fmax(neu1, regla_activa_9)
        activacion_salida_medio = np.fmin(neu2, salida_neutral)

        p1 = np.fmax(regla_activa_2, regla_activa_3)
        p2 = np.fmax(p1, regla_activa_6)
        activacion_salida_alto = np.fmin(p2, salida_positiva)

        salida_cero = np.zeros_like(x_salida)

        # Agregacion para calcular el sentimiento final.
        agregada = np.fmax(
            activacion_salida_bajo, np.fmax(activacion_salida_medio, activacion_salida_alto)
        )


        t_fuzz = time.time() - tiempo_inicio

        # Desfuzzificacion
        t_defuzz = time.time()

        salida = fuzz.defuzz(x_salida, agregada, "centroid")
        res_defuzz = round(salida, 2)

        t_defuzz = time.time() - t_defuzz

        sent_calculado = ""
        print("deffuzificado:" + str(res_defuzz))

        # Escala : Neg Neu Pos. Escala [0; 10]
        if res_defuzz > 0 and res_defuzz < 3.33:  # R
            sent_calculado = "Negativa"
            cant_negativo += 1

        elif res_defuzz > 3.34 and res_defuzz < 6.66:
            sent_calculado = "Neutra"
            cant_neutro += 1

        elif res_defuzz > 6.67 and res_defuzz < 10:
            sent_calculado = "Positiva"
            cant_positivo += 1

        print("sent_calculado: " + str(sent_calculado))

        # Medición del tiempo total que llevó calcular el texto
        exec_time = time.time() - tiempo_inicio
        total_time += exec_time
        resultados.append([
            prepate_text(linea, 1), positivo, negativo, res_defuzz, sent_calculado, exec_time
        ])

        # ----------------------------------------------

        # impresion datos del tweet
        t_total = t_fuzz + t_defuzz
        if(sent_calculado != "Neutra"):
            archivo_limpio.write(prepate_text(linea,1) + " "  + " " + str(positivo) + " " + str(negativo) + " " + sent_calculado + "\n")
            cantidad += 1
        
        
        print(f"Tiempo de ejecución: {exec_time:.4f} segundos")

    archivo_limpio.close()



    # Benchmark creado en archivo csv
    with open("resultado_benchmark.csv", "w", newline='') as archivo_salida:
        writer = csv.writer(archivo_salida)
        writer.writerow([
            "Oración original", "Label original", "Puntaje Positivo", "Puntaje Negativo", 
            "Resultado de Inferencia", "Resultado Calculado", "Tiempo de Ejecución"
        ])
        writer.writerows(resultados)

    # Benchmark en terminal
    promedio_tiempo = total_time / len(resultados)
    print("\nBenchmark")
    print(f"Total de positivos: {cant_positivo}")
    print(f"Total de negativos: {cant_negativo}")
    print(f"Total de neutrales: {cant_neutro}")
    print(f"Tiempo promedio de ejecución total: {promedio_tiempo:.4f} segundos")
