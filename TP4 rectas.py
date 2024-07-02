import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen (usa barras diagonales o cadena raw para la ruta del archivo)
imagen = cv2.imread('images/Sudoku_resuelto_completo.png')

# Convertir a escala de grises
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Aplicar desenfoque gaussiano para reducir el ruido
imagen_gris = cv2.GaussianBlur(imagen_gris, (5, 5), 0)

# Ajustar parámetros de detección de bordes Canny
bordes = cv2.Canny(imagen_gris, 50, 150, apertureSize=3)

# Ajustar parámetros de la Transformada de Hough
lineas = cv2.HoughLines(bordes, 1, np.pi / 180, 150)

# Función para filtrar líneas basadas en el ángulo
def filtrar_lineas(lineas, umbral_angulo=5):
    lineas_filtradas = []
    for linea in lineas:
        rho, theta = linea[0]  # Desempaquetar el primer (y único) elemento de cada array de línea
        # Convertir theta a grados
        angulo = np.degrees(theta)
        # Mantener solo líneas casi horizontales o casi verticales
        if angulo < umbral_angulo or (90 - umbral_angulo < angulo < 90 + umbral_angulo) or angulo > 180 - umbral_angulo:
            lineas_filtradas.append(linea[0])  # Agregar los valores desempaquetados
    return lineas_filtradas

# Filtrar las líneas
if lineas is not None:
    lineas_filtradas = filtrar_lineas(lineas)
else:
    lineas_filtradas = None

# Dibujar las líneas filtradas
if lineas_filtradas is not None:
    for rho, theta in lineas_filtradas:  # Ahora podemos desempaquetar directamente rho y theta
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(imagen, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Mostrar la imagen con las líneas detectadas
plt.figure(figsize=(12, 10))
plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
plt.title('Líneas detectadas (mejorado)')
plt.axis('off')
plt.show()