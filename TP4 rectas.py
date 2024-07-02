import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen
imagen = cv2.imread('images/Sudoku_resuelto_completo.png')

# Convertir a escala de grises
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Detectar bordes con Canny
bordes = cv2.Canny(imagen_gris, 50, 150)

# Aplicar la transformada de Hough
lineas = cv2.HoughLines(bordes, 1, np.pi / 180, 200)

# Dibujar las líneas detectadas
if lineas is not None:
    for rho, theta in lineas[:, 0]:
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
plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
plt.title('Líneas detectadas')
plt.show()
