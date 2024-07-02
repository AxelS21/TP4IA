import cv2
import numpy as np

def detect_circles(image_path):
    # Leer la imagen en color
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error al abrir la imagen {image_path}")
        return None

    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar suavizado Gaussiano
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detectar bordes
    edges = cv2.Canny(blurred, 50, 150)

    # Aplicar la Transformada de Hough con parámetros ajustados
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                               param1=50, param2=25, minRadius=10, maxRadius=100)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
        return img, circles
    else:
        print("No se encontraron circunferencias.")
        return img, None

# Ejemplo de uso
image_path = 'smarties.png'
detected_img, detected_circles = detect_circles(image_path)

if detected_img is not None:
    cv2.imshow('Detected Circles', detected_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()