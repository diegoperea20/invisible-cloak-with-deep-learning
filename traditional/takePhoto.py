import cv2
import time

# Inicializa la cámara
cap = cv2.VideoCapture(0)

# Espera 5 segundos
time.sleep(5)

# Verifica si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Captura un fotograma de la cámara
ret, frame = cap.read()

# Verifica si la captura fue exitosa
if not ret:
    print("Error: No se pudo capturar un fotograma.")
else:
    # Guarda la imagen en un archivo
    cv2.imwrite("foto_capturada.jpg", frame)
    print("Foto guardada como 'foto_capturada.jpg'.")

# Libera la cámara y cierra la ventana de vista previa si está abierta
cap.release()
cv2.destroyAllWindows()
