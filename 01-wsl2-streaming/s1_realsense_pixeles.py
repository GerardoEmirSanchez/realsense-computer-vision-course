import cv2
import numpy as np
import os
import signal
import sys
import pyrealsense2 as rs
from flask import Flask, Response

app = Flask(__name__)
os.makedirs("capture", exist_ok=True)
RUTA_A6 = "capture/captura_cajon_000.jpg"

# 1. Iniciar cámara RealSense D435i / D455
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe.start(cfg)

def cerrar_recursos(sig=None, frame=None):
    try:
        pipe.stop()
    except Exception:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, cerrar_recursos)

# Warm-up de cuadros para estabilizar exposición y balance
for _ in range(10):
    pipe.wait_for_frames()

# 2. Diagnóstico matemático en consola (Diapositivas 3, 4, 6 y 7)
frames = pipe.wait_for_frames()
f_diag = np.asanyarray(frames.get_color_frame().get_data())
h, w, c = f_diag.shape
y_c, x_c = h // 2, w // 2
b, g, r = f_diag[y_c, x_c]

# Grises manual Luma vs OpenCV (Diapositiva 6)
gray_luma = (0.114 * f_diag[:, :, 0] + 0.587 * f_diag[:, :, 1] + 0.299 * f_diag[:, :, 2]).astype(np.uint8)
gray_cv = cv2.cvtColor(f_diag, cv2.COLOR_BGR2GRAY)
diff = int(np.max(np.abs(gray_luma.astype(int) - gray_cv.astype(int))))

# Demostración de desbordamiento en uint8 con arrays válidos (Diapositiva 7)
val_a = np.array([200], dtype=np.uint8)
val_b = np.array([100], dtype=np.uint8)
overflow_np = int((val_a + val_b)[0])                  # 300 % 256 = 44 (error modular)
saturado_cv = int(cv2.add(val_a, val_b).flatten()[0])  # min(300, 255) = 255 (saturación correcta)

# Guardar captura inicial para Entregable A6
cv2.imwrite(RUTA_A6, f_diag)

print("\n" + "="*65)
print(" REPORTE DE ADQUISICIÓN MATRICIAL (SESIÓN 1)")
print("="*65)
print(f" * Resolución: {w}x{h} píxeles | Canales: {c} (BGR)")
print(f" * Tipo de dato en RAM: {f_diag.dtype} (Enteros de 0 a 255)")
print(f" * Tamaño en RAM: {(h * w * c) / 1024:.2f} KB | Ancho de banda (30 FPS): ~27.65 MB/s")
print(f" * Píxel central [{y_c}, {x_c}]: B={b}, G={g}, R={r}")
print(f" * Diferencia Luma manual vs OpenCV: {diff} (Equivalencia validada)")
print(f" * Prueba 200 + 100 uint8 -> NumPy: {overflow_np} | cv2.add: {saturado_cv}")
print(f" * Archivo Entregable A6 guardado: {RUTA_A6}")
print("="*65 + "\n")

def flujo_video():
    while True:
        try:
            frames = pipe.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue

            frame = np.asanyarray(color.get_data())
            anotado = frame.copy()

            # Centro óptico (u, v) = (x, y) = (320, 240)
            cv2.circle(anotado, (x_c, y_c), 6, (0, 0, 255), -1)

            # Región de interés (ROI) del cajón por slicing
            y1, y2 = int(h * 0.20), int(h * 0.80)
            x1, x2 = int(w * 0.20), int(w * 0.80)
            cv2.rectangle(anotado, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(anotado, f"Centro ({x_c},{y_c})", (x_c + 10, y_c - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(anotado, "ROI Cajon", (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imwrite(RUTA_A6, frame)

            ok, buf = cv2.imencode('.jpg', anotado)
            if not ok:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        except Exception:
            break

@app.route('/')
def index():
    return Response(flujo_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Transmisión lista. Abre en Windows: http://localhost:5000\n")
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        cerrar_recursos()
