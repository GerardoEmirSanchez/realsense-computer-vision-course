import os
import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response, render_template_string

app = Flask(__name__)

# Configuración de cámara
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe.start(cfg)

def capturar_frame():
    frames = pipe.wait_for_frames()
    color = frames.get_color_frame()
    return np.asanyarray(color.get_data()) if color else None

# -------------------------------------------------------------
# RETOS MATRICIALES (NUMPY SIN FOR)
# -------------------------------------------------------------
def resolver_reto1_sin_azul(frame):
    # Reto 1: Apagar canal Azul (B = Canal 0)
    res = frame.copy()
    res[:, :, 0] = 0
    return res

def resolver_reto2_negativo(frame):
    # Reto 2: Negativo exacto
    return 255 - frame

def resolver_reto3_punto_brillante(frame):
    # Reto 3: Localizar punto más luminoso
    gray = (0.114 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.299 * frame[:, :, 2]).astype(np.uint8)
    idx = np.argmax(gray)
    y_max, x_max = np.unravel_index(idx, gray.shape)
    
    res = frame.copy()
    cv2.circle(res, (x_max, y_max), 8, (0, 0, 255), -1)
    cv2.putText(res, f"Max: ({x_max},{y_max})", (x_max + 10, y_max),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return res

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Mini-Retos Sesion 1</title>
    <style>
        body { background:#0f172a; color:#fff; font-family:sans-serif; text-align:center; margin:10px; }
        .grid { display:grid; grid-template-columns:1fr 1fr; gap:15px; max-width:1100px; margin:auto; }
        .card { background:#1e293b; padding:10px; border-radius:8px; }
        img { width:100%; max-width:480px; border-radius:4px; }
    </style>
</head>
<body>
    <h2>MR3005C: Evaluación de Mini-Retos en Vivo</h2>
    <div class="grid">
        <div class="card"><h4>1. Original (BGR)</h4><img src="/feed/orig"></div>
        <div class="card"><h4>2. Reto 1: Sin Canal Azul</h4><img src="/feed/r1"></div>
        <div class="card"><h4>3. Reto 2: Negativo (255 - img)</h4><img src="/feed/r2"></div>
        <div class="card"><h4>4. Reto 3: Punto Máximo (argmax)</h4><img src="/feed/r3"></div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/feed/<modo>')
def feed(modo):
    def gen():
        while True:
            f = capturar_frame()
            if f is None:
                continue
            if modo == 'r1': out = resolver_reto1_sin_azul(f)
            elif modo == 'r2': out = resolver_reto2_negativo(f)
            elif modo == 'r3': out = resolver_reto3_punto_brillante(f)
            else: out = f
            
            ok, buf = cv2.imencode('.jpg', out)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Servidor de retos activo. Abre en Windows: http://localhost:5001\n")
    app.run(host='0.0.0.0', port=5001, threaded=True)
