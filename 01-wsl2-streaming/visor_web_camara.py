from flask import Flask, Response
import cv2
import numpy as np
import os
import pyrealsense2 as rs

app = Flask(__name__)
os.makedirs("capture", exist_ok=True)

# 1. Configuración del sensor Intel RealSense
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe.start(cfg)

def flujo_video():
    while True:
        frames = pipe.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            continue

        # Convertir buffer nativo a ndarray de NumPy
        frame = np.asanyarray(color.get_data())
        h, w, _ = frame.shape

        # Elementos didácticos: Centro óptico y ROI del cajón
        y_c, x_c = h // 2, w // 2
        cv2.circle(frame, (x_c, y_c), 6, (0, 0, 255), -1)
        
        y1, y2 = int(h * 0.2), int(h * 0.8)
        x1, x2 = int(w * 0.2), int(w * 0.8)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.putText(frame, f"Centro Optico ({x_c},{y_c})", (x_c + 10, y_c), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Persistencia continua del último frame (Muestra A6)
        cv2.imwrite("capture/captura_cajon_000.jpg", frame)

        # Codificación JPEG para transmisión HTTP multipart
        ok, buffer = cv2.imencode('.jpg', frame)
        if not ok:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def video_feed():
    return Response(flujo_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("\n=======================================================")
    print(" Transmision activa. Abre tu navegador en Windows y entra a:")
    print(" http://localhost:5000")
    print("=======================================================\n")
    app.run(host='0.0.0.0', port=5000, threaded=True)
