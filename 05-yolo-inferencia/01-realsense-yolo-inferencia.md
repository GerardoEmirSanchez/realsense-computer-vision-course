# realsense-yolo-inferencia

```text
realsense-yolo-inferencia/
├── .gitignore
├── requirements.txt
├── README.md
├── descargar_pesos.py
├── s5_inferencia_yolo.py
└── s5_inferencia_yolo_multiobj.py
```

---

## Paso 0: Configuración en Windows (Host)

### 1. Instalar `usbipd-win`

En **PowerShell (como Administrador)**:
```PowerShell
winget install --interactive --exact dorssel.usbipd-win
```

### 2. Enlazar la cámara Intel RealSense a WSL
Con la cámara RealSense D435i/D455 conectada a un puerto USB 3.0:
1. Lista los dispositivos USB conectados para localizar el `BUSID` de la RealSense:
```PowerShell
usbipd list
```
2. Comparte el puerto del dispositivo con WSL (solo se requiere una vez):
```PowerShell
usbipd bind --busid <TU-BUSID>
```
3. Conectar el dispositivo a la instancia activa de WSL con persistencia automática:
```PowerShell
usbipd attach --wsl --busid <TU-BUSID> --auto-attach
```

---

## Paso 1: Creación del Directorio Local y Configuración de Git

Ejecuta en la terminal de **Ubuntu (WSL2)**:

```bash
# 1. Crear y acceder a la carpeta del proyecto
mkdir -p ~/realsense-yolo-inferencia
cd ~/realsense-yolo-inferencia

# 2. Inicializar repositorio Git
git init
git branch -M main

# 3. Crear archivo .gitignore
cat << 'EOF' > .gitignore
vision_env/
__pycache__/
*.pyc
*.pt
*.onnx
runs/
capture/
*.jpg
*.png
.vscode/
EOF

# 4. Crear requirements.txt con soporte para Ultralytics y RealSense
cat << 'EOF' > requirements.txt
pyrealsense2
opencv-python
numpy
flask
ultralytics
EOF
```

---

## Paso 2: Script Utilitario de Descarga de Pesos Base (`descargar_pesos.py`)

Descarga el modelo base pre-entrenado en COCO (`yolov8n.pt`) para verificar el pipeline de inferencia sin esperar al entrenamiento de noviembre:

```bash
cat << 'EOF' > ~/realsense-yolo-inferencia/descargar_pesos.py
#!/usr/bin/env python3
"""
==============================================================================
MR3005C: Descarga y Validación del Punto de Control Base YOLOv8 (Nano)
==============================================================================
"""

import os
from ultralytics import YOLO

print("\n[INFO] Inicializando descarga de 'yolov8n.pt' desde el repositorio oficial...")
modelo = YOLO("yolov8n.pt")
modelo.info()

print("\n[OK] Modelo 'yolov8n.pt' listo y validado para inferencia en tiempo real.")
EOF
```

---

## Paso 3: Script de Inferencia de Objeto Objetivo y Guiado (`s5_inferencia_yolo.py`)

Prioriza la detección con mayor nivel de confianza para el guiado del robot continuum, calcula los errores cartesianos $(e_u, e_v)$ respecto al centro óptico $(320, 240)$ y estima la profundidad frontal $Z$ en centímetros mediante el modelo Pinhole:

```bash
cat << 'EOF' > ~/realsense-yolo-inferencia/s5_inferencia_yolo.py
#!/usr/bin/env python3
"""
==============================================================================
MR3005C: Sistemas Ciberfísicos — Módulo 8: Visión Artificial
Sesión 5: Inferencia YOLOv8 en Vivo y Telemetría Cinemática para el Continuum
==============================================================================
"""

import os
import sys
import signal
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response
from ultralytics import YOLO

app = Flask(__name__)
os.makedirs("capture", exist_ok=True)
RUTA_EVIDENCIA = "capture/deteccion_yolo_000.jpg"

# ============================================================================
# 1. PARÁMETROS MECATRÓNICOS Y MODELO ÓPTICO PINHOLE
# ============================================================================
# Ancho físico estimado del objeto objetivo en metros (5 cm = 0.05 m)
W_REAL_M = 0.05

# Distancia focal calibrada para Intel RealSense D455 / D435i a 640x480
F_X = 615.0

# Centro óptico principal del sensor CMOS
U_C = 320.0
V_C = 240.0

# Umbral de confianza operativa
CONF_MINIMA = 0.40

# Selección del modelo: prioriza 'best.pt' personalizado; si no existe, usa 'yolov8n.pt'
MODELO_PATH = "best.pt" if os.path.exists("best.pt") else "yolov8n.pt"
print(f"\n[INFO] Cargando red neuronal convolucional desde: '{MODELO_PATH}'...")
model = YOLO(MODELO_PATH)

# ============================================================================
# 2. INICIALIZACIÓN DE HARDWARE (INTEL REALSENSE)
# ============================================================================
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe.start(cfg)

def liberar_recursos(sig=None, frame=None):
    try:
        pipe.stop()
    except Exception:
        pass
    print("\n[INFO] Cámara liberada. Sesión de inferencia finalizada.")
    sys.exit(0)

signal.signal(signal.SIGINT, liberar_recursos)

# Warm-up del sensor CMOS para estabilizar exposición automática (AEC)
for _ in range(10):
    pipe.wait_for_frames()

# ============================================================================
# 3. PIPELINE DE PROCESAMIENTO E INFERENCIA EN TIEMPO REAL
# ============================================================================
def procesar_cuadro_yolo(frame):
    anotado = frame.copy()
    h, w, _ = frame.shape

    # Retícula del centro óptico (320, 240)
    cv2.circle(anotado, (int(U_C), int(V_C)), 5, (255, 255, 255), -1)
    cv2.rectangle(anotado, (int(U_C) - 40, int(V_C) - 40), (int(U_C) + 40, int(V_C) + 40), (255, 255, 255), 1)

    t_inicio = time.perf_counter()
    # Inferencia con red neuronal YOLOv8 en una sola pasada (Single Forward Pass)
    resultados = model.predict(frame, conf=CONF_MINIMA, imgsz=640, verbose=False)[0]
    latencia_ms = (time.perf_counter() - t_inicio) * 1000.0
    fps = 1000.0 / max(latencia_ms, 1e-5)

    cajas = resultados.boxes

    if len(cajas) > 0:
        # Extraer la detección con mayor nivel de confianza
        idx_max = int(cajas.conf.argmax())
        caja_sel = cajas[idx_max]

        # Coordenadas enteras de la caja delimitadora (xyxy)
        x1, y1, x2, y2 = map(int, caja_sel.xyxy[0].tolist())
        confianza = float(caja_sel.conf[0])
        clase_nombre = resultados.names[int(caja_sel.cls[0])]

        # Centroide (cx, cy) y dimensiones en píxeles
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        ancho_px = max(x2 - x1, 1)

        # 1. Vector de error cinemático de alineación respecto al centro óptico
        e_u = cx - int(U_C)
        e_v = cy - int(V_C)

        # 2. Estimación de distancia frontal en profundidad Z (Modelo Pinhole)
        z_est_m = (F_X * W_REAL_M) / ancho_px
        z_est_cm = z_est_m * 100.0

        # Criterio cinemático de acople para el robot continuum
        alineado = (abs(e_u) <= 40) and (abs(e_v) <= 40) and (15.0 <= z_est_cm <= 35.0)
        color_caja = (0, 255, 0) if alineado else (0, 165, 255)
        estado_texto = "ALINEADO PARA SUJECION" if alineado else "CENTRICIDAD REQUERIDA"

        # Trazado de primitivas gráficas
        cv2.rectangle(anotado, (x1, y1), (x2, y2), color_caja, 2)
        cv2.circle(anotado, (cx, cy), 6, (0, 0, 255), -1)
        cv2.line(anotado, (int(U_C), int(V_C)), (cx, cy), (255, 0, 0), 2)

        # Barra superior de telemetría de control
        cv2.rectangle(anotado, (10, 10), (630, 80), (15, 23, 42), -1)
        cv2.rectangle(anotado, (10, 10), (630, 80), (51, 65, 85), 1)

        cv2.putText(anotado, f"{clase_nombre.upper()} ({confianza*100:.1f}%) | Z Estimada: {z_est_cm:.1f} cm",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(anotado, f"Error: [eu:{e_u:+.0f}, ev:{e_v:+.0f}] px | {estado_texto} | {fps:.1f} FPS",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_caja, 2)

        cv2.imwrite(RUTA_EVIDENCIA, anotado)
    else:
        cv2.putText(anotado, f"BUSCANDO OBJETO... | Latencia: {latencia_ms:.1f} ms ({fps:.1f} FPS)",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 255), 2)

    return anotado

def generar_streaming():
    while True:
        try:
            frames = pipe.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue

            frame = np.asanyarray(color.get_data())
            anotado = procesar_cuadro_yolo(frame)

            ok, buffer = cv2.imencode('.jpg', anotado)
            if not ok:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception:
            break

@app.route('/')
def feed():
    return Response(generar_streaming(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("\n=======================================================")
    print(" SESIÓN 5: INFERENCIA YOLOV8 Y TELEMETRÍA CINEMÁTICA")
    print(" Monitor activo en: http://localhost:5000")
    print(f" Modelo en uso: '{MODELO_PATH}'")
    print(" Presiona Ctrl + C para terminar limpiamente.")
    print("=======================================================\n")
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        liberar_recursos()
EOF
```

---

## Paso 4: Script de Inferencia Multi-Objeto (`s5_inferencia_yolo_multiobj.py`)

Detecta y dibuja simultáneamente **todas las cajas y clases visibles** en la escena con colores independientes por clase, porcentaje de confianza y cálculo de distancia individual:

```bash
cat << 'EOF' > ~/realsense-yolo-inferencia/s5_inferencia_yolo_multiobj.py
#!/usr/bin/env python3
"""
==============================================================================
MR3005C: Sistemas Ciberfísicos — Módulo 8: Visión Artificial
Sesión 5: Inferencia Multi-Objeto en Tiempo Real con YOLOv8 y RealSense
==============================================================================
"""

import os
import sys
import signal
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response
from ultralytics import YOLO

app = Flask(__name__)
os.makedirs("capture", exist_ok=True)
RUTA_EVIDENCIA = "capture/deteccion_yolo_todas.jpg"

# ============================================================================
# 1. PARÁMETROS MECATRÓNICOS Y MODELO ÓPTICO PINHOLE
# ============================================================================
W_REAL_M = 0.05    # Ancho físico de referencia para estimación métrica (5 cm)
F_X = 615.0        # Distancia focal calibrada a 640x480
U_C = 320.0        # Centro óptico horizontal
V_C = 240.0        # Centro óptico vertical

CONF_MINIMA = 0.35 # Umbral de confianza más flexible para visualización general

MODELO_PATH = "best.pt" if os.path.exists("best.pt") else "yolov8n.pt"
print(f"\n[INFO] Cargando red neuronal convolucional desde: '{MODELO_PATH}'...")
model = YOLO(MODELO_PATH)

# ============================================================================
# 2. INICIALIZACIÓN DE HARDWARE (INTEL REALSENSE)
# ============================================================================
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe.start(cfg)

def liberar_recursos(sig=None, frame=None):
    try:
        pipe.stop()
    except Exception:
        pass
    print("\n[INFO] Cámara liberada. Sesión cerrada limpiamente.")
    sys.exit(0)

signal.signal(signal.SIGINT, liberar_recursos)

for _ in range(10):
    pipe.wait_for_frames()

# Paleta fija de colores pseudoaleatorios para diferenciar clases
np.random.seed(42)
COLORES = np.random.uniform(50, 255, size=(100, 3))

# ============================================================================
# 3. PROCESAMIENTO MULTI-OBJETO EN VIVO
# ============================================================================
def procesar_cuadro_yolo_multi(frame):
    anotado = frame.copy()
    h, w, _ = frame.shape

    # Retícula de referencia central (320, 240)
    cv2.circle(anotado, (int(U_C), int(V_C)), 4, (255, 255, 255), -1)
    cv2.rectangle(anotado, (int(U_C) - 30, int(V_C) - 30), (int(U_C) + 30, int(V_C) + 30), (255, 255, 255), 1)

    t_inicio = time.perf_counter()
    # Inferencia multi-objeto en una sola pasada
    resultados = model.predict(frame, conf=CONF_MINIMA, imgsz=640, verbose=False)[0]
    latencia_ms = (time.perf_counter() - t_inicio) * 1000.0
    fps = 1000.0 / max(latencia_ms, 1e-5)

    cajas = resultados.boxes
    n_detectados = len(cajas)

    # Iterar sobre TODAS las detecciones encontradas
    for b in cajas:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        conf = float(b.conf[0])
        cls_id = int(b.cls[0])
        clase_nombre = resultados.names[cls_id]

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        ancho_px = max(x2 - x1, 1)

        # Estimación de distancia frontal individual
        z_est_cm = ((F_X * W_REAL_M) / ancho_px) * 100.0

        color = [int(c) for c in COLORES[cls_id % len(COLORES)]]

        # Trazado de caja delimitadora y centroide
        cv2.rectangle(anotado, (x1, y1), (x2, y2), color, 2)
        cv2.circle(anotado, (cx, cy), 4, color, -1)
        cv2.line(anotado, (int(U_C), int(V_C)), (cx, cy), color, 1)

        texto = f"{clase_nombre} {conf*100:.0f}% | ~{z_est_cm:.0f}cm"
        y_text = max(y1 - 8, 20)
        cv2.putText(anotado, texto, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Barra superior con resumen global
    cv2.rectangle(anotado, (10, 10), (630, 45), (15, 23, 42), -1)
    cv2.rectangle(anotado, (10, 10), (630, 45), (51, 65, 85), 1)

    cv2.putText(anotado, f"OBJETOS: {n_detectados} | {latencia_ms:.1f} ms ({fps:.1f} FPS) | Conf: {CONF_MINIMA*100:.0f}%",
                (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    if n_detectados > 0:
        cv2.imwrite(RUTA_EVIDENCIA, anotado)

    return anotado

def generar_streaming():
    while True:
        try:
            frames = pipe.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue

            frame = np.asanyarray(color.get_data())
            anotado = procesar_cuadro_yolo_multi(frame)

            ok, buffer = cv2.imencode('.jpg', anotado)
            if not ok:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception:
            break

@app.route('/')
def feed():
    return Response(generar_streaming(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("\n=======================================================")
    print(" INFERENCIA MULTI-OBJETO YOLOV8 EN TIEMPO REAL")
    print(" Monitor activo en: http://localhost:5000")
    print(f" Modelo en uso: '{MODELO_PATH}'")
    print(" Presiona Ctrl + C para terminar limpiamente.")
    print("=======================================================\n")
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        liberar_recursos()
EOF
```

---

## Paso 5: Configurar Permisos y Dependencias en Ubuntu (WSL2)

Ejecuta en la terminal de **Ubuntu**:

```bash
# 1. Renovar permisos sobre el hardware USB
sudo chmod 666 /dev/video* 2>/dev/null
sudo chmod -R 777 /dev/bus/usb/ 2>/dev/null
lsusb
```
*Verifica que aparezca:* `Intel Corp. Intel(R) RealSense(TM) Depth Camera`[cite: 1, 8].

```bash
# 2. Activar el entorno virtual de visión
source ~/vision_env/bin/activate

# 3. Instalar ultralytics y dependencias en el entorno
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Paso 6: Descarga y Validación de Pesos

Ejecuta el script utilitario para inicializar y descargar los pesos base `yolov8n.pt`:

```bash
python descargar_pesos.py
```

---

## Paso 7: Ejecución y Validación

### Opción A: Objeto Objetivo y Telemetría Cinemática para el Continuum
Filtra el objeto de máxima confianza, reporta offsets $(e_u, e_v)$, distancia frontal y valida el acople:
```bash
python s5_inferencia_yolo.py
```

### Opción B: Inferencia Multi-Objeto (Detección de Todas las Cajas)
Muestra simultáneamente todos los objetos clasificados por la red con cajas independientes y colores distintos:
```bash
python s5_inferencia_yolo_multiobj.py
```

Abre en tu navegador de Windows (Chrome o Edge):
```text
http://localhost:5000
```

* Para detener cualquiera de los scripts, presiona `Ctrl + C` en la terminal de Ubuntu.

---

## Paso 8: `README.md` del Repositorio

Genera la documentación completa ejecutando este bloque:

```bash
    cat << 'EOF' > ~/realsense-yolo-inferencia/README.md
    # MR3005C: Inteligencia Artificial con YOLOv8, Inferencia en Vivo y Telemetría Robótica
    
    Sistema de detección de objetos en tiempo real basado en redes neuronales convolucionales (*Single-Stage Detector*) utilizando cámaras Intel RealSense (serie D400), el framework Ultralytics YOLOv8 y visualización web multipart vía Flask en entornos WSL2 (Ubuntu).
    
    ---
    
    ## Índice de Contenidos
    1. [Arquitectura de Red y Principio Anchor-Free](#arquitectura-de-red)
    2. [Protocolo de Enlace USB tras Reinicio](#protocolo-de-enlace-usb)
    3. [Instalación de Dependencias](#instalación-de-dependencias)
    4. [Descarga de Pesos Base (`descargar_pesos.py`)](#descarga-de-pesos)
    5. [Inferencia para Guiado Cinemático (`s5_inferencia_yolo.py`)](#inferencia-guiado)
    6. [Inferencia Multi-Objeto en Escena (`s5_inferencia_yolo_multiobj.py`)](#inferencia-multiobjeto)
    7. [Telemetría Métrica y Estimación de Distancia Pinhole](#telemetría-métrica)
    8. [Métricas Oficiales de Evaluación para Noviembre](#métricas-oficiales)
    9. [Solución de Problemas Frecuentes](#solución-de-problemas)
    
    ---
    
    ## 1. Arquitectura de Red y Principio Anchor-Free
    
    YOLOv8 reemplaza el paradigma tradicional de clasificación por recortes (*Two-Stage Detectors* como R-CNN) formulando la detección como un problema de regresión unificado en una sola pasada (*Single Forward Pass*):
    
    * **Backbone (CSPDarknet + Bloques C2f):** Extrae características visuales jerárquicas mediante convoluciones sucesivas a tres escalas espaciales:
      * **P3 ($80 \times 80$ celdas):** Especializado en piezas y objetos pequeños.
      * **P4 ($40 \times 40$ celdas):** Especializado en objetos de escala media.
      * **P5 ($20 \times 20$ celdas):** Especializado en estructuras globales grandes (ej. gaveta completa).
    * **Neck (PAN-FPN):** Red piramidal bidireccional que fusiona la información fina superficial con la semántica profunda.
    * **Head Desacoplada (*Anchor-Free*):** Separa la rama de regresión de cajas de la rama de clasificación. No utiliza cajas ancla rígidas; cada celda de la cuadrícula predice directamente cuatro distancias métricas continuas hacia las fronteras del objeto:
      $$\mathbf{caja} = [l, t, r, b] \quad (\text{left, top, right, bottom})$$
    
    ---
    
    ## 2. Protocolo de Enlace USB
    
    Al reiniciar la máquina host en Windows o reconectar el cable de la cámara:
    
    1. Abrir la terminal de **Ubuntu** desde el menú Inicio.
    2. En **PowerShell (como Administrador)** ejecutar:
       ```powershell
       usbipd attach --wsl --busid <TU-BUSID> --auto-attach
       ```
    3. En la terminal de Ubuntu renovar permisos de acceso al dispositivo:
       ```bash
       sudo chmod 666 /dev/video* 2>/dev/null
       sudo chmod -R 777 /dev/bus/usb/ 2>/dev/null
       ```
    
    ---
    
    ## 3. Instalación de Dependencias
    
    ```bash
    # Activar entorno virtual de visión
    source ~/vision_env/bin/activate
    
    # Instalar librerías
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    
    ---
    
    ## 4. Descarga de Pesos Base
    
    Para descargar y verificar los pesos base oficiales:
    ```bash
    python descargar_pesos.py
    ```
    
    ---
    
    ## 5. Inferencia para Guiado Cinemático (`s5_inferencia_yolo.py`)
    
    Aísla el objeto de mayor certeza, reporta errores cinemáticos $(e_u, e_v)$ y valida el acople:
    ```bash
    python s5_inferencia_yolo.py
    ```
    Abrir en el navegador de Windows: `http://localhost:5000`
    
    ---
    
    ## 6. Inferencia Multi-Objeto en Escena (`s5_inferencia_yolo_multiobj.py`)
    
    Detecta todas las clases y objetos visibles simultáneamente en el campo de visión:
    ```bash
    python s5_inferencia_yolo_multiobj.py
    ```
    Abrir en el navegador de Windows: `http://localhost:5000`
    
    ---
    
    ## 7. Telemetría Métrica
    
    A partir de las cuatro coordenadas entregadas por la red neuronal $[x_1, y_1, x_2, y_2]$:
    
    ### 1. Centroide y Vector de Error de Alineación:
    $$c_x = \frac{x_1 + x_2}{2}, \quad c_y = \frac{y_1 + y_2}{2}$$
    $$e_u = c_x - 320, \quad e_v = c_y - 240$$
    
    ### 2. Estimación de Distancia Frontal ($Z$) por Modelo Pinhole:
    Conociendo el ancho real de la pieza física ($W_{\text{real}} = 0.05\text{ m}$) y la distancia focal calibrada ($f_x \approx 615\text{ px}$):
    $$w_{\text{px}} = x_2 - x_1$$
    $$Z_{\text{est}} = \frac{f_x \cdot W_{\text{real}}}{w_{\text{px}}} = \frac{615 \cdot 0.05}{w_{\text{px}}} \quad (\text{en metros})$$
    
    ---
    
    ## 8. Métricas Oficiales de Evaluación para Noviembre
    
    Para la entrega del **Entregable A7 (12 de Noviembre)**:
    
    * **Intersection over Union (IoU):**
      $$\text{IoU} = \frac{\text{Área}(\text{Predicción} \cap \text{Ground Truth})}{\text{Área}(\text{Predicción} \cup \text{Ground Truth})}$$
      Se clasifica como acierto geométrico (*True Positive*) si $\text{IoU} \ge 0.50$.
    * **Mean Average Precision ($mAP_{50}$):** Área bajo la curva Precision-Recall.
      * **Criterio de aprobación:** $mAP_{50} \ge 0.85$ sobre el conjunto de prueba (*Test split*).
    
    ---
    
    ## 9. Solución de Problemas Frecuentes
    
    * **`ImportError: No module named 'ultralytics'`:** Confirma que el entorno virtual esté activo (`source ~/vision_env/bin/activate`) y ejecuta `pip install ultralytics`.
    * **Latencia elevada (< 10 FPS):** Verifica que la inferencia utilice resolución `imgsz=640` o `imgsz=480`. Si se ejecuta sobre CPU, un valor típico oscila entre 20 y 35 FPS para el modelo Nano (`yolov8n`).
    * **`RuntimeError: No device connected`:** Confirma que la RealSense esté enlazada a WSL2 con `usbipd list` y reconecta con `--auto-attach`.
    EOF
    ```
    
    ---
    
    ## Paso 9: Subir el Repositorio a GitHub
    
    1. Ve a [GitHub](https://github.com/new).
    2. Nombra el repositorio: **`realsense-yolo-inferencia`**.
    3. Selecciona **Public** y deja las casillas de inicialización desmarcadas.
    4. En tu terminal de Ubuntu ejecuta:
    
    ```bash
    cd ~/realsense-yolo-inferencia
    git add .
    git commit -m "feat: repositorio actualizado con inferencia guiada y multi-objeto yolov8"
    git remote add origin [https://github.com/GerardoEmirSanchez/realsense-yolo-inferencia.git](https://github.com/GerardoEmirSanchez/realsense-yolo-inferencia.git)
    git push -u origin main
```
