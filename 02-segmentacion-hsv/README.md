# realsense-segmentacion-hsv

```text
realsense-segmentacion-hsv/
├── .gitignore
├── requirements.txt
├── README.md
├── s2_segmentacion_hsv.py
├── s2_mini_retos_estudiantes.py
└── s2_mini_retos_resuelto.py
```

---

## Paso 0: Configuración en Windows (Host)

### 1. Instalar `usbipd-win`

En **PowerShell (como Administrador)**:
```PowerShell
winget install --interactive --exact dorssel.usbipd-win
```
### 2. Enlazar la cámara Intel RealSense a WSL
Con la cámara conectada a un puerto USB 3.0:
1. Lista los dispositivos USB conectados para localizar el `BUSID` de la RealSense:
```PowerShell
   usbipd list
   
```
2. Comparte el puerto del dispositivo con WSL (solo se requiere una vez):
```PowerShell
   usbipd bind --busid <TU-BUSID>
```
3. Conectar el dispositivo a la instancia activa de WSL:
```PowerShell
   usbipd attach --wsl --busid <TU-BUSID>
```


## Paso 1: Creación del Directorio Local y Configuración de Git
Ejecuta en la terminal de Ubuntu (WSL2):

```bash
# 1. Crear y acceder a la carpeta del nuevo proyecto
mkdir -p ~/realsense-segmentacion-hsv
cd ~/realsense-segmentacion-hsv
```
```bash
# 2. Inicializar repositorio local
git init
git branch -M main
```
```bash
# 3. Crear .gitignore
cat << 'EOF' > .gitignore
vision_env/
__pycache__/
*.pyc
capture/
*.jpg
*.png
.vscode/
EOF
```
```bash
# 4. Crear requirements.txt
cat << 'EOF' > requirements.txt
pyrealsense2
opencv-python
numpy
flask
EOF
```

---

## Paso 2: Script Principal de Clase (`s2_segmentacion_hsv.py`)
Genera el script del pipeline en vivo con mosaico de 4 cuadrantes en el puerto 5000:

```bash
cat << 'EOF' > ~/realsense-segmentacion-hsv/s2_segmentacion_hsv.py
#!/usr/bin/env python3
import os
import sys
import signal
import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response

app = Flask(__name__)
os.makedirs("capture", exist_ok=True)
RUTA_EVIDENCIA = "capture/segmentacion_cajon_000.jpg"

# 1. Inicialización de Hardware RealSense
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe.start(cfg)

def liberar_recursos(sig=None, frame=None):
    try:
        pipe.stop()
    except Exception:
        pass
    print("\n[INFO] Cámara liberada. Proceso terminado limpiamente.")
    sys.exit(0)

signal.signal(signal.SIGINT, liberar_recursos)

# Warm-up de la cámara para estabilizar exposición
for _ in range(10):
    pipe.wait_for_frames()

# Elemento estructurante para limpieza morfológica
KERNEL = np.ones((5, 5), np.uint8)

def procesar_cuadrantes(frame):
    h, w, _ = frame.shape
    u_c, v_c = w // 2, h // 2  # Centro óptico (320, 240)

    # 1. Transformación de Espacio de Color BGR -> HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. Umbralización Vectorial (Rango para pieza roja/cálida)
    mask1 = cv2.inRange(hsv, np.array([0, 100, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 100, 50]), np.array([179, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)

    # 3. Limpieza Morfológica (Apertura y Clausura)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, KERNEL)

    # 4. Cuadrante de Tracking Cinemático
    tracking = frame.copy()
    cv2.circle(tracking, (u_c, v_c), 5, (255, 255, 255), -1)  # Centro óptico blanco

    # 5. Cálculo de Momentos y Centroide
    M = cv2.moments(mask_clean)
    area = M["m00"]

    if area > 500:  # Filtro de área mínima contra ruido térmico
        cx = int(M["m10"] / area)
        cy = int(M["m01"] / area)
        e_u = cx - u_c
        e_v = cy - v_c

        # Dibujo de vector y centroide
        cv2.circle(tracking, (cx, cy), 8, (0, 255, 0), -1)
        cv2.line(tracking, (u_c, v_c), (cx, cy), (255, 0, 0), 2)
        cv2.putText(tracking, f"Centroide: ({cx},{cy})", (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(tracking, f"Error: [eu:{e_u}, ev:{e_v}]", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    else:
        cv2.putText(tracking, "BUSCANDO PIEZA...", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Guardar último frame de tracking para entregable oficial A6
    cv2.imwrite(RUTA_EVIDENCIA, tracking)

    # 6. Construcción del Mosaico 2x2
    orig_anotado = frame.copy()
    cv2.circle(orig_anotado, (u_c, v_c), 5, (0, 0, 255), -1)
    cv2.putText(orig_anotado, "1. BGR Original", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    hsv_vis = hsv.copy()
    cv2.putText(hsv_vis, "2. Espacio HSV", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    mask_bgr = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
    cv2.putText(mask_bgr, f"3. Mascara (Area: {int(area)})", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(tracking, "4. Tracking Cinematico", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    fila_arriba = np.hstack((orig_anotado, hsv_vis))
    fila_abajo = np.hstack((mask_bgr, tracking))
    mosaico = np.vstack((fila_arriba, fila_abajo))

    # Redimensionar a resolución nativa 640x480 para transmisión ligera
    return cv2.resize(mosaico, (640, 480))

def generar_frames():
    while True:
        try:
            frames = pipe.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue

            frame = np.asanyarray(color.get_data())
            mosaico = procesar_cuadrantes(frame)

            ok, buffer = cv2.imencode('.jpg', mosaico)
            if not ok:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception:
            break

@app.route('/')
def video_feed():
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("\n=======================================================")
    print(" SESIÓN 2: PIPELINE DE SEGMENTACIÓN Y TRACKING EN VIVO")
    print(" Monitor activo en: http://localhost:5000")
    print(" Presiona Ctrl + C para terminar la sesión de forma limpia.")
    print("=======================================================\n")
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        liberar_recursos()
EOF
```

---

## Paso 3: Plantilla para Estudiantes (`s2_mini_retos_estudiantes.py`)
Genera el script sin resolver con el dashboard en el puerto 5001:

```bash
cat << 'EOF' > ~/realsense-segmentacion-hsv/s2_mini_retos_estudiantes.py
#!/usr/bin/env python3
"""
==============================================================================
MR3005C: Sistemas Ciberfísicos — Módulo 8: Visión Artificial
Sesión 2: Mini-Retos Prácticos de Segmentación y Disparadores Robóticos
==============================================================================

INSTRUCCIONES PARA EL EQUIPO:
1. Trabajar en parejas durante 15 minutos.
2. Completar las tres funciones marcadas con operaciones de NumPy y OpenCV:
      - reto_1_segmentar_amarillo(frame, hsv)
      - reto_2_mascara_inversa(frame, hsv)
      - reto_3_trigger_proximidad(frame, hsv)
3. PROHIBIDO usar ciclos 'for'. Todo el procesamiento debe ser matricial.
4. Para evaluar su avance en vivo:
      - Ejecuten: python s2_mini_retos_estudiantes.py
      - Abran en Chrome/Edge en Windows: http://localhost:5001
"""

import os
import sys
import signal
import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response, render_template_string

# 1. Configuración de Hardware
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe.start(cfg)

def liberar_camara(sig=None, frame=None):
    try:
        pipe.stop()
    except Exception:
        pass
    print("\n[INFO] Cámara liberada correctamente.")
    sys.exit(0)

signal.signal(signal.SIGINT, liberar_camara)

def adquirir_frame():
    try:
        frames = pipe.wait_for_frames()
        c = frames.get_color_frame()
        return np.asanyarray(c.get_data()) if c else None
    except Exception:
        return None

# Umbral calibrado para detección de proximidad real (distancia de agarre)
UMBRAL_DISTANCIA_AGARRE = 25000

# ============================================================================
# MINI-RETOS EN PAREJAS (EDITAR ÚNICAMENTE ESTA SECCIÓN)
# ============================================================================

def reto_1_segmentar_amarillo(frame, hsv):
    """
    RETO 1: Calibración de Rango para Pieza Amarilla
    Objetivo: Aislar exclusivamente el tono amarillo (H: 20-35 en OpenCV).
    Retornar la máscara binaria en 3 canales (cv2.cvtColor a BGR).
    """
    salida = frame.copy()

    # ------------------------------------------------------------------------
    # [CÓDIGO ALUMNOS - RETO 1]
    # Definir lower_yellow y upper_yellow
    # Aplicar cv2.inRange y morfología matemática
    # ------------------------------------------------------------------------

    return salida


def reto_2_mascara_inversa(frame, hsv):
    """
    RETO 2: Segmentación Inversa (Eliminación de la Pieza)
    Objetivo: Conservar el entorno y colocar en negro absoluto la pieza amarilla.
    Restricción: Utilizar operaciones lógicas bitwise (cv2.bitwise_not / cv2.bitwise_and).
    """
    salida = frame.copy()

    # ------------------------------------------------------------------------
    # [CÓDIGO ALUMNOS - RETO 2]
    # Invertir la máscara y aplicarla sobre frame
    # ------------------------------------------------------------------------

    return salida


def reto_3_trigger_proximidad(frame, hsv):
    """
    RETO 3: Disparador Cinemático de Proximidad para el Robot Continuum
    Objetivo: Calcular el área M00 de la pieza:
      - Si Área >= 25,000 px (Cerca): Recuadro VERDE y texto "LISTO PARA AGARRE".
      - Si Área < 25,000 px (Lejos/Ausente): Recuadro ROJO y texto "FUERA DE RANGO".
    """
    salida = frame.copy()

    # ------------------------------------------------------------------------
    # [CÓDIGO ALUMNOS - RETO 3]
    # Calcular momentos espaciales de la máscara y condicionar alertas gráficas
    # ------------------------------------------------------------------------

    return salida

# ============================================================================
# SERVIDOR WEB DE MONITOREO MULTIPANTALLA (PORT 5001)
# ============================================================================
app = Flask(__name__)

HTML_DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>MR3005C - Mini-Retos Sesión 2</title>
    <style>
        body { background:#0b1120; color:#f8fafc; font-family:sans-serif; text-align:center; margin:0; padding:15px; }
        .grid { display:grid; grid-template-columns:1fr 1fr; gap:15px; max-width:1100px; margin:auto; }
        .card { background:#1e293b; padding:10px; border-radius:8px; border:1px solid #334155; }
        img { width:100%; max-width:480px; border-radius:4px; }
        h3 { margin:6px 0; color:#38bdf8; font-size:15px; }
    </style>
</head>
<body>
    <h2>MR3005C: Evaluación de Mini-Retos en Vivo (Sesión 2)</h2>
    <div class="grid">
        <div class="card"><h3>1. Entrada Original (BGR)</h3><img src="/stream/orig"></div>
        <div class="card"><h3>2. Reto 1: Máscara Amarilla</h3><img src="/stream/r1"></div>
        <div class="card"><h3>3. Reto 2: Fondo sin Pieza</h3><img src="/stream/r2"></div>
        <div class="card"><h3>4. Reto 3: Disparador de Agarre (Min: 25k px)</h3><img src="/stream/r3"></div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_DASHBOARD)

@app.route('/stream/<modo>')
def stream(modo):
    def gen():
        while True:
            f = adquirir_frame()
            if f is None:
                continue
            hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)

            if modo == 'r1': out = reto_1_segmentar_amarillo(f, hsv)
            elif modo == 'r2': out = reto_2_mascara_inversa(f, hsv)
            elif modo == 'r3': out = reto_3_trigger_proximidad(f, hsv)
            else: out = f

            if len(out.shape) == 2:
                out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

            ok, buf = cv2.imencode('.jpg', out)
            if not ok:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("\nServidor de Mini-Retos activo en: http://localhost:5001\n")
    try:
        app.run(host='0.0.0.0', port=5001, threaded=True)
    finally:
        liberar_camara()
EOF
```

---

## Paso 4: Código Resuelto de Referencia (`s2_mini_retos_resuelto.py`)
Crea la versión con las soluciones completas para proyectar o consultar:

```bash
cat << 'EOF' > ~/realsense-segmentacion-hsv/s2_mini_retos_resuelto.py
#!/usr/bin/env python3
import os
import sys
import signal
import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response, render_template_string

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe.start(cfg)

def liberar_camara(sig=None, frame=None):
    try:
        pipe.stop()
    except Exception:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, liberar_camara)

def adquirir_frame():
    try:
        frames = pipe.wait_for_frames()
        c = frames.get_color_frame()
        return np.asanyarray(c.get_data()) if c else None
    except Exception:
        return None

KERNEL = np.ones((5, 5), np.uint8)

# Rango óptimo para amarillo (rueda de prueba)
LOWER_YELLOW = np.array([20, 100, 100])
UPPER_YELLOW = np.array([35, 255, 255])

# Umbral calibrado: sólo activa si la pieza está cerca de la cámara
UMBRAL_AGARRE = 25000

def reto_1_segmentar_amarillo(frame, hsv):
    mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

def reto_2_mascara_inversa(frame, hsv):
    mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    mask_inv = cv2.bitwise_not(mask)
    return cv2.bitwise_and(frame, frame, mask=mask_inv)

def reto_3_trigger_proximidad(frame, hsv):
    salida = frame.copy()
    mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)

    M = cv2.moments(mask)
    area = M["m00"]

    if area >= UMBRAL_AGARRE:
        # Condición CUMPLIDA: Objeto cerca de la cámara
        cv2.rectangle(salida, (20, 20), (620, 460), (0, 255, 0), 4)
        cv2.putText(salida, f"LISTO PARA AGARRE (Area: {int(area)} px)", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    else:
        # Condición NO CUMPLIDA: Objeto lejos o ausente
        cv2.rectangle(salida, (20, 20), (620, 460), (0, 0, 255), 2)
        cv2.putText(salida, f"FUERA DE RANGO (Area: {int(area)} px / Min: {UMBRAL_AGARRE})", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    return salida

app = Flask(__name__)

HTML_DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8"><title>Mini-Retos Resueltos</title>
    <style>
        body { background:#0b1120; color:#f8fafc; font-family:sans-serif; text-align:center; padding:15px; }
        .grid { display:grid; grid-template-columns:1fr 1fr; gap:15px; max-width:1100px; margin:auto; }
        .card { background:#1e293b; padding:10px; border-radius:8px; }
        img { width:100%; max-width:480px; border-radius:4px; }
    </style>
</head>
<body>
    <h2>MR3005C: Solución de Mini-Retos (Sesión 2)</h2>
    <div class="grid">
        <div class="card"><h3>1. Original BGR</h3><img src="/stream/orig"></div>
        <div class="card"><h3>2. Reto 1 Resuelto</h3><img src="/stream/r1"></div>
        <div class="card"><h3>3. Reto 2 Resuelto</h3><img src="/stream/r2"></div>
        <div class="card"><h3>4. Reto 3: Disparador Calibrado (25k px)</h3><img src="/stream/r3"></div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_DASHBOARD)

@app.route('/stream/<modo>')
def stream(modo):
    def gen():
        while True:
            f = adquirir_frame()
            if f is None:
                continue
            hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
            if modo == 'r1': out = reto_1_segmentar_amarillo(f, hsv)
            elif modo == 'r2': out = reto_2_mascara_inversa(f, hsv)
            elif modo == 'r3': out = reto_3_trigger_proximidad(f, hsv)
            else: out = f

            ok, buf = cv2.imencode('.jpg', out)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)
EOF
```

---



### Paso 5: Configurar permisos en Ubuntu
```bash
sudo chmod 666 /dev/video* 2>/dev/null
sudo chmod -R 777 /dev/bus/usb/ 2>/dev/null
lsusb
```
*Verifica que aparezca:* `Intel Corp. Intel(R) RealSense(TM) Depth Camera`.

---

## 3. Instalación de Dependencias

Con la terminal de Ubuntu abierta en la raíz de este proyecto:

```bash
# Crear entorno virtual (si no existe)
python3 -m venv ~/vision_env

# Activar entorno
source ~/vision_env/bin/activate

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Ejecución del Pipeline Principal

Para ejecutar el seguimiento en tiempo real y desplegar los 4 cuadrantes de inspección:

```bash
python s2_segmentacion_hsv.py
```

Abre en tu navegador en Windows (Chrome o Edge):
```text
http://localhost:5000
```

### Cuadrantes Desplegados:
| Cuadrante | Contenido | Descripción Técnica |
| :--- | :--- | :--- |
| **1. Superior Izq.** | BGR Original | Frame con punto de referencia del centro óptico $(320, 240)$. |
| **2. Superior Der.** | Espacio HSV | Representación con matiz ($H$) desacoplado de la intensidad ($V$). |
| **3. Inferior Izq.** | Máscara Binaria | Imagen binarizada tras apertura y clausura morfológica con kernel $5\times 5$. |
| **4. Inferior Der.** | Tracking Cinemático | Centroide verde $(\bar{x}, \bar{y})$ y vector azul de guiado hacia el centro óptico. |

* **Evidencia automática:** Cada ciclo guarda el último cuadro procesado en `capture/segmentacion_cajon_000.jpg`.
* **Detención:** Presiona `Ctrl + C` en la terminal.

---

## 5. Dinámica de Mini-Retos

Para la sesión práctica en parejas (15 minutos):

```bash
python s2_mini_retos_estudiantes.py
```

Abre en tu navegador en Windows:
```text
http://localhost:5001
```

Completa las funciones en `s2_mini_retos_estudiantes.py` sin utilizar bucles `for`:
1. **Reto 1 (`reto_1_segmentar_amarillo`):** Configurar rangos HSV para aislar objetos amarillos ($H \in [20, 35]$).
2. **Reto 2 (`reto_2_mascara_inversa`):** Aplicar `cv2.bitwise_not` para ocultar la pieza y mostrar el entorno.
3. **Reto 3 (`reto_3_trigger_proximidad`):** Evaluar el área $M_{00}$. Si es $\ge 15,000\text{ px}$, activar alerta de sujeción.

---

## 6. Cálculo del Vector de Error Cinemático

A partir de la máscara limpia $I(y, x)$, los momentos se definen como:
$$M_{pq} = \sum_x \sum_y x^p y^q I(y, x)$$

El centroide se obtiene dividiendo los momentos de primer orden entre el área:
$$\bar{x} = \frac{M_{10}}{M_{00}}, \quad \bar{y} = \frac{M_{01}}{M_{00}}$$

El vector de error respecto al centro del sensor $(u_c, v_c) = (320, 240)$ corresponde a:
$$e_u = \bar{x} - 320, \quad e_v = \bar{y} - 240$$

---

## 7. Solución de Problemas Frecuentes

* **`usbipd: error: There is no WSL 2 distribution running`:** Abre primero la terminal de Ubuntu y luego corre el comando `usbipd attach`.
* **`RuntimeError: No device connected`:** La cámara se desconectó físicamente. Vuelve a ejecutar `usbipd attach --wsl --busid <BUSID> --auto-attach` en PowerShell y renueva permisos en Ubuntu (`sudo chmod 666 /dev/video*`).
* **La terminal ejecuta Python de Windows en VS Code:** Abre la paleta de comandos (`Ctrl + Shift + P`), selecciona `Terminal: Select Default Profile` y elige `Ubuntu (WSL)`.
EOF
```

---

## Paso 6: Subir el Nuevo Repositorio a GitHub
1. Ve a GitHub desde tu navegador.
2. Nombra el repositorio: `realsense-segmentacion-hsv`.
3. Selecciona **Public** y deja todas las casillas de inicialización desmarcadas.
4. En tu terminal de Ubuntu ejecuta:

```bash
cd ~/realsense-segmentacion-hsv
git add .
git commit -m "feat: implementacion completa sesion 2 segmentacion hsv y tracking"
git remote add origin [https://github.com/GerardoEmirSanchez/realsense-segmentacion-hsv.git](https://github.com/GerardoEmirSanchez/realsense-segmentacion-hsv.git)
git push -u origin main
```
