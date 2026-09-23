# realsense-bordes-contornos

```text
realsense-bordes-contornos/
├── .gitignore
├── requirements.txt
├── README.md
├── s3_bordes_contornos.py
├── s3_mini_retos_estudiantes.py
└── s3_mini_retos_resuelto.py

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

3. Conectar el dispositivo a la instancia activa de WSL con persistencia:

```PowerShell
usbipd attach --wsl --busid <TU-BUSID> --auto-attach

```

---

## Paso 1: Creación del Directorio Local y Configuración de Git

Ejecuta en la terminal de Ubuntu (WSL2):

```bash
# 1. Crear y acceder a la carpeta del nuevo proyecto
mkdir -p ~/realsense-bordes-contornos
cd ~/realsense-bordes-contornos

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

## Paso 2: Script Principal de Clase (`s3_bordes_contornos.py`)

Genera el script del pipeline en vivo con mosaico de 4 cuadrantes en el puerto 5000:

```bash
cat << 'EOF' > ~/realsense-bordes-contornos/s3_bordes_contornos.py
#!/usr/bin/env python3
"""
==============================================================================
MR3005C: Sistemas Ciberfísicos — Módulo 8: Visión Artificial
Sesión 3: Filtros Espaciales, Bordes Canny y Extracción de Contornos
==============================================================================
"""

import os
import sys
import signal
import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response

app = Flask(__name__)
os.makedirs("capture", exist_ok=True)
RUTA_EVIDENCIA = "capture/edges_cajon_000.jpg"

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

def procesar_cuadrantes(frame):
    h, w, _ = frame.shape
    u_c, v_c = w // 2, h // 2  # Centro óptico (320, 240)

    # 1. Transformación a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Suavizado Gaussiano 2D (5x5, sigma=1.2)
    gauss = cv2.GaussianBlur(gray, (5, 5), 1.2)

    # 3. Detector de Bordes Canny (Histéresis 50 / 150)
    edges = cv2.Canny(gauss, 50, 150)

    # 4. Extracción vectorial de contornos jerárquicos
    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos_validos = [c for c in contornos if cv2.contourArea(c) > 600]

    # Cuadrante de Tracking y Descriptores de Forma
    tracking = frame.copy()
    cv2.circle(tracking, (u_c, v_c), 5, (255, 255, 255), -1)  # Centro óptico blanco

    if contornos_validos:
        c_max = max(contornos_validos, key=cv2.contourArea)
        area_c = cv2.contourArea(c_max)
        perimetro = cv2.arcLength(c_max, closed=True)

        # Bounding Box Recto Cartesiano (Rojo)
        xr, yr, wr, hr = cv2.boundingRect(c_max)
        cv2.rectangle(tracking, (xr, yr), (xr + wr, yr + hr), (0, 0, 255), 1)

        # Bounding Box Mínimo Orientado (Verde)
        rect_rotado = cv2.minAreaRect(c_max)
        (cx, cy), (w_box, h_box), angulo = rect_rotado
        puntos = cv2.boxPoints(rect_rotado)
        caja_orientada = np.intp(puntos)
        cv2.drawContours(tracking, [caja_orientada], 0, (0, 255, 0), 2)
        cv2.circle(tracking, (int(cx), int(cy)), 5, (0, 255, 0), -1)

        # Vector de alineación hacia el centro óptico
        cv2.line(tracking, (u_c, v_c), (int(cx), int(cy)), (255, 0, 0), 2)

        # Factor de forma (Compacidad / Circularidad)
        compacidad = (4 * np.pi * area_c) / (perimetro**2 + 1e-6)

        # Telemetría en pantalla
        cv2.putText(tracking, f"Theta: {angulo:.1f} deg", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(tracking, f"Area: {int(area_c)} px | Compacidad: {compacidad:.2f}", (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    else:
        cv2.putText(tracking, "BUSCANDO CONTORNO...", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Guardar último frame para entregable oficial A6
    cv2.imwrite(RUTA_EVIDENCIA, tracking)

    # 5. Construcción del Mosaico 2x2
    orig_anotado = frame.copy()
    cv2.circle(orig_anotado, (u_c, v_c), 5, (0, 0, 255), -1)
    cv2.putText(orig_anotado, "1. BGR Original", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    gauss_bgr = cv2.cvtColor(gauss, cv2.COLOR_GRAY2BGR)
    cv2.putText(gauss_bgr, "2. Filtro Gaussiano 5x5", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.putText(edges_bgr, "3. Bordes Canny (1 px)", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(tracking, "4. Bounding Box Orientado", (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    fila_arriba = np.hstack((orig_anotado, gauss_bgr))
    fila_abajo = np.hstack((edges_bgr, tracking))
    mosaico = np.vstack((fila_arriba, fila_abajo))

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
    print(" SESIÓN 3: PIPELINE DE BORDES CANNY Y CONTORNOS 2D")
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

## Paso 3: Plantilla para Estudiantes (`s3_mini_retos_estudiantes.py`)

Genera el script sin resolver con el dashboard en el puerto 5001:

```bash
cat << 'EOF' > ~/realsense-bordes-contornos/s3_mini_retos_estudiantes.py
#!/usr/bin/env python3
"""
==============================================================================
MR3005C: Sistemas Ciberfísicos — Módulo 8: Visión Artificial
Sesión 3: Mini-Retos Prácticos de Convolución, Canny y Orientación
==============================================================================

INSTRUCCIONES PARA EL EQUIPO:
1. Trabajar en parejas durante 12 minutos.
2. Completar las tres funciones marcadas:
      - reto_1_canny_calibrado(frame, gray)
      - reto_2_filtro_area(frame, edges)
      - reto_3_min_area_rect(frame, edges)
3. PROHIBIDO usar ciclos 'for'. Todo debe ser matricial o funciones vectoriales.
4. Para evaluar su avance en vivo:
      - Ejecuten: python s3_mini_retos_estudiantes.py
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

def capturar_frame():
    try:
        frames = pipe.wait_for_frames()
        c = frames.get_color_frame()
        return np.asanyarray(c.get_data()) if c else None
    except Exception:
        return None

# ============================================================================
# MINI-RETOS EN PAREJAS (EDITAR ÚNICAMENTE ESTA SECCIÓN)
# ============================================================================

def reto_1_canny_calibrado(frame, gray):
    """
    RETO 1: Detector Canny Calibrado con Filtro Gaussiano
    Objetivo: Aplicar cv2.GaussianBlur y luego cv2.Canny para obtener bordes
              limpios y continuos de la gaveta y la pieza.
    Retorno: Matriz de 3 canales BGR (cv2.cvtColor a BGR).
    """
    salida = frame.copy()

    # ------------------------------------------------------------------------
    # [CÓDIGO ALUMNOS - RETO 1]
    # Aplicar GaussianBlur y Canny sobre gray.
    # Convertir a 3 canales BGR para visualización.
    # ------------------------------------------------------------------------

    return salida


def reto_2_filtro_area(frame, edges):
    """
    RETO 2: Extracción y Filtrado de Contornos por Área
    Objetivo: Obtener contornos (cv2.findContours), descartar aquellos con
              área menor a 800 px y dibujar los válidos en AMARILLO.
    """
    salida = frame.copy()

    # ------------------------------------------------------------------------
    # [CÓDIGO ALUMNOS - RETO 2]
    # findContours sobre edges (RETR_EXTERNAL, CHAIN_APPROX_SIMPLE).
    # Filtrar contornos con cv2.contourArea > 800.
    # cv2.drawContours sobre salida en color amarillo (0, 255, 255).
    # ------------------------------------------------------------------------

    return salida


def reto_3_min_area_rect(frame, edges):
    """
    RETO 3: Bounding Box Mínimo Orientado y Ángulo
    Objetivo: Sobre el contorno mayor, calcular la caja rotada (cv2.minAreaRect).
              Dibujar el polígono en VERDE y mostrar el ángulo theta en grados.
    """
    salida = frame.copy()

    # ------------------------------------------------------------------------
    # [CÓDIGO ALUMNOS - RETO 3]
    # Localizar c_max mediante max(contornos, key=cv2.contourArea).
    # Calcular cv2.minAreaRect(c_max).
    # Convertir vértices con cv2.boxPoints y np.intp.
    # Dibujar polígono rotado y mostrar ángulo con cv2.putText.
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
    <title>MR3005C - Mini-Retos Sesión 3</title>
    <style>
        body { background:#0b1120; color:#f8fafc; font-family:sans-serif; text-align:center; margin:0; padding:15px; }
        .grid { display:grid; grid-template-columns:1fr 1fr; gap:15px; max-width:1100px; margin:auto; }
        .card { background:#1e293b; padding:10px; border-radius:8px; border:1px solid #334155; }
        img { width:100%; max-width:480px; border-radius:4px; }
        h3 { margin:6px 0; color:#38bdf8; font-size:15px; }
    </style>
</head>
<body>
    <h2>MR3005C: Evaluación de Mini-Retos en Vivo (Sesión 3)</h2>
    <div class="grid">
        <div class="card"><h3>1. Entrada Original (BGR)</h3><img src="/stream/orig"></div>
        <div class="card"><h3>2. Reto 1: Canny Calibrado</h3><img src="/stream/r1"></div>
        <div class="card"><h3>3. Reto 2: Contornos (>800 px)</h3><img src="/stream/r2"></div>
        <div class="card"><h3>4. Reto 3: BBox Orientado y Ángulo</h3><img src="/stream/r3"></div>
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
            f = capturar_frame()
            if f is None:
                continue

            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            gauss = cv2.GaussianBlur(gray, (5, 5), 1.2)
            edges = cv2.Canny(gauss, 50, 150)

            if modo == 'r1': out = reto_1_canny_calibrado(f, gray)
            elif modo == 'r2': out = reto_2_filtro_area(f, edges)
            elif modo == 'r3': out = reto_3_min_area_rect(f, edges)
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

## Paso 4: Código Resuelto de Referencia (`s3_mini_retos_resuelto.py`)

Crea la versión con las soluciones completas para proyectar o consultar:

```bash
cat << 'EOF' > ~/realsense-bordes-contornos/s3_mini_retos_resuelto.py
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

def capturar_frame():
    try:
        frames = pipe.wait_for_frames()
        c = frames.get_color_frame()
        return np.asanyarray(c.get_data()) if c else None
    except Exception:
        return None

def reto_1_canny_calibrado(frame, gray):
    gauss = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(gauss, 50, 150)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def reto_2_filtro_area(frame, edges):
    salida = frame.copy()
    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    validos = [c for c in contornos if cv2.contourArea(c) > 800]
    cv2.drawContours(salida, validos, -1, (0, 255, 255), 2)
    cv2.putText(salida, f"Contornos: {len(validos)}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return salida

def reto_3_min_area_rect(frame, edges):
    salida = frame.copy()
    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    validos = [c for c in contornos if cv2.contourArea(c) > 800]

    if validos:
        c_max = max(validos, key=cv2.contourArea)
        rect = cv2.minAreaRect(c_max)
        (cx, cy), (ancho, alto), angulo = rect
        puntos = cv2.boxPoints(rect)
        caja = np.intp(puntos)
        cv2.drawContours(salida, [caja], 0, (0, 255, 0), 2)
        cv2.circle(salida, (int(cx), int(cy)), 5, (0, 255, 0), -1)
        cv2.putText(salida, f"Theta: {angulo:.1f} deg", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(salida, "SIN OBJETO DOMINANTE", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return salida

app = Flask(__name__)

HTML_DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8"><title>Mini-Retos Resueltos Sesion 3</title>
    <style>
        body { background:#0b1120; color:#f8fafc; font-family:sans-serif; text-align:center; padding:15px; }
        .grid { display:grid; grid-template-columns:1fr 1fr; gap:15px; max-width:1100px; margin:auto; }
        .card { background:#1e293b; padding:10px; border-radius:8px; }
        img { width:100%; max-width:480px; border-radius:4px; }
    </style>
</head>
<body>
    <h2>MR3005C: Solución de Mini-Retos (Sesión 3)</h2>
    <div class="grid">
        <div class="card"><h3>1. Original BGR</h3><img src="/stream/orig"></div>
        <div class="card"><h3>2. Reto 1 Resuelto</h3><img src="/stream/r1"></div>
        <div class="card"><h3>3. Reto 2 Resuelto</h3><img src="/stream/r2"></div>
        <div class="card"><h3>4. Reto 3 Resuelto</h3><img src="/stream/r3"></div>
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
            f = capturar_frame()
            if f is None:
                continue
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            gauss = cv2.GaussianBlur(gray, (5, 5), 1.2)
            edges = cv2.Canny(gauss, 50, 150)

            if modo == 'r1': out = reto_1_canny_calibrado(f, gray)
            elif modo == 'r2': out = reto_2_filtro_area(f, edges)
            elif modo == 'r3': out = reto_3_min_area_rect(f, edges)
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
python s3_bordes_contornos.py

```

Abre en tu navegador en Windows (Chrome o Edge):

```text
http://localhost:5000

```

### Cuadrantes Desplegados:

| Cuadrante | Contenido | Descripción Técnica |
| --- | --- | --- |
| **1. Superior Izq.** | BGR Original | Frame con punto de referencia del centro óptico $(320, 240)$.

 |
| **2. Superior Der.** | Filtro Gaussiano | Reducción de ruido térmico preservando fronteras estructurales.

 |
| **3. Inferior Izq.** | Bordes Canny | Mapa binarizado de 1 px tras supresión de no máximos e histéresis.

 |
| **4. Inferior Der.** | BBox Orientado | Bounding box rotado verde con ángulo de giro $\theta$ para la garra del robot continuum.

 |

* **Evidencia automática:** Cada ciclo guarda el último cuadro procesado en `capture/edges_cajon_000.jpg`.


* **Detención:** Presiona `Ctrl + C` en la terminal.

---

## 5. Dinámica de Mini-Retos

Para la sesión práctica en parejas (12 minutos):

```bash
python s3_mini_retos_estudiantes.py

```

Abre en tu navegador en Windows:

```text
http://localhost:5001

```

Completa las funciones en `s3_mini_retos_estudiantes.py` sin utilizar bucles `for`:

1. **Reto 1 (`reto_1_canny_calibrado`):** Implementar `cv2.GaussianBlur` + `cv2.Canny` con umbrales calibrados.


2. **Reto 2 (`reto_2_filtro_area`):** Agrupar contornos con `cv2.findContours` y filtrar por área $> 800\text{ px}^2$.


3. **Reto 3 (`reto_3_min_area_rect`):** Extraer `cv2.minAreaRect(c_max)` y dibujar la caja orientada verde con su ángulo $\theta$.



---

## 6. Formulación de Descriptores Geométricos

A partir del contorno vectorial $C = \{(x_i, y_i)\}_{i=1}^N$:

### Área de Green:

$$A = \frac{1}{2} \sum_{i=1}^{N} (x_i y_{i+1} - x_{i+1} y_i)$$

### Factor de Forma (Compacidad):

$$\text{Compacidad} = \frac{4\pi A}{P^2}$$

* Un círculo perfecto tiene compacidad $\approx 1.0$.
* Formas alargadas o irregulares (herramientas) tienen compacidad $\ll 1.0$.

### Orientación Angular ($\theta$):

Dada la matriz de covarianza de los momentos centrales $\mu'_{20}, \mu'_{02}, \mu'_{11}$:


$$\theta = \frac{1}{2} \arctan\left(\frac{2\mu'_{11}}{\mu'_{20} - \mu'_{02}}\right)$$

---

## 7. Solución de Problemas Frecuentes

* **`usbipd: error: There is no WSL 2 distribution running`:** Abre primero la terminal de Ubuntu y luego corre el comando `usbipd attach`.
* **`RuntimeError: No device connected`:** La cámara se desconectó físicamente. Vuelve a ejecutar `usbipd attach --wsl --busid <BUSID> --auto-attach` en PowerShell y renueva permisos en Ubuntu (`sudo chmod 666 /dev/video*`).


* **La terminal ejecuta Python de Windows en VS Code:** Abre la paleta de comandos (`Ctrl + Shift + P`), selecciona `Terminal: Select Default Profile` y elige `Ubuntu (WSL)`.
* **Bordes rotos o discontinuos en Canny:** Incrementa ligeramente el tamaño del kernel Gaussiano o reduce $T_{\text{low}}$ para conectar contornos tenues mediante histéresis.



---

## Paso 6: Subir el Nuevo Repositorio a GitHub

1. Ve a GitHub desde tu navegador.
2. Nombra el repositorio: `realsense-bordes-contornos`.
3. Selecciona **Public** y deja todas las casillas de inicialización desmarcadas.
4. En tu terminal de Ubuntu ejecuta:

```bash
cd ~/realsense-bordes-contornos
git add .
git commit -m "feat: implementacion completa sesion 3 canny y contornos orientados"
git remote add origin https://github.com/GerardoEmirSanchez/realsense-bordes-contornos.git
git push -u origin main

```
