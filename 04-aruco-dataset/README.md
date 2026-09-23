# realsense-aruco-dataset

```text
realsense-aruco-dataset/
├── .gitignore
├── requirements.txt
├── README.md
├── generar_marcador.py
├── s4_aruco_pose.py
└── capture_dataset.py
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
3. Conectar el dispositivo a la instancia activa de WSL con persistencia:
```PowerShell
usbipd attach --wsl --busid <TU-BUSID> --auto-attach
```

---

## Paso 1: Creación del Directorio Local y Configuración de Git

Ejecuta en la terminal de **Ubuntu (WSL2)**:

```bash
# 1. Crear y acceder a la carpeta del proyecto
mkdir -p ~/realsense-aruco-dataset
cd ~/realsense-aruco-dataset

# 2. Inicializar repositorio Git
git init
git branch -M main

# 3. Crear archivo .gitignore
cat << 'EOF' > .gitignore
vision_env/
__pycache__/
*.pyc
capture/
*.jpg
*.png
.vscode/
EOF

# 4. Crear requirements.txt con versiones compatibles
cat << 'EOF' > requirements.txt
pyrealsense2
opencv-python
numpy
flask
EOF
```

---

## Paso 2: Generador de Marcadores Imprimibles (`generar_marcador.py`)

Genera las imágenes PNG de alta resolución del diccionario oficial `DICT_4X4_50` listas para imprimir a escala $5\text{ cm} \times 5\text{ cm}$ ($L = 0.05\text{ m}$) o visualizar en pantalla móvil:

```bash
cat << 'EOF' > ~/realsense-aruco-dataset/generar_marcador.py
#!/usr/bin/env python3
"""
==============================================================================
MR3005C: Generador de Marcadores ArUco (DICT_4X4_50)
Genera marcadores de 500x500 píxeles listos para imprimir a escala 5x5 cm.
==============================================================================
"""

import os
import cv2

os.makedirs("marcadores_imprimibles", exist_ok=True)

# 1. Cargar el diccionario estándar utilizado en el curso (DICT_4X4_50)
diccionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# 2. Generar marcadores del ID 0 al ID 3
TAMANO_PX = 500  # Resolución cuadrada con margen de seguridad
for marker_id in range(4):
    imagen_marcador = cv2.aruco.generateImageMarker(diccionario, marker_id, TAMANO_PX)
    ruta = f"marcadores_imprimibles/aruco_4x4_id{marker_id}.png"
    cv2.imwrite(ruta, imagen_marcador)
    print(f"[OK] Generado marcador ID {marker_id} en: {ruta}")

print("\nImprime los archivos a un tamaño físico de exactamente 5 cm x 5 cm (L = 0.05 m).")
EOF
```

---

## Paso 3: Script de Estimación de Pose 3D y Telemetría (`s4_aruco_pose.py`)

Implementa la API moderna `cv2.aruco.ArucoDetector`, el modelo de proyección Pinhole con la matriz intrínseca $K$ de la RealSense D455 a $640 \times 480$, el algoritmo Perspective-n-Point (`cv2.solvePnP`) y la transmisión en vivo por Flask en `http://localhost:5000`:

```bash
cat << 'EOF' > ~/realsense-aruco-dataset/s4_aruco_pose.py
#!/usr/bin/env python3
"""
#!/usr/bin/env python3
"""
#!/usr/bin/env python3
"""
==============================================================================
MR3005C: Sistemas Ciberfísicos — Módulo 8: Visión Artificial
Sesión 4: Percepción Espacial 3D y Estimación de Pose con Marcadores ArUco
==============================================================================
"""

import os
import sys
import signal
import traceback
import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response

app = Flask(__name__)

# ============================================================================
# 1. PARÁMETROS GEOMÉTRICOS E INTRÍNSECOS DE LA CÁMARA (MODELO PINHOLE)
# ============================================================================
L_REAL = 0.05  # 5 cm = 0.05 m

U_C = 320.0
V_C = 240.0
F_X = 615.0
F_Y = 615.0

K_MAT = np.array([
    [F_X,  0.0, U_C],
    [0.0,  F_Y, V_C],
    [0.0,  0.0, 1.0]
], dtype=np.float64)

# 5 coeficientes de distorsión estándar (k1, k2, p1, p2, k3)
DIST_COEFFS = np.zeros((5, 1), dtype=np.float64)

# Vértices del marcador en su sistema local (Z=0, centrado en el origen)
# Orden: Superior-Izq, Superior-Der, Inferior-Der, Inferior-Izq
PUNTOS_OBJETO_3D = np.array([
    [-L_REAL / 2.0,  L_REAL / 2.0, 0.0],
    [ L_REAL / 2.0,  L_REAL / 2.0, 0.0],
    [ L_REAL / 2.0, -L_REAL / 2.0, 0.0],
    [-L_REAL / 2.0, -L_REAL / 2.0, 0.0]
], dtype=np.float64)

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

# Warm-up de 10 cuadros para estabilizar exposición automática
for _ in range(10):
    pipe.wait_for_frames()

# ============================================================================
# 3. CONFIGURACIÓN DEL DETECTOR ARUCO
# ============================================================================
diccionario_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Compatibilidad de inicialización del detector entre OpenCV < 4.7 y >= 4.7
if hasattr(cv2.aruco, "ArucoDetector"):
    parametros_deteccion = cv2.aruco.DetectorParameters()
    detector_aruco = cv2.aruco.ArucoDetector(diccionario_aruco, parametros_deteccion)
else:
    detector_aruco = None
    parametros_deteccion = cv2.aruco.DetectorParameters_create()

def detectar_marcadores(gray):
    if detector_aruco is not None:
        return detector_aruco.detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, diccionario_aruco, parameters=parametros_deteccion)

def dibujar_ejes_compatibles(img, k, dist, rvec, tvec, longitud=0.035):
    """Función de respaldo compatible con cualquier versión de OpenCV."""
    try:
        cv2.drawFrameAxes(img, k, dist, rvec, tvec, longitud)
    except AttributeError:
        cv2.aruco.drawAxis(img, k, dist, rvec, tvec, longitud)

# ============================================================================
# 4. PIPELINE DE TELEMETRÍA 3D Y ESTIMACIÓN DE POSE
# ============================================================================
def procesar_telemetria_aruco(frame):
    anotado = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Centro óptico de referencia (c_x, c_y)
    cv2.circle(anotado, (int(U_C), int(V_C)), 5, (255, 255, 255), -1)
    cv2.rectangle(anotado, (int(U_C) - 35, int(V_C) - 35), (int(U_C) + 35, int(V_C) + 35), (255, 255, 255), 1)

    esquinas, ids, _ = detectar_marcadores(gray)

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(anotado, esquinas, ids)
        ids_planos = ids.ravel()

        for i in range(len(ids)):
            puntos_2d = esquinas[i].reshape((4, 2)).astype(np.float64)

            # Intentar primero con SOLVEPNP_IPPE_SQUARE (óptimo para ArUco); fallback a ITERATIVE
            flag_pnp = getattr(cv2, 'SOLVEPNP_IPPE_SQUARE', cv2.SOLVEPNP_ITERATIVE)
            exito, rvec, tvec = cv2.solvePnP(
                PUNTOS_OBJETO_3D,
                puntos_2d,
                K_MAT,
                DIST_COEFFS,
                flags=flag_pnp
            )

            if exito:
                dibujar_ejes_compatibles(anotado, K_MAT, DIST_COEFFS, rvec, tvec, 0.035)

                # Desglose seguro usando .ravel()
                t_vec_flat = tvec.ravel()
                x_cm = t_vec_flat[0] * 100.0
                y_cm = t_vec_flat[1] * 100.0
                z_cm = t_vec_flat[2] * 100.0

                # Centroide proyectado en la imagen
                u_m = int(puntos_2d[:, 0].mean())
                v_m = int(puntos_2d[:, 1].mean())

                cv2.line(anotado, (int(U_C), int(V_C)), (u_m, v_m), (255, 0, 0), 2)
                cv2.circle(anotado, (u_m, v_m), 4, (0, 255, 0), -1)

                alineado = (abs(x_cm) <= 3.0) and (abs(y_cm) <= 3.0) and (15.0 <= z_cm <= 25.0)
                color_estado = (0, 255, 0) if alineado else (0, 165, 255)
                texto_estado = "LISTO PARA INSERCION" if alineado else "ALINEANDO ROBOT"

                # Barra de telemetría superior
                cv2.rectangle(anotado, (10, 10), (630, 80), (15, 23, 42), -1)
                cv2.rectangle(anotado, (10, 10), (630, 80), (51, 65, 85), 1)

                id_actual = ids_planos[i]
                cv2.putText(anotado, f"ARUCO ID: {id_actual} | Z (Profundidad): {z_cm:.1f} cm", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(anotado, f"Offsets: X={x_cm:+.1f} cm, Y={y_cm:+.1f} cm | {texto_estado}", (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_estado, 2)
    else:
        cv2.putText(anotado, "BUSCANDO MARCADOR ARUCO EN GAVETA...", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    return anotado

def generar_streaming():
    while True:
        try:
            frames = pipe.wait_for_frames(timeout_ms=5000)
            color = frames.get_color_frame()
            if not color:
                continue

            frame = np.asanyarray(color.get_data())
            anotado = procesar_telemetria_aruco(frame)

            ok, buffer = cv2.imencode('.jpg', anotado)
            if not ok:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e:
            # Imprime el error real en la terminal en lugar de matar el stream silenciosamente
            print(f"[ERROR EN FRAME]: {e}")
            traceback.print_exc()
            continue

@app.route('/')
def feed():
    return Response(generar_streaming(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("\n=======================================================")
    print(" SESIÓN 4: PERCEPCIÓN 3D Y LOCALIZACIÓN ARUCO")
    print(" Monitor activo en: http://localhost:5000")
    print(" Presiona Ctrl + C para finalizar la ejecución limpiamente.")
    print("=======================================================\n")
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        liberar_recursos()

EOF
```

---

## Paso 4: Herramienta de Captura del Dataset (`capture_dataset.py`)

Herramienta oficial de recolección fotográfica para el Entregable A6. Se ejecuta en consola pasando el identificador del equipo (`python capture_dataset.py equipo_1`) y despliega un panel web en `http://localhost:5001` con guardado en disco de imágenes puras a $640 \times 480$:

```bash
cat << 'EOF' > ~/realsense-aruco-dataset/capture_dataset.py
#!/usr/bin/env python3
"""
==============================================================================
MR3005C: Herramienta de Captura de Dataset (Entregable A6)
Uso: python capture_dataset.py <nombre_equipo>
Ejemplo: python capture_dataset.py equipo_1
==============================================================================
"""

import os
import sys
import signal
import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response, render_template_string

# Nombre del equipo recibido por parámetro de consola
equipo = sys.argv[1] if len(sys.argv) > 1 else "equipo_demo"
carpeta_salida = f"capture/{equipo}"
os.makedirs(carpeta_salida, exist_ok=True)

# Conteo automático para no sobreescribir capturas previas
n_capturas = len([f for f in os.listdir(carpeta_salida) if f.endswith('.jpg')])

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe.start(cfg)

def liberar(sig=None, frame=None):
    try:
        pipe.stop()
    except Exception:
        pass
    print(f"\n[INFO] Sesión finalizada. Total de imágenes almacenadas en {carpeta_salida}: {n_capturas}")
    sys.exit(0)

signal.signal(signal.SIGINT, liberar)

app = Flask(__name__)
ultimo_frame_puro = None

HTML_DASHBOARD = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Captura de Dataset A6 - {equipo}</title>
    <style>
        body {{ background:#0b1120; color:#f8fafc; font-family:sans-serif; margin:0; padding:15px; display:flex; gap:20px; justify-content:center; }}
        .monitor {{ text-align:center; }}
        img {{ width:640px; border-radius:8px; border:2px solid #38bdf8; box-shadow:0 4px 15px rgba(0,0,0,0.5); }}
        .panel {{ background:#1e293b; padding:20px; border-radius:8px; border:1px solid #334155; width:340px; }}
        h2 {{ margin-top:0; color:#38bdf8; font-size:18px; border-bottom:1px solid #334155; padding-bottom:8px; }}
        .btn {{ background:#0284c7; color:#fff; padding:14px 20px; font-size:16px; font-weight:bold; border:none; border-radius:6px; cursor:pointer; width:100%; margin-top:10px; transition:0.2s; }}
        .btn:hover {{ background:#0369a1; }}
        .counter {{ font-size:22px; color:#4ade80; margin:15px 0; font-weight:bold; text-align:center; }}
        .protocol {{ font-size:12px; color:#94a3b8; line-height:1.5; }}
        .protocol b {{ color:#e2e8f0; }}
        ul {{ padding-left:18px; margin:6px 0; }}
    </style>
</head>
<body>
    <div class="monitor">
        <img src="/feed">
    </div>
    <div class="panel">
        <h2>Capturador: {equipo}</h2>
        <div class="counter" id="lbl_count">Imágenes: {n_capturas} / Meta: 60-80</div>
        <button class="btn" onclick="capturarFoto()">📸 CAPTURAR IMAGEN</button>
        
        <div class="protocol" style="margin-top:20px;">
            <b>Matriz de Variabilidad Obligatoria (A6 - 09 de Noviembre):</b>
            <ul>
                <li><b>Iluminación:</b> Lugar de trabajo con luz media y sombras.</li>
                <li><b>Distancia:</b> Cerca (20 cm), Media (35 cm), Lejos (50 cm).</li>
                <li><b>Ángulo:</b> Frontal (0°), lateral izquierdo (+30°), lateral derecho (-30°).</li>
                <li><b>Oclusión:</b> Fuera de la gaveta, 50% metida, 80% ocluida.</li>
                <li><b>Muestras Negativas:</b> 10 fotos del cajón vacío sin la pieza.</li>
            </ul>
        </div>
    </div>

    <script>
        function capturarFoto() {{
            fetch('/capturar')
                .then(r => r.text())
                .then(num => {{
                    document.getElementById('lbl_count').innerText = "Imágenes: " + num + " / Meta: 60-80";
                }});
        }}
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_DASHBOARD)

@app.route('/capturar')
def capturar():
    global n_capturas, ultimo_frame_puro
    if ultimo_frame_puro is not None:
        ruta_archivo = f"{carpeta_salida}/{equipo}_{n_capturas:03d}.jpg"
        # Guardar frame nativo limpio (sin anotaciones gráficas)
        cv2.imwrite(ruta_archivo, ultimo_frame_puro)
        n_capturas += 1
        print(f"[DATASET A6] Foto guardada: {ruta_archivo}")
    return str(n_capturas)

@app.route('/feed')
def feed():
    def generador():
        global ultimo_frame_puro
        while True:
            frames = pipe.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue

            frame = np.asanyarray(color.get_data())
            ultimo_frame_puro = frame.copy()

            # Feedback visual con conteo actual
            anotado = frame.copy()
            cv2.putText(anotado, f"{equipo} | Captura #{n_capturas}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            ok, buffer = cv2.imencode('.jpg', anotado)
            if not ok:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(generador(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print(f"\n=======================================================")
    print(f" HERRAMIENTA DE CAPTURA A6 - {equipo}")
    print(f" Servidor activo en: http://localhost:5001")
    print(f" Carpeta destino: {carpeta_salida}/")
    print(f" Presiona Ctrl + C para terminar y consolidar el conteo.")
    print(f"=======================================================\n")
    try:
        app.run(host='0.0.0.0', port=5001, threaded=True)
    finally:
        liberar()
EOF
```

---

## Paso 5: Configurar Permisos y Dependencias en Ubuntu (WSL2)

Ejecuta en la terminal de **Ubuntu**:

```bash
# 1. Configurar permisos de acceso al dispositivo USB
sudo chmod 666 /dev/video* 2>/dev/null
sudo chmod -R 777 /dev/bus/usb/ 2>/dev/null
lsusb
```
*Verifica que aparezca:* `Intel Corp. Intel(R) RealSense(TM) Depth Camera`.

```bash
# 2. Activar entorno virtual e instalar requerimientos
source ~/vision_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Paso 6: Generar los Marcadores Físicos

Ejecuta el script para generar los marcadores ArUco del diccionario `DICT_4X4_50`:

```bash
python generar_marcador.py
```
Abre la carpeta resultante directamente en el explorador de Windows:
```bash
explorer.exe marcadores_imprimibles
```
*Abre `aruco_4x4_id0.png` en pantalla o imprímelo a escala $5\text{ cm} \times 5\text{ cm}$ ($L = 0.05\text{ m}$) para colocarlo en la gaveta.*

---

## Paso 7: Ejecución y Validación

### 1. Lanzamiento del Nodo de Telemetría 3D en Ubuntu (WSL2):
```bash
cd ~/realsense-aruco-dataset
source ~/vision_env/bin/activate
python s4_aruco_pose.py
```

### 2. Monitoreo en el Navegador de Windows:
```text
http://localhost:5000
```
* **Elementos Visualizados en Tiempo Real a 30 FPS:**
  * **Contorno verde:** Borde del marcador detectado con su número de ID.
  * **Ejes 3D en Realidad Aumentada:** Eje $X$ en **Rojo**, Eje $Y$ en **Verde**, Eje $Z$ en **Azul**.
  * **Barra de Telemetría:** Reporta $Z$ (Profundidad): `XX.X cm`, offsets laterales y el estado de inserción.
* Presiona `Ctrl + C` en la terminal para detener el proceso.

### 3. Ejecutar la Herramienta de Captura del Dataset A6:
Pasa como argumento el identificador del equipo:
```bash
python capture_dataset.py equipo_1
```
Abre en el navegador de Windows:
```text
http://localhost:5001
```
* Haz clic en el botón **📸 CAPTURAR IMAGEN** mientras mueves la cámara y la pieza siguiendo la matriz de variabilidad obligatoria.
* Al presionar `Ctrl + C`, las fotos quedan guardadas en `capture/equipo_1/`.

---

## Paso 8: `README.md` del Repositorio

Genera la documentación completa ejecutando este bloque:

```bash
cat << 'EOF' > ~/realsense-aruco-dataset/README.md
# MR3005C: Marcadores Fiduciarios ArUco, Pose 3D y Captura de Dataset A6

Sistema de localización espacial tridimensional y recolección controlada de imágenes para robótica colaborativa utilizando cámaras Intel RealSense (serie D400), OpenCV y servidores locales en Flask sobre WSL2 (Ubuntu).

---

## Índice de Contenidos
1. [Fundamento Matemático del Modelo Pinhole y ArUco](#fundamento-matemático)
2. [Protocolo de Enlace USB tras Reinicio](#protocolo-de-enlace-usb)
3. [Instalación de Dependencias](#instalación-de-dependencias)
4. [Generación de Marcadores Imprimibles](#generación-de-marcadores)
5. [Telemetría 3D y Estimación de Pose (`s4_aruco_pose.py`)](#telemetría-3d-y-pose)
6. [Protocolo de Captura para el Entregable A6 (`capture_dataset.py`)](#captura-dataset-a6)
7. [Solución de Problemas Frecuentes](#solución-de-problemas)

---

## 1. Fundamento Matemático

### Matriz Intrínseca de la Cámara ($K$):
La relación entre un punto en el espacio 3D métrico $\mathbf{P} = [X, Y, Z]^T$ y su proyección en píxeles $\mathbf{p} = [u, v]^T$ está gobernada por:
$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \frac{1}{Z} \mathbf{K} \begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = \frac{1}{Z} \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}$$

Para la cámara RealSense D455 a $640 \times 480$:
* Focal en píxeles: $f_x \approx 615\text{ px}, f_y \approx 615\text{ px}$.
* Centro óptico del sensor: $(c_x, c_y) = (320, 240)$.

### Algoritmo Perspective-n-Point (SolvePnP):
Al conocer las dimensiones reales del marcador cuadrado ($L = 0.05\text{ m}$), OpenCV resuelve la transformación euclidiana $[\mathbf{R} \mid \mathbf{t}]$:
* $\mathbf{t} = [X, Y, Z]^T$: Posición métrica en metros respecto al lente de la cámara ($Z$ representa la profundidad frontal directa).
* $\mathbf{r}$: Vector de rotación en formalismo de Rodrigues ($3 \times 1$).

---

## 2. Protocolo de Enlace USB

Al reiniciar Windows o reconectar la cámara:
1. Abre la terminal de **Ubuntu** en Windows.
2. En **PowerShell (Administrador)** ejecuta:
   ```powershell
   usbipd attach --wsl --busid <TU-BUSID> --auto-attach
   ```
3. En Ubuntu renueva permisos:
   ```bash
   sudo chmod 666 /dev/video* 2>/dev/null
   sudo chmod -R 777 /dev/bus/usb/ 2>/dev/null
   ```

---

## 3. Instalación de Dependencias

```bash
# Crear y activar entorno virtual
python3 -m venv ~/vision_env
source ~/vision_env/bin/activate

# Instalar librerías
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Generación de Marcadores

Para crear los archivos PNG del diccionario `DICT_4X4_50`:
```bash
python generar_marcador.py
```
Los archivos se guardarán en `marcadores_imprimibles/`. Imprime o despliega el marcador con lado exacto de $5\text{ cm}$.

---

## 5. Telemetría 3D y Pose

Para iniciar el streaming de pose y guiado cinemático:
```bash
python s4_aruco_pose.py
```
Abre en el navegador en Windows:
```text
http://localhost:5000
```
* **Ejes 3D:** Rojo ($X$), Verde ($Y$), Azul ($Z$).
* **Condición de Inserción:** Se activa en verde cuando $\vert{}X\vert{} \le 3\text{ cm}$, $\vert{}Y\vert{} \le 3\text{ cm}$ y la profundidad $Z$ se ubica entre $15$ y $25\text{ cm}$.

---

## 6. Captura Dataset A6 (Fecha de Entrega: 09 de Noviembre)

Para capturar las imágenes de entrenamiento para YOLOv8:
```bash
python capture_dataset.py <nombre_equipo>
```
Abre en el navegador en Windows:
```text
http://localhost:5001
```
Cumple rigurosamente con la **Matriz de Variabilidad**:
* **60 a 80 imágenes totales** por equipo.
* Variar iluminación (lugar de trabajo con luz media y sombras), distancia ($20$, $35$, $50\text{ cm}$) y rotación ($0^\circ, \pm 30^\circ$).
* Incluir piezas con oclusión parcial dentro del cajón (fuera, 50% metida, 80% ocluida).
* **Obligatorio:** 10 imágenes del cajón vacío sin la pieza (muestras negativas).

---

## 7. Solución de Problemas

* **`cv2.aruco.ArucoDetector` no existe:** Actualiza OpenCV a versión 4.7 o superior con `pip install --upgrade opencv-python`.
* **La distancia $Z$ calculada no coincide con la realidad:** Verifica que el marcador físico mida exactamente $5\text{ cm}$ de lado ($L = 0.05\text{ m}$) y que el foco no esté distorsionado.
* **`No device connected`:** Vuelve a vincular con `usbipd attach` en PowerShell y confirma con `lsusb` en Ubuntu.
EOF
```

---

## Paso 9: Subir el Nuevo Repositorio a GitHub

1. Ve a [GitHub](https://github.com/new).
2. Nombra el repositorio: **`realsense-aruco-dataset`**.
3. Selecciona **Public** y deja las casillas de inicialización desmarcadas.
4. En tu terminal de Ubuntu ejecuta:

```bash
cd ~/realsense-aruco-dataset
git add .
git commit -m "feat: implementacion completa sesion 4 aruco 3d y herramienta de captura a6"
git remote add origin [https://github.com/GerardoEmirSanchez/realsense-aruco-dataset.git](https://github.com/GerardoEmirSanchez/realsense-aruco-dataset.git)
git push -u origin main
```
