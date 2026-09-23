# Guía de Instalación, Configuración y Streaming de Intel RealSense en WSL2

Guía paso a paso para limpiar el entorno, reiniciar a cero y replicar todo el proceso de instalación y configuración de cámaras Intel RealSense (serie D400) en WSL2 con visualización web local.

---

## Índice de Contenidos
1. [Fase 0: Limpieza total para prueba desde cero](#fase-0-limpieza-total-para-prueba-desde-cero)
2. [Fase 1: Configuración en Windows (Host)](#fase-1-configuración-en-windows-host)
3. [Fase 2: Configuración en Ubuntu (WSL2)](#fase-2-configuración-en-ubuntu-wsl2)
4. [Fase 3: Scripts de Captura y Streaming](#fase-3-scripts)
5. [Fase 4: Ejecución y Validación](#fase-4-ejecución-y-validación)
6. [Fase 5: Protocolo de Arranque tras Reinicio y Reconexión Rápida](#fase-5-protocolo-de-arranque-tras-reinicio-y-reconexión-rápida)
7. [Fase 6: Solución a Problemas Frecuentes (Troubleshooting)](#fase-6-solución-a-problemas-frecuentes-troubleshooting)

---


## Fase 0: Limpieza total para prueba desde cero
Pasos para limpiar el entorno de trabajo actual y regresar al estado inicial sin reinstalar Ubuntu.

### 1. En la terminal de Ubuntu (WSL)
```bash
# Desactivar entorno si está activo
deactivate 2>/dev/null

# Detener cualquier proceso de Python residual
killall -9 python python3 2>/dev/null

# Eliminar el entorno virtual, scripts y capturas generadas
cd ~
rm -rf vision_env capture
rm -f s1_realsense_pixeles.py visor_web_camara.py
```
### 2. En Windows PowerShell (Administrador)

```PowerShell
# Apagar WSL por completo para liberar hardware y memoria
wsl --shutdown
```
## Fase 1: Configuración en Windows (Host)

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

## Fase 2: Configuración en Ubuntu (WSL2)

Toca la tecla `Windows` y escribe `Ubuntu` y ábrelo como **administrador**

En la terminal de **Ubuntu**:
### 1. Dependencias del sistema y soporte USB
```Bash
sudo apt update && sudo apt install -y usbutils python3-pip python3-venv libgl1 libglib2.0-0
```
### 2. Verificar detección de hardware

```Bash
lsusb
```
_Debe listar: `Intel Corp. Intel(R) RealSense(TM) Depth Camera`._
### 3. Permisos de hardware
```Bash
sudo usermod -aG video,plugdev $USER
sudo chmod 666 /dev/video*
sudo chmod -R 777 /dev/bus/usb/
```
### 4. Entorno virtual de Python
```Bash
# Crear entorno virtual limpio
python3 -m venv ~/vision_env

# Activar el entorno virtual
source ~/vision_env/bin/activate

# Actualizar pip e instalar librerías necesarias
pip install --upgrade pip
pip install opencv-python numpy pyrealsense2 flask
```
## Fase 3: Scripts 

### Captura y Streaming (`visor_web_camara.py`)

Con el entorno virtual activo (`vision_env`), genera el script ejecutando el siguiente bloque en la terminal:

```Bash
cat << 'EOF' > ~/visor_web_camara.py
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
EOF
```



### Código_01 (`s1_realsense_pixeles.py`)

Con el entorno virtual activo (`vision_env`), genera el script ejecutando el siguiente bloque en la terminal:

```Bash
cat << 'EOF' > ~/s1_realsense_pixeles.py
from flask import Flask, Response

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

EOF
```














### Miniretos (`s1_mini_retos_estudiantes.py`)

Con el entorno virtual activo (`vision_env`), genera el script ejecutando el siguiente bloque en la terminal:

```Bash
cat << 'EOF' > ~/s1_mini_retos_estudiantes.py
from flask import Flask, Response

#!/usr/bin/env python3
"""
==============================================================================
MR3005C: Sistemas Ciberfísicos — Módulo 8: Visión Artificial
Sesión 1: Fundamentos de Imágenes Digitales — Mini-Retos Prácticos en Parejas
==============================================================================

INSTRUCCIONES PARA EL ESTUDIANTE:
---------------------------------
1. Trabajar en parejas (o ternas) durante 15 minutos.
2. Completar el algoritmo matricial dentro de las tres funciones:
      - reto_1_filtrado_canal_azul(frame)
      - reto_2_inversion_negativo(frame)
      - reto_3_punto_mas_luminoso(frame)
3. REGLA ESTRICTA DE LA PRÁCTICA:
      - Queda PROHIBIDO el uso de bucles iterativos ('for', 'while').
      - Todo el procesamiento debe ser puramente vectorial usando NumPy.
4. Para evaluar su avance en tiempo real:
      - Ejecuten este script: python s1_mini_retos_estudiantes.py
      - Abran en su navegador en Windows: http://localhost:5001
"""

import os
import sys
import signal
import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response, render_template_string

# ============================================================================
# 1. GESTIÓN Y ADQUISICIÓN DE HARDWARE (INTEL REALSENSE)
# ============================================================================
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe.start(cfg)

def liberar_camara(sig=None, frame=None):
    """Cierra limpiamente la conexión con el hardware USB al salir."""
    try:
        pipe.stop()
    except Exception:
        pass
    print("\n[INFO] Cámara liberada. Sesión finalizada.")
    sys.exit(0)

signal.signal(signal.SIGINT, liberar_camara)

def capturar_frame():
    """Adquiere el cuadro actual de la cámara como ndarray (H, W, C)."""
    try:
        frames = pipe.wait_for_frames()
        color = frames.get_color_frame()
        return np.asanyarray(color.get_data()) if color else None
    except Exception:
        return None


# ============================================================================
# 2. SECCIÓN DE MINI-RETOS (CÓDIGO A DESARROLLAR POR EL EQUIPO)
# ============================================================================

def reto_1_filtrado_canal_azul(frame):
    """
    DESAFÍO 1: Filtrado Espectral de Canal
    --------------------------------------
    Objetivo:
        Anular completamente la contribución de la longitud de onda corta (Azul)
        en toda la escena, manteniendo intactos los otros dos canales.
        
    Entrada:
        frame: Matriz NumPy tridimensional (480, 640, 3) de tipo uint8 en orden BGR.
        
    Salida:
        Matriz con las mismas dimensiones donde la información azul sea nula.
        
    Pregunta de análisis:
        ¿Hacia qué tonalidad perceptiva vira la imagen resultante y qué fenómeno
        óptico/aditivo fundamenta ese cambio?
    """
    salida = frame.copy()
    
    # ------------------------------------------------------------------------
    # [DESARROLLO ALUMNOS - RETO 1]
    # Aplicar indexación y slicing sobre el tensor para anular el canal adecuado.
    # ------------------------------------------------------------------------

    return salida


def reto_2_inversion_negativo(frame):
    """
    DESAFÍO 2: Inversión Fotográfica Lineal (Negativo)
    --------------------------------------------------
    Objetivo:
        Calcular la transformación inversa completa de la imagen digital sin
        iterar píxel por píxel. Los valores máximos (blancos: 255) deben 
        convertirse en mínimos (negros: 0) y viceversa, manteniendo la linealidad.
        
    Entrada:
        frame: Matriz NumPy tridimensional (480, 640, 3) de tipo uint8.
        
    Salida:
        Matriz procesada de tipo uint8 con el negativo fotográfico de la escena.
        
    Restricción:
        Resolver exclusivamente mediante álgebra de arreglos (broadcasting nativo).
    """
    salida = frame.copy()

    # ------------------------------------------------------------------------
    # [DESARROLLO ALUMNOS - RETO 2]
    # Implementar la ecuación lineal de inversión elemento a elemento.
    # ------------------------------------------------------------------------

    return salida


def reto_3_punto_mas_luminoso(frame):
    """
    DESAFÍO 3: Localización y Marcado del Extremo de Luminancia
    ----------------------------------------------------------
    Objetivo:
        Determinar la posición espacial exacta del píxel más brillante de toda
        la imagen y destacar su ubicación visualmente sobre el cuadro a color.
        
    Procedimiento requerido:
        1. Convertir el frame a un mapa bidimensional de luminancia (escala de grises).
        2. Localizar el índice lineal del valor máximo en la matriz.
        3. Traducir el índice lineal a coordenadas espaciales bidimensionales.
        4. Dibujar un marcador circular rojo (radio = 8 px, sólido) centrado 
           en dicho punto sobre el frame a color original.
           
    Entrada:
        frame: Matriz NumPy tridimensional (480, 640, 3) de tipo uint8.
        
    Salida:
        Matriz a color con el marcador circular rojo superpuesto en la posición máxima.
        
    Punto crítico a cuidar:
        Tener presente la diferencia entre la indexación de matrices [fila, columna] 
        frente al sistema de dibujo de primitivas gráficas en OpenCV (x, y).
    """
    salida = frame.copy()

    # ------------------------------------------------------------------------
    # [DESARROLLO ALUMNOS - RETO 3]
    # Calcular luminancia, encontrar coordenadas máximas y dibujar marcador.
    # ------------------------------------------------------------------------

    return salida


# ============================================================================
# 3. INTERFAZ DE MONITOREO Y EVALUACIÓN MULTIPANTALLA (FLASK)
# ============================================================================
app = Flask(__name__)

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>MR3005C - Panel de Evaluación de Mini-Retos</title>
    <style>
        body { 
            background-color: #0b1120; 
            color: #f8fafc; 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
            margin: 0; 
            padding: 20px; 
            text-align: center; 
        }
        h1 { margin: 0 0 6px 0; font-size: 24px; color: #38bdf8; }
        p { margin: 0 0 20px 0; font-size: 14px; color: #94a3b8; }
        .grid { 
            display: grid; 
            grid-template-columns: repeat(2, 1fr); 
            gap: 18px; 
            max-width: 1100px; 
            margin: 0 auto; 
        }
        .card { 
            background-color: #1e293b; 
            border: 1px solid #334155; 
            border-radius: 10px; 
            padding: 12px; 
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.4); 
        }
        .card h3 { 
            margin: 0 0 10px 0; 
            font-size: 15px; 
            color: #e2e8f0; 
            display: flex; 
            justify-content: space-between; 
            padding: 0 5px; 
        }
        .card span.tag { 
            font-size: 11px; 
            padding: 2px 8px; 
            border-radius: 4px; 
            background: #0369a1; 
            color: #e0f2fe; 
        }
        .card img { 
            width: 100%; 
            max-width: 500px; 
            border-radius: 6px; 
            border: 1px solid #475569; 
            display: block; 
            margin: 0 auto; 
        }
    </style>
</head>
<body>
    <h1>MR3005C: Sistemas Ciberfísicos — Módulo 8</h1>
    <p>Sesión 1: Evaluación de Mini-Retos en Parejas (Prohibido el uso de ciclos 'for')</p>
    
    <div class="grid">
        <div class="card">
            <h3><span>1. Entrada de Sensor</span><span class="tag">Original BGR</span></h3>
            <img src="/feed/original" alt="Entrada Original">
        </div>
        <div class="card">
            <h3><span>2. Reto 1</span><span class="tag">Filtrado Canal Azul</span></h3>
            <img src="/feed/reto1" alt="Reto 1">
        </div>
        <div class="card">
            <h3><span>3. Reto 2</span><span class="tag">Inversión Fotográfica</span></h3>
            <img src="/feed/reto2" alt="Reto 2">
        </div>
        <div class="card">
            <h3><span>4. Reto 3</span><span class="tag">Punto Más Luminoso</span></h3>
            <img src="/feed/reto3" alt="Reto 3">
        </div>
    </div>
</body>
</html>
"""

def codificar_a_jpg(matriz):
    ok, buffer = cv2.imencode('.jpg', matriz, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buffer.tobytes() if ok else b''

@app.route('/')
def panel_principal():
    return render_template_string(DASHBOARD_HTML)

@app.route('/feed/<modo>')
def transmision(modo):
    def generador():
        while True:
            frame = capturar_frame()
            if frame is None:
                continue

            if modo == 'reto1':
                salida = reto_1_filtrado_canal_azul(frame)
            elif modo == 'reto2':
                salida = reto_2_inversion_negativo(frame)
            elif modo == 'reto3':
                salida = reto_3_punto_mas_luminoso(frame)
            else:
                salida = frame

            jpg_bytes = codificar_a_jpg(salida)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')

    return Response(generador(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("\n" + "="*65)
    print(" SERVIDOR DE MINI-RETOS EN VIVO (ESTUDIANTES)")
    print(" Abre tu navegador en Windows (Chrome o Edge) e ingresa a:")
    print(" http://localhost:5001")
    print(" Para detener el servidor presiona Ctrl + C en esta terminal.")
    print("="*65 + "\n")
    try:
        app.run(host='0.0.0.0', port=5001, threaded=True)
    finally:
        liberar_camara()

EOF
```

















## Fase 4: Ejecución y Validación

### 1. Iniciar servidor
```Bash
python ~/visor_web_camara.py
```
En la terminal debe verse el mensaje:
`* Running on http://127.0.0.1:5000`
### 2. Visualización en navegador
Abre Chrome, Edge o Firefox en Windows e ingresa a:
```Plaintext
http://localhost:5000
```
### 3. Verificar captura en Windows
Detén el servidor con `Ctrl + C` y abre la carpeta de capturas:
```Bash
explorer.exe capture
```
Abre `captura_cajon_000.jpg` con el visor de imágenes de Windows y confirma que mide $640 \times 480$ píxeles.

## Fase 5: Protocolo de Arranque tras Reinicio y Reconexión Rápida

Cada vez que la computadora se reinicie o se vuelva a conectar el hardware en el laboratorio, se debe seguir un orden estricto de dos pasos:
### 1. Iniciar primero la máquina virtual Ubuntu
Al reiniciar Windows, WSL2 se encuentra apagado. Para que el puente USB funcione, primero abre la terminal de **Ubuntu** desde el menú Inicio o dentro de VS Code y déjala abierta:
```Bash
# Verificar prompt activo
whoami
```
### 2. Enlazar la cámara con persistencia automática

Con la terminal de Ubuntu abierta en pantalla, abre **PowerShell como Administrador** en Windows y ejecuta:

```PowerShell
usbipd list
```
```PowerShell   
usbipd attach --wsl --busid <TU-BUSID> --auto-attach
```

> **Nota:** El parámetro `--auto-attach` mantiene el servicio escuchando en segundo plano. Si el cable se desconecta accidentalmente o se mueve durante la práctica, se reconectará automáticamente a Ubuntu sin tener que reescribir el comando.

### 3. Renovar permisos y lanzar el entorno (En Ubuntu)
Al reconectar la cámara, Linux crea nodos de hardware nuevos en `/dev/video*`. Ejecuta siempre este bloque antes de iniciar el código:
```Bash
source ~/vision_env/bin/activate
sudo chmod 666 /dev/video* 2>/dev/null
sudo chmod -R 777 /dev/bus/usb/ 2>/dev/null
```
Codigo:
```
python ~/visor_web_camara.py
```
```
python ~/s1_realsense_pixeles.py
```
```
python ~/s1_mini_retos_estudiantes.py
```

## Fase 6: Solución a Problemas Frecuentes (Troubleshooting)

### Error: `usbipd: error: There is no WSL 2 distribution running`

- **Causa:** Se intentó ejecutar `usbipd attach` en PowerShell antes de abrir Ubuntu. `usbipd` no puede conectar un dispositivo a una máquina virtual inactiva.
- **Solución:** Abre la terminal de **Ubuntu** en Windows primero, déjala abierta y después vuelve a ejecutar `usbipd attach --wsl --busid <TU-BUSID> --auto-attach` en PowerShell.

### Error: `RuntimeError: No device connected` en Python

- **Causa 1:** La cámara se desconectó físicamente y el enlace con WSL se rompió.
- **Causa 2:** El script se ejecutó en una terminal de Windows PowerShell en lugar de la terminal de Ubuntu (WSL).
- **Solución:** Confirma que el prompt sea `(vision_env) usuario@...:~$`, ejecuta `lsusb` para verificar que la cámara esté presente en Linux y corre nuevamente:
```PowerShell
  usbipd attach --wsl --busid <TU-BUSID> --auto-attach
```
### Error: `cv2.error: (-5:Bad argument) in function 'add'`

- **Causa:** `cv2.add()` espera arreglos de NumPy (`np.ndarray`) y falla si recibe escalares aislados de tipo `np.uint8`.
- **Solución:** Envolver los escalares en matrices unidimensionales:

```Python
  val_a = np.array([200], dtype=np.uint8)
  val_b = np.array([100], dtype=np.uint8)
  resultado = cv2.add(val_a, val_b)
  
```

### Ventana de OpenCV congelada o con `[Warn: Copy Mode]`
- **Causa:** Conflicto del backend Qt/X11 de OpenCV con el compositor gráfico WSLg de Windows.
- **Solución:** Descartar `cv2.imshow()` en WSL2 y desplegar el stream directamente a través del servidor local Flask en `http://localhost:5000`.
