# Guía Práctica: Generación de Dataset Propio, Etiquetado en Roboflow y Transfer Learning con YOLOv8


**Sesión:** Captura de Datos con RealSense, Etiquetado Normalizado y Fine-Tuning para Detección de Pieza Objetivo

**Fechas Clave:**

- **Entregable A6 (Generación de Dataset de Visión):** 9 de noviembre
    
- **Entregable A7 (Modelo Entrenado y Validación** $\text{mAP}_{50} \ge 0.85$**):** 12 de noviembre
    
    **Entorno de Ejecución:** Ubuntu en WSL2 con cámara Intel RealSense D435i/D455 sobre servidor Flask (`http://localhost:5000`), Roboflow y Google Colab / Local.
    

## Estructura Final del Flujo de Trabajo

Al completar todas las fases de esta guía, tu directorio de proyecto `~/realsense-yolo-inferencia/` tendrá la siguiente arquitectura:

```
realsense-yolo-inferencia/
├── dataset_raw/                 # [A6] Fotogramas vírgenes capturados con la RealSense (60-80 fotos)
├── dataset_yolo/                # [A6] Dataset exportado de Roboflow (train, valid, test) + data.yaml
├── s6_captura_dataset.py        # [A6] Capturador web interactivo de fotogramas crudos
├── s6_entrenar_yolo.py          # [A7] Script de Fine-Tuning a partir de yolov8n.pt
├── s6_validar_modelo.py         # [A7] Auditoría matemática de IoU, Precision, Recall y mAP50
├── best.pt                      # [A7] Pesos sinápticos finales listos para el robot continuum
└── s5_inferencia_yolo.py        # [A7] Script de inferencia en vivo (carga best.pt automáticamente)
```

## Fase 0: Limpieza de Entorno Previo (Reinicio en Limpio)

Si vienes de prácticas anteriores y deseas asegurar que ningún proceso residual bloquee la cámara Intel RealSense ni existan conflictos de librerías, ejecuta este protocolo de limpieza. _(Si ya tienes el repositorio de la Sesión 5 funcionando y solo deseas agregar los scripts nuevos, puedes saltar directamente a la **Fase 2, punto 3**)._

### 1. En la terminal de Ubuntu (WSL)

```
# Desactivar entorno si está activo
deactivate 2>/dev/null

# Detener cualquier proceso de Python residual que retenga el puerto 5000 o el USB
killall -9 python python3 2>/dev/null

# Eliminar el entorno virtual, scripts y capturas generadas anteriormente
cd ~
rm -rf vision_env capture
rm -f s1_realsense_pixeles.py visor_web_camara.py
```

### 2. En Windows PowerShell (Administrador)

```
# Apagar WSL por completo para liberar hardware y memoria
wsl --shutdown
```

## Fase 1: Configuración en Windows (Host)

### 1. Instalar `usbipd-win`

En **PowerShell (como Administrador)**:

```
winget install --interactive --exact dorssel.usbipd-win
```

### 2. Enlazar la cámara Intel RealSense a WSL

Con la cámara conectada a un puerto **USB 3.0** (conector azul):

1. Lista los dispositivos USB conectados para localizar el `BUSID` de la RealSense:
    

```
usbipd list
```

2. Comparte el puerto del dispositivo con WSL (solo se requiere la primera vez):
    

```
usbipd bind --busid <TU-BUSID>
```

3. Conecta el dispositivo a la instancia activa de WSL:
    

```
usbipd attach --wsl --busid <TU-BUSID>
```

## Fase 2: Configuración en Ubuntu (WSL2)

Presiona la tecla `Windows`, escribe `Ubuntu` y ábrelo como **Administrador**. En la terminal de **Ubuntu** ejecuta en orden:

### 1. Dependencias del sistema y soporte USB

```
sudo apt update && sudo apt install -y usbutils python3-pip python3-venv libgl1 libglib2.0-0 unzip curl
```

### 2. Verificar detección de hardware

```
lsusb
```

_Debe listar:_ `Intel Corp. Intel(R) RealSense(TM) Depth Camera`.

### 3. Permisos de hardware

```
sudo usermod -aG video,plugdev $USER
sudo chmod 666 /dev/video* 2>/dev/null
sudo chmod -R 777 /dev/bus/usb/ 2>/dev/null
```

### 4. Entorno virtual de Python y carpeta del proyecto

```
# Crear entorno virtual limpio
python3 -m venv ~/vision_env

# Activar el entorno virtual
source ~/vision_env/bin/activate

# Actualizar pip e instalar librerías necesarias (incluyendo Ultralytics YOLOv8)
pip install --upgrade pip
pip install opencv-python numpy pyrealsense2 flask ultralytics

# Crear y acceder a la carpeta del proyecto
mkdir -p ~/realsense-yolo-inferencia
cd ~/realsense-yolo-inferencia
```

## Fase 3: Scripts de Captura y Generación de Dataset (`s6_captura_dataset.py`)

Un error crítico al construir un dataset es guardar capturas de pantalla que contengan rectángulos dibujados o texto superpuesto (lo cual contamina los pesos de la red neuronal). El siguiente script despliega una interfaz en `http://localhost:5000` que muestra una guía visual en el navegador, pero almacena en disco **exclusivamente el fotograma crudo nativo** ($640 \times 480 \times 3\text{ BGR}$) dentro de la carpeta `dataset_raw/`.

### 1. Crear el script `s6_captura_dataset.py`

Ejecuta en tu terminal de **Ubuntu (WSL2)**:

```
cat << 'EOF' > ~/realsense-yolo-inferencia/s6_captura_dataset.py
#!/usr/bin/env python3
"""
==============================================================================
MR3005C: Sistemas Ciberfísicos — Módulo 8: Visión Artificial
Generador de Dataset Propio con Intel RealSense D435i/D455 para YOLOv8
==============================================================================
"""

import os
import sys
import signal
import time
import glob
import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response, jsonify, render_template_string

app = Flask(__name__)

# ============================================================================
# 1. CONFIGURACIÓN DEL DIRECTORIO DE MUESTRAS
# ============================================================================
DIR_CAPTURAS = "dataset_raw"
PREFIJO_CLASE = "pieza"
META_IMAGENES = 70  # Rango requerido para el Entregable A6: 60 a 80 fotos

os.makedirs(DIR_CAPTURAS, exist_ok=True)

# Contar imágenes previas para no sobrescribir sesiones anteriores
archivos_previos = sorted(glob.glob(os.path.join(DIR_CAPTURAS, f"{PREFIJO_CLASE}_*.jpg")))
contador_fotos = len(archivos_previos)

ultimo_frame_crudo = None
modo_rafaga = False
ultimo_tiempo_rafaga = 0.0
INTERVALO_RAFAGA_S = 1.5  # Segundos entre fotos automáticas

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
    print(f"\n[INFO] Cámara liberada. Total acumulado en '{DIR_CAPTURAS}/': {contador_fotos} imágenes.")
    sys.exit(0)

signal.signal(signal.SIGINT, liberar_recursos)

# Estabilización del sensor CMOS (Auto-Exposure)
for _ in range(15):
    pipe.wait_for_frames()

def guardar_fotograma_crudo():
    global contador_fotos, ultimo_frame_crudo
    if ultimo_frame_crudo is None:
        return False, contador_fotos
    contador_fotos += 1
    nombre_archivo = f"{PREFIJO_CLASE}_{contador_fotos:03d}.jpg"
    ruta_completa = os.path.join(DIR_CAPTURAS, nombre_archivo)
    # Guardamos el tensor limpio (sin líneas ni texto superpuesto)
    cv2.imwrite(ruta_completa, ultimo_frame_crudo, [cv2.IMWRITE_JPEG_QUALITY, 98])
    print(f"[CAPTURA OK] Guardada: {ruta_completa} ({contador_fotos}/{META_IMAGENES})")
    return True, contador_fotos

# ============================================================================
# 3. INTERFAZ WEB Y STREAMING MULTIPART
# ============================================================================
HTML_UI = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>MR3005C — Capturador de Dataset YOLOv8</title>
    <style>
        body { background: #0f172a; color: #f8fafc; font-family: system-ui, sans-serif; text-align: center; margin: 0; padding: 20px; }
        .contenedor { max-width: 720px; margin: auto; background: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; }
        img { border: 2px solid #38bdf8; border-radius: 8px; width: 640px; height: 480px; }
        .controles { margin-top: 15px; display: flex; justify-content: center; gap: 15px; }
        button { padding: 12px 24px; font-size: 16px; font-weight: bold; border: none; border-radius: 8px; cursor: pointer; transition: 0.2s; }
        .btn-foto { background: #22c55e; color: #052e16; }
        .btn-foto:hover { background: #4ade80; }
        .btn-rafaga { background: #eab308; color: #422006; }
        .estado { margin-top: 12px; font-size: 18px; color: #38bdf8; font-weight: bold; }
        .guia { text-align: left; font-size: 14px; color: #cbd5e1; margin-top: 15px; background: #0f172a; padding: 12px; border-radius: 8px; }
    </style>
</head>
<body>
    <div class="contenedor">
        <h2>Generador de Dataset Propio (Entregable A6) — MR3005C</h2>
        <img src="/video_feed" alt="Stream RealSense">
        <div class="controles">
            <button class="btn-foto" onclick="capturar()">Capturar Foto (Tecla ESPACIO)</button>
            <button class="btn-rafaga" id="btnRafaga" onclick="toggleRafaga()">Ráfaga Automática (1.5s): OFF</button>
        </div>
        <div class="estado" id="contador">Imágenes capturadas: {{ total }} / {{ meta }}</div>
        <div class="guia">
            <strong>Protocolo de Variabilidad Geométrica y Lumínica (60 a 80 fotos):</strong>
            <ul>
                <li><strong>Rotación y Escala:</strong> Gira la pieza cada 15 grados y varía la distancia entre 15 cm y 60 cm.</li>
                <li><strong>Iluminación y Fondo:</strong> Captura con luz directa, en penumbra lateral y sobre fondos distintos (mesa, gaveta, mano).</li>
                <li><strong>Oclusión Parcial:</strong> Incluye al menos 10 fotos donde el objeto esté parcialmente cubierto en un 20%.</li>
            </ul>
        </div>
    </div>
    <script>
        async function capturar() {
            const res = await fetch('/capturar', { method: 'POST' });
            const data = await res.json();
            document.getElementById('contador').innerText = `Imágenes capturadas: ${data.total} / {{ meta }}`;
        }
        async function toggleRafaga() {
            const res = await fetch('/rafaga', { method: 'POST' });
            const data = await res.json();
            const btn = document.getElementById('btnRafaga');
            btn.innerText = `Ráfaga Automática (1.5s): ${data.rafaga ? 'ON' : 'OFF'}`;
            btn.style.background = data.rafaga ? '#ef4444' : '#eab308';
            btn.style.color = data.rafaga ? '#ffffff' : '#422006';
        }
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space') { e.preventDefault(); capturar(); }
        });
        setInterval(async () => {
            const res = await fetch('/estado');
            const data = await res.json();
            document.getElementById('contador').innerText = `Imágenes capturadas: ${data.total} / {{ meta }}`;
        }, 1000);
    </script>
</body>
</html>
"""

def generar_streaming():
    global ultimo_frame_crudo, ultimo_tiempo_rafaga, contador_fotos, modo_rafaga
    while True:
        try:
            frames = pipe.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue

            frame = np.asanyarray(color.get_data())
            ultimo_frame_crudo = frame.copy()

            # Disparador de ráfaga automática
            ahora = time.time()
            if modo_rafaga and (ahora - ultimo_tiempo_rafaga >= INTERVALO_RAFAGA_S):
                guardar_fotograma_crudo()
                ultimo_tiempo_rafaga = ahora

            # Copia visual únicamente para el navegador (la foto guardada en disco NO tiene este texto)
            vista = frame.copy()
            cv2.rectangle(vista, (10, 10), (630, 48), (15, 23, 42), -1)
            progreso = min(100.0, (contador_fotos / META_IMAGENES) * 100.0)
            color_barra = (0, 255, 0) if contador_fotos >= META_IMAGENES else (0, 200, 255)
            estado_r = "RAFAGA ON" if modo_rafaga else "MANUAL [ESPACIO]"
            cv2.putText(vista, f"FOTOS: {contador_fotos}/{META_IMAGENES} ({progreso:.0f}%) | Modo: {estado_r}",
                        (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color_barra, 2)

            ok, buffer = cv2.imencode('.jpg', vista)
            if not ok:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception:
            break

@app.route('/')
def index():
    return render_template_string(HTML_UI, total=contador_fotos, meta=META_IMAGENES)

@app.route('/video_feed')
def video_feed():
    return Response(generar_streaming(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capturar', methods=['POST'])
def endpoint_capturar():
    ok, total = guardar_fotograma_crudo()
    return jsonify({"ok": ok, "total": total})

@app.route('/rafaga', methods=['POST'])
def endpoint_rafaga():
    global modo_rafaga
    modo_rafaga = not modo_rafaga
    return jsonify({"rafaga": modo_rafaga, "total": contador_fotos})

@app.route('/estado')
def endpoint_estado():
    return jsonify({"total": contador_fotos, "rafaga": modo_rafaga})

if __name__ == '__main__':
    print("\n=======================================================")
    print(" CAPTURADOR DE DATASET EN VIVO (REALSENSE -> YOLOv8)")
    print(" Abre en tu navegador: http://localhost:5000")
    print(f" Carpeta destino: '{DIR_CAPTURAS}/'")
    print(" Presiona Ctrl + C para finalizar.")
    print("=======================================================\n")
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        liberar_recursos()
EOF
```

### 2. Ejecutar la captura de imágenes

```
source ~/vision_env/bin/activate
cd ~/realsense-yolo-inferencia
python s6_captura_dataset.py
```

1. Abre `http://localhost:5000` en tu navegador de Windows.
    
2. Captura entre $60$ **y** $80$ **fotografías** variando ángulo, distancia ($15\text{ cm}$ a $60\text{ cm}$), iluminación y fondo.
    
3. Cuando llegues a la meta, presiona `Ctrl + C` en la terminal de Ubuntu para cerrar la cámara.
    

## Fase 4: Flujo de Trabajo Paso a Paso en Roboflow (Etiquetado y Exportación)

YOLOv8 requiere que cada fotografía `.jpg` vaya acompañada de un archivo `.txt` con el mismo nombre, donde cada línea describe un objeto mediante **coordenadas normalizadas invariantes a la escala** en el rango $[0, 1]$:

$$\text{Formato de línea: } \quad \text{class\_id} \quad x_n \quad y_n \quad w_n \quad h_n$$

donde las coordenadas respecto al ancho ($W = 640$) y alto ($H = 480$) de la imagen son:

$$x_n = \frac{c_x}{W}, \quad y_n = \frac{c_y}{H}, \quad w_n = \frac{w_{\text{px}}}{W}, \quad h_n = \frac{h_{\text{px}}}{H}$$

A continuación se detalla el procedimiento exacto en **Roboflow** para generar esta estructura sin errores:

### Paso 4.1: Crear cuenta y proyecto en Roboflow

1. Ingresa a [https://app.roboflow.com/](https://app.roboflow.com/ "null") y regístrate con tu correo institucional o cuenta de GitHub.
    
2. Al crear tu _Workspace_, selecciona el plan **Public (Free)** para disponer de créditos gratuitos de exportación.
    
3. Haz clic en el botón **Create New Project** y configura:
    
    - **Project Name:** `mr3005c-pieza-continuum`
        
    - **Project Type:** Selecciona **Object Detection (Bounding Boxes)**. _(¡Atención!: NO selecciones Instance Segmentation ni Classification)._
        
    - **Annotation Group:** Escribe `pieza` (o `objetos-celda`).
        
    - Haz clic en **Create Public Project**.
        

### Paso 4.2: Subir las fotos desde Ubuntu (WSL2) hacia Roboflow

Para abrir directamente tu carpeta de fotos de Ubuntu en una ventana del Explorador de Archivos de Windows, ejecuta este comando en tu terminal de **Ubuntu**:

```
explorer.exe ~/realsense-yolo-inferencia/dataset_raw
```

_(Alternativamente, puedes escribir `\\wsl.localhost\Ubuntu\home\<tu_usuario>\realsense-yolo-inferencia\dataset_raw` en la barra de direcciones del Explorador de Windows)._

1. Selecciona todas las imágenes (`pieza_001.jpg` a `pieza_070.jpg`) y arrástralas hacia el recuadro **Upload** de Roboflow en tu navegador.
    
2. Una vez procesadas las miniaturas, haz clic en el botón verde **Save and Continue** (esquina superior derecha).
    
3. En la ventana emergente de asignación de tareas, haz clic en **Assign Images** para asignarte el lote completo.
    

### Paso 4.3: Etiquetado manual de precisión (`Annotate`)

Haz clic sobre la primera imagen para abrir el editor de etiquetado:

1. Presiona la tecla **`B`** (herramienta _Bounding Box_).
    
2. Haz clic y arrastra el cursor formando un rectángulo que envuelva **exactamente los bordes físicos de tu pieza** (_Tight Bounding Box_).
    
    - **Regla Crítica de Ingeniería:** No dejes espacio vacío innecesario entre la pieza y el rectángulo, ni tampoco recortes extremos de la pieza. Recuerda que en la inferencia el modelo Pinhole calcula la distancia frontal $Z = \frac{f_x \cdot W_{\text{real}}}{w_{\text{px}}}$ usando el ancho en píxeles $w_{\text{px}}$ de esta caja.
        
3. En el cuadro de texto emergente, escribe el nombre de tu clase en minúsculas y sin espacios: `pieza` (si entrenas varios objetos distintos, usa etiquetas claras como `pieza_a`, `pieza_b`) y presiona **`Enter`**.
    
4. Avanza a la siguiente fotografía con la flecha derecha del teclado (`->`) y repite hasta completar las $60 - 80$ imágenes.
    
5. Cuando termines todas las fotos, regresa con la flecha superior izquierda y haz clic en el botón verde **Add X Approved Images to Dataset**.
    

### Paso 4.4: División estadística del conjunto (`Train / Valid / Test Split`)

Al hacer clic en _Add Images to Dataset_, Roboflow preguntará cómo dividir las muestras. Selecciona **Split Images Between Train/Valid/Test** y verifica que los porcentajes sean:

- **Train (**$70\%$ **—** $\approx 50$ **imágenes):** Muestras utilizadas para calcular el descenso de gradiente y actualizar los pesos de la red.
    
- **Valid (**$20\%$ **—** $\approx 14$ **imágenes):** Muestras utilizadas al final de cada época para medir la generalización y guardar el mejor punto de control (`best.pt`).
    
- **Test (**$10\%$ **—** $\approx 6$ **imágenes):** Muestras reservadas para la auditoría final del modelo.
    

Haz clic en **Add Images**.

### Paso 4.5: Preprocesamiento y Aumento de Datos (`Generate New Version`)

En el menú lateral izquierdo de Roboflow, entra a **Versions** (o _Generate_):

1. **Step 1 & 2 (Source Images & Train/Test Split):** Haz clic en _Continue_.
    
2. **Step 3 (Preprocessing):**
    
    - Deja activos **Auto-Orient** y **Resize: Stretch to** $640 \times 640$ (resolución nativa del tensor de entrada de YOLOv8). Haz clic en _Continue_.
        
3. **Step 4 (Augmentation — Opcional pero muy recomendado):**
    
    - Haz clic en **Add Augmentation Step** y selecciona:
        
        - **Flip:** _Horizontal_.
            
        - **Brightness:** _Between_ $-15\%$ _and_ $+15\%$ (simula cambios de luz en el laboratorio).
            
    - Haz clic en _Continue_.
        
4. **Step 5 (Create):** Selecciona la opción gratuita ($1\times$ o $2\times$**/**$3\times$ según tu límite de créditos) y presiona el botón **Create**.
    

### Paso 4.6: Exportar el Dataset en formato `YOLOv8` hacia Ubuntu (WSL2)

Una vez generada la versión (`v1`):

1. Haz clic en el botón **Export Dataset** (esquina superior derecha).
    
2. En el menú desplegable _Format_, selecciona **`YOLOv8`**.
    
3. Selecciona la opción **Show download code**, ve a la pestaña **Terminal** y copia el comando `curl -L "https://app.roboflow.com/ds/..."` que te proporciona Roboflow.
    
4. En tu terminal de **Ubuntu (WSL2)** ejecuta:
    

```
mkdir -p ~/realsense-yolo-inferencia/dataset_yolo
cd ~/realsense-yolo-inferencia/dataset_yolo

# Pega aquí tu comando curl de Roboflow (ejemplo):
# curl -L "https://app.roboflow.com/ds/TU_ENLACE_ROBOFLOW" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```

_(Si preferiste descargar el `.zip` directamente en Windows con **Download zip to computer**, descomprímelo dentro de `\\wsl.localhost\Ubuntu\home\<tu_usuario>\realsense-yolo-inferencia\dataset_yolo`)._

5. Verifica que el archivo `~/realsense-yolo-inferencia/dataset_yolo/data.yaml` exista y ajusta las rutas para que apunten a las carpetas locales:
    

```
cat << 'EOF' > ~/realsense-yolo-inferencia/dataset_yolo/data.yaml
path: .
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['pieza']
EOF
```

_(Nota: Si etiquetaste más de un objeto, actualiza `nc` con el número total de clases y respeta el orden exacto de `names` que generó Roboflow)._

## Fase 5: Requisitos Oficiales del Entregable "A6: Generación de Dataset de Visión" (9 de Noviembre)

Para acreditar el **Entregable A6** el día **9 de noviembre**, tu equipo debe contar con los siguientes elementos verificables en su repositorio y reporte de entrega:

### Lista de Cotejo (Checklist) del Entregable A6

|   |   |   |
|---|---|---|
|**Rubro**|**Especificación Técnica Requerida**|**Estado**|
|**1. Captura Cruda (`dataset_raw/`)**|Mínimo $60$ **a** $80$ **fotografías** nativas ($640 \times 480$) capturadas con `s6_captura_dataset.py` y la cámara Intel RealSense, libres de texto o cajas impresas.|$\square$|
|**2. Diversidad de Muestras**|El dataset incluye al menos $3$ distancias ($15\text{ cm}$, $30\text{ cm}$, $50\text{ cm}$), rotaciones en múltiples ángulos, cambios de iluminación y al menos $10$ fotos con oclusión parcial ($\approx 20\%$).|$\square$|
|**3. Etiquetado Normalizado**|Todas las imágenes etiquetadas con _Tight Bounding Boxes_ en Roboflow. Cada archivo `.jpg` tiene su par `.txt` con coordenadas normalizadas $[x_n, y_n, w_n, h_n] \in [0, 1]^4$.|$\square$|
|**4. Estructura `dataset_yolo/`**|Carpetas `train/images`, `train/labels`, `valid/images`, `valid/labels`, `test/images`, `test/labels` con partición $70\% / 20\% / 10\%$ y archivo maestro `data.yaml`.|$\square$|
|**5. Evidencias Documentales (Reporte A6)**|**a)** Enlace o archivo `.zip` de la versión exportada en formato `YOLOv8`.<br><br>  <br><br>**b)** Captura del panel _Dataset Health Check_ de Roboflow mostrando el total de imágenes y el _split_.<br><br>  <br><br>**c)** Captura de $3$ imágenes etiquetadas en Roboflow (luz normal, penumbra y oclusión) junto con la impresión de uno de sus archivos `.txt`.|$\square$|

Para verificar rápidamente desde la terminal de Ubuntu que tu dataset cumple con la estructura para el Entregable A6, ejecuta:

```
cd ~/realsense-yolo-inferencia/dataset_yolo
echo "Fotos Train : $(ls train/images/*.jpg 2>/dev/null | wc -l) | Labels Train : $(ls train/labels/*.txt 2>/dev/null | wc -l)"
echo "Fotos Valid : $(ls valid/images/*.jpg 2>/dev/null | wc -l) | Labels Valid : $(ls valid/labels/*.txt 2>/dev/null | wc -l)"
echo "Fotos Test  : $(ls test/images/*.jpg 2>/dev/null | wc -l) | Labels Test  : $(ls test/labels/*.txt 2>/dev/null | wc -l)"
head -n 5 train/labels/*.txt | head -n 10
```

## Fase 6: Reentrenamiento por Transfer Learning (`s6_entrenar_yolo.py`) — Hacia el Entregable A7 (12 de Noviembre)

Una vez validado tu dataset del Entregable A6, el siguiente paso es entrenar la red neuronal. Aplicamos **Transferencia de Aprendizaje (Transfer Learning)** partiendo de los pesos pre-entrenados `yolov8n.pt` (cuyo _Backbone_ CSPDarknet ya domina bordes, elipses y texturas metálicas sobre el dataset COCO) y realizamos **Fine-Tuning** sobre la cabeza desacoplada (_Decoupled Head_) durante $30$ épocas.

### Opción A: Entrenamiento Local en Ubuntu (WSL2)

Crea el script `s6_entrenar_yolo.py`:

```
cat << 'EOF' > ~/realsense-yolo-inferencia/s6_entrenar_yolo.py
#!/usr/bin/env python3
"""
==============================================================================
MR3005C: Sistemas Ciberfísicos — Módulo 8: Visión Artificial
Entrenamiento (Fine-Tuning) de YOLOv8n con Dataset Propio (Entregable A7)
==============================================================================
"""

import os
import shutil
import torch
from ultralytics import YOLO

RUTA_YAML = os.path.abspath("dataset_yolo/data.yaml")
EPOCAS = 30
RESOLUCION = 640
BATCH_SIZE = 16

if not os.path.exists(RUTA_YAML):
    raise FileNotFoundError(f"[ERROR] No se encontró el archivo de configuración en: {RUTA_YAML}")

dispositivo = "0" if torch.cuda.is_available() else "cpu"
print(f"\n[INFO] Dispositivo de cómputo detectado: {'GPU CUDA' if dispositivo == '0' else 'CPU Host'}")
print("[INFO] Cargando pesos pre-entrenados base 'yolov8n.pt' (Transfer Learning desde COCO)...")

# 1. Cargar checkpoint base
modelo = YOLO("yolov8n.pt")

# 2. Ejecutar optimización por Descenso de Gradiente
resultados = modelo.train(
    data=RUTA_YAML,
    epochs=EPOCAS,
    imgsz=RESOLUCION,
    batch=BATCH_SIZE,
    device=dispositivo,
    patience=10,         # Early stopping si no mejora en 10 épocas
    project="runs/detect",
    name="entregable_a7",
    exist_ok=True,
    pretrained=True,
    optimizer="auto",
    verbose=True
)

# 3. Copiar automáticamente el mejor punto de control a la raíz del proyecto
ruta_best = os.path.join("runs/detect/entregable_a7/weights/best.pt")
if os.path.exists(ruta_best):
    shutil.copy(ruta_best, "best.pt")
    print("\n==================================================================")
    print(" [OK] ENTRENAMIENTO FINALIZADO EXITOSAMENTE")
    print(" El archivo 'best.pt' se ha copiado a la raíz del repositorio.")
    print(" Al ejecutar 's5_inferencia_yolo.py', se usará tu nuevo modelo.")
    print("==================================================================\n")
EOF
```

Ejecútalo con:

```
cd ~/realsense-yolo-inferencia
python s6_entrenar_yolo.py
```

### Opción B: Entrenamiento Acelerado en Google Colab (GPU Nvidia T4 en $\approx 8\text{ min}$)

Si tu laptop no cuenta con GPU Nvidia dedicada, abre [Google Colab](https://colab.research.google.com/ "null"), cambia el entorno de ejecución a **GPU T4** y pega el código de descarga de Roboflow junto con el entrenamiento:

```
!pip install ultralytics roboflow

# 1. Pega aquí las 4 líneas de descarga que te dio Roboflow (Show download code -> Python)
# from roboflow import Roboflow
# rf = Roboflow(api_key="TU_API_KEY")
# project = rf.workspace("...").project("mr3005c-pieza-continuum")
# dataset = project.version(1).download("yolov8")

# 2. Entrenar YOLOv8n durante 30 épocas en GPU T4
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data=f"{dataset.location}/data.yaml", epochs=30, imgsz=640, batch=16, device=0)
```

Al concluir, descarga desde el panel de archivos de Colab el archivo `runs/detect/train/weights/best.pt` y los gráficos `results.png` y `confusion_matrix.png`, y coloca `best.pt` dentro de `~/realsense-yolo-inferencia/best.pt`.

## Fase 7: ¿Cómo Saber que el Modelo ya Está Bien Entrenado? (Auditoría de Métricas)

Un modelo **no** está listo solo porque el comando `model.train()` terminó de ejecutarse. Antes de conectarlo al lazo de control del robot continuum, debemos auditar sus métricas matemáticas.

### 1. Los 4 Indicadores de Convergencia que Debes Verificar

1. **Descenso Monótono de las Funciones de Pérdida (`results.png`):**
    
    - `train/box_loss` y `val/box_loss`: Error geométrico en las coordenadas $[l, t, r, b]$ de la caja delimitadora. Debe disminuir de forma continua y estabilizarse cerca de una asíntota horizontal.
        
    - `train/cls_loss` y `val/cls_loss`: Error en la clasificación de la pieza.
        
    - **Señal de Alerta (Overfitting / Sobreajuste):** Si `train/box_loss` sigue bajando pero `val/box_loss` comienza a **subir**, el modelo memorizó las fotos de entrenamiento. Por ello YOLO guarda dos archivos: `last.pt` y `best.pt`; **siempre** utilizamos `best.pt`.
        
2. **Intersección sobre Unión (**$\text{IoU} \ge 0.50$**):**  
    
    $$\text{IoU} = \frac{\text{Área}(\text{Predicción} \cap \text{Ground Truth})}{\text{Área}(\text{Predicción} \cup \text{Ground Truth})}$$
    
    Garantiza que la caja predicha envuelva fielmente la pieza física para no distorsionar el cálculo de distancia Pinhole.
    
3. **Balance entre** $\text{Precision}$ **y** $\text{Recall}$**:**  
    
    $$\text{Precision} = \frac{TP}{TP + FP}, \qquad \text{Recall} = \frac{TP}{TP + FN}$$
    - $\text{Precision}$ **alta (**$> 0.85$**):** Protege al robot contra _Falsas Alarmas_ ($FP$), evitando que intente sujetar sombras o reflejos.
        
    - $\text{Recall}$ **alto (**$> 0.80$**):** Protege contra _Omisiones_ ($FN$), asegurando que el robot no ignore la pieza cuando está frente a la cámara.
        
4. **Criterio Oficial de Aprobación para el Entregable A7 (**$\text{mAP}_{50} \ge 0.85$**):** El área bajo la curva Precision-Recall evaluada a un umbral $\text{IoU} = 0.50$ debe cumplir estrictamente:  
    
    $$\text{mAP}_{50} \ge 0.85$$

### 2. Script de Auditoría y Certificación Automática (`s6_validar_modelo.py`)

Ejecuta este script en Ubuntu para evaluar tu archivo `best.pt` y obtener el dictamen automático del Entregable A7:

```
cat << 'EOF' > ~/realsense-yolo-inferencia/s6_validar_modelo.py
#!/usr/bin/env python3
"""
==============================================================================
MR3005C: Auditoría Matemática de Convergencia del Modelo (Entregable A7)
Verifica Precision, Recall, mAP50 y emite dictamen de aprobación.
==============================================================================
"""

import os
from ultralytics import YOLO

RUTA_MODELO = "best.pt"
RUTA_YAML = os.path.abspath("dataset_yolo/data.yaml")
UMBRAL_APROBACION_MAP50 = 0.85

if not os.path.exists(RUTA_MODELO):
    raise FileNotFoundError("[ERROR] No se encontró 'best.pt'. Ejecuta primero el entrenamiento.")

print(f"\n[INFO] Evaluando pesos sinápticos '{RUTA_MODELO}' contra el conjunto de validación...")
modelo = YOLO(RUTA_MODELO)

metricas = modelo.val(data=RUTA_YAML, imgsz=640, conf=0.25, iou=0.50, verbose=False)

precision = float(metricas.box.mp)
recall = float(metricas.box.mr)
map50 = float(metricas.box.map50)
map50_95 = float(metricas.box.map)

print("\n======================================================================")
print(" REPORTE DE AUDITORÍA DE VISIÓN ARTIFICIAL — ENTREGABLE A7 (MR3005C)")
print("======================================================================")
print(f" Clases detectadas por el modelo : {modelo.names}")
print(f" 1. Precision (P)  [Anti-Falsas Alarmas] : {precision:.4f} ({precision*100:.1f}%)")
print(f" 2. Recall (R)     [Anti-Omisiones]      : {recall:.4f} ({recall*100:.1f}%)")
print(f" 3. mAP@50         [Umbral IoU = 0.50]   : {map50:.4f} ({map50*100:.1f}%)")
print(f" 4. mAP@50-95      [Rigor Geométrico]    : {map50_95:.4f} ({map50_95*100:.1f}%)")
print("----------------------------------------------------------------------")

if map50 >= UMBRAL_APROBACION_MAP50:
    print(f" DICTAMEN: [APROBADO] mAP50 >= {UMBRAL_APROBACION_MAP50}")
    print(" El modelo es seguro para integrarse al lazo de control del robot.")
else:
    print(f" DICTAMEN: [REENTRENAMIENTO REQUERIDO] mAP50 ({map50:.3f}) < {UMBRAL_APROBACION_MAP50}")
    print(" Acciones correctivas:")
    print("  - Revisa en Roboflow que las cajas estén ajustadas al borde real.")
    print("  - Agrega 20 imágenes adicionales en ángulos o luces donde falló.")
print("======================================================================\n")
EOF
```

## Fase 8: Prueba de Fuego en Tiempo Real con la RealSense

Dado que nuestros scripts `s5_inferencia_yolo.py`, `s5_inferencia_yolo_multiobj.py` y `s5_lab_yolo_experimentos.py` tienen programada la línea de prioridad automática:

```
MODELO_PATH = "best.pt" if os.path.exists("best.pt") else "yolov8n.pt"
```

Una vez que `best.pt` se encuentra en la carpeta `~/realsense-yolo-inferencia/`, simplemente lanza el nodo de inferencia habitual:

```
python s5_inferencia_yolo.py
```

Abre `http://localhost:5000` y comprueba en vivo:

1. Que en la terminal aparezca: `[INFO] Cargando red neuronal convolucional desde: 'best.pt'...`
    
2. Que al colocar tu objeto frente a la cámara RealSense se dibuje la caja delimitadora con el nombre de tu clase personalizada, un nivel de confianza $> 80\%$ y el cálculo de error cinemático $(e_u, e_v)$ y profundidad $Z$ operando a $30\text{ FPS}$.
