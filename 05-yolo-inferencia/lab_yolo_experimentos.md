# Guía de Laboratorio Práctico: Deconstrucción, Optimización y Telemetría Robótica con YOLOv8

**Módulo:** MR3005C — Sistemas Ciberfísicos (Módulo 8: Visión Artificial)
**Sesión:** Laboratorio Experimental de Inferencia, Ajuste de Hiperparámetros y Lógica Cinemática
**Duración:** 90 minutos
**Entorno de Ejecución:** Ubuntu en WSL2 con cámara Intel RealSense D435i/D455 sobre servidor Flask (`http://localhost:5000`)
### Script  (`s5_lab_yolo_experimentos.py`)
Crea este archivo en tu carpeta `~/realsense-yolo-inferencia/`. Integra todos los puntos de inyección de parámetros para los experimentos:


```Python
#!/usr/bin/env python3
"""
==============================================================================
MR3005C: Laboratorio Experimental de Inferencia y Telemetría con YOLOv8
Modifica las variables de control en cada bloque y recarga http://localhost:5000
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

# ============================================================================
# PANEL DE CONTROL DE HIPERPARÁMETROS (ZONA DE MODIFICACIÓN DEL ALUMNO)
# ============================================================================
CONF_UMBRAL = 0.40          # [EXP 1] Umbral de corte de confianza (0.01 a 0.99)
IOU_NMS = 0.50              # [EXP 2] Umbral de solapamiento de NMS (0.10 a 0.95)
RESOLUCION_INFERENCIA = 640 # [EXP 3] Tamaño del tensor: 160, 320, 640
CLASES_FILTRADAS = None     # [EXP 4] None o lista de IDs: ej. [39, 41, 64, 67]
AGNOSTIC_NMS = False        # [EXP 5] Fusión de cajas entre clases distintas
USAR_HALF = False           # [EXP 6] Inferencia en precisión media FP16
MAX_DETECCIONES = 300       # [EXP 7] Límite máximo de cajas por cuadro

# Parámetros ópticos y cinemáticos (Modelo Pinhole)
W_REAL_M = 0.05             # Ancho físico de referencia (5 cm = 0.05 m)
F_X = 615.0                 # Longitud focal calibrada (px)
U_C = 320.0                 # Centro óptico horizontal
V_C = 240.0                 # Centro óptico vertical

# ============================================================================
# INICIALIZACIÓN DE HARDWARE Y MODELO
# ============================================================================
app = Flask(__name__)
MODELO_PATH = "best.pt" if os.path.exists("best.pt") else "yolov8n.pt"
print(f"\n[INFO] Cargando red neuronal convolucional desde: '{MODELO_PATH}'...")
model = YOLO(MODELO_PATH)

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe.start(cfg)

def liberar(sig=None, frame=None):
    try:
        pipe.stop()
    except Exception:
        pass
    print("\n[INFO] Cámara liberada correctamente.")
    sys.exit(0)

signal.signal(signal.SIGINT, liberar)

for _ in range(10):
    pipe.wait_for_frames()

np.random.seed(42)
COLORES = np.random.uniform(50, 255, size=(100, 3))

# ============================================================================
# PIPELINE DE INFERENCIA EN TIEMPO REAL
# ============================================================================
def procesar_frame(frame):
    anotado = frame.copy()
    h, w, _ = frame.shape

    # Retícula de alineación del centro óptico (320, 240)
    cv2.circle(anotado, (int(U_C), int(V_C)), 4, (255, 255, 255), -1)
    x_tol_min, x_tol_max = int(U_C - 45), int(U_C + 45)
    y_tol_min, y_tol_max = int(V_C - 45), int(V_C + 45)
    cv2.rectangle(anotado, (x_tol_min, y_tol_min), (x_tol_max, y_tol_max), (255, 255, 255), 1)

    t0 = time.perf_counter()
    # Ejecución de inferencia con hiperparámetros dinámicos
    resultados = model.predict(
        frame,
        conf=CONF_UMBRAL,
        iou=IOU_NMS,
        imgsz=RESOLUCION_INFERENCIA,
        classes=CLASES_FILTRADAS,
        agnostic_nms=AGNOSTIC_NMS,
        half=USAR_HALF,
        max_det=MAX_DETECCIONES,
        verbose=False
    )[0]
    dt_ms = (time.perf_counter() - t0) * 1000.0
    fps = 1000.0 / max(dt_ms, 1e-5)

    cajas = resultados.boxes
    n_det = len(cajas)
    acople_valido = False

    for b in cajas:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        conf = float(b.conf[0])
        cls_id = int(b.cls[0])
        clase = resultados.names[cls_id]

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        ancho_px = max(x2 - x1, 1)

        # Distancia frontal Pinhole: Z = (fx * W) / w_px
        z_est_cm = ((F_X * W_REAL_M) / ancho_px) * 100.0

        # Validación de acople cinemático para el continuum
        centrado = (x_tol_min <= cx <= x_tol_max) and (y_tol_min <= cy <= y_tol_max)
        rango_z = (15.0 <= z_est_cm <= 30.0)

        if centrado and rango_z:
            color = (0, 255, 0)
            acople_valido = True
        else:
            color = [int(c) for c in COLORES[cls_id % len(COLORES)]]

        cv2.rectangle(anotado, (x1, y1), (x2, y2), color, 2)
        cv2.circle(anotado, (cx, cy), 4, color, -1)
        cv2.line(anotado, (int(U_C), int(V_C)), (cx, cy), color, 1)

        etiqueta = f"{clase} {conf*100:.0f}% | ~{z_est_cm:.0f}cm"
        cv2.putText(anotado, etiqueta, (x1, max(y1 - 6, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Panel superior de telemetría de ingeniería
    cv2.rectangle(anotado, (10, 10), (630, 48), (15, 23, 42), -1)
    color_status = (0, 255, 0) if acople_valido else (0, 200, 255)
    texto_status = "LISTO PARA INSERCION" if acople_valido else "ALINEANDO ROBOT"

    cv2.putText(anotado, f"Obj: {n_det} | {dt_ms:.1f}ms ({fps:.1f} FPS) | {texto_status}",
                (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color_status, 2)

    return anotado

def generar():
    while True:
        try:
            frames = pipe.wait_for_frames()
            c = frames.get_color_frame()
            if not c:
                continue
            frame = np.asanyarray(c.get_data())
            out = procesar_frame(frame)
            ok, buf = cv2.imencode('.jpg', out)
            if not ok:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        except Exception:
            break

@app.route('/')
def stream():
    return Response(generar(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("\nServidor de Laboratorio activo en: http://localhost:5000\n")
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        liberar()
```

## Experimento 1: El Dilema de la Confianza y Alucinaciones en Penumbra (`conf`)

- **Tiempo Asignado:** 10 minutos.
- **Concepto Teórico:** Cada celda de la rejilla de YOLOv8 predice probabilidades de clase mediante una función de activación sigmoide:
    $$\sigma(z) = \frac{1}{1 + e^{-z}} \in [0, 1]$$
    
    El hiperparámetro `conf` actúa como compuerta booleana dura: descarta cualquier vector candidato cuyo score de clase sea menor a dicho umbral.
### Procedimiento en Código
Modifica en `lab_yolo_experimentos.py` la variable `CONF_UMBRAL`:
1. `CONF_UMBRAL = 0.03` (Modo ultra-permisivo). Guarda con `Ctrl + S`, reinicia el script en la terminal (`Ctrl + C` $\rightarrow$ flecha arriba $\rightarrow$ `Enter`) y recarga.

![[Pasted image 20260922170027.png]]
1. `CONF_UMBRAL = 0.40` (Modo calibrado).
2. `CONF_UMBRAL = 0.90` (Modo restrictivo).
### Dinámica de Laboratorio

- Apaga las lámparas directas del banco o proyecta sombra con tu cuerpo sobre la gaveta de prueba.
- Con `0.03`, observa cómo grietas, texturas de la mesa o pliegues de ropa generan cajas erráticas etiquetadas como objetos inexistentes (alucinaciones).
- Con `0.90`, cubre un 25% de la pieza con la mano: el objeto desaparece de inmediato de la telemetría.

### Registro de Datos del Estudiante

|**Umbral conf**|**N° Falsos Positivos en Fondo**|**Estabilidad de Detección bajo Sombra**|**Riesgo Cinemático en el Robot**|
|---|---|---|---|
|`0.03`|Alto ($> 5$ cajas fantasma)|Inestable (cajas parpadeantes)|Movimientos erráticos/colisión del hexápodo.|
|`0.40`|Nulo ($0$ cajas fantasma)|Estable (sigue la pieza)|Óptimo para control cinemático.|
|`0.90`|Nulo ($0$ cajas fantasma)|Nula (pierde la pieza)|El brazo continuum se congela por timeout.|

## Experimento 2: Supresión de No Máximos (`iou` / NMS) y Oclusión Parcial

- **Tiempo Asignado:** 10 minutos.
- **Concepto Teórico:** Cuando un objeto es detectado, múltiples celdas contiguas en las rejillas P3, P4 y P5 generan cajas válidas. El algoritmo NMS (_Non-Maximum Suppression_) opera de forma secuencial:
    1. Ordena las predicciones de forma descendente según su confianza $s_i$.
    2. Selecciona la caja de máxima confianza $M$ y la conserva.
    3. Suprime cualquier otra caja vecina $B_j$ si su solapamiento geométrico supera el umbral:
        $$\text{IoU}(M, B_j) = \frac{\text{Área}(M \cap B_j)}{\text{Área}(M \cup B_j)} > \text{IOU\_NMS} \quad$$
        

### Procedimiento en Código

Fija `CONF_UMBRAL = 0.25` y modifica la variable `IOU_NMS`:
1. `IOU_NMS = 0.95` (NMS virtualmente desactivado).
2. `IOU_NMS = 0.50` (Estándar industrial).
3. `IOU_NMS = 0.08` (NMS ultra-agresivo).
### Dinámica de Laboratorio

- Coloca sobre la mesa dos objetos pequeños del mismo tipo (por ejemplo, dos botellas o dos marcadores) separados por $10\text{ cm}$.
- Con `IOU_NMS = 0.95`, observa la pieza: verás $3$ o $4$ rectángulos dibujados concéntricamente sobre el mismo objeto.
- Con `IOU_NMS = 0.08`, desliza lentamente un objeto detrás del otro hasta que se solapen un $20\%$. Observa cómo una de las piezas se elimina de la pantalla, dejando al robot ciego ante el segundo objeto.

## Experimento 3: Resolución de Inferencia (`imgsz`) vs. Costo Cuadrático $O(W \cdot H)$

- **Tiempo Asignado:** 10 minutos.
- **Concepto Teórico:** El cómputo en capas convolucionales depende directamente de las dimensiones espaciales del tensor. El costo operacional de un bloque convolucional con kernel $K \times K$ está acotado por:
    $$\text{FLOPs} \propto H \times W \times C_{\text{in}} \times C_{\text{out}} \times K^2 \quad$$
    
    Reducir `imgsz` a la mitad reduce las operaciones flotantes aproximadamente a una cuarta parte ($\frac{1}{4}$), acelerando la inferencia a costa de perder la resolución de las celdas de la rejilla P3 ($80 \times 80$).

### Procedimiento en Código

Modifica `RESOLUCION_INFERENCIA`:
1. `RESOLUCION_INFERENCIA = 160
2. `RESOLUCION_INFERENCIA = 320
3. `RESOLUCION_INFERENCIA = 640

### Dinámica de Laboratorio

- Coloca la pieza de prueba a una distancia de $1.2\text{ m}$ de la cámara.
- Anota la latencia en milisegundos y los FPS reportados en la barra superior.
### Registro de Datos del Estudiante

| **imgsz** | **Latencia en CPU (ms)**    | **Tasa de Cuadros (FPS)**    | **¿Detecta la pieza a 1.2 m?**                   |
| --------- | --------------------------- | ---------------------------- | ------------------------------------------------ |
| **160**   | $\approx 8 - 14\text{ ms}$  | $> 70\text{ FPS}$            | No (el objeto mide $< 8\text{ px}$ en el tensor) |
| **320**   | $\approx 18 - 25\text{ ms}$ | $\approx 40 - 50\text{ FPS}$ | Marginal                                         |
| **640**   | $\approx 32 - 45\text{ ms}$ | $\approx 22 - 30\text{ FPS}$ | **Sí (detección estable con forma íntegra)**     |

## Experimento 4: Filtrado Semántico de Clases de Interés (`classes`)

- **Tiempo Asignado:** 10 minutos.
- **Concepto Teórico:** El modelo base `yolov8n.pt` evalúa $80$ clases simultáneas del dataset COCO. En una celda robotizada, los operadores humanos, escritorios o cables no deben reportarse al lazo de control cinemático. El parámetro `classes` descarta vectores de predicción en el post-procesamiento sin requerir reentrenamiento de la red.

### Procedimiento en Código

Identifica los IDs estándar de objetos comunes en el laboratorio:
- `0`: Persona (`person`)
- `39`: Botella (`bottle`)
- `41`: Taza (`cup`)
- `64`: Ratón de computadora (`mouse`)
- `67`: Celular (`cell phone`)

Modifica la variable `CLASES_FILTRADAS`:

```Python
# Aislar exclusivamente celulares (67) y ratones (64)
	CLASES_FILTRADAS = [64, 67]
```

### Dinámica de Laboratorio

- Párate de cuerpo entero frente a la RealSense mientras sostienes un mouse o celular.
- Comprueba que tu silueta corporal sea ignorada por el pipeline, mientras que la herramienta en tu mano se rastrea con prioridad absoluta, protegiendo al controlador de registrar centroides humanos.
## Experimento 5: Agnostic NMS y Ambigüedad de Clases Solapadas (`agnostic_nms`)

- **Tiempo Asignado:** 10 minutos.
- **Concepto Teórico:** Por defecto en OpenCV y YOLO, NMS es _class-specific_: solo suprime cajas si pertenecen a la **misma clase**. Si una herramienta con reflejos produce dos hipótesis simultáneas con cajas casi idénticas (por ejemplo, clase `cell phone` al 49% y clase `remote` al 47%), NMS convencional conservará ambas cajas encimadas. Al habilitar `agnostic_nms=True`, la supresión se aplica puramente por geometría espacial, sin importar la etiqueta asignada.
### Procedimiento en Código

Configura:

```Python
CONF_UMBRAL = 0.20
IOU_NMS = 0.40
CLASES_FILTRADAS = None
AGNOSTIC_NMS = False # Prueba 5A
```

Luego cambia a:


```Python
AGNOSTIC_NMS = True  # Prueba 5B
```

### Dinámica de Laboratorio
- Apunta a un objeto híbrido o metálico con bordes rectos (una libreta, estuche o herramienta dentro del cajón) bajo luz indirecta.
- Observa cómo con `AGNOSTIC_NMS = False` se generan cajas duplicadas parpadeantes con nombres distintos sobre el mismo objeto. Al activar `AGNOSTIC_NMS = True`, sobrevive únicamente la hipótesis con el score más alto.

## Experimento 6: Cuantización y Media Precisión FP16 (`half`) en Hardware Embebido
- **Tiempo Asignado:** 8 minutos.
- **Concepto Teórico:** Los pesos sinápticos de `yolov8n.pt` se almacenan en formato de coma flotante de precisión simple de 32 bits (FP32). La media precisión FP16 reduce el ancho de palabra a 16 bits (1 bit de signo, 5 bits de exponente, 10 bits de mantisa):
    $$\text{VRAM}_{\text{FP16}} \approx \frac{1}{2} \text{VRAM}_{\text{FP32}}$$
    
    En tarjetas gráficas con núcleos Tensor (Tensor Cores de Nvidia), FP16 duplica el rendimiento de operaciones por segundo sin degradación perceptual en la detección. No obstante, en arquitecturas CPU puras x86 sin extensiones nativas AVX-512 FP16, PyTorch debe emular la aritmética de 16 bits, lo que puede inducir sobrecostos de conversión (_casting overhead_).
### Procedimiento en Código

Modifica en `lab_yolo_experimentos.py` la variable de precisión:
```Python
USAR_HALF = True   # Conmuta el cómputo del modelo a FP16
```

Guarda con `Ctrl + S`, reinicia el servidor en terminal (`Ctrl + C` $\rightarrow$ `python lab_yolo_experimentos.py`) y recarga el navegador.
### Dinámica de Laboratorio

- Ejecuta el script observando la latencia en milisegundos en la barra superior del monitor. 
- Si el equipo corre sobre la CPU del host a través de WSL2, observa si la latencia aumenta marginalmente o si la consola imprime una advertencia de compatibilidad.
- Si algún equipo dispone de aceleración CUDA por GPU mediante WSL2 (`nvidia-smi` activo), comparen el salto de velocidad frente a CPU.
### Registro de Datos del Estudiante

| **Modo de Precisión**        | **Tipo de Procesador (CPU / GPU CUDA)** | **Latencia Media (ms)** | **Tasa de Cuadros (FPS)** | **Diagnóstico Operativo**                   |
| ---------------------------- | --------------------------------------- | ----------------------- | ------------------------- | ------------------------------------------- |
| **FP32** (`USAR_HALF=False`) | CPU Intel/AMD (WSL2)                    |                         |                           | Estándar seguro para CPU.                   |
| **FP16** (`USAR_HALF=True`)  | CPU Intel/AMD (WSL2)                    |                         |                           | Evaluar si existe sobrecosto por emulación. |
| **FP16** (`USAR_HALF=True`)  | GPU Nvidia (Tensor Cores)               |                         |                           | Reducción drástica de latencia.             |

## Experimento 7: Control de Latencia y Peor Caso Determinístico (`max_det`)

- **Tiempo Asignado:** 8 minutos.
- **Concepto Teórico:** En sistemas ciberfísicos de tiempo real, el bucle de control exige **determinismo temporal**. Por defecto, YOLOv8 evalúa y post-procesa hasta 300 cajas simultáneas (`max_det=300`). En una escena industrial caótica (cables, tornillos, reflejos en el fondo), procesar cientos de candidatos satura el buffer de salida y dispara la fluctuación de la latencia (_jitter_). Acotar `max_det` fija una cota superior estricta para el tiempo de respuesta del sistema.

### Procedimiento en Código

Configura un escenario propenso a saturación forzando un umbral de confianza mínimo:
```Python
CONF_UMBRAL = 0.05       # Provoca sobre-generación de cajas candidatas
CLASES_FILTRADAS = None
MAX_DETECCIONES = 300    # Prueba 7A: Límite por defecto
```

Luego modifica a:
```Python
MAX_DETECCIONES = 3      # Prueba 7B: Límite estricto para el manipulador robótico
```

### Dinámica de Laboratorio

- Apunta la cámara RealSense hacia un área con textura densa (una mochila, teclado o interior del cajón con herramientas desordenadas).
- Con `MAX_DETECCIONES = 300`, observa cómo la pantalla se satura de rectángulos y la tasa de FPS desciende notablemente.
- Con `MAX_DETECCIONES = 3`, comprueba que el algoritmo suprime en el nivel de C++ todo candidato secundario, reteniendo únicamente las tres hipótesis de mayor score y restaurando la estabilidad del ciclo de reloj.
## Experimento 8: Geometría de Tensores: Píxeles Absolutos (`xyxy`) vs. Coordenadas Normalizadas (`xywhn`)

- **Tiempo Asignado:** 8 minutos.
- **Concepto Teórico:** Los algoritmos de dibujo matricial en OpenCV operan con píxeles enteros absolutos $[x_1, y_1, x_2, y_2] \in \mathbb{N}^4$. Sin embargo, los formatos de intercambio de datos para Deep Learning (como los archivos `.txt` de YOLO que exportarán en Roboflow para el Entregable A6) almacenan las coordenadas de forma normalizada e invariante a la escala:
    $$x_n = \frac{c_x}{W}, \quad y_n = \frac{c_y}{H}, \quad w_n = \frac{w_{\text{px}}}{W}, \quad h_n = \frac{h_{\text{px}}}{H} \quad \in [0, 1]$$
   
### Procedimiento en Código

Localiza el bucle `for b in cajas:` dentro de `lab_yolo_experimentos.py` y agrega la lectura del tensor normalizado:
```Python
# Extracción de coordenadas normalizadas nativas
xn, yn, wn, hn = b.xywhn[0].tolist()

# Impresión sobre la terminal para inspección directa
print(f"[YOLO Tensor] Centro Relativo: ({xn:.3f}, {yn:.3f}) | Dimensiones Relativas: ({wn:.3f}, {hn:.3f})")
```

### Dinámica de Laboratorio

- Guarda y ejecuta el script observando la consola de Ubuntu.
- Centra un objeto en el encuadre: verifica que $(x_n, y_n)$ marque valores cercanos a $(0.500, 0.500)$, sin importar si la resolución de la cámara es $640 \times 480$ o si el modelo infiere a $320 \times 320$.
- **Conexión con el Entregable A6:** Los alumnos comprenderán por qué un dataset etiquetado a una resolución puede reentrenarse a cualquier otra sin rehacer las cajas manuales.

## Experimento 9: Filtrado Temporal de Telemetría: Media Móvil Exponencial (EMA) sobre $Z$

- **Tiempo Asignado:** 10 minutos.
- **Concepto Teórico:** El ancho en píxeles $w_{\text{px}} = x_2 - x_1$ estimado por la red convolucional sufre fluctuaciones estocásticas de $\pm 2\text{ px}$ debido al ruido fotónico del sensor y la discretización del stride. Dada la relación no lineal del modelo Pinhole:
    $$Z = \frac{f_x \cdot W_{\text{real}}}{w_{\text{px}}}$$
    
    una pequeña variación en píxeles introduce oscilaciones de varios centímetros en la distancia calculada. Para estabilizar el guiado sin agregar latencia crítica por buffers circulares, implementamos un filtro de Media Móvil Exponencial (filtro paso-bajas de primer orden):
    $$Z_k^{\text{filt}} = \alpha \cdot Z_k^{\text{raw}} + (1 - \alpha) \cdot Z_{k-1}^{\text{filt}}, \quad \alpha \in (0, 1]$$
    
### Procedimiento en Código

1. En la zona de variables globales de `lab_yolo_experimentos.py`, añade la memoria del filtro:

```Python
Z_FILTRADA = None
ALFA_EMA = 0.25   # Factor de ponderación: valores bajos = mayor suavizado
```

2. Dentro de la función `procesar_frame(frame)`, sustituye la asignación directa de `z_est_cm` por:

```Python
global Z_FILTRADA
z_raw_cm = ((FOCAL_PX * ANCHO_REAL_OBJETO_M) / ancho_px) * 100.0

if Z_FILTRADA is None:
    Z_FILTRADA = z_raw_cm
else:
    Z_FILTRADA = (ALFA_EMA * z_raw_cm) + ((1.0 - ALFA_EMA) * Z_FILTRADA)

z_est_cm = Z_FILTRADA
```

### Dinámica de Laboratorio

- Deja la pieza inmóvil sobre la mesa a unos $25\text{ cm}$ de la cámara.
- **Prueba 9A:** Con $\alpha = 1.0$ (sin filtro), observa cómo el último dígito de la distancia oscila erráticamente ($\pm 3\text{ cm}$).
- **Prueba 9B:** Con $\alpha = 0.05$ (filtro pesado), la lectura es inmóvil, pero al acercar bruscamente la pieza se observa un retardo visual notable (_lag_ de fase).
- **Prueba 9C:** Ajusta $\alpha = 0.25$ para balancear inmunidad al ruido y respuesta transitoria.
## Experimento 10: Cerrojado Lógico con Persistencia Temporal (_Debounce Watchdog_) para el Continuum

- **Tiempo Asignado:** 10 minutos.
- **Concepto Teórico:** En celdas de automatización ciberfísica, un actuador de alta energía (como los servomotores que tensan los tendones del brazo flexible) **nunca** debe dispararse ante un único fotograma afirmativo. Un falso positivo transitorio provocado por un reflejo de luz dispararía el brazo en el vacío. Se implementa un contador de histéresis temporal (_debounce gate_): la condición espacial debe persistir de manera continua durante al menos $N$ cuadros consecutivos ($N \ge 15$, equivalente a $\approx 500\text{ ms}$) para validar el acoplamiento.
### Procedimiento en Código

1. Define las variables del cerrojo en la cabecera:
```Python
CONTEO_PERSISTENCIA = 0
UMBRAL_CUADROS_SEGUROS = 15 # Requiere medio segundo continuo de alineación estable
ESTADO_ACELERADOR_ACTUADORES = False
```
2. Sustituye la lógica de activación por el acumulador temporal:


```Python
global CONTEO_PERSISTENCIA, ESTADO_ACELERADOR_ACTUADORES

# Condición espacial instantánea
condicion_actual = centrado and rango_z

if condicion_actual:
    CONTEO_PERSISTENCIA += 1
else:
    CONTEO_PERSISTENCIA = max(0, CONTEO_PERSISTENCIA - 2) # Penalización por pérdida de visión

# Disparador biestable con histéresis
if CONTEO_PERSISTENCIA >= UMBRAL_CUADROS_SEGUROS:
    ESTADO_ACELERADOR_ACTUADORES = True
elif CONTEO_PERSISTENCIA == 0:
    ESTADO_ACELERADOR_ACTUADORES = False

# Feedback visual de la compuerta lógica
if ESTADO_ACELERADOR_ACTUADORES:
    color = (0, 255, 0)
    texto_status = f"INTERLOCK LIBERADO: ENVIAR COMANDO A TENDONES ({CONTEO_PERSISTENCIA})"
else:
    color = (0, 165, 255)
    texto_status = f"VALIDANDO ALINEACION: {CONTEO_PERSISTENCIA}/{UMBRAL_CUADROS_SEGUROS}"
```

### Dinámica de Laboratorio
- Pasa tu mano rápidamente frente a la cámara atravesando la zona de acople.
- Comprueba que, a pesar de que el objeto entró instantáneamente en la caja central por 1 o 2 cuadros, el interlock mecánico **permanece bloqueado**, evitando un accionamiento falso.
- Coloca el objeto en la posición de acople y mantenlo firme: observa cómo la barra acumula cuadros progresivamente hasta encender el indicador verde de inserción segura.
### Matriz de Calibración Final para el Proyecto

Al concluir los 10 experimentos, los estudiantes deben consolidar la configuración base que emplearán en su nodo de control para las pruebas de noviembre:
```Python
# Configuración balanceada de grado industrial recomendada
CONF_UMBRAL = 0.45
IOU_NMS = 0.45
RESOLUCION_INFERENCIA = 640
CLASES_FILTRADAS = None      # Se limitará al ID de 'pieza' tras entrenar A7
AGNOSTIC_NMS = True
USAR_HALF = False
MAX_DETECCIONES = 5
ALFA_EMA = 0.25
UMBRAL_CUADROS_SEGUROS = 15
```
