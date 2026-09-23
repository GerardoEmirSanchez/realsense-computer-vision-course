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