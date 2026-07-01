# Detector de Lengua de Señas Chilena (LSCh)

Proyecto final (Capstone Project) desarrollado para el programa **[Samsung Innovation Campus](https://innovationcampus.cl/): Inteligencia Artificial** (edición 2025), un bootcamp intensivo de 260 horas enfocado en el ciclo completo de IA. Presentado en las oficinas de Samsung Chile, Santiago.

El sistema detecta en tiempo real las 26 letras del alfabeto en lengua de señas chilena mediante un enfoque híbrido YOLO + MediaPipe (entrenado con un dataset extraído de Roboflow), desplegado como aplicación web con Django.

## Equipo de Desarrollo

- Franco Bernal
- Hugo Palomino
- Ariel Leiva
- Paula Henríquez

## Funcionalidades Principales

- **Detección en tiempo real:** Clasificación de las 26 letras (A-Z) a alta velocidad.
- **Modo Entrenamiento:** Módulo interactivo que propone señas aleatorias al usuario con límite de tiempo y feedback visual para practicar.
- **Arquitectura Web:** Detección impulsada por visión computacional y empaquetada en una interfaz limpia e intuitiva mediante Django.

## Requisitos

- **Python 3.12.8** - [Descargar aquí](https://www.python.org/downloads/release/python-3128/)
- Cámara web funcional
- Modelo entrenado en `runs/detect/train/weights/best.pt`

## Instalación

1. Dar permisos en PowerShell (solo la primera vez):
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

2. Crear y activar el entorno virtual:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

**Opción 1: Aplicación Web**
Ejecutar el script `run_server.bat` desde la raíz del proyecto. Este script activará el entorno, aplicará las migraciones e iniciará el servidor automáticamente.
Luego, abre tu navegador en `http://127.0.0.1:8000/`.

**Opción 2: Inferencia Directa (Notebook)**
Para ejecutar la detección de forma independiente (útil para pruebas o debug), activa el entorno virtual y ejecuta el archivo `inference.ipynb`. Presiona la tecla `q` en la ventana de video para salir.

## Estructura Principal

```text
SignAI/
├── dataset/                    # Dataset de entrenamiento y configuración
├── django_app/                 # Aplicación web (interfaz y backend Django)
├── runs/detect/train/weights/  # Carpeta contenedora de los pesos del modelo YOLO
├── inference.ipynb             # Notebook de detección independiente
├── train.ipynb                 # Notebook de entrenamiento
├── requirements.txt            # Dependencias del sistema
└── run_server.bat              # Script ejecutable de inicio rápido
```

---

> **Nota sobre este repositorio:** Como uno de los autores originales y desarrollador principal del proyecto, decidí crear este repositorio luego de que el original quedara inactivo (donde se conserva todo mi historial previo de commits). El objetivo de esta nueva versión es realizar una limpieza profunda del código, refactorizar la arquitectura y mantener una versión oficial, estable y ordenada del sistema.