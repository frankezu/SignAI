## Detector de Lenguaje de Señas Chileno (LSCh)

Detector en tiempo real de letras del alfabeto en lenguaje de señas chileno usando YOLO + MediaPipe.

## Requisitos

### Python
- **Python 3.12.8** - [Descargar aquí](https://www.python.org/downloads/release/python-3128/)

### Configuración del entorno virtual

#### 1. Dar permisos en PowerShell (solo la primera vez):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 2. Activar el entorno virtual:
```powershell
.\venv\Scripts\Activate.ps1
```

#### 3. Instalar dependencias:

Con el entorno virtual activado, ejecutar:

```bash
pip install ultralytics mediapipe opencv-python numpy
```

O instalar paso a paso:

```bash
pip install ultralytics
pip install mediapipe  
pip install opencv-python
pip install numpy
```

## Uso

1. **Dar permisos en PowerShell** (solo primera vez)
2. **Activar el entorno virtual** con `.\venv\Scripts\Activate.ps1`
3. **Instalar dependencias** con pip
4. Ejecutar el notebook `model.ipynb`
3. La primera celda importa las librerías
4. La segunda celda inicia el detector con cámara
5. Presionar 'q' para salir

## Características

- ✅ Detección en tiempo real de 26 letras (A-Z)
- ✅ Sistema híbrido YOLO + MediaPipe
- ✅ Verificación cruzada para mayor precisión
- ✅ Interfaz visual con porcentaje de confianza
- ✅ Optimizado para lenguaje de señas de una mano