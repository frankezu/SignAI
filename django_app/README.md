# Detector de Lengua de Señas Chilena - Aplicación Web Django

Esta aplicación web permite la detección en tiempo real de señas del alfabeto chileno usando tu modelo YOLO entrenado.

## Características

- 🎯 **Detección en tiempo real** usando YOLO + MediaPipe
- 🤖 **Optimización inteligente** - Solo procesa cuando detecta una mano
- 🌐 **Interfaz web moderna** y responsive
- ⚡ **Alto rendimiento** con streaming de video
- 🎨 **Visualización clara** con bounding boxes y porcentajes de confianza

## Requisitos del Sistema

- Python 3.8 o superior
- Cámara web funcional
- Modelo entrenado en `runs/detect/train/weights/best.pt`

## Instalación

1. **Navegar al directorio de la aplicación:**
   ```bash
   cd django_app
   ```

2. **Crear un entorno virtual (recomendado):**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # En Windows
   # source venv/bin/activate  # En Linux/Mac
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Aplicar migraciones de Django:**
   ```bash
   python manage.py migrate
   ```

## Uso

1. **Iniciar el servidor de desarrollo:**
   ```bash
   python manage.py runserver
   ```

2. **Abrir el navegador y visitar:**
   ```
   http://127.0.0.1:8000/
   ```

3. **Usar la aplicación:**
   - Haz clic en "Iniciar Detector"
   - Permite el acceso a la cámara si el navegador lo solicita
   - Coloca tu mano frente a la cámara
   - Realiza señas del alfabeto chileno
   - Observa las detecciones en tiempo real

## Estructura del Proyecto

```
django_app/
├── manage.py
├── requirements.txt
├── sign_detector/          # Configuración principal de Django
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── detector/               # Aplicación del detector
│   ├── __init__.py
│   ├── apps.py
│   ├── views.py           # Lógica principal del detector
│   ├── urls.py
│   ├── models.py
│   ├── admin.py
│   └── tests.py
└── templates/
    └── detector/
        └── index.html     # Interfaz web principal
```

## Funcionamiento Técnico

### Backend (Django)
- **SignDetector Class**: Maneja la inicialización del modelo YOLO y MediaPipe
- **Video Streaming**: Utiliza `StreamingHttpResponse` para transmisión en tiempo real
- **API Endpoints**: 
  - `/video_feed/` - Stream de video procesado
  - `/start_detection/` - Iniciar detección
  - `/stop_detection/` - Detener detección

### Frontend
- **Interfaz Responsive**: Adaptable a diferentes tamaños de pantalla
- **Control de Estado**: Manejo inteligente de botones y estado de la aplicación
- **Visualización en Tiempo Real**: Muestra el video con detecciones superpuestas

### Proceso de Detección
1. **Captura de Frame**: La cámara captura frames a 30 FPS
2. **Filtrado MediaPipe**: Solo procesa frames donde se detecta una mano
3. **Inferencia YOLO**: Predice la seña cuando hay una mano presente
4. **Post-procesamiento**: Dibuja bounding boxes y etiquetas
5. **Streaming**: Envía el frame procesado al navegador

## Personalización

### Ajustar Confianza
En `detector/views.py`, línea ~45:
```python
conf=0.4,  # Cambiar este valor (0.1 - 1.0)
```

### Modificar Colores
En `detector/views.py`, líneas ~65-66:
```python
color = (0, 255, 0) if conf > 0.7 else (0, 255, 255)  # Verde/Amarillo
```

### Cambiar Resolución
En `detector/views.py`, líneas ~32-33:
```python
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Ancho
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Alto
```

## Solución de Problemas

### Error: "No se puede cargar el modelo"
- Verifica que el archivo `runs/detect/train/weights/best.pt` existe
- Asegúrate de que la ruta en `settings.py` es correcta

### Error: "No se puede acceder a la cámara"
- Verifica que la cámara no esté siendo usada por otra aplicación
- En algunos navegadores, necesitas HTTPS para acceso a cámara

### Rendimiento lento
- Reduce la resolución de video
- Aumenta el threshold de confianza
- Verifica que MediaPipe esté filtrando correctamente

## Licencia

Este proyecto utiliza el modelo YOLO entrenado específicamente para lengua de señas chilena.

## Soporte

Para problemas técnicos o preguntas sobre el modelo, consulta la documentación del proyecto principal.
