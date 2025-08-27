# Detector de Lengua de Señas Chilena - Modo Entrenamiento

## 🎯 Nueva Funcionalidad: Modo Entrenamiento

Hemos agregado un **Modo Entrenamiento** interactivo que permite a los usuarios practicar las señas del alfabeto chileno de manera guiada.

### ✨ Características del Modo Entrenamiento

- **🎲 Letras Aleatorias**: El sistema selecciona una letra al azar para practicar
- **⏱️ Temporizador de 10 segundos**: Tienes exactamente 10 segundos para hacer la seña correcta
- **📸 Imágenes de Referencia**: Se muestra una imagen real del dataset4 como ejemplo
- **✅ Retroalimentación en Tiempo Real**: 
  - ✅ Verde cuando haces la seña correcta
  - ❌ Rojo cuando la seña es incorrecta
  - 📍 Indicación de qué letra estás haciendo si es incorrecta
- **🎨 Interfaz Intuitiva**: Modo fácil de cambiar entre detección normal y entrenamiento

### 🚀 Cómo Usar el Modo Entrenamiento

1. **Inicia la aplicación**:
   ```bash
   # Ejecuta el script de Windows
   run_training_mode.bat
   
   # O manualmente:
   .venv\Scripts\activate
   cd django_app
   python manage.py runserver
   ```

2. **Accede a la aplicación**: Abre http://127.0.0.1:8000/ en tu navegador

3. **Cambia al Modo Entrenamiento**: Haz clic en el botón "Modo Entrenamiento"

4. **Inicia una sesión de entrenamiento**:
   - Haz clic en "Letra Aleatoria"
   - Se mostrará una letra objetivo y una imagen de referencia
   - Tienes 10 segundos para hacer la seña
   - La cámara te dará retroalimentación visual

### 📁 Estructura de Archivos Actualizada

```
django_app/
├── detector/
│   ├── views.py          # ✨ Actualizado con lógica de entrenamiento
│   ├── urls.py           # ✨ Nuevas rutas para entrenamiento
│   └── ...
├── templates/detector/
│   └── index.html        # ✨ Nueva interfaz con dos modos
└── ...

dataset4/                 # 📸 Fuente de imágenes de referencia
├── train/
│   ├── images/          # Imágenes utilizadas como referencia
│   └── labels/          # Etiquetas para mapear letras
└── data.yaml            # Configuración de clases A-Z
```

### 🔧 Nuevos Endpoints API

```javascript
// Obtener letra aleatoria con imagen de referencia
GET /get_random_letter/

// Iniciar sesión de entrenamiento
POST /start_training/
{
  "letter": "A"
}

// Detener entrenamiento
POST /stop_training/

// Servir imagen de referencia
GET /reference_image/{letter}/{filename}
```

### 🎮 Funciones del Frontend

- **switchMode(mode)**: Cambiar entre modo detección y entrenamiento
- **startRandomTraining()**: Iniciar nueva sesión con letra aleatoria
- **stopTraining()**: Detener sesión actual
- **startTrainingTimer(duration)**: Temporizador visual de 10 segundos

### 📊 Mejoras Técnicas

1. **Mapeo Inteligente de Imágenes**:
   - Busca patrones como `L1_jpg.rf.*.jpg`, `M-1-_jpeg_jpg.rf.*.jpg`
   - Selecciona hasta 3 imágenes aleatorias por letra
   - Carga automática al inicializar la aplicación

2. **Procesamiento de Frames Mejorado**:
   - Código de colores: Verde (correcto), Rojo (incorrecto)
   - Mensajes de retroalimentación en tiempo real
   - Temporizador visual integrado en el video

3. **Interfaz Responsiva**:
   - Diseño adaptable para diferentes dispositivos
   - Sección lateral para controles de entrenamiento
   - Indicadores visuales claros de estado

### 🐛 Solución de Problemas

**Error: "No se encuentran imágenes de referencia"**
- Verifica que el directorio `dataset4/train/images/` existe
- Asegúrate de que hay archivos .jpg en el directorio

**Error: "Letra aleatoria no funciona"**
- Revisa la consola del navegador para errores JavaScript
- Verifica que el servidor Django esté ejecutándose

**Error: "La cámara no inicia"**
- Asegúrate de que tienes permisos de cámara en el navegador
- Verifica que no hay otras aplicaciones usando la cámara

### 📈 Estadísticas Cargadas

El sistema actualmente carga imágenes de referencia para **17 letras** del alfabeto chileno desde el dataset4.

### 🎉 ¡Empezar a Entrenar!

1. Ejecuta `run_training_mode.bat`
2. Ve a http://127.0.0.1:8000/
3. Haz clic en "Modo Entrenamiento"
4. ¡Presiona "Letra Aleatoria" y comienza a practicar!

---

**¡Disfruta practicando lengua de señas chilena con tu nuevo entrenador personal de IA!** 🤟
