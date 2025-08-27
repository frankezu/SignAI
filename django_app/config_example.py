# Configuración para desarrollo local
# Copia este archivo y renómbralo a 'local_settings.py' para personalizar

# Configuración de debug
DEBUG = True

# Hosts permitidos (agregar tu IP local si necesitas acceso desde otros dispositivos)
ALLOWED_HOSTS = ['localhost', '127.0.0.1', '192.168.1.100']  # Ajusta según tu red

# Configuración de cámara
CAMERA_INDEX = 0  # Cambiar si tienes múltiples cámaras
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Configuración del modelo
MODEL_CONFIDENCE = 0.4  # Threshold de confianza (0.1 - 1.0)
HIGH_CONFIDENCE = 0.7   # Threshold para color verde

# MediaPipe configuración
MP_MIN_DETECTION_CONFIDENCE = 0.6
MP_MIN_TRACKING_CONFIDENCE = 0.3

# Configuración de streaming
STREAM_QUALITY = 'medium'  # 'low', 'medium', 'high'
