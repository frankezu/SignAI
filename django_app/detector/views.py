from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import cv2
# import mediapipe as mp
from ultralytics import YOLO
import threading
import json
import time
import random
import os
import glob
from datetime import datetime, timedelta
from collections import deque
import numpy as np

class SignDetector:
    def __init__(self):
        self.model = None
        self.mp_hands = None
        self.hands = None
        self.cap = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # Estado del modo de entrenamiento
        self.training_mode = False
        self.target_letter = None
        self.training_start_time = None
        self.training_duration = 10  # 10 segundos
        self.reference_images = {}
        
        # Configuración del mecanismo de avance automático
        self.correct_detection_start = None  # Cuando empezó la detección correcta
        self.auto_advance_duration = 5  # 5 segundos de detección correcta para avanzar
        self.last_detected_letter = None
        self.should_auto_advance = False
        
        # Memoria intermedia para estabilizar predicciones
        self.prediction_buffer = deque(maxlen=5)
        
        # Métricas de rendimiento
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Región de Interés (ROI) dinámica basada en detección de extremidad
        self.roi_buffer = deque(maxlen=3)
        
        self.load_reference_images()
        
    def load_reference_images(self):
        """Inicializa la carga de imágenes de referencia desde el conjunto de datos."""
        try:
            dataset_path = os.path.join(os.path.dirname(settings.BASE_DIR), 'dataset', 'train', 'images')
            
            # Asignación de clases según configuración de data.yaml
            class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
            
            # Iteración para localizar imágenes correspondientes a cada clase
            for letter in class_names:
                letter_images = []
                
                # Búsqueda mediante patrones de nomenclatura
                patterns = [
                    f"{letter}[0-9]*_*.jpg",         # L1_jpg, L10_jpg, etc.
                    f"{letter}-*_*.jpg",             # M-1-_jpeg_jpg, etc.
                    f"{letter.lower()}-*_*.jpg",     # Búsqueda de variantes en minúsculas con separador
                    f"{letter.lower()}_*.jpg",       # Búsqueda de variantes en minúsculas con separador bajo
                    f"{letter}*.jpg"                 # Cualquier archivo que empiece con la letra
                ]
                
                for pattern in patterns:
                    try:
                        image_files = glob.glob(os.path.join(dataset_path, pattern))
                        letter_images.extend(image_files)
                    except:
                        continue
                
                # Eliminación de entradas duplicadas
                letter_images = list(set(letter_images))
                
                if letter_images:
                    # Selección aleatoria de un máximo de 3 imágenes por clase
                    self.reference_images[letter] = random.sample(
                        letter_images, 
                        min(3, len(letter_images))
                    )
                    
            print(f"Cargadas imágenes de referencia para {len(self.reference_images)} letras")
            
        except Exception as e:
            print(f"Error cargando imágenes de referencia: {e}")
            self.reference_images = {}
    
    def reset_training_state(self):
        """Restablece el estado de entrenamiento para un nuevo objetivo."""
        self.correct_detection_start = None
        self.should_auto_advance = False
        self.last_detected_letter = None
    
    def get_reference_image_for_letter(self, letter):
        """Recupera una imagen de referencia aleatoria para la clase especificada."""
        if letter in self.reference_images and self.reference_images[letter]:
            return random.choice(self.reference_images[letter])
        return None
    
    def get_hand_roi(self, frame, hand_landmarks):
        """Calcula la región de interés (ROI) delimitada por los puntos clave de la extremidad."""
        h, w = frame.shape[:2]
        
        # Extracción de coordenadas espaciales de los puntos clave
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        # Determinación de la caja delimitadora con margen adicional
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Incorporación de un margen de seguridad del 20%
        padding_x = int((x_max - x_min) * 0.2)
        padding_y = int((y_max - y_min) * 0.2)
        
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(w, x_max + padding_x)
        y_max = min(h, y_max + padding_y)
        
        return (x_min, y_min, x_max, y_max)
    
    def smooth_prediction(self, letter, confidence):
        """Aplica un filtro de suavizado sobre las predicciones recientes."""
        self.prediction_buffer.append((letter, confidence))
        
        # Verificación de muestras suficientes en el buffer
        if len(self.prediction_buffer) >= 3:
            # Cálculo de frecuencia de ocurrencia por clase
            letter_counts = {}
            
            for pred_letter, pred_conf in self.prediction_buffer:
                if pred_letter not in letter_counts:
                    letter_counts[pred_letter] = {'count': 0, 'total_conf': 0}
                letter_counts[pred_letter]['count'] += 1
                letter_counts[pred_letter]['total_conf'] += pred_conf
            
            # Identificación de la clase predominante
            best_letter = max(letter_counts.keys(), 
                            key=lambda x: letter_counts[x]['count'])
            avg_conf = letter_counts[best_letter]['total_conf'] / letter_counts[best_letter]['count']
            
            return best_letter, avg_conf
        
        return letter, confidence
    
    def calculate_fps(self):
        """Calcula los fotogramas por segundo (FPS) de forma continua."""
        self.frame_count += 1
        if self.frame_count % 10 == 0:  # Frecuencia de actualización establecida en 10 fotogramas
            current_time = time.time()
            elapsed = current_time - self.fps_start_time
            self.current_fps = 10 / elapsed
            self.fps_start_time = current_time
    
    def draw_info_panel(self, frame, letter=None, confidence=None, hand_detected=False):
        """Renderiza el panel de información sobre el flujo de video."""
        h, w = frame.shape[:2]
        
        # Elementos gráficos de fondo
        cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 100), (255, 255, 255), 2)
        
        # Despliegue de métricas y estado
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        status = "MANO DETECTADA" if hand_detected else "SIN MANO"
        color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.putText(frame, f"Estado: {status}", 
                   (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        if letter and confidence:
            cv2.putText(frame, f"Letra: {letter} ({confidence:.1%})", 
                       (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def draw_info_panel_simplified(self, frame, letter=None, confidence=None):
        """Renderiza un panel de información simplificado sin telemetría de extremidad."""
        h, w = frame.shape[:2]
        
        # Elementos gráficos de fondo
        cv2.rectangle(frame, (10, 10), (300, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 80), (255, 255, 255), 2)
        
        # Despliegue de métricas y estado
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(frame, "Modo: Solo YOLO", 
                   (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        if letter and confidence:
            cv2.putText(frame, f"Letra: {letter} ({confidence:.1%})", 
                       (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    def initialize(self):
        """Instancia e inicializa el modelo de detección YOLO."""
        try:
            # Carga de los pesos del modelo YOLO
            self.model = YOLO(settings.MODEL_PATH)
            
            # Configuración del dispositivo de captura de video
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)  # Restricción de la tasa de captura a 30 FPS
            
            print("Modelo YOLO cargado exitosamente")
            print("NOTA: MediaPipe no disponible en Python 3.13, usando solo YOLO")
            
            return True
        except Exception as e:
            print(f"Error inicializando detector: {e}")
            return False
    
    def process_frame(self, frame):
        """Ejecuta el pipeline de inferencia sobre un fotograma completo."""
        # Inversión horizontal de la imagen (efecto espejo)
        frame = cv2.flip(frame, 1)
        
        # Despliegue de telemetría para el modo de entrenamiento
        if self.training_mode and self.target_letter and self.training_start_time:
            # Cálculo del tiempo restante de la sesión
            elapsed_time = time.time() - self.training_start_time
            remaining_time = max(0, self.training_duration - elapsed_time)
            
            # Renderizado de instrucciones de la sesión
            cv2.putText(frame, f"Entrena la letra: {self.target_letter}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Condición de término por finalización del tiempo
            if remaining_time <= 0:
                self.training_mode = False
                self.target_letter = None
                self.training_start_time = None
        
        current_letter = None
        current_confidence = None
        
        # Ejecución del modelo YOLO sobre el fotograma sin recorte
        results = self.model.predict(
            source=frame,
            conf=0.3,  # Umbral de confianza permisivo para maximizar recuperaciones
            verbose=False,
            stream=False
        )
        
        # Análisis y selección de detecciones
        best_detection = None
        best_conf = 0
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        cls = int(box.cls[0])
                        letter = self.model.names[cls]
                        
                        # Extracción de coordenadas de la caja delimitadora
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        best_detection = {
                            'letter': letter,
                            'confidence': conf,
                            'bbox': (x1, y1, x2, y2)
                        }
        
        # Renderizado de la detección con mayor grado de confianza
        if best_detection and best_detection['confidence'] > 0.4:
            # Aplicación de estabilización a la predicción
            smooth_letter, smooth_conf = self.smooth_prediction(
                best_detection['letter'], 
                best_detection['confidence']
            )
            
            current_letter = smooth_letter
            current_confidence = smooth_conf
            
            # Renderizado gráfico de la caja delimitadora
            x1, y1, x2, y2 = best_detection['bbox']
            color = (0, 255, 0) if smooth_conf > 0.7 else (0, 255, 255)
            
            # Modificación de color basada en coincidencia con objetivo
            if self.training_mode and self.target_letter:
                if smooth_letter == self.target_letter:
                    color = (0, 255, 0)  # Tonalidad verde indica coincidencia
                else:
                    color = (0, 0, 255)  # Tonalidad roja indica discrepancia
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Renderizado optimizado de la etiqueta
            label = f"{smooth_letter} {smooth_conf:.1%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Renderizado del fondo de la etiqueta
            cv2.rectangle(frame, (x1, y1 - text_h - 10), 
                         (x1 + text_w + 10, y1), color, -1)
            
            # Renderizado del texto de la etiqueta
            cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                       font, font_scale, (0, 0, 0), thickness)
            
            # Lógica específica del modo de entrenamiento interactivo
            if self.training_mode and self.target_letter:
                current_time = time.time()
                
                if smooth_letter == self.target_letter and smooth_conf > 0.6:
                    # Confirmación de predicción correcta
                    if self.correct_detection_start is None:
                        self.correct_detection_start = current_time
                    
                    # Cálculo de la duración sostenida de la predicción correcta
                    correct_duration = current_time - self.correct_detection_start
                    remaining_time = max(0, self.auto_advance_duration - correct_duration)
                    
                    # Notificación visual con temporizador de transición
                    if remaining_time > 0:
                        msg = f"CORRECTO! Siguiente en: {remaining_time:.1f}s"
                        text_x = max(10, frame.shape[1]//2 - 200)
                        cv2.putText(frame, msg, 
                                   (text_x, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        # Umbral superado; activando indicador de transición automática
                        self.should_auto_advance = True
                        msg = "¡Perfecto! Cambiando letra..."
                        text_x = max(10, frame.shape[1]//2 - 150)
                        cv2.putText(frame, msg, 
                                   (text_x, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    # Gestión de predicción incorrecta
                    self.correct_detection_start = None
                    
                    if smooth_letter and smooth_letter != self.target_letter:
                        msg = f"Incorrecto. Haces: {smooth_letter.upper()}"
                        text_y = max(frame.shape[0] - 50, 50)
                        cv2.putText(frame, msg, 
                                   (10, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Registro de la predicción más reciente
                self.last_detected_letter = smooth_letter
        else:
            # Purgado de la memoria intermedia ante ausencia de detecciones
            self.prediction_buffer.clear()
            
            # Restablecimiento del estado ante pérdida de seguimiento
            if self.training_mode:
                self.correct_detection_start = None
        
        # Renderizado del panel informativo (variante básica)
        self.draw_info_panel_simplified(frame, current_letter, current_confidence)
        
        # Actualización de métricas de rendimiento
        self.calculate_fps()
        
        return frame
    
    def generate_frames(self):
        """Genera un flujo continuo de fotogramas para transmisión web."""
        while self.is_running:
            with self.lock:
                if self.cap is None or not self.cap.isOpened():
                    break
                    
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Ejecución del pipeline de procesamiento visual
                processed_frame = self.process_frame(frame)
                
                # Codificación del fotograma a formato JPEG
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    def start(self):
        """Endpoint para iniciar el motor de detección."""
        with self.lock:
            if not self.is_running:
                if self.initialize():
                    self.is_running = True
                    return True
            return False
    
    def stop(self):
        """Endpoint para detener el motor de detección."""
        with self.lock:
            self.is_running = False
            if self.cap:
                self.cap.release()
                self.cap = None

# Instancia global del servicio de detección
detector = SignDetector()

def index(request):
    """Renderiza la vista principal de la aplicación."""
    return render(request, 'detector/index.html')

def video_feed(request):
    """Provee la conexión de transmisión de video en tiempo real."""
    if not detector.is_running:
        detector.start()
    
    return StreamingHttpResponse(
        detector.generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

@csrf_exempt
def start_detection(request):
    """Endpoint para iniciar el motor de detección."""
    if request.method == 'POST':
        success = detector.start()
        return JsonResponse({'success': success})
    return JsonResponse({'success': False})

@csrf_exempt
def stop_detection(request):
    """Endpoint para detener el motor de detección."""
    if request.method == 'POST':
        detector.stop()
        return JsonResponse({'success': True})
    return JsonResponse({'success': False})

@csrf_exempt
def start_training(request):
    """Endpoint para iniciar una sesión de entrenamiento."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            letter = data.get('letter', '').upper()
            
            # Validación de la entrada recibida
            if not letter or letter not in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']:
                return JsonResponse({'success': False, 'error': 'Letra inválida'})
            
            # Verificación del estado del servicio de detección
            if not detector.is_running:
                detector.start()
            
            # Inicialización del contexto para la sesión de entrenamiento
            detector.training_mode = True
            detector.target_letter = letter
            detector.training_start_time = time.time()
            detector.reset_training_state()  # Reinicialización del estado para un nuevo objetivo
            
            # Recuperación del recurso visual de referencia
            reference_image_path = detector.get_reference_image_for_letter(letter)
            reference_image_url = None
            
            if reference_image_path:
                # Construcción de la ruta accesible para el recurso visual
                relative_path = os.path.relpath(reference_image_path, os.path.dirname(settings.BASE_DIR))
                reference_image_url = f"/reference_image/{letter}/{os.path.basename(reference_image_path)}"
            
            return JsonResponse({
                'success': True, 
                'letter': letter,
                'duration': detector.training_duration,
                'reference_image': reference_image_url
            })
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False})

@csrf_exempt
def stop_training(request):
    """Endpoint para interrumpir la sesión de entrenamiento actual."""
    if request.method == 'POST':
        detector.training_mode = False
        detector.target_letter = None
        detector.training_start_time = None
        detector.reset_training_state()
        return JsonResponse({'success': True})
    return JsonResponse({'success': False})

@csrf_exempt
def check_auto_advance(request):
    """Endpoint para consultar el estado de transición automática."""
    if request.method == 'GET':
        if detector.should_auto_advance:
            detector.should_auto_advance = False  # Restablecimiento del indicador de transición
            return JsonResponse({'should_advance': True})
        return JsonResponse({'should_advance': False})
    return JsonResponse({'success': False})

@csrf_exempt
def get_random_letter(request):
    """Endpoint para solicitar un objetivo aleatorio de entrenamiento."""
    if request.method == 'GET':
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        random_letter = random.choice(letters)
        
        # Recuperación del recurso visual de referencia
        reference_image_path = detector.get_reference_image_for_letter(random_letter)
        reference_image_url = None
        
        if reference_image_path:
            # Crear URL relativa para servir la imagen
            relative_path = os.path.relpath(reference_image_path, os.path.dirname(settings.BASE_DIR))
            reference_image_url = f"/reference_image/{random_letter}/{os.path.basename(reference_image_path)}"
        
        return JsonResponse({
            'letter': random_letter,
            'reference_image': reference_image_url
        })
    return JsonResponse({'success': False})

@csrf_exempt
def serve_reference_image(request, letter, filename):
    """Endpoint para servir los archivos de imagen de referencia."""
    try:
        dataset_path = os.path.join(os.path.dirname(settings.BASE_DIR), 'dataset', 'train', 'images')
        image_path = os.path.join(dataset_path, filename)
        
        if os.path.exists(image_path):
            with open(image_path, 'rb') as image_file:
                from django.http import HttpResponse
                response = HttpResponse(image_file.read(), content_type="image/jpeg")
                response['Content-Disposition'] = f'inline; filename="{filename}"'
                return response
        else:
            from django.http import Http404
            raise Http404("Imagen no encontrada")
            
    except Exception as e:
        from django.http import Http404
        raise Http404("Error al cargar imagen")
