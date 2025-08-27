from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import cv2
import mediapipe as mp
from ultralytics import YOLO
import threading
import json
import time
import random
import os
import glob
from datetime import datetime, timedelta

class SignDetector:
    def __init__(self):
        self.model = None
        self.mp_hands = None
        self.hands = None
        self.cap = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # Variables para modo entrenamiento
        self.training_mode = False
        self.target_letter = None
        self.training_start_time = None
        self.training_duration = 10  # 10 segundos
        self.reference_images = {}
        
        # Variables para auto-avance
        self.correct_detection_start = None  # Cuando empezó la detección correcta
        self.auto_advance_duration = 5  # 5 segundos de detección correcta para avanzar
        self.last_detected_letter = None
        self.should_auto_advance = False
        
        self.load_reference_images()
        
    def load_reference_images(self):
        """Cargar imágenes de referencia del dataset4"""
        try:
            dataset_path = os.path.join(os.path.dirname(settings.BASE_DIR), 'dataset4', 'train', 'images')
            
            # Mapeo de clases según data.yaml
            class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
            
            # Para cada letra, buscar sus imágenes correspondientes
            for letter in class_names:
                letter_images = []
                
                # Buscar patrones de nombres que contengan la letra
                patterns = [
                    f"{letter}[0-9]*_*.jpg",         # L1_jpg, L10_jpg, etc.
                    f"{letter}-*_*.jpg",             # M-1-_jpeg_jpg, etc.
                    f"{letter.lower()}-*_*.jpg",     # Para letras en minúscula con guión
                    f"{letter.lower()}_*.jpg",       # Para letras en minúscula con guión bajo
                    f"{letter}*.jpg"                 # Cualquier archivo que empiece con la letra
                ]
                
                for pattern in patterns:
                    try:
                        image_files = glob.glob(os.path.join(dataset_path, pattern))
                        letter_images.extend(image_files)
                    except:
                        continue
                
                # Remover duplicados
                letter_images = list(set(letter_images))
                
                if letter_images:
                    # Seleccionar hasta 3 imágenes aleatorias para cada letra
                    self.reference_images[letter] = random.sample(
                        letter_images, 
                        min(3, len(letter_images))
                    )
                    
            print(f"Cargadas imágenes de referencia para {len(self.reference_images)} letras")
            
        except Exception as e:
            print(f"Error cargando imágenes de referencia: {e}")
            self.reference_images = {}
    
    def reset_training_state(self):
        """Resetear estado de entrenamiento para nueva letra"""
        self.correct_detection_start = None
        self.should_auto_advance = False
        self.last_detected_letter = None
    
    def get_reference_image_for_letter(self, letter):
        """Obtener una imagen de referencia aleatoria para una letra"""
        if letter in self.reference_images and self.reference_images[letter]:
            return random.choice(self.reference_images[letter])
        return None
        
    def initialize(self):
        """Inicializar el modelo y MediaPipe"""
        try:
            # Cargar modelo YOLO
            self.model = YOLO(settings.MODEL_PATH)
            
            # MediaPipe para filtrar
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.3
            )
            
            # Configurar cámara
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            return True
        except Exception as e:
            print(f"Error inicializando detector: {e}")
            return False
    
    def process_frame(self, frame):
        """Procesar un frame con detección de señas"""
        frame = cv2.flip(frame, 1)
        
        # Modo entrenamiento
        if self.training_mode and self.target_letter and self.training_start_time:
            # Calcular tiempo restante
            elapsed_time = time.time() - self.training_start_time
            remaining_time = max(0, self.training_duration - elapsed_time)
            
            # Mostrar información del entrenamiento
            cv2.putText(frame, f"Entrena la letra: {self.target_letter}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Si se acabó el tiempo, terminar entrenamiento
            if remaining_time <= 0:
                self.training_mode = False
                self.target_letter = None
                self.training_start_time = None
        
        # Verificar si hay mano
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_results = self.hands.process(rgb_frame)
        hand_present = mp_results.multi_hand_landmarks is not None
        
        # YOLO solo si hay mano
        if hand_present:
            results = self.model.predict(
                source=frame,
                conf=0.4,
                verbose=False,
                stream=False
            )
            
            # Procesar detecciones
            detected_letter = None
            best_confidence = 0
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Obtener datos del bbox
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        letter = self.model.names[cls]
                        
                        # Solo mostrar si confianza es buena
                        if conf > 0.4:
                            # Convertir a enteros
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # Color del bbox
                            color = (0, 255, 0) if conf > 0.7 else (0, 255, 255)
                            
                            # En modo entrenamiento, cambiar color si coincide con letra objetivo
                            if self.training_mode and self.target_letter and letter == self.target_letter:
                                color = (0, 255, 0)  # Verde para letra correcta
                                if conf > best_confidence:
                                    detected_letter = letter
                                    best_confidence = conf
                            elif self.training_mode and self.target_letter and letter != self.target_letter:
                                color = (0, 0, 255)  # Rojo para letra incorrecta
                            
                            # Dibujar bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Preparar texto
                            label = f"{letter} {conf:.1%}"
                            
                            # Calcular tamaño del texto
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.8
                            thickness = 2
                            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                            
                            # Posición del texto (arriba del bbox)
                            text_x = x1
                            text_y = y1 - 10
                            
                            # Si el texto se sale arriba, ponerlo abajo
                            if text_y < text_h:
                                text_y = y2 + text_h + 10
                            
                            # Fondo para el texto
                            cv2.rectangle(frame, 
                                         (text_x, text_y - text_h - 5), 
                                         (text_x + text_w + 5, text_y + 5), 
                                         color, -1)
                            
                            # Texto en negro
                            cv2.putText(frame, label, (text_x + 2, text_y), 
                                       font, font_scale, (0, 0, 0), thickness)
            
            # En modo entrenamiento, mostrar feedback y manejar auto-avance
            if self.training_mode and self.target_letter:
                current_time = time.time()
                
                if detected_letter == self.target_letter and best_confidence > 0.6:
                    # Detección correcta
                    if self.correct_detection_start is None:
                        # Primera vez que detecta correctamente
                        self.correct_detection_start = current_time
                    
                    # Calcular tiempo de detección correcta
                    correct_duration = current_time - self.correct_detection_start
                    remaining_time = max(0, self.auto_advance_duration - correct_duration)
                    
                    # Mostrar mensaje de correcto con countdown
                    if remaining_time > 0:
                        msg = f"CORRECTO! Siguiente en: {remaining_time:.1f}s"
                        text_x = max(10, frame.shape[1]//2 - 200)  # Posición segura
                        cv2.putText(frame, msg, 
                                   (text_x, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        # Han pasado 5 segundos, marcar para auto-avance
                        self.should_auto_advance = True
                        msg = "¡Perfecto! Cambiando letra..."
                        text_x = max(10, frame.shape[1]//2 - 150)  # Posición segura
                        cv2.putText(frame, msg, 
                                   (text_x, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    # Detección incorrecta o no hay detección
                    self.correct_detection_start = None
                    
                    if detected_letter and detected_letter != self.target_letter and len(detected_letter) == 1 and detected_letter.isalpha():
                        # Asegurar que tenemos una letra válida y posición segura
                        msg = f"Incorrecto. Haces: {detected_letter.upper()}"
                        text_y = max(frame.shape[0] - 50, 50)  # Posición segura
                        cv2.putText(frame, msg, 
                                   (10, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Almacenar la última letra detectada para referencia
                self.last_detected_letter = detected_letter
        
        return frame
    
    def generate_frames(self):
        """Generar frames para streaming"""
        while self.is_running:
            with self.lock:
                if self.cap is None or not self.cap.isOpened():
                    break
                    
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Procesar frame
                processed_frame = self.process_frame(frame)
                
                # Convertir a JPEG
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    def start(self):
        """Iniciar detección"""
        with self.lock:
            if not self.is_running:
                if self.initialize():
                    self.is_running = True
                    return True
            return False
    
    def stop(self):
        """Detener detección"""
        with self.lock:
            self.is_running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            if self.hands:
                self.hands.close()
                self.hands = None

# Instancia global del detector
detector = SignDetector()

def index(request):
    """Página principal"""
    return render(request, 'detector/index.html')

def video_feed(request):
    """Stream de video"""
    if not detector.is_running:
        detector.start()
    
    return StreamingHttpResponse(
        detector.generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

@csrf_exempt
def start_detection(request):
    """Iniciar detección"""
    if request.method == 'POST':
        success = detector.start()
        return JsonResponse({'success': success})
    return JsonResponse({'success': False})

@csrf_exempt
def stop_detection(request):
    """Detener detección"""
    if request.method == 'POST':
        detector.stop()
        return JsonResponse({'success': True})
    return JsonResponse({'success': False})

@csrf_exempt
def start_training(request):
    """Iniciar modo entrenamiento"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            letter = data.get('letter', '').upper()
            
            # Validar letra
            if not letter or letter not in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']:
                return JsonResponse({'success': False, 'error': 'Letra inválida'})
            
            # Asegurar que el detector esté iniciado
            if not detector.is_running:
                detector.start()
            
            # Configurar modo entrenamiento
            detector.training_mode = True
            detector.target_letter = letter
            detector.training_start_time = time.time()
            detector.reset_training_state()  # Resetear estado para nueva letra
            
            # Obtener imagen de referencia
            reference_image_path = detector.get_reference_image_for_letter(letter)
            reference_image_url = None
            
            if reference_image_path:
                # Crear URL relativa para la imagen
                relative_path = os.path.relpath(reference_image_path, os.path.dirname(settings.BASE_DIR))
                reference_image_url = f"/static/reference_images/{letter}/{os.path.basename(reference_image_path)}"
            
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
    """Detener modo entrenamiento"""
    if request.method == 'POST':
        detector.training_mode = False
        detector.target_letter = None
        detector.training_start_time = None
        detector.reset_training_state()
        return JsonResponse({'success': True})
    return JsonResponse({'success': False})

@csrf_exempt
def check_auto_advance(request):
    """Verificar si hay auto-avance pendiente"""
    if request.method == 'GET':
        if detector.should_auto_advance:
            detector.should_auto_advance = False  # Resetear flag
            return JsonResponse({'should_advance': True})
        return JsonResponse({'should_advance': False})
    return JsonResponse({'success': False})

@csrf_exempt
def get_random_letter(request):
    """Obtener una letra aleatoria para entrenar"""
    if request.method == 'GET':
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        random_letter = random.choice(letters)
        
        # Obtener imagen de referencia
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
    """Servir imagen de referencia"""
    try:
        dataset_path = os.path.join(os.path.dirname(settings.BASE_DIR), 'dataset4', 'train', 'images')
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
