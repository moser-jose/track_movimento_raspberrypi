import cv2
import numpy as np
from ultralytics import YOLO
import config
import time
import csv
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from calcular_angulo import YOLOAngleCalculator, MediaPipeAngleCalculator
from picamera2 import Picamera2
from libcamera import controls
import os
from PIL import ImageFont, ImageDraw, Image

os.environ['GLOG_minloglevel'] = '2'  # 0 = DEBUG, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class MovementAnalyzer:
    def __init__(self, engine=config.ENGINE):
        self.engine = engine.lower()
        self.recording = False
        self.video_writer = None
        self.recording_start_time = None
        
        # Inicializar o modelo YOLO para detecção
        if self.engine == 'yolo':
            self.model = YOLO(config.MODEL_PATH, verbose=False)
            self.angle_calculator = YOLOAngleCalculator()
        elif self.engine == 'mediapipe':
            self.model = None  # Não precisamos do modelo YOLO para MediaPipe
            self.angle_calculator = MediaPipeAngleCalculator()
            # Definir as conexões do esqueleto para MediaPipe
            self.mp_pose = self.angle_calculator.mp_pose
            self.mediapipe_connections = [
                # Braço direito
                (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW),
                (self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST),
                (self.mp_pose.PoseLandmark.RIGHT_WRIST, self.mp_pose.PoseLandmark.RIGHT_PINKY),
                (self.mp_pose.PoseLandmark.RIGHT_WRIST, self.mp_pose.PoseLandmark.RIGHT_INDEX),
                (self.mp_pose.PoseLandmark.RIGHT_WRIST, self.mp_pose.PoseLandmark.RIGHT_THUMB),
                (self.mp_pose.PoseLandmark.RIGHT_PINKY, self.mp_pose.PoseLandmark.RIGHT_INDEX),
                
                # Braço esquerdo
                (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW),
                (self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST),
                (self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.LEFT_PINKY),
                (self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.LEFT_INDEX),
                (self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.LEFT_THUMB),
                (self.mp_pose.PoseLandmark.LEFT_PINKY, self.mp_pose.PoseLandmark.LEFT_INDEX),
                
                # Tronco
                (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
                (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP),
                (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_HIP),
                (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP),
                
                # Perna direita
                (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE),
                (self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE),
                (self.mp_pose.PoseLandmark.RIGHT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_HEEL),
                (self.mp_pose.PoseLandmark.RIGHT_HEEL, self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
                (self.mp_pose.PoseLandmark.RIGHT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
                
                # Perna esquerda
                (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
                (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE),
                (self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.LEFT_HEEL),
                (self.mp_pose.PoseLandmark.LEFT_HEEL, self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
                (self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
            ]
        else:
            raise ValueError(f"Engine não reconhecido: {engine}. Use 'yolo' ou 'mediapipe'.")
        
        self.angle_buffer = []
        self.start_time = time.time()

    def start_recording(self, output_path=None):
        """Inicia a gravação do vídeo."""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{config.OUTPUT_DIR}/movement_analysis_{self.engine}_{timestamp}.mp4"
        
        # Configurar o VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            config.CAMERA_FPS,  # FPS
            (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)  # Resolução
        )
        self.recording = True
        self.recording_start_time = time.time()
        print(f"Gravação iniciada: {output_path}")

    def stop_recording(self):
        """Para a gravação do vídeo."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False
        print("Gravação finalizada")

    def process_frame(self, frame):
        """Processa um único quadro e detecta poses."""
        # Se estiver gravando, salvar o frame processado
        if self.recording and self.video_writer:
            self.video_writer.write(frame)
        
        if self.engine == 'yolo':
            # Usando YOLO para detecção e cálculo de ângulos
            results = self.model(frame, conf=config.CONFIDENCE_THRESHOLD, verbose=False)
            
            for result in results:
                if result.keypoints is not None:
                    keypoints = result.keypoints.data[0].cpu().numpy()
                    
                    # Definir conexões do esqueleto (pares de índices dos pontos-chave)
                    skeleton = [
                        # Corpo superior
                        (5, 7), (7, 9), (6, 8), (8, 10),
                        # Tronco
                        (5, 6), (5, 11), (6, 12), (11, 12),
                        # Corpo inferior
                        (11, 13), (13, 15), (12, 14), (14, 16)
                    ]
                    
                    # Desenhar linhas do esqueleto e pontos-chave em uma única passagem
                    for connection in skeleton:
                        if all(keypoints[connection[0]] != 0) and all(keypoints[connection[1]] != 0):
                            pt1 = (int(keypoints[connection[0]][0]), int(keypoints[connection[0]][1]))
                            pt2 = (int(keypoints[connection[1]][0]), int(keypoints[connection[1]][1]))
                            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                            cv2.circle(frame, pt1, 4, (0, 0, 255), -1)
                            cv2.circle(frame, pt2, 4, (0, 0, 255), -1)
                    
                    # Calcular ângulos utilizando o calculador YOLO
                    angles = self.angle_calculator.process_keypoints(keypoints)
                    
                    # Armazenar ângulos no buffer se disponíveis
                    if angles:
                        self.angle_buffer.append({
                            'timestamp': time.time() - self.start_time,
                            'angles': angles
                        })
                        
                        # Manter apenas as últimas medições BUFFER_SIZE
                        if len(self.angle_buffer) > config.BUFFER_SIZE:
                            self.angle_buffer.pop(0)
        
        elif self.engine == 'mediapipe':
            # Usar MediaPipe para detecção e cálculo de ângulos
            angles, _, landmarks = self.angle_calculator.process_frame(frame)
            
            # Processar o frame com MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.angle_calculator.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Desenhar conexões do esqueleto em uma única passagem
                for connection in self.mediapipe_connections:
                    start_idx, end_idx = connection
                    if (results.pose_landmarks.landmark[start_idx].visibility > 0.5 and
                        results.pose_landmarks.landmark[end_idx].visibility > 0.5):
                        
                        start_point = (
                            int(results.pose_landmarks.landmark[start_idx].x * frame.shape[1]),
                            int(results.pose_landmarks.landmark[start_idx].y * frame.shape[0])
                        )
                        
                        end_point = (
                            int(results.pose_landmarks.landmark[end_idx].x * frame.shape[1]),
                            int(results.pose_landmarks.landmark[end_idx].y * frame.shape[0])
                        )
                        
                        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
                        # Desenhar pontos apenas para conexões não faciais
                        if start_idx not in [self.mp_pose.PoseLandmark.NOSE, 
                                          self.mp_pose.PoseLandmark.LEFT_EYE_INNER,
                                          self.mp_pose.PoseLandmark.LEFT_EYE,
                                          self.mp_pose.PoseLandmark.LEFT_EYE_OUTER,
                                          self.mp_pose.PoseLandmark.RIGHT_EYE_INNER,
                                          self.mp_pose.PoseLandmark.RIGHT_EYE,
                                          self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
                                          self.mp_pose.PoseLandmark.MOUTH_LEFT,
                                          self.mp_pose.PoseLandmark.MOUTH_RIGHT,
                                          self.mp_pose.PoseLandmark.LEFT_EAR,
                                          self.mp_pose.PoseLandmark.RIGHT_EAR]:
                            cv2.circle(frame, start_point, 4, (0, 0, 255), -1)
                        if end_idx not in [self.mp_pose.PoseLandmark.NOSE, 
                                        self.mp_pose.PoseLandmark.LEFT_EYE_INNER,
                                        self.mp_pose.PoseLandmark.LEFT_EYE,
                                        self.mp_pose.PoseLandmark.LEFT_EYE_OUTER,
                                        self.mp_pose.PoseLandmark.RIGHT_EYE_INNER,
                                        self.mp_pose.PoseLandmark.RIGHT_EYE,
                                        self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
                                        self.mp_pose.PoseLandmark.MOUTH_LEFT,
                                        self.mp_pose.PoseLandmark.MOUTH_RIGHT,
                                        self.mp_pose.PoseLandmark.LEFT_EAR,
                                        self.mp_pose.PoseLandmark.RIGHT_EAR]:
                            cv2.circle(frame, end_point, 4, (0, 0, 255), -1)
            
            # Armazenar ângulos no buffer se disponíveis
            if angles:
                self.angle_buffer.append({
                    'timestamp': time.time() - self.start_time,
                    'angles': angles
                })
                
                # Manter apenas as últimas medições BUFFER_SIZE
                if len(self.angle_buffer) > config.BUFFER_SIZE:
                    self.angle_buffer.pop(0)
        
        # Desenhar ângulos no quadro (independente do motor)
        y_offset = 30
        if self.angle_buffer and 'angles' in self.angle_buffer[-1]:
            angles = self.angle_buffer[-1]['angles']
            
            # Converter o frame OpenCV para PIL Image para usar fonte personalizada
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)
            
            # Carregar a fonte Arial
            try:
                font = ImageFont.truetype("Arial.ttf", 20)
            except:
                # Se Arial não estiver disponível, tentar outra fonte comum
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", 20)
                except:
                    # Se nenhuma fonte personalizada estiver disponível, usar a fonte padrão
                    font = ImageFont.load_default()
            
            for joint, angle in angles.items():
                text = f"{joint}: {angle:.1f}°"
                draw.text((10, y_offset), text, font=font, fill=(0, 255, 0))
                y_offset += 30
            
            # Converter de volta para OpenCV
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        
        # Mostrar qual motor está sendo usado
        cv2.putText(frame, f"Motor: {self.engine.upper()}", 
                   (frame.shape[1] - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return frame
    
    def save_data(self):
        """Salvar dados coletados em arquivo CSV."""
        print("Salvando dados em arquivo CSV...")
        if config.SAVE_DATA and self.angle_buffer:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
            # Salvar como CSV
            csv_filename = f"{config.OUTPUT_DIR}/dados_dos_movimentos_{self.engine}_{timestamp}.csv"
            if self.angle_buffer:
                # Obter todos os nomes de articulações únicos
                joint_names = set()
                for data in self.angle_buffer:
                    joint_names.update(data['angles'].keys())
                joint_names = sorted(list(joint_names))
                
                # Escrever cabeçalho CSV
                with open(csv_filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp'] + joint_names)
                    
                    # Escrever linhas de dados
                    for data in self.angle_buffer:
                        row = [data['timestamp']]
                        for joint in joint_names:
                            row.append(data['angles'].get(joint, ''))
                        writer.writerow(row)
            print("Dados salvos em arquivo CSV com sucesso!")
            print("Salvando os gráficos...")
            # Criar gráfico de ângulos
            if self.angle_buffer:
                plt.figure(figsize=(10, 6))
                times = [d['timestamp'] for d in self.angle_buffer]
                
                for joint in joint_names:
                    angles = []
                    for d in self.angle_buffer:
                        if joint in d['angles']:
                            angles.append(d['angles'][joint])
                        else:
                            angles.append(None)  # Usar None para dados ausentes
                    
                    # Filtrar None antes de plotar
                    valid_times = []
                    valid_angles = []
                    for t, a in zip(times, angles):
                        if a is not None:
                            valid_times.append(t)
                            valid_angles.append(a)
                    
                    if valid_times and valid_angles:
                        plt.plot(valid_times, valid_angles, label=joint)
                
                plt.xlabel('Tempo (s)')
                plt.ylabel('Ângulo (graus)')
                plt.title(f'Ângulos das articulações ao longo do tempo - {self.engine.upper()}')
                plt.legend()
                plt.grid(True)
                image_path = f"{config.OUTPUT_DIR}/angulos_articulacoes_{self.engine}_{timestamp}.png"
                plt.savefig(image_path)
                plt.close()
                print(f"Gráfico salvo em: {image_path}")
            print("Gráficos salvos com sucesso!")
    def run(self):
        # Inicializar a câmera
        try:
            self.picam2 = Picamera2()
            
            # Configurar a câmera com configurações otimizadas
            camera_config = self.picam2.create_preview_configuration(
                main={"format": 'RGB888',"size": (640, 480)},
            )
            
            # Tentar configurar a câmera
            self.picam2.configure(camera_config)
            
            # Iniciar a câmera
            self.picam2.start()
            
            print("Câmera inicializada com sucesso!")
            print("Pressione 'r' para iniciar/parar a gravação")
            print("Pressione 'q' para sair")
            
            try:
                while True:
                    # Capturar frame da câmera
                    frame = self.picam2.capture_array()
                    
                    # Processar o frame
                    processed_frame = self.process_frame(frame)
                    
                    # Mostrar o frame
                    cv2.imshow('Movement Analysis', processed_frame)
                    
                    # Verificar comandos do teclado
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        if not self.recording:
                            self.start_recording()
                        else:
                            self.stop_recording()
                        
            finally:
                if self.recording:
                    self.stop_recording()
                self.picam2.stop()
                cv2.destroyAllWindows()
                self.save_data()
                if self.engine == 'mediapipe' and hasattr(self.angle_calculator, 'pose'):
                    self.angle_calculator.pose.close()
        except Exception as e:
            print(f"Erro ao inicializar a câmera: {str(e)}")

if __name__ == "__main__":
    # Configurar argumentos de linha de comando
    parser = argparse.ArgumentParser(description='Analisador de Movimento com YOLO e MediaPipe')
    parser.add_argument('--engine', type=str, default=config.ENGINE,
                        choices=['yolo', 'mediapipe'],
                        help='Motor de análise a ser usado (yolo ou mediapipe)')
    args = parser.parse_args()
    
    # Inicializar e executar o analisador
    analyzer = MovementAnalyzer(engine=args.engine)
    analyzer.run() 
