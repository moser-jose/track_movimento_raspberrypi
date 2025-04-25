import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configurações da câmera
CAMERA_FPS = int(os.getenv('CAMERA_FPS', '30'))
CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', '640'))
CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', '480'))

# Configurações do modelo
MODEL_PATH = os.getenv('MODEL_PATH', 'YOLO11n-pose.pt')
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.3'))

# Motor de análise (YOLO ou MediaPipe)
ENGINE = os.getenv('ENGINE', 'yolo').lower()

# Configurações de dados
SAVE_DATA = os.getenv('SAVE_DATA', 'true').lower() == 'true'
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'dados_dos_movimentos')
BUFFER_SIZE = int(os.getenv('BUFFER_SIZE', '30'))

# Criar diretório de saída se não existir
if SAVE_DATA:
    os.makedirs(OUTPUT_DIR, exist_ok=True) 