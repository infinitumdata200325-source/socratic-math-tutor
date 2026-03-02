#!/bin/bash
# ══════════════════════════════════════════════════════════════
# install.sh — Instalación para QLoRA en RTX 5050 (Blackwell)
# ══════════════════════════════════════════════════════════════
# Ejecutar: chmod +x install.sh && ./install.sh
# ══════════════════════════════════════════════════════════════

set -e

echo "============================================"
echo " Instalacion QLoRA - RTX 5050 Blackwell"
echo "============================================"
echo ""

# --- Paso 0: Crear entorno virtual ---
echo "[0/4] Creando entorno virtual..."
python3 -m venv venv
source venv/bin/activate

# --- Paso 1: PyTorch nightly con CUDA 12.8 (soporte sm_120) ---
echo ""
echo "[1/4] Instalando PyTorch nightly con soporte Blackwell (sm_120)..."
echo "       Esto puede tardar varios minutos..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# --- Paso 2: Verificar que CUDA funciona ---
echo ""
echo "[2/4] Verificando soporte CUDA..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
if torch.cuda.is_available():
    print(f'CUDA disponible: True')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    cap = torch.cuda.get_device_capability(0)
    print(f'Compute capability: sm_{cap[0]*10 + cap[1]}')
else:
    print('ERROR: CUDA no detectado')
    exit(1)
"

# --- Paso 3: Instalar dependencias de fine-tuning ---
echo ""
echo "[3/4] Instalando librerias de fine-tuning..."
pip install -r requirements.txt

# --- Paso 4: Verificación final ---
echo ""
echo "[4/4] Verificacion final..."
python3 -c "
import torch
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer
import bitsandbytes
print('Todas las librerias cargadas correctamente')
print(f'  torch: {torch.__version__}')
print(f'  bitsandbytes: {bitsandbytes.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "============================================"
echo " Instalacion completada!"
echo " Para activar el entorno en el futuro:"
echo "   source venv/bin/activate"
echo " Para entrenar:"
echo "   python train_qlora_7b.py"
echo "============================================"
