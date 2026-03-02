@echo off
REM ══════════════════════════════════════════════════════════════
REM install.bat — Instalación para QLoRA en RTX 5050 (Blackwell)
REM ══════════════════════════════════════════════════════════════
REM Ejecutar desde la carpeta del proyecto con Python 3.10+ instalado
REM ══════════════════════════════════════════════════════════════

echo ============================================
echo  Instalacion QLoRA - RTX 5050 Blackwell
echo ============================================
echo.

REM --- Paso 0: Crear entorno virtual ---
echo [0/4] Creando entorno virtual...
python -m venv venv
call venv\Scripts\activate.bat

REM --- Paso 1: PyTorch nightly con CUDA 12.8 (soporte sm_120) ---
echo.
echo [1/4] Instalando PyTorch nightly con soporte Blackwell (sm_120)...
echo        Esto puede tardar varios minutos...
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

REM --- Paso 2: Verificar que CUDA funciona ---
echo.
echo [2/4] Verificando soporte CUDA...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else print('ERROR: CUDA no detectado')"
if errorlevel 1 (
    echo.
    echo ERROR: PyTorch no detecta CUDA. Verifica que:
    echo   1. Los drivers NVIDIA estan actualizados
    echo   2. CUDA Toolkit 12.8+ esta instalado
    echo   3. Reinicia el PC despues de actualizar drivers
    pause
    exit /b 1
)

REM --- Paso 3: Instalar dependencias de fine-tuning ---
echo.
echo [3/4] Instalando librerias de fine-tuning...
pip install -r requirements.txt

REM --- Paso 4: Verificación final ---
echo.
echo [4/4] Verificacion final...
python -c "import torch; from transformers import AutoTokenizer; from peft import LoraConfig; from trl import SFTTrainer; import bitsandbytes; print('Todas las librerias cargadas correctamente'); print(f'  torch: {torch.__version__}'); print(f'  bitsandbytes: {bitsandbytes.__version__}'); print(f'  CUDA: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0)}')"

echo.
echo ============================================
echo  Instalacion completada!
echo  Para activar el entorno en el futuro:
echo    venv\Scripts\activate.bat
echo  Para entrenar:
echo    python train_qlora_7b.py
echo ============================================
pause
