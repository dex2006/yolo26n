@echo off
chcp 65001 >nul
cd /d "%~dp0"

set PY=C:\Users\user\Desktop\yolo\.venv\Scripts\python.exe

echo.
echo ===== Night Agent Install =====
echo Python: %PY%
echo.

"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -U ultralytics gradio opencv-python Pillow requests pydantic
"%PY%" -m pip install -U transformers accelerate bitsandbytes

echo.
echo ===== Check GPU =====
"%PY%" -c "import torch; print('Torch:',torch.__version__); print('CUDA:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

echo.
echo 安裝完成
pause
