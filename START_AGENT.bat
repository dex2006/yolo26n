@echo off
chcp 65001 >nul
cd /d "%~dp0"

"C:\Users\user\Desktop\yolo\.venv\Scripts\python.exe" app.py

pause
