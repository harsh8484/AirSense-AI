@echo off
echo ==============================================================
echo              AirSense AI - Launching System
echo ==============================================================
echo.

:: Check for virtual environment
if not exist venv (
    echo [INFO] Creating Python virtual environment...
    python -m venv venv
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo [INFO] Installing/verifying dependencies...
pip install -r requirements.txt

echo [INFO] Launching FastAPI web server...
python main.py

pause
