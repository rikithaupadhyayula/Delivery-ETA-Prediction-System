@echo off
REM =============================================================================
REM Delivery ETA Prediction System - Run Script (Windows)
REM =============================================================================
REM This script sets up and runs all components of the project
REM Usage: run.bat [command]
REM Commands: setup, train, api, dashboard, all
REM =============================================================================

setlocal enabledelayedexpansion

set PROJECT_DIR=%~dp0
set VENV_DIR=%PROJECT_DIR%venv

REM Remove trailing backslash
if "%PROJECT_DIR:~-1%"=="\" set PROJECT_DIR=%PROJECT_DIR:~0,-1%

echo.
echo =============================================
echo   Delivery ETA Prediction System
echo =============================================
echo.

if "%1"=="" goto help
if "%1"=="setup" goto setup
if "%1"=="train" goto train
if "%1"=="api" goto api
if "%1"=="dashboard" goto dashboard
if "%1"=="all" goto all
if "%1"=="help" goto help
if "%1"=="--help" goto help
if "%1"=="-h" goto help

echo [ERROR] Unknown command: %1
goto help

:setup
echo [STEP] Setting up the project...
echo.

REM Create virtual environment if it doesn't exist
if not exist "%VENV_DIR%" (
    echo [STEP] Creating virtual environment...
    python -m venv "%VENV_DIR%"
)

REM Activate and install
echo [STEP] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

echo [STEP] Installing dependencies...
pip install --upgrade pip
pip install -r "%PROJECT_DIR%\requirements.txt"

echo.
echo [SUCCESS] Setup complete!
echo.
echo Next steps:
echo   1. Run training: run.bat train
echo   2. Start API: run.bat api
echo   3. Start Dashboard: run.bat dashboard
echo.
goto end

:train
echo [STEP] Training the model...
echo.

call "%VENV_DIR%\Scripts\activate.bat"
cd /d "%PROJECT_DIR%"
python src/models/train_model.py

echo.
echo [SUCCESS] Training complete! Model saved to models/best_model.pkl
echo.
goto end

:api
echo [STEP] Starting FastAPI server...
echo.

call "%VENV_DIR%\Scripts\activate.bat"
cd /d "%PROJECT_DIR%"

echo.
echo API Documentation: http://localhost:8000/docs
echo Health Check: http://localhost:8000/health
echo.
echo Press Ctrl+C to stop the server
echo.

uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
goto end

:dashboard
echo [STEP] Starting Streamlit dashboard...
echo.

call "%VENV_DIR%\Scripts\activate.bat"
cd /d "%PROJECT_DIR%"

echo.
echo Dashboard: http://localhost:8501
echo.
echo Press Ctrl+C to stop the dashboard
echo.

streamlit run dashboard/app.py --server.port 8501
goto end

:all
echo [STEP] Running complete pipeline...
echo.

call "%VENV_DIR%\Scripts\activate.bat"
cd /d "%PROJECT_DIR%"

REM Train first
echo [STEP] Step 1/3: Training model...
python src/models/train_model.py

echo.
echo [SUCCESS] Training complete!
echo.

REM Start API in a new window
echo [STEP] Step 2/3: Starting API server (new window)...
start "Delivery ETA API" cmd /k "call "%VENV_DIR%\Scripts\activate.bat" && uvicorn api.app:app --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak > nul

REM Start dashboard
echo [STEP] Step 3/3: Starting dashboard...
echo.
echo API running at: http://localhost:8000/docs
echo Dashboard running at: http://localhost:8501
echo.

streamlit run dashboard/app.py --server.port 8501
goto end

:help
echo Usage: run.bat [command]
echo.
echo Commands:
echo   setup      - Create virtual environment and install dependencies
echo   train      - Train the ML model
echo   api        - Start the FastAPI prediction server
echo   dashboard  - Start the Streamlit dashboard
echo   all        - Train model and start both servers
echo   help       - Show this help message
echo.
echo Examples:
echo   run.bat setup     # First time setup
echo   run.bat train     # Train the model
echo   run.bat api       # Start API only
echo   run.bat dashboard # Start dashboard only
echo   run.bat all       # Run everything
echo.
goto end

:end
endlocal
