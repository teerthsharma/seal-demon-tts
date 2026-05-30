@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
title DemonTTS 6-Hour Autonomous Training Pipeline
color 0A

set "LOGFILE=batch_launch.log"
set "START_TIME=%date% %time%"

echo ======================================== >> %LOGFILE%
echo   BATCH LAUNCHER STARTED: %START_TIME% >> %LOGFILE%
echo ======================================== >> %LOGFILE%

call :log "=== DEMON TTS 6-HOUR PIPELINE ==="
call :log "Launcher started: %START_TIME%"

echo.
echo   ^╔══════════════════════════════════════════════════════════════╗
echo   ^║                                                              ^║
echo   ^║     DEMON TTS — 6 HOUR AUTONOMOUS TRAINING PIPELINE          ^║
echo   ^║                                                              ^║
echo   ^║     Master Chapter: 2. Bound By Will                         ^║
echo   ^║     Training: Faraday + Aether on Chapter 2 data             ^║
echo   ^║     Then: Generate full Threshold's Pursuit audiobook        ^║
echo   ^║                                                              ^║
echo   ^║     DO NOT CLOSE THIS WINDOW. Let it cook.                   ^║
echo   ^║                                                              ^║
echo   ^╚══════════════════════════════════════════════════════════════╝
echo.

:: ================================================================
:: STEP 1: Check Python
:: ================================================================
call :log "[CHECK] Looking for Python..."
python --version >nul 2>&1
if errorlevel 1 (
    call :log "[FAIL] Python not found in PATH"
    call :log "[INFO] Please install Python 3.11 from python.org"
    call :fatal "Python is required but not found."
)
for /f "tokens=*" %%a in ('python --version 2^>^&1') do set "PYVER=%%a"
call :log "[PASS] Found %PYVER%"

:: ================================================================
:: STEP 2: Check pip and install dependencies
:: ================================================================
call :log "[CHECK] Verifying pip..."
python -m pip --version >nul 2>&1
if errorlevel 1 (
    call :log "[FAIL] pip not found"
    call :fatal "pip is required but not found."
)
call :log "[PASS] pip is available"

call :log "[CHECK] Verifying required packages..."
set "PACKAGES=torch transformers accelerate bitsandbytes torchaudio soundfile numpy scipy matplotlib tqdm"
set "MISSING=0"

for %%p in (%PACKAGES%) do (
    python -c "import %%p" >nul 2>&1
    if errorlevel 1 (
        call :log "[MISSING] Package not installed: %%p"
        set "MISSING=1"
    ) else (
        call :log "[PASS] Package OK: %%p"
    )
)

if %MISSING% equ 1 (
    call :log "[INSTALL] Installing missing packages..."
    python -m pip install -q torch transformers accelerate bitsandbytes torchaudio soundfile numpy scipy matplotlib tqdm
    if errorlevel 1 (
        call :log "[FAIL] pip install failed"
        call :fatal "Failed to install required packages. Check internet connection."
    )
    call :log "[PASS] Packages installed"
) else (
    call :log "[PASS] All packages already installed"
)

:: ================================================================
:: STEP 3: Check Git Bash
:: ================================================================
call :log "[CHECK] Looking for Git Bash..."
set "BASH_PATH="

if exist "C:\Program Files\Git\bin\bash.exe" (
    set "BASH_PATH=C:\Program Files\Git\bin\bash.exe"
)
if not defined BASH_PATH (
    if exist "C:\Program Files (x86)\Git\bin\bash.exe" (
        set "BASH_PATH=C:\Program Files (x86)\Git\bin\bash.exe"
    )
)
if not defined BASH_PATH (
    for /f "delims=" %%i in ('where bash 2^>nul') do (
        set "BASH_PATH=%%i"
    )
)

if not defined BASH_PATH (
    call :log "[FAIL] Git Bash not found"
    call :log "[INFO] Download: https://git-scm.com/download/win"
    call :fatal "Git Bash is required. Please install Git for Windows."
)
call :log "[PASS] Git Bash found: %BASH_PATH%"

:: ================================================================
:: STEP 4: Check required files
:: ================================================================
call :log "[CHECK] Verifying required files..."

if not exist "cook_for_6_hours.sh" (
    call :log "[FAIL] cook_for_6_hours.sh not found in current directory"
    call :fatal "Main script missing."
)
call :log "[PASS] cook_for_6_hours.sh found"

if not exist "pipeline_chapter2_master.py" (
    call :log "[FAIL] pipeline_chapter2_master.py not found"
    call :fatal "Pipeline script missing."
)
call :log "[PASS] pipeline_chapter2_master.py found"

if not exist "book_parsed\Threshold's Pursuit_6b3bb078d03bc9c4.json" (
    call :log "[FAIL] Book JSON not found"
    call :fatal "Parsed book data missing."
)
call :log "[PASS] Book JSON found"

if not exist "models\faraday.pt" (
    call :log "[WARN] models\faraday.pt not found — will train from scratch"
)
if not exist "models\aether.pt" (
    call :log "[WARN] models\aether.pt not found — will train from scratch"
)

:: ================================================================
:: STEP 5: Check GPU
:: ================================================================
call :log "[CHECK] Verifying CUDA GPU..."
python -c "import torch; assert torch.cuda.is_available(); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')" > gpu_check.txt 2>&1
if errorlevel 1 (
    call :log "[FAIL] CUDA GPU not available or not working"
    type gpu_check.txt >> %LOGFILE%
    call :fatal "NVIDIA GPU with CUDA required."
)
type gpu_check.txt >> %LOGFILE%
del gpu_check.txt
call :log "[PASS] GPU verified"

:: ================================================================
:: STEP 6: Check disk space
:: ================================================================
call :log "[CHECK] Verifying disk space..."
for /f "tokens=3" %%a in ('dir ^| findstr "bytes free"') do set "FREE_BYTES=%%a"
set "FREE_GB=%FREE_BYTES:~0,-9%"
if not defined FREE_GB set "FREE_GB=0"
if %FREE_GB% lss 10 (
    call :log "[WARN] Low disk space: ~%FREE_GB% GB free. Recommend 20+ GB."
) else (
    call :log "[PASS] Disk space OK: ~%FREE_GB% GB free"
)

:: ================================================================
:: STEP 7: Launch the pipeline
:: ================================================================
call :log "[LAUNCH] Starting cook_for_6_hours.sh ..."
call :log "[LAUNCH] Working directory: %CD%"
call :log "[LAUNCH] Timestamp: %date% %time%"
call :log "========================================"

echo.
echo ================================================================
echo  ALL CHECKS PASSED. LAUNCHING 6-HOUR PIPELINE.
echo ================================================================
echo.
echo  Do NOT close this window. Training is in progress.
echo  Log: training_6hr.log (inside bash)
echo  Batch log: %LOGFILE% (this file)
echo.

"%BASH_PATH%" --login -i -c "cd '%CD%' && bash cook_for_6_hours.sh"

set "EXIT_CODE=%ERRORLEVEL%"
set "END_TIME=%date% %time%"

echo.
echo ================================================================
if %EXIT_CODE% equ 0 (
    call :log "[SUCCESS] Pipeline completed successfully."
    echo  [SUCCESS] Training pipeline completed.
) else (
    call :log "[ERROR] Pipeline exited with code %EXIT_CODE%"
    echo  [ERROR] Pipeline failed with exit code %EXIT_CODE%.
)
call :log "End time: %END_TIME%"
call :log "Exit code: %EXIT_CODE%"
call :log "========================================"

echo.
echo  Check training_6hr.log for full training details.
echo  Check %LOGFILE% for launcher diagnostics.
echo  Output audio: audiobook/final/
echo.
pause
exit /b %EXIT_CODE%

:: ================================================================
:: Helper functions
:: ================================================================
:log
    echo [%time%] %~1
    echo [%time%] %~1 >> %LOGFILE%
    exit /b 0

:fatal
    echo.
    echo  [FATAL ERROR] %~1
    echo.
    echo  See %LOGFILE% for details.
    pause
    exit /b 1
