@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
title DemonTTS Master Pipeline — Admin Mode
color 0A

set "LOGFILE=master_launch.log"
set "START_TIME=%date% %time%"

:: ================================================================
:: ADMIN ELEVATION
:: ================================================================
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [Master] Admin privileges confirmed. >> %LOGFILE%
) else (
    echo [Master] Requesting admin privileges...
    powershell -Command "Start-Process '%~f0' -Verb runAs"
    exit /b
)

echo ======================================== >> %LOGFILE%
echo   MASTER BAT STARTED: %START_TIME% >> %LOGFILE%
echo ======================================== >> %LOGFILE%

call :log "=== DEMON TTS MASTER PIPELINE ==="
call :log "Launcher started: %START_TIME%"

echo.
echo   ^╔══════════════════════════════════════════════════════════════╗
echo   ^║                                                              ^║
echo   ^║     DEMON TTS — MASTER AUTONOMOUS PIPELINE                   ^║
echo   ^║                                                              ^║
echo   ^║     Voice: Human Male                                        ^║
echo   ^║     Training: Faraday + Aether + Student (Council)           ^║
echo   ^║     Then: Generate full audiobook                            ^║
echo   ^║                                                              ^║
echo   ^║     DO NOT CLOSE THIS WINDOW. Let it cook.                   ^║
echo   ^║                                                              ^║
echo   ^╚══════════════════════════════════════════════════════════════╝
echo.

:: ================================================================
:: STEP 1: HYGIENE
:: ================================================================
call :log "[1/5] Running hygiene..."

echo [Hygiene] Cleaning Python cache files...
for /f "delims=" %%d in ('dir /s /b /ad __pycache__ 2^>nul') do (
    rd /s /q "%%d" 2>nul
)
for /f "delims=" %%f in ('dir /s /b *.pyc 2^>nul') do (
    del "%%f" 2>nul
)
for /f "delims=" %%f in ('dir /s /b *.pyo 2^>nul') do (
    del "%%f" 2>nul
)

:: Clean backup / temp files (but NEVER .bat files)
echo [Hygiene] Cleaning temp and backup files...
del /q gpu_check.txt 2>nul
for /f "delims=" %%f in ('dir /s /b *.bak 2^>nul') do del "%%f" 2>nul
for /f "delims=" %%f in ('dir /s /b *.new 2^>nul') do del "%%f" 2>nul
for /f "delims=" %%f in ('dir /s /b *.transformer 2^>nul') do del "%%f" 2>nul

:: Keep only the 3 most recent .log files
echo [Hygiene] Trimming old logs (keep newest 3)...
for /f "skip=3 delims=" %%f in ('dir /b /o-d *.log 2^>nul') do (
    del "%%f" 2>nul
)

call :log "[1/5] Hygiene complete."

:: ================================================================
:: STEP 2: Check Python
:: ================================================================
call :log "[2/5] Looking for Python..."
python --version >nul 2>&1
if errorlevel 1 (
    call :log "[FAIL] Python not found in PATH"
    call :fatal "Python is required but not found."
)
for /f "tokens=*" %%a in ('python --version 2^>^&1') do set "PYVER=%%a"
call :log "[PASS] Found %PYVER%"

:: ================================================================
:: STEP 3: Check Git Bash
:: ================================================================
call :log "[3/5] Looking for Git Bash..."
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
    call :fatal "Git Bash is required. Please install Git for Windows."
)
call :log "[PASS] Git Bash found: %BASH_PATH%"

:: ================================================================
:: STEP 4: Check GPU
:: ================================================================
call :log "[4/5] Verifying CUDA GPU..."
python -c "import torch; assert torch.cuda.is_available(); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')" > gpu_check.txt 2>&1
if errorlevel 1 (
    call :log "[FAIL] CUDA GPU not available"
    type gpu_check.txt >> %LOGFILE%
    call :fatal "NVIDIA GPU with CUDA required."
)
type gpu_check.txt >> %LOGFILE%
del gpu_check.txt
call :log "[PASS] GPU verified"

:: ================================================================
:: STEP 5: Check disk space
:: ================================================================
call :log "[5/5] Verifying disk space..."
for /f "tokens=3" %%a in ('dir ^| findstr "bytes free"') do set "FREE_BYTES=%%a"
set "FREE_GB=%FREE_BYTES:~0,-9%"
if not defined FREE_GB set "FREE_GB=0"
if %FREE_GB% lss 10 (
    call :log "[WARN] Low disk space: ~%FREE_GB% GB free. Recommend 20+ GB."
) else (
    call :log "[PASS] Disk space OK: ~%FREE_GB% GB free"
)

:: ================================================================
:: STEP 6: Launch Master SH
:: ================================================================
call :log "[LAUNCH] Starting master.sh ..."
call :log "[LAUNCH] Working directory: %CD%"
call :log "========================================"

echo.
echo ================================================================
echo  ALL CHECKS PASSED. LAUNCHING MASTER PIPELINE.
echo ================================================================
echo.
echo  Do NOT close this window. Training is in progress.
echo  Log: training_7hr.log
echo  Master log: %LOGFILE%
echo.

set "MASTER_MODE=1"
"%BASH_PATH%" --login -i -c "cd '%CD%' && export MASTER_MODE=1 && bash master.sh"

set "EXIT_CODE=%ERRORLEVEL%"
set "END_TIME=%date% %time%"

echo.
echo ================================================================
if %EXIT_CODE% equ 0 (
    call :log "[SUCCESS] Master pipeline completed successfully."
    echo  [SUCCESS] Master pipeline completed.
) else (
    call :log "[ERROR] Master pipeline exited with code %EXIT_CODE%"
    echo  [ERROR] Master pipeline failed with exit code %EXIT_CODE%.
)
call :log "End time: %END_TIME%"
call :log "Exit code: %EXIT_CODE%"
call :log "========================================"

echo.
echo  Output audio: audiobook/final_7hr/
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
