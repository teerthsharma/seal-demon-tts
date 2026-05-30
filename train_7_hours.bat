@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
title DemonTTS 7-Hour Autonomous Training Pipeline
color 0A

set "LOGFILE=batch_launch_7hr.log"
set "START_TIME=%date% %time%"

echo ======================================== >> %LOGFILE%
echo   BATCH LAUNCHER STARTED: %START_TIME% >> %LOGFILE%
echo ======================================== >> %LOGFILE%

call :log "=== DEMON TTS 7-HOUR PIPELINE ==="
call :log "Launcher started: %START_TIME%"

echo.
echo   ^╔══════════════════════════════════════════════════════════════╗
echo   ^║                                                              ^║
echo   ^║     DEMON TTS — 7 HOUR AUTONOMOUS TRAINING PIPELINE          ^║
echo   ^║                                                              ^║
echo   ^║     Fixes: Robotic voice / backward pass artifacts           ^║
echo   ^║     Voice: Human Male (White)                                ^║
echo   ^║     Training: Faraday + Aether + Student                     ^║
echo   ^║     Then: Generate full audiobook                            ^║
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
    call :fatal "Python is required but not found."
)
for /f "tokens=*" %%a in ('python --version 2^>^&1') do set "PYVER=%%a"
call :log "[PASS] Found %PYVER%"

:: ================================================================
:: STEP 2: Check Git Bash
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
    call :fatal "Git Bash is required. Please install Git for Windows."
)
call :log "[PASS] Git Bash found: %BASH_PATH%"

:: ================================================================
:: STEP 3: Launch the pipeline
:: ================================================================
call :log "[LAUNCH] Starting train_7_hours.sh ..."
call :log "[LAUNCH] Working directory: %CD%"
call :log "========================================"

echo.
echo ================================================================
echo  ALL CHECKS PASSED. LAUNCHING 7-HOUR PIPELINE.
echo ================================================================
echo.

"%BASH_PATH%" --login -i -c "cd '%CD%' && bash train_7_hours.sh"

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
call :log "========================================"

echo.
echo  Output audio: audiobook/final_7hr/
echo.
pause
exit /b %EXIT_CODE%

:log
    echo [%time%] %~1
    echo [%time%] %~1 >> %LOGFILE%
    exit /b 0

:fatal
    echo.
    echo  [FATAL ERROR] %~1
    echo.
    pause
    exit /b 1