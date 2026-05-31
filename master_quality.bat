@echo off
REM Master Quality Pipeline — Seal-Approved RTX 4060 Optimized
REM Trains Student TTS (quality-focused) → Auto-generates judged audiobook
REM 
REM Specs: RTX 4060 Laptop 8GB | i7-14700HX | 16-mixed AMP | batch_size=2
REM 
REM DO NOT CLOSE THIS WINDOW. Minimize it and let it cook.

title DemonTTS Quality Pipeline — DO NOT CLOSE

echo ============================================================
echo   DEMON TTS — QUALITY PIPELINE
echo   Target: Train Student + Generate Judged Audiobook
echo   GPU: RTX 4060 Laptop 8GB
echo   DO NOT CLOSE THIS WINDOW
echo ============================================================
echo.

cd /d "%~dp0"

REM Find Git Bash
set "BASH_PATH="
if exist "C:\Program Files\Git\bin\bash.exe" (
    set "BASH_PATH=C:\Program Files\Git\bin\bash.exe"
) else if exist "C:\Program Files (x86)\Git\bin\bash.exe" (
    set "BASH_PATH=C:\Program Files (x86)\Git\bin\bash.exe"
) else (
    for %%i in (bash.exe) do set "BASH_PATH=%%~$PATH:i"
)

if not defined BASH_PATH (
    echo [ERROR] Git Bash not found. Install Git for Windows.
    pause
    exit /b 1
)

REM Launch the quality pipeline in a persistent window
start "DemonTTS Quality Pipeline" /MIN "%BASH_PATH%" --login -c "cd '%CD%' && bash master_quality.sh"

echo.
echo [OK] Pipeline launched in background window.
echo [OK] Training will auto-resume from last checkpoint.
echo [OK] Audiobook generation starts after training completes.
echo.
echo Check training_student.log for live progress.
echo Check audiobook/final_7hr/ for generated chapters.
echo.
pause
