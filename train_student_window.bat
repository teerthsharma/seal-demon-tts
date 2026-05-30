@echo off
REM Student Training Window Launcher
REM Opens a standalone Git Bash window and runs student training for 2 hours.

title DemonTTS Student Training

echo Finding Git Bash...
set "BASH_PATH="

if exist "C:\Program Files\Git\bin\bash.exe" (
    set "BASH_PATH=C:\Program Files\Git\bin\bash.exe"
) else if exist "C:\Program Files (x86)\Git\bin\bash.exe" (
    set "BASH_PATH=C:\Program Files (x86)\Git\bin\bash.exe"
) else (
    for %%i in (bash.exe) do set "BASH_PATH=%%~$PATH:i"
)

if not defined BASH_PATH (
    echo ERROR: Git Bash not found. Install Git for Windows.
    pause
    exit /b 1
)

echo Launching student training in new window...
echo.
echo Command: bash train_student_only.sh
echo.

start "Student Training" "%BASH_PATH%" --login -i -c "cd '%CD%' && export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 && bash train_student_only.sh"

echo Window opened. Training will run for ~2 hours.
echo Close the window ONLY if you want to stop training early.
pause
