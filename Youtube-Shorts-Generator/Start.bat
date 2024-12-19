@echo off
setlocal

rem Check if venv directory exists
if not exist "%~dp0\venv" (
    echo Virtual environment not found. Creating one...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        exit /b 1
    )
    echo Virtual environment created successfully.
)

rem Activate venv
call %~dp0\venv\Scripts\activate

echo Virtual environment is now active.
echo Type 'deactivate' to exit the virtual environment.

rem Keep the window open
cmd /k