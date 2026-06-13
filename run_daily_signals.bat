@echo off
REM Get the directory where this batch file is located
cd /d "%~dp0"

REM Use virtual environment Python
set PYTHON=.venv\Scripts\python.exe

REM Fallback: try python from PATH
if not exist "%PYTHON%" (
    set PYTHON=python
)

echo [%date% %time%] Starting daily signals run...
"%PYTHON%" run_all_local_tw_to_excel.py

echo [%date% %time%] Done.
