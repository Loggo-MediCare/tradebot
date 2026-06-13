@echo off
REM run_bot.bat - run run_all_local_tw_to_excel.py
REM Usage: run_bot.bat [--txt-only] [additional args]

cd /d %~dp0
echo Running run_all_local_tw_to_excel.py...

REM Try using `python` first
python "%~dp0run_all_local_tw_to_excel.py" %*
if %ERRORLEVEL% NEQ 0 (
	echo python failed, trying py -3 launcher...
	py -3 "%~dp0run_all_local_tw_to_excel.py" %*
)

if %ERRORLEVEL% NEQ 0 (
	echo.
	echo ERROR: Could not run Python. Ensure Python is installed and on PATH, or use the full path to python.exe.
	echo Example: C:\Path\To\Python\python.exe run_all_local_tw_to_excel.py --txt-only
	pause
)

exit /b %ERRORLEVEL%