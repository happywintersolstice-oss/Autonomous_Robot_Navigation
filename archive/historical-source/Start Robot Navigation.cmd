@echo off
setlocal
cd /d "%~dp0"
if not exist "autonomous-robot-navigation-rl\.venv\Scripts\python.exe" (
  echo Set up the Python environment first. See docs\developer-guide.md.
  pause
  exit /b 1
)
"autonomous-robot-navigation-rl\.venv\Scripts\python.exe" run_app.py
if errorlevel 1 pause
