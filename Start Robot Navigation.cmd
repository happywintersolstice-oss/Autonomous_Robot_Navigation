@echo off
cd /d "%~dp0"
py -3 run_app.py
if errorlevel 1 pause
