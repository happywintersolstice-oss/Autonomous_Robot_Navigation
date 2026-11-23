@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m medinav pipeline --config configs\smoke.yaml
echo.
echo Done - figures are in runs\smoke\results
pause
