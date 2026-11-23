@echo off
setlocal
cd /d "%~dp0"
echo Creating virtual environment...
py -3 -m venv .venv || goto :err
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || goto :err
pip install -e . || goto :err
python -m pytest -q
echo.
echo Setup finished. Next: "Run Smoke Test.cmd" then "Start App.cmd".
pause
exit /b 0
:err
echo Setup failed - see the messages above.
pause
exit /b 1
