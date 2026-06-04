@echo off
cd /d "%~dp0.."
echo Iniciando Backend Quantum Trader (porta 8080)...
echo Deixe esta janela aberta!
echo.
.venv\Scripts\python.exe -m uvicorn cloud_api.main:app --host 0.0.0.0 --port 8080
pause
