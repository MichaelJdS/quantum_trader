@echo off
cd /d "%~dp0"
echo ====================================================
echo        Iniciando Quantum Trader (Tudo em Um)
echo ====================================================
echo.

echo [1/2] Iniciando o Motor de Inteligencia Artificial...
start "Quantum Trader - Motor AI" cmd /c ".venv\Scripts\python.exe -m uvicorn cloud_api.main:app --host 0.0.0.0 --port 8080"

echo Aguardando 5 segundos para o motor carregar e abrir a porta 8080...
timeout /t 5 /nobreak >nul

echo [2/2] Iniciando o Dashboard Grafico (App Desktop)...
.venv\Scripts\python.exe desktop_app\main.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERRO] O app encerrou de forma inesperada com codigo %ERRORLEVEL%
    pause
)
