@echo off
cd /d "%~dp0.."
echo Iniciando Quantum Trader Desktop...
echo Backend deve estar rodando em http://localhost:8080
echo.
.venv\Scripts\python.exe desktop_app\main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERRO] O app encerrou com codigo %ERRORLEVEL%
    pause
)
