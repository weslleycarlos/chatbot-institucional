@echo off
TITLE GovBot Launcher

echo =====================================================
echo      INICIANDO SISTEMA GOVBOT (INTRANET)
echo =====================================================

:: 1. Iniciar o Backend (Python) em uma nova janela minimizada
echo [1/2] Iniciando Servidor Python (API)...
cd backend
:: --host 0.0.0.0 permite que outros computadores acessem a API
start /min "GovBot API" uvicorn main:app --host 0.0.0.0 --port 8000

:: Voltar para a raiz
cd ..

:: 2. Iniciar o Frontend (React/Vite)
echo [2/2] Iniciando Interface React...
cd frontend
:: --host permite que o Vite sirva para a rede externa
start /min "GovBot Interface" npm run dev -- --host

echo.
echo =====================================================
echo SISTEMA ONLINE!
echo.
echo 1. Para acessar nesta maquina: http://localhost:5173
echo 2. Para acessar da Intranet:   http://SEU_IP_AQUI:5173
echo.
echo (Nao feche as janelas pretas que abriram)
echo =====================================================
pause