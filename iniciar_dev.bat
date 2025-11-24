@echo off
TITLE Launcher GovBot
echo =====================================================
echo      INICIANDO AMBIENTE DE DESENVOLVIMENTO
echo =====================================================

:: 1. Iniciar o Backend (Python) em nova janela
echo [1/2] Subindo API Python (Uvicorn)...
:: AQUI ESTAVA O ERRO: Adicionei --host 0.0.0.0 para aceitar conexoes do IP 10.x.x.x
start "GovBot API (Backend)" cmd /k "cd backend && venv\Scripts\activate && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

:: 2. Iniciar o Frontend (React) em nova janela
echo [2/2] Subindo Interface React (Vite)...
start "GovBot Interface (Frontend)" cmd /k "cd frontend && npm run dev"

echo.
echo ✅ Tudo rodando!
echo - API Local: http://localhost:8000/docs
echo - API Rede:  http://10.61.172.6:8000/docs
echo - App:       http://localhost:5173
echo.
echo (Nao feche as janelas pretas que se abriram)
echo =====================================================
pause