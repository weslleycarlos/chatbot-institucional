@echo off
echo ==========================================
echo      CONFIGURACAO DO GOVBOT (SAFE)
echo ==========================================
echo.

:: --- BACKEND ---
echo [1/2] Configurando Backend (Python)...
cd backend

:: Verifica se o Python esta acessivel
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado! Instale o Python e marque "Add to PATH".
    pause
    exit /b 1
)

:: Cria a venv se nao existir
if not exist venv (
    echo    Criando ambiente virtual...
    python -m venv venv
)

:: Ativa a venv
echo    Ativando venv...
call venv\Scripts\activate

:: PASSOS DE CORRECAO DE ERRO DE COMPILACAO
echo    1. Atualizando pip...
python -m pip install --upgrade pip

echo    2. Instalando Numpy separadamente (evita erro de compilacao)...
pip install numpy

:: Instala dependencias
echo    3. Instalando o restante das bibliotecas...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERRO] Falha ao instalar dependencias do Python.
    pause
    exit /b 1
)

:: --- FRONTEND ---
echo.
echo [2/2] Configurando Frontend (Node.js)...
cd ..\frontend

:: Verifica se npm esta acessivel
call npm --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Node.js/NPM nao encontrado! Instale o Node.js.
    pause
    exit /b 1
)

echo    Instalando dependencias do React...
call npm install
if errorlevel 1 (
    echo [ERRO] Falha ao instalar dependencias do Frontend.
    pause
    exit /b 1
)

:: --- FINALIZAÇÃO ---
cd ..
echo.
echo ==========================================
echo      CONFIGURACAO CONCLUIDA!
echo ==========================================
echo.
echo Proximos passos:
echo 1. Configure sua API Key no backend/main.py
echo 2. Execute iniciar_dev.bat
echo.
pause