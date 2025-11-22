@echo off
echo 🚀 Iniciando Projeto GovBot...
echo.

echo 🔧 Configurando Backend...
cd backend
echo Instalando dependências Python...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Erro ao instalar dependências do backend
    pause
    exit /b 1
)

echo.
echo ⚙️ Configurando Frontend...
cd ..\frontend
echo Instalando dependências Node.js...
npm install
if errorlevel 1 (
    echo ❌ Erro ao instalar dependências do frontend
    pause
    exit /b 1
)

echo.
echo ✅ Configuração concluída!
echo.
echo 📝 PRÓXIMOS PASSOS:
echo 1. Configure o arquivo .env no backend com sua GOOGLE_API_KEY
echo 2. Execute o servidor backend: cd backend && python main.py
echo 3. Execute o frontend: cd frontend && npm run dev
echo.
pause