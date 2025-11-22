@echo off
echo 🛠️ Modo Desenvolvimento - GovBot
echo.

echo 📚 Iniciando Backend (FastAPI)...
start cmd /k "cd backend && python main.py"

timeout /t 3 /nobreak >nul

echo 🌐 Iniciando Frontend (React)...
start cmd /k "cd frontend && npm run dev"

echo.
echo ✅ Ambientes iniciados!
echo 📍 Backend: http://localhost:8000
echo 📍 Frontend: http://localhost:5173
echo.
pause