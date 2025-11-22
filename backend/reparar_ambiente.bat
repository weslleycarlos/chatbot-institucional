@echo off
echo =================================================
echo      REPARADOR DE AMBIENTE PYTHON (GOVBOT)
echo =================================================
echo.
echo 1. Verificando ambiente virtual...
if not exist venv (
    echo [AVISO] Pasta venv nao encontrada. Criando do zero...
    python -m venv venv
)

echo 2. Ativando ambiente...
call venv\Scripts\activate

echo.
echo 3. FORCANDO REINSTALACAO DO LANGCHAIN...
echo (Isso resolve o erro 'No module named langchain.chains')
echo.
:: Desinstala versoes que podem estar em conflito
pip uninstall -y langchain langchain-community langchain-core langchain-google-genai

:: Instala as versoes corretas do requirements.txt
pip install -r requirements.txt

echo.
echo =================================================
echo               CONCLUIDO!
echo =================================================
echo Tente rodar o servidor novamente com:
echo uvicorn main:app --reload
echo.
pause