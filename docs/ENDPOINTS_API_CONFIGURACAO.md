# Configuração de Endpoints das APIs (Gemini, GROQ, OpenAI)

## 🔧 Onde estão as configurações?

### **Arquivo: `backend/main.py` (linhas 233-252)**

```python
def _get_llm_instance():
    """
    Instancia o LLM conforme configurado em LLM_PROVIDER e LLM_MODEL
    Suporta: Gemini (Google), GROQ, OpenAI
    """
    if LLM_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)
    
    elif LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL, temperature=0.2)
    
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(api_key=OPENAI_API_KEY, model=LLM_MODEL, temperature=0.2)
```

## 📍 Endpoints Utilizados (Automáticos via LangChain)

As bibliotecas LangChain usam endpoints padrão e **não requerem configuração manual de URL**. Eles já estão pré-configurados:

### **1. Gemini (Google)**
- **Biblioteca**: `langchain-google-genai`
- **Endpoint**: `https://generativelanguage.googleapis.com/` (automático)
- **Autenticação**: Via `GOOGLE_API_KEY`
- **Configuração**: `ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)`

**Seu setup atual**:
```env
GOOGLE_API_KEY=AIzaSyB0-Gu8pOLIxSWeG-AQbzAxcraapXr_YAc
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
```

### **2. GROQ**
- **Biblioteca**: `langchain-groq`
- **Endpoint**: `https://api.groq.com/` (automático)
- **Autenticação**: Via `GROQ_API_KEY`
- **Configuração**: `ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL, temperature=0.2)`

**Para usar GROQ**:
```env
GROQ_API_KEY=gsk_seu_token_aqui
LLM_PROVIDER=groq
LLM_MODEL=mixtral-8x7b-32768
```

### **3. OpenAI**
- **Biblioteca**: `langchain-openai`
- **Endpoint**: `https://api.openai.com/v1/` (automático)
- **Autenticação**: Via `OPENAI_API_KEY`
- **Configuração**: `ChatOpenAI(api_key=OPENAI_API_KEY, model=LLM_MODEL, temperature=0.2)`

**Para usar OpenAI**:
```env
OPENAI_API_KEY=sk_seu_token_aqui
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
```

## ✅ Como Verificar se o Caminho está Correto?

### **1. Teste de Importação Básico**
```bash
cd backend
.\venv\Scripts\activate.ps1
python -c "import main; print('✅ Imports OK')"
```

Se aparecer erro de `ModuleNotFoundError`, falta instalar:
```bash
pip install langchain-groq langchain-openai langchain-google-genai
```

### **2. Teste de Conexão com a API**
```bash
python -c "
from main import _get_llm_instance
try:
    llm = _get_llm_instance()
    response = llm.invoke('Olá, tudo bem?')
    print('✅ API funcionando!')
    print(f'Resposta: {response.content[:100]}...')
except Exception as e:
    print(f'❌ Erro: {e}')
"
```

### **3. Verificar Chave de API**
```bash
# Gemini
echo %GOOGLE_API_KEY%

# GROQ
echo %GROQ_API_KEY%

# OpenAI
echo %OPENAI_API_KEY%
```

### **4. Teste Completo do Sistema**
```bash
python -m uvicorn main:app --reload
# Acesse http://localhost:8000/docs
# Teste POST /chat com pergunta simples
```

## 🔍 O que Verificar no Console

### ✅ **Sucesso**
```
🤖 LLM: GEMINI | Modelo: gemini-2.5-flash
✅ Carregando modelo local: C:\...\modelo_local\all-MiniLM-L6-v2
✅ Modelo LOCAL 'all-MiniLM-L6-v2' carregado com sucesso!
🔄 Gerando variações da pergunta...
HyDE (doc hipotético): ...
✅ Total de documentos únicos encontrados: 20
```

### ❌ **Erro - Chave não configurada**
```
RuntimeError: GOOGLE_API_KEY não configurada no .env
```
→ Solução: Adicione `GOOGLE_API_KEY=...` no `.env`

### ❌ **Erro - Rede bloqueada**
```
ConnectionError: Max retries exceeded with url: /api.groq.com
```
→ Motivo: Proxy da sua rede bloqueando GROQ
→ Solução: Use Gemini (que você já tem configurado)

### ❌ **Erro - Modelo não existe**
```
ValueError: Could not find model gpt-5000 (typo)
```
→ Solução: Verifique `LLM_MODEL` no `.env` (modelos válidos listados acima)

## 🎯 Resumo das Localizações

| Componente | Localização | Configuração |
|-----------|-----------|--------------|
| **Provider selection** | `backend/main.py` linha 233 | `LLM_PROVIDER` env var |
| **Gemini endpoint** | `langchain-google-genai` (automático) | `GOOGLE_API_KEY` |
| **GROQ endpoint** | `langchain-groq` (automático) | `GROQ_API_KEY` |
| **OpenAI endpoint** | `langchain-openai` (automático) | `OPENAI_API_KEY` |
| **Model name** | `backend/main.py` linha 237, etc | `LLM_MODEL` env var |
| **Temperature** | `backend/main.py` linha 237, etc | `temperature=0.2` (hardcoded) |

## 📝 Variáveis de Ambiente Requeridas

**Em `backend/.env`:**
```env
# Obrigatório: escolha o provider
LLM_PROVIDER=gemini

# Obrigatório: nome do modelo
LLM_MODEL=gemini-2.5-flash

# Chaves de API (apenas para o provider escolhido)
GOOGLE_API_KEY=...          # Para Gemini
GROQ_API_KEY=               # Para GROQ (deixar vazio se usar Gemini)
OPENAI_API_KEY=             # Para OpenAI (deixar vazio se usar Gemini)
```

## 🚀 Status Atual do Seu Sistema

```
✅ GEMINI: Configurado e funcionando
   - Endpoint: api.generativelanguage.googleapis.com
   - API Key: ✓ Presente
   - Modelo: gemini-2.5-flash
   
❌ GROQ: Não recomendado (sua rede bloqueada)
   - Endpoint: api.groq.com
   - API Key: Disponível mas rede bloqueada por proxy
   
⚪ OpenAI: Não configurado
   - Endpoint: api.openai.com
   - API Key: Não definida
```

---

**Tl;dr**: Os endpoints estão automáticos nas bibliotecas LangChain. Você só precisa garantir que:
1. ✅ `LLM_PROVIDER` está definido
2. ✅ `LLM_MODEL` é válido para esse provider
3. ✅ Chave de API está no `.env`
4. ✅ Rede permite conexão com a API (seu Gemini já funciona!)
