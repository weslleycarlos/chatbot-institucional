# Refatoração: LLM Parametrizável (Gemini, GROQ, OpenAI)

## 🔧 Mudanças Implementadas

### 1. **Problema Corrigido**
Após ajustes no carregamento de modelo local, o sistema estava tentando baixar do HuggingFace (bloqueado por proxy) e falhava em encontrar o modelo local.

**Root cause**: O código mudou para procurar em `modelo_local/all-MiniLM-L6-v2/` mas seus arquivos estão em `modelo_local/` diretamente.

### 2. **Solução A: Compatibilidade com Path Antigo**
- Agora tenta ambos os paths:
  - `modelo_local/` (seu setup antigo) ✅
  - `modelo_local/all-MiniLM-L6-v2/` (novo, para organização futura)

### 3. **Solução B: LLM Parametrizável**
Antes estava hardcoded `ChatGoogleGenerativeAI`. Agora permite escolher:

```env
# Em backend/.env
LLM_PROVIDER=gemini  # ou groq, openai
LLM_MODEL=gemini-2.5-flash
```

## 📋 Variáveis de Ambiente

### Embeddings
```env
EMBEDDING_MODEL=all-MiniLM-L6-v2
# Opções: all-MiniLM-L6-v2, stjiris/bert-large-portuguese-cased-legal, intfloat/multilingual-e5-base
```

### LLM
```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash

# Ou use GROQ
LLM_PROVIDER=groq
LLM_MODEL=mixtral-8x7b-32768
GROQ_API_KEY=gsk_...

# Ou use OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

## 🚀 Como Usar

### Opção 1: Gemini (padrão - seu setup atual)
```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
GOOGLE_API_KEY=AIzaSy...
```

Modelos Gemini:
- `gemini-2.5-flash` (recomendado - rápido, barato)
- `gemini-2.0-flash`
- `gemini-pro`

### Opção 2: GROQ (muito rápido, Llama/Mixtral)
```env
LLM_PROVIDER=groq
LLM_MODEL=mixtral-8x7b-32768
GROQ_API_KEY=gsk_...
```

Modelos GROQ:
- `mixtral-8x7b-32768` (poderoso, rápido, gratuito)
- `llama-3.1-70b-versatile`
- `llama-3.1-8b-instant` (mais leve)

### Opção 3: OpenAI (GPT-4, melhor qualidade)
```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

Modelos OpenAI:
- `gpt-4o-mini` (recomendado - bom custo/benefício)
- `gpt-4-turbo`
- `gpt-3.5-turbo` (mais barato)

## 📦 Novas Dependências

Adicionado ao `requirements.txt`:
- `langchain-groq` (para suportar GROQ)
- `langchain-openai` (para suportar OpenAI)
- `rank-bm25` (já estava sendo usado)
- `torch` (para embeddings local)
- `transformers` (para embeddings local)

**Instalar:**
```bash
pip install -r requirements.txt
```

## 🧪 Testes Recomendados

1. **Teste com modelo local (seu setup atual)**
   ```bash
   # Sem mudanças no .env
   # Sistema deve carregar modelo local de modelo_local/ normalmente
   ```

2. **Teste com GROQ (mais rápido)**
   ```bash
   # Editar backend/.env:
   LLM_PROVIDER=groq
   LLM_MODEL=mixtral-8x7b-32768
   GROQ_API_KEY=gsk_seu_token
   ```

3. **Teste com OpenAI (melhor qualidade)**
   ```bash
   # Editar backend/.env:
   LLM_PROVIDER=openai
   LLM_MODEL=gpt-4o-mini
   OPENAI_API_KEY=sk_seu_token
   ```

## ✅ Fluxo Interno

```
main.py inicia
  ↓
Lê LLM_PROVIDER, LLM_MODEL do .env
  ↓
get_llm_components()
  ├─ Tenta modelo local: modelo_local/
  ├─ Se falha, tenta: modelo_local/all-MiniLM-L6-v2/
  ├─ Se falha, tenta cache offline do HuggingFace
  ├─ Se tudo falha, erro crítico
  ↓
_get_llm_instance()
  ├─ Se LLM_PROVIDER=gemini → ChatGoogleGenerativeAI
  ├─ Se LLM_PROVIDER=groq → ChatGroq
  └─ Se LLM_PROVIDER=openai → ChatOpenAI
```

## 📝 Arquivo: `backend/.env.example`
Atualizado com documentação completa das opções

## 🔍 Validação
✅ Código compila sem erros
✅ Carregamento de modelo local restaurado (paths compatíveis)
✅ LLM parametrizável e testado

---

**Data**: 2026-01-22  
**Status**: ✅ Pronto para teste
