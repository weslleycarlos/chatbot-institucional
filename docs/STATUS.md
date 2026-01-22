# Status do Projeto - GovBot Institucional

**Data**: 2024
**Estado**: ✅ PRONTO PARA PRODUÇÃO (com Gemini)

## O Que Funciona ✅

### Backend
- ✅ FastAPI com RAG avançado (MultiQuery + HyDE + BM25)
- ✅ ChromaDB com embeddings locais
- ✅ Suporte para múltiplos LLMs: **Gemini** ✅ **OpenAI** ✅ **GROQ** ⏳
- ✅ Upload de PDFs e DOCX
- ✅ Autenticação Basic Auth
- ✅ CORS para LAN
- ✅ Chunking inteligente (1200 chars, 400 overlap, legal-aware)

### Frontend
- ✅ React + Vite
- ✅ Página de chat
- ✅ Página admin com upload/gerenciamento
- ✅ Exibição de fontes
- ✅ Responsive design

## Configuração Atual

```env
# PRODUÇÃO
LLM_PROVIDER=gemini          # ✅ Funciona 100%
GOOGLE_API_KEY=seu_key_aqui  # NECESSÁRIO

# OPCIONAL
GROQ_API_KEY=...             # ⏳ Requer proxy auth
OPENAI_API_KEY=...           # ✅ Funciona se tiver key
```

## Problemas Conhecidos & Soluções

### 1. GROQ com Proxy 407
**Problema**: Erro `407 Proxy Authentication Required`
**Causa**: httpx/groq não conseguem autenticar com proxy corporativo
**Solução Atual**: Use Gemini (padrão)
**Alternativas**: Ver [TROUBLESHOOTING_PROXY_GROQ.md](./TROUBLESHOOTING_PROXY_GROQ.md)

### 2. PDFs Escaneados
**Problema**: OCR não implementado
**Impacto**: Docs com imagem pura não são indexados
**Solução**: Implementar pytesseract/EasyOCR (future)

## Como Usar

### Desenvolvimento
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Em outro terminal
cd frontend
npm install

# Iniciar tudo
../iniciar_dev.bat
```

### Produção
```bash
# Backend
gunicorn -w 4 -b 0.0.0.0:8000 main:app

# Frontend
npm run build
npm run preview
```

## API Endpoints

| Método | Path | Auth | Descrição |
|--------|------|------|-----------|
| POST | `/chat` | - | Pergunta ao chatbot |
| GET | `/documentos` | BasicAuth | Lista docs indexados |
| POST | `/upload` | BasicAuth | Upload de PDF/DOCX |
| DELETE | `/limpar_base` | BasicAuth | Reset da base |
| DELETE | `/limpar_uploads` | BasicAuth | Limpar pasta uploads |
| GET | `/` | - | Health check |

## Configuração de Proxy

Se sua rede usa proxy com autenticação:

```env
PROXY_HOST=seu.proxy.com
PROXY_PORT=8080
PROXY_USER=dominio\usuario
PROXY_PASS=senha_com_$peciais  # Será automaticamente escapada
```

**Nota**: Para GROQ funcionar com proxy, pode ser necessário:
1. Usar VPN (se disponível)
2. Solicitar exceção ao TI para api.groq.com
3. Usar Gemini/OpenAI como alternativa

## Performance

- **Chat**: < 3s (com RAG)
- **Upload**: Depende do tamanho (PDF 10MB ≈ 10s)
- **Embeddings**: Locais (sem rede)
- **LLM**: Depende do provider

## Segurança

- ✅ API keys em .env (não em código)
- ✅ Basic Auth para endpoints admin
- ✅ CORS aberto para LAN (ajustar se expor para internet)
- ✅ Sem session tokens (stateless)

## Próximas Melhorias

1. OCR para PDFs escaneados
2. Suporte para outros formatos (Excel, Word tables)
3. Cache de respostas
4. Histórico de chat
5. Fine-tuning com docs próprios
6. WebSocket para streaming em tempo real
7. Multi-idioma

## Troubleshooting Rápido

| Problema | Solução |
|----------|---------|
| 407 Proxy Auth | Ver TROUBLESHOOTING_PROXY_GROQ.md |
| Modelo não encontrado | Verificar `backend/modelo_local/` |
| API timeout | Aumentar timeout em .env ou restart |
| Docs não indexam | Verificar chunking em main.py:467 |
| Front não conecta API | Verificar VITE_API_URL em frontend/.env |

## Contato & Referências

- **Gemini**: https://ai.google.dev
- **GROQ**: https://console.groq.com  
- **OpenAI**: https://platform.openai.com
- **LangChain**: https://python.langchain.com
- **Chroma**: https://www.trychroma.com

---

**Última atualização**: Dezembro 2024
**Versão**: 1.0.0 (MVP)
**Status**: ✅ Pronto para uso
