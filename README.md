# 🏛️ GovBot - Chatbot Institucional com RAG

Chatbot inteligente baseado em RAG (Retrieval-Augmented Generation) para ambientes corporativos e institucionais. Integra búsca semântica + BM25 com LLM (Google Gemini) para respostas precisas sobre documentos institucionais.

## 🎯 Funcionalidades

- ✅ **Upload de Documentos**: PDF e DOCX com análise automática de conteúdo
- 📊 **RAG Híbrido**: Busca semântica + keyword (BM25) + optional HyDE
- 🧠 **LLM Flexível**: Suporte para Gemini 2.5/2.0/3 Flash Preview
- 🔐 **Autenticação**: Admin panel com credenciais
- 📱 **Interface Moderno**: React + Vite + Tailwind CSS
- 🚀 **Performance**: Query variations otimizadas (instantâneas)
- 🌐 **CORS Aberto**: Fácil integração em LAN

## 📋 Pré-requisitos

- **Python 3.9+**
- **Node.js 16+**
- **Google Gemini API Key** (obtenha em [AI Studio](https://aistudio.google.com))

## 🚀 Instalação Rápida

### 1️⃣ Clonar e Preparar Ambiente

```bash
git clone <seu-repo>
cd chatbot-institucional
configurar_ambiente.bat  # Windows: Cria venv + instala dependências Python + npm
```

**Linux/Mac:**
```bash
python -m venv backend/venv
source backend/venv/bin/activate
pip install -r backend/requirements.txt
cd frontend && npm install && cd ..
```

### 2️⃣ Configurar `.env`

Copie `.env.example` para `.env` e preencha:

```env
GOOGLE_API_KEY=sua-chave-gemini-aqui
ADMIN_PASS=sua-senha-admin-segura
```

**Opcional:**
```env
LLM_MODEL=gemini-2.5-flash          # ou gemini-3-flash-preview
HYDE_ENABLED=false                   # true para ativar HyDE (mais lento)
HTTP_PROXY=http://proxy:porta        # Se em rede corporativa
HTTPS_PROXY=http://proxy:porta
```

### 3️⃣ Desenvolver

```bash
iniciar_dev.bat  # Windows: Inicia backend + frontend em modo dev
```

**Linux/Mac:**
```bash
# Terminal 1: Backend
cd backend
source venv/bin/activate
uvicorn main:app --reload

# Terminal 2: Frontend
cd frontend
npm run dev
```

### 4️⃣ Acessar

- 🔵 **Chat**: [http://localhost:5173](http://localhost:5173)
- 🟠 **Admin**: [http://localhost:5173/admin](http://localhost:5173/admin)
- 🔴 **API**: [http://localhost:8000](http://localhost:8000)

## 📚 Uso da API

### Chat (Público)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Qual é a portaria X?"}'
```

**Response:**
```json
{
  "answer": "A portaria X estabelece...",
  "sources": [
    {"name": "Boletim.pdf", "page": 12},
    {"name": "Normas.docx", "page": "N/A"}
  ]
}
```

### Admin Endpoints (Autenticação Basic Auth)

**Listar Documentos:**
```bash
curl -X GET http://localhost:8000/documentos \
  -H "Authorization: Basic YWRtaW46c2VudGhhMTIz"
```

**Upload:**
```bash
curl -X POST http://localhost:8000/upload \
  -H "Authorization: Basic YWRtaW46c2VudGhhMTIz" \
  -F "file=@documento.pdf"
```

**Limpar Base:**
```bash
curl -X DELETE http://localhost:8000/limpar_base \
  -H "Authorization: Basic YWRtaW06c2VudGhhMTIz"
```

## 🏗️ Arquitetura

```
┌─────────────────┐
│  React UI       │ (http://localhost:5173)
│  /admin         │
│  /chat          │
└────────┬────────┘
         │ HTTP
         ↓
┌─────────────────────────────────────┐
│        FastAPI Backend              │
│  - POST /chat                       │
│  - GET /documentos                  │
│  - POST /upload                     │
│  - DELETE /limpar_base              │
└────────┬────────────────────────────┘
         │
    ┌────┴────┐
    ↓         ↓
┌─────────┐ ┌──────────────┐
│Chroma   │ │Sentence      │
│Vector   │ │Transformers  │
│Store    │ │(embeddings)  │
└─────────┘ └──────────────┘
    
    ↓ (Recupera contexto)
    
┌──────────────────────────┐
│  Google Gemini LLM       │
│  Gera respostas          │
└──────────────────────────┘
```

## 🔄 Pipeline RAG

1. **Query Variations** (~50ms): Gera 2 variações localmente
2. **HyDE** (~2-3s, opcional): Gera documento hipotético (se ativado)
3. **Semantic Search** (~200ms): Busca em embeddings (k=20)
4. **BM25 Search** (~50ms): Busca por keywords
5. **Merge & Filter**: Deduplica e combina resultados
6. **LLM Answer** (~3-5s): Gemini gera resposta
7. **Cleanup** (~10ms): Remove markdown

**Total: ~4-6 segundos (otimizado)**

## 📁 Estrutura do Projeto

```
chatbot-institucional/
├── backend/
│   ├── main.py                    # API FastAPI + RAG logic
│   ├── proxy_config.py           # Configuração de proxy corporativo
│   ├── sharepoint_watcher.py    # Monitor de pasta sincronizada
│   ├── requirements.txt          # Dependências Python
│   ├── modelo_local/            # Embeddings offline
│   ├── db_chroma/               # 🚫 Vector store (git ignored)
│   ├── uploads/                 # 🚫 Arquivos upados (git ignored)
│   └── venv/                    # 🚫 Virtualenv (git ignored)
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── ChatbotPage.jsx  # Interface de chat
│   │   │   └── AdminPage.jsx    # Painel administrativo
│   │   ├── main.jsx
│   │   └── index.css
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── 🚫 node_modules/ (git ignored)
│
├── .env.example                 # Template de configuração
├── .gitignore                   # Git ignore list
├── configurar_ambiente.bat      # Setup initial (Windows)
├── iniciar_dev.bat              # Dev mode (Windows)
├── README.md                    # Este arquivo
└── .github/
    └── copilot-instructions.md  # Instruções para AI agents
```

## ⚙️ Configuração Avançada

### Embeddings Offline

Coloque arquivos de `sentence-transformers/all-MiniLM-L6-v2` em `backend/modelo_local/`:

```
backend/modelo_local/
├── config.json
├── model.safetensors
├── tokenizer.json
├── vocab.txt
└── special_tokens_map.json
```

Se vazio, tentará download online (pode ser bloqueado por proxy).

### SharePoint Watcher

Monitora pasta sincronizada do SharePoint e faz upload automático:

```env
PASTA_SHAREPOINT=C:\Usuarios\Seu_Usuario\Documentos\SharePoint
API_URL=http://localhost:8000
ADMIN_USER=admin
ADMIN_PASS=sua-senha
```

Execute: `python sharepoint_watcher.py`

### Proxy Corporativo

Se está em rede corporativa com proxy:

```env
HTTP_PROXY=http://usuario:senha@proxy-ip:porta
HTTPS_PROXY=http://usuario:senha@proxy-ip:porta
```

O `proxy_config.py` é chamado automaticamente no startup.

## 🐛 Troubleshooting

| Erro | Solução |
|------|---------|
| `407 Proxy Authentication Required` | Configure HTTP_PROXY com credenciais em `.env` |
| `'list' object has no attribute` | Compatibilidade com Gemini 3 Flash (já fixado) |
| `Embeddings download bloqueado` | Coloque modelo em `backend/modelo_local/` |
| `Porta 8000/5173 em uso` | Altere em `main.py` e `vite.config.js` |

## 🔐 Segurança

⚠️ **Importante:**
- Nunca commite `.env` (já em `.gitignore`)
- Mude `ADMIN_PASS` em produção
- Use HTTPS em produção
- Considere rate limiting para `/chat`

## 📦 Deploy

### Produção com Docker

```dockerfile
FROM python:3.11
WORKDIR /app
COPY backend requirements.txt .
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
```

### Ambiente Virtual Simples

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

pip install -r backend/requirements.txt
cd backend && uvicorn main:app --host 0.0.0.0
```

## 🤝 Contribuindo

1. Fork o repositório
2. Crie branch (`git checkout -b feature/AmazingFeature`)
3. Commit mudanças (`git commit -m 'Add AmazingFeature'`)
4. Push para branch (`git push origin feature/AmazingFeature`)
5. Abra Pull Request

## 📝 Licença

MIT License

## 📞 Suporte

Para issues e dúvidas, abra uma [issue no GitHub](https://github.com/seu-usuario/chatbot-institucional/issues)

---

**Desenvolvido com ❤️ para instituições**
