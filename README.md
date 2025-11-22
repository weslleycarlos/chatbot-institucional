# 🏛️ GovBot - Chatbot Institucional

Sistema de chatbot com RAG para ambientes corporativos.

## 🚀 Como Usar

### Pré-requisitos
- Python 3.9+
- Node.js 16+
- Google Gemini API Key

### Instalação Rápida
1. Execute `iniciar_projeto.bat`
2. Configure o `.env` no backend
3. Execute `desenvolver.bat`

### Desenvolvimento
- Backend: FastAPI (http://localhost:8000)
- Frontend: React (http://localhost:5173)

## 📁 Estrutura
projeto/
├── backend/ # FastAPI + LangChain
├── frontend/ # React + Vite
├── deploy_git.bat # Deploy para Git
├── desenvolver.bat # Desenvolvimento
└── README.md


## 🔧 Configuração
Crie `backend/.env`:
```env
GOOGLE_API_KEY=sua_chave_aqui
ADMIN_PASS=sua_senha_aqui