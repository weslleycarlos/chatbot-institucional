🏛️ Chatbot Institucional com RAG (Intranet)

Assistente Virtual Inteligente projetado para operar em ambientes corporativos restritos (Intranet / Shadow IT).
Utiliza RAG (Retrieval-Augmented Generation) para responder perguntas usando documentos internos (PDFs, Portarias, Manuais) mantendo mínimas alucinações.

🎯 Funcionalidades

RAG Local: Indexação de documentos PDF e DOCX em banco vetorial local (ChromaDB).

Zero-Admin: Arquitetura projetada para rodar sem permissões de administrador no Windows.

SharePoint Sync: Watcher para detectar arquivos sincronizados via OneDrive e indexá-los automaticamente.

Interface Segura: Painel administrativo protegido por Basic Auth.

Proxy Aware: Configuração automática para operar atrás de proxies corporativos.

🏗️ Arquitetura

O sistema segue o modelo Hub-and-Spoke local:

Frontend (React/Vite): Interface do usuário.

Backend (FastAPI): Gerencia LangChain + Google Gemini.

Storage (ChromaDB): Banco vetorial local persistido em arquivos (sem dependência de SQL Server, Postgres etc.).

Diagrama (Mermaid)
graph LR
    A[Usuário Intranet] -->|Browser| B(React Frontend)
    B -->|JSON| C(FastAPI Backend)
    C -->|Busca| D[(ChromaDB Local)]
    C -->|Contexto + Prompt| E[Google Gemini API]
    F[SharePoint Watcher] -->|Novo Arquivo| C

🚀 Como Rodar
✔ Pré-requisitos

Python 3.9+

Node.js (para desenvolvimento do front)

Google Gemini API Key

🛠️ Instalação
1. Clone o repositório
git clone <seu-repo>

2. Instale o Backend
cd backend
pip install -r requirements.txt

3. Instale o Frontend
cd frontend
npm install

4. Configure a sua API Key

No arquivo:

backend/main.py


ou use variáveis de ambiente:

set GEMINI_API_KEY=sua-chave

▶️ Execução

Na raiz do projeto, execute:

INICIAR_SISTEMA.bat

🛡️ Segurança e Privacidade

Auth: Upload de arquivos exige autenticação.

Dados Locais: Todos os documentos permanecem na rede corporativa.

Privacidade: Apenas trechos anonimizados são enviados à LLM para interpretação semântica.

Projeto desenvolvido para fins de portfólio de Engenharia de Software e IA.