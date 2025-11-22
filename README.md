🏛️ Chatbot Institucional com RAG (Intranet)Este projeto é um Assistente Virtual Inteligente projetado para operar em ambientes corporativos restritos (Intranet/Shadow IT). Ele utiliza RAG (Retrieval-Augmented Generation) para responder perguntas baseadas em documentos internos (PDFs, Portarias, Manuais) sem alucinações.🎯 FuncionalidadesRAG Local: Indexação de documentos PDF e DOCX em banco vetorial local (ChromaDB).Zero-Admin: Arquitetura desenhada para rodar sem permissões de administrador no Windows.SharePoint Sync: Script watcher que detecta arquivos sincronizados via OneDrive e indexa automaticamente.Interface Segura: Painel administrativo protegido por Basic Auth.Proxy Aware: Configuração automática para lidar com proxies corporativos.🏗️ ArquiteturaO sistema opera em modelo Hub-and-Spoke local:Frontend (React/Vite): Interface do usuário.Backend (FastAPI): API que gerencia o LangChain e Google Gemini.Storage (ChromaDB): Persistência vetorial em arquivos (sem instalação de banco SQL).graph LR
    A[Usuário Intranet] -->|Browser| B(React Frontend)
    B -->|JSON| C(FastAPI Backend)
    C -->|Busca| D[(ChromaDB Local)]
    C -->|Contexto + Prompt| E[Google Gemini API]
    F[SharePoint Watcher] -->|Novo Arquivo| C
🚀 Como RodarPré-requisitosPython 3.9+Node.js (para desenvolvimento do front)Google Gemini API KeyInstalaçãoClone o repositório.Instale as dependências do Backend:cd backend
pip install -r requirements.txt
Instale as dependências do Frontend:cd frontend
npm install
Configure sua API Key no arquivo backend/main.py (ou variáveis de ambiente).ExecuçãoBasta rodar o script de inicialização na raiz:INICIAR_SISTEMA.bat🛡️ Segurança e PrivacidadeAuth: O upload de arquivos exige autenticação.Dados: Os arquivos processados residem na rede local. Apenas trechos anonimizados são enviados para a LLM para processamento semântico.Projeto desenvolvido para fins de portfólio de Engenharia de Software e IA.