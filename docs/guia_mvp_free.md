Guia Técnico: Chatbot RAG "Custo Zero" e MultiformatoEste documento detalha como implementar os requisitos avançados (Upload Admin, Feedback e Stack Gratuita) no seu projeto de portfólio.1. Stack Tecnológica 100% Free (MVP)Para viabilizar o projeto sem custos mensais, utilizaremos serviços com tiers gratuitos generosos.ComponenteFerramenta RecomendadaPor que?BackendFastAPI (Python)Mais rápido que Flask e gera documentação automática (Swagger) para seu portfólio.HospedagemRender.comPermite hospedar serviços Python gratuitamente (Web Services). Obs: Desliga após inatividade (cold start), mas é free.LLMGoogle Gemini APIO modelo gemini-1.5-flash tem um tier gratuito excelente e suporta janelas de contexto longas.Banco VetorialChromaDB (Local/Ephemeral)Roda dentro do container do Render. Não precisa de servidor externo. Os dados persistem no disco do servidor (cuidado com reinicializações no free tier) ou use Pinecone (tem free tier cloud).FrontendVercelMelhor lugar para hospedar React/Next.js grátis.2. Implementando Múltiplos FormatosO segredo para aceitar PDF, DOCX e Web não é a IA, mas o extrator de texto (Parser).Bibliotecas Python Essenciais:No seu requirements.txt, você precisará de:langchain-community
pypdf          # Para PDFs
python-docx    # Para arquivos Word
beautifulsoup4 # Para raspar sites (HTML)
yt_dlp         # (Opcional) Se quiser transcrever vídeos do YouTube
openai-whisper # (Opcional) Para áudio -> texto
Exemplo de Função de "Roteamento de Arquivo":def load_file(file_path):
    extension = file_path.split('.')[-1]
    
    if extension == 'pdf':
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(file_path).load()
        
    elif extension == 'docx':
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(file_path).load()
        
    elif extension == 'txt':
        from langchain_community.document_loaders import TextLoader
        return TextLoader(file_path).load()
        
    else:
        raise ValueError("Formato não suportado")
3. Sistema de Feedback (RLHF Simplificado)Quando o usuário clica no "Joinha para Baixo" (Thumbs Down), o frontend deve enviar um JSON para o backend. Como não queremos pagar um banco SQL (Postgres), podemos usar soluções NoSQL gratuitas ou arquivos simples para o MVP.Payload do Feedback:{
  "message_id": "msg_123",
  "user_query": "Como tiro passaporte?",
  "bot_response": "Vá na prefeitura...",
  "feedback": "negative",
  "timestamp": "2024-05-20T10:00:00Z"
}
Onde salvar de graça:Google Sheets API: Use uma planilha do Google como banco de dados! É grátis e fácil de visualizar.Firebase Firestore: O tier Spark (gratuito) aguenta muito tráfego de texto simples.4. Workflow da Interface AdministrativaPara o upload funcionar no Admin Panel:O Frontend envia o arquivo via multipart/form-data para o endpoint /upload.O Backend salva o arquivo temporariamente.O Backend roda o load_file() + text_splitter.O Backend gera embeddings e salva no VectorStore.O Backend retorna "Sucesso" e o frontend atualiza a lista.Dica de Ouro para PortfólioNo seu README do GitHub, crie um diagrama (pode ser no Mermaid.js) mostrando esse fluxo:Usuário -> React -> FastAPI -> ChromaDB -> Gemini API.Isso demonstra maturidade técnica muito além de apenas "usar uma API".