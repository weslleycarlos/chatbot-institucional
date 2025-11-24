import os
import shutil
import glob
import time
import secrets
import sys
import gc
from contextlib import asynccontextmanager
from typing import List, Optional
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

vectorstore = None
_chroma_client = None

# --- DIAGNÓSTICO DE IMPORTAÇÃO ---
try:
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough

except ImportError as e:
    print("\n" + "="*60)
    print("ERRO CRÍTICO DE BIBLIOTECA FALTANDO")
    print(f"O Python não encontrou: {e}")
    print("SOLUÇÃO: pip install langchain-huggingface python-dotenv psutil")
    print("="*60 + "\n")
    sys.exit(1)

# Framework API
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from proxy_config import configurar_proxy
except ImportError:
    def configurar_proxy(): pass

# --- 1. CONFIGURAÇÕES ---

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PERSIST_DIRECTORY = "./db_chroma"
UPLOAD_DIR = "./uploads"
MODELO_LOCAL_DIR = "./modelo_local" # Pasta onde os arquivos baixados DEVEM estar

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODELO_LOCAL_DIR, exist_ok=True) # Cria a pasta se não existir

ADMIN_USER = "admin"
ADMIN_PASS = "senha_secreta_123"

# --- 2. APP ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 Inicializando GovBot API...")
    if not GOOGLE_API_KEY:
        print("⚠️ AVISO: GOOGLE_API_KEY não encontrada no arquivo .env")
    
    # Verificação CRÍTICA do Modelo Offline
    if not os.listdir(MODELO_LOCAL_DIR):
        print("\n" + "!"*60)
        print("❌ ERRO FATAL: MODELO DE IA NÃO ENCONTRADO LOCALMENTE")
        print(f"A pasta '{MODELO_LOCAL_DIR}' está vazia.")
        print("Devido ao Proxy da sua rede, o download automático falhou.")
        print("SOLUÇÃO: Baixe os arquivos do modelo 'sentence-transformers/all-MiniLM-L6-v2'")
        print("e coloque-os manualmente dentro da pasta backend/modelo_local.")
        print("!"*60 + "\n")
    
    configurar_proxy()
    yield
    print("🛑 Desligando GovBot API...")
    close_vectorstore()

app = FastAPI(title="GovBot Intranet API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBasic()

# --- 3. AUXILIARES ---

def verificar_credenciais(credentials: HTTPBasicCredentials = Depends(security)):
    is_user_ok = secrets.compare_digest(credentials.username, ADMIN_USER)
    is_pass_ok = secrets.compare_digest(credentials.password, ADMIN_PASS)
    if not (is_user_ok and is_pass_ok):
        raise HTTPException(
            status_code=401, 
            detail="Não autorizado", 
            headers={"WWW-Authenticate": "Basic"}
        )
    return credentials.username

def get_llm_components():
    """
    Retorna embeddings (Local) e LLM (Google).
    Agora força o uso da pasta local para evitar travamento no Proxy.
    """
    # Se a pasta estiver vazia, usamos o nome online como fallback, 
    # mas isso provavelmente falhará no seu proxy.
    model_source = MODELO_LOCAL_DIR
    if not os.listdir(MODELO_LOCAL_DIR):
        print("⚠️ AVISO: Usando download online (pode falhar no proxy)...")
        model_source = "sentence-transformers/all-MiniLM-L6-v2"
    else:
        print(f"✅ Usando modelo OFFLINE carregado de: {MODELO_LOCAL_DIR}")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_source,
        model_kwargs={'device': 'cpu'}
    )
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite-preview-02-05", temperature=0.2)
    return embeddings, llm

def get_vectorstore():
    global vectorstore, _chroma_client
    
    if vectorstore is None:
        embeddings, _ = get_llm_components()
        
        import chromadb
        from chromadb.config import Settings
        
        _chroma_client = chromadb.PersistentClient(
            path=PERSIST_DIRECTORY,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        
        vectorstore = Chroma(
            client=_chroma_client,
            embedding_function=embeddings,
            collection_name="govbot_docs"
        )
    return vectorstore

def close_vectorstore():
    global vectorstore, _chroma_client
    print("🔒 Fechando Chroma...")
    try:
        if vectorstore:
            vectorstore = None
        if _chroma_client:
            try: _chroma_client.clear_system_cache()
            except: pass
            _chroma_client = None
        
        gc.collect()
        time.sleep(1) 
    except Exception as e:
        print(f"⚠️ Erro ao fechar: {e}")

# --- 4. ENDPOINTS ---

class QueryRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    try:
        embeddings, llm = get_llm_components()
        vs = get_vectorstore()
        
        # Busca mais chunks para ter mais contexto
        retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        
        system_prompt = (
            "Você é um assistente especializado em documentos oficiais. "
            "Responda com base EXCLUSIVAMENTE nos contextos fornecidos abaixo. "
            "Se a resposta não estiver no contexto, diga que não sabe. "
            "Cite o nome do documento fonte sempre que possível."
            "\n\nContextos:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])
        
        def format_docs(docs):
            return "\n\n".join(f"[Doc: {d.metadata.get('source', 'Desconhecido')}] {d.page_content}" for d in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
        )
        
        response = rag_chain.invoke(request.question)
        
        # Recupera fontes para o frontend
        sources = []
        docs = retriever.invoke(request.question)
        for doc in docs:
            sources.append({
                "name": os.path.basename(doc.metadata.get('source', 'Desconhecido')),
                "page": doc.metadata.get('page', 'N/A')
            })

        return {"answer": response.content, "sources": sources}

    except Exception as e:
        print(f"❌ Erro no Chat: {e}")
        # Retorna erro JSON em vez de 500 para o frontend tratar
        return {"answer": "Desculpe, ocorreu um erro interno no servidor de IA.", "sources": []}

@app.get("/documentos")
async def listar_documentos(username: str = Depends(verificar_credenciais)):
    try:
        vs = get_vectorstore()
        docs = vs.get()
        unique_docs = list(set(
            os.path.basename(m.get('source', '')) for m in docs['metadatas'] if m.get('source')
        ))
        return {"documentos": unique_docs, "total_chunks": len(docs['ids'])}
    except Exception as e:
        return {"documentos": [], "erro": str(e)}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), username: str = Depends(verificar_credenciais)):
    try:
        print(f"📥 Recebendo upload: {file.filename}")
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if file.filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.filename.lower().endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            return {"error": "Apenas PDF ou DOCX"}
        
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)

        vs = get_vectorstore()
        vs.add_documents(chunks)
        
        print(f"✅ Indexado: {len(chunks)} trechos.")
        return {"status": "sucesso", "chunks": len(chunks), "documento": file.filename}

    except Exception as e:
        print(f"❌ Erro Upload: {e}")
        return {"status": "erro", "detalhes": str(e)}

@app.delete("/limpar_base")
async def limpar_base(username: str = Depends(verificar_credenciais)):
    global vectorstore, _chroma_client
    try:
        if _chroma_client:
            _chroma_client.reset()
        close_vectorstore()
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR)
        
        # Re-inicializa
        get_vectorstore()
        return {"status": "Base limpa com sucesso"}
    except Exception as e:
        return {"status": "erro", "detalhes": str(e)}

@app.delete("/limpar_uploads")
async def limpar_uploads(username: str = Depends(verificar_credenciais)):
    try:
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR)
        return {"status": "Uploads limpos"}
    except Exception as e:
        return {"status": "erro", "detalhes": str(e)}

@app.get("/")
async def root():
    return {"status": "Online", "version": "7.0 - Offline Model Forced"}

if __name__ == "__main__":
    import uvicorn
    # Host 0.0.0.0 é vital para acesso na rede
    uvicorn.run(app, host="0.0.0.0", port=8000)