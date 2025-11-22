import os
import shutil
import secrets
import sys
from contextlib import asynccontextmanager
from typing import List, Optional

# --- DIAGNÓSTICO DE IMPORTAÇÃO (Para não quebrar com erro feio) ---
try:
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    
    # O erro estava aqui:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate

except ImportError as e:
    print("\n" + "="*60)
    print("ERRO CRÍTICO DE BIBLIOTECA FALTANDO")
    print("="*60)
    print(f"O Python não encontrou uma biblioteca essencial: {e.name}")
    print("SOLUÇÃO: Rode o arquivo 'reparar_ambiente.bat' dentro da pasta backend.")
    print("="*60 + "\n")
    sys.exit(1) # Para o programa aqui para você ver a mensagem

# Framework API
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Tenta importar a configuração de proxy
try:
    from proxy_config import configurar_proxy
except ImportError:
    def configurar_proxy():
        pass

# --- 1. CONFIGURAÇÕES ---

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "SUA_CHAVE_AQUI")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

PERSIST_DIRECTORY = "./db_chroma"
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ADMIN_USER = "admin"
ADMIN_PASS = "senha_secreta_123"

# --- 2. APP ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 Inicializando GovBot API...")
    configurar_proxy()
    yield
    print("🛑 Desligando GovBot API...")

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
        raise HTTPException(status_code=401, detail="Não autorizado", headers={"WWW-Authenticate": "Basic"})
    return credentials.username

def get_llm_components():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    return embeddings, llm

# --- 4. ENDPOINTS ---

class QueryRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    try:
        embeddings, llm = get_llm_components()
        vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        
        # --- LÓGICA RAG (LCEL) ---
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        system_prompt = (
            "Você é um assistente jurídico de um órgão público. "
            "Use os contextos abaixo para responder à pergunta. "
            "Se não souber a resposta baseada no contexto, diga que não encontrou a informação. "
            "Cite o nome do documento se possível."
            "\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        response = rag_chain.invoke({"input": request.question})
        
        sources = []
        if "context" in response:
            for doc in response["context"]:
                sources.append({
                    "name": os.path.basename(doc.metadata.get('source', 'Desconhecido')),
                    "page": doc.metadata.get('page', 'N/A')
                })

        return {"answer": response["answer"], "sources": sources}

    except Exception as e:
        print(f"Erro: {e}")
        return {"answer": "Erro ao processar sua pergunta. Verifique se há documentos indexados.", "sources": [], "debug": str(e)}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), username: str = Depends(verificar_credenciais)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if file.filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.filename.lower().endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            return {"error": "Formato inválido"}
        
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        embeddings, _ = get_llm_components()
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=PERSIST_DIRECTORY)
        vectorstore.persist()

        return {"status": "sucesso", "chunks": len(chunks)}

    except Exception as e:
        return {"status": "erro", "detalhes": str(e)}

@app.delete("/limpar_base")
async def limpar_base(username: str = Depends(verificar_credenciais)):
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
    return {"status": "Base apagada"}