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

load_dotenv()
# Variável global para controlar o vectorstore
vectorstore = None

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
    print("="*60)
    print(f"O Python não encontrou: {e}")
    print("SOLUÇÃO: Execute:")
    print("pip install langchain-huggingface")
    print("="*60 + "\n")
    sys.exit(1)

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

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


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
        raise HTTPException(
            status_code=401, 
            detail="Não autorizado", 
            headers={"WWW-Authenticate": "Basic"}
        )
    return credentials.username

def get_llm_components():
    """Retorna embeddings locais e modelo Gemini"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)
    return embeddings, llm

def get_vectorstore():
    """Obtém ou cria o vectorstore com gerenciamento de conexão"""
    global vectorstore
    if vectorstore is None:
        embeddings, _ = get_llm_components()
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY, 
            embedding_function=embeddings
        )
    return vectorstore

def close_vectorstore():
    """Fecha explicitamente o vectorstore para liberar arquivos"""
    global vectorstore
    if vectorstore is None:
        return
    
    try:
        vectorstore = None
        gc.collect()
        print("✅ Vectorstore fechado")
    except Exception as e:
        print(f"⚠️ Erro ao fechar vectorstore: {e}")

# --- 4. ENDPOINTS ---

class QueryRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    """Endpoint público para chat com a base de conhecimento"""
    try:
        embeddings, llm = get_llm_components()
        vs = get_vectorstore()
        
        # DEBUG: Ver o que está na base
        docs = vs.get()
        print(f"📊 Total de chunks na base: {len(docs['ids'])}")
        
        # LÓGICA RAG MODERNA (LCEL)
        retriever = vs.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}
        )
        
        # DEBUG: Ver o que está sendo recuperado
        retrieved_docs = retriever.invoke(request.question)
        print(f"🔍 Chunks recuperados para a pergunta: {len(retrieved_docs)}")
        
        for i, doc in enumerate(retrieved_docs):
            print(f"📄 Chunk {i+1}: {doc.page_content[:200]}...")
        
        system_prompt = (
            "Você é um assistente especializado em documentos oficiais da Polícia Federal. "
            "Analise cuidadosamente os contextos fornecidos para encontrar informações ESPECÍFICAS como nomes, matrículas, cargos e datas. "
            "SE os contextos contiverem listas, tabelas ou designações de servidores, EXTRAIA TODOS os nomes e informações relevantes. "
            "NÃO generalize nem resuma demais - seja preciso e detalhado com informações de designação de pessoal. "
            "Se a informação não estiver completa nos contextos, indique o que foi possível encontrar e sugira verificar o documento original."
            "\n\n"
            "Contextos relevantes:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
        )
        
        response = rag_chain.invoke(request.question)
        
        # Coleta fontes
        sources = []
        for doc in retrieved_docs:
            sources.append({
                "name": os.path.basename(doc.metadata.get('source', 'Desconhecido')),
                "page": doc.metadata.get('page', 'N/A')
            })

        return {"answer": response.content, "sources": sources}

    except Exception as e:
        print(f"Erro: {e}")
        return {"answer": "Erro ao processar sua pergunta.", "sources": []}

@app.get("/documentos")
async def listar_documentos(username: str = Depends(verificar_credenciais)):
    """Lista documentos indexados na base (protegido)"""
    try:
        vs = get_vectorstore()
        docs = vs.get()
        documentos_unicos = set()
        
        for metadata in docs['metadatas']:
            if 'source' in metadata:
                doc_name = os.path.basename(metadata['source'])
                if doc_name and doc_name != 'Desconhecido':
                    documentos_unicos.add(doc_name)
        
        return {
            "documentos": list(documentos_unicos), 
            "total_chunks": len(docs['ids'])
        }
    
    except Exception as e:
        return {"documentos": [], "total_chunks": 0, "erro": str(e)}

@app.delete("/limpar_uploads")
async def limpar_uploads(username: str = Depends(verificar_credenciais)):
    """Limpa apenas os arquivos uploadados, mantém a base"""
    try:
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True)
        return {"status": "Uploads limpos com sucesso"}
    except Exception as e:
        return {"status": "erro", "detalhes": str(e)}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), username: str = Depends(verificar_credenciais)):
    """Upload e indexação de documentos PDF/DOCX"""
    try:
        # Salva o arquivo
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Carrega o documento
        if file.filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.filename.lower().endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            return {"error": "Formato inválido. Use PDF ou DOCX."}
        
        docs = loader.load()
        
        # Divide em chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
        chunks = text_splitter.split_documents(docs)

        # ✅ CORREÇÃO: Adiciona os chunks ao vectorstore
        vs = get_vectorstore()
        vs.add_documents(chunks)
        
        print(f"✅ Documento '{file.filename}' indexado com {len(chunks)} chunks")

        return {"status": "sucesso", "chunks": len(chunks), "documento": file.filename}

    except Exception as e:
        print(f"❌ Erro no upload: {e}")
        return {"status": "erro", "detalhes": str(e)}

@app.delete("/limpar_base")
async def limpar_base(username: str = Depends(verificar_credenciais)):
    """Limpeza completa da base com gerenciamento de conexões"""
    global vectorstore
    
    try:
        print("🧹 INICIANDO LIMPEZA COM GESTÃO DE CONEXÕES...")
        
        # 1. Fecha o vectorstore atual
        print("🔒 Fechando conexões Chroma...")
        close_vectorstore()
        time.sleep(2)
        
        # 2. Tenta remover com múltiplas estratégias
        max_attempts = 3
        success = False
        
        for attempt in range(max_attempts):
            try:
                if not os.path.exists(PERSIST_DIRECTORY):
                    print("ℹ️ Pasta já não existe")
                    success = True
                    break
                    
                print(f"🗑️ Tentativa {attempt + 1} - Removendo {PERSIST_DIRECTORY}...")
                shutil.rmtree(PERSIST_DIRECTORY)
                print("✅ Pasta removida com sucesso!")
                success = True
                break
                
            except PermissionError as e:
                print(f"⚠️ PermissionError na tentativa {attempt + 1}: {e}")
                
                if attempt < max_attempts - 1:
                    try:
                        print("🔄 Tentando remover arquivos individualmente...")
                        for root, dirs, files in os.walk(PERSIST_DIRECTORY, topdown=False):
                            for name in files:
                                file_path = os.path.join(root, name)
                                try:
                                    os.chmod(file_path, 0o777)
                                    os.remove(file_path)
                                    print(f"   ✅ Removido: {name}")
                                except Exception as file_error:
                                    print(f"   ❌ Falha em {name}: {file_error}")
                            
                            for name in dirs:
                                dir_path = os.path.join(root, name)
                                try:
                                    os.rmdir(dir_path)
                                except Exception as dir_error:
                                    print(f"   ❌ Falha na pasta {name}: {dir_error}")
                        
                        if os.path.exists(PERSIST_DIRECTORY):
                            os.rmdir(PERSIST_DIRECTORY)
                        success = True
                        break
                        
                    except Exception as inner_error:
                        print(f"❌ Estratégia individual falhou: {inner_error}")
                        time.sleep(2)
                else:
                    print("❌ Todas as tentativas falharam!")
        
        if not success:
            return {
                "status": "erro", 
                "detalhes": "Não foi possível remover os arquivos. O ChromaDB pode estar bloqueado. Reinicie o servidor."
            }
        
        # 3. Remove arquivos Chroma soltos
        chroma_files = ["chroma.sqlite3", "chroma.sqlite3-wal", "chroma.sqlite3-shm"]
        for file_pattern in chroma_files:
            for file_path in glob.glob(f"./{file_pattern}"):
                try:
                    if os.path.exists(file_path):
                        os.chmod(file_path, 0o777)
                        os.remove(file_path)
                        print(f"🗑️ Removido: {file_path}")
                except Exception as e:
                    print(f"⚠️ Não foi possível remover {file_path}: {e}")
        
        # 4. Limpa uploads
        if os.path.exists(UPLOAD_DIR):
            try:
                shutil.rmtree(UPLOAD_DIR)
                print(f"🗑️ Uploads removidos: {UPLOAD_DIR}")
            except Exception as e:
                print(f"⚠️ Erro ao remover uploads: {e}")
        
        # 5. Recria estrutura
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # 6. Recria o vectorstore vazio
        print("🆕 Recriando Chroma vazio...")
        vs = get_vectorstore()
        
        # 7. Verificação final
        docs_after = vs.get()
        chunks_remaining = len(docs_after['ids'])
        
        print(f"🎉 LIMPEZA CONCLUÍDA! Chunks restantes: {chunks_remaining}")
        
        return {
            "status": "Base COMPLETAMENTE limpa", 
            "chunks_restantes": chunks_remaining
        }
        
    except Exception as e:
        print(f"💥 ERRO CRÍTICO: {e}")
        return {"status": "erro", "detalhes": str(e)}

@app.get("/verificar_limpeza")
async def verificar_limpeza(username: str = Depends(verificar_credenciais)):
    """Verifica o estado atual da base"""
    try:
        import psutil
        
        chroma_processes = []
        if os.path.exists(PERSIST_DIRECTORY):
            for proc in psutil.process_iter(['pid', 'name', 'open_files']):
                try:
                    if proc.info['open_files']:
                        for file in proc.info['open_files']:
                            if PERSIST_DIRECTORY in file.path:
                                chroma_processes.append({
                                    'pid': proc.info['pid'],
                                    'name': proc.info['name'],
                                    'file': file.path
                                })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        
        return {
            "chroma_directory_exists": os.path.exists(PERSIST_DIRECTORY),
            "chroma_files": os.listdir(PERSIST_DIRECTORY) if os.path.exists(PERSIST_DIRECTORY) else [],
            "processos_bloqueando": chroma_processes,
            "pasta_db_chroma_tamanho": sum(
                f.stat().st_size for f in os.scandir(PERSIST_DIRECTORY) if f.is_file()
            ) if os.path.exists(PERSIST_DIRECTORY) else 0
        }
    except ImportError:
        return {"erro": "psutil não instalado. Execute: pip install psutil"}
    except Exception as e:
        return {"erro": str(e)}

@app.get("/debug_chroma")
async def debug_chroma(username: str = Depends(verificar_credenciais)):
    """Debug completo do ChromaDB"""
    try:
        vs = get_vectorstore()
        docs = vs.get()
        
        chroma_files = []
        if os.path.exists(PERSIST_DIRECTORY):
            for root, dirs, files in os.walk(PERSIST_DIRECTORY):
                for file in files:
                    chroma_files.append(os.path.join(root, file))
        
        return {
            "chroma_persist_directory": PERSIST_DIRECTORY,
            "chroma_directory_exists": os.path.exists(PERSIST_DIRECTORY),
            "total_chunks": len(docs['ids']),
            "documentos": list(set(
                os.path.basename(meta.get('source', 'Unknown')) 
                for meta in docs['metadatas'] if 'source' in meta
            )),
            "chroma_files": chroma_files,
            "upload_dir_files": os.listdir(UPLOAD_DIR) if os.path.exists(UPLOAD_DIR) else []
        }
    
    except Exception as e:
        return {"erro": str(e)}

@app.get("/")
async def root():
    return {
        "message": "GovBot API está rodando!", 
        "version": "5.1 - Corrigido",
        "status": "Gemini + Embeddings locais"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)