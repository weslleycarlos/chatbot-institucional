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
    global vectorstore, _chroma_client
    
    if vectorstore is None:
        embeddings, _ = get_llm_components()
        
        # ✅ Cria cliente persistente para ter controle sobre ele
        import chromadb
        from chromadb.config import Settings
        
        _chroma_client = chromadb.PersistentClient(
            path=PERSIST_DIRECTORY,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True  # ✅ Permite reset
            )
        )
        
        vectorstore = Chroma(
            client=_chroma_client,
            embedding_function=embeddings,
            collection_name="govbot_docs"
        )
    
    return vectorstore

def close_vectorstore():
    """Fecha COMPLETAMENTE o vectorstore e libera arquivos"""
    global vectorstore, _chroma_client
    
    print("🔒 Iniciando fechamento do Chroma...")
    
    try:
        # 1. Remove referência do vectorstore
        if vectorstore is not None:
            # Tenta acessar e limpar o client interno
            if hasattr(vectorstore, '_client'):
                try:
                    vectorstore._client.clear_system_cache()
                except:
                    pass
            vectorstore = None
            print("   ✅ Vectorstore liberado")
        
        # 2. Fecha o client do Chroma
        if _chroma_client is not None:
            try:
                # ✅ Método correto para fechar conexões
                _chroma_client.clear_system_cache()
            except Exception as e:
                print(f"   ⚠️ Erro ao limpar cache: {e}")
            
            try:
                # Força a desconexão do SQLite
                if hasattr(_chroma_client, '_identifier_to_system'):
                    _chroma_client._identifier_to_system.clear()
            except:
                pass
            
            _chroma_client = None
            print("   ✅ Client Chroma liberado")
        
        # 3. Força coleta de lixo agressiva
        gc.collect()
        gc.collect()  # Duas vezes para garantir
        
        # 4. No Windows, precisa esperar mais
        time.sleep(3)
        
        print("✅ Vectorstore completamente fechado")
        
    except Exception as e:
        print(f"⚠️ Erro ao fechar vectorstore: {e}")
        vectorstore = None
        _chroma_client = None

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
    """Limpeza completa da base - versão Windows-safe"""
    global vectorstore, _chroma_client
    
    try:
        print("🧹 INICIANDO LIMPEZA (Windows-safe)...")
        
        # ============================================
        # ESTRATÉGIA 1: Reset via API do Chroma
        # ============================================
        print("🔄 Tentando reset via API do Chroma...")
        
        try:
            if _chroma_client is not None:
                # ✅ Método mais limpo: usa o reset do próprio Chroma
                _chroma_client.reset()
                print("✅ Reset do Chroma executado!")
                
                # Fecha tudo
                close_vectorstore()
                time.sleep(2)
                
                # Remove pasta agora que está liberada
                if os.path.exists(PERSIST_DIRECTORY):
                    shutil.rmtree(PERSIST_DIRECTORY)
                    print("✅ Pasta removida após reset")
                
                # Limpa uploads
                if os.path.exists(UPLOAD_DIR):
                    shutil.rmtree(UPLOAD_DIR)
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                
                # Recria vectorstore vazio
                vs = get_vectorstore()
                docs = vs.get()
                
                return {
                    "status": "Base COMPLETAMENTE limpa (via reset)",
                    "chunks_restantes": len(docs['ids'])
                }
                
        except Exception as reset_error:
            print(f"⚠️ Reset falhou: {reset_error}")
        
        # ============================================
        # ESTRATÉGIA 2: Deletar coleção e recriar
        # ============================================
        print("🔄 Tentando deletar coleção...")
        
        try:
            if _chroma_client is not None:
                # Deleta a coleção em vez da pasta
                try:
                    _chroma_client.delete_collection("govbot_docs")
                    print("✅ Coleção deletada!")
                except:
                    pass
                
                # Fecha e recria
                close_vectorstore()
                time.sleep(2)
                
                # Tenta remover pasta
                if os.path.exists(PERSIST_DIRECTORY):
                    try:
                        shutil.rmtree(PERSIST_DIRECTORY)
                    except:
                        pass
                
                # Limpa uploads
                if os.path.exists(UPLOAD_DIR):
                    try:
                        shutil.rmtree(UPLOAD_DIR)
                    except:
                        pass
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                
                # Recria
                vs = get_vectorstore()
                docs = vs.get()
                
                return {
                    "status": "Base limpa (coleção recriada)",
                    "chunks_restantes": len(docs['ids'])
                }
                
        except Exception as delete_error:
            print(f"⚠️ Delete coleção falhou: {delete_error}")
        
        # ============================================
        # ESTRATÉGIA 3: Forçar fechamento e deletar
        # ============================================
        print("🔄 Tentando força bruta...")
        
        # Fecha tudo
        close_vectorstore()
        
        # Espera mais tempo no Windows
        print("⏳ Aguardando liberação de arquivos (5s)...")
        time.sleep(5)
        
        # Força gc novamente
        gc.collect()
        gc.collect()
        
        # Tenta deletar
        if os.path.exists(PERSIST_DIRECTORY):
            try:
                shutil.rmtree(PERSIST_DIRECTORY)
                print("✅ Pasta removida!")
            except PermissionError:
                # ============================================
                # ESTRATÉGIA 4: Marcar para deletar no restart
                # ============================================
                print("⚠️ Arquivos bloqueados - marcando para limpeza no restart")
                
                # Cria arquivo de flag
                with open(".cleanup_needed", "w") as f:
                    f.write("cleanup")
                
                # Limpa uploads pelo menos
                if os.path.exists(UPLOAD_DIR):
                    try:
                        shutil.rmtree(UPLOAD_DIR)
                    except:
                        pass
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                
                return {
                    "status": "parcial",
                    "detalhes": "Arquivos bloqueados pelo Windows. REINICIE O SERVIDOR para completar a limpeza.",
                    "acao_necessaria": "Pare o servidor (Ctrl+C) e inicie novamente"
                }
        
        # Limpa uploads
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # Recria
        vs = get_vectorstore()
        docs = vs.get()
        
        return {
            "status": "Base COMPLETAMENTE limpa",
            "chunks_restantes": len(docs['ids'])
        }
        
    except Exception as e:
        print(f"💥 ERRO: {e}")
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