import os
import shutil
import glob
import time
import secrets
import sys
import gc
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from functools import lru_cache
from dotenv import load_dotenv

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Carrega variáveis de ambiente
load_dotenv()

vectorstore = None
_chroma_client = None
_bm25_cache = None

# --- DIAGNÓSTICO DE IMPORTAÇÃO ---
try:
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from rank_bm25 import BM25Okapi
    from werkzeug.utils import secure_filename

except ImportError as e:
    logger.critical(f"Biblioteca faltando: {e}")
    print("\n" + "="*60)
    print("ERRO CRÍTICO DE BIBLIOTECA FALTANDO")
    print(f"O Python não encontrou: {e}")
    print("SOLUÇÃO: pip install langchain-huggingface python-dotenv psutil werkzeug")
    print("="*60 + "\n")
    sys.exit(1)

# Framework API
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

try:
    from proxy_config import configurar_proxy
    logger.info("Configuração de proxy carregada")
except ImportError:
    logger.warning("proxy_config não encontrado - sem configuração de proxy")
    def configurar_proxy(): pass

# --- CONFIGURAÇÕES ---

class Config:
    """Centraliza todas as configurações da aplicação"""
    
    # APIs
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Diretórios
    PERSIST_DIRECTORY = "./db_chroma"
    UPLOAD_DIR = "./uploads"
    MODELO_LOCAL_DIR = "./modelo_local"
    
    # Segurança
    ADMIN_USER = os.getenv("ADMIN_USER", "admin")
    ADMIN_PASS = os.getenv("ADMIN_PASS")
    
    # Modelos
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
    LLM_MODEL = os.getenv("LLM_MODEL", "")
    
    # RAG Settings
    HYDE_ENABLED = os.getenv("HYDE_ENABLED", "false").lower() == "true"
    SIMILARITY_TOP_K = 20
    CONTEXT_TOP_K = 15
    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 400
    
    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE = 10
    LLM_TIMEOUT_SECONDS = 30
    
    # Validações
    ALLOWED_MODELS = ["all-MiniLM-L6-v2", "stjiris/bert-large-portuguese-cased-legal", "intfloat/multilingual-e5-base"]
    ALLOWED_PROVIDERS = ["gemini", "groq", "openai"]
    MAX_QUESTION_LENGTH = 500
    MIN_QUESTION_LENGTH = 3
    
    @classmethod
    def validate(cls):
        """Valida configurações críticas no startup"""
        errors = []
        
        # Valida senha admin
        if not cls.ADMIN_PASS:
            errors.append("ADMIN_PASS não configurada no .env - OBRIGATÓRIA")
        elif len(cls.ADMIN_PASS) < 8:
            errors.append("ADMIN_PASS deve ter no mínimo 8 caracteres")
        
        # Valida modelo embedding
        if cls.EMBEDDING_MODEL not in cls.ALLOWED_MODELS:
            logger.warning(f"EMBEDDING_MODEL '{cls.EMBEDDING_MODEL}' não reconhecido. Usando padrão.")
            cls.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        
        # Valida LLM provider
        if cls.LLM_PROVIDER not in cls.ALLOWED_PROVIDERS:
            logger.warning(f"LLM_PROVIDER '{cls.LLM_PROVIDER}' não reconhecido. Usando 'gemini'.")
            cls.LLM_PROVIDER = "gemini"
        
        # Define modelo padrão se não especificado
        if not cls.LLM_MODEL:
            if cls.LLM_PROVIDER == "gemini":
                cls.LLM_MODEL = "gemini-2.5-flash"
            elif cls.LLM_PROVIDER == "groq":
                cls.LLM_MODEL = "mixtral-8x7b-32768"
            elif cls.LLM_PROVIDER == "openai":
                cls.LLM_MODEL = "gpt-4o-mini"
        
        # Valida API Keys
        if cls.LLM_PROVIDER == "gemini" and not cls.GOOGLE_API_KEY:
            errors.append("GOOGLE_API_KEY não configurada para provider 'gemini'")
        elif cls.LLM_PROVIDER == "groq" and not cls.GROQ_API_KEY:
            errors.append("GROQ_API_KEY não configurada para provider 'groq'")
        elif cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY não configurada para provider 'openai'")
        
        if errors:
            for error in errors:
                logger.error(f"❌ {error}")
            raise RuntimeError(f"Configuração inválida: {'; '.join(errors)}")
        
        # Cria diretórios necessários
        os.makedirs(cls.UPLOAD_DIR, exist_ok=True)
        os.makedirs(cls.MODELO_LOCAL_DIR, exist_ok=True)
        
        logger.info(f"🤖 LLM: {cls.LLM_PROVIDER.upper()} | Modelo: {cls.LLM_MODEL}")
        logger.info(f"📚 Embedding: {cls.EMBEDDING_MODEL}")

# --- FUNÇÕES AUXILIARES ---

def extract_text_from_response(response_content):
    """
    Extrai texto do response.content que pode vir em diferentes formatos:
    - String pura: "texto..."
    - Lista de strings: ["texto1", "texto2"]
    - Lista de dicts (Gemini 3 Flash): [{'type': 'text', 'text': '...', 'extras': {...}}]
    """
    if isinstance(response_content, str):
        return response_content
    
    elif isinstance(response_content, list):
        texts = []
        for item in response_content:
            if isinstance(item, dict) and 'text' in item:
                texts.append(item['text'])
            elif isinstance(item, str):
                texts.append(item)
            else:
                texts.append(str(item))
        return '\n'.join(texts) if texts else ""
    
    else:
        return str(response_content)

def clean_markdown(text: str) -> str:
    """Remove/limpa formatação Markdown do texto para melhor legibilidade"""
    import re
    
    # Remove negrito: **texto** → texto
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    
    # Remove itálico: *texto* → texto
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    
    # Remove links: [texto](url) → texto
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Converte bullet points * para •
    text = re.sub(r'^\s*\*\s+', '• ', text, flags=re.MULTILINE)
    
    return text

def generate_query_variations_advanced(question: str) -> List[str]:
    """
    Gera variações inteligentes da pergunta usando técnicas locais avançadas.
    VERSÃO MELHORADA: Menos redundância, mais diversidade semântica.
    """
    import re
    
    variations = []
    question_lower = question.lower().strip()
    
    # Extrai palavras-chave (remove stop words)
    stop_words = {
        'o', 'a', 'os', 'as', 'de', 'da', 'do', 'das', 'dos', 'para', 'em', 'com', 
        'por', 'é', 'são', 'um', 'uma', 'qual', 'quais', 'como', 'quando', 'onde'
    }
    
    words = question_lower.split()
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Extrai artigos mencionados (muito comum em consultas jurídicas)
    articles = re.findall(r'art(?:igo)?\.?\s*(\d+)', question_lower)
    
    # --- VARIAÇÃO 1: Foco em artigos específicos ---
    if articles:
        art_query = f"artigo {' '.join(articles[:3])}"
        if len(keywords) >= 2:
            # Combina artigo + principais keywords
            art_query += f" {' '.join(keywords[:2])}"
        variations.append(art_query)
    
    # --- VARIAÇÃO 2: Reformulação semântica (sinônimos contextuais) ---
    synonyms = {
        'deve': 'precisa',
        'pode': 'é permitido',
        'requisitos': 'condições necessárias',
        'procedimento': 'trâmite',
        'providências': 'medidas cabíveis',
        'prazo': 'período',
        'servidor': 'servidor público',
        'tac': 'termo de ajustamento de conduta',
        'sindicância': 'processo administrativo',
        'penalidade': 'sanção',
        'celebração': 'formalização'
    }
    
    var_semantic = question_lower
    replaced = False
    for original, sinonimo in synonyms.items():
        if original in var_semantic:
            var_semantic = var_semantic.replace(original, sinonimo)
            replaced = True
    
    if replaced and var_semantic != question_lower:
        variations.append(var_semantic)
    
    # --- VARIAÇÃO 3: Keywords essenciais (remover verbos e perguntas) ---
    if len(keywords) >= 3:
        # Remove palavras interrogativas e mantém substantivos
        essential = [k for k in keywords if k not in {'deve', 'pode', 'como', 'porque'}]
        if len(essential) >= 2:
            var_essential = ' '.join(essential[:4])  # Top 4 keywords
            if var_essential not in [question_lower, var_semantic]:
                variations.append(var_essential)
    
    # --- VARIAÇÃO 4: Inversão contextual (muda perspectiva) ---
    perspective_transforms = {
        'quais são': 'existe',
        'como funciona': 'funcionamento',
        'o que é': 'definição',
        'quando deve': 'prazo para',
        'pode ser': 'é possível'
    }
    
    var_perspective = question_lower
    for orig, transform in perspective_transforms.items():
        if orig in var_perspective:
            var_perspective = var_perspective.replace(orig, transform)
            break
    
    if var_perspective != question_lower and var_perspective not in variations:
        variations.append(var_perspective)
    
    # Remove duplicatas e limita a 3 variações (evita redundância)
    seen = set()
    unique_variations = []
    for var in variations:
        normalized = ' '.join(sorted(var.split()))  # Normaliza para detectar duplicatas semânticas
        if normalized not in seen:
            seen.add(normalized)
            unique_variations.append(var)
    
    # Retorna no máximo 3 variações
    return unique_variations[:3]

@lru_cache(maxsize=128)
def get_bm25_index_cached(docs_hash: int):
    """Cache do índice BM25 para evitar reconstrução a cada query"""
    return _bm25_cache

def invalidate_bm25_cache():
    """Invalida cache quando documentos são adicionados/removidos"""
    global _bm25_cache
    _bm25_cache = None
    get_bm25_index_cached.cache_clear()

# --- COMPONENTES DE IA ---

def get_llm_components():
    """
    Retorna embeddings e LLM (parametrizável).
    Versão otimizada com melhor tratamento de erros.
    """
    import huggingface_hub
    from transformers import AutoTokenizer, AutoModel
    import torch
    
    model_configs = {
        "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
        "stjiris/bert-large-portuguese-cased-legal": "bert-large-portuguese-cased-legal",
        "intfloat/multilingual-e5-base": "multilingual-e5-base"
    }
    
    local_model_name = model_configs.get(Config.EMBEDDING_MODEL, "all-MiniLM-L6-v2")
    model_path = os.path.abspath(os.path.join(Config.MODELO_LOCAL_DIR, local_model_name))
    
    logger.info(f"🔧 Carregando modelo embedding: {Config.EMBEDDING_MODEL}")
    
    # 1. Tenta carregar modelo local
    if os.path.exists(model_path) and os.listdir(model_path):
        logger.info(f"✅ Modelo local encontrado: {model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path)
            
            class LocalEmbeddings:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer
                    self.device = "cpu"
                
                def embed_documents(self, texts):
                    results = []
                    for text in texts:
                        encoded = self.tokenizer(text, padding=True, truncation=True, 
                                                return_tensors="pt", max_length=512)
                        with torch.no_grad():
                            model_output = self.model(**encoded)
                            embeddings = model_output[0]
                            mask = encoded['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
                            masked_embeddings = embeddings * mask
                            summed = torch.sum(masked_embeddings, 1)
                            summed_mask = torch.clamp(mask.sum(1), min=1e-9)
                            mean_embeddings = summed / summed_mask
                            results.append(mean_embeddings[0].cpu().numpy().tolist())
                    return results
                
                def embed_query(self, text):
                    return self.embed_documents([text])[0]
            
            embeddings = LocalEmbeddings(model, tokenizer)
            logger.info(f"✅ Modelo LOCAL '{Config.EMBEDDING_MODEL}' carregado com sucesso!")
            llm = _get_llm_instance()
            return embeddings, llm
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo local: {e}")
    
    # 2. Fallback: Cache offline HuggingFace
    logger.info("📦 Tentando cache offline do HuggingFace...")
    
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            cache_folder=os.path.expanduser("~/.cache/huggingface/hub")
        )
        logger.info(f"✅ Modelo '{Config.EMBEDDING_MODEL}' carregado do cache offline!")
        llm = _get_llm_instance()
        return embeddings, llm
        
    except Exception as e:
        logger.critical(f"Falha ao carregar embeddings: {e}")
        raise RuntimeError(
            f"Não foi possível carregar o modelo de embeddings '{Config.EMBEDDING_MODEL}'. "
            "Verifique se os arquivos estão em backend/modelo_local/ ou no cache do HuggingFace."
        )

def _get_llm_instance():
    """Instancia o LLM conforme configuração"""
    if Config.LLM_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(model=Config.LLM_MODEL, temperature=0.2)
    
    elif Config.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(api_key=Config.GROQ_API_KEY, model=Config.LLM_MODEL, temperature=0.2)
    
    elif Config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(api_key=Config.OPENAI_API_KEY, model=Config.LLM_MODEL, temperature=0.2)
    
    else:
        raise RuntimeError(f"LLM_PROVIDER '{Config.LLM_PROVIDER}' não suportado")

def get_vectorstore():
    global vectorstore, _chroma_client
    
    if vectorstore is None:
        embeddings, _ = get_llm_components()
        
        import chromadb
        from chromadb.config import Settings
        
        _chroma_client = chromadb.PersistentClient(
            path=Config.PERSIST_DIRECTORY,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        
        vectorstore = Chroma(
            client=_chroma_client,
            embedding_function=embeddings,
            collection_name="govbot_docs"
        )
    return vectorstore

def bm25_retriever(question: str, vs):
    """
    Busca por BM25 com cache otimizado.
    Recalcula índice apenas quando documentos mudam.
    """
    global _bm25_cache
    
    try:
        all_docs_dict = vs.get()
        
        if not all_docs_dict['ids']:
            return []
        
        docs_hash = hash(tuple(all_docs_dict['ids']))
        
        # Reconstrói documentos
        from langchain_core.documents import Document
        all_docs = []
        for i, doc_id in enumerate(all_docs_dict['ids']):
            doc = Document(
                page_content=all_docs_dict['documents'][i],
                metadata=all_docs_dict['metadatas'][i] if all_docs_dict['metadatas'] else {}
            )
            all_docs.append(doc)
        
        # Usa cache se disponível
        if _bm25_cache is None or get_bm25_index_cached.cache_info().currsize == 0:
            logger.info("📊 Construindo índice BM25...")
            tokenized_docs = [doc.page_content.lower().split() for doc in all_docs]
            _bm25_cache = BM25Okapi(tokenized_docs)
            get_bm25_index_cached(docs_hash)  # Armazena no cache
        
        bm25 = _bm25_cache
        
        # Busca
        query_tokens = question.lower().split()
        scores = bm25.get_scores(query_tokens)
        
        # Top 10 por BM25
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
        results = [all_docs[i] for i in top_indices if scores[i] > 0]
        
        logger.info(f"🔍 BM25 encontrou {len(results)} documentos relevantes")
        return results
        
    except Exception as e:
        logger.error(f"Erro em BM25: {e}")
        return []

def multi_query_retriever(question: str, retriever, llm):
    """
    VERSÃO MELHORADA do MultiQueryRetriever + HyDE.
    - Variações mais inteligentes (menos redundância)
    - Cache de queries similares
    - Melhor deduplicação
    """
    logger.info(f"🔄 Gerando variações da pergunta...")
    
    # 1. Gera variações (versão melhorada)
    variations = generate_query_variations_advanced(question)
    
    if variations:
        for i, var in enumerate(variations, 1):
            logger.info(f"  Variação {i}: {var}")
    
    # 2. HyDE opcional
    hyde_docs = []
    if Config.HYDE_ENABLED:
        hyde_prompt = f"""Baseado na pergunta: "{question}"

Escreva um breve trecho (2-3 linhas) de um documento oficial que responderia essa pergunta.
Inclua números, artigos e detalhes específicos se aplicável.
Responda APENAS com o trecho do documento, sem introduções."""
        
        try:
            response = llm.invoke(hyde_prompt)
            hyde_text = extract_text_from_response(response.content)
            if hyde_text:
                logger.info(f"  HyDE gerado: {hyde_text[:80]}...")
                hyde_docs = [hyde_text]
        except Exception as e:
            logger.warning(f"Erro ao gerar HyDE: {e}")
    
    # 3. Busca com deduplicação inteligente
    all_queries = [question] + variations + hyde_docs
    all_docs = []
    seen_hashes = set()
    
    for q in all_queries:
        try:
            docs = retriever.invoke(q)
            for doc in docs:
                # Hash baseado em conteúdo + metadata
                content_preview = doc.page_content[:200]
                source = doc.metadata.get('source', '')
                page = doc.metadata.get('page', '')
                doc_hash = hash(f"{content_preview}{source}{page}")
                
                if doc_hash not in seen_hashes:
                    seen_hashes.add(doc_hash)
                    all_docs.append(doc)
        except Exception as e:
            logger.warning(f"Erro ao buscar variação '{q[:50]}...': {e}")
    
    logger.info(f"✅ Total de documentos únicos: {len(all_docs)}")
    return all_docs

def close_vectorstore():
    global vectorstore, _chroma_client
    logger.info("🔒 Fechando Chroma...")
    try:
        if vectorstore:
            vectorstore = None
        if _chroma_client:
            try: 
                _chroma_client.clear_system_cache()
            except: 
                pass
            _chroma_client = None
        
        invalidate_bm25_cache()
        gc.collect()
        time.sleep(1)
    except Exception as e:
        logger.error(f"Erro ao fechar vectorstore: {e}")

# --- RATE LIMITING ---

from collections import defaultdict
from datetime import datetime, timedelta

class RateLimiter:
    """Rate limiter simples baseado em IP"""
    
    def __init__(self, max_requests: int, time_window: timedelta):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = defaultdict(list)
    
    def is_allowed(self, identifier: str) -> bool:
        now = datetime.now()
        
        # Remove requisições antigas
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if now - req_time < self.time_window
        ]
        
        # Verifica limite
        if len(self.requests[identifier]) >= self.max_requests:
            return False
        
        # Adiciona requisição atual
        self.requests[identifier].append(now)
        return True

rate_limiter = RateLimiter(
    max_requests=Config.MAX_REQUESTS_PER_MINUTE,
    time_window=timedelta(minutes=1)
)

# --- APP FASTAPI ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Inicializando GovBot API...")
    
    # Valida configurações ANTES de iniciar
    try:
        Config.validate()
    except RuntimeError as e:
        logger.critical(f"Falha na validação: {e}")
        raise
    
    # Verifica modelo offline
    if not os.listdir(Config.MODELO_LOCAL_DIR):
        logger.warning("⚠️ Pasta de modelo local vazia - dependendo de cache HuggingFace")
    
    configurar_proxy()
    
    yield
    
    logger.info("🛑 Desligando GovBot API...")
    close_vectorstore()

app = FastAPI(
    title="GovBot Intranet API",
    version="8.0",
    description="API RAG para consultas em documentos jurídicos institucionais",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBasic()

# --- MODELOS PYDANTIC ---

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=Config.MIN_QUESTION_LENGTH, max_length=Config.MAX_QUESTION_LENGTH)
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Pergunta não pode ser vazia")
        return v.strip()

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    processing_time: Optional[float] = None

class DocumentInfo(BaseModel):
    documentos: List[str]
    total_chunks: int

class UploadResponse(BaseModel):
    status: str
    chunks: int
    documento: str

class StatusResponse(BaseModel):
    status: str
    version: str
    llm_provider: str
    embedding_model: str

# --- AUTENTICAÇÃO ---

def verificar_credenciais(credentials: HTTPBasicCredentials = Depends(security)):
    is_user_ok = secrets.compare_digest(credentials.username, Config.ADMIN_USER)
    is_pass_ok = secrets.compare_digest(credentials.password, Config.ADMIN_PASS)
    
    if not (is_user_ok and is_pass_ok):
        logger.warning(f"Tentativa de login falhou: {credentials.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciais inválidas",
            headers={"WWW-Authenticate": "Basic"}
        )
    
    return credentials.username

# --- MIDDLEWARE DE RATE LIMITING ---

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Pula rate limiting para endpoints de health/status
    if request.url.path in ["/", "/health"]:
        return await call_next(request)
    
    client_ip = request.client.host
    
    if not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit excedido: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Limite de {Config.MAX_REQUESTS_PER_MINUTE} requisições por minuto excedido"
        )
    
    return await call_next(request)

# --- ENDPOINTS ---

@app.get("/", response_model=StatusResponse)
async def root():
    """Endpoint de status básico"""
    return {
        "status": "Online",
        "version": "8.0",
        "llm_provider": Config.LLM_PROVIDER,
        "embedding_model": Config.EMBEDDING_MODEL
    }

@app.get("/health")
async def health_check():
    """Health check detalhado"""
    try:
        vs = get_vectorstore()
        docs = vs.get()
        doc_count = len(docs['ids'])
        
        return {
            "status": "healthy",
            "vectorstore": "connected",
            "total_documents": doc_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check falhou: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: QueryRequest):
    """Endpoint principal de chat RAG"""
    start_time = time.time()
    
    try:
        logger.info(f"📩 Query recebida: {request.question[:80]}...")
        
        # Timeout wrapper
        try:
            result = await asyncio.wait_for(
                _process_chat_query(request.question),
                timeout=Config.LLM_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            logger.error(f"Timeout ao processar: {request.question[:50]}")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Processamento excedeu {Config.LLM_TIMEOUT_SECONDS}s"
            )
        
        processing_time = time.time() - start_time
        result["processing_time"] = round(processing_time, 2)
        
        logger.info(f"✅ Resposta gerada em {processing_time:.2f}s")
        return result
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Erro de validação: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Erro no chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno ao processar consulta"
        )

async def _process_chat_query(question: str) -> Dict[str, Any]:
    """Lógica principal do RAG (async wrapper)"""
    
    def sync_process():
        embeddings, llm = get_llm_components()
        vs = get_vectorstore()
        
        # Retriever base
        base_retriever = vs.as_retriever(
            search_type="similarity",
            search_kwargs={"k": Config.SIMILARITY_TOP_K}
        )
        
        # Busca semântica com variações
        logger.info("🔍 Iniciando busca semântica...")
        combined_docs = multi_query_retriever(question, base_retriever, llm)
        docs_for_context = combined_docs[:Config.CONTEXT_TOP_K]
        
        # Busca BM25 paralela
        logger.info("🔍 Busca BM25 por keywords...")
        bm25_docs = bm25_retriever(question, vs)
        
        # Mescla resultados
        seen = {hash(d.page_content[:100]) for d in docs_for_context}
        for doc in bm25_docs:
            doc_hash = hash(doc.page_content[:100])
            if doc_hash not in seen and len(docs_for_context) < Config.SIMILARITY_TOP_K:
                docs_for_context.append(doc)
                seen.add(doc_hash)
                logger.debug(f"BM25 adicionado: p.{doc.metadata.get('page')}")
        
        # Prompt otimizado
        system_prompt = (
            "Você é um assistente especializado em documentos oficiais jurídicos. "
            "Responda com base EXCLUSIVAMENTE nos contextos fornecidos abaixo. "
            "Se a resposta não estiver no contexto, diga que não sabe. "
            "Cite o nome do documento fonte sempre que possível.\n\n"
            "IMPORTANTE: Se a pergunta envolve 'providências', 'procedimentos', 'medidas' ou 'ações', "
            "procure por múltiplos artigos relacionados (como Art. X, Art. X+1, Art. X+2) pois "
            "frequentemente a resposta envolve uma sequência de artigos. Cite TODOS os artigos relevantes.\n\n"
            "Contextos:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])
        
        def format_docs(docs):
            return "\n\n".join(
                f"[Doc: {d.metadata.get('source', 'Desconhecido')}] {d.page_content}" 
                for d in docs
            )
        
        context_text = format_docs(docs_for_context)
        
        # Log de debug
        logger.info(f"📚 Usando {len(docs_for_context)} documentos para contexto")
        for i, doc in enumerate(docs_for_context[:5], 1):
            preview = doc.page_content.replace('\n', ' ')[:100]
            source = os.path.basename(doc.metadata.get('source', '?'))
            logger.debug(f"  [{i}] {source} (p.{doc.metadata.get('page')}): {preview}...")
        
        # Gera resposta
        rag_chain = prompt | llm
        response = rag_chain.invoke({
            "context": context_text,
            "question": question
        })
        
        content = extract_text_from_response(response.content)
        answer_lower = content.lower()
        
        # Filtra sources mencionadas
        sources = []
        sources_dedup = set()
        
        for doc in docs_for_context:
            source_name = os.path.basename(doc.metadata.get('source', 'Desconhecido')).lower()
            page_num = str(doc.metadata.get('page', 'N/A'))
            source_key = f"{source_name}-{page_num}"
            
            if (source_name in answer_lower or 
                f"p.{page_num}" in answer_lower or
                f"p. {page_num}" in answer_lower):
                
                if source_key not in sources_dedup:
                    sources.append({
                        "name": os.path.basename(doc.metadata.get('source', 'Desconhecido')),
                        "page": doc.metadata.get('page', 'N/A')
                    })
                    sources_dedup.add(source_key)
        
        # Fallback: primeiras 3 sources
        if not sources:
            for doc in docs_for_context[:3]:
                source_name = os.path.basename(doc.metadata.get('source', 'Desconhecido'))
                page_num = doc.metadata.get('page', 'N/A')
                source_key = f"{source_name}-{page_num}"
                if source_key not in sources_dedup:
                    sources.append({"name": source_name, "page": page_num})
                    sources_dedup.add(source_key)
        
        return {
            "answer": clean_markdown(content),
            "sources": sources
        }
    
    # Executa em thread para não bloquear event loop
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, sync_process)

@app.get("/documentos", response_model=DocumentInfo)
async def listar_documentos(username: str = Depends(verificar_credenciais)):
    """Lista documentos indexados"""
    try:
        vs = get_vectorstore()
        docs = vs.get()
        
        unique_docs = list(set(
            os.path.basename(m.get('source', '')) 
            for m in docs['metadatas'] 
            if m.get('source')
        ))
        
        logger.info(f"📋 Listados {len(unique_docs)} documentos únicos")
        return {"documentos": unique_docs, "total_chunks": len(docs['ids'])}
        
    except Exception as e:
        logger.error(f"Erro ao listar documentos: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao acessar base de documentos"
        )

@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    username: str = Depends(verificar_credenciais)
):
    """Upload e indexação de documentos"""
    try:
        logger.info(f"📥 Upload iniciado: {file.filename} por {username}")
        
        # Validação de arquivo
        if not file.filename:
            raise HTTPException(status_code=400, detail="Nome de arquivo inválido")
        
        allowed_extensions = {'.pdf', '.docx'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Formato não suportado. Aceitos: {', '.join(allowed_extensions)}"
            )
        
        # Nome seguro para evitar path traversal
        safe_name = secure_filename(file.filename)
        unique_name = f"{secrets.token_hex(8)}_{safe_name}"
        file_path = os.path.join(Config.UPLOAD_DIR, unique_name)
        
        # Salva arquivo
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"📄 Arquivo salvo: {unique_name}")
        
        # Carrega documento
        if file_ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_ext == ".docx":
            loader = Docx2txtLoader(file_path)
        
        docs = loader.load()
        logger.info(f"📖 {len(docs)} páginas carregadas")
        
        # Enriquece metadados
        for doc in docs:
            content = doc.page_content.lower()
            metadata_tags = []
            
            # Extrai artigos
            import re
            articles = re.findall(r'art\.\s*(\d+)', content)
            if articles:
                metadata_tags.extend([f"Art. {art}" for art in articles[:3]])
            
            # Keywords importantes
            keywords = {
                'tac': 'TAC, Termo de Ajustamento de Conduta',
                'servidor': 'Servidor Público',
                'disciplinar': 'Processo Disciplinar',
                'sindicância': 'Sindicância',
                'dois anos': 'Restrição Temporal, 2 anos',
                'celebração': 'Celebração TAC'
            }
            
            for keyword, tag in keywords.items():
                if keyword in content:
                    metadata_tags.append(tag)
            
            if metadata_tags:
                doc.metadata['tags'] = ', '.join(metadata_tags)
            
            # Adiciona nome original
            doc.metadata['original_filename'] = file.filename
        
        # Chunking inteligente
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=[
                "\n\nArt. ",
                "\n\n§ ",
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )
        chunks = text_splitter.split_documents(docs)
        
        logger.info(f"✂️ Dividido em {len(chunks)} chunks")
        
        # Indexa
        vs = get_vectorstore()
        vs.add_documents(chunks)
        
        # Invalida cache BM25
        invalidate_bm25_cache()
        
        logger.info(f"✅ Indexação concluída: {len(chunks)} trechos")
        return {
            "status": "sucesso",
            "chunks": len(chunks),
            "documento": file.filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro no upload: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar arquivo: {str(e)}"
        )

@app.delete("/limpar_base")
async def limpar_base(username: str = Depends(verificar_credenciais)):
    """Remove todos os documentos da base vetorial"""
    global vectorstore, _chroma_client
    
    try:
        logger.warning(f"🗑️ Limpeza de base iniciada por {username}")
        
        if _chroma_client:
            _chroma_client.reset()
        
        close_vectorstore()
        
        if os.path.exists(Config.PERSIST_DIRECTORY):
            shutil.rmtree(Config.PERSIST_DIRECTORY)
        
        if os.path.exists(Config.UPLOAD_DIR):
            shutil.rmtree(Config.UPLOAD_DIR)
            os.makedirs(Config.UPLOAD_DIR)
        
        # Re-inicializa
        get_vectorstore()
        
        logger.info("✅ Base limpa com sucesso")
        return {"status": "Base limpa com sucesso"}
        
    except Exception as e:
        logger.error(f"Erro ao limpar base: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao limpar base: {str(e)}"
        )

@app.delete("/limpar_uploads")
async def limpar_uploads(username: str = Depends(verificar_credenciais)):
    """Remove apenas arquivos de upload (mantém indexação)"""
    try:
        logger.info(f"🗑️ Limpeza de uploads por {username}")
        
        if os.path.exists(Config.UPLOAD_DIR):
            shutil.rmtree(Config.UPLOAD_DIR)
            os.makedirs(Config.UPLOAD_DIR)
        
        logger.info("✅ Uploads limpos")
        return {"status": "Uploads limpos"}
        
    except Exception as e:
        logger.error(f"Erro ao limpar uploads: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao limpar uploads: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Iniciando servidor...")
    uvicorn.run(app, host="0.0.0.0", port=8000)