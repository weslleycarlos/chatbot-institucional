# 🏢 Guia de Implementação em Ambiente Corporativo

Documentação técnica completa para implantar o GovBot em infraestrutura corporativa on-premise com modelos open source.

## 📋 Índice

1. [Arquitetura Corporativa](#arquitetura-corporativa)
2. [Modelos LLM Open Source](#modelos-llm-open-source)
3. [Modelos de Embeddings](#modelos-de-embeddings)
4. [Banco de Dados](#banco-de-dados)
5. [Infraestrutura Recomendada](#infraestrutura-recomendada)
6. [Ampliações e Features](#ampliações-e-features)
7. [Deployment e DevOps](#deployment-e-devops)

---

## Arquitetura Corporativa

### Arquitetura Atual (Cloud)

```
┌─────────────────────┐
│   React Frontend    │
│   (5173)            │
└──────────┬──────────┘
           │ HTTP/CORS
           ▼
┌─────────────────────┐
│  FastAPI Backend    │
│  (8000)             │
└──────────┬──────────┘
           │
    ┌──────┴──────┬─────────────┐
    ▼             ▼             ▼
┌────────┐  ┌─────────────┐  ┌──────────────┐
│ Chroma │  │Sentence-    │  │Google Gemini │
│VectorDB│  │Transformers │  │API (Pago)    │
└────────┘  └─────────────┘  └──────────────┘
```

### Arquitetura Corporativa (On-Premise)

```
┌─────────────────────────────────────────────────────────────┐
│              REDE CORPORATIVA (Intranet)                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         SERVIDOR DE APLICAÇÃO                        │  │
│  │         (4-16 CPU cores, 32-128GB RAM)              │  │
│  │                                                      │  │
│  │  ┌────────────────────────────────────────────────┐ │  │
│  │  │    FastAPI Backend (8000)                      │ │  │
│  │  │    • RAG Pipeline                              │ │  │
│  │  │    • Multi-Query Retriever                     │ │  │
│  │  │    • LLM Inference                             │ │  │
│  │  └────────────────────────────────────────────────┘ │  │
│  │                                                      │  │
│  │  ┌────────────────────────────────────────────────┐ │  │
│  │  │    Frontend (React/Vite - 5173)                │ │  │
│  │  │    • Chat UI                                   │ │  │
│  │  │    • Admin Panel                               │ │  │
│  │  │    • HTTPS/Authentication                      │ │  │
│  │  └────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────┘  │
│                         │                                  │
│      ┌──────────────────┼──────────────────┐               │
│      ▼                  ▼                  ▼               │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  PostgreSQL │  │  Milvus/     │  │  Ollama/     │     │
│  │             │  │  Qdrant      │  │  LM Studio   │     │
│  │ • Histórico │  │ Vector DB    │  │ LLM Local    │     │
│  │ • Metadados │  │              │  │              │     │
│  │ • Audit Log │  │              │  │ (GPU opcional)      │
│  └─────────────┘  └──────────────┘  └──────────────┘     │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         ARMAZENAMENTO                               │  │
│  │  • /data/documents  (PDFs/DOCXs)                   │  │
│  │  • /data/models     (Embeddings + LLM)             │  │
│  │  • /data/backups    (PostgreSQL backups)           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

OPCIONAL: Sincronização
┌──────────────────────────────────────────────────────────────┐
│  SharePoint Watcher / SFTP Sync / API Integration           │
└──────────────────────────────────────────────────────────────┘
```

---

## Modelos LLM Open Source

### 1. **Ollama** (Recomendado para Corporativo)

**Características:**
- Executa localmente modelos GGUF quantizados
- Muito leve (~5GB RAM mínimo)
- Suporte a GPU (NVIDIA/AMD/Metal)
- API simples similar a OpenAI
- Perfeito para corporativo (privado, offline)

**Instalação:**
```bash
# Download em https://ollama.ai
ollama serve

# Em outro terminal, pull de modelos
ollama pull llama2           # 3.8B, fast, light
ollama pull mistral          # 7B, balanced
ollama pull neural-chat      # 7B, conversação
ollama pull openhermes       # 34B, melhor qualidade
ollama pull mistral-openorca # 7B, excelente português
```

**Modelos Recomendados:**

| Modelo | Tamanho | Velocidade | Qualidade | Português | Use Case |
|--------|---------|-----------|-----------|-----------|----------|
| llama2 | 3.8B | ⚡⚡⚡ | ⭐⭐ | Regular | Prototipagem rápida |
| mistral | 7B | ⚡⚡ | ⭐⭐⭐ | Bom | Produção leve |
| openhermes | 34B | ⚡ | ⭐⭐⭐⭐ | Excelente | Produção robusta |
| neural-chat | 7B | ⚡⚡ | ⭐⭐⭐ | Bom | Chat especializados |
| dolphin-mixtral | 8.7B | ⚡ | ⭐⭐⭐⭐ | Excelente | Análise de docs |

**Integração com FastAPI:**

```python
# backend/llm_config.py
import ollama
from langchain.llms.base import LLM

class OllamaLLM(LLM):
    model: str = "mistral"
    
    def _call(self, prompt: str, **kwargs) -> str:
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            stream=False
        )
        return response['response']

# Uso em main.py
llm = OllamaLLM(model="openhermes")
```

### 2. **LM Studio** (GUI Friendly)

**Características:**
- Interface gráfica amigável
- Gerenciamento automático de modelos
- Servidor local com API OpenAI-compatible
- Perfeito para usuários não-técnicos

**Setup:**
1. Download: https://lmstudio.ai
2. Carrega modelo na GUI
3. Inicia servidor na porta 1234
4. Conecta FastAPI via HTTP

### 3. **text-generation-webui** (Avançado)

**Características:**
- Interface web rica
- Suporte para vários formatos
- Fine-tuning integrado
- Métricas e logging detalhados

**Docker Compose:**
```yaml
services:
  text-gen-webui:
    image: ghcr.io/oobabooga/text-generation-webui:latest
    ports:
      - "7860:7860"
    volumes:
      - ./models:/app/models
    environment:
      CUDA_VISIBLE_DEVICES: "0"
```

### 4. **vLLM** (Alta Performance)

**Características:**
- Otimizado para batch inference
- Suporte para GPU multi-GPU
- PagedAttention para economia de memória
- APIs OpenAI-compatible

**Instalação:**
```bash
pip install vllm

# Servidor
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.1 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9
```

---

## Modelos de Embeddings

### Alternativas ao `all-MiniLM-L6-v2`

**Cenário Atual:**
- **all-MiniLM-L6-v2**: 384 dimensões, ~80MB, rápido, adequado para docs pequenos

**Alternativas Robustas para Corporativo:**

#### 1. **multilingual-e5-base** (Recomendado para Português)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/multilingual-e5-base')
# 768 dimensões, 438MB
# Multilíngue, melhor qualidade que MiniLM
# Suporta instruct prompts: "passage: ..." vs "query: ..."
```

**Características:**
- ✅ Excelente para português/jurídico
- ✅ 768 dimensões (melhor semântica)
- ✅ Suporta ~100+ idiomas
- ⏱️ 2x mais lento que MiniLM
- 💾 438MB

#### 2. **bge-base-pt-v1.5** (Otimizado para Português)

```python
model = SentenceTransformer('BAAI/bge-base-pt-v1.5')
# 768 dimensões, 438MB
# Treinado especificamente em português
# Excelente para jurídico/corporativo
```

**Características:**
- 🇧🇷 Otimizado para português
- ⭐ Melhor para documentos jurídicos
- 📊 768 dimensões
- ✅ Suporta sentence_transformers

#### 3. **Legal-BERT-based Embeddings**

```python
# Para domínio jurídico específico
model = SentenceTransformer('nlpaueb/legal-bert-base-uncased')
# Treinado em corpus jurídico (contratos, leis, etc)
```

#### 4. **bge-large-pt-v1.5** (Robusto)

```python
model = SentenceTransformer('BAAI/bge-large-pt-v1.5')
# 1024 dimensões, melhor qualidade
# Mais pesado: 1.2GB, 2-3x mais lento
# Para sistemas com recursos suficientes
```

#### 5. **jina-embeddings-v2-base-pt** (Recente)

```python
model = SentenceTransformer('jinaai/jina-embeddings-v2-base-pt')
# 768 dimensões
# Suporte a contexto de até 8K tokens
# Excelente para documentos longos
```

### Comparativo de Modelos

| Modelo | Dimensões | Tamanho | Velocidade | Português | Jurídico | Recomendado Para |
|--------|-----------|---------|-----------|-----------|----------|------------------|
| all-MiniLM-L6-v2 | 384 | 80MB | ⚡⚡⚡ | Regular | Regular | Prototipagem |
| multilingual-e5-base | 768 | 438MB | ⚡⚡ | Bom | Bom | Produção padrão |
| bge-base-pt-v1.5 | 768 | 438MB | ⚡⚡ | Excelente | Excelente | **Recomendado** |
| bge-large-pt-v1.5 | 1024 | 1.2GB | ⚡ | Excelente | Excelente | Alta qualidade |
| legal-bert-base | 768 | 440MB | ⚡⚡ | Bom | ⭐⭐⭐⭐ | Documentos jurídicos |

### Implementação em Produção

```python
# backend/embeddings.py
from sentence_transformers import SentenceTransformer
import os

def get_embeddings_model():
    """Carrega modelo conforme env var"""
    model_name = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-base-pt-v1.5')
    model_path = f'./modelos_local/{model_name.split("/")[1]}'
    
    # Tenta carregar local, se não existir, baixa
    try:
        model = SentenceTransformer(model_path)
    except:
        print(f"Baixando {model_name}...")
        model = SentenceTransformer(model_name)
        model.save(model_path)
    
    return model

embeddings = get_embeddings_model()
```

---

## Banco de Dados

### Dados Armazenados Atualmente

**Chroma (Vector DB):**
- Chunks de documentos
- Embeddings (384D ou mais)
- Metadados (source, page, date)

### Banco de Dados Recomendado para Corporativo

#### 1. **PostgreSQL + pgvector** (Recomendado)

**Setup:**
```sql
-- Extensão para vetores
CREATE EXTENSION IF NOT EXISTS vector;

-- Tabela de documentos
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500),
    upload_date TIMESTAMP DEFAULT NOW(),
    file_size INT,
    source VARCHAR(50),
    uploaded_by VARCHAR(100),
    department VARCHAR(100),
    status VARCHAR(20) DEFAULT 'ativo',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Tabela de chunks/embeddings
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    document_id INT REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_order INT,
    embedding vector(768),  -- Dimensão do embedder
    page_number INT,
    section VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (document_id) REFERENCES documents(id)
);

-- Índice para busca semântica rápida
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops);

-- Tabela de histórico de chat
CREATE TABLE chat_history (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100),
    question TEXT NOT NULL,
    answer TEXT,
    sources JSONB,
    embedding vector(768),  -- Query embedding para análise
    created_at TIMESTAMP DEFAULT NOW(),
    response_time_ms INT,
    model_used VARCHAR(50)
);

-- Tabela de audit
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100),
    action VARCHAR(50),  -- 'upload', 'delete', 'chat', 'admin'
    resource VARCHAR(255),
    details JSONB,
    ip_address VARCHAR(15),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Tabela de usuários (para multi-tenant)
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    email VARCHAR(100),
    department VARCHAR(100),
    role VARCHAR(20),  -- 'admin', 'user', 'analyst'
    active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Tabela de permissões
CREATE TABLE permissions (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    document_id INT REFERENCES documents(id),
    permission_type VARCHAR(20),  -- 'read', 'write', 'delete'
    created_at TIMESTAMP DEFAULT NOW()
);

-- Índices para performance
CREATE INDEX idx_documents_upload_date ON documents(upload_date);
CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chat_created_at ON chat_history(created_at);
CREATE INDEX idx_audit_user_id ON audit_log(user_id);
```

**Integração com Python:**

```python
# backend/database.py
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
from datetime import datetime
import os

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost/govbot')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500))
    upload_date = Column(DateTime, default=datetime.now)
    file_size = Column(Integer)
    source = Column(String(50))
    uploaded_by = Column(String(100))
    department = Column(String(100))
    status = Column(String(20), default='ativo')

class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    chunk_text = Column(String)
    chunk_order = Column(Integer)
    embedding = Column(Vector(768))
    page_number = Column(Integer)
    section = Column(String(100))

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(100))
    question = Column(String)
    answer = Column(String)
    sources = Column(JSON)
    embedding = Column(Vector(768))
    created_at = Column(DateTime, default=datetime.now)
    response_time_ms = Column(Integer)
    model_used = Column(String(50))

class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(100))
    action = Column(String(50))
    resource = Column(String(255))
    details = Column(JSON)
    ip_address = Column(String(15))
    created_at = Column(DateTime, default=datetime.now)

# Criar tabelas
Base.metadata.create_all(bind=engine)
```

#### 2. **Milvus** (Vector DB Especializado)

**Características:**
- Vector database nativo (não precisa pgvector)
- Escalável horizontalmente
- Suporta 1 bilhão+ vetores
- Cluster-ready

**Docker:**
```yaml
version: '3.8'
services:
  milvus:
    image: milvusdb/milvus:latest
    ports:
      - "19530:19530"
    volumes:
      - ./milvus_data:/var/lib/milvus
    environment:
      COMMON_STORAGETYPE: local
```

**Integração:**
```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema

connections.connect("default", host="localhost", port=19530)

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
]
schema = CollectionSchema(fields)
collection = Collection("documents", schema)

# Inserir dados
collection.insert([
    [1, 2, 3],  # IDs
    ["doc1", "doc2", "doc3"],  # Textos
    [embeddings]  # Vetores
])
```

#### 3. **Qdrant** (Moderno e Rápido)

**Características:**
- Vector database moderno
- Payload filtering robusto
- Replicação automática
- Armazenamento eficiente

**Docker:**
```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

### Comparativo de Vector DBs

| Aspecto | Chroma | PostgreSQL+pgvector | Milvus | Qdrant |
|--------|--------|-------------------|--------|--------|
| Tipo | Embarcado | SQL + Vector | Vector nativo | Vector nativo |
| Escalabilidade | Baixa | Média | Alta | Alta |
| Cluster | Não | Sim | Sim | Sim |
| Persistência | Local | Forte | Forte | Forte |
| Simplicidade | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Produção | ✅ | ✅✅ | ✅✅✅ | ✅✅ |

**Recomendação:** PostgreSQL + pgvector para corporativo (já tem dados estruturados)

---

## Infraestrutura Recomendada

### Servidor Mínimo

```yaml
CPU: 4 cores
RAM: 16GB (8GB Ollama + 8GB aplicação)
Armazenamento: 256GB SSD
GPU: Opcional (NVIDIA RTX 3060 6GB mínimo)
Rede: 1Gbps
SO: Ubuntu 20.04 LTS / CentOS 8
```

**Custo:** ~$50-100/mês em cloud privada

### Servidor Recomendado (Produção)

```yaml
CPU: 8-16 cores (Intel Xeon / AMD EPYC)
RAM: 64GB (16GB Ollama + 32GB DB + 16GB aplicação)
Armazenamento: 1TB+ SSD (RAID 1)
GPU: NVIDIA A100 / RTX 4090 (48GB VRAM)
        Permite rodar modelos 70B+ quantizados
Rede: 10Gbps
SO: Ubuntu 22.04 LTS / RHEL 8
```

**Custo:** ~$500-1000/mês em cloud privada

### Setup com GPU

**NVIDIA RTX 4090 (48GB):**
- Roda modelos até 70B quantizados
- Perfeito para multi-tenant corporativo
- Amortiza-se em ~3 meses vs APIs pagas

**Instalação CUDA:**
```bash
# Ubuntu 22.04
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb | sudo dpkg -i -
sudo apt update
sudo apt install cuda-12-3

# Verificar
nvidia-smi
```

**Otimização para Ollama:**
```bash
# .bashrc ou docker env
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_NUM_GPU=1
export OLLAMA_NUM_PREDICT=2048

# Executar Ollama com GPU
ollama serve --gpu all
```

---

## Ampliações e Features

### 1. Histórico de Conversas

**Estrutura:**
```python
# models/chat.py
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, JSON, Float

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), index=True)
    session_name = Column(String(255))
    started_at = Column(DateTime, default=datetime.now)
    ended_at = Column(DateTime, nullable=True)
    total_messages = Column(Integer, default=0)
    
class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    role = Column(String(20))  # 'user', 'assistant'
    content = Column(String)
    embedding = Column(Vector(768))  # Para busca semântica em histórico
    sources = Column(JSON)
    created_at = Column(DateTime, default=datetime.now)
    response_time_ms = Column(Integer, nullable=True)

# Novo endpoint
@app.get("/history/{user_id}")
async def get_chat_history(user_id: str, limit: int = 50):
    """Recupera histórico de conversas do usuário"""
    db = SessionLocal()
    sessions = db.query(ChatSession).filter(
        ChatSession.user_id == user_id
    ).order_by(ChatSession.started_at.desc()).limit(limit).all()
    
    return {
        "sessions": sessions,
        "total": len(sessions)
    }

@app.get("/history/session/{session_id}")
async def get_session_messages(session_id: int):
    """Recupera mensagens de uma sessão específica"""
    db = SessionLocal()
    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.created_at).all()
    
    return {"messages": messages}
```

### 2. Fine-tuning em Modelos

**LoRA (Low-Rank Adaptation) - Recomendado:**

```python
# backend/finetune.py
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Configurar LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Carregar modelo base
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Aplicar LoRA
model = get_peft_model(model, lora_config)

# Treinar com documentos específicos
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./lora_adapter",
    learning_rate=2e-4,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    save_strategy="epoch",
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,  # Dataset de documentos internos
)

trainer.train()

# Salvar adapter
model.save_pretrained("./lora_adapter_final")
```

**Dataset de Treinamento:**
```python
from datasets import Dataset

# Dados formato: chat/instruction
data = [
    {
        "instruction": "O que diz a portaria X?",
        "input": "",
        "output": "A portaria X estabelece que...",
        "source_doc": "Portaria_X.pdf"
    },
    # ... mais exemplos
]

dataset = Dataset.from_list(data)
```

### 3. Análise de Sentimento e Classificação

```python
# backend/analytics.py
from transformers import pipeline
from sqlalchemy.orm import Session

class ChatAnalytics:
    def __init__(self):
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
        self.zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli"
        )
    
    def analyze_query(self, question: str):
        """Analisa sentimento e categoria da pergunta"""
        
        # Sentimento
        sentiment = self.sentiment_analyzer(question)[0]
        
        # Classificação de categoria
        categories = ["Norma/Portaria", "Procedimento", "Formulario", "Dúvida Geral"]
        classification = self.zero_shot_classifier(
            question,
            categories,
            multi_class=False
        )
        
        return {
            "sentiment": sentiment,
            "category": classification["labels"][0],
            "confidence": classification["scores"][0]
        }
    
    def generate_report(self, user_id: str, days: int = 30):
        """Gera relatório de uso para o usuário"""
        db = SessionLocal()
        
        messages = db.query(ChatMessage).join(
            ChatSession
        ).filter(
            ChatSession.user_id == user_id,
            ChatMessage.created_at >= datetime.now() - timedelta(days=days)
        ).all()
        
        analytics = {
            "total_queries": len(messages),
            "avg_response_time": sum(m.response_time_ms for m in messages) / len(messages),
            "top_categories": self._get_top_categories(messages),
            "sentiment_distribution": self._analyze_sentiments(messages),
        }
        
        return analytics
```

### 4. Re-ranking com Cross-Encoder

```python
# backend/reranking.py
from sentence_transformers import CrossEncoder

class SemanticReranker:
    def __init__(self):
        # Modelo cross-encoder treinado em relevância
        self.model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    
    def rerank_results(self, query: str, documents: list, top_k: int = 5):
        """
        Re-ranking dos documentos recuperados
        Melhora significativamente a qualidade dos resultados
        """
        
        # Calcular scores de relevância
        pairs = [[query, doc['content']] for doc in documents]
        scores = self.model.predict(pairs)
        
        # Ordenar por score
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [doc for doc, score in ranked[:top_k]]
```

### 5. Moderação e Segurança

```python
# backend/moderation.py
from transformers import pipeline

class ContentModerator:
    def __init__(self):
        self.toxicity_classifier = pipeline(
            "text-classification",
            model="siebert/sentiment-roberta-large-english",
            device=0  # GPU
        )
    
    def check_prompt(self, text: str) -> bool:
        """Verifica se prompt é apropriado"""
        
        # Lista negra de palavras
        blocked_words = ['hack', 'ataque', 'vírus']
        if any(word in text.lower() for word in blocked_words):
            return False
        
        # Análise de toxicidade
        result = self.toxicity_classifier(text[:512])[0]
        if result['label'] == 'NEGATIVE' and result['score'] > 0.8:
            return False
        
        return True
    
    def check_response(self, response: str):
        """Valida resposta do LLM antes de enviar ao usuário"""
        # Similar ao check_prompt
        return True
```

### 6. Multi-Tenant e Controle de Acesso

```python
# backend/auth.py
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

class TenantManager:
    async def get_current_user(self, token: str):
        """Valida token JWT e retorna usuário"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            user_id = payload.get("sub")
        except:
            raise HTTPException(status_code=401)
        
        db = SessionLocal()
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user or not user.active:
            raise HTTPException(status_code=401)
        
        return user
    
    def has_permission(self, user: User, document_id: int):
        """Verifica se usuário tem acesso ao documento"""
        db = SessionLocal()
        
        permission = db.query(Permission).filter(
            Permission.user_id == user.id,
            Permission.document_id == document_id,
            Permission.permission_type.in_(['read', 'write'])
        ).first()
        
        return permission is not None

# Uso em endpoint
@app.post("/chat")
async def chat(
    question: str,
    token: str = Header(...),
    tenant_manager: TenantManager = Depends()
):
    user = await tenant_manager.get_current_user(token)
    
    # Recuperar apenas documentos que usuário tem acesso
    db = SessionLocal()
    accessible_docs = db.query(Document).join(
        Permission
    ).filter(
        Permission.user_id == user.id,
        Permission.permission_type == 'read'
    ).all()
    
    # RAG pipeline com documentos permitidos
    # ...
```

### 7. Monitoramento e Observabilidade

```python
# backend/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Métricas
query_counter = Counter(
    'chat_queries_total',
    'Total de queries',
    ['model', 'user_department']
)

query_duration = Histogram(
    'chat_query_duration_seconds',
    'Duração das queries',
    buckets=[1, 2, 5, 10, 30]
)

active_sessions = Gauge(
    'chat_active_sessions',
    'Sessões ativas'
)

# Middleware para logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    # Log estruturado
    logger.info({
        "endpoint": request.url.path,
        "method": request.method,
        "status": response.status_code,
        "duration_ms": duration * 1000,
        "user_agent": request.headers.get("user-agent"),
    })
    
    return response
```

---

## Deployment e DevOps

### Docker Compose para Produção

```yaml
version: '3.8'

services:
  # Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.prod
    ports:
      - "80:5173"
    environment:
      VITE_API_URL: http://localhost:8000
    depends_on:
      - backend

  # Backend FastAPI
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://govbot:password@postgres:5432/govbot_db
      EMBEDDING_MODEL: BAAI/bge-base-pt-v1.5
      OLLAMA_API: http://ollama:11434
      REDIS_URL: redis://redis:6379
    depends_on:
      - postgres
      - redis
      - ollama
    volumes:
      - ./data/uploads:/app/uploads
      - ./data/models:/app/modelos_local
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3

  # PostgreSQL com pgvector
  postgres:
    image: pgvector/pgvector:pg15-latest
    environment:
      POSTGRES_USER: govbot
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: govbot_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init_db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U govbot"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Ollama LLM
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
      - ollama_cache:/root/.cache
    environment:
      NVIDIA_VISIBLE_DEVICES: all
      OLLAMA_NUM_GPU: 1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "ollama", "list"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Redis para cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  # Milvus Vector DB (alternativa a pgvector)
  milvus:
    image: milvusdb/milvus:latest
    ports:
      - "19530:19530"
    environment:
      COMMON_STORAGETYPE: local
    volumes:
      - milvus_data:/var/lib/milvus
    command: milvus run standalone

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - backend
      - frontend

volumes:
  postgres_data:
  ollama_models:
  ollama_cache:
  redis_data:
  milvus_data:
```

### Nginx Config (HTTPS)

```nginx
server {
    listen 443 ssl http2;
    server_name govbot.empresa.com;

    ssl_certificate /etc/nginx/certs/cert.pem;
    ssl_certificate_key /etc/nginx/certs/key.pem;

    # Frontend
    location / {
        proxy_pass http://frontend:5173;
    }

    # Backend API
    location /api/ {
        proxy_pass http://backend:8000/;
        proxy_set_header Authorization $http_authorization;
        
        # WebSocket (para chat em tempo real)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts para requisições longas
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # Prometheus metrics (protegido)
    location /metrics {
        allow 10.0.0.0/8;
        deny all;
        proxy_pass http://backend:8000/metrics;
    }
}
```

### Kubernetes Deployment

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: govbot

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
  namespace: govbot
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: backend
        image: govbot-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: govbot-secrets
              key: database-url
        - name: OLLAMA_API
          value: "http://ollama-service:11434"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "16Gi"
            cpu: "8"
        livenessProbe:
          httpGet:
            path: /
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 8000
          initialDelaySeconds: 20
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: backend-service
  namespace: govbot
spec:
  selector:
    app: backend
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
  namespace: govbot
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## Estimativas de Custos

### On-Premise (Infrastructure)

| Item | Custo Inicial | Custo/Mês | 3 Anos |
|------|--------------|-----------|--------|
| Servidor (hardware) | $5,000 | $100 | $8,600 |
| Licenças OS/DB | $1,000 | $50 | $2,800 |
| Infraestrutura (eletricidade, cooling) | - | $200 | $7,200 |
| Suporte técnico | - | $500 | $18,000 |
| **Total 3 anos** | **$6,000** | **$850/mês** | **$36,600** |

### Cloud (AWS/Azure)

| Item | Custo/Mês | 3 Anos |
|------|-----------|---------|
| EC2 c5.4xlarge | $600 | $21,600 |
| RDS PostgreSQL | $200 | $7,200 |
| EBS Storage | $100 | $3,600 |
| **Total** | **$900/mês** | **$32,400** |

### APIs Pagas (Comparação)

| Provider | Custo/1M queries |
|----------|-----------------|
| OpenAI API | $3-20 |
| Google Gemini | $5 |
| Azure OpenAI | $3-15 |
| Anthropic Claude | $8-24 |
| **Ollama Local** | ~$0.10 (amortizado) |

**ROI:** Payback em ~6-12 meses com alta utilização (>10K queries/mês)

---

## Conclusão

**Recomendação para Corporativo:**

1. ✅ **LLM:** Ollama + Mistral/OpenHermes (7B-34B)
2. ✅ **Embeddings:** BAAI/bge-base-pt-v1.5 (768D)
3. ✅ **Vector DB:** PostgreSQL + pgvector
4. ✅ **Histórico:** PostgreSQL com ChatSession/ChatMessage
5. ✅ **Escalabilidade:** Docker Compose → Kubernetes
6. ✅ **Segurança:** HTTPS + JWT + RBAC + Audit Logs
7. ✅ **Monitoring:** Prometheus + Grafana
8. ✅ **Ampliações:** Fine-tuning, Re-ranking, Analytics

**Timeline:**
- Semana 1-2: Setup servidor + PostgreSQL + Ollama
- Semana 3-4: Migrar backend + testes
- Semana 5: Frontend + integração
- Semana 6+: Fine-tuning e otimizações
