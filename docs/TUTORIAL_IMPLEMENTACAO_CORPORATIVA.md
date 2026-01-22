# 🚀 Tutorial: Implementação Prática On-Premise

Guia passo-a-passo para colocar o GovBot em produção corporativa.

## Pré-requisitos

- Servidor Ubuntu 22.04 LTS (4+ cores, 16GB+ RAM)
- SSH acesso com sudo
- Conexão à internet (apenas para downloads iniciais)
- (Opcional) GPU NVIDIA com CUDA 12.3+

---

## Etapa 1: Setup do Servidor

### 1.1 Atualizar Sistema

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl wget git net-tools htop vim
```

### 1.2 Instalar Docker

```bash
# Instalar Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Adicionar usuário ao grupo docker
sudo usermod -aG docker $USER
newgrp docker

# Instalar Docker Compose
sudo apt install -y docker-compose

# Verificar
docker --version
docker-compose --version
```

### 1.3 Instalar NVIDIA Drivers (se houver GPU)

```bash
# Verificar GPU
lspci | grep -i nvidia

# Instalar drivers NVIDIA
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verificar
docker run --rm --gpus all nvidia/cuda:12.3.0-runtime-ubuntu22.04 nvidia-smi
```

### 1.4 Criar Diretório de Dados

```bash
sudo mkdir -p /data/{postgres,ollama,uploads,models,backups}
sudo chmod 755 /data
sudo chown $USER:$USER /data

# Criar arquivo de variáveis de ambiente
cat > /data/.env << 'EOF'
# Database
DB_USER=govbot
DB_PASSWORD=$(openssl rand -base64 32)
DB_NAME=govbot_db

# API
ADMIN_USER=admin
ADMIN_PASS=$(openssl rand -base64 32)
EMBEDDING_MODEL=BAAI/bge-base-pt-v1.5
OLLAMA_MODEL=mistral

# Security
SECRET_KEY=$(openssl rand -base64 64)
JWT_SECRET=$(openssl rand -base64 64)
EOF

cat /data/.env
```

---

## Etapa 2: Preparar Aplicação

### 2.1 Clonar e Configurar Repositório

```bash
cd /opt
sudo git clone https://github.com/seu-usuario/chatbot-institucional.git
cd chatbot-institucional

# Copiar env de exemplo
cp .env.example .env
nano .env  # Editar com valores reais
```

### 2.2 Atualizar Backend para Ollama

Editar `backend/main.py`:

```python
# Substituir Gemini por Ollama
import os
from langchain.llms import Ollama

# Configuração
OLLAMA_BASE_URL = os.getenv('OLLAMA_API', 'http://ollama:11434')
LLM_MODEL = os.getenv('OLLAMA_MODEL', 'mistral')

# Inicializar LLM
llm = Ollama(
    base_url=OLLAMA_BASE_URL,
    model=LLM_MODEL,
    temperature=0.3,
    top_p=0.9,
    num_ctx=2048
)
```

### 2.3 Atualizar Embeddings

Editar `backend/main.py`:

```python
from sentence_transformers import SentenceTransformer
import os

# Configurar modelo
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-base-pt-v1.5')
MODEL_PATH = '/app/modelos_local'

def get_embeddings_model():
    """Carrega embedding model, preferindo local"""
    try:
        return SentenceTransformer(f'{MODEL_PATH}/bge-base-pt-v1.5')
    except:
        print(f"Baixando {EMBEDDING_MODEL}...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        model.save(f'{MODEL_PATH}/bge-base-pt-v1.5')
        return model

embeddings = get_embeddings_model()
```

---

## Etapa 3: Docker Compose

### 3.1 Criar docker-compose.yml

```bash
cat > docker-compose.prod.yml << 'EOF'
version: '3.8'

services:
  # PostgreSQL com pgvector
  postgres:
    image: pgvector/pgvector:pg15-latest
    container_name: govbot-postgres
    environment:
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: ${DB_NAME}
    ports:
      - "5432:5432"
    volumes:
      - /data/postgres:/var/lib/postgresql/data
      - ./init_db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: always
    networks:
      - govbot-net

  # Ollama LLM
  ollama:
    image: ollama/ollama:latest
    container_name: govbot-ollama
    ports:
      - "11434:11434"
    volumes:
      - /data/ollama:/root/.ollama
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
      test: ["CMD", "curl", "-f", "http://localhost:11434"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: always
    networks:
      - govbot-net

  # Backend FastAPI
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: govbot-backend
    environment:
      DATABASE_URL: postgresql://${DB_USER}:${DB_PASSWORD}@postgres:5432/${DB_NAME}
      OLLAMA_API: http://ollama:11434
      OLLAMA_MODEL: ${OLLAMA_MODEL}
      EMBEDDING_MODEL: ${EMBEDDING_MODEL}
      ADMIN_PASS: ${ADMIN_PASS}
      SECRET_KEY: ${SECRET_KEY}
    ports:
      - "8000:8000"
    volumes:
      - /data/uploads:/app/uploads
      - /data/models:/app/modelos_local
      - ./backend:/app
    depends_on:
      postgres:
        condition: service_healthy
      ollama:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: always
    networks:
      - govbot-net

  # Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.prod
    container_name: govbot-frontend
    environment:
      VITE_API_URL: http://localhost:8000
    ports:
      - "5173:5173"
    depends_on:
      - backend
    restart: always
    networks:
      - govbot-net

  # Redis para cache
  redis:
    image: redis:7-alpine
    container_name: govbot-redis
    ports:
      - "6379:6379"
    volumes:
      - /data/redis:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: always
    networks:
      - govbot-net

networks:
  govbot-net:
    driver: bridge
EOF
```

### 3.2 Criar init_db.sql

```bash
cat > init_db.sql << 'EOF'
-- Criar extensão pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Tabela de documentos
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500),
    upload_date TIMESTAMP DEFAULT NOW(),
    file_size INT,
    source VARCHAR(50),
    uploaded_by VARCHAR(100),
    department VARCHAR(100),
    status VARCHAR(20) DEFAULT 'ativo'
);

-- Tabela de chunks com embeddings
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    document_id INT REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_order INT,
    embedding vector(768),
    page_number INT,
    section VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Índice para busca semântica
CREATE INDEX IF NOT EXISTS idx_chunks_embedding 
ON chunks USING ivfflat (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id 
ON chunks(document_id);

-- Tabela de histórico de chat
CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100),
    question TEXT NOT NULL,
    answer TEXT,
    sources JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    response_time_ms INT
);

-- Tabela de audit
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100),
    action VARCHAR(50),
    resource VARCHAR(255),
    details JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Índices
CREATE INDEX IF NOT EXISTS idx_chat_history_created_at 
ON chat_history(created_at);

CREATE INDEX IF NOT EXISTS idx_audit_user_id 
ON audit_log(user_id);
EOF
```

---

## Etapa 4: Deploy

### 4.1 Iniciar Serviços

```bash
# Carregar variáveis de ambiente
export $(cat /data/.env | xargs)

# Iniciar stack
cd /opt/chatbot-institucional
docker-compose -f docker-compose.prod.yml up -d

# Verificar status
docker-compose -f docker-compose.prod.yml ps

# Logs
docker-compose -f docker-compose.prod.yml logs -f backend
docker-compose -f docker-compose.prod.yml logs -f ollama
```

### 4.2 Baixar Modelo Ollama

```bash
# Entrar no container Ollama
docker exec -it govbot-ollama ollama pull mistral

# Ou outro modelo
docker exec -it govbot-ollama ollama pull openhermes
docker exec -it govbot-ollama ollama pull neural-chat

# Listar modelos
docker exec -it govbot-ollama ollama list
```

### 4.3 Verificar Integrações

```bash
# Testar backend
curl http://localhost:8000/

# Testar Ollama
curl http://localhost:11434/api/tags

# Testar PostgreSQL
docker exec -it govbot-postgres psql -U govbot -d govbot_db -c "SELECT version();"

# Testar chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Olá, quem é você?"}'
```

---

## Etapa 5: Configuração de Produção

### 5.1 Nginx com HTTPS

```bash
# Criar config nginx
sudo mkdir -p /etc/nginx/sites-available

cat > /etc/nginx/sites-available/govbot << 'EOF'
server {
    listen 443 ssl http2;
    server_name govbot.empresa.com.br;

    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Frontend
    location / {
        proxy_pass http://localhost:5173;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # API Backend
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Authorization $http_authorization;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}

server {
    listen 80;
    server_name govbot.empresa.com.br;
    return 301 https://$server_name$request_uri;
}
EOF

# Ativar
sudo ln -s /etc/nginx/sites-available/govbot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 5.2 Backup Automático

```bash
# Criar script backup
cat > /usr/local/bin/backup-govbot.sh << 'EOF'
#!/bin/bash

BACKUP_DIR="/data/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup PostgreSQL
docker exec govbot-postgres pg_dump -U govbot govbot_db | \
  gzip > "$BACKUP_DIR/govbot_db_$DATE.sql.gz"

# Backup uploads
tar -czf "$BACKUP_DIR/uploads_$DATE.tar.gz" /data/uploads

# Manter apenas últimos 7 dias
find $BACKUP_DIR -type f -mtime +7 -delete

echo "Backup concluído: $DATE"
EOF

chmod +x /usr/local/bin/backup-govbot.sh

# Cron job diário
(crontab -l 2>/dev/null; echo "0 2 * * * /usr/local/bin/backup-govbot.sh") | crontab -
```

### 5.3 Monitoramento

```bash
# Instalar Prometheus + Grafana (opcional)
docker run -d \
  -p 9090:9090 \
  -v /data/prometheus:/etc/prometheus \
  prom/prometheus

docker run -d \
  -p 3000:3000 \
  grafana/grafana

# Health check script
cat > /usr/local/bin/health-check-govbot.sh << 'EOF'
#!/bin/bash

echo "=== GovBot Health Check ==="

# Backend
if curl -f http://localhost:8000/ > /dev/null 2>&1; then
  echo "✅ Backend: OK"
else
  echo "❌ Backend: DOWN"
fi

# Ollama
if curl -f http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "✅ Ollama: OK"
else
  echo "❌ Ollama: DOWN"
fi

# PostgreSQL
if docker exec govbot-postgres pg_isready -U govbot > /dev/null 2>&1; then
  echo "✅ PostgreSQL: OK"
else
  echo "❌ PostgreSQL: DOWN"
fi

# Redis
if docker exec govbot-redis redis-cli ping > /dev/null 2>&1; then
  echo "✅ Redis: OK"
else
  echo "❌ Redis: DOWN"
fi

# Disco
DISK_USAGE=$(df /data | tail -1 | awk '{print $5}' | cut -d'%' -f1)
echo "📊 Disco: $DISK_USAGE% utilizado"

# RAM
RAM_USAGE=$(free | grep Mem | awk '{printf("%.0f", $3/$2 * 100)}')
echo "💾 RAM: $RAM_USAGE% utilizado"
EOF

chmod +x /usr/local/bin/health-check-govbot.sh

# Executar check
/usr/local/bin/health-check-govbot.sh
```

---

## Etapa 6: Testes

### 6.1 Teste de Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Qual é o propósito deste sistema?"
  }'

# Resposta esperada:
# {
#   "answer": "Este é um sistema de chat baseado em IA...",
#   "sources": []
# }
```

### 6.2 Teste de Upload

```bash
# Criar documento teste
echo "Portaria nº 123/2024 - Este é um documento de teste." > documento.txt

# Upload
curl -X POST http://localhost:8000/upload \
  -H "Authorization: Basic YWRtaW46c2VudGhhMTIz" \
  -F "file=@documento.txt"
```

### 6.3 Teste de Performance

```bash
# Instalar Apache Bench
sudo apt install -y apache2-utils

# Teste de carga
ab -n 100 -c 10 http://localhost:8000/

# Resultado esperado:
# Requests per second: 10-20 (depende do hardware)
```

---

## Troubleshooting

### Container não inicia

```bash
# Ver logs detalhados
docker-compose -f docker-compose.prod.yml logs backend

# Reconstruir imagem
docker-compose -f docker-compose.prod.yml build --no-cache

# Limpar e recomeçar
docker-compose -f docker-compose.prod.yml down -v
docker-compose -f docker-compose.prod.yml up -d
```

### Ollama muito lento

```bash
# Verificar se usa GPU
docker exec govbot-ollama nvidia-smi

# Se não aparecer, verificar driver NVIDIA
nvidia-smi

# Reiniciar docker
sudo systemctl restart docker
```

### PostgreSQL cheio

```bash
# Verificar tamanho
docker exec govbot-postgres du -sh /var/lib/postgresql/data

# Vacuumar banco
docker exec govbot-postgres psql -U govbot -d govbot_db -c "VACUUM ANALYZE;"

# Remover chats antigos
docker exec govbot-postgres psql -U govbot -d govbot_db -c \
  "DELETE FROM chat_history WHERE created_at < NOW() - INTERVAL '90 days';"
```

---

## Checklist Final

- [ ] Servidor provisionado e atualizado
- [ ] Docker + Docker Compose instalados
- [ ] NVIDIA drivers instalados (se houver GPU)
- [ ] Diretórios /data criados
- [ ] Repositório clonado e configurado
- [ ] docker-compose.prod.yml criado
- [ ] Variáveis de ambiente definidas
- [ ] Serviços iniciados e saudáveis
- [ ] Modelo Ollama baixado
- [ ] HTTPS configurado
- [ ] Backup automático configurado
- [ ] Monitoramento ativo
- [ ] Testes passando

---

## Operação Diária

```bash
# Verificar saúde
/usr/local/bin/health-check-govbot.sh

# Ver logs em tempo real
docker-compose -f docker-compose.prod.yml logs -f

# Restart de um serviço
docker-compose -f docker-compose.prod.yml restart backend

# Stop/Start
docker-compose -f docker-compose.prod.yml stop
docker-compose -f docker-compose.prod.yml start

# Update imagem
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d
```

---

**Tempo estimado para setup:** ~4 horas  
**Suporte:** Consulte a documentação em /docs
