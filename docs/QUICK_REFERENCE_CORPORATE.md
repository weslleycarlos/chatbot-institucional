# 📊 Quick Reference - Implementação Corporativa vs Cloud

## Comparativo de Arquiteturas

### Cloud Atual (APIs Pagas)

```
┌──────────────────────────────┐
│  Vantagens                   │
├──────────────────────────────┤
│ ✅ Zero setup/manutenção     │
│ ✅ Escalabilidade automática │
│ ✅ Modelos SOTA (GPT-4, etc) │
│ ✅ Suporte 24/7              │
│ ❌ Custo alto ($5-20/M req)  │
│ ❌ Dados na cloud            │
│ ❌ Limite de taxa             │
│ ❌ Latência de rede          │
└──────────────────────────────┘
```

**Custo/Ano:** $60K - $240K (com alto uso)

---

### On-Premise (Open Source)

```
┌──────────────────────────────┐
│  Vantagens                   │
├──────────────────────────────┤
│ ✅ Custo 90% menor           │
│ ✅ Dados 100% privados       │
│ ✅ Offline (sem internet)    │
│ ✅ Customizável              │
│ ✅ Sem limites de taxa       │
│ ⚠️ Requer setup/manutenção   │
│ ⚠️ Modelos menos poderosos   │
│ ⚠️ Precisa suporte técnico   │
└──────────────────────────────┘
```

**Custo/Ano:** $10K - $36K (one-time + operacional)

---

## Decision Matrix

**Use Cloud APIs se:**
- Documentos não-sensíveis
- Orçamento flexível
- Máxima qualidade importante
- Poucos usuários (< 100/mês)

**Use On-Premise se:**
- 🔐 Dados confidenciais/jurídicos
- 💰 Orçamento limitado
- 🚀 Alta utilização (> 10K req/mês)
- 🏢 Ambiente corporativo fechado
- 🔌 Possibilidade de downtime

---

## Guia de Modelos (Escolha Rápida)

### Para POC (Prototipagem)

```
LLM: llama2 (3.8B)
Embeddings: all-MiniLM-L6-v2 (80MB)
DB: SQLite (local)
Recursos: 2 CPU, 4GB RAM
Custo: $0
```

### Para Produção Leve

```
LLM: Mistral (7B) ou Neural-Chat (7B)
Embeddings: bge-base-pt-v1.5 (438MB)
DB: PostgreSQL + pgvector
Recursos: 4 CPU, 16GB RAM
Custo: ~$100/mês on-prem
```

### Para Produção Robusto

```
LLM: OpenHermes (34B) ou Dolphin-Mixtral (8.7B)
Embeddings: bge-large-pt-v1.5 (1.2GB)
DB: PostgreSQL + pgvector + Milvus
Recursos: 8-16 CPU, 64GB RAM + GPU (NVIDIA)
Custo: ~$500/mês on-prem
```

---

## Modelos Recomendados por Caso

### 1. Instituto Público / Tribunal

**Requisitos:**
- Documentos jurídicos
- Conformidade legal
- Histórico de requisições
- Offline capability

**Recomendação:**
```
LLM: openhermes-neural-chat-pt (7B português)
Embeddings: legal-bert-base (para jurídico)
BD: PostgreSQL + pgvector
Auditoria: Full compliance logging
Multi-tenant: Sim, com RBAC
Custo: $300-500/mês
```

### 2. Universidade / Biblioteca Digital

**Requisitos:**
- Muitos documentos
- Múltiplos idiomas
- Busca semântica forte
- Análise de similaridade

**Recomendação:**
```
LLM: mistral-instruct (7B) multilíngue
Embeddings: multilingual-e5-base (768D)
BD: Milvus (escalável para 1M+ docs)
Analytics: Sim, com relatórios
Custo: $200-400/mês
```

### 3. Empresa Privada / Knowledge Base

**Requisitos:**
- Propriedade intelectual
- Integração com Sharepoint
- Análise de sentimento
- Fine-tuning possível

**Recomendação:**
```
LLM: mistral (7B) + LoRA finetuned
Embeddings: bge-large-pt-v1.5 (alta qualidade)
BD: PostgreSQL + pgvector + Milvus
Analytics: Completo
Fine-tuning: Sim, com LoRA
Custo: $400-800/mês
```

### 4. Startup / MVP

**Requisitos:**
- Rapidez
- Baixo custo
- Iteração rápida
- Escalável depois

**Recomendação:**
```
LLM: Ollama + mistral (7B)
Embeddings: sentence-transformers (local)
BD: SQLite → PostgreSQL (depois)
Analytics: Básico
Custo: $50-100/mês (cloud mínimo)
```

---

## Checklist de Implementação

### Semana 1: Infraestrutura

- [ ] Provisionar servidor (bare metal ou cloud)
- [ ] Instalar Ubuntu 22.04 LTS
- [ ] Configurar SSH/VPN
- [ ] Instalar Docker + Docker Compose
- [ ] Instalar NVIDIA drivers (se houver GPU)
- [ ] Configurar storage

### Semana 2: Serviços Base

- [ ] Deploy PostgreSQL
- [ ] Instalar pgvector extension
- [ ] Deploy Ollama
- [ ] Baixar modelo LLM (mistral/openhermes)
- [ ] Deploy Redis
- [ ] Testes de conectividade

### Semana 3: Aplicação

- [ ] Atualizar backend para usar Ollama
- [ ] Atualizar embeddings (bge-pt-v1.5)
- [ ] Migrar Chroma → PostgreSQL + pgvector
- [ ] Testes de RAG pipeline
- [ ] Configurar logging/monitoring

### Semana 4: Produção

- [ ] Configurar Nginx + HTTPS
- [ ] Backup automático (PostgreSQL)
- [ ] Monitoring (Prometheus + Grafana)
- [ ] Documentação deployment
- [ ] Treinamento usuários

---

## Performance Benchmarks

### Tempo de Resposta (ms)

| Etapa | CPU/GPU |
|-------|---------|
| Query variations | 30-50ms |
| Semantic search (k=20) | 100-300ms |
| BM25 search | 50-100ms |
| LLM inference (mistral 7B) | 2000-5000ms |
| Total por query | 2.5-5.5s |

**Com GPU (RTX 4090):**
- LLM inference: 500-1500ms
- Total: 1-2s por query

### Throughput

| Configuração | Queries/seg | Latência p95 |
|--------------|------------|-------------|
| CPU 4c, 16GB | ~0.2-0.5 q/s | 3-5s |
| CPU 8c, 32GB | ~0.5-1 q/s | 2-3s |
| + GPU RTX3060 | ~1-2 q/s | 1-2s |
| + GPU RTX4090 | ~3-5 q/s | 0.5-1s |

---

## Troubleshooting Comum

### Ollama muito lento

```bash
# Verificar se está usando GPU
ollama list
nvidia-smi

# Se não, ativar GPU
export OLLAMA_NUM_GPU=32  # Max GPUs disponíveis
ollama serve
```

### PostgreSQL + pgvector lento

```sql
-- Criar índice adequado
CREATE INDEX idx_chunks_embedding 
ON chunks USING ivfflat (embedding vector_cosine_ops);

-- Analisar query
EXPLAIN ANALYZE
SELECT * FROM chunks
ORDER BY embedding <-> '[0.1, 0.2, ...]'::vector
LIMIT 10;

-- Se ainda lento, aumentar probes
SET ivfflat.probes = 40;
```

### OOM (Out of Memory)

```bash
# Verificar uso
free -h
top -p $(pidof -x python)

# Solução:
# 1. Usar modelo menor (llama2 vs openhermes)
# 2. Quantização (GGUF Q4)
# 3. Aumentar RAM
# 4. Usar swap (temporário)

# Criar swap de 32GB
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## Recursos Úteis

### Documentação

- **Ollama**: https://github.com/ollama/ollama
- **LangChain**: https://python.langchain.com/
- **pgvector**: https://github.com/pgvector/pgvector
- **Sentence Transformers**: https://www.sbert.net/

### Modelos Disponíveis

```bash
# Listar modelos
ollama list

# Pull de modelos
ollama pull mistral          # 3.8GB
ollama pull openhermes       # 13GB
ollama pull dolphin-mixtral  # 8.7GB
ollama pull neural-chat      # 4.7GB
```

### Comunidades

- HuggingFace: https://huggingface.co/models
- Ollama Discord: https://discord.gg/ollama
- LangChain Community: https://discord.gg/langchain

---

## Calculadora de Recursos

```
RAM Total Necessária:
= 2GB (SO) 
+ Tamanho do Modelo LLM
+ 2GB (Ollama overhead)
+ 2-4GB (PostgreSQL)
+ 1GB (Redis)
+ 2GB (Buffer)

Exemplo Mistral 7B:
= 2 + 15 + 2 + 4 + 1 + 2 = 26GB RAM
Recomendado: 32GB

Exemplo OpenHermes 34B:
= 2 + 35 + 2 + 4 + 1 + 2 = 46GB RAM
Recomendado: 64GB
```

---

## Roadmap de Features

### MVP (Mês 1-2)
- ✅ Chat básico
- ✅ Histórico de conversas
- ✅ Admin panel

### v1.0 (Mês 3-4)
- 📋 Multi-tenant
- 📊 Analytics dashboard
- 🔐 Audit logs
- 📝 Fine-tuning

### v2.0 (Mês 5+)
- 🤖 Agents com ferramentas
- 🔄 Retrieval feedback loop
- 📚 Knowledge base management
- 🌐 API pública
- 🔌 Integrações (SharePoint, LDAP)

---

**Versão:** 1.0  
**Data:** Janeiro 2026  
**Status:** Production Ready
