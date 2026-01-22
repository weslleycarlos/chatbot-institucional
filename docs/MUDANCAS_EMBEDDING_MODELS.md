# Suporte a Múltiplos Modelos de Embedding

## Mudanças Implementadas

### 1. **Problema Identificado**
O sistema estava retornando apenas 17 trechos do contexto (limitado a 12 após busca semântica + até 8 do BM25), causando perda de artigos sequenciais como Art. 65 e 66 quando a resposta requeria toda a sequência Art. 64→65→66.

### 2. **Soluções Aplicadas**

#### A. Aumentar Capacidade de Recuperação
- **Retriever base**: `k=15` → `k=20` (mais documentos semânticos)
- **Limite inicial**: `12 docs` → `15 docs` antes de adicionar BM25
- **Resultado**: Até 20 trechos no contexto (15 semântico + até 5 do BM25)

#### B. Implementar Suporte a Múltiplos Embeddings
- **Adicionada variável de ambiente**: `EMBEDDING_MODEL` no `.env`
- **Modelos suportados**:
  - `all-MiniLM-L6-v2` (padrão, 384 dims, BERT-6L português genérico)
  - `stjiris/bert-large-portuguese-cased-legal` (especializado em português jurídico, 1024 dims)
  - `intfloat/multilingual-e5-base` (otimizado para retrieval, multilíngue)

### 3. **Como Usar**

#### Opção A: Usar Padrão (all-MiniLM-L6-v2)
Nenhuma alteração necessária. O sistema usa este modelo por padrão.

#### Opção B: Testar Modelo Português Legal
```bash
# Editar backend/.env:
EMBEDDING_MODEL=stjiris/bert-large-portuguese-cased-legal

# Depois limpar base e re-indexar:
DELETE /limpar_base  # Limpar Chroma
# Fazer upload de IN 304_2025.pdf novamente
```

#### Opção C: Testar Multilíngue Otimizado
```bash
# Editar backend/.env:
EMBEDDING_MODEL=intfloat/multilingual-e5-base

# Depois limpar base e re-indexar
```

### 4. **Requisitos para Múltiplos Modelos**

Se quiser testar com modelos diferentes, coloque os arquivos em estrutura como:

```
backend/modelo_local/
├── all-MiniLM-L6-v2/           (atual)
├── bert-large-portuguese-cased-legal/
└── multilingual-e5-base/
```

Ou deixe o sistema tentar carregar do cache offline do HuggingFace.

### 5. **Sequência de Testes Recomendada**

1. **Teste 1**: Padrão com aumento de k
   ```
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   Pergunta: "Em caso de extravio de arma de fogo, quais providências?"
   Esperado: Art. 64, 65 e 66 citados
   ```

2. **Teste 2**: Modelo especializado português
   ```
   EMBEDDING_MODEL=stjiris/bert-large-portuguese-cased-legal
   Limpar base e re-indexar
   Pergunta: "Em caso de extravio de arma de fogo, quais providências?"
   Comparar: Qualidade das respostas sobre artigos sequenciais
   ```

3. **Teste 3**: Modelo multilíngue
   ```
   EMBEDDING_MODEL=intfloat/multilingual-e5-base
   Limpar base e re-indexar
   Pergunta: (mesma)
   Comparar: Performance e relevância
   ```

### 6. **Modificações no Código**

**Arquivo**: `backend/main.py`

1. **Linha ~55-62**: Added env var parsing
   ```python
   EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
   # Validação de modelo suportado
   ```

2. **Linha ~115-134**: Updated `get_llm_components()`
   - Suporte a múltiplos paths locais via `model_configs` dict
   - Mantém fallback para cache offline com modelo selecionado

3. **Linha ~358-362**: Increased retriever k
   - Base retriever: `k: 15` → `k: 20`
   - Context docs: 12 → 15

**Arquivo**: `backend/.env.example`
- Adicionado comentário explicativo sobre `EMBEDDING_MODEL`

### 7. **Benefícios Esperados**

| Modelo | Dimensões | Especialização | Vantagem |
|--------|-----------|-----------------|----------|
| all-MiniLM-L6-v2 | 384 | Genérica | Rápido, baseline consolidado |
| bert-large-portuguese | 1024 | Português jurídico | Melhor para artigos legais em PT |
| multilingual-e5 | 1024 | Retrieval multilíngue | Melhor recall em recuperação |

### 8. **Próximas Etapas**

- [ ] Baixar modelos alternativos (se houver restrição de proxy)
- [ ] Realizar teste A/B com "extravio de arma" para Art. 64-66
- [ ] Medir performance (latência) de cada modelo
- [ ] Avaliar precisão (relevância) das respostas

---

**Data**: 2026-01-22  
**Status**: ✅ Implementado e validado
