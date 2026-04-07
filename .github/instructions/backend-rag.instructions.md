---
description: "Use when editing FastAPI backend, RAG pipeline, retrieval logic, embeddings, auth, or API contracts in GovBot. Covers /chat compatibility, Chroma lifecycle, env validation, and corporate proxy constraints."
name: "GovBot Backend RAG Guidelines"
applyTo: "backend/**/*.py"
---
# GovBot Backend RAG Guidelines

## Scope
- Applies to `backend/main.py`, `backend/proxy_config.py`, and backend support scripts.
- Keep behavior aligned with frontend expectations for payload shape and errors.

## API Contract Rules
- Keep `/chat` response contract as `{ answer, sources }` with friendly JSON errors for UI consumption.
- Preserve source metadata shape as `{ name, page }`.
- If endpoint names or response fields change, update frontend callers in `frontend/src/pages/ChatbotPage.jsx` and `frontend/src/pages/AdminPage.jsx` in the same change.

## Config and Startup Rules
- `ADMIN_PASS` is mandatory and must remain validated on startup.
- Respect provider-based config: `LLM_PROVIDER` (`gemini`, `groq`, `openai`) and optional `LLM_MODEL` default assignment.
- Do not assume open internet access; corporate proxy may block model downloads.

## Retrieval and Data Rules
- Keep chunking defaults (`CHUNK_SIZE=1200`, `CHUNK_OVERLAP=400`) unless there is a documented reason to change.
- Preserve hybrid retrieval behavior (semantic + BM25 merge) and explicit deduplication.
- Keep Chroma lifecycle explicit: cleanup on shutdown and reinitialize logic during `limpar_base`.

## Performance and Reliability
- Treat `HYDE_ENABLED=true` as higher latency mode.
- Keep timeout/error handling user-friendly (avoid uncaught HTTP 500 for expected failures).
- Prefer incremental, low-risk changes and keep logs meaningful.

## Link, Do Not Duplicate
- Setup and run details: `README.md`, `docs/QUICKSTART.md`.
- API details: `docs/ENDPOINTS_API_CONFIGURACAO.md`.
- Deployment/proxy details: `docs/IMPLEMENTACAO_CORPORATIVA.md`, `docs/CONFIGURACAO_PROXY_GROQ.md`, `docs/TROUBLESHOOTING_PROXY_GROQ.md`.
- LLM/model decisions: `docs/REFATOR_LLM_PARAMETRIZAVEL.md`, `docs/CATALOGO_MODELOS_OPEN_SOURCE.md`, `docs/MUDANCAS_EMBEDDING_MODELS.md`.
