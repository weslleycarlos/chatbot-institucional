# GovBot - Workspace Instructions

Concise guidance for coding agents to be productive quickly in this repo.

## Overview
- Stack: FastAPI RAG backend + React/Vite frontend.
- Purpose: institutional-document chat (PDF/DOCX) with cited sources.
- Primary files: `backend/main.py`, `backend/proxy_config.py`, `frontend/src/pages/ChatbotPage.jsx`, `frontend/src/pages/AdminPage.jsx`.

## Build and Run
- Windows first setup: `configurar_ambiente.bat`.
- Windows dev start: `iniciar_dev.bat`.
- Backend manual run (from `backend/` with venv active): `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`.
- Frontend run (from `frontend/`): `npm run dev`.
- Frontend build: `npm run build`.
- Tests: no automated test suite is configured; validate via API calls and UI flows.

## Runtime and Config
- Required backend env: `ADMIN_PASS` plus provider-specific API key.
- LLM is configurable via `LLM_PROVIDER` (`gemini`, `groq`, `openai`) and optional `LLM_MODEL`.
- If `LLM_MODEL` is empty, defaults are assigned by provider in `Config.validate()`.
- Frontend API base uses `VITE_API_URL`.
- `backend/proxy_config.py` configures proxy variables at startup when needed.

## Architecture and Data Flow
- Upload flow: Admin uploads PDF/DOCX -> loader (`PyPDFLoader` or `Docx2txtLoader`) -> split (`CHUNK_SIZE=1200`, `CHUNK_OVERLAP=400`) -> Chroma persist.
- Storage paths (backend working dir): `./db_chroma` for vectors and `./uploads` for uploaded files.
- Chat flow: question validation -> query variations (+ optional HyDE) -> semantic retrieval + BM25 merge -> LLM -> markdown cleanup -> `{ answer, sources }`.
- Source metadata returned as `{ name, page }`, and frontend deduplicates by `name-page`.

## API Surface
- `GET /`: status summary (includes selected provider/model metadata).
- `GET /health`: runtime health and vectorstore status.
- `POST /chat`: main RAG endpoint.
- `GET /documentos` (Basic Auth): indexed docs + chunk counts.
- `POST /upload` (Basic Auth): index PDF/DOCX.
- `DELETE /limpar_base` (Basic Auth): reset Chroma and clear uploads.
- `DELETE /limpar_uploads` (Basic Auth): clear uploads only.

## Project Conventions
- Keep `/chat` behavior compatible with frontend: return friendly JSON error payloads instead of uncaught HTTP 500 responses.
- Keep backend and frontend in sync when changing endpoint names or response shapes.
- Admin auth is Basic Auth (`ADMIN_USER` defaults to `admin`, password from `ADMIN_PASS`); frontend keeps admin password in local component state.
- Chroma lifecycle is explicit: cleanup/reinitialize logic is part of `limpar_base` and shutdown flow.
- Favor local/offline embeddings model under `backend/modelo_local/all-MiniLM-L6-v2` in corporate/proxy environments.

## Gotchas
- `ADMIN_PASS` is mandatory and must have at least 8 characters; startup validation fails otherwise.
- Corporate proxy can block model downloads and some providers; do not assume open internet access.
- `HYDE_ENABLED=true` increases latency noticeably.
- Deleting `db_chroma` removes indexed knowledge.

## Link, Do Not Duplicate
- Setup and usage details: `README.md`, `docs/QUICKSTART.md`.
- Full API reference: `docs/ENDPOINTS_API_CONFIGURACAO.md`.
- Corporate deployment/proxy guidance: `docs/IMPLEMENTACAO_CORPORATIVA.md`, `docs/CONFIGURACAO_PROXY_GROQ.md`, `docs/TROUBLESHOOTING_PROXY_GROQ.md`.
- LLM/model decisions: `docs/REFATOR_LLM_PARAMETRIZAVEL.md`, `docs/CATALOGO_MODELOS_OPEN_SOURCE.md`, `docs/MUDANCAS_EMBEDDING_MODELS.md`.
- Contribution conventions: `.github/CONTRIBUTING.md`.

If changing core behavior, update both code and linked docs in the same PR.
