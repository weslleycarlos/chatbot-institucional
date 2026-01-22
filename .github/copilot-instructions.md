# GovBot – AI Agent Instructions

Concise guidance for AI coding agents to be immediately productive in this repo. Focus on the actual patterns and workflows used here.

## Overview
- Stack: FastAPI RAG backend + React/Vite frontend.
- Purpose: Chat over institutional docs (PDF/DOCX) with sources via Chroma + HuggingFace embeddings and Google Gemini LLM.
- Key paths: backend main API in `backend/main.py`; admin UI in `frontend/src/pages/AdminPage.jsx`; chat UI in `frontend/src/pages/ChatbotPage.jsx`.

## Runtime & Environment
- Backend env: set `GOOGLE_API_KEY` and `ADMIN_PASS` in `backend/.env` (see `README.md`).
- Frontend env: set `VITE_API_URL` (e.g. `http://localhost:8000`). The code falls back to a static IP if unset.
- Proxy: `backend/proxy_config.py` sets `HTTP_PROXY/HTTPS_PROXY` if missing; `configurar_proxy()` is called on startup.
- Offline embeddings model: place the files for `sentence-transformers/all-MiniLM-L6-v2` under `backend/modelo_local/`. If this folder is empty, the app attempts an online download (likely blocked by proxy).

## Data Flow
- Upload: Admin uploads PDF/DOCX → `PyPDFLoader`/`Docx2txtLoader` → `RecursiveCharacterTextSplitter` (`chunk_size=800`, `chunk_overlap=100`) → add to Chroma.
- Store: Persistent Chroma at `./db_chroma` (relative to backend). Uploads stored in `./uploads`.
- Chat: `/chat` builds RAG chain with retriever `k=6`, formats docs, applies system prompt, invokes Gemini; returns `answer` + `sources`.
- Sources: backend returns `[ { name: <file>, page: <page|N/A> }, ... ]`; frontend deduplicates by `name-page`.

## API Surface (backend/main.py)
- `POST /chat`: body `{ question: string }` → `{ answer, sources }`; returns friendly JSON errors (not HTTP 500) for UI handling.
- `GET /documentos` (Basic Auth): returns indexed docs + `total_chunks`.
- `POST /upload` (Basic Auth): accepts `file` (PDF/DOCX); indexes into Chroma.
- `DELETE /limpar_base` (Basic Auth): resets Chroma and clears uploads.
- `DELETE /limpar_uploads` (Basic Auth): clears only `./uploads`.
- `GET /`: health with version string.
- Auth: `ADMIN_USER="admin"`; password from env (`ADMIN_PASS`). Frontend sends `Authorization: Basic`.

## Frontend Patterns
- Routing: `/` → chat; `/admin` → admin panel (`react-router-dom` in `src/main.jsx`).
- Admin: loads docs/stats with Basic Auth; supports file input + drag&drop; calls `/limpar_base` and `/limpar_uploads`.
- Chat: `POST ${VITE_API_URL}/chat` with 60s timeout + AbortController; shows errors user-friendly; displays source chips.
- Vite dev server: `vite.config.js` sets `host: true`, `port: 5173` (LAN access).

## Developer Workflows (Windows)
- First-time setup: run `configurar_ambiente.bat` (creates venv, installs Python deps incl. `numpy`, then runs `npm install`).
- Dev run: `iniciar_dev.bat` (starts `uvicorn main:app --host 0.0.0.0 --port 8000 --reload` and `npm run dev`).
- Repair Python env: `backend/reparar_ambiente.bat` (uninstalls conflicting LangChain pkgs, reinstalls from `requirements.txt`).
- Manual backend: from `backend/venv`, run `uvicorn main:app --reload`.

## Conventions & Gotchas
- Embeddings forced to local folder when available; proxy may block online downloads.
- Chroma lifecycle: global client; `close_vectorstore()` cleans caches; `limpar_base` resets and reinitializes.
- CORS open: `allow_origins=["*"]` for LAN access.
- Admin endpoints require Basic Auth; UI stores password in state (no sessions).
- Error handling: `/chat` returns JSON error messages the UI renders—don’t change to HTTP 500 unless UI is updated.
- SharePoint watcher: `backend/sharepoint_watcher.py` monitors a local synced folder, retries uploads to `/upload`, and uses Basic Auth; configure `PASTA_SHAREPOINT`, `API_URL`, and credentials.

## Examples
- Call chat:
  - Request: `{ "question": "Qual é a portaria X?" }`
  - Response: `{ "answer": "...", "sources": [{ "name": "Boletim.pdf", "page": 12 }] }`
- Admin upload: send `multipart/form-data` with `file: <PDF/DOCX>`; expect `{ status: "sucesso", chunks, documento }`.
- Clearing base: `DELETE /limpar_base` with Basic Auth → `{ status: "Base limpa com sucesso" }`.

Keep changes aligned with these patterns. If altering endpoints or response shapes, update both backend and the corresponding frontend calls/components.
