---
description: "Use when editing React/Vite frontend pages, API integration, admin upload flow, or chat UI behavior in GovBot. Covers backend contract compatibility, timeout/error UX, and source rendering conventions."
name: "GovBot Frontend API Contract Guidelines"
applyTo:
  - "frontend/src/**/*.js"
  - "frontend/src/**/*.jsx"
---
# GovBot Frontend API Contract Guidelines

## Scope
- Applies to frontend app code, especially `frontend/src/pages/ChatbotPage.jsx` and `frontend/src/pages/AdminPage.jsx`.
- Prioritize stable UX and backend compatibility over stylistic refactors.

## API Integration Rules
- Keep API base from `VITE_API_URL` and preserve sensible local defaults.
- For chat, maintain compatibility with backend shape: success `{ answer, sources }`; errors as friendly JSON payloads.
- Keep source rendering compatible with `{ name, page }` and deduplicate by `name-page`.

## Admin/Auth Rules
- Admin routes rely on Basic Auth (`ADMIN_USER`/`ADMIN_PASS` on backend).
- Preserve existing admin workflow for upload, list documents, clear base, and clear uploads.
- If auth/endpoint behavior changes, update backend and docs in the same change.

## UX and Resilience Rules
- Keep explicit request timeout and cancellation behavior in chat.
- Show actionable error messages to users, not raw stack traces.
- Avoid introducing breaking changes to route structure (`/`, `/admin`) without corresponding updates.

## Build/Run Expectations
- Frontend commands: `npm run dev`, `npm run build`, `npm run preview`.
- Validate critical flows manually after API-related changes: chat request, source display, upload, documents list, cleanup actions.

## Link, Do Not Duplicate
- App setup: `README.md`, `docs/QUICKSTART.md`.
- Endpoint contracts: `docs/ENDPOINTS_API_CONFIGURACAO.md`.
- Project context: `docs/guia_projeto.md`, `.github/copilot-instructions.md`.
