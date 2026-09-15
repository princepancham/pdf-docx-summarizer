# PDF & DOCX Summarizer

AI-powered web app to upload PDF/DOCX documents and generate summaries.
Phase 1 foundation: FastAPI backend + Streamlit frontend + env configuration.

## Phase 1 scope

- FastAPI with `GET /` and `GET /api/health`
- Streamlit starter with backend connectivity check
- Env-based config (`backend/config.py`, `.env.example`)
- No upload, parsing, summarization, or database yet (Phases 2+)

## Project structure

```text
backend/        FastAPI app (main.py, config.py)
frontend/       Streamlit app (app.py, requests -> FastAPI only)
tests/          Automated tests (from Phase 9)
uploads/        Runtime upload dir (ignored by Git)
.env            Local secrets (ignored by Git, never commit)
.env.example    Safe template for required env vars
```

## Setup (Windows PowerShell)

```powershell
# 1. Create and activate venv
py -3 -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Configure environment
Copy-Item .env.example .env
# OPENROUTER_API_KEY may stay empty in Phase 1 (needed from Phase 4)
```

## Run

```powershell
# Backend (http://localhost:8000, docs at /docs)
python -m uvicorn backend.main:app --reload --port 8000

# Frontend (new terminal, venv activated)
python -m streamlit run frontend/app.py
```

## Verify

- Backend root: `GET http://127.0.0.1:8000/` -> `{"message": ..., "docs": "/docs"}`
- Health: `GET http://127.0.0.1:8000/api/health` -> `{"status": "ok", ...}`
- Streamlit shows "Connected" when backend is running, else a clear error.

## Security notes

- `.env`, `uploads/`, `*.db`, `venv/` are Git-ignored.
- Never commit API keys. Copy `.env.example` instead.
