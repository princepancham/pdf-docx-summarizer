# PDF & DOCX Summarizer

AI-powered web app to upload PDF/DOCX documents and generate summaries,
with persistent history stored in SQLite.

## Current functionality (Phases 1–3)

- FastAPI with `GET /`, `GET /api/health`
- Secure upload: `POST /api/documents/upload` (PDF/DOCX, UUID filenames)
- AI summarization: `POST /api/documents/summarize` (extract → clean →
  chunk → OpenRouter), persisted to SQLite with success/failed status
- History API: `GET /api/documents` (newest first, `limit`/`offset`
  pagination) and `GET /api/documents/{id}` (full record + summary)
- Streamlit UI: backend check, file upload + summary display, history
  list with previous-summary viewer
- Env-based config (`backend/config.py`, `.env.example`)

## Project structure

```text
backend/        FastAPI app (main.py, config.py, database.py, models.py,
                schemas.py, extract.py, textutil.py, llm.py)
frontend/       Streamlit app (app.py, requests -> FastAPI only)
tests/          Automated tests (pytest)
uploads/        Runtime upload dir (ignored by Git)
documents.db    SQLite history database (ignored by Git)
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
# OPENROUTER_API_KEY is required for summarization (leave empty to run
# the app without AI; the summarize endpoint then returns 503)
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
- Upload: `POST /api/documents/upload` with a PDF/DOCX file -> `201`
- Summarize: `POST /api/documents/summarize` -> `200` with summary + `id`
- History: `GET http://127.0.0.1:8000/api/documents` -> newest-first list
- Streamlit shows "Connected" when backend is running, else a clear error.

## Security notes

- `.env`, `uploads/`, `*.db`, `venv/` are Git-ignored.
- Never commit API keys. Copy `.env.example` instead.
