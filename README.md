# PDF & DOCX Summarizer

AI-powered web app to upload PDF/DOCX documents and generate summaries,
with persistent history stored in SQLite.

## Current functionality (Phases 1–4)

- FastAPI with `GET /`, `GET /api/health`
- Secure upload: `POST /api/documents/upload` (PDF/DOCX, UUID filenames)
- Agentic summarization: `POST /api/documents/summarize` — a bounded-loop
  agent profiles each document, picks a strategy (`direct`, `map_reduce`,
  `key_points_first`), executes it, then checks deterministic quality
  gates (`quality_score`) with at most one repair retry. Returns summary,
  key points, strategy, quality score, and agent notes.
- History API: `GET /api/documents` (newest first, `limit`/`offset`
  pagination, `status` + `q` filters) and `GET /api/documents/{id}`
  (full record + summary), persisted to SQLite with success/failed status
- Streamlit UI: backend check, file upload + summary display (strategy,
  key points, quality score), searchable/filterable history with
  previous-summary viewer
- Env-based config (`backend/config.py`, `.env.example`)

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | (empty) | Secret key for OpenRouter; empty disables AI (503) |
| `OPENROUTER_MODEL` | `openrouter/free` | Chat model for summaries and planner |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | API base URL |
| `OPENROUTER_TIMEOUT_S` | `45` | Per-request timeout (seconds) |
| `OPENROUTER_MAX_TOKENS` | `500` | Max summary tokens |
| `CHUNK_CHARS` / `CHUNK_OVERLAP` / `MAX_CHUNKS` | `4000` / `200` / `12` | Long-document chunking |
| `AGENT_MAX_REPAIRS` | `1` | Bounded repair retries after failed quality gates |
| `AGENT_PLAN_TOKENS` | `150` | Token budget for the planning call |
| `AGENT_MIN_COVERAGE` | `0.3` | Required key-term coverage fraction |
| `DATABASE_URL` | `sqlite:///./documents.db` | History database location |

## Project structure

```text
backend/        FastAPI app (main.py, config.py, database.py, models.py,
                schemas.py, agent.py, extract.py, textutil.py, llm.py)
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
