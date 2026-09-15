"""FastAPI entrypoint — upload + agentic AI summarization + history.

Exposes a root endpoint, a health check, PDF/DOCX file upload,
agent-driven document summarization via OpenRouter, and SQLite-backed
document history. No multi-agent logic (single bounded-loop agent only).
"""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend import agent
from backend.config import settings, validate_settings
from backend.database import get_db, init_db
from backend.extract import ExtractionError, extract_text
from backend.llm import LLMError
from backend.models import Document
from backend.schemas import DocumentListOut, DocumentOut
from backend.textutil import clean_text

logger = logging.getLogger("pdf_summarizer")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    validate_settings()
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    description="AI-powered PDF & DOCX summarization API.",
    version="0.5.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach a short request ID to every response for traceability."""
    response = await call_next(request)
    response.headers["X-Request-ID"] = uuid.uuid4().hex[:8]
    return response

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
CHUNK_SIZE = 1024 * 1024  # 1 MiB per read for bounded memory + early abort


class UploadResponse(BaseModel):
    original_filename: str
    stored_filename: str
    content_type: str
    size_bytes: int


class SummarizeResponse(BaseModel):
    id: int
    filename: str
    stored_filename: str
    summary: str
    chunks: int
    chars: int
    truncated: bool
    model: str
    strategy: str
    key_points: list[str] = []
    quality_score: float = 0.0
    notes: list[str] = []


@app.get("/")
def read_root() -> dict[str, str]:
    """Root endpoint pointing to interactive docs."""
    return {"message": settings.app_name, "docs": "/docs"}


@app.get("/api/health")
def health_check() -> dict[str, str]:
    """Health check used by the Streamlit frontend and smoke tests."""
    return {"status": "ok", "app": settings.app_name, "env": settings.app_env}


async def _store_validated_upload(
    file: UploadFile,
) -> tuple[str, str, str, str, Path, int]:
    """Validate and store an upload; return (original, ext, ctype, stored, path, size).

    Same rules as before: extension -> MIME type -> magic bytes -> size limit.
    Partial files are deleted on any failure.
    """
    original_filename = Path(file.filename or "").name.strip()
    if not original_filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = Path(original_filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only PDF and DOCX are allowed.",
        )

    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported content type. Only PDF and DOCX are allowed.",
        )

    stored_filename = f"{uuid.uuid4()}{ext}"

    upload_dir = settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / stored_filename
    # Resolve and verify containment (defense in depth; name is UUID-generated).
    try:
        resolved_dest = dest.resolve()
        if resolved_dest.parent != upload_dir.resolve():
            raise HTTPException(status_code=500, detail="Storage path error.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Storage path error.")

    max_bytes = settings.max_file_size_mb * 1024 * 1024

    out = open(resolved_dest, "wb")
    try:
        total = 0
        first_chunk = True
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            if first_chunk:
                first_chunk = False
                if ext == ".pdf" and not chunk.startswith(b"%PDF"):
                    raise HTTPException(
                        status_code=400, detail="Invalid PDF file signature."
                    )
                if ext == ".docx" and not chunk.startswith(b"PK\x03\x04"):
                    raise HTTPException(
                        status_code=400, detail="Invalid DOCX file signature."
                    )
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"File too large. Max size is "
                        f"{settings.max_file_size_mb} MB."
                    ),
                )
            out.write(chunk)
        out.close()
        if total == 0:
            raise HTTPException(status_code=400, detail="Empty file.")
    except HTTPException:
        try:
            out.close()
        except Exception:
            pass
        try:
            resolved_dest.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    except Exception:
        try:
            out.close()
        except Exception:
            pass
        try:
            resolved_dest.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="Failed to store file.")
    finally:
        try:
            await file.close()
        except Exception:
            pass

    return (
        original_filename,
        ext,
        content_type,
        stored_filename,
        resolved_dest,
        total,
    )


@app.post("/api/documents/upload", status_code=201, response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Upload a PDF or DOCX file using chunked reads with early abort.

    Validation order: extension -> MIME type -> magic bytes -> size limit.
    The stored filename is a UUID4 with dashes plus the original extension.
    The original filename is preserved only as metadata, never as a path.
    Partial files are deleted on any failure.
    """
    original_filename, _, content_type, stored_filename, _, total = (
        await _store_validated_upload(file)
    )
    return UploadResponse(
        original_filename=original_filename,
        stored_filename=stored_filename,
        content_type=content_type,
        size_bytes=total,
    )


@app.post("/api/documents/summarize", response_model=SummarizeResponse)
async def summarize_document(
    file: UploadFile = File(...), db: Session = Depends(get_db)
) -> SummarizeResponse:
    """Upload a document and return an agent-produced summary.

    Flow: validate -> save -> extract -> clean -> agent -> persist.
    Successful and failed runs are recorded in the history database.
    """
    started = time.perf_counter()
    original_filename, ext, content_type, stored_filename, dest, total = (
        await _store_validated_upload(file)
    )

    def _record_failure(
        *, chars: int, detail: str, strategy: str = "unknown"
    ) -> None:
        db.add(
            Document(
                original_filename=original_filename,
                stored_filename=stored_filename,
                file_type=ext,
                content_type=content_type,
                size_bytes=total,
                status="failed",
                summary=None,
                chunks=0,
                chars=chars,
                truncated=False,
                model=settings.openrouter_model,
                error=detail,
                strategy=strategy,
                quality_score=0.0,
                key_points=None,
                notes=None,
            )
        )
        db.commit()
        logger.warning(
            "summarize failed file=%s strategy=%s detail=%s",
            stored_filename,
            strategy,
            detail,
        )

    try:
        raw_text = extract_text(dest)
    except ExtractionError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Failed to extract text.")

    cleaned = clean_text(raw_text)
    if not cleaned:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail="No extractable text found (scanned image?).",
        )

    if not (settings.openrouter_api_key or "").strip():
        detail = "AI service is not configured."
        _record_failure(chars=len(cleaned), detail=detail)
        raise HTTPException(status_code=503, detail=detail)

    try:
        result = agent.run(cleaned)
    except LLMError as exc:
        strategy = getattr(exc, "strategy", None) or "unknown"
        _record_failure(chars=len(cleaned), detail=exc.message, strategy=strategy)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)

    record = Document(
        original_filename=original_filename,
        stored_filename=stored_filename,
        file_type=ext,
        content_type=content_type,
        size_bytes=total,
        status="completed",
        summary=result.summary,
        chunks=result.chunks,
        chars=len(cleaned),
        truncated=result.truncated,
        model=settings.openrouter_model,
        error=None,
        strategy=result.strategy,
        quality_score=result.quality_score,
        key_points=json.dumps(result.key_points),
        notes="\n".join(result.notes) if result.notes else None,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "summarized id=%s strategy=%s chunks=%s score=%.2f ms=%s",
        record.id,
        result.strategy,
        result.chunks,
        result.quality_score,
        elapsed_ms,
    )

    return SummarizeResponse(
        id=record.id,
        filename=original_filename,
        stored_filename=stored_filename,
        summary=result.summary,
        chunks=result.chunks,
        chars=len(cleaned),
        truncated=result.truncated,
        model=settings.openrouter_model,
        strategy=result.strategy,
        key_points=result.key_points,
        quality_score=result.quality_score,
        notes=result.notes,
    )


def _to_document_out(record: Document) -> DocumentOut:
    """Convert a row to its schema, decoding stored JSON/lines fields."""
    try:
        key_points = json.loads(record.key_points) if record.key_points else []
    except ValueError:
        key_points = []
    if not isinstance(key_points, list):
        key_points = []
    notes = record.notes.split("\n") if record.notes else []
    return DocumentOut(
        id=record.id,
        original_filename=record.original_filename,
        stored_filename=record.stored_filename,
        file_type=record.file_type,
        content_type=record.content_type,
        size_bytes=record.size_bytes,
        status=record.status,
        summary=record.summary,
        chunks=record.chunks,
        chars=record.chars,
        truncated=record.truncated,
        model=record.model,
        error=record.error,
        strategy=record.strategy,
        quality_score=record.quality_score or 0.0,
        key_points=[str(p) for p in key_points],
        notes=[n for n in notes if n],
        created_at=record.created_at,
    )


@app.get("/api/documents", response_model=DocumentListOut)
def list_documents(
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
    q: str | None = None,
    db: Session = Depends(get_db),
) -> DocumentListOut:
    """List document history, newest first, with offset pagination."""
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    query = db.query(Document)
    if status is not None:
        if status not in ("completed", "failed"):
            raise HTTPException(
                status_code=422,
                detail="Invalid status filter. Use completed or failed.",
            )
        query = query.filter(Document.status == status)
    if q is not None and q.strip():
        query = query.filter(
            Document.original_filename.ilike(f"%{q.strip()}%")
        )
    total = query.count()
    rows = (
        query.order_by(Document.created_at.desc(), Document.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return DocumentListOut(documents=rows, total=total)


@app.get("/api/documents/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: int, db: Session = Depends(get_db)) -> DocumentOut:
    """Return a single document record with its summary."""
    record = db.query(Document).filter(Document.id == doc_id).first()
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return _to_document_out(record)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
    )
