"""FastAPI entrypoint — document upload + AI summarization + history.

Exposes a root endpoint, a health check, PDF/DOCX file upload,
single-call document summarization via OpenRouter, and SQLite-backed
document history. No agent logic yet (deferred to later phases).
"""

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import get_db, init_db
from backend.extract import ExtractionError, extract_text
from backend.llm import LLMError, summarize_chunks, summarize_text
from backend.models import Document
from backend.schemas import DocumentListOut, DocumentOut
from backend.textutil import clean_text, split_chunks


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    description="AI-powered PDF & DOCX summarization API.",
    version="0.4.0",
    lifespan=lifespan,
)

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
    """Upload a document and return an OpenRouter-generated summary.

    Flow: validate -> save -> extract -> clean -> chunk if needed -> LLM.
    Successful and failed runs are recorded in the history database.
    """
    original_filename, ext, content_type, stored_filename, dest, total = (
        await _store_validated_upload(file)
    )

    def _record_failure(
        *, chars: int, detail: str, model: str | None = None
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
                model=model if model is not None else settings.openrouter_model,
                error=detail,
            )
        )
        db.commit()

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

    chars = len(cleaned)
    try:
        if chars <= settings.chunk_chars:
            summary = summarize_text(cleaned)
            num_chunks, truncated = 1, False
        else:
            all_chunks = split_chunks(
                cleaned, settings.chunk_chars, settings.chunk_overlap
            )
            summary, truncated = summarize_chunks(all_chunks)
            num_chunks = min(len(all_chunks), settings.max_chunks)
    except LLMError as exc:
        _record_failure(chars=chars, detail=exc.message)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)

    record = Document(
        original_filename=original_filename,
        stored_filename=stored_filename,
        file_type=ext,
        content_type=content_type,
        size_bytes=total,
        status="completed",
        summary=summary,
        chunks=num_chunks,
        chars=chars,
        truncated=truncated,
        model=settings.openrouter_model,
        error=None,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return SummarizeResponse(
        id=record.id,
        filename=original_filename,
        stored_filename=stored_filename,
        summary=summary,
        chunks=num_chunks,
        chars=chars,
        truncated=truncated,
        model=settings.openrouter_model,
    )


@app.get("/api/documents", response_model=DocumentListOut)
def list_documents(
    limit: int = 50, offset: int = 0, db: Session = Depends(get_db)
) -> DocumentListOut:
    """List document history, newest first, with offset pagination."""
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    total = db.query(Document).count()
    rows = (
        db.query(Document)
        .order_by(Document.created_at.desc(), Document.id.desc())
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
    return record


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
    )
