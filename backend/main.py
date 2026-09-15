"""FastAPI entrypoint — document upload + AI summarization.

Exposes a root endpoint, a health check, PDF/DOCX file upload,
and single-call document summarization via OpenRouter.
No database logic yet (deferred to later phases).
"""

import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.config import settings
from backend.extract import ExtractionError, extract_text
from backend.llm import LLMError, summarize_chunks, summarize_text
from backend.textutil import clean_text, split_chunks

app = FastAPI(
    title=settings.app_name,
    description="AI-powered PDF & DOCX summarization API.",
    version="0.3.0",
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
async def summarize_document(file: UploadFile = File(...)) -> SummarizeResponse:
    """Upload a document and return an OpenRouter-generated summary.

    Flow: validate -> save -> extract -> clean -> chunk if needed -> LLM.
    """
    original_filename, _, _, stored_filename, dest, _ = (
        await _store_validated_upload(file)
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
        raise HTTPException(
            status_code=503, detail="AI service is not configured."
        )

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
        raise HTTPException(status_code=exc.status_code, detail=exc.message)

    return SummarizeResponse(
        filename=original_filename,
        stored_filename=stored_filename,
        summary=summary,
        chunks=num_chunks,
        chars=chars,
        truncated=truncated,
        model=settings.openrouter_model,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
    )
