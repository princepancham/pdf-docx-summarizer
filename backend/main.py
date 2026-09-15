"""FastAPI entrypoint — Phase 2 upload endpoint.

Exposes a root endpoint, a health check, and PDF/DOCX file upload.
No parsing, summarization, or database logic yet (deferred to later phases).
"""

import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.config import settings

app = FastAPI(
    title=settings.app_name,
    description="AI-powered PDF & DOCX summarization API (Phase 2 upload).",
    version="0.2.0",
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


@app.get("/")
def read_root() -> dict[str, str]:
    """Root endpoint pointing to interactive docs."""
    return {"message": settings.app_name, "docs": "/docs"}


@app.get("/api/health")
def health_check() -> dict[str, str]:
    """Health check used by the Streamlit frontend and smoke tests."""
    return {"status": "ok", "app": settings.app_name, "env": settings.app_env}


@app.post("/api/documents/upload", status_code=201, response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Upload a PDF or DOCX file using chunked reads with early abort.

    Validation order: extension -> MIME type -> magic bytes -> size limit.
    The stored filename is a UUID4 with dashes plus the original extension.
    The original filename is preserved only as metadata, never as a path.
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

    return UploadResponse(
        original_filename=original_filename,
        stored_filename=stored_filename,
        content_type=content_type,
        size_bytes=total,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
    )
