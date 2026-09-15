"""FastAPI entrypoint — Phase 1 foundation only.

Exposes a root endpoint and a health check. No upload, parsing,
summarization, or database logic yet (deferred to later phases).
"""

from fastapi import FastAPI

from backend.config import settings

app = FastAPI(
    title=settings.app_name,
    description="AI-powered PDF & DOCX summarization API (Phase 1 foundation).",
    version="0.1.0",
)


@app.get("/")
def read_root() -> dict[str, str]:
    """Root endpoint pointing to interactive docs."""
    return {"message": settings.app_name, "docs": "/docs"}


@app.get("/api/health")
def health_check() -> dict[str, str]:
    """Health check used by the Streamlit frontend and smoke tests."""
    return {"status": "ok", "app": settings.app_name, "env": settings.app_env}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
    )
