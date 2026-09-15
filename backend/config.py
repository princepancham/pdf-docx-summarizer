"""Application configuration loaded from environment variables.

OPENROUTER_API_KEY stays optional so tests and local runs without
AI credentials still work; the summarize endpoint returns 503 then.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _get_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def _get_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _get_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_env: str
    debug: bool
    backend_host: str
    backend_port: int
    upload_dir: Path
    max_file_size_mb: int
    openrouter_api_key: str | None
    openrouter_model: str
    openrouter_base_url: str
    openrouter_timeout_s: int
    openrouter_max_tokens: int
    chunk_chars: int
    chunk_overlap: int
    max_chunks: int


def get_settings() -> Settings:
    """Build settings from environment with safe defaults."""
    upload_dir_raw = _get_str("UPLOAD_DIR", "uploads")
    upload_dir = Path(upload_dir_raw)
    if not upload_dir.is_absolute():
        upload_dir = BASE_DIR / upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)

    raw_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    return Settings(
        app_name=_get_str("APP_NAME", "PDF & DOCX Summarizer"),
        app_env=_get_str("APP_ENV", "dev"),
        debug=_get_bool("DEBUG", True),
        backend_host=_get_str("BACKEND_HOST", "127.0.0.1"),
        backend_port=_get_int("BACKEND_PORT", 8000),
        upload_dir=upload_dir,
        max_file_size_mb=_get_int("MAX_FILE_SIZE_MB", 20),
        openrouter_api_key=raw_key or None,
        openrouter_model=_get_str("OPENROUTER_MODEL", "openrouter/free"),
        openrouter_base_url=_get_str(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ),
        openrouter_timeout_s=_get_int("OPENROUTER_TIMEOUT_S", 45),
        openrouter_max_tokens=_get_int("OPENROUTER_MAX_TOKENS", 500),
        chunk_chars=_get_int("CHUNK_CHARS", 4000),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 200),
        max_chunks=_get_int("MAX_CHUNKS", 12),
    )


settings = get_settings()
