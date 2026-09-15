"""Pydantic response schemas for document history."""

from datetime import datetime

from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: int
    original_filename: str
    stored_filename: str
    file_type: str
    content_type: str | None = None
    size_bytes: int
    status: str
    summary: str | None = None
    chunks: int
    chars: int
    truncated: bool
    model: str | None = None
    error: str | None = None
    strategy: str | None = None
    quality_score: float = 0.0
    key_points: list[str] = []
    notes: list[str] = []
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentListItem(BaseModel):
    id: int
    original_filename: str
    file_type: str
    status: str
    chunks: int
    chars: int
    truncated: bool
    model: str | None = None
    strategy: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentListOut(BaseModel):
    documents: list[DocumentListItem]
    total: int
