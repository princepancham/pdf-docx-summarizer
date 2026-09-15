"""Summarize endpoint tests with a mocked LLM (no network calls)."""

import dataclasses
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.llm as llm_module
import backend.main as main_module
from backend.config import settings as prod_settings
from backend.llm import LLMError
from backend.main import app
from tests.pdf_fixture import minimal_pdf_bytes

PDF_MIME = "application/pdf"
DOCX_MIME = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)

MINIMAL_PDF = minimal_pdf_bytes()


def _docx_bytes(paragraphs: list[str]) -> bytes:
    from docx import Document

    doc = Document()
    for para in paragraphs:
        doc.add_paragraph(para)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _isolated_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **overrides
):
    """Point both main and llm settings at tmp_path with a fake API key."""
    values = {
        "upload_dir": tmp_path,
        "max_file_size_mb": 20,
        "openrouter_api_key": "test-key",
    }
    values.update(overrides)
    isolated = dataclasses.replace(prod_settings, **values)
    monkeypatch.setattr(main_module, "settings", isolated)
    monkeypatch.setattr(llm_module, "settings", isolated)
    return isolated


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    _isolated_settings(monkeypatch, tmp_path)
    return TestClient(app)


def test_summarize_short_docx(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        llm_module,
        "_chat",
        lambda messages: calls.append(messages[-1]["content"]) or "SHORT SUMMARY",
    )
    body = _docx_bytes(["Hello summarizer.", "Second paragraph here."])
    response = client.post(
        "/api/documents/summarize",
        files={"file": ("notes.docx", body, DOCX_MIME)},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["filename"] == "notes.docx"
    assert data["summary"] == "SHORT SUMMARY"
    assert data["chunks"] == 1
    assert data["truncated"] is False
    assert data["chars"] > 0
    assert data["model"] == main_module.settings.openrouter_model
    assert len(calls) == 1
    assert (tmp_path / data["stored_filename"]).is_file()


def test_summarize_short_pdf(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm_module, "_chat", lambda messages: "PDF SUMMARY")
    response = client.post(
        "/api/documents/summarize",
        files={"file": ("doc.pdf", MINIMAL_PDF, PDF_MIME)},
    )
    assert response.status_code == 200, response.text
    assert response.json()["summary"] == "PDF SUMMARY"


def test_summarize_long_doc_uses_chunks(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def fake_chat(messages):
        seen.append(messages[-1]["content"])
        if messages[-1]["content"].startswith("Combine"):
            return "FINAL SUMMARY"
        return f"PART {len(seen)}"

    monkeypatch.setattr(llm_module, "_chat", fake_chat)
    paras = ["Lorem ipsum dolor sit amet. " * 20 for _ in range(30)]
    response = client.post(
        "/api/documents/summarize",
        files={"file": ("long.docx", _docx_bytes(paras), DOCX_MIME)},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["chunks"] > 1
    assert data["truncated"] is False
    assert data["summary"] == "FINAL SUMMARY"
    # N chunk calls + 1 combine call.
    assert len(seen) == data["chunks"] + 1


def test_summarize_truncation_flagged(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolated_settings(monkeypatch, tmp_path, max_chunks=2)
    monkeypatch.setattr(llm_module, "_chat", lambda messages: "X")
    paras = ["Wordy paragraph content here. " * 30 for _ in range(30)]
    response = client.post(
        "/api/documents/summarize",
        files={"file": ("big.docx", _docx_bytes(paras), DOCX_MIME)},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["chunks"] == 2
    assert data["truncated"] is True
    assert "truncated" in data["summary"].lower()


def test_summarize_missing_key_returns_503(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolated_settings(monkeypatch, tmp_path, openrouter_api_key=None)

    def fail_if_called(messages):
        raise AssertionError("LLM must not be called without a key")

    monkeypatch.setattr(llm_module, "_chat", fail_if_called)
    response = client.post(
        "/api/documents/summarize",
        files={"file": ("doc.pdf", MINIMAL_PDF, PDF_MIME)},
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "AI service is not configured."


def test_summarize_llm_timeout_returns_504(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(messages):
        raise LLMError("AI service timed out.", 504)

    monkeypatch.setattr(llm_module, "_chat", boom)
    response = client.post(
        "/api/documents/summarize",
        files={"file": ("doc.pdf", MINIMAL_PDF, PDF_MIME)},
    )
    assert response.status_code == 504


def test_summarize_invalid_extension_400(client: TestClient) -> None:
    response = client.post(
        "/api/documents/summarize",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_summarize_blank_pdf_returns_422(
    client: TestClient, tmp_path: Path
) -> None:
    from pypdf import PdfWriter

    buf = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(buf)
    response = client.post(
        "/api/documents/summarize",
        files={"file": ("blank.pdf", buf.getvalue(), PDF_MIME)},
    )
    assert response.status_code == 422
    assert list(tmp_path.iterdir()) == []


def test_summarize_corrupt_pdf_leaves_no_orphan(
    client: TestClient, tmp_path: Path
) -> None:
    response = client.post(
        "/api/documents/summarize",
        files={"file": ("broken.pdf", b"%PDF-1.4 not really a pdf", PDF_MIME)},
    )
    assert response.status_code == 422
    assert list(tmp_path.iterdir()) == []
