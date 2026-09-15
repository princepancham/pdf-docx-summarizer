"""Document history API tests (isolated tmp SQLite DB, mocked LLM)."""

import dataclasses
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import backend.agent as agent_module
import backend.database as db_module
import backend.llm as llm_module
import backend.main as main_module
from backend.config import BASE_DIR, settings as prod_settings
from backend.database import Base, get_db
from backend.llm import LLMError
from backend.main import app
from tests.pdf_fixture import minimal_pdf_bytes

DOCX_MIME = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
PDF_MIME = "application/pdf"
MINIMAL_PDF = minimal_pdf_bytes()


def _docx_bytes(paragraphs: list[str]) -> bytes:
    from docx import Document

    doc = Document()
    for para in paragraphs:
        doc.add_paragraph(para)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient with tmp upload dir, tmp SQLite DB, and fake API key."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    isolated = dataclasses.replace(
        prod_settings,
        upload_dir=upload_dir,
        max_file_size_mb=20,
        openrouter_api_key="test-key",
    )
    monkeypatch.setattr(main_module, "settings", isolated)
    monkeypatch.setattr(llm_module, "settings", isolated)
    monkeypatch.setattr(agent_module, "settings", isolated)

    def _plan_unavailable(system: str, user: str) -> dict:
        raise LLMError("AI service returned an invalid response.", 502)

    monkeypatch.setattr(llm_module, "chat_json", _plan_unavailable)

    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    TestingSession = sessionmaker(
        bind=engine, autoflush=False, autocommit=False
    )
    from backend import models  # noqa: F401  (register tables)

    Base.metadata.create_all(bind=engine)

    def override_get_db():
        session = TestingSession()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def _summarize(client: TestClient, name="a.docx"):
    body = _docx_bytes([f"Content for {name}.", "Second paragraph here."])
    return client.post(
        "/api/documents/summarize", files={"file": (name, body, DOCX_MIME)}
    )


def test_summarize_persists_completed_row(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm_module, "_chat", lambda messages: "S1")
    response = _summarize(client)
    assert response.status_code == 200, response.text
    doc_id = response.json()["id"]
    assert isinstance(doc_id, int)

    detail = client.get(f"/api/documents/{doc_id}")
    assert detail.status_code == 200
    data = detail.json()
    assert data["status"] == "completed"
    assert data["summary"] == "S1"
    assert data["original_filename"] == "a.docx"
    assert data["error"] is None


def test_failed_run_persisted_but_error_returned(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(messages):
        raise LLMError("AI service timed out.", 504)

    monkeypatch.setattr(llm_module, "_chat", boom)
    response = _summarize(client, "bad.docx")
    assert response.status_code == 504

    listing = client.get("/api/documents")
    assert listing.status_code == 200
    docs = listing.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["status"] == "failed"

    detail = client.get(f"/api/documents/{docs[0]['id']}")
    assert detail.status_code == 200
    data = detail.json()
    assert data["status"] == "failed"
    assert data["summary"] is None
    assert data["error"] == "AI service timed out."


def test_missing_key_persisted_as_failed(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolated = dataclasses.replace(
        prod_settings,
        upload_dir=tmp_path / "uploads",
        openrouter_api_key=None,
    )
    monkeypatch.setattr(main_module, "settings", isolated)
    monkeypatch.setattr(llm_module, "settings", isolated)
    monkeypatch.setattr(agent_module, "settings", isolated)
    response = _summarize(client, "nokey.docx")
    assert response.status_code == 503
    docs = client.get("/api/documents").json()["documents"]
    assert len(docs) == 1
    assert docs[0]["status"] == "failed"


def test_list_newest_first_and_pagination(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm_module, "_chat", lambda messages: "S")
    for name in ("one.docx", "two.docx", "three.docx"):
        assert _summarize(client, name).status_code == 200

    page1 = client.get("/api/documents", params={"limit": 2})
    assert page1.status_code == 200
    body = page1.json()
    assert body["total"] == 3
    assert [d["original_filename"] for d in body["documents"]] == [
        "three.docx",
        "two.docx",
    ]

    page2 = client.get("/api/documents", params={"limit": 2, "offset": 2})
    body = page2.json()
    assert body["total"] == 3
    assert [d["original_filename"] for d in body["documents"]] == ["one.docx"]


def test_limit_clamped(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_module, "_chat", lambda messages: "S")
    assert _summarize(client).status_code == 200
    response = client.get("/api/documents", params={"limit": 500})
    assert response.status_code == 200
    assert len(response.json()["documents"]) <= 200
    response = client.get("/api/documents", params={"limit": 0})
    assert response.status_code == 200
    assert len(response.json()["documents"]) == 1


def test_detail_unknown_id_404(client: TestClient) -> None:
    response = client.get("/api/documents/9999")
    assert response.status_code == 404


def test_upload_not_persisted(client: TestClient) -> None:
    response = client.post(
        "/api/documents/upload",
        files={"file": ("plain.pdf", MINIMAL_PDF, PDF_MIME)},
    )
    assert response.status_code == 201, response.text
    body = client.get("/api/documents").json()
    assert body["total"] == 0
    assert body["documents"] == []


def test_real_database_file_untouched() -> None:
    assert not (BASE_DIR / "documents.db").exists()


def test_db_module_import_does_not_create_file() -> None:
    assert db_module.engine is not None
    assert not (BASE_DIR / "documents.db").exists()


def test_history_status_filter(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm_module, "_chat", lambda messages: "S")
    assert _summarize(client, "good.docx").status_code == 200

    def boom(messages):
        raise LLMError("AI service timed out.", 504)

    monkeypatch.setattr(llm_module, "_chat", boom)
    assert _summarize(client, "bad.docx").status_code == 504

    completed = client.get("/api/documents", params={"status": "completed"})
    assert completed.status_code == 200
    assert [d["original_filename"] for d in completed.json()["documents"]] == [
        "good.docx"
    ]
    failed = client.get("/api/documents", params={"status": "failed"})
    assert [d["original_filename"] for d in failed.json()["documents"]] == [
        "bad.docx"
    ]
    bogus = client.get("/api/documents", params={"status": "bogus"})
    assert bogus.status_code == 422


def test_history_q_search(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm_module, "_chat", lambda messages: "S")
    assert _summarize(client, "quarterly-report.docx").status_code == 200
    assert _summarize(client, "meeting-notes.docx").status_code == 200

    response = client.get("/api/documents", params={"q": "quarter"})
    assert response.status_code == 200
    docs = response.json()["documents"]
    assert [d["original_filename"] for d in docs] == ["quarterly-report.docx"]

    response = client.get("/api/documents", params={"q": "QUARTER"})
    assert len(response.json()["documents"]) == 1

    response = client.get("/api/documents", params={"q": "nothing-here"})
    assert response.json() == {"documents": [], "total": 0}


def test_responses_carry_request_id(client: TestClient) -> None:
    assert "x-request-id" in {k.lower() for k in client.get("/").headers}
    assert "x-request-id" in {
        k.lower() for k in client.get("/api/documents").headers
    }


def test_history_lists_strategy(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm_module, "_chat", lambda messages: "S")
    assert _summarize(client, "strat.docx").status_code == 200
    docs = client.get("/api/documents").json()["documents"]
    assert docs[0]["strategy"] == "direct"
