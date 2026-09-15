"""Phase 2 upload endpoint tests.

Covers POST /api/documents/upload validation, storage, and cleanup,
plus regression checks for GET / and GET /api/health.
"""

import dataclasses
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.main as main_module
from backend.config import settings as prod_settings
from backend.main import app

PDF_MIME = "application/pdf"
DOCX_MIME = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
PDF_BYTES = b"%PDF-1.4 fake pdf content for tests"
DOCX_BYTES = b"PK\x03\x04 fake docx content for tests"
UUID_DASHED = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient with uploads isolated to tmp_path (20 MB limit)."""
    isolated = dataclasses.replace(
        prod_settings, upload_dir=tmp_path, max_file_size_mb=20
    )
    monkeypatch.setattr(main_module, "settings", isolated)
    return TestClient(app)


def _upload(
    client: TestClient,
    filename: str,
    content: bytes,
    mime: str,
):
    return client.post(
        "/api/documents/upload",
        files={"file": (filename, content, mime)},
    )


def test_root_ok(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_health_ok(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_valid_pdf_201(client: TestClient, tmp_path: Path) -> None:
    response = _upload(client, "report.pdf", PDF_BYTES, PDF_MIME)
    assert response.status_code == 201, response.text
    data = response.json()
    assert data["original_filename"] == "report.pdf"
    assert data["content_type"] == PDF_MIME
    assert data["size_bytes"] == len(PDF_BYTES)
    assert UUID_DASHED.fullmatch(Path(data["stored_filename"]).stem)
    assert data["stored_filename"].endswith(".pdf")
    stored = tmp_path / data["stored_filename"]
    assert stored.is_file()
    assert stored.read_bytes() == PDF_BYTES


def test_valid_docx_201(client: TestClient, tmp_path: Path) -> None:
    response = _upload(client, "report.docx", DOCX_BYTES, DOCX_MIME)
    assert response.status_code == 201, response.text
    data = response.json()
    assert data["original_filename"] == "report.docx"
    assert data["content_type"] == DOCX_MIME
    assert data["size_bytes"] == len(DOCX_BYTES)
    assert UUID_DASHED.fullmatch(Path(data["stored_filename"]).stem)
    assert data["stored_filename"].endswith(".docx")
    assert (tmp_path / data["stored_filename"]).read_bytes() == DOCX_BYTES


def test_uppercase_extension_201(client: TestClient, tmp_path: Path) -> None:
    response = _upload(client, "REPORT.PDF", PDF_BYTES, PDF_MIME)
    assert response.status_code == 201, response.text
    data = response.json()
    assert data["original_filename"] == "REPORT.PDF"
    assert data["stored_filename"].endswith(".pdf")
    assert (tmp_path / data["stored_filename"]).is_file()


def test_unsupported_extension_400(client: TestClient, tmp_path: Path) -> None:
    response = _upload(client, "notes.txt", b"hello", "text/plain")
    assert response.status_code == 400
    assert list(tmp_path.iterdir()) == []


def test_invalid_mime_400(client: TestClient, tmp_path: Path) -> None:
    response = _upload(client, "report.pdf", PDF_BYTES, "application/octet-stream")
    assert response.status_code == 400
    assert list(tmp_path.iterdir()) == []


def test_spoofed_pdf_magic_400(client: TestClient, tmp_path: Path) -> None:
    response = _upload(client, "evil.pdf", b"MZ not a pdf", PDF_MIME)
    assert response.status_code == 400
    assert "signature" in response.json()["detail"].lower()
    assert list(tmp_path.iterdir()) == []


def test_spoofed_docx_magic_400(client: TestClient, tmp_path: Path) -> None:
    response = _upload(client, "fake.docx", b"%PDF not a docx", DOCX_MIME)
    assert response.status_code == 400
    assert "signature" in response.json()["detail"].lower()
    assert list(tmp_path.iterdir()) == []


def test_empty_file_400(client: TestClient, tmp_path: Path) -> None:
    response = _upload(client, "empty.pdf", b"", PDF_MIME)
    assert response.status_code == 400
    assert list(tmp_path.iterdir()) == []


def test_oversized_file_413(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        main_module,
        "settings",
        dataclasses.replace(
            prod_settings, upload_dir=tmp_path, max_file_size_mb=1
        ),
    )
    big = b"%PDF" + b"A" * (2 * 1024 * 1024)
    response = _upload(client, "big.pdf", big, PDF_MIME)
    assert response.status_code == 413
    assert list(tmp_path.iterdir()) == []


def test_traversal_filename_safely_stored(
    client: TestClient, tmp_path: Path
) -> None:
    response = _upload(client, "../../evil.pdf", PDF_BYTES, PDF_MIME)
    assert response.status_code == 201, response.text
    data = response.json()
    # Original is metadata only, stripped of any directory components.
    assert "/" not in data["original_filename"]
    assert "\\" not in data["original_filename"]
    assert UUID_DASHED.fullmatch(Path(data["stored_filename"]).stem)
    stored = (tmp_path / data["stored_filename"]).resolve()
    assert stored.parent == tmp_path.resolve()
    assert stored.read_bytes() == PDF_BYTES


def test_partial_files_cleaned_up_after_failures(
    client: TestClient, tmp_path: Path
) -> None:
    cases = [
        ("notes.txt", b"hello", "text/plain"),
        ("evil.pdf", b"MZ not a pdf", PDF_MIME),
        ("fake.docx", b"nope", DOCX_MIME),
        ("empty.pdf", b"", PDF_MIME),
        ("report.pdf", PDF_BYTES, "application/octet-stream"),
    ]
    for filename, content, mime in cases:
        response = _upload(client, filename, content, mime)
        assert response.status_code in {400, 413}, (filename, response.text)
    assert list(tmp_path.iterdir()) == []
