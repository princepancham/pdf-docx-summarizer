"""Streamlit entrypoint — upload, health check, and AI summarization.

Uses `requests` only for Streamlit -> FastAPI HTTP communication.
"""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


def check_backend_health(base_url: str, timeout_seconds: int = 5) -> tuple[bool, str]:
    """Ping FastAPI health endpoint. Returns (ok, message)."""
    try:
        response = requests.get(f"{base_url}/api/health", timeout=timeout_seconds)
    except requests.RequestException:
        return False, f"Cannot reach backend at {base_url}. Is FastAPI running?"
    if response.status_code == 200:
        try:
            data = response.json()
        except ValueError:
            return False, "Backend responded but returned invalid JSON."
        return True, f"Connected ({data.get('app', 'API')}, env={data.get('env', '?')})."
    return False, f"Backend returned status {response.status_code}."


def main() -> None:
    st.title("PDF & DOCX Summarizer")
    st.caption(f"Backend: {BACKEND_URL}")

    if st.button("Check backend connection"):
        st.rerun()

    ok, message = check_backend_health(BACKEND_URL)
    if ok:
        st.success(message)
    else:
        st.error(message)
        st.info("Start the backend with: uvicorn backend.main:app --reload --port 8000")

    st.divider()
    st.subheader("Summarize a document")

    uploaded = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])
    if uploaded is not None and st.button("Summarize"):
        with st.spinner("Summarizing..."):
            ok, result = summarize_document(
                uploaded.name, uploaded.getvalue(), uploaded.type
            )
        if ok:
            assert isinstance(result, dict)
            st.write(result.get("summary", ""))
            details = (
                f"Chunks: {result.get('chunks')} | "
                f"Chars: {result.get('chars')} | "
                f"Model: {result.get('model')}"
            )
            st.caption(details)
            if result.get("truncated"):
                st.warning("Long document was truncated to the first sections.")
            st.rerun()
        else:
            assert isinstance(result, str)
            st.error(result)

    st.divider()
    st.subheader("History")
    render_history()


def summarize_document(
    filename: str, content: bytes, mime_type: str | None
) -> tuple[bool, dict | str]:
    """POST a file to the summarize endpoint. Returns (ok, data|error)."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    mime = (mime_type or "").strip() or (
        "application/pdf"
        if ext == "pdf"
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/documents/summarize",
            files={"file": (filename, content, mime)},
            timeout=120,
        )
    except requests.RequestException:
        return False, f"Cannot reach backend at {BACKEND_URL}. Is FastAPI running?"
    try:
        data = response.json()
    except ValueError:
        return False, f"Backend returned status {response.status_code}."
    if response.status_code == 200:
        return True, data
    detail = data.get("detail") if isinstance(data, dict) else None
    return False, str(detail or f"Backend returned status {response.status_code}.")


def fetch_history(limit: int = 50) -> tuple[bool, dict | str]:
    """GET the document history list. Returns (ok, data|error)."""
    try:
        response = requests.get(
            f"{BACKEND_URL}/api/documents", params={"limit": limit}, timeout=30
        )
    except requests.RequestException:
        return False, f"Cannot reach backend at {BACKEND_URL}. Is FastAPI running?"
    try:
        data = response.json()
    except ValueError:
        return False, f"Backend returned status {response.status_code}."
    if response.status_code == 200:
        return True, data
    detail = data.get("detail") if isinstance(data, dict) else None
    return False, str(detail or f"Backend returned status {response.status_code}.")


def fetch_document(doc_id: int) -> tuple[bool, dict | str]:
    """GET a single document record. Returns (ok, data|error)."""
    try:
        response = requests.get(
            f"{BACKEND_URL}/api/documents/{doc_id}", timeout=30
        )
    except requests.RequestException:
        return False, f"Cannot reach backend at {BACKEND_URL}. Is FastAPI running?"
    try:
        data = response.json()
    except ValueError:
        return False, f"Backend returned status {response.status_code}."
    if response.status_code == 200:
        return True, data
    detail = data.get("detail") if isinstance(data, dict) else None
    return False, str(detail or f"Backend returned status {response.status_code}.")


def render_history() -> None:
    """Render the document history list with a detail viewer."""
    ok, result = fetch_history()
    if not ok:
        assert isinstance(result, str)
        st.error(result)
        return
    assert isinstance(result, dict)
    documents = result.get("documents", [])
    if not documents:
        st.info("No documents yet. Summarize a file above to build history.")
        return
    for doc in documents:
        label = f"#{doc.get('id')} {doc.get('original_filename')} ({doc.get('status')})"
        if st.button(label, key=f"doc-{doc.get('id')}"):
            st.session_state.selected_id = doc.get("id")
    selected = st.session_state.get("selected_id")
    if selected is not None:
        ok, result = fetch_document(selected)
        if not ok:
            assert isinstance(result, str)
            st.error(result)
            return
        assert isinstance(result, dict)
        st.subheader(f"Summary for {result.get('original_filename')}")
        if result.get("status") == "failed":
            st.error(result.get("error") or "Processing failed.")
        else:
            st.write(result.get("summary", ""))
        st.caption(
            f"Chunks: {result.get('chunks')} | "
            f"Chars: {result.get('chars')} | "
            f"Model: {result.get('model')}"
        )
        if result.get("truncated"):
            st.warning("Long document was truncated to the first sections.")


if __name__ == "__main__":
    main()
