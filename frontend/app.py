"""Streamlit entrypoint — Phase 1 foundation only.

Shows app title and backend connectivity status.
No upload, parsing, or summarization logic yet.
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
    st.write("Phase 1 foundation. Upload and summarization arrive in later phases.")
    st.caption(f"Backend: {BACKEND_URL}")

    if st.button("Check backend connection"):
        st.rerun()

    ok, message = check_backend_health(BACKEND_URL)
    if ok:
        st.success(message)
    else:
        st.error(message)
        st.info("Start the backend with: uvicorn backend.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
