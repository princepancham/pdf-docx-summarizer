"""OpenRouter summarization via the OpenAI-compatible Chat Completions API.

All calls go through _chat() so tests can patch a single seam and
no test ever hits the network. The API key is read from settings
at call time and is never logged.
"""

import httpx

from backend.config import settings

SYSTEM_PROMPT = (
    "You are a helpful assistant that summarizes documents. "
    "Produce a concise, faithful summary: a short paragraph followed "
    "by the key points as bullet items."
)
CHUNK_PROMPT_TEMPLATE = "Summarize this section of a longer document (part {i} of {n}):\n\n{chunk}"
COMBINE_PROMPT_TEMPLATE = (
    "Combine the following section summaries into one coherent "
    "document summary:\n\n{partials}"
)
TRUNCATION_NOTE = "\n\n(Note: long document truncated to the first {n} sections.)"


class LLMError(Exception):
    """LLM failure with an HTTP status code for the API layer."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _chat(messages: list[dict[str, str]]) -> str:
    """Send chat messages to OpenRouter and return the reply text."""
    api_key = (settings.openrouter_api_key or "").strip()
    if not api_key:
        raise LLMError("AI service is not configured.", 503)
    url = f"{settings.openrouter_base_url.rstrip('/')}/chat/completions"
    try:
        response = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "pdf-docx-summarizer",
            },
            json={
                "model": settings.openrouter_model,
                "temperature": 0.2,
                "max_tokens": settings.openrouter_max_tokens,
                "messages": messages,
            },
            timeout=settings.openrouter_timeout_s,
        )
    except httpx.TimeoutException as exc:
        raise LLMError("AI service timed out.", 504) from exc
    except httpx.RequestError as exc:
        raise LLMError("AI service request failed.", 502) from exc

    if response.status_code == 429:
        raise LLMError("AI service is rate limited. Try again later.", 503)
    if response.status_code in (401, 403):
        raise LLMError("AI service configuration error.", 500)
    if response.status_code >= 400:
        raise LLMError("AI service request failed.", 502)

    try:
        content = response.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        raise LLMError("AI service returned an invalid response.", 502) from exc
    if not content or not content.strip():
        raise LLMError("AI service returned an empty response.", 502)
    return content.strip()


def summarize_text(text: str) -> str:
    """Summarize short text with a single OpenRouter call."""
    return _chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Summarize this document:\n\n{text}"},
        ]
    )


def summarize_chunks(chunks: list[str]) -> tuple[str, bool]:
    """Summarize each chunk, then combine. Returns (summary, truncated)."""
    if not chunks:
        raise LLMError("Nothing to summarize.", 502)
    if len(chunks) == 1:
        return summarize_text(chunks[0]), False

    truncated = len(chunks) > settings.max_chunks
    used = chunks[: settings.max_chunks]
    partials = [
        _chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": CHUNK_PROMPT_TEMPLATE.format(
                        i=i, n=len(used), chunk=chunk
                    ),
                },
            ]
        )
        for i, chunk in enumerate(used, start=1)
    ]
    combined = _chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": COMBINE_PROMPT_TEMPLATE.format(
                    partials="\n\n".join(partials)
                ),
            },
        ]
    )
    if truncated:
        combined += TRUNCATION_NOTE.format(n=settings.max_chunks)
    return combined, truncated
