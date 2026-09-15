"""OpenRouter calls via the OpenAI-compatible Chat Completions API.

All network traffic goes through _chat(), so tests patch a single seam
and never hit the network. The API key is read from settings
at call time and is never logged.
"""

import json

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
BULLETS_PROMPT_TEMPLATE = (
    "List the key points of this document section (part {i} of {n}) "
    "as short bullet lines, one per line:\n\n{chunk}"
)
PLAN_SYSTEM_PROMPT = (
    "You are a summarization planner. Reply with a single JSON object only, "
    ' shaped like {"strategy": "direct|map_reduce|key_points_first", '
    '"focus": "<one short phrase or empty>", "max_sections": <int>}.'
)


class LLMError(Exception):
    """LLM failure with an HTTP status code for the API layer."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _chat(messages: list[dict[str, str]]) -> str:
    """Send chat messages to OpenRouter and return the reply text."""
    return _chat_raw(
        messages,
        max_tokens=settings.openrouter_max_tokens,
        temperature=0.2,
    )


def _chat_raw(
    messages: list[dict[str, str]], *, max_tokens: int, temperature: float
) -> str:
    """Shared Chat Completions call with explicit sampling parameters."""
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
                "temperature": temperature,
                "max_tokens": max_tokens,
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


def summarize_bullets(chunk: str, i: int, n: int) -> str:
    """Extract the key points of one section as bullet lines."""
    return _chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": BULLETS_PROMPT_TEMPLATE.format(i=i, n=n, chunk=chunk),
            },
        ]
    )


def chat_json(system: str, user: str) -> dict:
    """Chat call that must reply with a JSON object (for the planner)."""
    raw = _chat_raw(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=settings.agent_plan_tokens,
        temperature=0.0,
    )
    try:
        return json.loads(raw)
    except ValueError:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            pass
    raise LLMError("AI service returned an invalid response.", 502)
