"""Single bounded-loop summarization agent: profile -> plan -> execute -> evaluate.

The agent perceives each document (text-only profile signals), picks a
strategy from a fixed menu (via one cheap planning call with a
deterministic fallback), executes with the existing llm tools, then
checks the result against deterministic quality gates with at most one
repair retry. No frameworks, no multi-agent orchestration.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from backend import llm as llm_tools
from backend.config import settings
from backend.llm import (
    COMBINE_PROMPT_TEMPLATE,
    PLAN_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    TRUNCATION_NOTE,
    LLMError,
)
from backend.textutil import split_chunks

STRATEGIES = ("direct", "map_reduce", "key_points_first")

_STOPWORDS = frozenset(
    "the a an and or of to in on for with is are was were be been by as at "
    "from that this these those it its into over such than then them they "
    "their there here which while will would can could should have has had "
    "not but all any each more most other some what when where who whom "
    "your you our out about also after before between during under again "
    "once only same very just don now".split()
)
_BULLET_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*\S)\s*$")
_HEADING_HASH = re.compile(r"^\s*#{1,6}\s+\S")


@dataclass
class DocumentProfile:
    chars: int
    paragraphs: int
    headings: int
    avg_para_len: int
    digit_ratio: float
    bullet_ratio: float
    density: str  # sparse | normal | dense


@dataclass
class Plan:
    strategy: str
    focus: str = ""
    max_sections: int = 12


@dataclass
class AgentResult:
    summary: str
    key_points: list[str] = field(default_factory=list)
    strategy: str = "direct"
    quality_score: float = 0.0
    notes: list[str] = field(default_factory=list)
    chunks: int = 1
    truncated: bool = False
    calls: int = 0


def profile_text(text: str) -> DocumentProfile:
    """Measure text-only characteristics of a cleaned document."""
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    lines = [ln for ln in text.split("\n") if ln.strip()]
    headings = sum(
        1
        for ln in lines
        if _HEADING_HASH.match(ln)
        or (
            len(ln) <= 80
            and ln == ln.upper()
            and sum(c.isalpha() for c in ln) >= 4
        )
    )
    bullets = sum(1 for ln in lines if _BULLET_LINE.match(ln))
    chars = len(text)
    digits = sum(c.isdigit() for c in text)
    avg_para = chars // len(paras) if paras else 0
    digit_ratio = round(digits / chars, 4) if chars else 0.0
    bullet_ratio = round(bullets / len(lines), 4) if lines else 0.0
    if avg_para > 800 or bullet_ratio > 0.25 or digit_ratio > 0.08:
        density = "dense"
    elif len(paras) <= 3 and avg_para < 300:
        density = "sparse"
    else:
        density = "normal"
    return DocumentProfile(
        chars=chars,
        paragraphs=len(paras),
        headings=headings,
        avg_para_len=avg_para,
        digit_ratio=digit_ratio,
        bullet_ratio=bullet_ratio,
        density=density,
    )


def deterministic_plan(profile: DocumentProfile) -> Plan:
    """Rule-based strategy choice (also the planner fallback)."""
    if profile.chars <= settings.chunk_chars:
        return Plan(strategy="direct", max_sections=settings.max_chunks)
    if profile.density == "dense" or profile.headings >= 3:
        return Plan(strategy="key_points_first", max_sections=settings.max_chunks)
    return Plan(strategy="map_reduce", max_sections=settings.max_chunks)


def plan_strategy(profile: DocumentProfile) -> tuple[Plan, bool]:
    """Ask the LLM for a strategy; fall back to rules on any soft failure.

    Returns (plan, used_fallback). A missing-key 503 is fatal and propagates.
    """
    user = (
        "Document profile: "
        f"{profile.chars} chars, {profile.paragraphs} paragraphs, "
        f"{profile.headings} headings, avg paragraph {profile.avg_para_len} chars, "
        f"density {profile.density}. "
        f"Chunk budget is {settings.chunk_chars} chars x {settings.max_chunks} sections. "
        "Which strategy fits best?"
    )
    try:
        raw = llm_tools.chat_json(PLAN_SYSTEM_PROMPT, user)
    except LLMError as exc:
        if exc.status_code == 503 and "not configured" in exc.message:
            raise
        return deterministic_plan(profile), True
    strategy = raw.get("strategy")
    if strategy not in STRATEGIES:
        return deterministic_plan(profile), True
    try:
        max_sections = int(raw.get("max_sections", settings.max_chunks))
    except (TypeError, ValueError):
        max_sections = settings.max_chunks
    return (
        Plan(
            strategy=strategy,
            focus=str(raw.get("focus") or "")[:120],
            max_sections=max(1, min(max_sections, settings.max_chunks)),
        ),
        False,
    )


def _parse_bullets(text: str) -> list[str]:
    points = []
    for line in text.split("\n"):
        match = _BULLET_LINE.match(line)
        if match:
            points.append(match.group(1).strip())
    return points


def _execute(
    plan: Plan, text: str, repair_notes: list[str] | None = None
) -> tuple[str, list[str], int, bool, int]:
    """Run one strategy; returns (summary, key_points, chunks, truncated, calls)."""
    repair = ""
    if repair_notes:
        repair = (
            "\n\nPrevious attempt had these problems: "
            + "; ".join(repair_notes)
            + ". Try again, addressing each problem."
        )
    if plan.strategy == "direct":
        summary = llm_tools.summarize_text(text + repair)
        return summary, [], 1, False, 1
    if plan.strategy == "key_points_first":
        chunks = split_chunks(text, settings.chunk_chars, settings.chunk_overlap)
        used = chunks[: plan.max_sections]
        truncated = len(chunks) > plan.max_sections
        partials = [
            llm_tools.summarize_bullets(c, i, len(used))
            for i, c in enumerate(used, 1)
        ]
        key_points = [p for part in partials for p in _parse_bullets(part)][:10]
        combined = summarize_chunks_combine(partials, repair)
        if truncated:
            combined += TRUNCATION_NOTE.format(n=plan.max_sections)
        return combined, key_points, len(used), truncated, len(used) + 1
    # map_reduce
    chunks = split_chunks(text, settings.chunk_chars, settings.chunk_overlap)
    summary, truncated = llm_tools.summarize_chunks(chunks)
    if repair:
        summary = summarize_chunks_combine([summary], repair)
    return summary, [], min(len(chunks), settings.max_chunks), truncated, len(
        chunks[: settings.max_chunks]
    ) + 1 + (1 if repair else 0)


def summarize_chunks_combine(partials: list[str], repair: str = "") -> str:
    """Combine partial summaries (shared by map-reduce repair and key-points)."""
    return llm_tools._chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": COMBINE_PROMPT_TEMPLATE.format(
                    partials="\n\n".join(partials)
                )
                + repair,
            },
        ]
    )


def _content_words(text: str) -> list[str]:
    return [
        w
        for w in re.findall(r"[a-z]{5,}", text.lower())
        if w not in _STOPWORDS
    ]


def evaluate(
    summary: str,
    key_points: list[str],
    source: str,
    *,
    strategy: str,
    truncated: bool,
) -> tuple[float, list[str]]:
    """Deterministic quality gates -> (quality_score in [0,1], notes)."""
    notes: list[str] = []
    passed = 0
    total = 0

    total += 1
    if len(summary) >= 50:
        passed += 1
    else:
        notes.append("summary too short")

    total += 1
    if len(summary) >= max(50, int(len(source) * 0.02)):
        passed += 1
    else:
        notes.append("summary short relative to source")

    total += 1
    top = [w for w, _ in Counter(_content_words(source)).most_common(15)]
    if not top:
        passed += 1
    else:
        need = max(1, int(len(top) * settings.agent_min_coverage))
        summary_words = set(_content_words(summary))
        hit = sum(1 for w in top if w in summary_words)
        if hit >= need:
            passed += 1
        else:
            notes.append(f"low key-term coverage ({hit}/{len(top)})")

    total += 1
    if strategy == "key_points_first":
        if 3 <= len(key_points) <= 10:
            passed += 1
        else:
            notes.append("key points missing or excessive")
    else:
        passed += 1

    total += 1
    note_present = "truncated to the first" in summary
    if note_present == truncated:
        passed += 1
    else:
        notes.append("truncation flag inconsistent")

    return round(passed / total, 2), notes


def run(text: str) -> AgentResult:
    """Profile -> plan -> execute -> evaluate (≤1 repair). Returns the result."""
    profile = profile_text(text)
    plan, used_fallback = plan_strategy(profile)
    notes: list[str] = []
    if used_fallback:
        notes.append("planner fallback to deterministic strategy")

    calls = 1  # planning call (or its failed attempt)
    try:
        summary, key_points, num_chunks, truncated, used_calls = _execute(
            plan, text
        )
    except LLMError as exc:
        exc.strategy = plan.strategy  # type: ignore[attr-defined]
        raise
    calls += used_calls

    score, gate_notes = evaluate(
        summary, key_points, text, strategy=plan.strategy, truncated=truncated
    )
    notes.extend(gate_notes)

    repairs = 0
    while score < 1.0 and repairs < settings.agent_max_repairs and gate_notes:
        repairs += 1
        try:
            summary, key_points, num_chunks, truncated, used_calls = _execute(
                plan, text, repair_notes=gate_notes
            )
        except LLMError as exc:
            exc.strategy = plan.strategy  # type: ignore[attr-defined]
            raise
        calls += used_calls
        score, gate_notes = evaluate(
            summary, key_points, text, strategy=plan.strategy, truncated=truncated
        )
        notes.append(f"repair attempt {repairs} applied")
        notes.extend(gate_notes)

    return AgentResult(
        summary=summary,
        key_points=key_points,
        strategy=plan.strategy,
        quality_score=score,
        notes=notes,
        chunks=num_chunks,
        truncated=truncated,
        calls=calls,
    )
