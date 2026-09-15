"""Unit tests for text cleaning and chunking (pure functions)."""

from backend.textutil import clean_text, split_chunks


def test_clean_collapses_whitespace_and_drops_empties() -> None:
    raw = "  Hello   world  \r\n\r\n\n  second   line\t\twith tabs  \n\n\n"
    assert clean_text(raw) == "Hello world\n\nsecond line with tabs"


def test_clean_empty() -> None:
    assert clean_text("") == ""
    assert clean_text("   \n \t \n  ") == ""


def test_split_short_text_single_chunk() -> None:
    assert split_chunks("hello", chunk_chars=4000, overlap=200) == ["hello"]


def test_split_empty() -> None:
    assert split_chunks("", chunk_chars=4000, overlap=200) == []


def test_split_long_text_respects_size_and_overlap() -> None:
    words = [f"word{i}" for i in range(2000)]
    text = " ".join(words)
    chunks = split_chunks(text, chunk_chars=4000, overlap=200)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 4000
    # Overlap: consecutive chunks share text.
    assert chunks[0][-100:] in chunks[1][:500]
    # No mid-word cuts: chunks join back to the original words.
    rejoined_words = " ".join(
        w for chunk in chunks for w in chunk.split(" ")
    ).split(" ")
    seen: list[str] = []
    for word in rejoined_words:
        if not seen or seen[-1] != word or word not in seen[-3:]:
            seen.append(word)
    for word in words:
        assert word in seen


def test_split_no_spaces_hard_cut() -> None:
    text = "A" * 9000
    chunks = split_chunks(text, chunk_chars=4000, overlap=200)
    assert len(chunks) == 3
    for chunk in chunks:
        assert len(chunk) <= 4000
    # Overlap duplicates content: strip it to reconstruct the original.
    assert chunks[0][-200:] == chunks[1][:200]
    assert chunks[1][-200:] == chunks[2][:200]
    assert chunks[0] + chunks[1][200:] + chunks[2][200:] == text
