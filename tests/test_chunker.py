import random

import pytest

from docuquery.chunker import overlap_chunked_texts, split_text
from docuquery.chunker import recursive_semantic_chunking as chunk

SEPARATORS = ["\n\n", "\n", ".", "!", "?", ",", " "]


def _without_separators(text):
    """The characters of a text, without the separators the chunker may drop."""
    return "".join(c for c in text if c not in "\n., !?")


def test_short_text_is_one_chunk():
    assert chunk("hello", 512, 50) == ["hello"]


def test_text_at_the_limit_is_one_chunk():
    assert chunk("x" * 512, 512, 50) == ["x" * 512]


def test_no_text_is_lost_without_separators():
    chunks = chunk("x" * 2000, 512, 0)
    assert "".join(chunks) == "x" * 2000


def test_text_without_separators_is_cut_at_the_limit():
    assert [len(c) for c in split_text("x" * 2000, 512, SEPARATORS)] == [
        512,
        512,
        512,
        464,
    ]


def test_no_chunk_is_over_the_limit():
    text = ("word " * 200) + "\n\n" + ("y" * 1500)
    assert max(len(c) for c in chunk(text, 512, 50)) <= 512


def test_long_piece_is_split_again_with_the_next_separator():
    # One paragraph that is too long, but has sentences inside it
    paragraph = ". ".join(["This is a sentence"] * 60)
    chunks = split_text(paragraph, 100, SEPARATORS)
    assert len(chunks) > 1
    assert max(len(c) for c in chunks) <= 100
    assert _without_separators("".join(chunks)) == _without_separators(paragraph)


def test_trailing_newline_does_not_make_an_oversized_chunk():
    # Regression: "x" * 2000 + "\n" once gave one 2000-character chunk
    chunks = split_text("x" * 2000 + "\n", 512, SEPARATORS)
    assert max(len(c) for c in chunks) <= 512
    assert "".join(chunks) == "x" * 2000


def test_paragraphs_are_kept_together_when_they_fit():
    text = "First paragraph.\n\nSecond paragraph.\n\n" + "z" * 600
    chunks = split_text(text, 512, SEPARATORS)
    assert chunks[0] == "First paragraph.\n\nSecond paragraph."


def test_split_never_returns_empty_chunks():
    text = "a\n\n\n\n" + "b" * 600 + "\n\n\n\nc"
    assert "" not in split_text(text, 100, SEPARATORS)


@pytest.mark.parametrize("seed", range(5))
def test_random_text_never_goes_over_the_limit_or_loses_text(seed):
    rng = random.Random(seed)
    text = "".join(rng.choice("ab .,\n") for _ in range(rng.randint(0, 3000)))
    chunks = split_text(text, 100, SEPARATORS)
    assert max((len(c) for c in chunks), default=0) <= 100
    assert _without_separators("".join(chunks)) == _without_separators(text)


def test_overlap_adds_the_end_of_the_previous_chunk():
    assert overlap_chunked_texts(["aaaa", "bbbb"], 2, 10) == ["aaaa", "aabbbb"]


def test_overlap_is_skipped_when_it_would_pass_the_limit():
    assert overlap_chunked_texts(["aaaa", "bbbb"], 2, 5) == ["aaaa", "bbbb"]


def test_zero_overlap_does_not_change_the_chunks():
    # Regression: [-0:] is the whole previous chunk, which doubled every chunk
    chunks = ["a" * 512, "b" * 512, "c" * 100]
    assert overlap_chunked_texts(chunks, 0, 512) == chunks


def test_overlapped_chunks_stay_within_the_limit():
    text = ". ".join(["Sentence number"] * 200)
    assert max(len(c) for c in chunk(text, 200, 50)) <= 200
