"""Exact source-span and Unicode regressions, including boundary-sized inputs."""

import random
import pytest
from euhnn.schema import TextPage, ValidationError, IndexConfig
from euhnn.text import words, search_terms, fts_query, iter_chunks


def test_unicode_terms_preserve_accents_and_non_latin_words():
    assert words("Memoria \u00f3ptica, acci\u00f3n y \u6f22\u5b57") == [
        "memoria",
        "optica",
        "accion",
        "y",
        "\u6f22\u5b57",
    ]
    assert search_terms("la memoria \u00f3ptica") == ["memoria", "optica"]


@pytest.mark.parametrize("count", [0, 1, 15, 16, 17, 31, 32, 33, 64, 1001])
@pytest.mark.parametrize("overlap", [0, 1, 5, 15])
def test_all_characters_and_source_locations_survive_chunking(count, overlap):
    rng = random.Random(count)
    text = "  \n" + "".join(f"word{i}" + rng.choice([" ", "\n", ", ", "\n\n"]) for i in range(count)) + "!\n"
    chunks = list(iter_chunks(TextPage(text, 3), 16, overlap))
    covered = bytearray(len(text))
    previous = -1
    for ordinal, c in enumerate(chunks):
        assert c.ordinal == ordinal
        assert c.text == text[c.start : c.end]
        assert c.start > previous
        assert c.page == 3
        assert c.line_start == 1 + text.count("\n", 0, c.start)
        assert c.line_end == c.line_start + c.text.count("\n")
        covered[c.start : c.end] = b"\x01" * (c.end - c.start)
        previous = c.start
    assert all(covered)
    if count <= 16:
        assert len(chunks) == 1


def test_empty_whitespace_has_no_chunks():
    assert list(iter_chunks(TextPage(" \n\t"))) == []


@pytest.mark.parametrize("query", ['" OR 1=1 --', "a NEAR(b, c)", "{abc}:def", "*", "_", "the and of", "\u0000"])
def test_query_operators_are_not_executed(query):
    if "\u0000" in query:
        with pytest.raises(ValidationError):
            fts_query(query)
    else:
        result = fts_query(query)
        assert result is None or result.startswith('"')


@pytest.mark.parametrize(
    "values",
    [
        {"source_count": True},
        {"detector_count": 0},
        {"seed": -1},
        {"overlap_words": 192},
        {"chunk_words": 8},
        {"wavelengths": (0, 1, 2)},
        {"format_version": True},
    ],
)
def test_invalid_configuration(values):
    with pytest.raises(ValidationError):
        IndexConfig(**values)


def test_configuration_round_trip():
    value = IndexConfig()
    assert IndexConfig.from_dict(value.to_dict()) == value
    with pytest.raises(ValidationError):
        IndexConfig.from_dict({**value.to_dict(), "unknown": 1})


def test_exact_phrase_is_never_silently_truncated():
    with pytest.raises(ValidationError):
        fts_query(" ".join(f"word{i}" for i in range(65)), phrase=True)
    with pytest.raises(ValidationError):
        fts_query("x" * 129, phrase=True)
