"""Unicode tokenization and bounded-window chunking without losing provenance."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterator
import re
import unicodedata

from .schema import TextChunk, TextPage, ValidationError, bounded_int, bounded_text

# This tokenizer targets whitespace-delimited languages and source documentation.
# It is not a language model or a specialist CJK word segmenter.
WORD = re.compile(r"[^\W_]+(?:['\u2019-][^\W_]+)*", re.UNICODE)
STOP_WORDS = frozenset(
    """
a an and are as at be been being but by can could did do does for from had has have
he her here him his how i if in into is it its me my not of on or our she should
so than that the their them then there these they this those to was we were what
when where which who why will with would you your
al algo como con cual cuando de del desde donde el ella en entre es esa ese eso
esta este esto fue ha hay la las le les lo los me mi no nos o para pero por que
se si sin sobre son su sus te tu un una uno unos unas y ya
""".split()
)


def normalize_word(word: str) -> str:
    """Case-fold and remove combining accents, without changing source text."""
    return "".join(c for c in unicodedata.normalize("NFKD", word.casefold()) if not unicodedata.combining(c))


def words(text: str, *, remove_stop: bool = False, limit: int | None = None) -> list[str]:
    """Extract bounded terms; excessively long tokens are excluded, not truncated."""
    result: list[str] = []
    for match in WORD.finditer(text):
        token = normalize_word(match.group())
        if len(token) > 128 or (remove_stop and token in STOP_WORDS):
            continue
        result.append(token)
        if limit is not None and len(result) >= limit:
            break
    return result


def search_terms(query: str) -> list[str]:
    """Return distinct literal terms; no user-provided FTS syntax is executed."""
    bounded_text(query, "query", 4096, empty=True)
    terms = words(query, remove_stop=True, limit=64)
    return list(dict.fromkeys(terms))


def fts_query(query: str, *, phrase: bool = False) -> str | None:
    """Quote every token and use only application-generated FTS operators."""
    if type(phrase) is not bool:
        raise ValidationError("phrase must be a boolean.")
    bounded_text(query, "query", 4096, empty=True)
    if phrase:
        if any(len(normalize_word(match.group())) > 128 for match in WORD.finditer(query)):
            raise ValidationError("A phrase token exceeds the 128-character limit.")
        terms = words(query, limit=65)
        if len(terms) > 64:
            raise ValidationError("Exact phrase queries support at most 64 words; the phrase was not truncated.")
    else:
        terms = search_terms(query)
    if not terms:
        return None
    if phrase:
        return '"' + " ".join(terms).replace('"', '""') + '"'
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)


def feature_counts(text: str) -> dict[str, float]:
    """Stable lexical/subword features used as the optical source illumination.

    Word features carry most of the energy. Character trigrams support spelling
    variation; low-weight adjacent bigrams retain limited sequence information.
    These are deterministic features, not pretrained semantic embeddings.
    """
    tokens = words(text, remove_stop=True, limit=4096)
    counts = Counter(tokens)
    features: dict[str, float] = {}
    import math

    for token, count in counts.items():
        weight = 1.0 + math.log(count)
        features["w:" + token] = weight
        padded = "^" + token + "$"
        grams = set(padded[i : i + 3] for i in range(max(0, len(padded) - 2)))
        scale = 0.24 * weight / max(1, len(grams)) ** 0.5
        for gram in grams:
            key = "g:" + gram
            features[key] = features.get(key, 0.0) + scale
    for left, right in zip(tokens, tokens[1:]):
        key = "b:" + left + " " + right
        features[key] = features.get(key, 0.0) + 0.12
    return features


def iter_chunks(page: TextPage, chunk_words: int = 192, overlap_words: int = 32) -> Iterator[TextChunk]:
    """Yield original text slices using a bounded look-ahead token window.

    Memory is O(chunk_words), apart from the page text. Every character appears
    in at least one slice, including separators and final punctuation. Adjacent
    chunks overlap by the configured number of words, never by fabricated text.
    """
    bounded_int(chunk_words, "chunk_words", 16, 1024)
    bounded_int(overlap_words, "overlap_words", 0, chunk_words - 1)
    if not isinstance(page, TextPage) or not isinstance(page.text, str):
        raise ValidationError("Expected a TextPage containing text.")
    if page.page is not None:
        bounded_int(page.page, "page", 1, 1_000_000)
    window: deque[tuple[int, int, int]] = deque()
    matches = iter(WORD.finditer(page.text))
    previous_end, line, ordinal = 0, 1, 0
    cursor_start = 0
    for match in matches:
        line += page.text.count("\n", previous_end, match.start())
        previous_end = match.end()
        window.append((match.start(), match.end(), line))
        if len(window) <= chunk_words:
            continue
        # One look-ahead token proves that the previous full window is not final.
        next_start = window[-1][0]
        start = 0 if ordinal == 0 else cursor_start
        first_line = 1 if ordinal == 0 else window[0][2]
        end = next_start
        yield TextChunk(
            page.text[start:end],
            start,
            end,
            first_line,
            first_line + page.text.count("\n", start, end),
            page.page,
            ordinal,
        )
        ordinal += 1
        for _ in range(chunk_words - overlap_words):
            window.popleft()
        cursor_start = window[0][0] if overlap_words else next_start
    if window:
        start = 0 if ordinal == 0 else cursor_start
        first_line = 1 if ordinal == 0 else window[0][2]
        end = len(page.text)
        yield TextChunk(
            page.text[start:end],
            start,
            end,
            first_line,
            first_line + page.text.count("\n", start, end),
            page.page,
            ordinal,
        )
    elif page.text.strip():
        yield TextChunk(page.text, 0, len(page.text), 1, 1 + page.text.count("\n"), page.page, 0)
