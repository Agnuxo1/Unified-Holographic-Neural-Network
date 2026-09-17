"""Validated, serializable public types shared by the library, CLI and adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import math


class EUHNNError(Exception):
    """An actionable application error, distinct from an empty search result."""


class ValidationError(EUHNNError, ValueError):
    """Input failed a documented size, shape or schema requirement."""


class IndexFormatError(EUHNNError):
    """The index is unsupported or damaged; it must not be silently recreated."""


def bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    """Validate integers without accepting booleans, floats or numeric strings."""
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValidationError(f"{name} must be an integer from {minimum} to {maximum}.")
    return value


def bounded_text(value: Any, name: str, maximum: int, *, empty: bool = False) -> str:
    """Validate a UTF-8-encodable string while retaining its original characters."""
    if not isinstance(value, str) or len(value) > maximum or (not empty and not value.strip()):
        raise ValidationError(f"{name} must contain {'0' if empty else '1'}-{maximum} characters.")
    if "\x00" in value:
        raise ValidationError(f"{name} must not contain NUL characters.")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ValidationError(f"{name} contains an invalid Unicode surrogate.") from exc
    return value


@dataclass(frozen=True)
class IndexConfig:
    """Immutable feature/chunking contract stored with each index.

    Geometry distances and RGB wavelengths use the same simulation length unit.
    This is a classical straight-ray phase-delay model, not quantum hardware.
    """

    format_version: int = 2
    source_count: int = 96
    detector_count: int = 128
    sphere_count: int = 18
    seed: int = 20240917
    chunk_words: int = 192
    overlap_words: int = 32
    wavelengths: tuple[float, float, float] = (0.63, 0.53, 0.46)

    def __post_init__(self) -> None:
        if type(self.format_version) is not int or self.format_version != 2:
            raise ValidationError("Only index format 2 is supported.")
        bounded_int(self.source_count, "source_count", 8, 512)
        bounded_int(self.detector_count, "detector_count", 8, 512)
        bounded_int(self.sphere_count, "sphere_count", 0, 128)
        bounded_int(self.seed, "seed", 0, 2**32 - 1)
        bounded_int(self.chunk_words, "chunk_words", 16, 1024)
        bounded_int(self.overlap_words, "overlap_words", 0, self.chunk_words - 1)
        if (
            not isinstance(self.wavelengths, (tuple, list))
            or len(self.wavelengths) != 3
            or any(type(x) not in (int, float) or not math.isfinite(x) or not 0.05 <= x <= 5 for x in self.wavelengths)
        ):
            raise ValidationError("wavelengths must contain three finite values from 0.05 to 5.")
        object.__setattr__(self, "wavelengths", tuple(float(x) for x in self.wavelengths))

    @property
    def dimensions(self) -> int:
        """Real, imaginary and centered intensity features for three colors."""
        return 9 * self.detector_count

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexConfig":
        if not isinstance(data, dict) or set(data) != set(cls.__dataclass_fields__):
            raise ValidationError("The stored index configuration has an invalid schema.")
        return cls(**data)


@dataclass(frozen=True)
class TextPage:
    """One extracted PDF page or one ordinary text source, with stable offsets."""

    text: str
    page: int | None = None


@dataclass(frozen=True)
class TextChunk:
    """A slice of a TextPage; start/end are Python Unicode-character offsets."""

    text: str
    start: int
    end: int
    line_start: int
    line_end: int
    page: int | None
    ordinal: int


@dataclass(frozen=True)
class SearchHit:
    """A retrieved passage with sufficient provenance to verify its quotation."""

    chunk_id: str
    document_id: str
    source: str
    title: str
    text: str
    page: int | None
    start: int
    end: int
    line_start: int
    line_end: int
    score: float
    lexical_score: float = 0.0
    optical_score: float = 0.0
    learned_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        location = f"page {self.page}, " if self.page is not None else ""
        return f"{self.source}: {location}lines {self.line_start}-{self.line_end} [{self.chunk_id}]"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["citation"] = self.citation
        return data
