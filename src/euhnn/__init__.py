"""EUHNN: ray-traced optical memory and source-grounded document retrieval."""

__version__ = "2.0.0"
from .index import HolographicIndex
from .schema import IndexConfig, SearchHit, TextPage, ValidationError, IndexFormatError

__all__ = [
    "HolographicIndex",
    "IndexConfig",
    "SearchHit",
    "TextPage",
    "ValidationError",
    "IndexFormatError",
    "__version__",
]
