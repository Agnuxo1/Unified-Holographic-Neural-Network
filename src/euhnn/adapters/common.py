"""A read-only retrieval binding shared by optional native framework adapters."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
from ..index import HolographicIndex
from ..schema import SearchHit, ValidationError, bounded_int


@dataclass(frozen=True)
class RetrievalBinding:
    """Bind an operator-selected local index; model arguments cannot change it.

    Passage text is deliberately returned to the calling application. It may
    contain untrusted instructions or sensitive source data. The caller chooses
    which documents may enter its model and applies its own prompt/data policy.
    """

    index_path: str
    top_k: int = 5
    mode: str = "hybrid"
    backend: str = "cpu"

    def __post_init__(self) -> None:
        path = Path(self.index_path).expanduser().resolve()
        if not path.is_file():
            raise ValidationError("The adapter requires an existing local EUHNN index.")
        bounded_int(self.top_k, "top_k", 1, 100)
        if self.mode not in {"hybrid", "lexical", "optical"} or self.backend not in {"cpu", "cuda", "auto"}:
            raise ValidationError("Invalid adapter retrieval mode or backend.")
        object.__setattr__(self, "index_path", str(path))

    def retrieve(self, query: str) -> list[SearchHit]:
        """Open the existing index, retrieve exact passages and release resources."""
        with HolographicIndex(self.index_path, backend=self.backend) as index:
            return index.search(query, top_k=self.top_k, mode=self.mode)

    def execute(self, query: str) -> str:
        """Return citation-bearing JSON, including an explicit empty result list."""
        return json.dumps(
            {
                "hits": [hit.to_dict() for hit in self.retrieve(query)],
                "kind": "retrieved_source_passages",
                "generated_answer": False,
            },
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

    def function(self):
        """Build a narrowly scoped native function with an explicit input schema."""
        binding = self

        def euhnn_search(query: str) -> str:
            """Retrieve original local document passages with source citations for a query.

            Args:
                query: Terms or a question to match against the operator-selected library.
            """
            return binding.execute(query)

        return euhnn_search
