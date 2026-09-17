"""A serializable native Haystack retriever component returning Documents."""

from __future__ import annotations
from dataclasses import asdict
from haystack import component, Document, default_to_dict, default_from_dict
from .common import RetrievalBinding


@component
class EUHNNRetriever:
    """Use an existing local library inside a Haystack Pipeline.

    Deserializing this component authorizes access to its configured index path.
    Only restore a pipeline configuration supplied by a trusted operator.
    """

    def __init__(self, index_path: str, top_k: int = 5, mode: str = "hybrid", backend: str = "cpu"):
        self.binding = RetrievalBinding(index_path, top_k, mode, backend)

    @component.output_types(documents=list[Document])
    def run(self, query: str):
        documents = []
        for hit in self.binding.retrieve(query):
            metadata = hit.to_dict()
            metadata.pop("text")
            documents.append(Document(id=hit.chunk_id, content=hit.text, score=hit.score, meta=metadata))
        return {"documents": documents}

    def to_dict(self):
        return default_to_dict(self, **asdict(self.binding))

    @classmethod
    def from_dict(cls, data):
        return default_from_dict(cls, data)


def make(binding: RetrievalBinding):
    return EUHNNRetriever(**asdict(binding))
