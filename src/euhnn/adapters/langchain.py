"""Native LangChain retriever that returns Document objects with provenance."""

from __future__ import annotations
import asyncio
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import PrivateAttr
from .common import RetrievalBinding


class EUHNNRetriever(BaseRetriever):
    """Retrieve exact passages without an embedding API or a remote model."""

    _binding: RetrievalBinding = PrivateAttr()

    def __init__(self, binding: RetrievalBinding, **kwargs):
        super().__init__(**kwargs)
        self._binding = binding

    def _get_relevant_documents(self, query: str, *, run_manager):
        documents = []
        for hit in self._binding.retrieve(query):
            metadata = hit.to_dict()
            metadata.pop("text")
            documents.append(Document(page_content=hit.text, metadata=metadata, id=hit.chunk_id))
        return documents

    async def _aget_relevant_documents(self, query: str, *, run_manager):
        return await asyncio.to_thread(self._get_relevant_documents, query, run_manager=run_manager)


def make(binding: RetrievalBinding):
    return EUHNNRetriever(binding)
