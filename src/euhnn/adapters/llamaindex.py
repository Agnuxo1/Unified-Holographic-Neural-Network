"""Native LlamaIndex BaseRetriever using EUHNN source-grounded passage scores."""

from __future__ import annotations
import asyncio
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from .common import RetrievalBinding


class EUHNNRetriever(BaseRetriever):
    """Expose optical/hybrid retrieval directly to LlamaIndex query engines."""

    def __init__(self, binding: RetrievalBinding):
        super().__init__()
        self._binding = binding

    def _retrieve(self, query_bundle):
        nodes = []
        for hit in self._binding.retrieve(query_bundle.query_str):
            metadata = hit.to_dict()
            metadata.pop("text")
            metadata.pop("score")
            node = TextNode(text=hit.text, id_=hit.chunk_id, metadata=metadata)
            nodes.append(NodeWithScore(node=node, score=hit.score))
        return nodes

    async def _aretrieve(self, query_bundle):
        return await asyncio.to_thread(self._retrieve, query_bundle)


def make(binding: RetrievalBinding):
    return EUHNNRetriever(binding)
