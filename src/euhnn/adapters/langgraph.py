"""An executable LangGraph retrieval node with optional local checkpoints."""

from __future__ import annotations
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from .common import RetrievalBinding


class RetrievalState(TypedDict, total=False):
    query: str
    result: str


def make(binding: RetrievalBinding):
    """Return a compiled graph; its checkpoints contain retrieved source text."""

    def retrieve(state: RetrievalState) -> RetrievalState:
        return {"result": binding.execute(state["query"])}

    graph = StateGraph(RetrievalState)
    graph.add_node("euhnn_retrieve", retrieve)
    graph.add_edge(START, "euhnn_retrieve")
    graph.add_edge("euhnn_retrieve", END)
    return graph.compile(checkpointer=InMemorySaver())
