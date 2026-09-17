"""Lazy native integrations: optional SDKs are loaded only when selected."""

from importlib import import_module
from .common import RetrievalBinding
from ..schema import ValidationError

SUPPORTED_FRAMEWORKS = (
    "langchain",
    "langgraph",
    "llamaindex",
    "crewai",
    "haystack",
    "agent_framework",
    "smolagents",
    "agno",
    "pydantic_ai",
    "autogen",
)


def make_adapter(framework: str, index_path: str, *, top_k: int = 5, mode: str = "hybrid", backend: str = "cpu"):
    """Construct a real native retriever/component/tool bound to one local index."""
    if framework not in SUPPORTED_FRAMEWORKS:
        raise ValidationError("Unknown framework. Choose one of: " + ", ".join(SUPPORTED_FRAMEWORKS))
    binding = RetrievalBinding(index_path, top_k, mode, backend)
    module = import_module("." + framework, __name__)
    return module.make(binding)


__all__ = ["make_adapter", "SUPPORTED_FRAMEWORKS", "RetrievalBinding"]
