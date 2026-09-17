"""Native PydanticAI Tool without remote inference or implicit model creation."""

from pydantic_ai import Tool
from .common import RetrievalBinding


def make(binding: RetrievalBinding):
    return Tool(binding.function(), name="euhnn_search", takes_ctx=False)
