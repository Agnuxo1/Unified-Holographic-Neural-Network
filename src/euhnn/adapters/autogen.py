"""Native AutoGen Core function tool returning exact cited passages."""

from autogen_core.tools import FunctionTool
from .common import RetrievalBinding


def make(binding: RetrievalBinding):
    return FunctionTool(
        binding.function(), description="Retrieve original local passages with source citations.", name="euhnn_search"
    )
