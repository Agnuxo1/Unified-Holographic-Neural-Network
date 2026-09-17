"""Native Microsoft Agent Framework function tool."""

from agent_framework import tool
from .common import RetrievalBinding


def make(binding: RetrievalBinding):
    return tool(binding.function(), name="euhnn_search")
