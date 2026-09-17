"""Native CrewAI tool for citation-bearing local passage retrieval."""

from crewai.tools import tool
from .common import RetrievalBinding


def make(binding: RetrievalBinding):
    return tool("euhnn_search")(binding.function())
