"""Native Agno callable exposing only a query, not a filesystem path."""

from agno.tools.function import Function
from .common import RetrievalBinding


def make(binding: RetrievalBinding):
    return Function.from_callable(binding.function(), name="euhnn_search")
