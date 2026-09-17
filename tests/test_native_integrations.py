"""Execute native third-party APIs against a real, persistent synthetic index."""

from __future__ import annotations
import asyncio
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import uuid
import pytest

from euhnn import HolographicIndex, IndexConfig
from euhnn.adapters import make_adapter, SUPPORTED_FRAMEWORKS

pytestmark = pytest.mark.integrations
PACKAGES = {
    "langchain": "langchain-core",
    "langgraph": "langgraph",
    "llamaindex": "llama-index-core",
    "crewai": "crewai",
    "haystack": "haystack-ai",
    "agent_framework": "agent-framework-core",
    "smolagents": "smolagents",
    "agno": "agno",
    "pydantic_ai": "pydantic-ai-slim",
    "autogen": "autogen-core",
}
MODULES = {
    **{k: k for k in PACKAGES},
    "langchain": "langchain_core",
    "llamaindex": "llama_index.core",
    "autogen": "autogen_core",
}
SOURCE = """# Optical integration test
The blue wavelength is 0.46 simulation units.
A reversible holographic memory uses RGB phase channels and a Fourier transform.
Every retrieved quotation must retain the original source identifier and line range.
"""


@pytest.fixture(scope="module")
def native_library(tmp_path_factory):
    directory = tmp_path_factory.mktemp("native-library")
    path = directory / "library.sqlite"
    config = IndexConfig(source_count=24, detector_count=32, sphere_count=8, chunk_words=64, overlap_words=8)
    with HolographicIndex(path, create=True, config=config, backend="cpu") as index:
        index.ingest_text(SOURCE, source="synthetic-optical-guide.md", metadata={"fixture": True})
        index.ingest_text(
            "The steam locomotive inspection interval is 120 operating hours.", source="synthetic-railway.md"
        )
    evidence = []
    yield path, evidence
    target = os.environ.get("EUHNN_NATIVE_EVIDENCE")
    if target:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text(
            json.dumps(
                {
                    "version": "2.0.0",
                    "synthetic_data_only": True,
                    "native_framework_execution": True,
                    "remote_model_calls": 0,
                    "upstream_adoption_claimed": False,
                    "frameworks": evidence,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def awaited(value):
    if inspect.isawaitable(value):

        async def run():
            return await value

        return asyncio.run(run())
    return value


def invoke(framework, adapter, query):
    """Use each upstream framework's actual invocation/serialization interfaces."""
    state_checked = False
    if framework == "langchain":
        documents = adapter.invoke(query)
        assert [d.page_content for d in adapter.invoke(query)] == [
            d.page_content for d in awaited(adapter.ainvoke(query))
        ]
        hits = [{"text": d.page_content, **d.metadata} for d in documents]
        assert all(d.id == h["chunk_id"] for d, h in zip(documents, hits))
    elif framework == "llamaindex":
        nodes = adapter.retrieve(query)
        assert [n.node.text for n in nodes] == [n.node.text for n in awaited(adapter.aretrieve(query))]
        hits = [{"text": n.node.text, "score": n.score, **n.node.metadata} for n in nodes]
    elif framework == "langgraph":
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        value = adapter.invoke({"query": query}, config=config)["result"]
        assert adapter.get_state(config).values["result"] == value
        assert list(adapter.get_state_history(config))
        state_checked = True
        hits = json.loads(value)["hits"]
    elif framework == "haystack":
        from haystack import Pipeline

        pipeline = Pipeline()
        pipeline.add_component("retriever", adapter)
        serialized = pipeline.dumps()
        restored = Pipeline.loads(serialized, allowed_modules=["euhnn.adapters.haystack"])
        documents = restored.run({"retriever": {"query": query}})["retriever"]["documents"]
        state_checked = True
        hits = [{"text": d.content, **d.meta} for d in documents]
    elif framework == "crewai":
        hits = json.loads(adapter.run(query=query))["hits"]
    elif framework == "agent_framework":
        hits = json.loads(awaited(adapter.invoke(arguments={"query": query}, skip_parsing=True)))["hits"]
    elif framework == "smolagents":
        hits = json.loads(adapter(query=query))["hits"]
    elif framework == "agno":
        from agno.tools.function import FunctionCall

        result = FunctionCall(function=adapter, arguments={"query": query}).execute()
        hits = json.loads(result.result)["hits"]
    elif framework == "autogen":
        from autogen_core import CancellationToken

        hits = json.loads(awaited(adapter.run_json({"query": query}, CancellationToken())))["hits"]
    elif framework == "pydantic_ai":
        from pydantic_ai import Agent
        from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
        from pydantic_ai.models.function import FunctionModel

        def local_model(messages, info):
            if any(getattr(p, "part_kind", "") == "tool-return" for m in messages for p in m.parts):
                return ModelResponse(parts=[TextPart("Synthetic retrieval completed")])
            return ModelResponse(parts=[ToolCallPart("euhnn_search", {"query": query}, tool_call_id="fixture-query")])

        run = Agent(model=FunctionModel(local_model), tools=[adapter]).run_sync(
            "Retrieve from the selected local library"
        )
        values = [
            p.content for m in run.all_messages() for p in m.parts if getattr(p, "part_kind", "") == "tool-return"
        ]
        assert len(values) == 1
        state_checked = True
        hits = json.loads(values[0])["hits"]
    else:
        raise AssertionError("Missing native invocation")
    return hits, state_checked


@pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
def test_native_retrieval_keeps_quotes_citations_and_returns_empty_on_no_match(framework, native_library):
    pytest.importorskip(MODULES[framework])
    path, evidence = native_library
    adapter = make_adapter(framework, str(path), top_k=2)
    hits, state_checked = invoke(framework, adapter, "blue wavelength")
    assert hits
    for hit in hits:
        assert hit["source"] == "synthetic-optical-guide.md"
        assert hit["text"] == SOURCE[hit["start"] : hit["end"]]
        assert hit["line_start"] == 1 + SOURCE.count("\n", 0, hit["start"])
        assert hit["source"] in hit["citation"] and hit["chunk_id"] in hit["citation"]
    missing, _ = invoke(framework, make_adapter(framework, str(path), top_k=2), "absentxyzterm")
    assert missing == []
    evidence.append(
        {
            "framework": framework,
            "package": PACKAGES[framework],
            "version": version(PACKAGES[framework]),
            "native_class": type(adapter).__module__ + "." + type(adapter).__name__,
            "retrieval_executed": True,
            "exact_source_and_citation_verified": True,
            "nonmatch_returns_empty": True,
            "native_state_or_pipeline_serialization_checked": state_checked,
        }
    )
