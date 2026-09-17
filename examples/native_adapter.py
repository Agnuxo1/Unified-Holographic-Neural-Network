"""Run a native adapter end-to-end on a temporary original demonstration library."""

from __future__ import annotations
import argparse
import asyncio
import inspect
import json
from pathlib import Path
import tempfile
from euhnn import HolographicIndex
from euhnn.sample import load_demo
from euhnn.adapters import make_adapter, SUPPORTED_FRAMEWORKS


def resolve(value):
    if inspect.isawaitable(value):

        async def wait():
            return await value

        return asyncio.run(wait())
    return value


def main():
    options = argparse.ArgumentParser(description=__doc__)
    options.add_argument("framework", choices=SUPPORTED_FRAMEWORKS)
    args = options.parse_args()
    with tempfile.TemporaryDirectory(prefix="euhnn-native-demo-") as temporary:
        path = Path(temporary) / "library.sqlite"
        with HolographicIndex(path, create=True, backend="cpu") as index:
            load_demo(index)
        adapter = make_adapter(args.framework, str(path), top_k=2)
        query = "blue wavelength"
        if args.framework == "langchain":
            results = adapter.invoke(query)
            output = [{"text": item.page_content, **item.metadata} for item in results]
        elif args.framework == "llamaindex":
            output = [{"text": item.node.text, **item.node.metadata} for item in adapter.retrieve(query)]
        elif args.framework == "langgraph":
            output = json.loads(
                adapter.invoke({"query": query}, {"configurable": {"thread_id": "local-example"}})["result"]
            )["hits"]
        elif args.framework == "haystack":
            from haystack import Pipeline

            pipeline = Pipeline()
            pipeline.add_component("retriever", adapter)
            result = pipeline.run({"retriever": {"query": query}})
            output = [{"text": item.content, **item.meta} for item in result["retriever"]["documents"]]
        elif args.framework == "crewai":
            output = json.loads(adapter.run(query=query))["hits"]
        elif args.framework == "agent_framework":
            output = json.loads(resolve(adapter.invoke(arguments={"query": query}, skip_parsing=True)))["hits"]
        elif args.framework == "smolagents":
            output = json.loads(adapter(query=query))["hits"]
        elif args.framework == "agno":
            from agno.tools.function import FunctionCall

            output = json.loads(FunctionCall(function=adapter, arguments={"query": query}).execute().result)["hits"]
        elif args.framework == "autogen":
            from autogen_core import CancellationToken

            output = json.loads(resolve(adapter.run_json({"query": query}, CancellationToken())))["hits"]
        else:
            from pydantic_ai import Agent
            from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
            from pydantic_ai.models.function import FunctionModel

            def local_model(messages, info):
                if any(
                    getattr(part, "part_kind", "") == "tool-return" for message in messages for part in message.parts
                ):
                    return ModelResponse(parts=[TextPart("Local retrieval completed")])
                return ModelResponse(parts=[ToolCallPart("euhnn_search", {"query": query}, tool_call_id="example")])

            completed = Agent(model=FunctionModel(local_model), tools=[adapter]).run_sync(
                "Read the local optical manual"
            )
            values = [
                part.content
                for message in completed.all_messages()
                for part in message.parts
                if getattr(part, "part_kind", "") == "tool-return"
            ]
            output = json.loads(values[0])["hits"]
        assert output and any("0.46" in item["text"] for item in output)
        print(json.dumps({"framework": args.framework, "remote_model_calls": 0, "hits": output}, indent=2))


if __name__ == "__main__":
    main()
