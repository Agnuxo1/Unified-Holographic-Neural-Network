# Native integrations

`make_adapter(framework, index_path, top_k=5, mode="hybrid", backend="cpu")`
binds an existing local library selected by the operator. Only the query is exposed
to a model/tool caller; the caller cannot choose an arbitrary file path through the
retrieval arguments. The returned passages contain exact text, source, page, line
range, character offsets, chunk ID and ranking values.

| Framework argument | Install extra | Native interface |
| --- | --- | --- |
| `langchain` | `langchain` | `BaseRetriever.invoke` / `ainvoke`, returning Documents |
| `langgraph` | `langgraph` | Compiled StateGraph, query/result state and local checkpointing |
| `llamaindex` | `llamaindex` | `BaseRetriever.retrieve` / `aretrieve`, returning NodeWithScore |
| `crewai` | `crewai` | Native tool `.run(query=...)` |
| `haystack` | `haystack` | Serializable retriever component in a native Pipeline |
| `agent_framework` | `agent-framework` | Native Microsoft function tool `.invoke` |
| `smolagents` | `smolagents` | Native Tool subclass, callable with query |
| `agno` | `agno` | Native Function / FunctionCall execution |
| `pydantic_ai` | `pydantic-ai` | Native Tool, exercised inside a local FunctionModel agent loop |
| `autogen` | `autogen` | AutoGen Core FunctionTool / run_json with CancellationToken |

Install a selected extra using `python -m pip install ".[llamaindex]"`, for example.
`.[integrations]` installs all ten declared SDKs. They are optional: importing the
EUHNN core does not initialize them. [Runnable examples](../examples/README.md)
create the library themselves and need no provider account.

[Native execution evidence](../audit/native-integrations.json) records the installed
SDK versions, actual native classes, executed retrieval, exact citation checks and
empty nonmatches. Haystack serialization/restoration and LangGraph checkpoints are
also checked. This is evidence of native execution, not a claim that the upstream
projects have adopted or certified this package, and not an LLM reasoning benchmark.

## Data and configuration boundaries

Passages deliberately enter the caller's memory and may enter its logs, checkpoints
or selected model. Choose only documents that may be exposed to that application.
Document text is untrusted input, not permission to execute instructions. The core
and adapters do not call a provider model by themselves.

A serialized Haystack component includes the operator-selected index path. Restore
only trusted pipeline configurations and allow only the intended adapter module.
Create a new component instance for a second Pipeline; Haystack does not permit the
same component object to belong to two independent pipelines.
