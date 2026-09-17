# Runnable examples

After installing the selected extra, run a complete temporary-library example:

```bash
python -m pip install ".[workbench,llamaindex]"
python examples/native_adapter.py llamaindex
```

Replace the framework argument with one listed by `--help`; install its matching
extra from the integration guide. The example creates its own original demo corpus,
executes the real native API, prints source passages and citations, and cleans up its
temporary index. It needs no API keys or manual source edits. For all ten examples,
install `.[workbench,integrations]`.

The PydanticAI example uses a deterministic local FunctionModel to exercise a real
tool-call loop. This is not remote LLM inference or a language-model quality benchmark.
