# Python API reference

Signatures and descriptions below are generated from the current source's
public docstrings at site build time. These interfaces are experimental; there
is no compatibility-stable release promise. The core workflow is available
through `import xgraph as xg`; the original module imports remain available.

| Task | Public entry points |
|---|---|
| Define a graph | `Entity`, `Relation`, `KnowledgeGraph` |
| Bind a tokenizer and language | `ServedTokenizer`, `SpanSyntax` |
| Compile and persist | `build_kg_index`, `save_kg_index`, `load_kg_index` |
| Resolve on CPU | `Oracle`, `OracleState`, `ClosedSpan` |
| Constrain on GPU | `Engine`; vLLM registers `xgraph.vllm:XGraphLogitsProcessor` |

```{toctree}
:maxdepth: 1

api/core
api/index
api/runtime
api/versions
api/rdf
api/execution
```

Research orchestration, benchmark loaders and study-specific command modules
retain their own source docstrings. Historical study records are outside this public profile; see [reproduction scope](reproducing.md).
They are not prerequisites for the library workflow.
