# Quick start

Try xgraph on a tiny knowledge graph (KG): `Atlas --creator--> Ada`.
xgraph checks text inside markers, called a *span*, as each token is written.
For `⟦ Atlas · creator · Ada⟧`, the names and fact must exist in the graph.
Text outside spans stays free.

Complete the [CPU installation](installation.md#cpu), then run this example
from the repository root. It checks a supplied fact one token at a time,
without a dataset, network, model weights or GPU.

## Check your first fact

```{literalinclude} ../examples/quick_start.py
:language: python
```

The result is `(('atlas', 'creator', 'ada'),)`: the fact as IDs in our KG.
The byte-level tokenizer above is a demonstration vocabulary. When serving an
LLM, use `ServedTokenizer.from_file` with that model's local `tokenizer.json`.
Build the index using that exact tokenizer and the chosen span syntax.

Each entity has a stable ID and names, called *labels*; each fact is a *triple*
of entity, relation and entity IDs. Labels are what the model writes.
`build_kg_index` checks the graph and compiles its labels and connections into
token lookup tables, called a *KG index*. The CPU checker, called the *oracle*,
starts a new state for each output sequence. `advance` returns the next state
and rejects a disallowed token.

The example encodes delimiters and prefixed labels as separate pieces, keeping
their boundaries consistent with the index. A generated token is consumed
directly; the sampling loop does not re-tokenize completed text. Read
[tokenization and the index](concepts.md#tokenization-and-the-kg-index) before
replaying text from another source.

## More spans and a saved index

```bash
uv run --no-sync python examples/toy_kg.py --out target/examples/toy-index
```

This [complete CPU example](../examples/toy_kg.py) saves and reloads an index,
resolves entity and fact spans, executes a path, binds its answer, and rejects
an answer outside the reached frontier. It also checks unresolved escapes and
invalid anchors. Its labels and IDs are local to the example.

Continue with [path mode and execution](guides/paths.md),
[versions and edits](guides/kg-versions.md), or [your own KG](guides/bring-your-own-kg.md).

## First constrained generation

The CPU example validates the library workflow by replaying known tokens.
LLM generation uses the [vLLM tutorial](guides/generation.md) and a CUDA build.
Prompts teach span syntax and your graph's labels; the engine then applies the
KG constraints inside the sampling loop.

## Install on CPU

Installation commands and optional dependencies are on the
[installation page](installation.md#cpu).
