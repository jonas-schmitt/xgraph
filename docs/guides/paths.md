# Path mode and execution

Start with the [quick start](../getting-started.md). Ordinary mode constrains
entity and fact spans. In path mode the model writes a start entity and relation
labels, and the KG executes each relation while it is written:

`⟦ Atlas · creator · born in⟧` reaches Cedar in the example graph. The next
span, `Answer: ⟦ Cedar⟧`, is an entity answer bound to that frontier. The model
does not write the intermediate entity Ada.

## Execute and bind on CPU

The complete [toy example](../../examples/toy_kg.py) creates the index and
tokenizer before the following block:

```{literalinclude} ../../examples/toy_kg.py
:language: python
:start-at:     paths = Oracle(
:end-at:     answer = walk(
```

The example tests that the path and answer resolve to `place:cedar`, and that
the answer `Delta` raises `TokenNotAllowed`. `paths=True` selects the language;
`committed_paths=True` resolves to the best-ranked surviving start's endpoints.
These settings must match in the oracle, GPU engine and offline executor.

For an anchored request, use `anchored_state` with IDs from your index. An
explicit anchor with no writable rows is rejected. Validate all supplied IDs
before serving; the [vLLM guide](vllm.md#request-settings) describes the stricter
whole-request check.

## Inspect the execution

Read `state.last_span.entity_ids` when the path closes. A path close records
endpoints, while a fact close records matching triples. `last_span` retains the
previous result until another span closes, so an open or token-limited span
does not produce a new resolved result.

For offline execution, use `MatchedExecutor` on the same index and tokenizer.
`execute` returns a `MatchedPath`; `execute_set` reports deliverable endpoints,
an uncapped fixed-plan diagnostic and cap events. The [witness tutorial](witnesses.md)
shows a complete executable call. Match anchors, versions, ranking and any
selections to the original request. Offline execution does not reproduce the
model's probability of choosing that plan or its answer spelling.

## Limits and policies

Frontiers and answer rows are capped, so execution can omit valid uncapped
answers. A relation label can denote several relation IDs; execution follows
their union. An escaped path closes unresolved. The weaker `vocabulary` and
`successor` relation supports execute no path and do not bind answers.

See [guarantees](../concepts.md#guarantees-and-contract) and
[span modes and policies](span-modes.md) for derivation mode, escapes, observations,
menus and stopping behavior.
