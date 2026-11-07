# Choosing span modes and policies

Choose the language your application needs before tuning optional policies.
The same delimiters mean different things in fact and path mode.

| Need | Oracle/engine setting | vLLM environment | Behavior |
|---|---|---|---|
| Entity mentions and standalone facts | Defaults | Defaults | Entity spans can continue into subject/relation/object fact spans |
| Write intermediate facts and bind successive hops | `derivation=True` | `XGRAPH_DERIVATION=1` | A fact's objects bind subsequent subjects |
| Execute relation paths during decoding | `paths=True` | `XGRAPH_PATHS=1` | Start, then relation-only hops; next span is a bound entity answer |
| Commit ambiguous starts on path closure | `paths=True, committed_paths=True` | Also `XGRAPH_COMMITTED_PATHS=1` | Resolve from the highest-ranked surviving start |

There is no entity-only constructor mode; ordinary mode admits both entity and
fact spans. A bound answer after a path is entity-only. There is no independent
per-request switch between fact and path languages in a single engine.

## Small configuration first

Keep escapes enabled and use the default frontier relation support for grounded
paths. Begin with degree ranking, especially for a custom KG without sitelink
metadata. Inspect ambiguity, exclusions and cap events on representative paths.
Use anchors when your application already has trustworthy entity IDs; anchors
restrict the first span by ID, not by spelling alone.

| Policy | Default | Effect and prerequisite |
|---|---|---|
| `escapes` / `XGRAPH_ESCAPES` | Enabled | Permit unresolved text inside a span |
| `bound_escapes` / `XGRAPH_BOUND_ESCAPES` | Enabled | Permit an anchored/bound subject or answer to escape; setting false requires a writable bound answer |
| `repeats` / `XGRAPH_REPEATS` | Enabled | False blocks repeated fact-label triples; escapes must remain enabled |
| `start_rank` / `XGRAPH_START_RANK` | `degree` | `sitelinks` ranks counts first and needs the optional count array |
| `unquoted_starts` / `XGRAPH_UNQUOTED_STARTS` | False | Exclude quote-leading labels only for unbound starts |
| `end_after_answer` / `XGRAPH_END_AFTER_ANSWER` | False | After a normally closed path answer allow only explicit served-model EOS IDs |
| `escape_bias` / `XGRAPH_ESCAPE_BIAS` | Zero | GPU logit bias for tokens committing to escape; not an oracle membership rule |
| `release_mass`, `release_top` / `XGRAPH_RELEASE_MASS`, `XGRAPH_RELEASE_TOP` | Off | GPU policies steering an incompatible step to escape; require escapes |

Environment booleans use `1`/`0`. These variables configure the vLLM adapter;
calling `Oracle` or `Engine` directly uses Python arguments instead.

## Relation support and advanced interfaces

`relations="frontier"` executes against the current frontier. `"vocabulary"`
allows all writable relation labels; `"successor"` allows relations with some
two-hop witness anywhere in the graph. Those weaker modes are experimental
planning controls: they execute no live path and do not bind an answer. Never
interpret their outputs as grounded paths without separate execution.

Optional frontier observations (`observe`), menus (`menu`), frontier selection,
audit, terminal readback, and residual-goal STOP change the interface or stopping
policy. They have configuration prerequisites; constructor errors describe
invalid combinations. They are experimental and opt in explicitly. See the
[API](../api.md) and [engine contract](../engine-contract.md) for the
current policies and limits.
Do not enable research policies merely to run the basic examples.

Date/value expressions require committed frontier paths, expression-compatible
index syntax, a date table, escapes, and, for value verdicts, the verdict
entities. Their operators are defined in `SpanSyntax.operators`. They are not
general SPARQL, set algebra, qualifier queries, or arbitrary arithmetic.

## Completion policy

The model must choose to open a span unless you use a forced opener or replayed
prefill. `xgraph_open=1` requires a prompt ending in the opener;
`xgraph_prefill` requires a prompt ending in the supplied text and replays it
through the engine. They are mutually exclusive. A declined replay is a failed
constrained request. The generation helpers correlate worker receipts by
`xgraph_replay_id`; missing or invalid confirmation also fails closed while
retaining text and all costs. A decline without a status consumer errors.
See [request settings](vllm.md#request-settings). Inspect output completion and
escape status as well as visible span delimiters.
