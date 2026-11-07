# Troubleshooting and FAQ

Start with the [CPU examples](../../examples/README.md), then isolate the
smallest graph and token sequence that fails. Include the source commit,
tokenizer fingerprint, syntax, version, options, and full exception when
[reporting an issue](../../CONTRIBUTING.md).

| Symptom | Cause to check | Next step |
|---|---|---|
| Installation starts fetching torch or training packages | Default development group was selected | Use `uv sync --locked --no-default-groups` and `uv run --no-sync` for library use |
| A CPU-only install invokes CUDA | A CUDA compiler was discovered | Use the explicit CPU CMake setting in [getting started](../getting-started.md#install-on-cpu) |
| Source build cannot configure CMake | CMake/toolchain unavailable | Install a C++ compiler and the CMake version required by `CMakeLists.txt`; retain the build log |
| `EngineUnavailable: xgraph was built without its GPU engine` | CPU build or no CUDA compiler at build time | CPU index/oracle tools still work; for generation, rebuild in the GPU worker environment with `nvcc` available |
| `UnsupportedTokenizer` | The tokenizer cannot be mapped to supported BPE token bytes | Use a supported byte-level or byte-fallback BPE `tokenizer.json`; do not substitute another model's tokenizer |
| `IndexMismatch` | Index tokenizer fingerprint differs, or index rows are inconsistent | Rebuild using the exact served tokenizer and syntax; keep builds in separate directories |
| `KGIndexFormatError` | Missing index header/array or unsupported saved format | Supply a saved KG index, rather than a KG source directory, and rebuild an incomplete index |
| `KGFormatError` during in-memory build | A triple references an undefined ID | Define that entity/relation or correct the triple |
| `UnknownId` during a query | Queried entity/relation is absent from the index | Inspect `labels.owner_ids` / `relations.owner_ids` and your source-ID mapping |
| `TokenNotAllowed` during replay | Wrong mode, missing label prefix, tokenization boundary, unsupported spelling, or unavailable relation | Replay the canonical pieces from the examples and inspect the state before the rejected token |
| Desired label/fact is absent but build succeeded | Labels were excluded or triples were unwritable | Inspect both excluded-label lists and `unwritable_triples`; an alias may provide a writable spelling |
| A path answer cannot name the expected entity | Version, start sense, relation direction, or a cap changed reachability | Inspect `MatchedExecutor.execute_set` with matching options and version |
| Version registration succeeds but a pair has no objects | Its replacement object was unwritable | Inspect `VersionStats.unwritable_edits` and the version's `object_rows` |
| Sitelink ranking or endpoint witnesses refuse an index | Required count metadata is absent | Use degree ranking, or supply declared counts/neutral zero counts for a numeric custom graph |
| Witness reconstruction fails on custom string IDs | Current witness tie-breaker assumes numeric-suffix IDs | Use the documented numeric mapping; ordinary oracle/index use still supports arbitrary IDs |
| vLLM cannot load the logits processor | Wrong worker environment, runtime version or model runner | Install xgraph in the worker and explicitly select Model Runner V2 |
| vLLM generates ordinary prose | No opt-in, or model never opened a span | Send `xgraph=1` in `vllm_xargs` and teach span syntax in the prompt |
| A replay is declined (`tail mismatch` or `disallowed`) | Prompt tail differs or the prefill is inadmissible | Treat the request as unconstrained failure; correct the prompt tail/tokenization and replay options |

## Does xgraph need Wikidata?

No. [In-memory graphs](bring-your-own-kg.md#in-memory-triples) accept your own
string IDs. Existing disk loaders use Q/P numbers as a local storage mapping.
The names of modules such as `wikidata.build_index` reflect their origin;
the index, oracle, and engine operate on the graph you supply.

## Does a resolved span mean the answer is true?

It means the span is supported by the indexed KG and execution policy. A graph
can be outdated, incomplete, or wrong; the model can choose an irrelevant
supported path. Free text, escapes and truncated spans are not grounded results.
See [the contract boundary](../concepts.md#guarantees-and-contract).

## Can I generate on CPU, in Transformers, or in SGLang?

The CPU oracle is a small-graph reference, with a vocabulary scan for each mask.
No production CPU backend, Transformers logits processor, or SGLang adapter is
implemented here. The implemented generation integration is
[vLLM on GPU](vllm.md). Training code using Transformers does not constitute a
generation adapter.

## Can I add inverse relations or new labels?

Ordinary execution follows outgoing edges. To traverse inverses, explicitly
add inverse relation IDs, labels and triples before building the index. New
labels and relations require a rebuild. Registered versions replace adjacency
over existing labels; they are not live append-only ingestion.

## Can I enforce entity-only spans or every claim in the answer?

There is no general entity-only mode switch. Ordinary spans can become facts;
path answers are entity-only. The engine allows free text outside spans.
Applications needing all content constrained must define and validate their
completion policy; a span mask alone does not enforce that policy.

## Where are serving and research-launcher failures documented?

[DEVELOPMENT.md](../../DEVELOPMENT.md#troubleshooting) covers shared GPU headroom,
KV cache sizing, container launch diagnostics, quality-gate locking and build
repair. Those operational procedures are separate from CPU onboarding.
