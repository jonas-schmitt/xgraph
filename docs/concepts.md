# Concepts

xgraph takes a knowledge graph, a served tokenizer, and a span syntax. It
compiles labels and adjacency into a KG index, then computes which vocabulary
tokens can continue a span. The [oracle](../src/xgraph/oracle.py) specifies the
language on CPU; the [engine](../src/xgraph/engine.py) implements it on GPU.

Source triples and labels → tokenizer-bound KG index → per-request advance and
mask → resolved spans. The GPU engine implements the sampling-loop contract;
the CPU oracle provides an independent reference for the same span language.

## Spans and execution

| Span | Example | What a normal close establishes |
|---|---|---|
| Entity | `⟦ Atlas⟧` | Its label names at least one indexed entity |
| Fact | `⟦ Atlas · creator · Ada⟧` | At least one matching subject–relation–object triple exists |
| Relation path, in path mode | `⟦ Atlas · creator · born in⟧` | The written relations execute from an admitted start under the configured frontier policy |
| Escape | `⟦ ⁇ an unlisted item⟧` | Nothing about KG membership; the span is unresolved |

In ordinary mode an entity span can become a fact span by writing a separator
after its subject. The next segments are a relation and an object. In path mode,
the next segments are all relations. Select the mode for the oracle/engine;
there is no per-span fact-versus-path marker.

Path mode walks the graph while relations are decoded. After `creator` in the
example, the frontier contains Ada; after `born in`, Cedar. Intermediate entity
names are not generated. The next span is an entity-only answer bound to that
frontier. It cannot start another path. An entity label can name several nodes;
relations operate over the surviving senses. With `committed_paths=True`, a
path closes onto the best-ranked surviving start's reach. Start ranking uses
adjacency degree by default, with label-index row order breaking ties.

Derivation mode binds separate fact spans through their object entities. It is
useful when you want intermediate facts written out. Path mode implies that
binding behavior while using relation-only hops.

## Tokenization and the KG index

An in-memory `KnowledgeGraph` holds entities, relations, and ID triples. IDs can
be arbitrary strings. Labels are what the model writes; the first label is
canonical and the rest are aliases. Several entities may share a label.

The KG index contains sorted tokenized label rows, relation rows, and adjacency
ordered by relation and object label row. A prefix narrows a row range. Each
object alias contributes a separate adjacency entry, so alias policy affects
memory and degree ranking as well as spelling coverage.

Labels are encoded after `SpanSyntax.label_prefix`, a space by default.
Delimiters may occupy several tokens but must finish at a token boundary.
Tokens merging a delimiter with following text can therefore be blocked even
when their text looks plausible. For deterministic replay, encode delimiter
pieces separately and use prefixed/indexed label tokens, as the
[CPU example](../examples/toy_kg.py) does. The model's sampled tokens are checked
directly; generation does not re-tokenize a completed string.

The index records a fingerprint of the serialized tokenizer and its span syntax.
The oracle and engine refuse a tokenizer mismatch. Rebuild for another tokenizer
or syntax; copying IDs between different model vocabularies is invalid.
Supported tokenizer families are documented in
[installation](installation.md#tokenizers-and-saved-artifacts).

Blank labels, labels containing reserved span syntax, and labels containing
special-token text are excluded. Their reasons appear in `LabelIndex.excluded`.
New builds also enforce **exact label encoding** (`exact-v1`): concatenate the
known bytes of every emitted token and require exact equality to the UTF-8 bytes
of `label_prefix + source_label`. Empty encodings, tokens without known bytes
and mismatches are excluded and reported. This applies to entity labels,
aliases, generated initial-case aliases and relation labels in the in-memory
and streamed directory builders. Tokenizer normalization (including NFC) does
not implicitly make two source labels equivalent. Normalize source data only
as an explicit, separately versioned application policy before building.

KG builders record `build_config["label_encoding"] = "exact-v1"`, binding the
admission rule into the base fingerprint. A saved index without this field
retains its historical admission policy and rows on load; loading does not
revalidate or migrate it. Audit proposed exclusions before rebuilding a frozen
index, and create a separately named index and version binding when adopting
the rule. Existing studies must retain their original artifacts and pins.

A triple with no writable subject, relation, or object is counted in
`KGIndex.unwritable_triples`. Inspect both before serving.

A [KG directory](guides/bring-your-own-kg.md#kg-directory-format) is source data
on disk. A saved KG index is its tokenizer-bound compilation. A KG store is a
separate SQLite representation for tools and offline readers. These artifacts
have different formats and cannot substitute for one another.

## Escapes and free text

Text outside spans is free, subject to valid opener boundaries. The engine does
not require the model to open a span or make every claim inside one. Prompt the
model in the syntax of your graph and check completion in your application.

The escape marker releases the rest of a span to free text until the closer.
`ClosedSpan.resolved` is false after an escape. Disabling escapes can steer a
model toward an irrelevant but writable label when the desired fact is absent.
An incomplete span at a token limit is not a resolved result.

Escapes are enabled by default. Bound escapes control whether an anchored start
or bound answer can escape; release and escape bias are separate GPU policies.
See [choosing policies](guides/span-modes.md).

## Frontiers and caps

Execution retains bounded sets and label rows, rather than enumerating every
possible KG path. [`MAX_REACHED`, `MAX_PATH_RELATIONS`, and `MAX_FACTS`](../src/xgraph/oracle.py)
define the frontier/answer-row, hop, and no-repeat memory limits mirrored by the
CUDA engine. Ranking and aliases can affect which entities survive a cap.
These are implementation limits, not user-adjustable constructor arguments.

`MatchedExecutor.execute_set` reports `reached`, `uncapped_reached`, and
`cap_events` for a fixed plan. Its uncapped replay is a diagnostic of that
written plan; it cannot recover relation decisions lost during capped decoding.
The weaker `vocabulary` and `successor` relation supports execute no path and
leave answers unbound. Their closed relation plans are distinct from resolved
paths even when every relation label is valid.

## Versions

A KG version is a named replacement edit set over an exact base index. States
pin the base or one registered version; labels remain shared. Versions do not
add entities, relations, or labels. Unknown/unwritable edits can remove old
objects without adding a writable replacement, so inspect registration stats.
See [versions and edits](guides/kg-versions.md).

## Witnesses

A witness is an entity-ID chain supporting one endpoint of a fixed executed
plan. `MatchedExecutor.execute_set(..., witnesses=True)` supplies these chains
and cap events on CPU. Witnesses establish graph support for the plan; they do
not explain why the model chose it or prove that it answers the question.

The current witness selector requires an index with sitelink counts and
numeric-suffix entity IDs (use the Q-number mapping in the directory guide).
For a custom numeric graph, `build_kg_index(..., sitelinks={})` supplies neutral
zero counts. The basic index/oracle API supports arbitrary string IDs, but the
witness selector's numeric-ID ordering is a narrower experimental interface.
The engine's `closed_span` reports span kinds; it does not return a full path
witness. Re-execute with matching version, ranking, anchors, selections, and
caps to inspect a path.

## Guarantees and contract

Resolved spans are grounded in the indexed graph under the configured policy.
They do not guarantee real-world truth, relevance, coverage of all spellings,
untruncated output, or a complete set of all uncapped KG answers. Anchors guide
starts; component bounds and versions are not tenant authorization mechanisms.

The [engine contract](engine-contract.md) states the scoped path and bound-answer
soundness premise, its witness induction and deliberate frontier-cap incompleteness.
The executable reference is the [oracle module](../src/xgraph/oracle.py), with
[engine design](engine-design.md), [oracle tests](../tests/test_oracle.py),
and [engine agreement tests](../tests/test_engine.py). CPU examples do not
establish GPU agreement.

```{toctree}
:maxdepth: 1

guides/span-modes
```
