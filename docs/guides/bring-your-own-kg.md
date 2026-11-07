# Bring your own knowledge graph

Choose the in-memory API for small graphs and arbitrary string IDs. For larger
graphs, write a numeric KG directory and use the existing bounded-memory builder.
For RDF, install the optional `rdf` extra and use the [RDFLib adapter](rdflib.md),
which preserves IRIs and literal-value IDs through its `rdf-ids.json` mapping.
Both paths produce the same kind of tokenizer-bound KG index.

## In-memory triples

The [complete toy example](../../examples/toy_kg.py) builds, saves and reloads
an index. Its source graph uses application IDs and is constructed here:

```{literalinclude} ../../examples/toy_kg.py
:language: python
:pyobject: graph
```

Define every entity and relation referenced by a triple; unknown triple IDs
raise `KGFormatError`. Use unique IDs and supply nonempty label tuples, with
the canonical label first. Duplicate owner/label pairs and duplicate adjacency
edges are deduplicated; shared labels retain their different owners. The
in-memory builder materializes the graph and sorted edges, so it is not a
streaming triples importer.

`xgraph.kg.validate(kg)` checks the source without a tokenizer; the in-memory
builder calls it automatically. Empty label tuples are accepted as unwritable
owners, and blank/syntax-bearing labels retain the existing exclusion behavior.

Literal values need entity representations and labels. The core triple model
does not accept a raw string or numeric literal as a separate RDF term type.
The existing [literal loader](../../src/xgraph/literals.py) handles its supported
date and quantity representations; qualifiers and general RDF semantics are
outside ordinary path execution.

## KG directory format

The current directory builder lives in `xgraph.wikidata` but is also used by
MINTQA, Freebase component loaders, and generated worlds. It uses local
Wikidata-shaped **numeric identifiers**, independently of where the data came
from. Preserve your original-ID-to-number mapping alongside the source data.
Keep entity and relation numbering stable across versions and rebuilds.

| File | Representation | Required by |
|---|---|---|
| `entity_labels.jsonl` | UTF-8 JSON array per line: `["Q1", "Atlas", "label"]`; aliases use `"alias"` in the third field | Directory index builder and KG store |
| `relation_labels.jsonl` | Same layout, relation IDs: `["P1", "creator", "label"]` | Directory index builder and KG store |
| `triples.npy` | NumPy `int32` array of shape `(n, 3)`; columns are subject number, relation number, object number | Directory index builder and KG store |
| `sitelinks.jsonl` | UTF-8 arrays `["Q1", "Page title"]`; empty when there are no Wikipedia titles | KG-store and source-directory workflows; not needed by the bare index builder |
| `manifest.json` | JSON counts, source identity, configuration and provenance | Source inventory and reproducibility; the bare index builder does not validate it |
| `sitelink_counts.npy` | Optional `int32` shape `(n, 2)`, sorted by entity number: `[entity_number, count]` | Sitelink ranking and endpoint witnesses |
| `date_values.npy` | Optional `int64` shape `(n, 4)`: entity number, year, month, day | Merged dates; see `DATE_VALUES` in the literal loader and `_dates_of` in the directory builder |

The ordinary example below has no date table. For a custom graph, do not invent
sitelink counts as evidence of popularity; omit them or use explicit zeros when
an interface requires the array. Counts must be nonnegative. Numbers used by
triples must fit `int32`; keep Q/P IDs in their matching label files. The files
contain data, not Python objects or pickle arrays.

Here is a complete minimal directory writer. Run it from the repository root
with the installed environment:

```python
import json
from pathlib import Path
import numpy as np

kg_dir = Path("target/custom-kg")
kg_dir.mkdir(parents=True, exist_ok=True)
files = {
    "entity_labels.jsonl": [
        ["Q1", "Atlas", "label"],
        ["Q2", "Ada", "label"],
        ["Q3", "Cedar", "label"],
    ],
    "relation_labels.jsonl": [["P1", "creator", "label"], ["P2", "born in", "label"]],
    "sitelinks.jsonl": [],
}
for name, rows in files.items():
    (kg_dir / name).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
np.save(kg_dir / "triples.npy", np.array([[1, 1, 2], [2, 2, 3]], dtype=np.int32))
(kg_dir / "manifest.json").write_text(
    json.dumps(
        {"source": "hand-written example", "counts": {"entities": 3, "relations": 2, "triples": 2}}
    )
    + "\n",
    encoding="utf-8",
)
```

Index this directory using a real tokenizer already present locally:

```bash
export MODEL_DIR=/path/to/local/model
uv run --no-sync python -m xgraph.kg validate --kg target/custom-kg
uv run --no-sync python -m xgraph.wikidata index --kg target/custom-kg \
    --tokenizer "$MODEL_DIR/tokenizer.json" --out target/custom-index
```

The Python equivalent is
`xgraph.wikidata.build_index(kg_dir, tokenizer, syntax, out)`, with `Path`
directories, a `ServedTokenizer`, and a `SpanSyntax`. It writes the saved index
and returns `None`; use `load_kg_index(out)` to inspect it.
`--unique-aliases` excludes relation aliases shared by several relations;
the directory builder always drops an alias colliding with another relation's
canonical label, ignoring case. The in-memory builder preserves the supplied
aliases. Apply the same policy yourself if comparing the two build paths.

## Saved KG index

Use `save_kg_index` and `load_kg_index`, rather than editing arrays by hand.
`kg-index.json` records format, tokenizer fingerprint, syntax, owner IDs,
exclusions, and optional arrays. Label and relation arrays use
`labels.tokens.npy`, `labels.offsets.npy`, `labels.owners.npy` and their
`relations.*.npy` equivalents. Adjacency uses `adjacency_offsets.npy`,
`edge_relations.npy`, and `edge_objects.npy`. The loader memory-maps the arrays.
`kg_index_fingerprint(directory)` hashes the saved index files for provenance.
New builds also store `base_fingerprint`, `source_graph_hash`, and `build_config`.
`load_kg_index` retains that identity without scanning arrays; `verify=True`
recomputes it to check file integrity. `strict=True` also refuses legacy indexes
without a complete recorded identity. Default binding trusts headers and does
not detect modified files. See [versions](kg-versions.md) for binding and migration.

Save different builds in separate directories. The GPU engine copies index
tables to its device; a memory-mapped host index does not imply host-resident
GPU execution or automatic sharding.

## Import boundaries

Use the in-memory `KnowledgeGraph` interface above for application triples, or
the optional [RDF projection](rdflib.md) for RDFLib graphs. The public source
also contains dataset-specific adapter modules; their presence does not supply
datasets, access rights or complete evaluation recipes. Review their source
docstrings and upstream terms before use.

The directory builder's `python -m xgraph.wikidata index --help` describes its
indexing CLI. Build with the exact served tokenizer and retain the source graph,
alias policy and index identity. Historical research recipes and released-data
preparation records are separate from this archive; see
[reproduction scope](../reproducing.md).

## Optional initial-case aliases

`build_kg_index(..., initial_case_aliases=True)` adds the opt-in initial-case
entity surfaces described in the [glossary](../../GLOSSARY.md).
The directory CLI exposes the same policy as `python -m xgraph.wikidata index
--initial-case-aliases`. Existing surfaces and homonym resolution remain primary;
the default build does not add these aliases. Save the variant separately and
record its index fingerprint with its exact tokenizer.
