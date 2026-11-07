# Bring an RDF knowledge graph

The optional `xgraph.rdf` adapter turns an RDFLib `Graph` into xgraph's
`KnowledgeGraph` or a KG directory. Compile it with the existing index builder
for your served tokenizer. The oracle works on CPU; serving still needs the GPU
engine. The adapter performs no reasoning or retrieval.

Install the optional `rdf` extra using the [CPU installation](../installation.md#optional-rdf-support). The `rdf` extra pins RDFLib in `pyproject.toml`;
the development group installs it too, so RDF tests run in the stable gate.
Core imports, index building and saved-index loading do not import RDFLib.

The [runnable CPU example](../../examples/rdf_graph.py) exercises directory
indexing, content-pinned string-ID versions, explicit pair deletion and fact
export. Its commands are in [examples](../../examples/README.md).

## Library

The complete CPU example creates its RDF data in memory; no external files are
needed. Run it with a fresh output directory:

```bash
uv run --no-sync python examples/rdf_graph.py --out target/examples/rdf
```

```{literalinclude} ../../examples/rdf_graph.py
:language: python
:pyobject: main
```

`from_rdflib` returns a source KG; `to_rdflib` returns an RDF graph; and
`write_rdflib` writes a source KG directory. Their subsequent in-memory or
directory index build records a complete `base_fingerprint` with
`fingerprint_status="verified"`. Check saved contents explicitly with
`load_kg_index("my-index", strict=True)`. That status describes the recorded
identity; strict loading additionally recomputes it. Indexes saved before base
fingerprints were introduced load with `fingerprint_status="legacy"` and are
refused by strict loading. Rebuild those indexes from the source KG to upgrade.
Export omits label metadata, so reimporting exported facts can create a different
source identity; bind versions to the newly built index.

`from_rdflib` takes keyword options:

| Option | Default and meaning |
| --- | --- |
| `label_predicates` | `RDFS.label`, `SKOS.prefLabel`, `SKOS.altLabel`, `SDO.name`, in that order; `SDO` uses `https://schema.org/` |
| `relation_label_predicates` | Same predicates as entity labels; supply another ordered iterable for relation labels |
| `alias_predicates` | `SKOS.altLabel`; these predicates supply aliases for both kinds of label and never the canonical label |
| `languages` | `("en",)`; exact, case insensitive tags in preference order, with untagged labels afterwards; `None` accepts all tags in lexical order, and `()` accepts only untagged labels |
| `include_literals` | `False`; opt in to literal values as terminal entity objects |
| `blank_nodes` | `"skip"`; omit triples incident on blank nodes; `"error"` rejects them |
| `missing_labels` | `"iri"`; use the complete IRI as canonical label; `"error"` rejects missing accepted labels |

For example, `label_predicates=(SKOS.prefLabel, RDFS.label)`,
`relation_label_predicates=(RDFS.label,)`, `languages=("fr", "en")` chooses
preferred labels first, then French before English within each predicate.
Alias predicates are always consulted in addition to those lists. Pass
`alias_predicates=()` to disable them or let an explicitly listed alternative
predicate supply canonical labels.

The RDF projection has these rules:

- Entities are the IRI subjects and objects of retained data triples, plus IRI
  subjects of configured label annotations, including labels rejected by the
  language filter. An IRI used only as a relation and its label subject is not
  also an entity. An IRI used in both roles retains both roles. IRIs must be
  absolute; their spelling is preserved without namespace shortening.
- Relations are the predicates of retained data triples. Configured entity,
  relation and alias label predicates are metadata throughout the graph and
  produce no data edges. A label object must be a literal. Empty or whitespace
  labels are ignored.
- Canonical labels are chosen by predicate order, then language order, then
  lexical text order. Other accepted primary labels become aliases; alias-only
  labels follow them. Duplicated surfaces within one owner collapse. An
  alias-only node gets the IRI fallback before its aliases. Shared labels keep
  every owner, with the existing homonym resolution semantics.
- Blank nodes are not entities. To retain them, skolemize them to stable IRIs
  before importing, using your own persistent mapping or RDFLib's
  `Graph.skolemize`. Parsing-generated blank-node names are not stable ids.
- `owl:sameAs`, `rdf:type` and schema relations are ordinary asserted edges.
  There is no entity merging, inverse closure, RDFS or OWL inference. Run any
  desired reasoning before import. Pick a particular graph from an RDFLib
  `Dataset` explicitly; the adapter rejects datasets to avoid implicit unions.
- With literal values enabled, every non-label literal can be an object,
  including strings and language-tagged strings. Its opaque id encodes
  `[lexical form, language, datatype]` as JSON, distinct from an absolute IRI.
  RDFLib's terms as supplied are preserved, including datatype and language;
  language tags are lowercased so their case does not change RDF term identity.
  RDFLib can normalize lexical forms during parsing: construct literals with
  `normalize=False` before import when exact lexical preservation matters.
- Literal labels use their lexical form; dates (`xsd:dateTime`) and quantities
  (`xsd:decimal`) also use the existing `xgraph.literals` renderings. Values with
  different tags or datatypes remain distinct even when labels coincide. The
  language filter applies to label annotations, not literal data. Blank value
  labels and labels containing span syntax face the normal index exclusions.
  No date comparison table, unit conversion or datatype inference is added.
- Entities, relations and triples are sorted and deduplicated. No random seed
  is needed. Label exclusions, unwritable triples, frontier caps and the
  tokenizer fingerprint retain their existing index and oracle semantics.

## Files to a KG directory and index

```bash
uv run --no-sync python -m xgraph.rdf import people.ttl extra.nt --out my-kg --include-literals
uv run --no-sync python -m xgraph.wikidata index --kg my-kg \
    --tokenizer tokenizer.json --out my-index
```

The parser format is inferred per file; `--format` selects one explicitly.
Repeat `--language` for ordered language preferences. `--blank-nodes error` and
`--missing-labels error` select strict policies. The CLI accepts local files;
for another store or RDFLib parser configuration, construct a `Graph` yourself
and call `write_rdflib(graph, "my-kg", include_literals=True)`.

The destination must be empty. `entity_labels.jsonl`, `relation_labels.jsonl`,
`triples.npy` and the empty `sitelinks.jsonl` use the existing directory layout.
`rdf-ids.json` maps directory-local integer storage ids back to RDF ids. The
existing builder restores those ids before saving the index; saved indexes need
neither the sidecar nor RDFLib. Keep the sidecar with the KG directory and do not
interpret its local `Q`/`P` storage ids as Wikidata identifiers.

The directory builder retains its existing relation-alias policy: an alias
equal to another relation's canonical label is omitted; canonical homonyms
remain. Its optional `--unique-aliases` also removes shared relation aliases.
The in-memory builder keeps the supplied aliases without this directory policy.

`manifest.json` records import options, the RDFLib version, a digest of the
projected KG and provenance. The CLI hashes every source file. For programmatic
imports pass `inputs=[...]` to `write_rdflib` when source files exist; otherwise
the projection digest identifies the retained content. These directories are
supported by the index builder. Wikidata-specific KG mutation and SQLite store
tools still assume Wikidata identifiers and do not restore this mapping.

## Pinned versions

Use the same import options as the base. Supply either a complete updated graph
or sets of RDF triples to add and remove. The [example](../../examples/rdf_graph.py)
computes both a replacement and a
pair deletion with `rdflib_edits`, writes content-pinned JSONL edits and creates
an oracle reader in the updated version. It checks the deleted pair separately
through the compiled version view.

`rdflib_edits` leaves its inputs untouched. It computes the difference of the
two RDF projections, then emits **all remaining objects** for each changed
subject–relation pair. Removing the last object emits `(subject, relation, None)`
to delete that pair. Unchanged pairs inherit the base. Removing an absent RDF
triple or adding and removing the same triple is an error. New entity or
relation ids and changes to retained labels require rebuilding the index.
Labels for nodes no longer occurring in the updated projection stay available
in the base label index, as for every KG version.
Removing a relation's final data edge keeps its label annotations as metadata;
it does not require adding a new entity row. Those annotations must stay
unchanged. Using that relation IRI as a new data entity still requires a rebuild.

String-ID version files use JSONL edits, check their hashes on read, and record
the complete base fingerprint for admission by `read_kg_versions`, the oracle
and `load_kg_versions`. Another recorded base with the same dimensions is
rejected. Older string-ID files retain their content pin with legacy status;
strict loading refuses them. Numeric `.npy` versions can also record a base
fingerprint with `write_kg_versions(..., index=index)`; omitting `index` keeps
their legacy unbound format. The engine and vLLM version loader consume these same
tables; CPU tests establish loader, index and oracle behavior. CUDA agreement
and model serving are separate validation layers.

## Back to RDF

The same example exports the resolved fact with `to_rdflib(spans=[fact])` and
checks the resulting relation-ID triple. Call the returned graph's `serialize`
method to write RDF in a format supported by RDFLib.

Export follows `xgraph.turtle.span_triples`: resolved fact spans assert their
actual triples; entity spans, escapes and unresolved spans assert nothing.
Labels, language preferences and other discarded source metadata are not
reconstructed. Import/export round-trips the retained data facts and literal
terms, rather than the entire original document or its syntax.

A path result stores endpoints, not intermediate assertions. Supply a list of
resolved subject/relation/object ID triples, with relation **ids**, through the
`witnesses` argument to `to_rdflib`. Its [API reference](../api/rdf.md) describes
the accepted inputs; [witness inspection](witnesses.md) explains how to obtain
the supporting entity chains.

Do not infer an edge from two endpoints or from an ambiguous relation label.
The exporter accepts caller-supplied triples and witnesses; it does not validate
their membership in an index. RDF set semantics deduplicate facts. The existing
`xgraph.turtle.turtle` formatter remains the prefixed Wikidata formatter; use
`to_rdflib(...).serialize(...)` for arbitrary RDF ids and literal values.

The import operates over an already materialized RDFLib graph and builds
in-memory Python collections. Large RDF stores need an appropriately sized
projection; this is not the bounded-memory Wikidata dump streamer. No inference,
named-graph preservation or blank-node identity round-trip is provided.

RDFLib's [release pin](https://pypi.org/project/rdflib/7.6.0/) and
[Graph API](https://rdflib.readthedocs.io/en/7.6.0/apidocs/rdflib.graph/)
describe the parser, term and serialization contracts used here.

Before starting an anchored generation run over the saved RDF index, validate all
request anchors with [anchor pre-validation](vllm.md#request-settings). RDF entity
IDs stay strings, including IRIs and the adapter's literal-value IDs; use those
IDs rather than labels. The optional `[rdf]` extra is needed for RDF projection
and the RDF example; the anchor check itself uses core dependencies.
