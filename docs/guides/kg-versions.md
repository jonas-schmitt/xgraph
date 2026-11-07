# KG versions and edits

A version is a named edit set over one base KG index, registered at engine
construction and selected per request. The base and all versions share labels,
relations, and tokenization. New labels require rebuilding the index.

## Replacement semantics

Each `(subject, relation, object)` edit names IDs. Edits are grouped by
`(subject, relation)`; the listed objects replace **all** base objects of that
pair. Multiple listed objects form a set, not an ordered sequence of writes.
Unmentioned pairs retain their base edges. A version can add a pair whose
entities and relation already have writable index rows.

An unknown ID or an ID with no writable label produces an unwritable edit.
For a known writable subject/relation pair, an unwritable replacement object
still removes the old objects and adds no edge. Check
`KGVersions.stats[*].unwritable_edits`; successful registration alone does not
establish that the intended replacement is readable. Use `(subject, relation, None)`
to delete a pair explicitly; do not use unknown IDs as a deletion interface.
Adding labels requires an index rebuild.

## Python API

Arbitrary application IDs work in memory and in content-pinned JSONL files.
Run the complete [string-ID example](../../examples/string_versions.py) from the
repository root. It creates its own graph, tokenizer and index, persists
replacement and deletion edits, then interleaves base and edited readers on CPU:

```{literalinclude} ../../examples/string_versions.py
:language: python
:pyobject: main
```

The oracle takes the edit mapping, while `Engine(..., versions=versions)` takes
compiled `KGVersions` tables. The oracle states are immutable and retain their
version. For the engine, call `begin`, then `set_version(slot, name)` once before
decoding. `None` selects the base. `begin` resets a reused slot's version.
Registering another version requires a new engine/server; there is no live
overlay reload protocol.

Every hop and bound answer reads that same version. Other registered versions
cannot supply a missing edge, even when the union of their edges would complete
a path. Union paths combine entity senses within the selected version. Both
`begin` and `reset` return slots to the base; after preemption the adapter needs
the original request arguments to reinstall the version before replaying its
prompt tail and generated history. See the
[engine contract](../engine-contract.md) for scope and executable checks.
Selecting an existing version does not measure ingestion speed or establish
instant editing. Compiled tables copy the whole adjacency of each edited subject,
so edits to hubs can cost substantially more memory than their edit files.

Compiled versions bind to `KGIndex.base_fingerprint`, covering every source
entity, relation, label and triple (including excluded content), tokenizer
identity, build configuration and compiled table. Builders compute it once.
`KGVersions.matches`, views and engine admission compare recorded identities,
without scanning arrays, and refuse another recorded base even with identical
shape and tokenizer. Source order and directory serialization are significant.
Index tables and metadata must remain immutable during use.
Keep `kg_index_fingerprint` and version-file hashes in application provenance
as separate hashes of the saved artifact bytes.

## Persisted version directory

The legacy `.npy` interface uses Q/P numbers even for custom graphs. Use the
[numeric mapping](bring-your-own-kg.md#kg-directory-format) consistently in the
base index and edits. Do not persist arbitrary IDs by stripping their prefixes.

The [numeric version example](../../examples/kg_versions.py) exercises the
legacy array writer and strict loading, then resolves the same base and edited
paths. It is executed by the CPU example tests.

`kg-versions.json` contains `format`, plus an ordered `versions` list with
`name`, `edits` (the array filename), and `count`. Each edit array is `int32`
shape `(n, 3)`, including shape `(0, 3)` for an empty version. Names start with a
letter or digit and then use letters, digits, `_`, `.`, or `-`. The writer
accepts extra JSON metadata, such as input hashes. With `index=...`, it records
`base_fingerprint`; loading or serving it against another base raises a clear
fingerprint mismatch. `read_kg_versions` returns a dict-compatible `VersionEdits`
whose binding survives use by `Oracle` and `build_kg_versions`.

Existing indexes and version files load without modification. When an index has
no recorded fingerprint, `fingerprint_status` is `"legacy"`; it uses the previous
tokenizer/entity/relation/label-row count check for compiled-version admission. An unbound
legacy version file also reports `"legacy"`: its original source base cannot be
recovered from the edits alone. `load_kg_index(..., strict=True)` and
`load_kg_versions(..., strict=True)` refuse missing identities. The version
writer without `index` retains legacy behavior for existing preparation
workflows; use a fingerprinted index for newly pinned application versions.
Rebuild from the original source to obtain complete source provenance rather
than inventing a fingerprint for archived files. A bound file is specific to
one tokenizer/build; write separate bound files for different base indexes.

[kg_versions.py](../../examples/kg_versions.py) is a complete CPU round-trip
with an interleaved base reader and an edited reader. It uses local numeric IDs,
with no external graph dependency.

`write_kg_edits` persists application IDs and explicit pair deletions. The
[string-ID example above](../../examples/string_versions.py) supplies the exact
base index, checks strict loading and shows the unchanged base reader.

The example uses the string-ID toy graph. The format records each JSONL
file's hash and the complete `base_fingerprint` of a fingerprinted index.
`read_kg_versions` reads both formats and retains the binding through oracle
admission; `load_kg_versions` compares the recorded identity before building GPU
tables. New string-ID writes require a complete base fingerprint. Archived
string-ID files with only `base_index_identity` remain readable with legacy
status: compatibility readers recompute that older content-only pin, including
when the edits go directly to the oracle. Strict loading refuses those files.
A pair cannot mix explicit deletion with replacement objects in the
same saved version. See [RDF graph deltas](rdflib.md#pinned-versions) for graph
differences projected to these edits.

## vLLM requests

Set `XGRAPH_KG_VERSIONS` to the version directory before starting vLLM. Each
opted-in request selects a registered name:

```json
{"vllm_xargs": {"xgraph": 1, "xgraph_kg_version": "updated"}}
```

Default binding catches wrong index/version pairings by recorded identity; it
trusts those headers and does not detect modified index arrays or numeric edit
arrays. String-ID JSONL files also verify their recorded edit hashes on read.
`load_kg_index(..., verify=True)` and `python -m xgraph.kg validate --index <index>`
also recompute index contents against its record. Legacy verification has no
original complete record to compare and cannot establish file integrity.
`fingerprint_status="verified"` denotes a complete recorded identity, not a
load-time verification result. Verification does not authenticate headers or
recheck the original source KG; preserve external artifact hashes as well.

Set `XGRAPH_STRICT_FINGERPRINT=1` at server startup to recompute the index
fingerprint and require recorded complete identities for both index and version
files. `load_kg_index(..., strict=True)` provides the same index check directly.

Omit `xgraph_kg_version` to use the base. Requests using different versions may
share a batch. Versions change adjacency and version-dependent degree ranking;
the model prompt need not carry replacement facts. See the
[vLLM guide](vllm.md) for launch and request examples.

The optional `components` metadata scopes some SQLite readers to question-local
entity ranges. It does not constrain unanchored engine requests and is not an
access-control boundary. See the historical
[version ADR](../kg-versions-adr.md) for the original design decision.
