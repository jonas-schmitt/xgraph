# Inspecting witnesses

A path result contains endpoints. To inspect a supporting entity-ID chain,
re-execute the same plan and request settings with `MatchedExecutor` on CPU.
This example uses a tiny local graph and needs no model:

```{literalinclude} ../../examples/witnesses.py
:language: python
```

The witness `Q1 → Q2 → Q3` supports endpoint `Q3` through `creator` and `born in`.
`cap_events` is empty for this graph; `uncapped_reached` is a diagnostic of this
fixed plan with execution caps removed. Removing a cap cannot recover relation
decisions that were never made during decoding.

## Selector requirements

The experimental witness selector requires sitelink counts and entity IDs with
a numeric suffix, such as `Q1`. Empty `sitelinks={}` supplies neutral zero counts
for a custom numeric graph. Preserve the original-ID mapping if you use local
numbers. These restrictions apply to this selector; the core KG index, oracle
and string-ID version API accept arbitrary string IDs.

Match the original index, tokenizer, version, start ranking, anchors and
selections when replaying a generated path. Witnesses come from the capped
execution, with ties resolved by counts and numeric IDs. They demonstrate
support in that KG, without proving truth or question relevance. The GPU
engine does not emit a full path witness stream.

To export support as RDF, supply actual relation-ID triples to
`to_rdflib(witnesses=...)`. Endpoint IDs alone do not identify those triples;
an ambiguous relation label can denote several relations. See
[RDF export](rdflib.md#back-to-rdf) and the [execution API](../api/execution.md).
