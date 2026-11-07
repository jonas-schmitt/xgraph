# KG index

The in-memory builder is available as `xgraph.build_kg_index`. A source KG
directory is indexed by `xgraph.wikidata.build_index`; the historical module
name does not restrict it to Wikidata. Keep tokenizer and index identities with
saved artifacts and version edits.

```{autofunction} xgraph.index.build_kg_index
```

```{autoclass} xgraph.index.KGIndex
:members: tokenizer_fingerprint, relations_of, object_rows, edges_at, date_of, compute_base_fingerprint
```

```{autoclass} xgraph.index.LabelIndex
:members: num_rows, row_tokens, owner_id, token_at, prefix_range
```

```{autoclass} xgraph.index.ExcludedLabel
```

```{autofunction} xgraph.index.save_kg_index
```

```{autofunction} xgraph.index.load_kg_index
```

```{autofunction} xgraph.index.kg_index_fingerprint
```

```{autofunction} xgraph.wikidata.build_index
```

```{autoclass} xgraph.index.IndexMismatch
```

```{autoclass} xgraph.index.KGIndexFormatError
```

```{autoclass} xgraph.index.UnknownId
```
