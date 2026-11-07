# KG versions

An oracle consumes edit mappings. An engine consumes compiled `KGVersions`.
Both pin a reader before decoding and retain shared labels. Use content-pinned
string-ID files for application IDs or RDF IRIs; numeric files remain supported.

```{autofunction} xgraph.kgversions.build_kg_versions
```

```{autoclass} xgraph.kgversions.KGVersions
:members: view, matches, number, check_base
```

```{autoclass} xgraph.kgversions.VersionStats
```

```{autoclass} xgraph.kgversions.VersionView
:members: relations_of, object_rows, edges_at
```

```{autofunction} xgraph.kgversions.write_kg_edits
```

```{autofunction} xgraph.kgversions.write_kg_versions
```

```{autofunction} xgraph.kgversions.read_kg_versions
```

```{autofunction} xgraph.kgversions.load_kg_versions
```

```{autoclass} xgraph.kgversions.KGVersionsFormatError
```
