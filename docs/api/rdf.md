# RDFLib adapter

Install the `rdf` extra before using these functions. Core library imports and
saved-index loading need no RDFLib. The projection retains asserted facts under
the configured policies; it performs no RDF inference or identity merging.

```{autofunction} xgraph.rdf.from_rdflib
```

```{autofunction} xgraph.rdf.write_rdflib
```

```{autofunction} xgraph.rdf.rdflib_edits
```

```{autofunction} xgraph.rdf.to_rdflib
```
