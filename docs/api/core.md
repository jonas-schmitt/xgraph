# Graphs, tokenizers and syntax

These objects are also available from `import xgraph as xg`. IDs belong to your
application; labels belong to the language a model writes. Constructing a
source graph does not tokenize it or initialize a model.

```{autoclass} xgraph.kg.Entity
```

```{autoclass} xgraph.kg.Relation
```

```{autoclass} xgraph.kg.KnowledgeGraph
```

```{autofunction} xgraph.kg.validate
```

```{autoclass} xgraph.kg.KGFormatError
```

```{autoclass} xgraph.tokenizer.ServedTokenizer
:members:
```

```{autoclass} xgraph.tokenizer.UnsupportedTokenizer
```

```{autoclass} xgraph.syntax.SpanSyntax
:members:
```
