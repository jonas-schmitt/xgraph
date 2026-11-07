# Offline execution

Use the same KG index, tokenizer and policy as decoding. This API executes a
fixed plan; it does not reproduce a model's decisions. Witness selection has
the narrower requirements in the [witness tutorial](../guides/witnesses.md).

```{autoclass} xgraph.programs.MatchedExecutor
:members: execute, execute_set
```

```{autoclass} xgraph.programs.MatchedPath
:members: recorded_plan
```

```{autofunction} xgraph.programs.parse_path
```
