# Integration guide

The implemented generation adapter connects xgraph's GPU engine to vLLM's
Model Runner V2 sampling loop. The CPU oracle supplies a small reference for
implementing and testing the same contract in another engine.

```{toctree}
:maxdepth: 1

../guides/vllm
another-engine
```

RDFLib integrates at the data boundary, before index compilation and after
resolved fact extraction. See the [RDF tutorial](../guides/rdflib.md).
Transformers and SGLang generation adapters are not implemented.
