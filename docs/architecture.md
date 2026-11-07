# Library architecture

The [glossary](../GLOSSARY.md) defines the project's terms. The public workflow
starts with a `KnowledgeGraph`, the served tokenizer and a `SpanSyntax`.

| Boundary | Implementation | Responsibility |
|---|---|---|
| Graph and labels | [kg.py](../src/xgraph/kg.py) | Application identifiers, labels and triples |
| Tokenizer | [tokenizer.py](../src/xgraph/tokenizer.py) | Served-tokenizer bytes and supported vocabulary boundary |
| Compiled KG index | [index.py](../src/xgraph/index.py) | Tokenized labels, relation tables and adjacency |
| CPU reference | [oracle.py](../src/xgraph/oracle.py) | Legal token sets, state transitions and resolved spans |
| GPU host boundary | [engine.py](../src/xgraph/engine.py) | Native table construction and batched engine API |
| Native engine | [engine.cu](../csrc/engine.cu) | CUDA advance/mask implementation |
| Serving adapter | [vllm.py](../src/xgraph/vllm.py) | Request lifecycle and worker integration |
| Pinned edits | [kgversions.py](../src/xgraph/kgversions.py) | Registered replacements over one index |

The CPU oracle can run without CUDA, a model or vLLM. The optional native engine
uses CUDA C++ bound through nanobind and exchanges buffers through DLPack.
The [vLLM adapter](guides/vllm.md) needs its compatible worker environment;
other generation adapters are not implemented.

Start with the [quick start](getting-started.md), then
[bring your own KG](guides/bring-your-own-kg.md). The
[engine contract](engine-contract.md) states what a resolved span guarantees.
