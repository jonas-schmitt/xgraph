# Engine design

The [oracle](../src/xgraph/oracle.py) is the CPU reference for allowed next
tokens and span resolution. The [native engine](../csrc/engine.cu) implements
the same language for batched GPU requests; [engine.py](../src/xgraph/engine.py)
builds and owns its tables. See the [engine contract](engine-contract.md).

Tokenized labels and relation labels are compiled with KG adjacency. A sequence
tracks its delimiter/label position, candidate rows and path frontier. The mask
allows only tokens compatible with the current state and enabled policies.
Outside a constrained span, ordinary generation remains unconstrained.

Request slots have independent state. The serving adapter advances committed
generated history, resets reused slots and handles speculative draft positions
without committing them. Integration must preserve that lifecycle, not only
apply a mask. See [another-engine integration](integration/another-engine.md).

Pinned [KG versions](kg-versions-adr.md) replace complete subject/relation pairs
over one fixed label inventory. Registered version tables are immutable while
the engine is serving; adding versions requires rebuilding the engine. Memory
depends on the full adjacency of edited subjects, not only edit-file size.

CPU oracle agreement is a correctness reference. Native mask agreement,
worker compatibility and end-to-end serving performance require separate GPU
validation. See [measurement boundaries](performance.md).
