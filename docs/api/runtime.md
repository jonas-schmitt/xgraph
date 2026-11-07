# Oracle and GPU engine

The oracle is a CPU reference with immutable states. The GPU engine shares
compiled tables and keeps mutable state in independent request slots. The
same span language and configured policy apply. Read the
[integration contract](../integration/another-engine.md) before embedding either.

```{autoclass} xgraph.oracle.Oracle
:members: initial_state, anchored_state, opened_state, allowed, advance
```

```{autoclass} xgraph.oracle.OracleState
```

```{autoclass} xgraph.oracle.ClosedSpan
:members:
```

```{autoclass} xgraph.oracle.TokenNotAllowed
```

```{autoclass} xgraph.engine.Engine
:members: begin, end, reset, anchor, set_version, open_span, advance, sync, fill_mask, mask_host, apply_mask, states_host, errors
```

```{autoclass} xgraph.engine.EngineUnavailable
```

```{autofunction} xgraph.engine.closed_span
```

The vLLM processor is registered by class path. Client code supplies request
options; the runtime calls its callbacks. Importing its reference does not
construct a worker or require vLLM to be installed.

```{autoclass} xgraph.vllm.XGraphLogitsProcessor
:members: validate_params
```

For replayed prefills, also construct the LLM with
`worker_extension_cls="xgraph.vllm.XGraphWorkerExtension"` so the client can
consume worker receipts by name. Missing receipts remain unconfirmed.

```{autoclass} xgraph.vllm.XGraphWorkerExtension
:members: xgraph_collect_replay_status
```

```{autofunction} xgraph.vllm.replay_statuses
```
