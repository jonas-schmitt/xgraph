# vLLM integration

The implemented adapter is
[`XGraphLogitsProcessor`](../../src/xgraph/vllm.py) for vLLM's **Model Runner
V2** interface. The repository targets vLLM 0.31; this internal interface is
experimental, so a different version requires adapter and GPU validation.
Install vLLM separately in the worker environment. The Python API below does
not require Docker or a research launcher.

## Build and prerequisites

Use Linux, an NVIDIA GPU, the CUDA toolkit with `nvcc` on `PATH`, CMake 3.26 or
newer, Python 3.12, and a compatible vLLM environment. Install/build xgraph in
the **same environment as the vLLM worker**. The library's core dependencies do
not install vLLM. For a source checkout with a GPU toolchain:

```bash
uv sync --locked --no-default-groups \
    --config-setting "cmake.define.CMAKE_CUDA_COMPILER=$(command -v nvcc)"
```

After a CPU build, explicitly replace the cached CUDA-disable value with the
installed compiler path and rebuild:

```bash
uv sync --locked --no-default-groups --reinstall-package xgraph \
    --config-setting "cmake.define.CMAKE_CUDA_COMPILER=$(command -v nvcc)"
```

Removing a command-line setting alone may leave `CMAKE_CUDA_COMPILER=NOTFOUND`
in the persistent CMake cache. The explicit path above overrides it. Check that
`command -v nvcc` succeeds first.

For an existing compatible vLLM environment:

```bash
python -m pip install . \
    --config-settings "cmake.define.CMAKE_CUDA_COMPILER=$(command -v nvcc)"
```

Its nanobind extension uses DLPack and does not link torch's C++
ABI. Supported CUDA targets are defined in [CMakeLists.txt](../../CMakeLists.txt).
Index tables occupy GPU memory alongside model weights and KV cache; budget
them before launch. See [performance measurement](../performance.md).

Build your KG index with the exact `tokenizer.json` served by this model. Run
the [GPU example](../../examples/vllm_generation.py) only in an environment
meeting these requirements; CPU examples exercise different validation scope.

## Serve an OpenAI-compatible endpoint

Set paths for your installation, then launch from that environment:

```bash
export MODEL_DIR=/path/to/local/model
export KG_INDEX_DIR=/path/to/saved/kg-index
export VLLM_USE_V2_MODEL_RUNNER=1
export XGRAPH_KG_INDEX="$KG_INDEX_DIR"
export XGRAPH_TOKENIZER="$MODEL_DIR/tokenizer.json"
export XGRAPH_PATHS=1
export XGRAPH_COMMITTED_PATHS=1
vllm serve "$MODEL_DIR" --logits-processors xgraph.vllm:XGraphLogitsProcessor
```

Force V2 explicitly: the runtime can otherwise select an incompatible runner.
For ordinary entity/fact spans, omit the two path settings. Prompts must teach
the selected language and your graph's relation labels; the adapter alone does
not force every answer into a span.

An online request opts in through `vllm_xargs` in its JSON body:

```json
{
  "model": "your-served-model",
  "messages": [{"role": "user", "content": "Write a KG path and an Answer span."}],
  "temperature": 0,
  "max_tokens": 128,
  "vllm_xargs": {"xgraph": 1}
}
```

With an OpenAI-compatible client, pass the last field through its `extra_body`
argument. The [generation tutorial](generation.md) provides the complete Python
worker example, its local CPU dry run and matching toy graph. A client package
is not a core xgraph dependency.

Use your graph's labels and examples for your application. Requests omitting
`xgraph=1` remain unconstrained.

## Offline Python generation

In an existing vLLM environment, register the same class path with
`LLM(..., logits_processors=["xgraph.vllm:XGraphLogitsProcessor"])` and opt in
with `SamplingParams(extra_args={"xgraph": 1})`. Set the same environment before
constructing `LLM`. The complete [GPU example](../../examples/vllm_generation.py)
does this for the toy KG and adds an anchor. No one-call xgraph setup factory
exists yet.

## Request settings

These keys go inside `vllm_xargs` online, or `SamplingParams.extra_args` offline.
They are not ordinary top-level OpenAI request fields.

| Key | Value | Meaning |
|---|---|---|
| `xgraph` | `1` or `0` | Opt in/out |
| `xgraph_anchor` | List of entity ID strings | Bind the first subject in derivation/path mode |
| `xgraph_kg_version` | Registered version name | Read an edit set; needs `XGRAPH_KG_VERSIONS` at startup |
| `xgraph_open` | `1` | Begin inside a span; prompt must end with the opener |
| `xgraph_prefill` | Nonempty text | Replay the prompt's trailing assistant prefill; excludes `xgraph_open` |
| `xgraph_replay_id` | Unique opaque string | Correlate a replay with its worker receipt; required by a replay-status consumer |
| `xgraph_question` | Nonempty question text | Required for STOP, audit, and relation menus |
| `xgraph_stop` | `0` | Disable configured STOP for this request |
| `xgraph_path_options` | Dictionary | Experimental per-request observation/menu settings; requires `XGRAPH_REQUEST_PATH_OPTIONS=1` and allocated capabilities |

Anchors are IDs in your index, including arbitrary strings in an in-memory-built
index. IDs are deduplicated before label rows are capped. An explicit empty,
unknown-only or exclusively unwritable anchor raises `ValueError` during request
preparation. Mixed anchors keep the writable IDs; an omitted anchor is unanchored.
Validate a whole anchored request set before launching generation. With an existing
saved index and your JSONL request file:

```bash
export REQUESTS_FILE=/path/to/requests.jsonl
python -I -m xgraph.anchors --kg-index "$KG_INDEX_DIR" \
    --requests "$REQUESTS_FILE" --require-anchors --out target/anchor-validation.json
```

The CPU command scans the complete set, prints all failing request IDs and anchor
IDs, writes input hashes and the index fingerprint, and exits nonzero on any empty,
malformed, unknown or unwritable anchor. Omit `--require-anchors` to allow unanchored
requests; supplied anchors are still checked. Duplicate known IDs are accepted;
an unknown ID in a mixed anchor is an input error in this stricter pre-validation.
The engine/oracle rules above remain the runtime contract.

`xgraph.generate` validates supplied anchors before importing vLLM or constructing
its model for constrained, fallback and masked-agent generation. The launcher
validates before building its wheel or checking GPU headroom. Direct endpoint or
Python vLLM callers should run this command or `xgraph.anchors.validate_anchors`
over their complete request set before starting the server/model.

A replay tail mismatch or disallowed token declines masking. Generation attaches
`xgraph_replay_id` and calls `xgraph.vllm.replay_statuses(llm, keys)` after the
call to collect sticky per-request worker receipts. Records with a declined,
missing or invalid receipt retain text and costs but receive no constrained
answer or plan credit. A decline without a status consumer raises an error.
The generation CLI saves records and its summary before returning nonzero.
Applications calling vLLM directly must provide unique correlation keys and
collect the receipts; aggregate `replays` counters cannot establish request status.

## Lifecycle and validation

One index/engine is shared per worker; slots hold separate request state and
version. The adapter skips prompt tokens, follows committed generated history,
and uses scratch state for speculative draft positions. Reused slots reset
before their next request. Its hooks are called by vLLM, not by client code.

The [adapter tests](../../tests/test_vllm_adapter.py) include CPU stand-ins
for host callbacks and replay lifecycle, plus native CUDA cases. The
[engine agreement tests](../../tests/test_engine.py) include native GPU cases.
A real serving smoke also requires a compatible GPU worker and model.
CPU tests alone do not qualify a runtime upgrade or demonstrate serving speed.

## Other runtimes

| Runtime | Current support |
|---|---|
| vLLM Model Runner V2 | Implemented GPU adapter |
| Hugging Face Transformers | No xgraph `LogitsProcessor` adapter; oracle replay is a reference only |
| SGLang | No xgraph backend implemented |

Do not substitute a grammar backend name such as `xgrammar` for the xgraph
processor: they enforce different constraints. See
[troubleshooting](troubleshooting.md) and [span policies](span-modes.md).
