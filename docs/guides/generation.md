# Constrained generation with vLLM

Use the [quick start](../getting-started.md) to check the CPU API, then follow
[GPU installation and vLLM compatibility](vllm.md#build-and-prerequisites).
The worker needs a CUDA build of xgraph, a compatible Model Runner V2 runtime,
local model weights, and an index built for that model's exact tokenizer.

## Prepare your KG index on CPU

The toy graph makes the generation setup self-contained. Set `MODEL_DIR` to a
local model directory, then build the same graph with its tokenizer:

```bash
export MODEL_DIR=/path/to/local/model
uv run --no-sync python examples/toy_kg.py \
    --tokenizer "$MODEL_DIR/tokenizer.json" --out target/examples/model-index
```

This opens only the tokenizer. For your application, substitute the graph from
the [KG tutorial](bring-your-own-kg.md) and use its relation labels in the prompt.

## Check configuration on CPU

```bash
uv run --no-sync python examples/vllm_generation.py --dry-run \
    --model "$MODEL_DIR" --index target/examples/model-index \
    --tokenizer "$MODEL_DIR/tokenizer.json"
```

The dry run prints environment settings and the full worker constructor
arguments, without importing vLLM or loading weights. Check the processor class
path, tokenizer path, index path and memory settings before launching a worker.

## Generate in the GPU worker

In the compatible worker environment, run the same script without `--dry-run`:

```bash
python examples/vllm_generation.py --model "$MODEL_DIR" \
    --index target/examples/model-index --tokenizer "$MODEL_DIR/tokenizer.json"
```

The generation call registers the logits processor and opts in per request:

```{literalinclude} ../../examples/vllm_generation.py
:language: python
:pyobject: generate
```

The prompt teaches a start/relation path followed by a bound entity answer.
It uses the example's `creator` and `born in` labels and `item:atlas` anchor.
Output wording is model-dependent; the graph constrains normally closed spans.
Free text and escaped or truncated spans carry no KG guarantee.

`--max-model-len` and `--kv-cache-gib` configure the worker's context and fixed
KV cache. Budget model, index and request memory together. These example values
are configuration, not a memory-capacity recommendation.

## Requests and validation

Requests omitting `xgraph=1` remain unconstrained. Validate all anchor IDs before
starting a workload; select a named KG version in request extras when needed.
The [integration guide](vllm.md#request-settings) covers online requests,
forced openers, prefill replay and per-request status receipts.

The CPU tests execute the configuration and worker call with a stand-in runtime.
They do not load vLLM, validate native masks or establish real serving behavior.
Native oracle/engine agreement and a worker smoke require a GPU owner run.
