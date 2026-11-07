# Runnable examples

Run these commands from the repository root after the
[CPU installation](../docs/installation.md#cpu).
The CPU examples use only Python, the local tokenizer library, and tiny graphs.
They train a demonstration tokenizer locally by default; no weights or network
access are needed. Their JSON output shows resolved IDs rather than model answers.

```bash
.venv/bin/python examples/toy_kg.py
.venv/bin/python examples/kg_versions.py
```

[toy_kg.py](toy_kg.py) builds and reloads an index over application string IDs,
walks entity, fact, escaped and relation-path spans, and rejects an answer outside
the reached frontier. Output goes to `target/examples/toy-index`.
[kg_versions.py](kg_versions.py) interleaves base and edited readers and round-trips
a persisted version. Output goes to `target/examples/versions`. Its Q/P IDs are
application-assigned numbers for the disk format, with no Wikidata dependency.

The optional [RDF example](rdf_graph.py) also needs the `rdf` extra:

```bash
uv sync --locked --no-default-groups --extra rdf \
    --config-setting cmake.define.CMAKE_CUDA_COMPILER=NOTFOUND
uv run --no-sync python examples/rdf_graph.py --out target/examples/rdf
```

It imports local RDF triples, restores their IRIs after directory indexing,
loads content-pinned replacements and pair deletion, and exports a resolved
fact through RDFLib. Choose a fresh output directory for each run.

To use your model's tokenizer without loading its weights:

```bash
export MODEL_DIR=/path/to/local/model
.venv/bin/python examples/toy_kg.py --tokenizer "$MODEL_DIR/tokenizer.json" \
    --out target/examples/model-index
.venv/bin/python examples/kg_versions.py --tokenizer "$MODEL_DIR/tokenizer.json"
```

The tokenizer must use a supported BPE vocabulary; see
[tokenization](../docs/concepts.md#tokenization-and-the-kg-index).
The examples deliberately encode delimiters and prefixed labels in separate
pieces, preserving the indexed token boundaries.

[test_documentation_examples.py](../tests/test_documentation_examples.py) runs
the CPU scripts without network, checking concrete graph answers and replacement
semantics. The public [source smoke](../tests/test_source_smoke.py) provides a smaller
standard-library onboarding check; see [development](../DEVELOPMENT.md).

Additional CPU examples are [quick_start.py](quick_start.py), the exact README
program; [string_versions.py](string_versions.py), application-ID replacement
and deletion; [witnesses.py](witnesses.py), fixed-plan endpoint support; and
[sampling_loop.py](sampling_loop.py), masking synthetic logits before sampling.

## GPU generation

[vllm_generation.py](vllm_generation.py) **requires an NVIDIA GPU**, CUDA-built
xgraph, compatible vLLM, and model weights. It loads the model and registers the
GPU logits processor. Follow the [vLLM installation and integration
guide](../docs/guides/vllm.md) before running:

```bash
python examples/vllm_generation.py --model "$MODEL_DIR" \
    --index target/examples/model-index --tokenizer "$MODEL_DIR/tokenizer.json"
```

Use the index built with the exact served tokenizer. The example assumes the
toy graph's IDs and labels. It teaches span syntax in the prompt; constraints
apply only after the model opens a span. Its memory settings are example
configuration, not a capacity recommendation. Its `--dry-run` path and a stand-in worker call run in the CPU tests; real
model generation is a separate GPU check.
