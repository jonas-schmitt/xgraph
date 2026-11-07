# Installation

Install from a source checkout with Python 3.12 or newer. The core library uses
NumPy and the Hugging Face `tokenizers` package; it does not install a model or
vLLM. A C++ compiler is needed for CMake configuration even for a CPU build.

## CPU

From the repository root, with uv:

```bash
uv sync --locked --no-default-groups \
    --config-setting cmake.define.CMAKE_CUDA_COMPILER=NOTFOUND
uv run --no-sync python examples/quick_start.py
```

The explicit CMake setting disables CUDA compilation even on a GPU host.
`--no-default-groups` omits contributor tools and training dependencies. Use
`uv run --no-sync` afterwards to keep that library-only environment.

With pip inside an existing virtual environment:

```bash
python -m pip install . --config-settings=cmake.define.CMAKE_CUDA_COMPILER=NOTFOUND
```

CPU support covers KG validation, index building, the reference oracle, versions
and offline execution. The CPU oracle is intended for small examples and
correctness checks; it scans the vocabulary to compute each mask.

## Optional RDF support

```bash
uv sync --locked --no-default-groups --extra rdf \
    --config-setting cmake.define.CMAKE_CUDA_COMPILER=NOTFOUND
```

This installs the pinned RDFLib adapter dependency. Read the
[RDF tutorial](guides/rdflib.md) for projection and export limits.

## GPU engine

Use Linux, an NVIDIA GPU, a supported CUDA toolkit, CMake and `nvcc`. The native
build targets are declared in [CMakeLists.txt](../CMakeLists.txt). Check that
`command -v nvcc` finds the installed compiler before building:

```bash
uv sync --locked --no-default-groups --reinstall-package xgraph \
    --config-setting "cmake.define.CMAKE_CUDA_COMPILER=$(command -v nvcc)"
```

After a CPU installation, this explicitly replaces the cached
`CMAKE_CUDA_COMPILER=NOTFOUND` value. Omitting the old setting alone does not
reset the persistent CMake cache. `EngineUnavailable` means the installed
package has no native engine.

For generation, install/build xgraph in the same environment as the vLLM worker.
The adapter targets the repository's Model Runner V2 interface; see
[vLLM compatibility and setup](guides/vllm.md#build-and-prerequisites).
GPU tables share device memory with model weights, KV cache and request state.

## Tokenizers and saved artifacts

`ServedTokenizer` accepts byte-level BPE and SentencePiece-style BPE with byte
fallback. A `tokenizer.json` is sufficient for index building; weights and
remote repository code are unnecessary. Other tokenizer families raise
`UnsupportedTokenizer`.

Build a separate index for each tokenizer and span syntax. The tokenizer
fingerprint is checked at oracle/engine construction. Keep source graph IDs
stable across edits and preserve the index's recorded identity. Continue with
the [quick start](getting-started.md).
