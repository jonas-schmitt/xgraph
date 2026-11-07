# Development

This public source profile contains the library, tiny local examples, public
documentation and a curated synthetic test subset. Research datasets, model
weights, experiment queues and historical results are distributed separately.

## CPU environment

Use Python 3.12 or newer, `uv`, a C++ compiler and an internet connection for the
initial dependency installation. No model, private dataset or GPU is needed for
the CPU examples. From the source root:

```bash
uv sync --locked --no-default-groups --extra docs --extra rdf \
    --config-setting cmake.define.CMAKE_CUDA_COMPILER=NOTFOUND
```

The compiler is a CMake configuration prerequisite; the CPU profile does not
build the CUDA engine. The default development dependency group includes GPU
tools, so use `--no-default-groups` for this profile. See
[installation](docs/installation.md) for a CUDA build and cache transitions.

## Testing and validation layers

Run the installed-package onboarding smoke from the source root:

```bash
uv run --no-sync python -I tests/test_source_smoke.py
```

It runs the README program and tiny toy/version examples in temporary working
directories, checking literal graph identifiers. It uses only the standard
library as its test runner and installed core dependencies. It does not import
a model, use the network, or exercise CUDA. The manual CI workflow runs this
same CPU check; it does not publish a release.

Build the public documentation:

```bash
uv run --no-sync python -m sphinx -W --keep-going -b html docs target/docs/html
```

The curated `tests/test_oracle.py`, `tests/test_engine.py`,
`tests/test_vllm_adapter.py` and `tests/test_documentation_examples.py` retain
synthetic correctness checks. Their shared fixtures also use pytest, torch,
transformers and RDFLib from the development environment. CUDA-marked cases
need a built native extension and GPU. Choose a focused test in that environment;
passing the onboarding smoke does not qualify those cases or a serving runtime.

Research scoring, latency protocols and the full development gate are outside
this profile. Keep their results distinct from library onboarding and
documentation validation. See [reproduction scope](docs/reproducing.md).

## Troubleshooting

If imports fail, repeat the install command in the intended Python environment.
The smoke uses Python isolated mode and a temporary working directory so it
cannot rely on a checkout-local `PYTHONPATH`.

After a CPU build, explicitly replace CMake's cached
`CMAKE_CUDA_COMPILER=NOTFOUND` when building for CUDA; follow the
[vLLM prerequisites](docs/guides/vllm.md#build-and-prerequisites). A compatible
worker environment, model tokenizer and saved KG index are separate requirements.

See [user troubleshooting](docs/guides/troubleshooting.md) and the
[contribution guide](CONTRIBUTING.md) for a minimal bug report.
