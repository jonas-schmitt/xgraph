# xgraph

**Constrain LLM names and facts to your knowledge graph.**

## The idea

An LLM can invent names and facts missing
from your knowledge graph (KG).

[XGrammar](https://github.com/mlc-ai/xgrammar)
restricts tokens to a grammar. xgraph restricts
them to your KG inside marked text called
*spans*, like `⟦ Atlas⟧`.
Text outside spans stays free.

Each marked fact is checked as it is written,
without a lookup after generation.

A *resolved span* matches your graph when it closes,
but does not guarantee truth, relevance or all possible answers
([details](docs/concepts.md#guarantees-and-contract)).

## How it works

Take this toy graph:

```text
 Atlas
   |
   | creator
   |
   v
  Ada
```

1. **Compile for your model's tokenizer.**
   xgraph turns the graph into token lookup
   tables, called a *KG index*.

2. **Open a span.** The model writes `⟦`
   to start the part you want checked.

3. **Allow only valid next tokens.** After `⟦ Atlas · creator ·`,
   only tokens that continue `Ada` are allowed for the name.
   Closing the span gives `⟦ Atlas · creator · Ada⟧`.

4. **Let the graph find the answer.** In *path mode*, the model writes
   a starting name followed by relations: `⟦ Atlas · creator⟧`.
   xgraph follows `creator` to Ada during decoding.
   The next answer span can only name `⟦ Ada⟧`.

[Get started](docs/getting-started.md) · [Documentation](docs/index.md) ·
[Python API](docs/api.md) · [Examples](examples/README.md)

## Features

- **Names and facts:** check marked text while leaving the surrounding prose free.
  An escape marker lets a span finish unresolved when the KG lacks an answer.
- **Paths:** follow relations without generating intermediate entity names.
- **Graph updates:** choose a fixed [set of graph edits](docs/guides/kg-versions.md)
  for each request, without rebuilding shared lookup tables.
- **CPU and GPU:** try the CPU reference checker or use the GPU engine inside
  an inference server.
- **Your data:** supply your own IDs, names and facts, or import RDF with the
  optional [RDFLib adapter](docs/guides/rdflib.md).

## Install

From a source checkout with Python 3.12 or newer, uv and a C++ compiler,
install the CPU library:

```bash
uv sync --locked --no-default-groups \
    --config-setting cmake.define.CMAKE_CUDA_COMPILER=NOTFOUND
```

CPU use needs no model, dataset or CUDA. For CUDA builds and the vLLM worker
environment, follow [installation](docs/installation.md). Source installation
is the supported distribution route.

## Quick start

Run this on CPU after installing. It builds our Atlas graph and a tiny tokenizer.
The CPU checker, called the *oracle*, checks a supplied fact one token at a time
and returns the graph's IDs:

```python
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

import xgraph as xg

alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
raw = Tokenizer(models.BPE({char: i for i, char in enumerate(alphabet)}, []))
raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
raw.decoder = decoders.ByteLevel()
tokenizer = xg.ServedTokenizer(raw)
kg = xg.KnowledgeGraph(
    (xg.Entity("atlas", ("Atlas",)), xg.Entity("ada", ("Ada",))),
    (xg.Relation("creator", ("creator",)),),
    (("atlas", "creator", "ada"),),
)
index = xg.build_kg_index(kg, tokenizer, xg.SpanSyntax())
oracle = xg.Oracle(index, tokenizer, index.span_syntax)
state = oracle.initial_state()
for piece in ("⟦", " Atlas", " ·", " creator", " ·", " Ada", "⟧"):
    for token in tokenizer.encode(piece):
        state = oracle.advance(state, token)
print(state.last_span.triples)  # (('atlas', 'creator', 'ada'),)
```

This is [examples/quick_start.py](examples/quick_start.py). Run it with
`uv run --no-sync python examples/quick_start.py`. For a model, build the index
with that model's actual tokenizer. Continue with [paths and bound answers](docs/guides/paths.md)
or [bring your own KG](docs/guides/bring-your-own-kg.md).

## Integrations

| Integration | Support |
|---|---|
| [vLLM](docs/guides/vllm.md) | GPU logits processor for Model Runner V2; requests opt in |
| [RDFLib](docs/guides/rdflib.md) | Optional RDF projection, string-ID edits and RDF export |
| [Another engine](docs/integration/another-engine.md) | Mask/advance contract and CPU reference loop |
| Transformers / SGLang | Generation adapters are not implemented |

The APIs are experimental. See [concepts](docs/concepts.md),
[versions and edits](docs/guides/kg-versions.md), [performance notes](docs/performance.md),
[FAQ](docs/guides/troubleshooting.md), and [contributing](CONTRIBUTING.md).

## License

Project-owned code is licensed under [Apache License 2.0](LICENSE)
(`Apache-2.0`), including commercial use under its terms. Vendored GWalk code
retains its separate CC BY 4.0 license; datasets, model weights and dependencies
retain their own terms. See [licensing](docs/licensing.md) for redistribution
and attribution details.

## Security

Follow the [security policy](SECURITY.md) to report vulnerabilities privately.
Keep credentials and private data out of public issues.

## Citation

Use the software metadata in [CITATION.cff](CITATION.cff) and record the exact
source commit used in your work. The paper citation remains withheld during
anonymous review; see [citation scope](docs/citation.md). No DOI is claimed.
