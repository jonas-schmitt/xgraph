# Integrating another engine

Keep one tokenizer-bound KG index per graph/tokenizer/syntax combination.
Share its immutable tables among requests. Give each sequence its own constraint
state, anchor and pinned KG version.

## Sampling loop

Before sampling, compute the allowed set and set all other logits to negative
infinity. Sample from the masked distribution, then advance by the sampled
token exactly once. The following complete CPU example uses synthetic scores
in place of model logits:

```{literalinclude} ../../examples/sampling_loop.py
:language: python
```

The example deliberately gives a disallowed token the highest raw score inside
the span. Masking removes it and the graph's intended fact still resolves.
It is a reference loop over the scalar oracle; it is not a CPU generation
adapter or a model-speed measurement.

## Request lifecycle

- Start a fresh state or reset the reused slot before each request. Register
  versions during setup and select the version before advancing any token.
- Skip ordinary prompt tokens. If the prompt ends with a forced span opener,
  use `opened_state` or the native `open_span` operation. A replayed prefill must be
  validated in the correct anchor/version and have a per-request success receipt.
- Consume each newly committed output token once before the next mask. Keep
  request IDs separate from batch row positions; rows can move or disappear.
- Keep speculative draft state separate from committed state. Acceptance,
  rollback, preemption and resume must preserve the exact committed history.
- Check completion and escapes. If every permitted logit is nonfinite, stop
  with an explicit request failure. An empty mask is not permission to sample
  outside it. A declined replay gets no constrained-result credit.

## Native GPU interface

`Engine` owns per-slot GPU state. `begin` resets slots; `sync` advances their
committed token history; `apply_mask` masks logits through the DLPack tensor
interface, including scratch state for draft positions. The native
`fill_mask`/`mask_host` pair is useful for diagnostics, but copying masks to the
host at every step changes serving cost. See the [runtime API](../api/runtime.md)
for actual tensor shapes and stream arguments.

The model's logits width may include padding beyond `tokenizer.vocab_size`.
Respect vocabulary IDs, special tokens, EOS configuration, device, contiguity,
stream ownership and tensor lifetime. Validate those boundaries before serving.

## Validate an adapter

Compare every mask and transition against the independent oracle on fixed
synthetic spans and seeded token walks. Include aliases, homonyms, escapes,
bound answers, versions, slot reuse, prompt skipping, invalid anchors, replay
declines and speculative acceptance. CPU stand-ins validate host callbacks;
native oracle/engine agreement and a real worker smoke are additional checks.
The existing [adapter tests](../../tests/test_vllm_adapter.py) and
[engine tests](../../tests/test_engine.py) are the reference cases.
