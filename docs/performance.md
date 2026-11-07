# Measuring performance

Measure the costs your application pays, with correctness and completion status
beside them. Store local reports under `target/`. This page describes
methods, not benchmark results. Historical experiment artifacts are outside this
source profile; see [measurement scope](reproducing.md#systems-measurements).

## Index setup and memory

Time the complete index build, including input decoding, tokenization, sorting,
adjacency expansion and writing. Record the source triple/label inventory,
alias policy, tokenizer fingerprint, syntax, input hashes, command and commit.
Measure peak resident memory and disk output, not only NumPy array sizes.
Directory building is bounded in chunks but still retains label arrays and
subject runs; in-memory building materializes the input and edges.

Measure engine construction separately: host-table preparation, version-table
building, device allocation and upload. GPU-resident KG tables compete with
model weights, KV cache, and per-request state. Each object alias expands
adjacency; report the alias policy and version inventory. The CPU oracle builds
plain graph sets and is a correctness reference, not a scaling benchmark.

Use `kg_index_fingerprint(directory)` and the tokenizer fingerprint to identify
the exact compiled input. `xgraph.provenance.provenance()` records file hashes,
command, commit, dirty status and package versions for local reports.

## Engine step diagnostic — GPU required

[`xgraph.bench`](../src/xgraph/bench.py) replays token streams through engine
sync and masking. Use `python -m xgraph.bench --help` for its actual CLI. Supply the saved index,
served tokenizer, constrained output streams, model logits width, batch sizes
and a local report destination.

Match the generation's paths, escapes, ranking, versions and opener options;
consult `python -m xgraph.bench --help`. Logits width is the model's actual
output width, including padding, rather than an assumed tokenizer vocabulary
size. Reject replay errors. Record stream assignment and finished-slot padding,
and include wide prefixes, ambiguous labels, aliases and high-degree subjects.
Synchronize device measurements appropriately. Report request setup separately
from per-step sync/mask time and construction.

This diagnostic excludes model forward passes, server scheduling, prompt
prefill and client waiting. It cannot establish end-to-end latency or throughput.
It also does not measure a sampler-bypass speedup: a label continuation is not
forced while an escape or another branch remains allowed.

## End-to-end serving

Use identical model weights/revision, tokenizer, prompts, request order,
sampling, seeds, token limits, concurrency and cache conditions across paired
arms. Compare the same workload with and without the constraint. State whether
the interval includes startup, warm-up, queueing, prefill, CPU/tool work and
final delivery. Measure an idle dedicated device for cost claims; retain failed
attempts and unsuccessful answers in accounting.

Report whole-question latency distributions, visible time to first token,
throughput, prompt/generated tokens, model/tool calls, CPU time, peak host/device
memory and completion/escape/truncation outcomes. Forced observations and menus
are generated tokens; count them once. A multi-call policy must include every
consumed call and CPU stage, even when it retains an earlier answer.

A serving harness must retain matched cache, repetition, admission and
provenance records. See the [timing scope](reproducing.md#controlled-serving-timing-protocol).
Repeat matched trials with randomized arm order; separate repeatability noise
from uncertainty in sampled questions. Do not extrapolate oracle timing or a
single engine-step measurement into a serving claim.

## Correctness alongside cost

Replay fixed synthetic spans and version edits through the oracle, then compare
GPU masks and resolutions at every step using the existing engine tests.
Record cap events on representative plans. Keep correctness validation,
microbenchmarks, serving measurements and answer-quality experiments distinct.
The [CPU examples](../examples/README.md) verify onboarding behavior and do not
claim GPU performance. See [development validation layers](../DEVELOPMENT.md#testing-and-validation-layers).
