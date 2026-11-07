# Engine soundness contract

This is a design contract for the implemented path language. The oracle argument
below is separate from the engine's universal agreement obligation. CPU checks
alone do not discharge that obligation for CUDA or an LLM server.

## Scope and statement

Fix a served tokenizer, span syntax, valid KG index, KG version, anchor, start
rank, execution and answer-row caps, and policy settings for a sequence. The
index and version tables remain immutable while the sequence reads them.
Use `paths=True`, `relations="frontier"`, `escapes=False` and
`expressions=False`. Both union paths and committed paths are included.
Only requests admitted to the constraint engine, and tokens advanced under its
mask, are covered. A checked replayed prefill is subject to the same condition.
Free text outside spans has no grounding guarantee.

KG versions are registered at engine construction. Selecting a registered
version avoids rebuilding the base index; adding a version requires a new
engine/server, and adding labels requires rebuilding the index. This contract
does not establish hot ingestion or ingest-to-answer latency.

**Version isolation.** Every hop, degree lookup and answer binding reads the
request's one version. Registering other versions cannot add edges to that
reader. Union paths combine senses within the selected version, never edges
across registered versions. An answer witnessed only in the edge union of
incompatible versions must remain unreachable in each of them.

Version names must be nonempty strings; `None` is reserved for the base reader.
Oracle construction and native-table compilation reject invalid names before
decoding, so a registered edit mapping cannot overwrite the base graph entry.

At the slot API, call `begin`, then select the version once with `set_version`
before the first `sync`, opener or replay. `begin` and `reset` clear the version
to the base. The adapter restores the request's version after `begin`, before
replaying either a prefilled prompt tail or generated history on re-admission
after preemption. Callers must retain the original per-request arguments on
re-admission. Directly changing a slot's version during decoding is outside
the pinned-reader premise.

An explicit anchor must resolve to at least one writable entity-label row.
`Engine.anchor`, `Oracle.anchored_state` and native `set_reached` reject an empty
binding with an error; an empty list, unknown IDs or exclusively excluded labels
cannot turn an anchored request into an unanchored request. Anchor IDs have set
semantics: deduplicate them before expanding and sorting label rows, then apply
the answer-row cap. Repeated IDs cannot displace another entity's rows. Unknown
or unwritable IDs in a mixed anchor contribute no rows. Omitting the anchor is
the explicit way to request an unanchored start.

Ordinary generation records each replayed request's worker outcome in
`replay_status`: only `replayed` confirms admission. A `declined` or `unconfirmed`
outcome is a generation failure in every answer/plan scorer, regardless of the
raw completion's contents. The raw text and token costs remain in the output
record; the CLI writes those records and costs before exiting unsuccessfully.
Stored replayed-prefill records without a confirming receipt also fail closed.
A caller without a request status consumer receives an error on a declined
replay. No unmasked continuation receives constrained success credit.
Replay runs in the adapter's first `apply` call once prompt history is ready,
after `apply_staged_writes` admits the request. A caller that consumes outcomes
supplies `xgraph_replay_id` and reads its receipt through the worker RPC
`collect_replay_status`; the receipt records a decline even if the same request
is later re-admitted.
The processor and collector resolve one worker-local receipt store, including
when processor code is loaded separately. Callers register
`worker_extension_cls="xgraph.vllm.XGraphWorkerExtension"` at LLM construction;
ordinary generation and the smoke driver do this automatically. The collector
runs through the named worker method `xgraph_collect_replay_status`: vLLM's
default frontend Msgpack transport rejects a raw function object. Absent,
partial or invalid replies still remain unconfirmed.

The **writable KG** of version v contains the ID triples represented by that
version's adjacency. Start with the base triples, replace all objects of each
edited subject-relation pair, then omit triples whose subject, relation or
object has no writable label. An unwritable edit still removes the pair's old
objects. Aliases supply additional label rows, not additional ID edges. This is
the semantics constructed by `build_kg_index` in
[`index.py`](../src/xgraph/index.py), `build_kg_versions` and
`KGVersions.view` in [`kgversions.py`](../src/xgraph/kgversions.py), and
independently by `Oracle.__init__` in [`oracle.py`](../src/xgraph/oracle.py).

For every normally resolved positive-hop path, **each resolved endpoint** has a
witness edge sequence in this pinned version's writable KG. A witness starts at
an entity denoted by the written start label and admitted by the anchor or prior
binding, follows one edge per executed relation label, and ends at that endpoint.
Each edge's relation ID must be one denoted by that label. With committed paths,
the witness starts at the surviving committed start sense. Every entity ID in
the following normally resolved bound answer is such a witnessed endpoint.

This statement concerns resolved IDs. A homonymous printed label does not
uniquely identify an entity to a reader. It makes no assertion about question
relevance, answer correctness, truth outside the KG, unique paths, sampling
probabilities or eventual termination. Entity-only spans and expression
comparisons need different statements. Under vocabulary or successor relation
support, a relation plan is unexecuted and does not receive this guarantee.

## Witness invariant and implementation obligations

Maintain the invariant that every retained frontier entity, together with its
stored start rank, has a witness for the executed relation prefix. The witness
is a proof object; the engine does not store its edge sequence.

1. **Start.** `Oracle.initial_state`, `anchored_state` and `opened_state` retain
   the version and binding. `_candidates` and `_segment_candidates` admit indexed
   start rows, restricting a bound start to `state.reached`. `_next_segment`
   removes starts without outgoing triples, ranks them with `_best_connected`
   and caps them with `_kept`. Each retained start has a length-zero witness.
   `_observed_frontier` performs the corresponding start calculation when an
   observation commits the label. Native counterparts in
   [`engine.cu`](../csrc/engine.cu) include `candidate_in`, `ranked_starts` and
   the subject-separator branch of `advance_span`.
2. **Relation.** `_segment_candidates` admits relations supported by the live
   versioned frontier. `_frontier` maps the complete written label to all its
   indexed relation IDs, reads `_graph(state).from_subjects`, and appends only
   those edges. `_next_segment` installs this frontier after a separator; the
   path-closing branch of `_span_step` executes the final relation too. Appending
   an edge extends an existing witness. When endpoints merge, retaining the
   least incoming start rank retains a witness of that rank. Native
   `edges_of`/`object_slice` select the version's edges; `next_frontier`,
   `path_frontier` and `advance_span` perform the corresponding execution.
3. **Restriction and display.** `_kept` takes a subset, even at overflow.
   `_span_step`'s frontier-selection branch intersects the current frontier with
   owners of the selected label, retaining their ranks. DEFER preserves that
   frontier. `_observation`, `_menu_display` and `_after_observation` render or
   schedule displays; they do not add reachable entities. STOP, loop guards and
   the hop limit restrict continuation. Their native obligations live in
   `advance_span`, `prepare_observation`, `after_observation` and
   `residual_goal_stop`. These operations preserve the invariant.
4. **Closure and answer.** `_span_step` keeps the least surviving rank when
   committed paths are enabled, records the path endpoints in `ClosedSpan`, and
   sets `state.reached` to their first `MAX_REACHED` label rows. `_answers_path`
   keeps the next span entity-only; `_segment_candidates` admits only those
   rows and `_resolve` chooses an owner within them. A shared answer label cannot
   introduce owners outside the binding. Native `advance_span`, `close_span`
   and `reached_in` have the corresponding obligations. Forced answer prefixes,
   terminal displays and EOS handling add no edges.

For audit mode, apply the statement to the **resulting executed plan**, the
`ClosedSpan.labels` returned by `_audit_finish`, rather than just the original
printed blind plan. `_audit_step` restricts decisions to supported entities or
relations. KEEP preserves witnesses; SELECT takes a subset; DROP restores the
saved penultimate frontier and removes the last relation; APPEND and REPLACE
call `_frontier` from the terminal or penultimate frontier and update the plan.
Native `advance_audit` must preserve the same correspondence. Audit display
labels are not extra executed relations.

Induction on executed relations, with subset-preserving operations between
them, proves the witness invariant and the closure/answer statement for the
oracle specification. The argument depends on the named code paths satisfying
their obligations; it is not a machine-checked proof of either implementation.

## Engine/oracle agreement premise

Lift the oracle statement to the engine only if all of these hold:

- Initial states correspond, including the tokenizer and syntax, index and
  version tables, anchor/bound rows, caps, ranks and per-request policy settings.
- At **every reachable corresponding state**, the GPU mask equals
  `Oracle.allowed`, over the complete served vocabulary.
- Advancing any allowed token preserves state correspondence, including
  frontier entities and ranks, version, bindings, branch progress, audit edits
  and closed-span resolutions. Request admission, replay, slot moves, reset and
  draft-row handling must preserve the appropriate initial/history relation.
- The sampler applies the mask and advances the corresponding sequence with
  the sampled token; no unmasked fallback is treated as constrained output.

`Oracle.allowed` and `Oracle.advance` share `_step`, making acceptance and
advance consistent on the oracle side. Induction on token length under the
premises above gives equal finite accepted token languages and corresponding
resolutions. Mask equality without transition/resolution correspondence is
insufficient. Native entry points include `write_mask`/`mask_kernel`,
`row_mask_kernel` and `advance_state`/`advance_kernel` in `engine.cu`.

The agreement harness `run_agreement` in
[`tests/test_engine.py`](../tests/test_engine.py) compares complete token masks
and the public state projection through `expected_state`, `engine_state` and
`assert_agrees` at each sampled step. Fixed-span and seeded-walk cases exercise
versions (also against rebuilt indexes), anchors, committed starts, overflowing
frontiers, observations, selections, audit decisions, STOP, slot lifecycle and
draft rows. [`tests/test_vllm_adapter.py`](../tests/test_vllm_adapter.py) adds
adapter request/history checks with stand-ins.

The fixed `test_version_isolation_masks_and_transitions_equal_rebuilt_indexes`
and `test_version_isolation_rejects_the_union_only_answer` cases in the span
language seam use literal rebuilt KGs for the incompatible-world adversary,
inner-hop and terminal replacements, and pair deletion.
`test_version_isolation_mixed_slots_and_reset_equal_rebuilt_engines` adds native
mixed-slot and reset agreement. The adapter's
`test_version_isolation_adapter_replay_preemption_and_slot_reuse` and
`test_version_isolation_adapter_rejects_another_base_with_identical_shape`
exercise CPU stand-ins and separately queued CUDA parameters. A CPU stand-in
checks adapter binding and histories; it supplies no native agreement evidence.

These are finite checks of selected KGs, tokenizers, histories and policy
combinations. The public state projection is not every internal memory field;
future-mask and resolution checks give additional observable coverage. Shared
index construction can hide a common defect. Passing tests neither proves the
universal premise nor qualifies real serving, concurrency or hardware that was
not exercised. CPU-only skips provide no CUDA agreement evidence.

## Deliberate incompleteness

The frontier cap can discard the sole valid continuation of a writable graph
path. `test_frontier_cap_discards_the_only_valid_path_continuation` in
[`tests/test_oracle.py`](../tests/test_oracle.py) fixes `MAX_REACHED` at 64:
Start has 65 r-successors, B000 through B064, and only B064 has t to Goal.
All successors have the same start rank. The capped oracle drops B064, accepts
the one-hop path, and rejects Start-r-t-Goal in both union and committed modes.
The literal two-edge witness exists in the input KG. A process-local cap of 65
accepts the same trace and bound Goal answer; the source/index are unchanged.
This is intended coverage loss, not a soundness failure.

The answer cap counts label rows, so aliases can also remove otherwise reached
endpoints from answer support. Start caps, excluded labels, tokenization context,
the hop limit, selections, STOP and finite generation budgets impose further
coverage limits. A nonempty exact KG path does not imply that this language
admits it or that a locally allowed prefix has every desired completion.

`test_resolved_random_paths_and_bound_answers_have_versioned_edge_witnesses`
checks the oracle statement on seeded random KGs, with aliases, homonymous
entity/relation labels, anchors, union/committed paths and replacement versions
including unwritable edits. It searches the raw writable ID triples for actual
edge sequences independently of oracle frontier/ranking helpers and checks every
accepted answer label of sampled paths. This property test covers ordinary
frontier paths; the existing feature tests cover observation/audit behavior.
Neither establishes unrestricted path completeness or native agreement.
