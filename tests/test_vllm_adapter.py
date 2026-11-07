"""vLLM adapter: the engine as a Model Runner V2 logits processor, driven like vLLM drives it.

vLLM is not installed in the development environment, so these tests stand in for
its request state and per-step context with objects of the same shape (see
vllm/v1/worker/gpu/sample/logits_processor/interface.py at v0.31.0): one row of
token history per request slot, int32 lengths on the GPU, and a row-to-slot map.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from xgraph.index import build_kg_index, save_kg_index
from xgraph.oracle import Oracle
from xgraph.syntax import SpanSyntax
from xgraph.tokenizer import ServedTokenizer

torch = pytest.importorskip("torch")

from xgraph.vllm import XGraphLogitsProcessor  # noqa: E402  (after the torch check)

SLOTS = 4
WIDTH = 64  # tokens of history per slot


@pytest.fixture
def served(toy_tokenizer):
    return ServedTokenizer(toy_tokenizer)


@pytest.fixture
def cuda():
    if shutil.which("nvidia-smi") is None or os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        pytest.skip("no visible NVIDIA GPU for the adapter tests")


@pytest.fixture
def vllm(request, toy_kg, toy_tokenizer, served, tmp_path, monkeypatch):
    """A processor as vLLM builds it, and the request state it reads."""
    device = getattr(request, "param", "cuda")
    if device == "cuda":
        request.getfixturevalue("cuda")
    if device == "cpu":
        monkeypatch.setattr("xgraph.engine.Engine", ReplayOracleEngine)
        monkeypatch.setattr(XGraphLogitsProcessor, "_stream", lambda self: 0)
    save_kg_index(build_kg_index(toy_kg, served, SpanSyntax()), tmp_path / "index")
    toy_tokenizer.save(str(tmp_path / "tokenizer.json"))
    monkeypatch.setenv("XGRAPH_KG_INDEX", str(tmp_path / "index"))
    monkeypatch.setenv("XGRAPH_TOKENIZER", str(tmp_path / "tokenizer.json"))
    gpu = torch.device(device)
    state = SimpleNamespace(
        device=gpu,
        max_num_reqs=SLOTS,
        vocab_size=served.vocab_size + 11,  # model vocabularies are padded
        all_token_ids=SimpleNamespace(
            gpu=torch.zeros((SLOTS, WIDTH), dtype=torch.int32, device=gpu)
        ),
        prompt_len=SimpleNamespace(gpu=torch.zeros(SLOTS, dtype=torch.int32, device=gpu)),
        prefill_len=SimpleNamespace(gpu=torch.zeros(SLOTS, dtype=torch.int32, device=gpu)),
        total_len=SimpleNamespace(gpu=torch.zeros(SLOTS, dtype=torch.int32, device=gpu)),
    )
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=object()), state)
    return processor, state


class ReplayOracleEngine:
    """CPU stand-in for isolated replay lifecycle tests, without native agreement evidence."""

    def __init__(self, index, tokenizer, syntax, batch, **options):
        from xgraph.kgversions import read_kg_versions

        self.versions = options.get("versions")
        self.oracle = Oracle(
            index,
            tokenizer,
            syntax,
            paths=options["paths"],
            committed_paths=options.get("committed_paths", False),
            derivation=options.get("derivation", False),
            escapes=options.get("escapes", True),
            versions=read_kg_versions(os.environ["XGRAPH_KG_VERSIONS"], index=index)
            if self.versions is not None
            else None,
            end_after_answer=options["end_after_answer"],
            eos_token_ids=options["eos_token_ids"],
        )
        self.states, self.cursors, self.errors = {}, {}, set()

    def set_version(self, slot, version, stream):
        self.versions.number(version)
        self.states[slot] = replace(self.states[slot], version=version)

    def anchor(self, slot, entities, stream):
        binding = self.oracle.anchored_state(entities)
        self.states[slot] = replace(self.states[slot], reached=binding.reached)

    def open_span(self, slot, stream):
        self.states[slot] = self.oracle.opened_state(self.states[slot])

    def begin(self, slots, stream):
        for slot in slots.tolist():
            self.states[slot] = self.oracle.initial_state()
            self.cursors.pop(slot, None)
            self.errors.discard(slot)

    def end(self, slots, stream):
        for slot in slots.tolist():
            self.states.pop(slot, None)

    def sync(self, slots, history, start, stop, stream):
        from xgraph.oracle import TokenNotAllowed

        for slot in slots.tolist():
            if slot not in self.states or slot in self.errors:
                continue
            cursor = self.cursors.get(slot, int(start[slot]))
            for token in history[slot, cursor : int(stop[slot])].tolist():
                try:
                    self.states[slot] = self.oracle.advance(self.states[slot], token)
                except TokenNotAllowed:
                    self.errors.add(slot)
                    break
            self.cursors[slot] = int(stop[slot])

    def states_host(self, count):
        from xgraph.engine import PHASE_ERROR

        return {"phase": [PHASE_ERROR if slot in self.errors else 0 for slot in range(count)]}

    def apply_mask(self, logits, slots, stream):
        for row, slot in enumerate(slots.tolist()):
            if slot in self.states:
                mask = torch.ones(logits.shape[1], dtype=torch.bool)
                mask[list(self.oracle.allowed(self.states[slot]))] = False
                logits[row, mask] = -float("inf")


def params(**extra_args):
    return SimpleNamespace(extra_args=extra_args or None)


@pytest.mark.parametrize("vllm", ["cpu"], indirect=True)
def test_relation_ranks_are_staged_per_slot_and_legacy_callbacks_stay_identical(vllm):
    processor, _ = vllm
    calls = []
    processor._menu = 5
    processor._engine = SimpleNamespace(
        begin=lambda slots, stream: calls.append(("begin", slots.tolist())),
        end=lambda slots, stream: calls.append(("end", slots.tolist())),
        set_relation_question=lambda slot, question, stream, **kw: calls.append(
            (slot, question, kw)
        ),
    )
    ranks = [["P26", 0], ["P19", 1]]
    processor.add_request(
        2, params(xgraph=1, xgraph_question="Dense?", xgraph_relation_ranks=ranks)
    )
    processor.add_request(0, params(xgraph=1, xgraph_question="Lexical?"))
    assert calls == []
    processor.apply_staged_writes()
    assert calls == [("begin", [2, 0]), (2, "Dense?", {"ranked": ranks}), (0, "Lexical?", {})]
    calls.clear()
    # Slot reuse and repeated callbacks cannot leak the prior dense list.
    processor.add_request(
        2, params(xgraph=1, xgraph_question="Dense?", xgraph_relation_ranks=ranks)
    )
    processor.add_request(2, params(xgraph=1, xgraph_question="Again?"))
    processor.apply_staged_writes()
    assert calls == [("begin", [2, 2]), (2, "Again?", {})]


@pytest.mark.parametrize(
    "ranks, message",
    [
        (None, "list"),
        (["P1"], "pair"),
        ([["P19", 64]], "rank must"),
        ([["P19", -1]], "rank must"),
        ([["P19", True]], "rank must"),
        ([["P19", 0.5]], "rank must"),
        ([["P19", 0], ["P19", 1]], "duplicate"),
        ([["unknown", 0]], "unknown relation"),
        ([[f"P{n}", 0] for n in range(65)], "at most 64"),
    ],
)
@pytest.mark.parametrize("vllm", ["cpu"], indirect=True)
def test_relation_ranks_reject_malformed_payloads_even_without_menus(vllm, ranks, message):
    processor, _ = vllm
    with pytest.raises(ValueError, match=message):
        processor.add_request(0, params(xgraph=1, xgraph_relation_ranks=ranks))


@pytest.mark.parametrize("vllm", ["cpu"], indirect=True)
def test_staged_admissions_preserve_batches_optional_fields_and_repeated_callbacks(vllm):
    processor, _ = vllm
    calls = []

    def record(name):
        def call(*args):
            values = [
                value.tolist() if isinstance(value, torch.Tensor) else value for value in args
            ]
            calls.append((name, *values))

        return call

    processor._engine = SimpleNamespace(
        **{
            name: record(name)
            for name in (
                "begin",
                "end",
                "set_path_options",
                "anchor",
                "set_version",
                "set_stop",
                "set_relation_question",
                "open_span",
            )
        }
    )
    processor._audit = True
    processor._stop = "residual-goal"
    assert processor.add_request(
        2,
        params(
            xgraph=1,
            xgraph_anchor=["Q1"],
            xgraph_kg_version="edit",
            xgraph_open=1,
            xgraph_question="First?",
        ),
    )
    assert not processor.add_request(1, params())
    assert processor.add_request(0, params(xgraph=1, xgraph_anchor=[], xgraph_question="Empty?"))
    # Repeated callbacks retain the earlier anchor when the argument is absent,
    # but replace version, question and opener settings before the flush.
    assert processor.add_request(2, params(xgraph=1, xgraph_question="Last?", xgraph_stop=0))
    assert calls == []
    processor.apply_staged_writes()
    assert calls[:2] == [("begin", [2, 0, 2], 0), ("end", [1], 0)]
    for slot, expected in (
        (
            0,
            [
                ("anchor", 0, [], 0),
                ("set_stop", 0, "Empty?", 0),
                ("set_relation_question", 0, "Empty?", 0),
            ],
        ),
        (2, [("anchor", 2, ["Q1"], 0), ("set_relation_question", 2, "Last?", 0)]),
    ):
        assert [call for call in calls[2:] if call[1] == slot] == expected
    calls.clear()
    processor.apply_staged_writes()
    assert calls == []
    assert processor.add_request(2, params(xgraph=1, xgraph_question="Reuse?"))
    processor.apply_staged_writes()
    assert calls == [
        ("begin", [2], 0),
        ("set_stop", 2, "Reuse?", 0),
        ("set_relation_question", 2, "Reuse?", 0),
    ]


@pytest.mark.parametrize("vllm", ["cpu"], indirect=True)
def test_failed_admission_flush_preserves_completed_stages_and_retry_order(vllm):
    processor, _ = vllm
    processor._request_path_options = True
    calls = []

    def anchor(slot, entities, stream):
        calls.append(("anchor", slot, entities))
        raise ValueError("empty anchor")

    processor._engine = SimpleNamespace(
        begin=lambda slots, stream: calls.append(("begin", slots.tolist())),
        end=lambda *args: None,
        set_path_options=lambda slot, options, stream: calls.append(("options", slot)),
        anchor=anchor,
    )
    processor.add_request(0, params(xgraph=1, xgraph_anchor=[]))
    processor.add_request(1, params(xgraph=1, xgraph_path_options={}))
    for _ in range(2):
        with pytest.raises(ValueError, match="empty anchor"):
            processor.apply_staged_writes()
    assert calls == [("begin", [0, 1]), ("options", 1), ("anchor", 0, []), ("anchor", 0, [])]


def admit(processor, state, slot, prompt, on, **extra):
    """vLLM admitting a request into a slot: history and lengths, then add_request."""
    state.all_token_ids.gpu[slot].zero_()
    state.all_token_ids.gpu[slot, : len(prompt)] = torch.tensor(prompt, dtype=torch.int32)
    for name in ("prompt_len", "prefill_len", "total_len"):
        getattr(state, name).gpu[slot] = len(prompt)
    return processor.add_request(slot, params(xgraph=1, **extra) if on else params())


def step(processor, state, slots):
    """One decoding step: mask fresh logits for these slots, in batch order."""
    mapping = torch.tensor(slots, dtype=torch.int32, device=state.device)
    ctx = SimpleNamespace(idx_mapping=mapping, expanded_idx_mapping=mapping)
    logits = torch.randn((len(slots), state.vocab_size), device=state.device)
    original = logits.clone()
    return original, processor.apply(logits, ctx)


def sample(state, slot, token):
    """vLLM committing a sampled token to a slot's history."""
    at = int(state.total_len.gpu[slot])
    state.all_token_ids.gpu[slot, at] = token
    state.total_len.gpu[slot] = at + 1


def allowed(served, logits_row) -> set[int]:
    return set(torch.nonzero(~torch.isneginf(logits_row)).flatten().tolist()) & set(
        range(served.vocab_size)
    )


@pytest.mark.parametrize("order", ["identity", "reversed", "random"])
def test_terminal_adapter_forces_readback_and_copies_speculative_and_prefill_state(
    vllm, tmp_path, monkeypatch, order
):
    from tests.conftest import byte_level_tokenizer
    from tests.test_oracle import audit_world

    _, index, served, syntax = audit_world()
    _, state = vllm
    save_kg_index(index, tmp_path / "terminal-index")
    byte_level_tokenizer([]).save(str(tmp_path / "terminal-tokenizer.json"))
    monkeypatch.setenv("XGRAPH_KG_INDEX", str(tmp_path / "terminal-index"))
    monkeypatch.setenv("XGRAPH_TOKENIZER", str(tmp_path / "terminal-tokenizer.json"))
    monkeypatch.setenv("XGRAPH_PATHS", "1")
    monkeypatch.setenv("XGRAPH_OBSERVE_TERMINAL", "2")
    monkeypatch.setenv("XGRAPH_TERMINAL_ORDER", order)
    monkeypatch.setenv("XGRAPH_RELEASE_TOP", "1")
    monkeypatch.setenv("XGRAPH_RELEASE_MASS", "0.99")
    monkeypatch.setenv("XGRAPH_ESCAPE_BIAS", "1000")
    monkeypatch.setitem(
        sys.modules,
        "vllm.tokenizers",
        SimpleNamespace(
            cached_tokenizer_from_config=lambda config: SimpleNamespace(eos_token_id=0)
        ),
    )
    config = SimpleNamespace(
        speculative_config=object(),
        model_config=SimpleNamespace(try_get_generation_config=lambda: {"eos_token_id": 0}),
    )
    state.vocab_size = served.vocab_size + 11
    state.all_token_ids.gpu = torch.zeros((SLOTS, 512), dtype=torch.int32, device="cuda")
    processor = XGraphLogitsProcessor(config, state)
    oracle = Oracle(
        index,
        served,
        syntax,
        paths=True,
        observe_terminal=2,
        terminal_order=order,
        eos_token_ids=[0],
    )
    expected = oracle.initial_state()
    assert admit(processor, state, 0, [], True)
    assert not admit(processor, state, 1, [], False)
    processor.apply_staged_writes()
    for token in served.encode("⟦ Start · via⟧"):
        sample(state, 0, token)
        expected = oracle.advance(expected, token)
    original, logits = step(processor, state, [1, 0])
    assert torch.equal(original[0], logits[0])
    assert allowed(served, logits[1]) == oracle.allowed(expected)
    display = "Gamma ; Beta" if order == "identity" else "Beta ; Gamma"
    drafts = served.encode(f"\nEndpoints: ⟦ {display}⟧.\nAnswer: ⟦ Beta⟧")
    rows = len(drafts) + 1
    mapping = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ctx = SimpleNamespace(
        idx_mapping=mapping[:1],
        expanded_idx_mapping=mapping,
        expanded_local_pos=torch.arange(rows, dtype=torch.int32, device="cuda"),
        input_ids=torch.tensor([0, *drafts], dtype=torch.int32, device="cuda"),
    )
    logits = processor.apply(torch.zeros((rows, state.vocab_size), device="cuda"), ctx)
    copied = expected
    for row in range(rows):
        assert allowed(served, logits[row]) == oracle.allowed(copied)
        if row < len(drafts):
            copied = oracle.advance(copied, drafts[row])
    _, logits = step(processor, state, [0])
    assert allowed(served, logits[0]) == oracle.allowed(expected)
    # Reset/replay in the middle of the forced payload; copied drafts left it untouched.
    prefill = f"⟦ Start · via⟧\nEndpoints: ⟦ {display[0]}"
    assert admit(processor, state, 0, served.encode(prefill), True, xgraph_prefill=prefill)
    processor.apply_staged_writes()
    replayed = oracle.initial_state()
    for token in served.encode(prefill):
        replayed = oracle.advance(replayed, token)
    _, logits = step(processor, state, [0])
    assert allowed(served, logits[0]) == oracle.allowed(replayed)
    assert processor.replays["replayed"] == 1


def test_request_path_options_adapter_prefill_drafts_and_slot_reuse(vllm, tmp_path, monkeypatch):
    from tests.conftest import byte_level_tokenizer
    from tests.test_oracle import observed_world

    kg, served, syntax = observed_world()
    index = build_kg_index(kg, served, syntax)
    save_kg_index(index, tmp_path / "mixed-index")
    byte_level_tokenizer([]).save(str(tmp_path / "mixed-tokenizer.json"))
    monkeypatch.setenv("XGRAPH_KG_INDEX", str(tmp_path / "mixed-index"))
    monkeypatch.setenv("XGRAPH_TOKENIZER", str(tmp_path / "mixed-tokenizer.json"))
    for key, value in dict(PATHS=1, MENU=2, OBSERVE=1, OBSERVE_SELECT=1,
                           OBSERVE_ANSWER=1, OBSERVE_REQUIRED=1, OBSERVE_NO_LOOP=1,
                           END_AFTER_ANSWER=1, REQUEST_PATH_OPTIONS=1).items():  # fmt: skip
        monkeypatch.setenv("XGRAPH_" + key, str(value))
    monkeypatch.setitem(sys.modules, "vllm.tokenizers", SimpleNamespace(
        cached_tokenizer_from_config=lambda c: SimpleNamespace(eos_token_id=0)))  # fmt: skip
    config = SimpleNamespace(speculative_config=object(), model_config=SimpleNamespace(
        try_get_generation_config=lambda: {"eos_token_id": 0}))  # fmt: skip
    _, state = vllm
    state.all_token_ids.gpu = torch.zeros((SLOTS, 512), dtype=torch.int32, device="cuda")
    state.vocab_size = served.vocab_size + 11
    processor = XGraphLogitsProcessor(config, state)
    base = dict(paths=True, end_after_answer=True, eos_token_ids=[0])
    view = dict(menu=2, observe=1, observe_select=True, observe_answer=True,
                observe_required=True, observe_no_loop=True)  # fmt: skip
    oracles = [Oracle(index, served, syntax, **base),
               Oracle(index, served, syntax, **base, **view)]  # fmt: skip
    prefixes = ["⟦ Start · via", "⟦ Start ·→ Start ⟦? via⟧ · via"]
    states = []
    for slot, (oracle, prefix) in enumerate(zip(oracles, prefixes, strict=True)):
        options = {"menu": 0, "observe": 0, "path_demos": 4} if slot == 0 else view
        assert admit(processor, state, slot, served.encode(prefix), True,
                     xgraph_question="Which next?", xgraph_prefill=prefix,
                     xgraph_path_options=options)  # fmt: skip
        expected = oracle.initial_state(question="Which next?")
        for token in served.encode(prefix):
            expected = oracle.advance(expected, token)
        states.append(expected)
    assert not admit(processor, state, 2, [], False)
    processor.apply_staged_writes()
    original, logits = step(processor, state, [1, 2, 0])
    assert torch.equal(original[1], logits[1])
    assert processor.replays["tail mismatch"] == processor.replays["disallowed"] == 0
    for slot, row in ((0, 2), (1, 0)):
        assert allowed(served, logits[row]) == oracles[slot].allowed(states[slot])
    # Speculative rows use each slot's view; neither live state advances.
    drafts = [served.encode("⟧."), served.encode(" ·→")]
    mapping = torch.tensor([0, 0, 0, 1, 1, 1, 2], dtype=torch.int32, device="cuda")
    ctx = SimpleNamespace(
        idx_mapping=torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda"),
        expanded_idx_mapping=mapping,
        expanded_local_pos=torch.tensor([0, 1, 2, 0, 1, 2, 0], dtype=torch.int32, device="cuda"),
        input_ids=torch.tensor(
            [0, *drafts[0][:2], 0, *drafts[1][:2], 0], dtype=torch.int32, device="cuda"
        ),
    )
    before = processor._engine.states_host()
    logits = processor.apply(torch.zeros((7, state.vocab_size), device="cuda"), ctx)
    for slot, oracle in enumerate(oracles):
        expected = states[slot]
        for k in range(3):
            assert allowed(served, logits[slot * 3 + k]) == oracle.allowed(expected)
            if k < 2:
                expected = oracle.advance(expected, drafts[slot][k])
    assert all(np.array_equal(v, processor._engine.states_host()[k]) for k, v in before.items())
    assert admit(processor, state, 0, [], True, xgraph_question="Which next?")
    processor.apply_staged_writes()
    fresh = oracles[1].initial_state(question="Which next?")
    for token in served.encode("⟦ Start"):
        sample(state, 0, token)
        fresh = oracles[1].advance(fresh, token)
    _, logits = step(processor, state, [0])
    assert allowed(served, logits[0]) == oracles[1].allowed(fresh)


def test_menu_v2_adapter_prefill_drafts_forced_choices_and_slot_isolation(
    vllm, tmp_path, monkeypatch
):
    from tests.conftest import byte_level_tokenizer
    from tests.test_engine import allowed_from_mask
    from tests.test_oracle import observed_world

    kg, served, syntax = observed_world()
    index = build_kg_index(kg, served, syntax)
    save_kg_index(index, tmp_path / "compact-index")
    byte_level_tokenizer([]).save(str(tmp_path / "compact-tokenizer.json"))
    monkeypatch.setenv("XGRAPH_KG_INDEX", str(tmp_path / "compact-index"))
    monkeypatch.setenv("XGRAPH_TOKENIZER", str(tmp_path / "compact-tokenizer.json"))
    for name, value in (
        ("PATHS", "1"),
        ("MENU", "2"),
        ("OBSERVE", "1"),
        ("OBSERVE_WIDTH", "8"),
        ("OBSERVE_DEFER", "2"),
        ("MENU_CLOSE_NEXT", "1"),
        ("OBSERVE_REQUIRED", "1"),
        ("OBSERVE_NO_LOOP", "1"),
        ("RELEASE_TOP", "1"),
        ("RELEASE_MASS", "0.99"),
        ("ESCAPE_BIAS", "1000"),
    ):
        monkeypatch.setenv("XGRAPH_" + name, value)
    monkeypatch.setitem(
        sys.modules,
        "vllm.tokenizers",
        SimpleNamespace(
            cached_tokenizer_from_config=lambda config: SimpleNamespace(eos_token_id=0)
        ),
    )
    config = SimpleNamespace(
        speculative_config=object(),
        model_config=SimpleNamespace(try_get_generation_config=lambda: {"eos_token_id": 0}),
    )
    _, state = vllm
    state.vocab_size = served.vocab_size + 11
    state.all_token_ids.gpu = torch.zeros((SLOTS, 512), dtype=torch.int32, device="cuda")
    processor = XGraphLogitsProcessor(config, state)
    oracle = Oracle(
        index,
        served,
        syntax,
        paths=True,
        menu=2,
        observe=1,
        observe_width=8,
        observe_defer=2,
        menu_close_next=True,
        observe_required=True,
        observe_no_loop=True,
        eos_token_ids=[0],
    )
    prefix = "⟦ Start ·→ Start ⟦? via⟧ · via ·→ Gamma ; Beta ; Delta ⟦=DE"
    expected = oracle.initial_state(question="Which next?")
    for token in served.encode(prefix):
        expected = oracle.advance(expected, token)
    for slot in (0, 1):
        assert admit(
            processor,
            state,
            slot,
            served.encode(prefix),
            True,
            xgraph_question="Which next?",
            xgraph_prefill=prefix,
        )
    assert not admit(processor, state, 2, [], False)
    processor.apply_staged_writes()
    _, logits = step(processor, state, [1, 2, 0])
    assert allowed(served, logits[0]) == allowed(served, logits[2]) == oracle.allowed(expected)
    saved = processor._engine.states_host()
    drafts = served.encode(
        "FER⟧ ⟦!NEXT⟧ ⟦? next⟧ · next ·→ Beta ; Delta"
        " ⟦= Delta ·→ Delta ⟦!CLOSE⟧⟧.\nAnswer: ⟦ Delta⟧"
    )
    mapping = torch.zeros(len(drafts) + 1, dtype=torch.int32, device="cuda")
    ctx = SimpleNamespace(
        idx_mapping=mapping[:1],
        expanded_idx_mapping=mapping,
        expanded_local_pos=torch.arange(len(mapping), dtype=torch.int32, device="cuda"),
        input_ids=torch.tensor([0, *drafts], dtype=torch.int32, device="cuda"),
    )
    raw = torch.full((len(mapping), state.vocab_size), -100.0, device="cuda")
    walked = expected
    masks = []
    for row in range(len(mapping)):
        masks.append(oracle.allowed(walked))
        raw[row, list(masks[-1])] = 100.0
        if row < len(drafts):
            walked = oracle.advance(walked, drafts[row])
    logits = processor.apply(raw, ctx)
    for row, wanted in enumerate(masks):
        assert allowed(served, logits[row]) == wanted
    processor._engine.fill_mask()
    assert allowed_from_mask(processor._engine.mask_host()[0]) == oracle.allowed(expected)
    assert all(np.array_equal(saved[k], v) for k, v in processor._engine.states_host().items())
    for token in drafts:
        raw = torch.full((3, state.vocab_size), -100.0, device="cuda")
        raw[2, list(oracle.allowed(expected))] = 100.0
        mapping = torch.tensor([1, 2, 0], dtype=torch.int32, device="cuda")
        ctx = SimpleNamespace(idx_mapping=mapping, expanded_idx_mapping=mapping)
        untouched = raw[1].clone()
        logits = processor.apply(raw, ctx)
        assert torch.equal(untouched, logits[1])
        assert allowed(served, logits[2]) == oracle.allowed(expected)
        sample(state, 0, token)
        expected = oracle.advance(expected, token)
    assert oracle.allowed(expected) == {0}
    assert admit(processor, state, 0, [], True, xgraph_question="Which via?")
    processor.apply_staged_writes()
    assert processor._engine.states_host()["defer_count"][0] == 0
    assert processor._engine.states_host()["decision_phase"][0] == 0


@pytest.mark.parametrize("audit", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_relation_display_adapter_forces_canonical_and_scored_fallback(
    vllm, tmp_path, monkeypatch, audit, legacy
):
    import json
    from dataclasses import replace

    from tests.conftest import byte_level_tokenizer
    from tests.test_oracle import relation_display_world

    kg, index, served, _ = relation_display_world()
    _, state = vllm
    if legacy:
        index = replace(index, relation_primary_rows=None)
        source = tmp_path / "relation_labels.jsonl"
        source.write_text(
            "".join(
                json.dumps([r.id, name, "label" if n == 0 else "alias"]) + "\n"
                for r in kg.relations
                for n, name in reversed(list(enumerate(r.labels)))
            )
        )
        monkeypatch.setenv("XGRAPH_RELATION_LABELS", str(source))
    save_kg_index(index, tmp_path / "display-index")
    byte_level_tokenizer([]).save(str(tmp_path / "display-tokenizer.json"))
    monkeypatch.setenv("XGRAPH_KG_INDEX", str(tmp_path / "display-index"))
    monkeypatch.setenv("XGRAPH_TOKENIZER", str(tmp_path / "display-tokenizer.json"))
    monkeypatch.setenv("XGRAPH_PATHS", "1")
    monkeypatch.setenv("XGRAPH_ESCAPES", "0")
    monkeypatch.setenv("XGRAPH_AUDIT", str(int(audit)))
    monkeypatch.setenv("XGRAPH_MENU", "0" if audit else "3")
    monkeypatch.setitem(
        sys.modules,
        "vllm.tokenizers",
        SimpleNamespace(
            cached_tokenizer_from_config=lambda config: SimpleNamespace(eos_token_id=0)
        ),
    )
    config = SimpleNamespace(
        speculative_config=object(),
        model_config=SimpleNamespace(try_get_generation_config=lambda: {"eos_token_id": 0}),
    )
    state.vocab_size = served.vocab_size + 11
    state.all_token_ids.gpu = torch.zeros((SLOTS, 512), dtype=torch.int32, device="cuda")
    processor = XGraphLogitsProcessor(config, state)
    for slot, question in enumerate(("Which capital?", "Which unrelated?")):
        assert admit(processor, state, slot, [], True, xgraph_question=question)
    assert not admit(processor, state, 2, [], False)
    processor.apply_staged_writes()
    prefix = "⟦ Start · country⟧" if audit else "⟦ Start"
    for token in served.encode(prefix):
        for slot in (0, 1):
            sample(state, slot, token)
    display = (
        "\nExecuted: ⟦ Start⟧"
        "\nRelations (terminal): ⟦ capital ; b capital ; country⟧"
        "\nRelations (penultimate): ⟦ capital ; b capital ; country⟧\nDecision: ⟦"
        if audit
        else " ⟦? capital ; b capital ; country⟧"
    )
    other = display.replace("capital ; b capital ; country", "a unrelated ; country ; capital")
    displays = [served.encode(display), served.encode(other)]
    for at in range(max(map(len, displays))):
        original, logits = step(processor, state, [1, 2, 0])
        assert torch.equal(original[1], logits[1])
        for slot, row in ((0, 2), (1, 0)):
            if at < len(displays[slot]):
                token = displays[slot][at]
                assert allowed(served, logits[row]) == {token}
                sample(state, slot, token)
    # Reusing a slot installs a new question before any forced display bytes.
    assert admit(processor, state, 0, [], True, xgraph_question="Which unrelated?")
    processor.apply_staged_writes()
    for token in served.encode(prefix):
        sample(state, 0, token)
    for token in displays[1]:
        _, logits = step(processor, state, [0])
        assert allowed(served, logits[0]) == {token}
        sample(state, 0, token)


@pytest.mark.parametrize("observed", [False, True])
def test_menu_adapter_question_ranks_forced_tokens_drafts_and_slot_reuse(
    vllm, tmp_path, monkeypatch, observed
):
    from tests.conftest import byte_level_tokenizer
    from tests.test_oracle import audit_world

    _, index, served, syntax = audit_world()
    _, state = vllm
    save_kg_index(index, tmp_path / "menu-index")
    byte_level_tokenizer([]).save(str(tmp_path / "menu-tokenizer.json"))
    monkeypatch.setenv("XGRAPH_KG_INDEX", str(tmp_path / "menu-index"))
    monkeypatch.setenv("XGRAPH_TOKENIZER", str(tmp_path / "menu-tokenizer.json"))
    for name, value in (
        ("XGRAPH_PATHS", "1"),
        ("XGRAPH_MENU", "1"),
        ("XGRAPH_RELEASE_TOP", "1"),
        ("XGRAPH_RELEASE_MASS", "0.99"),
        ("XGRAPH_ESCAPE_BIAS", "1000"),
    ):
        monkeypatch.setenv(name, value)
    if observed:
        monkeypatch.setenv("XGRAPH_OBSERVE", "1")
        monkeypatch.setenv("XGRAPH_OBSERVE_REQUIRED", "1")
        monkeypatch.setenv("XGRAPH_OBSERVE_NO_LOOP", "1")
    state.vocab_size = served.vocab_size + 11
    state.all_token_ids.gpu = torch.zeros((SLOTS, 512), dtype=torch.int32, device="cuda")
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=object()), state)
    oracle = Oracle(
        index,
        served,
        syntax,
        paths=True,
        menu=1,
        observe=int(observed),
        observe_required=observed,
        observe_no_loop=observed,
    )
    questions = ["Which next?", "Which alternate?"]
    states = [oracle.initial_state(question=q) for q in questions]
    with pytest.raises(ValueError, match="question"):
        admit(processor, state, 3, [], True)
    for slot, question in enumerate(questions):
        assert admit(processor, state, slot, [], True, xgraph_question=question)
    assert not admit(processor, state, 2, [], False)
    processor.apply_staged_writes()
    # Different request scores also change the start menu, so drain each prefix
    # independently, using its scalar state through all forced bytes.
    for slot in (0, 1):
        prefix = "⟦ Start ·→" if observed else "⟦ Start ⟦?"
        for token in served.encode(prefix):
            sample(state, slot, token)
            states[slot] = oracle.advance(states[slot], token)
    displays = [b"", b""]
    while any(s.audit_phase == 6 or s.observation_offset < len(s.observation) for s in states):
        original, logits = step(processor, state, [1, 2, 0])
        assert torch.equal(original[1], logits[1])
        for slot, row in ((0, 2), (1, 0)):
            if states[slot].audit_phase != 6 and states[slot].observation_offset == len(
                states[slot].observation
            ):
                continue
            assert allowed(served, logits[row]) == oracle.allowed(states[slot])
            (token,) = oracle.allowed(states[slot])
            displays[slot] += served.token_bytes(token)
            sample(state, slot, token)
            states[slot] = oracle.advance(states[slot], token)
    assert b" via (+1 more)" in displays[0]
    assert b" alternate (+1 more)" in displays[1]
    # Speculate through a relation, its observation and its menu. Rejecting the
    # draft must leave the live boundary and per-slot scores intact.
    drafts = served.encode(
        " · via" + (" ·→ Beta (+1 more) ⟦? next (+1 more)⟧" if observed else " ⟦? next (+1 more)⟧")
    )
    mapping = torch.zeros(len(drafts) + 1, dtype=torch.int32, device="cuda")
    ctx = SimpleNamespace(
        idx_mapping=mapping[:1],
        expanded_idx_mapping=mapping,
        expanded_local_pos=torch.arange(len(mapping), dtype=torch.int32, device="cuda"),
        input_ids=torch.tensor([0, *drafts], dtype=torch.int32, device="cuda"),
    )
    raw = torch.full((len(mapping), state.vocab_size), -100.0, device="cuda")
    expected = states[0]
    for row in range(len(mapping)):
        raw[row, list(oracle.allowed(expected))] = 100.0
        if row < len(drafts):
            expected = oracle.advance(expected, drafts[row])
    logits = processor.apply(raw, ctx)
    expected = states[0]
    for row in range(len(mapping)):
        assert allowed(served, logits[row]) == oracle.allowed(expected)
        if row < len(drafts):
            expected = oracle.advance(expected, drafts[row])
    processor._engine.fill_mask()
    from tests.test_engine import allowed_from_mask

    assert allowed_from_mask(processor._engine.mask_host()[0]) == oracle.allowed(states[0])
    prefill = "⟦ Start" + (" ·→ Start" if observed else "") + " ⟦? alternate (+1 more)⟧"
    assert admit(
        processor,
        state,
        0,
        served.encode(prefill),
        True,
        xgraph_question=questions[1],
        xgraph_prefill=prefill,
    )
    processor.apply_staged_writes()
    expected = oracle.initial_state(question=questions[1])
    for token in served.encode(prefill):
        expected = oracle.advance(expected, token)
    mapping = torch.zeros(1, dtype=torch.int32, device="cuda")
    ctx = SimpleNamespace(idx_mapping=mapping, expanded_idx_mapping=mapping)
    raw = torch.full((1, state.vocab_size), -100.0, device="cuda")
    raw[0, list(oracle.allowed(expected))] = 100.0
    logits = processor.apply(raw, ctx)  # apply replays prompt tokens, as vLLM does
    assert allowed(served, logits[0]) == oracle.allowed(expected)
    assert processor.replays["replayed"] == 1


def test_audit_adapter_installs_request_scores_and_forces_tokens_in_isolated_slots(
    vllm, tmp_path, monkeypatch
):
    from tests.conftest import byte_level_tokenizer
    from tests.test_oracle import audit_world

    _, index, served, syntax = audit_world()
    _, state = vllm
    save_kg_index(index, tmp_path / "audit-index")
    served_tokenizer = tmp_path / "audit-tokenizer.json"
    byte_level_tokenizer([]).save(str(served_tokenizer))
    monkeypatch.setenv("XGRAPH_KG_INDEX", str(tmp_path / "audit-index"))
    monkeypatch.setenv("XGRAPH_TOKENIZER", str(served_tokenizer))
    for name, value in (
        ("XGRAPH_PATHS", "1"),
        ("XGRAPH_AUDIT", "1"),
        ("XGRAPH_AUDIT_RELATIONS", "1"),
        ("XGRAPH_RELEASE_TOP", "1"),
        ("XGRAPH_RELEASE_MASS", "0.99"),
        ("XGRAPH_ESCAPE_BIAS", "1000"),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.setitem(
        sys.modules,
        "vllm.tokenizers",
        SimpleNamespace(
            cached_tokenizer_from_config=lambda config: SimpleNamespace(eos_token_id=0)
        ),
    )
    config = SimpleNamespace(
        speculative_config=object(),
        model_config=SimpleNamespace(try_get_generation_config=lambda: {"eos_token_id": 0}),
    )
    state.vocab_size = served.vocab_size + 11
    state.all_token_ids.gpu = torch.zeros((SLOTS, 512), dtype=torch.int32, device="cuda")
    processor = XGraphLogitsProcessor(config, state)
    oracle = Oracle(
        index, served, syntax, paths=True, audit=True, audit_relations=1, eos_token_ids=[0]
    )
    questions = ["Which next?", "Which alternate?"]
    states = [oracle.initial_state(question=q) for q in questions]
    for slot, question in enumerate(questions):
        assert admit(processor, state, slot, [], True, xgraph_question=question)
    assert not admit(processor, state, 2, [], False)
    processor.apply_staged_writes()
    for token in served.encode("⟦ Start · via⟧"):
        for slot in (0, 1):
            sample(state, slot, token)
            states[slot] = oracle.advance(states[slot], token)
    displays = [b"", b""]
    while any(s.audit_phase == 1 for s in states):
        original, logits = step(processor, state, [1, 2, 0])
        assert torch.equal(original[1], logits[1])
        for slot, row in ((0, 2), (1, 0)):
            assert allowed(served, logits[row]) == oracle.allowed(states[slot])
            if states[slot].audit_phase != 1:
                continue
            (token,) = oracle.allowed(states[slot])
            displays[slot] += served.token_bytes(token)
            sample(state, slot, token)
            states[slot] = oracle.advance(states[slot], token)
    assert b" next (+1 more)" in displays[0]
    assert b" alternate (+1 more)" in displays[1]
    drafts = served.encode("APPEND · next⟧\nExecuted: ⟦ End⟧.\nAnswer: ⟦ End⟧")
    rows = len(drafts) + 1
    mapping = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ctx = SimpleNamespace(
        idx_mapping=mapping[:1],
        expanded_idx_mapping=mapping,
        expanded_local_pos=torch.arange(rows, dtype=torch.int32, device="cuda"),
        input_ids=torch.tensor([0, *drafts], dtype=torch.int32, device="cuda"),
    )
    logits = processor.apply(torch.zeros((rows, state.vocab_size), device="cuda"), ctx)
    expected = states[0]
    for row in range(rows):
        assert allowed(served, logits[row]) == oracle.allowed(expected)
        if row < len(drafts):
            expected = oracle.advance(expected, drafts[row])
    _, logits = step(processor, state, [0])
    assert allowed(served, logits[0]) == oracle.allowed(states[0])
    # Canonical replay may end in a forced block; a fresh slot installs new scores
    # before replaying the prompt and does not inherit its earlier request's state.
    prefill = "⟦ Start · via⟧" + displays[1].decode()
    assert admit(
        processor,
        state,
        0,
        served.encode(prefill),
        True,
        xgraph_question=questions[1],
        xgraph_prefill=prefill,
    )
    processor.apply_staged_writes()
    _, logits = step(processor, state, [0])
    assert allowed(served, logits[0]) == oracle.allowed(states[1])
    assert processor.replays["replayed"] == 1


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("v3", [False, True])
def test_observed_adapter_forces_tokens_with_slot_isolation_and_replayed_prefills(
    vllm, served, toy_kg, monkeypatch, v2, v3
):
    _, state = vllm
    state.all_token_ids.gpu = torch.zeros((SLOTS, 512), dtype=torch.int32, device="cuda")
    for name, value in (
        ("XGRAPH_PATHS", "1"),
        ("XGRAPH_COMMITTED_PATHS", "1"),
        ("XGRAPH_OBSERVE", "2"),
        ("XGRAPH_ESCAPES", "0"),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("XGRAPH_OBSERVE_SELECT", str(int(v2)))
    monkeypatch.setenv("XGRAPH_OBSERVE_ANSWER", str(int(v2)))
    monkeypatch.setenv("XGRAPH_OBSERVE_REQUIRED", str(int(v3)))
    monkeypatch.setenv("XGRAPH_OBSERVE_NO_LOOP", str(int(v3)))
    monkeypatch.setitem(
        sys.modules,
        "vllm.tokenizers",
        SimpleNamespace(
            cached_tokenizer_from_config=lambda config: SimpleNamespace(eos_token_id=0)
        ),
    )
    config = SimpleNamespace(
        speculative_config=object(),
        model_config=SimpleNamespace(try_get_generation_config=lambda: {"eos_token_id": 0}),
    )
    processor = XGraphLogitsProcessor(config, state)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    oracle = Oracle(
        index,
        served,
        SpanSyntax(),
        paths=True,
        committed_paths=True,
        escapes=False,
        observe=2,
        observe_select=v2,
        observe_answer=v2,
        observe_required=v3,
        observe_no_loop=v3,
        eos_token_ids=[0] if v2 else (),
    )
    prefix = "⟦ Marie Curie"
    expected = oracle.initial_state()
    for token in served.encode(prefix):
        expected = oracle.advance(expected, token)
    assert admit(processor, state, 0, served.encode(prefix), True, xgraph_prefill=prefix)
    assert not admit(processor, state, 1, served.encode("Hi"), False)
    processor.apply_staged_writes()
    chunks = (
        " ·→",
        " Maria Skłodowska",
        " ·",
        " spouse",
        " ·→",
        " Pierre Curie",
        *([" ⟦=", " Pierre Curie", " ·→", " Pierre Curie"] if v2 else []),
        "⟧",
        *([".\nAnswer: ", "⟦"] if v2 else ["\nAnswer: ⟦"]),
        " Pierre Curie",
        "⟧",
    )
    for token in [t for c in chunks for t in served.encode(c)]:
        original, logits = step(processor, state, [1, 0])
        assert torch.equal(logits[0], original[0])
        assert allowed(served, logits[1]) == oracle.allowed(expected)
        sample(state, 0, token)
        expected = oracle.advance(expected, token)
    _, logits = step(processor, state, [0])
    assert allowed(served, logits[0]) == oracle.allowed(expected)
    if v2:
        # Reuse the slot for a canonical prefill ending inside the forced prefix.
        prefill = prefix + "".join(chunks[:-3])
        tokens = [*served.encode(prefix), *(t for c in chunks[:-3] for t in served.encode(c))]
        expected = oracle.initial_state()
        for token in tokens:
            expected = oracle.advance(expected, token)
        assert admit(processor, state, 0, tokens, True, xgraph_prefill=prefill)
        processor.apply_staged_writes()
        original, logits = step(processor, state, [1, 0])
        assert torch.equal(logits[0], original[0])
        assert allowed(served, logits[1]) == oracle.allowed(expected)
        assert admit(processor, state, 0, served.encode("Hi"), True)
        processor.apply_staged_writes()
        _, logits = step(processor, state, [0])
        assert allowed(served, logits[0]) == oracle.allowed(oracle.initial_state())


def test_required_selection_opener_cannot_be_released_and_drafts_do_not_advance_it(
    vllm, tmp_path, monkeypatch
):
    from tests.conftest import byte_level_tokenizer
    from tests.test_oracle import observed_world

    _, state = vllm
    kg, served, syntax = observed_world()
    index = tmp_path / "required-index"
    save_kg_index(build_kg_index(kg, served, syntax), index)
    byte_level_tokenizer([]).save(str(tmp_path / "required-tokenizer.json"))
    settings = {
        "XGRAPH_KG_INDEX": str(index),
        "XGRAPH_TOKENIZER": str(tmp_path / "required-tokenizer.json"),
        "XGRAPH_PATHS": "1",
        "XGRAPH_OBSERVE": "1",
        "XGRAPH_OBSERVE_REQUIRED": "1",
        "XGRAPH_OBSERVE_NO_LOOP": "1",
        "XGRAPH_RELEASE_TOP": "1",
        "XGRAPH_ESCAPE_BIAS": "-50",
    }
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=object()), state)
    state.all_token_ids.gpu = torch.zeros((SLOTS, 512), dtype=torch.int32, device="cuda")
    prefix = "⟦ Start ·→ Start · via ·→ Gamma (+2 more)"
    assert admit(processor, state, 0, served.encode(prefix), True, xgraph_prefill=prefix)
    assert not admit(processor, state, 1, served.encode("Hi"), False)
    processor.apply_staged_writes()
    drafts = served.encode(" ⟦=")
    mapping = torch.zeros(len(drafts), dtype=torch.int32, device="cuda")
    context = SimpleNamespace(
        idx_mapping=mapping[:1],
        expanded_idx_mapping=mapping,
        expanded_local_pos=torch.arange(len(drafts), dtype=torch.int32, device="cuda"),
        input_ids=torch.tensor([0, *drafts[:-1]], dtype=torch.int32, device="cuda"),
    )
    logits = torch.zeros((len(drafts), state.vocab_size), device="cuda")
    logits[:, 0] = 100  # release_top must not override a required opener
    masked = processor.apply(logits, context)
    for row, token in enumerate(drafts):
        assert allowed(served, masked[row]) == {token}
    original, real = step(processor, state, [1, 0])
    assert torch.equal(original[0], real[0])
    assert allowed(served, real[1]) == {drafts[0]}


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("v3", [False, True])
def test_observed_adapter_speculative_rows_copy_observation_progress(
    vllm, served, toy_kg, monkeypatch, v2, v3
):
    _, state = vllm
    for name, value in (("XGRAPH_PATHS", "1"), ("XGRAPH_OBSERVE", "1"), ("XGRAPH_ESCAPES", "0")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("XGRAPH_OBSERVE_SELECT", str(int(v2)))
    monkeypatch.setenv("XGRAPH_OBSERVE_ANSWER", str(int(v2)))
    monkeypatch.setenv("XGRAPH_OBSERVE_REQUIRED", str(int(v3)))
    monkeypatch.setenv("XGRAPH_OBSERVE_NO_LOOP", str(int(v3)))
    monkeypatch.setitem(
        sys.modules,
        "vllm.tokenizers",
        SimpleNamespace(
            cached_tokenizer_from_config=lambda config: SimpleNamespace(eos_token_id=0)
        ),
    )
    config = SimpleNamespace(
        speculative_config=object(),
        model_config=SimpleNamespace(try_get_generation_config=lambda: {"eos_token_id": 0}),
    )
    processor = XGraphLogitsProcessor(config, state)
    oracle = Oracle(
        build_kg_index(toy_kg, served, SpanSyntax()),
        served,
        SpanSyntax(),
        paths=True,
        escapes=False,
        observe=1,
        observe_select=v2,
        observe_answer=v2,
        observe_required=v3,
        observe_no_loop=v3,
        eos_token_ids=[0] if v2 else (),
    )
    assert admit(processor, state, 0, served.encode("Hi"), True)
    processor.apply_staged_writes()
    expected = oracle.initial_state()
    state.all_token_ids.gpu = torch.zeros((SLOTS, 512), dtype=torch.int32, device="cuda")
    prefix = "⟦ Marie Curie" + (" ·→ Maria Skłodowska · spouse ·→ Pierre Curie ⟦=" if v2 else "")
    for token in served.encode(prefix):
        step(processor, state, [0])
        sample(state, 0, token)
        expected = oracle.advance(expected, token)
    chunks = (
        [" Pierre Curie", " ·→", " Pierre Curie", "⟧", ".\nAnswer: ", "⟦", " Pierre Curie", "⟧"]
        if v2
        else [" ·→", " Maria Skłodowska"]
    )
    drafts = [t for chunk in chunks for t in served.encode(chunk)]
    rows = len(drafts) + 1
    mapping = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ctx = SimpleNamespace(
        idx_mapping=mapping[:1],
        expanded_idx_mapping=mapping,
        expanded_local_pos=torch.arange(rows, dtype=torch.int32, device="cuda"),
        input_ids=torch.tensor([0, *drafts], dtype=torch.int32, device="cuda"),
    )
    logits = processor.apply(torch.zeros((rows, state.vocab_size), device="cuda"), ctx)
    for k in range(rows):
        assert allowed(served, logits[k]) == oracle.allowed(expected)
        if k < len(drafts):
            expected = oracle.advance(expected, drafts[k])
    # Draft progress must not consume the real slot's forced observation.
    _, real = step(processor, state, [0])
    assert allowed(served, real[0]) == {drafts[0]}


def test_observation_payload_cannot_be_released_or_escape_biased(vllm, served, toy_kg, monkeypatch):
    _, state = vllm
    for name, value in (
        ("XGRAPH_PATHS", "1"),
        ("XGRAPH_OBSERVE", "1"),
        ("XGRAPH_RELEASE_TOP", "1"),
        ("XGRAPH_RELEASE_MASS", "0.99"),
        ("XGRAPH_ESCAPE_BIAS", "1000"),
    ):
        monkeypatch.setenv(name, value)
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=None), state)
    prefix = "⟦ Marie Curie ·→"
    assert admit(processor, state, 0, served.encode(prefix), True, xgraph_prefill=prefix)
    processor.apply_staged_writes()
    oracle = Oracle(
        build_kg_index(toy_kg, served, SpanSyntax()), served, SpanSyntax(), paths=True, observe=1
    )
    expected = oracle.initial_state()
    for token in served.encode(prefix):
        expected = oracle.advance(expected, token)
    mapping = torch.tensor([0], dtype=torch.int32, device="cuda")
    logits = torch.zeros((1, state.vocab_size), device="cuda")
    logits[0, 0] = 100000  # disallowed top token, almost no mass on the observation
    ctx = SimpleNamespace(idx_mapping=mapping, expanded_idx_mapping=mapping)
    result = processor.apply(logits, ctx)
    assert len(oracle.allowed(expected)) == 1
    assert allowed(served, result[0]) == oracle.allowed(expected)


def test_opted_in_requests_are_masked_like_the_oracle_and_others_are_not(vllm, served, toy_kg):
    processor, state = vllm
    oracle = Oracle(build_kg_index(toy_kg, served, SpanSyntax()), served, SpanSyntax())
    prompt = served.encode("Example: ⟦ Paris")  # a span left open inside the prompt
    assert admit(processor, state, 2, prompt, on=True)
    assert not admit(processor, state, 0, prompt, on=False)
    processor.apply_staged_writes()
    reply = served.encode(" She met ⟦ Marie Curie · spouse · Pierre Curie⟧ there")
    expected = oracle.initial_state()
    for token in reply:
        original, logits = step(processor, state, [2, 0])
        assert allowed(served, logits[0]) == oracle.allowed(expected)
        assert torch.isneginf(logits[0, served.vocab_size :]).all()  # padding columns
        assert torch.equal(logits[1], original[1])  # the declined request
        expected = oracle.advance(expected, token)
        sample(state, 2, token)
        sample(state, 0, token)


def test_a_reused_slot_starts_afresh(vllm, served):
    processor, state = vllm
    admit(processor, state, 1, served.encode("Hi"), on=True)
    processor.apply_staged_writes()
    for token in served.encode(" ⟦ Marie"):  # the first request leaves a span open
        step(processor, state, [1])
        sample(state, 1, token)
    admit(processor, state, 1, served.encode("Hello there"), on=True)
    processor.apply_staged_writes()
    _, logits = step(processor, state, [1])
    eos = served.encode("<|endoftext|>")[0]
    assert eos in allowed(served, logits[0])  # free text again: end of text is allowed


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("eos_ids", [1, [1, 36]])
def test_end_after_answer_eos_metadata_reaches_engine_on_cpu(
    toy_kg, toy_tokenizer, tmp_path, monkeypatch, enabled, eos_ids
):
    from xgraph import engine

    served, syntax = ServedTokenizer(toy_tokenizer), SpanSyntax()
    save_kg_index(build_kg_index(toy_kg, served, syntax), tmp_path / "index")
    toy_tokenizer.save(str(tmp_path / "tokenizer.json"))
    monkeypatch.setenv("XGRAPH_KG_INDEX", str(tmp_path / "index"))
    monkeypatch.setenv("XGRAPH_TOKENIZER", str(tmp_path / "tokenizer.json"))
    monkeypatch.setenv("XGRAPH_PATHS", "1")
    monkeypatch.setenv("XGRAPH_END_AFTER_ANSWER", "1" if enabled else "0")
    monkeypatch.setitem(
        sys.modules,
        "vllm.tokenizers",
        SimpleNamespace(
            cached_tokenizer_from_config=lambda config: SimpleNamespace(eos_token_id=0)
        ),
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(try_get_generation_config=lambda: {"eos_token_id": eos_ids})
    )
    captured = []
    monkeypatch.setattr(engine, "Engine", lambda *args, **kwargs: captured.append(kwargs))
    XGraphLogitsProcessor(config, SimpleNamespace(max_num_reqs=2))
    assert captured[0]["end_after_answer"] is enabled
    eos = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
    assert set(captured[0]["eos_token_ids"]) == (eos | {0} if enabled else set())


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("eos_ids", [0, [0, 1]])
@pytest.mark.parametrize("delivery", ["live", "open", "prefill", "complete prefill"])
@pytest.mark.parametrize("relations", ["frontier", "vocabulary", "successor"])
def test_end_after_answer_flag_eos_and_slots(
    vllm, served, toy_kg, tmp_path, monkeypatch, enabled, eos_ids, delivery, relations
):
    from xgraph.kgversions import write_kg_versions

    _, state = vllm
    monkeypatch.setenv("XGRAPH_PATHS", "1")
    monkeypatch.setenv("XGRAPH_ESCAPES", "0")
    monkeypatch.setenv("XGRAPH_END_AFTER_ANSWER", "1" if enabled else "0")
    monkeypatch.setenv("XGRAPH_RELATIONS", relations)
    loaded = []
    monkeypatch.setitem(
        sys.modules,
        "vllm.tokenizers",
        SimpleNamespace(
            cached_tokenizer_from_config=lambda config: (
                loaded.append(config) or SimpleNamespace(eos_token_id=0)
            )
        ),
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(try_get_generation_config=lambda: {"eos_token_id": eos_ids})
    )
    write_kg_versions(tmp_path / "versions", {"edit": np.array([[1, 108, 2]], dtype=np.int32)})
    monkeypatch.setenv("XGRAPH_KG_VERSIONS", str(tmp_path / "versions"))
    processor = XGraphLogitsProcessor(config, state)
    eos = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
    assert bool(loaded) is enabled  # the default does not even resolve EOS metadata
    oracle = Oracle(build_kg_index(toy_kg, served, SpanSyntax()), served, SpanSyntax(),
                    paths=True, escapes=False, relations=relations, end_after_answer=enabled,
                    eos_token_ids=eos, versions={"edit": [("Q1", "P108", "Q2")]})  # fmt: skip
    path, answer = "⟦ Marie Curie · employer⟧ so ", "⟦ Pierre Curie⟧"
    prefill = path + answer if delivery == "complete prefill" else path
    extra = dict(xgraph_anchor=["Q1"], xgraph_kg_version="edit")
    if delivery in ("prefill", "complete prefill"):
        prompt = served.encode("Hi ") + served.encode(prefill)
        extra["xgraph_prefill"] = prefill
        reply = "" if delivery == "complete prefill" else answer
    elif delivery == "open":
        prompt, reply = served.encode("Hi ⟦"), path[1:] + answer
        extra["xgraph_open"] = 1
    else:
        prompt, reply = served.encode("Hi"), path + answer
    assert admit(processor, state, 0, prompt, on=True, **extra)
    assert admit(processor, state, 1, served.encode("Hi"), on=True)
    assert not admit(processor, state, 2, served.encode("Hi"), on=False)
    processor.apply_staged_writes()
    expected = oracle.anchored_state(["Q1"], "edit")
    if delivery == "open":
        expected = oracle.opened_state(expected)
    elif delivery in ("prefill", "complete prefill"):
        for token in served.encode(prefill):
            expected = oracle.advance(expected, token)
    for token in served.encode(reply):
        original, logits = step(processor, state, [2, 1, 0])
        assert torch.equal(logits[0], original[0])
        assert allowed(served, logits[2]) == oracle.allowed(expected)
        expected = oracle.advance(expected, token)
        sample(state, 0, token)
    original, logits = step(processor, state, [1, 0, 2])
    assert allowed(served, logits[1]) == oracle.allowed(expected)
    assert (allowed(served, logits[1]) == eos) is enabled
    assert allowed(served, logits[0]) != eos
    assert torch.equal(logits[2], original[2])
    assert torch.isneginf(logits[1, served.vocab_size :]).all()
    admit(processor, state, 0, served.encode("Hi"), on=True)
    processor.apply_staged_writes()
    _, logits = step(processor, state, [0])
    assert allowed(served, logits[0]) == oracle.allowed(oracle.initial_state())


@pytest.mark.parametrize("vllm", ["cpu", "cuda"], indirect=True)
def test_end_after_answer_declines_a_prefill_with_trailing_text(vllm, served, monkeypatch):
    from xgraph.vllm import collect_replay_status

    _, state = vllm
    monkeypatch.setenv("XGRAPH_PATHS", "1")
    monkeypatch.setenv("XGRAPH_END_AFTER_ANSWER", "1")
    monkeypatch.setitem(
        sys.modules,
        "vllm.tokenizers",
        SimpleNamespace(
            cached_tokenizer_from_config=lambda config: SimpleNamespace(eos_token_id=0)
        ),
    )
    config = SimpleNamespace(model_config=SimpleNamespace(try_get_generation_config=lambda: {}))
    processor = XGraphLogitsProcessor(config, state)
    prefill = "⟦ Marie Curie · spouse⟧ so ⟦ Pierre Curie⟧ Wait"
    key = "trailing-text"
    admit(
        processor,
        state,
        0,
        served.encode(prefill),
        on=True,
        xgraph_prefill=prefill,
        xgraph_replay_id=key,
    )
    processor.apply_staged_writes()
    original, logits = step(processor, state, [0])
    assert torch.equal(original, logits)
    assert processor.replays["disallowed"] == 1
    assert collect_replay_status(None, [key]) == {key: "disallowed"}
    assert collect_replay_status(None, [key]) == {}


@pytest.mark.parametrize("support", ["frontier", "vocabulary", "successor"])
@pytest.mark.parametrize("forced", [False, True])
def test_agent_rounds_opt_in_with_fresh_states_versions_and_optional_prefill(
    vllm, served, toy_kg, tmp_path, monkeypatch, support, forced
):
    from xgraph.generate import sampling_params
    from xgraph.kgversions import write_kg_versions

    _, state = vllm
    for key, value in {
        "PATHS": "1",
        "COMMITTED_PATHS": "1",
        "ESCAPES": "0",
        "RELATIONS": support,
    }.items():
        monkeypatch.setenv("XGRAPH_" + key, value)
    edits = [("Q1", "P108", "Q3")]
    write_kg_versions(tmp_path / "versions", {"v": np.array([[1, 108, 3]], dtype=np.int32)})
    monkeypatch.setenv("XGRAPH_KG_VERSIONS", str(tmp_path / "versions"))
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=None), state)
    oracle = Oracle(
        build_kg_index(toy_kg, served, SpanSyntax()),
        served,
        SpanSyntax(),
        paths=True,
        committed_paths=True,
        escapes=False,
        relations=support,
        versions={"v": edits},
    )
    opener = "Action: execute_path ⟦"
    for round_number, version in enumerate(["v", None]):
        prefill = opener if forced and round_number == 0 else None
        prompt = served.encode("Earlier: ⟦ Paris⟧ Observation: done\n" + (prefill or ""))
        for slot in (0, 1):
            admit(processor, state, slot, prompt, on=False)
        processor.apply_staged_writes()
        opted = sampling_params(
            "constrained", 256, SimpleNamespace, kg_version=version, prefill=prefill
        )
        free = sampling_params("agent", 256, SimpleNamespace)
        assert processor.add_request(0, opted)
        assert not processor.add_request(1, free)
        processor.apply_staged_writes()
        expected = oracle.initial_state(version)
        if prefill:
            for token in served.encode(prefill):
                assert token in oracle.allowed(expected)
                expected = oracle.advance(expected, token)
        text = ("" if prefill else opener) + " Marie Curie · employer⟧"
        for token in served.encode(text):
            original, logits = step(processor, state, [1, 0])
            assert torch.equal(original[0], logits[0])
            assert allowed(served, logits[1]) == oracle.allowed(expected)
            assert token in oracle.allowed(expected)
            sample(state, 0, token)
            expected = oracle.advance(expected, token)
        if support == "frontier":
            assert expected.last_span.entity_ids == (("Q3",) if version else ("Q7",))


@pytest.mark.parametrize("value", ["yes", 2, [1]])
def test_odd_opt_in_values_are_refused(value):
    with pytest.raises(ValueError, match="xgraph"):
        XGraphLogitsProcessor.validate_params(params(xgraph=value))


def test_draft_rows_are_masked_after_the_drafts_before_them(vllm, served, toy_kg):
    """Under speculative decoding vLLM hands one row per draft position of a request."""
    processor, state = vllm
    oracle = Oracle(build_kg_index(toy_kg, served, SpanSyntax()), served, SpanSyntax())
    prompt = served.encode("Hi")
    admit(processor, state, 3, prompt, on=True)
    processor.apply_staged_writes()
    committed = served.encode(" She met ⟦ Marie")
    for token in committed:
        step(processor, state, [3])
        sample(state, 3, token)
    drafts = served.encode(" Curie⟧ in")
    rows = len(drafts) + 1
    ctx = SimpleNamespace(
        idx_mapping=torch.tensor([3], dtype=torch.int32, device="cuda"),
        expanded_idx_mapping=torch.full((rows,), 3, dtype=torch.int32, device="cuda"),
        expanded_local_pos=torch.arange(rows, dtype=torch.int32, device="cuda"),
        input_ids=torch.tensor([committed[-1], *drafts], dtype=torch.int32, device="cuda"),
    )
    logits = processor.apply(torch.randn((rows, state.vocab_size), device="cuda"), ctx)
    for k in range(rows):
        expected = oracle.initial_state()
        for token in [*committed, *drafts[:k]]:
            expected = oracle.advance(expected, token)
        assert allowed(served, logits[k]) == oracle.allowed(expected), k


@pytest.mark.usefixtures("cuda")
def test_the_span_syntax_comes_from_the_index(toy_kg, toy_tokenizer, served, tmp_path, monkeypatch):
    from xgraph.syntax import preset

    wiki = preset("wiki")
    index = build_kg_index(toy_kg, served, wiki)
    save_kg_index(index, tmp_path / "wiki-index")
    toy_tokenizer.save(str(tmp_path / "tokenizer.json"))
    monkeypatch.setenv("XGRAPH_KG_INDEX", str(tmp_path / "wiki-index"))
    monkeypatch.setenv("XGRAPH_TOKENIZER", str(tmp_path / "tokenizer.json"))
    gpu = torch.device("cuda")
    state = SimpleNamespace(
        device=gpu,
        max_num_reqs=SLOTS,
        vocab_size=served.vocab_size,
        all_token_ids=SimpleNamespace(
            gpu=torch.zeros((SLOTS, WIDTH), dtype=torch.int32, device=gpu)
        ),
        prompt_len=SimpleNamespace(gpu=torch.zeros(SLOTS, dtype=torch.int32, device=gpu)),
        prefill_len=SimpleNamespace(gpu=torch.zeros(SLOTS, dtype=torch.int32, device=gpu)),
        total_len=SimpleNamespace(gpu=torch.zeros(SLOTS, dtype=torch.int32, device=gpu)),
    )
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=None), state)
    oracle = Oracle(index, served, wiki)
    admit(processor, state, 0, served.encode("Hi"), on=True)
    processor.apply_staged_writes()
    expected = oracle.initial_state()
    for token in served.encode(" She met [[Marie Curie]] there"):
        _, logits = step(processor, state, [0])
        assert allowed(served, logits[0]) == oracle.allowed(expected)
        expected = oracle.advance(expected, token)
        sample(state, 0, token)


def test_escapes_can_be_switched_off(vllm, served, toy_kg, monkeypatch):
    _, state = vllm
    monkeypatch.setenv("XGRAPH_ESCAPES", "0")
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=None), state)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    oracle = Oracle(index, served, SpanSyntax(), escapes=False)
    with_escapes = Oracle(index, served, SpanSyntax())
    admit(processor, state, 0, served.encode("Hi"), on=True)
    processor.apply_staged_writes()
    expected = escaping = oracle.initial_state()
    for token in served.encode(" ⟦ "):
        _, logits = step(processor, state, [0])
        assert allowed(served, logits[0]) == oracle.allowed(expected)
        sample(state, 0, token)
        expected = oracle.advance(expected, token)
        escaping = with_escapes.advance(escaping, token)
    _, logits = step(processor, state, [0])
    assert allowed(served, logits[0]) == oracle.allowed(expected)
    assert oracle.allowed(expected) < with_escapes.allowed(escaping)  # the escape is gone


def test_repeats_can_be_switched_off(vllm, served, toy_kg, monkeypatch):
    _, state = vllm
    monkeypatch.setenv("XGRAPH_REPEATS", "0")
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=None), state)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    oracle = Oracle(index, served, SpanSyntax(), repeats=False)
    admit(processor, state, 0, served.encode("Hi"), on=True)
    processor.apply_staged_writes()
    fact = " ⟦ Marie Curie · award received · Nobel Prize"
    expected = oracle.initial_state()
    for token in served.encode(f"{fact}⟧ and{fact}"):
        _, logits = step(processor, state, [0])
        assert allowed(served, logits[0]) == oracle.allowed(expected)
        sample(state, 0, token)
        expected = oracle.advance(expected, token)
    _, logits = step(processor, state, [0])
    assert allowed(served, logits[0]) == oracle.allowed(expected)
    assert served.encode("⟧")[0] not in oracle.allowed(expected)  # the repeat cannot close


def test_release_can_be_switched_on(vllm, served, toy_kg, monkeypatch):
    from tests.test_engine import _released_mask

    _, state = vllm
    monkeypatch.setenv("XGRAPH_RELEASE_TOP", "1")
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=None), state)
    oracle = Oracle(build_kg_index(toy_kg, served, SpanSyntax()), served, SpanSyntax())
    admit(processor, state, 0, served.encode("Hi"), on=True)
    processor.apply_staged_writes()
    expected = oracle.initial_state()
    for token in served.encode(" ⟦ Marie"):
        step(processor, state, [0])
        sample(state, 0, token)
        expected = oracle.advance(expected, token)
    torch.manual_seed(0)
    original, logits = step(processor, state, [0])
    top = int(original[0, : served.vocab_size].argmax())
    allowed_set = oracle.allowed(expected)
    assert top not in allowed_set  # random logits favour a token the KG does not allow
    assert allowed(served, logits[0]) == _released_mask(oracle, expected)


def test_derivation_can_be_switched_on(vllm, served, toy_kg, monkeypatch):
    _, state = vllm
    monkeypatch.setenv("XGRAPH_DERIVATION", "1")
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=None), state)
    oracle = Oracle(
        build_kg_index(toy_kg, served, SpanSyntax()), served, SpanSyntax(), derivation=True
    )
    admit(processor, state, 0, served.encode("Hi"), on=True)
    processor.apply_staged_writes()
    expected = oracle.initial_state()
    for token in served.encode(" ⟦ Curie · place of birth · Paris⟧ so ⟦"):
        _, logits = step(processor, state, [0])
        assert allowed(served, logits[0]) == oracle.allowed(expected)
        sample(state, 0, token)
        expected = oracle.advance(expected, token)
    _, logits = step(processor, state, [0])
    assert allowed(served, logits[0]) == oracle.allowed(expected)
    assert served.encode(" Marie")[0] not in oracle.allowed(expected)  # bound to Paris


def test_an_escape_bias_can_be_set(vllm, served, toy_kg, monkeypatch):
    _, state = vllm
    monkeypatch.setenv("XGRAPH_ESCAPE_BIAS", "-2.5")
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=None), state)
    admit(processor, state, 0, served.encode("Hi"), on=True)
    processor.apply_staged_writes()
    for token in served.encode(" ⟦ Marie"):
        step(processor, state, [0])
        sample(state, 0, token)
    original, logits = step(processor, state, [0])
    shifted = [
        t for t in range(served.vocab_size)
        if not torch.isneginf(logits[0, t]) and logits[0, t] != original[0, t]
    ]  # fmt: skip
    assert shifted and all(
        logits[0, t].item() == pytest.approx(original[0, t].item() - 2.5) for t in shifted
    )


def test_a_request_can_anchor_its_derivation(vllm, served, toy_kg, monkeypatch):
    _, state = vllm
    monkeypatch.setenv("XGRAPH_DERIVATION", "1")
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=None), state)
    oracle = Oracle(
        build_kg_index(toy_kg, served, SpanSyntax()), served, SpanSyntax(), derivation=True
    )
    slot = 0
    state.all_token_ids.gpu[slot].zero_()
    prompt = served.encode("Hi")
    state.all_token_ids.gpu[slot, : len(prompt)] = torch.tensor(prompt, dtype=torch.int32)
    for name in ("prompt_len", "prefill_len", "total_len"):
        getattr(state, name).gpu[slot] = len(prompt)
    processor.add_request(slot, params(xgraph=1, xgraph_anchor=["Q6"]))
    processor.apply_staged_writes()
    expected = oracle.anchored_state(["Q6"])
    for token in served.encode(" ⟦ Curie · place of birth ·"):
        _, logits = step(processor, state, [slot])
        assert allowed(served, logits[0]) == oracle.allowed(expected)
        sample(state, slot, token)
        expected = oracle.advance(expected, token)
    _, logits = step(processor, state, [slot])
    assert allowed(served, logits[0]) == oracle.allowed(expected)  # Q6's Paris, not Q17's


@pytest.mark.parametrize("relations", ["frontier", "vocabulary", "successor"])
def test_support_ladder_flag_and_slot_isolation(
    vllm, served, toy_kg, tmp_path, monkeypatch, relations
):
    from xgraph.kgversions import write_kg_versions

    _, state = vllm
    monkeypatch.setenv("XGRAPH_PATHS", "1")
    monkeypatch.setenv("XGRAPH_COMMITTED_PATHS", "1")
    monkeypatch.setenv("XGRAPH_ESCAPES", "0")
    monkeypatch.setenv("XGRAPH_RELATIONS", relations)
    write_kg_versions(
        tmp_path / "support-versions",
        {
            "no-location": np.array([[7, 131, 99], [7, 276, 99]], dtype=np.int32),
        },
    )
    monkeypatch.setenv("XGRAPH_KG_VERSIONS", str(tmp_path / "support-versions"))
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=None), state)
    oracle = Oracle(
        build_kg_index(toy_kg, served, SpanSyntax()),
        served,
        SpanSyntax(),
        paths=True,
        committed_paths=True,
        escapes=False,
        relations=relations,
        versions={"no-location": [("Q7", "P131", "Q99"), ("Q7", "P276", "Q99")]},
    )
    prompt = served.encode("Answer: ⟦")
    admit(
        processor,
        state,
        0,
        prompt,
        on=True,
        xgraph_open=1,
        xgraph_anchor=["Q1"],
        xgraph_kg_version="no-location",
    )
    admit(processor, state, 1, prompt, on=True, xgraph_open=1, xgraph_anchor=["Q2"])
    admit(processor, state, 2, prompt, on=False)
    processor.apply_staged_writes()
    tokens = {
        0: served.encode(" Marie Curie · employer⟧ unbound answer"),
        1: served.encode(" Pierre Curie · spouse · employer · located in⟧ text"),
    }
    expected = {
        0: oracle.opened_state(oracle.anchored_state(["Q1"], "no-location")),
        1: oracle.opened_state(oracle.anchored_state(["Q2"])),
    }
    for i in range(max(map(len, tokens.values()))):
        slots = [slot for slot in (1, 0) if i < len(tokens[slot])] + [2]
        original, logits = step(processor, state, slots)
        for row, slot in enumerate(slots[:-1]):
            assert allowed(served, logits[row]) == oracle.allowed(expected[slot]), (i, slot)
            sample(state, slot, tokens[slot][i])
            expected[slot] = oracle.advance(expected[slot], tokens[slot][i])
        assert torch.equal(original[-1], logits[-1])
    # Reuse clears predecessor support, version and anchor; the first relation is global.
    admit(processor, state, 0, prompt, on=True, xgraph_open=1)
    processor.apply_staged_writes()
    fresh = oracle.opened_state(oracle.initial_state())
    for token in served.encode(" Pierre Curie · employer"):
        _, logits = step(processor, state, [0])
        assert allowed(served, logits[0]) == oracle.allowed(fresh)
        sample(state, 0, token)
        fresh = oracle.advance(fresh, token)
    _, logits = step(processor, state, [0])
    assert allowed(served, logits[0]) == oracle.allowed(fresh)


def test_path_mode_can_be_switched_on(vllm, served, toy_kg, monkeypatch):
    _, state = vllm
    monkeypatch.setenv("XGRAPH_PATHS", "1")
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=None), state)
    oracle = Oracle(build_kg_index(toy_kg, served, SpanSyntax()), served, SpanSyntax(), paths=True)
    admit(processor, state, 0, served.encode("Hi"), on=True)
    processor.apply_staged_writes()
    expected = oracle.initial_state()
    for token in served.encode(" ⟦ Pierre Curie · spouse ·"):
        _, logits = step(processor, state, [0])
        assert allowed(served, logits[0]) == oracle.allowed(expected)
        sample(state, 0, token)
        expected = oracle.advance(expected, token)
    _, logits = step(processor, state, [0])
    assert allowed(served, logits[0]) == oracle.allowed(expected)  # Marie Curie's relations


def test_committed_paths_can_be_switched_on(vllm, served, toy_kg, monkeypatch):
    _, state = vllm
    monkeypatch.setenv("XGRAPH_PATHS", "1")
    monkeypatch.setenv("XGRAPH_COMMITTED_PATHS", "1")
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=None), state)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    oracle = Oracle(index, served, SpanSyntax(), paths=True, committed_paths=True)
    union = Oracle(index, served, SpanSyntax(), paths=True)
    admit(processor, state, 0, served.encode("Hi"), on=True)
    processor.apply_staged_writes()
    expected, other = oracle.initial_state(), union.initial_state()
    for token in served.encode(" ⟦ Curie · place of birth⟧ in ⟦ Paris⟧ near ⟦ Paris"):
        _, logits = step(processor, state, [0])
        assert allowed(served, logits[0]) == oracle.allowed(expected)
        sample(state, 0, token)
        expected, other = oracle.advance(expected, token), union.advance(other, token)
    _, logits = step(processor, state, [0])
    # Committed to Q17, the answer is its Paris (Q4), which no later path can start from.
    assert allowed(served, logits[0]) == oracle.allowed(expected) != union.allowed(other)


def test_unquoted_starts_can_be_switched_on(vllm, tmp_path, monkeypatch):
    from tests.conftest import byte_level_tokenizer
    from tests.test_oracle import QUOTE_MERGES, QUOTED_KG

    _, state = vllm
    tokenizer = byte_level_tokenizer(QUOTE_MERGES)
    served = ServedTokenizer(tokenizer)
    index = build_kg_index(QUOTED_KG, served, SpanSyntax())
    save_kg_index(index, tmp_path / "quoted")
    tokenizer.save(str(tmp_path / "quoted.json"))
    monkeypatch.setenv("XGRAPH_KG_INDEX", str(tmp_path / "quoted"))
    monkeypatch.setenv("XGRAPH_TOKENIZER", str(tmp_path / "quoted.json"))
    monkeypatch.setenv("XGRAPH_UNQUOTED_STARTS", "1")
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=None), state)
    oracle = Oracle(index, served, SpanSyntax(), unquoted_starts=True)
    admit(processor, state, 0, served.encode("Hi"), on=True)
    processor.apply_staged_writes()
    expected = oracle.initial_state()
    for token in served.encode(" ⟦ Kleb · by · Ann⟧ wrote ⟦"):
        _, logits = step(processor, state, [0])
        assert allowed(served, logits[0]) == oracle.allowed(expected)
        sample(state, 0, token)
        expected = oracle.advance(expected, token)
    _, logits = step(processor, state, [0])
    assert allowed(served, logits[0]) == oracle.allowed(expected)
    assert served.encode(' "')[0] not in oracle.allowed(expected)  # no quoted title starts


def test_the_sitelink_rank_can_be_switched_on(vllm, served, toy_kg, tmp_path, monkeypatch):
    _, state = vllm
    index = build_kg_index(toy_kg, served, SpanSyntax(), sitelinks={"Q6": 40, "Q17": 3})
    save_kg_index(index, tmp_path / "ranked")
    monkeypatch.setenv("XGRAPH_PATHS", "1")
    monkeypatch.setenv("XGRAPH_COMMITTED_PATHS", "1")
    monkeypatch.setenv("XGRAPH_START_RANK", "sitelinks")
    with pytest.raises(ValueError, match="sitelink counts"):  # the fixture's index has none
        XGraphLogitsProcessor(SimpleNamespace(speculative_config=None), state)
    monkeypatch.setenv("XGRAPH_KG_INDEX", str(tmp_path / "ranked"))
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=None), state)
    modes = {"paths": True, "committed_paths": True}
    oracle = Oracle(index, served, SpanSyntax(), start_rank="sitelinks", **modes)
    degree = Oracle(index, served, SpanSyntax(), **modes)
    admit(processor, state, 0, served.encode("Hi"), on=True)
    processor.apply_staged_writes()
    expected, other = oracle.initial_state(), degree.initial_state()
    for token in served.encode(" ⟦ Curie · place of birth⟧ in ⟦ Paris⟧ near ⟦ Paris"):
        _, logits = step(processor, state, [0])
        assert allowed(served, logits[0]) == oracle.allowed(expected)
        sample(state, 0, token)
        expected, other = oracle.advance(expected, token), degree.advance(other, token)
    _, logits = step(processor, state, [0])
    # Committed to Q6, the answer is its Paris (Q3), from which a later path can start.
    assert allowed(served, logits[0]) == oracle.allowed(expected) != degree.allowed(other)


def test_bound_escapes_can_be_switched_off(vllm, served, toy_kg, monkeypatch):
    _, state = vllm
    monkeypatch.setenv("XGRAPH_PATHS", "1")
    monkeypatch.setenv("XGRAPH_BOUND_ESCAPES", "0")
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=None), state)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    oracle = Oracle(index, served, SpanSyntax(), paths=True, bound_escapes=False)
    admit(processor, state, 0, served.encode("Hi"), on=True)
    processor.apply_staged_writes()
    expected = oracle.initial_state()
    for token in served.encode(" ⟦ Pierre Curie · spouse⟧ so ⟦"):
        _, logits = step(processor, state, [0])
        assert allowed(served, logits[0]) == oracle.allowed(expected)
        sample(state, 0, token)
        expected = oracle.advance(expected, token)
    _, logits = step(processor, state, [0])
    assert allowed(served, logits[0]) == oracle.allowed(expected)  # no escape at the answer


# Edits by Wikidata number, as a version file holds them: Marie Curie's employer becomes
# Paris (Q3), and Q4's Paris gains a relation.
MOVED = [[1, 108, 3], [4, 131, 12]]


@pytest.mark.parametrize("vllm", ["cpu", "cuda"], indirect=True)
@pytest.mark.parametrize("delivery", ["plain", "prefill"])
@pytest.mark.parametrize("answer", ["Cedar", "Delta"])
def test_version_isolation_adapter_replay_preemption_and_slot_reuse(
    vllm, served, tmp_path, monkeypatch, delivery, answer
):
    from tests.test_engine import assert_agrees, engine_state, expected_state
    from tests.test_oracle import (
        ISOLATION_ANSWERS,
        ISOLATION_EDITS,
        ISOLATION_GRAPHS,
        ISOLATION_KG,
    )
    from xgraph.kgversions import write_kg_edits
    from xgraph.vllm import collect_replay_status

    torch.manual_seed(17)
    _, state = vllm
    syntax = SpanSyntax()
    index = build_kg_index(ISOLATION_KG, served, syntax)
    save_kg_index(index, tmp_path / "isolated-index")
    write_kg_edits(tmp_path / "isolated-versions", ISOLATION_EDITS, index=index)
    monkeypatch.setenv("XGRAPH_KG_INDEX", str(tmp_path / "isolated-index"))
    monkeypatch.setenv("XGRAPH_KG_VERSIONS", str(tmp_path / "isolated-versions"))
    for setting, value in (
        ("XGRAPH_PATHS", "1"),
        ("XGRAPH_COMMITTED_PATHS", "1"),
        ("XGRAPH_ESCAPES", "0"),
    ):
        monkeypatch.setenv(setting, value)
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=object()), state)
    options = dict(paths=True, committed_paths=True, escapes=False)
    oracle = Oracle(index, served, syntax, versions=ISOLATION_EDITS, **options)
    state.all_token_ids.gpu = torch.zeros((SLOTS, 256), dtype=torch.int32, device=state.device)
    for round_id, bindings in enumerate(
        (
            (None, "hybrid-source", "inner", "deleted"),
            ("terminal", "inner", "hybrid-source", None),
            (None, None, None, None),
        )
    ):
        refs, ours, expected, arguments, texts = {}, {}, {}, {}, {}
        prefill = "⟦ Atlas ·" if delivery == "prefill" else ""
        for slot, version in enumerate(bindings):
            ref_index = build_kg_index(
                replace(ISOLATION_KG, triples=ISOLATION_GRAPHS[version]), served, syntax
            )
            refs[slot] = (ref_index, Oracle(ref_index, served, syntax, **options))
            expected[slot] = refs[slot][1].anchored_state(["Q1"])
            ours[slot] = oracle.anchored_state(["Q1"], version)
            arguments[slot] = {"xgraph_anchor": ["Q1"]}
            if version is not None:
                arguments[slot]["xgraph_kg_version"] = version
            if prefill:
                arguments[slot].update(
                    xgraph_prefill=prefill, xgraph_replay_id=f"isolation-{round_id}-{slot}"
                )
            prompt = served.encode("Ignore this example: ⟦ Delta⟧. ") + served.encode(prefill)
            admit(processor, state, slot, prompt, True, **arguments[slot])
            for token in served.encode(prefill):
                ours[slot] = oracle.advance(ours[slot], token)
                expected[slot] = refs[slot][1].advance(expected[slot], token)
            text = " creator" if prefill else "⟦ Atlas · creator"
            texts[slot] = served.encode(text + f" · born⟧ Answer: ⟦ {answer}⟧")
        processor.apply_staged_writes()
        rejected = set()
        for position in range(len(texts[0]) + 1):
            # Move requests to different slots, then reconstruct from their
            # prompt tails and generated histories, as after vLLM preemption.
            if position == 2:
                previous = (1, 3, 0, 2)
                for field in ("all_token_ids", "prompt_len", "prefill_len", "total_len"):
                    buffer = getattr(state, field).gpu
                    buffer.copy_(buffer[list(previous)].clone())
                refs, ours, expected, arguments, texts = (
                    {slot: values[old] for slot, old in enumerate(previous)}
                    for values in (refs, ours, expected, arguments, texts)
                )
                bindings = tuple(bindings[old] for old in previous)
                for slot in (3, 1, 2, 0):
                    processor.add_request(slot, params(xgraph=1, **arguments[slot]))
                processor.apply_staged_writes()
            _, logits = step(processor, state, [3, 1, 2, 0])
            for row, slot in enumerate((3, 1, 2, 0)):
                ref_index, ref_oracle = refs[slot]
                legal = ref_oracle.allowed(expected[slot])
                assert allowed(served, logits[row]) == legal, (round_id, slot, position)
                assert oracle.allowed(ours[slot]) == legal
                assert replace(ours[slot], version=None) == expected[slot]
                if state.device.type == "cpu":
                    assert replace(processor._engine.states[slot], version=None) == expected[slot]
                else:
                    assert_agrees(
                        expected_state(expected[slot], ref_index, syntax),
                        engine_state(processor._engine.states_host(), slot, ref_index),
                        frontier=True,
                    )
                if slot in rejected or position == len(texts[slot]):
                    continue
                token = texts[slot][position]
                if token not in legal:
                    rejected.add(slot)
                    continue
                sample(state, slot, token)
                ours[slot] = oracle.advance(ours[slot], token)
                expected[slot] = ref_oracle.advance(expected[slot], token)
        for slot, version in enumerate(bindings):
            assert (slot in rejected) == (ISOLATION_ANSWERS[version] != answer)
            if slot not in rejected:
                assert expected[slot].last_span.entity_ids == (
                    ("Q4",) if answer == "Cedar" else ("Q5",)
                )
        if prefill:
            keys = [arguments[slot]["xgraph_replay_id"] for slot in range(SLOTS)]
            assert collect_replay_status(None, keys) == dict.fromkeys(keys, "replayed")


@pytest.mark.parametrize("vllm", ["cpu", "cuda"], indirect=True)
def test_version_isolation_adapter_rejects_another_base_with_identical_shape(
    vllm, served, toy_kg, tmp_path, monkeypatch
):
    from xgraph.index import IndexMismatch
    from xgraph.kgversions import write_kg_edits

    _, state = vllm
    base = build_kg_index(toy_kg, served, SpanSyntax())
    other = build_kg_index(replace(toy_kg, triples=toy_kg.triples[::-1]), served, SpanSyntax())
    assert base.labels.num_rows == other.labels.num_rows
    assert base.base_fingerprint != other.base_fingerprint
    write_kg_edits(tmp_path / "bound-versions", {"edit": [("Q1", "P108", "Q3")]}, index=base)
    save_kg_index(other, tmp_path / "other-base")
    monkeypatch.setenv("XGRAPH_KG_INDEX", str(tmp_path / "other-base"))
    monkeypatch.setenv("XGRAPH_KG_VERSIONS", str(tmp_path / "bound-versions"))
    with pytest.raises(IndexMismatch, match="base fingerprint"):
        XGraphLogitsProcessor(SimpleNamespace(speculative_config=None), state)


def test_requests_in_one_batch_read_their_own_kg_versions(
    vllm, served, toy_kg, tmp_path, monkeypatch
):
    from xgraph.kgversions import write_kg_versions

    _, state = vllm
    write_kg_versions(tmp_path / "versions", {"moved": np.array(MOVED, dtype=np.int32)})
    monkeypatch.setenv("XGRAPH_KG_VERSIONS", str(tmp_path / "versions"))
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=None), state)
    oracle = Oracle(
        build_kg_index(toy_kg, served, SpanSyntax()), served, SpanSyntax(),
        versions={"moved": [("Q1", "P108", "Q3"), ("Q4", "P131", "Q12")]},
    )  # fmt: skip
    prompt = served.encode("Hi")
    admit(processor, state, 0, prompt, on=True)
    processor.add_request(0, params(xgraph=1, xgraph_kg_version="moved"))
    admit(processor, state, 1, prompt, on=True)
    processor.apply_staged_writes()
    tokens = {
        0: served.encode(" ⟦ Marie Curie · employer · Paris⟧ ok"),
        1: served.encode(" ⟦ Marie Curie · employer · University of Paris⟧ ok"),
    }
    expected = {0: oracle.initial_state("moved"), 1: oracle.initial_state()}
    for i in range(max(map(len, tokens.values()))):
        slots = [slot for slot in (0, 1) if i < len(tokens[slot])]
        _, logits = step(processor, state, slots)
        for row, slot in enumerate(slots):
            assert allowed(served, logits[row]) == oracle.allowed(expected[slot]), (i, slot)
            sample(state, slot, tokens[slot][i])
            expected[slot] = oracle.advance(expected[slot], tokens[slot][i])
    assert expected[0].last_span.triples == (("Q1", "P108", "Q3"),)
    # The slot's next request, without a version, reads the base KG again.
    admit(processor, state, 0, prompt, on=True)
    processor.apply_staged_writes()
    base = oracle.initial_state()
    for token in tokens[1]:
        _, logits = step(processor, state, [0])
        assert allowed(served, logits[0]) == oracle.allowed(base)
        sample(state, 0, token)
        base = oracle.advance(base, token)


def test_kg_version_values_are_checked(tmp_path, monkeypatch):
    from xgraph.kgversions import write_kg_versions

    monkeypatch.delenv("XGRAPH_KG_VERSIONS", raising=False)
    with pytest.raises(ValueError, match="no KG versions are registered"):
        XGraphLogitsProcessor.validate_params(params(xgraph=1, xgraph_kg_version="w0"))
    write_kg_versions(tmp_path / "versions", {"w0": np.zeros((0, 3), dtype=np.int32)})
    monkeypatch.setenv("XGRAPH_KG_VERSIONS", str(tmp_path / "versions"))
    XGraphLogitsProcessor.validate_params(params(xgraph=1, xgraph_kg_version="w0"))
    with pytest.raises(ValueError, match="no KG version 'w1'"):
        XGraphLogitsProcessor.validate_params(params(xgraph=1, xgraph_kg_version="w1"))
    with pytest.raises(ValueError, match="xgraph_kg_version"):
        XGraphLogitsProcessor.validate_params(params(xgraph=1, xgraph_kg_version=3))


def test_a_request_can_start_inside_a_span(vllm, served, toy_kg, tmp_path, monkeypatch):
    """xgraph_open: the prompt ends with the opener (a prefilled assistant turn), which the
    engine never reads, and the request's first token continues the span; it composes with
    an anchor and a KG version, and a request beside it, or after it in its slot, starts in
    free text."""
    from xgraph.kgversions import write_kg_versions

    _, state = vllm
    write_kg_versions(tmp_path / "versions", {"moved": np.array(MOVED, dtype=np.int32)})
    monkeypatch.setenv("XGRAPH_KG_VERSIONS", str(tmp_path / "versions"))
    monkeypatch.setenv("XGRAPH_DERIVATION", "1")
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=None), state)
    oracle = Oracle(
        build_kg_index(toy_kg, served, SpanSyntax()), served, SpanSyntax(), derivation=True,
        versions={"moved": [("Q1", "P108", "Q3"), ("Q4", "P131", "Q12")]},
    )  # fmt: skip
    prompt = served.encode("Answer: ⟦")
    admit(processor, state, 0, prompt, on=True)
    processor.add_request(
        0, params(xgraph=1, xgraph_open=1, xgraph_anchor=["Q1"], xgraph_kg_version="moved")
    )
    admit(processor, state, 1, prompt, on=True)  # the same prompt, not opened
    processor.apply_staged_writes()
    tokens = {
        0: served.encode(" Marie Curie · employer · Paris⟧ ok"),
        1: served.encode(" Fine ⟦ Paris⟧"),
    }
    expected = {
        0: oracle.opened_state(oracle.anchored_state(["Q1"], "moved")),
        1: oracle.initial_state(),
    }
    for i in range(max(map(len, tokens.values()))):
        slots = [slot for slot in (0, 1) if i < len(tokens[slot])]
        _, logits = step(processor, state, slots)
        for row, slot in enumerate(slots):
            assert allowed(served, logits[row]) == oracle.allowed(expected[slot]), (i, slot)
            sample(state, slot, tokens[slot][i])
            expected[slot] = oracle.advance(expected[slot], tokens[slot][i])
    assert expected[0].last_span.triples == (("Q1", "P108", "Q3"),)
    admit(processor, state, 0, prompt, on=True)  # the slot's next request: not opened
    processor.apply_staged_writes()
    _, logits = step(processor, state, [0])
    assert allowed(served, logits[0]) == oracle.allowed(oracle.initial_state())


def test_open_values_are_checked():
    XGraphLogitsProcessor.validate_params(params(xgraph=1, xgraph_open=1))
    XGraphLogitsProcessor.validate_params(params(xgraph=1, xgraph_open=0))
    for value in ("yes", 2, 1.0, [1]):
        with pytest.raises(ValueError, match="xgraph_open"):
            XGraphLogitsProcessor.validate_params(params(xgraph=1, xgraph_open=value))
    with pytest.raises(ValueError, match="needs the engine"):
        XGraphLogitsProcessor.validate_params(params(xgraph_open=1))


PATH_SPAN = "⟦ Marie Curie · employer⟧"


def replaying(toy_kg, served, tmp_path, monkeypatch, state):
    """A path-mode processor with the MOVED KG version, and its oracle."""
    from xgraph.kgversions import write_kg_versions

    write_kg_versions(tmp_path / "versions", {"moved": np.array(MOVED, dtype=np.int32)})
    monkeypatch.setenv("XGRAPH_KG_VERSIONS", str(tmp_path / "versions"))
    monkeypatch.setenv("XGRAPH_PATHS", "1")
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=None), state)
    oracle = Oracle(
        build_kg_index(toy_kg, served, SpanSyntax()), served, SpanSyntax(), paths=True,
        versions={"moved": [("Q1", "P108", "Q3"), ("Q4", "P131", "Q12")]},
    )  # fmt: skip
    return processor, oracle


@pytest.mark.parametrize("family", ["qwen", "gemma"])
@pytest.mark.parametrize("delivery", ["plain", "opened", "prefill", "complete prefill"])
@pytest.mark.parametrize("end_after_answer", [False, True])
def test_thought_prompt_is_skipped_before_the_answer_mask(
    vllm, served, toy_kg, tmp_path, monkeypatch, family, delivery, end_after_answer
):
    """A native thought can contain invalid, closed and unfinished draft paths.
    None is replayed: only generated answer tokens and its explicit prefill bind.
    Beside it, the unconstrained thought-stage request is never masked.
    """
    _, state = vllm
    state.all_token_ids.gpu = torch.zeros((SLOTS, 512), dtype=torch.int32, device="cuda")
    processor, oracle = replaying(toy_kg, served, tmp_path, monkeypatch, state)
    eos = served.encode("<|endoftext|>")[0]
    if end_after_answer:
        monkeypatch.setenv("XGRAPH_END_AFTER_ANSWER", "1")
        monkeypatch.setitem(sys.modules, "vllm.tokenizers", SimpleNamespace(
            cached_tokenizer_from_config=lambda config: SimpleNamespace(eos_token_id=eos),
        ))  # fmt: skip
        config = SimpleNamespace(model_config=SimpleNamespace(
            try_get_generation_config=lambda: {"eos_token_id": [eos]},
        ))  # fmt: skip
        processor = XGraphLogitsProcessor(config, state)
        oracle = Oracle(
            build_kg_index(toy_kg, served, SpanSyntax()), served, SpanSyntax(), paths=True,
            versions={"moved": [("Q1", "P108", "Q3"), ("Q4", "P131", "Q12")]},
            end_after_answer=True, eos_token_ids=[eos],
        )  # fmt: skip
    draft = "⟦ Paris · located in⟧ ⟦ Zorblax"
    thought = (
        f"<think>\n{draft}\n</think>\n\n"
        if family == "qwen"
        else f"<|channel>thought\n{draft}\n<channel|>"
    )
    extra = {"xgraph_anchor": ["Q1"], "xgraph_kg_version": "moved"}
    expected = oracle.anchored_state(["Q1"], "moved")
    prefix = ""
    if delivery == "opened":
        prefix = "⟦"
        extra["xgraph_open"] = 1
        expected = oracle.opened_state(expected)
    elif delivery in ("prefill", "complete prefill"):
        prefix = PATH_SPAN + ("\nAnswer: ⟦ Paris⟧" if delivery == "complete prefill" else "")
        extra["xgraph_prefill"] = prefix
        for token in served.encode(prefix):
            expected = oracle.advance(expected, token)
    prompt = served.encode("Where?\n" + thought) + served.encode(prefix)
    assert admit(processor, state, 0, prompt, on=True, **extra)
    assert not admit(processor, state, 1, served.encode("Where?"), on=False)
    processor.apply_staged_writes()
    answer = "\nAnswer: ⟦ Paris⟧"
    if delivery == "plain":
        answer = PATH_SPAN + answer
    elif delivery == "opened":
        answer = PATH_SPAN.removeprefix("⟦") + answer
    elif delivery == "complete prefill":
        answer = ""
    tokens = served.encode(answer) + ([eos] if end_after_answer else [])
    for position, token in enumerate(tokens):
        if position == 3:  # Resume the same prompt+answer history after preemption.
            processor.add_request(0, params(xgraph=1, **extra))
            processor.apply_staged_writes()
        original, logits = step(processor, state, [0, 1])
        assert allowed(served, logits[0]) == oracle.allowed(expected)
        assert torch.equal(logits[1], original[1])
        assert token in oracle.allowed(expected)
        sample(state, 0, token)
        sample(state, 1, token)
        expected = oracle.advance(expected, token)
    assert expected.last_span.entity_ids == ("Q3",)
    if end_after_answer:
        # Preemption after a completed answer must retain the EOS-only boundary.
        processor.add_request(0, params(xgraph=1, **extra))
        processor.apply_staged_writes()
        original, logits = step(processor, state, [0, 1])
        assert allowed(served, logits[0]) == {eos}
        assert torch.equal(logits[1], original[1])
    assert processor.replays["tail mismatch"] == processor.replays["disallowed"] == 0


def test_a_request_can_replay_its_prefilled_path_span(vllm, served, toy_kg, tmp_path, monkeypatch):
    """xgraph_prefill: the prompt ends with a prefilled assistant turn holding a path span,
    and the engine advances over those prompt tokens, through the request's anchor and KG
    version, so the answer after it is bound to what the path reached; a request beside it
    with the same prompt, and the slot's next request, start at the end of the prompt."""
    _, state = vllm
    processor, oracle = replaying(toy_kg, served, tmp_path, monkeypatch, state)
    prefill = served.encode(PATH_SPAN)
    prompt = served.encode("Where? ") + prefill  # the prefill's own tokens end the prompt
    admit(processor, state, 0, prompt, on=True)
    arguments = {"xgraph_anchor": ["Q1"], "xgraph_kg_version": "moved"}
    processor.add_request(0, params(xgraph=1, xgraph_prefill=PATH_SPAN, **arguments))
    admit(processor, state, 1, prompt, on=True)  # the same prompt, nothing replayed
    processor.apply_staged_writes()
    expected = {0: oracle.anchored_state(["Q1"], "moved"), 1: oracle.initial_state()}
    for token in prefill:  # what the engine must reach by replaying the prompt's tail
        expected[0] = oracle.advance(expected[0], token)
    tokens = {0: served.encode(".\nAnswer: ⟦ Paris⟧"), 1: served.encode(" ok ⟦ Paris⟧")}
    for i in range(max(map(len, tokens.values()))):
        slots = [slot for slot in (0, 1) if i < len(tokens[slot])]
        _, logits = step(processor, state, slots)
        for row, slot in enumerate(slots):
            assert allowed(served, logits[row]) == oracle.allowed(expected[slot]), (i, slot)
            sample(state, slot, tokens[slot][i])
            expected[slot] = oracle.advance(expected[slot], tokens[slot][i])
    # Under the version, Marie Curie's employer is Q3's Paris: the answer named it.
    assert expected[0].last_span.entity_ids == ("Q3",)
    assert processor.replays == {"replayed": 1, "tail mismatch": 0, "disallowed": 0}
    admit(processor, state, 0, prompt, on=True)  # the slot's next request: not replayed
    processor.apply_staged_writes()
    _, logits = step(processor, state, [0])
    assert allowed(served, logits[0]) == oracle.allowed(oracle.initial_state())


def test_a_prefill_that_shares_a_token_with_the_text_before_it_is_replayed_whole(
    vllm, served, toy_kg, tmp_path, monkeypatch
):
    """The prefill's first bytes may share a token with the text before them (here the
    space before the opener): that token is replayed whole, from free text."""
    _, state = vllm
    processor, oracle = replaying(toy_kg, served, tmp_path, monkeypatch, state)
    prompt = served.encode("Where? " + PATH_SPAN)
    assert prompt[-len(served.encode(PATH_SPAN)) :] != served.encode(PATH_SPAN)
    admit(processor, state, 2, prompt, on=True)
    processor.add_request(2, params(xgraph=1, xgraph_prefill=PATH_SPAN))
    processor.apply_staged_writes()
    expected = oracle.initial_state()
    for token in served.encode(" " + PATH_SPAN):
        expected = oracle.advance(expected, token)
    for token in served.encode(".\nAnswer: ⟦ University of Paris⟧"):
        _, logits = step(processor, state, [2])
        assert allowed(served, logits[0]) == oracle.allowed(expected)
        sample(state, 2, token)
        expected = oracle.advance(expected, token)
    assert processor.replays["replayed"] == 1


@pytest.mark.parametrize(
    ("prompt_text", "prefill", "reason"),
    [
        ("Where? ⟦ Marie Curie · spouse⟧", PATH_SPAN, "tail mismatch"),  # another span
        ("Where? " + PATH_SPAN + "\n", PATH_SPAN, "tail mismatch"),  # text after it
        (PATH_SPAN, "Where? " + PATH_SPAN, "tail mismatch"),  # longer than the prompt
        ("Where? ⟦ Zorblax · employer⟧", "⟦ Zorblax · employer⟧", "disallowed"),  # not the KG
    ],
)
def test_a_prefill_the_engine_cannot_replay_is_declined_and_counted(
    vllm, served, toy_kg, tmp_path, monkeypatch, prompt_text, prefill, reason
):
    """A prompt that does not end with the prefill, or a replayed token the mask
    disallows, leaves the request unmasked (declined), never masked from a wrong state."""
    _, state = vllm
    processor, _ = replaying(toy_kg, served, tmp_path, monkeypatch, state)
    admit(processor, state, 1, served.encode(prompt_text), on=True)
    key = f"decline-{reason}-{prompt_text}"
    processor.add_request(1, params(xgraph=1, xgraph_prefill=prefill, xgraph_replay_id=key))
    admit(processor, state, 0, served.encode("Hi"), on=True)  # beside it, masked as usual
    processor.apply_staged_writes()
    for token in served.encode(".\nAnswer: ⟦ Paris⟧"):
        original, logits = step(processor, state, [1, 0])
        assert torch.equal(logits[0], original[0])
        sample(state, 1, token)
        sample(state, 0, token)
    assert not torch.equal(logits[1], original[1])  # inside a span: masked
    expected = {"replayed": 0, "tail mismatch": 0, "disallowed": 0} | {reason: 1}
    assert processor.replays == expected

    from xgraph.vllm import collect_replay_status

    assert collect_replay_status(None, [key]) == {key: reason}


def test_prefill_values_are_checked():
    XGraphLogitsProcessor.validate_params(params(xgraph=1, xgraph_prefill=PATH_SPAN))
    for value in ("", 1, [1, 2]):
        with pytest.raises(ValueError, match="xgraph_prefill"):
            XGraphLogitsProcessor.validate_params(params(xgraph=1, xgraph_prefill=value))
    with pytest.raises(ValueError, match="needs the engine"):
        XGraphLogitsProcessor.validate_params(params(xgraph_prefill=PATH_SPAN))
    with pytest.raises(ValueError, match="xgraph_open"):
        XGraphLogitsProcessor.validate_params(
            params(xgraph=1, xgraph_prefill=PATH_SPAN, xgraph_open=1)
        )


@pytest.mark.parametrize(
    ("paths", "committed", "escapes", "anchor"),
    [
        (False, False, True, None),
        (True, False, True, None),
        (True, True, False, None),
        (True, True, False, ["Q1"]),
        (True, False, False, ["Q1"]),
    ],
)
def test_training_masks_equal_the_served_masks_under_the_same_settings(
    vllm, served, tmp_path, monkeypatch, paths, committed, escapes, anchor
):
    """The masked span loss (xgraph.train, xgraph.policy) normalizes over the tokens the
    served engine allows: for the same reply, the trainer's mask at every position equals
    the processor's under the same paths, committed paths, escapes, and anchor; a trainer
    left at its default escapes differs from a server without them."""
    from xgraph.policy import mask_reply
    from xgraph.train import engines

    _, state = vllm
    monkeypatch.setenv("XGRAPH_PATHS", "1" if paths else "0")
    monkeypatch.setenv("XGRAPH_COMMITTED_PATHS", "1" if committed else "0")
    monkeypatch.setenv("XGRAPH_ESCAPES", "1" if escapes else "0")
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=None), state)
    reply = served.encode(" ⟦ Marie Curie · employer · located in⟧\nAnswer: ⟦ Paris⟧ ok")
    admit(processor, state, 0, served.encode("Where?"), on=True)
    if anchor:
        processor.add_request(0, params(xgraph=1, xgraph_anchor=anchor))
    processor.apply_staged_writes()
    served_masks = []
    for token in reply:
        _, logits = step(processor, state, [0])
        served_masks.append(allowed(served, logits[0]))
        sample(state, 0, token)

    worlds = [f"original={tmp_path / 'index'}"]

    def trainer_masks(**settings):
        engine = engines(worlds, served, paths=paths, committed_paths=committed, **settings)
        logits = torch.zeros((len(reply), state.vocab_size), device="cuda")
        masked = mask_reply(engine["original"], logits, reply, anchor)
        return [allowed(served, row) for row in masked]

    assert trainer_masks(escapes=escapes) == served_masks
    if not escapes:
        assert trainer_masks() != served_masks  # the default keeps the escape marker


def test_expressions_can_be_switched_on(vllm, tmp_path, monkeypatch):
    """XGRAPH_EXPRESSIONS=1 (with path mode and committed paths) lets a span be a date
    comparison, masked as the oracle masks it, the answer after it bound to its result."""
    from tests.test_oracle import EXPRESSIONS, compare_kg, compare_tokenizer

    _, state = vllm
    kg, dates, served = compare_kg()
    index = build_kg_index(kg, served, SpanSyntax(), dates=dates)
    save_kg_index(index, tmp_path / "dated")
    compare_tokenizer(kg).save(str(tmp_path / "dated.json"))
    monkeypatch.setenv("XGRAPH_KG_INDEX", str(tmp_path / "dated"))
    monkeypatch.setenv("XGRAPH_TOKENIZER", str(tmp_path / "dated.json"))
    monkeypatch.setenv("XGRAPH_EXPRESSIONS", "1")
    monkeypatch.setenv("XGRAPH_PATHS", "1")
    with pytest.raises(ValueError, match="XGRAPH_COMMITTED_PATHS=1"):
        XGraphLogitsProcessor(SimpleNamespace(speculative_config=None), state)
    monkeypatch.setenv("XGRAPH_COMMITTED_PATHS", "1")
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=None), state)
    oracle = Oracle(index, served, SpanSyntax(), **EXPRESSIONS)
    admit(processor, state, 0, served.encode("Hi"), on=True)
    processor.apply_staged_writes()
    expected = oracle.initial_state()
    for token in served.encode(" ⟦ earlier: Alien | Duel · publication date⟧ so ⟦"):
        _, logits = step(processor, state, [0])
        assert allowed(served, logits[0]) == oracle.allowed(expected)
        sample(state, 0, token)
        expected = oracle.advance(expected, token)
    _, logits = step(processor, state, [0])
    assert allowed(served, logits[0]) == oracle.allowed(expected)
    assert expected.last_span.entity_ids == ("Q5",)  # Duel, 13 November 1971
    assert served.encode(" Alien")[0] not in oracle.allowed(expected)


@pytest.mark.parametrize("delivery", ["normal", "opened", "prefill"])
def test_residual_goal_stop_questions_reach_their_slots(
    vllm, served, toy_kg, tmp_path, monkeypatch, delivery
):
    from xgraph.kgversions import write_kg_versions

    _, state = vllm
    monkeypatch.setenv("XGRAPH_STOP", "residual-goal")
    monkeypatch.setenv("XGRAPH_PATHS", "1")
    monkeypatch.setenv("XGRAPH_COMMITTED_PATHS", "1")
    edits = [("Q7", "P131", "Q99"), ("Q7", "P276", "Q99")]
    write_kg_versions(
        tmp_path / "stop-versions",
        {
            "no-location": np.array([[7, 131, 99], [7, 276, 99]], dtype=np.int32),
        },
    )
    monkeypatch.setenv("XGRAPH_KG_VERSIONS", str(tmp_path / "stop-versions"))
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=object()), state)
    oracle = Oracle(
        build_kg_index(toy_kg, served, SpanSyntax()),
        served,
        SpanSyntax(),
        paths=True,
        committed_paths=True,
        stop="residual-goal",
        versions={"no-location": edits},
    )
    prefix = "⟦ Marie Curie · employer"
    questions = {
        0: "Which employer has the location of Marie Curie?",
        2: "Which employer has Marie Curie?",
        3: "Which employer has the location of Marie Curie?",
    }
    expected = {}
    for slot in (3, 0, 2):  # admission and logits order both differ from slot order
        version = "no-location" if slot == 3 else None
        extra = {"xgraph_question": questions[slot], "xgraph_anchor": ["Q1"]}
        if version:
            extra["xgraph_kg_version"] = version
        if delivery == "opened":
            extra["xgraph_open"] = 1
        if delivery == "prefill":
            extra["xgraph_prefill"] = prefix
        prompt = prefix if delivery == "prefill" else ("⟦" if delivery == "opened" else "Hi")
        admit(processor, state, slot, served.encode(prompt), on=True, **extra)
        expected[slot] = oracle.anchored_state(["Q1"], version, question=questions[slot])
        if delivery == "opened":
            expected[slot] = oracle.opened_state(expected[slot])
        if delivery == "prefill":
            for token in served.encode(prefix):
                expected[slot] = oracle.advance(expected[slot], token)
    admit(processor, state, 1, served.encode("Hi"), on=False)
    processor.apply_staged_writes()
    continuation = (
        ""
        if delivery == "prefill"
        else prefix.removeprefix("⟦")
        if delivery == "opened"
        else prefix
    )
    for token in served.encode(continuation):
        original, logits = step(processor, state, [3, 1, 0, 2])
        assert torch.equal(original[1], logits[1])
        for row, slot in ((0, 3), (2, 0), (3, 2)):
            assert allowed(served, logits[row]) == oracle.allowed(expected[slot])
            sample(state, slot, token)
            expected[slot] = oracle.advance(expected[slot], token)
    original, logits = step(processor, state, [3, 1, 0, 2])
    assert torch.equal(original[1], logits[1])
    for row, slot in ((0, 3), (2, 0), (3, 2)):
        assert allowed(served, logits[row]) == oracle.allowed(expected[slot])
    assert allowed(served, logits[0]) == allowed(served, logits[3]) != allowed(served, logits[2])
    # Reuse the stopped slot with a question whose residual location keeps it open.
    admit(
        processor,
        state,
        2,
        served.encode(prefix),
        on=True,
        xgraph_question=questions[0],
        xgraph_prefill=prefix,
    )
    processor.apply_staged_writes()
    _, reused = step(processor, state, [2])
    assert allowed(served, reused[0]) == allowed(served, logits[2])


def test_residual_goal_stop_is_applied_to_adapter_draft_rows(vllm, served, toy_kg, monkeypatch):
    _, state = vllm
    monkeypatch.setenv("XGRAPH_STOP", "residual-goal")
    monkeypatch.setenv("XGRAPH_PATHS", "1")
    monkeypatch.setenv("XGRAPH_COMMITTED_PATHS", "1")
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=object()), state)
    question = "Which employer has Marie Curie?"
    admit(processor, state, 2, served.encode("Hi"), on=True, xgraph_question=question)
    processor.apply_staged_writes()
    committed = served.encode("⟦ Marie Curie")
    for token in committed:
        step(processor, state, [2])
        sample(state, 2, token)
    drafts = served.encode(" · employer⟧ so ⟦ University of Paris⟧")
    rows = len(drafts) + 1
    ctx = SimpleNamespace(
        idx_mapping=torch.tensor([2], dtype=torch.int32, device="cuda"),
        expanded_idx_mapping=torch.full((rows,), 2, dtype=torch.int32, device="cuda"),
        expanded_local_pos=torch.arange(rows, dtype=torch.int32, device="cuda"),
        input_ids=torch.tensor([committed[-1], *drafts], dtype=torch.int32, device="cuda"),
    )
    oracle = Oracle(
        build_kg_index(toy_kg, served, SpanSyntax()),
        served,
        SpanSyntax(),
        paths=True,
        committed_paths=True,
        stop="residual-goal",
    )
    expected = oracle.initial_state(question=question)
    for token in committed:
        expected = oracle.advance(expected, token)
    logits = processor.apply(torch.zeros((rows, state.vocab_size), device="cuda"), ctx)
    for k in range(rows):
        assert allowed(served, logits[k]) == oracle.allowed(expected)
        if k < len(drafts):
            expected = oracle.advance(expected, drafts[k])


def test_residual_goal_stop_requires_question_text_in_request_params(monkeypatch):
    monkeypatch.setenv("XGRAPH_STOP", "residual-goal")
    for value in (None, "", "   ", 3, []):
        with pytest.raises(ValueError, match="xgraph_question"):
            XGraphLogitsProcessor.validate_params(params(xgraph=1, xgraph_question=value))
    XGraphLogitsProcessor.validate_params(params(xgraph=1, xgraph_question="Which employer?"))
    XGraphLogitsProcessor.validate_params(params())  # declined requests need no question


def test_delivery_stop_override_matches_no_stop_oracle_and_clears_reused_slot(
    vllm, served, toy_kg, monkeypatch
):
    _, state = vllm
    monkeypatch.setenv("XGRAPH_PATHS", "1")
    monkeypatch.setenv("XGRAPH_COMMITTED_PATHS", "1")
    monkeypatch.setenv("XGRAPH_STOP", "residual-goal")
    processor = XGraphLogitsProcessor(SimpleNamespace(speculative_config=object()), state)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    prefix = "⟦ Marie Curie · employer"
    question = "Which employer has Marie Curie?"
    for disabled in (False, True):
        oracle = Oracle(
            index,
            served,
            SpanSyntax(),
            paths=True,
            committed_paths=True,
            stop=None if disabled else "residual-goal",
        )
        expected = oracle.anchored_state(["Q1"], question=None if disabled else question)
        for token in served.encode(prefix):
            expected = oracle.advance(expected, token)
        admit(
            processor,
            state,
            0,
            served.encode(prefix),
            on=True,
            xgraph_anchor=["Q1"],
            xgraph_prefill=prefix,
            **({"xgraph_stop": 0} if disabled else {"xgraph_question": question}),
        )
        processor.apply_staged_writes()
        _, logits = step(processor, state, [0])
        assert allowed(served, logits[0]) == oracle.allowed(expected)


@pytest.mark.parametrize("reason", ["tail mismatch", "disallowed"])
@pytest.mark.parametrize("vllm", ["cpu", "cuda"], indirect=True)
def test_declined_replay_without_status_consumer_fails_request(
    vllm, served, toy_kg, tmp_path, monkeypatch, reason
):
    _, state = vllm
    processor, _ = replaying(toy_kg, served, tmp_path, monkeypatch, state)
    prefill = PATH_SPAN if reason == "tail mismatch" else "⟦ Zorblax · employer⟧"
    prompt = "Where? ⟦ Marie Curie · spouse⟧" if reason == "tail mismatch" else prefill
    admit(processor, state, 0, served.encode(prompt), True, xgraph_prefill=prefill)
    processor.apply_staged_writes()
    assert processor.replays == dict.fromkeys(processor.replays, 0)
    with pytest.raises(RuntimeError, match=r"replay declined.*status consumer"):
        step(processor, state, [0])
    assert processor.replays[reason] == 1


@pytest.mark.parametrize("reason", ["tail mismatch", "disallowed"])
@pytest.mark.parametrize("vllm", ["cpu", "cuda"], indirect=True)
def test_replay_status_survives_readmission_and_slot_reuse(
    vllm, served, toy_kg, tmp_path, monkeypatch, reason
):
    from xgraph.vllm import collect_replay_status

    _, state = vllm
    processor, oracle = replaying(toy_kg, served, tmp_path, monkeypatch, state)
    key, replacement = f"sticky-{reason}", f"replacement-{reason}"
    prefill = PATH_SPAN if reason == "tail mismatch" else "⟦ Zorblax · employer⟧"
    prompt = "Where? ⟦ Marie Curie · spouse⟧" if reason == "tail mismatch" else prefill
    admit(
        processor,
        state,
        0,
        served.encode(prompt),
        True,
        xgraph_prefill=prefill,
        xgraph_replay_id=key,
    )
    processor.apply_staged_writes()
    original, logits = step(processor, state, [0])
    assert torch.equal(original, logits)
    # Re-admission with a now-valid history cannot undo the unmasked interval.
    admit(
        processor,
        state,
        0,
        served.encode(PATH_SPAN),
        True,
        xgraph_prefill=PATH_SPAN,
        xgraph_replay_id=key,
    )
    processor.apply_staged_writes()
    step(processor, state, [0])
    # A different request in the same slot receives its own success receipt.
    admit(
        processor,
        state,
        0,
        served.encode(PATH_SPAN),
        True,
        xgraph_prefill=PATH_SPAN,
        xgraph_replay_id=replacement,
    )
    processor.apply_staged_writes()
    expected = oracle.initial_state()
    for token in served.encode(PATH_SPAN):
        expected = oracle.advance(expected, token)
    _, logits = step(processor, state, [0])
    assert allowed(served, logits[0]) == oracle.allowed(expected)
    assert collect_replay_status(None, [key, replacement]) == {key: reason, replacement: "replayed"}
    assert collect_replay_status(None, [key, replacement]) == {}
    admit(processor, state, 0, served.encode("Hi"), False)
    processor.apply_staged_writes()
    admit(processor, state, 0, served.encode(prompt), True, xgraph_prefill=prefill)
    processor.apply_staged_writes()
    with pytest.raises(RuntimeError, match=r"replay declined.*status consumer"):
        step(processor, state, [0])


@pytest.mark.parametrize("vllm", ["cpu"], indirect=True)
@pytest.mark.parametrize("reason", ["replayed", "tail mismatch", "disallowed"])
def test_replay_status_from_separately_loaded_worker_processor(vllm, served, monkeypatch, reason):
    """Worker-loaded processor code and the RPC import must share replay receipts."""
    import importlib.util

    import xgraph.vllm as adapter

    spec = importlib.util.spec_from_file_location("worker_adapter", adapter.__file__)
    worker_adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker_adapter)
    monkeypatch.setattr(worker_adapter.XGraphLogitsProcessor, "_stream", lambda self: 0)
    _, state = vllm
    processor = worker_adapter.XGraphLogitsProcessor(SimpleNamespace(), state)
    key = f"worker-module-{reason}"
    prefill = "⟦ Zorblax" if reason == "disallowed" else "⟦ Marie Curie"
    prompt = "Hi" if reason == "tail mismatch" else prefill
    admit(
        processor, state, 0, served.encode(prompt), True,
        xgraph_prefill=prefill, xgraph_replay_id=key,
    )  # fmt: skip
    processor.apply_staged_writes()
    step(processor, state, [0])
    assert processor.replays[reason] == 1

    class LLM:
        def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
            worker = adapter.XGraphWorkerExtension()
            return [getattr(worker, method)(*args, **(kwargs or {}))]

    status = adapter.replay_statuses(LLM(), [key, "missing-worker-replay"])
    assert status[key]["replay_status"] == ("replayed" if reason == "replayed" else "declined")
    if reason != "replayed":
        assert status[key]["replay_reason"] == reason
    assert status["missing-worker-replay"]["replay_status"] == "unconfirmed"
    assert adapter.replay_statuses(LLM(), [key])[key]["replay_status"] == "unconfirmed"
