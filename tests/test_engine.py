"""Engine agreement seam: the GPU engine against the oracle, step by step."""

from __future__ import annotations

import math
import os
import random
import shutil
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from tests.conftest import byte_level_tokenizer, train_byte_level_bpe
from tests.test_oracle import (
    EXPRESSIONS,
    HAND_ENTITIES,
    HAND_SYNTAX,
    ISOLATION_ANSWERS,
    ISOLATION_EDITS,
    ISOLATION_GRAPHS,
    ISOLATION_KG,
    MERGES,
    QUOTED_KG,
    STOP_CASES,
    STOP_KG,
    STOP_VERSIONS,
    SUPPORT_KG,
    SUPPORT_SYNTAX,
    SUPPORT_VERSIONS,
    TOY_SITELINKS,
    VERSION_MODES,
    case_alias_world,
    compare_kg,
    crowded_start_kg,
    crowded_tokenizer,
    dated_tokenizer,
    dated_world,
    edited_kg,
    quoted_tokenizer,
    random_dated_kg,
    random_sitelinks,
    random_versioned_kg,
    terminal_display_tokens,
    variable_stride_cases,
    versioned_tokenizer,
)
from xgraph.engine import CLOSED, PHASE_ERROR, PHASES, SEGMENTS, Engine, closed_span
from xgraph.index import IndexMismatch, build_kg_index, load_kg_index, save_kg_index
from xgraph.kg import Entity, KnowledgeGraph, Relation
from xgraph.kgversions import build_kg_versions
from xgraph.oracle import MAX_FACTS, MAX_REACHED, Oracle, OracleState, Phase, Segment
from xgraph.syntax import SpanSyntax
from xgraph.tokenizer import ServedTokenizer

pytestmark = pytest.mark.skipif(
    shutil.which("nvidia-smi") is None or os.environ.get("CUDA_VISIBLE_DEVICES") == "",
    reason="no visible NVIDIA GPU for engine agreement",
)

CODES = {Phase.FREE: 0, Phase.SPAN: 1, Phase.ESCAPED: 2}


@pytest.mark.parametrize(
    "kind", ["complete", "missing-byte", "unknown-fallback", "normalizer", "empty"]
)
def test_exact_label_byte_admission_agrees_with_native_masks(kind):
    from tests.test_label_index import label_byte_tokenizer

    served, syntax = label_byte_tokenizer(kind), SpanSyntax()
    kg = KnowledgeGraph(
        (
            Entity("a", ("A",)),
            Entity("az", ("Az",)),
            Entity("goal", ("Goal",)),
            Entity("unicode", ("é", "e\u0301")),
        ),
        (Relation("r", ("r",)),),
        (("az", "r", "goal"),),
    )
    index = build_kg_index(kg, served, syntax)
    lost = {
        "complete": set(),
        "missing-byte": {"Az"},
        "unknown-fallback": {"Az"},
        "normalizer": {"e\u0301"},
        "empty": {"Az"},
    }[kind]
    assert {entry.label for entry in index.labels.excluded} == lost
    oracle = Oracle(index, served, syntax, escapes=False)
    engine = Engine(index, served, syntax, 1, escapes=False)
    for text, owners in (("⟦ A⟧", ("a",)), ("⟦ Goal⟧", ("goal",)), ("⟦ é⟧", ("unicode",))):
        engine.reset()
        assert _agree_on_text(oracle, engine, index, served, text).last_span.entity_ids == owners
    engine.reset()
    state = _agree_on_text(oracle, engine, index, served, "⟦ A")
    assert served.encode(syntax.separator)[0] not in oracle.allowed(state)


@pytest.mark.parametrize("rank", ["degree", "sitelinks"])
def test_initial_case_alias_masks_and_shared_owner_resolutions_agree(rank):
    kg, served, syntax = case_alias_world()
    index = build_kg_index(
        kg, served, syntax, sitelinks={"Q1": 1, "Q2": 10}, initial_case_aliases=True
    )
    options = dict(paths=True, committed_paths=True, escapes=False, start_rank=rank)
    oracle = Oracle(index, served, syntax, **options)
    engine = Engine(index, served, syntax, 1, **options)
    rng = random.Random(17)
    for start in ("Sual", "sual"):
        for _ in range(8):
            text = "⟦ " + " · ".join([start, *(["via"] * rng.randint(1, 4))]) + "⟧"
            engine.reset()
            _agree_on_text(oracle, engine, index, served, text)
        text = f"⟦ {start} · via⟧ Answer: ⟦ "
        text += "new endpoint⟧" if rank == "sitelinks" else "old endpoint⟧"
        engine.reset()
        _agree_on_text(oracle, engine, index, served, text)


@pytest.mark.parametrize("seed", [2402, 2403])
@pytest.mark.parametrize("version", [None, "edit"])
def test_variable_stride_expanded_seeded_walks_and_fixed_spans_agree(
    toy_kg, toy_tokenizer, seed, version
):
    from xgraph.programs import MatchedExecutor

    served, syntax = ServedTokenizer(toy_tokenizer), SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax, sitelinks=TOY_SITELINKS)
    edits = {"edit": [("Q1", "P108", "Q2")]}
    versions = build_kg_versions(index, edits)
    options = dict(paths=True, committed_paths=True, escapes=False, start_rank="sitelinks")
    oracle = Oracle(index, served, syntax, versions=edits, **options)
    engine = Engine(index, served, syntax, 1, versions=versions, **options)
    executor = MatchedExecutor(index, served, versions=versions, start_rank="sitelinks")
    for pieces, result in variable_stride_cases(toy_kg, executor, seed, version):
        engine.reset()
        engine.set_version(0, version)
        expected = oracle.initial_state(version)
        for i, piece in enumerate(pieces):
            expected = _agree_on_text(oracle, engine, index, served, piece, expected)
            if i == 0:
                assert set(expected.last_span.entity_ids) == set(result["reached"])
        assert set(expected.last_span.entity_ids) <= set(result["witnesses"])
        assert (
            closed_span(index, engine.states_host(), 0).entity_ids == expected.last_span.entity_ids
        )


@pytest.mark.parametrize("bpe", [False, True])
def test_terminal_engine_forces_display_and_keeps_undisplayed_answer(bpe):
    from tests.test_oracle import audit_world

    _, index, served, syntax = audit_world(bpe)
    options = dict(
        paths=True,
        committed_paths=True,
        escapes=False,
        observe_terminal=1,
        eos_token_ids=sorted(served.special_ids),
    )
    oracle = Oracle(index, served, syntax, **options)
    engine = Engine(index, served, syntax, 1, **options)
    tokens = [t for p in ("⟦", " Start", " ·", " via", "⟧") for t in served.encode(p)]
    tokens += terminal_display_tokens(served)
    tokens += served.encode(" Beta") + served.encode("⟧")
    state = oracle.initial_state()
    for token in tokens:
        engine.fill_mask()
        assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(state)
        state = oracle.advance(state, token)
        engine.advance(np.array([token], dtype=np.int32))
    engine.fill_mask()
    assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(state) == served.special_ids
    assert closed_span(index, engine.states_host(), 0).entity_ids == ("Q2",)


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("order", ["identity", "reversed", "random"])
def test_terminal_seeded_walks_agree_with_versioned_and_rebuilt_engines(committed, order):
    from tests.test_oracle import audit_world

    _, _, served, syntax = audit_world()
    # Every ordinary relation has ambiguous endpoints, so seeded relation walks
    # exercise the terminal display rather than mostly reaching singleton sinks.
    kg = KnowledgeGraph(
        tuple(Entity(f"Q{i}", (name,)) for i, name in enumerate(("Start", "Beta", "Gamma"), 1)),
        (Relation("P1", ("via",)), Relation("P2", ("next",))),
        tuple((f"Q{s}", f"P{r}", f"Q{o}") for s in (1, 2, 3) for r in (1, 2) for o in (2, 3)),
    )
    seen = run_agreement(
        served,
        kg,
        syntax,
        sequences=12,
        steps=500,
        seed=842,
        paths=True,
        escapes=False,
        committed_paths=committed,
        observe_terminal=2,
        terminal_order=order,
        terminal_seed=18,
        start_rank="sitelinks",
        sitelinks={"Q3": 100, "Q2": 10},
        versions={"edit": [("Q2", "P2", "Q1")]},
    )
    assert seen["terminal display steps"] > 0 and seen["versioned steps"] > 0


@pytest.mark.parametrize(
    "order,display",
    [
        ("identity", "Gamma ; Beta (+1 more)"),
        ("reversed", "Beta ; Gamma (+1 more)"),
        ("random", "Beta ; Gamma (+1 more)"),
    ],
)
def test_terminal_order_engine_keeps_display_subset_and_undisplayed_binding(order, display):
    from tests.test_oracle import observed_world

    kg, served, syntax = observed_world()
    index = build_kg_index(kg, served, syntax, sitelinks={"Q3": 100, "Q2": 10})
    options = dict(
        paths=True,
        committed_paths=True,
        escapes=False,
        observe_terminal=2,
        terminal_order=order,
        terminal_seed=17,
        eos_token_ids=[0],
    )
    oracle = Oracle(index, served, syntax, **options)
    engine = Engine(index, served, syntax, 1, **options)
    text = f"⟦ Start · via⟧\nEndpoints: ⟦ {display}⟧.\nAnswer: ⟦ Delta⟧"
    _agree_on_text(oracle, engine, index, served, text)
    engine.reset()
    _agree_on_text(oracle, engine, index, served, text)


@pytest.mark.parametrize("aliases", [False, True])
@pytest.mark.parametrize("order", ["identity", "reversed", "random"])
def test_terminal_engine_caps_and_start_commitment_precede_display(aliases, order):
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    kg = KnowledgeGraph(
        (
            Entity("Q1", ("Start",)),
            Entity("Q2", ("Start",)),
            Entity("Q3", tuple(f"A{i:02}" for i in range(70)) if aliases else ("Alpha",)),
            Entity("Q4", ("Beta",)),
            Entity("Q5", ("Other start",)),
        ),
        (Relation("P1", ("via",)),),
        (("Q1", "P1", "Q3"), ("Q1", "P1", "Q4"), ("Q2", "P1", "Q5")),
    )
    index = build_kg_index(kg, served, syntax, sitelinks={"Q1": 100, "Q4": 1000, "Q5": 9000})
    options = dict(
        paths=True,
        committed_paths=True,
        start_rank="sitelinks",
        observe_terminal=8,
        terminal_order=order,
        eos_token_ids=[0],
    )
    oracle, engine = (
        Oracle(index, served, syntax, **options),
        Engine(index, served, syntax, 1, **options),
    )
    display = "Beta ; Alpha" if order == "identity" else "Alpha ; Beta"
    text = "⟦ Start · via⟧" + ("" if aliases else f"\nEndpoints: ⟦ {display}⟧")
    text += ".\nAnswer: ⟦ " + ("A00" if aliases else "Alpha") + "⟧"
    _agree_on_text(oracle, engine, index, served, text)
    engine.reset()
    _agree_on_text(oracle, engine, index, served, text)


@pytest.mark.parametrize("terminal", [False, True])
def test_request_path_options_mix_blind_and_menu_observed_slots_and_reset(terminal):
    from tests.test_oracle import observed_world

    torch = pytest.importorskip("torch")
    kg, served, syntax = observed_world()
    index = build_kg_index(kg, served, syntax, sitelinks={})
    base = dict(paths=True, escapes=False, committed_paths=True, end_after_answer=True,
                eos_token_ids=[0])  # fmt: skip
    view = dict(menu=2, observe=1, observe_select=True, observe_answer=True,
                observe_required=True, observe_no_loop=True)  # fmt: skip
    engine = Engine(
        index,
        served,
        syntax,
        2,
        **base,
        **view,
        observe_terminal=8 if terminal else 0,
        request_path_options=True,
    )
    slots = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    engine.begin(slots)
    engine.set_path_options(
        0, {"menu": 0, "observe": 0, "path_demos": 5, "observe_terminal": 8 if terminal else 0}
    )
    engine.set_path_options(1, {**view, "path_demos": 5})
    oracles = [Oracle(index, served, syntax, **base, observe_terminal=8 if terminal else 0),
               Oracle(index, served, syntax, **base, **view)]  # fmt: skip
    states = [o.initial_state(question="Which next?") for o in oracles]
    engine.set_relation_question(1, "Which next?")
    texts = [
        "⟦ Start · via⟧"
        + ("\nEndpoints: ⟦ Gamma ; Beta ; Delta⟧" if terminal else "")
        + ".\nAnswer: ⟦ Gamma⟧",
        "⟦ Start ·→ Start ⟦? via⟧ · via ·→ Gamma (+2 more) ⟦? next⟧"
        " ⟦= Delta ·→ Delta ⟦? ∅⟧⟧.\nAnswer: ⟦ Delta⟧",
    ]
    streams = [served.encode(t) for t in texts]
    for step in range(max(map(len, streams)) + 1):
        engine.fill_mask()
        masks = engine.mask_host()
        tokens = []
        for slot, oracle in enumerate(oracles):
            assert allowed_from_mask(masks[slot]) == oracle.allowed(states[slot]), (slot, step)
            token = streams[slot][step] if step < len(streams[slot]) else 0
            assert token in oracle.allowed(states[slot]), (slot, step)
            states[slot] = oracle.advance(states[slot], token)
            tokens.append(token)
        engine.advance(np.array(tokens, dtype=np.int32))
    assert engine.errors() == 0
    # Reused slots inherit constructor defaults, never the previous blind override.
    engine.begin(slots)
    for token in served.encode("⟦ Start"):
        engine.advance(np.array([token, token], dtype=np.int32))
    engine.fill_mask()
    assert np.array_equal(*engine.mask_host())
    with pytest.raises(ValueError, match="capabilit"):
        engine.set_path_options(0, {"menu": 3, "observe": 1})
    plain = Engine(index, served, syntax, 1, **base)
    with pytest.raises(ValueError, match="request_path_options"):
        plain.set_path_options(0, {"menu": 0, "observe": 0})


@pytest.mark.parametrize("view,text", [
    ({"menu": 2}, "⟦ Start ⟦? via⟧ · via ⟦? next⟧⟧.\nAnswer: ⟦ Gamma⟧"),
    ({"observe": 1, "observe_required": True, "observe_answer": True},
     "⟦ Start ·→ Start · via ·→ Gamma (+2 more)"
     " ⟦= Delta ·→ Delta⟧.\nAnswer: ⟦ Delta⟧"),
])  # fmt: skip
def test_request_path_options_menu_only_and_observed_only_use_their_own_markers(view, text):
    from tests.test_oracle import observed_world

    torch = pytest.importorskip("torch")
    kg, served, syntax = observed_world()
    index = build_kg_index(kg, served, syntax)
    base = dict(paths=True, end_after_answer=True, eos_token_ids=[0])
    engine = Engine(
        index,
        served,
        syntax,
        1,
        **base,
        menu=2,
        observe=1,
        observe_required=True,
        observe_answer=True,
        request_path_options=True,
    )
    engine.begin(torch.tensor([0], dtype=torch.int32, device="cuda"))
    engine.set_path_options(0, view)
    reference = Engine(index, served, syntax, 1, **base, **view)
    oracle = Oracle(index, served, syntax, **base, **view)
    state = oracle.initial_state()
    for token in [*served.encode(text), 0]:
        engine.fill_mask()
        reference.fill_mask()
        assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(state)
        assert np.array_equal(engine.mask_host(), reference.mask_host())
        actual = engine.states_host()
        assert all(np.array_equal(actual[k], v) for k, v in reference.states_host().items())
        state = oracle.advance(state, token)
        engine.advance(np.array([token], dtype=np.int32))
        reference.advance(np.array([token], dtype=np.int32))
    assert engine.errors() == reference.errors() == 0


@pytest.mark.parametrize("width,display", [(2, "Gamma (+2 more)"), (8, "Gamma ; Beta ; Delta")])
def test_adaptive_observation_width_changes_only_display(width, display):
    from tests.test_oracle import observed_world

    kg, served, syntax = observed_world()
    index = build_kg_index(kg, served, syntax)
    options = dict(paths=True, escapes=False, observe=1, observe_width=width)
    oracle = Oracle(index, served, syntax, **options)
    engine = Engine(index, served, syntax, 1, **options)
    _agree_on_text(oracle, engine, index, served, f"⟦ Start ·→ Start · via ·→ {display}⟧")


@pytest.mark.parametrize("bpe", [False, True])
def test_compact_menu_engine_defer_next_close_and_forced_delivery(bpe):
    from tests.test_oracle import observed_world

    kg, served, syntax = observed_world()
    if bpe:
        served = ServedTokenizer(
            train_byte_level_bpe(
                [
                    "CLOSE",
                    "NEXT",
                    "DEFER",
                    *[name for item in (*kg.entities, *kg.relations) for name in item.labels],
                ],
                vocab_size=384,
            )
        )
    index = build_kg_index(kg, served, syntax)
    options = dict(
        paths=True,
        committed_paths=True,
        escapes=False,
        observe=1,
        observe_width=8,
        observe_required=True,
        observe_defer=2,
        menu=2,
        menu_close_next=True,
        observe_no_loop=True,
        eos_token_ids=sorted(served.special_ids),
    )
    oracle = Oracle(index, served, syntax, **options)
    engine = Engine(index, served, syntax, 1, **options)
    text = (
        "⟦ Start ·→ Start ⟦? via⟧ · via ·→ Gamma ; Beta ; Delta"
        " ⟦=DEFER⟧ ⟦!NEXT⟧ ⟦? next⟧ · next ·→ Beta ; Delta"
        " ⟦= Delta ·→ Delta ⟦!CLOSE⟧⟧.\nAnswer: ⟦ Delta⟧"
    )
    _agree_on_text(oracle, engine, index, served, text)


@pytest.mark.parametrize("prefix", [" ", ""])
def test_defer_engine_limit_alias_collision_and_path_reset(prefix):
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax(label_prefix=prefix)
    kg = KnowledgeGraph(
        (Entity("Q1", ("Start",)), Entity("Q2", ("DEFER",))),
        (Relation("P1", ("loop",)),),
        (("Q1", "P1", "Q1"), ("Q1", "P1", "Q2"), ("Q2", "P1", "Q1")),
    )
    index = build_kg_index(kg, served, syntax)
    options = dict(
        paths=True,
        escapes=False,
        observe=2,
        observe_required=True,
        observe_defer=2,
        eos_token_ids=sorted(served.special_ids),
    )
    oracle, engine = (
        Oracle(index, served, syntax, **options),
        Engine(index, served, syntax, 1, **options),
    )
    text = (
        f"⟦{prefix}Start ·→{prefix}Start"
        + f" ·{prefix}loop ·→{prefix}Start ;{prefix}DEFER ⟦=DEFER⟧" * 2
    )
    for _ in range(2):
        state = _agree_on_text(
            oracle,
            engine,
            index,
            served,
            text + f" ·{prefix}loop ·→{prefix}Start ;{prefix}DEFER ⟦=",
        )
        assert engine.states_host()["defer_count"][0] == 2
        if prefix:
            assert served.encode("D")[0] not in oracle.allowed(state)
        _agree_on_text(
            oracle,
            engine,
            index,
            served,
            f"{prefix}DEFER ·→{prefix}DEFER⟧.\nAnswer: ⟦{prefix}DEFER⟧",
            state,
        )
        engine.reset()
        assert engine.states_host()["defer_count"][0] == 0


@pytest.mark.parametrize("stop", [False, True])
def test_compact_selection_precedes_forced_close_at_stop_or_hop_cap(stop):
    from tests.test_oracle import observed_world

    if stop:
        kg, served, syntax = observed_world()
        prefix = "⟦ Start ·→ Start ⟦? via⟧ · via ·→ Gamma (+2 more)"
        question = "Which via does Start reach?"
    else:
        served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
        kg = KnowledgeGraph(
            tuple(Entity(f"Q{i}", (f"N{i}",)) for i in range(10)),
            (Relation("P1", ("via",)),),
            (
                *((f"Q{i}", "P1", f"Q{i + 1}") for i in range(8)),
                ("Q7", "P1", "Q9"),
                ("Q8", "P1", "Q9"),
            ),
        )
        prefix = "⟦ N0 ·→ N0 ⟦? via⟧" + "".join(
            f" · via ·→ N{i} ⟦!NEXT⟧ ⟦? via⟧" for i in range(1, 8)
        )
        prefix += " · via ·→ N8 (+1 more)"
        question = "Which via?"
    index = build_kg_index(kg, served, syntax)
    options = dict(
        paths=True,
        escapes=False,
        committed_paths=True,
        observe=1,
        menu=1,
        observe_required=True,
        observe_defer=2,
        menu_close_next=True,
        observe_no_loop=True,
        eos_token_ids=sorted(served.special_ids),
        stop="residual-goal" if stop else None,
    )
    oracle, engine = (
        Oracle(index, served, syntax, **options),
        Engine(index, served, syntax, 1, **options),
    )
    if stop:
        engine.set_stop(0, question)
    state = _agree_on_text(
        oracle, engine, index, served, prefix, oracle.initial_state(question=question)
    )
    state = _agree_on_text(oracle, engine, index, served, " ⟦=DEFER⟧ ⟦!", state)
    assert oracle.allowed(state) == {served.encode("C")[0]}
    _agree_on_text(oracle, engine, index, served, "CLOSE⟧⟧", state)


@pytest.mark.parametrize("compact,defer", [(True, 0), (False, 2), (True, 2)])
@pytest.mark.parametrize("committed", [False, True])
def test_menu_v2_seeded_walks_versions_caps_and_slot_decisions(compact, defer, committed):
    from tests.test_oracle import audit_world

    kg, _, served, syntax = audit_world()
    seen = run_agreement(
        served,
        kg,
        syntax,
        sequences=4,
        steps=500,
        seed=731,
        paths=True,
        escapes=False,
        committed_paths=committed,
        versions={"edit": [("Q2", "P2", "Q5")]},
        sitelinks={"Q3": 100, "Q2": 10},
        start_rank="sitelinks",
        menu=2,
        observe=1,
        observe_width=8,
        observe_select=True,
        observe_required=True,
        observe_no_loop=True,
        menu_close_next=compact,
        observe_defer=defer,
        questions=("Which next?", "Which alternate?", "Which via?"),
    )
    assert seen["answers"] > 0 and seen["versioned steps"] > 0
    if compact:
        assert seen["decision steps"] > 0


@pytest.mark.parametrize("audit", [False, True])
@pytest.mark.parametrize("bpe", [False, True])
def test_relation_display_engine_uses_canonical_and_question_scored_fallback(audit, bpe):
    from tests.test_oracle import relation_display_world

    _, index, served, syntax = relation_display_world(bpe)
    options = {"audit": True, "eos_token_ids": sorted(served.special_ids)} if audit else {"menu": 3}
    oracle = Oracle(index, served, syntax, paths=True, escapes=False, **options)
    engine = Engine(index, served, syntax, 1, paths=True, escapes=False, **options)
    engine.set_relation_question(0, "Which capital?")
    prefix = "⟦ Start · country⟧" if audit else "⟦ Start"
    display = (
        "\nExecuted: ⟦ Start⟧"
        "\nRelations (terminal): ⟦ capital ; b capital ; country⟧"
        "\nRelations (penultimate): ⟦ capital ; b capital ; country⟧\nDecision: ⟦"
        if audit
        else " ⟦? capital ; b capital ; country⟧"
    )
    _agree_on_text(
        oracle,
        engine,
        index,
        served,
        prefix + display,
        state=oracle.initial_state(question="Which capital?"),
    )


@pytest.mark.parametrize("observed", [False, True])
@pytest.mark.parametrize("bpe", [False, True])
def test_menu_engine_forces_ranked_suggestions_and_keeps_hidden_relations(observed, bpe):
    from dataclasses import replace

    from tests.test_oracle import audit_world

    kg, index, served, syntax = audit_world()
    if bpe:
        corpus = [" ".join(item.labels) for item in (*kg.entities, *kg.relations)]
        corpus += [" ⟦? alternate ; via (+1 more)⟧", " ·→ Gamma (+1 more) ⟦= Beta"]
        served = ServedTokenizer(train_byte_level_bpe(corpus, vocab_size=384))
        index = build_kg_index(kg, served, syntax, sitelinks={"Q3": 100, "Q2": 10})
    options = dict(paths=True, committed_paths=True, escapes=False, menu=1, start_rank="sitelinks")
    if observed:
        options.update(
            observe=1,
            observe_required=True,
            observe_no_loop=True,
            observe_answer=True,
            eos_token_ids=sorted(served.special_ids),
        )
    oracle = Oracle(index, served, syntax, **options)
    engine = Engine(index, served, syntax, 1, **options)
    scores = (("P3", 0), ("P2", 0), ("P1", 1))
    engine.set_relation_scores(0, scores)
    state = replace(oracle.initial_state(), audit_scores=scores)
    pieces = ["⟦", " Start"]
    if observed:
        pieces += [" ·→", " Start"]
    pieces += [" ⟦?", " alternate", " (+1 more)", "⟧", " ·", " via"]
    if observed:
        pieces += [" ·→", " Gamma", " (+1 more)"]
    pieces += [" ⟦?", " next", " (+1 more)", "⟧"]
    if observed:
        pieces += [" ⟦=", " Beta", " ·→", " Beta", " ⟦?", " next", "⟧"]
    pieces += [" ·", " next"]
    if observed:
        pieces += [" ·→", " End"]
    pieces += [
        " ⟦?",
        " via",
        "⟧",
        "⟧",
        ".\nAnswer: " if observed else " Answer: ",
        "⟦",
        " End",
        "⟧",
    ]
    for piece in pieces:
        for token in served.encode(piece):
            engine.fill_mask()
            assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(state)
            assert token in oracle.allowed(state), piece
            state = oracle.advance(state, token)
            engine.advance(np.array([token], dtype=np.int32))
            assert engine.states_host()["sep_pos"][0] == len(state.separating or b"")
    if observed:
        for token in sorted(served.special_ids):
            engine.fill_mask()
            assert (
                allowed_from_mask(engine.mask_host()[0])
                == oracle.allowed(state)
                == served.special_ids
            )
            state = oracle.advance(state, token)
            engine.advance(np.array([token], dtype=np.int32))
    assert closed_span(index, engine.states_host(), 0).entity_ids == ("Q4",)
    assert state.last_span.entity_ids == ("Q4",)


@pytest.mark.parametrize("observed", [0, 1, 3])
@pytest.mark.parametrize("committed", [False, True])
def test_menu_seeded_walks_versions_caps_and_per_slot_question_ranks(observed, committed):
    from tests.test_oracle import audit_world

    kg, _, served, syntax = audit_world()
    seen = run_agreement(
        served,
        kg,
        syntax,
        sequences=3,
        steps=650,
        seed=9821,
        paths=True,
        escapes=False,
        committed_paths=committed,
        versions={"edit": [("Q2", "P2", "Q5")]},
        sitelinks={"Q3": 100, "Q2": 10},
        start_rank="sitelinks" if committed else "degree",
        menu=3,
        observe=observed,
        observe_select=bool(observed),
        observe_required=observed == 3,
        observe_no_loop=observed == 3,
        questions=("Which next?", "Which alternate?", "Which via?"),
    )
    assert seen["menu steps"] > 0 and seen["versioned steps"] > 0
    assert seen["resolved"] > 0


def test_menu_zero_keeps_native_tables_masks_and_public_states_identical():
    from tests.test_oracle import audit_world
    from xgraph.engine import native_tables

    _, index, served, syntax = audit_world()
    baseline = native_tables(index, served, syntax)
    explicit = native_tables(index, served, syntax, menu=0)
    assert baseline.keys() == explicit.keys()
    assert all(np.array_equal(baseline[k], explicit[k]) for k in baseline)
    a = Engine(index, served, syntax, 1, paths=True)
    b = Engine(index, served, syntax, 1, paths=True, menu=0)
    for engine in (a, b):
        with pytest.raises(ValueError, match=r"^audit is not enabled$"):
            engine.set_audit(0, "Which next?")
        with pytest.raises(ValueError, match=r"^audit is not enabled$"):
            engine.set_relation_scores(0, ())
    for token in served.encode("⟦ Start · via · next⟧ Answer: ⟦ End⟧"):
        a.fill_mask()
        b.fill_mask()
        assert np.array_equal(a.mask_host(), b.mask_host())
        assert a.states_host().keys() == b.states_host().keys()
        assert all(np.array_equal(v, b.states_host()[k]) for k, v in a.states_host().items())
        a.advance(np.array([token], dtype=np.int32))
        b.advance(np.array([token], dtype=np.int32))


@pytest.mark.parametrize("k", [3, 5, 8, 16])
def test_menu_truncation_counts_distinct_ids_and_allows_an_omitted_relation(k):
    from dataclasses import replace

    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    kg = KnowledgeGraph(
        (Entity("Q1", ("Start",)), Entity("Q2", ("End",))),
        tuple(Relation(f"P{r:03d}", (f"r{r:02d}",)) for r in range(20)),
        tuple(("Q1", f"P{r:03d}", "Q2") for r in range(20)),
    )
    index = build_kg_index(kg, served, syntax)
    options = dict(paths=True, menu=k, escapes=False)
    oracle, engine = (
        Oracle(index, served, syntax, **options),
        Engine(index, served, syntax, 1, **options),
    )
    scores = (("P019", 0),)
    engine.set_relation_scores(0, scores)
    state = replace(oracle.initial_state(), audit_scores=scores)
    labels = ["r19", *(f"r{r:02d}" for r in range(k - 1))]
    menu = " ⟦? " + " ; ".join(labels) + f" (+{20 - k} more)⟧"
    state = _agree_on_text(
        oracle, engine, index, served, "⟦ Start" + menu + " · r18 ⟦? ∅⟧⟧", state=state
    )
    assert state.last_span.entity_ids == ("Q2",)


def test_menu_loop_guard_displays_best_scored_alias_when_canonical_is_blocked():
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    kg = KnowledgeGraph(
        (Entity("Q1", ("A",)),),
        (
            Relation("P1", ("a shared", "p1")),
            Relation("P2", ("a shared", "a unrelated", "z capital", "b capital")),
        ),
        (("Q1", "P1", "Q1"), ("Q1", "P2", "Q1")),
    )
    index = build_kg_index(kg, served, syntax)
    options = dict(paths=True, escapes=False, observe=1, observe_no_loop=True, menu=2)
    oracle, engine = (
        Oracle(index, served, syntax, **options),
        Engine(index, served, syntax, 1, **options),
    )
    engine.set_relation_question(0, "Which capital?")
    state = _agree_on_text(
        oracle,
        engine,
        index,
        served,
        "⟦ A ·→ A ⟦? a shared ; a shared⟧ · p1 ·→ A ⟦? b capital⟧ · b capital ·→ A ⟦? ∅⟧⟧",
        state=oracle.initial_state(question="Which capital?"),
    )
    assert state.last_span.entity_ids == ("Q1",)


@pytest.mark.parametrize(
    "plan,decision,changed,answer",
    [
        ("via", "KEEP", "", "Beta"),
        ("via", "SELECT ⟦= Gamma", "", "Gamma"),
        ("via", "APPEND · next", "End", "End"),
        ("via", "REPLACE · alternate", "Other", "Other"),
        ("via · next", "DROP", "Gamma ; Beta", "Beta"),
    ],
)
@pytest.mark.parametrize("bpe", [False, True])
def test_audit_engine_decisions_agree_with_oracle(plan, decision, changed, answer, bpe):
    from tests.test_oracle import audit_world

    kg, index, served, syntax = audit_world()
    if bpe:
        corpus = [" ".join(item.labels) for item in (*kg.entities, *kg.relations)]
        corpus += [
            "\nExecuted: ⟦ Gamma ; Beta (+1 more)⟧\nDecision: ⟦KEEP⟧",
            "APPEND · next⟧ REPLACE · alternate⟧ SELECT ⟦= Gamma⟧ DROP⟧.\nAnswer: ⟦",
        ]
        served = ServedTokenizer(train_byte_level_bpe(corpus, vocab_size=384))
        index = build_kg_index(kg, served, syntax, sitelinks={"Q3": 100, "Q2": 10})
    options = dict(
        paths=True,
        committed_paths=True,
        escapes=False,
        audit=True,
        audit_relations=0,
        eos_token_ids=sorted(served.special_ids),
    )
    oracle = Oracle(index, served, syntax, **options)
    engine = Engine(index, served, syntax, 1, **options)
    initial = "End" if plan.endswith("next") else "Gamma ; Beta"
    text = (
        f"⟦ Start · {plan}⟧\nExecuted: ⟦ {initial}⟧\nDecision: ⟦{decision}⟧"
        + (f"\nExecuted: ⟦ {changed}⟧" if changed else "")
        + f".\nAnswer: ⟦ {answer}⟧"
    )
    # Compile independent punctuation and label pieces: whole-text BPE may have
    # different boundaries, which must not be used for canonical forced blocks.
    pieces = ["⟦", " Start"]
    for relation in plan.split(" · "):
        pieces += [" ·", " " + relation]
    pieces += ["⟧", "\nExecuted: ⟦"]
    for n, label in enumerate(initial.split(" ; ")):
        pieces += ([" ;"] if n else []) + [" " + label]
    pieces += ["⟧", "\nDecision: ⟦"]
    action = decision.split()[0]
    pieces += [action]
    if action == "SELECT":
        pieces += [" ⟦=", decision.split("⟦=")[1]]
    elif action in ("APPEND", "REPLACE"):
        pieces += [" ·", decision.split(" ·")[1]]
    pieces += ["⟧"]
    if changed:
        pieces += ["\nExecuted: ⟦"]
        for n, label in enumerate(changed.split(" ; ")):
            pieces += ([" ;"] if n else []) + [" " + label]
        pieces += ["⟧"]
    pieces += [".\nAnswer: ", "⟦", " " + answer, "⟧"]
    tokens = [t for piece in pieces for t in served.encode(piece)]
    assert b"".join(served.token_bytes(t) for t in tokens).decode() == text
    state = oracle.initial_state()
    for token in tokens:
        engine.fill_mask()
        assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(state)
        state = oracle.advance(state, token)
        engine.advance(np.array([token], dtype=np.int32))
    engine.fill_mask()
    assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(state) == served.special_ids
    assert closed_span(index, engine.states_host(), 0).entity_ids == state.last_span.entity_ids


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("menus", [0, 2])
def test_audit_seeded_walks_cover_decisions_versions_slot_scores_and_resets(committed, menus):
    from tests.test_oracle import audit_world

    torch = pytest.importorskip("torch")
    _, index, served, syntax = audit_world()
    versions = {"edit": [("Q2", "P2", "Q5")]}
    options = dict(
        paths=True,
        escapes=False,
        committed_paths=committed,
        audit=True,
        audit_relations=menus,
        audit_entities=1,
        eos_token_ids=sorted(served.special_ids),
    )
    oracle = Oracle(index, served, syntax, versions=versions, **options)
    engine = Engine(
        index, served, syntax, 4, versions=build_kg_versions(index, versions), **options
    )
    rng, seen = random.Random(825), set()
    states, queues, histories = [None] * 4, [[] for _ in range(4)], [b""] * 4
    episodes = [0] * 4
    for _ in range(1300):
        for slot in range(4):
            if states[slot] is None or states[slot].answer_complete:
                if histories[slot]:
                    decision = histories[slot].split(b"\nDecision: ")[1]
                    seen.add(decision.decode().split("⟦", 1)[1].split()[0].split("⟧")[0])
                engine.begin(torch.tensor([slot], dtype=torch.int32, device="cuda"))
                version = "edit" if slot % 2 else None
                question = "Which next?" if (slot + episodes[slot]) % 2 else "Which alternate?"
                engine.set_version(slot, version)
                engine.set_audit(slot, question)
                states[slot] = oracle.initial_state(version, question=question)
                queues[slot] = served.encode(
                    "⟦ Start · via" + (" · next" if slot > 1 else "") + "⟧"
                )
                histories[slot] = b""
                episodes[slot] += 1
        engine.fill_mask()
        masks, native = engine.mask_host(), engine.states_host()
        chosen = []
        for slot, state in enumerate(states):
            allowed = oracle.allowed(state)
            assert allowed and allowed_from_mask(masks[slot]) == allowed, (slot, state)
            assert native["phase"][slot] == CODES[state.phase]
            assert native["audit_phase"][slot] == state.audit_phase
            assert native["answer_pos"][slot] == state.answer_pos
            assert native["answer_complete"][slot] == state.answer_complete
            if state.audit_phase in (3, 4) and state.typed is not None:
                rows = index.labels if state.audit_phase == 3 else index.relations
                assert (native["lo"][slot], native["hi"][slot]) == rows.prefix_range(state.typed)
                assert native["depth"][slot] == len(state.typed)
            token = queues[slot].pop(0) if queues[slot] else rng.choice(sorted(allowed))
            chosen.append(token)
            histories[slot] += served.token_bytes(token) or b""
            states[slot] = oracle.advance(state, token)
        engine.advance(np.array(chosen, dtype=np.int32))
    assert seen == {"KEEP", "SELECT", "APPEND", "REPLACE", "DROP"}


def test_audit_caps_do_not_promote_omitted_entities_or_limit_selection_to_display():
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    kg = KnowledgeGraph(
        (Entity("Q0", ("Start",)), *(Entity(f"Q{i + 1}", (f"N{i:02d}",)) for i in range(70))),
        (Relation("P1", ("via",)),),
        tuple(("Q0", "P1", f"Q{i + 1}") for i in range(70)),
    )
    index = build_kg_index(kg, served, syntax, sitelinks={"Q70": 10000, "Q64": 100})
    options = dict(
        paths=True,
        committed_paths=True,
        escapes=False,
        audit=True,
        audit_entities=1,
        audit_relations=0,
        eos_token_ids=sorted(served.special_ids),
    )
    oracle, engine = (
        Oracle(index, served, syntax, **options),
        Engine(index, served, syntax, 1, **options),
    )
    text = "⟦ Start · via⟧\nExecuted: ⟦ N63 (+63 more)⟧\nDecision: ⟦SELECT ⟦= N00⟧.\nAnswer: ⟦ N00⟧"
    state = oracle.initial_state()
    for token in served.encode(text):
        engine.fill_mask()
        assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(state)
        state = oracle.advance(state, token)
        engine.advance(np.array([token], dtype=np.int32))
    assert state.last_span.entity_ids == ("Q1",)


def test_audit_stop_closes_the_blind_plan_before_a_single_explicit_edit():
    from tests.test_oracle import audit_world

    _, index, served, syntax = audit_world()
    options = dict(
        paths=True,
        committed_paths=True,
        escapes=False,
        stop="residual-goal",
        audit=True,
        audit_relations=0,
        eos_token_ids=sorted(served.special_ids),
    )
    oracle, engine = (
        Oracle(index, served, syntax, **options),
        Engine(index, served, syntax, 1, **options),
    )
    question = "Which via does Start reach?"
    engine.set_stop(0, question)
    state = oracle.initial_state(question=question)
    text = (
        "⟦ Start · via⟧\nExecuted: ⟦ Gamma ; Beta⟧\nDecision: ⟦APPEND · next⟧"
        "\nExecuted: ⟦ End⟧.\nAnswer: ⟦ End⟧"
    )
    before_closer = len(served.encode("⟦ Start · via"))
    for step, token in enumerate(served.encode(text)):
        engine.fill_mask()
        assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(state)
        if step == before_closer:
            assert oracle.allowed(state) == {token}
        state = oracle.advance(state, token)
        engine.advance(np.array([token], dtype=np.int32))
    assert state.last_span.entity_ids == ("Q4",)


def test_audit_full_size_menus_count_unlisted_relations_and_reset_scores():
    from dataclasses import replace

    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    kg = KnowledgeGraph(
        (
            Entity("Q0", ("Start",)),
            Entity("Q100", ("End",)),
            *(Entity(f"Q{i + 1}", (f"N{i:02d}",)) for i in range(64)),
        ),
        (Relation("P1", ("via",)), *(Relation(f"P{i + 2}", (f"R{i:02d}",)) for i in range(70))),
        (
            *(("Q0", "P1", f"Q{i + 1}") for i in range(64)),
            *((f"Q{i}", f"P{r + 2}", "Q100") for i in range(65) for r in range(70)),
        ),
    )
    index = build_kg_index(kg, served, syntax, sitelinks={})
    options = dict(
        paths=True,
        escapes=False,
        audit=True,
        audit_entities=64,
        audit_relations=64,
        eos_token_ids=sorted(served.special_ids),
    )
    oracle, engine = (
        Oracle(index, served, syntax, **options),
        Engine(index, served, syntax, 1, **options),
    )
    for scores in ((("P71", 0), ("P70", 0)), ()):
        engine.reset()
        if scores:
            engine.set_relation_scores(0, scores)
        state = replace(oracle.initial_state(), audit_scores=scores)
        for token in served.encode("⟦ Start · via⟧"):
            state = oracle.advance(state, token)
            engine.advance(np.array([token], dtype=np.int32))
        display = b""
        while state.audit_phase == 1:
            engine.fill_mask()
            assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(state)
            (token,) = oracle.allowed(state)
            display += served.token_bytes(token)
            state = oracle.advance(state, token)
            engine.advance(np.array([token], dtype=np.int32))
        terminal = b"R68 ; R69 ; R00" if scores else b"R00 ; R01 ; R02"
        assert b"Relations (terminal): " + "⟦ ".encode() + terminal in display
        assert b"(+6 more)" in display and b"(+7 more)" in display


@pytest.mark.parametrize("required", [False, True])
def test_observed_v3_cycle_closes_without_a_dead_end(required):
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    kg = KnowledgeGraph(
        (Entity("Q1", ("A",)), Entity("Q2", ("B",))),
        (Relation("P1", ("via", "alias")),),
        (("Q1", "P1", "Q2"), ("Q2", "P1", "Q1")),
    )
    index = build_kg_index(kg, served, syntax)
    options = dict(
        paths=True,
        escapes=False,
        observe=1,
        observe_select=True,
        observe_required=required,
        observe_no_loop=True,
    )
    oracle = Oracle(index, served, syntax, **options)
    engine = Engine(index, served, syntax, 1, **options)
    state = _agree_on_text(
        oracle, engine, index, served, "⟦ A ·→ A · via ·→ B · alias ·→ A ⟦= A ·→ A"
    )
    # Both names of P1 have been consumed from {A}. Even a selection of A
    # cannot bypass the guard; the closer must survive and no separator may start.
    for token in served.encode("⟧"):
        assert oracle.allowed(state) == {token}
        engine.fill_mask()
        assert allowed_from_mask(engine.mask_host()[0]) == {token}
        state = oracle.advance(state, token)
        engine.advance(np.array([token], dtype=np.int32))
    assert state.last_span.entity_ids == ("Q1",)
    from xgraph.programs import MatchedExecutor, record_path

    path = record_path({"text": "⟦ A ·→ A · via ·→ B · alias ·→ A ⟦= A ·→ A⟧"})
    plain = MatchedExecutor(index, served).execute(*path)
    guarded = MatchedExecutor(
        index, served, observe_no_loop=True, observe_required=required
    ).execute(*path)
    assert plain == guarded
    assert set(guarded.reached) == set(state.last_span.entity_ids)


def test_observed_v3_required_opener_is_forced_before_stop_and_closer():
    from tests.test_oracle import observed_world

    kg, served, syntax = observed_world()
    index = build_kg_index(kg, served, syntax)
    options = dict(
        paths=True,
        committed_paths=True,
        observe=1,
        escapes=False,
        observe_required=True,
        observe_no_loop=True,
        stop="residual-goal",
    )
    oracle = Oracle(index, served, syntax, **options)
    engine = Engine(index, served, syntax, 1, **options)
    question = "Which via does Start reach?"
    engine.set_stop(0, question)
    state = _agree_on_text(
        oracle,
        engine,
        index,
        served,
        "⟦ Start ·→ Start · via ·→ Gamma (+2 more)",
        oracle.initial_state(question=question),
    )
    assert state.stop.head
    for token in served.encode(" ⟦="):
        assert oracle.allowed(state) == {token}
        engine.fill_mask()
        assert allowed_from_mask(engine.mask_host()[0]) == {token}
        state = oracle.advance(state, token)
        engine.advance(np.array([token], dtype=np.int32))
    state = _agree_on_text(oracle, engine, index, served, " Beta ·→ Beta⟧", state)
    assert state.last_span.entity_ids == ("Q2",)


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("committed", [False, True])
def test_observed_v3_seeded_walks_shared_labels_versions_and_slots(required, committed):
    from tests.test_oracle import observed_world

    kg, served, syntax = observed_world()
    kg = KnowledgeGraph(
        (*kg.entities, Entity("Q5", ("Beta", "Twin"))),
        (*kg.relations, Relation("P3", ("next", "back"))),
        (
            *kg.triples,
            ("Q1", "P1", "Q5"),
            ("Q5", "P2", "Q3"),
            ("Q4", "P3", "Q1"),
            ("Q2", "P3", "Q2"),
        ),
    )
    seen = run_agreement(
        served,
        kg,
        syntax,
        sequences=6,
        steps=550,
        seed=73,
        paths=True,
        committed_paths=committed,
        observe=1,
        observe_select=True,
        observe_required=required,
        observe_no_loop=True,
        opened=True,
        start_rank="sitelinks",
        sitelinks={"Q3": 100, "Q2": 10},
        versions={"edit": [("Q2", "P2", "Q1")]},
    )
    assert seen["selecting"] > 10


def test_observed_v3_frontier_keys_compare_exact_sets_after_hash_collision():
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    # These distinct six-row sets share the native prefilter hash, 3226504003.
    # Entity identity, rather than that hash, determines whether P1 may repeat.
    a, b = (18, 23, 29, 49, 51, 53), (3, 30, 51, 54, 57, 61)
    kg = KnowledgeGraph(
        tuple(
            Entity(f"Q{i}", (f"N{i:02d}", "Start") if i in a else (f"N{i:02d}",)) for i in range(64)
        ),
        (Relation("P1", ("via",)),),
        tuple(sorted({(f"Q{i}", "P1", f"Q{j}") for i in (*a, *b) for j in b})),
    )
    index = build_kg_index(kg, served, syntax)
    options = dict(paths=True, escapes=False, observe=6, observe_no_loop=True)
    oracle, engine = (
        Oracle(index, served, syntax, **options),
        Engine(index, served, syntax, 1, **options),
    )
    display = lambda ids: " ;".join(f" N{i:02d}" for i in ids)  # noqa: E731
    text = f"⟦ Start ·→{display(a)} · via ·→{display(b)} · via ·→{display(b)}⟧"
    state = _agree_on_text(oracle, engine, index, served, text)
    assert set(state.last_span.entity_ids) == {f"Q{i}" for i in b}
    _agree_on_text(oracle, engine, index, served, "\nAnswer: ⟦ N03⟧", state)
    assert closed_span(index, engine.states_host(), 0).entity_ids == ("Q3",)


def test_observed_v3_frontier_keys_ignore_start_ranks_and_iteration_order():
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    kg = KnowledgeGraph(
        (Entity("Q1", ("A", "Shared")), Entity("Q2", ("B", "Shared"))),
        (Relation("P1", ("swap",)),),
        (("Q1", "P1", "Q2"), ("Q2", "P1", "Q1")),
    )
    index = build_kg_index(kg, served, syntax)
    options = dict(paths=True, committed_paths=True, escapes=False, observe=2, observe_no_loop=True)
    oracle, engine = (
        Oracle(index, served, syntax, **options),
        Engine(index, served, syntax, 1, **options),
    )
    state = _agree_on_text(oracle, engine, index, served, "⟦ Shared ·→ A ; B · swap ·→ A ; B")
    for token in served.encode("⟧"):
        assert oracle.allowed(state) == {token}
        state = oracle.advance(state, token)
        engine.advance(np.array([token], dtype=np.int32))
    assert state.last_span.entity_ids == ("Q2",)
    _agree_on_text(oracle, engine, index, served, "\nAnswer: ⟦ B⟧", state)
    assert closed_span(index, engine.states_host(), 0).entity_ids == ("Q2",)


def test_observed_v3_table_fills_at_hop_cap_and_required_selection_still_precedes_closer():
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    kg = KnowledgeGraph(
        tuple(Entity(f"Q{i}", (f"N{i}",)) for i in range(10)),
        (Relation("P1", ("via",)),),
        (*((f"Q{i}", "P1", f"Q{i + 1}") for i in range(8)), ("Q7", "P1", "Q9")),
    )
    index = build_kg_index(kg, served, syntax)
    options = dict(
        paths=True, escapes=False, observe=2, observe_required=True, observe_no_loop=True
    )
    oracle, engine = (
        Oracle(index, served, syntax, **options),
        Engine(index, served, syntax, 1, **options),
    )
    text = "⟦ N0 ·→ N0" + "".join(f" · via ·→ N{i}" for i in range(1, 8))
    state = _agree_on_text(oracle, engine, index, served, text + " · via ·→ N8 ; N9")
    assert int(engine.states_host()["loop_count"][0]) == 8
    for token in served.encode(" ⟦="):
        assert oracle.allowed(state) == {token}
        state = oracle.advance(state, token)
        engine.advance(np.array([token], dtype=np.int32))
    state = _agree_on_text(oracle, engine, index, served, " N9 ·→ N9⟧", state)
    assert state.last_span.entity_ids == ("Q9",)


@pytest.mark.parametrize("version", [None, "edit"])
@pytest.mark.parametrize("answer", [False, True])
def test_observed_selection_of_hidden_entity_narrows_execution(version, answer):
    from tests.test_oracle import observed_world

    kg, served, syntax = observed_world()
    index = build_kg_index(kg, served, syntax)
    versions = {"edit": [("Q2", "P2", "Q3")]}
    options = dict(
        paths=True,
        committed_paths=True,
        escapes=False,
        observe=1,
        observe_select=True,
        observe_answer=answer,
        eos_token_ids=[256] if answer else (),
    )
    oracle = Oracle(index, served, syntax, versions=versions, **options)
    engine = Engine(
        index, served, syntax, 1, versions=build_kg_versions(index, versions), **options
    )
    engine.set_version(0, version)
    end = "Gamma (+2 more)"
    # Beta is not displayed in the base version, but remains selectable.
    result = "Gamma" if version else "Delta"
    state = _agree_on_text(
        oracle,
        engine,
        index,
        served,
        f"⟦ Start ·→ Start · via ·→ {end} ⟦= Beta ·→ Beta · next ·→ {result}⟧"
        + (f".\nAnswer: ⟦ {result}⟧" if answer else ""),
        state=oracle.initial_state(version),
    )
    assert state.last_span.entity_ids == (("Q3",) if version else ("Q4",))


@pytest.mark.parametrize("committed", [False, True])
def test_observed_selection_seeded_walks_include_shared_labels_and_versions(committed):
    from tests.test_oracle import observed_world

    kg, served, syntax = observed_world()
    kg = KnowledgeGraph(
        (*kg.entities, Entity("Q5", ("Beta", "Twin"))),
        kg.relations,
        (*kg.triples, ("Q1", "P1", "Q5"), ("Q5", "P2", "Q3")),
    )
    seen = run_agreement(
        served,
        kg,
        syntax,
        sequences=6,
        steps=650,
        seed=43,
        paths=True,
        committed_paths=committed,
        observe=1,
        observe_select=True,
        opened=True,
        start_rank="sitelinks",
        sitelinks={"Q3": 100, "Q2": 10},
        versions={"edit": [("Q2", "P2", "Q3")]},
    )
    assert seen["selecting"] > 10


def test_observed_selection_keeps_every_shared_label_owner_and_alias():
    from tests.test_oracle import observed_world

    kg, served, syntax = observed_world()
    kg = KnowledgeGraph(
        (
            *kg.entities,
            Entity("Q5", ("Beta", "Twin")),
            Entity("Q6", tuple(f"Alias{i:02d}" for i in range(80))),
        ),
        kg.relations,
        (*kg.triples, ("Q1", "P1", "Q5"), ("Q1", "P1", "Q6"), ("Q5", "P2", "Q3")),
    )
    index = build_kg_index(kg, served, syntax)
    options = dict(paths=True, committed_paths=True, escapes=False, observe=1, observe_select=True)
    oracle, engine = (
        Oracle(index, served, syntax, **options),
        Engine(index, served, syntax, 1, **options),
    )
    state = _agree_on_text(
        oracle,
        engine,
        index,
        served,
        "⟦ Start ·→ Start · via ·→ Gamma (+4 more) ⟦= Beta ·→ Beta (+1 more)"
        " · next ·→ Gamma (+1 more)⟧",
    )
    assert set(state.last_span.entity_ids) == {"Q3", "Q4"}


@pytest.mark.parametrize("menu", [0, 1])
def test_observed_selection_stop_uses_only_relations_after_narrowing(menu):
    from tests.test_oracle import observed_world

    kg, served, syntax = observed_world()
    index = build_kg_index(kg, served, syntax)
    options = dict(
        paths=True,
        committed_paths=True,
        escapes=False,
        observe=1,
        observe_select=True,
        observe_answer=True,
        eos_token_ids=[256],
        stop="residual-goal",
        menu=menu,
    )
    oracle, engine = (
        Oracle(index, served, syntax, **options),
        Engine(index, served, syntax, 1, **options),
    )
    question = "Which next does Start reach via?"
    engine.set_stop(0, question)
    if menu:
        engine.set_relation_question(0, question)

    def hint(label):
        return f" ⟦? {label}⟧" if menu else ""

    state = _agree_on_text(
        oracle,
        engine,
        index,
        served,
        "⟦ Start ·→ Start"
        + hint("via")
        + " · via ·→ Gamma (+2 more)"
        + hint("next")
        + " ⟦= Beta ·→ Beta"
        + hint("next")
        + " · next ·→ Delta"
        + hint("∅"),
        state=oracle.initial_state(question=question),
    )
    assert oracle.allowed(state) == {served.encode("⟧")[0]}


@pytest.mark.parametrize("observe", [1, 2, 4])
@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("rank", ["degree", "sitelinks"])
def test_observed_seeded_walks_with_ranked_frontiers_and_versions(observe, committed, rank):
    from tests.test_oracle import observed_world

    kg, served, syntax = observed_world()
    kg = KnowledgeGraph((*kg.entities, Entity("Q999", (" ",))), kg.relations, kg.triples)
    run_agreement(
        served,
        kg,
        syntax,
        sequences=6,
        steps=350,
        seed=19,
        paths=True,
        committed_paths=committed,
        observe=observe,
        opened=True,
        start_rank=rank,
        sitelinks={"Q4": 100, "Q2": 10},
        versions={"edit": [("Q1", "P1", "Q4")], "dead": [("Q1", "P1", "Q999")]},
    )


def test_observed_engine_fixed_stop_and_empty_start():
    from tests.test_oracle import observed_world

    kg, served, syntax = observed_world()
    index = build_kg_index(kg, served, syntax)
    options = dict(
        paths=True,
        committed_paths=True,
        escapes=False,
        observe=1,
        stop="residual-goal",
        end_after_answer=True,
        eos_token_ids=[256],
    )
    oracle = Oracle(index, served, syntax, **options)
    engine = Engine(index, served, syntax, 1, **options)
    question = "Which next does Start reach via?"
    engine.set_stop(0, question)
    final = _agree_on_text(
        oracle,
        engine,
        index,
        served,
        "⟦ Start ·→ Start · via ·→ Gamma (+2 more) · next ·→ Beta (+1 more)⟧\nAnswer: ⟦ Delta⟧",
        state=oracle.initial_state(question=question),
    )
    assert oracle.allowed(final) == {256}
    engine.reset()
    engine.set_stop(0, question)
    _agree_on_text(
        oracle,
        engine,
        index,
        served,
        "⟦ Delta ·→ ∅⟧",
        state=oracle.initial_state(question=question),
    )


def test_observed_engine_zero_has_identical_masks_and_public_states(toy_kg, toy_tokenizer):
    served, syntax = ServedTokenizer(toy_tokenizer), SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    a = Engine(index, served, syntax, 1, paths=True)
    b = Engine(index, served, syntax, 1, paths=True, observe=0)
    for token in served.encode("⟦ Marie Curie · employer⟧\nAnswer: ⟦ University of Paris⟧"):
        a.fill_mask()
        b.fill_mask()
        assert np.array_equal(a.mask_host(), b.mask_host())
        sa, sb = a.states_host(), b.states_host()
        assert sa.keys() == sb.keys()
        assert all(sa[key].tobytes() == sb[key].tobytes() for key in sa)
        a.advance(np.array([token], dtype=np.int32))
        b.advance(np.array([token], dtype=np.int32))


def test_observed_v2_disabled_matches_v1_byte_for_byte():
    from tests.test_oracle import observed_world

    kg, served, syntax = observed_world()
    index = build_kg_index(kg, served, syntax)
    options = dict(paths=True, observe=1, escapes=False)
    a, b = (
        Engine(index, served, syntax, 1, **options),
        Engine(
            index,
            served,
            syntax,
            1,
            **options,
            observe_select=False,
            observe_answer=False,
            observe_required=False,
            observe_no_loop=False,
            observe_width=0,
            observe_defer=0,
            menu_close_next=False,
        ),
    )
    oa, ob = (
        Oracle(index, served, syntax, **options),
        Oracle(
            index,
            served,
            syntax,
            **options,
            observe_select=False,
            observe_answer=False,
            observe_required=False,
            observe_no_loop=False,
            observe_width=0,
            observe_defer=0,
            menu_close_next=False,
        ),
    )
    sa, sb = oa.initial_state(), ob.initial_state()
    for token in served.encode("⟦ Start ·→ Start · via ·→ Gamma (+2 more)⟧\nAnswer: ⟦ Beta⟧"):
        a.fill_mask()
        b.fill_mask()
        assert a.mask_host().tobytes() == b.mask_host().tobytes()
        states_a, states_b = a.states_host(), b.states_host()
        assert states_a.keys() == states_b.keys()
        assert all(states_a[name].tobytes() == states_b[name].tobytes() for name in states_a)
        assert oa.allowed(sa) == ob.allowed(sb)
        sa, sb = oa.advance(sa, token), ob.advance(sb, token)
        assert sa == sb
        a.advance(np.array([token], dtype=np.int32))
        b.advance(np.array([token], dtype=np.int32))


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("guard", [False, True])
@pytest.mark.parametrize("menu", [0, 1])
def test_observed_engine_keeps_the_cap_and_start_sense_commitment(committed, guard, menu):
    from xgraph.programs import MatchedExecutor, parse_path

    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    index = build_kg_index(crowded_start_kg(), served, syntax)
    options = dict(
        paths=True,
        committed_paths=committed,
        escapes=False,
        observe=3,
        observe_no_loop=guard,
        menu=menu,
    )
    oracle, engine = (
        Oracle(index, served, syntax, **options),
        Engine(index, served, syntax, 1, **options),
    )
    start_menu = (" ⟦? r (+1 more)⟧" if committed else " ⟦? r⟧") if menu else ""
    text = (
        "⟦ X ·→ X ; X ; X (+61 more)"
        + start_menu
        + " · r ·→ O0 ; O1 ; O10 (+61 more)"
        + (" ⟦? t⟧" if menu else "")
        + "⟧"
    )
    state = _agree_on_text(oracle, engine, index, served, text)
    assert len(state.last_span.entity_ids) == (1 if committed else MAX_REACHED)
    executor = MatchedExecutor(index, served, committed_paths=committed, observe_no_loop=guard)
    assert set(executor.execute(*parse_path(text)).reached) == set(state.last_span.entity_ids)
    if committed:
        assert state.last_span.entity_ids == ("Q1065",)


@pytest.mark.parametrize("menu,observe", [(0, 1), (1, 0), (1, 1)])
def test_observed_engine_hop_limit_shows_the_last_observation(menu, observe):
    from xgraph.oracle import MAX_PATH_RELATIONS

    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    kg = KnowledgeGraph(
        (Entity("Q1", ("Start",)),), (Relation("P1", ("loop",)),), (("Q1", "P1", "Q1"),)
    )
    index = build_kg_index(kg, served, syntax)
    options = dict(paths=True, escapes=False, observe=observe, menu=menu)
    oracle = Oracle(index, served, syntax, **options)
    engine = Engine(index, served, syntax, 1, **options)
    suffix = (" ·→ Start" if observe else "") + (" ⟦? loop⟧" if menu else "")
    state = _agree_on_text(
        oracle,
        engine,
        index,
        served,
        "⟦ Start" + suffix + (" · loop" + suffix) * MAX_PATH_RELATIONS,
    )
    assert oracle.allowed(state) == {served.encode("⟧")[0]}


def test_observation_marker_prefix_is_not_biased_as_escape_only():
    from tests.test_oracle import observed_world

    torch = pytest.importorskip("torch")
    kg, served, syntax = observed_world()
    engine = Engine(
        build_kg_index(kg, served, syntax),
        served,
        syntax,
        1,
        paths=True,
        observe=1,
        escape_bias=-3.0,
    )
    slots = torch.tensor([0], dtype=torch.int32, device="cuda")
    engine.begin(slots)
    for token in served.encode("⟦ Start"):
        engine.advance(np.array([token], dtype=np.int32))
    logits = torch.zeros((1, served.vocab_size), device="cuda")
    engine.apply_mask(logits, slots)
    # The byte token for space starts both " ·→" and the escape " ⁇".
    # It has not committed to an escape and must keep its original logit.
    assert logits[0, served.encode(" ")[0]].item() == 0.0


def allowed_from_mask(words: np.ndarray) -> set[int]:
    bits = np.unpackbits(np.ascontiguousarray(words).view(np.uint8), bitorder="little")
    return {int(t) for t in np.flatnonzero(bits)}


def partial_match(tail: bytes, delimiter: bytes) -> int:
    """How many leading bytes of the delimiter the text ends with (short of all of them)."""
    return max(k for k in range(len(delimiter)) if tail.endswith(delimiter[:k]))


def last_span(state: OracleState) -> tuple[str, tuple, tuple] | None:
    closed = state.last_span
    if closed is None:
        return None
    if not closed.resolved:
        if closed.labels:
            return ("relation plan", (), ())
        return ("unresolved", (), ())
    if closed.operator:
        return ("comparison", closed.entity_ids, ())
    if not closed.triples and (
        len(closed.labels) > 1 or closed.path
    ):  # the engine records no frontier
        return ("path span", (), ())
    kind = "fact span" if closed.triples else "entity span"
    return (kind, closed.entity_ids, closed.triples)


def expected_state(state: OracleState, index, syntax: SpanSyntax) -> dict[str, object]:
    """The engine's state fields that the oracle's state implies."""
    fields: dict[str, object] = {
        "phase": CODES[state.phase],
        "last": last_span(state),
        "resolved": state.last_span is not None and state.last_span.resolved,
        "answer_complete": state.answer_complete,
        "answer_pos": state.answer_pos,
    }
    if state.phase is Phase.SPAN:
        fields["segment"] = SEGMENTS.index(state.segment.value)
        rows = index.relations if state.segment is Segment.RELATION else index.labels
        fields["range"] = None if state.typed is None else rows.prefix_range(state.typed)
        if state.observed and state.typed is not None:
            lo, hi = fields["range"]
            fields["range"] = (
                lo,
                next((r for r in range(lo, hi) if rows.row_tokens(r) != state.typed), hi),
            )
        fields["depth"] = None if state.typed is None else len(state.typed)
        fields["close_pos"] = len(state.closing or b"")
        fields["sep_pos"] = len(state.separating or b"")
        fields["esc_pos"] = len(state.escaping or b"")
        fields["op_pos"] = len(state.operating or b"")
        fields["op"] = state.operator
        fields["subjects"] = set(state.subjects)
        fields["relations"] = set(state.relations)
        fields["observe_pos"] = len(state.observing or ())
        fields["observe_offset"] = state.observation_offset
        fields["observe_length"] = len(state.observation)
        fields["observed"] = state.observed
        fields["select_pos"] = len(state.selection or ())
        fields["selecting"] = state.selecting
        fields["selected"] = state.selected
        fields["loop_count"] = len(state.loop_history)
        for name in (
            "defer_count",
            "defer_pos",
            "decision_phase",
            "decision_pos",
            "decision_choices",
        ):
            fields[name] = getattr(state, name)
    elif state.phase is Phase.FREE:
        fields["pos"] = partial_match(state.tail, syntax.opener.encode("utf-8"))
    else:
        fields["pos"] = partial_match(state.tail, syntax.closer.encode("utf-8"))
    return fields


def engine_state(states: dict[str, np.ndarray], seq: int, index) -> dict[str, object]:
    """The engine's state of one sequence, with its candidate rows read as KG ids."""
    field = {name: int(values[seq]) for name, values in states.items()}
    subjects = {
        index.labels.owner_id(r)
        for r in range(field["subj_lo"], field["subj_hi"])
        if index.relations_of(index.labels.owner_id(r))
    }
    relations = {
        index.relations.owner_id(r)
        for r in range(field["rel_lo"], field["rel_hi"])
        if any(index.relations.owner_id(r) in index.relations_of(s) for s in subjects)
    }
    closed = closed_span(index, states, seq)
    return {
        **field,
        "subjects": subjects,
        "relations": relations,
        "last": None if closed.kind == "none" else (closed.kind, closed.entity_ids, closed.triples),
        "resolved": closed.resolved,
    }


def assert_agrees(
    expected: dict[str, object], got: dict[str, object], frontier: bool = False
) -> None:
    """frontier: path mode, whose relation segments' subjects live in the engine's memory."""
    assert got["phase"] == expected["phase"]
    assert got["resolved"] == expected["resolved"]
    assert got["answer_complete"] == expected["answer_complete"]
    if "answer_pos" in got:
        assert got["answer_pos"] == expected["answer_pos"]
    if expected["phase"] == CODES[Phase.SPAN]:
        assert got["segment"] == expected["segment"]
        if expected["range"] is None:
            assert got["lo"] < 0  # the label branch is dead
        else:
            assert (got["lo"], got["hi"]) == expected["range"]
            assert got["depth"] == expected["depth"]
        for name in ("close_pos", "sep_pos", "esc_pos", "op_pos", "op"):
            assert got[name] == expected[name], name
        for name in (
            "observe_pos",
            "observe_offset",
            "observe_length",
            "observed",
            "select_pos",
            "selecting",
            "selected",
            "loop_count",
            "defer_count",
            "defer_pos",
            "decision_phase",
            "decision_pos",
            "decision_choices",
        ):
            if name in got:
                assert got[name] == expected[name], name
        if expected["segment"] != SEGMENTS.index("subject") and not frontier:
            assert got["subjects"] == expected["subjects"]
        if expected["segment"] == SEGMENTS.index("object"):
            assert got["relations"] == expected["relations"]
    else:
        assert got["pos"] == expected["pos"]
    assert got["last"] == expected["last"]


def run_agreement(
    served,
    kg: KnowledgeGraph,
    syntax: SpanSyntax,
    sequences: int,
    steps: int,
    seed: int,
    escapes: bool = True,
    repeats: bool = True,
    derivation: bool = False,
    paths: bool = False,
    bound_escapes: bool = True,
    committed_paths: bool = False,
    versions: dict[str, list[tuple[str, str, str]]] | None = None,
    start_rank: str = "degree",
    sitelinks: dict[str, int] | None = None,
    opened: bool = False,
    unquoted_starts: bool = False,
    expressions: bool = False,
    dates: dict | None = None,
    stop: str | None = None,
    questions: tuple[str, ...] = (),
    relations: str = "frontier",
    end_after_answer: bool = False,
    observe: int = 0,
    observe_width: int = 0,
    observe_defer: int = 0,
    menu_close_next: bool = False,
    observe_select: bool = False,
    observe_answer: bool = False,
    observe_required: bool = False,
    observe_no_loop: bool = False,
    menu: int = 0,
    observe_terminal: int = 0,
    terminal_order: str = "identity",
    terminal_seed: int = 17,
) -> Counter:
    """Random walks through the oracle's allowed sets, one per sequence, mirrored by the engine.

    Sequences open spans and close escapes often enough to visit every phase.
    Returns how many sequence-steps were spent in each phase and how many spans closed, resolved
    or not.

    With versions (KG versions by name), the sequences read the base KG and each version in
    turn, and each version's sequences are also mirrored by an engine on the KG index rebuilt
    from the edited triples (xgraph.mquake.apply_edits), which must give the same masks and
    the same states, read as KG ids. sitelinks are the KG indexes' sitelink counts. With
    opened, every sequence starts inside a span (Engine.open_span, Oracle.opened_state).
    dates are the KG indexes' date table (by entity id); with expressions, the KG indexes
    hold the verdict entities, and the walks also write operators and operand separators,
    and close comparisons, value comparisons among them.
    """
    modes = {
        "escapes": escapes,
        "repeats": repeats,
        "derivation": derivation,
        "paths": paths,
        "bound_escapes": bound_escapes,
        "committed_paths": committed_paths,
        "start_rank": start_rank,
        "unquoted_starts": unquoted_starts,
        "expressions": expressions,
        "stop": stop,
        "relations": relations,
        "end_after_answer": end_after_answer,
        "observe": observe,
        "observe_width": observe_width,
        "observe_defer": observe_defer,
        "menu_close_next": menu_close_next,
        "observe_select": observe_select,
        "observe_answer": observe_answer,
        "observe_required": observe_required,
        "observe_no_loop": observe_no_loop,
        "menu": menu,
        "observe_terminal": observe_terminal,
        "terminal_order": terminal_order,
        "terminal_seed": terminal_seed,
        "eos_token_ids": sorted(served.special_ids)
        if end_after_answer
        or observe_answer
        or observe_defer
        or menu_close_next
        or observe_terminal
        else (),
    }
    index = build_kg_index(
        kg, served, syntax, sitelinks=sitelinks, dates=dates, verdicts=expressions
    )
    oracle = Oracle(index, served, syntax, versions=versions, **modes)
    kg_versions = build_kg_versions(index, versions) if versions else None
    engine = Engine(index, served, syntax, max_sequences=sequences, versions=kg_versions, **modes)
    names = [None, *(versions or {})]
    version_of = [names[seq % len(names)] for seq in range(sequences)]
    views = {None: index}
    rebuilt = {}  # version: (rebuilt index, its engine, the sequences that read the version)
    for name, edits in (versions or {}).items():
        views[name] = kg_versions.view(index, name)
        own = [seq for seq in range(sequences) if version_of[seq] == name]
        rebuilt_index = build_kg_index(
            edited_kg(kg, edits),
            served,
            syntax,
            sitelinks=sitelinks,
            dates=dates,
            verdicts=expressions,
        )
        rebuilt_engine = Engine(rebuilt_index, served, syntax, max_sequences=len(own), **modes)
        rebuilt[name] = (rebuilt_index, rebuilt_engine, own)
    question_of = [
        questions[seq % len(questions)] if questions else None for seq in range(sequences)
    ]
    for seq, name in enumerate(version_of):
        engine.set_version(seq, name)
        if stop:
            engine.set_stop(seq, question_of[seq])
        if menu:
            engine.set_relation_question(seq, question_of[seq])
        if opened:
            engine.open_span(seq)
    for _, rebuilt_engine, own in rebuilt.values():
        for j, seq in enumerate(own):
            if stop:
                rebuilt_engine.set_stop(j, question_of[seq])
            if menu:
                rebuilt_engine.set_relation_question(j, question_of[seq])
            if opened:
                rebuilt_engine.open_span(j)
    rng = random.Random(seed)
    opener, closer = served.encode(" " + syntax.opener), served.encode(syntax.closer)
    escape, separator = served.encode(syntax.escape), served.encode(syntax.separator)
    from xgraph.observations import selection_marker

    select = served.encode(selection_marker(syntax)) if observe_select else []
    operand = served.encode(syntax.operand_separator) if expressions else []
    operators = [served.encode(syntax.label_prefix + op) for op in syntax.operators]
    regular = [t for t in range(served.vocab_size) if t not in served.special_ids]
    # Tokens that may be part of a delimiter inside a span; the rest of an allowed set
    # continues a label.
    delimiters = (syntax.closer, syntax.separator, syntax.escape)
    delimiter_bytes = [
        d.encode() for d in (*delimiters, *([syntax.operand_separator] if expressions else []))
    ]
    delimiter_like = {
        t
        for t in regular
        if any(
            d.startswith(served.token_bytes(t), k) for d in delimiter_bytes for k in range(len(d))
        )
    }
    states = [
        oracle.initial_state(version_of[seq], question=question_of[seq]) for seq in range(sequences)
    ]
    if opened:
        states = [oracle.opened_state(state) for state in states]
    queues: list[list[int]] = [[] for _ in range(sequences)]
    seen: Counter = Counter()
    for step in range(steps):
        engine.fill_mask()
        masks = engine.mask_host()
        native = engine.states_host()
        mirrored = {}  # sequence: (mask, state) of the engine on the rebuilt index
        for rebuilt_index, rebuilt_engine, own in rebuilt.values():
            rebuilt_engine.fill_mask()
            rebuilt_masks, rebuilt_native = rebuilt_engine.mask_host(), rebuilt_engine.states_host()
            for j, seq in enumerate(own):
                mirrored[seq] = (
                    rebuilt_masks[j],
                    engine_state(rebuilt_native, j, rebuilt_index),
                )
        tokens = []
        for seq in range(sequences):
            state = states[seq]
            if observe_terminal:
                assert native["audit_phase"][seq] == state.audit_phase
                seen["terminal display steps"] += state.audit_phase == 5
            if observe_no_loop:
                pairs = [(key, r) for key, relations in state.loop_history for r in relations]
                assert len(pairs) == len(set(pairs)), (step, seq, pairs)
            allowed = oracle.allowed(state)
            assert allowed_from_mask(masks[seq]) == allowed, (step, seq, state)
            got = engine_state(native, seq, views[version_of[seq]])
            assert_agrees(expected_state(state, index, syntax), got, paths)
            if seq in mirrored:
                mask, theirs = mirrored[seq]
                assert allowed_from_mask(mask) == allowed, (step, seq, state)
                for name in IDS_FIELDS:
                    assert got[name] == theirs[name], (step, seq, name)
                seen["versioned steps"] += 1
            if not queues[seq]:
                # Bias the walk toward whole spans: open often, take the separator
                # and (in the object segment) the closer when they can start, and
                # escape now and then.
                in_span = state.phase is Phase.SPAN
                if (
                    observe_select
                    and state.observed
                    and not state.selected
                    and not state.selection
                    and rng.random() < 0.8
                ):
                    queues[seq] = list(select)
                elif state.phase is Phase.FREE and rng.random() < 0.3:
                    queues[seq] = list(opener)
                elif (
                    expressions
                    and in_span
                    and state.typed == ()
                    and not state.operator
                    and rng.random() < 0.5
                ):
                    queues[seq] = list(rng.choice(operators))
                elif expressions and in_span and operand[0] in allowed and rng.random() < 0.9:
                    queues[seq] = list(operand)
                elif (
                    expressions
                    and in_span
                    and state.operator
                    and state.segment is Segment.RELATION
                    and closer[0] in allowed
                    and rng.random() < 0.5
                ):
                    queues[seq] = list(closer)
                elif (
                    in_span
                    and separator[0] in allowed
                    and rng.random() < (0.9 if expressions else 0.6)
                ):
                    queues[seq] = list(separator)
                elif (
                    in_span
                    and state.segment is Segment.OBJECT
                    and closer[0] in allowed
                    and rng.random() < 0.6
                ):
                    queues[seq] = list(closer)
                elif in_span and rng.random() < 0.05:
                    queues[seq] = list(escape)
                elif state.phase is Phase.ESCAPED and rng.random() < 0.3:
                    queues[seq] = list(closer)
            label_tokens = sorted(allowed - delimiter_like)
            if queues[seq] and queues[seq][0] in allowed:
                token = queues[seq].pop(0)
            elif state.phase is Phase.SPAN and label_tokens and rng.random() < 0.85:
                queues[seq] = []
                token = rng.choice(label_tokens)
            else:
                queues[seq] = []
                choices = sorted(allowed & set(regular)) or sorted(allowed)
                token = rng.choice(choices)
            seen[state.phase] += 1
            seen["menu steps"] += state.audit_phase == 6
            seen["decision steps"] += bool(state.decision_phase)
            seen["selecting"] += state.selecting
            if state.phase is Phase.SPAN and (*state.written, state.typed) in state.facts:
                seen["repeats reached"] += 1
            tokens.append(token)
            states[seq] = oracle.advance(state, token)
            seen["answers"] += states[seq].answer_complete and not state.answer_complete
            closed = states[seq].last_span
            if closed is not None and closed is not state.last_span:
                seen["resolved" if closed.resolved else "unresolved"] += 1
                seen["fact spans"] += bool(closed.triples)
                seen["comparisons"] += bool(closed.operator)
                seen["value comparisons"] += closed.operator == "same"
        engine.advance(np.array(tokens, dtype=np.int32))
        for _, rebuilt_engine, own in rebuilt.values():
            rebuilt_engine.advance(np.array([tokens[seq] for seq in own], dtype=np.int32))
    assert engine.errors() == 0
    assert all(rebuilt_engine.errors() == 0 for _, rebuilt_engine, _ in rebuilt.values())
    return seen


IDS_FIELDS = (
    "phase", "pos", "segment", "depth", "close_pos", "sep_pos", "esc_pos", "hops", "op_pos",
    "op_kind", "op", "operand", "answer_complete",
    "subjects", "relations", "last",
)  # fmt: skip
"""The state fields that mean the same in two KG indexes over the same labels: everything but
raw rows, with candidate subjects, relations, and the last closed span read as KG ids."""


# The defaults, and delimiters that overlap themselves ("[[[" holds "[[" twice), so
# that the opener and closer automata need their KMP fallback.
SYNTAXES = [SpanSyntax(), SpanSyntax(opener="[[", closer="]]", escape="??")]


def test_engine_matches_the_oracle_without_escapes(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    seen = run_agreement(
        served, toy_kg, SpanSyntax(), sequences=24, steps=80, seed=7, escapes=False
    )
    assert seen[Phase.ESCAPED] == 0 and seen[Phase.SPAN] > 50 and seen["fact spans"] > 0, seen


@pytest.mark.parametrize("syntax", SYNTAXES, ids=["default", "self-overlapping"])
def test_engine_matches_the_oracle_on_random_streams(toy_kg, toy_tokenizer, syntax):
    served = ServedTokenizer(toy_tokenizer)
    seen = run_agreement(served, toy_kg, syntax, sequences=24, steps=80, seed=7)
    # The walks visit every phase and close entity spans, fact spans, and escapes.
    assert min(seen[Phase.FREE], seen[Phase.SPAN], seen[Phase.ESCAPED]) > 50, seen
    assert min(seen["resolved"], seen["unresolved"], seen["fact spans"]) > 5, seen


def test_engine_matches_the_oracle_on_an_empty_label_index(toy_tokenizer):
    """With no label to write, a span can still be escaped: no dead end after the opener."""
    served = ServedTokenizer(toy_tokenizer)
    kg = KnowledgeGraph(entities=(Entity("Q1", ("A⟧B",)),))  # excluded: holds the closer
    assert build_kg_index(kg, served, SpanSyntax()).labels.num_rows == 0
    seen = run_agreement(served, kg, SpanSyntax(), sequences=8, steps=40, seed=3)
    assert seen[Phase.ESCAPED] > 0 and seen["unresolved"] > 0, seen


def test_codes_match_the_kernels():
    from xgraph import _engine

    assert (_engine.PHASES, _engine.SEGMENTS, _engine.CLOSED) == (PHASES, SEGMENTS, CLOSED)
    assert _engine.MAX_FACTS == MAX_FACTS and _engine.MAX_REACHED == MAX_REACHED


def test_a_failed_allocation_does_not_fail_later_calls(toy_kg, toy_tokenizer):
    """A CUDA error is reported by the call that caused it, not by the next one.

    The native class is called directly: Engine rejects a bad max_sequences before
    any allocation, and a real out-of-memory would have to fill the machine.
    """
    from xgraph import _engine
    from xgraph.engine import mask_words, native_tables

    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    with pytest.raises(RuntimeError, match="cudaMalloc"):
        tables = native_tables(index, served, SpanSyntax())
        _engine.Engine(tables, served.vocab_size, mask_words(served.vocab_size), -1)
    engine = Engine(index, served, SpanSyntax(), max_sequences=1)
    engine.advance(np.array(served.encode("⟦")[:1], dtype=np.int32))
    engine.fill_mask()
    assert engine.errors() == 0


def test_committed_paths_need_path_mode(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    with pytest.raises(ValueError, match="committed_paths needs path mode"):
        Engine(index, served, SpanSyntax(), max_sequences=1, committed_paths=True)


@pytest.mark.parametrize("max_sequences", [0, -1])
def test_max_sequences_must_be_positive(toy_kg, toy_tokenizer, max_sequences):
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    with pytest.raises(ValueError, match="max_sequences"):
        Engine(index, served, SpanSyntax(), max_sequences=max_sequences)


def test_engine_matches_the_oracle_where_a_token_serves_two_branches():
    tokenizer = byte_level_tokenizer(MERGES)
    served = ServedTokenizer(tokenizer)
    index = build_kg_index(KnowledgeGraph(entities=HAND_ENTITIES), served, HAND_SYNTAX)
    oracle = Oracle(index, served, HAND_SYNTAX)
    streams = [
        ["âŁ¦", "Bra", "âģ", "ĩ", "âŁ§"],
        ["âŁ¦", "Bra", "âģ", "Ħ", "âŁ§"],
        ["âŁ¦", "Bra", "âŁ", "§", "x"],
        ["âŁ¦", "Bra", "âŁ", "©", "âŁ§"],
        ["xâŁ¦", "â", "ģ", "ĩ", "âŁ§"],
        ["âŁ", "¦", "Bra", "â", "Ł", "§"],
    ]
    engine = Engine(index, served, HAND_SYNTAX, max_sequences=len(streams))
    states = [oracle.initial_state() for _ in streams]
    for step in range(len(streams[0])):
        engine.fill_mask()
        masks, native = engine.mask_host(), engine.states_host()
        for seq, stream in enumerate(streams):
            assert allowed_from_mask(masks[seq]) == oracle.allowed(states[seq]), (step, seq)
            assert_agrees(
                expected_state(states[seq], index, HAND_SYNTAX), engine_state(native, seq, index)
            )
            states[seq] = oracle.advance(states[seq], tokenizer.token_to_id(stream[step]))
        engine.advance(np.array([tokenizer.token_to_id(s[step]) for s in streams], dtype=np.int32))
    engine.fill_mask()
    native = engine.states_host()
    for seq in range(len(streams)):
        assert_agrees(
            expected_state(states[seq], index, HAND_SYNTAX), engine_state(native, seq, index)
        )


@pytest.mark.parametrize(
    "text",
    [
        "She met ⟦ Marie Curie · award received · Nobel Prize⟧ and more.",
        "Then ⟦ Pierre Curie · award · Nobel Prize in Physics⟧ too.",
        "At ⟦ University of Paris · located in · Paris⟧ now.",
        "Also ⟦ Paris · located in · Rhône–Alpes⟧, ⟦ Marie Curie · spouse · ⁇ unsure⟧ too.",
        "Both ⟦ Curie · place of birth · Paris⟧ and ⟦ Curie · employer · University of Paris⟧ .",
    ],
)
def test_engine_matches_the_oracle_on_fact_spans(toy_kg, toy_tokenizer, text):
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    oracle = Oracle(index, served, SpanSyntax())
    engine = Engine(index, served, SpanSyntax(), max_sequences=1)
    state = oracle.initial_state()
    for step, token in enumerate(served.encode(text)):
        engine.fill_mask()
        assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(state), step
        assert_agrees(
            expected_state(state, index, SpanSyntax()), engine_state(engine.states_host(), 0, index)
        )
        state = oracle.advance(state, token)
        engine.advance(np.array([token], dtype=np.int32))
    assert engine.errors() == 0


def test_a_token_outside_the_mask_is_an_error(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    engine = Engine(index, served, SpanSyntax(), max_sequences=2)
    eos = toy_tokenizer.token_to_id("<|endoftext|>")
    engine.advance(np.array(served.encode("⟦")[:1] * 2, dtype=np.int32))
    engine.advance(np.array(served.encode("⟦")[1:] * 2, dtype=np.int32))
    engine.advance(np.array([eos, served.encode(" Paris")[0]], dtype=np.int32))
    assert engine.errors() == 1
    assert int(engine.states_host()["phase"][0]) == PHASE_ERROR


def test_tokens_on_the_device_advance_like_tokens_on_the_host(toy_kg, toy_tokenizer):
    torch = pytest.importorskip("torch")
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    on_device = Engine(index, served, SpanSyntax(), max_sequences=2)
    on_host = Engine(index, served, SpanSyntax(), max_sequences=2)
    stream = served.encode("She met ⟦ Marie")
    for token in stream:
        pair = [token, stream[0]]
        on_device.advance(torch.tensor(pair, dtype=torch.int32, device="cuda"))
        on_host.advance(np.array(pair, dtype=np.int32))
    for engine in (on_device, on_host):
        engine.fill_mask()
    assert int(on_device.states_host()["phase"][0]) == CODES[Phase.SPAN]
    for field, values in on_host.states_host().items():
        assert np.array_equal(on_device.states_host()[field], values), field
    mask = torch.from_dlpack(on_device.mask)
    assert mask.is_cuda and mask.dtype == torch.int32
    assert np.array_equal(mask.cpu().numpy(), on_host.mask_host())


def test_index_built_for_another_tokenizer_is_rejected(toy_kg, toy_tokenizer):
    index = build_kg_index(toy_kg, ServedTokenizer(toy_tokenizer), SpanSyntax())
    other = ServedTokenizer(train_byte_level_bpe(["a different corpus"], vocab_size=300))
    with pytest.raises(IndexMismatch):
        Engine(index, other, SpanSyntax(), max_sequences=1)


def test_slots_follow_a_token_history_from_the_prompt_on(toy_kg, toy_tokenizer):
    """sync advances each active slot over its history from prompt_len to total_len.

    The history, slot mapping, and lengths live on the GPU, as vLLM keeps them; the
    prompt (which may hold span syntax of its own) is never advanced over.
    """
    torch = pytest.importorskip("torch")
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    oracle = Oracle(index, served, SpanSyntax())
    prompt = served.encode("Example: ⟦ Paris")  # leaves a span open inside the prompt
    replies = [served.encode(" She met ⟦ Marie Curie⟧ in"), served.encode(" ⟦ Curie · employer")]
    history = torch.zeros((4, 64), dtype=torch.int32, device="cuda")
    prompt_len = torch.zeros(4, dtype=torch.int32, device="cuda")
    total_len = torch.zeros(4, dtype=torch.int32, device="cuda")
    slots = [2, 0]  # batch position -> slot
    engine = Engine(index, served, SpanSyntax(), max_sequences=4)
    engine.begin(torch.tensor(slots, dtype=torch.int32, device="cuda"))
    for slot, reply in zip(slots, replies, strict=True):
        tokens = prompt + reply
        history[slot, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
        prompt_len[slot] = len(prompt)
    for step in range(max(len(r) for r in replies) + 1):
        for slot, reply in zip(slots, replies, strict=True):
            total_len[slot] = len(prompt) + min(step, len(reply))
        engine.sync(
            torch.tensor(slots, dtype=torch.int32, device="cuda"), history, prompt_len, total_len
        )
    states = engine.states_host()
    for slot, reply in zip(slots, replies, strict=True):
        expected = oracle.initial_state()
        for token in reply:
            expected = oracle.advance(expected, token)
        got = engine_state({k: v[[slot]] for k, v in states.items()}, 0, index)
        assert_agrees(expected_state(expected, index, SpanSyntax()), got)


def test_rows_are_masked_through_their_slots_and_logits_in_place(toy_kg, toy_tokenizer):
    torch = pytest.importorskip("torch")
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    oracle = Oracle(index, served, SpanSyntax())
    engine = Engine(index, served, SpanSyntax(), max_sequences=3)
    engine.begin(torch.tensor([0, 1], dtype=torch.int32, device="cuda"))  # slot 2 stays inactive
    opener = served.encode("⟦")
    for token in opener:  # slot 0 opens a span; slot 1 writes free text
        engine.advance(
            torch.tensor([token, served.encode("x")[0]], dtype=torch.int32, device="cuda")
        )
    rows = torch.tensor([1, 2, 0], dtype=torch.int32, device="cuda")  # row -> slot
    vocab = served.vocab_size + 37  # the model's logits are wider than the tokenizer
    logits = torch.randn((3, vocab), device="cuda")
    original = logits.clone()
    engine.apply_mask(logits, rows)
    in_span = oracle.initial_state()
    for token in opener:
        in_span = oracle.advance(in_span, token)
    free = oracle.advance(oracle.initial_state(), served.encode("x")[0])
    for row, state in ((0, free), (2, in_span)):
        allowed = torch.zeros(vocab, dtype=torch.bool)
        allowed[sorted(oracle.allowed(state))] = True
        got = logits[row].cpu()
        assert torch.equal(got[allowed], original[row].cpu()[allowed])
        assert torch.isneginf(got[~allowed]).all()
    assert torch.equal(logits[1], original[1])  # inactive slot: untouched


def test_engine_runs_on_a_saved_and_loaded_kg_index(toy_kg, toy_tokenizer, tmp_path):
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    save_kg_index(index, tmp_path)
    engines = [
        Engine(i, served, SpanSyntax(), max_sequences=1) for i in (index, load_kg_index(tmp_path))
    ]
    for token in served.encode("⟦ Marie Curie · award received · Nobel Prize⟧ ok"):
        masks = []
        for engine in engines:
            engine.fill_mask()
            masks.append(engine.mask_host())
            engine.advance(np.array([token], dtype=np.int32))
        assert np.array_equal(*masks)
    assert all(engine.errors() == 0 for engine in engines)


@pytest.mark.parametrize("wide_rows", [4096, 4], ids=["scanned", "precomputed"])
def test_masks_of_wide_ranges_match_the_oracle(wide_rows, monkeypatch):
    """Hundreds of labels share their first token, so each thread jumps over many runs.

    After " Alpha" every row holds a different word (runs of one row, two rows per
    thread); after " Beta" each word starts eight rows (itself and seven variants).
    With a low threshold, the same ranges use precomputed next-token masks instead.
    """
    import xgraph.engine

    monkeypatch.setattr(xgraph.engine, "WIDE_ROWS", wide_rows)
    words = [f"{a}{b}{c}" for a in "bcdfgh" for b in "aeiou" for c in "klmnprstvz"]
    variants = ("one", "two", "three", "four", "five", "six", "seven")
    labels = [f"Alpha {w}" for w in words] + [f"Beta {w}" for w in words]
    labels += [f"Beta {w} {v}" for w in words for v in variants]
    entities = tuple(Entity(f"Q{i}", (label,)) for i, label in enumerate(labels))
    # A vocabulary large enough that every word is one token.
    served = ServedTokenizer(train_byte_level_bpe([" " + label for label in labels] * 2, 2000))
    index = build_kg_index(KnowledgeGraph(entities), served, SpanSyntax())
    oracle = Oracle(index, served, SpanSyntax())
    engine = Engine(index, served, SpanSyntax(), max_sequences=1)
    wide = xgraph.engine.native_tables(index, served, SpanSyntax())["wide_keys"]
    assert (len(wide) > 0) == (wide_rows == 4)
    for text in ("Alpha bak", "Beta dum seven"):
        label = served.encode(" " + text)
        for depth in range(1, len(label)):
            state = oracle.initial_state()
            engine.reset()
            for token in [*served.encode("⟦"), *label[:depth]]:
                state = oracle.advance(state, token)
                engine.advance(np.array([token], dtype=np.int32))
            engine.fill_mask(1)
            got = allowed_from_mask(engine.mask_host(1)[0])
            assert got == set(oracle.allowed(state)), (text, depth)


def test_draft_rows_are_masked_with_the_state_after_their_drafts(toy_kg, toy_tokenizer):
    """Under speculative decoding a request owns one logits row per draft position.

    Row k of a request is masked with the state after the committed tokens and the
    input tokens of the request's rows 1..k (the drafts before it); the committed
    state itself does not move.
    """
    torch = pytest.importorskip("torch")
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    oracle = Oracle(index, served, SpanSyntax())
    engine = Engine(index, served, SpanSyntax(), max_sequences=3)
    slots = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    engine.begin(slots)
    committed = [served.encode(" She met ⟦ Marie"), served.encode(" At ⟦ University")]
    history = torch.zeros((3, 32), dtype=torch.int32, device="cuda")
    total = torch.zeros(3, dtype=torch.int32, device="cuda")
    for slot, tokens in zip((0, 2), committed, strict=True):
        history[slot, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
        total[slot] = len(tokens)
    engine.sync(slots, history, torch.zeros(3, dtype=torch.int32, device="cuda"), total)
    drafts = [served.encode(" Curie⟧ in"), served.encode(" of Paris")]
    rows, local, inputs = [], [], []
    for slot, tokens, draft in zip((0, 2), committed, drafts, strict=True):
        for k in range(len(draft) + 1):  # row 0 is fed the last committed token
            rows.append(slot)
            local.append(k)
            inputs.append(tokens[-1] if k == 0 else draft[k - 1])
    vocab = served.vocab_size + 5
    logits = torch.zeros((len(rows), vocab), device="cuda")
    tensor = lambda v: torch.tensor(v, dtype=torch.int32, device="cuda")  # noqa: E731
    before = engine.states_host()
    engine.apply_mask(logits, tensor(rows), local_pos=tensor(local), input_ids=tensor(inputs))
    for i, (slot, k) in enumerate(zip(rows, local, strict=True)):
        tokens = committed[(0, 2).index(slot)]
        draft = drafts[(0, 2).index(slot)]
        state = oracle.initial_state()
        for token in [*tokens, *draft[:k]]:
            state = oracle.advance(state, token)
        allowed = torch.zeros(vocab, dtype=torch.bool)
        allowed[sorted(oracle.allowed(state))] = True
        assert torch.isneginf(logits[i].cpu()[~allowed]).all(), (slot, k)
        assert (logits[i].cpu()[allowed] == 0).all(), (slot, k)
    after = engine.states_host()
    assert all((before[f] == after[f]).all() for f in before)  # drafts are tentative


def test_an_index_built_for_another_span_syntax_is_rejected(toy_kg, toy_tokenizer):
    from xgraph.syntax import preset

    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, preset("wiki"))
    with pytest.raises(IndexMismatch, match="span syntax"):
        Engine(index, served, SpanSyntax(), max_sequences=1)


def test_engine_matches_the_oracle_without_repeats(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    seen = run_agreement(
        served, toy_kg, SpanSyntax(), sequences=24, steps=160, seed=11, repeats=False
    )
    assert seen["fact spans"] > 5 and seen["repeats reached"] > 0, seen
    # A fact written twice: the second time only the escape (or a longer object) is left.
    fact = "⟦ Marie Curie · award received · Nobel Prize"
    text = f"{fact}⟧ and {fact} ⁇ again⟧ and {fact} in Physics⟧ and {fact} in Physics ⁇ so⟧ and"
    index = build_kg_index(toy_kg, served, SpanSyntax())
    oracle = Oracle(index, served, SpanSyntax(), repeats=False)
    engine = Engine(index, served, SpanSyntax(), max_sequences=1, repeats=False)
    state = oracle.initial_state()
    for step, token in enumerate(served.encode(text)):
        engine.fill_mask()
        assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(state), step
        state = oracle.advance(state, token)
        engine.advance(np.array([token], dtype=np.int32))
    assert engine.errors() == 0 and len(state.facts) == 2


@pytest.mark.parametrize(
    ("committed", "draft"),
    [
        # Drafts that close a fact and write it again.
        (" She met ⟦ Marie", " Curie · award received · Nobel Prize⟧ and ⟦ Marie Curie · award"
         " received · Nobel Prize"),
        # Drafts that write again a fact the committed tokens closed.
        (" ⟦ Marie Curie · award received · Nobel Prize⟧ and", " ⟦ Marie Curie · award received"
         " · Nobel Prize"),
    ],
    ids=["closed in drafts", "closed before"],
)  # fmt: skip
def test_draft_rows_see_the_facts_closed_without_repeats(toy_kg, toy_tokenizer, committed, draft):
    torch = pytest.importorskip("torch")
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    oracle = Oracle(index, served, SpanSyntax(), repeats=False)
    engine = Engine(index, served, SpanSyntax(), max_sequences=2, repeats=False)
    tensor = lambda v: torch.tensor(v, dtype=torch.int32, device="cuda")  # noqa: E731
    slots = tensor([1])
    engine.begin(slots)
    tokens, drafted = served.encode(committed), served.encode(draft)
    history = torch.zeros((2, 64), dtype=torch.int32, device="cuda")
    history[1, : len(tokens)] = tensor(tokens)
    total = tensor([0, len(tokens)])
    engine.sync(slots, history, tensor([0, 0]), total)
    rows = [1] * (len(drafted) + 1)
    local = list(range(len(drafted) + 1))
    inputs = [tokens[-1], *drafted]
    vocab = served.vocab_size + 5
    logits = torch.zeros((len(rows), vocab), device="cuda")
    engine.apply_mask(logits, tensor(rows), local_pos=tensor(local), input_ids=tensor(inputs))
    state = oracle.initial_state()
    for token in tokens:
        state = oracle.advance(state, token)
    for k in local:
        allowed = torch.zeros(vocab, dtype=torch.bool)
        allowed[sorted(oracle.allowed(state))] = True
        assert torch.isneginf(logits[k].cpu()[~allowed]).all(), k
        assert (logits[k].cpu()[allowed] == 0).all(), k
        if k < len(drafted):
            state = oracle.advance(state, drafted[k])
    assert served.encode("⟧")[0] not in oracle.allowed(state)  # the last row blocks the repeat


def test_without_repeats_the_first_max_facts_are_remembered():
    labels = [f"Item {i}" for i in range(MAX_FACTS + 6)]
    kg = KnowledgeGraph(
        entities=(
            Entity(id="Q1", labels=("Hub",)),
            *(Entity(id=f"Q{i + 2}", labels=(label,)) for i, label in enumerate(labels)),
        ),
        relations=(Relation(id="P1", labels=("has",)),),
        triples=tuple(("Q1", "P1", f"Q{i + 2}") for i in range(len(labels))),
    )
    served = ServedTokenizer(train_byte_level_bpe(["Hub", "has", *labels, "⟦ · ⟧ ⁇"], 400))
    index = build_kg_index(kg, served, SpanSyntax())
    oracle = Oracle(index, served, SpanSyntax(), repeats=False)
    engine = Engine(index, served, SpanSyntax(), max_sequences=1, repeats=False)
    closer = served.encode("⟧")[0]

    def write(text: str, state):
        for token in served.encode(text):
            state = oracle.advance(state, token)
            engine.advance(np.array([token], dtype=np.int32))
        return state

    state = write("".join(f" ⟦ Hub · has · {label}⟧ and" for label in labels[: MAX_FACTS + 1]),
                  oracle.initial_state())  # fmt: skip
    for label, remembered in ((labels[MAX_FACTS - 1], True), (labels[MAX_FACTS], False)):
        before_closer = write(f" ⟦ Hub · has · {label}", state)
        engine.fill_mask()
        assert (closer in oracle.allowed(before_closer)) is not remembered, label
        assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(before_closer), label
        state = write(" ⁇⟧ and", before_closer)
    assert engine.errors() == 0


def test_reset_and_begin_forget_the_facts_closed(toy_kg, toy_tokenizer):
    torch = pytest.importorskip("torch")
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    engine = Engine(index, served, SpanSyntax(), max_sequences=1, repeats=False)
    fact = served.encode(" ⟦ Marie Curie · award received · Nobel Prize")
    closer = served.encode("⟧")

    def closer_allowed_after_fact() -> bool:
        for token in fact:
            engine.advance(np.array([token], dtype=np.int32))
        engine.fill_mask()
        return closer[0] in allowed_from_mask(engine.mask_host()[0])

    for forget in (engine.reset, lambda: engine.begin(torch.zeros(1, dtype=torch.int32,
                                                                   device="cuda"))):  # fmt: skip
        assert closer_allowed_after_fact()
        for token in [*closer, *served.encode(" and")]:
            engine.advance(np.array([token], dtype=np.int32))
        assert not closer_allowed_after_fact()
        forget()
        assert closer_allowed_after_fact()
        engine.reset()


def _released_mask(oracle, state):
    """The tokens of the allowed set that keep only the escape: those after which the
    escape marker is being typed or complete."""
    released = set()
    for token in oracle.allowed(state):
        after = oracle.advance(state, token)
        if after.phase is Phase.ESCAPED or (after.phase is Phase.SPAN and after.escaping):
            released.add(token)
    return released


@pytest.mark.parametrize(
    ("release", "favourite", "released"),
    [
        ({"release_top": True}, "disallowed", True),
        ({"release_top": True}, "allowed", False),
        ({"release_mass": 0.5}, "disallowed", True),  # the allowed mass is about 0.3
        ({"release_mass": 0.2}, "disallowed", False),
        ({}, "disallowed", False),
    ],
)
def test_rows_with_high_rejected_mass_are_released_to_the_escape(
    toy_kg, toy_tokenizer, release, favourite, released
):
    torch = pytest.importorskip("torch")
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    oracle = Oracle(index, served, SpanSyntax())
    engine = Engine(index, served, SpanSyntax(), max_sequences=2, **release)
    tensor = lambda v: torch.tensor(v, dtype=torch.int32, device="cuda")  # noqa: E731
    slots = tensor([0, 1])
    engine.begin(slots)
    texts = [" She met ⟦ Marie", " She met"]  # inside a span; in free text
    history = torch.zeros((2, 32), dtype=torch.int32, device="cuda")
    states = []
    for slot, text in enumerate(texts):
        tokens = served.encode(text)
        history[slot, : len(tokens)] = tensor(tokens)
        state = oracle.initial_state()
        for token in tokens:
            state = oracle.advance(state, token)
        states.append(state)
    total = tensor([len(served.encode(t)) for t in texts])
    engine.sync(slots, history, tensor([0, 0]), total)
    allowed = oracle.allowed(states[0])
    disallowed = next(t for t in range(served.vocab_size) if t not in allowed)
    # Three tenths of the probability on the allowed set, the rest on one token.
    logits = torch.full((2, served.vocab_size), -1e4, device="cuda")
    logits[:, sorted(allowed)] = math.log(0.3 / len(allowed))
    top = disallowed if favourite == "disallowed" else min(allowed)
    logits[:, top] = (
        math.log(0.7) if favourite == "disallowed" else math.log(0.7 + 0.3 / len(allowed))
    )
    engine.apply_mask(logits, tensor([0, 1]))
    expected = _released_mask(oracle, states[0]) if released else allowed
    assert {t for t in range(served.vocab_size) if not torch.isneginf(logits[0, t])} == expected
    assert (expected and expected != allowed) or not released
    free = {t for t in range(served.vocab_size) if not torch.isneginf(logits[1, t])}
    assert free == oracle.allowed(states[1])  # free text is never released


def test_release_needs_escapes(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    with pytest.raises(ValueError, match="escape"):
        Engine(index, served, SpanSyntax(), max_sequences=1, escapes=False, release_top=True)


def test_a_started_escape_marker_is_released_to_its_continuation():
    """After "⟦Bra", the token E2 81 starts both the escape marker ⁇ and the label
    "Bra⁄": released, only the byte that completes the marker remains."""
    torch = pytest.importorskip("torch")
    tokenizer = byte_level_tokenizer(MERGES)
    served = ServedTokenizer(tokenizer)
    index = build_kg_index(KnowledgeGraph(entities=HAND_ENTITIES), served, HAND_SYNTAX)
    oracle = Oracle(index, served, HAND_SYNTAX)
    engine = Engine(index, served, HAND_SYNTAX, max_sequences=1, release_top=True)
    tokens = [tokenizer.token_to_id(piece) for piece in ("âŁ¦", "Bra", "âģ")]
    state = oracle.initial_state()
    for token in tokens:
        state = oracle.advance(state, token)
    allowed = oracle.allowed(state)
    assert state.escaping and len(allowed) == 2  # the marker or the label goes on
    slots = torch.tensor([0], dtype=torch.int32, device="cuda")
    engine.begin(slots)
    history = torch.zeros((1, 8), dtype=torch.int32, device="cuda")
    history[0, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
    lengths = torch.tensor([len(tokens)], dtype=torch.int32, device="cuda")
    engine.sync(slots, history, torch.zeros_like(lengths), lengths)
    logits = torch.zeros((1, served.vocab_size), device="cuda")
    logits[0, next(t for t in range(served.vocab_size) if t not in allowed)] = 50.0
    engine.apply_mask(logits, slots)
    got = {t for t in range(served.vocab_size) if not torch.isneginf(logits[0, t])}
    assert got == _released_mask(oracle, state) == {tokenizer.token_to_id("ĩ")}


def test_engine_matches_the_oracle_in_derivation_mode(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    seen = run_agreement(
        served, toy_kg, SpanSyntax(), sequences=24, steps=200, seed=5, derivation=True
    )
    assert seen["fact spans"] > 5, seen
    index = build_kg_index(toy_kg, served, SpanSyntax())
    oracle = Oracle(index, served, SpanSyntax(), derivation=True)
    engine = Engine(index, served, SpanSyntax(), max_sequences=1, derivation=True)
    text = (
        "So ⟦ Curie · employer · University of Paris⟧, ⟦ University of Paris · located in · "
        "Paris⟧ and ⟦ Paris⟧ in ⟦ Paris · located in · Rhône–Alpes⟧ ok"
    )
    state = oracle.initial_state()
    for step, token in enumerate(served.encode(text)):
        engine.fill_mask()
        assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(state), step
        assert_agrees(
            expected_state(state, index, SpanSyntax()), engine_state(engine.states_host(), 0, index)
        )
        state = oracle.advance(state, token)
        engine.advance(np.array([token], dtype=np.int32))
    assert engine.errors() == 0


def test_engine_matches_the_oracle_in_derivation_mode_without_repeats(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    for seed in (5, 6, 7):
        seen = run_agreement(
            served, toy_kg, SpanSyntax(), sequences=24, steps=200, seed=seed,
            derivation=True, repeats=False,
        )  # fmt: skip
        assert seen["fact spans"] > 5, seen


def test_draft_rows_see_the_rows_reached_in_derivation_mode(toy_kg, toy_tokenizer):
    torch = pytest.importorskip("torch")
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    oracle = Oracle(index, served, SpanSyntax(), derivation=True)
    engine = Engine(index, served, SpanSyntax(), max_sequences=2, derivation=True)
    tensor = lambda v: torch.tensor(v, dtype=torch.int32, device="cuda")  # noqa: E731
    for committed, draft in (
        (" ⟦ Curie · place of", " birth · Paris⟧ so ⟦ Paris"),  # reached within the drafts
        (" ⟦ Curie · place of birth · Paris⟧ so", " ⟦ Paris⟧"),  # reached before them
    ):
        slots = tensor([1])
        engine.begin(slots)
        tokens, drafted = served.encode(committed), served.encode(draft)
        history = torch.zeros((2, 64), dtype=torch.int32, device="cuda")
        history[1, : len(tokens)] = tensor(tokens)
        engine.sync(slots, history, tensor([0, 0]), tensor([0, len(tokens)]))
        local = list(range(len(drafted) + 1))
        logits = torch.zeros((len(local), served.vocab_size + 5), device="cuda")
        engine.apply_mask(
            logits, tensor([1] * len(local)), local_pos=tensor(local),
            input_ids=tensor([tokens[-1], *drafted]),
        )  # fmt: skip
        state = oracle.initial_state()
        for token in tokens:
            state = oracle.advance(state, token)
        for k in local:
            got = {t for t in range(served.vocab_size) if not torch.isneginf(logits[k, t])}
            assert got == oracle.allowed(state), (committed, k)
            if k < len(drafted):
                state = oracle.advance(state, drafted[k])


def test_without_repeats_a_fact_repeats_by_label_even_when_bound_to_another_entity():
    """Two entities are labelled X; the first span binds the better-connected one, a
    chain then reaches the other, and writing the first fact again must still count as
    a repeat: facts are identified by their labels."""
    kg = KnowledgeGraph(
        entities=(
            Entity(id="Q1", labels=("X",)), Entity(id="Q2", labels=("X",)),
            Entity(id="Q3", labels=("Y",)),
        ),
        relations=(Relation(id="P1", labels=("r one",)), Relation(id="P2", labels=("r two",)),
                   Relation(id="P4", labels=("r four",))),
        triples=(("Q1", "P1", "Q3"), ("Q1", "P2", "Q3"), ("Q2", "P1", "Q3"), ("Q3", "P4", "Q2")),
    )  # fmt: skip
    served = ServedTokenizer(train_byte_level_bpe(["X", "Y", "r one", "r two", "r four"], 300))
    index = build_kg_index(kg, served, SpanSyntax())
    oracle = Oracle(index, served, SpanSyntax(), derivation=True, repeats=False)
    engine = Engine(index, served, SpanSyntax(), max_sequences=1, derivation=True, repeats=False)
    state = oracle.initial_state()
    text = "⟦ X · r one · Y⟧ ⟦ Y · r four · X⟧ ⟦ X · r one · Y"
    for token in served.encode(text):
        state = oracle.advance(state, token)
        engine.advance(np.array([token], dtype=np.int32))
    engine.fill_mask()
    assert served.encode("⟧")[0] not in oracle.allowed(state)  # the repeat cannot close
    assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(state)


def test_engine_lets_a_reached_entity_be_named_by_any_of_its_labels(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    oracle = Oracle(index, served, SpanSyntax(), derivation=True)
    engine = Engine(index, served, SpanSyntax(), max_sequences=1, derivation=True)
    text = "⟦ Pierre Curie · spouse · Marie Curie⟧ so ⟦ Maria Skłodowska · award received · "
    state = oracle.initial_state()
    for step, token in enumerate(served.encode(text + "Nobel Prize⟧ ok")):
        engine.fill_mask()
        assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(state), step
        state = oracle.advance(state, token)
        engine.advance(np.array([token], dtype=np.int32))
    assert engine.errors() == 0


@pytest.mark.parametrize("span", [" She met ⟦ Marie", " She met ⟦ Marie Curie"])
def test_an_escape_bias_shifts_only_tokens_that_commit_to_the_escape(toy_kg, toy_tokenizer, span):
    torch = pytest.importorskip("torch")
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    oracle = Oracle(index, served, SpanSyntax())
    engine = Engine(index, served, SpanSyntax(), max_sequences=2, escape_bias=-3.0)
    tensor = lambda v: torch.tensor(v, dtype=torch.int32, device="cuda")  # noqa: E731
    slots = tensor([0, 1])
    engine.begin(slots)
    texts = [span, " She met"]  # inside a span (mid-label or complete); in free text
    history = torch.zeros((2, 32), dtype=torch.int32, device="cuda")
    states = []
    for slot, text in enumerate(texts):
        tokens = served.encode(text)
        history[slot, : len(tokens)] = tensor(tokens)
        state = oracle.initial_state()
        for token in tokens:
            state = oracle.advance(state, token)
        states.append(state)
    engine.sync(slots, history, tensor([0, 0]), tensor([len(served.encode(t)) for t in texts]))
    logits = torch.zeros((2, served.vocab_size), device="cuda")
    engine.apply_mask(logits, tensor([0, 1]))
    allowed = oracle.allowed(states[0])

    # The tokens that commit to the escape: they start the marker and end every other branch.
    def commits(after):
        return after.phase is Phase.ESCAPED or (
            after.typed is None and after.separating is None and after.closing is None
        )

    escapes = {
        t for t in _released_mask(oracle, states[0]) if commits(oracle.advance(states[0], t))
    }
    assert escapes
    for t in range(served.vocab_size):
        expected = (-3.0 if t in escapes else 0.0) if t in allowed else float("-inf")
        assert logits[0, t].item() == expected, t
    assert all(logits[1, t].item() in (0.0, float("-inf")) for t in range(served.vocab_size))


def test_an_anchored_slot_starts_its_derivation_from_the_question_entities(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    oracle = Oracle(index, served, SpanSyntax(), derivation=True)
    engine = Engine(index, served, SpanSyntax(), max_sequences=1, derivation=True)
    engine.anchor(0, ["Q6"])
    state = oracle.anchored_state(["Q6"])
    for step, token in enumerate(served.encode("⟦ Curie · place of birth · Paris⟧ ok")):
        engine.fill_mask()
        assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(state), step
        state = oracle.advance(state, token)
        engine.advance(np.array([token], dtype=np.int32))
    assert closed_span(index, engine.states_host(), 0).triples == (("Q6", "P19", "Q3"),)


@pytest.mark.parametrize("relations", ["frontier", "vocabulary", "successor"])
@pytest.mark.parametrize("opened", [False, True])
def test_engine_support_ladder_random_walks_with_versions(toy_kg, toy_tokenizer, relations, opened):
    served = ServedTokenizer(toy_tokenizer)
    for seed in (31, 32):
        seen = run_agreement(
            served, toy_kg, SpanSyntax(), sequences=12, steps=200, seed=seed,
            paths=True, committed_paths=True, escapes=False, relations=relations,
            versions=TOY_VERSIONS, opened=opened, unquoted_starts=True,
            start_rank="sitelinks", sitelinks=TOY_SITELINKS,
        )  # fmt: skip
        assert seen["versioned steps"] > 0


@pytest.mark.parametrize("relations", ["frontier", "vocabulary", "successor"])
@pytest.mark.parametrize("version", [None, "no-join"])
@pytest.mark.parametrize("opened", [False, True])
def test_engine_support_ladder_fixed_texts(relations, version, opened):
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SUPPORT_SYNTAX
    index = build_kg_index(SUPPORT_KG, served, syntax)
    modes = {"paths": True, "committed_paths": True, "escapes": False, "relations": relations}
    oracle = Oracle(index, served, syntax, versions=SUPPORT_VERSIONS, **modes)
    engine = Engine(index, served, syntax, 1,
                    versions=build_kg_versions(index, SUPPORT_VERSIONS), **modes)  # fmt: skip
    # The first y is absent from A's frontier. x|x is never a schema successor.
    texts = ["A|x] free answer", "A|y] anything", "A|x|y]", "A|x|x]"]
    for tail in texts:
        engine.reset()
        engine.set_version(0, version)
        state = oracle.anchored_state(["Q1"], version)
        engine.anchor(0, ["Q1"])
        if opened:
            engine.open_span(0)
            state = oracle.opened_state(state)
        for token in served.encode(("" if opened else "[") + tail):
            engine.fill_mask()
            assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(state)
            assert_agrees(
                expected_state(state, index, syntax),
                engine_state(engine.states_host(), 0, engine.view(version)),
                True,
            )
            if token not in oracle.allowed(state):
                break
            state = oracle.advance(state, token)
            engine.advance(np.array([token], dtype=np.int32))
        else:
            engine.fill_mask()
            assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(state)
            assert_agrees(
                expected_state(state, index, syntax),
                engine_state(engine.states_host(), 0, engine.view(version)),
                True,
            )
        assert engine.errors() == 0


@pytest.mark.parametrize("relations", ["frontier", "vocabulary", "successor"])
def test_engine_support_ladder_drafts_do_not_mutate_slots(toy_kg, toy_tokenizer, relations):
    torch = pytest.importorskip("torch")
    served, syntax = ServedTokenizer(toy_tokenizer), SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    edits = {"no-location": [("Q7", "P131", "Q99"), ("Q7", "P276", "Q99")]}
    modes = {"paths": True, "committed_paths": True, "escapes": False, "relations": relations}
    oracle = Oracle(index, served, syntax, versions=edits, **modes)
    engine = Engine(index, served, syntax, 2, versions=build_kg_versions(index, edits), **modes)
    tensor = lambda values: torch.tensor(values, dtype=torch.int32, device="cuda")  # noqa: E731
    engine.begin(tensor([0, 1]))
    engine.set_version(1, "no-location")
    prefixes = [served.encode("⟦ Pierre Curie · spouse"), served.encode("⟦ Marie Curie · employer")]
    suffixes = [" · employer⟧ so ⟦ University of Paris⟧", "⟧ so ⟦ University of Paris⟧"]
    if relations != "frontier":
        suffixes[1] = "⟧ so ⟦ Pierre Curie⟧"  # an answer outside any executed frontier
    history = torch.zeros((2, 100), dtype=torch.int32, device="cuda")
    for slot, prefix in enumerate(prefixes):
        history[slot, : len(prefix)] = tensor(prefix)
    engine.sync(tensor([0, 1]), history, tensor([0, 0]), tensor(list(map(len, prefixes))))
    engine.fill_mask()
    before_masks, before_states = engine.mask_host().copy(), engine.states_host()
    rows, positions, inputs, expected = [], [], [], []
    for slot in (1, 0):
        state = oracle.initial_state("no-location" if slot == 1 else None)
        for token in prefixes[slot]:
            state = oracle.advance(state, token)
        drafts = [prefixes[slot][-1], *served.encode(suffixes[slot])]
        for offset, token in enumerate(drafts):
            if offset:
                state = oracle.advance(state, token)
            rows.append(slot)
            positions.append(offset)
            inputs.append(token)
            expected.append(oracle.allowed(state))
    logits = torch.zeros((len(rows), served.vocab_size + 3), device="cuda")
    engine.apply_mask(logits, tensor(rows), local_pos=tensor(positions), input_ids=tensor(inputs))
    for row, allowed in enumerate(expected):
        assert set(torch.nonzero(torch.isfinite(logits[row])).flatten().tolist()) == allowed
    engine.fill_mask()
    assert np.array_equal(engine.mask_host(), before_masks)
    for name, values in before_states.items():
        assert np.array_equal(engine.states_host()[name], values), name


@pytest.mark.parametrize("relations", ["frontier", "vocabulary", "successor"])
def test_engine_support_ladder_crosses_bitmap_words(relations):
    kg = KnowledgeGraph(
        (Entity("Q1", ("A",)), Entity("Q2", ("B", "Bee"))),
        tuple(Relation(f"P{r}", (f"r{r}",)) for r in range(35)),
        (("Q1", "P0", "Q2"), ("Q2", "P34", "Q1")),
    )
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    index = build_kg_index(kg, served, syntax)
    modes = {"paths": True, "committed_paths": True, "escapes": False, "relations": relations}
    oracle = Oracle(index, served, syntax, **modes)
    engine = Engine(index, served, syntax, 1, **modes)
    _agree_on_text(oracle, engine, index, served, "⟦ A · r0 · r34 · r0⟧ so ⟦ B⟧")


def test_engine_matches_the_oracle_in_path_mode(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    for seed in (5, 6):
        seen = run_agreement(
            served, toy_kg, SpanSyntax(), sequences=24, steps=200, seed=seed, paths=True
        )
        assert seen["resolved"] > 5, seen
    index = build_kg_index(toy_kg, served, SpanSyntax())
    oracle = Oracle(index, served, SpanSyntax(), paths=True)
    loop = "⟦ Pierre Curie" + " · spouse" * 8 + "⟧"  # eight relations: the path must close
    texts = [
        loop + " then ⟦ Pierre Curie⟧ and ⟦ Pierre Curie · spouse · employer · located in⟧"
        " so ⟦ Paris⟧ ok",
        "⟦ Pierre Curie · spouse · employer⟧ so ⟦ University of Paris⟧ in"
        " ⟦ University of Paris · located in⟧",  # the answer takes no separator; later spans do
        "⟦ Curie · place of birth⟧ in ⟦ Paris⟧",  # both entities labelled Curie start
    ]
    for text in texts:
        engine = Engine(index, served, SpanSyntax(), max_sequences=1, paths=True)
        state = oracle.initial_state()
        for step, token in enumerate(served.encode(text)):
            engine.fill_mask()
            assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(state), step
            expected = expected_state(state, index, SpanSyntax())
            assert_agrees(expected, engine_state(engine.states_host(), 0, index), frontier=True)
            state = oracle.advance(state, token)
            engine.advance(np.array([token], dtype=np.int32))
        assert engine.errors() == 0


def test_engine_matches_the_oracle_without_bound_escapes(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    for seed in (7, 8):
        seen = run_agreement(
            served, toy_kg, SpanSyntax(), sequences=24, steps=200, seed=seed, paths=True,
            bound_escapes=False,
        )  # fmt: skip
        assert seen["resolved"] > 5, seen


def _agree_on_text(oracle, engine, index, served, text, state=None):
    """Step a single-slot engine and the oracle through the text, checking every mask and
    state; returns the oracle's state."""
    state = oracle.initial_state() if state is None else state
    for step, token in enumerate(served.encode(text)):
        engine.fill_mask()
        assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(state), step
        expected = expected_state(state, index, SpanSyntax())
        assert_agrees(expected, engine_state(engine.states_host(), 0, index), frontier=True)
        state = oracle.advance(state, token)
        engine.advance(np.array([token], dtype=np.int32))
    engine.fill_mask()
    assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(state)
    assert engine.errors() == 0
    return state


@pytest.mark.parametrize("relations", ["frontier", "vocabulary", "successor"])
@pytest.mark.parametrize("opened", [False, True])
@pytest.mark.parametrize("committed", [False, True])
def test_engine_end_after_answer_random_walks(relations, opened, committed):
    seen = run_agreement(
        ServedTokenizer(byte_level_tokenizer([])), SUPPORT_KG, SUPPORT_SYNTAX,
        sequences=12, steps=500, seed=27, paths=True, escapes=False,
        committed_paths=committed, relations=relations, versions=SUPPORT_VERSIONS,
        opened=opened, end_after_answer=True,
    )  # fmt: skip
    assert seen["answers"] > 0 and seen["versioned steps"] > 0, seen


@pytest.mark.parametrize("relations", ["frontier", "vocabulary", "successor"])
@pytest.mark.parametrize("version", [None, "edit"])
@pytest.mark.parametrize("opened", [False, True])
@pytest.mark.parametrize("escaped", [False, True])
def test_engine_end_after_answer_fixed_texts(toy_kg, relations, version, opened, escaped):
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    edits = {"edit": [("Q1", "P108", "Q2")]}
    modes = dict(paths=True, committed_paths=True, relations=relations,
                 end_after_answer=True, eos_token_ids=[0, 1], escapes=escaped)  # fmt: skip
    oracle = Oracle(index, served, syntax, versions=edits, **modes)
    engine = Engine(index, served, syntax, 1, versions=build_kg_versions(index, edits), **modes)
    engine.set_version(0, version)
    state = oracle.anchored_state(["Q1"], version)
    engine.anchor(0, ["Q1"])
    if opened:
        engine.open_span(0)
        state = oracle.opened_state(state)
    prefix = ("" if opened else "⟦") + " Marie Curie · employer⟧.\nAnswer: ⟦ "
    answer = "Pierre Curie" if version else "University of Paris"
    suffix = "⁇ unknown⟧ trailing text" if escaped else answer + "⟧"
    state = _agree_on_text(oracle, engine, engine.view(version), served, prefix + suffix, state)
    assert state.answer_complete is (not escaped)
    if not escaped:
        assert allowed_from_mask(engine.mask_host()[0]) == {0, 1}
        engine.advance(np.array([1], dtype=np.int32))
        engine.fill_mask()
        assert allowed_from_mask(engine.mask_host()[0]) == {0, 1}
        engine.advance(np.array(served.encode("W"), dtype=np.int32))
        assert engine.errors() == 1  # mask and transition both reject trailing text
    engine.reset()
    engine.fill_mask()
    assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(oracle.initial_state())


@pytest.mark.parametrize("text", [
    "⟦ earlier: Jaws | Star Wars · director · date of birth⟧ so ⟦ Star Wars⟧",
    "⟦ same: Jaws | Star Wars · director⟧ so ⟦ no⟧",
])  # fmt: skip
def test_engine_end_after_answer_comparisons(tmp_path, text):
    index, served, _ = dated_world(tmp_path)
    modes = dict(**EXPRESSIONS, end_after_answer=True, eos_token_ids=[0])
    oracle = Oracle(index, served, SpanSyntax(), **modes)
    engine = Engine(index, served, SpanSyntax(), 1, **modes)
    state = _agree_on_text(oracle, engine, index, served, text)
    assert state.answer_complete and allowed_from_mask(engine.mask_host()[0]) == {0}


@pytest.mark.parametrize("relations", ["frontier", "vocabulary", "successor"])
@pytest.mark.parametrize("stop", [False, True])
def test_engine_end_after_answer_drafts_and_slot_reuse(toy_kg, relations, stop):
    if stop and relations != "frontier":
        pytest.skip("STOP requires frontier relation support")
    torch = pytest.importorskip("torch")
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    modes = dict(paths=True, committed_paths=True, relations=relations, escapes=False,
                 end_after_answer=True, eos_token_ids=[0, 1],
                 stop="residual-goal" if stop else None)  # fmt: skip
    edits = {"edit": [("Q1", "P108", "Q2")]}
    oracle = Oracle(index, served, syntax, versions=edits, **modes)
    engine = Engine(index, served, syntax, 2, versions=build_kg_versions(index, edits), **modes)
    tensor = lambda v: torch.tensor(v, dtype=torch.int32, device="cuda")  # noqa: E731
    slots = tensor([0, 1])
    engine.begin(slots)
    engine.set_version(1, "edit")
    question = "Which employer has Marie Curie?" if stop else None
    if stop:
        for slot in (0, 1):
            engine.set_stop(slot, question)
    prefixes = [
        served.encode(f"⟦ Marie Curie · employer⟧.\nAnswer: ⟦ {answer}")
        for answer in ("University of Paris", "Pierre Curie")
    ]
    prefix = prefixes[0]
    history = torch.zeros((2, 256), dtype=torch.int32, device="cuda")
    for slot, own in enumerate(prefixes):
        history[slot, : len(own)] = tensor(own)
    engine.sync(slots, history, tensor([0, 0]), tensor(list(map(len, prefixes))))
    # One draft group completes the answer, tries forbidden text, then reaches a row
    # verification cannot visit. The other leaves the answer open; slot order is reversed.
    drafts = [*served.encode("⟧"), *served.encode("W")]
    rows = [1, *([0] * (len(drafts) + 1))]
    positions = [0, *range(len(drafts) + 1)]
    inputs = [prefixes[1][-1], prefix[-1], *drafts]
    logits = torch.zeros((len(rows), served.vocab_size + 3), device="cuda")
    engine.fill_mask()
    before_mask, before_state = engine.mask_host().copy(), engine.states_host()
    engine.apply_mask(logits, tensor(rows), local_pos=tensor(positions), input_ids=tensor(inputs))
    state = oracle.initial_state(question=question)
    for token in prefix:
        state = oracle.advance(state, token)
    other = oracle.initial_state("edit", question=question)
    for token in prefixes[1]:
        other = oracle.advance(other, token)
    assert set(torch.nonzero(torch.isfinite(logits[0])).flatten().tolist()) == oracle.allowed(other)
    for k in range(len(drafts)):
        expected = oracle.allowed(state)
        assert set(torch.nonzero(torch.isfinite(logits[k + 1])).flatten().tolist()) == expected
        state = oracle.advance(state, drafts[k]) if drafts[k] in expected else state
    # The row immediately after the closer admits every supplied EOS, and no padding.
    assert set(torch.nonzero(torch.isfinite(logits[-2])).flatten().tolist()) == {0, 1}
    assert torch.isfinite(logits[-1]).all()  # after the disallowed draft
    engine.fill_mask()
    assert np.array_equal(before_mask, engine.mask_host())
    assert all(np.array_equal(v, engine.states_host()[k]) for k, v in before_state.items())
    # Committing the same closer ends just slot 0; then begin/end/reset clear completion.
    tail = served.encode("⟧")
    history[0, len(prefix) : len(prefix) + len(tail)] = tensor(tail)
    engine.sync(slots, history, tensor([0, 0]), tensor([len(prefix) + len(tail), len(prefixes[1])]))
    engine.fill_mask()
    assert allowed_from_mask(engine.mask_host()[0]) == {0, 1}
    assert allowed_from_mask(engine.mask_host()[1]) != {0, 1}
    engine.end(tensor([0]))
    engine.begin(tensor([0]))
    engine.fill_mask()
    assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(
        oracle.initial_state(question=question)
    )


def test_engine_matches_the_oracle_with_committed_paths(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    for seed in (5, 6):
        seen = run_agreement(
            served, toy_kg, SpanSyntax(), sequences=24, steps=200, seed=seed, paths=True,
            committed_paths=True,
        )  # fmt: skip
        assert seen["resolved"] > 5, seen
    index = build_kg_index(toy_kg, served, SpanSyntax())
    oracle = Oracle(index, served, SpanSyntax(), paths=True, committed_paths=True)
    texts = {
        "⟦ Curie · place of birth⟧ in ⟦ Paris⟧": ("Q4",),  # Q17's, not the union's Q3
        "⟦ Curie · place of birth · located in⟧ in ⟦ Rhône–Alpes⟧": ("Q12",),  # Q6's
        "⟦ Pierre Curie · spouse · employer · located in⟧ so ⟦ Paris⟧": ("Q3",),
    }
    for text, answer in texts.items():
        engine = Engine(
            index, served, SpanSyntax(), max_sequences=1, paths=True, committed_paths=True
        )
        state = _agree_on_text(oracle, engine, index, served, text)
        assert state.last_span.entity_ids == answer
        assert closed_span(index, engine.states_host(), 0).entity_ids == answer
    # Q4 has no triples: no path starts from the committed answer.
    engine = Engine(index, served, SpanSyntax(), max_sequences=1, paths=True, committed_paths=True)
    state = _agree_on_text(oracle, engine, index, served, "⟦ Curie · place of birth⟧ in ⟦ Paris")
    assert served.encode(" ·")[0] not in oracle.allowed(state)


def test_engine_and_oracle_reach_merged_dates_along_a_path(tmp_path):
    """Dates merged onto an entity KG (merge_dates) are entities like any other: a path
    whose last relation reaches a date binds the answer to it, written at the date's
    stored precision in either day format."""
    from tests.conftest import write_dated_kgs
    from xgraph.literals import FIRST_ID, merge_dates
    from xgraph.wikidata import build_index

    literal, entity = write_dated_kgs(tmp_path)
    merge_dates(literal, entity, tmp_path / "dated")
    served = ServedTokenizer(byte_level_tokenizer([]))
    build_index(tmp_path / "dated", served, SpanSyntax(), tmp_path / "index")
    index = load_kg_index(tmp_path / "index")
    oracle = Oracle(index, served, SpanSyntax(), paths=True, committed_paths=True)
    texts = {
        "⟦ Jaws · director · date of birth⟧ so ⟦ 18 December 1946⟧": f"Q{FIRST_ID}",
        "⟦ Jaws · director · date of birth⟧ so ⟦ December 18, 1946⟧": f"Q{FIRST_ID}",
        "⟦ Jaws · publication date⟧ in ⟦ June 1975⟧": f"Q{FIRST_ID + 1}",
        "⟦ Star Wars · publication date⟧ in ⟦ 1977⟧": f"Q{FIRST_ID + 4}",
    }
    for text, answer in texts.items():
        engine = Engine(
            index, served, SpanSyntax(), max_sequences=1, paths=True, committed_paths=True
        )
        state = _agree_on_text(oracle, engine, index, served, text)
        assert state.last_span.entity_ids == (answer,)
        assert closed_span(index, engine.states_host(), 0).entity_ids == (answer,)


@pytest.mark.parametrize("committed", [False, True], ids=["union", "committed"])
def test_engine_matches_the_oracle_when_a_frontier_overflows(committed):
    """More entities share the start label than a frontier holds (MAX_REACHED): the engine
    keeps the same ones as the oracle, by row, or committed by start rank."""
    kg = crowded_start_kg()
    served = crowded_tokenizer(kg)
    index = build_kg_index(kg, served, SpanSyntax())
    oracle = Oracle(index, served, SpanSyntax(), paths=True, committed_paths=committed)
    for text in ("⟦ X · r⟧ so ⟦", "⟦ X · r · t⟧ so ⟦"):
        engine = Engine(
            index, served, SpanSyntax(), max_sequences=1, paths=True, committed_paths=committed
        )
        _agree_on_text(oracle, engine, index, served, text)
    engine = Engine(
        index, served, SpanSyntax(), max_sequences=1, paths=True, committed_paths=committed
    )
    answer = "T1" if committed else "T0"
    _agree_on_text(oracle, engine, index, served, f"⟦ X · r · t⟧ so ⟦ {answer}⟧")
    assert closed_span(index, engine.states_host(), 0).entity_ids == (
        ("Q2001",) if committed else ("Q2000",)
    )


@pytest.mark.parametrize("anchor", [["Q6"], ["Q17"]])
def test_committing_leaves_an_anchored_path_unchanged_in_the_engine(toy_kg, toy_tokenizer, anchor):
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    union = Oracle(index, served, SpanSyntax(), paths=True)
    text = "⟦ Curie · place of birth⟧ in ⟦ Paris⟧"
    expected = union.anchored_state(anchor)
    for token in served.encode(text):
        expected = union.advance(expected, token)
    oracle = Oracle(index, served, SpanSyntax(), paths=True, committed_paths=True)
    engine = Engine(index, served, SpanSyntax(), max_sequences=1, paths=True, committed_paths=True)
    engine.anchor(0, anchor)
    state = _agree_on_text(oracle, engine, index, served, text, oracle.anchored_state(anchor))
    assert state.last_span == expected.last_span
    assert closed_span(index, engine.states_host(), 0).entity_ids == expected.last_span.entity_ids


@pytest.mark.parametrize(
    "mode",
    [{"paths": True, "committed_paths": True}, {"derivation": True}],
    ids=["committed", "derivation"],
)
def test_engine_matches_the_oracle_with_the_sitelink_rank(toy_kg, toy_tokenizer, mode):
    served = ServedTokenizer(toy_tokenizer)
    for seed in (5, 6):
        seen = run_agreement(
            served, toy_kg, SpanSyntax(), sequences=24, steps=200, seed=seed,
            start_rank="sitelinks", sitelinks=TOY_SITELINKS, **mode,
        )  # fmt: skip
        assert seen["resolved"] > 5, seen


def test_the_sitelink_rank_and_degree_disagree_alike_in_engine_and_oracle(toy_kg, toy_tokenizer):
    """Curie: Q6 has more sitelinks, Q17 more triples; Paris: Q4 more sitelinks, Q3 more
    triples. Each text resolves to the sitelink rank's entity in both, not the degree's."""
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax(), sitelinks=TOY_SITELINKS)
    texts = {
        "committed": {
            "⟦ Curie · place of birth⟧ in ⟦ Paris⟧": ("Q3",),  # Q6's birthplace
            "⟦ Curie · employer⟧ at ⟦ University of Paris⟧": ("Q7",),  # Q6 has none: Q17
        },
        "derivation": {
            "⟦ Curie · place of birth · Paris⟧": (("Q6", "P19", "Q3"),),
            "⟦ University of Paris · located in · Paris⟧ so ⟦ Paris⟧": ("Q4",),
        },
    }
    modes = {
        "committed": {"paths": True, "committed_paths": True},
        "derivation": {"derivation": True},
    }
    for mode, resolved in texts.items():
        oracle = Oracle(index, served, SpanSyntax(), start_rank="sitelinks", **modes[mode])
        for text, expected in resolved.items():
            engine = Engine(
                index, served, SpanSyntax(), max_sequences=1, start_rank="sitelinks",
                **modes[mode],
            )  # fmt: skip
            state = _agree_on_text(oracle, engine, index, served, text)
            closed = closed_span(index, engine.states_host(), 0)
            got = closed.triples if closed.triples else closed.entity_ids
            assert got == expected == (state.last_span.triples or state.last_span.entity_ids)


def test_the_sitelink_rank_needs_an_index_with_sitelink_counts(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    with pytest.raises(ValueError, match="sitelink counts"):
        Engine(
            index, served, SpanSyntax(), max_sequences=1, derivation=True, start_rank="sitelinks"
        )
    with pytest.raises(ValueError, match="start_rank"):
        Engine(index, served, SpanSyntax(), max_sequences=1, start_rank="fame")


# KG versions (docs/kg-versions-adr.md): the engine with a version against the oracle with it,
# and against the engine on the KG index rebuilt from the edited triples.

TOY_VERSIONS = {
    "moved": [("Q1", "P108", "Q3"), ("Q4", "P131", "Q12")],  # a replaced and a new pair
    "q6": [("Q6", "P108", "Q7"), ("Q6", "P26", "Q2")],  # Q6 becomes the best-connected Curie
    "spouse": [("Q1", "P26", "Q10")],  # unwritable: Marie Curie loses her spouses
}
TIED_SITELINKS = {**TOY_SITELINKS, "Q6": 5, "Q17": 5}
"""The Curies tie on sitelinks, so that the degree of the version read decides between them."""


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("anchored", [False, True])
def test_version_isolation_mixed_slots_and_reset_equal_rebuilt_engines(committed, anchored):
    """Every exercised mask and public transition, including rejected hybrid answers.

    Reassign all slots, then reset without set_version to test return to base.
    Native engines on literal per-version KGs provide the second reference.
    """
    torch = pytest.importorskip("torch")
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    index = build_kg_index(ISOLATION_KG, served, syntax)
    options = dict(paths=True, committed_paths=committed, escapes=False)
    oracle = Oracle(index, served, syntax, versions=ISOLATION_EDITS, **options)
    versions = build_kg_versions(index, ISOLATION_EDITS)
    names = list(ISOLATION_GRAPHS)
    engine = Engine(index, served, syntax, len(names), versions=versions, **options)
    tensor = lambda v: torch.tensor(v, dtype=torch.int32, device="cuda")  # noqa: E731
    slots = tensor(list(range(len(names))))
    for reset, bindings in ((False, names), (False, names[::-1]), (True, [None] * len(names))):
        if reset:
            engine.reset()
        # reset deactivates slots; sync only advances requests admitted by begin.
        engine.begin(slots)
        refs, states, rebuilt_states, texts, views = [], [], [], [], []
        for slot, version in enumerate(bindings):
            ref_index = build_kg_index(
                replace(ISOLATION_KG, triples=ISOLATION_GRAPHS[version]), served, syntax
            )
            ref_oracle = Oracle(ref_index, served, syntax, **options)
            ref_engine = Engine(ref_index, served, syntax, 1, **options)
            if version is not None:
                engine.set_version(slot, version)
            if anchored:
                engine.anchor(slot, ["Q1"])
                ref_engine.anchor(0, ["Q1"])
            refs.append((ref_index, ref_oracle, ref_engine))
            states.append(
                oracle.anchored_state(["Q1"], version)
                if anchored
                else oracle.initial_state(version)
            )
            rebuilt_states.append(
                ref_oracle.anchored_state(["Q1"]) if anchored else ref_oracle.initial_state()
            )
            # Delta must be blocked in the base and hybrid-source, and allowed
            # after inner/terminal replacement. Deletion blocks the second hop.
            texts.append(served.encode("⟦ Atlas · creator · born⟧ Answer: ⟦ Delta⟧"))
            views.append(versions.view(index, version))
        history = torch.zeros((len(names), 256), dtype=torch.int32, device="cuda")
        lengths, rejected = [0] * len(names), set()
        for step in range(max(map(len, texts)) + 1):
            engine.fill_mask()
            masks, native = engine.mask_host(), engine.states_host()
            advancing = []
            for slot, (ref_index, ref_oracle, ref_engine) in enumerate(refs):
                ref_engine.fill_mask()
                expected = ref_oracle.allowed(rebuilt_states[slot])
                assert allowed_from_mask(masks[slot]) == expected
                assert allowed_from_mask(ref_engine.mask_host()[0]) == expected
                assert oracle.allowed(states[slot]) == expected
                assert replace(states[slot], version=None) == rebuilt_states[slot]
                got = engine_state(native, slot, views[slot])
                assert got == engine_state(ref_engine.states_host(), 0, ref_index)
                assert_agrees(
                    expected_state(rebuilt_states[slot], ref_index, syntax), got, frontier=True
                )
                if step == len(texts[slot]) or slot in rejected:
                    continue
                token = texts[slot][step]
                if token not in expected:
                    rejected.add(slot)
                    continue
                states[slot] = oracle.advance(states[slot], token)
                rebuilt_states[slot] = ref_oracle.advance(rebuilt_states[slot], token)
                ref_engine.advance(np.array([token], np.int32))
                history[slot, lengths[slot]] = token
                lengths[slot] += 1
                advancing.append(slot)
            if advancing:
                engine.sync(tensor(advancing), history, tensor([0] * len(names)), tensor(lengths))
        for slot, version in enumerate(bindings):
            assert (slot in rejected) == (ISOLATION_ANSWERS[version] != "Delta")
            if slot not in rejected:
                assert closed_span(views[slot], engine.states_host(), slot).entity_ids == ("Q5",)
        assert engine.errors() == 0


@pytest.mark.parametrize("paths", [False, True])
def test_version_isolation_pair_deletion_matches_rebuilt_engine(paths):
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    kg = replace(ISOLATION_KG, triples=(*ISOLATION_KG.triples, ("Q2", "P2", "Q5")))
    index = build_kg_index(kg, served, syntax)
    options = dict(paths=paths, escapes=False)
    versions = build_kg_versions(index, {"deleted": ISOLATION_EDITS["deleted"]})
    engine = Engine(index, served, syntax, 1, versions=versions, **options)
    for version in (None, "deleted", None):
        engine.reset()
        engine.set_version(0, version)
        ref_index = build_kg_index(
            replace(kg, triples=ISOLATION_GRAPHS["deleted"]) if version else kg, served, syntax
        )
        oracle = Oracle(ref_index, served, syntax, **options)
        reference = Engine(ref_index, served, syntax, 1, **options)
        state = oracle.initial_state()
        text = (
            "⟦ Ada⟧"
            if version
            else ("⟦ Ada · born⟧ Answer: ⟦ Delta⟧" if paths else "⟦ Ada · born · Delta⟧")
        )
        tokens = served.encode(text)
        for at in range(len(tokens) + 1):
            engine.fill_mask()
            reference.fill_mask()
            assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(state)
            assert np.array_equal(engine.mask_host(), reference.mask_host())
            got = engine_state(engine.states_host(), 0, versions.view(index, version))
            assert got == engine_state(reference.states_host(), 0, ref_index)
            assert_agrees(expected_state(state, ref_index, syntax), got, frontier=paths)
            if at < len(tokens):
                state = oracle.advance(state, tokens[at])
                token = np.array([tokens[at]], np.int32)
                engine.advance(token)
                reference.advance(token)


@pytest.mark.parametrize("mode", VERSION_MODES)
@pytest.mark.parametrize("seed", [0, 1])
def test_engine_matches_the_oracle_and_rebuilt_indexes_with_kg_versions(seed, mode):
    kg, versions = random_versioned_kg(seed)
    seen = run_agreement(
        versioned_tokenizer(), kg, SpanSyntax(), sequences=12, steps=120, seed=seed,
        versions=versions, sitelinks=random_sitelinks(kg, seed), **VERSION_MODES[mode],
    )  # fmt: skip
    assert seen["versioned steps"] > 500 and seen["resolved"] > 5, seen


@pytest.mark.parametrize("mode", VERSION_MODES)
def test_engine_matches_the_oracle_and_rebuilt_indexes_with_toy_kg_versions(
    toy_kg, toy_tokenizer, mode
):
    served = ServedTokenizer(toy_tokenizer)
    seen = run_agreement(
        served, toy_kg, SpanSyntax(), sequences=16, steps=150, seed=5, versions=TOY_VERSIONS,
        sitelinks=TIED_SITELINKS, **VERSION_MODES[mode],
    )  # fmt: skip
    assert seen["versioned steps"] > 1000 and seen["resolved"] > 5, seen


def test_a_slot_reads_its_kg_version_until_it_begins_again(toy_kg, toy_tokenizer):
    """Through the slot API: a version set after begin holds over spans and draft rows (the
    scratch memory of apply_mask), and the next begin returns the slot to the base KG."""
    torch = pytest.importorskip("torch")
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    oracle = Oracle(index, served, SpanSyntax(), versions=TOY_VERSIONS)
    engine = Engine(
        index, served, SpanSyntax(), max_sequences=2,
        versions=build_kg_versions(index, TOY_VERSIONS),
    )  # fmt: skip
    tensor = lambda v: torch.tensor(v, dtype=torch.int32, device="cuda")  # noqa: E731
    committed = served.encode(" ⟦ Pierre Curie · employer · University of Paris⟧ and ⟦ Marie")
    drafted = served.encode(" Curie · employer ·")
    history = torch.zeros((2, 64), dtype=torch.int32, device="cuda")
    history[:, : len(committed)] = tensor(committed)
    slots = tensor([0, 1])
    for version in ("moved", None):
        engine.begin(slots)
        engine.set_version(1, version)
        engine.sync(slots, history, tensor([0, 0]), tensor([len(committed)] * 2))
        local = list(range(len(drafted) + 1))
        rows = [0] * len(local) + [1] * len(local)
        logits = torch.zeros((len(rows), served.vocab_size), device="cuda")
        engine.apply_mask(
            logits, tensor(rows), local_pos=tensor(local * 2),
            input_ids=tensor([committed[-1], *drafted] * 2),
        )  # fmt: skip
        allowed = ~torch.isneginf(logits).cpu()
        for slot, name in ((0, None), (1, version)):
            state = oracle.initial_state(name)
            for token in committed:
                state = oracle.advance(state, token)
            for k in local:
                got = set(torch.nonzero(allowed[slot * len(local) + k]).flatten().tolist())
                assert got == oracle.allowed(state), (version, slot, k)
                if k < len(drafted):
                    state = oracle.advance(state, drafted[k])
        if version == "moved":  # the last draft row is Marie Curie's employers: Q3's Paris
            assert not torch.equal(logits[len(local) - 1], logits[-1])
    assert engine.errors(2) == 0


def test_a_kg_version_changes_what_a_fact_span_resolves_to(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    versions = build_kg_versions(index, TOY_VERSIONS)
    engine = Engine(index, served, SpanSyntax(), max_sequences=1, versions=versions)
    engine.set_version(0, "moved")
    for token in served.encode("⟦ Marie Curie · employer · Paris⟧"):
        engine.advance(np.array([token], dtype=np.int32))
    states = engine.states_host()
    assert engine.errors() == 0
    assert closed_span(versions.view(index, "moved"), states, 0).triples == (("Q1", "P108", "Q3"),)
    assert closed_span(index, states, 0).triples == ()  # the base KG holds no such triple


def test_an_unknown_kg_version_is_refused_by_the_engine(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    engine = Engine(
        index, served, SpanSyntax(), max_sequences=1,
        versions=build_kg_versions(index, TOY_VERSIONS),
    )  # fmt: skip
    with pytest.raises(ValueError, match="no KG version 'w9'"):
        engine.set_version(0, "w9")
    plain = Engine(index, served, SpanSyntax(), max_sequences=1)
    with pytest.raises(ValueError, match="no KG version 'moved'"):
        plain.set_version(0, "moved")


@pytest.mark.parametrize("difference", ["shape", "content"])
def test_kg_versions_of_another_kg_index_are_refused(toy_kg, toy_tokenizer, difference):
    from dataclasses import replace

    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    other_kg = (
        KnowledgeGraph(entities=toy_kg.entities[:5], relations=toy_kg.relations)
        if difference == "shape"
        else replace(toy_kg, triples=toy_kg.triples[::-1])
    )
    other = build_kg_versions(build_kg_index(other_kg, served, SpanSyntax()), {"v": []})
    with pytest.raises(IndexMismatch, match="KG versions were built for another KG index"):
        Engine(index, served, SpanSyntax(), max_sequences=1, versions=other)


@pytest.mark.parametrize("version", [None, "moved"])
def test_legacy_index_masks_equal_fingerprinted_index_with_versions(
    toy_kg, toy_tokenizer, tmp_path, version
):
    import json

    served, syntax = ServedTokenizer(toy_tokenizer), SpanSyntax()
    toy_index = build_kg_index(toy_kg, served, syntax)
    save_kg_index(toy_index, tmp_path)
    path = tmp_path / "kg-index.json"
    header = json.loads(path.read_text())
    for name in ("base_fingerprint", "source_graph_hash", "build_config"):
        header.pop(name, None)
    path.write_text(json.dumps(header) + "\n")
    legacy = load_kg_index(tmp_path)
    assert legacy.fingerprint_status == "legacy"
    oracle = Oracle(toy_index, served, syntax, versions=TOY_VERSIONS)
    engines = [
        Engine(index, served, syntax, 1, versions=build_kg_versions(index, TOY_VERSIONS))
        for index in (toy_index, legacy)
    ]
    for engine in engines:
        engine.set_version(0, version)
    state = oracle.initial_state(version)
    endpoint = "Paris" if version else "University of Paris"
    for token in served.encode(f"⟦ Marie Curie · employer · {endpoint}⟧"):
        for engine in engines:
            engine.fill_mask()
            assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(state)
        assert engines[0].mask_host().tobytes() == engines[1].mask_host().tobytes()
        state = oracle.advance(state, token)
        for engine in engines:
            engine.advance(np.array([token], np.int32))
    assert all(engine.errors() == 0 for engine in engines)


# A forced opener: a slot that starts inside a span, its prompt ending with the opener.


@pytest.mark.parametrize("mode", ["plain", "paths", "committed by sitelinks"])
def test_engine_matches_the_oracle_from_an_opened_span(toy_kg, toy_tokenizer, mode):
    served = ServedTokenizer(toy_tokenizer)
    seen = run_agreement(
        served, toy_kg, SpanSyntax(), sequences=16, steps=100, seed=3, versions=TOY_VERSIONS,
        sitelinks=TIED_SITELINKS, opened=True, **VERSION_MODES[mode],
    )  # fmt: skip
    assert seen["versioned steps"] > 500 and seen["resolved"] > 5, seen


def test_an_opened_slot_starts_in_a_span_with_its_anchor_and_version(toy_kg, toy_tokenizer):
    """Through the slot API: open_span after begin composes with anchor and set_version, in
    either order; the slot's history from its prompt's end on continues the span, and the
    next begin returns the slot to free text."""
    torch = pytest.importorskip("torch")
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    oracle = Oracle(index, served, SpanSyntax(), derivation=True, versions=TOY_VERSIONS)
    engine = Engine(
        index, served, SpanSyntax(), max_sequences=3, derivation=True,
        versions=build_kg_versions(index, TOY_VERSIONS),
    )  # fmt: skip
    tensor = lambda v: torch.tensor(v, dtype=torch.int32, device="cuda")  # noqa: E731
    prompt = served.encode("Answer: ⟦")  # the prefilled opener, never advanced over
    base = " Marie Curie · employer · University of Paris⟧ so ⟦ University"
    replies = {  # slot 2 reads the version, where Marie Curie's employer is Paris (Q3)
        0: served.encode(base),
        1: served.encode(base),
        2: served.encode(" Marie Curie · employer · Paris⟧ near ⟦ Paris"),
    }
    history = torch.zeros((3, 64), dtype=torch.int32, device="cuda")
    for slot, reply in replies.items():
        history[slot, : len(prompt) + len(reply)] = tensor(prompt + reply)
    slots = tensor([0, 1, 2])
    states = {
        0: oracle.initial_state(),
        1: oracle.opened_state(oracle.initial_state()),
        2: oracle.opened_state(oracle.anchored_state(["Q1"], "moved")),
    }
    engine.begin(slots)
    engine.open_span(1)
    engine.open_span(2)  # before its anchor and version: the order does not matter
    engine.anchor(2, ["Q1"])
    engine.set_version(2, "moved")
    for done in range(max(map(len, replies.values())) + 1):
        lengths = [len(prompt) + min(done, len(replies[slot])) for slot in range(3)]
        engine.sync(slots, history, tensor([len(prompt)] * 3), tensor(lengths))
        logits = torch.zeros((3, served.vocab_size), device="cuda")
        engine.apply_mask(logits, slots)
        allowed = ~torch.isneginf(logits).cpu()
        for slot, reply in replies.items():
            if done > len(reply):
                continue
            got = set(torch.nonzero(allowed[slot]).flatten().tolist())
            assert got == oracle.allowed(states[slot]), (slot, done)
            if done < len(reply):
                states[slot] = oracle.advance(states[slot], reply[done])
    assert engine.errors(3) == 0
    assert states[1].last_span.triples == (("Q1", "P108", "Q7"),)
    assert states[2].last_span.triples == (("Q1", "P108", "Q3"),)  # the version's employer
    assert states[0].last_span is None  # free text: no span closed
    engine.begin(tensor([1]))  # the next request in slot 1 starts in free text
    engine.fill_mask(3)
    free = allowed_from_mask(engine.mask_host(3)[1])
    assert free == oracle.allowed(oracle.initial_state())


def test_open_span_refuses_a_slot_out_of_range(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    engine = Engine(
        build_kg_index(toy_kg, served, SpanSyntax()), served, SpanSyntax(), max_sequences=2
    )
    with pytest.raises(ValueError, match="slot"):
        engine.open_span(2)


# Unquoted starts: an unbound subject cannot name a label that starts with a quotation mark.

QUOTED_VERSIONS = {
    "moved": [("Q4", "P800", "Q5"), ("Q1", "P800", "Q3")],  # a replaced and a new pair
    "dropped": [("Q2", "P50", "Q1")],
}


@pytest.mark.parametrize(
    ("opened", "wide_rows"), [(False, 4096), (True, 1)], ids=["free-scanned", "opened-precomputed"]
)
@pytest.mark.parametrize("mode", ["plain", "derivation", "committed by sitelinks"])
def test_engine_matches_the_oracle_with_unquoted_starts(mode, opened, wide_rows, monkeypatch):
    """Scanned or, with every prefix wide, precomputed: the next tokens of the prefix ' “'
    and ' –' share hold the quotation mark's continuation, which an unbound subject must
    not offer."""
    import xgraph.engine

    monkeypatch.setattr(xgraph.engine, "WIDE_ROWS", wide_rows)
    seen = Counter()
    for seed in (0, 1, 2):
        seen += run_agreement(
            quoted_tokenizer(), QUOTED_KG, SpanSyntax(), sequences=12, steps=120, seed=seed,
            versions=QUOTED_VERSIONS, sitelinks=random_sitelinks(QUOTED_KG, seed),
            opened=opened, unquoted_starts=True, **VERSION_MODES[mode],
        )  # fmt: skip
    assert seen["versioned steps"] > 500 and seen["resolved"] > 5, seen


def test_engine_matches_the_oracle_on_quoted_labels_with_unquoted_starts():
    """Through texts: objects, answers after a path, and anchored subjects name quoted labels,
    and an anchored path composes with a KG version and a forced opener."""
    served = quoted_tokenizer()
    index = build_kg_index(QUOTED_KG, served, SpanSyntax())
    versions = build_kg_versions(index, QUOTED_VERSIONS)
    modes = {"paths": True, "committed_paths": True, "unquoted_starts": True}
    oracle = Oracle(index, served, SpanSyntax(), versions=QUOTED_VERSIONS, **modes)
    texts = [
        (None, None, False, "⟦ Kleb · by · work⟧ so ⟦ “Hero”⟧", "Q3"),
        (
            None, None, False,
            '⟦ –Dash · by⟧ is ⟦ Ann⟧, of ⟦ Ann · work⟧: ⟦ "Klis"⟧',
            "Q2",
        ),
        (["Q2"], None, True, ' "Klis" · by · work⟧ so ⟦ “Hero”⟧', "Q3"),
        (["Q1"], "moved", True, " Kleb · work⟧ so ⟦ “Hero”⟧", "Q3"),
    ]  # fmt: skip
    for anchor, version, opened, text, answer in texts:
        engine = Engine(index, served, SpanSyntax(), max_sequences=1, versions=versions, **modes)
        state = oracle.initial_state(version)
        if anchor:
            engine.anchor(0, anchor)
            state = oracle.anchored_state(anchor, version)
        engine.set_version(0, version)
        if opened:
            engine.open_span(0)
            state = oracle.opened_state(state)
        state = _agree_on_text(oracle, engine, index, served, text, state)
        assert state.last_span.entity_ids == (answer,), text
        assert closed_span(engine.view(version), engine.states_host(), 0).entity_ids == (answer,)


def test_a_quotation_mark_starting_an_unbound_subject_is_an_error():
    served = quoted_tokenizer()
    index = build_kg_index(QUOTED_KG, served, SpanSyntax())
    for unquoted_starts, errors in ((False, 0), (True, 1)):
        engine = Engine(
            index, served, SpanSyntax(), max_sequences=1, unquoted_starts=unquoted_starts
        )
        for token in served.encode('⟦ "Klis"'):
            engine.advance(np.array([token], dtype=np.int32))
        assert engine.errors() == errors


# Expressions: date comparisons (`⟦ earlier: A | B · relation⟧`) over KG indexes with dates.


def test_expressions_are_refused_without_committed_paths_escapes_or_dates(toy_kg, toy_tokenizer):
    kg, dates, served = compare_kg()
    index = build_kg_index(kg, served, SpanSyntax(), dates=dates)
    with pytest.raises(ValueError, match="committed paths"):
        Engine(index, served, SpanSyntax(), max_sequences=1, paths=True, expressions=True)
    with pytest.raises(ValueError, match="escapes"):
        Engine(index, served, SpanSyntax(), max_sequences=1, escapes=False, **EXPRESSIONS)
    undated = build_kg_index(toy_kg, ServedTokenizer(toy_tokenizer), SpanSyntax())
    with pytest.raises(ValueError, match="date table"):
        Engine(
            undated, ServedTokenizer(toy_tokenizer), SpanSyntax(), max_sequences=1, **EXPRESSIONS
        )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_engine_matches_the_oracle_on_comparisons(seed):
    """Random walks that write operators, operands, relations, and closers over random dated
    KGs (nested and overlapping dates, shared labels), and over the hand-built one."""
    kg, dates, _ = random_dated_kg(seed)
    seen = run_agreement(
        dated_tokenizer(), kg, SpanSyntax(), sequences=32, steps=250, seed=seed, dates=dates,
        **EXPRESSIONS,
    )  # fmt: skip
    assert seen["comparisons"] > 3 and seen["resolved"] > 5, seen
    assert seen["value comparisons"] > 0, seen
    kg, dates, served = compare_kg()
    seen = run_agreement(
        served, kg, SpanSyntax(), sequences=24, steps=150, seed=seed, dates=dates, **EXPRESSIONS
    )
    assert seen["comparisons"] > 3, seen


@pytest.mark.parametrize("seed", [0, 1])
def test_engine_matches_the_oracle_and_rebuilt_indexes_on_comparisons_with_kg_versions(seed):
    kg, dates, versions = random_dated_kg(seed)
    seen = run_agreement(
        dated_tokenizer(), kg, SpanSyntax(), sequences=12, steps=150, seed=seed, dates=dates,
        versions=versions, sitelinks=random_sitelinks(kg, seed), start_rank="sitelinks",
        **EXPRESSIONS,
    )  # fmt: skip
    assert seen["versioned steps"] > 500 and seen["comparisons"] > 0, seen


def test_engine_matches_the_oracle_on_comparisons_from_an_opened_span():
    kg, dates, _ = random_dated_kg(3)
    seen = run_agreement(
        dated_tokenizer(), kg, SpanSyntax(), sequences=16, steps=100, seed=3, dates=dates,
        opened=True, **EXPRESSIONS,
    )  # fmt: skip
    assert seen["comparisons"] > 0, seen


def test_engine_resolves_comparisons_as_the_oracle_does(tmp_path):
    """The hand-worked comparisons of the span-language tests through the engine: the same
    masks and states every step, and the same result and operands decoded from the state."""
    from xgraph.index import VERDICTS

    kg, dates, served = compare_kg()
    index = build_kg_index(kg, served, SpanSyntax(), dates=dates, verdicts=True)
    oracle = Oracle(index, served, SpanSyntax(), **EXPRESSIONS)
    texts = {
        "⟦ same: Alien | Hair · country of origin⟧ so ⟦ yes⟧": (VERDICTS["yes"],),
        "⟦ same: Duel | Hair · country of origin⟧ so ⟦ no⟧": (VERDICTS["no"],),
        "⟦ same: Jaws | Hair · country of origin⟧ so ⟦ yes⟧": (VERDICTS["yes"],),
        "⟦ same: Alien | Alien · publication date⟧ so ⟦ yes⟧": (VERDICTS["yes"],),
        "⟦ same: Alien | Hair · publication date⟧ so ⟦ no⟧": (VERDICTS["no"],),
        "⟦ earlier: Alien | Hair · publication date⟧ so ⟦ Hair⟧": ("Q1", "Q2"),
        "⟦ later: Duel | Alien · publication date⟧ so ⟦ Alien⟧": ("Q1",),
        "⟦ earlier: Jaws | Alien · publication date⟧ so ⟦ Jaws⟧": ("Q4",),
        "⟦ later: Jaws | Duel · publication date⟧ so ⟦ Jaws⟧": ("Q4",),
        "⟦ earlier: Alien | Alien · publication date⟧ so ⟦ Alien⟧": ("Q1",),
    }
    for text, answer in texts.items():
        comparison, rest = text.split(" so ")
        engine = Engine(index, served, SpanSyntax(), max_sequences=1, **EXPRESSIONS)
        state = _agree_on_text(oracle, engine, index, served, comparison)
        closed = closed_span(index, engine.states_host(), 0)
        assert closed.kind == "comparison"
        assert closed.entity_ids == answer == state.last_span.entity_ids
        assert closed.operator == state.last_span.operator
        assert closed.operands == state.last_span.operands
        _agree_on_text(oracle, engine, index, served, " so " + rest, state)
    index, served, _ = dated_world(tmp_path)
    oracle = Oracle(index, served, SpanSyntax(), **EXPRESSIONS)
    for text, answer in {
        "⟦ earlier: Jaws | Star Wars · director · date of birth⟧ so ⟦ Star Wars⟧": "Q3",
        "⟦ later: Steven Spielberg | George Lucas · date of birth⟧ so ⟦ Steven": "Q2",
    }.items():
        engine = Engine(index, served, SpanSyntax(), max_sequences=1, **EXPRESSIONS)
        comparison, rest = text.split(" so ")
        state = _agree_on_text(oracle, engine, index, served, comparison)
        assert closed_span(index, engine.states_host(), 0).entity_ids == (answer,)
        _agree_on_text(oracle, engine, index, served, " so " + rest, state)


def test_draft_rows_see_both_frontiers_of_a_comparison():
    """apply_mask's draft rows advance a scratch copy of the slot's memory, which must hold
    a comparison's second frontier and its operands' starts."""
    torch = pytest.importorskip("torch")
    kg, dates, served = compare_kg()
    index = build_kg_index(kg, served, SpanSyntax(), dates=dates)
    oracle = Oracle(index, served, SpanSyntax(), **EXPRESSIONS)
    engine = Engine(index, served, SpanSyntax(), max_sequences=1, **EXPRESSIONS)
    slots = torch.tensor([0], dtype=torch.int32, device="cuda")
    engine.begin(slots)
    committed = served.encode(" A ⟦ earlier: Alien | Duel")
    draft = served.encode(" · publication date⟧ so ⟦ Duel⟧")
    history = torch.zeros((1, 64), dtype=torch.int32, device="cuda")
    history[0, : len(committed)] = torch.tensor(committed, dtype=torch.int32)
    total = torch.tensor([len(committed)], dtype=torch.int32, device="cuda")
    engine.sync(slots, history, torch.zeros(1, dtype=torch.int32, device="cuda"), total)
    rows = list(range(len(draft) + 1))
    inputs = [committed[-1], *draft]
    logits = torch.zeros((len(rows), served.vocab_size), device="cuda")
    tensor = lambda v: torch.tensor(v, dtype=torch.int32, device="cuda")  # noqa: E731
    engine.apply_mask(
        logits, tensor([0] * len(rows)), local_pos=tensor(rows), input_ids=tensor(inputs)
    )
    state = oracle.initial_state()
    for token in committed:
        state = oracle.advance(state, token)
    for k in rows:
        allowed = torch.zeros(served.vocab_size, dtype=torch.bool)
        allowed[sorted(oracle.allowed(state))] = True
        assert torch.isneginf(logits[k].cpu()[~allowed]).all(), k
        assert (logits[k].cpu()[allowed] == 0).all(), k
        if k < len(draft):
            state = oracle.advance(state, draft[k])


def test_engine_refuses_a_value_comparison_of_two_kinds_and_unbound_verdicts():
    kg, dates, served = compare_kg()
    index = build_kg_index(kg, served, SpanSyntax(), dates=dates, verdicts=True)
    oracle = Oracle(index, served, SpanSyntax(), **EXPRESSIONS)
    yes, same = served.encode(" yes")[0], served.encode(" same")[0]
    for text in ("⟦ same: Jaws | Alien · publication date", "⟦", "⟦ same:"):
        engine = Engine(index, served, SpanSyntax(), max_sequences=1, **EXPRESSIONS)
        state = _agree_on_text(oracle, engine, index, served, text)
        assert yes not in oracle.allowed(state)  # no unbound subject names a verdict
    plain = build_kg_index(kg, served, SpanSyntax(), dates=dates)  # no verdicts: no same:
    oracle = Oracle(plain, served, SpanSyntax(), **EXPRESSIONS)
    engine = Engine(plain, served, SpanSyntax(), max_sequences=1, **EXPRESSIONS)
    state = _agree_on_text(oracle, engine, plain, served, "⟦")
    assert same not in oracle.allowed(state)


@pytest.mark.parametrize("opened", [False, True])
def test_engine_matches_the_oracle_with_residual_goal_stop(opened):
    seen = run_agreement(
        ServedTokenizer(byte_level_tokenizer([])),
        STOP_KG,
        SpanSyntax(),
        sequences=12,
        steps=600,
        seed=31,
        paths=True,
        committed_paths=True,
        versions=STOP_VERSIONS,
        opened=opened,
        stop="residual-goal",
        questions=(
            "Which country borders the country of Seed?",
            "Which country has the capital of Seed?",
            "Where is Seed?",
        ),
    )
    assert seen["resolved"] > 0 and seen["versioned steps"] > 0, seen


@pytest.mark.parametrize("opened", [False, True])
@pytest.mark.parametrize("start, relations, question, version, anchor, expected", STOP_CASES)
def test_engine_residual_goal_stop_on_fixed_paths(
    start, relations, question, version, anchor, expected, opened
):
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    index = build_kg_index(STOP_KG, served, syntax)
    oracle = Oracle(
        index,
        served,
        syntax,
        paths=True,
        committed_paths=True,
        stop="residual-goal",
        versions=STOP_VERSIONS,
    )
    engine = Engine(
        index,
        served,
        syntax,
        1,
        paths=True,
        committed_paths=True,
        stop="residual-goal",
        versions=build_kg_versions(index, STOP_VERSIONS),
    )
    engine.set_stop(0, question)
    engine.set_version(0, version)
    if anchor:
        engine.anchor(0, anchor)
        state = oracle.anchored_state(anchor, version, question=question)
    else:
        state = oracle.initial_state(version, question=question)
    if opened:
        engine.open_span(0)
        state = oracle.opened_state(state)
    prefix = "" if opened else syntax.opener
    kept = relations[:expected] if expected else relations
    text = prefix + syntax.label_prefix + " · ".join([start, *kept])
    state = _agree_on_text(oracle, engine, engine.view(version), served, text, state)
    _agree_on_text(oracle, engine, engine.view(version), served, syntax.closer, state)


@pytest.mark.parametrize("release", [False, True])
def test_residual_goal_stop_draft_rows_and_slot_reuse(toy_kg, toy_tokenizer, release):
    torch = pytest.importorskip("torch")
    served, syntax = ServedTokenizer(toy_tokenizer), SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    oracle = Oracle(index, served, syntax, paths=True, committed_paths=True, stop="residual-goal")
    engine = Engine(
        index,
        served,
        syntax,
        3,
        paths=True,
        committed_paths=True,
        stop="residual-goal",
        release_top=release,
        escape_bias=-2.0,
    )
    tensor = lambda v: torch.tensor(v, dtype=torch.int32, device="cuda")  # noqa: E731
    engine.begin(tensor([0, 2]))
    questions = {0: "Which employer has Marie Curie?", 2: "Which location has Marie Curie?"}
    for slot, question in questions.items():
        engine.set_stop(slot, question)
    committed = served.encode("⟦ Marie Curie")
    history = torch.zeros((3, 64), dtype=torch.int32, device="cuda")
    history[0, : len(committed)] = history[2, : len(committed)] = tensor(committed)
    engine.sync(
        tensor([0, 2]), history, tensor([0, 0, 0]), tensor([len(committed), 0, len(committed)])
    )
    drafts = served.encode(" · employer⟧ so ⟦ University of Paris⟧")
    # Reversed slot order, and drafts that cross a STOP, its closer and bound answer.
    rows = [slot for slot in (2, 0) for _ in range(len(drafts) + 1)]
    local = list(range(len(drafts) + 1)) * 2
    inputs = [committed[-1], *drafts] * 2
    logits = torch.zeros((len(rows), served.vocab_size), device="cuda")
    expected_masks = []
    for i, (slot, k) in enumerate(zip(rows, local, strict=True)):
        state = oracle.initial_state(question=questions[slot])
        for token in [*committed, *drafts[:k]]:
            state = oracle.advance(state, token)
        expected_masks.append(oracle.allowed(state))
        # Release should leave ordinary rows alone, and cannot override a forced close
        # even when the model's top token there is disallowed.
        is_stop = slot == 0 and state.typed == tuple(served.encode(" employer"))
        logits[i, 0 if is_stop else min(expected_masks[-1])] = 100.0
    before = engine.states_host()
    engine.apply_mask(logits, tensor(rows), local_pos=tensor(local), input_ids=tensor(inputs))
    for i, expected in enumerate(expected_masks):
        assert set(torch.nonzero(~torch.isneginf(logits[i])).flatten().tolist()) == expected
    assert all(np.array_equal(before[f], engine.states_host()[f]) for f in before)
    engine.begin(tensor([0]))  # the next request inherits neither the question nor its used bits
    engine.set_stop(0, "Which location has Marie Curie?")
    engine.open_span(0)
    _agree_on_text(
        oracle,
        engine,
        index,
        served,
        " Marie Curie · employer · located in⟧",
        oracle.opened_state(oracle.initial_state(question="Which location has Marie Curie?")),
    )


def test_residual_goal_stop_overrides_a_pending_escape_under_release():
    torch = pytest.importorskip("torch")
    served = ServedTokenizer(byte_level_tokenizer([]))
    syntax = SpanSyntax(opener="[", label_prefix="", closer="]", separator="/", escape="ry?")
    kg = KnowledgeGraph(
        entities=(Entity("Q1", ("Seed",)), Entity("Q2", ("End",))),
        relations=(Relation("P1", ("count",)), Relation("P2", ("country",))),
        triples=(("Q1", "P1", "Q2"), ("Q1", "P2", "Q2"), ("Q2", "P1", "Q1")),
    )
    index = build_kg_index(kg, served, syntax)
    oracle = Oracle(index, served, syntax, paths=True, committed_paths=True, stop="residual-goal")
    engine = Engine(
        index,
        served,
        syntax,
        1,
        paths=True,
        committed_paths=True,
        stop="residual-goal",
        release_top=True,
    )
    slots = torch.tensor([0], dtype=torch.int32, device="cuda")
    engine.begin(slots)
    engine.set_stop(0, "Which country has Seed?")
    state = oracle.initial_state(question="Which country has Seed?")
    for token in served.encode("[Seed/country"):
        state = oracle.advance(state, token)
        engine.advance(np.array([token], dtype=np.int32))
    assert state.escaping == b"ry"  # began at count, beside the longer label
    logits = torch.zeros((1, served.vocab_size), device="cuda")
    engine.apply_mask(logits, slots)  # top token is disallowed; STOP still has priority
    assert set(torch.nonzero(~torch.isneginf(logits[0])).flatten().tolist()) == oracle.allowed(
        state
    )


@pytest.mark.parametrize("word_count", [33, 64])
def test_residual_goal_stop_carries_high_word_bits(word_count):
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    words = [f"item{i:03}" for i in range(word_count - 1)]
    start = " ".join(words)
    kg = KnowledgeGraph(
        entities=(Entity("Q1", (start,)), Entity("Q2", ("End",)), Entity("Q3", ("Tail",))),
        relations=(
            Relation("P1", ("country",)),
            Relation("P2", ("zulu", "border country", *words)),
        ),
        triples=(("Q1", "P1", "Q2"), ("Q2", "P2", "Q3"), ("Q3", "P1", "Q1")),
    )
    index = build_kg_index(kg, served, syntax)
    oracle = Oracle(index, served, syntax, paths=True, committed_paths=True, stop="residual-goal")
    engine = Engine(
        index, served, syntax, 1, paths=True, committed_paths=True, stop="residual-goal"
    )
    question = f"Which country has the zulu of {start}?"
    engine.set_stop(0, question)
    # The start removes all low bits, leaving the highest word, zulu. It blocks the
    # first close rule; following it consumes the bit and its country alias names the head.
    state = _agree_on_text(
        oracle,
        engine,
        index,
        served,
        f"⟦ {start} · country · zulu",
        oracle.initial_state(question=question),
    )
    _agree_on_text(oracle, engine, index, served, syntax.closer, state)


@pytest.mark.parametrize("text, question", [
    ("⟦ same: Alien | Hair · country of origin⟧ so ⟦ yes⟧", "Which country has Alien?"),
    ("⟦ earlier: Alien | Hair · publication date⟧ so ⟦ Alien⟧", "Which date has Alien?"),
])  # fmt: skip
def test_residual_goal_stop_preserves_comparisons_in_engine(text, question):
    kg, dates, served = compare_kg()
    syntax = SpanSyntax()
    index = build_kg_index(kg, served, syntax, dates=dates, verdicts=True)
    oracle = Oracle(index, served, syntax, **EXPRESSIONS, stop="residual-goal")
    engine = Engine(index, served, syntax, 1, **EXPRESSIONS, stop="residual-goal")
    engine.set_stop(0, question)
    _agree_on_text(oracle, engine, index, served, text, oracle.initial_state(question=question))


@pytest.mark.parametrize("paths", [False, True])
@pytest.mark.parametrize("anchor", [[], ["UNKNOWN"], ["BAD"]])
def test_anchor_without_writable_rows_matches_oracle_rejection(paths, anchor):
    from tests.test_oracle import anchor_world

    index, served, syntax, *_ = anchor_world()
    options = dict(derivation=True, paths=paths, committed_paths=paths, escapes=False)
    oracle = Oracle(index, served, syntax, **options)
    engine = Engine(index, served, syntax, max_sequences=1, **options)
    with pytest.raises(ValueError, match=r"anchor.*writable"):
        oracle.anchored_state(anchor)
    with pytest.raises(ValueError, match=r"anchor.*writable"):
        engine.anchor(0, anchor)
    # A rejected admission must not corrupt a later valid binding.
    engine.anchor(0, ["A"])
    _agree_on_text(oracle, engine, index, served, "⟦ A000⟧", oracle.anchored_state(["A"]))


@pytest.mark.parametrize("paths", [False, True])
@pytest.mark.parametrize("aliases", [40, 65])
def test_duplicate_anchor_ids_match_oracle_at_native_row_cap(paths, aliases):
    from tests.test_oracle import anchor_world

    index, served, syntax, *_ = anchor_world(aliases)
    options = dict(derivation=True, paths=paths, committed_paths=paths, escapes=False)
    oracle = Oracle(index, served, syntax, **options)
    engine = Engine(index, served, syntax, max_sequences=1, **options)
    anchor = ["A", "A", "B", "UNKNOWN"]
    engine.anchor(0, anchor)
    state = oracle.anchored_state(anchor)
    text = "⟦ Z⟧" if aliases == 40 else "⟦ A000⟧"
    state = _agree_on_text(oracle, engine, index, served, text, state)
    assert state.last_span.entity_ids == (("B",) if aliases == 40 else ("A",))


@pytest.mark.parametrize("paths", [False, True])
@pytest.mark.parametrize("deleted", [False, True])
def test_rdflib_versions_match_oracle_with_string_anchors_and_pair_deletion(paths, deleted):
    from tests.test_oracle import rdf_version_world

    torch = pytest.importorskip("torch")
    index, served, syntax, edits, start = rdf_version_world(deleted)
    options = dict(paths=paths, committed_paths=paths, derivation=True, escapes=False)
    oracle = Oracle(index, served, syntax, versions=edits, **options)
    engine = Engine(
        index, served, syntax, max_sequences=1, versions=build_kg_versions(index, edits), **options
    )
    slots = torch.tensor([0], dtype=torch.int32, device="cuda")
    for version, endpoint in ((None, "Old"), ("changed", "New"), (None, "Old")):
        engine.begin(slots)
        engine.set_version(0, version)
        engine.anchor(0, [start])
        state = oracle.anchored_state([start], version)
        text = (
            "⟦ Start⟧"
            if deleted and version
            else ("⟦ Start · route⟧" if paths else f"⟦ Start · route · {endpoint}⟧")
        )
        state = _agree_on_text(oracle, engine, index, served, text, state)
        if not (deleted and version):
            target = f"https://example.org/{endpoint.lower()}"
            assert target in (state.last_span.entity_ids if paths else state.last_span.triples[0])


def test_native_empty_anchor_rows_are_rejected():
    from tests.test_oracle import anchor_world

    index, served, syntax, *_ = anchor_world()
    engine = Engine(index, served, syntax, max_sequences=1, paths=True, escapes=False)
    with pytest.raises(ValueError, match=r"anchor.*writable"):
        engine._native.set_reached(0, np.array([], dtype=np.int32), 0)


@pytest.mark.parametrize("trained", [False, True])
def test_unbound_verdict_prefix_cannot_close(trained, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer if trained else byte_level_tokenizer([]))
    syntax = SpanSyntax()
    index = build_kg_index(
        KnowledgeGraph((Entity("application", ("yesman",)),), (), ()),
        served,
        syntax,
        verdicts=True,
    )
    oracle = Oracle(index, served, syntax)
    engine = Engine(index, served, syntax, 1)
    state = _agree_on_text(oracle, engine, index, served, "⟦ yes")
    engine.fill_mask()
    assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(state)
    assert served.encode(syntax.closer)[0] not in oracle.allowed(state)


@pytest.mark.parametrize("trained", [False, True])
@pytest.mark.parametrize("version", [None, "edit"])
def test_unbound_shared_verdict_label_excludes_reserved_owner(trained, version, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer if trained else byte_level_tokenizer([]))
    syntax = SpanSyntax()
    index = build_kg_index(
        KnowledgeGraph((Entity("application", ("yes",)),), (), ()),
        served,
        syntax,
        verdicts=True,
    )
    edits = {"edit": []}
    oracle = Oracle(index, served, syntax, versions=edits)
    engine = Engine(index, served, syntax, 1, versions=build_kg_versions(index, edits))
    engine.set_version(0, version)
    state = _agree_on_text(oracle, engine, index, served, "⟦ yes⟧", oracle.initial_state(version))
    assert (
        closed_span(engine.view(version), engine.states_host(), 0).entity_ids
        == (state.last_span.entity_ids)
        == ("application",)
    )


def test_verdict_closures_with_served_tokenizers():
    """Owner supplies cached served tokenizer files; no model loading or downloads."""
    from pathlib import Path

    paths = os.environ.get("XG_NATIVE_TOKENIZER_PATHS", "").split(os.pathsep)
    if paths == [""]:
        pytest.skip("owner must supply XG_NATIVE_TOKENIZER_PATHS for served-tokenizer agreement")
    for path in paths:
        assert Path(path).is_file(), path
        served, syntax = ServedTokenizer.from_file(path), SpanSyntax()
        for label in ("yes", "yes man"):
            index = build_kg_index(
                KnowledgeGraph((Entity("application", (label,)),), (), ()),
                served,
                syntax,
                verdicts=True,
            )
            oracle = Oracle(index, served, syntax)
            engine = Engine(index, served, syntax, 1)
            state = oracle.initial_state()
            row = next(
                r for r in range(index.labels.num_rows) if index.labels.owner_id(r) == "application"
            )
            tokens = served.encode(syntax.opener) + list(index.labels.row_tokens(row))
            for token in tokens:
                engine.fill_mask()
                assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(state)
                state = oracle.advance(state, token)
                engine.advance(np.array([token], dtype=np.int32))
            _agree_on_text(oracle, engine, index, served, syntax.closer, state)
            assert closed_span(index, engine.states_host(), 0).entity_ids == ("application",)


@pytest.mark.parametrize("good", [False, True])
def test_mixed_anchor_with_trailing_unwritable_owner_matches_oracle(good):
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    entities = ((Entity("good", ("Good",)),) if good else ()) + (Entity("bad", ("Bad⟧",)),)
    index = build_kg_index(KnowledgeGraph(entities, (), ()), served, syntax)
    options = dict(derivation=True, escapes=False)
    oracle = Oracle(index, served, syntax, **options)
    engine = Engine(index, served, syntax, 1, **options)
    with pytest.raises(ValueError, match=r"anchor.*writable"):
        engine.anchor(0, ["bad"])
    with pytest.raises(ValueError, match=r"anchor.*writable"):
        oracle.anchored_state(["bad"])
    if good:
        anchor = ["good", "bad", "good", "UNKNOWN"]
        engine.anchor(0, anchor)
        _agree_on_text(oracle, engine, index, served, "⟦ Good⟧", oracle.anchored_state(anchor))


def test_native_anchor_slot_rejects_negative_index():
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    index = build_kg_index(KnowledgeGraph((Entity("good", ("Good",)),), (), ()), served, syntax)
    options = dict(derivation=True, escapes=False)
    oracle = Oracle(index, served, syntax, **options)
    engine = Engine(index, served, syntax, 2, **options)
    rows = np.array([0], dtype=np.int32)
    for slot in (-1, -2, engine.max_sequences, np.iinfo(np.int32).max):
        with pytest.raises(ValueError, match="slot"):
            engine.anchor(int(slot), ["good"])
        with pytest.raises(ValueError, match="slot"):
            engine._native.set_reached(int(slot), rows, 0)
    for slot in (0, engine.max_sequences - 1):
        engine.anchor(slot, ["good"])
    state = oracle.anchored_state(["good"])
    for token in served.encode("⟦ Good⟧"):
        engine.fill_mask()
        assert all(allowed_from_mask(mask) == oracle.allowed(state) for mask in engine.mask_host())
        state = oracle.advance(state, token)
        engine.advance(np.array([token, token], dtype=np.int32))
    assert all(
        closed_span(index, engine.states_host(), slot).entity_ids == ("good",) for slot in (0, 1)
    )


def _check_escape_bias_logits(index, served, syntax, biased, options, bias):
    torch = pytest.importorskip("torch")
    oracle = Oracle(index, served, syntax, **options)
    engine = Engine(index, served, syntax, 1, escape_bias=bias, **options)
    engine.begin(torch.tensor([0], dtype=torch.int32, device="cuda"))
    state = oracle.initial_state()
    for token in served.encode(syntax.opener):
        engine.fill_mask()
        assert allowed_from_mask(engine.mask_host()[0]) == oracle.allowed(state)
        state = oracle.advance(state, token)
        engine.advance(np.array([token], dtype=np.int32))
    allowed = oracle.allowed(state)
    engine.fill_mask()
    assert allowed_from_mask(engine.mask_host()[0]) == allowed
    # Grow, reuse and grow again; cross the bias kernel's block boundary. Padding
    # makes the logits stride differ from the vocabulary and bias-token strides.
    for rows in (1, 3, 2, 129):
        bases = [(1.25, 3.5, -1.75)[row % 3] for row in range(rows)]
        storage = torch.tensor([[base] * (served.vocab_size + 3) for base in bases], device="cuda")
        logits = storage[:, : served.vocab_size]
        engine.apply_mask(logits, torch.zeros(rows, dtype=torch.int32, device="cuda"))
        expected = np.array(
            [
                [
                    (base + bias if token in biased else base) if token in allowed else -np.inf
                    for token in range(served.vocab_size)
                ]
                for base in bases
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(logits.cpu().numpy(), expected)
        np.testing.assert_array_equal(
            storage[:, served.vocab_size :].cpu().numpy(),
            np.array([[base] * 3 for base in bases], dtype=np.float32),
        )


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("bias", [-3.0, -0.5])
def test_escape_bias_covers_more_than_eight_prefix_tokens(shared, bias):
    from tests.test_engine_cpu import escape_bias_world

    index, served, syntax, biased = escape_bias_world(shared)
    _check_escape_bias_logits(index, served, syntax, biased, {}, bias)


def test_escape_bias_excludes_shared_operator_prefix_tokens():
    from tests.test_engine_cpu import operator_escape_bias_world

    index, served, syntax, biased, options = operator_escape_bias_world()
    _check_escape_bias_logits(index, served, syntax, biased, options, -0.5)
