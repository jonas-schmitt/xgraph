"""Span language seam: which spans the oracle lets the model write and close."""

from __future__ import annotations

import random
from collections import defaultdict
from contextlib import suppress
from dataclasses import replace
from itertools import pairwise

import pytest

from tests.conftest import TOY_RELATIONS, TOY_TRIPLES, byte_level_tokenizer, train_byte_level_bpe
from tests.test_label_index import KEPT
from xgraph.index import IndexMismatch, build_kg_index
from xgraph.kg import Entity, KnowledgeGraph, Relation
from xgraph.oracle import MAX_PATH_RELATIONS, MAX_REACHED, Oracle, Phase, TokenNotAllowed
from xgraph.syntax import SpanSyntax
from xgraph.tokenizer import ServedTokenizer

# The round-26 isolation probe: base and hybrid-source each reach Cedar, but
# their edge union reaches Delta too. Rebuilt references are literal graphs,
# independent of the production edit compiler and the oracle's edit handling.
ISOLATION_KG = KnowledgeGraph(
    entities=(
        Entity("Q1", ("Atlas", "Book")),
        Entity("Q2", ("Ada",)),
        Entity("Q3", ("Bert",)),
        Entity("Q4", ("Cedar",)),
        Entity("Q5", ("Delta",)),
    ),
    relations=(Relation("P1", ("creator",)), Relation("P2", ("born",))),
    triples=(("Q1", "P1", "Q2"), ("Q2", "P2", "Q4")),
)
ISOLATION_EDITS = {
    "inner": [("Q1", "P1", "Q3"), ("Q3", "P2", "Q5")],
    "terminal": [("Q2", "P2", "Q5")],
    "deleted": [("Q2", "P2", None)],
    "hybrid-source": [("Q1", "P1", "Q3"), ("Q3", "P2", "Q4"), ("Q2", "P2", "Q5")],
}
ISOLATION_GRAPHS = {
    None: ISOLATION_KG.triples,
    "inner": (("Q1", "P1", "Q3"), ("Q2", "P2", "Q4"), ("Q3", "P2", "Q5")),
    "terminal": (("Q1", "P1", "Q2"), ("Q2", "P2", "Q5")),
    "deleted": (("Q1", "P1", "Q2"),),
    "hybrid-source": (("Q1", "P1", "Q3"), ("Q2", "P2", "Q5"), ("Q3", "P2", "Q4")),
}
ISOLATION_ANSWERS = {None: "Cedar", "inner": "Delta", "terminal": "Delta",
                     "deleted": None, "hybrid-source": "Cedar"}  # fmt: skip


@pytest.mark.parametrize("mode", ["derivation", "paths", "committed"])
@pytest.mark.parametrize("anchored", [False, True])
@pytest.mark.parametrize("bpe", [False, True])
def test_version_isolation_masks_and_transitions_equal_rebuilt_indexes(mode, anchored, bpe):
    from xgraph.kgversions import build_kg_versions

    words = [label for item in (*ISOLATION_KG.entities, *ISOLATION_KG.relations)
             for label in item.labels]  # fmt: skip
    token = train_byte_level_bpe(words, 280) if bpe else byte_level_tokenizer([])
    served, syntax = ServedTokenizer(token), SpanSyntax()
    index = build_kg_index(ISOLATION_KG, served, syntax)
    options = {**VERSION_MODES[mode], "escapes": False}
    oracle = Oracle(index, served, syntax, versions=ISOLATION_EDITS, **options)
    compiled = build_kg_versions(index, ISOLATION_EDITS)
    for version, triples in ISOLATION_GRAPHS.items():
        rebuilt_index = build_kg_index(replace(ISOLATION_KG, triples=triples), served, syntax)
        rebuilt = Oracle(rebuilt_index, served, syntax, **options)
        view = compiled.view(index, version)
        for entity in ISOLATION_KG.entities:
            assert view.relations_of(entity.id) == rebuilt_index.relations_of(entity.id)
            for relation in ISOLATION_KG.relations:
                assert view.object_rows(entity.id, relation.id) == rebuilt_index.object_rows(
                    entity.id, relation.id
                )
        if mode == "derivation":
            traces = [
                (f"⟦ {s} · {r} · {o}⟧", (sid, rid, oid) in triples)
                for s, sid in (("Atlas", "Q1"), ("Book", "Q1"), ("Ada", "Q2"), ("Bert", "Q3"))
                for r, rid in (("creator", "P1"), ("born", "P2"))
                for o, oid in (("Ada", "Q2"), ("Bert", "Q3"), ("Cedar", "Q4"), ("Delta", "Q5"))
                if not anchored or sid == "Q1"
            ]
        else:
            traces = [
                (
                    f"⟦ {start} · creator · born⟧ Answer: ⟦ {answer}⟧",
                    answer == ISOLATION_ANSWERS[version],
                )
                for start in ("Atlas", "Book")
                for answer in ("Cedar", "Delta")
            ]
            first = "Bert" if version in ("inner", "hybrid-source") else "Ada"
            traces += [(f"⟦ Atlas · creator⟧ Answer: ⟦ {first}⟧", True)]
        for text, accepted in traces:
            ours = (
                oracle.anchored_state(["Q1"], version)
                if anchored
                else oracle.initial_state(version)
            )
            theirs = rebuilt.anchored_state(["Q1"]) if anchored else rebuilt.initial_state()
            completed = True
            for token_id in served.encode(text):
                assert replace(ours, version=None) == theirs, (version, text)
                allowed = rebuilt.allowed(theirs)
                assert oracle.allowed(ours) == allowed, (version, text)
                if token_id not in allowed:
                    with pytest.raises(TokenNotAllowed):
                        oracle.advance(ours, token_id)
                    with pytest.raises(TokenNotAllowed):
                        rebuilt.advance(theirs, token_id)
                    completed = False
                    break
                ours, theirs = oracle.advance(ours, token_id), rebuilt.advance(theirs, token_id)
            assert replace(ours, version=None) == theirs
            assert oracle.allowed(ours) == rebuilt.allowed(theirs)
            assert completed == accepted, (version, text)


@pytest.mark.parametrize("committed", [False, True])
def test_version_isolation_rejects_the_union_only_answer(committed):
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    union_triples = tuple(
        sorted(set(ISOLATION_GRAPHS[None]) | set(ISOLATION_GRAPHS["hybrid-source"]))
    )
    options = dict(paths=True, committed_paths=committed, escapes=False)
    union = Oracle(build_kg_index(replace(ISOLATION_KG, triples=union_triples), served, syntax),
                   served, syntax, **options)  # fmt: skip
    text = "⟦ Atlas · creator · born⟧ Answer: ⟦ Delta⟧"
    assert walk_from(union, union.initial_state(), served, text).last_span.entity_ids == ("Q5",)
    pinned = Oracle(build_kg_index(ISOLATION_KG, served, syntax), served, syntax,
                    versions=ISOLATION_EDITS, **options)  # fmt: skip
    for version in (None, "hybrid-source"):
        with pytest.raises(TokenNotAllowed):
            walk_from(pinned, pinned.initial_state(version), served, text)


@pytest.mark.parametrize("paths", [False, True])
def test_version_isolation_pair_deletion_removes_every_object(paths):
    from xgraph.kgversions import build_kg_versions

    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    kg = replace(ISOLATION_KG, triples=(*ISOLATION_KG.triples, ("Q2", "P2", "Q5")))
    index = build_kg_index(kg, served, syntax)
    edits = {"deleted": ISOLATION_EDITS["deleted"]}
    versions = build_kg_versions(index, edits)
    assert versions.stats[0].unwritable_edits == 0
    view = versions.view(index, "deleted")
    assert len(index.object_rows("Q2", "P2")) == 2
    assert view.object_rows("Q2", "P2") == []
    assert view.relations_of("Q2") == []
    oracle = Oracle(index, served, syntax, versions=edits, paths=paths, escapes=False)
    rebuilt = Oracle(build_kg_index(replace(kg, triples=ISOLATION_GRAPHS["deleted"]),
                                    served, syntax), served, syntax,
                     paths=paths, escapes=False)  # fmt: skip
    for suffix in ("⟧", " · born⟧" if paths else " · born · Cedar⟧"):
        ours, theirs = oracle.initial_state("deleted"), rebuilt.initial_state()
        completed = True
        for token in served.encode("⟦ Ada" + suffix):
            legal = rebuilt.allowed(theirs)
            assert oracle.allowed(ours) == legal
            assert replace(ours, version=None) == theirs
            if token not in legal:
                with pytest.raises(TokenNotAllowed):
                    oracle.advance(ours, token)
                completed = False
                break
            ours, theirs = oracle.advance(ours, token), rebuilt.advance(theirs, token)
        assert replace(ours, version=None) == theirs
        assert oracle.allowed(ours) == rebuilt.allowed(theirs)
        assert completed == (suffix == "⟧")


@pytest.mark.parametrize("name", [None, "", 1])
def test_version_isolation_refuses_invalid_names_before_replacing_the_base(name):
    from xgraph.kgversions import build_kg_versions

    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    index = build_kg_index(ISOLATION_KG, served, syntax)
    edits = {name: ISOLATION_EDITS["inner"]}
    # None is reserved for the base reader. Accepting this mapping silently
    # overwrites its graph; other invalid names disagree with engine admission.
    with pytest.raises(ValueError, match="KG version needs a non-empty name"):
        Oracle(index, served, syntax, versions=edits, paths=True, escapes=False)
    with pytest.raises(ValueError, match="KG version needs a non-empty name"):
        build_kg_versions(index, edits)


def rdf_version_world(deleted):
    """Independent tiny RDF world shared by CPU and native agreement cases."""
    from rdflib import Graph, Namespace

    from xgraph.rdf import from_rdflib, rdflib_edits

    graph = Graph().parse(
        data="""
        @prefix ex: <https://example.org/> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        ex:start rdfs:label "Start" ; ex:route ex:old .
        ex:old rdfs:label "Old" . ex:new rdfs:label "New" .
        ex:route rdfs:label "route" .
    """,
        format="turtle",
    )
    ex = Namespace("https://example.org/")
    edits = rdflib_edits(
        graph,
        removed=[(ex.start, ex.route, ex.old)],
        added=[] if deleted else [(ex.start, ex.route, ex.new)],
    )
    served = ServedTokenizer(byte_level_tokenizer([]))
    syntax = SpanSyntax()
    index = build_kg_index(from_rdflib(graph), served, syntax)
    return index, served, syntax, {"changed": edits}, str(ex.start)


@pytest.mark.parametrize("paths", [False, True])
@pytest.mark.parametrize("deleted", [False, True])
def test_rdflib_versions_bind_anchors_after_replacement_and_pair_deletion(paths, deleted):
    index, served, syntax, edits, start = rdf_version_world(deleted)
    oracle = Oracle(
        index,
        served,
        syntax,
        versions=edits,
        paths=paths,
        committed_paths=paths,
        derivation=True,
        escapes=False,
    )
    for version, endpoint in ((None, "Old"), ("changed", "New")):
        state = oracle.anchored_state([start], version)
        if deleted and version:
            state = walk_from(oracle, state, served, "⟦ Start")
            assert served.encode(syntax.separator)[0] not in oracle.allowed(state)
            continue
        text = "⟦ Start · route⟧" if paths else f"⟦ Start · route · {endpoint}⟧"
        state = walk_from(oracle, state, served, text)
        target = f"https://example.org/{endpoint.lower()}"
        if paths:
            assert state.last_span.entity_ids == (target,)
        else:
            assert state.last_span.triples == ((start, "https://example.org/route", target),)


def test_answer_mass_certificate_compares_unseen_answers_failures_and_ties():
    from fractions import Fraction

    from xgraph.certification import strict_winner

    lower = {"answer:Q1": Fraction(3, 5)}
    assert strict_winner(lower, {"answer:Q1": 1, "answer:Q2": Fraction(2, 5)}) == "answer:Q1"
    assert strict_winner(lower, {"answer:Q1": 1, "answer:Q2": Fraction(3, 5)}) is None
    assert strict_winner(lower, {"answer:Q1": 1, "failure:horizon": Fraction(4, 5)}) is None


def mass_world(first=None):
    """A literal two-program distribution, with forced observations and delivery."""
    from fractions import Fraction

    from xgraph.certification import OraclePolicy

    served = ServedTokenizer(byte_level_tokenizer([]))
    syntax = SpanSyntax(
        opener="<", closer=">", separator="|", escape="?", label_prefix="", operators=()
    )
    kg = KnowledgeGraph(
        (Entity("Q1", ("A",)), Entity("Q2", ("B",)), Entity("Q3", ("C",))),
        (Relation("P1", ("r",)), Relation("P2", ("s",))),
        (("Q1", "P1", "Q2"), ("Q1", "P2", "Q3")),
    )
    eos = next(iter(served.special_ids))
    first = Fraction(1, 3) if first is None else first
    programs = {
        (*served.encode("A|→A|r|→B>.\nAnswer: <B>"), eos): first,
        (*served.encode("A|→A|s|→C>.\nAnswer: <C>"), eos): 1 - first,
    }

    def weights(history, allowed):
        return {
            token: sum(
                (
                    p
                    for trace, p in programs.items()
                    if trace[: len(history) + 1] == (*history, token)
                ),
                Fraction(),
            )
            for token in allowed
        }

    return OraclePolicy(build_kg_index(kg, served, syntax), served, syntax, weights, anchor=("Q1",))


def test_answer_mass_search_certifies_executed_id_with_forced_observations():
    from fractions import Fraction

    from xgraph.certification import search

    snapshots = list(search(mass_world()))
    assert snapshots[-1].winner == "answer:Q3"
    assert snapshots[-1].lower["answer:Q3"] == Fraction(2, 3)
    assert snapshots[-1].forced_steps > 0


def test_answer_mass_graph_envelope_can_stop_before_the_global_bound():
    from fractions import Fraction

    from xgraph.certification import OraclePolicy, search

    base = mass_world()
    kg = KnowledgeGraph(
        tuple(Entity(f"Q{n}", (name,)) for n, name in enumerate("ABCD", 1)),
        tuple(Relation(f"P{n}", (label,)) for n, label in enumerate("rst", 1)),
        tuple(("Q1", f"P{n}", f"Q{n + 1}") for n in range(1, 4)),
    )
    eos = next(iter(base.tokenizer.special_ids))
    traces = [
        ((*base.tokenizer.encode(f"A|→A|{r}|→{name}>.\nAnswer: <{name}>"), eos), mass)
        for r, name, mass in zip("rst", "BCD", [40, 35, 25], strict=True)
    ]

    def weights(history, allowed):
        return {
            token: sum(p for trace, p in traces if trace[: len(history) + 1] == (*history, token))
            for token in allowed
        }

    def policy():
        return OraclePolicy(
            build_kg_index(kg, base.tokenizer, base.syntax),
            base.tokenizer,
            base.syntax,
            weights,
            anchor=("Q1",),
        )

    exact, _ = mass_enumeration(policy())
    assert exact == {
        "answer:Q2": Fraction(2, 5),
        "answer:Q3": Fraction(7, 20),
        "answer:Q4": Fraction(1, 4),
    }
    reachable = list(search(policy()))[-1]
    global_bound = list(search(policy(), bound="global"))[-1]
    assert reachable.winner == global_bound.winner == "answer:Q2"
    assert reachable.forward_steps < global_bound.forward_steps
    # Failures remain competitors even in this favorable, three-program example.
    assert reachable.upper["failure:horizon"] < reachable.lower["answer:Q2"]


def mass_enumeration(policy):
    """Independent DFS and probability arithmetic; no search/bound/outcome helpers."""
    from fractions import Fraction

    totals, leaves = defaultdict(Fraction), []
    stack = [((), policy.initial, Fraction(1))]
    while stack:
        history, state, probability = stack.pop()
        outcome = None
        if history and history[-1] in policy.tokenizer.special_ids:
            if state.answer_complete and state.last_span and len(state.last_span.entity_ids) == 1:
                outcome = f"answer:{state.last_span.entity_ids[0]}"
            else:
                outcome = "failure:early_eos" if not state.answer_complete else "failure:ambiguous"
        elif state.phase is Phase.FREE and state.answer_pos == 0 and not state.answer_complete:
            outcome = "failure:no_path"
        elif len(history) == policy.horizon:
            outcome = "failure:horizon"
        if outcome is None:
            allowed = policy.oracle.allowed(state)
            if len(allowed) == 1:
                transitions = {next(iter(allowed)): Fraction(1)}
            elif allowed:
                raw = {t: Fraction(w) for t, w in policy.weights(history, allowed).items()}
                denominator = sum(raw.values(), Fraction())
                transitions = (
                    {t: w / denominator for t, w in raw.items() if w} if denominator else {}
                )
            else:
                transitions = {}
            if not transitions:
                outcome = "failure:zero_weight" if allowed else "failure:empty"
        if outcome is not None:
            totals[outcome] += probability
            leaves.append((history, outcome, probability))
            continue
        for token, conditional in transitions.items():
            stack.append(
                ((*history, token), policy.oracle.advance(state, token), probability * conditional)
            )
    return dict(totals), leaves


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("bits", [None, 4, 12])
def test_answer_mass_certificates_enclose_exhaustive_seeded_worlds(monkeypatch, seed, bits):
    from fractions import Fraction

    from xgraph.certification import OraclePolicy, search

    rng = random.Random(seed)
    monkeypatch.setattr("xgraph.oracle.MAX_REACHED", 1 + seed % 3)
    monkeypatch.setattr("xgraph.oracle.MAX_PATH_RELATIONS", 1 + seed % 2)
    served = ServedTokenizer(byte_level_tokenizer([]))
    syntax = SpanSyntax(
        opener="<", closer=">", separator="|", escape="?", label_prefix="", operators=()
    )
    kg = KnowledgeGraph(
        tuple(Entity(f"Q{n}", (name, name.lower())) for n, name in enumerate("ABBC")),
        (Relation("P0", ("r", "t")), Relation("P1", ("s", "t"))),
        tuple(
            (f"Q{s}", f"P{r}", f"Q{o}")
            for s in range(4)
            for r in range(2)
            for o in range(4)
            if rng.random() < 0.2
        ),
    )

    def weights(history, allowed):
        # Integer arithmetic is stable across Python versions and independent of RNG order.
        return {t: 1 + (sum(history) * 7 + t * 11 + seed) % 9 for t in allowed}

    policy = OraclePolicy(
        build_kg_index(kg, served, syntax),
        served,
        syntax,
        weights,
        horizon=(0, 12, 50)[seed % 3],
        anchor=("Q0", "Q1"),
        versions={"v": (("Q0", "P0", "Q3"), ("Q1", "P1", "unwritable"))},
        version="v" if seed % 2 else None,
    )
    exact, leaves = mass_enumeration(policy)
    assert sum(exact.values()) == 1
    for kind in ("reachable", "global"):
        for snapshot in search(policy, bound=kind, rounding_bits=bits, max_expansions=2000):
            for outcome in policy.universe:
                assert snapshot.lower.get(outcome, 0) <= exact.get(outcome, 0)
                assert exact.get(outcome, 0) <= snapshot.upper[outcome]
            histories = sorted(node.history for node in snapshot.pending)
            assert all(b[: len(a)] != a for a, b in pairwise(histories))
            for node in snapshot.pending:
                descendants = [
                    leaf for leaf in leaves if leaf[0][: len(node.history)] == node.history
                ]
                assert node.low <= sum((leaf[2] for leaf in descendants), Fraction()) <= node.high
                assert {leaf[1] for leaf in descendants} <= node.reachable
            if snapshot.winner:
                assert all(
                    exact[snapshot.winner] > p
                    for other, p in exact.items()
                    if other != snapshot.winner
                )


def test_answer_mass_rounding_never_turns_a_tie_or_near_tie_into_a_certificate():
    from fractions import Fraction

    from xgraph.certification import search

    assert list(search(mass_world(Fraction(1, 2))))[-1].winner is None
    near_tie = Fraction(1, 2) + Fraction(1, 2**64)
    assert list(search(mass_world(near_tie)))[-1].winner == "answer:Q2"
    assert list(search(mass_world(near_tie), rounding_bits=12))[-1].winner is None


def test_answer_mass_reachability_never_recaps_a_union(monkeypatch):
    from fractions import Fraction

    from xgraph.certification import OraclePolicy

    monkeypatch.setattr("xgraph.oracle.MAX_REACHED", 1)
    base = mass_world()
    policy = OraclePolicy(base.index, base.tokenizer, base.syntax, base.weights, anchor=("Q1",))
    envelope = policy.reachable(policy.initial)
    # Each of two independently capped one-hop trajectories reaches a different ID.
    assert {"answer:Q2", "answer:Q3"} <= envelope
    assert len(envelope - {x for x in envelope if x.startswith("failure:")}) > policy.cap
    exact, _ = mass_enumeration(policy)
    assert exact == {"answer:Q2": Fraction(1, 3), "answer:Q3": Fraction(2, 3)}


def test_answer_mass_envelope_uses_the_chosen_current_relation(monkeypatch):
    monkeypatch.setattr("xgraph.oracle.MAX_REACHED", 1)
    policy = mass_world()
    state = policy.initial
    for token in policy.tokenizer.encode("A|→A|"):
        state = policy.oracle.advance(state, token)
    # The empty current relation has two separately capped continuations.
    assert {"answer:Q2", "answer:Q3"} <= policy.reachable(state)
    for token in policy.tokenizer.encode("r|→B"):
        state = policy.oracle.advance(state, token)
    # Its old subject set is still Q1; the observed current relation already reaches Q2.
    assert state.subjects == frozenset({"Q1"})
    assert "answer:Q2" in policy.reachable(state)
    assert "answer:Q3" not in policy.reachable(state)
    assert "answer:Q1" not in policy.reachable(state)


@pytest.mark.parametrize("failure", ["horizon", "empty", "zero_weight"])
def test_answer_mass_failures_retain_their_probability(failure):
    from xgraph.certification import OraclePolicy, search

    base = mass_world()
    # F08 rejects an invalid explicit anchor during preparation. An unanchored
    # empty index supplies a real empty policy support without weakening F08.
    index = (
        build_kg_index(KnowledgeGraph((), (), ()), base.tokenizer, base.syntax)
        if failure == "empty"
        else base.index
    )
    policy = OraclePolicy(
        index,
        base.tokenizer,
        base.syntax,
        lambda history, allowed: dict.fromkeys(allowed, 0),
        anchor=() if failure == "empty" else ("Q1",),
        horizon=0 if failure == "horizon" else 64,
    )
    exact, _ = mass_enumeration(policy)
    assert exact == {f"failure:{failure}": 1}
    result = list(search(policy))[-1]
    assert result.winner == f"failure:{failure}"
    assert result.lower[result.winner] == 1


def test_answer_mass_budget_preserves_unexpanded_mass_and_rejects_cap_mutation(monkeypatch):
    from xgraph.certification import search

    policy = mass_world()
    result = list(search(policy, max_expansions=1))[-1]
    assert result.winner is None and result.pending
    assert sum(n.high for n in result.pending) + sum(result.lower.values()) == 1
    monkeypatch.setattr("xgraph.oracle.MAX_REACHED", policy.cap + 1)
    with pytest.raises(ValueError, match="caps changed"):
        list(search(policy))


@pytest.mark.parametrize("bad", ["missing_tail", "negative", "nan", "infinite"])
def test_answer_mass_refuses_invalid_or_incomplete_distributions(bad):
    from xgraph.certification import OraclePolicy, search

    base = mass_world()

    def weights(history, allowed):
        result = dict.fromkeys(allowed, 1)
        token = min(allowed)
        if bad == "missing_tail":
            del result[token]
        else:
            result[token] = {"negative": -1, "nan": float("nan"), "infinite": float("inf")}[bad]
        return result

    policy = OraclePolicy(base.index, base.tokenizer, base.syntax, weights, anchor=("Q1",))
    with pytest.raises(ValueError, match=r"every legal token|nonnegative|finite"):
        list(search(policy))


def case_alias_world():
    """A shared case variant whose less popular sense has more outgoing edges."""
    kg = KnowledgeGraph(
        (
            Entity("Q1", ("Sual",)),
            Entity("Q2", ("sual",)),
            Entity("Q3", ("Old endpoint",)),
            Entity("Q4", ("New endpoint",)),
        ),
        (Relation("P1", ("via",)),),
        (("Q1", "P1", "Q3"), ("Q1", "P1", "Q1"), ("Q2", "P1", "Q4")),
    )
    return kg, ServedTokenizer(byte_level_tokenizer(MERGES)), SpanSyntax()


def test_identity_surface_normalization_reaches_internal_case_variants():
    from xgraph.identity import IdentityOracle

    kg = KnowledgeGraph((Entity("Q1", ("Town hall of Lurate Caccivio",)),))
    oracle = IdentityOracle(kg)
    assert oracle.candidates("Town Hall of Lurate Caccivio") == frozenset({"Q1"})
    assert oracle.candidates("TOWN HALL OF LURATE CACCIVIO") == frozenset({"Q1"})


def test_identity_binding_resolves_shared_labels_without_popularity_fallback():
    from xgraph.identity import IdentityOracle

    kg, _, _ = case_alias_world()
    oracle = IdentityOracle(kg)
    assert oracle.candidates("SUAL") == frozenset({"Q1", "Q2"})
    root = oracle.bind("SUAL", "Q2")
    assert root.qid == "Q2"
    assert oracle.execute(root, ("P1",)).endpoints == frozenset({"Q4"})
    with pytest.raises(ValueError, match="candidate"):
        oracle.bind("SUAL", "Q3")
    with pytest.raises(ValueError, match="candidate"):
        oracle.bind("missing", "Q2")


def test_identity_binding_pins_version_component_and_anchor():
    from xgraph.identity import IdentityOracle

    kg, _, _ = case_alias_world()
    oracle = IdentityOracle(kg, versions={"edit": (("Q2", "P1", "Q3"),)})
    base = oracle.bind("sual", "Q2")
    edited = oracle.bind("sual", "Q2", version="edit")
    assert oracle.execute(base, ("P1",)).endpoints == frozenset({"Q4"})
    assert oracle.execute(edited, ("P1",)).endpoints == frozenset({"Q3"})
    assert oracle.execute(base, ("P1",)).endpoints == frozenset({"Q4"})
    assert oracle.candidates("Sual", component={"Q2", "Q4"}) == frozenset({"Q2"})
    with pytest.raises(ValueError, match="candidate"):
        oracle.bind("sual", "Q1", anchor={"Q2"})
    with pytest.raises(ValueError, match="version"):
        oracle.bind("sual", "Q2", version="absent")
    scoped = oracle.bind("sual", "Q2", component={"Q2"})
    assert not oracle.execute(scoped, ("P1",)).endpoints
    # An unwritable edit still replaces the old pair; no fallback to another root.
    erased = IdentityOracle(kg, versions={"edit": (("Q2", "P1", "Q999"),)})
    assert not erased.execute(erased.bind("sual", "Q2", version="edit"), ("P1",)).endpoints


def test_identity_binding_snapshot_does_not_borrow_mutable_version_edits():
    from xgraph.identity import IdentityOracle

    kg, _, _ = case_alias_world()
    edits = [["Q2", "P1", "Q3"]]
    oracle = IdentityOracle(kg, versions={"edit": edits})
    bound = oracle.bind("sual", "Q2", version="edit")
    edits[0][2] = "Q4"
    assert oracle.execute(bound, ("P1",)).endpoints == frozenset({"Q3"})


def test_identity_root_protocol_is_utf8_and_qid_exact():
    from xgraph.identity import IdentityOracle, root_prefix

    kg = KnowledgeGraph((Entity("Q1", ("Café",)), Entity("Q10", ("Café",))))
    oracle = IdentityOracle(kg)
    wire = root_prefix("Cafe\u0301", "Q10").encode()
    for chunks in ([wire], [wire[i : i + 1] for i in range(len(wire))]):
        assert oracle.read_root(chunks).qid == "Q10"
    for text in ["⟦ Café⟧", "⟦ Café ⟦# Q01⟧", "⟦ Café ⟦# Q100⟧", "⟦ Café ⟦# Q1⟧junk"]:
        with pytest.raises(ValueError):
            oracle.read_root([text.encode()])
    with pytest.raises(ValueError):
        oracle.read_root([b"\xff"])


def test_identity_qid_mask_refuses_an_oversized_root_before_offering_ids():
    from xgraph.identity import MAX_ROOT_BYTES, IdentityOracle

    # One byte fits for Q1 but not Q10. Refuse the reference rather than pruning
    # the longer ID out of a complete identity set because of the root-byte cap.
    surface = "A" * (MAX_ROOT_BYTES - len("⟦  ⟦# Q1⟧".encode()))
    oracle = IdentityOracle(KnowledgeGraph((Entity("Q1", (surface,)), Entity("Q10", (surface,)))))
    assert oracle.candidates(surface) == frozenset({"Q1", "Q10"})
    with pytest.raises(ValueError, match="byte limit"):
        oracle.binding_suffixes(surface)
    with pytest.raises(ValueError, match="byte limit"):
        oracle.bind(surface, "Q1")


def test_identity_qid_mask_preserves_prefix_collisions_and_multibyte_closer():
    from xgraph.identity import IdentityOracle

    kg = KnowledgeGraph((Entity("Q1", ("Sual",)), Entity("Q10", ("sual",))))
    oracle = IdentityOracle(kg)
    choices = oracle.binding_suffixes("SUAL")
    assert choices == ("Q1⟧".encode(), "Q10⟧".encode())
    vocab = {i: bytes([i]) for i in range(256)} | {256: "1⟧".encode(), 257: b"10", 258: b""}
    # Independently enumerate the two literal accepted byte strings at every prefix.
    for accepted in ("Q1⟧".encode(), "Q10⟧".encode()):
        for end in range(len(accepted)):
            prefix = accepted[:end]
            expected = {
                token
                for token, piece in vocab.items()
                if piece
                and any(
                    w.startswith(prefix + piece) for w in (b"Q1\xe2\x9f\xa7", b"Q10\xe2\x9f\xa7")
                )
            }
            assert oracle.binding_mask("SUAL", prefix, vocab) == expected
    assert not oracle.binding_mask("SUAL", "Q1⟧".encode(), vocab)


@pytest.mark.parametrize(
    "surface,expected",
    [
        ("Cafe\u0301", {"Q1"}),
        ("CAFé", {"Q1"}),
        ("CAFÉ", set()),  # Unicode case folding is not in nfc-ascii-v1.
        ("Cafe", set()),
        ("Café ", set()),
        ("I", {"Q2", "Q3"}),
        ("İ", {"Q4"}),
        ("\uff26oo", set()),
        ("Foo", {"Q5"}),
        ("Straße", {"Q6"}),
        ("STRASSE", set()),
        ("O\u2019Neal", {"Q7"}),
        ("O'Neal", set()),
    ],
)
def test_identity_normalization_classes_have_declared_boundaries(surface, expected):
    from xgraph.identity import IdentityOracle

    labels = (("Café",), ("I",), ("i", "i"), ("İ",), ("Foo",), ("Straße",), ("O\u2019Neal",))
    kg = KnowledgeGraph(tuple(Entity(f"Q{i}", ls) for i, ls in enumerate(labels, 1)))
    assert IdentityOracle(kg).candidates(surface) == frozenset(expected)


def test_identity_alias_dedup_exclusions_overflow_and_root_provenance():
    from xgraph.identity import IdentityOracle, normalize_surface

    kg = KnowledgeGraph(
        (
            Entity("Q1", ("Name", "name", "Name")),
            Entity("Q2", ("name",)),
            Entity("Q3", ("  ", "Bad⟧label", "<eos>")),
        )
    )
    oracle = IdentityOracle(kg, special_tokens=("<eos>",), max_candidates=1)
    assert oracle.candidates("NAME") == frozenset({"Q1", "Q2"})
    with pytest.raises(ValueError, match="overflow"):
        oracle.bind("NAME", "Q1")
    assert not oracle.candidates("<eos>")
    assert not oracle.candidates("Bad⟧label")
    assert not oracle.candidates("  ")
    for s in ("H\u0331", "ẖ", "Cafe\u0301", "İ", "TOWN HALL"):
        key = normalize_surface(s)
        assert normalize_surface(key) == key
    exact = IdentityOracle(kg, normalization="exact-v1")
    assert not exact.candidates("NAME")
    assert exact.candidates("Name") == frozenset({"Q1"})
    root = exact.bind("Name", "Q1")
    with pytest.raises(ValueError, match="fingerprint"):
        oracle.execute(root, ())


def test_identity_surface_and_binding_extend_aliases_but_factor_through_an_anchor():
    from xgraph.identity import IdentityOracle

    kg = KnowledgeGraph(
        (Entity("Q1", ("Town hall of Vale",)), Entity("Q2", ("Endpoint",))),
        (Relation("P1", ("via",)),),
        (("Q1", "P1", "Q2"),),
    )
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    alias_index = build_kg_index(kg, served, syntax, initial_case_aliases=True)
    alias = Oracle(alias_index, served, syntax, paths=True, committed_paths=True, escapes=False)
    with pytest.raises(TokenNotAllowed):
        walk(alias, served.encode("⟦ Town Hall of Vale · via⟧"))
    identity = IdentityOracle(kg)
    root = identity.bind("Town Hall of Vale", "Q1")
    assert identity.execute(root, ("P1",)).endpoints == frozenset({"Q2"})

    kg, served, syntax = case_alias_world()
    index = build_kg_index(kg, served, syntax, initial_case_aliases=True, sitelinks={"Q1": 100})
    identity = IdentityOracle(kg)
    for rank in ("degree", "sitelinks"):
        legacy = Oracle(
            index, served, syntax, paths=True, committed_paths=True, escapes=False, start_rank=rank
        )
        text = "⟦ sual · via⟧"
        assert "Q4" not in walk(legacy, served.encode(text)).last_span.entity_ids
        anchored = walk_from(legacy, legacy.anchored_state(["Q2"]), served, text)
        chosen = identity.execute(identity.bind("SUAL", "Q2"), ("P1",))
        assert chosen.endpoints == frozenset(anchored.last_span.entity_ids) == frozenset({"Q4"})
        assert chosen.witnesses == (("Q2", "P1", "Q4"),)


def test_initial_case_alias_bulk_spans_keep_homonym_commitment_and_bound_answers(tmp_path):
    import json

    import numpy as np

    from xgraph.index import load_kg_index
    from xgraph.wikidata import build_index

    kg, served, syntax = case_alias_world()
    (tmp_path / "entity_labels.jsonl").write_text(
        "".join(json.dumps([e.id, e.labels[0], "label"]) + "\n" for e in kg.entities)
    )
    (tmp_path / "relation_labels.jsonl").write_text('["P1", "via", "label"]\n')
    np.save(tmp_path / "triples.npy", np.array([[1, 1, 3], [1, 1, 1], [2, 1, 4]], np.int32))
    np.save(tmp_path / "sitelink_counts.npy", np.array([[1, 1], [2, 10]], np.int32))
    build_index(tmp_path, served, syntax, tmp_path / "index", initial_case_aliases=True)
    index = load_kg_index(tmp_path / "index")
    for label in ("Sual", "sual"):
        plain = Oracle(index, served, syntax, escapes=False)
        assert walk(plain, served.encode(f"⟦ {label}⟧")).last_span.entity_ids == ("Q1", "Q2")
        for rank, expected in (("sitelinks", "Q4"), ("degree", "Q3")):
            oracle = Oracle(
                index,
                served,
                syntax,
                paths=True,
                committed_paths=True,
                escapes=False,
                start_rank=rank,
            )
            prefix = f"⟦ {label} · via⟧"
            assert expected in walk(oracle, served.encode(prefix)).last_span.entity_ids
            if rank == "sitelinks":
                for answer in ("New endpoint", "new endpoint"):
                    state = walk(oracle, served.encode(prefix + f" Answer: ⟦ {answer}⟧"))
                    assert state.last_span.entity_ids == ("Q4",)
                with pytest.raises(TokenNotAllowed):
                    walk(oracle, served.encode(prefix + " Answer: ⟦ old endpoint⟧"))
        with pytest.raises(TokenNotAllowed):
            walk(plain, served.encode("⟦ NEW endpoint⟧"))


@pytest.mark.parametrize("part,kind", [("labels", "entity"), ("relations", "relation")])
@pytest.mark.parametrize(
    "rows,valid",
    [
        ((), True),
        (((10,),), True),
        (((9,), (10,)), True),
        (((10,), (10,), (10, 11)), True),
        (((10,), (9,)), False),
        (((10, 11), (10,)), False),
        (((10, 11), (10, 9)), False),
    ],
)
def test_oracle_checks_label_row_order_as_token_tuples(part, kind, rows, valid):
    import numpy as np

    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    kg = KnowledgeGraph(
        (Entity("Q1", ("A",)), Entity("Q2", ("B",))),
        (Relation("P1", ("first",)), Relation("P2", ("second",))),
    )
    index = build_kg_index(kg, served, syntax)
    table = replace(
        getattr(index, part),
        tokens=np.asarray([token for row in rows for token in row], dtype=np.int32),
        offsets=np.asarray([0, *np.cumsum([len(row) for row in rows])], dtype=np.int64),
        # Equal token tuples may have owners in either order.
        owners=np.asarray([1, 0, 1][: len(rows)], dtype=np.int32),
    )
    changed = replace(index, **{part: table})
    if valid:
        Oracle(changed, served, syntax)
    else:
        with pytest.raises(IndexMismatch, match=rf"{kind} label rows.*token tuples.*rows 0 and 1"):
            Oracle(changed, served, syntax)


# Byte-level alphabet: E2 = "â", 9F = "Ł", A6 = "¦", A7 = "§", A9 = "©", 81 = "ģ", 84 = "Ħ",
# 87 = "ĩ". So ⟦ = "âŁ¦", ⟧ = "âŁ§", ⟩ = "âŁ©" (left unmerged), ⁄ = "âģĦ", and the
# escape marker ⁇ = "âģĩ".
MERGES = [
    ("â", "Ł"),
    ("â", "ģ"),  # E2 81: starts both ⁄ and the escape marker ⁇
    ("âŁ", "¦"),
    ("âŁ", "§"),
    ("x", "âŁ¦"),  # x⟦: the opener ends with the token
    ("âŁ¦", "y"),  # ⟦y: the opener completes before the token ends
    ("âŁ§", "y"),  # ⟧y: the closer completes before the token ends
    ("B", "r"),
    ("Br", "a"),
]


HAND_ENTITIES = (Entity("Q1", ("Bra",)), Entity("Q2", ("Bra⟩",)), Entity("Q3", ("Bra⁄",)))
# The mechanism tests write labels and the escape marker without the default
# spaces, which keeps their hand-worked token sequences short.
HAND_SYNTAX = SpanSyntax(label_prefix="", escape="⁇")


@pytest.fixture
def hand():
    """A hand-built tokenizer and an oracle over the labels "Bra", "Bra⟩", and "Bra⁄".

    "Bra⟩" tokenizes as Bra, âŁ, ©, so after the complete label "Bra" the token
    âŁ (E2 9F) both continues a label and starts the closer ⟧ (E2 9F A7).
    "Bra⁄" tokenizes as Bra, âģ, Ħ, so the token âģ (E2 81) both continues a
    label and starts the escape marker ⁇ (E2 81 87).
    """
    tokenizer = byte_level_tokenizer(MERGES)
    served = ServedTokenizer(tokenizer)
    kg = KnowledgeGraph(entities=HAND_ENTITIES)
    oracle = Oracle(build_kg_index(kg, served, HAND_SYNTAX), served, HAND_SYNTAX)

    def ids(*pieces: str) -> list[int]:
        return [tokenizer.token_to_id(piece) for piece in pieces]

    return oracle, ids, tokenizer.token_to_id("<|endoftext|>")


def walk(oracle: Oracle, tokens: list[int]):
    state = oracle.initial_state()
    for token in tokens:
        state = oracle.advance(state, token)
    return state


def variable_stride_cases(kg, executor, seed, version):
    """Expanded actions, with syntax and free text encoded at separate boundaries."""
    from xgraph.stride import action_menu

    rng = random.Random(seed)
    names = {e.id: e.labels[0] for e in kg.entities}
    plans = [["spouse", "spouse"], ["spouse"] * 8]
    for _ in range(12):
        prefix = []
        while len(prefix) < 8:
            menu = action_menu(executor, "Marie Curie", prefix, "spouse employer", version=version)
            if not menu:
                break
            pairs = [a for a in menu if len(a) == 2]
            prefix.extend(rng.choice(pairs or menu))
            if rng.random() < 0.4:
                break
        if prefix:
            plans.append(prefix)
    for prefix in plans:
        result = executor.execute_set("Marie Curie", prefix, version=version, witnesses=True)
        assert result["reached"]
        for name in sorted({names[e] for e in result["reached"]}):
            path = "⟦ " + " · ".join(["Marie Curie", *prefix]) + "⟧"
            # Encoding the concatenated text can merge the closer with the dot.
            # Such a token is correctly disallowed by both oracle and engine.
            yield [path, ".\nAnswer: ", f"⟦ {name}⟧"], result


@pytest.mark.parametrize("seed", [2402, 2403])
@pytest.mark.parametrize("version", [None, "edit"])
def test_variable_stride_expanded_seeded_walks_match_executor(toy_kg, toy_tokenizer, seed, version):
    from xgraph.kgversions import build_kg_versions
    from xgraph.programs import MatchedExecutor

    served, syntax = ServedTokenizer(toy_tokenizer), SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax, sitelinks=TOY_SITELINKS)
    edits = {"edit": [("Q1", "P108", "Q2")]}
    oracle = Oracle(
        index,
        served,
        syntax,
        versions=edits,
        paths=True,
        committed_paths=True,
        escapes=False,
        start_rank="sitelinks",
    )
    executor = MatchedExecutor(
        index, served, versions=build_kg_versions(index, edits), start_rank="sitelinks"
    )
    for pieces, result in variable_stride_cases(toy_kg, executor, seed, version):
        state = oracle.initial_state(version)
        for i, piece in enumerate(pieces):
            for token in served.encode(piece):
                state = oracle.advance(state, token)
            if i == 0:
                assert set(state.last_span.entity_ids) == set(result["reached"])
        assert set(state.last_span.entity_ids) <= set(result["witnesses"])


def observed_world():
    """A hand-ranked branch: Gamma (degree 2), Beta (1), Delta (0)."""
    served = ServedTokenizer(byte_level_tokenizer([]))
    syntax = SpanSyntax()
    kg = KnowledgeGraph(
        entities=tuple(
            Entity(e, (label,))
            for e, label in (("Q1", "Start"), ("Q2", "Beta"), ("Q3", "Gamma"), ("Q4", "Delta"))
        ),
        relations=(Relation("P1", ("via",)), Relation("P2", ("next",))),
        triples=(
            ("Q1", "P1", "Q2"),
            ("Q1", "P1", "Q3"),
            ("Q1", "P1", "Q4"),
            ("Q2", "P2", "Q4"),
            ("Q3", "P2", "Q2"),
            ("Q3", "P2", "Q4"),
        ),
    )
    return kg, served, syntax


@pytest.mark.parametrize("width,display", [(2, "Gamma (+2 more)"), (8, "Gamma ; Beta ; Delta")])
def test_adaptive_width_preserves_full_answer_support(width, display):
    kg, served, syntax = observed_world()
    oracle = Oracle(
        build_kg_index(kg, served, syntax),
        served,
        syntax,
        paths=True,
        escapes=False,
        observe=1,
        observe_width=width,
    )
    state = walk(oracle, served.encode(f"⟦ Start ·→ Start · via ·→ {display}⟧"))
    assert state.last_span.entity_ids == ("Q2", "Q3", "Q4")


@pytest.mark.parametrize(
    "extra,message",
    [
        ({"observe_width": -1}, "observe_width"),
        ({"observe_width": 65}, "observe_width"),
        ({"observe": 0, "observe_width": 8}, "observe_width"),
        ({"observe_defer": 9}, "observe_defer"),
        ({"observe_defer": 2, "observe_required": False}, "observe_defer"),
        ({"menu_close_next": True, "menu": 0}, "menu_close_next"),
    ],
)
def test_menu_v2_invalid_options_fail_explicitly(extra, message):
    kg, served, syntax = observed_world()
    options = dict(
        paths=True,
        observe=1,
        observe_required=True,
        menu=2,
        eos_token_ids=sorted(served.special_ids),
    )
    options.update(extra)
    with pytest.raises(ValueError, match=message):
        Oracle(build_kg_index(kg, served, syntax), served, syntax, **options)


def test_compact_menu_defers_then_requests_next_and_closes_with_bound_answer():
    kg, served, syntax = observed_world()
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
    oracle = Oracle(build_kg_index(kg, served, syntax), served, syntax, **options)
    text = (
        "⟦ Start ·→ Start ⟦? via⟧ · via ·→ Gamma ; Beta ; Delta"
        " ⟦=DEFER⟧ ⟦!NEXT⟧ ⟦? next⟧ · next ·→ Beta ; Delta"
        " ⟦= Delta ·→ Delta ⟦!CLOSE⟧⟧.\nAnswer: ⟦ Delta⟧"
    )
    state = walk(oracle, served.encode(text))
    assert state.last_span.entity_ids == ("Q4",)
    assert oracle.allowed(state) == served.special_ids


def test_defer_limit_is_per_path_and_does_not_shadow_an_entity_named_defer():
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    kg = KnowledgeGraph(
        (Entity("Q1", ("Start",)), Entity("Q2", ("DEFER",))),
        (Relation("P1", ("loop",)),),
        (("Q1", "P1", "Q1"), ("Q1", "P1", "Q2"), ("Q2", "P1", "Q1")),
    )
    oracle = Oracle(
        build_kg_index(kg, served, syntax),
        served,
        syntax,
        paths=True,
        escapes=False,
        observe=2,
        observe_required=True,
        observe_defer=2,
        eos_token_ids=sorted(served.special_ids),
    )
    prefix = "⟦ Start ·→ Start" + " · loop ·→ Start ; DEFER ⟦=DEFER⟧" * 2
    state = walk(oracle, served.encode(prefix + " · loop ·→ Start ; DEFER ⟦="))
    assert served.encode("D")[0] not in oracle.allowed(state)
    assert served.encode(" ")[0] in oracle.allowed(state)
    for token in served.encode(" DEFER ·→ DEFER⟧.\nAnswer: ⟦ DEFER⟧"):
        state = oracle.advance(state, token)
    assert state.last_span.entity_ids == ("Q2",)


def audit_world(bpe=False):
    """Two terminal choices and different append/replace support, with a cycle."""
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    kg = KnowledgeGraph(
        tuple(
            Entity(f"Q{i}", (name,))
            for i, name in enumerate(("Start", "Beta", "Gamma", "End", "Other"), 1)
        ),
        (Relation("P1", ("via",)), Relation("P2", ("next",)), Relation("P3", ("alternate",))),
        (
            ("Q1", "P1", "Q2"),
            ("Q1", "P1", "Q3"),
            ("Q1", "P3", "Q5"),
            ("Q2", "P2", "Q4"),
            ("Q3", "P3", "Q5"),
            ("Q4", "P1", "Q4"),
        ),
    )
    if bpe:
        served = ServedTokenizer(
            train_byte_level_bpe(
                [
                    "\nEndpoints: ⟦ Gamma (+1 more)⟧.\nAnswer: ⟦ Beta⟧",
                    *[name for item in (*kg.entities, *kg.relations) for name in item.labels],
                ],
                384,
            )
        )
    index = build_kg_index(kg, served, syntax, sitelinks={"Q3": 100, "Q2": 10})
    return kg, index, served, syntax


@pytest.mark.parametrize(
    "order,seed,width,display",
    [
        ("identity", 17, 2, "Gamma ; Beta (+1 more)"),
        ("reversed", 17, 2, "Beta ; Gamma (+1 more)"),
        ("random", 17, 2, "Beta ; Gamma (+1 more)"),
        ("random", 18, 8, "Delta ; Gamma ; Beta"),
    ],
)
def test_terminal_display_order_preserves_ranked_subset_and_answer_support(
    order, seed, width, display
):
    kg, served, syntax = observed_world()
    index = build_kg_index(kg, served, syntax, sitelinks={"Q3": 100, "Q2": 10})
    options = dict(
        paths=True, committed_paths=True, escapes=False, observe_terminal=width, eos_token_ids=[0]
    )
    original = Oracle(index, served, syntax, **options)
    oracle = Oracle(index, served, syntax, **options, terminal_order=order, terminal_seed=seed)
    state = walk(oracle, served.encode("⟦ Start · via⟧"))
    assert state.reached == walk(original, served.encode("⟦ Start · via⟧")).reached
    for token in served.encode(f"\nEndpoints: ⟦ {display}⟧.\nAnswer: ⟦"):
        assert oracle.allowed(state) == {token}
        state = oracle.advance(state, token)
    reference = walk(original, served.encode("⟦ Start · via⟧"))
    while len(original.allowed(reference)) == 1:
        reference = original.advance(reference, next(iter(original.allowed(reference))))
    while len(oracle.allowed(state)) == 1:
        state = oracle.advance(state, next(iter(oracle.allowed(state))))
    assert oracle.allowed(state) == original.allowed(reference)
    state = walk(
        oracle, served.encode(f"⟦ Start · via⟧\nEndpoints: ⟦ {display}⟧.\nAnswer: ⟦ Delta⟧")
    )
    assert state.last_span.entity_ids == ("Q4",)


@pytest.mark.parametrize("bpe", [False, True])
def test_terminal_observation_forces_deliverable_display_then_undisplayed_answer(bpe):
    _, index, served, syntax = audit_world(bpe)
    oracle = Oracle(
        index,
        served,
        syntax,
        paths=True,
        committed_paths=True,
        escapes=False,
        observe_terminal=1,
        eos_token_ids=sorted(served.special_ids),
    )
    state = walk(oracle, served.encode("⟦ Start · via⟧"))
    reached = state.reached
    for token in terminal_display_tokens(served):
        assert oracle.allowed(state) == {token}
        state = oracle.advance(state, token)
        assert state.reached == reached
    for token in served.encode(" Beta"):
        state = oracle.advance(state, token)
    assert served.encode(" ·")[0] not in oracle.allowed(state)
    for token in served.encode("⟧"):
        state = oracle.advance(state, token)
    assert state.last_span.entity_ids == ("Q2",)
    assert oracle.allowed(state) == served.special_ids


@pytest.mark.parametrize(
    "extra",
    [
        {"terminal_order": "popularity"},
        {"terminal_seed": -1},
        {"terminal_seed": 2**32},
        {"terminal_seed": True},
        {"observe_terminal": 0, "terminal_order": "reversed"},
    ],
)
def test_terminal_order_rejects_invalid_configuration(extra):
    _, index, served, syntax = audit_world()
    options = dict(paths=True, observe_terminal=8, eos_token_ids=[0])
    options.update(extra)
    with pytest.raises(ValueError, match=r"terminal_(order|seed)"):
        Oracle(index, served, syntax, **options)


def terminal_display_tokens(served):
    """Canonical forced display: punctuation and label rows keep their boundaries."""
    pieces = ["\nEndpoints: ⟦", " Gamma", " (+1 more)", "⟧", ".\nAnswer: ", "⟦"]
    return [t for piece in pieces for t in served.encode(piece)]


@pytest.mark.parametrize("aliases", [False, True])
def test_terminal_display_follows_start_commitment_and_answer_row_cap(aliases):
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
    oracle = Oracle(
        index,
        served,
        syntax,
        paths=True,
        committed_paths=True,
        start_rank="sitelinks",
        observe_terminal=8,
        eos_token_ids=[0],
    )
    state = walk(oracle, served.encode("⟦ Start · via⟧"))
    forced = "" if aliases else "\nEndpoints: ⟦ Beta ; Alpha⟧"
    for token in served.encode(forced + ".\nAnswer: ⟦"):
        assert oracle.allowed(state) == {token}
        state = oracle.advance(state, token)
    # Other-start support is never displayed; an alias-heavy capped singleton
    # bypasses the display even though the uncapped final frontier is ambiguous.
    assert {index.labels.owner_id(r) for r in state.reached} == (
        {"Q3"} if aliases else {"Q3", "Q4"}
    )


@pytest.mark.parametrize(
    "options",
    [
        {"observe_terminal": -1},
        {"observe_terminal": 65},
        {"observe_terminal": True},
        {"observe_terminal": 8, "paths": False},
        {"observe_terminal": 8, "relations": "vocabulary"},
        {"observe_terminal": 8, "observe": 1},
        {"observe_terminal": 8, "menu": 1},
        {"observe_terminal": 8, "audit": True},
    ],
)
def test_terminal_admission_refuses_incompatible_options(options):
    _, index, served, syntax = audit_world()
    with pytest.raises(ValueError, match="observe_terminal"):
        Oracle(index, served, syntax, **{"paths": True, "eos_token_ids": [0], **options})


def relation_display_world(bpe=False):
    """Canonical labels need not sort first; an excluded one needs a scored alias."""
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    kg = KnowledgeGraph(
        (Entity("Q1", ("Start",)),),
        (
            Relation("P1", ("country", "as of")),
            Relation("P2", ("capital", "administrative centre")),
            Relation("P3", ("unsafe⟦", "a unrelated", "z capital", "b capital")),
        ),
        tuple(("Q1", f"P{i}", "Q1") for i in range(1, 4)),
    )
    if bpe:
        served = ServedTokenizer(
            train_byte_level_bpe(
                [name for item in (*kg.entities, *kg.relations) for name in item.labels], 384
            )
        )
    return kg, build_kg_index(kg, served, syntax, sitelinks={}), served, syntax


@pytest.mark.parametrize("audit", [False, True])
@pytest.mark.parametrize("bpe", [False, True])
def test_relation_display_prefers_canonical_then_best_writable_alias(audit, bpe):
    _, index, served, syntax = relation_display_world(bpe)
    options = {"audit": True, "eos_token_ids": sorted(served.special_ids)} if audit else {"menu": 3}
    oracle = Oracle(index, served, syntax, paths=True, escapes=False, **options)
    state = oracle.initial_state(question="Which capital?")
    prefix = "⟦ Start · country⟧" if audit else "⟦ Start"
    for token in served.encode(prefix):
        state = oracle.advance(state, token)
    display = (
        "\nExecuted: ⟦ Start⟧"
        "\nRelations (terminal): ⟦ capital ; b capital ; country⟧"
        "\nRelations (penultimate): ⟦ capital ; b capital ; country⟧\nDecision: ⟦"
        if audit
        else " ⟦? capital ; b capital ; country⟧"
    )
    for token in served.encode(display):
        assert oracle.allowed(state) == {token}
        state = oracle.advance(state, token)


def test_menu_language_forces_only_display_and_preserves_available_choices():
    _, index, served, syntax = audit_world()
    oracle = Oracle(index, served, syntax, paths=True, escapes=False, menu=1)
    state = oracle.initial_state(question="Which alternate?")
    for token in served.encode("⟦ Start"):
        state = oracle.advance(state, token)
    for token in served.encode(" ⟦? alternate (+1 more)⟧"):
        assert oracle.allowed(state) == {token}
        state = oracle.advance(state, token)
    # The displayed relation is a suggestion; the omitted via remains writable.
    for token in served.encode(" · via ⟦? alternate (+1 more)⟧⟧"):
        assert oracle.allowed(state)
        state = oracle.advance(state, token)
    assert state.last_span.entity_ids == ("Q2", "Q3")


@pytest.mark.parametrize(
    "options,match",
    [
        ({"menu": -1}, "menu"),
        ({"menu": 17}, "menu"),
        ({"menu": True}, "menu"),
        ({"menu": 1, "paths": False}, "frontier path"),
        ({"menu": 1, "relations": "vocabulary"}, "frontier path"),
        ({"menu": 1, "expressions": True}, "frontier path"),
        ({"menu": 1, "audit": True}, "audit"),
    ],
)
def test_menu_admission(options, match):
    _, index, served, syntax = audit_world()
    with pytest.raises(ValueError, match=match):
        Oracle(index, served, syntax, **{"paths": True, **options})


def test_audit_keep_forces_readback_then_bound_answer_and_eos():
    _, index, served, syntax = audit_world()
    oracle = Oracle(
        index,
        served,
        syntax,
        paths=True,
        committed_paths=True,
        escapes=False,
        audit=True,
        eos_token_ids=sorted(served.special_ids),
    )
    state = walk(oracle, served.encode("⟦ Start · via⟧"))
    forced = (
        "\nExecuted: ⟦ Gamma ; Beta⟧"
        "\nRelations (terminal): ⟦ next ; alternate⟧"
        "\nRelations (penultimate): ⟦ via ; alternate⟧\nDecision: ⟦"
    )
    for token in served.encode(forced):
        assert oracle.allowed(state) == {token}
        state = oracle.advance(state, token)
    for token in served.encode("KEEP⟧"):
        state = oracle.advance(state, token)
    for token in served.encode(".\nAnswer: ⟦"):
        assert oracle.allowed(state) == {token}
        state = oracle.advance(state, token)
    for token in served.encode(" Beta⟧"):
        state = oracle.advance(state, token)
    assert state.last_span.entity_ids == ("Q2",)
    assert oracle.allowed(state) == served.special_ids


@pytest.mark.parametrize(
    "plan,expected",
    [
        (
            "via",
            {
                "KEEP⟧",
                "SELECT ⟦= Beta⟧",
                "SELECT ⟦= Gamma⟧",
                "APPEND · next⟧",
                "APPEND · alternate⟧",
                "REPLACE · via⟧",
                "REPLACE · alternate⟧",
            },
        ),
        (
            "via · next",
            {
                "KEEP⟧",
                "SELECT ⟦= End⟧",
                "APPEND · via⟧",
                "REPLACE · next⟧",
                "REPLACE · alternate⟧",
                "DROP⟧",
            },
        ),
        ("via · next" + " · via" * 6, {"KEEP⟧", "SELECT ⟦= End⟧", "REPLACE · via⟧", "DROP⟧"}),
    ],
)
def test_audit_decision_language_is_exact_and_has_no_dead_ends(plan, expected):
    _, index, served, syntax = audit_world()
    oracle = Oracle(
        index,
        served,
        syntax,
        paths=True,
        audit=True,
        audit_entities=1,
        audit_relations=0,
        escapes=False,
        eos_token_ids=sorted(served.special_ids),
    )
    state = walk(oracle, served.encode(f"⟦ Start · {plan}⟧"))
    while state.audit_phase == 1:
        allowed = oracle.allowed(state)
        assert len(allowed) == 1
        state = oracle.advance(state, next(iter(allowed)))
    stack, language = [(state, b"")], set()
    while stack:
        state, text = stack.pop()
        if state.answer_pos:
            language.add(text.decode())
            continue
        if state.audit_phase == 5:
            language.add(text.decode())
            continue
        allowed = oracle.allowed(state)
        assert allowed, text
        for token in allowed:
            stack.append((oracle.advance(state, token), text + served.token_bytes(token)))
    assert language == expected


def test_audit_scores_and_versions_rank_menus_without_restricting_support():
    _, index, served, syntax = audit_world()
    oracle = Oracle(
        index,
        served,
        syntax,
        paths=True,
        audit=True,
        audit_relations=1,
        versions={"edit": [("Q2", "P2", "Q5")]},
        eos_token_ids=sorted(served.special_ids),
    )
    state = oracle.initial_state("edit", question="Which alternate?")
    for token in served.encode("⟦ Start · via⟧"):
        state = oracle.advance(state, token)
    text = b""
    while state.audit_phase == 1:
        (token,) = oracle.allowed(state)
        text += served.token_bytes(token)
        state = oracle.advance(state, token)
    expected = (
        "\nExecuted: ⟦ Gamma ; Beta⟧\nRelations (terminal): ⟦ alternate (+1 more)⟧"
        "\nRelations (penultimate): ⟦ alternate (+1 more)⟧\nDecision: ⟦"
    )
    assert text == expected.encode()
    for token in served.encode("APPEND · next⟧\nExecuted: ⟦ Other⟧.\nAnswer: ⟦ Other⟧"):
        state = oracle.advance(state, token)
    assert state.last_span.entity_ids == ("Q5",)


@pytest.mark.parametrize(
    "options,match",
    [
        ({}, "path mode"),
        ({"paths": True, "observe": 1}, "observations off"),
        ({"paths": True, "relations": "vocabulary"}, "frontier"),
        ({"paths": True, "audit_entities": 0}, "audit_entities"),
        ({"paths": True, "audit_relations": 65}, "audit_relations"),
    ],
)
def test_audit_admission_refuses_incompatible_options(options, match):
    _, index, served, syntax = audit_world()
    with pytest.raises(ValueError, match=match):
        Oracle(
            index, served, syntax, audit=True, eos_token_ids=sorted(served.special_ids), **options
        )


def test_audit_requires_eos_sitelinks_and_nonconflicting_punctuation():
    kg, index, served, syntax = audit_world()
    with pytest.raises(ValueError, match="EOS"):
        Oracle(index, served, syntax, paths=True, audit=True)
    with pytest.raises(ValueError, match="sitelink counts"):
        Oracle(
            build_kg_index(kg, served, syntax),
            served,
            syntax,
            paths=True,
            audit=True,
            eos_token_ids=sorted(served.special_ids),
        )
    syntax = SpanSyntax(closer="KEEP")
    with pytest.raises(ValueError, match="audit punctuation"):
        Oracle(
            build_kg_index(kg, served, syntax, sitelinks={}),
            served,
            syntax,
            paths=True,
            audit=True,
            eos_token_ids=sorted(served.special_ids),
        )


def test_observed_v3_required_counts_entities_even_with_one_displayed_shared_label():
    kg, served, syntax = observed_world()
    kg = KnowledgeGraph(
        (*kg.entities, Entity("Q5", ("Beta",))), kg.relations, (*kg.triples, ("Q1", "P1", "Q5"))
    )
    oracle = Oracle(
        build_kg_index(kg, served, syntax),
        served,
        syntax,
        paths=True,
        escapes=False,
        observe=1,
        observe_required=True,
        observe_no_loop=True,
    )
    state = walk(oracle, served.encode("⟦ Start ·→ Start · via ·→ Gamma (+3 more)"))
    for token in served.encode(" ⟦="):
        assert oracle.allowed(state) == {token}
        state = oracle.advance(state, token)
    for token in served.encode(" Beta ·→ Beta (+1 more)⟧"):
        state = oracle.advance(state, token)
    # Retain both label owners. Another identical selection could never terminate.
    assert set(state.last_span.entity_ids) == {"Q2", "Q5"}


@pytest.mark.parametrize("guard", [False, True])
@pytest.mark.parametrize("menu", [0, 1])
def test_observed_v3_required_span_language_is_exact_and_has_no_dead_ends(guard, menu):
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    kg = KnowledgeGraph(
        (Entity("Q1", ("Start",)), Entity("Q2", ("A",)), Entity("Q3", ("B",))),
        (Relation("P1", ("via",)),),
        (("Q1", "P1", "Q2"), ("Q1", "P1", "Q3")),
    )
    oracle = Oracle(
        build_kg_index(kg, served, syntax),
        served,
        syntax,
        paths=True,
        escapes=False,
        observe=1,
        observe_required=True,
        observe_no_loop=guard,
        menu=menu,
    )
    stack = [(oracle.opened_state(oracle.initial_state()), "⟦".encode())]
    closed = set()
    while stack:
        state, text = stack.pop()
        if state.phase is Phase.FREE:
            closed.add(text.decode())
            continue
        allowed = oracle.allowed(state)
        assert allowed, text
        for token in allowed:
            stack.append((oracle.advance(state, token), text + served.token_bytes(token)))
    via, empty = (" ⟦? via⟧", " ⟦? ∅⟧") if menu else ("", "")
    prefixes = ("⟦ Start ·→ Start" + via, "⟦ Start ·→ Start" + via + " ⟦= Start ·→ Start" + via)
    expected = {prefix + "⟧" for prefix in prefixes} | {
        "⟦ A ·→ ∅" + empty + "⟧",
        "⟦ B ·→ ∅" + empty + "⟧",
    }
    expected |= {
        prefix + f" · via ·→ A (+1 more){empty} ⟦= {label} ·→ {label}{empty}⟧"
        for prefix in prefixes
        for label in ("A", "B")
    }
    assert closed == expected


@pytest.mark.parametrize("menu", [0, 1])
def test_observed_v3_guard_span_language_has_no_dead_ends(menu):
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    kg = KnowledgeGraph(
        (Entity("Q1", ("A",)),), (Relation("P1", ("x", "alias")),), (("Q1", "P1", "Q1"),)
    )
    oracle = Oracle(
        build_kg_index(kg, served, syntax),
        served,
        syntax,
        paths=True,
        escapes=False,
        observe=1,
        observe_no_loop=True,
        menu=menu,
    )
    stack = [(oracle.opened_state(oracle.initial_state()), "⟦".encode())]
    closed = set()
    while stack:
        state, text = stack.pop()
        if state.phase is Phase.FREE:
            closed.add(text.decode())
            continue
        allowed = oracle.allowed(state)
        assert allowed, text
        for token in allowed:
            stack.append((oracle.advance(state, token), text + served.token_bytes(token)))
    hint, exhausted = (" ⟦? x⟧", " ⟦? ∅⟧") if menu else ("", "")
    assert closed == {
        "⟦ A ·→ A" + hint + "⟧",
        "⟦ A ·→ A" + hint + " · x ·→ A" + exhausted + "⟧",
        "⟦ A ·→ A" + hint + " · alias ·→ A" + exhausted + "⟧",
    }


def test_observed_selection_span_language_and_forced_answer():
    kg, served, syntax = observed_world()
    oracle = Oracle(
        build_kg_index(kg, served, syntax),
        served,
        syntax,
        paths=True,
        committed_paths=True,
        observe=1,
        observe_select=True,
        observe_answer=True,
        eos_token_ids=[256],
    )
    state = walk(oracle, served.encode("⟦ Start ·→ Start · via ·→ Gamma (+2 more) ⟦="))
    # The entire capped frontier is writable, not just the display; Start is excluded.
    for label in ("Beta", "Gamma", "Delta"):
        trial = state
        for token in served.encode(" " + label + " ·→ " + label):
            trial = oracle.advance(trial, token)
    with pytest.raises(TokenNotAllowed):
        trial = state
        for token in served.encode(" Start"):
            trial = oracle.advance(trial, token)
    for token in served.encode(" Beta ·→ Beta · next ·→ Delta⟧"):
        state = oracle.advance(state, token)
    for token in served.encode(".\nAnswer: ⟦"):
        assert oracle.allowed(state) == {token}
        state = oracle.advance(state, token)
    for token in served.encode(" Delta⟧"):
        state = oracle.advance(state, token)
    assert state.last_span.entity_ids == ("Q4",)
    assert oracle.allowed(state) == {256}


@pytest.mark.parametrize("committed", [False, True])
def test_selection_retains_all_label_owners_and_all_labels_past_answer_row_cap(committed):
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
    oracle = Oracle(
        build_kg_index(kg, served, syntax),
        served,
        syntax,
        paths=True,
        committed_paths=committed,
        escapes=False,
        observe=1,
        observe_select=True,
    )
    state = walk(oracle, served.encode("⟦ Start ·→ Start · via ·→ Gamma (+4 more) ⟦="))
    for label in ("Beta", "Twin", "Alias79", "Delta"):
        trial = state
        for token in served.encode(" " + label):
            trial = oracle.advance(trial, token)
    for token in served.encode(" Beta ·→ Beta (+1 more) · next ·→ Gamma (+1 more)⟧"):
        state = oracle.advance(state, token)
    assert set(state.last_span.entity_ids) == {"Q3", "Q4"}


def test_start_selection_can_close_without_consuming_a_relation():
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    kg = KnowledgeGraph(
        (Entity("Q1", ("Start", "A")), Entity("Q2", ("Start", "B"))),
        (Relation("P1", ("via",)),),
        (("Q1", "P1", "Q1"), ("Q2", "P1", "Q2")),
    )
    oracle = Oracle(
        build_kg_index(kg, served, syntax),
        served,
        syntax,
        paths=True,
        committed_paths=True,
        observe=2,
        observe_select=True,
        observe_answer=True,
        eos_token_ids=[256],
    )
    state = walk(oracle, served.encode("⟦ Start ·→ A ; B ⟦= B ·→ B⟧"))
    assert state.last_span.entity_ids == ("Q2",)
    for token in served.encode(".\nAnswer: ⟦ B⟧"):
        state = oracle.advance(state, token)
    assert oracle.allowed(state) == {256}


def test_observed_path_forces_ranked_frontiers_then_binds_the_same_answer():
    kg, served, syntax = observed_world()
    index = build_kg_index(kg, served, syntax)
    oracle = Oracle(
        index,
        served,
        syntax,
        paths=True,
        committed_paths=True,
        escapes=False,
        observe=2,
        end_after_answer=True,
        eos_token_ids=[256],
    )
    state = oracle.opened_state(oracle.initial_state())

    def write(text, forced=False):
        nonlocal state
        for token in served.encode(text):
            if forced:
                assert oracle.allowed(state) == {token}
            state = oracle.advance(state, token)

    write(" Start")
    write(" ·→ Start", forced=True)
    write(" · via")
    frontier = state.subjects
    write(" ·→ Gamma ; Beta (+1 more)", forced=True)
    assert state.subjects == frontier  # observing executes no extra hop
    write(" · next")
    write(" ·→ Beta ; Delta", forced=True)
    write("⟧\nAnswer: ⟦ Delta⟧")
    assert state.last_span.entity_ids == ("Q4",)
    assert oracle.allowed(state) == {256}


def test_observed_path_zero_is_identical_at_every_step(toy_kg, toy_tokenizer):
    served, syntax = ServedTokenizer(toy_tokenizer), SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    a = Oracle(index, served, syntax, paths=True)
    b = Oracle(index, served, syntax, paths=True, observe=0)
    sa, sb = a.initial_state(), b.initial_state()
    for token in served.encode("⟦ Marie Curie · employer⟧\nAnswer: ⟦ University of Paris⟧"):
        assert a.allowed(sa) == b.allowed(sb)
        sa, sb = a.advance(sa, token), b.advance(sb, token)
        assert sa == sb


@pytest.mark.parametrize("version,frontier", [(None, "Gamma (+2 more)"), ("edit", "Delta")])
def test_observations_use_versioned_frontiers_and_can_escape(version, frontier):
    kg, served, syntax = observed_world()
    index = build_kg_index(kg, served, syntax)
    oracle = Oracle(
        index, served, syntax, paths=True, observe=1, versions={"edit": [("Q1", "P1", "Q4")]}
    )
    state = oracle.opened_state(oracle.initial_state(version))
    for text in (" Start", " ·→", " Start", " ·", " via", " ·→", " " + frontier):
        for token in served.encode(text):
            state = oracle.advance(state, token)
    for token in served.encode(" ⁇ unknown⟧"):
        state = oracle.advance(state, token)
    assert not state.last_span.resolved


def test_dead_start_observes_empty_without_inventing_an_edge():
    kg, served, syntax = observed_world()
    oracle = Oracle(
        build_kg_index(kg, served, syntax), served, syntax, paths=True, escapes=False, observe=3
    )
    state = walk(oracle, served.encode("⟦ Delta"))
    for token in served.encode(" ·→ ∅"):
        assert oracle.allowed(state) == {token}
        state = oracle.advance(state, token)
    with pytest.raises(TokenNotAllowed):
        oracle.advance(state, served.encode(" ·")[0])
    for token in served.encode("⟧"):
        state = oracle.advance(state, token)
    assert state.last_span.entity_ids == ("Q4",)


@pytest.mark.parametrize("committed", [False, True])
def test_observations_count_the_capped_frontier_without_changing_commitment(committed):
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    ends = tuple(Entity(f"Q{n + 10}", (f"N{n:02d}",)) for n in range(70))
    kg = KnowledgeGraph(
        entities=(Entity("Q1", ("Start",)), *ends),
        relations=(Relation("P1", ("via",)),),
        triples=tuple(("Q1", "P1", e.id) for e in ends),
    )
    index = build_kg_index(kg, served, syntax)
    oracle = Oracle(
        index, served, syntax, paths=True, committed_paths=committed, escapes=False, observe=3
    )
    state = walk(oracle, served.encode("⟦ Start ·→ Start · via"))
    for token in served.encode(" ·→ N00 ; N01 ; N02 (+61 more)"):
        assert oracle.allowed(state) == {token}
        state = oracle.advance(state, token)
    for token in served.encode("⟧"):
        state = oracle.advance(state, token)
    assert len(state.last_span.entity_ids) == MAX_REACHED
    assert "Q79" not in state.last_span.entity_ids


def test_observations_precede_stop_and_the_last_permitted_relation():
    kg, served, syntax = observed_world()
    index = build_kg_index(kg, served, syntax)
    oracle = Oracle(
        index,
        served,
        syntax,
        paths=True,
        committed_paths=True,
        escapes=False,
        observe=1,
        stop="residual-goal",
    )
    state = oracle.initial_state(question="Which next does Start reach via?")
    for text in (
        "⟦ Start",
        " ·→ Start",
        " · via",
        " ·→ Gamma (+2 more)",
        " · next",
        " ·→ Beta (+1 more)",
    ):
        for token in served.encode(text):
            state = oracle.advance(state, token)
    assert oracle.allowed(state) == {served.encode("⟧")[0]}

    loop = KnowledgeGraph(
        (Entity("Q1", ("Start",)),), (Relation("P1", ("loop",)),), (("Q1", "P1", "Q1"),)
    )
    oracle = Oracle(
        build_kg_index(loop, served, syntax), served, syntax, paths=True, escapes=False, observe=1
    )
    state = oracle.initial_state()
    for token in served.encode("⟦ Start ·→ Start" + " · loop ·→ Start" * MAX_PATH_RELATIONS):
        state = oracle.advance(state, token)
    assert oracle.allowed(state) == {served.encode("⟧")[0]}


@pytest.mark.parametrize("rank,first", [("degree", "Gamma"), ("sitelinks", "Delta")])
def test_observation_ranking_and_unique_aliases(rank, first):
    from dataclasses import replace

    from xgraph.wikidata import relation_names

    kg, served, syntax = observed_world()
    names = relation_names(
        [
            ("P1", "via", "label"),
            ("P1", "alias", "alias"),
            ("P1", "shared", "alias"),
            ("P2", "next", "label"),
            ("P2", "shared", "alias"),
        ],
        unique_aliases=True,
    )
    assert "shared" not in [label for labels in names.values() for label in labels]
    kg = replace(kg, relations=tuple(Relation(r, tuple(labels)) for r, labels in names.items()))
    index = build_kg_index(kg, served, syntax, sitelinks={"Q4": 100})
    oracle = Oracle(index, served, syntax, paths=True, observe=1, start_rank=rank, escapes=False)
    state = walk(oracle, served.encode("⟦ Start ·→ Start · alias"))
    for token in served.encode(" ·→ " + first + " (+2 more)"):
        assert oracle.allowed(state) == {token}
        state = oracle.advance(state, token)


def test_observation_does_not_consume_longer_labels_or_offer_them_after_it():
    kg, served, syntax = observed_world()
    kg = KnowledgeGraph(
        entities=(*kg.entities, Entity("Q5", ("Start Extended",))),
        relations=(*kg.relations, Relation("P3", ("via extended",))),
        triples=(*kg.triples, ("Q5", "P3", "Q4")),
    )
    oracle = Oracle(
        build_kg_index(kg, served, syntax), served, syntax, paths=True, observe=1, escapes=False
    )
    for start, relation, result in (
        ("Start", "via", "Gamma (+2 more)"),
        ("Start Extended", "via extended", "Delta"),
    ):
        state = oracle.initial_state()
        for text in (
            "⟦",
            " " + start,
            " ·→",
            " " + start,
            " ·",
            " " + relation,
            " ·→",
            " " + result,
            "⟧",
        ):
            for token in served.encode(text):
                state = oracle.advance(state, token)
        assert state.last_span.resolved


@pytest.mark.parametrize(
    "options",
    [
        {"observe": -1},
        {"observe": 65},
        {"observe": True},
        {"observe": 1},
        {"observe": 1, "paths": True, "relations": "vocabulary"},
        {"observe": 1, "paths": True, "relations": "successor"},
        {"observe": 1, "paths": True, "expressions": True},
    ],
)
def test_observation_configuration_fails_explicitly(options):
    kg, served, syntax = observed_world()
    with pytest.raises(ValueError, match="observe"):
        Oracle(build_kg_index(kg, served, syntax), served, syntax, **options)


def test_observed_paths_preserve_literal_arrows_in_indexed_labels():
    from xgraph.programs import record_path

    kg, served, syntax = observed_world()
    kg = KnowledgeGraph(
        (*kg.entities, Entity("Q9", ("X → Y",))), kg.relations, (*kg.triples, ("Q9", "P1", "Q4"))
    )
    index = build_kg_index(kg, served, syntax)
    oracle = Oracle(index, served, syntax, paths=True, observe=1, escapes=False)
    text = "⟦ X → Y ·→ X → Y · via ·→ Delta⟧"
    assert walk(oracle, served.encode(text)).last_span.entity_ids == ("Q4",)
    assert record_path({"text": text}) == ("X → Y", ["via"])


@pytest.mark.parametrize("relations", ["frontier", "vocabulary", "successor"])
@pytest.mark.parametrize("committed", [False, True])
def test_end_after_answer_allows_only_eos_after_the_answer_closes(
    toy_kg, toy_tokenizer, relations, committed
):
    served, syntax = ServedTokenizer(toy_tokenizer), SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    # Distinct EOS ids are supplied by the served model, never guessed from token text.
    eos = (toy_tokenizer.token_to_id("<|endoftext|>"), served.encode("x")[0])
    options = dict(paths=True, escapes=False, committed_paths=committed, relations=relations)
    plain = Oracle(index, served, syntax, **options)
    ending = Oracle(index, served, syntax, **options, end_after_answer=True, eos_token_ids=eos)
    before, state = plain.initial_state(), ending.initial_state()
    chunks = ("⟦ Marie Curie⟧", ". ", "⟦ Marie Curie · spouse⟧", ".\nAnswer: ", "⟦ Pierre Curie⟧")
    for token in [t for chunk in chunks for t in served.encode(chunk)]:
        assert ending.allowed(state) == plain.allowed(before)
        before, state = plain.advance(before, token), ending.advance(state, token)
    assert ending.allowed(state) == frozenset(eos)
    assert ending.allowed(ending.advance(state, eos[0])) == frozenset(eos)
    with pytest.raises(TokenNotAllowed):
        ending.advance(state, served.encode(" Wait")[0])


def test_end_after_answer_does_not_end_after_an_escape(toy_kg, toy_tokenizer):
    served, syntax = ServedTokenizer(toy_tokenizer), SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    options = dict(paths=True, end_after_answer=True, eos_token_ids=[0])
    ending = Oracle(index, served, syntax, **options)
    plain = Oracle(index, served, syntax, paths=True)
    text = "⟦ Marie Curie · spouse⟧\nAnswer: ⟦ ⁇ unknown⟧ More ⟦ Pierre Curie⟧"
    a, b = ending.initial_state(), plain.initial_state()
    for token in served.encode(text):
        assert ending.allowed(a) == plain.allowed(b)
        a, b = ending.advance(a, token), plain.advance(b, token)
    assert ending.allowed(a) == plain.allowed(b)


@pytest.mark.parametrize("ids", [(), (-1,), (1000000,), (True,), ("0",)])
def test_end_after_answer_requires_valid_eos_ids(toy_kg, toy_tokenizer, ids):
    served, syntax = ServedTokenizer(toy_tokenizer), SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    with pytest.raises(ValueError, match="EOS"):
        Oracle(index, served, syntax, paths=True, end_after_answer=True, eos_token_ids=ids)
    # Off means no change, even without an EOS configuration.
    Oracle(index, served, syntax, end_after_answer=False)
    with pytest.raises(ValueError, match="path mode"):
        Oracle(index, served, syntax, end_after_answer=True, eos_token_ids=[0])


@pytest.mark.parametrize("opened", [False, True])
@pytest.mark.parametrize("version", [None, "edit"])
def test_end_after_answer_with_opened_spans_versions_and_stop(
    toy_kg, toy_tokenizer, opened, version
):
    served, syntax = ServedTokenizer(toy_tokenizer), SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    oracle = Oracle(index, served, syntax, paths=True, committed_paths=True, escapes=False,
                    stop="residual-goal", versions={"edit": [("Q1", "P108", "Q2")]},
                    end_after_answer=True, eos_token_ids=[0])  # fmt: skip
    state = oracle.anchored_state(["Q1"], version, question="Which employer has Marie Curie?")
    if opened:
        state = oracle.opened_state(state)
    path = ("" if opened else "⟦") + " Marie Curie · employer⟧"
    answer = "Pierre Curie" if version else "University of Paris"
    for token in served.encode(path) + served.encode(f"\nAnswer: ⟦ {answer}⟧"):
        state = oracle.advance(state, token)
    assert oracle.allowed(state) == {0}


@pytest.mark.parametrize("expression,answer", [
    ("earlier: Jaws | Star Wars · director · date of birth", "Star Wars"),
    ("later: Jaws | Star Wars · publication date", "Star Wars"),
    ("same: Jaws | Star Wars · director", "no"),
])  # fmt: skip
def test_end_after_answer_with_comparisons(tmp_path, expression, answer):
    index, served, _ = dated_world(tmp_path)
    oracle = Oracle(index, served, SpanSyntax(), **EXPRESSIONS,
                    end_after_answer=True, eos_token_ids=[0])  # fmt: skip
    state = walk(oracle, served.encode(f"⟦ {expression}⟧ so ⟦ {answer}⟧"))
    assert oracle.allowed(state) == {0}


def test_vocabulary_relations_do_not_execute_or_bind_the_answer(toy_kg, toy_tokenizer):
    served, syntax = ServedTokenizer(toy_tokenizer), SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    modes = {"paths": True, "committed_paths": True, "escapes": False}
    frontier = Oracle(index, served, syntax, **modes)
    vocabulary = Oracle(index, served, syntax, **modes, relations="vocabulary")
    text = "⟦ Pierre Curie · place of birth · spouse⟧"
    with pytest.raises(TokenNotAllowed):
        walk(frontier, served.encode(text))
    state = walk(vocabulary, served.encode(text))
    assert state.reached is None
    assert state.last_span.labels
    assert not state.last_span.resolved  # a plan, without an execution witness
    assert not state.last_span.entity_ids
    assert walk_from(vocabulary, state, served, " Unbound answer: Paris.")
    assert walk_from(vocabulary, state, served, " ⟦ Marie Curie⟧")


SUPPORT_KG = KnowledgeGraph(
    entities=tuple(Entity(f"Q{n}", (name,)) for n, name in enumerate("ABCD", 1)),
    relations=(Relation("P1", ("x",)), Relation("P2", ("y",))),
    triples=(("Q1", "P1", "Q2"), ("Q2", "P2", "Q3"), ("Q4", "P2", "Q1")),
)
SUPPORT_VERSIONS = {"no-join": [("Q1", "P1", "Q3")]}
SUPPORT_SYNTAX = SpanSyntax(opener="[", closer="]", separator="|", label_prefix="", escape="!")


@pytest.mark.parametrize("relations", ["frontier", "vocabulary", "successor"])
@pytest.mark.parametrize("opened", [False, True])
@pytest.mark.parametrize("version", [None, "no-join"])
def test_support_ladder_span_language_is_exact(relations, opened, version):
    """Every token path from the opener, through the full hop cap, has exactly the
    independently enumerated language. In particular A|y is vocabulary/schema-only,
    and x|x is vocabulary-only. No admitted prefix is a dead end."""
    from itertools import pairwise, product

    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SUPPORT_SYNTAX
    index = build_kg_index(SUPPORT_KG, served, syntax)
    oracle = Oracle(index, served, syntax, paths=True, committed_paths=True,
                    escapes=False, relations=relations, versions=SUPPORT_VERSIONS)  # fmt: skip
    initial = oracle.initial_state(version)
    initial = oracle.opened_state(initial) if opened else walk_from(oracle, initial, served, "[")
    stack = [(initial, b"[")]
    actual = set()
    while stack:
        state, text = stack.pop()
        if state.phase is Phase.FREE:
            actual.add(text.decode())
            if len(state.last_span.labels) > 1 and relations != "frontier":
                assert not state.last_span.resolved
                assert state.reached is None
            continue
        allowed = oracle.allowed(state)
        assert allowed, text
        for token in allowed:
            after = oracle.advance(state, token)
            if relations != "frontier":
                assert not after.subjects and not after.ranks
            stack.append((after, text + served.token_bytes(token)))
    expected = {f"[{name}]" for name in "ABCD"}
    if relations == "frontier":
        expected |= {"[A|x]", "[A|x|y]", "[B|y]", "[D|y]", "[D|y|x]", "[D|y|x|y]"}
        if version:
            expected -= {"[A|x|y]", "[D|y|x|y]"}
    else:
        for hops in range(1, MAX_PATH_RELATIONS + 1):
            for path in product("xy", repeat=hops):
                if relations == "successor" and any(a == b for a, b in pairwise(path)):
                    continue
                if relations == "successor" and version and "x" in path[:-1]:
                    continue
                expected |= {"[" + "|".join((name, *path)) + "]" for name in "ABD"}
    assert actual == expected


def test_successor_relations_preserve_only_compatible_senses_of_a_shared_label():
    kg = KnowledgeGraph(
        tuple(Entity(f"Q{n}", (label,)) for n, label in enumerate("ABCDEF", 1)),
        (Relation("P1", ("a",)), Relation("P2", ("b", "x")),
         Relation("P3", ("c", "x")), Relation("P4", ("d",))),
        (("Q1", "P1", "Q2"), ("Q2", "P2", "Q3"),
         ("Q4", "P3", "Q5"), ("Q5", "P4", "Q6")),
    )  # fmt: skip
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SUPPORT_SYNTAX
    oracle = Oracle(build_kg_index(kg, served, syntax), served, syntax,
                    paths=True, escapes=False, relations="successor")  # fmt: skip
    assert walk(oracle, served.encode("[A|x|d]"))
    assert walk(oracle, served.encode("[A|a|x]"))
    with pytest.raises(TokenNotAllowed):
        walk(oracle, served.encode("[A|a|x|d]"))


@pytest.mark.parametrize("relations", ["vocabulary", "successor"])
@pytest.mark.parametrize("anchor", [None, ["Q1"]])
def test_support_ladder_keeps_start_constraints_and_unbinds_answers(
    toy_kg, toy_tokenizer, relations, anchor
):
    served, syntax = ServedTokenizer(toy_tokenizer), SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax, sitelinks=TOY_SITELINKS)
    modes = {
        "paths": True,
        "committed_paths": True,
        "start_rank": "sitelinks",
        "unquoted_starts": True,
        "escapes": False,
    }
    baseline = Oracle(index, served, syntax, **modes)
    weaker = Oracle(index, served, syntax, **modes, relations=relations)
    a = baseline.anchored_state(anchor) if anchor else baseline.initial_state()
    b = weaker.anchored_state(anchor) if anchor else weaker.initial_state()
    for token in served.encode("⟦ Marie Curie"):
        assert baseline.allowed(a) == weaker.allowed(b)
        a, b = baseline.advance(a, token), weaker.advance(b, token)
    closed = walk_from(weaker, b, served, " · employer⟧")
    assert closed.reached is None
    assert walk_from(weaker, closed, served, " so ⟦ Paris⟧")
    # The answer span keeps path mode's entity-only syntax, even though it is unbound.
    with pytest.raises(TokenNotAllowed):
        walk_from(weaker, closed, served, " so ⟦ Marie Curie ·")


@pytest.mark.parametrize("relations", ["vocabulary", "successor"])
@pytest.mark.parametrize(
    "bad", [{}, {"paths": True, "stop": "residual-goal"}, {"paths": True, "expressions": True}]
)
def test_weaker_relations_require_plain_path_mode(toy_kg, toy_tokenizer, relations, bad):
    served, syntax = ServedTokenizer(toy_tokenizer), SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    with pytest.raises(ValueError, match="weaker relations need path mode"):
        Oracle(index, served, syntax, relations=relations, **bad)


def test_explicit_frontier_matches_the_default_on_every_mask(toy_kg, toy_tokenizer):
    served, syntax = ServedTokenizer(toy_tokenizer), SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    modes = {"paths": True, "committed_paths": True, "escapes": False}
    default = Oracle(index, served, syntax, **modes)
    explicit = Oracle(index, served, syntax, **modes, relations="frontier")
    a, b = default.initial_state(), explicit.initial_state()
    rng = random.Random(91)
    for _ in range(200):
        assert default.allowed(a) == explicit.allowed(b)
        assert a == b
        token = rng.choice(sorted(default.allowed(a)))
        a, b = default.advance(a, token), explicit.advance(b, token)


@pytest.mark.parametrize("relations", ["frontier", "vocabulary", "successor"])
def test_support_ladder_keeps_unquoted_start_admission(relations):
    served, syntax = quoted_tokenizer(), SpanSyntax()
    index = build_kg_index(QUOTED_KG, served, syntax)
    oracle = Oracle(index, served, syntax, paths=True, committed_paths=True,
                    unquoted_starts=True, escapes=False, relations=relations)  # fmt: skip
    for name in ('"Klis"', "“Hero”", "«Bo»"):
        with pytest.raises(TokenNotAllowed):
            walk(oracle, served.encode(f"⟦ {name}"))
    # An anchor still admits its quoted label, identically in all arms.
    assert walk_from(oracle, oracle.anchored_state(["Q2"]), served, '⟦ "Klis" · by⟧')
    assert walk(oracle, served.encode("⟦ Bo · by⟧"))


@pytest.mark.parametrize("relations", ["vocabulary", "successor"])
@pytest.mark.parametrize("unique", [False, True])
def test_support_ladder_obeys_the_indexes_unique_alias_policy(relations, unique):
    from xgraph.wikidata import relation_names

    names = relation_names(
        [
            ("P1", "by", "label"),
            ("P1", "shared", "alias"),
            ("P2", "work", "label"),
            ("P2", "shared", "alias"),
        ],
        unique_aliases=unique,
    )
    kg = KnowledgeGraph((Entity("Q1", ("A",)),),
                        tuple(Relation(r, tuple(labels)) for r, labels in names.items()),
                        (("Q1", "P1", "Q1"), ("Q1", "P2", "Q1")))  # fmt: skip
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    oracle = Oracle(build_kg_index(kg, served, syntax), served, syntax,
                    paths=True, escapes=False, relations=relations)  # fmt: skip
    assert walk(oracle, served.encode("⟦ A · by · work⟧"))
    if unique:
        with pytest.raises(TokenNotAllowed):
            walk(oracle, served.encode("⟦ A · shared⟧"))
    else:
        assert walk(oracle, served.encode("⟦ A · shared · shared⟧"))


@pytest.mark.parametrize("question, stops", [
    ("Which employer has Marie Curie?", True),
    ("Which employer has the location of Marie Curie?", False),
    ("Which employer in a fictional realm has Marie Curie?", True),
    ("Which location has Marie Curie?", False),
    ("Who employs Marie Curie?", False),
])  # fmt: skip
def test_residual_goal_stop_forces_only_the_closer(toy_kg, toy_tokenizer, question, stops):
    served, syntax = ServedTokenizer(toy_tokenizer), SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    oracle = Oracle(index, served, syntax, paths=True, committed_paths=True, stop="residual-goal")
    state = oracle.initial_state(question=question)
    for token in served.encode("⟦ Marie Curie · employer"):
        state = oracle.advance(state, token)
    closer = served.encode(syntax.closer)
    if stops:
        # Check every byte boundary, including a split closer, with escapes enabled.
        for token in closer:
            remaining = syntax.closer.encode()[len(state.closing or b"") :]
            assert oracle.allowed(state) == {
                t
                for t in range(served.vocab_size)
                if served.token_bytes(t) and remaining.startswith(served.token_bytes(t))
            }
            state = oracle.advance(state, token)
        assert state.last_span.entity_ids == ("Q7",)
    else:
        for token in served.encode(syntax.separator):
            state = oracle.advance(state, token)


STOP_KG = KnowledgeGraph(
    entities=(Entity("Q1", ("Seed", "Borders Seed", "Country Seed", "Straße Location")),
              Entity("Q2", ("Seed",)), Entity("Q3", ("Harbor",)),
              Entity("Q4", ("Inland",)), Entity("Q5", ("Ward",)), Entity("Q6", ("City",)),
              Entity("Q99", (" ",))),  # declared but unwritable, including in rebuilt KGs
    relations=(Relation("P1", ("citizenship", "country")),
               Relation("P2", ("capital", "capital city")),
               Relation("P3", ("borders", "border country")),
               Relation("P4", ("located in", "location")),
               Relation("P5", ("visits",)), Relation("P6", ("countryish",))),
    triples=(("Q1", "P1", "Q3"), ("Q2", "P1", "Q4"), ("Q3", "P3", "Q5"),
             ("Q4", "P2", "Q6"), ("Q4", "P3", "Q5"), ("Q5", "P1", "Q3"),
             ("Q3", "P4", "Q5"), ("Q5", "P5", "Q3"), ("Q1", "P6", "Q5"),
             ("Q1", "P2", "Q3"), ("Q3", "P1", "Q5")),
)  # fmt: skip
STOP_VERSIONS = {"no-capital": [("Q4", "P2", "Q99")]}
STOP_CASES = [
    ("Seed", ["citizenship", "borders", "country"], "Which country has Seed?", None, None, 1),
    (
        "Seed",
        ["country", "borders", "country"],
        "Which country borders the country of Seed?",
        None,
        None,
        2,
    ),
    (
        "Seed",
        ["citizenship", "borders"],
        "Which country has the capital of Seed?",
        None,
        None,
        None,
    ),
    (
        "Seed",
        ["citizenship", "borders"],
        "Which country has the capital of Seed?",
        "no-capital",
        None,
        1,
    ),
    ("Seed", ["citizenship", "borders"], "Which country has the capital of Seed?", None, ["Q1"], 1),
    ("Borders Seed", ["country", "borders"], "Which country has Borders Seed?", None, None, 1),
    ("Country Seed", ["capital", "country"], "Which capital has Country Seed?", None, None, 1),
    ("Seed", ["capital", "country"], "Which capital has the country of Seed?", None, None, None),
    (
        "Straße Location",
        ["capital", "located in"],
        "Which capital has STRASSE LOCATION?",
        None,
        None,
        1,
    ),
    ("Seed", ["country", "borders"], "Which countries has Seed?", None, None, None),
    ("Seed", ["country", "borders"], "Which countrys has Seed?", None, None, 1),
    ("Seed", ["countryish", "visits"], "Which country has Seed?", None, None, None),
    ("Seed", ["country", "borders"], "Where is Seed?", None, None, None),
]


@pytest.mark.parametrize("opened", [False, True])
@pytest.mark.parametrize("start, relations, question, version, anchor, expected", STOP_CASES)
def test_residual_goal_stop_matches_cpu_truncation(
    start, relations, question, version, anchor, expected, opened
):
    from xgraph.kgversions import build_kg_versions
    from xgraph.programs import MatchedExecutor
    from xgraph.stopping import residual_goal_stop

    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    index = build_kg_index(STOP_KG, served, syntax)
    oracle = Oracle(
        index,
        served,
        syntax,
        paths=True,
        committed_paths=True,
        versions=STOP_VERSIONS,
        stop="residual-goal",
    )
    matched = MatchedExecutor(index, served, versions=build_kg_versions(index, STOP_VERSIONS))
    assert (
        residual_goal_stop(question, start, relations, matched, version=version, anchor=anchor)
        == expected
    )
    state = (
        oracle.anchored_state(anchor, version, question=question)
        if anchor
        else oracle.initial_state(version, question=question)
    )
    if opened:
        state = oracle.opened_state(state)
    for token in served.encode(("" if opened else syntax.opener) + syntax.label_prefix + start):
        state = oracle.advance(state, token)
    forced_at = None
    for hop, relation in enumerate(relations, 1):
        for token in served.encode(syntax.separator + syntax.label_prefix + relation):
            state = oracle.advance(state, token)
        if hop == len(relations):
            break  # CPU replay counts truncations, not a close at the final hop.
        continuation = state
        try:
            for token in served.encode(syntax.separator):
                continuation = oracle.advance(continuation, token)
        except TokenNotAllowed:
            forced_at = hop
            for token in served.encode(syntax.closer):
                state = oracle.advance(state, token)
            assert set(state.last_span.entity_ids) == set(
                matched.execute(start, relations[:hop], version=version, anchor=anchor).reached
            )
            break
    assert forced_at == expected


@pytest.mark.parametrize("opened", [False, True])
def test_residual_goal_stop_version_rebuild_removes_unwritable_capital(opened):
    from xgraph.kgversions import build_kg_versions

    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    index = build_kg_index(STOP_KG, served, syntax)
    rebuilt_index = build_kg_index(edited_kg(STOP_KG, STOP_VERSIONS["no-capital"]), served, syntax)
    versions = build_kg_versions(index, STOP_VERSIONS)
    assert versions.stats[0].unwritable_edits == rebuilt_index.unwritable_triples == 1
    assert set(index.relations_of("Q4")) == {"P2", "P3"}
    assert versions.view(index, "no-capital").relations_of("Q4") == ["P3"]
    assert rebuilt_index.relations_of("Q4") == ["P3"]
    modes = {"paths": True, "committed_paths": True, "stop": "residual-goal"}
    oracle = Oracle(index, served, syntax, versions=STOP_VERSIONS, **modes)
    rebuilt = Oracle(rebuilt_index, served, syntax, **modes)
    question = "Which country has the capital of Seed?"
    state = oracle.initial_state("no-capital", question=question)
    other = rebuilt.initial_state(question=question)
    base = oracle.initial_state(question=question)
    if opened:
        state, other, base = (
            oracle.opened_state(state),
            rebuilt.opened_state(other),
            oracle.opened_state(base),
        )
    text = ("" if opened else syntax.opener) + " Seed · citizenship"
    for token in served.encode(text):
        assert oracle.allowed(state) == rebuilt.allowed(other)
        state = oracle.advance(state, token)
        other = rebuilt.advance(other, token)
        base = oracle.advance(base, token)
    # The lower-ranked start still offers capital in the base KG. Removing its
    # unwritable replacement makes STOP fire in both the version and rebuilt KG.
    assert oracle.allowed(base) != oracle.allowed(state) == rebuilt.allowed(other)
    for token in served.encode(syntax.separator):
        base = oracle.advance(base, token)
    with pytest.raises(TokenNotAllowed):
        for token in served.encode(syntax.separator):
            state = oracle.advance(state, token)
    for token in served.encode(syntax.closer):
        assert oracle.allowed(state) == rebuilt.allowed(other)
        state, other = oracle.advance(state, token), rebuilt.advance(other, token)
    assert state.last_span.entity_ids == other.last_span.entity_ids == ("Q3",)


def test_residual_goal_stop_does_not_complete_a_capital_or_change_default(toy_kg, toy_tokenizer):
    served, syntax = ServedTokenizer(toy_tokenizer), SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    for stop in (None, "residual-goal"):
        oracle = Oracle(index, served, syntax, paths=True, committed_paths=True, stop=stop)
        state = oracle.initial_state(question="Which capital has Marie Curie?")
        for token in served.encode("⟦ Marie Curie · employer⟧"):
            state = oracle.advance(state, token)
        assert state.last_span.entity_ids == ("Q7",)


def test_residual_goal_stop_requires_committed_paths_and_a_question(toy_kg, toy_tokenizer):
    served, syntax = ServedTokenizer(toy_tokenizer), SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    with pytest.raises(ValueError, match="committed paths"):
        Oracle(index, served, syntax, paths=True, stop="residual-goal")
    with pytest.raises(ValueError, match="unknown STOP"):
        Oracle(index, served, syntax, paths=True, committed_paths=True, stop="capital")
    oracle = Oracle(index, served, syntax, paths=True, committed_paths=True, stop="residual-goal")
    with pytest.raises(ValueError, match="question"):
        oracle.initial_state()


def test_residual_goal_stop_handles_shared_delimiter_and_label_prefixes():
    served = ServedTokenizer(byte_level_tokenizer([]))
    syntax = SpanSyntax(opener="[", label_prefix="", closer="@a", separator="@b", escape="@c")
    kg = KnowledgeGraph(
        entities=(Entity("Q1", ("Seed",)), Entity("Q2", ("End",))),
        relations=(Relation("P1", ("country",)), Relation("P2", ("countryish",))),
        triples=(("Q1", "P1", "Q2"), ("Q1", "P2", "Q2"), ("Q2", "P1", "Q1")),
    )
    index = build_kg_index(kg, served, syntax)
    oracle = Oracle(index, served, syntax, paths=True, committed_paths=True, stop="residual-goal")
    state = oracle.initial_state(question="Which country has Seed?")
    for token in served.encode("[Seed@bcountry"):
        state = oracle.advance(state, token)
    # A longer relation remains writable. Among delimiter branches only the closer lives.
    longer = state
    for token in served.encode("ish@a"):
        longer = oracle.advance(longer, token)
    assert longer.last_span.labels[-1] == tuple(served.encode("countryish"))
    for token in served.encode("@"):
        state = oracle.advance(state, token)
    assert oracle.allowed(state) == set(served.encode("a"))
    for token in served.encode("b") + served.encode("c"):
        with pytest.raises(TokenNotAllowed):
            oracle.advance(state, token)


def test_residual_goal_stop_preserves_expression_masks():
    kg, dates, served = compare_kg()
    syntax = SpanSyntax()
    index = build_kg_index(kg, served, syntax, dates=dates, verdicts=True)
    for text, question in [
        ("⟦ same: Alien | Hair · country of origin⟧ so ⟦ yes⟧", "Which country has Alien?"),
        ("⟦ earlier: Alien | Hair · publication date⟧ so ⟦ Alien⟧", "Which date has Alien?"),
    ]:
        oracle = Oracle(index, served, syntax, **EXPRESSIONS, stop="residual-goal")
        plain = Oracle(index, served, syntax, **EXPRESSIONS)
        state, other = oracle.initial_state(question=question), plain.initial_state()
        for token in served.encode(text):
            assert oracle.allowed(state) == plain.allowed(other)
            state, other = oracle.advance(state, token), plain.advance(other, token)


def test_residual_goal_stop_uses_aliases_of_every_relation_sharing_the_label(toy_kg, toy_tokenizer):
    from xgraph.kgversions import build_kg_versions
    from xgraph.programs import MatchedExecutor
    from xgraph.stopping import residual_goal_stop

    served, syntax = ServedTokenizer(toy_tokenizer), SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    versions = {"p131-only": [("Q7", "P276", "Q99")]}
    question = "Which location has Marie Curie?"
    matched = MatchedExecutor(index, served, versions=build_kg_versions(index, versions))
    assert (
        residual_goal_stop(
            question,
            "Marie Curie",
            ["employer", "located in", "located in"],
            matched,
            version="p131-only",
        )
        == 2
    )
    oracle = Oracle(
        index,
        served,
        syntax,
        paths=True,
        committed_paths=True,
        versions=versions,
        stop="residual-goal",
    )
    state = oracle.initial_state("p131-only", question=question)
    for token in served.encode("⟦ Marie Curie · employer · located in"):
        state = oracle.advance(state, token)
    # Only P131 is offered, but P276 shares this label and its location alias names the head.
    assert all(
        syntax.closer.encode().startswith(served.token_bytes(t)) for t in oracle.allowed(state)
    )


def test_opener_at_the_end_of_a_token_opens_a_span(hand):
    oracle, ids, _ = hand
    assert walk(oracle, ids("x", "xâŁ¦")).phase is Phase.SPAN


def test_opener_split_across_tokens_opens_a_span(hand):
    oracle, ids, _ = hand
    assert walk(oracle, ids("âŁ")).phase is Phase.FREE
    assert walk(oracle, ids("âŁ", "¦")).phase is Phase.SPAN
    assert walk(oracle, ids("â", "Ł", "¦")).phase is Phase.SPAN


def test_token_that_completes_the_opener_before_its_end_is_not_allowed(hand):
    oracle, ids, _ = hand
    (token,) = ids("âŁ¦y")
    assert token not in oracle.allowed(oracle.initial_state())
    with pytest.raises(TokenNotAllowed):
        oracle.advance(oracle.initial_state(), token)


def test_end_of_text_is_allowed_only_outside_spans(hand):
    oracle, ids, eos = hand
    assert eos in oracle.allowed(walk(oracle, ids("x")))
    assert eos not in oracle.allowed(walk(oracle, ids("âŁ¦")))
    assert eos not in oracle.allowed(walk(oracle, ids("âŁ¦", "Bra")))
    assert eos not in oracle.allowed(walk(oracle, ids("âŁ¦", "â", "ģ", "ĩ")))


def test_closer_is_allowed_only_after_a_complete_label(hand):
    oracle, ids, _ = hand
    (closer,) = ids("âŁ§")
    assert closer not in oracle.allowed(walk(oracle, ids("âŁ¦")))
    assert closer in oracle.allowed(walk(oracle, ids("âŁ¦", "Bra")))
    assert closer not in oracle.allowed(walk(oracle, ids("âŁ¦", "Bra", "âŁ")))


def test_a_token_can_both_continue_a_label_and_start_the_closer(hand):
    oracle, ids, _ = hand
    short = walk(oracle, ids("âŁ¦", "Bra", "âŁ", "§"))
    assert short.phase is Phase.FREE
    assert short.last_span.entity_ids == ("Q1",)
    assert short.last_span.labels == (tuple(ids("Bra")),)
    long = walk(oracle, ids("âŁ¦", "Bra", "âŁ", "©", "âŁ§"))
    assert long.phase is Phase.FREE
    assert long.last_span.entity_ids == ("Q2",)
    assert long.last_span.labels == (tuple(ids("Bra", "âŁ", "©")),)


def test_a_token_can_both_continue_a_label_and_start_the_escape_marker(hand):
    oracle, ids, _ = hand
    allowed = oracle.allowed(walk(oracle, ids("âŁ¦", "Bra", "âģ")))
    assert set(ids("Ħ", "ĩ")) <= allowed  # finish ⁄ (the label) or ⁇ (the escape marker)
    assert ids("âģ")[0] not in allowed  # no fresh escape marker while one is pending
    assert walk(oracle, ids("âŁ¦", "Bra", "âģ", "ĩ")).phase is Phase.ESCAPED
    closed = walk(oracle, ids("âŁ¦", "Bra", "âģ", "Ħ", "âŁ§"))
    assert closed.last_span.entity_ids == ("Q3",)


def test_closer_must_end_at_a_token_boundary(hand):
    oracle, ids, _ = hand
    (glued,) = ids("âŁ§y")
    assert glued not in oracle.allowed(walk(oracle, ids("âŁ¦", "Bra")))
    assert glued not in oracle.allowed(walk(oracle, ids("âŁ¦", "â", "ģ", "ĩ")))


def test_escape_turns_the_rest_of_the_span_into_free_text(hand):
    oracle, ids, _ = hand
    assert ids("â")[0] in oracle.allowed(walk(oracle, ids("âŁ¦")))
    escaped = walk(oracle, ids("âŁ¦", "Bra", "â", "ģ", "ĩ", "x", "y", "Br"))
    assert escaped.phase is Phase.ESCAPED
    closed = oracle.advance(escaped, ids("âŁ§")[0])
    assert closed.phase is Phase.FREE
    assert not closed.last_span.resolved
    assert closed.last_span.entity_ids == ()


@pytest.mark.parametrize("builder", ["memory", "stream", "case-stream"])
def test_lossy_label_cannot_ground_another_printed_name(tmp_path, builder):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    from xgraph.index import load_kg_index
    from xgraph.wikidata import ItemBitmap, build_index, filter_truthy

    alphabet = sorted(set(pre_tokenizers.ByteLevel.alphabet()) - {"z"})
    raw = Tokenizer(models.BPE({c: i for i, c in enumerate(alphabet)}, []))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    served, syntax = ServedTokenizer(raw), SpanSyntax()
    kg = KnowledgeGraph(
        (Entity("Q1", ("A",)), Entity("Q2", ("Az",)), Entity("Q3", ("Goal",))),
        (Relation("P1", ("r",)),),
        (("Q2", "P1", "Q3"),),
    )
    if builder == "memory":
        index = build_kg_index(kg, served, syntax)
    else:
        entity = "http://www.wikidata.org/entity/"
        label = "<http://www.w3.org/2000/01/rdf-schema#label>"
        lines = [
            f'<{entity}{item.id}> {label} "{item.labels[0]}"@en .'
            for item in (*kg.entities, *kg.relations)
        ]
        lines.append(f"<{entity}Q2> <http://www.wikidata.org/prop/direct/P1> <{entity}Q3> .")
        filter_truthy(lines, ItemBitmap.of([1, 2, 3]), tmp_path / "kg")
        build_index(
            tmp_path / "kg",
            served,
            syntax,
            tmp_path / "index",
            initial_case_aliases=builder == "case-stream",
        )
        index = load_kg_index(tmp_path / "index", strict=True)
    oracle = Oracle(index, served, syntax, escapes=False)
    assert walk(oracle, served.encode("⟦ A⟧")).last_span.entity_ids == ("Q1",)
    assert all(not triples for _, _, triples in closed_spans(oracle, served.encode(syntax.opener)))
    assert index.unwritable_triples == 1


@pytest.mark.parametrize("mutation", ["added", "special", "normalizer", "decoder", "pretokenizer"])
def test_caller_tokenizer_mutation_cannot_change_indexed_span_bytes(mutation):
    import hashlib

    from tokenizers import decoders, normalizers, pre_tokenizers

    raw = byte_level_tokenizer([])
    served, syntax = ServedTokenizer(raw), SpanSyntax()
    before = served.encode(" Atlas")
    stream = [*served.encode(syntax.opener), *before, *served.encode(syntax.closer)]
    fingerprint, size, specials = served.fingerprint, served.vocab_size, served.special_ids
    byte_table = [served.token_bytes(t) for t in range(size)]
    if mutation == "added":
        raw.add_tokens([" Atlas"])
    elif mutation == "special":
        raw.add_special_tokens(["Atlas"])
    elif mutation == "normalizer":
        raw.normalizer = normalizers.Replace("Atlas", "Other")
    elif mutation == "decoder":
        raw.decoder = decoders.WordPiece()
    else:
        raw.pre_tokenizer = pre_tokenizers.Whitespace()
    assert hashlib.sha256(raw.to_str().encode()).hexdigest() != fingerprint
    index = build_kg_index(KnowledgeGraph((Entity("atlas", ("Atlas",)),)), served, syntax)
    oracle = Oracle(index, served, syntax, escapes=False)
    closed = walk(oracle, stream)
    assert closed.last_span is not None
    assert closed.last_span.entity_ids == ("atlas",)
    assert served.encode(" Atlas") == before
    assert served.encode_batch([" Atlas", " Atlas"]) == [before, before]
    assert served.fingerprint == index.tokenizer_fingerprint == fingerprint
    assert served.vocab_size == size and served.special_ids == specials
    assert [served.token_bytes(t) for t in range(size)] == byte_table


@pytest.mark.parametrize(
    "field,value",
    [
        ("opener", "[["),
        ("label_prefix", "_"),
        ("closer", "]]"),
        ("separator", " ::"),
        ("escape", " ?"),
        ("operand_separator", " &&"),
        ("operators", ("before:", "after:", "equal:")),
    ],
)
def test_oracle_refuses_every_span_syntax_mismatch(tmp_path, field, value):
    from xgraph.index import load_kg_index, save_kg_index

    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    index = build_kg_index(KnowledgeGraph((Entity("a", ("A",)),)), served, syntax)
    save_kg_index(index, tmp_path / "index")
    saved = load_kg_index(tmp_path / "index", strict=True)
    for candidate in (index, saved):
        with pytest.raises(IndexMismatch, match="span syntax"):
            Oracle(candidate, served, replace(syntax, **{field: value}))
        oracle = Oracle(candidate, served, syntax, escapes=False)
        assert walk(oracle, served.encode("⟦ A⟧")).last_span.entity_ids == ("a",)


@pytest.mark.parametrize("field,value", [("closer", "1"), ("escape", "=")])
def test_oracle_checks_syntax_binding_before_preparing_menu(field, value):
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    index = build_kg_index(KnowledgeGraph((Entity("a", ("A",)),)), served, syntax)
    with pytest.raises(IndexMismatch, match="span syntax"):
        Oracle(index, served, replace(syntax, **{field: value}), paths=True, menu=1)


def test_index_built_for_another_tokenizer_is_rejected(toy_kg, toy_tokenizer):
    index = build_kg_index(toy_kg, ServedTokenizer(toy_tokenizer), SpanSyntax())
    other = ServedTokenizer(train_byte_level_bpe(["a different corpus"], vocab_size=300))
    with pytest.raises(IndexMismatch):
        Oracle(index, other, SpanSyntax())


def test_escape_marker_and_closer_must_not_prefix_each_other():
    with pytest.raises(ValueError, match="prefix"):
        SpanSyntax(closer="⟧", escape="⟧?")


def closed_spans(oracle: Oracle, opener: list[int]) -> set:
    """Every span reachable from just after the opener that closes resolved.

    Each is (the token labels of its segments, its entities, its triples). Fails on
    a dead end: a reachable state inside a span with nothing allowed.
    """
    spans, stack = set(), [walk(oracle, opener)]
    while stack:
        state = stack.pop()
        allowed = oracle.allowed(state)
        assert allowed, f"dead end in {state}"
        for token in allowed:
            following = oracle.advance(state, token)
            if following.phase is Phase.FREE:
                closed = following.last_span
                spans.add((closed.labels, closed.entity_ids, closed.triples))
            elif following.phase is Phase.SPAN:
                stack.append(following)
    return spans


def test_spans_that_can_close_are_exactly_the_labels_and_triples(toy_kg, toy_tokenizer):
    """The exact-mask contract for entity and fact spans, from the KG directly."""
    served = ServedTokenizer(toy_tokenizer)
    oracle = Oracle(build_kg_index(toy_kg, served, SpanSyntax()), served, SpanSyntax())

    def written(label: str) -> tuple[int, ...]:
        return tuple(served.encode(" " + label))

    labels_of = defaultdict(list)
    for entity, label in KEPT:
        labels_of[entity].append(label)
    entities_by_label = defaultdict(set)
    for entity, label in KEPT:
        entities_by_label[written(label)].add(entity)
    expected = {
        ((tokens,), tuple(sorted(entities)), ()) for tokens, entities in entities_by_label.items()
    }
    triples_by_labels = defaultdict(set)
    for s, r, o in TOY_TRIPLES:
        for ls in labels_of[s]:
            for lr in dict(TOY_RELATIONS)[r]:
                for lo in labels_of[o]:
                    triples_by_labels[written(ls), written(lr), written(lo)].add((s, r, o))
    expected |= {(labels, (), tuple(sorted(t))) for labels, t in triples_by_labels.items()}
    assert closed_spans(oracle, served.encode("⟦")) == expected


@pytest.fixture
def toy_oracle(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    return Oracle(build_kg_index(toy_kg, served, SpanSyntax()), served, SpanSyntax()), served


def test_an_ambiguous_subject_unites_the_candidates_of_its_entities(toy_oracle):
    oracle, served = toy_oracle
    state = walk(oracle, served.encode("⟦ Curie · place of birth · Paris⟧"))
    assert state.last_span.triples == (("Q17", "P19", "Q4"), ("Q6", "P19", "Q3"))
    employer = walk(oracle, served.encode("⟦ Curie · employer"))
    assert employer.phase is Phase.SPAN  # only Q17 has it, and that is enough


def test_ambiguous_relation_and_object_resolve_to_every_matching_triple(toy_oracle):
    oracle, served = toy_oracle
    state = walk(oracle, served.encode("At ⟦ University of Paris · located in · Paris⟧"))
    assert state.last_span.triples == (("Q7", "P131", "Q3"), ("Q7", "P276", "Q4"))


def test_separator_only_after_a_subject_with_triples(toy_oracle):
    oracle, served = toy_oracle
    separator = served.encode(" ·")[0]
    assert separator not in oracle.allowed(walk(oracle, served.encode("⟦ Nobel Prize")))  # none
    assert separator in oracle.allowed(walk(oracle, served.encode("⟦ Paris")))  # Q3 has one


def test_no_closer_in_the_relation_segment(toy_oracle):
    oracle, served = toy_oracle
    closer = served.encode("⟧")[0]
    assert closer not in oracle.allowed(walk(oracle, served.encode("⟦ Marie Curie · spouse")))
    assert closer in oracle.allowed(
        walk(oracle, served.encode("⟦ Marie Curie · spouse · Pierre Curie"))
    )


def test_escape_inside_a_fact_span(toy_oracle):
    oracle, served = toy_oracle
    state = walk(oracle, served.encode("⟦ Marie Curie · spouse · ⁇ a physicist⟧"))
    assert state.phase is Phase.FREE and not state.last_span.resolved


@pytest.mark.parametrize(
    "delimiters",
    [{"separator": " ⁇x"}, {"separator": "⟧x"}, {"escape": " ·x"}, {"closer": " ·"}],
)
def test_delimiters_inside_a_span_must_not_prefix_each_other(delimiters):
    with pytest.raises(ValueError, match="prefix"):
        SpanSyntax(**delimiters)


@pytest.mark.parametrize(
    ("text", "resolved"),
    [
        ("She met ⟦ Marie Curie⟧ there", ("Q1",)),
        ("She met ⟦ ⁇ an unlisted lab⟧ there", None),
        ("She met ⟦ Marie ⁇ née Skłodowska⟧ there", None),
    ],
    ids=["label", "escape after the opener", "escape after a label token"],
)
def test_default_syntax_follows_the_tokenizer(toy_kg, toy_tokenizer, text, resolved):
    """Labels and the escape marker are written after a space, as the tokenizer splits them."""
    served = ServedTokenizer(toy_tokenizer)
    oracle = Oracle(build_kg_index(toy_kg, served, SpanSyntax()), served, SpanSyntax())
    state = walk(oracle, served.encode(text))
    assert state.phase is Phase.FREE
    if resolved is None:
        assert not state.last_span.resolved
    else:
        assert state.last_span.entity_ids == resolved


def test_advance_accepts_exactly_the_allowed_set(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    oracle = Oracle(build_kg_index(toy_kg, served, SpanSyntax()), served, SpanSyntax())
    prompt = served.encode("She met ⟦")
    rng = random.Random(1234)
    for _ in range(30):
        state = walk(oracle, prompt)
        for _ in range(12):
            allowed = oracle.allowed(state)
            for token in range(served.vocab_size):
                try:
                    oracle.advance(state, token)
                    accepted = True
                except TokenNotAllowed:
                    accepted = False
                assert accepted == (token in allowed), (state, token)
            state = oracle.advance(state, rng.choice(sorted(allowed)))


def test_without_escapes_a_span_can_only_name_the_kg():
    tokenizer = byte_level_tokenizer(MERGES)
    served = ServedTokenizer(tokenizer)
    index = build_kg_index(KnowledgeGraph(entities=HAND_ENTITIES), served, HAND_SYNTAX)
    oracle = Oracle(index, served, HAND_SYNTAX, escapes=False)

    def ids(*pieces: str) -> list[int]:
        return [tokenizer.token_to_id(piece) for piece in pieces]

    # "â" (E2) starts only the escape marker right after the opener; without escapes it
    # is not allowed, and the escape marker after a label is rejected.
    assert ids("â")[0] not in oracle.allowed(walk(oracle, ids("âŁ¦")))
    with pytest.raises(TokenNotAllowed):
        walk(oracle, ids("âŁ¦", "Bra", "â", "ģ", "ĩ"))


def test_without_repeats_a_fact_span_cannot_close_twice(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    syntax = SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    closer = served.encode(syntax.closer)[0]
    fact = "⟦ Marie Curie · award received · Nobel Prize"

    def before_closer(oracle: Oracle, text: str):
        state = oracle.initial_state()
        for token in served.encode(text):
            state = oracle.advance(state, token)
        return state

    once = fact + "⟧ and "
    for oracle in (Oracle(index, served, syntax), Oracle(index, served, syntax, repeats=False)):
        assert closer in oracle.allowed(before_closer(oracle, fact))
    repeats = Oracle(index, served, syntax)
    assert closer in repeats.allowed(before_closer(repeats, once + fact))
    no_repeats = Oracle(index, served, syntax, repeats=False)
    again = before_closer(no_repeats, once + fact)
    assert closer not in no_repeats.allowed(again)
    # The escape still leaves the span, and another object of the pair still closes.
    escaped = before_closer(no_repeats, once + fact + syntax.escape + " again")
    assert closer in no_repeats.allowed(escaped)
    other = before_closer(
        no_repeats, once + "⟦ Marie Curie · award received · Nobel Prize in Physics"
    )
    assert closer in no_repeats.allowed(other)


def test_repeats_can_only_be_switched_off_with_escapes(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    with pytest.raises(ValueError, match="escape"):
        Oracle(index, served, SpanSyntax(), escapes=False, repeats=False)


def _walk_text(oracle, served, text):
    state = oracle.initial_state()
    for token in served.encode(text):
        state = oracle.advance(state, token)
    return state


def test_derivation_binds_the_first_subject_to_its_best_connected_sense(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    derive = Oracle(index, served, SpanSyntax(), derivation=True)
    plain = Oracle(index, served, SpanSyntax())
    text = "⟦ Curie · place of birth · Paris⟧"
    # "Curie" is Q6 (one triple) and Q17 (two): derivation follows Q17 alone.
    assert {t[0] for t in _walk_text(plain, served, text).last_span.triples} == {"Q6", "Q17"}
    assert _walk_text(derive, served, text).last_span.triples == (("Q17", "P19", "Q4"),)


def test_derivation_binds_later_spans_to_the_entities_reached(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    syntax = SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    derive = Oracle(index, served, syntax, derivation=True)
    plain = Oracle(index, served, syntax)
    reached = "⟦ Curie · place of birth · Paris⟧ so "  # reaches Q4's Paris, which has no triples
    separator = served.encode(syntax.separator)[0]
    for oracle, continues in ((plain, True), (derive, False)):
        assert (separator in oracle.allowed(_walk_text(oracle, served, reached + "⟦ Paris"))) is (
            continues
        )
    # Entity spans after a fact name a reached entity: Paris yes, Marie Curie no.
    assert _walk_text(derive, served, reached + "⟦ Paris⟧").last_span.entity_ids == ("Q4",)
    assert served.encode(" Marie")[0] not in derive.allowed(
        _walk_text(derive, served, reached + "⟦")
    )
    # A chain goes on from the object it reached.
    chain = "⟦ Curie · employer · University of Paris⟧, ⟦ University of Paris · located in · Paris⟧"
    assert {t[2] for t in _walk_text(derive, served, chain).last_span.triples} == {"Q3", "Q4"}


def test_derivation_lets_a_reached_entity_be_named_by_any_of_its_labels(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    syntax = SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    derive = Oracle(index, served, syntax, derivation=True)
    reached = "⟦ Pierre Curie · spouse · Marie Curie⟧ so "  # reaches Q1, also Maria Skłodowska
    alias = _walk_text(
        derive, served, reached + "⟦ Maria Skłodowska · award received · Nobel Prize⟧"
    )
    assert alias.last_span.triples == (("Q1", "P166", "Q9"),)


def test_an_anchored_derivation_starts_from_the_question_entities(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    derive = Oracle(index, served, SpanSyntax(), derivation=True)
    text = "⟦ Curie · place of birth · Paris⟧"
    # Unanchored, "Curie" is its best-connected sense Q17; anchored to Q6, it is Q6.
    assert _walk_text(derive, served, text).last_span.triples == (("Q17", "P19", "Q4"),)
    state = derive.anchored_state(["Q6"])
    for token in served.encode(text):
        state = derive.advance(state, token)
    assert state.last_span.triples == (("Q6", "P19", "Q3"),)


def test_an_opened_state_is_the_state_right_after_the_span_opener(toy_kg, toy_tokenizer):
    """A request whose prompt ends with the opener starts in an open span: the same state as
    writing the opener, from the initial, an anchored, or a versioned state."""
    served = ServedTokenizer(toy_tokenizer)
    syntax = SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    edits = {"moved": [("Q1", "P108", "Q3")]}
    oracle = Oracle(index, served, syntax, paths=True, versions=edits)
    for start in (
        oracle.initial_state(),
        oracle.anchored_state(["Q6"]),
        oracle.initial_state("moved"),
        oracle.anchored_state(["Q1"], "moved"),
    ):
        opened = oracle.opened_state(start)
        assert opened.phase is Phase.SPAN
        assert opened == walk_from(oracle, start, served, syntax.opener)
    state = walk_from(oracle, oracle.initial_state(), served, "⟦ Pierre Curie · spouse⟧ ok")
    assert oracle.opened_state(state).last_span == state.last_span  # the memory is kept
    with pytest.raises(ValueError, match="free text"):
        oracle.opened_state(oracle.opened_state(oracle.initial_state()))


def test_a_path_span_follows_relations_the_frontier_has(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    syntax = SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    paths = Oracle(index, served, syntax, paths=True)
    # Pierre Curie --spouse--> Marie Curie --employer--> University of Paris.
    state = _walk_text(paths, served, "⟦ Pierre Curie · spouse · employer⟧")
    assert state.last_span.entity_ids == ("Q7",)
    # The answer must name the frontier: University of Paris, not Paris.
    assert _walk_text(
        paths, served, "⟦ Pierre Curie · spouse · employer⟧ so ⟦ University of Paris⟧"
    )
    assert served.encode(" Paris")[0] not in paths.allowed(
        _walk_text(paths, served, "⟦ Pierre Curie · spouse · employer⟧ so ⟦")
    )
    # A relation the frontier lacks is not allowed: Pierre Curie has no place of birth.
    with pytest.raises(TokenNotAllowed):
        _walk_text(paths, served, "⟦ Pierre Curie · place of birth")
    # Three hops; "located in" names two properties, so the frontier holds both objects.
    long = _walk_text(paths, served, "⟦ Pierre Curie · spouse · employer · located in⟧")
    assert long.last_span.entity_ids == ("Q3", "Q4")


def test_the_span_after_a_path_names_a_reached_entity_and_goes_no_further(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    syntax = SpanSyntax()
    paths = Oracle(build_kg_index(toy_kg, served, syntax), served, syntax, paths=True)
    path = "⟦ Pierre Curie · spouse · employer⟧ so "
    # The answer after a path is an entity span: no separator turns it into another path.
    answered = _walk_text(paths, served, path + "⟦ University of Paris⟧")
    assert answered.last_span.entity_ids == ("Q7",)
    with pytest.raises(TokenNotAllowed):
        _walk_text(paths, served, path + "⟦ University of Paris · located in")
    # After that entity span (or before any path, anchored), a path may start again.
    again = _walk_text(
        paths, served, path + "⟦ University of Paris⟧ in ⟦ University of Paris · located in⟧"
    )
    assert again.last_span.entity_ids == ("Q3", "Q4")
    state = paths.anchored_state(["Q2"])
    for token in served.encode("⟦ Pierre Curie · spouse⟧"):
        state = paths.advance(state, token)
    assert state.last_span.entity_ids == ("Q1",)


def test_a_path_ends_after_eight_relations(toy_kg, toy_tokenizer):
    from xgraph.oracle import MAX_PATH_RELATIONS

    served = ServedTokenizer(toy_tokenizer)
    syntax = SpanSyntax()
    paths = Oracle(build_kg_index(toy_kg, served, syntax), served, syntax, paths=True)
    assert MAX_PATH_RELATIONS == 8
    # spouse loops between Pierre and Marie Curie: seven relations may go on, eight must close
    _walk_text(paths, served, "⟦ Pierre Curie" + " · spouse" * 7 + " ·")
    with pytest.raises(TokenNotAllowed):
        _walk_text(paths, served, "⟦ Pierre Curie" + " · spouse" * 8 + " ·")
    closed = _walk_text(paths, served, "⟦ Pierre Curie" + " · spouse" * 8 + "⟧")
    assert closed.last_span.entity_ids == ("Q2",)  # back at Pierre after an even count


def test_a_path_starts_from_every_entity_with_the_label(toy_kg, toy_tokenizer):
    """Two entities are labelled Curie (Q6 and Q17, the better connected); the path keeps
    both, and the relations written decide which senses go on."""
    served = ServedTokenizer(toy_tokenizer)
    syntax = SpanSyntax()
    paths = Oracle(build_kg_index(toy_kg, served, syntax), served, syntax, paths=True)
    born = _walk_text(paths, served, "⟦ Curie · place of birth⟧")
    assert born.last_span.entity_ids == ("Q3", "Q4")  # Q6 born in Q3, Q17 in Q4
    employed = _walk_text(paths, served, "⟦ Curie · employer⟧")
    assert employed.last_span.entity_ids == ("Q7",)  # only Q17 has an employer


def test_a_bound_span_cannot_escape_without_bound_escapes(toy_kg, toy_tokenizer):
    """With bound_escapes False, a span the KG has bound (the answer after a path) names
    what the KG reached; escapes stay open in a path's relations and in unbound spans."""
    served = ServedTokenizer(toy_tokenizer)
    syntax = SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    strict = Oracle(index, served, syntax, paths=True, bound_escapes=False)
    _walk_text(strict, served, "⟦ ⁇ someone⟧ and ⟦ Pierre Curie · ⁇ some relation⟧")
    with pytest.raises(TokenNotAllowed):
        _walk_text(strict, served, "⟦ Pierre Curie · spouse · employer⟧ so ⟦ ⁇ elsewhere⟧")
    lenient = Oracle(index, served, syntax, paths=True)
    _walk_text(lenient, served, "⟦ Pierre Curie · spouse · employer⟧ so ⟦ ⁇ elsewhere⟧")


def test_a_committed_path_resolves_to_its_best_connected_start(toy_kg, toy_tokenizer):
    """Curie names Q6 and Q17 (the better connected). A path keeps both for its relations,
    but with committed paths it resolves to what Q17 alone reaches, so the answer cannot
    pick the other sense's value; without, it resolves to the union."""
    served = ServedTokenizer(toy_tokenizer)
    syntax = SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    union = Oracle(index, served, syntax, paths=True)
    committed = Oracle(index, served, syntax, paths=True, committed_paths=True)
    born = "⟦ Curie · place of birth⟧"
    assert _walk_text(union, served, born).last_span.entity_ids == ("Q3", "Q4")
    assert _walk_text(committed, served, born).last_span.entity_ids == ("Q4",)
    # The answer names the committed start's Paris (Q4), not the better-connected Paris
    # (Q3, Q6's birthplace) that the union offers.
    answer = born + " in ⟦ Paris⟧"
    assert _walk_text(union, served, answer).last_span.entity_ids == ("Q3",)
    assert _walk_text(committed, served, answer).last_span.entity_ids == ("Q4",)
    # Q4 has no triples, so a path cannot start from the committed Paris after the answer.
    after = born + " in ⟦ Paris⟧ near ⟦ Paris"
    assert served.encode(" ·")[0] not in committed.allowed(_walk_text(committed, served, after))
    assert served.encode(" ·")[0] in union.allowed(_walk_text(union, served, after))


def test_a_committed_path_falls_back_to_the_best_start_that_reaches_something(
    toy_kg, toy_tokenizer
):
    """Q17's birthplace (Q4) is located nowhere, so the path commits to Q6, the best start
    whose whole path is non-empty."""
    served = ServedTokenizer(toy_tokenizer)
    syntax = SpanSyntax()
    committed = Oracle(
        build_kg_index(toy_kg, served, syntax), served, syntax, paths=True, committed_paths=True
    )
    text = "⟦ Curie · place of birth · located in⟧"
    assert _walk_text(committed, served, text).last_span.entity_ids == ("Q12",)


def test_a_committed_path_offers_the_relations_of_every_start(toy_kg, toy_tokenizer):
    """Committing changes what a path resolves to, not which relations it may name."""
    served = ServedTokenizer(toy_tokenizer)
    syntax = SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    union = Oracle(index, served, syntax, paths=True)
    committed = Oracle(index, served, syntax, paths=True, committed_paths=True)
    text = "⟦ Curie · place of birth · located in"
    state, other = union.initial_state(), committed.initial_state()
    for token in served.encode(text):
        assert union.allowed(state) == committed.allowed(other)
        state, other = union.advance(state, token), committed.advance(other, token)
    assert union.allowed(state) == committed.allowed(other)


@pytest.mark.parametrize("anchor", [["Q6"], ["Q17"], ["Q2", "Q6"]])
def test_committing_leaves_an_anchored_path_unchanged(toy_kg, toy_tokenizer, anchor):
    """An anchored path starts from the anchor's one entity with the label: nothing to commit."""
    served = ServedTokenizer(toy_tokenizer)
    syntax = SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    union = Oracle(index, served, syntax, paths=True)
    committed = Oracle(index, served, syntax, paths=True, committed_paths=True)
    for text in ("⟦ Curie · place of birth⟧ in ⟦ Paris⟧", "⟦ Curie · employer · located in⟧"):
        state, other = union.anchored_state(anchor), committed.anchored_state(anchor)
        if "Q17" not in anchor and "employer" in text:
            continue  # Q6 has no employer
        for token in served.encode(text):
            assert union.allowed(state) == committed.allowed(other)
            state, other = union.advance(state, token), committed.advance(other, token)
        assert state.last_span == other.last_span


def test_a_committed_path_is_the_first_start_whose_own_path_is_non_empty(toy_kg, toy_tokenizer):
    """The ranks' claim, on every path of up to three relations from Curie: the committed
    path resolves to what the path reaches from one start alone (anchored there), the
    first start in best-connected order (Q17, then Q6) for which that is non-empty."""
    from itertools import product

    served = ServedTokenizer(toy_tokenizer)
    syntax = SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax)
    union = Oracle(index, served, syntax, paths=True)
    committed = Oracle(index, served, syntax, paths=True, committed_paths=True)

    def walk(oracle, state, text):
        for token in served.encode(text):
            state = oracle.advance(state, token)
        return state.last_span.entity_ids

    labels = sorted({label for _, names in TOY_RELATIONS for label in names})
    checked = 0
    for hops in (1, 2, 3):
        for path in product(labels, repeat=hops):
            text = "⟦ Curie" + "".join(f" · {label}" for label in path) + "⟧"
            alone = []
            for start in ("Q17", "Q6"):
                try:
                    alone.append(walk(union, union.anchored_state([start]), text))
                except TokenNotAllowed:
                    alone.append(())
            if not any(alone):
                continue
            expected = next(reached for reached in alone if reached)
            assert walk(committed, committed.initial_state(), text) == expected, text
            checked += 1
    assert checked > 5


def test_a_merged_date_is_named_only_at_its_stored_precision(tmp_path):
    """After a path reaches a date (merge_dates), the answer names it as its stored
    precision writes it: a day as "18 December 1946" or "December 18, 1946", a month as
    "June 1975", a year as "1977"; never in ISO form or at another precision."""
    from tests.conftest import write_dated_kgs
    from xgraph.index import load_kg_index
    from xgraph.literals import FIRST_ID, merge_dates
    from xgraph.wikidata import build_index

    literal, entity = write_dated_kgs(tmp_path)
    merge_dates(literal, entity, tmp_path / "dated")
    served = ServedTokenizer(byte_level_tokenizer([]))
    syntax = SpanSyntax()
    build_index(tmp_path / "dated", served, syntax, tmp_path / "index")
    paths = Oracle(load_kg_index(tmp_path / "index"), served, syntax, paths=True)
    born = "⟦ Jaws · director · date of birth⟧ so "
    for answer in ("18 December 1946", "December 18, 1946"):
        state = _walk_text(paths, served, born + f"⟦ {answer}⟧")
        assert state.last_span.entity_ids == (f"Q{FIRST_ID}",)
    released = "⟦ Jaws · publication date⟧ in "
    assert _walk_text(paths, served, released + "⟦ June 1975⟧").last_span.entity_ids == (
        f"Q{FIRST_ID + 1}",
    )
    for refused in (
        born + "⟦ 1946-",
        born + "⟦ 1946⟧",
        released + "⟦ 1 June",
        released + "⟦ 1975⟧",
    ):
        with pytest.raises(TokenNotAllowed):
            _walk_text(paths, served, refused)
    # The quantity stays out (Jaws has no duration), and so does the date of an item
    # the entity KG lacks.
    with pytest.raises(TokenNotAllowed):
        _walk_text(paths, served, "⟦ Jaws · duration")
    with pytest.raises(TokenNotAllowed):
        _walk_text(paths, served, "⟦ 2 March 1900")


def crowded_start_kg() -> KnowledgeGraph:
    """More entities labelled X than a frontier holds: S0 ... S65, in label index row order.
    Each Si has r to its own object Oi, which has t to T0; the last, past MAX_REACHED in row
    order, is the best-connected (it also has s) and its object O65 has t to T1."""
    count = MAX_REACHED + 2
    starts = [Entity(f"Q{i}", ("X",)) for i in range(count)]
    objects = [Entity(f"Q{1000 + i}", (f"O{i}",)) for i in range(count)]
    ends = [Entity("Q2000", ("T0",)), Entity("Q2001", ("T1",))]
    triples = [(f"Q{i}", "P1", f"Q{1000 + i}") for i in range(count)]
    triples += [
        (f"Q{1000 + i}", "P3", "Q2001" if i == count - 1 else "Q2000") for i in range(count)
    ]
    triples.append((f"Q{count - 1}", "P2", "Q1000"))
    return KnowledgeGraph(
        entities=(*starts, *objects, *ends),
        relations=(Relation("P1", ("r",)), Relation("P2", ("s",)), Relation("P3", ("t",))),
        triples=tuple(triples),
    )


def crowded_tokenizer(kg: KnowledgeGraph) -> ServedTokenizer:
    labels = [label for item in (*kg.entities, *kg.relations) for label in item.labels]
    return ServedTokenizer(
        train_byte_level_bpe([*labels, "⟦ X · r · t⟧ so ⟦ T1⟧ ⟦ ⁇ no⟧"], vocab_size=400)
    )


@pytest.mark.parametrize("committed", [False, True])
def test_frontier_cap_discards_the_only_valid_path_continuation(committed, monkeypatch):
    """A writable graph path need not survive the execution frontier cap.

    All successors share one start rank. Only B064 has the next relation, so
    committed-start ranking cannot rescue it. The larger cap is process-local.
    """
    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    middle = tuple(Entity(f"Q{100 + i}", (f"B{i:03d}",)) for i in range(65))
    kg = KnowledgeGraph(
        entities=(Entity("Q1", ("Start",)), *middle, Entity("Q999", ("Goal",))),
        relations=(Relation("P1", ("r",)), Relation("P2", ("t",))),
        triples=(*(("Q1", "P1", e.id) for e in middle), ("Q164", "P2", "Q999")),
    )
    assert MAX_REACHED == 64
    # An independent, literal witness in the input graph, including the discarded bridge.
    assert {("Q1", "P1", "Q164"), ("Q164", "P2", "Q999")} <= set(kg.triples)
    index = build_kg_index(kg, served, syntax)

    def oracle():
        return Oracle(index, served, syntax, paths=True, committed_paths=committed, escapes=False)

    capped = oracle()
    one = walk_from(capped, capped.anchored_state(["Q1"]), served, "⟦ Start · r⟧")
    assert one.last_span.entity_ids == tuple(sorted(e.id for e in middle[:-1]))
    assert len(one.last_span.entity_ids) == 64
    trace = "⟦ Start · r · t⟧ so ⟦ Goal⟧"
    with pytest.raises(TokenNotAllowed):
        walk_from(capped, capped.anchored_state(["Q1"]), served, trace)

    # The same tokens and index work if that single omitted frontier entity is retained.
    monkeypatch.setattr("xgraph.oracle.MAX_REACHED", 65)
    uncapped = oracle()
    answer = walk_from(uncapped, uncapped.anchored_state(["Q1"]), served, trace)
    assert answer.last_span.resolved
    assert answer.last_span.entity_ids == ("Q999",)


def test_a_committed_path_keeps_its_best_starts_when_the_label_names_too_many():
    """A frontier holds MAX_REACHED entities: by first label row, or, committed, by start
    rank, so that the best-connected start is kept even when it comes last in row order."""
    kg = crowded_start_kg()
    served = crowded_tokenizer(kg)
    syntax = SpanSyntax()
    index = build_kg_index(kg, served, syntax)
    union = Oracle(index, served, syntax, paths=True)
    committed = Oracle(index, served, syntax, paths=True, committed_paths=True)
    one = "⟦ X · r⟧"
    assert len(_walk_text(union, served, one).last_span.entity_ids) == MAX_REACHED
    assert f"Q{1000 + MAX_REACHED + 1}" not in _walk_text(union, served, one).last_span.entity_ids
    assert _walk_text(committed, served, one).last_span.entity_ids == (
        f"Q{1000 + MAX_REACHED + 1}",
    )
    two = "⟦ X · r · t⟧"
    assert _walk_text(union, served, two).last_span.entity_ids == ("Q2000",)
    assert _walk_text(committed, served, two).last_span.entity_ids == ("Q2001",)


def test_committed_paths_need_path_mode(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    with pytest.raises(ValueError, match="committed_paths needs path mode"):
        Oracle(index, served, SpanSyntax(), derivation=True, committed_paths=True)


# The sitelink rank: an ambiguous label's entities ordered by sitelink count, then degree.

TOY_SITELINKS = {"Q6": 40, "Q17": 3, "Q4": 9, "Q3": 2}
"""Curie: Q6 (one triple) has more sitelinks than Q17 (two), and Paris: Q4 (no triple) more
than Q3 (one), so that the sitelink rank and best-connected order disagree on both."""


def test_the_sitelink_rank_commits_a_path_to_the_start_with_the_most_sitelinks(
    toy_kg, toy_tokenizer
):
    served = ServedTokenizer(toy_tokenizer)
    syntax = SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax, sitelinks=TOY_SITELINKS)
    degree = Oracle(index, served, syntax, paths=True, committed_paths=True)
    ranked = Oracle(index, served, syntax, paths=True, committed_paths=True, start_rank="sitelinks")
    born = "⟦ Curie · place of birth⟧"
    assert _walk_text(degree, served, born).last_span.entity_ids == ("Q4",)  # Q17's
    assert _walk_text(ranked, served, born).last_span.entity_ids == ("Q3",)  # Q6's
    # Q6 has no employer: the path commits to the next start whose path is non-empty, Q17.
    employer = "⟦ Curie · employer⟧"
    assert _walk_text(ranked, served, employer).last_span.entity_ids == ("Q7",)
    # The rank changes what a path resolves to, not which relations it may name.
    state, other = degree.initial_state(), ranked.initial_state()
    for token in served.encode(born + " in"):
        assert degree.allowed(state) == ranked.allowed(other)
        state, other = degree.advance(state, token), ranked.advance(other, token)


def test_the_sitelink_rank_also_picks_derivation_modes_entity(toy_kg, toy_tokenizer):
    """Derivation mode shares the rank: a fact span's subject, and an entity span bound to
    the entities a fact span reached, name the one with the most sitelinks."""
    served = ServedTokenizer(toy_tokenizer)
    syntax = SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax, sitelinks=TOY_SITELINKS)
    degree = Oracle(index, served, syntax, derivation=True)
    ranked = Oracle(index, served, syntax, derivation=True, start_rank="sitelinks")
    fact = "⟦ Curie · place of birth · Paris⟧"
    assert _walk_text(degree, served, fact).last_span.triples == (("Q17", "P19", "Q4"),)
    assert _walk_text(ranked, served, fact).last_span.triples == (("Q6", "P19", "Q3"),)
    # University of Paris is located in Q3 (P131) and Q4 (P276): both reached, one named.
    named = "⟦ University of Paris · located in · Paris⟧ so ⟦ Paris⟧"
    assert _walk_text(degree, served, named).last_span.entity_ids == ("Q3",)
    assert _walk_text(ranked, served, named).last_span.entity_ids == ("Q4",)


def test_the_sitelink_rank_breaks_ties_by_the_versions_degree(toy_kg, toy_tokenizer):
    """Equal sitelinks leave best-connected order, by the degree of the KG version read."""
    served = ServedTokenizer(toy_tokenizer)
    syntax = SpanSyntax()
    index = build_kg_index(toy_kg, served, syntax, sitelinks={"Q6": 5, "Q17": 5})
    edits = {"q6": [("Q6", "P108", "Q7"), ("Q6", "P26", "Q2")]}  # Q6 gains two triples
    ranked = Oracle(
        index, served, syntax, paths=True, committed_paths=True, start_rank="sitelinks",
        versions=edits,
    )  # fmt: skip
    born = "⟦ Curie · place of birth⟧"
    assert _walk_text(ranked, served, born).last_span.entity_ids == ("Q4",)  # Q17's
    q6 = walk_from(ranked, ranked.initial_state("q6"), served, born)
    assert q6.last_span.entity_ids == ("Q3",)  # Q6's


def test_the_sitelink_rank_needs_sitelink_counts(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    with pytest.raises(ValueError, match="sitelink counts"):
        Oracle(index, served, SpanSyntax(), paths=True, start_rank="sitelinks")
    with pytest.raises(ValueError, match="start_rank"):
        Oracle(index, served, SpanSyntax(), paths=True, start_rank="fame")


# Unquoted starts: an unbound subject cannot name a label that starts with a quotation mark.

QUOTED_WORDS = ("Kleb", "Klis", "Hero", "Ann", "Dash", "Bo", "by", "work")
# Byte-level alphabet: 20 = "Ġ", E2 = "â", 80 = "Ģ", 9C = "ľ", 93 = "ĵ". So ' "' is one token,
# while ' “' (20 E2 80 9C) and ' –' (20 E2 80 93) share their first token "Ġâ", which also
# starts the escape marker ' ⁇' (20 E2 81 87), and ' «' (20 C2 AB) starts with "Ġ" alone, as
# the separator ' ·' (20 C2 B7) does. Each word is one token, with or without a space before.
QUOTE_MERGES = [
    ("Ġ", '"'),
    ("Ġ", "â"),
    ("Ģ", "ľ"),
    ("Ģ", "ĵ"),
    *dict.fromkeys(
        merge
        for word in QUOTED_WORDS
        for merge in [*((word[:i], word[i]) for i in range(1, len(word))), ("Ġ", word)]
    ),
]
QUOTED_KG = KnowledgeGraph(
    entities=(
        Entity("Q1", ("Kleb",)),
        Entity("Q2", ('"Klis"',)),
        Entity("Q3", ("“Hero”",)),
        Entity("Q4", ("Ann",)),
        Entity("Q5", ("–Dash",)),
        Entity("Q6", ("«Bo»", "Bo")),
    ),
    relations=(Relation("P50", ("by",)), Relation("P800", ("work",))),
    triples=(
        ("Q1", "P50", "Q4"),
        ("Q2", "P50", "Q4"),
        ("Q3", "P50", "Q4"),
        ("Q5", "P50", "Q4"),
        ("Q6", "P50", "Q4"),
        ("Q4", "P800", "Q2"),
        ("Q4", "P800", "Q3"),
        ("Q4", "P800", "Q6"),
    ),
)
"""Titles in straight, typographic, and angle quotation marks, a dash-led label whose first
token a quoted one shares, and a quoted label with an unquoted alias (Q6)."""


def quoted_tokenizer() -> ServedTokenizer:
    served = ServedTokenizer(byte_level_tokenizer(QUOTE_MERGES))
    first = {text: served.encode(text)[0] for text in (' "Klis"', " “Hero”", " –Dash", " «Bo»")}
    assert served.token_bytes(first[' "Klis"']) == b' "'
    assert first[" “Hero”"] == first[" –Dash"]
    assert served.token_bytes(first[" «Bo»"]) == b" "
    assert len(served.encode(" Kleb")) == 1
    return served


def test_an_unbound_subject_cannot_start_with_a_quotation_mark():
    """A model copying a quoted title's opening quote from the question is not steered into
    some other label that starts with one: with unquoted starts no such label is offered."""
    served = quoted_tokenizer()
    index = build_kg_index(QUOTED_KG, served, SpanSyntax())
    plain = Oracle(index, served, SpanSyntax())
    unquoted = Oracle(index, served, SpanSyntax(), unquoted_starts=True)
    for text, resolved in {
        '⟦ "Klis"⟧': ("Q2",),
        "⟦ “Hero”⟧": ("Q3",),
        "⟦ «Bo»⟧": ("Q6",),
    }.items():
        assert _walk_text(plain, served, text).last_span.entity_ids == resolved
        with pytest.raises(TokenNotAllowed):
            _walk_text(unquoted, served, text)
    opened = _walk_text(unquoted, served, "⟦")
    assert served.encode(' "')[0] not in unquoted.allowed(opened)
    assert served.encode(' "')[0] in plain.allowed(_walk_text(plain, served, "⟦"))
    # The first token ' “' shares with ' –' stays; only the quotation mark's rows are gone.
    shared, dash = served.encode(" –Dash")[:2]
    quote = served.encode(" “Hero”")[1]
    assert dash in unquoted.allowed(unquoted.advance(opened, shared))
    assert quote not in unquoted.allowed(unquoted.advance(opened, shared))
    assert quote in plain.allowed(plain.advance(_walk_text(plain, served, "⟦"), shared))
    for subject, entity in {"Kleb": "Q1", "–Dash": "Q5", "Bo": "Q6"}.items():
        fact = _walk_text(unquoted, served, f"⟦ {subject} · by · Ann⟧")
        assert fact.last_span.triples == ((entity, "P50", "Q4"),)
    assert _walk_text(unquoted, served, "⟦ ⁇ unknown⟧").last_span.resolved is False


def test_quoted_labels_stay_writable_where_the_kg_binds_the_span():
    """Unquoted starts restrict only unbound subjects: an object, the answer after a path, and
    an anchored subject name what the KG holds, quotation marks and all."""
    served = quoted_tokenizer()
    index = build_kg_index(QUOTED_KG, served, SpanSyntax())
    plain = Oracle(index, served, SpanSyntax(), unquoted_starts=True)
    fact = _walk_text(plain, served, '⟦ Ann · work · "Klis"⟧')
    assert fact.last_span.triples == (("Q4", "P800", "Q2"),)
    paths = Oracle(index, served, SpanSyntax(), paths=True, unquoted_starts=True)
    answer = _walk_text(paths, served, "⟦ Ann · work⟧ so ⟦ “Hero”⟧")
    assert answer.last_span.entity_ids == ("Q3",)
    anchored = walk_from(paths, paths.anchored_state(["Q2"]), served, '⟦ "Klis" · by⟧')
    assert anchored.last_span.entity_ids == ("Q4",)


# KG versions: named edit sets over one KG index (docs/kg-versions-adr.md).


def walk_from(oracle: Oracle, state, served, text: str):
    for token in served.encode(text):
        state = oracle.advance(state, token)
    return state


def test_a_kg_version_replaces_the_objects_of_its_edited_pairs(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    # Marie Curie's employer becomes Paris (Q3); Q4's Paris gains a relation it lacked.
    edits = {"moved": [("Q1", "P108", "Q3"), ("Q4", "P131", "Q12")]}
    oracle = Oracle(index, served, SpanSyntax(), versions=edits)
    moved = oracle.initial_state("moved")
    fact = "⟦ Marie Curie · employer · "
    versioned = walk_from(oracle, moved, served, fact + "Paris⟧")
    assert versioned.last_span.triples == (("Q1", "P108", "Q3"),)
    assert versioned.version == "moved"
    with pytest.raises(TokenNotAllowed):  # the base object is gone
        walk_from(oracle, moved, served, fact + "University")
    # The base KG and unedited pairs are untouched.
    assert walk(oracle, served.encode(fact + "University of Paris⟧")).last_span.triples == (
        ("Q1", "P108", "Q7"),
    )
    pierre = "⟦ Pierre Curie · employer · University of Paris⟧"
    assert walk_from(oracle, moved, served, pierre).last_span.resolved
    located = "⟦ Paris · located in · Rhône–Alpes⟧"
    assert walk_from(oracle, moved, served, located).last_span.triples == (
        ("Q3", "P131", "Q12"),
        ("Q4", "P131", "Q12"),
    )
    assert walk(oracle, served.encode(located)).last_span.triples == (("Q3", "P131", "Q12"),)


def test_a_kg_version_ranks_best_connected_by_its_own_degree(toy_kg, toy_tokenizer):
    """Curie is Q6 (one triple) and Q17 (two); a version giving Q6 two more makes Q6 the
    best-connected sense, as an index rebuilt from the edited triples would."""
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    edits = {"q6": [("Q6", "P108", "Q7"), ("Q6", "P26", "Q2")]}
    derive = Oracle(index, served, SpanSyntax(), derivation=True, versions=edits)
    text = "⟦ Curie · place of birth · Paris⟧"
    assert walk(derive, served.encode(text)).last_span.triples == (("Q17", "P19", "Q4"),)
    assert walk_from(derive, derive.initial_state("q6"), served, text).last_span.triples == (
        ("Q6", "P19", "Q3"),
    )


def test_an_unwritable_edit_still_replaces_its_pair(toy_kg, toy_tokenizer):
    """Q10's only label is excluded: the edit writes no edge, yet Marie Curie's spouses
    are replaced, so she has none in the version (the rebuilt index's unwritable triple)."""
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    oracle = Oracle(index, served, SpanSyntax(), versions={"v": [("Q1", "P26", "Q10")]})
    assert walk(oracle, served.encode("⟦ Marie Curie · spouse · Pierre Curie⟧")).last_span.resolved
    with pytest.raises(TokenNotAllowed):
        walk_from(oracle, oracle.initial_state("v"), served, "⟦ Marie Curie · spouse")


def test_an_unknown_kg_version_is_refused(toy_kg, toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    index = build_kg_index(toy_kg, served, SpanSyntax())
    oracle = Oracle(index, served, SpanSyntax(), versions={"v": [("Q1", "P26", "Q2")]})
    with pytest.raises(ValueError, match="no KG version 'w0'"):
        oracle.initial_state("w0")
    with pytest.raises(ValueError, match="no KG version 'w0'"):
        oracle.anchored_state(["Q1"], "w0")


VERSION_LABELS = ("Ann", "Ann Lee", "Bo", "Bo Ray", "Cy", "Lee")
VERSION_RELATIONS = (
    Relation("P1", ("born",)),
    Relation("P2", ("born in",)),
    Relation("P3", ("works",)),
    Relation("P4", ("likes", "works")),  # an alias that is another relation's label
)


def random_versioned_kg(seed: int) -> tuple[KnowledgeGraph, dict[str, list[tuple[str, str, str]]]]:
    """A small KG over shared and prefix labels, with an entity whose only label is
    excluded (Q11), and two KG versions of random edits: replaced pairs, new pairs, and
    edits naming Q11 (unwritable)."""
    rng = random.Random(seed)
    entities = [
        Entity(f"Q{i}", tuple(rng.sample(VERSION_LABELS, rng.choice((1, 1, 2)))))
        for i in range(1, 11)
    ]
    entities.append(Entity("Q11", ("A⟧B",)))
    ids = [e.id for e in entities]
    relations = [r.id for r in VERSION_RELATIONS]
    triples = sorted(
        {(rng.choice(ids[:6]), rng.choice(relations), rng.choice(ids)) for _ in range(22)}
    )
    versions = {}
    for name in ("v1", "v2"):
        edits = []
        for _ in range(rng.randint(1, 4)):
            if rng.random() < 0.6:  # an existing pair
                subject, relation, _ = rng.choice(triples)
            else:
                subject, relation = rng.choice(ids), rng.choice(relations)
            edits += [(subject, relation, rng.choice(ids)) for _ in range(rng.randint(1, 2))]
        versions[name] = edits
    kg = KnowledgeGraph(
        entities=tuple(entities), relations=VERSION_RELATIONS, triples=tuple(triples)
    )
    return kg, versions


def versioned_tokenizer() -> ServedTokenizer:
    labels = [*VERSION_LABELS, *(label for r in VERSION_RELATIONS for label in r.labels)]
    return ServedTokenizer(
        train_byte_level_bpe([*labels, "⟦ Ann Lee · born in · Bo Ray⟧ ⟦ ⁇ no⟧"], vocab_size=320)
    )


def edited_kg(kg: KnowledgeGraph, edits: list[tuple[str, str, str]]) -> KnowledgeGraph:
    """The KG with the edits applied by xgraph.mquake.apply_edits, the reference semantics."""
    import numpy as np

    from xgraph.mquake import apply_edits

    def numbers(rows):
        return np.array([[int(x[1:]) for x in row] for row in rows], dtype=np.int32).reshape(-1, 3)

    triples = apply_edits(numbers(kg.triples), numbers(edits))
    return replace(
        kg, triples=tuple(sorted({(f"Q{s}", f"P{p}", f"Q{o}") for s, p, o in triples.tolist()}))
    )


def random_sitelinks(kg: KnowledgeGraph, seed: int) -> dict[str, int]:
    """Sitelink counts from a small range, so that the sitelink rank both decides and ties."""
    rng = random.Random(seed)
    return {e.id: rng.randint(0, 2) for e in kg.entities}


VERSION_MODES = {
    "plain": {},
    "derivation": {"derivation": True},
    "paths": {"paths": True},
    "committed": {"paths": True, "committed_paths": True},
    "derivation by sitelinks": {"derivation": True, "start_rank": "sitelinks"},
    "committed by sitelinks": {"paths": True, "committed_paths": True, "start_rank": "sitelinks"},
}
"""Engine and oracle options to run KG versions in; the KG indexes carry sitelink counts."""


@pytest.mark.parametrize(
    "mode,terminal",
    [(mode, 0) for mode in VERSION_MODES]
    + [(mode, 1) for mode, options in VERSION_MODES.items() if options.get("paths")],
)
@pytest.mark.parametrize("seed", range(4))
def test_a_kg_version_equals_the_index_rebuilt_from_its_edited_triples(seed, mode, terminal):
    """Random walks through a versioned oracle and an oracle on the rebuilt index: the
    same allowed sets and the same states, every step."""
    kg, versions = random_versioned_kg(seed)
    sitelinks = random_sitelinks(kg, seed)
    served = versioned_tokenizer()
    syntax = SpanSyntax()
    index = build_kg_index(kg, served, syntax, sitelinks=sitelinks)
    options = {
        **VERSION_MODES[mode],
        "observe_terminal": terminal,
        "eos_token_ids": sorted(served.special_ids) if terminal else (),
    }
    versioned = Oracle(index, served, syntax, versions=versions, **options)
    rng = random.Random(seed)
    opener, separator, closer = (served.encode(d) for d in ("⟦", " ·", "⟧"))
    resolved = 0
    for name, edits in versions.items():
        rebuilt_index = build_kg_index(edited_kg(kg, edits), served, syntax, sitelinks=sitelinks)
        rebuilt = Oracle(rebuilt_index, served, syntax, **options)
        for _ in range(8):
            ours, theirs, queue = versioned.initial_state(name), rebuilt.initial_state(), []
            for _ in range(40):
                allowed = rebuilt.allowed(theirs)
                assert versioned.allowed(ours) == allowed
                assert replace(ours, version=None) == theirs
                if not queue and theirs.phase is Phase.FREE and rng.random() < 0.4:
                    queue = list(opener)
                elif not queue and theirs.phase is Phase.SPAN and rng.random() < 0.5:
                    queue = list(rng.choice((separator, closer)))
                token = queue.pop(0) if queue and queue[0] in allowed else None
                if token is None:
                    queue = []
                    token = rng.choice(sorted(allowed))
                ours, theirs = versioned.advance(ours, token), rebuilt.advance(theirs, token)
                resolved += ours.last_span is not None and ours.last_span.resolved
    assert resolved > 0


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("anchored", [False, True])
def test_resolved_random_paths_and_bound_answers_have_versioned_edge_witnesses(
    seed, committed, anchored
):
    """Check accepted paths against raw triples, independently of oracle execution.

    Seeded KGs include shared entity/relation labels, aliases and unwritable
    edits. The witness search deliberately has no frontier cap or start ranking:
    it checks soundness, without assuming complete path coverage.
    """
    kg, versions = random_versioned_kg(seed)
    served, syntax = versioned_tokenizer(), SpanSyntax()
    index = build_kg_index(kg, served, syntax)
    oracle = Oracle(
        index,
        served,
        syntax,
        paths=True,
        escapes=False,
        committed_paths=committed,
        versions=versions,
    )
    rng = random.Random(seed)

    def owners(table):
        result = defaultdict(set)
        for row in range(table.num_rows):
            result[table.row_tokens(row)].add(table.owner_id(row))
        return result

    entity_labels, relation_labels = owners(index.labels), owners(index.relations)
    entities = set().union(*entity_labels.values())
    relations = set().union(*relation_labels.values())
    opener, separator, closer = (served.encode(d) for d in ("⟦", " ·", "⟧"))

    def advance(state, tokens):
        for token in tokens:
            state = oracle.advance(state, token)
        return state

    def accepted(state, labels, suffix):
        result = []
        for label in sorted(labels):
            with suppress(TokenNotAllowed):
                result.append((label, advance(state, [*label, *suffix])))
        return result

    for version, edits in [(None, []), *versions.items()]:
        # Literal replacement semantics, without oracle graph/frontier helpers.
        replaced = {(s, r) for s, r, _ in edits}
        triples = {edge for edge in kg.triples if edge[:2] not in replaced} | set(edits)
        writable = {
            (s, r, o) for s, r, o in triples if s in entities and o in entities and r in relations
        }
        checked = 0
        for _ in range(6):
            anchor = [rng.choice(sorted({s for s, _, _ in writable}))] if anchored else None
            initial = (
                oracle.anchored_state(anchor, version) if anchor else oracle.initial_state(version)
            )
            starts = accepted(advance(initial, opener), entity_labels, separator)
            if not starts:  # an anchor's writable label rows may be capped
                continue
            _, state = rng.choice(starts)
            for _ in range(4):
                endings = accepted(state, relation_labels, closer)
                if not endings:
                    break
                label, closed = rng.choice(endings)
                path = closed.last_span
                assert path.resolved and len(path.labels) > 1 and path.entity_ids
                starts = entity_labels[path.labels[0]]
                if anchor:
                    starts = starts & set(anchor)
                witnesses = {e: () for e in starts}
                for written in path.labels[1:]:
                    witnesses = {
                        o: (*witnesses[s], (s, r, o))
                        for s, r, o in sorted(writable)
                        if s in witnesses and r in relation_labels[written]
                    }
                assert set(path.entity_ids) <= witnesses.keys(), (seed, version, path)
                answers = accepted(advance(closed, opener), entity_labels, closer)
                assert answers
                for _, answered in answers:
                    answer = answered.last_span
                    assert answer.resolved and answer.entity_ids
                    assert set(answer.entity_ids) <= set(path.entity_ids)
                    for endpoint in answer.entity_ids:
                        witness = witnesses[endpoint]
                        assert len(witness) == len(path.labels) - 1
                        assert witness[0][0] in starts and witness[-1][2] == endpoint
                        assert all(edge in writable for edge in witness)
                        assert all(a[2] == b[0] for a, b in pairwise(witness))
                checked += 1
                try:
                    state = advance(state, [*label, *separator])
                except TokenNotAllowed:  # a witnessed endpoint can have no outgoing edge
                    break
        assert checked > 0, (seed, version, committed, anchored)


# Expressions: date comparisons written as one span, `⟦ earlier: A | B · relation⟧`, with
# committed paths over a KG index with merged dates (xgraph.literals merge-dates).

EXPRESSIONS = {"paths": True, "committed_paths": True, "expressions": True}


def dated_world(root):
    """The dated toy KG (tests/conftest.py write_dated_kgs) merged into one KG directory,
    its index for a tokenizer trained on its labels, and the literal KG directory:
    (index, served tokenizer, literal KG directory)."""
    import json

    from tests.conftest import write_dated_kgs
    from xgraph.index import load_kg_index
    from xgraph.literals import merge_dates
    from xgraph.wikidata import build_index

    literal, entity = write_dated_kgs(root)
    merge_dates(literal, entity, root / "dated")
    labels = [
        json.loads(line)[1]
        for name in ("entity_labels.jsonl", "relation_labels.jsonl")
        for line in (root / "dated" / name).read_text(encoding="utf-8").splitlines()
    ]
    texts = [
        "⟦ earlier: Jaws | Star Wars · director · date of birth⟧ so ⟦ Star Wars⟧",
        "⟦ later: Steven Spielberg | George Lucas · date of birth⟧ ⟦ ⁇ no⟧",
        "⟦ same: Jaws | Star Wars · director⟧ so ⟦ no⟧ ⟦ yes⟧",
    ]
    served = ServedTokenizer(train_byte_level_bpe(labels + texts, vocab_size=400))
    build_index(root / "dated", served, SpanSyntax(), root / "index", verdicts=True)
    return load_kg_index(root / "index"), served, literal


def dated_executor(literal, root):
    """xgraph.expressions' CPU executor over the literal KG's KG store and date store."""
    from xgraph.expressions import DateStore, Executor, Relations, build_date_store
    from xgraph.toolkg import KGStore, build_store

    build_store(literal, root / "kg.sqlite")
    build_date_store(literal, root / "dates.sqlite")
    return Executor(
        KGStore(root / "kg.sqlite"),
        DateStore(root / "dates.sqlite"),
        Relations.read(literal / "relation_labels.jsonl"),
    )


def test_expressions_need_committed_paths_escapes_and_an_index_with_dates(
    toy_kg, toy_tokenizer, tmp_path
):
    index, served, _ = dated_world(tmp_path)
    with pytest.raises(ValueError, match="committed paths"):
        Oracle(index, served, SpanSyntax(), paths=True, expressions=True)
    with pytest.raises(ValueError, match="escapes"):
        Oracle(index, served, SpanSyntax(), escapes=False, **EXPRESSIONS)
    undated = build_kg_index(toy_kg, ServedTokenizer(toy_tokenizer), SpanSyntax())
    with pytest.raises(ValueError, match="date table"):
        Oracle(undated, ServedTokenizer(toy_tokenizer), SpanSyntax(), **EXPRESSIONS)


def test_a_date_comparison_binds_the_answer_to_the_operand_it_returns(tmp_path):
    """Jaws (June 1975, directed by Spielberg, born 18 December 1946) against Star Wars (1977,
    directed by Lucas, born 14 May 1944): the comparison returns the operand whose date is
    earlier (later), and the answer span after it may only name that operand."""
    index, served, _ = dated_world(tmp_path)
    oracle = Oracle(index, served, SpanSyntax(), **EXPRESSIONS)
    cases = {
        "⟦ earlier: Jaws | Star Wars · publication date⟧": "Q1",
        "⟦ later: Jaws | Star Wars · publication date⟧": "Q3",
        "⟦ earlier: Jaws | Star Wars · director · date of birth⟧": "Q3",
        "⟦ later: Star Wars | Jaws · director · date of birth⟧": "Q1",
        "⟦ earlier: Steven Spielberg | George Lucas · date of birth⟧": "Q4",
    }
    labels = {"Q1": "Jaws", "Q3": "Star Wars", "Q4": "George Lucas"}
    others = {"Q1": "Star Wars", "Q3": "Jaws", "Q4": "Steven Spielberg"}
    for text, answer in cases.items():
        state = _walk_text(oracle, served, text)
        closed = state.last_span
        assert (closed.resolved, closed.entity_ids) == (True, (answer,)), text
        assert closed.operator == text.split(":")[0][2:], text
        assert len(closed.operands) == 2 and answer in closed.operands, text
        answered = walk_from(oracle, state, served, f" so ⟦ {labels[answer]}⟧")
        assert answered.last_span.entity_ids == (answer,), text
        with pytest.raises(TokenNotAllowed):
            walk_from(oracle, state, served, f" so ⟦ {others[answer]}⟧")
        with pytest.raises(TokenNotAllowed):  # the answer is an entity span
            walk_from(oracle, state, served, f" so ⟦ {labels[answer]} ·")


def _text_of(served, tokens: tuple[int, ...]) -> str:
    return b"".join(served.token_bytes(t) for t in tokens).decode("utf-8")[1:]


def test_comparisons_that_can_close_are_exactly_those_the_executor_runs(tmp_path):
    """The exact-mask contract for date comparisons: every comparison the oracle lets close,
    from the operator on, is one that xgraph.expressions' executor runs to a value (or to
    undetermined, two dates that overlap) over the literal KG's stores, and it returns what
    the executor returns: the operand it gives, or both operands when undetermined."""
    import json

    from xgraph.expressions import Compare, Start

    index, served, literal = dated_world(tmp_path)
    executor = dated_executor(literal, tmp_path)
    oracle = Oracle(index, served, SpanSyntax(), **EXPRESSIONS)
    found = set()
    stack = [walk(oracle, served.encode(f"⟦ {op}:")) for op in ("earlier", "later")]
    while stack:
        state = stack.pop()
        allowed = oracle.allowed(state)
        assert allowed, f"dead end in {state}"
        for token in allowed:
            following = oracle.advance(state, token)
            if following.phase is Phase.SPAN:
                stack.append(following)
            elif following.phase is Phase.FREE and following.last_span.operator:
                closed = following.last_span
                a, b, *via = (_text_of(served, label) for label in closed.labels)
                found.add((closed.operator, a, b, tuple(via), closed.entity_ids))
    starts = [
        json.loads(line)[1]
        for line in (tmp_path / "dated" / "entity_labels.jsonl").read_text().splitlines()
    ]
    relations = ["director", "date of birth", "publication date", "duration"]
    vias = [(r,) for r in relations] + [(r, s) for r in relations for s in relations]
    expected = set()
    for op in ("earlier", "later"):
        for a in starts:
            for b in starts:
                for via in vias:
                    result = executor.run(Compare(op, (Start(a), Start(b)), via))
                    if result.status == "value":
                        expected.add((op, a, b, via, (f"Q{result.answer}",)))
                    elif result.status == "undetermined":
                        both = tuple(sorted({f"Q{s}" for s in result.starts}))
                        expected.add((op, a, b, via, both))
    assert len(expected) > 10
    assert found == expected


def test_a_comparison_offers_the_relations_both_operands_have_and_closes_on_dates(tmp_path):
    index, served, _ = dated_world(tmp_path)
    oracle = Oracle(index, served, SpanSyntax(), **EXPRESSIONS)
    closer, separator, operand = (served.encode(d)[0] for d in ("⟧", " ·", " |"))
    first = _walk_text(oracle, served, "⟦ earlier: Jaws")
    allowed = oracle.allowed(first)
    assert operand in allowed and closer not in allowed and separator not in allowed
    second = _walk_text(oracle, served, "⟦ earlier: Jaws | Star Wars")
    allowed = oracle.allowed(second)
    assert separator in allowed and closer not in allowed and operand not in allowed
    directors = _walk_text(oracle, served, "⟦ earlier: Jaws | Star Wars · director")
    allowed = oracle.allowed(directors)  # the directors are no dates, but have triples
    assert separator in allowed and closer not in allowed
    dated = _walk_text(oracle, served, "⟦ earlier: Jaws | Star Wars · publication date")
    allowed = oracle.allowed(dated)  # dates, which have no triples
    assert closer in allowed and separator not in allowed
    # Jaws has a director and a publication date, George Lucas only a date of birth: no
    # relation is shared, so only the escape marker is left.
    unshared = _walk_text(oracle, served, "⟦ earlier: Jaws | George Lucas ·")
    escape = " ⁇".encode()
    assert all(escape.startswith(served.token_bytes(t)) for t in oracle.allowed(unshared))
    with pytest.raises(TokenNotAllowed):
        _walk_text(oracle, served, "⟦ earlier: Jaws | George Lucas · date of birth")


def test_an_operator_starts_only_an_unbound_span(tmp_path):
    index, served, _ = dated_world(tmp_path)
    oracle = Oracle(index, served, SpanSyntax(), **EXPRESSIONS)
    assert _walk_text(oracle, served, "⟦ later:").operator == 2
    for refused in (
        "⟦ Jaws · director⟧ so ⟦ earlier:",  # the answer after a path is bound
        "⟦ earlier: later:",  # one operator, at the start
        "⟦ earlier: Jaws | Star Wars | ",  # two operands
        "⟦ Jaws | ",  # no operand separator without an operator
    ):
        with pytest.raises(TokenNotAllowed):
            _walk_text(oracle, served, refused)
    paths = Oracle(index, served, SpanSyntax(), paths=True, committed_paths=True)
    with pytest.raises(TokenNotAllowed):  # without expressions there are no operators
        _walk_text(paths, served, "⟦ earlier:")
    escaped = _walk_text(oracle, served, "⟦ earlier: Jaws | ⁇ a remake⟧")
    assert escaped.phase is Phase.FREE and not escaped.last_span.resolved


def compare_kg():
    """A dated KG with the hard cases of a comparison: overlapping dates (Alien's 1979 holds
    Hair's 15 June 1979), an operand with two dates (Duel, 13 November 1971 and 1983), and an
    ambiguous start (Jaws) whose best-connected sense, the novel, reaches no date by its
    publication date (an unknown value) while the film does: (KG, dates, tokenizer)."""
    from xgraph.expressions import stored_date

    entities = {
        "Q1": "Alien", "Q2": "Hair", "Q3": "Jaws", "Q4": "Jaws", "Q5": "Duel",
        "Q50": "unknown value", "Q60": "Peter Benchley", "Q61": "Ann", "Q62": "Bo",
        "Q100": "1979", "Q101": "15 June 1979", "Q102": "June 1975", "Q103": "13 November 1971",
        "Q104": "1983", "Q70": "United States", "Q71": "United Kingdom", "Q72": "France",
    }  # fmt: skip
    dates = {
        "Q100": stored_date(1979, 1, 1),
        "Q101": stored_date(1979, 6, 15),
        "Q102": stored_date(1975, 6, 1),
        "Q103": stored_date(1971, 11, 13),
        "Q104": stored_date(1983, 1, 1),
    }
    triples = (
        ("Q1", "P577", "Q100"), ("Q2", "P577", "Q101"), ("Q4", "P577", "Q102"),
        ("Q5", "P577", "Q103"), ("Q5", "P577", "Q104"), ("Q3", "P577", "Q50"),
        ("Q3", "P50", "Q60"), ("Q3", "P50", "Q61"), ("Q3", "P50", "Q62"),
        ("Q1", "P495", "Q70"), ("Q1", "P495", "Q71"), ("Q2", "P495", "Q70"),
        ("Q4", "P495", "Q70"), ("Q5", "P495", "Q72"),
    )  # fmt: skip
    kg = KnowledgeGraph(
        entities=tuple(Entity(q, (label,)) for q, label in entities.items()),
        relations=(
            Relation("P577", ("publication date",)),
            Relation("P50", ("author",)),
            Relation("P495", ("country of origin",)),
        ),
        triples=triples,
    )
    return kg, dates, ServedTokenizer(compare_tokenizer(kg))


def compare_tokenizer(kg: KnowledgeGraph):
    """compare_kg's tokenizer, trained on its labels and a few comparisons."""
    texts = [
        "⟦ earlier: Jaws | Alien · publication date⟧ ⟦ later: Duel | Hair⟧ ⟦ ⁇ no⟧",
        "⟦ same: Alien | Hair · country of origin⟧ so ⟦ yes⟧ ⟦ no⟧",
    ]
    labels = [label for item in (*kg.entities, *kg.relations) for label in item.labels]
    return train_byte_level_bpe(labels + texts, vocab_size=400)


def test_comparisons_bind_both_operands_on_overlap_and_compare_earliest_dated_starts():
    kg, dates, served = compare_kg()
    index = build_kg_index(kg, served, SpanSyntax(), dates=dates)
    oracle = Oracle(index, served, SpanSyntax(), **EXPRESSIONS)
    cases = {
        # Overlap: neither 1979 nor 15 June 1979 is before the other, so both bind.
        "⟦ earlier: Alien | Hair · publication date⟧": ("Q1", "Q2"),
        "⟦ later: Hair | Alien · publication date⟧": ("Q1", "Q2"),
        # Duel is compared by its earliest date, 13 November 1971.
        "⟦ later: Duel | Alien · publication date⟧": ("Q1",),
        "⟦ earlier: Alien | Duel · publication date⟧": ("Q5",),
        # Jaws commits to the film: the novel ranks first but reaches no date.
        "⟦ earlier: Jaws | Alien · publication date⟧": ("Q4",),
        "⟦ later: Jaws | Duel · publication date⟧": ("Q4",),
        # The same operand twice: its dates are equal, so it binds alone.
        "⟦ earlier: Alien | Alien · publication date⟧": ("Q1",),
    }
    for text, answer in cases.items():
        closed = _walk_text(oracle, served, text).last_span
        assert closed.entity_ids == answer, text
    state = _walk_text(oracle, served, "⟦ earlier: Alien | Hair · publication date⟧")
    for name, entity in (("Alien", "Q1"), ("Hair", "Q2")):
        assert walk_from(oracle, state, served, f" so ⟦ {name}⟧").last_span.entity_ids == (entity,)
    jaws = _walk_text(oracle, served, "⟦ earlier: Jaws | Alien · publication date⟧")
    assert jaws.last_span.operands == ("Q4", "Q1")


DATE_LABELS = {
    "Q20": ("1979", (1979, 1, 1)),
    "Q21": ("June 1979", (1979, 6, 1)),
    "Q22": ("15 June 1979", (1979, 6, 15)),
    "Q23": ("1980", (1980, 1, 1)),
    "Q24": ("2 March 1900", (1900, 3, 2)),
}
"""Dates for random dated KGs: nested (1979 holds June 1979, which holds 15 June 1979),
apart, and a label ("1979") that is a prefix of others."""


def random_dated_kg(seed: int):
    """random_versioned_kg with date entities (DATE_LABELS) as objects of random triples, and
    versions whose edits may name them: (KG, dates by entity id, versions)."""
    from xgraph.expressions import stored_date

    rng = random.Random(seed)
    kg, versions = random_versioned_kg(seed)
    dates = [Entity(q, (label,)) for q, (label, _) in DATE_LABELS.items()]
    ids = [e.id for e in kg.entities[:6]]
    relations = [r.id for r in VERSION_RELATIONS]
    dated = {
        (rng.choice(ids), rng.choice(relations), rng.choice(list(DATE_LABELS))) for _ in range(9)
    }
    for edits in versions.values():
        edits.append((rng.choice(ids), rng.choice(relations), rng.choice(list(DATE_LABELS))))
    kg = KnowledgeGraph(
        entities=(*kg.entities, *dates),
        relations=kg.relations,
        triples=tuple(sorted({*kg.triples, *dated})),
    )
    return kg, {q: stored_date(*value) for q, (_, value) in DATE_LABELS.items()}, versions


def dated_tokenizer() -> ServedTokenizer:
    labels = [
        *VERSION_LABELS,
        *(label for r in VERSION_RELATIONS for label in r.labels),
        *(label for label, _ in DATE_LABELS.values()),
    ]
    texts = ["⟦ earlier: Ann Lee | Bo Ray · born in · born⟧ ⟦ later: Cy⟧ ⟦ same: Lee⟧ ⟦ ⁇ no⟧"]
    return ServedTokenizer(train_byte_level_bpe([*labels, *texts], vocab_size=360))


class ExpressionWalk:
    """Random walk choices through allowed sets that write operators, operand separators,
    separators, and closers often, so that random walks write and close comparisons."""

    def __init__(self, served: ServedTokenizer, rng: random.Random) -> None:
        self.rng, self.queue = rng, []
        syntax = SpanSyntax()
        self.operators = [served.encode(syntax.label_prefix + op) for op in syntax.operators]
        self.opener = served.encode(syntax.opener)
        self.delimiters = [
            served.encode(d) for d in (syntax.closer, syntax.separator, syntax.operand_separator)
        ]
        written = [d.encode() for d in (syntax.closer, syntax.separator, syntax.escape)] + [
            syntax.operand_separator.encode()
        ]
        self.delimiter_like = {
            t
            for t in range(served.vocab_size)
            if served.token_bytes(t)
            and any(d.startswith(served.token_bytes(t), k) for d in written for k in range(len(d)))
        }

    def next(self, state, allowed: frozenset[int]) -> int:
        rng = self.rng
        if not self.queue and state.phase is Phase.FREE and rng.random() < 0.4:
            self.queue = list(self.opener)
        elif not self.queue and state.phase is Phase.ESCAPED and rng.random() < 0.3:
            self.queue = list(self.delimiters[0])
        elif not self.queue and state.phase is Phase.SPAN:
            if state.typed == () and not state.operator and rng.random() < 0.6:
                self.queue = list(rng.choice(self.operators))
            else:
                ready = [d for d in self.delimiters if d[0] in allowed]
                if ready and rng.random() < 0.6:
                    self.queue = list(rng.choice(ready))
        if self.queue and self.queue[0] in allowed:
            return self.queue.pop(0)
        self.queue = []
        label = sorted(allowed - self.delimiter_like)
        if state.phase is Phase.SPAN and label and rng.random() < 0.9:
            return rng.choice(label)
        return rng.choice(sorted(allowed))


@pytest.mark.parametrize("seed", range(3))
def test_a_kg_version_equals_the_rebuilt_index_for_comparisons_too(seed):
    """KG versions need no code of their own for expressions: a versioned oracle and an
    oracle on the index rebuilt from the edited triples agree on every step of random walks
    that write comparisons, and the walks close some."""
    kg, dates, versions = random_dated_kg(seed)
    served = dated_tokenizer()
    syntax = SpanSyntax()
    index = build_kg_index(kg, served, syntax, dates=dates, verdicts=True)
    versioned = Oracle(index, served, syntax, versions=versions, **EXPRESSIONS)
    walk = ExpressionWalk(served, random.Random(seed))
    compared = 0
    for name, edits in versions.items():
        rebuilt_index = build_kg_index(
            edited_kg(kg, edits), served, syntax, dates=dates, verdicts=True
        )
        rebuilt = Oracle(rebuilt_index, served, syntax, **EXPRESSIONS)
        for _ in range(10):
            ours, theirs = versioned.initial_state(name), rebuilt.initial_state()
            for _ in range(80):
                allowed = rebuilt.allowed(theirs)
                assert versioned.allowed(ours) == allowed
                assert replace(ours, version=None) == theirs
                token = walk.next(theirs, allowed)
                ours, theirs = versioned.advance(ours, token), rebuilt.advance(theirs, token)
                closed = theirs.last_span
                if closed is not None and closed.operator and closed is not ours.last_span:
                    compared += 1
    assert compared > 3, compared


def test_value_comparisons_that_can_close_are_exactly_those_the_executor_runs(tmp_path):
    """The exact-mask contract for value comparisons (`same:`): every one the oracle lets
    close, from the operator on, over the merged dated toy KG, is one that xgraph.expressions'
    executor runs to a value, and its verdict entity is the executor's yes or no."""
    import json

    from xgraph.expressions import Same, Start
    from xgraph.index import VERDICTS

    index, served, literal = dated_world(tmp_path)
    executor = dated_executor(literal, tmp_path)
    oracle = Oracle(index, served, SpanSyntax(), **EXPRESSIONS)
    found = set()
    stack = [walk(oracle, served.encode("⟦ same:"))]
    while stack:
        state = stack.pop()
        allowed = oracle.allowed(state)
        assert allowed, f"dead end in {state}"
        for token in allowed:
            following = oracle.advance(state, token)
            if following.phase is Phase.SPAN:
                stack.append(following)
            elif following.phase is Phase.FREE and following.last_span.operator:
                closed = following.last_span
                a, b, *via = (_text_of(served, label) for label in closed.labels)
                found.add((a, b, tuple(via), closed.entity_ids))
    starts = [
        json.loads(line)[1]
        for line in (tmp_path / "dated" / "entity_labels.jsonl").read_text().splitlines()
    ]
    relations = ["director", "date of birth", "publication date", "duration"]
    vias = [(r,) for r in relations] + [(r, s) for r in relations for s in relations]
    expected = set()
    for a in starts:
        for b in starts:
            for via in vias:
                result = executor.run(Same((Start(a), Start(b)), via))
                if result.status == "value":
                    verdict = VERDICTS["yes" if result.answer else "no"]
                    expected.add((a, b, via, (verdict,)))
    assert {verdict for *_, (verdict,) in expected} == set(VERDICTS.values())
    assert found == expected


def test_a_value_comparison_binds_its_verdict_and_needs_values_of_one_kind():
    """Entities are shared or not (Alien and Hair are both from the United States; Duel is
    French); dates are shared only when equal, so 1979 and 15 June 1979 are no shared date,
    though they overlap; Jaws commits to its best-connected sense with any value, the novel,
    whose unknown value is an entity where Alien has a date, so that comparison cannot close.
    The verdict binds the answer, and an unbound span cannot name a verdict."""
    from xgraph.index import VERDICTS

    kg, dates, served = compare_kg()
    index = build_kg_index(kg, served, SpanSyntax(), dates=dates, verdicts=True)
    oracle = Oracle(index, served, SpanSyntax(), **EXPRESSIONS)
    cases = {
        "⟦ same: Alien | Hair · country of origin⟧": "yes",
        "⟦ same: Duel | Hair · country of origin⟧": "no",
        "⟦ same: Jaws | Hair · country of origin⟧": "yes",  # the film: the novel has none
        "⟦ same: Alien | Alien · publication date⟧": "yes",
        "⟦ same: Alien | Hair · publication date⟧": "no",
    }
    for text, verdict in cases.items():
        state = _walk_text(oracle, served, text)
        assert state.last_span.operator == "same", text
        assert state.last_span.entity_ids == (VERDICTS[verdict],), text
        answered = walk_from(oracle, state, served, f" so ⟦ {verdict}⟧").last_span
        assert answered.entity_ids == (VERDICTS[verdict],)
        other = "no" if verdict == "yes" else "yes"
        with pytest.raises(TokenNotAllowed):
            walk_from(oracle, state, served, f" so ⟦ {other}⟧")
    mixed = _walk_text(oracle, served, "⟦ same: Jaws | Alien · publication date")
    assert served.encode("⟧")[0] not in oracle.allowed(mixed)
    assert _walk_text(oracle, served, "⟦ same: Jaws | Alien").last_span is None
    for refused in ("⟦ yes⟧", "⟦ no⟧"):
        with pytest.raises(TokenNotAllowed):
            _walk_text(oracle, served, refused)
    # Without verdicts in the index there is no value comparison to write.
    plain = build_kg_index(kg, served, SpanSyntax(), dates=dates)
    with pytest.raises(TokenNotAllowed):
        _walk_text(Oracle(plain, served, SpanSyntax(), **EXPRESSIONS), served, "⟦ same:")


def anchor_world(aliases=40):
    from types import SimpleNamespace

    from xgraph.engine import Engine

    served, syntax = ServedTokenizer(byte_level_tokenizer([])), SpanSyntax()
    kg = KnowledgeGraph(
        (
            Entity("BAD", ("A⟧B",)),
            Entity("A", tuple(f"A{k:03}" for k in range(aliases))),
            Entity("B", ("Z",)),
        ),
        (Relation("R", ("via",)),),
        (("A", "R", "B"),),
    )
    index = build_kg_index(kg, served, syntax)
    oracle = Oracle(index, served, syntax, paths=True, committed_paths=True, escapes=False)
    uploaded = []
    engine = Engine.__new__(Engine)
    engine._index, engine._entity_index = index, None
    engine._native = SimpleNamespace(set_reached=lambda slot, rows, stream: uploaded.append(rows))
    return index, served, syntax, oracle, engine, uploaded


@pytest.mark.parametrize("anchor", [[], ["UNKNOWN"], ["BAD"]])
@pytest.mark.parametrize("implementation", ["oracle", "engine"])
def test_anchor_without_writable_rows_is_rejected_before_admission(anchor, implementation):
    *_, oracle, engine, uploaded = anchor_world()
    with pytest.raises(ValueError, match=r"anchor.*writable"):
        oracle.anchored_state(anchor) if implementation == "oracle" else engine.anchor(0, anchor)
    assert uploaded == []


@pytest.mark.parametrize("aliases", [40, 65])
def test_duplicate_anchor_ids_do_not_consume_the_native_row_cap(aliases):
    index, _, _, oracle, engine, uploaded = anchor_world(aliases)
    engine.anchor(0, ["A", "A", "B", "UNKNOWN"])
    expected = oracle.anchored_state(["A", "A", "B", "UNKNOWN"]).reached
    rows = list(map(int, uploaded[0]))
    assert rows == sorted(set(rows)) and len(rows) == aliases + 1
    assert frozenset(rows[:MAX_REACHED]) == expected
    if aliases == 40:
        assert {index.labels.owner_id(row) for row in rows} == {"A", "B"}


def test_answer_mass_unknown_anchor_is_rejected_before_search():
    from xgraph.certification import OraclePolicy

    base = mass_world()
    with pytest.raises(ValueError, match="anchor has no writable label rows"):
        OraclePolicy(base.index, base.tokenizer, base.syntax, base.weights, anchor=("missing",))
