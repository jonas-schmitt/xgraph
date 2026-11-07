"""CPU certificates for the mode of a finite, locally masked executed-answer policy.

This module never changes an oracle mask. It certifies the represented-weight
policy, not the real-valued neural softmax or the factual correctness of an ID.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import count

from xgraph import oracle as oracle_module
from xgraph.index import KGIndex
from xgraph.oracle import Oracle, OracleState, Phase, Segment
from xgraph.syntax import SpanSyntax
from xgraph.tokenizer import ServedTokenizer

FAILURES = frozenset(
    f"failure:{name}"
    for name in ("no_path", "empty", "horizon", "zero_weight", "early_eos", "ambiguous")
)
WeightFunction = Callable[[tuple[int, ...], frozenset[int]], Mapping[int, int | float | Fraction]]


@dataclass
class OraclePolicy:
    """One opened, optionally anchored path, forced observations and a bound ID answer.

    The weights callback is pure and history-sensitive. It supplies *every*
    legal token, including zero weights. A singleton mask has probability one.
    The index, callback and oracle settings must remain immutable during search.
    """

    index: KGIndex
    tokenizer: ServedTokenizer
    syntax: SpanSyntax
    weights: WeightFunction
    horizon: int = 64
    anchor: tuple[str, ...] = ()
    versions: Mapping[str, tuple[tuple[str, str, str], ...]] | None = None
    version: str | None = None
    oracle: Oracle = field(init=False, repr=False)
    initial: OracleState = field(init=False)
    universe: frozenset[str] = field(init=False)
    cap: int = field(init=False)
    hop_cap: int = field(init=False)
    graph_scans: int = field(init=False, default=0)
    _adjacency: dict[str, set[str]] | None = field(init=False, default=None, repr=False)
    _reach_cache: dict = field(init=False, default_factory=dict, repr=False)
    _frontier_cache: dict = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self):
        if type(self.horizon) is not int or self.horizon < 0:
            raise ValueError("horizon must be a nonnegative integer")
        self.anchor = tuple(self.anchor)
        self.versions = {name: tuple(edits) for name, edits in (self.versions or {}).items()}
        self.cap, self.hop_cap = oracle_module.MAX_REACHED, oracle_module.MAX_PATH_RELATIONS
        self.oracle = Oracle(
            self.index,
            self.tokenizer,
            self.syntax,
            paths=True,
            committed_paths=True,
            escapes=False,
            observe=1,
            observe_answer=True,
            eos_token_ids=sorted(self.tokenizer.special_ids)[:1],
            versions=self.versions,
        )
        state = (
            self.oracle.anchored_state(list(self.anchor), self.version)
            if self.anchor
            else self.oracle.initial_state(self.version)
        )
        self.initial = self.oracle.opened_state(state)
        self.universe = frozenset(f"answer:{e}" for e in self.index.labels.owner_ids) | FAILURES

    def check_pinned(self):
        if (self.cap, self.hop_cap) != (
            oracle_module.MAX_REACHED,
            oracle_module.MAX_PATH_RELATIONS,
        ):
            raise ValueError("oracle caps changed during the fixed policy")

    def outcome(self, history, state):
        """Terminal partition; EOS completion takes priority over the token horizon."""
        self.check_pinned()
        if history and history[-1] in self.tokenizer.special_ids:
            if not state.answer_complete:
                return "failure:early_eos"
            span = state.last_span
            if span is None or not span.resolved or len(span.entity_ids) != 1:
                return "failure:ambiguous"
            return f"answer:{span.entity_ids[0]}"
        if state.phase is Phase.FREE and not state.answer_pos and not state.answer_complete:
            return "failure:no_path"
        if len(history) >= self.horizon:
            return "failure:horizon"
        return None

    def probabilities(self, history, allowed):
        """Exact normalization of finite represented weights; no top-k renormalization."""
        if len(allowed) == 1:
            return {next(iter(allowed)): Fraction(1)}
        weights = self.weights(history, allowed)
        if weights.keys() != allowed:
            raise ValueError("weights must cover every legal token, with no omitted tail")
        try:
            exact = {token: Fraction(weight) for token, weight in weights.items()}
        except (ValueError, OverflowError, TypeError) as error:
            raise ValueError("weights must be finite represented numbers") from error
        if any(weight < 0 for weight in exact.values()):
            raise ValueError("weights must be nonnegative")
        total = sum(exact.values(), Fraction())
        return {token: weight / total for token, weight in exact.items() if weight} if total else {}

    def _build_adjacency(self):
        """A lazy ID graph from public index arrays, independently of oracle helpers."""
        labels, relations = self.index.labels, self.index.relations
        owners = {labels.owner_id(row) for row in range(labels.num_rows)}
        writable_relations = {relations.owner_id(row) for row in range(relations.num_rows)}
        edges = set()
        for number, subject in enumerate(labels.owner_ids):
            lo, hi = self.index.adjacency_offsets[number : number + 2]
            for edge in range(int(lo), int(hi)):
                self.graph_scans += 1
                relation = relations.owner_ids[int(self.index.edge_relations[edge])]
                obj = labels.owner_id(int(self.index.edge_objects[edge]))
                edges.add((subject, relation, obj))
        edits = self.versions.get(self.version, ())
        replaced = {(s, r) for s, r, _ in edits}
        edges = {edge for edge in edges if edge[:2] not in replaced}
        edges.update(
            (s, r, obj)
            for s, r, obj in edits
            if s in owners and obj in owners and r in writable_relations
        )
        self._adjacency = defaultdict(set)
        for subject, _, obj in edges:
            self._adjacency[subject].add(obj)

    def reachable(self, state):
        """An uncapped envelope of all legal future answers, plus conservative failures.

        Every live trajectory still receives the oracle's ordinary cap. This
        envelope follows *all* relations from its live state and never recaps
        their union. Current incomplete labels may only make it too large.
        """
        self.check_pinned()
        last = state.last_span
        answer = state.answer_complete or (last is not None and (len(last.labels) > 1 or last.path))
        if answer:
            seeds = {self.index.labels.owner_id(row) for row in state.reached or ()}
            hops = 0
        elif state.segment is Segment.RELATION:
            # The oracle executes a relation lazily: subjects still precedes the
            # current label even after its forced observation. Enumerate possible
            # labels, cap EACH execution, then union without recapping.
            labels = set()
            degree = self.oracle._graph(state).degree
            scans = sum(degree.get(entity, 0) for entity in state.subjects)
            if state.typed is not None:
                self.graph_scans += scans
                labels.update(
                    tokens
                    for tokens, _, _ in self.oracle._candidates(
                        state, state.typed, exact=state.observed
                    )
                )
            for progress, label in (
                (state.closing, state.closing_label),
                (state.separating, state.separating_label),
                (state.observing, state.observing_label),
            ):
                if progress is not None:
                    labels.add(label)
            seeds = set()
            for label in labels:
                key = (state.version, state.subjects, state.ranks, label)
                if key not in self._frontier_cache:
                    self.graph_scans += scans
                    self._frontier_cache[key] = frozenset(self.oracle._frontier(state, label))
                seeds.update(self._frontier_cache[key])
            hops = max(0, self.hop_cap - len(state.written))
        else:
            seeds = (
                {self.index.labels.owner_id(row) for row in state.reached}
                if state.reached is not None
                else set(self.index.labels.owner_ids)
            )
            hops = self.hop_cap
        key = (frozenset(seeds), hops)
        if key not in self._reach_cache:
            if hops and self._adjacency is None:
                self._build_adjacency()
            reached, layer = set(seeds), set(seeds)
            for _ in range(hops):
                following = set()
                for subject in layer:
                    neighbors = self._adjacency.get(subject, ())
                    self.graph_scans += len(neighbors)
                    following.update(neighbors)
                layer = following - reached
                reached.update(following)
                if not layer:
                    break
            self._reach_cache[key] = frozenset(f"answer:{e}" for e in reached) | FAILURES
        return self._reach_cache[key]


@dataclass(frozen=True)
class Pending:
    history: tuple[int, ...]
    state: OracleState
    low: Fraction
    high: Fraction
    reachable: frozenset[str]


@dataclass(frozen=True)
class Snapshot:
    lower: dict[str, Fraction]
    upper: dict[str, Fraction]
    pending: tuple[Pending, ...]
    winner: str | None
    expansions: int
    forward_steps: int
    forced_steps: int
    graph_scans: int


def enclose(low, high, bits):
    """Round outwards on an exact dyadic grid; None keeps exact rational masses."""
    if bits is None:
        return low, high
    scale = 1 << bits
    return Fraction(int(low * scale), scale), Fraction(-int(-(high * scale) // 1), scale)


def search(policy, *, bound="reachable", rounding_bits=None, max_expansions=10000):
    """Yield prefix-free search snapshots, retaining every unexplored branch."""
    if bound not in ("reachable", "global"):
        raise ValueError("bound must be reachable or global")
    if rounding_bits is not None and (type(rounding_bits) is not int or rounding_bits < 0):
        raise ValueError("rounding_bits must be None or a nonnegative integer")
    if type(max_expansions) is not int or max_expansions < 0:
        raise ValueError("max_expansions must be a nonnegative integer")
    policy.check_pinned()
    serial = count()
    queue = []
    completed_low, completed_high = defaultdict(Fraction), defaultdict(Fraction)

    def put(history, state, low, high):
        terminal = policy.outcome(history, state)
        if terminal:
            completed_low[terminal] += low
            completed_high[terminal] += high
        else:
            reach = policy.reachable(state) if bound == "reachable" else policy.universe
            node = Pending(history, state, low, high, reach)
            heapq.heappush(queue, (-high, history, next(serial), node))

    put((), policy.initial, Fraction(1), Fraction(1))
    expansions = forwards = forced = 0
    while True:
        if bound == "global":
            remainder = sum((item[3].high for item in queue), Fraction())
            upper = {e: completed_high.get(e, 0) + remainder for e in policy.universe}
        else:
            upper = dict.fromkeys(policy.universe, Fraction())
            upper.update(completed_high)
            for _, _, _, node in queue:
                for outcome in node.reachable:
                    upper[outcome] += node.high
        lower = dict(completed_low)
        winner = strict_winner(lower, upper)
        yield Snapshot(
            lower,
            upper,
            tuple(item[3] for item in queue),
            winner,
            expansions,
            forwards,
            forced,
            policy.graph_scans,
        )
        if winner or not queue or expansions >= max_expansions:
            return
        _, _, _, node = heapq.heappop(queue)
        expansions += 1
        allowed = policy.oracle.allowed(node.state)
        probabilities = policy.probabilities(node.history, allowed) if allowed else {}
        if not probabilities:
            outcome = "failure:zero_weight" if allowed else "failure:empty"
            completed_low[outcome] += node.low
            completed_high[outcome] += node.high
            continue
        forwards += 1  # Forced observations still need a forward/KV update.
        forced += len(allowed) == 1
        for token, probability in sorted(probabilities.items()):
            low, high = enclose(node.low * probability, node.high * probability, rounding_bits)
            following = policy.oracle.advance(node.state, token)
            put((*node.history, token), following, low, high)


def strict_winner(lower, upper):
    """Return an outcome only when its lower bound beats all other upper bounds."""
    for outcome, mass in sorted(lower.items()):
        if mass > 0 and all(mass > bound for other, bound in upper.items() if other != outcome):
            return outcome
    return None
