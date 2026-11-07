"""Historical four-way Holm simulator. Not the registered fixed-sequence analysis."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from xgraph.cost import StageTimer, request_cost
from xgraph.editing import case_clusters, case_members, update_events
from xgraph.provenance import provenance
from xgraph.score import load_labels, record_text, replay_failure, spans, wwq_accuracy

ENDPOINTS = (
    "H1.question",
    "H1.any",
    "H2.question",
    "H2.any",
    "H2.tokens",
    "H2.gpu_seconds",
    "H3.question",
    "H4.t",
    "H4.fidelity",
)
HYPOTHESES = ("H1", "H2", "H3", "H4")
COMPONENTS = ((0, 1), (2, 3, 4, 5), (6,), (7, 8))
NULLS = np.array([0, -0.03, -0.03, -0.03, 3, 3, 0, -0.03, -0.03])
LEVELS = 0.05 / np.arange(4, 0, -1)


def decision_rule(pvalues: np.ndarray, effects: np.ndarray) -> dict:
    """Conjunction p-values, Holm within each family, then H1's practical gate.

    The last two axes are model families and ENDPOINTS; preceding axes may index
    simulated studies. Missing endpoints must be completed before calling this.
    """
    pvalues, effects = np.asarray(pvalues), np.asarray(effects)
    if pvalues.shape != effects.shape or pvalues.shape[-1] != len(ENDPOINTS):
        raise ValueError("p-values and effects need matching (..., families, 9) shapes")
    if not np.isfinite(pvalues).all() or ((pvalues < 0) | (pvalues > 1)).any():
        raise ValueError("p-values must be finite and between zero and one")
    hypotheses = np.stack([pvalues[..., list(c)].max(axis=-1) for c in COMPONENTS], axis=-1)
    order = np.argsort(hypotheses, axis=-1, kind="stable")
    sorted_p = np.take_along_axis(hypotheses, order, axis=-1)
    sorted_pass = np.logical_and.accumulate(sorted_p <= LEVELS, axis=-1)
    passes = np.take_along_axis(sorted_pass, np.argsort(order, axis=-1), axis=-1)
    passes[..., 0] &= effects[..., 0] >= 0.05 - 1e-12
    both = passes.all(axis=-2)
    return {
        "hypothesis_pvalues": hypotheses,
        "passes": passes,
        "both_families": both,
        "all_pass": both.all(axis=-1),
    }


def bootstrap_test(
    numerators: np.ndarray,
    denominators: np.ndarray,
    clusters: list[str],
    nulls: np.ndarray,
    *,
    resamples: int = 100_000,
    seed: int = 0,
    rng: np.random.Generator | None = None,
    batch_size: int = 256,
    crossed: list[str] | None = None,
    count_cache: dict | None = None,
) -> dict:
    """Paired cluster bootstrap of ratios of sums, with null-centred upper tails.

    Arrays are (cases, families, endpoints). All endpoints and families share
    each cluster draw. With crossed, independent sequence and event cluster
    multiplicities are multiplied (pigeonhole bootstrap), never intersections.
    Four stored critical deviations suffice to invert any Holm step later.
    Unknown columns remain NaN. Memory is bounded by batches plus the bootstrap
    statistics, rather than (resamples, cases, families, endpoints).
    """
    n, d = np.asarray(numerators, dtype=float), np.asarray(denominators, dtype=float)
    if n.ndim != 3 or n.shape != d.shape or not len(n) or len(clusters) != len(n):
        raise ValueError("bootstrap needs nonempty matching (cases, families, endpoints) arrays")
    if resamples < 1 or batch_size < 1:
        raise ValueError("resamples and batch_size must be positive")
    if crossed is not None and len(crossed) != len(n):
        raise ValueError("crossed cluster labels must cover every case")
    rng = np.random.default_rng(seed) if rng is None else rng
    shape = n.shape[1:]
    n, d = n.reshape(len(n), -1), d.reshape(len(d), -1)
    known = np.isfinite(n).all(axis=0) & np.isfinite(d).all(axis=0)
    estimate = np.full(n.shape[1], np.nan)
    pvalues = np.full_like(estimate, np.nan)
    critical = np.full((len(estimate), 4), np.nan)
    if not known.any():
        return {
            "estimate": estimate.reshape(shape),
            "pvalues": pvalues.reshape(shape),
            "critical": critical.reshape(*shape, 4),
        }
    n, d = n[:, known], d[:, known]
    if (d <= 0).any():
        raise ValueError("known bootstrap denominators must be strictly positive for every case")
    observed = n.sum(axis=0) / d.sum(axis=0)
    _, group = np.unique(clusters, return_inverse=True)
    k = int(group.max()) + 1
    if crossed is None:
        sums = np.zeros((k, n.shape[1] * 2))
        np.add.at(sums, group, np.concatenate([n, d], axis=1))
    else:
        _, event = np.unique(crossed, return_inverse=True)
        events = int(event.max()) + 1
        sums = np.concatenate([n, d], axis=1)
    # Refuse an oversized statistic cache before allocating it. This leaves
    # ample room below the task's process limit, including the input panels.
    if resamples * len(observed) * 8 > 512 * 1024**2:
        raise ValueError("bootstrap statistics exceed 512 MiB; reduce resamples or families")
    batch_size = min(batch_size, max(1, 32 * 1024**2 // (8 * max(k, len(n)))))

    def counts(size: int, start: int, stop: int, dimension: str):
        # Common Monte Carlo draws at the fixed design, shared by simulated
        # studies. Compact integer counts, with a bounded total cache.
        key = (dimension, size, resamples)
        dtype = np.uint16 if size <= np.iinfo(np.uint16).max else np.uint32
        if count_cache is not None and key not in count_cache:
            occupied = sum(v.nbytes for v in count_cache.values())
            if occupied + resamples * size * np.dtype(dtype).itemsize <= 256 * 1024**2:
                cached = np.empty((resamples, size), dtype=dtype)
                for begin in range(0, resamples, batch_size):
                    end = min(begin + batch_size, resamples)
                    cached[begin:end] = rng.multinomial(
                        size, np.full(size, 1 / size), size=end - begin
                    )
                count_cache[key] = cached
        if count_cache is not None and key in count_cache:
            return count_cache[key][start:stop].astype(float)
        return rng.multinomial(size, np.full(size, 1 / size), size=stop - start).astype(float)

    deviations = np.empty((resamples, len(observed)))
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        weights = counts(k, start, stop, "main")
        if crossed is not None:
            other = counts(events, start, stop, "crossed")
            weights = weights[:, group] * other[:, event]
            # Very small crossed fixtures can select no occupied cell. Redraw
            # these empty replicates rather than turning an undefined ratio
            # into a successful test.
            empty = weights.sum(axis=1) == 0
            while empty.any():
                w = rng.multinomial(k, np.full(k, 1 / k), size=int(empty.sum()))
                v = rng.multinomial(events, np.full(events, 1 / events), size=int(empty.sum()))
                weights[empty] = w[:, group] * v[:, event]
                empty = weights.sum(axis=1) == 0
        totals = weights @ sums
        width = len(observed)
        deviations[start:stop] = totals[:, :width] / totals[:, width:] - observed
    null = np.broadcast_to(nulls, shape).ravel()[known]
    # The +1 correction avoids reporting zero p-values from a finite run.
    pvalues[known] = (1 + (deviations >= observed - null).sum(axis=0)) / (resamples + 1)
    estimate[known] = observed
    indices = resamples - np.floor(LEVELS * (resamples + 1) - 1).astype(int) - 1
    valid = indices < resamples
    bounds = np.full((len(observed), 4), np.inf)
    if valid.any():
        partitioned = np.partition(deviations, indices[valid], axis=0)
        bounds[:, valid] = partitioned[indices[valid]].T
    critical[known] = bounds
    return {
        "estimate": estimate.reshape(shape),
        "pvalues": pvalues.reshape(shape),
        "critical": critical.reshape(*shape, 4),
    }


class _Inputs:
    def __init__(self, config: Path):
        self.base = config.resolve().parent
        self.paths: set[Path] = {config.resolve(), Path(__file__).resolve()}

    def path(self, value: str) -> Path:
        path = (self.base / value).resolve()
        self.paths.add(path)
        return path

    def read(self, value: str):
        return json.loads(self.path(value).read_text(encoding="utf-8"))


@dataclass
class _Panel:
    cases: list[str]
    sequences: list[str]
    events: list[str] | None
    weights: np.ndarray
    right: np.ndarray  # cases, families, P/X/A, paraphrases (or joint fidelity)
    cost: np.ndarray  # cases, families, P/X/A, tokens/GPU-seconds (sums over questions)
    description: dict

    @cached_property
    def sequence_members(self):
        return self.members(self.sequences)

    @cached_property
    def event_members(self):
        return self.members(self.events) if self.events is not None else None

    @staticmethod
    def members(labels):
        groups = {}
        for i, label in enumerate(labels):
            groups.setdefault(label, []).append(i)
        return {s: np.array(group) for s, group in sorted(groups.items())}

    @cached_property
    def by_hops(self):
        groups = {}
        for sequence in self.sequence_members:
            groups.setdefault(sequence.count(">"), []).append(sequence)
        return groups


@dataclass
class _Population:
    cases: list[str]
    sequences: list[str]
    events: list[str] | None

    def summary(self) -> dict:
        return {
            "cases": len(self.cases),
            "sequences": len(set(self.sequences)),
            "events": len(set(self.events)) if self.events is not None else None,
        }


def _flag(value, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context}: correctness must be a boolean")
    return value


def _cost(value, context: str) -> float:
    if value is None:
        return np.nan
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}: cost must be a nonnegative finite number or null")
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{context}: cost must be a nonnegative finite number or null")
    return float(value)


def _read_panel(
    spec: dict, families: list[str], inputs: _Inputs, *, updates=False, temporal=False
) -> _Panel:
    questions = inputs.read(spec["questions"])
    qids = [q["id"] for q in questions]
    if not qids or len(set(qids)) != len(qids):
        raise ValueError(f"{spec['questions']}: questions must be nonempty and have unique ids")
    members = {q: [q] for q in qids} if updates else case_members(questions)
    cases = list(members)
    if not updates and any(len(ids) != 3 for ids in members.values()):
        raise ValueError("CF and T panels must ask exactly three questions per case")
    sequences = (
        ["updates"] * len(cases)
        if updates
        else case_clusters(inputs.read(spec["clusters"]), members, cases)
    )
    events = (
        case_clusters(inputs.read(spec["events"]), members, cases) if spec.get("events") else None
    )
    if (updates or temporal) and events is None:
        raise ValueError("temporal and update fidelity panels need event clusters")
    weights = inputs.read(spec["panel"]) if spec.get("panel") else {c: 1 for c in cases}
    weights = weights.get("weights", weights)
    missing = set(cases) - weights.keys()
    if missing:
        raise ValueError(f"panel weights missing cases: {sorted(missing)[:5]}")
    weights = np.array([weights[c] for c in cases], dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("case weights must be finite and positive")
    raw = bool(spec.get("arms"))
    if raw == bool(spec.get("outcomes")):
        raise ValueError(
            "each panel needs exactly one of arms (records) or outcomes (paired scores)"
        )
    data = spec["arms"] if raw else inputs.read(spec["outcomes"])
    if set(data) - set(families):
        raise ValueError(f"panel names unexpected families: {sorted(set(data) - set(families))}")
    gold, labels = None, None
    if raw:
        if updates:
            raise ValueError("updates need per-question scored current/rollback/off-chain fidelity")
        gold = inputs.read(spec["gold_new"])
        gold = {
            "answers": {q: v for q, v in gold["answers"].items() if q in set(qids)},
            "failed": {q: v for q, v in gold.get("failed", {}).items() if q in set(qids)},
        }
        wanted = {e for answer in gold["answers"].values() for e in answer["entities"]}
        paths = spec["labels"] if isinstance(spec["labels"], list) else [spec["labels"]]
        labels = load_labels([inputs.path(p) for p in paths], wanted)
    k = 1 if updates else 3
    right = np.full((len(cases), len(families), 3, k), np.nan)
    cost = np.full((len(cases), len(families), 3, 2), np.nan)
    description = {}
    for f, family in enumerate(families):
        if family not in data:
            continue
        needed = {"P", "A"} if updates or temporal else {"P", "X", "A"}
        if not needed <= data[family].keys():
            raise ValueError(f"{family}: panel missing arms {sorted(needed - data[family].keys())}")
        description[family] = {}
        for a, arm in enumerate(("P", "X", "A")):
            if arm not in data[family]:
                continue
            records = data[family][arm]
            if raw:
                records = {}
                with inputs.path(data[family][arm]).open(encoding="utf-8") as file:
                    for line in file:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if row["id"] in records:
                            raise ValueError(f"{family}/{arm}: duplicate record id {row['id']}")
                        records[row["id"]] = row
                texts = {q: record_text(records.get(q, {})) for q in qids}
                texts = {q: t if isinstance(t, str) else "" for q, t in texts.items()}
                scored = wwq_accuracy(texts, gold, labels)["correct"]
            flags, costs = {}, {}
            failures = 0
            for q in qids:
                context = f"{family}/{arm}/{q}"
                value = records.get(q, {})
                if isinstance(value, bool):
                    value = {"correct": value}
                if not isinstance(value, dict):
                    raise ValueError(f"{context}: outcome must be a boolean or an object")
                if updates:
                    fidelity = value.get("fidelity", [False] * 3)
                    if not isinstance(fidelity, list) or len(fidelity) != 3:
                        raise ValueError(
                            f"{context}: fidelity needs current, rollback and off-chain booleans"
                        )
                    flags[q] = all([_flag(v, context) for v in fidelity])
                else:
                    flags[q] = (
                        scored.get(q, False) if raw else _flag(value.get("correct", False), context)
                    )
                text = value.get("text")
                invalid = (
                    replay_failure(value)
                    or any(
                        value.get(key)
                        for key in ("declined", "truncated", "escaped", "malformed", "error")
                    )
                    or value.get("finish_reason")
                    in {"length", "max_tokens", "error", "abort", "declined"}
                    or (isinstance(text, str) and any(kind == "escaped" for kind, _ in spans(text)))
                )
                if invalid:
                    flags[q] = False
                    failures += 1
                usage = request_cost(value) if raw else {}
                tokens = value.get("total_tokens")
                if (
                    tokens is None
                    and raw
                    and all(usage.get(t) is not None for t in ("prompt_tokens", "generated_tokens"))
                ):
                    tokens = usage["prompt_tokens"] + usage["generated_tokens"]
                costs[q] = [_cost(tokens, context), _cost(value.get("gpu_seconds"), context)]
            for c, case in enumerate(cases):
                right[c, f, a] = [flags[q] for q in members[case]]
                cost[c, f, a] = np.array([costs[q] for q in members[case]]).sum(axis=0)
            mean = right[:, f, a].mean(axis=-1)
            description[family][arm] = {
                "source": data[family][arm] if raw else spec["outcomes"],
                "missing_records": len(set(qids) - records.keys()),
                "protocol_failures": failures,
                "per_question": float(mean.mean()),
                "per_question_weighted": float(np.average(mean, weights=weights)),
                "any_of_three_weighted": float(
                    np.average(right[:, f, a].max(axis=-1), weights=weights)
                ),
                "cost_known_cases": dict(
                    zip(
                        ("tokens", "gpu_seconds"),
                        np.isfinite(cost[:, f, a]).sum(axis=0).tolist(),
                        strict=True,
                    )
                ),
            }
    return _Panel(
        cases,
        list(map(str, sequences)),
        list(map(str, events)) if events else None,
        weights,
        right,
        cost,
        description,
    )


def _read_population(spec: dict, inputs: _Inputs) -> _Population:
    registry = inputs.read(spec["registry"])
    records = dict(registry["cases"])
    ids = registry["strata"][spec["stratum"]]["ids"]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"{spec['stratum']}: population ids must be nonempty and unique")
    sequences = [records[c]["relation_sequence"] for c in ids]
    events = None
    if spec.get("events"):
        mapping = inputs.read(spec["events"])
        events = [str(mapping[c]) for c in ids]
    elif spec.get("data"):
        cases = inputs.read(spec["data"])
        selected = set(ids)
        mapping = update_events(
            {f"mquake-{c['case_id']}": c for c in cases if f"mquake-{c['case_id']}" in selected}
        )
        events = [mapping[c] for c in ids]
    return _Population(ids, sequences, events)


def _transport(panel: _Panel, population: _Population, rng, cache: dict, *, events=False):
    source = panel.events if events else panel.sequences
    target = population.events if events else population.sequences
    if source is None or target is None:
        raise ValueError("event transport requires source and target event clusters")
    members = panel.event_members if events else panel.sequence_members
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(target):
        groups.setdefault(s, []).append(i)
    draw = np.empty(len(target), dtype=int)
    for s, indices in groups.items():
        key = (id(panel), s, events)
        if key not in cache:
            pool = list(members) if events else panel.by_hops.get(s.count(">"), list(members))
            pool = pool or list(members)
            cache[key] = s if s in members else pool[rng.integers(len(pool))]
        donor = members[cache[key]]
        probability = panel.weights[donor] / panel.weights[donor].sum()
        draw[indices] = rng.choice(donor, size=len(indices), replace=True, p=probability)
    return draw


def _case_statistics(panel: _Panel, draw: np.ndarray, endpoints: list[int]):
    right, costs = panel.right[draw], panel.cost[draw]
    p, x, a = (right[:, :, i] for i in range(3))
    question_x = (p - x).mean(axis=-1)
    question_a = (p - a).mean(axis=-1)
    values = {
        0: question_x,
        1: p.max(axis=-1) - x.max(axis=-1),
        2: question_a,
        3: p.max(axis=-1) - a.max(axis=-1),
        4: costs[:, :, 2, 0],
        5: costs[:, :, 2, 1],
        6: question_x,
        7: question_a,
        8: question_a,
    }
    n = np.stack([values[e] for e in endpoints], axis=-1)
    d = np.ones_like(n)
    for j, e in enumerate(endpoints):
        if e in (4, 5):
            d[:, :, j] = costs[:, :, 0, e - 4]
    return n, d


def _probability(low: np.ndarray, high: np.ndarray, complete: bool) -> dict:
    def interval(flags):
        # Wilson Monte Carlo interval. This measures simulation error only.
        n, p, z = len(flags), float(flags.mean()), 1.959963984540054
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return [max(0.0, float(centre - radius)), min(1.0, float(centre + radius))]

    return {
        "probability": float(low.mean()) if complete else None,
        "bounds": [float(low.mean()), float(high.mean())],
        "status": "complete" if complete else "incomplete",
        "monte_carlo_95_interval_bounds": [interval(low)[0], interval(high)[1]],
    }


def _minimum_effect(pvalues, estimates, critical, h: int, families: list[str], step: float):
    components = list(COMPONENTS[h])
    accuracy = [e for e in components if e not in (4, 5)]
    known_means = {
        f: {
            ENDPOINTS[e]: float(estimates[:, i, e].mean())
            for e in accuracy
            if np.isfinite(estimates[:, i, e]).all()
        }
        for i, f in enumerate(families)
    }
    means = [v for values in known_means.values() for v in values.values()]
    complete = bool(np.isfinite(pvalues).all())
    lower = max([-2.0, *[-1 - v for v in means]])
    upper = min([2.0, *[1 - v for v in means]])
    grid = np.arange(np.ceil(lower / step), np.floor(upper / step) + 1) * step

    def invert(optimistic):
        fill = 0 if optimistic else 1
        completed = np.where(np.isfinite(pvalues), pvalues, fill)
        hp = np.stack([completed[..., list(c)].max(axis=-1) for c in COMPONENTS], axis=-1)
        others = np.sort(np.delete(hp, h, axis=-1), axis=-1)
        rank = np.zeros(hp.shape[:-1], dtype=int)
        active = np.ones_like(rank, dtype=bool)
        for j in range(3):
            active &= others[..., j] <= LEVELS[rank]
            rank += active
        threshold = np.full(rank.shape, -np.inf)
        for e in components:
            known = np.isfinite(pvalues[..., e])
            if e in (4, 5):
                required = np.where(completed[..., e] <= LEVELS[rank], -np.inf, np.inf)
            else:
                bound = np.take_along_axis(critical[..., e, :], rank[..., None], axis=-1)[..., 0]
                required = np.nextafter(NULLS[e] + bound - estimates[..., e], np.inf)
            required = np.where(known, required, -np.inf if optimistic else np.inf)
            threshold = np.maximum(threshold, required)
        if h == 0:
            gate = 0.05 - 1e-12 - estimates[..., 0]
            gate = np.where(np.isfinite(gate), gate, -np.inf if optimistic else np.inf)
            threshold = np.maximum(threshold, gate)
        required = threshold.max(axis=-1)  # same shift required in every family
        powers = (grid[:, None] >= required[None, :]).mean(axis=1)
        candidates = np.flatnonzero(powers >= 0.8)
        effect = None
        if len(candidates) and means:
            delta = float(grid[candidates[0]])
            effect = {
                "shift": delta,
                "power": float(powers[candidates[0]]),
                "effects": {
                    f: {name: value + delta for name, value in scores.items()}
                    for f, scores in known_means.items()
                },
            }
        return {"effect_for_80": effect, "maximum_power": float(powers[-1]) if len(powers) else 0.0}

    pessimistic, optimistic = invert(False), invert(True)
    return {
        "effect_for_80": pessimistic["effect_for_80"] if complete else None,
        "maximum_power_bounds": [pessimistic["maximum_power"], optimistic["maximum_power"]],
        "missing_fail": pessimistic,
        "missing_pass": optimistic,
        "status": "complete" if complete else "incomplete",
        "varied_endpoints": [ENDPOINTS[e] for e in accuracy],
    }


def simulate(
    config: str | Path,
    *,
    simulations=200,
    resamples=100_000,
    seed=0,
    batch_size=256,
    effect_step=0.0025,
    crossed=False,
) -> dict:
    """Transport paired case blocks to the registered populations and test every study.

    See docs/power.md for the manifest and the location-shift definition of the
    minimum effect. No marginal powers are multiplied, including across families.
    """
    if simulations < 1 or resamples < 79 or batch_size < 1:
        raise ValueError(
            "simulations and batch_size must be positive; resamples must be at least 79"
        )
    if not 0 < effect_step <= 0.1:
        raise ValueError("effect_step must be in (0, 0.1]")
    timer = StageTimer()
    config = Path(config)
    inputs = _Inputs(config)
    spec = inputs.read(str(config.resolve()))
    families = spec.get("families", ["qwen25", "gemma31"])
    if len(families) != 2 or len(set(families)) != 2:
        raise ValueError("the joint rule requires exactly two distinct model families")
    panels = {
        name: _read_panel(p, families, inputs, updates=name == "updates", temporal=name == "t")
        for name, p in spec["panels"].items()
    }
    if "cf" not in panels or set(panels) - {"cf", "t", "updates"}:
        raise ValueError("panels need cf and may also contain t and updates")
    populations = {n: _read_population(spec["populations"][n], inputs) for n in ("cf", "h3", "t")}
    cf, h3, temporal = (populations[n] for n in ("cf", "h3", "t"))
    index = {c: i for i, c in enumerate(cf.cases)}
    if not set(h3.cases) <= index.keys():
        raise ValueError("H3 must be a subset of the primary CF population")
    subset = np.array([index[c] for c in h3.cases])
    if [cf.sequences[i] for i in subset] != h3.sequences:
        raise ValueError("H3 relation sequences disagree with the primary CF registry")
    if cf.events is not None:
        h3.events = [cf.events[i] for i in subset]
    if temporal.events is None:
        raise ValueError("T population needs event clusters (events or data)")
    if crossed and cf.events is None:
        raise ValueError("crossed CF bootstrap needs population event clusters (events or data)")
    if "updates" in panels:
        panel = panels["updates"]
        if "updates" in spec["populations"]:
            events = inputs.read(spec["populations"]["updates"]["events"])
            populations["updates"] = _Population(
                list(events), ["updates"] * len(events), list(events.values())
            )
        else:
            populations["updates"] = _Population(panel.cases, panel.sequences, panel.events)
    if spec.get("transport_t_from_cf"):
        if "t" in panels:
            raise ValueError("transport_t_from_cf cannot also supply a T panel")
        panels["t"] = panels["cf"]
    effects = np.full((simulations, len(families), len(ENDPOINTS)), np.nan)
    pvalues = np.full_like(effects, np.nan)
    critical = np.full((*effects.shape, 4), np.nan)
    outer_seed, inner_seed = np.random.SeedSequence(seed).spawn(2)
    outer, inner = np.random.default_rng(outer_seed), np.random.default_rng(inner_seed)
    count_cache = {}
    for study in range(simulations):
        cache = {}
        draw = _transport(panels["cf"], cf, outer, cache)
        tasks = [
            (panels["cf"], draw, cf.sequences, cf.events if crossed else None, list(range(6))),
            (
                panels["cf"],
                draw[subset],
                h3.sequences,
                [cf.events[i] for i in subset] if crossed else None,
                [6],
            ),
        ]
        if "t" in panels:
            draw_t = _transport(
                panels["t"],
                temporal,
                outer,
                cache,
                events=not spec.get("transport_t_from_cf", False),
            )
            tasks.append((panels["t"], draw_t, temporal.events, None, [7]))
        if "updates" in panels:
            population = populations["updates"]
            draw_u = _transport(panels["updates"], population, outer, cache, events=True)
            tasks.append((panels["updates"], draw_u, population.events, None, [8]))
        for panel, chosen, clusters, events, endpoints in tasks:
            n, d = _case_statistics(panel, chosen, endpoints)
            result = bootstrap_test(
                n,
                d,
                clusters,
                NULLS[endpoints],
                resamples=resamples,
                rng=inner,
                batch_size=batch_size,
                crossed=events,
                count_cache=count_cache,
            )
            effects[study][:, endpoints] = result["estimate"]
            pvalues[study][:, endpoints] = result["pvalues"]
            critical[study][:, endpoints] = result["critical"]
    missing = ~np.isfinite(pvalues)
    low = decision_rule(np.where(missing, 1, pvalues), np.where(missing, 0, effects))
    high = decision_rule(np.where(missing, 0, pvalues), np.where(missing, 0.05, effects))
    complete = bool(not missing.any())

    def hypothesis_reports(selected):
        lo, hi = low["passes"][:, selected], high["passes"][:, selected]
        if lo.ndim == 2:
            lo, hi = lo[:, None, :], hi[:, None, :]
        return {
            "hypotheses": {
                h: _probability(
                    lo[..., j].all(axis=1),
                    hi[..., j].all(axis=1),
                    bool(not missing[:, selected].any()),
                )
                for j, h in enumerate(HYPOTHESES)
            },
            "all_hypotheses": _probability(
                lo.all(axis=(1, 2)), hi.all(axis=(1, 2)), bool(not missing[:, selected].any())
            ),
        }

    reports = {}
    for i, family in enumerate(families):
        report = hypothesis_reports(i)
        report["endpoints"] = {}
        for e, name in enumerate(ENDPOINTS):
            known = np.isfinite(pvalues[:, i, e])
            report["endpoints"][name] = {
                "known_studies": int(known.sum()),
                "mean_effect": float(effects[known, i, e].mean()) if known.any() else None,
                "effect_sd": float(effects[known, i, e].std()) if known.any() else None,
                "mean_lower_bound_at_0.0125": float(
                    (effects[known, i, e] - critical[known, i, e, 0]).mean()
                )
                if known.any()
                else None,
            }
        report["minimum_effect"] = {
            h: _minimum_effect(
                pvalues[:, i : i + 1],
                effects[:, i : i + 1],
                critical[:, i : i + 1],
                j,
                [family],
                effect_step,
            )
            for j, h in enumerate(HYPOTHESES)
        }
        reports[family] = report
    joint = hypothesis_reports(slice(None))
    joint["minimum_effect"] = {
        h: _minimum_effect(pvalues, effects, critical, j, families, effect_step)
        for j, h in enumerate(HYPOTHESES)
    }
    return {
        "artifact_kind": "xgraph.legacy_joint_power",
        "version": 1,
        "settings": {
            "config": str(config.resolve()),
            "simulations": simulations,
            "resamples": resamples,
            "seed": seed,
            "batch_size": batch_size,
            "effect_step": effect_step,
            "crossed_cf_bootstrap": crossed,
            "registered_resamples": 100_000,
            "registered_resamples_used": resamples == 100_000,
            "transport_t_from_cf": bool(spec.get("transport_t_from_cf")),
            "monte_carlo_pvalue": "(1 + null-centred upper-tail count) / (resamples + 1)",
            "common_inner_cluster_draws": True,
        },
        "populations": {n: p.summary() for n, p in populations.items()},
        "panels": {n: p.description for n, p in panels.items()},
        "transport": {
            name: {
                "source_cases": len(panels["cf" if name == "h3" else name].cases),
                "source_singleton_sequences": sum(
                    len(v) == 1
                    for v in panels["cf" if name == "h3" else name].sequence_members.values()
                ),
                "target_cases_with_matched_sequence": sum(
                    s in panels["cf" if name == "h3" else name].sequence_members
                    for s in p.sequences
                ),
            }
            for name, p in populations.items()
            if ("cf" if name == "h3" else name) in panels
        },
        "complete": complete,
        "families": reports,
        "joint": joint,
        "missing_endpoints": {
            f: [ENDPOINTS[e] for e in range(len(ENDPOINTS)) if missing[:, i, e].any()]
            for i, f in enumerate(families)
        },
        "minimum_effect_model": (
            "Common additive shift to every accuracy contrast within one hypothesis, "
            "in all families; fixed bootstrap deviations, cost ratios and other "
            "hypotheses. Holm and the point gate are reinverted. Fractions, not points."
        ),
        "interpretation": (
            "Transported development design sensitivity, not a forecast. Missing endpoints "
            "are bounded by p=1 versus p=0; they never count as measured successes. "
            "Monte Carlo intervals exclude donor, selection and transport uncertainty."
        ),
        "provenance": provenance(sorted(inputs.paths)),
        "resources": timer.summary(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("simulate", help="simulate the joint registered decision rule")
    command.add_argument(
        "--config", type=Path, required=True, help="input manifest; see docs/power.md"
    )
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--simulations", type=int, default=200)
    command.add_argument("--resamples", type=int, default=100_000)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--batch-size", type=int, default=256)
    command.add_argument("--effect-step", type=float, default=0.0025)
    command.add_argument(
        "--crossed", action="store_true", help="CF sequence x event robustness bootstrap"
    )
    args = parser.parse_args(argv)
    try:
        result = simulate(
            args.config,
            simulations=args.simulations,
            resamples=args.resamples,
            seed=args.seed,
            batch_size=args.batch_size,
            effect_step=args.effect_step,
            crossed=args.crossed,
        )
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "joint": result["joint"],
                "resources": result["resources"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1:])
