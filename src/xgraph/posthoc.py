"""Small, deterministic reducers for retrospective paired analyses (CPU, stdlib only)."""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict


def indexed(rows, expected=None):
    """Reject duplicate records and incomplete paired ID matrices."""
    result = {}
    for row in rows:
        if row["id"] in result:
            raise ValueError(f"duplicate record: {row['id']}")
        result[row["id"]] = row
    if expected is not None and set(result) != set(expected):
        raise ValueError("record IDs differ from the declared panel")
    return result


def relation_sequence(relations, mapping):
    """An ordered sequence of indexed ID unions, preserving aliases and ambiguity."""
    if relations is None:
        return None
    result = []
    for label in relations:
        if not mapping.get(label):
            raise ValueError(f"unknown relation: {label!r}")
        result.append(tuple(sorted(mapping[label])))
    return tuple(result)


def executed_sequence(executor, start, relations, anchor=None):
    """Resolve actual relation-ID unions using the matched, capped execution trace.

    Selection at a boundary affects the next frontier. All surviving start ranks
    participate, matching execution; this is not a single inferred gold path.
    This adapter intentionally uses the same private trace API as the retained
    menu-control support reducer. It never requests an uncapped replay.
    """
    index = executor._index
    start = executor._read_start(index, start, anchor)
    trace = []
    executor._walk(index, start, relations, anchor, False, trace)
    frontier = {e: 0 for e in executor._start_entities(index, start, anchor)}
    selections = getattr(relations, "selections", {})
    if 0 in selections:
        named = executor._rows_owned(index.labels, selections[0])
        frontier = {e: rank for e, rank in frontier.items() if e in named}
    result = []
    for label, reached in zip(relations, trace, strict=False):
        followed = executor._rows_owned(index.relations, label) & executor._have(index, frontier)
        result.append(tuple(sorted(index.relations.owner_ids[r] for r in followed)))
        frontier = reached
    return tuple(result)


def ordinal_choice(text, menu, stride):
    """Classify a complete ordinal against the actual displayed menu entry."""
    if re.fullmatch(r"\s*[0-9]+\s*", text) is None:
        return "nonordinal"
    choice = int(text)
    if choice == 0:
        return "stop"
    if choice > len(menu):
        return "out_of_range"
    size = len(menu[choice - 1]["relations"])
    if size == 1:
        return "atomic"
    if size == 2:
        return "pair" if stride == 2 else "forbidden_pair"
    raise ValueError("menu entry must contain one or two relations")


def set_metrics(prediction, gold, *, failed=False):
    """Ranked first hit and exact label-set F1; empty sets remain in the denominator."""
    predicted, wanted = set(prediction), set(gold)
    return {
        "hits_at_1": int(not failed and bool(prediction) and prediction[0] in wanted),
        "f1": 0.0
        if failed
        else (
            2 * len(predicted & wanted) / (len(predicted) + len(wanted))
            if predicted or wanted
            else 1.0
        ),
        "strict_subset": predicted < wanted,
        "prediction_size": len(predicted),
        "gold_size": len(wanted),
    }


def _interval(point, draws):
    defined = sorted(d for d in draws if d is not None)

    def quantile(p):
        if not defined:
            return None
        at = (len(defined) - 1) * p
        low = math.floor(at)
        value = defined[low] + (defined[math.ceil(at)] - defined[low]) * (at - low)
        return 100 * value

    return {
        "point": None if point is None else 100 * point,
        "low": quantile(0.025),
        "high": quantile(0.975),
        "valid_resamples": len(defined),
    }


def _bootstrap(members, columns, resamples, seed):
    """Uniform whole-cluster draws within hops; calibrate to frozen full hop mass."""
    mass = math.fsum(m["weight"] for m in members)
    strata = defaultdict(lambda: defaultdict(list))
    for i, member in enumerate(members):
        strata[member["hops"]][member["cluster"]].append(i)
    draws = [[0.0] * resamples for _ in columns]
    rng = random.Random(seed)
    for hop in sorted(strata):
        clusters = strata[hop]
        groups = [clusters[k] for k in sorted(clusters)]
        weights = [math.fsum(members[i]["weight"] for i in group) for group in groups]
        sums = [
            [math.fsum(members[i]["weight"] * col[i] for i in group) for group in groups]
            for col in columns
        ]
        hop_share = math.fsum(weights) / mass
        population = range(len(groups))
        for b in range(resamples):
            counts = Counter(rng.choices(population, k=len(groups)))
            chosen = list(counts.items())
            denominator = math.fsum(weights[i] * count for i, count in chosen)
            for j, totals in enumerate(sums):
                draws[j][b] += (
                    math.fsum(totals[i] * count for i, count in chosen) / denominator * hop_share
                )
    return draws


def paired_summary(members, first, second, masks, *, interactions=None, resamples=10000, seed=17):
    """Paired contrasts, subset-conditional effects and additive population contributions.

    Subsets are classified once from retained records. Every bootstrap draw keeps
    both arms and these classifications together. A subset's conditional contrast
    divides its resampled weighted numerator by its resampled mass. Its population
    contribution divides by the full panel mass. Empty conditional draws are
    undefined and counted explicitly, never assigned zero or silently dropped.
    """
    n = len(members)
    indexed(members)
    if not n or len(first) != n or len(second) != n or resamples < 2:
        raise ValueError("nonempty aligned vectors and at least two resamples required")
    if any(not math.isfinite(m["weight"]) or m["weight"] <= 0 for m in members):
        raise ValueError("each design weight must be finite and positive")
    if any(not math.isfinite(v) for v in [*first, *second]):
        raise ValueError("non-finite score")
    if "all" in masks or any(len(mask) != n for mask in masks.values()):
        raise ValueError("aligned masks required; 'all' is reserved")
    masks = {"all": [True] * n, **masks}
    delta = [a - b for a, b in zip(first, second, strict=True)]
    mass = math.fsum(m["weight"] for m in members)
    columns = []
    for mask in masks.values():
        columns.extend(([d * bool(keep) for d, keep in zip(delta, mask, strict=True)], mask))
    draws = _bootstrap(members, columns, resamples, seed)
    groups, conditional = {}, {}
    for j, (name, mask) in enumerate(masks.items()):
        selected = [i for i, keep in enumerate(mask) if keep]
        weight = math.fsum(members[i]["weight"] for i in selected)
        gain = math.fsum(members[i]["weight"] * delta[i] for i in selected)
        numerator, denominator = draws[2 * j : 2 * j + 2]
        cd = [a / b if b else None for a, b in zip(numerator, denominator, strict=True)]
        point = gain / weight if weight else None
        conditional[name] = point, cd
        groups[name] = {
            "n": len(selected),
            "clusters": len({members[i]["cluster"] for i in selected}),
            "weight_sum": weight,
            "population_weight_sum": mass,
            "population_share_pct": 100 * weight / mass,
            "first_mean_pct": 100
            * math.fsum(members[i]["weight"] * first[i] for i in selected)
            / weight
            if weight
            else None,
            "second_mean_pct": 100
            * math.fsum(members[i]["weight"] * second[i] for i in selected)
            / weight
            if weight
            else None,
            "first_better_records": sum(delta[i] > 0 for i in selected),
            "second_better_records": sum(delta[i] < 0 for i in selected),
            "difference_pp": _interval(point, cd),
            "population_contribution_pp": _interval(gain / mass, numerator),
        }
    contrasts = {}
    for name, (a, b) in (interactions or {}).items():
        ap, ad = conditional[a]
        bp, bd = conditional[b]
        contrasts[name] = _interval(
            ap - bp if ap is not None and bp is not None else None,
            [
                x - y if x is not None and y is not None else None
                for x, y in zip(ad, bd, strict=True)
            ],
        )
    return {
        "all": groups.pop("all"),
        "groups": groups,
        "interactions": contrasts,
        "bootstrap": {
            "resamples": resamples,
            "seed": seed,
            "method": "paired whole clusters within hops, fixed full-panel hop mass",
            "rng": "stdlib random.Random.choices",
            "interval": "95% linear percentile",
        },
    }
