"""Registered fixed-sequence analysis and selection-aware joint assurance (CPU)."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from xgraph.calibration import Panel, select_arms
from xgraph.cost import StageTimer, dedicated_gpu_seconds, request_cost
from xgraph.editing import case_clusters, case_members, update_events
from xgraph.inference import (
    ALPHA,
    ENDPOINTS,
    HYPOTHESES,
    NULLS,
    bootstrap_test,
    decision_rule,
    serving_pairs,
)
from xgraph.inference import (
    COMPONENTS as COMPONENTS,
)
from xgraph.provenance import provenance
from xgraph.score import load_labels, record_failure, record_outputs, wwq_accuracy


def _resources(timer):
    limit = resource.getrlimit(resource.RLIMIT_AS)[0]
    return {
        **timer.summary(),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "address_space_limit_bytes": None if limit == resource.RLIM_INFINITY else limit,
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


class _Inputs:
    def __init__(self, config):
        self.base = Path(config).resolve().parent
        self.paths = {
            Path(config).resolve(),
            Path(__file__).resolve(),
            Path(sys.modules[bootstrap_test.__module__].__file__).resolve(),
            Path(sys.modules[select_arms.__module__].__file__).resolve(),
            Path(sys.modules[record_outputs.__module__].__file__).resolve(),
            Path(sys.modules[case_clusters.__module__].__file__).resolve(),
            Path(sys.modules[dedicated_gpu_seconds.__module__].__file__).resolve(),
        }

    def path(self, path):
        path = (self.base / path).resolve()
        self.paths.add(path)
        return path

    def read(self, path):
        return json.loads(self.path(path).read_text(encoding="utf-8"))


@dataclass
class _Panel:
    panel: Panel
    sequences: list[str]
    events: list[str]
    weights: np.ndarray
    right: dict
    tokens: dict
    description: dict

    @property
    def cases(self):
        return self.panel.cases


@dataclass
class _Population:
    cases: list[str]
    sequences: list[str]
    events: list[str]

    def summary(self):
        return {
            "cases": len(self.cases),
            "sequences": len(set(self.sequences)),
            "events": len(set(self.events)),
        }


def _read_population(spec, inputs):
    registry = inputs.read(spec["registry"])
    records = dict(registry["cases"])
    ids = registry["strata"][spec["stratum"]]["ids"]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"{spec['stratum']}: population ids must be nonempty and unique")
    sequences = [records[c]["relation_sequence"] for c in ids]
    if spec.get("events"):
        mapping = inputs.read(spec["events"])
    elif spec.get("data"):
        selected = set(ids)
        mapping = update_events(
            {
                f"mquake-{c['case_id']}": c
                for c in inputs.read(spec["data"])
                if f"mquake-{c['case_id']}" in selected
            }
        )
    else:
        raise ValueError("population needs frozen event clusters (events or data)")
    missing = set(ids) - mapping.keys()
    if missing:
        raise ValueError(f"population event map missing cases {sorted(missing)[:5]}")
    return _Population(ids, list(map(str, sequences)), [str(mapping[c]) for c in ids])


def _read_panel(spec, families, inputs, *, candidates=False):
    questions = inputs.read(spec["questions"])
    qids = [q["id"] for q in questions]
    members = case_members(questions)
    cases = list(members)
    if not qids or len(set(qids)) != len(qids) or any(len(q) != 3 for q in members.values()):
        raise ValueError("panel needs unique ids and exactly three questions per case")
    seq = list(map(str, case_clusters(inputs.read(spec["clusters"]), members, cases)))
    if not spec.get("events"):
        raise ValueError("calibration/CF panel needs frozen event clusters")
    events = list(map(str, case_clusters(inputs.read(spec["events"]), members, cases)))
    weights = inputs.read(spec["panel"]) if spec.get("panel") else {c: 1 for c in cases}
    weights = weights.get("weights", weights)
    if set(cases) - weights.keys():
        raise ValueError("panel weights missing cases")
    panel = Panel(qids, dict(zip(cases, seq, strict=True)), weights)
    raw = bool(spec.get("arms")) and not candidates
    key = "candidate_outcomes" if candidates else "outcomes"
    if not raw and not spec.get(key):
        raise ValueError(f"panel needs {key}; selected-only scores cannot establish assurance")
    data = spec["arms"] if raw else inputs.read(spec[key])
    if set(data) != set(families):
        raise ValueError("panel must supply both model families")
    gold = labels = None
    if raw:
        gold = inputs.read(spec["gold_new"])
        paths = spec["labels"] if isinstance(spec["labels"], list) else [spec["labels"]]
        labels = load_labels(
            [inputs.path(p) for p in paths],
            {e for g in gold["answers"].values() for e in g["entities"]},
        )
    right, tokens, description = {}, {}, {}
    for family in families:
        right[family], tokens[family], description[family] = {}, {}, {}
        for arm, records in data[family].items():
            if raw:
                rows = [
                    json.loads(line)
                    for line in inputs.path(records).read_text().splitlines()
                    if line.strip()
                ]
                outputs = record_outputs(rows, qids)
                records = {r["id"]: r for r in rows}
                flags = wwq_accuracy(outputs, gold, labels)["correct"]
            else:
                flags = {}
            score, usage = [], []
            failures = 0
            for q in qids:
                value = records.get(q, {})
                value = {"correct": value} if isinstance(value, bool) else value
                if not isinstance(value, dict):
                    raise ValueError(f"{family}/{arm}/{q}: outcome must be a boolean or object")
                flag = flags.get(q, False) if raw else value.get("correct", False)
                if not isinstance(flag, bool):
                    raise ValueError(f"{family}/{arm}/{q}: correctness must be a boolean")
                invalid = record_failure(value, require_text=raw)
                score.append(flag and not invalid)
                failures += invalid
                cost = value.get("total_tokens")
                if raw and cost is None:
                    c = request_cost(value)
                    if c["prompt_tokens"] is not None and c["generated_tokens"] is not None:
                        cost = c["prompt_tokens"] + c["generated_tokens"]
                if cost is not None and (
                    isinstance(cost, bool)
                    or not isinstance(cost, (int, float))
                    or not np.isfinite(cost)
                    or cost < 0
                ):
                    raise ValueError(f"{family}/{arm}/{q}: tokens must be finite and nonnegative")
                usage.append(np.nan if cost is None else cost)
            grid = panel.by_case(np.array(score))
            right[family][arm] = grid
            tokens[family][arm] = panel.by_case(np.array(usage)).sum(axis=1)
            description[family][arm] = {
                "per_question": float(grid.mean()),
                "any_of_three": float(grid.any(axis=1).mean()),
                "per_question_weighted": float(
                    np.average(grid.mean(axis=1), weights=panel.case_weight)
                ),
                "missing_records": len(set(qids) - records.keys()),
                "protocol_failures": int(failures),
                "known_token_cases": int(np.isfinite(tokens[family][arm]).sum()),
            }
    return _Panel(panel, seq, events, panel.case_weight, right, tokens, description)


def _case_statistics(panel, families, selection, mechanism, draw=None):
    draw = np.arange(len(panel.cases)) if draw is None else draw
    n = np.full((len(draw), len(families), 4), np.nan)
    for f, family in enumerate(families):
        chosen = {r: selection[family][r]["arm"] for r in ("P", "X", "A")}
        chosen.update(mechanism.get(family, {"F": "F", "V": "V"}))
        arms = {r: panel.right[family].get(a) for r, a in chosen.items()}
        if arms["F"] is not None and arms["V"] is not None:
            n[:, f, 0] = (arms["F"][draw].astype(float) - arms["V"][draw]).mean(axis=1)
        p, x, a = (arms[r] for r in ("P", "X", "A"))
        if p is not None and x is not None:
            n[:, f, 1] = (p[draw].astype(float) - x[draw]).mean(axis=1)
            n[:, f, 2] = p[draw].any(axis=1).astype(float) - x[draw].any(axis=1)
        if p is not None and a is not None:
            n[:, f, 3] = (p[draw].astype(float) - a[draw]).mean(axis=1)
    return n


def _families(spec):
    families = spec.get("families", ["qwen25", "gemma31"])
    if len(families) != 2 or len(set(families)) != 2:
        raise ValueError("the registered rule needs two distinct model families")
    return families


def _selection(spec, families):
    return spec.get("selection", {f: {r: {"arm": r} for r in ("P", "X", "A")} for f in families})


def _endpoint_report(result, family):
    return {
        name: {
            "estimate": float(result["estimate"][family, e]),
            "pvalue": float(result["pvalues"][family, e]),
            "lower_97_5": float(result["lower"][family, e]),
            "interval_95": result["interval"][family, e].tolist(),
            "null": float(NULLS[e]),
        }
        for e, name in enumerate(ENDPOINTS)
    }


def _finite_json(value):
    if isinstance(value, dict):
        return {k: _finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _timing_reports(spec, inputs):
    # Files contain the serving study's complete per-cell trial reports, with
    # trial.repeat/arm/cache/concurrency/seed and summary.requests/gpu_seconds.
    reports = inputs.read(spec) if isinstance(spec, str) else spec
    if isinstance(reports, dict):
        reports = reports["trials"]
    result = []
    for report in reports:
        if "summary" in report:
            result.append(report)
        elif report.get("output"):
            # Plan entries name per-cell reports; retain failed entries in the
            # ledger, never use them as successful repetitions.
            if report.get("status") == "completed":
                result.append(inputs.read(report["output"]))
            else:
                result.append({"status": report.get("status"), "trial": report})
        else:
            raise ValueError("timing input needs complete trial reports or a trials plan")
    return result


def _secondary_analysis(spec, families, inputs, *, resamples, seed, batch_size):
    """T and fidelity share whole-event draws, including events in both populations."""
    arrays, denominators, events, present = [], [], [], set()
    panels = spec.get("panels", {})
    if "t" in panels:
        t = _read_panel(panels["t"], families, inputs)
        n, d = np.zeros((len(t.cases), 2, 2)), np.zeros((len(t.cases), 2, 2))
        for f, family in enumerate(families):
            n[:, f, 0] = (t.right[family]["P"].astype(float) - t.right[family]["A"]).mean(axis=1)
            d[:, f, 0] = 1
        arrays.append(n)
        denominators.append(d)
        events.extend(t.events)
        present.add(0)
    if "updates" in panels:
        panel = panels["updates"]
        qids = [q["id"] for q in inputs.read(panel["questions"])]
        if not qids or len(qids) != len(set(qids)):
            raise ValueError("update question ids must be nonempty and unique")
        data, mapping = inputs.read(panel["outcomes"]), inputs.read(panel["events"])
        n, d = np.zeros((len(qids), 2, 2)), np.zeros((len(qids), 2, 2))
        for f, family in enumerate(families):
            flags = {}
            for arm in ("P", "A"):
                flags[arm] = []
                for q in qids:
                    row = data[family][arm].get(q, {})
                    fidelity = row.get("fidelity", [False] * 3)
                    if (
                        not isinstance(fidelity, list)
                        or len(fidelity) != 3
                        or any(type(v) is not bool for v in fidelity)
                    ):
                        raise ValueError(
                            "update fidelity needs current/rollback/off-chain booleans"
                        )
                    flags[arm].append(all(fidelity) and not record_failure(row, require_text=False))
            n[:, f, 1] = np.array(flags["P"], dtype=float) - flags["A"]
            d[:, f, 1] = 1
        arrays.append(n)
        denominators.append(d)
        events.extend(str(mapping[q]) for q in qids)
        present.add(1)
    if not arrays:
        return {}
    n, d = np.concatenate(arrays), np.concatenate(denominators)
    for endpoint in set(range(2)) - present:
        n[:, :, endpoint] = d[:, :, endpoint] = np.nan
    result = bootstrap_test(
        n,
        d,
        events,
        np.array([-0.03, -0.03]),
        resamples=resamples,
        seed=seed,
        batch_size=batch_size,
    )
    return {
        "confirmatory": False,
        "clusters": "whole update events, shared across populations",
        "events": len(set(events)),
        "families": {
            family: {
                name: {
                    "estimate": result["estimate"][f, e],
                    "interval_95": result["interval"][f, e],
                }
                for e, name in enumerate(("T.question", "updates.fidelity"))
                if e in present
            }
            for f, family in enumerate(families)
        },
    }


def analyze(config, *, resamples=100_000, seed=0, batch_size=256, clustering="crossed"):
    """Analyze fixed F/V/P/X/A outcomes; downstream estimates are always reported."""
    timer, inputs = StageTimer(), _Inputs(config)
    spec = inputs.read(str(Path(config).resolve()))
    families = _families(spec)
    panel = _read_panel(spec["panels"]["cf"], families, inputs)
    fixed = {f: {r: {"arm": r} for r in ("P", "X", "A")} for f in families}
    n = _case_statistics(panel, families, fixed, {})
    groups = panel.events if clustering == "event" else panel.sequences
    result = bootstrap_test(
        n,
        np.ones_like(n),
        groups,
        NULLS[:4],
        resamples=resamples,
        seed=seed,
        batch_size=batch_size,
        crossed=panel.events if clustering == "crossed" else None,
    )
    full = {k: np.full((2, 5, 2) if k == "interval" else (2, 5), np.nan) for k in result}
    for k in result:
        full[k][:, :4] = result[k]
    serving = {}
    for f, family in enumerate(families):
        timing = spec.get("serving", {}).get(family)
        if timing:
            a, p, questions = serving_pairs(
                _timing_reports(timing["trials"], inputs),
                **{k: timing[k] for k in ("p_arm", "a_arm", "cache", "concurrency")},
            )
            serving[family] = {
                "repetitions": len(a),
                "questions": int(questions.sum()),
                "cost_per_question": {
                    "A": float(a.sum() / questions.sum()),
                    "P": float(p.sum() / questions.sum()),
                },
            }
            cost = bootstrap_test(
                a[:, None, None],
                p[:, None, None],
                list(map(str, range(len(a)))),
                NULLS[4:],
                resamples=resamples,
                seed=seed,
                batch_size=batch_size,
            )
            for k in full:
                full[k][f, 4] = cost[k][0, 0]
    known = np.isfinite(full["pvalues"])
    decisions = decision_rule(np.where(known, full["pvalues"], 1), full["estimate"])
    return _finite_json(
        {
            "artifact_kind": "xgraph.registered_analysis",
            "version": 2,
            "families": {f: _endpoint_report(full, i) for i, f in enumerate(families)},
            "decision": decisions,
            "serving": serving,
            "decision_scope": "primary" if clustering == "crossed" else "sensitivity",
            "secondary": _secondary_analysis(
                spec, families, inputs, resamples=resamples, seed=seed, batch_size=batch_size
            ),
            "complete": bool(known.all()),
            "settings": {
                "resamples": resamples,
                "seed": seed,
                "alpha": ALPHA,
                "clustering": clustering,
                "registered_resamples_used": resamples == 100_000,
            },
            "provenance": provenance(sorted(inputs.paths)),
            "resources": _resources(timer),
        }
    )


def _outer_counts(panel, rng):
    _, seq = np.unique(panel.sequences, return_inverse=True)
    _, event = np.unique(panel.events, return_inverse=True)
    ns, ne = int(seq.max()) + 1, int(event.max()) + 1
    while True:
        sc = rng.multinomial(ns, np.full(ns, 1 / ns))
        ec = rng.multinomial(ne, np.full(ne, 1 / ne))
        counts = sc[seq] * ec[event]
        if counts.any():
            return counts


def _reselect(panel, families, catalog, counts):
    selected = {}
    for family in families:
        inverse = np.argsort(np.argsort(panel.panel.case_of_question, kind="stable"))
        arms = {a: grid.ravel()[inverse] for a, grid in panel.right[family].items()}
        costs = {}
        for a, values in panel.tokens[family].items():
            used = counts > 0
            total = float(values[used] @ counts[used]) if np.isfinite(values[used]).all() else None
            costs[a] = {"total_tokens": total}
        selected[family] = select_arms(
            panel.panel, arms, catalog[family], costs, case_multiplicities=counts
        )
    return selected


def _predict_cases(panel, target, counts, rng, *, matched_probability, event_share):
    """Crossed empirical block prediction, with new sequence and event donors.

    Named donors are mixed with fresh clusters from the outer empirical library.
    Both donor labels are shared over a target cluster. Use their occupied source
    cell when available; otherwise a predeclared sequence/event mixture. Draw whole
    paired case blocks, never independent candidate or paraphrase outcomes.
    """

    def groups(labels):
        return {
            g: np.flatnonzero((np.array(labels) == g) & (counts > 0))
            for g in sorted(set(labels))
            if np.any((np.array(labels) == g) & (counts > 0))
        }

    seq, events = groups(panel.sequences), groups(panel.events)

    def donors(names, source, by_hops):
        chosen = {}
        for name in sorted(set(names)):
            pool = [s for s in source if not by_hops or s.count(">") == name.count(">")]
            pool = pool or list(source)
            if name in source and rng.random() < matched_probability:
                chosen[name] = name
            else:
                mass = np.array([counts[source[s]].sum() for s in pool], dtype=float)
                chosen[name] = pool[rng.choice(len(pool), p=mass / mass.sum())]
        return chosen

    sd = donors(target.sequences, seq, True)
    ed = donors(target.events, events, False)
    cells = {}
    for i, cell in enumerate(zip(target.sequences, target.events, strict=True)):
        cells.setdefault(cell, []).append(i)
    draw = np.empty(len(target.cases), dtype=int)
    fallback = 0
    for (s, e), indices in cells.items():
        pool = np.intersect1d(seq[sd[s]], events[ed[e]], assume_unique=True)
        if not len(pool):
            fallback += 1
            event_pool = events[ed[e]]
            compatible = np.array(
                [panel.sequences[i].count(">") == s.count(">") for i in event_pool]
            )
            event_pool = event_pool[compatible]
            pool = event_pool if len(event_pool) and rng.random() < event_share else seq[sd[s]]
        probability = counts[pool] * panel.weights[pool]
        draw[indices] = rng.choice(pool, size=len(indices), p=probability / probability.sum())
    return draw, fallback / len(cells)


class _Timing:
    def __init__(self, spec, inputs, families):
        self.spec = inputs.read(spec) if isinstance(spec, str) else spec or {}
        self.kind = self.spec.get("kind", "missing")
        if self.kind not in {"pilot", "placeholder", "missing"}:
            raise ValueError("timing kind must be pilot or placeholder")
        self.families, self.native, self.validated = families, {}, set()
        samples = self.spec.get("samples", [])
        cells = self.spec.get("families", {})
        self.paired_families = self.spec.get("paired_families", not bool(cells))
        if not isinstance(self.paired_families, bool):
            raise ValueError("paired_families must be a boolean")
        self.rows = {f: [s.get(f, {}) for s in samples] for f in families}
        repeats = {}
        if cells:
            for family in families:
                cell = cells[family]
                reports = _timing_reports(cell["trials"], inputs)
                self.native[family] = (reports, cell)
                rows, seen = {}, set()
                for report in reports:
                    trial = report["trial"]
                    if (trial["cache"], trial["concurrency"]) != (
                        cell["cache"],
                        cell["concurrency"],
                    ):
                        continue
                    row = rows.setdefault(trial["repeat"], {})
                    if report.get("status") != "completed":
                        continue
                    identity = (trial["repeat"], trial["arm"])
                    if identity in seen:
                        raise ValueError("duplicate completed pilot repetition/arm")
                    seen.add(identity)
                    row[trial["arm"]] = {
                        "gpu_seconds": report["summary"]["gpu_seconds"],
                        "requests": report["summary"]["requests"],
                    }
                repeats[family] = sorted(rows)
                self.rows[family] = [rows[r] for r in repeats[family]]
            if self.paired_families and repeats[families[0]] != repeats[families[1]]:
                raise ValueError("paired family pilots need the same repetition identifiers")
        self.pilot_sizes = {f: len(self.rows[f]) for f in families}
        self.repetitions = self.spec.get("repetitions", max(self.pilot_sizes.values())) or 2
        if not isinstance(self.repetitions, int) or self.repetitions < 2:
            raise ValueError("timing simulation needs at least two target repetitions")

    def draw(self, selection, rng):
        a, p = (np.full((self.repetitions, 2, 1), np.nan) for _ in range(2))

        def indices(size):
            outer = rng.integers(size, size=size)
            return rng.choice(outer, size=self.repetitions)

        shared = (
            indices(self.pilot_sizes[self.families[0]])
            if (self.paired_families and self.pilot_sizes[self.families[0]])
            else None
        )
        for f, family in enumerate(self.families):
            rows = self.rows[family]
            if not rows:
                continue
            pilot = []
            for row in rows:
                pair = []
                for role in ("A", "P"):
                    arm = selection[family][role]["arm"]
                    value = row.get(arm)
                    if value is None and self.kind == "placeholder":
                        value = row.get(role)
                    pair.append(value)
                if any(v is None for v in pair):
                    break
                qs = [v["requests"] for v in pair]
                if (
                    any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in qs)
                    or qs[0] != qs[1]
                ):
                    raise ValueError("timing samples need matched question counts for A/P")
                costs = [dedicated_gpu_seconds({"summary": v}) for v in pair]
                pilot.append(costs)
            # Unknown selected pairs remain unknown, even if a future draw could
            # happen to avoid every unmeasured pilot repetition.
            if len(pilot) != len(rows):
                continue
            key = (family, selection[family]["P"]["arm"], selection[family]["A"]["arm"])
            if family in self.native and key not in self.validated:
                reports, cell = self.native[family]
                serving_pairs(
                    reports,
                    p_arm=key[1],
                    a_arm=key[2],
                    cache=cell["cache"],
                    concurrency=cell["concurrency"],
                )
                self.validated.add(key)
            future = shared if shared is not None else indices(len(rows))
            pairs = np.asarray(pilot)[future]
            a[:, f, 0], p[:, f, 0] = pairs.T
        return a, p


def _probability(low, high):
    n, lo, hi = len(low), float(np.mean(low)), float(np.mean(high))
    identified = np.array_equal(low, high)

    def wilson(p):
        z = 1.959963984540054
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return [max(0.0, float(centre - radius)), min(1.0, float(centre + radius))]

    return {
        "probability": lo if identified else None,
        "bounds": [lo, hi],
        "monte_carlo_se": float(np.sqrt(lo * (1 - lo) / n)) if identified else None,
        "bound_endpoint_monte_carlo_se": [float(np.sqrt(v * (1 - v) / n)) for v in (lo, hi)],
        "monte_carlo_95_interval_bounds": [wilson(lo)[0], wilson(hi)[1]],
        "simulations": n,
    }


def _assurance_report(pvalues, effects, families):
    missing = ~np.isfinite(pvalues)
    low = decision_rule(np.where(missing, 1, pvalues), effects)
    high = decision_rule(np.where(missing, 0, pvalues), effects)
    result = {"families": {}, "joint": {}}
    for f, family in enumerate(families):
        result["families"][family] = {
            "prefixes": {
                h: _probability(low["family_prefixes"][:, f, j], high["family_prefixes"][:, f, j])
                for j, h in enumerate(HYPOTHESES)
            },
            "steps": {
                h: _probability(
                    low["hypothesis_pvalues"][:, f, j] <= ALPHA,
                    high["hypothesis_pvalues"][:, f, j] <= ALPHA,
                )
                for j, h in enumerate(HYPOTHESES)
            },
        }
    result["joint"]["prefixes"] = {
        h: _probability(low["both_families"][:, j], high["both_families"][:, j])
        for j, h in enumerate(HYPOTHESES)
    }
    result["joint"]["all_hypotheses"] = result["joint"]["prefixes"]["H2"]
    result["complete"] = bool(not missing.any())
    result["missing_endpoints"] = {
        family: [name for e, name in enumerate(ENDPOINTS) if missing[:, f, e].any()]
        for f, family in enumerate(families)
    }
    return result


def _infer_studies(
    values,
    sequences,
    events,
    *,
    resamples,
    seed,
    batch_size,
    study_batch,
    clustering="crossed",
    count_cache=None,
):
    # Collapse a small batch of studies into the endpoint/family axes so BLAS
    # processes them together. Never allocate simulations x resamples x cases.
    m = values.shape[1]
    result = {
        k: np.empty((m, 2, 4, 2) if k == "interval" else (m, 2, 4))
        for k in ("estimate", "pvalues", "lower", "interval")
    }
    for start in range(0, m, study_batch):
        stop = min(start + study_batch, m)
        n = values[:, start:stop].reshape(len(values), -1, 4)
        found = bootstrap_test(
            n,
            np.ones_like(n),
            events if clustering == "event" else sequences,
            NULLS[:4],
            resamples=resamples,
            seed=seed,
            batch_size=batch_size,
            crossed=events if clustering == "crossed" else None,
            count_cache=count_cache,
        )
        for key in result:
            result[key][start:stop] = found[key].reshape(result[key][start:stop].shape)
    return result


def simulate(
    config,
    *,
    simulations=200,
    resamples=100_000,
    seed=0,
    batch_size=256,
    study_batch=8,
    crossed=True,
    sensitivities=True,
):
    """Nested cluster/selection/predictive assurance for the complete ordered rule."""
    if simulations < 1 or resamples < 39 or study_batch < 1 or batch_size < 1:
        raise ValueError("positive simulation/batch sizes and at least 39 resamples are required")
    if not crossed:
        raise ValueError("registered assurance requires crossed CF resampling")
    timer, inputs = StageTimer(), _Inputs(config)
    spec = inputs.read(str(Path(config).resolve()))
    families = _families(spec)
    panel = _read_panel(spec["panels"]["cf"], families, inputs, candidates=True)
    target = _read_population(spec["populations"]["cf"], inputs)
    catalog = spec.get("candidates")
    mechanism = spec.get("mechanism")
    if not catalog or not mechanism:
        raise ValueError("assurance needs all eligible candidates and fixed H1a F/V names")
    for f in families:
        needed = {c.split("=", 1)[-1] for c in catalog[f]} | set(mechanism[f].values())
        if needed - panel.right[f].keys() or set(mechanism[f]) != {"F", "V"}:
            raise ValueError(f"{f}: complete candidate and fixed F/V outcomes are required")
    model = {"matched_probability": 0.5, "event_share": 0.5, **spec.get("assurance", {})}
    for key in ("matched_probability", "event_share"):
        if not 0 <= model[key] <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    scenarios = {"main": {k: model[k] for k in ("matched_probability", "event_share")}}
    if sensitivities:
        scenarios.update(
            {
                "all_fresh_clusters": {**scenarios["main"], "matched_probability": 0.0},
                "named_cluster_donors": {**scenarios["main"], "matched_probability": 1.0},
                "sequence_donor_fallback": {**scenarios["main"], "event_share": 0.0},
                "event_donor_fallback": {**scenarios["main"], "event_share": 1.0},
            }
        )
    timing = _Timing(spec.get("timing"), inputs, families)
    values = {name: np.empty((len(target.cases), simulations, 2, 4)) for name in scenarios}
    cost_a, cost_p = [], []
    selections, fallbacks = [], {name: [] for name in scenarios}
    outer_seed, predictive_seed, timing_seed = np.random.SeedSequence(seed).spawn(3)
    outer, future, clock = (
        np.random.default_rng(s) for s in (outer_seed, predictive_seed, timing_seed)
    )
    for study in range(simulations):
        counts = _outer_counts(panel, outer)
        selected = _reselect(panel, families, catalog, counts)
        selections.append(selected)
        for name, settings in scenarios.items():
            draw, fallback = _predict_cases(panel, target, counts, future, **settings)
            values[name][:, study] = _case_statistics(panel, families, selected, mechanism, draw)
            fallbacks[name].append(fallback)
        a, p = timing.draw(selected, clock)
        cost_a.append(a)
        cost_p.append(p)
    a = np.stack(cost_a, axis=1).reshape(len(cost_a[0]), -1, 1)
    p = np.stack(cost_p, axis=1).reshape(len(cost_p[0]), -1, 1)

    def infer_costs(numerator):
        result = {
            k: np.empty((simulations, 2, 1, 2) if k == "interval" else (simulations, 2, 1))
            for k in ("estimate", "pvalues", "lower", "interval")
        }
        for start in range(0, simulations, study_batch):
            stop = min(start + study_batch, simulations)
            found = bootstrap_test(
                numerator[:, 2 * start : 2 * stop],
                p[:, 2 * start : 2 * stop],
                list(map(str, range(len(a)))),
                NULLS[4:],
                resamples=resamples,
                seed=seed,
                batch_size=batch_size,
            )
            for key in result:
                result[key][start:stop] = found[key].reshape(result[key][start:stop].shape)
        return result

    costs = infer_costs(a)
    cache, inferred = {}, {}
    for name, study_values in values.items():
        found = _infer_studies(
            study_values,
            target.sequences,
            target.events,
            resamples=resamples,
            seed=seed,
            batch_size=batch_size,
            study_batch=study_batch,
            count_cache=cache,
        )
        inferred[name] = {}
        for key in found:
            c = costs[key].reshape(
                (simulations, 2, 1, 2) if key == "interval" else (simulations, 2, 1)
            )
            inferred[name][key] = np.concatenate([found[key], c], axis=2)
    main = inferred["main"]
    report = _assurance_report(main["pvalues"], main["estimate"], families)
    sensitivity_reports = {
        name: _assurance_report(r["pvalues"], r["estimate"], families)
        for name, r in inferred.items()
        if name != "main"
    }
    if sensitivities:
        for clustering in ("sequence", "event"):
            r = _infer_studies(
                values["main"],
                target.sequences,
                target.events,
                resamples=resamples,
                seed=seed,
                batch_size=batch_size,
                study_batch=study_batch,
                clustering=clustering,
                count_cache=cache,
            )
            pv = np.concatenate([r["pvalues"], main["pvalues"][:, :, 4:]], axis=2)
            sensitivity_reports[f"{clustering}_only"] = _assurance_report(
                pv, main["estimate"], families
            )
        slower = infer_costs(a * 0.9)
        pv = main["pvalues"].copy()
        pv[:, :, 4] = slower["pvalues"][:, :, 0]
        ef = main["estimate"].copy()
        ef[:, :, 4] *= 0.9
        sensitivity_reports["A_cost_10_percent_lower"] = _assurance_report(pv, ef, families)
    for f, family in enumerate(families):
        report["families"][family]["endpoints"] = {
            name: {
                "known_studies": int(np.isfinite(main["estimate"][:, f, e]).sum()),
                "mean_estimate": float(np.nanmean(main["estimate"][:, f, e]))
                if np.isfinite(main["estimate"][:, f, e]).any()
                else None,
                "mean_lower_97_5": float(np.nanmean(main["lower"][:, f, e]))
                if np.isfinite(main["lower"][:, f, e]).any()
                else None,
            }
            for e, name in enumerate(ENDPOINTS)
        }
    return _finite_json(
        {
            "artifact_kind": "xgraph.selection_aware_assurance",
            "input_purpose": spec.get("purpose", "development_assurance"),
            "version": 2,
            **report,
            "registered_inference_inputs_complete": report["complete"]
            and timing.kind == "pilot"
            and resamples == 100_000,
            "timing_status": timing.kind,
            "timing_pilot_repetitions": timing.pilot_sizes,
            "target_serving_repetitions": timing.repetitions,
            "timing_paired_families": timing.paired_families,
            "settings": {
                "simulations": simulations,
                "resamples": resamples,
                "seed": seed,
                "alpha": ALPHA,
                "batch_size": batch_size,
                "study_batch": study_batch,
                "registered_resamples_used": resamples == 100_000,
                "selection_aware": True,
                "crossed_cf_bootstrap": True,
                "predictive_model": model,
                "sensitivities": sensitivities,
                "common_inner_draws": "paired endpoints/families; cached across studies",
            },
            "populations": {"cf": target.summary()},
            "panels": {"cf": panel.description},
            "selection_frequencies": {
                f: {
                    role: {
                        name: sum(s[f][role]["arm"] == name for s in selections) / simulations
                        for name in sorted({s[f][role]["arm"] for s in selections})
                    }
                    for role in ("P", "X", "A")
                }
                for f in families
            },
            "donor_coverage": {
                "source_cases": len(panel.cases),
                "source_sequences": len(set(panel.sequences)),
                "source_events": len(set(panel.events)),
                "source_singleton_sequences": sum(
                    panel.sequences.count(s) == 1 for s in set(panel.sequences)
                ),
                "source_singleton_events": sum(
                    panel.events.count(e) == 1 for e in set(panel.events)
                ),
                "target_cases_with_matched_sequence": sum(
                    s in panel.sequences for s in target.sequences
                ),
                "target_cases_with_matched_event": sum(e in panel.events for e in target.events),
                "empty_donor_cell_fraction": {n: float(np.mean(v)) for n, v in fallbacks.items()},
            },
            "sensitivities": sensitivity_reports,
            "studies": {**main, "selections": selections},
            "interpretation": (
                "Nested empirical crossed-cluster assurance conditional on the eligible catalog, "
                "the explicit donor prediction model and timing distribution. New target sequence "
                "and event donors are drawn inside every outer calibration bootstrap. Empty source "
                "cells use the declared sequence/event mixture. Sparse donors do not identify "
                "unobserved cluster effects; sensitivities vary matching and clustering. "
                "Monte Carlo "
                "SE covers simulation error, not identification of these assumptions. Placeholder "
                "timing never supplies measured serving evidence or permits a freeze. "
                "No confirmation "
                "outcomes are used to select policies. No marginal probabilities are multiplied."
            ),
            "provenance": provenance(sorted(inputs.paths)),
            "resources": _resources(timer),
        }
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("analyze", "simulate"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--config", type=Path, required=True)
        cmd.add_argument("--out", type=Path, required=True)
        cmd.add_argument("--resamples", type=int, default=100_000)
        cmd.add_argument("--seed", type=int, default=0)
        cmd.add_argument("--batch-size", type=int, default=256)
        if name == "simulate":
            cmd.add_argument("--simulations", type=int, default=200)
            cmd.add_argument("--study-batch", type=int, default=8)
            cmd.add_argument(
                "--crossed",
                action="store_true",
                default=True,
                help="registered CF crossed bootstrap (always enabled)",
            )
            cmd.add_argument("--no-sensitivities", action="store_false", dest="sensitivities")
        else:
            cmd.add_argument(
                "--clustering", choices=("crossed", "sequence", "event"), default="crossed"
            )
    args = vars(parser.parse_args(argv))
    command, output = args.pop("command"), args.pop("out")
    try:
        report = (simulate if command == "simulate" else analyze)(**args)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("complete", "resources")}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
