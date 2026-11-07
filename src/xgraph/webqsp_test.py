"""Prospective WebQSP test registration; CPU preparation and explicit owner launch."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import shlex
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist

from xgraph.provenance import sha256_file
from xgraph.queue_accounting import gpu_occupancy

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("qwen38", "gemma31")
REFERENCES = {"qwen38": "agent-strong-8", "gemma31": "agent-feedback-8"}
CORE_ARMS = ("W", "WT", "B2", "agent-strong-8", "agent-feedback-8", "gcr-native", "subgraphrag")
SEED = 17
REPAIRED_PILOT_COMMIT = "32e3b61d9e85d1c96eadcc9ac75a234314155481"
SUPERVISED_PILOT_COMMIT = "8821f0e57cd207d13432f8d49a36d5bcc0c964ae"
OCCUPANCY_PILOT_COMMIT = "0ad21bc57c231513bf4fa5240ea144ccc0c82c87"
WATCHDOG_TESTED_PILOT_COMMIT = "f6549a13cd97fd42f460be503764571dada91153"
GCR_GROUP_COMMIT = "778872bfca5e528140c60cb52643a9c9293abbaa"


def unique_ids(ids):
    ids = list(ids)
    if any(not isinstance(q, str) or not q.strip() for q in ids):
        raise ValueError("question ids must be nonempty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate question ids")
    return ids


def select_ids(ids, count, seed=SEED):
    ids = unique_ids(ids)
    if type(count) is not int or not 0 < count <= len(ids):
        raise ValueError("sample count must be positive and fit the ID population")
    return sorted(ids, key=lambda q: (hashlib.sha256(f"{seed}:{q}".encode()).digest(), q))[:count]


def population_audit(rog, ml):
    r, m = set(unique_ids(rog)), set(unique_ids(ml))
    return {
        "rog": sorted(r),
        "ml": sorted(m),
        "intersection": sorted(r & m),
        "rog_only": sorted(r - m),
        "ml_only": sorted(m - r),
        "rule": "R intersect M; no answer, graph-coverage or retrieval-availability filter",
    }


def parquet_ids(paths):
    """Project the ID column, without requesting any gold or prediction column."""
    import pyarrow.parquet as pq

    return unique_ids(
        [
            q
            for path in paths
            for q in pq.ParquetFile(path)
            .read(columns=["id"], use_threads=False)
            .column("id")
            .to_pylist()
        ]
    )


def positive(value, context):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{context} must be finite and positive")
    return value


def verify_occupancy_prices(registration, timings):
    """Admit measured occupancy only with the producer's completion receipt."""
    pilot = registration["pilot"]
    expected = pilot["occupancy_receipt"]
    receipt = json.loads(Path(expected["path"]).read_text())
    if any(receipt.get(k) != expected[k] for k in ("seal_sha256", "command")):
        raise ValueError("pilot occupancy receipt changed serving seal or command")
    path = Path(pilot["timings"])
    files = [path, path.with_name("queue.json"), path.with_name("gpu-ledger.jsonl")]
    charges, fresh_charges, walls = [], [], []
    declared = pilot.get("restart", {}).get("completed", [])
    reused = []
    for row in timings["per_arm"]:
        if row.get("accounting") != "measured" or row.get("container_launched") is not True:
            raise ValueError("pilot occupancy price requires measured container intervals")
        charge = positive(row.get("gpu_seconds"), "pilot occupancy seconds")
        launcher = positive(row.get("wall_seconds"), "pilot occupancy launcher wall seconds")
        if charge > launcher:
            raise ValueError("pilot occupancy exceeds launcher wall")
        event = row.get("gpu_events", {})
        if not event.get("path") or receipt.get("files", {}).get(event["path"]) != event.get(
            "sha256"
        ):
            raise ValueError("pilot occupancy event identity missing or changed")
        files.append(Path(event["path"]))
        if "reused_receipt" in row:
            item = row["reused_receipt"]
            if item not in declared or item in reused:
                raise ValueError("pilot occupancy reuse receipt is undeclared or repeated")
            verify_identity(item)
            saved = json.loads(Path(item["path"]).read_text())
            if saved["timing"] != {k: v for k, v in row.items() if k != "reused_receipt"}:
                raise ValueError("pilot occupancy reused timing changed")
            files.append(Path(item["path"]))
            reused.append(item)
        else:
            fresh_charges.append(charge)
        charges.append(charge)
        walls.append(launcher)
    # Match the producer's sum reductions, including Python's float summation.
    seconds, fresh_seconds, wall = sum(charges), sum(fresh_charges), sum(walls)
    if len(reused) != len(declared) or any(item not in reused for item in declared):
        raise ValueError("pilot occupancy reuse receipt matrix is incomplete")
    for file in files:
        if receipt.get("files", {}).get(str(file)) != sha256_file(file):
            raise ValueError(f"pilot occupancy receipt identity missing or changed: {file}")
    if json.loads(path.read_text()) != timings:
        raise ValueError("pilot occupancy timings differ from the receipted file")
    restart = pilot.get("restart")
    prior = 0
    if restart is not None:
        prior = positive(restart.get("charged_seconds"), "pilot occupancy prior charge")
        verify_identity(restart["ledger"])
        historical = Path(restart["ledger"]["path"]).read_bytes()
        active = path.with_name("gpu-ledger.jsonl")
        rows = [json.loads(line) for line in active.read_bytes().splitlines()]
        starts = [r["attempt"] for r in rows if r["event"] == "start"]
        ends = {r["attempt"]: r["seconds"] for r in rows if r["event"] == "end"}
        if (
            timings.get("prior_gpu_seconds") != prior
            or not active.read_bytes().startswith(historical)
            or len(starts) != len(set(starts))
            or set(starts) != ends.keys()
            or len(rows) != 2 * len(starts)
            or not math.isclose(sum(ends.values()), prior + fresh_seconds, rel_tol=0, abs_tol=1e-8)
        ):
            raise ValueError("pilot occupancy prior charge or ledger changed")
    if (
        timings.get("measured_gpu_wall_seconds") != seconds
        or timings.get("charged_gpu_seconds") != prior + fresh_seconds
        or timings.get("launcher_wall_seconds") != wall
        or prior + fresh_seconds > 16 * 3600
        or timings.get("pilot_budget_exceeded", False)
    ):
        raise ValueError("pilot occupancy totals unknown, inconsistent or over cap")


def size_protocol(registration, timings=None, gate=None, *, baseline_rates=None):
    gate = apply_registered_gate(registration, gate)
    pricing = registration["pricing"]
    cap, reserve = pricing["cap_seconds"], pricing["reserve_seconds"]
    if not 0 <= reserve < cap <= 40 * 3600:
        raise ValueError("invalid hard cap or reserve")
    measured = {}
    occupancy = registration.get("pilot", {}).get("timing_basis") == "container_occupancy"
    if timings is not None:
        if timings.get("complete") is not True:
            raise ValueError("pilot timings must be complete")
        if occupancy:
            verify_occupancy_prices(registration, timings)
        for row in timings["per_arm"]:
            key = row["family"], row["name"]
            if key in measured:
                raise ValueError("duplicate pilot stage timing")
            if row["returncode"] != 0:
                raise ValueError(f"failed pilot stage: {key}")
            if (
                type(row["questions"]) is not int
                or row["questions"] != registration["pilot"]["questions"]
            ):
                raise ValueError("pilot question count changed")
            field = "gpu_seconds" if occupancy else "wall_seconds"
            measured[key] = positive(row[field], f"pilot {field}") / row["questions"]
        needed = [tuple(v) for v in registration["pilot_stages"]]
        if len(set(needed)) != len(needed) or set(needed) - measured.keys():
            raise ValueError("incomplete unique pilot stage matrix")
        base = sum(measured[key] for key in needed)
    elif pricing.get("cohort_rule") == "two_tier":
        return two_tier_sizing(registration, {}, gate, baseline_rates)
    else:
        base = positive(pricing["unmeasured_base_seconds_per_question"], "price scenario")
    if pricing.get("cohort_rule") == "two_tier":
        result = two_tier_sizing(registration, measured, gate, baseline_rates)
        if occupancy:
            result.update(
                pilot_timing_basis="container_occupancy",
                pilot_gpu_seconds=timings["charged_gpu_seconds"],
                pilot_launcher_wall_seconds=timings["launcher_wall_seconds"],
            )
        return result
    if pricing.get("cohort_rule") == "core_first":
        return core_first_sizing(registration, measured, base, gate, timings is not None)
    extra = pricing["extra_envelope_seconds_per_question"]
    included = []
    if "extra_rates" in pricing:
        extra = 0
        for row in pricing["extra_rates"]:
            condition = row.get("condition")
            if condition and gate is not None and not gate[condition]["admit"]:
                continue
            if condition == "wt_pc" and gate is not None:
                copies = gate["wt_pc"]["K"] * (2 if gate["wt_pc"]["prior"] else 1)
            else:
                copies = row.get("copies", 1)
            rate = max(row["envelope"], measured.get(tuple(row.get("pilot_stage", [])), 0))
            extra += rate * copies
            included.append({**row, "rate": rate, "copies": copies, "measured": False})
    rate = positive(base + extra, "combined stage rate")
    n = min(len(registration["population"]["intersection"]), math.floor((cap - reserve) / rate))
    if n < 1:
        raise ValueError("no complete paired cohort fits the hard cap")
    result = {
        "status": "PILOT_MEASURED_WITH_DECLARED_ENVELOPES" if timings else "UNMEASURED",
        "pilot_measured": timings is not None,
        "all_rates_measured": not extra and timings is not None,
        "questions_per_arm": n,
        "base_seconds_per_question": base,
        "extra_seconds_per_question": extra,
        "extra_rates": included,
        "projected_seconds_with_reserve": reserve + n * rate,
        "cap_seconds": cap,
        "reserve_seconds": reserve,
        "formula": (
            "n=min(|R intersect M|, floor((40*3600-reserve-sum(startups))/sum(unique stage rates)))"
        ),
        "limits": "projection only; owner ledger/watchdog enforces the cap; no test outcomes used",
    }
    if registration.get("prior"):
        result["power"] = power_report(registration["prior"], n)
        result["power"]["joint80_required_n_within_population"] = {
            scenario: next(
                (
                    count
                    for count in range(1, len(registration["population"]["intersection"]) + 1)
                    if power_report(registration["prior"], count)[field] >= 0.8
                ),
                None,
            )
            for scenario, field in (
                ("zero_effect", "joint_NI_lower_bound_zero_effect"),
                ("development_effect", "joint_NI_lower_bound_development_effect"),
            )
        }
    return result


def cluster_subpanel(ids, assignments, limit, seed=SEED):
    """One seeded cluster order; take its longest whole-cluster prefix within limit."""
    groups = {}
    for q in ids:
        groups.setdefault(assignments[q], []).append(q)
    ordered = sorted(groups, key=lambda c: (hashlib.sha256(f"{seed}:{c}".encode()).digest(), c))
    selected = []
    for cluster in ordered:
        if len(selected) + len(groups[cluster]) > limit:
            break
        selected.extend(sorted(groups[cluster]))
    return selected


def descriptive_power(n, clusters):
    """Declared worst-case variance screen, never measured baseline assurance."""
    z = NormalDist()
    summary = cluster_summary(clusters) if clusters else None
    numerator = z.inv_cdf(0.975) + z.inv_cdf(0.9)
    return {
        "questions": n,
        "clusters": summary,
        "paired_variance_assumption": 1,
        "joint80_detectable_difference": numerator / math.sqrt(n) if n else None,
        "joint80_detectable_difference_perfect_cluster_dependence": numerator
        / math.sqrt(summary["effective_clusters"])
        if n
        else None,
        "basis": "Normal screen, paired binary variance bound 1, union bound over two families; "
        "unknown within-cluster dependence can reduce power; descriptive only",
    }


def semantic_hops(support):
    """Semantic depth of authoritative forward-chain annotations, never generated plans."""
    if not support or not support.get("paths") or "operations" not in support:
        return {
            "status": "UNCLASSIFIED",
            "reason": "missing authoritative chain/operation annotation",
        }
    if any(op != "traverse" for op in support["operations"]):
        return {"status": "UNSUPPORTED", "reason": "operation outside forward entity-chain support"}
    depths = set()
    for path in support["paths"]:
        nodes, relations = path.get("nodes", []), path.get("relations", [])
        if (
            len(nodes) < 2
            or len(relations) != len(nodes) - 1
            or any(not n.get("id") or type(n.get("cvt")) is not bool for n in nodes)
            or any(not isinstance(r, str) or not r for r in relations)
        ):
            return {"status": "UNCLASSIFIED", "reason": "missing chain or authoritative CVT type"}
        if nodes[0]["cvt"] or nodes[-1]["cvt"] or len({n["id"] for n in nodes}) != len(nodes):
            return {"status": "UNSUPPORTED", "reason": "CVT endpoint or cyclic chain"}
        depths.add(len(relations) - sum(n["cvt"] for n in nodes[1:-1]))
    if len(depths) != 1:
        return {"status": "AMBIGUOUS", "reason": "released parses disagree on semantic depth"}
    return {"status": "SUPPORTED", "hops": depths.pop()}


def depth_interaction(pairs, hops, clusters, *, resamples=10000, seed=SEED, minimum_clusters=20):
    """Resample whole worlds/clusters together, preserving cross-depth dependence."""
    import numpy as np

    groups, excluded = {}, []
    counts = [0, 0]
    cluster_sets = [set(), set()]
    for q, (wt, agent) in pairs.items():
        annotation = hops.get(q, {})
        h = annotation.get("hops")
        if annotation.get("status") != "SUPPORTED" or type(h) is not int or h < 1:
            excluded.append(q)
            continue
        stratum = 0 if h >= 2 else 1
        row = groups.setdefault(clusters[q], [0.0, 0.0, 0.0, 0.0])
        row[2 * stratum] += wt - agent
        row[2 * stratum + 1] += 1
        counts[stratum] += 1
        cluster_sets[stratum].add(clusters[q])
    result = {
        "excluded_ids": sorted(excluded),
        "questions_long_one": counts,
        "clusters": len(groups),
        "resamples": resamples,
        "seed": seed,
        "method": "paired whole-world/cluster draws jointly across both depths",
        "multiplicity": "separate secondary family, outside primary",
        "passes_unadjusted": False,
        "p_value": 1.0,
    }
    if not all(counts):
        return {**result, "status": "UNAVAILABLE", "reason": "empty supported depth stratum"}
    values = np.asarray(list(groups.values()))
    sums = values.sum(axis=0)
    point = sums[0] / sums[1] - sums[2] / sums[3]
    rng, samples = np.random.default_rng(seed), []
    for start in range(0, resamples, 128):
        selected = rng.integers(0, len(values), size=(min(128, resamples - start), len(values)))
        draw = values[selected].sum(axis=1)
        valid = (draw[:, 1] > 0) & (draw[:, 3] > 0)
        samples.extend(draw[valid, 0] / draw[valid, 1] - draw[valid, 2] / draw[valid, 3])
    if not samples:
        return {**result, "status": "UNAVAILABLE", "reason": "no complete bootstrap draw"}
    draws = np.asarray(samples)
    low, high = np.quantile(draws, [0.025, 0.975])
    p = (1 + np.count_nonzero(draws - point >= point)) / (len(draws) + 1)
    enough = (
        all(len(c) >= minimum_clusters for c in cluster_sets) and len(draws) >= 0.95 * resamples
    )
    return {
        **result,
        "status": "COMPLETE",
        "difference": float(point),
        "low": float(low),
        "high": float(high),
        "p_value": float(p),
        "one_sided_lower": float(low),
        "enough_clusters": enough,
        "valid_resamples": len(draws),
        "passes_unadjusted": bool(enough and low > 0 and p <= 0.025),
    }


def score_depth(registration, execution, results):
    from xgraph.mintqa_validation import intersection_holm

    definition = registration["analysis"].get("depth_interaction")
    if definition is None:
        return None
    sealed = execution["gate"].get("depth_support")
    if sealed is None:
        return {
            "status": "UNAVAILABLE",
            "outside_primary": True,
            "reason": "No authoritative chain/CVT source identity was frozen before generation",
        }
    if set(sealed.get("sources", {})) != {"annotations", "cvt_types"}:
        raise ValueError("semantic depth needs frozen annotation and authoritative CVT source pins")
    for item in [sealed["file"], *sealed["sources"].values()]:
        verify_identity(item)
    data = json.loads(Path(sealed["file"]["path"]).read_text())
    if (
        data.get("definition") != definition["definition"]
        or data.get("sources") != sealed["sources"]
    ):
        raise ValueError("semantic depth annotation definition or source identity changed")
    unique_ids(row["id"] for row in data["questions"])
    support = {r["id"]: semantic_hops(r) for r in data["questions"]}
    by_family = {}
    for family in FAMILIES:
        reference = registration.get("strongest_agents", REFERENCES)[family]
        a = {r["id"]: r["gcr_hit"] for r in results[f"primary/{family}/WT"]["per_question"]}
        b = {
            r["id"]: r["gcr_hit"] for r in results[f"primary/{family}/{reference}"]["per_question"]
        }
        by_family[family] = {
            "DEPTH": depth_interaction(
                {q: (a[q], b[q]) for q in execution["ids"]},
                support,
                execution["cluster"],
                resamples=registration["analysis"].get("resamples", 10000),
                minimum_clusters=registration["analysis"].get("minimum_clusters", 20),
            )
        }
    return {
        "outside_primary": True,
        "definition": definition,
        "source": sealed,
        "classifications": {
            q: support.get(q, {"status": "UNCLASSIFIED"}) for q in execution["ids"]
        },
        "by_family": by_family,
        "secondary_intersection_holm": intersection_holm(
            by_family, ["DEPTH"], FAMILIES, alpha=0.025
        ),
    }


def two_tier_sizing(registration, measured, gate, baseline_rates):
    pricing = registration["pricing"]
    cap, reserve = pricing["cap_seconds"], pricing["reserve_seconds"]
    ceiling = positive(pricing["baseline_cap_seconds"], "baseline ceiling")
    minimum = pricing["minimum_baseline_questions"]
    if ceiling > 12 * 3600 or type(minimum) is not int or minimum < 100:
        raise ValueError("baseline ceiling must be <=12 GPU-h and minimum at least 100 questions")
    formula = (
        "n_core=min(|U|,floor((cap-reserve)/C_core)); "
        "B_panel=separate post-freeze allocation <=12*3600; "
        "share=B_panel/number of admitted baseline groups; "
        "n_panel=longest seed-17 whole-cluster prefix of core within "
        "min(n_core,min(floor(share/C_baseline))); "
        "minimum 100; no baseline may shrink core"
    )
    if not measured:
        return {
            "status": "UNMEASURED",
            "pilot_measured": False,
            "all_rates_measured": False,
            "execution_ready": False,
            "questions_per_arm": None,
            "formula": formula,
            "limits": "No inherited envelope is a measured two-tier price",
        }
    core = [tuple(s) for s in pricing["core_stages"]]
    if len(set(core)) != len(core) or set(core) - measured.keys():
        raise ValueError("incomplete unique core pilot stage matrix")
    rate = positive(sum(measured[s] for s in core), "core stage rate")
    n = min(len(registration["population"]["intersection"]), math.floor((cap - reserve) / rate))
    if n < 1:
        raise ValueError("no complete core cohort fits the hard cap")
    ids = select_ids(registration["population"]["intersection"], n)
    method = registration["analysis"]["cluster_method"]
    assignments = registration["question_cluster_audit"]["methods"][method]["assignments"]
    budget = ceiling
    decisions, candidates = {}, {}
    baseline_rates = baseline_rates or {}
    for name in pricing["baseline_order"]:
        row = {"admit": False}
        if gate is not None and name in gate and gate[name]["admit"] is False:
            row.update(status="DEVELOPMENT_EXCLUDED", reason=gate[name]["reason"])
        elif name == "dog" and (gate is None or not gate["dog"]["admit"]):
            row.update(status="PENDING_DEVELOPMENT_GATE")
        elif name not in baseline_rates:
            row.update(status="PENDING_MEASURED_PRICE")
        elif baseline_rates[name].get("unavailable") is True:
            row.update(status="NATIVE_PRICING_UNAVAILABLE", reason=baseline_rates[name]["reason"])
        else:
            expected = {"/".join(s) for s in pricing["baseline_stages"][name]}
            if set(baseline_rates[name]) != expected:
                raise ValueError(f"{name}: incomplete unique matched baseline stage matrix")
            cost = sum(positive(v, "matched baseline rate") for v in baseline_rates[name].values())
            row.update(seconds_per_question=cost, minimum_questions=minimum)
            candidates[name] = cost
        decisions[name] = row
    panel_ids = []
    # Split the independent bucket equally. Remove the most expensive infeasible
    # group first, then recompute shares; registered order breaks equal-rate ties.
    while candidates:
        share = budget / len(candidates)
        expensive = max(candidates, key=candidates.get)
        limit = min(n, math.floor(share / candidates[expensive]))
        panel_ids = cluster_subpanel(ids, assignments, limit)
        if len(panel_ids) >= minimum:
            for name, cost in candidates.items():
                decisions[name].update(
                    admit=True,
                    status="ADMITTED_MEASURED",
                    budget_seconds=share,
                    maximum_questions=min(n, math.floor(share / cost)),
                )
                if name == "gcr" and registration.get("gcr_amortization"):
                    decisions[name]["launcher_wall_caps_seconds"] = {
                        family: share * baseline_rates[name][f"{family}/gcr-native"] / cost
                        for family in FAMILIES
                    }
            break
        decisions[expensive].update(
            status="INFEASIBLE_AT_BOUND",
            budget_seconds=share,
            maximum_questions=limit,
            reason="minimum whole-cluster subpanel does not fit its equal share of the "
            "separate post-freeze published-baseline allocation",
        )
        del candidates[expensive]
        panel_ids = []
    panel_rate = sum(candidates.values())
    result = {
        "status": "PILOT_MEASURED",
        "pilot_measured": True,
        "all_rates_measured": all(
            not d["status"].startswith("PENDING") for d in decisions.values()
        ),
        "execution_ready": all(not d["status"].startswith("PENDING") for d in decisions.values()),
        "cohort_rule": "two_tier",
        "questions_per_arm": n,
        "core_questions_per_arm": n,
        "core_seconds_per_question": rate,
        "core_projected_seconds_with_reserve": reserve + n * rate,
        "baseline_budget_seconds": budget,
        "baseline_ceiling_seconds": ceiling,
        "baseline_allocation": "separate_post_freeze",
        "baseline_projected_seconds": len(panel_ids) * panel_rate,
        "baseline_seconds_per_question": panel_rate,
        "baseline_ids": panel_ids,
        "baseline_questions_per_arm": len(panel_ids),
        "baseline_admission": decisions,
        "baseline_power": descriptive_power(len(panel_ids), {q: assignments[q] for q in panel_ids}),
        "projected_seconds_with_reserve": reserve + n * rate + len(panel_ids) * panel_rate,
        "cap_seconds": cap,
        "reserve_seconds": reserve,
        "formula": formula,
        "limits": "Baseline contrasts secondary/descriptive on one subpanel; competitive claim "
        "limited to frozen strongest-agent NI on the core; ledger enforces separate "
        "core and published-baseline lifetime caps",
    }
    if registration.get("gcr_native_pricing") and "gcr" in decisions:
        from xgraph.webqsp_gcr_pricing import PUBLISHED_COMPARISON

        decision = decisions["gcr"]
        if decision["admit"]:
            decision["comparison"] = "FULL_COHORT" if len(panel_ids) == n else "SEEDED_SUBPANEL"
        elif decision["status"].startswith("PENDING"):
            decision["comparison"] = "PENDING_NATIVE_WEBQSP_PRICE"
        else:
            decision.update(
                comparison="PUBLISHED_NON_PAIRED", published_comparison=PUBLISHED_COMPARISON
            )
        result["gcr_power_consequence"] = {
            "core_questions": n,
            "paired_questions": len(panel_ids) if decision["admit"] else 0,
            "standard_error_multiplier": math.sqrt(n / len(panel_ids))
            if decision["admit"]
            else None,
            "basis": "sqrt(n_core/n_panel) at equal paired variance; baseline contrasts remain "
            "secondary/descriptive; clustering can further reduce power; no primary NI change",
        }
    if registration.get("prior"):
        result["power"] = power_report(registration["prior"], n)
        result["core_power_qualified"] = result["power"]["joint_NI_lower_bound_zero_effect"] >= 0.8
        result["execution_ready"] &= result["core_power_qualified"]
    else:
        result["core_power_qualified"] = None
    return result


def core_first_sizing(registration, measured, base, gate, pilot_measured):
    """Fix the complete core cohort before testing optional arms against its unused budget."""
    pricing = registration["pricing"]
    cap, reserve = pricing["cap_seconds"], pricing["reserve_seconds"]
    core, optional = [], {}
    for row in pricing["extra_rates"]:
        condition = row.get("condition")
        copies = row.get("copies", 1)
        if condition == "wt_pc" and gate is not None and gate[condition]["admit"]:
            copies = gate[condition]["K"] * (2 if gate[condition]["prior"] else 1)
        positive(copies, "stage copies")
        # A declared matched stage may replace its unmeasured envelope. Other
        # measurements (e.g. WT revision as a PC proxy) only increase the bound.
        matched = [tuple(s) for s in row.get("matched_pilot_stages", [])]
        qualified = bool(matched) and all(s in measured for s in matched)
        rate = (
            sum(measured[s] for s in matched)
            if qualified
            else max(
                positive(row["envelope"], "stage envelope"),
                measured.get(tuple(row.get("pilot_stage", [])), 0),
            )
        )
        item = {**row, "rate": rate, "copies": copies, "measured": qualified}
        if condition:
            optional.setdefault(condition, []).append(item)
        else:
            core.append(item)
    order = pricing["optional_order"]
    if len(set(order)) != len(order) or optional.keys() - set(order):
        raise ValueError("optional arms need a unique registered admission order")
    core_extra = sum(r["rate"] * r["copies"] for r in core)
    core_rate = positive(base + core_extra, "core stage rate")
    n = min(
        len(registration["population"]["intersection"]), math.floor((cap - reserve) / core_rate)
    )
    if n < 1:
        raise ValueError("no complete core cohort fits the hard cap")
    occupied = reserve + n * core_rate
    included, decisions = list(core), {}
    for name in order:
        if name not in optional:
            continue
        rows = optional[name]
        seconds = n * sum(r["rate"] * r["copies"] for r in rows)
        qualified = gate is not None and gate[name]["admit"]
        fits = occupied + seconds <= cap
        admit = qualified and fits
        reason = (
            "development gate failed (resolved)"
            if name == "wt_pc" and registration.get("conditional_rule", {}).get("wt_pc_resolution")
            else "development gate pending"
            if gate is None
            else "development gate excluded"
            if not qualified
            else "fits the complete core cohort"
            if fits
            else "excluded: complete core cohort plus this optional arm exceeds the cap"
        )
        decisions[name] = {
            "development_qualified": qualified,
            "admit": admit,
            "budget_fits": fits,
            "reason": reason,
            "seconds_at_core_n": seconds,
            "rates": rows,
        }
        if admit:
            occupied += seconds
            included.extend(rows)
    extra = sum(r["rate"] * r["copies"] for r in included)
    result = {
        "status": "PILOT_MEASURED_WITH_DECLARED_ENVELOPES" if pilot_measured else "UNMEASURED",
        "pilot_measured": pilot_measured,
        "all_rates_measured": pilot_measured and all(r["measured"] for r in included),
        "cohort_rule": "core_first",
        "core_questions_per_arm": n,
        "questions_per_arm": n,
        "base_seconds_per_question": base,
        "core_seconds_per_question": core_rate,
        "core_projected_seconds_with_reserve": reserve + n * core_rate,
        "extra_seconds_per_question": extra,
        "extra_rates": included,
        "optional_admission": decisions,
        "projected_seconds_with_reserve": occupied,
        "cap_seconds": cap,
        "reserve_seconds": reserve,
        "formula": (
            "n_core=min(|U|, floor((cap-reserve)/sum(unique mandatory-core stage rates))); "
            "optional arms in registered order require reserve+n_core*sum(admitted rates)<=cap"
        ),
        "limits": "projection only; never shrink the core for an optional arm; ledger enforces cap",
    }
    if registration.get("prior"):
        result["power"] = power_report(registration["prior"], n)
        result["power"]["joint80_required_n_within_population"] = {
            scenario: next(
                (
                    count
                    for count in range(1, len(registration["population"]["intersection"]) + 1)
                    if power_report(registration["prior"], count)[field] >= 0.8
                ),
                None,
            )
            for scenario, field in (
                ("zero_effect", "joint_NI_lower_bound_zero_effect"),
                ("development_effect", "joint_NI_lower_bound_development_effect"),
            )
        }
    return result


def apply_budget_admission(gate, sizing):
    """Retain qualification evidence while recording a distinct prospective budget exclusion."""
    effective = copy.deepcopy(gate)
    if sizing.get("cohort_rule") == "two_tier":
        effective["baseline_admission"] = sizing["baseline_admission"]
        dog = sizing["baseline_admission"]["dog"]
        if effective["dog"]["admit"] and not dog["admit"]:
            effective["dog"].update(admit=False, reason=dog.get("reason", dog["status"]))
        return effective
    for name, row in sizing.get("optional_admission", {}).items():
        if effective[name]["admit"] and not row["admit"]:
            effective[name].update(admit=False, reason=row["reason"])
    return effective


def apply_registered_gate(registration, gate):
    """Keep a resolved negative development condition from being re-admitted."""
    if registration.get("gcr_native_pricing") and gate is not None and "gcr" in gate:
        raise ValueError(
            "native WebQSP GCR decisions require its registered price, not a bench gate"
        )
    resolution = registration.get("conditional_rule", {}).get("wt_pc_resolution")
    if gate is None or resolution is None:
        return gate
    if resolution.get("admit") is not False or gate.get("wt_pc", {}).get("admit") is True:
        raise ValueError("resolved WT-PC development gate excludes this secondary")
    effective = copy.deepcopy(gate)
    effective["wt_pc"] = copy.deepcopy(resolution)
    return effective


def power_report(prior, n, *, effective_questions=None):
    normal = NormalDist()
    rows = {}
    effective = n if effective_questions is None else positive(effective_questions, "effective n")
    for family, row in prior["families"].items():
        variance = positive(row["transport_variance"], "development transport variance")
        se = math.sqrt(variance / effective)
        rows[family] = {
            "NI_at_zero_effect": normal.cdf(0.03 / se - normal.inv_cdf(0.975)),
            "NI_at_development_effect": normal.cdf(
                (row["effect"] + 0.03) / se - normal.inv_cdf(0.975)
            ),
            "superiority_at_development_effect_Holm_two": normal.cdf(
                row["effect"] / se - normal.inv_cdf(0.9875)
            ),
            "discordance": row["discordance"],
            "design_effect": row["design_effect"],
        }
    return {
        "questions": n,
        "assumed_effective_questions": effective,
        "families": rows,
        "joint_NI_lower_bound_zero_effect": max(
            0, 1 - sum(1 - r["NI_at_zero_effect"] for r in rows.values())
        ),
        "joint_NI_lower_bound_development_effect": max(
            0, 1 - sum(1 - r["NI_at_development_effect"] for r in rows.values())
        ),
        "basis": (
            "MINTQA development discordance and cluster variance; "
            "normal approximation; transport not guaranteed"
        ),
    }


def gate_decision(gate):
    decided = datetime.fromisoformat(gate["decided_at"])
    if decided.tzinfo is None or decided > datetime(2026, 11, 13, 22, 59, 59, tzinfo=UTC):
        raise ValueError("development gate deadline is November 13 in Europe/Berlin")
    if gate.get("development_only") is not True or gate.get("webqsp_test_accessed") is not False:
        raise ValueError("gate must certify development-only decisions without WebQSP test access")
    result = {}
    for name in ("wt_pc", "dog"):
        row = gate[name]
        if type(row.get("admit")) is not bool:
            raise ValueError(f"{name}: a recorded yes/no development decision is required")
        result[name] = row["admit"]
        if not row["admit"]:
            if not row.get("reason"):
                raise ValueError(f"{name}: excluded gate needs a reason")
            continue
        if not row.get("evidence") or not row.get("serving_qualified"):
            raise ValueError(
                f"{name}: admission needs hash-bound development evidence and serving qualification"
            )
        if name == "wt_pc":
            if row.get("K") not in (2, 4) or type(row.get("prior")) is not bool:
                raise ValueError("WT-PC must freeze shared K=2/4 and prior on/off")
            for field, threshold in (
                ("crossfit_improvement", 0.005),
                ("loso_min", 0),
                ("equal_compute_improvement", 0),
            ):
                values = row.get(field, {})
                if set(values) != set(FAMILIES) or any(
                    not math.isfinite(v) or (v < threshold if threshold else v <= 0)
                    for v in values.values()
                ):
                    raise ValueError(f"WT-PC failed development evidence: {field}")
            if (
                row.get("order_sensitivity_reduced") != dict.fromkeys(FAMILIES, True)
                or row.get("cuda_agreement") is not True
            ):
                raise ValueError("WT-PC needs reduced order sensitivity and CUDA/oracle agreement")
            if not row.get("source_root") or len(row.get("source_commit", "")) != 40:
                raise ValueError("WT-PC needs a frozen implementation commit")
        elif any(
            row.get(k) is not True
            for k in ("pilot_complete", "usage_conserved", "released_graph_qualified")
        ):
            raise ValueError("DoG needs complete pilot and released-graph qualification")
    if "gcr" in gate:
        row = gate["gcr"]
        decided = datetime.fromisoformat(row.get("decided_at", gate["decided_at"]))
        if (
            row.get("admit") is not False
            or not row.get("reason")
            or not row.get("evidence")
            or decided.tzinfo is None
            or decided > datetime(2026, 11, 23, 22, 59, 59, tzinfo=UTC)
        ):
            raise ValueError("GCR exclusion needs hash-bound, pre-freeze bench infeasibility")
        for semantics in ("hf-group", "hf"):
            cell = row.get("bench", {}).get(semantics, {})
            if (
                cell.get("native") is not True
                or cell.get("batch_size") != 1
                or cell.get("questions") != 8
                or type(cell.get("returncode")) is not int
            ):
                raise ValueError(f"GCR {semantics}: missing native eight-ID b1 bench evidence")
            failed = cell["returncode"] != 0 or cell.get("timeout") is True
            if not failed:
                if cell.get("complete") is not True:
                    raise ValueError(f"GCR {semantics}: unfinished bench evidence stays pending")
                failed = positive(cell.get("wall_seconds"), "GCR bench wall seconds") / 8 > 600
            if not failed:
                raise ValueError(f"GCR {semantics}: feasible at the registered per-question bound")
        result["gcr"] = False
    return result


def require_complete(records, ids):
    from xgraph.mintqa_rematch import audit_complete_usage

    ids = unique_ids(ids)
    actual = unique_ids(r["id"] for r in records)
    if set(actual) != set(ids):
        raise ValueError("incomplete arm: missing or foreign selected question ids")
    # generate() emits exactly one completion for these native arms. Match the
    # rematch `source` adapter's contract, without editing the sealed records or
    # guessing counts for thought, fallback, composite or multi-call readers.
    accounted = []
    for row in records:
        if (
            row.get("arm") in ("baseline", "constrained")
            and "usage" not in row
            and "model_calls" not in row
            and not any(
                k in row for k in ("rounds", "calls", "turns", "thought", "stages", "cost_sources")
            )
            and isinstance(row.get("token_ids"), list)
            and row.get("generated_tokens", len(row["token_ids"])) == len(row["token_ids"])
        ):
            row = {**row, "model_calls": 1}
        accounted.append(row)
    try:
        return audit_complete_usage(accounted, ids)
    except KeyError as error:
        raise ValueError(f"incomplete call/token accounting: {error}") from error


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_identity(item):
    if identity(item["path"]) != item:
        raise ValueError(f"frozen input changed: {item['path']}")


def write_sealed(path, value):
    path = Path(path)
    if path.exists() or Path(f"{path}.sha256").exists():
        raise FileExistsError(f"immutable seal already exists: {path}")
    write_json(path, value)
    with Path(f"{path}.sha256").open("x") as out:
        out.write(sha256_file(path) + "\n")


def read_sealed(path):
    path = Path(path)
    if sha256_file(path) != Path(f"{path}.sha256").read_text().strip():
        raise ValueError(f"SHA-256 seal mismatch: {path}")
    return json.loads(path.read_text())


def cpu_job(family, name, *argv, output=None, outputs=None, scope="primary"):
    return {
        "family": family,
        "name": name,
        "scope": scope,
        "kind": "cpu",
        "env": {
            "CUDA_VISIBLE_DEVICES": "",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "XGRAPH_LEXICAL_SCHEMA": "freebase",
        },
        "command": [
            "nice",
            "-n",
            "10",
            "prlimit",
            "--as=4294967296",
            str(ROOT / ".venv/bin/python"),
            "-I",
            "-m",
            "xgraph.webqsp_test",
            *map(str, argv),
        ],
        "outputs": list(map(str, outputs or [output])),
    }


def build_jobs(config, gate=None):
    """Construct the complete DAG without importing a model/server or reading gold."""
    from xgraph.webqsp_catalog import catalog_plan

    if config.get("cohort_rule") == "two_tier":
        return two_tier_jobs(config, gate)
    out = Path(config["out"]) / "execution"
    data, run = out / "inputs", out / "runs"
    models = {k: Path(v) for k, v in config["models"].items()}
    include_gcr = gate is None or gate.get("gcr", {}).get("admit", True)
    plan = catalog_plan(
        data,
        run,
        models,
        arms=["W", "WT", "B2", "agent-strong-8", *(["gcr-native"] if include_gcr else [])],
        timing_root=Path(config["source_root"]),
        gcr_model=Path(config["gcr_model"]),
    )
    jobs, arms = [], []
    for original in plan["jobs"]:
        name = original["name"]
        if name.endswith(("-score", "-declared-score")) or name == "agent-strong-8-sets":
            continue
        job = copy.deepcopy(original)
        for field in (
            "questions",
            "assumed_seconds_per_question",
            "timing_assumption",
            "startup_seconds",
        ):
            job.pop(field, None)
        job["scope"] = "primary"
        command = job["command"]
        if job["kind"] == "gpu":
            at = next(i for i, s in enumerate(command) if s.endswith("/scripts/generate.sh"))
            job["requests"] = command[at + 2]
            job["outputs"] = [command[at + 4]]
            job["env"].update(
                XG_NEED_GIB="75" if job["family"] == "gemma31" else "60",
                XG_AVAILABLE_WAIT="43200",
                XG_START_ATTEMPTS="1",
            )
        else:
            job["command"] = ["--as=4294967296" if s.startswith("--as=") else s for s in command]
            job["outputs"] = [command[command.index("--out") + 1]]
        jobs.append(job)
        if name == "agent-strong-8":
            feedback = copy.deepcopy(job)
            feedback["name"] = "agent-feedback-8"
            at = feedback["command"].index(str(models[job["family"]]))
            feedback["command"][at + 3] = str(run / job["family"] / "agent-feedback-8.jsonl")
            feedback["command"].append("--agent-feedback")
            feedback["outputs"] = [feedback["command"][at + 3]]
            jobs.append(feedback)
    for family in FAMILIES:
        folder = run / family
        for arm in CORE_ARMS[:-1]:
            if arm == "gcr-native" and not include_gcr:
                continue
            arms.append(
                {
                    "family": family,
                    "arm": arm,
                    "scope": "primary",
                    "output": str(
                        folder
                        / (f"{arm}-sets.jsonl" if arm in ("W", "WT", "B2") else f"{arm}.jsonl")
                    ),
                }
            )
        base = next(j for j in jobs if j["family"] == family and j["name"] == "agent-strong-8")
        neutral = {
            **base["env"],
            "XG_TOOLS_DB": "",
            "XG_KG_INDEX": "",
            "XG_TOKENIZER": "",
            "XG_KG_VERSIONS": "",
            "XG_RELATION_LABELS": "",
        }
        first, retry_requests, retry, merged = [
            folder / f"subgraphrag-{n}.jsonl"
            for n in ("first", "retry-requests", "retry", "answers")
        ]
        request = data / "requests-subgraphrag.jsonl"
        for name, requests, output in (("first", request, first), ("retry", retry_requests, retry)):
            gpu = {
                "kind": "gpu",
                "family": family,
                "scope": "primary",
                "name": f"subgraphrag-{name}",
                "env": neutral,
                "requests": str(requests),
                "outputs": [str(output)],
                "command": [
                    "nice",
                    "-n",
                    "10",
                    str(ROOT / "scripts/generate.sh"),
                    str(models[family]),
                    str(requests),
                    "baseline",
                    str(output),
                    "--max-tokens",
                    "256",
                    "--max-model-len",
                    "16384",
                    "--max-num-seqs",
                    "8",
                    "--kv-cache-gib",
                    "16" if family == "gemma31" else "8",
                    "--seed",
                    "17",
                    *(["--ban-tokens", "<|channel>"] if family == "gemma31" else []),
                ],
            }
            if name == "retry":
                gpu["condition"] = "nonempty_requests"
            jobs.append(gpu)
            if name == "first":
                jobs.append(
                    cpu_job(
                        family,
                        "subgraphrag-retry-requests",
                        "reader-retry",
                        "--requests",
                        request,
                        "--first",
                        first,
                        "--prompts",
                        config["prompts"],
                        "--out",
                        retry_requests,
                        output=retry_requests,
                    )
                )
        jobs.append(
            cpu_job(
                family,
                "subgraphrag-merge",
                "reader-merge",
                "--requests",
                request,
                "--first",
                first,
                "--second",
                retry,
                "--out",
                merged,
                output=merged,
            )
        )
        arms.append(
            {"family": family, "arm": "subgraphrag", "scope": "primary", "output": str(merged)}
        )
    if gate and gate["dog"]["admit"]:
        for family in FAMILIES:
            keep = {
                "W",
                "W-sets",
                "WT-requests",
                "WT-raw",
                "WT-select",
                "WT-sets",
                REFERENCES[family],
            }
            for original in list(jobs):
                if (
                    original["family"] != family
                    or original["name"] not in keep
                    or original["scope"] != "primary"
                ):
                    continue
                job = json.loads(
                    json.dumps(original)
                    .replace(str(data), str(out / "dog-inputs"))
                    .replace(str(run), str(out / "dog-runs"))
                )
                job["scope"] = "dog"
                jobs.append(job)
            folder = out / "dog-runs" / family
            request = folder / "dog-requests.jsonl"
            jobs.append(
                cpu_job(
                    family,
                    "dog-requests",
                    "dog-requests",
                    "--data",
                    out / "dog-inputs",
                    "--out",
                    request,
                    output=request,
                    scope="dog",
                )
            )
            raw = folder / "dog.jsonl"
            template = copy.deepcopy(
                next(j for j in jobs if j["family"] == family and j["name"] == "agent-strong-8")
            )
            template.update(name="dog", scope="dog", requests=str(request), outputs=[str(raw)])
            template["env"].update(XG_GCR_SYNTHESIS_MODEL="", XG_TOOLS_DB="")
            template["command"] = [
                str(ROOT / "scripts/generate.sh"),
                str(models[family]),
                str(request),
                "dog",
                str(raw),
                "--dog-prompts",
                config["dog_prompts"],
                "--dog-beam",
                "1",
                "--dog-steps",
                "5",
                "--dog-batch-size",
                "32",
                "--max-tokens",
                "512",
                "--max-model-len",
                "16384",
                "--kv-cache-gib",
                "16" if family == "gemma31" else "8",
                "--seed",
                "17",
            ]
            jobs.append(template)
            for arm in ("WT", REFERENCES[family], "dog"):
                arms.append(
                    {
                        "family": family,
                        "arm": arm,
                        "scope": "dog",
                        "output": str(
                            folder / ("WT-sets.jsonl" if arm == "WT" else f"{arm}.jsonl")
                        ),
                    }
                )
    if gate and gate["wt_pc"]["admit"]:
        for family in FAMILIES:
            folder = run / family
            pc = gate["wt_pc"]
            files = [
                folder / f"pc-{kind}-{view}.requests.jsonl"
                for kind in (("question", "prior") if pc["prior"] else ("question",))
                for view in range(pc["K"])
            ]
            jobs.append(
                cpu_job(
                    family,
                    "WT-PC-requests",
                    "pc-requests",
                    "--execution",
                    out,
                    "--family",
                    family,
                    output=folder / "pc-support.json",
                    outputs=[folder / "pc-support.json", *files],
                )
            )
            for request in files:
                kind, view = request.name.removesuffix(".requests.jsonl").split("-")[1:3]
                view_seed = 17 + int(view)
                template = copy.deepcopy(
                    next(
                        j
                        for j in jobs
                        if j["family"] == family
                        and j["name"] == "WT-raw"
                        and j["scope"] == "primary"
                    )
                )
                raw = request.with_name(request.name.replace(".requests", ""))
                template.update(
                    name=f"WT-PC-{kind}-{view}",
                    requests=str(request),
                    outputs=[str(raw)],
                    condition="nonempty_requests",
                )
                command = template["command"]
                at = next(i for i, s in enumerate(command) if s.endswith("/scripts/generate.sh"))
                command[at] = str(Path(pc["source_root"]) / "scripts/generate.sh")
                command[at + 2], command[at + 4] = str(request), str(raw)
                template["env"].update(XG_TERMINAL_ORDER="random", XG_TERMINAL_SEED=str(view_seed))
                command += [
                    "--sample",
                    "--temperature",
                    "1",
                    "--top-p",
                    "1",
                    "--terminal-order",
                    "random",
                    "--terminal-seed",
                    str(view_seed),
                ]
                jobs.append(template)
            output = folder / "WT-PC.jsonl"
            jobs.append(
                cpu_job(
                    family,
                    "WT-PC-merge",
                    "pc-merge",
                    "--execution",
                    out,
                    "--family",
                    family,
                    output=output,
                )
            )
            arms.append(
                {"family": family, "arm": "WT-PC", "scope": "primary", "output": str(output)}
            )
    for job in jobs:
        if job["kind"] == "gpu":
            at = next(i for i, s in enumerate(job["command"]) if s.endswith("/scripts/generate.sh"))
            code = "endpoint-code" if job["name"].startswith("WT-PC-") else "code"
            if job["name"] == "gcr-native" and config.get("gcr_beam_semantics"):
                if config["gcr_beam_semantics"] not in ("hf-group", "hf"):
                    raise ValueError("registered GCR semantics must be hf-group or its hf fallback")
                code = "gcr-code"
                job["command"] += [
                    "--gcr-beam-semantics",
                    config["gcr_beam_semantics"],
                    "--gcr-beam-early-stopping",
                    "false",
                    "--gcr-beam-length-penalty",
                    "1",
                    "--gcr-beam-return-sequences",
                    "10",
                ]
            job["command"][at] = str(out / code / "scripts/generate.sh")
    return jobs, arms


def two_tier_jobs(config, gate):
    """Keep the catalog recipes; change only prospectively registered tiers and inputs."""
    jobs, arms = build_jobs({**config, "cohort_rule": "legacy"}, gate)
    base = Path(config["out"]) / "execution"
    admitted = gate.get("baseline_admission") if gate else None

    def group(row):
        if row["scope"] == "dog":
            return "dog"
        name = row.get("name", row.get("arm"))
        return (
            "gcr"
            if name == "gcr-native"
            else "subgraphrag"
            if name.startswith("subgraphrag")
            else None
        )

    def retain(row):
        name = row.get("name", row.get("arm"))
        baseline = group(row)
        if baseline:
            return admitted is None or admitted[baseline]["admit"]
        return not name.startswith("agent-") or name == REFERENCES[row["family"]]

    def transform(row):
        if group(row) in ("gcr", "subgraphrag"):
            row = json.loads(
                json.dumps(row)
                .replace(str(base / "inputs"), str(base / "baseline-inputs"))
                .replace(str(base / "runs"), str(base / "baseline-runs"))
            )
            row["scope"] = "baseline"
        if (
            group(row) == "gcr"
            and row.get("kind") == "gpu"
            and config.get("gcr_funded_wall_guard")
            and admitted is not None
        ):
            row["launcher_wall_cap_seconds"] = admitted["gcr"]["launcher_wall_caps_seconds"][
                row["family"]
            ]
        if "command" in row:
            command = row["command"]
            if command[:2] == ["nice", "-n"]:
                command = command[3:]
            row["command"] = ["nice", "-n", "19", "taskset", "-c", "1-8", *command]
        return row

    return ([transform(j) for j in jobs if retain(j)], [transform(a) for a in arms if retain(a)])


def baseline_price_contract(registration, name, semantics="hf-group"):
    """Hash exact model/reader recipes with only execution destinations normalized."""
    config = {**registration["config"], "gcr_beam_semantics": semantics}
    jobs, _ = build_jobs(config, {"wt_pc": {"admit": False}, "dog": {"admit": True}})
    selected = [
        j
        for j in jobs
        if j["kind"] == "gpu"
        and (
            j["scope"] == "dog"
            if name == "dog"
            else j["name"] == "gcr-native"
            if name == "gcr"
            else j["name"].startswith("subgraphrag-")
        )
    ]
    value = [{k: j[k] for k in ("family", "name", "env", "command")} for j in selected]
    encoded = json.dumps(value, sort_keys=True).replace(
        str(Path(config["out"]) / "execution"), "$EXECUTION"
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def gcr_bench_feasible(cell):
    if (
        cell.get("native") is not True
        or cell.get("batch_size") != 1
        or cell.get("questions") != 8
        or type(cell.get("returncode")) is not int
    ):
        raise ValueError("GCR matched validation price requires native eight-ID b1 bench evidence")
    if cell["returncode"] != 0 or cell.get("timeout") is True:
        return False
    if cell.get("complete") is not True:
        raise ValueError("GCR matched validation feasibility is unfinished")
    return positive(cell.get("wall_seconds"), "GCR bench wall") / 8 <= 600


def read_baseline_prices(registration, path, pilot_ids):
    """Read qualified validation costs; missing groups stay pending."""
    data = json.loads(Path(path).read_text())
    if data.get("scope") != "validation" or data.get("complete") is not True:
        raise ValueError("matched validation baseline timings must be complete")
    if set(data.get("groups", {})) - set(registration["pricing"]["baseline_order"]):
        raise ValueError("unregistered matched validation baseline price group")
    result = {}
    for name, group in data.get("groups", {}).items():
        if name == "gcr" and registration.get("gcr_native_pricing"):
            from xgraph.webqsp_gcr_pricing import amortized_native_price, read_native_price

            result[name] = (
                amortized_native_price(registration, group)["rates"]
                if registration.get("gcr_amortization")
                else read_native_price(registration, group)
            )
            continue
        semantics = group.get("semantics", "hf-group")
        questions = 2 if name == "gcr" else 100
        producer = (
            registration["config"].get("gcr_source_commit")
            if name == "gcr"
            else registration["commit"]
        )
        if (
            group.get("producer_commit") != producer
            or group.get("native_qualified") is not True
            or group.get("validation_ids") != pilot_ids[:questions]
            or len(pilot_ids) != 100
            or group.get("recipe_sha256") != baseline_price_contract(registration, name, semantics)
            or not group.get("evidence")
        ):
            raise ValueError(
                f"{name}: matched validation price changed producer, recipe, IDs or native checks"
            )
        for item in group["evidence"]:
            verify_identity(item)
        if name == "gcr":
            bench = group.get("bench", {})
            preferred = gcr_bench_feasible(bench.get("hf-group", {}))
            if semantics != ("hf-group" if preferred else "hf") or not gcr_bench_feasible(
                bench.get(semantics, {})
            ):
                raise ValueError(
                    "GCR matched validation price violates timing-only feasibility fallback"
                )
        if name == "dog" and group.get("graph_contract") != "released-top120-validation":
            raise ValueError("DoG matched validation price cannot use the BM25 proxy")
        rates = {}
        for row in group["per_arm"]:
            key = f"{row['family']}/{row['name']}"
            if key in rates or row.get("returncode") != 0 or row.get("questions") != questions:
                raise ValueError(f"{name}: incomplete/duplicate/failed matched validation timing")
            rates[key] = (
                positive(row["wall_seconds"], "matched validation launcher wall") / questions
            )
        expected = {"/".join(s) for s in registration["pricing"]["baseline_stages"][name]}
        if set(rates) != expected:
            raise ValueError(f"{name}: incomplete matched validation stage matrix")
        result[name] = rates
    return result


def matched_prices(registration, path, pilot_path):
    if path is None:
        return {}
    plan = json.loads(Path(pilot_path).with_name("queue.json").read_text())
    return read_baseline_prices(registration, path, plan["pilot"]["cohort_ids"])


def require_native_checks(gate):
    native = gate.get("native_qualification", {})
    if native.get("families") != dict.fromkeys(FAMILIES, True) or not native.get("evidence"):
        raise ValueError("early execution requires hash-bound native checks in both families")
    if datetime.fromisoformat(gate["decided_at"]) > datetime.now(UTC):
        raise ValueError("early execution cannot use future development/native checks")
    charge = gate.get("protocol_qualification_gpu_seconds")
    if (
        isinstance(charge, bool)
        or not isinstance(charge, (int, float))
        or not math.isfinite(charge)
        or not 0 <= charge <= 4 * 3600
    ):
        raise ValueError(
            "native checks require an explicit qualification GPU charge within reserve"
        )
    for item in native["evidence"]:
        verify_identity(item)


def baseline_timing_path(registration):
    path = registration["config"].get("baseline_timings")
    return Path(path) if path and Path(path).is_file() else None


def native_pricing_charge(registration, baseline_path):
    """All native pricing attempts, including a terminal failure, enter the core reserve."""
    if not registration.get("gcr_native_pricing"):
        return 0
    if baseline_path is None:
        raise ValueError("native WebQSP GCR pricing receipt is required before execution")
    group = json.loads(Path(baseline_path).read_text()).get("groups", {}).get("gcr")
    if group is None:
        raise ValueError("native WebQSP GCR pricing charge is missing")
    from xgraph.webqsp_gcr_pricing import read_native_price

    read_native_price(registration, group)
    return group["charged_seconds"]


def resolved_config(registration, baseline_path):
    config = copy.deepcopy(registration["config"])
    if baseline_path:
        group = json.loads(Path(baseline_path).read_text()).get("groups", {}).get("gcr")
        if group:
            config["gcr_beam_semantics"] = group["semantics"]
    return config


class IdRangeReader(io.RawIOBase):
    """Bounded HTTP ranges for Arrow ID projection; a whole-shard response is refused."""

    def __init__(self, url, size):
        self.url, self.size, self.position, self.transferred = url, size, 0, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = (
            offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        )
        if not 0 <= position <= self.size:
            raise ValueError("invalid Parquet range seek")
        self.position = position
        return position

    def read(self, size=-1):
        import urllib.request

        if size < 0 or size > 1_000_000 or self.transferred + size > 4_000_000:
            raise ValueError("ID projection exceeded the bounded range-read budget")
        size = min(size, self.size - self.position)
        if not size:
            return b""
        end = self.position + size - 1
        request = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={self.position}-{end}", "Accept-Encoding": "identity"},
        )
        response = urllib.request.urlopen(request, timeout=60)
        try:
            if (
                response.status != 206
                or response.headers.get("Content-Range")
                != f"bytes {self.position}-{end}/{self.size}"
            ):
                raise ValueError("server did not honour exact range request; whole file not read")
            raw = response.read(size + 1)
            if len(raw) != size:
                raise ValueError("incorrect range response length")
        finally:
            response.close()
        self.position += size
        self.transferred += size
        return raw

    def readinto(self, buffer):
        raw = self.read(len(buffer))
        buffer[: len(raw)] = raw
        return len(raw)


def question_clusters(rows, method="transitive"):
    """Question-only dependence units, with alternatives that do not bridge topics."""
    if method not in ("transitive", "primary_topic", "exact_question"):
        raise ValueError(f"unknown question cluster method: {method}")
    rows = list(rows)
    unique_ids(r["id"] for r in rows)
    parent = {r["id"]: r["id"] for r in rows}

    def root(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    seen = {}
    for row in rows:
        q = row["id"]
        text = " ".join(row["question"].casefold().split())
        if method == "primary_topic":
            # Released order, not a gold-derived assertion about semantic primacy.
            keys = [("topic", str(row["q_entity"][0]))] if row["q_entity"] else [("id", q)]
        elif method == "exact_question":
            keys = [("question", text)]
        else:
            keys = [("topic", str(e)) for e in row["q_entity"]]
            keys.append(("question", text))
        for key in keys:
            if key in seen:
                a, b = sorted((root(q), root(seen[key])))
                parent[b] = a
            seen[key] = q
    return {q: root(q) for q in sorted(parent)}


def cluster_summary(assignments):
    sizes = Counter(assignments.values())
    if not sizes:
        raise ValueError("empty question cluster population")
    n, largest = len(assignments), max(sizes.values())
    effective = n * n / sum(size * size for size in sizes.values())
    return {
        "questions": n,
        "clusters": len(sizes),
        "sizes_descending": sorted(sizes.values(), reverse=True),
        "size_distribution": {str(k): v for k, v in sorted(Counter(sizes.values()).items())},
        "largest_component": largest,
        "largest_share": largest / n,
        "effective_clusters": effective,
        "degenerate": largest / n > 0.10 or effective < 200,
    }


def audit_question_clusters(rows, population, counts, prior):
    """Audit a closed question-only projection; select once, cluster on the full population."""
    rows = list(rows)
    if any(set(row) != {"id", "question", "q_entity"} for row in rows):
        raise ValueError("cluster audit requires a question-only id/question/q_entity projection")
    supplied = set(unique_ids(row["id"] for row in rows))
    if supplied != set(population.get("rog", population["intersection"])):
        raise ValueError("question projection differs from the registered RoG ID census")
    ids = unique_ids(population["intersection"])
    if not set(ids) <= supplied:
        raise ValueError("question projection is missing common-population IDs")

    def summarize(mapping):
        summary = cluster_summary(mapping)
        if prior:
            n = len(mapping)
            summary["power_transport"] = power_report(prior, n)
            summary["power_perfect_dependence_stress"] = power_report(
                prior, n, effective_questions=summary["effective_clusters"]
            )
            summary["power_perfect_dependence_stress"]["basis"] = (
                "Extra perfect within-cluster dependence stress on top of the MINTQA "
                "transport variance; deliberately conservative, not measured assurance"
            )
        return summary

    methods = {}
    for method in ("transitive", "primary_topic", "exact_question"):
        mapping = question_clusters(rows, method)
        mapping = {q: mapping[q] for q in sorted(ids)}
        methods[method] = {
            "assignments": mapping,
            "population": summarize(mapping),
            "samples": {
                str(count): {
                    "ids": selected,
                    **summarize({q: mapping[q] for q in selected}),
                }
                for count in counts
                for selected in [select_ids(ids, count)]
            },
        }
    return {
        "columns": ["id", "question", "q_entity"],
        "seed": SEED,
        "methods": methods,
        "threshold_scope": "Apply the population diagnostic to U; also report sample diagnostics",
        "limits": (
            "Kish sum(n)^2/sum(n^2) is cluster-size concentration, not an observed variance. "
            "Actual clustered joint assurance is unidentified without outcomes; normal transport "
            "and perfect-dependence stress are scenarios only; no finite-population correction."
        ),
    }


def development_prior(source):
    """Re-score only selected MINTQA development rows from primary artifacts."""
    import numpy as np

    from xgraph.mintqa_bench import answer_contains
    from xgraph.mintqa_rematch import clustered_difference
    from xgraph.prompts import final_answer
    from xgraph.score import record_failure

    source = Path(source).resolve()
    inputs = [source / "target/mintqa-rematch/panels/panels.json"]
    members = json.loads(inputs[0].read_text())["panels"]["development"]["members"]
    ids = unique_ids(r["id"] for r in members)
    gold = {}
    for split in ("pop", "ti"):
        wanted = {int(q.rsplit("-", 1)[1]): q for q in ids if q.startswith(f"mintqa-{split}-")}
        path = source / f"data/external/mintqa-full/questions/{split}.jsonl"
        inputs.append(path)
        with path.open() as stream:
            for i, line in enumerate(stream):
                if i in wanted:
                    gold[wanted[i]] = json.loads(line)["answer"]
    if set(gold) != set(ids):
        raise ValueError("incomplete MINTQA development gold")
    weights = np.array([r["weight"] for r in members], dtype=float)
    weights /= weights.sum()
    result = {}
    for family in FAMILIES:
        wt = (
            source
            / ".claude/worktrees/codex-integrate/target/mintqa-width8/terminal/menuobs-width8"
            / family
            / "development/menuobs-width8-two-pass-terminal.jsonl"
        )
        reference = source / (
            f"target/mintqa-rematch/{family}/development/agent-strong-8.jsonl"
            if family == "qwen38"
            else f"target/agent-qual/{family}/development/agent-strong-8-feedback.jsonl"
        )
        inputs.extend([wt, reference])
        vectors = []
        for path in (wt, reference):
            rows = {r["id"]: r for r in read_lines(path)}
            if set(rows) != set(ids):
                raise ValueError("incomplete development arm")
            vectors.append(
                np.array(
                    [
                        float(
                            answer_contains(
                                final_answer("" if record_failure(rows[q]) else rows[q]["text"]),
                                gold[q],
                            )
                        )
                        for q in ids
                    ]
                )
            )
        a, b = vectors
        delta = a - b
        effect, discordance = float(weights @ delta), float(weights @ (delta * delta))
        bootstrap = clustered_difference(a, b, members, 10000, SEED)
        from xgraph.mintqa_rematch import _paired_bootstrap

        _, draws, _ = _paired_bootstrap(a, b, members, 10000, SEED)
        variance = float(np.var(draws[:, 0] - draws[:, 1], ddof=1))
        iid = discordance - effect * effect
        design = max(1.0, variance * len(ids) / iid) if iid else 1.0
        result[family] = {
            "reference": REFERENCES[family],
            "questions": len(ids),
            "effect": effect,
            "discordance": discordance,
            "discordant_count": int(np.count_nonzero(delta)),
            "design_effect": design,
            "transport_variance": design * iid,
            "cluster_bootstrap_variance": variance,
            "bootstrap": bootstrap,
            "kish_n": float(1 / (weights @ weights)),
        }
    return {
        "families": result,
        "inputs": [identity(p) for p in inputs],
        "seed": SEED,
        "scope": "MINTQA development only; no WebQSP answers, predictions or validation outputs",
    }


def read_lines(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def bounded_launcher(source):
    """Adapt an isolated source snapshot; TERM cleans up only its own container."""
    original = "trap 'rm -rf \"${run}\"' EXIT"
    if source.count(original) != 1:
        raise ValueError("launcher cleanup seam changed; bounded snapshot must be requalified")
    return source.replace(
        original,
        """cleanup_webqsp_test() {
    status=$?
    trap - EXIT TERM INT
    if [[ -n "${container:-}" ]]; then
        timeout --signal=KILL 10 docker kill "${container}" >/dev/null 2>&1 || true
    fi
    rm -rf "${run}"
    return "${status}"
}
trap cleanup_webqsp_test EXIT
trap 'exit 143' TERM
trap 'exit 130' INT""",
    )


def rank_endpoint_set(wt, endpoints, samples, prior=None):
    from xgraph.mintqa_rematch import combined_usage

    consumed = [*samples, *(prior or [])]
    result = {
        **wt,
        "usage": combined_usage(wt, *consumed),
        "endpoint_pc": {"K": len(samples), "prior": prior is not None},
    }
    support = sorted(endpoints)
    if not samples or any(r.get("endpoint_entity_id") not in endpoints for r in consumed):
        result["endpoint_pc"]["fallback"] = "unusable revision or unresolved sample"
        return result
    if prior is not None and len(prior) != len(samples):
        raise ValueError("WT-PC prior must pair every question view")
    counts = Counter(r["endpoint_entity_id"] for r in samples)
    scores = {e: math.log((counts[e] + 1) / (len(samples) + len(support))) for e in support}
    if prior is not None:
        counts0 = Counter(r["endpoint_entity_id"] for r in prior)
        scores = {
            e: s - math.log((counts0[e] + 1) / (len(prior) + len(support)))
            for e, s in scores.items()
        }
    ranked = list(
        dict.fromkeys(endpoints[e] for e in sorted(support, key=lambda e: (-scores[e], e)))
    )
    if set(ranked) != set(wt["answers"]):
        raise ValueError("WT-PC may rank but cannot change the complete WT answer set")
    return {**result, "answers": ranked, "endpoint_pc": {**result["endpoint_pc"], "scores": scores}}


def enforce_embargo(registration, directory, override=None):
    """Check metadata before any output, receipt, graph, gold or release is opened."""
    deadline = registration.get("score_not_before")
    if deadline is None or datetime.now(UTC) >= datetime.fromisoformat(deadline):
        return None
    if override is None:
        raise ValueError(
            f"WebQSP output embargo through November 23 23:59 Europe/Berlin; "
            f"score opens at {deadline}; an owner override file must record why"
        )
    record = json.loads(Path(override).read_text())
    if any(
        not isinstance(record.get(k), str) or not record[k].strip()
        for k in ("owner", "reason", "authorized_at")
    ):
        raise ValueError("embargo override requires owner, reason and authorization time")
    when = datetime.fromisoformat(record["authorized_at"])
    if when.tzinfo is None or when > datetime.now(UTC):
        raise ValueError(
            "embargo override authorization must be dated, timezone-aware and not future"
        )
    path = Path(directory) / "execution.json"
    execution = read_sealed(path)
    if (
        record.get("execution_sha256") != sha256_file(path)
        or record.get("registration_sha256") != execution["registration"]["sha256"]
    ):
        raise ValueError("embargo override must bind the registration and execution seals")
    return {"file": identity(override), "record": record}


def score_execution(registration, execution, directory, *, embargo_override=None):
    """Check completeness before gold; a sealed GCR failure invalidates its contrasts only."""
    override = enforce_embargo(registration, directory, embargo_override)
    rows, incomplete_gcr = {}, {}
    for arm in execution["arms"]:
        path = Path(arm["output"])
        try:
            if not path.is_file():
                raise ValueError(f"incomplete arm: {path}")
            records = read_lines(path)
            expected = execution_ids(execution, arm["scope"])
            require_complete(records, expected)
        except ValueError as error:
            if arm["arm"] != "gcr-native":
                raise
            incomplete_gcr[arm["family"]] = str(error)
            continue
        rows[arm["scope"], arm["family"], arm["arm"]] = records
    verify_receipts(execution, Path(directory), incomplete_gcr=incomplete_gcr)
    for family in incomplete_gcr:
        for scope in ("primary", "baseline"):
            rows.pop((scope, family, "gcr-native"), None)
    references = registration.get("strongest_agents", REFERENCES)
    for family in FAMILIES:
        if references[family] == "gcr-native" and ("primary", family, "gcr-native") not in rows:
            raise ValueError(f"incomplete primary: sealed strongest agent gcr-native/{family}")
    audit_policy_sources(execution, rows)
    return finish_scoring(
        registration,
        execution,
        rows,
        Path(directory),
        incomplete_gcr,
        **({"embargo_override": override} if override else {}),
    )


def execution_ids(execution, scope):
    return execution[{"dog": "dog_ids", "baseline": "baseline_ids"}.get(scope, "ids")]


def audit_policy_sources(execution, rows):
    from xgraph.mintqa_rematch import COST_FIELDS, combined_usage, usage

    stages = {
        (j["scope"], j["family"], j["name"]): {r["id"]: r for r in read_lines(j["outputs"][0])}
        for j in execution["jobs"]
        if j["kind"] == "gpu"
        and (j["name"] != "gcr-native" or (j["scope"], j["family"], j["name"]) in rows)
    }
    for (scope, family, name), records in rows.items():
        sources = {
            "W": ["W"],
            "WT": ["W", "WT-raw"],
            "B2": ["blind5", "B2-raw"],
            "subgraphrag": ["subgraphrag-first", "subgraphrag-retry"],
        }.get(name, [name])
        if name == "WT-PC":
            sources = [
                "W",
                "WT-raw",
                *[
                    key[2]
                    for key in stages
                    if key[:2] == (scope, family) and key[2].startswith("WT-PC-")
                ],
            ]
        for row in records:
            consumed = []
            for source in sources:
                key = scope, family, source
                if key not in stages:
                    # Unverified minimal synthetic dictionaries are supported by unit seams;
                    # the CLI verifies the complete production stage DAG before this call.
                    if not execution["jobs"]:
                        continue
                    raise ValueError(f"incomplete charged policy source: {key}")
                stage = stages[key]
                if (
                    row["id"] not in stage
                    and source not in ("subgraphrag-retry",)
                    and not source.startswith("WT-PC-")
                ):
                    raise ValueError(f"incomplete charged source: {key}/{row['id']}")
                consumed.append(stage.get(row["id"]))
            if execution["jobs"] and any(
                usage(row)[field] != combined_usage(*consumed)[field] for field in COST_FIELDS
            ):
                raise ValueError(
                    f"policy call/token usage does not conserve charged sources: {name}/{row['id']}"
                )


def projected_rows(paths, columns):
    import pyarrow.parquet as pq

    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=1 if "graph" in columns else 16, columns=columns, use_threads=False
        ):
            yield from batch.to_pylist()


def goldfree_graphs(paths, ids, out):
    from xgraph.toolkg import build_store
    from xgraph.webqsp import convert

    rows = list(projected_rows(paths, ["id", "question", "q_entity"]))
    unique_ids(r["id"] for r in rows)
    selected = set(ids)
    if selected - {r["id"] for r in rows}:
        raise ValueError("selected IDs missing from graph projection")
    convert(
        (
            r
            for r in projected_rows(paths, ["id", "question", "q_entity", "graph"])
            if r["id"] in selected
        ),
        out,
        gold=False,
    )
    build_store(out / "kg", out / "kg.sqlite")
    build_store(out / "kg-gcr", out / "kg-gcr.sqlite")
    return {
        "clusters": question_clusters(rows),
        "graph_columns": ["id", "question", "q_entity", "graph"],
        "gold_loaded": False,
    }


def git_commit(root=ROOT, *, clean=True):
    if (
        clean
        and subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True
        ).strip()
    ):
        raise ValueError(f"source tree must be clean: {root}")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def model_inventory(path):
    path = Path(path).resolve()
    metadata = [
        identity(p) for p in sorted(path.iterdir()) if p.suffix in (".json", ".jinja", ".txt")
    ]
    weights = [
        {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in sorted(path.glob("*.safetensors"))
    ]
    if not metadata or not weights:
        raise ValueError(f"missing model metadata or weights: {path}")
    download = path / ".cache/huggingface/download"
    config_metadata = download / "config.json.metadata"
    revision = (
        config_metadata.read_text().split("\n", 1)[0].strip() if config_metadata.is_file() else None
    )
    download_metadata = [
        identity(p)
        for p in sorted(download.glob("*.metadata"))
        if p.name == "config.json.metadata" or p.name.endswith(".safetensors.metadata")
    ]
    return {
        "path": str(path),
        "metadata": metadata,
        "weights": weights,
        "revision": revision,
        "download_metadata": download_metadata,
        "weight_identity": "size_mtime; revision and tokenizer hashes retained; no weights loaded",
    }


def register(config_path, destination):
    from xgraph.policy_catalog import fingerprint, recipe
    from xgraph.provenance import provenance

    if Path(destination).exists() or Path(f"{destination}.sha256").exists():
        raise FileExistsError(f"immutable registration already exists: {destination}")
    config = json.loads(Path(config_path).read_text())
    commit = git_commit()
    population = json.loads(Path(config["population"]).read_text())
    unique_ids(population["intersection"])
    if not population["intersection"]:
        raise ValueError("empty common population")
    prior = json.loads(Path(config["prior"]).read_text())
    prices = json.loads(Path(config["prices"]).read_text())["webqsp-pilot"]
    jobs, arms = build_jobs(config)
    maximum = {
        "wt_pc": {"admit": True, "K": 4, "prior": True, "source_root": config["endpoint_root"]},
        "dog": {"admit": True},
    }
    conditional, conditional_arms = build_jobs(config, maximum)
    out = Path(config["out"])
    queue = out / "queue.sh"
    with queue.open("x") as queue_file:
        queue_file.write(f"""#!/usr/bin/env bash
set -euo pipefail
cd {shlex.quote(str(ROOT))}
mode="${{1:---dry-run}}"
case "${{mode}}" in --dry-run|--run) ;; *) echo 'use --dry-run or --run' >&2; exit 2;; esac
exec {shlex.quote(str(ROOT / ".venv/bin/python"))} -m xgraph.webqsp_test queue \\
  --registration {shlex.quote(str(Path(destination).resolve()))} \\
  --execution {shlex.quote(str(out / "execution"))} "${{mode}}"
""")
    queue.chmod(0o755)
    inputs = list(
        dict.fromkeys(
            [
                str(Path(config_path).resolve()),
                config["population"],
                config["prior"],
                config["prices"],
                config["prompts"],
                config["retrieval_scores"],
                str(queue),
                *config["evidence_inputs"],
            ]
        )
    )
    extra = []
    for family in FAMILIES:
        extra.extend(
            [
                {"family": family, "name": "SubgraphRAG two calls", "envelope": 204},
                {
                    "family": family,
                    "name": "WT-PC answer views",
                    "condition": "wt_pc",
                    "envelope": 30,
                    "copies": 8,
                    "pilot_stage": [family, "WT-raw"],
                },
                *[
                    {
                        "family": family,
                        "name": "DoG same-graph " + name,
                        "condition": "dog",
                        "envelope": envelope,
                        "pilot_stage": [family, name],
                    }
                    for name, envelope in (
                        ("W", 6),
                        ("WT-raw", 6),
                        (REFERENCES[family], 14),
                        ("dog", 30),
                    )
                ],
            ]
        )
    registration = {
        "artifact_kind": "xgraph.webqsp_test_registration",
        "version": 1,
        "commit": commit,
        "config": config,
        "inputs": [identity(p) for p in inputs],
        "model_inventory": {
            f: model_inventory(p)
            for f, p in {**config["models"], "gcr": config["gcr_model"]}.items()
        },
        "data_pins": config["data_pins"],
        "population": population,
        "prior": prior,
        "sampling": {
            "seed": SEED,
            "rule": (
                "sort ascending SHA-256 UTF-8 '17:'+exact question ID, then exact ID; "
                "common prefix; no replacement or gold-derived strata"
            ),
        },
        "policies": {
            f: {
                p: {"recipe": recipe(p, f), "sha256": fingerprint(recipe(p, f))}
                for p in ("WT", "W", "B2", "agent-strong-8", "agent-feedback-8")
            }
            for f in FAMILIES
        },
        "strongest_agents": REFERENCES,
        "jobs": jobs,
        "arms": arms,
        "conditional_maximum_jobs": conditional,
        "conditional_maximum_arms": conditional_arms,
        "conditional_rule": {
            "gate_file": config["gate"],
            "deadline": "2026-11-13T22:59:59+00:00",
            "WT_PC_K": [2, 4],
            "prior": [False, True],
            "shared_configuration": True,
            "missing_gate_blocks": True,
        },
        "pilot": {
            "timings": config["timings"],
            "commit": config["pilot_commit"],
            "questions": 100,
            "scope": "validation; no test timing for sample size",
        },
        "pilot_stages": [
            [j["family"], j["name"]]
            for j in jobs
            if j["kind"] == "gpu" and not j["name"].startswith("subgraphrag")
        ],
        "pricing": {
            "cap_seconds": 144000,
            "reserve_seconds": 14400,
            "unmeasured_base_seconds_per_question": prices["high_gpu_hours"]
            * 3600
            / prices["questions_per_arm"],
            "extra_rates": extra,
            "extra_envelope_seconds_per_question": sum(
                r["envelope"] * r.get("copies", 1) for r in extra
            ),
            "basis": (
                "prices.json pilot upper scenario plus declared new-arm envelopes; "
                "reserve covers qualification/retries/startup uncertainty"
            ),
        },
        "analysis": {
            "primary": "gcr_hit",
            "alpha": 0.025,
            "NI_margin": -0.03,
            "bootstrap_seed": SEED,
            "resamples": 10000,
            "cluster": (
                "transitive topic overlap or duplicate whitespace-normalized casefolded question"
            ),
            "minimum_clusters": 20,
            "multiplicity": "max over families then Holm within gatekept stage",
            "primary_claim": "WT vs strongest agent NI",
            "secondary_claims": [
                "WT superiority on gcr_hit",
                "conditional WT-PC vs WT superiority on ranked Hits@1",
            ],
            "secondary_gate": "primary joint NI passes",
            "WT_PC_set_invariance": True,
        },
        "freeze_deadline": "2026-11-23T22:59:59+00:00",
        "generation_not_before": "2026-11-23T23:00:00+00:00",
        "provenance": provenance([Path(config_path)]),
        "status": "SEALED_PREREGISTRATION_PENDING_MEASURED_TIMINGS_AND_DEVELOPMENT_GATE",
    }
    if git_commit() != commit:
        raise ValueError("source changed during registration")
    write_sealed(destination, registration)
    return registration


def science_sha256(registration):
    """Frozen science, normalizing only execution destinations and pilot source pins."""
    science = copy.deepcopy(registration)
    for field in (
        "version",
        "commit",
        "amendment",
        "inputs",
        "pilot",
        "status",
        "provenance",
        "parent_provenance",
    ):
        science.pop(field, None)
    for field in ("timings", "pilot_commit"):
        science["config"].pop(field, None)
    science["config"]["out"] = "{OUT}"
    serialized = json.dumps(science, sort_keys=True).replace(
        registration["config"]["out"] + "/execution", "{EXECUTION}"
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def amend_registration(parent_path, destination, protocol_path, audit_path, record_path):
    """Seal a prospective amendment beside its immutable parent."""
    from xgraph.provenance import provenance

    destination = Path(destination).resolve()
    if destination.exists() or Path(f"{destination}.sha256").exists():
        raise FileExistsError(f"immutable amendment already exists: {destination}")
    original = read_sealed(parent_path)
    if datetime.now(UTC) > datetime.fromisoformat(original["freeze_deadline"]):
        raise ValueError("prospective amendment deadline has passed")
    if (Path(original["config"]["out"]) / "execution").exists():
        raise ValueError("amendment requires no existing execution or test output")
    active = ROOT / "target/webqsp-test/active-registration.json"
    if active.is_file():
        reference = json.loads(active.read_text())
        verify_identity(reference)
        current = read_sealed(reference["path"])
        if (Path(current["config"]["out"]) / "execution").exists():
            raise ValueError(
                "active execution already sealed; no further amendment or sample change"
            )
    record = json.loads(Path(record_path).read_text())
    number = record.get("amendment")
    if number not in (1, 2, 3, 4, 5, 6, 8) or not record.get("date") or not record.get("rationale"):
        raise ValueError("amendment needs its number, date and rationale")
    audit = json.loads(Path(audit_path).read_text())
    child_8a = number == 8 and record.get("addendum") == "8a"
    if number == 8 and record.get("addendum") not in (None, "8a"):
        raise ValueError("unreviewed amendment-8 addendum")
    if child_8a:
        if original.get("amendment", {}).get("amendment") != 8 or original["amendment"].get(
            "addendum"
        ):
            raise ValueError("amendment 8a requires amendment 8 as its parent")
        if record["parent_sha256"] != identity(parent_path)["sha256"]:
            raise ValueError("amendment 8a parent hash changed")
        if record["preserved_science_sha256"] != science_sha256(original):
            raise ValueError("amendment 8a frozen science changed")
        for field in ("cohort_ids", "questions", "scope", "timing_basis", "timing_contract"):
            if record["pilot_evidence"].get(field) != original["pilot"].get(field):
                raise ValueError(f"amendment 8a pilot science changed: {field}")
    amended = copy.deepcopy(original)
    amended.update(version=original["version"] + 1, commit=git_commit())
    amended["status"] = "SEALED_AMENDMENT_1_PENDING_MEASURED_CORE_GATES_AND_GCR_BEAM_SEMANTICS"
    amended["config"]["out"] = str(destination.parent)
    amended["pricing"].update(cohort_rule="core_first", optional_order=["wt_pc", "dog"])
    primary = (
        "primary_topic"
        if audit["methods"]["transitive"]["population"]["degenerate"]
        else "transitive"
    )
    amended["analysis"].update(
        cluster_method=primary,
        cluster=(
            "transitive topic overlap or duplicate whitespace-normalized casefolded question"
            if primary == "transitive"
            else "own first released q_entity only; no secondary-topic or duplicate-text chaining"
        ),
        sensitivity_cluster_methods=[
            m for m in ("transitive", "primary_topic", "exact_question") if m != primary
        ],
        WT_PC_framing="contextual calibration",
        WT_PC_score=(
            "prior on: log(n_q(e)+1)-log(n_0(e)+1), equal K/support denominators cancel; "
            "prior off: log((n_q(e)+1)/(K+|E|)); unchanged set, ranked Hits@1 claim only"
        ),
    )
    amended["question_cluster_audit"] = audit
    amended["conditional_rule"]["budget_gate"] = "full core cohort preserved within 40 GPU-h"
    amended["amendment_slots"] = [
        {
            "name": "GCR beam semantics",
            "status": "PENDING",
            "source": "/home/jonas/git/xgraph/.claude/worktrees/codex-gcr2/target/gcr-speed/",
            "deadline": original["freeze_deadline"],
            "rule": "Keep current GCR configuration until a new prospective bench-based amendment",
            "blocks_execution": True,
        }
    ]
    if number == 2:
        parent_number = 2 if record.get("clarification") else 1
        if original.get("amendment", {}).get("amendment") != parent_number:
            raise ValueError(f"amendment 2 requires amendment {parent_number} as its parent")
        # Preserve the complete analysis and prices. The pending measured
        # resolution may replace only the two GCR cells at the anchored cohort.
        for key in ("analysis", "pricing", "conditional_rule", "question_cluster_audit"):
            if key in original:
                amended[key] = copy.deepcopy(original[key])
        if record.get("wt_pc_resolution"):
            amended["conditional_rule"]["wt_pc_resolution"] = record["wt_pc_resolution"]
        amended["pilot"]["commit"] = REPAIRED_PILOT_COMMIT
        amended["config"].update(
            pilot_commit=REPAIRED_PILOT_COMMIT,
            gcr_beam_semantics="hf-group",
            gcr_source_root="/home/jonas/git/xgraph/.claude/worktrees/codex-gcr2",
            gcr_source_commit=GCR_GROUP_COMMIT,
        )
        amended["gcr_semantics"] = {
            "preferred": "hf-group",
            "fallback": "hf",
            "max_launcher_seconds_per_question": 600,
            "decision_basis": "published implementation; no answer or accuracy input",
            "release": "9518e8e69197d5e577a6e414060fb898771617b9",
            "beam_arguments": {
                "do_sample": False,
                "num_beams": 10,
                "num_return_sequences": 10,
                "num_beam_groups": 10,
                "diversity_penalty": 1.0,
                "early_stopping": False,
                "length_penalty": 1.0,
            },
            "feasibility": (
                "Evaluate hf-group independently of old/new identity and hf accuracy. "
                "Require a complete native eight-ID b1 bench cell, exit 0, and launcher "
                "wall/8 <= 600 seconds. Missing/unfinished evidence stays pending. An "
                "explicit crash, timeout or wall/8 > 600 selects ordinary hf (10 beams, "
                "one group, no diversity, early_stopping=False, length_penalty=1); require "
                "the same native completeness and bound for hf. If both are infeasible, "
                "drop gcr-native prospectively before the execution seal and report "
                "'infeasible at the registered per-question bound' with bench evidence. "
                "Primary and other secondary claims proceed. A sealed GCR failure makes "
                "GCR comparisons INCOMPLETE; primary is affected only if GCR is the sealed "
                "strongest-agent reference (neither registered family uses GCR). "
                "No fallback to the old vllm port, no retries selected by answers."
            ),
            "repricing": (
                "Bench models, questions, empty-trie policy and token cap differ: bench "
                "seconds are feasibility only, never WebQSP prices. Freeze n_anchor from "
                "the repaired original-semantics pilot and amendment-1 core-first sizing. "
                "Before execution, time only gcr-native in each family on the first two "
                "registered validation pilot IDs, b1, with the selected semantics and "
                "otherwise exact WebQSP argv/models/inputs. Rate = complete launcher "
                "wall/2 per family, including both model loads; replace only those two "
                "pilot cells. Preserve n_anchor, all other rates, IDs and arms; recheck "
                "optional admission at that n. If the repriced core exceeds 40 GPU-h, "
                "block execution, never shrink it or choose semantics for budget/accuracy. "
                "Charge all timing attempts to at most 2 GPU-h within the existing "
                "4-hour reserve, stage timeout 1200 seconds, including startup/admission; "
                "a matched crash/timeout uses the same hf fallback, then otherwise "
                "GCR comparisons are INCOMPLETE. The prospective bench-based exclusion "
                "above needs no matched GCR timing and retains the original priced core "
                "cohort. No timing run or price is asserted here."
            ),
        }
        amended["amendment_slots"] = [
            {
                "name": "GCR beam semantics",
                "status": "RESOLVED",
                "blocks_execution": False,
                "rule": "hf-group by release faithfulness; fixed feasibility-only hf fallback",
            },
            {
                "name": "GCR feasibility and matched timing",
                "status": "PENDING",
                "blocks_execution": True,
                "deadline": original["freeze_deadline"],
                "rule": "Resolve only with the registered timing-only evidence and anchored cohort",
            },
        ]
        amended["status"] = "SEALED_AMENDMENT_2_PENDING_MEASURED_CORE_GATES_AND_GCR_FEASIBILITY"
    if number == 3:
        if original.get("amendment", {}).get("amendment") != 2:
            raise ValueError("amendment 3 requires amendment 2 as its parent")
        if original["pilot"]["commit"] != REPAIRED_PILOT_COMMIT:
            raise ValueError("amendment 3 requires the reviewed amendment-2 producer pin")
        # This amendment changes only the producer pin and its future repair rule.
        # Preserve the owner's clarification, pending gates and scientific design.
        amended = copy.deepcopy(original)
        amended.update(version=original["version"] + 1, commit=git_commit())
        amended["config"].update(out=str(destination.parent), pilot_commit=SUPERVISED_PILOT_COMMIT)
        amended["pilot"].update(
            commit=SUPERVISED_PILOT_COMMIT,
            producer_repair_rule={
                "before_pilot": (
                    "Any further producer repair before the pilot runs requires the same "
                    "prospective diff-justified re-pin, preserving input and recipe identities."
                ),
                "after_pilot": (
                    "Any producer repair after the pilot runs requires a diff-justified "
                    "re-pin and a complete pilot re-run; never reuse superseded timings."
                ),
            },
        )
        amended["status"] = "SEALED_AMENDMENT_3_PENDING_MEASURED_CORE_GATES_AND_GCR_FEASIBILITY"
    if number == 4:
        if original.get("amendment", {}).get("amendment") != 3:
            raise ValueError("amendment 4 requires amendment 3 as its parent")
        if original["pilot"]["commit"] != SUPERVISED_PILOT_COMMIT:
            raise ValueError("amendment 4 preserves the reviewed amendment-3 producer pin")
        amended = copy.deepcopy(original)
        amended.update(
            version=original["version"] + 1,
            commit=git_commit(),
            status="SEALED_AMENDMENT_4_PENDING_QUALIFIED_TWO_TIER_PRICES_AND_NATIVE_CHECKS",
            generation_not_before="execution_seal",
            score_not_before="2026-11-23T23:00:00+00:00",
        )
        amended["config"].update(
            out=str(destination.parent),
            cohort_rule="two_tier",
            baseline_timings=str(destination.parent / "baseline-timings.json"),
        )
        core = [
            [f, s] for f in FAMILIES for s in ("blind5", "W", "WT-raw", "B2-raw", REFERENCES[f])
        ]
        amended["pilot_stages"] = core
        amended["pricing"] = {
            "cohort_rule": "two_tier",
            "cap_seconds": 40 * 3600,
            "reserve_seconds": 4 * 3600,
            "baseline_cap_seconds": 12 * 3600,
            "baseline_allocation": "separate_post_freeze",
            "baseline_share_rule": "equal_among_admitted_groups",
            "minimum_baseline_questions": 100,
            "core_stages": core,
            "baseline_order": ["gcr", "subgraphrag", "dog"],
            "baseline_stages": {
                "gcr": [[f, "gcr-native"] for f in FAMILIES],
                "subgraphrag": [[f, "subgraphrag-reader"] for f in FAMILIES],
                "dog": [[f, s] for f in FAMILIES for s in ("W", "WT-raw", REFERENCES[f], "dog")],
            },
            "basis": "Qualified measured validation launcher rates only; shared physical stages "
            "once. Baselines have a separate 12 GPU-h published-baseline subpanel bucket "
            "in the 114 GPU-h post-freeze plan, split equally among admitted groups. "
            "DoG includes same-graph WT/agent controls; retrieval history remains unknown.",
        }
        amended["conditional_rule"]["budget_gate"] = (
            "one measured seeded baseline subpanel; never shrink core"
        )
        amended["analysis"]["depth_interaction"] = {
            "definition": "freebase-semantic-hop-v1",
            "long_minimum_hops": 2,
            "contrast": "(WT-agent at >=2 semantic hops) - (WT-agent at 1 semantic hop)",
            "direction": "positive in both families",
            "endpoint": "gcr_hit",
            "resampling": "paired whole KG worlds or question clusters jointly across depths",
            "multiplicity": "separate secondary family: max over families then Holm at .025; "
            "outside primary",
            "support": "Authoritative linear forward entity chains only; contract maximal internal "
            "CVT runs to one semantic edge using pinned type annotations. No opaque-MID "
            "heuristic. Filters, joins, inverses, ordering, aggregates and unsupported or "
            "ambiguous parses remain unclassified; never alter the primary denominator.",
        }
        amended["gcr_semantics"]["repricing"] = (
            "Measured native matched validation timing on the first two registered pilot IDs, "
            "selected hf-group/hf semantics, launcher wall/2 including both loads. Baseline-tier "
            "prices only; supersedes the previous anchored-cohort repricing rule. Bench rates "
            "are feasibility evidence only. All attempts stay inside reserve and lifetime cap."
        )
        amended["amendment_slots"] = [
            {
                "name": "two-tier prices and native checks",
                "status": "PENDING",
                "blocks_execution": False,
                "rule": "dynamically checked before sealing; no inherited envelopes",
            }
        ]
        amended["early_launch"] = {
            "target": "November 3-6 Europe/Berlin",
            "backstop": "November 10",
            "rule": "seal once qualified prices, producer pins and native checks exist; "
            "no policy, arm, sample or definition changes after seal; embargo through "
            "November 23 23:59 Europe/Berlin",
        }
    if number == 5:
        addendum = record.get("addendum")
        if addendum not in (None, "5a"):
            raise ValueError("unreviewed amendment-5 addendum")
        parent_number = 5 if addendum else 4
        prior_commit = OCCUPANCY_PILOT_COMMIT if addendum else SUPERVISED_PILOT_COMMIT
        producer = WATCHDOG_TESTED_PILOT_COMMIT if addendum else OCCUPANCY_PILOT_COMMIT
        if original.get("amendment", {}).get("amendment") != parent_number:
            raise ValueError(f"amendment 5 requires amendment {parent_number} as its parent")
        if original["pilot"]["commit"] != prior_commit:
            raise ValueError("amendment 5 requires the reviewed parent producer pin")
        amended = copy.deepcopy(original)
        amended.update(version=original["version"] + 1, commit=git_commit())
        amended["config"].update(out=str(destination.parent), pilot_commit=producer)
        amended["pilot"].update(
            **record["pilot_evidence"],
            commit=producer,
            timing_basis="container_occupancy",
        )
        source = ROOT / "src/xgraph/webqsp_test.py"
        prior = [item for item in amended["inputs"] if Path(item["path"]) == source]
        if prior:
            current = identity(source)
            amended["inputs"] = [
                current if Path(item["path"]) == source else item for item in amended["inputs"]
            ]
            record["consumer_source_repin"] = {"before": prior, "after": current}
        if addendum:
            helper = ROOT / "scripts/webqsp-amendment-5.py"
            prior_helpers = [item for item in amended["inputs"] if Path(item["path"]) == helper]
            if prior_helpers:
                current_helper = identity(helper)
                amended["inputs"] = [
                    current_helper if Path(item["path"]) == helper else item
                    for item in amended["inputs"]
                ]
                record["consumer_helper_repin"] = {"before": prior_helpers, "after": current_helper}
            seal = record["serving_seal"]
            verify_identity(seal)
            prior_seals = [item for item in amended["inputs"] if item["path"] == seal["path"]]
            if not prior_seals or any(
                item["sha256"] != original["pilot"]["occupancy_receipt"]["seal_sha256"]
                for item in prior_seals
            ):
                raise ValueError("amendment 5a requires the parent's serving seal identity")
            if seal["sha256"] != amended["pilot"]["occupancy_receipt"]["seal_sha256"]:
                raise ValueError("amendment 5a serving seal and receipt differ")
            amended["inputs"] = [
                seal if item["path"] == seal["path"] else item for item in amended["inputs"]
            ]
            record["producer_metadata_repin"] = {"before": prior_seals, "after": seal}
        amended["status"] = "SEALED_AMENDMENT_5_PENDING_QUALIFIED_TWO_TIER_PRICES_AND_NATIVE_CHECKS"
    if number == 6:
        if (
            original.get("amendment", {}).get("addendum") != "5a"
            or original["pilot"]["commit"] != WATCHDOG_TESTED_PILOT_COMMIT
        ):
            raise ValueError("amendment 6 requires the reviewed amendment-5a parent")
        amended = copy.deepcopy(original)
        amended.update(version=original["version"] + 1, commit=git_commit())
        amended["config"].update(
            out=str(destination.parent),
            baseline_timings=str(destination.parent / "baseline-timings.json"),
            gcr_source_commit=record["gcr_source_commit"],
            gcr_beam_semantics="hf-group",
        )
        amended["gcr_native_pricing"] = copy.deepcopy(record["gcr_native_pricing"])
        amended["gcr_semantics"] = {
            **original["gcr_semantics"],
            "preferred": "hf-group",
            "fallback": None,
            "feasibility": "MINTQA eight-ID cost is cross-dataset evidence only; never an "
            "exclusion or price for native WebQSP. Use the bounded seeded validation stage.",
            "repricing": "Two fixed seed-17 validation IDs in both families; native hf-group "
            "recipe, launcher wall/2 including both model loads; at most 2 GPU-h from the "
            "existing core reserve. One attempt per family; no hf or old-port fallback. "
            "Full core if its equal baseline share fits, otherwise the shared seed-17 whole-"
            "cluster subpanel with minimum 100. Below that bound or on terminal native "
            "pricing failure, cite the published non-paired result. Never shrink core.",
        }
        amended["gcr_semantics"].pop("max_launcher_seconds_per_question", None)
        amended["amendment_slots"] = [
            {
                "name": "two-tier prices and native checks",
                "status": "PENDING",
                "blocks_execution": False,
                "rule": "native WebQSP prices and all admission decisions required before sealing",
            }
        ]
        source = ROOT / "src/xgraph/webqsp_test.py"
        previous = [item for item in amended["inputs"] if Path(item["path"]) == source]
        amended["inputs"] = [
            identity(source) if Path(item["path"]) == source else item for item in amended["inputs"]
        ]
        record["consumer_source_repin"] = {"before": previous, "after": identity(source)}
        amended["status"] = "SEALED_AMENDMENT_6_PENDING_NATIVE_WEBQSP_PRICE_AND_CORE_GATES"
    if number == 8 and not child_8a:
        spec = record["gcr_amortization"]
        if (
            original.get("amendment", {}).get("amendment") != 6
            or len(original["pilot"]["cohort_ids"]) != 100
            or type(spec.get("basis_questions")) is not int
            or spec["basis_questions"] != 100
            or spec.get("rule") != "fixed_over_pilot_plus_workload"
        ):
            raise ValueError("amendment 8 requires amendment 6 and its fixed pilot basis")
        verify_identity(record["supersedes"])
        previous_decision = read_sealed(record["supersedes"]["path"])
        if (
            previous_decision["parent"] != identity(parent_path)
            or previous_decision["gcr"]["status"] != "INFEASIBLE_AT_BOUND"
        ):
            raise ValueError("amendment 8 must retain the parent's GCR exclusion decision")
        amended = copy.deepcopy(original)
        amended.update(version=original["version"] + 1, commit=git_commit())
        amended["config"].update(
            out=str(destination.parent),
            baseline_timings=str(destination.parent / "baseline-timings.json"),
            gcr_funded_wall_guard=True,
        )
        amended["gcr_amortization"] = copy.deepcopy(spec)
        amended["gcr_semantics"]["repricing"] = (
            "Amendment 8: F/100+w per family from the complete receipted launcher wall "
            "minus its measured generation workload. The frozen pilot cohort sets 100; "
            "no component is waived. Amendment-4 allocation and minimum remain unchanged. "
            "Split GCR's equal group share between families in proportion to their prices; "
            "each launcher has that hard wall cap, including cleanup. Overrun is INCOMPLETE; "
            "no top-up or retry. Published rows remain context."
        )
        source = Path(__file__).resolve()
        source_root = source.parents[2]
        repins = {
            source,
            source.with_name("webqsp_gcr_pricing.py"),
            source_root / "scripts/webqsp-amendment-6.py",
        }
        previous = [item for item in amended["inputs"] if Path(item["path"]) in repins]
        amended["inputs"] = [
            identity(item["path"]) if Path(item["path"]) in repins else item
            for item in amended["inputs"]
        ]
        record["consumer_source_repin"] = {
            "before": previous,
            "after": [identity(p) for p in sorted(repins)],
        }
        amended["status"] = "SEALED_AMENDMENT_8_PENDING_CORE_AND_OTHER_BASELINE_PRICES"
    if child_8a:
        amended = copy.deepcopy(original)
        amended.update(version=original["version"] + 1, commit=git_commit())
        amended["pilot"] = copy.deepcopy(record["pilot_evidence"])
        if amended["pilot"]["commit"] != "9708b863a2217caaf4591e29af160efd704cdf03":
            raise ValueError("amendment 8a requires the accepted 6e producer")
        amended["config"].update(
            out=str(destination.parent),
            timings=amended["pilot"]["timings"],
            pilot_commit=amended["pilot"]["commit"],
        )
        replacements = {p["before"]["path"]: p["after"] for p in record.get("input_repins", [])}
        for item in record.get("input_repins", []):
            if item["before"] not in original["inputs"]:
                raise ValueError("amendment 8a input re-pin has no parent identity")
            verify_identity(item["after"])
        amended["inputs"] = [replacements.get(p["path"], p) for p in amended["inputs"]]
        amended["status"] = "SEALED_AMENDMENT_8A_PENDING_CORE_AND_OTHER_BASELINE_PRICES"
    amended["amendment"] = {
        **record,
        "parent": identity(parent_path),
        "record": identity(record_path),
        "protocol": identity(protocol_path),
        "cluster_audit": identity(audit_path),
        "future_optional_arms": (
            "Any novelty/competitor arm needs a separate pre-freeze amendment with exact commands, "
            "qualification, complete cost and the same core-first admission; none is admitted here"
        ),
    }
    amended["jobs"], amended["arms"] = build_jobs(amended["config"])
    maximum = {
        "wt_pc": {
            "admit": True,
            "K": 4,
            "prior": True,
            "source_root": amended["config"]["endpoint_root"],
        },
        "dog": {"admit": True},
    }
    if "wt_pc_resolution" in amended["conditional_rule"]:
        maximum["wt_pc"] = amended["conditional_rule"]["wt_pc_resolution"]
    amended["conditional_maximum_jobs"], amended["conditional_maximum_arms"] = build_jobs(
        amended["config"], maximum
    )
    if child_8a and science_sha256(amended) != record["preserved_science_sha256"]:
        raise ValueError("amendment 8a changed frozen science or generation arguments")
    destination.parent.mkdir(parents=True, exist_ok=True)
    bound_paths = [Path(p) for p in (parent_path, protocol_path, audit_path, record_path)]
    if number not in (4, 5):
        queue = destination.parent / "queue.sh"
        setup = (
            f"export TMPDIR={shlex.quote(str(ROOT / 'target/tmp'))} PYTHONDONTWRITEBYTECODE=1\n"
            'mkdir -p "$TMPDIR"\n'
            if number == 8
            else ""
        )
        with queue.open("x") as stream:
            stream.write(f"""#!/usr/bin/env bash
set -euo pipefail
cd {shlex.quote(str(ROOT))}
mode="${{1:---dry-run}}"
if (($# > 1)) || [[ "$mode" != --dry-run && "$mode" != --run ]]; then
    printf 'usage: queue.sh [--dry-run|--run]\\n' >&2; exit 2
fi
{setup}exec nice -n 10 {shlex.quote(str(ROOT / ".venv/bin/python"))} -I \\
  -m xgraph.webqsp_test queue \\
  --registration {shlex.quote(str(destination))} \\
  --execution {shlex.quote(str(destination.parent / "execution"))} "$mode"
""")
        queue.chmod(0o755)
        amended["amendment"]["queue"] = identity(queue)
        bound_paths.append(queue)
    amended["inputs"].extend(identity(p) for p in bound_paths)
    amended["inputs"].extend(record.get("evidence", []))
    amended["parent_provenance"] = amended.get("provenance")
    amended["provenance"] = provenance(bound_paths)
    if git_commit() != amended["commit"]:
        raise ValueError("source changed during amendment")
    write_sealed(destination, amended)
    return amended


def verify_registration(path, *, metadata_only=False):
    registration = read_sealed(path)
    if git_commit() != registration["commit"]:
        raise ValueError("registered code commit changed")
    if metadata_only:
        # A pre-execution preview may inspect the closed question-only audit and
        # the registration, never inherited retrievals, graphs or output inputs.
        for field in ("parent", "protocol", "record", "cluster_audit", "queue"):
            if field in registration.get("amendment", {}):
                verify_identity(registration["amendment"][field])
        for item in registration.get("amendment", {}).get("evidence", []):
            verify_identity(item)
        return registration
    for item in registration["inputs"]:
        verify_identity(item)
    for saved in registration["model_inventory"].values():
        if model_inventory(saved["path"]) != saved:
            raise ValueError("frozen model inventory changed")
    return registration


def require_active_registration(path):
    """Keep immutable old registrations readable, but refuse superseded sizing commands."""
    pointer = ROOT / "target/webqsp-test/active-registration.json"
    if not Path(path).resolve().is_relative_to(pointer.parent):
        return
    if pointer.is_file():
        active = json.loads(pointer.read_text())
        verify_identity(active)
        if Path(path).resolve() != Path(active["path"]).resolve():
            raise ValueError(f"superseded registration; use --registration {active['path']}")


def verify_pilot(registration, path):
    timings = json.loads(Path(path).read_text())
    plan = json.loads(Path(path).with_name("queue.json").read_text())
    if (
        plan.get("provenance", {}).get("repo_commit") != registration["pilot"]["commit"]
        or plan.get("provenance", {}).get("repo_dirty") is not False
    ):
        raise ValueError("pilot source commit/clean state changed")
    if (
        not plan.get("pilot")
        or plan["pilot"].get("scope") != "validation"
        or plan["pilot"].get("questions_per_arm") != 100
    ):
        raise ValueError("the measured pilot must be the registered validation cohort")
    if timings.get("catalog_signature") != plan.get("catalog_signature"):
        raise ValueError("measured pilot signature changed")
    if registration["pilot"].get("timing_basis") == "container_occupancy":
        if Path(path).resolve() != Path(registration["pilot"]["timings"]).resolve():
            raise ValueError("pilot occupancy timings path changed")
        if (
            plan.get("catalog_signature") != registration["pilot"]["catalog_signature"]
            or plan["pilot"].get("cohort_ids") != registration["pilot"]["cohort_ids"]
            or plan.get("pilot_restart") != registration["pilot"].get("restart")
        ):
            raise ValueError("registered occupancy pilot signature or cohort changed")
    pilot = plan["pilot"]
    if (
        pilot.get("seed") != SEED
        or len(unique_ids(pilot.get("cohort_ids", []))) != 100
        or pilot.get("cohort_ids") != pilot.get("per_arm_ids")
        or plan.get("arms")
        != ["W", "WT", "B2", "agent-strong-8", "agent-feedback-8", "dog", "gcr-native"]
    ):
        raise ValueError("registered validation pilot selection or arms changed")
    for family in FAMILIES:
        if (
            plan.get("frozen_policies", {}).get(family, {}).get("recipe")
            != registration["policies"][family]["WT"]["recipe"]
        ):
            raise ValueError("validation pilot changed the frozen WT recipe")
    size_protocol(registration, timings)
    return timings


def snapshot_code(source, commit, out):
    import tarfile

    out.mkdir(parents=True)
    archive = subprocess.check_output(["git", "archive", commit], cwd=source)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(out, filter="data")
    subprocess.run(["git", "init", "--quiet", str(out)], check=True)
    common = subprocess.check_output(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=source, text=True
    ).strip()
    (out / ".git/objects/info/alternates").write_text(str(Path(common) / "objects") + "\n")
    subprocess.run(["git", "update-ref", "HEAD", commit], cwd=out, check=True)
    subprocess.run(["git", "read-tree", commit], cwd=out, check=True)
    launcher = out / "scripts/generate.sh"
    launcher.write_text(bounded_launcher(launcher.read_text()))
    if (Path(source) / ".venv").is_dir():
        snapshot_environment(Path(source) / ".venv", out)
    return identity(launcher)


def snapshot_environment(base, snapshot):
    """Own environment metadata; dependency .pth files and editable imports stay inactive."""
    import shutil

    base, snapshot = Path(base).resolve(), Path(snapshot).resolve()
    environment = snapshot / ".venv"
    (environment / "bin").mkdir(parents=True)
    shutil.copyfile(base / "pyvenv.cfg", environment / "pyvenv.cfg")
    interpreter = (base / "bin/python").resolve()
    for name in ("python", "python3", f"python{sys.version_info.major}.{sys.version_info.minor}"):
        (environment / "bin" / name).symlink_to(interpreter)
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site = environment / "lib" / version / "site-packages"
    site.mkdir(parents=True)
    dependencies = base / "lib" / version / "site-packages"
    if not dependencies.is_dir():
        raise ValueError("snapshot dependencies have a different Python version")
    (site / "snapshot.pth").write_text(
        "import sys; sys.dont_write_bytecode = True\n"
        + str(snapshot / "src")
        + "\n"
        + str(dependencies)
        + "\n"
    )


def prepare_execution(registration_path, directory):
    from xgraph.method_prompts import PromptSource
    from xgraph.subgraphrag import PROMPT_NAMES
    from xgraph.webqsp import prepare_catalog

    registration = verify_registration(registration_path)
    if datetime.now(UTC) > datetime.fromisoformat(registration["freeze_deadline"]):
        raise ValueError("execution seal deadline has passed")
    config, directory = registration["config"], Path(directory).resolve()
    if directory != Path(config["out"]) / "execution" or directory.exists():
        raise ValueError("execution requires the registered fresh destination")
    development_gate = json.loads(Path(config["gate"]).read_text())
    gate = apply_registered_gate(registration, development_gate)
    decision = gate_decision(gate)
    tiered = registration["pricing"].get("cohort_rule") == "two_tier"
    if tiered:
        require_native_checks(gate)
    if any(
        s.get("status") == "PENDING"
        and s.get("blocks_execution", True)
        and not (s["name"] == "GCR feasibility and matched timing" and decision.get("gcr") is False)
        for s in registration.get("amendment_slots", [])
    ):
        raise ValueError("pending pre-freeze amendment slot must be resolved before execution")
    for name in decision:
        for item in gate[name].get("evidence", []):
            verify_identity(item)
    timings = verify_pilot(registration, config["timings"])
    baseline_path = baseline_timing_path(registration) if tiered else None
    rates = matched_prices(registration, baseline_path, config["timings"]) if tiered else None
    sizing = size_protocol(registration, timings, gate, baseline_rates=rates)
    if tiered and not sizing["execution_ready"]:
        raise ValueError(
            "two-tier prices/qualification pending; cannot seal or change sample later"
        )
    config = resolved_config(registration, baseline_path)
    gate = apply_budget_admission(gate, sizing)
    decision = gate_decision(gate)
    ids = select_ids(registration["population"]["intersection"], sizing["questions_per_arm"])
    for source in config["rog_sources"]:
        verify_identity({key: source[key] for key in ("path", "bytes", "sha256")})
    directory.mkdir(parents=True)
    graph = goldfree_graphs([r["path"] for r in config["rog_sources"]], ids, directory / "graphs")
    audited_clusters = registration.get("question_cluster_audit", {}).get("methods", {})
    cluster_method = registration["analysis"].get("cluster_method", "transitive")
    if audited_clusters:
        frozen = audited_clusters[cluster_method]["assignments"]
        if cluster_method == "transitive" and any(graph["clusters"][q] != frozen[q] for q in ids):
            raise ValueError("question clusters differ from the prospective question-only audit")
        graph["clusters"] = frozen
    prepare_catalog(
        directory / "graphs",
        directory / "inputs",
        {f: Path(p) for f, p in config["models"].items()},
    )
    questions = json.loads((directory / "inputs/questions.json").read_text())
    if tiered:
        baseline_ids = sizing["baseline_ids"]
        admitted = sizing["baseline_admission"]
        if baseline_ids and any(admitted[k]["admit"] for k in ("gcr", "subgraphrag")):
            goldfree_graphs(
                [r["path"] for r in config["rog_sources"]],
                baseline_ids,
                directory / "baseline-graphs",
            )
            prepare_catalog(
                directory / "baseline-graphs",
                directory / "baseline-inputs",
                {f: Path(p) for f, p in config["models"].items()},
            )
        if admitted["subgraphrag"]["admit"]:
            prompts = PromptSource.load(Path(config["prompts"]), PROMPT_NAMES)
            selected = [q for q in questions if q["id"] in set(baseline_ids)]
            requests = reader_requests_from_cache(config["retrieval_scores"], prompts, selected)
            write_lines(directory / "baseline-inputs/requests-subgraphrag.jsonl", requests)
    else:
        baseline_ids = []
        prompts = PromptSource.load(Path(config["prompts"]), PROMPT_NAMES)
        requests = reader_requests_from_cache(config["retrieval_scores"], prompts, questions)
        write_lines(directory / "inputs/requests-subgraphrag.jsonl", requests)
    dog_ids = []
    if decision["dog"]:
        dog_ids = prepare_dog(config, baseline_ids if tiered else ids, directory)
        if not dog_ids:
            raise ValueError("admitted DoG has no selected common IDs")
    snapshots = [snapshot_code(ROOT, registration["commit"], directory / "code")]
    if (
        config.get("gcr_beam_semantics")
        and decision.get("gcr") is not False
        and (not tiered or sizing["baseline_admission"]["gcr"]["admit"])
    ):
        snapshots.append(
            snapshot_code(
                config["gcr_source_root"], config["gcr_source_commit"], directory / "gcr-code"
            )
        )
    if decision["wt_pc"]:
        pc = gate["wt_pc"]
        if Path(pc["source_root"]).resolve() != Path(config["endpoint_root"]).resolve():
            raise ValueError("WT-PC must come from the registered endpoint checkout")
        if git_commit(Path(pc["source_root"])) != pc["source_commit"]:
            raise ValueError("WT-PC development source changed")
        snapshots.append(
            snapshot_code(Path(pc["source_root"]), pc["source_commit"], directory / "endpoint-code")
        )
    jobs, arms = build_jobs(config, gate)
    initial = gate.get("protocol_qualification_gpu_seconds", 0) + native_pricing_charge(
        registration, baseline_path
    )
    if (
        not isinstance(initial, (int, float))
        or not math.isfinite(initial)
        or not 0 <= initial <= registration["pricing"]["reserve_seconds"]
    ):
        raise ValueError("qualification GPU seconds must be recorded within the reserve")
    files = [
        identity(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and not {".venv", ".git", "build", "__pycache__"}.intersection(p.parts)
    ]
    for code in ("code", "endpoint-code", "gcr-code"):
        environment = directory / code / ".venv"
        if environment.exists():
            files.append(identity(environment / "pyvenv.cfg"))
            files.extend(
                identity(p) for p in environment.glob("lib/python*/site-packages/snapshot.pth")
            )
    execution = {
        "artifact_kind": "xgraph.webqsp_test_execution",
        "registration": identity(registration_path),
        "commit": registration["commit"],
        "gate": gate,
        "development_gate": development_gate,
        "gate_identity": identity(config["gate"]),
        "timing_identity": identity(config["timings"]),
        "pilot_plan_identity": identity(Path(config["timings"]).with_name("queue.json")),
        "baseline_timing_identity": identity(baseline_path) if baseline_path else None,
        "resolved_config": config,
        "ids": ids,
        "baseline_ids": baseline_ids,
        "dog_ids": dog_ids,
        "sizing": sizing,
        "cluster": {q: graph["clusters"][q] for q in ids},
        "cluster_sensitivities": {
            method: {q: audited_clusters[method]["assignments"][q] for q in ids}
            for method in registration["analysis"].get("sensitivity_cluster_methods", [])
        },
        "jobs": jobs,
        "arms": arms,
        "inputs": files,
        "adapted_launchers": snapshots,
        "created_at": datetime.now(UTC).isoformat(),
        "initial_gpu_seconds": initial,
    }
    write_sealed(directory / "execution.json", execution)
    ledger = {"spent_seconds": initial, "active": None, "attempts": []}
    if tiered:
        ledger["buckets"] = {
            name: {"cap_seconds": cap, "spent_seconds": initial if name == "core" else 0}
            for name, cap in gpu_budget_caps(registration).items()
        }
    write_json(directory / "ledger.json", ledger)
    return execution


def write_lines(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def reader_requests_from_cache(path, prompts, panel):
    from xgraph.subgraphrag import prepare_requests

    wanted = {r["id"]: r for r in panel}
    unique_ids(r["id"] for r in panel)
    prepared, seen = {}, set()
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            q = row["id"]
            if q in seen:
                raise ValueError("duplicate published retrieval ID")
            seen.add(q)
            if q in wanted:
                prepared[q] = prepare_requests({q: row}, prompts, [wanted[q]], top_k=100)[0]
    return [
        prepared[q] if q in prepared else prepare_requests({}, prompts, [r], top_k=100)[0]
        for q, r in wanted.items()
    ]


def prepare_dog(config, ids, directory):
    from xgraph.webqsp import prepare_catalog

    for item in config.get("dog_data_inputs", []):
        verify_identity(item)
    source = Path(config["dog_data"])
    metadata = json.loads((source / "manifest.json").read_text())
    if (
        metadata.get("population") != "dog_top120_answer_present"
        or metadata.get("top_triples") != 120
    ):
        raise ValueError("DoG must use its predeclared released top-120 population")
    requests = read_lines(source / "requests-paths.jsonl")
    selected = [r for r in requests if r["id"] in set(ids)]
    data = directory / "dog-graphs"
    data.mkdir()
    for name in ("kg", "kg.sqlite", "versions"):
        (data / name).symlink_to(
            (source / name).resolve(), target_is_directory=(source / name).is_dir()
        )
    panel = [
        {
            "id": r["id"],
            "utterance": r["question"],
            "anchor": r["anchor"],
            "kg_version": r["kg_version"],
            "answers": [],
            "missing_subgraph": False,
            "coverage": {},
        }
        for r in selected
    ]
    write_json(data / "questions.json", panel)
    write_json(data / "questions-released.json", panel)
    write_json(
        data / "manifest.json",
        {**metadata, "counts": {"questions": len(panel)}, "gold_loaded": False},
    )
    prepare_catalog(
        data, directory / "dog-inputs", {f: Path(p) for f, p in config["models"].items()}
    )
    return [r["id"] for r in selected]


def verify_execution(registration_path, directory):
    from xgraph.webqsp import verify_catalog

    sealed = read_sealed(Path(directory) / "execution.json")
    if sealed.get("artifact_kind") == "xgraph.webqsp_test_continuation":
        from xgraph.webqsp_continuation import verify_continuation

        return verify_continuation(registration_path, directory)
    registration = verify_registration(registration_path)
    execution = sealed
    if (
        execution["registration"] != identity(registration_path)
        or execution["commit"] != registration["commit"]
    ):
        raise ValueError("execution registration changed")
    gate_decision(execution["gate"])
    for item in [
        execution["gate_identity"],
        execution["timing_identity"],
        execution["pilot_plan_identity"],
        *execution["inputs"],
    ]:
        verify_identity(item)
    tiered = registration["pricing"].get("cohort_rule") == "two_tier"
    baseline_path = None
    if tiered and execution.get("baseline_timing_identity"):
        verify_identity(execution["baseline_timing_identity"])
        baseline_path = Path(execution["baseline_timing_identity"]["path"])
    if registration.get("gcr_native_pricing") and execution["initial_gpu_seconds"] != (
        execution["gate"].get("protocol_qualification_gpu_seconds", 0)
        + native_pricing_charge(registration, baseline_path)
    ):
        raise ValueError("native WebQSP GCR pricing charge changed in the execution reserve")
    verify_catalog(Path(directory) / "inputs")
    development_gate = execution.get("development_gate", execution["gate"])
    if development_gate != json.loads(Path(execution["gate_identity"]["path"]).read_text()):
        raise ValueError("execution development qualification record changed")
    if tiered:
        require_native_checks(development_gate)
    sizing = size_protocol(
        registration,
        json.loads(Path(execution["timing_identity"]["path"]).read_text()),
        development_gate,
        baseline_rates=matched_prices(
            registration, baseline_path, execution["timing_identity"]["path"]
        )
        if tiered
        else None,
    )
    if execution["gate"] != apply_budget_admission(
        apply_registered_gate(registration, development_gate), sizing
    ):
        raise ValueError("execution budget admission changed")
    if execution["gate"]["dog"]["admit"]:
        verify_catalog(Path(directory) / "dog-inputs")
    config = resolved_config(registration, baseline_path)
    if tiered:
        if (
            not sizing["execution_ready"]
            or execution["sizing"] != sizing
            or execution.get("baseline_ids") != sizing["baseline_ids"]
            or execution.get("resolved_config") != config
        ):
            raise ValueError("sealed two-tier sizes, prices or commands changed")
        if any(sizing["baseline_admission"][k]["admit"] for k in ("gcr", "subgraphrag")):
            verify_catalog(Path(directory) / "baseline-inputs")
    expected_jobs, expected_arms = build_jobs(config, execution["gate"])
    if expected_jobs != execution["jobs"] or expected_arms != execution["arms"]:
        raise ValueError("execution commands changed")
    n = sizing["questions_per_arm"]
    if execution["ids"] != select_ids(registration["population"]["intersection"], n):
        raise ValueError("execution ID sample changed")
    audited = registration.get("question_cluster_audit", {}).get("methods", {})
    if audited:
        method = registration["analysis"]["cluster_method"]
        expected = {q: audited[method]["assignments"][q] for q in execution["ids"]}
        sensitivities = {
            method: {q: audited[method]["assignments"][q] for q in execution["ids"]}
            for method in registration["analysis"]["sensitivity_cluster_methods"]
        }
        if execution["cluster"] != expected or execution["cluster_sensitivities"] != sensitivities:
            raise ValueError("execution differs from the frozen question-only clusters")
    return registration, execution


def stage_id(job):
    return f"{job['scope']}-{job['family']}-{job['name']}"


def receipt_path(directory, job):
    return Path(directory) / "receipts" / f"{stage_id(job)}.json"


def verify_receipts(execution, directory, *, complete=True, incomplete_gcr=None):
    for job in execution["jobs"]:
        path = receipt_path(directory, job)
        if not path.is_file():
            if job["name"] == "gcr-native" and incomplete_gcr is not None:
                incomplete_gcr[job["family"]] = f"incomplete stage receipt: {stage_id(job)}"
                continue
            if complete:
                raise ValueError(f"incomplete stage receipt: {stage_id(job)}")
            continue
        receipt = read_sealed(path)
        if (
            receipt["execution_sha256"] != sha256_file(Path(directory) / "execution.json")
            or receipt["job"] != job
        ):
            raise ValueError("stage receipt command/seal changed or failed")
        for item in receipt["outputs"]:
            verify_identity(item)
        if receipt["returncode"] != 0:
            if job["name"] == "gcr-native" and incomplete_gcr is not None:
                incomplete_gcr[job["family"]] = f"failed stage receipt: {stage_id(job)}"
            else:
                raise ValueError("stage receipt command/seal changed or failed")


def gpu_budget_bucket(job):
    return "core" if job["scope"] == "primary" else "published_baseline_subpanel"


def gpu_budget_caps(registration):
    pricing = registration["pricing"]
    caps = {"core": pricing["cap_seconds"]}
    if pricing.get("cohort_rule") == "two_tier":
        caps["published_baseline_subpanel"] = pricing["baseline_cap_seconds"]
    return caps


def verify_gpu_ledger(registration, execution, directory, *, ledger=None):
    if ledger is None:
        ledger = json.loads((Path(directory) / "ledger.json").read_text())
    if "continuation" in execution:
        continuation = execution["continuation"]
        history = continuation["history"]["snapshot"]
        verify_identity(history)
        parent = json.loads(Path(history["path"]).read_text())
        if (
            ledger.get("history") != history
            or execution["initial_gpu_seconds"] != parent["spent_seconds"]
            or continuation["parent_charge"]
            != {k: v["spent_seconds"] for k, v in parent["buckets"].items()}
        ):
            raise ValueError("continuation ledger lost or changed its parent history")
    jobs = {stage_id(job): job for job in execution["jobs"] if job["kind"] == "gpu"}
    caps = gpu_budget_caps(registration)
    charges = [execution["initial_gpu_seconds"], ledger["spent_seconds"]]
    charges.extend(attempt["elapsed_seconds"] for attempt in ledger["attempts"])
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in charges
    ):
        raise ValueError("GPU ledger charges must be finite and nonnegative")
    charged = execution["initial_gpu_seconds"] + sum(
        attempt["elapsed_seconds"] for attempt in ledger["attempts"]
    )
    if "continuation" in execution:
        charged = execution["initial_gpu_seconds"]
        for attempt in ledger["attempts"]:
            charged += attempt["elapsed_seconds"]
        if ledger["spent_seconds"] != charged:
            raise ValueError("continuation cumulative charge changed")
    if (
        ledger["active"] is not None
        or not math.isclose(ledger["spent_seconds"], charged, rel_tol=0, abs_tol=1e-6)
        or charged > sum(caps.values())
        or any(attempt["stage"] not in jobs for attempt in ledger["attempts"])
    ):
        raise ValueError("GPU ledger is incomplete, inconsistent or over cap")
    if len(caps) > 1:
        if set(ledger.get("buckets", {})) != set(caps):
            raise ValueError("GPU ledger must record both separate allocations")
        bucket_charges = dict.fromkeys(caps, 0)
        bucket_charges["core"] = execution["initial_gpu_seconds"]
        for attempt in ledger["attempts"]:
            bucket = gpu_budget_bucket(jobs[attempt["stage"]])
            if attempt.get("budget_bucket") != bucket:
                raise ValueError("GPU ledger attempt charged to the wrong allocation")
            bucket_charges[bucket] += attempt["elapsed_seconds"]
        for name, cap in caps.items():
            account = ledger["buckets"][name]
            spent = account.get("spent_seconds")
            if "continuation" in execution and spent != bucket_charges[name]:
                raise ValueError("continuation bucket charge changed")
            if (
                isinstance(spent, bool)
                or not isinstance(spent, (int, float))
                or not math.isfinite(spent)
                or account.get("cap_seconds") != cap
                or not math.isclose(spent, bucket_charges[name], rel_tol=0, abs_tol=1e-6)
                or not 0 <= spent <= cap
            ):
                raise ValueError(f"GPU ledger allocation inconsistent or over cap: {name}")
    for stage, job in jobs.items():
        if "launcher_wall_cap_seconds" in job:
            funded = positive(job["launcher_wall_cap_seconds"], "funded launcher wall cap")
            attempts = [a for a in ledger["attempts"] if a["stage"] == stage]
            if len(attempts) > 1 or any(
                a.get("maximum_seconds", funded) > funded
                or (a["elapsed_seconds"] > funded and a["returncode"] == 0)
                for a in attempts
            ):
                raise ValueError("funded launcher wall exceeded or retried; GCR is INCOMPLETE")
        path = receipt_path(directory, job)
        if not path.exists():
            continue
        receipt = read_sealed(path)
        attempts = [
            attempt
            for attempt in ledger["attempts"]
            if attempt["stage"] == stage
            and attempt["returncode"] == receipt["returncode"]
            and (receipt["returncode"] != 0 or not attempt["timeout"])
        ]
        if receipt["skipped_empty_requests"]:
            matches = not attempts and receipt["gpu_seconds"] == 0
        else:
            matches = len(attempts) == 1 and math.isclose(
                attempts[0]["elapsed_seconds"], receipt["gpu_seconds"], rel_tol=0, abs_tol=1e-6
            )
        if not matches:
            raise ValueError(f"GPU ledger does not match completed receipt: {stage}")
        if (
            "continuation" in execution
            and attempts
            and attempts[0]["elapsed_seconds"] != receipt["gpu_seconds"]
        ):
            raise ValueError(f"GPU ledger does not match completed receipt: {stage}")
    return ledger


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    temporary.replace(path)


def wait_occupancy_stage(process, job, allowance, started, events, offset):
    """Operator a78320a's watchdog, with a separate preparation/headroom wall bound."""
    import time
    from contextlib import suppress

    code = None
    while True:
        now = time.monotonic()
        seconds, launched = gpu_occupancy(events, offset, started) if events else (0, False)
        left = job["wall_timeout_seconds"] - (now - started)
        if left <= 0 or (launched and seconds >= allowance):
            raise subprocess.TimeoutExpired(job["command"], allowance if left > 0 else 0)
        if code is not None:
            return code
        with suppress(subprocess.TimeoutExpired):
            code = process.wait(
                timeout=min(1, left, allowance - seconds) if launched else min(1, left)
            )


def execute_gpu_stage(
    job,
    ledger_path,
    cap_seconds,
    log_path,
    *,
    launcher_env=None,
    cleanup=None,
    occupancy_accounting=False,
):
    """Bound the process group; CWQ opts into durable container occupancy charging."""
    import signal
    import time
    from contextlib import suppress

    from xgraph.load_progress import LoadProgressTimeout, StartupTimeout, wait_for_load

    load_seconds = (
        positive(job["load_progress_seconds"], "load silence")
        if "load_progress_seconds" in job
        else None
    )
    ledger = json.loads(Path(ledger_path).read_text())
    if ledger["active"] is not None:
        raise ValueError(
            "unreconciled GPU launch; preserve artifacts and reconcile its charge first"
        )
    bucket = gpu_budget_bucket(job) if "buckets" in ledger else None
    account = ledger["buckets"][bucket] if bucket else ledger
    if bucket and account["cap_seconds"] != cap_seconds:
        raise ValueError("GPU budget allocation cap changed")
    remaining = cap_seconds - account["spent_seconds"]
    funded_cap = job.get("launcher_wall_cap_seconds")
    if funded_cap is not None:
        funded_cap = positive(funded_cap, "funded launcher wall cap")
        if any(a["stage"] == stage_id(job) for a in ledger["attempts"]):
            raise ValueError("GPU budget exhausted for this funded arm; no retry or top-up")
        remaining = min(remaining, funded_cap)
    maximum = 43200 if bucket == "published_baseline_subpanel" else 144000
    needs_gpu = not occupancy_accounting or job["kind"] == "gpu"
    if not math.isfinite(remaining) or (needs_gpu and remaining <= 30) or cap_seconds > maximum:
        raise ValueError("GPU budget exhausted; no further stage launched")
    started = time.monotonic()
    wall_bound = (
        job.get("wall_timeout_seconds", 24 * 3600) if occupancy_accounting else remaining - 30
    )
    events, offset = None, 0
    if occupancy_accounting and needs_gpu:
        events = Path(launcher_env["XG_GPU_EVENTS"])
        with events.open("ab") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        offset = events.stat().st_size
    watchdog = [
        "timeout",
        "--signal=TERM",
        "--kill-after=15",
        str(wall_bound),
        *job["command"],
    ]
    ledger["active"] = {
        "stage": stage_id(job),
        "started_at": datetime.now(UTC).isoformat(),
        "maximum_seconds": remaining if needs_gpu else 0,
        "watchdog_argv": watchdog,
    }
    if launcher_env:
        ledger["active"]["launcher_env"] = launcher_env
    if bucket:
        ledger["active"]["budget_bucket"] = bucket
    if occupancy_accounting:
        ledger["active"].update(
            started_monotonic=started,
            wall_timeout_seconds=wall_bound,
            gpu_allowance_seconds=remaining - 30 if needs_gpu else 0,
            gpu_events=str(events) if events else None,
            events_offset=offset,
        )
    atomic_json(ledger_path, ledger)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("XG_", "XGRAPH_"))}
    env.update(job["env"])
    env.update(launcher_env or {})
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    timed_out = False
    load_stalled = False
    startup_timed_out = False
    cleanup_error = None
    event_error = None
    process, pending_signal, finishing = None, None, False

    def interrupted(signum, _frame):
        nonlocal pending_signal
        pending_signal = signum
        # Popen must return its child handle before interruption can unwind it.
        # During cleanup, keep the timeout machinery alive and defer the signal.
        if process is not None and not finishing:
            raise InterruptedError(signum)

    def terminate():
        if process is None:
            return
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=1)

    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        with Path(log_path).open("x") as log:
            try:
                if pending_signal is not None:
                    raise InterruptedError(pending_signal)
                process = subprocess.Popen(
                    watchdog,
                    env=env,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                if pending_signal is not None:
                    raise InterruptedError(pending_signal)
                status = (
                    wait_occupancy_stage(
                        process,
                        {**job, "wall_timeout_seconds": wall_bound},
                        remaining - 30,
                        started,
                        events,
                        offset,
                    )
                    if occupancy_accounting
                    else wait_for_load(process, log_path, load_seconds, remaining - 30)
                    if load_seconds is not None
                    else process.wait(timeout=remaining - 30)
                )
                timed_out = status == 124  # GNU timeout may beat the outer watchdog.
                if (
                    status == 0
                    and events
                    and not gpu_occupancy(events, offset, started, require_end=True)[1]
                ):
                    raise ValueError("stage succeeded without a GPU launch event")
            except subprocess.TimeoutExpired as error:
                finishing = True
                timed_out = True
                load_stalled = isinstance(error, LoadProgressTimeout)
                startup_timed_out = isinstance(error, StartupTimeout)
                if load_stalled:
                    log.write(
                        f"\n[xgraph-load] no load progress for {error.timeout} seconds; stopping\n"
                    )
                    log.flush()
                elif startup_timed_out:
                    log.write(
                        f"\n[xgraph-load] total startup bound {error.timeout} seconds; stopping\n"
                    )
                    log.flush()
                status = 125 if load_stalled else 124
                terminate()
            except InterruptedError as error:
                finishing = True
                status = 128 + error.args[0]
                terminate()
            except (ValueError, KeyError, TypeError, OSError) as error:
                if not occupancy_accounting:
                    raise
                finishing = True
                event_error = str(error)
                status = 125
                terminate()
            finally:
                finishing = True
            if cleanup is not None and process is not None:
                # No surviving launcher may submit a container after inspection.
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
        cleared = True
        if cleanup is None and (timed_out or status) and job["env"].get("XG_CONTAINER_NAME"):
            from xgraph.container_cleanup import stop_container

            try:
                stop_container(job["env"]["XG_CONTAINER_NAME"], job["env"]["XG_GPU_EVENTS"])
            except (ValueError, KeyError, OSError) as error:
                cleanup_error = error
                cleared = False
                ledger["active"]["uncertain_cleanup"] = job["env"]["XG_CONTAINER_NAME"]
        if cleanup is not None:
            try:
                ledger["active"]["cleanup"] = cleanup()
            except Exception as error:
                ledger["active"]["cleanup"] = {"confirmed_absent": False, "error": str(error)}
            cleared = ledger["active"]["cleanup"].get("confirmed_absent") is True
        elif job["scope"] == "cwq-validation" and job.get("kind") != "cpu":
            cleared = False
            ledger["active"]["cleanup"] = {
                "confirmed_absent": False,
                "error": "CWQ container absence was not verified",
            }
        if pending_signal is not None and not timed_out:
            status = 128 + pending_signal
        if not cleared and status == 0:
            status = 125
        elapsed = time.monotonic() - started
        if funded_cap is not None and elapsed > remaining:
            status, timed_out = 124, True
        if occupancy_accounting:
            ledger["active"]["wall_seconds"] = elapsed
            try:
                if event_error or not cleared:
                    raise ValueError(event_error or "container cleanup unconfirmed")
                elapsed, launched = gpu_occupancy(events, offset, started) if events else (0, False)
                if events and not launched and ledger["active"]["cleanup"].get("container_id"):
                    raise ValueError("owned container has no GPU launch event")
                ledger["active"].update(accounting="measured", container_launched=launched)
            except (ValueError, KeyError, TypeError, OSError) as error:
                # Unknown occupancy retains the whole admitted reservation and blocks resume.
                elapsed, cleared = remaining, False
                ledger["active"].update(accounting="reserved", accounting_error=str(error))
                status = status or 125
            if events and events.is_file():
                ledger["active"]["gpu_events_identity"] = identity(events)
        ledger["spent_seconds"] += elapsed
        if bucket:
            account["spent_seconds"] += elapsed
        ledger["attempts"].append(
            {
                **ledger["active"],
                "elapsed_seconds": elapsed,
                "returncode": status,
                "timeout": timed_out,
                **({"load_progress_timeout": True} if load_stalled else {}),
                **({"startup_timeout": True} if startup_timed_out else {}),
            }
        )
        if cleared:
            ledger["active"] = None
        else:
            ledger["active"]["charged_through"] = datetime.now(UTC).isoformat()
            ledger["active"]["elapsed_seconds"] = elapsed
        atomic_json(ledger_path, ledger)
        if cleanup_error:
            raise ValueError("unknown GCR cleanup; reservation retained") from cleanup_error
        if account["spent_seconds"] > cap_seconds:
            raise RuntimeError("hard GPU cap exceeded during cleanup; no scores are admissible")
        return status, elapsed
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def preview(registration, execution=None):
    jobs = execution["jobs"] if execution else registration["conditional_maximum_jobs"]
    if execution is None:
        print(
            "# SEALED RULES; execution blocked pending measured validation timings "
            "and recorded development gate"
        )
        print("# UNMEASURED sizing: " + json.dumps(size_protocol(registration)))
        print("# Conditional maximum below is a preview, not an admission decision")
    for job in jobs:
        print(f"# {job['scope']} {job['family']} {job['name']} {job.get('condition', 'always')}")
        print(shlex.join(["env", *[f"{k}={v}" for k, v in job["env"].items()], *job["command"]]))


def run_owner(registration_path, directory):
    import fcntl
    import time

    registration, execution = verify_execution(registration_path, directory)
    directory = Path(directory)
    prefix = execution.get("continuation", {}).get("status_prefix", "webqsp-test")

    def progress_line(job, status):
        progress = Path("/home/jonas/git/xgraph/target/queue/webqsp-test.log")
        progress.parent.mkdir(parents=True, exist_ok=True)
        label = job["name"] if job["scope"] == "primary" else job["scope"] + "-" + job["name"]
        with progress.open("a") as stream:
            stream.write(
                f"{prefix} {job['family']} {label} {status} "
                f"{datetime.now().astimezone().isoformat(timespec='seconds')}\n"
            )

    start = (
        execution["created_at"]
        if registration["generation_not_before"] == "execution_seal"
        else registration["generation_not_before"]
    )
    if datetime.now(UTC) < datetime.fromisoformat(start):
        raise ValueError(f"WebQSP test generation starts {start}")
    with Path("/tmp/xgraph-webqsp-queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for job in execution["jobs"]:
            verify_receipts(execution, directory, complete=False)
            ledger = verify_gpu_ledger(registration, execution, directory)
            receipt = receipt_path(directory, job)
            if receipt.exists():
                if "continuation" in execution:
                    progress_line(job, "skip-receipt")
                continue
            if job["name"] == "gcr-native" and any(
                a["stage"] == stage_id(job) for a in ledger["attempts"]
            ):
                if "continuation" in execution:
                    progress_line(job, "INCOMPLETE-no-retry")
                continue  # Retain the sealed arm as INCOMPLETE; never retry after seeing outputs.
            outputs = list(map(Path, job["outputs"]))
            if any(p.exists() or p.is_symlink() for p in outputs):
                raise ValueError(f"orphaned/partial output prevents reuse: {stage_id(job)}")
            for path in outputs:
                path.parent.mkdir(parents=True, exist_ok=True)
            logs = directory / "logs"
            logs.mkdir(exist_ok=True)
            log = logs / f"{stage_id(job)}-{time.time_ns()}.log"
            if "continuation" in execution:
                progress_line(job, "start")
            elapsed, skipped = 0.0, False
            if job.get("condition") == "nonempty_requests" and not read_lines(job["requests"]):
                for output in outputs:
                    write_lines(output, [])
                status, skipped = 0, True
            elif job["kind"] == "gpu":
                cap = (
                    ledger["buckets"][gpu_budget_bucket(job)]["cap_seconds"]
                    if "buckets" in ledger
                    else registration["pricing"]["cap_seconds"]
                )
                status, elapsed = execute_gpu_stage(job, directory / "ledger.json", cap, log)
            else:
                with log.open("x") as stream:
                    result = subprocess.run(
                        job["command"],
                        cwd=ROOT,
                        env={
                            **{
                                k: v
                                for k, v in os.environ.items()
                                if not k.startswith(("XG_", "XGRAPH_"))
                            },
                            **job["env"],
                        },
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                status = result.returncode
            progress_line(job, f"exit {status}")
            if status:
                if job["name"] == "gcr-native":
                    continue
                raise RuntimeError(f"stage {stage_id(job)} exited {status}; outputs preserved")
            if job["kind"] == "gpu":
                try:
                    require_complete(
                        read_lines(outputs[0]), [r["id"] for r in read_lines(job["requests"])]
                    )
                    if not skipped:
                        for suffix in (".summary.json", ".provenance.json"):
                            sidecar = Path(str(outputs[0]) + suffix)
                            if not sidecar.is_file():
                                raise ValueError(f"incomplete generation sidecar: {sidecar}")
                            outputs.append(sidecar)
                except (ValueError, FileNotFoundError):
                    if "continuation" in execution:
                        progress_line(job, "audit-failed")
                    if job["name"] == "gcr-native":
                        continue
                    raise
            write_sealed(
                receipt,
                {
                    "execution_sha256": sha256_file(directory / "execution.json"),
                    "job": job,
                    "returncode": status,
                    "gpu_seconds": elapsed,
                    "skipped_empty_requests": skipped,
                    "outputs": [identity(p) for p in outputs],
                },
            )
        verify_receipts(execution, directory, incomplete_gcr={})


def pc_actions(directory, family, *, merge=False):
    from xgraph.index import load_kg_index
    from xgraph.kgversions import load_kg_versions
    from xgraph.programs import MatchedExecutor, record_path
    from xgraph.score import record_failure
    from xgraph.tokenizer import ServedTokenizer
    from xgraph.toolkg import KGStore, version_stores
    from xgraph.webqsp import answers_of, unrender_label

    directory = Path(directory)
    execution = read_sealed(directory / "execution.json")
    registration = read_sealed(execution["registration"]["path"])
    pc = execution["gate"]["wt_pc"]
    data, folder = directory / "inputs", directory / "runs" / family
    wt = {r["id"]: r for r in read_lines(folder / "WT-sets.jsonl")}
    if merge:
        support = json.loads((folder / "pc-support.json").read_text())
        views = {}
        for kind in ("question", "prior") if pc["prior"] else ("question",):
            views[kind] = []
            for view in range(pc["K"]):
                rows = read_lines(folder / f"pc-{kind}-{view}.jsonl")
                requests = read_lines(folder / f"pc-{kind}-{view}.requests.jsonl")
                require_complete(rows, [r["id"] for r in requests])
                annotated = {}
                for row in rows:
                    labels = answers_of(row)
                    owners = [e for e, label in support[row["id"]].items() if labels == [label]]
                    annotated[row["id"]] = {
                        **row,
                        "endpoint_entity_id": owners[0]
                        if len(owners) == 1 and not record_failure(row)
                        else None,
                    }
                views[kind].append(annotated)
        merged = [
            rank_endpoint_set(
                wt[q],
                support.get(q, {}),
                [v[q] for v in views["question"] if q in v],
                prior=[v[q] for v in views["prior"] if q in v] if pc["prior"] else None,
            )
            for q in execution["ids"]
        ]
        write_lines(folder / "WT-PC.jsonl", merged)
        return
    requests = {r["id"]: r for r in read_lines(folder / "WT-requests.jsonl")}
    raw = {r["id"]: r for r in read_lines(folder / "WT-raw.jsonl")}
    store = KGStore(data / "kg.sqlite")
    versions = version_stores(
        store, data / "versions", {r["kg_version"] for r in requests.values()}
    )
    index = load_kg_index(data / f"index-{family}")
    tokenizer = ServedTokenizer.from_file(
        Path(registration["config"]["models"][family]) / "tokenizer.json"
    )
    executor = MatchedExecutor(
        index,
        tokenizer,
        versions=load_kg_versions(data / "versions", index),
        start_rank="sitelinks",
        unquoted_starts=True,
    )
    support, eligible = {}, {}
    for q in execution["ids"]:
        parsed = record_path(raw[q])
        if (
            record_failure(raw[q])
            or parsed is None
            or getattr(parsed[1], "selections", {})
            or getattr(parsed[1], "audit", None)
        ):
            continue
        start, relations = parsed
        result = executor.execute_set(
            start, relations, requests[q].get("anchor"), requests[q].get("kg_version")
        )
        read = versions[requests[q]["kg_version"]]
        endpoints = {e: unrender_label(read.label(int(e[1:]))) for e in result["reached"]}
        if not endpoints or set(endpoints.values()) != set(wt[q]["answers"]):
            continue
        support[q] = endpoints
        eligible[q] = "⟦ " + " · ".join([start, *relations]) + "⟧"
    write_json(folder / "pc-support.json", support)
    for kind in ("question", "prior") if pc["prior"] else ("question",):
        for view in range(pc["K"]):
            rows = []
            for q, path in eligible.items():
                request = copy.deepcopy(requests[q])
                messages = request["messages"]
                if kind == "prior":
                    messages = [
                        *messages[:-3],
                        {
                            "role": "user",
                            "content": "Continue with one bound Answer span after the KG display.",
                        },
                    ]
                request.update(
                    messages=[*messages, {"role": "assistant", "content": path}],
                    prefill_replay=True,
                    seed=17 + view,
                )
                rows.append(request)
            write_lines(folder / f"pc-{kind}-{view}.requests.jsonl", rows)


def released_reader_metrics(question, prediction, gold, *, rog=False):
    """Data-only reimplementation of the pinned release's corrected reader measures."""

    from xgraph.twowiki import rog_normalize

    raw = prediction if isinstance(prediction, str) else "\n".join(prediction)
    lines = raw.split("\n")
    predicted = list(
        dict.fromkeys(
            lines
            if rog
            else [
                p
                for p in lines
                if "ans:" in p
                and "none" not in p.lower()
                and "ans: not available" not in p.lower()
                and "ans: no information available" not in p.lower()
            ]
        )
    )
    answers = sorted(dict.fromkeys(gold), key=len, reverse=True)
    if "when" in question.lower() or "what year" in question.lower():
        answers = [
            a.split("-")[0] if "-" in a and a.split("-")[0].isdigit() else a for a in answers
        ]
    double = any(
        k in question.lower()
        for k in (
            "when",
            "what year",
            "which year",
            "where",
            "sport",
            "what countr",
            "language",
            "nba finals",
            "world series",
        )
    )

    def matches(p, a):
        return rog_normalize(a) in rog_normalize(p) or (
            double and rog_normalize(p.split("ans:")[-1].strip()) in rog_normalize(a)
        )

    remaining = sorted(predicted, key=len, reverse=True)
    matched = 0
    for answer in answers:
        for p in remaining:
            if matches(p, answer):
                matched += 1
                remaining.remove(p)
                break
    precision = matched / len(predicted) if predicted else 0
    recall = matched / len(answers) if answers else 0
    return {
        "corrected_hits_at_1": int(
            bool(predicted) and any(matches(predicted[0], a) for a in answers)
        ),
        "corrected_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
        "corrected_precision": precision,
        "corrected_recall": recall,
        "published_any_hit": int(
            any(matches(p, a) for a in answers for p in (predicted if "ans:" in raw else lines))
        ),
        "matched": matched,
        "predictions": len(predicted),
        "gold": len(answers),
        "empty_gold_extension": not answers,
    }


def load_gold(registration, ids):
    from xgraph.webqsp import graph_coverage

    config = registration["config"]
    for pin in config["rog_sources"]:
        if sha256_file(Path(pin["path"])) != pin["sha256"]:
            raise ValueError("pinned test data changed; gold not scored")
    rows = projected_rows(
        [p["path"] for p in config["rog_sources"]],
        ["id", "answer", "question", "q_entity", "graph"],
    )
    selected = set(ids)
    panel = [
        {
            "id": r["id"],
            "utterance": r["question"],
            "answers": r["answer"],
            "coverage": graph_coverage(r),
            "empty_graph": not r["graph"],
            "missing_subgraph": False,
        }
        for r in rows
        if r["id"] in selected
    ]
    if {q["id"] for q in panel} != selected:
        raise ValueError("incomplete primary gold after all arms completed")
    return panel


def verify_released_pin(path, pin):
    path = Path(path)
    if path.stat().st_size != pin["bytes"]:
        raise ValueError("released output size mismatch")
    saved = identity(path)
    if pin.get("sha256") is not None:
        if saved["sha256"] != pin["sha256"]:
            raise ValueError("released output SHA-256 mismatch")
    elif pin.get("git_blob_sha1"):
        digest = hashlib.sha1(f"blob {pin['bytes']}\0".encode(), usedforsecurity=False)
        with path.open("rb") as stream:
            while block := stream.read(1 << 20):
                digest.update(block)
        if digest.hexdigest() != pin["git_blob_sha1"]:
            raise ValueError("released output Git blob identity mismatch")
    else:
        raise ValueError("released output has no immutable published hash")
    return saved


def released_outputs(registration, panel, directory):
    import re
    import urllib.request

    from xgraph.webqsp import score_records

    report = {}
    ids = {q["id"] for q in panel}
    for pin in registration["config"]["released_outputs"]:
        path = directory / "released" / (pin["label"] + ".jsonl")
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(".part")
            with (
                urllib.request.urlopen(pin["url"], timeout=60) as response,
                temporary.open("xb") as stream,
            ):
                count = 0
                while block := response.read(1 << 20):
                    count += len(block)
                    if count > pin["bytes"]:
                        raise ValueError("released output exceeds pinned size")
                    stream.write(block)
            verify_released_pin(temporary, pin)
            temporary.replace(path)
        verify_released_pin(path, pin)
        raw = read_lines(path)
        unique_ids(r["id"] for r in raw)
        by_id = {r["id"]: r for r in raw}
        missing = sorted(ids - by_id.keys())
        if missing:
            write_json(
                directory / f"{pin['label']}-incomplete.json",
                {"missing_ids": missing, "required_population": sorted(ids)},
            )
            raise ValueError(f"incomplete released arm: {pin['label']}; missing IDs reported")
        records, native = [], []
        for q in panel:
            prediction = by_id[q["id"]]["prediction"]
            if pin["reader"] == "subgraphrag":
                text = prediction if isinstance(prediction, str) else "\n".join(prediction)
                answers = [
                    a.strip()
                    for a in re.findall(r"^\s*ans:\s*([^\n]+)", text, re.I | re.M)
                    if a.strip().casefold() not in ("not available", "no information available")
                ]
            else:
                answers = prediction.split("\n") if isinstance(prediction, str) else prediction
            records.append({"id": q["id"], "answers": answers})
            native.append(
                released_reader_metrics(
                    q["utterance"], prediction, q["answers"], rog=pin["reader"] == "rog"
                )
            )
        report[pin["label"]] = {
            "kind": "released outputs",
            "source": pin,
            "identity": identity(path),
            "new_gpu_seconds": 0,
            "historical_inference_and_retrieval_cost": "unknown, not zero",
            "our_metrics": score_records(records, panel),
            "released_corrected_reader_metrics": {
                k: sum(r[k] for r in native) / len(native)
                for k in (
                    "corrected_hits_at_1",
                    "corrected_f1",
                    "corrected_precision",
                    "corrected_recall",
                    "published_any_hit",
                )
            },
            "native_mid_f1": (
                "unavailable: released label outputs do not recover authoritative MIDs"
            ),
        }
    return report


def omission_census(registration, directory):
    """Post-completion gold-column audit of IDs outside the common population."""
    import pyarrow.parquet as pq

    source = registration["config"].get("ml_projection")
    if source is None:
        return {"status": "not supplied"}
    metadata = json.loads(Path(source).read_text())
    rows = []
    for pin in metadata["sources"]:
        reader = IdRangeReader(pin["url"], pin["bytes"])
        rows.extend(
            pq.ParquetFile(reader).read(columns=["id", "answer"], use_threads=False).to_pylist()
        )
    unique_ids(r["id"] for r in rows)
    if {r["id"] for r in rows} != set(registration["population"]["ml"]):
        raise ValueError("ml1996 gold-column IDs differ from the registered ID projection")
    path = directory / "ml-gold-column-projection.json"
    write_json(path, rows)
    outside = set(registration["population"]["ml_only"])
    return {
        "identity": identity(path),
        "population": "ml1996 test; outside-intersection counts never enter paired accuracy",
        "empty_gold_ids": [r["id"] for r in rows if not r["answer"]],
        "ml_only_answer_counts": {r["id"]: len(r["answer"]) for r in rows if r["id"] in outside},
        "native_mid_scores": "not recoverable from label outputs",
    }


def dog_metric_panel(panel, requests):
    from xgraph.webqsp import graph_coverage, unrender_label

    unique_ids(r["id"] for r in requests)
    graphs = {r["id"]: r for r in requests}
    if set(graphs) != {q["id"] for q in panel}:
        raise ValueError("DoG coverage requires the same complete secondary population")
    result = []
    for question in panel:
        request = graphs[question["id"]]
        labels = {
            row["ids"][i]: unrender_label(row["labels"][i])
            for row in request["dog_graph"]
            for i in (0, 2)
        }
        graph = [[unrender_label(v) for v in row["labels"]] for row in request["dog_graph"]]
        topics = [labels[int(t[1:])] for t in request["topics"] if int(t[1:]) in labels]
        coverage = graph_coverage(
            {"graph": graph, "q_entity": topics, "answer": question["answers"]}
        )
        result.append({**question, "coverage": coverage, "empty_graph": not graph})
    return result


def finish_scoring(
    registration, execution, rows, directory, incomplete_gcr=None, *, embargo_override=None
):
    import numpy as np

    from xgraph.mintqa_rematch import _paired_bootstrap, clustered_difference, usage
    from xgraph.mintqa_validation import intersection_holm
    from xgraph.webqsp import score_records

    ledger = verify_gpu_ledger(registration, execution, directory)
    panel = load_gold(registration, execution["ids"])
    omissions = omission_census(registration, directory)
    released = released_outputs(registration, panel, directory)
    results, comparisons, costs = {}, {}, {}
    members = [
        {"id": q, "hops": 1, "cluster": execution["cluster"][q], "weight": 1.0}
        for q in execution["ids"]
    ]
    for (scope, family, name), records in rows.items():
        questions = [q for q in panel if q["id"] in set(execution_ids(execution, scope))]
        if scope == "dog":
            questions = dog_metric_panel(
                questions, read_lines(directory / "dog-runs" / family / "dog-requests.jsonl")
            )
        key = f"{scope}/{family}/{name}"
        results[key] = score_records(records, questions, failed_zero=True)
        values = [usage(r) for r in records]
        costs[key] = {
            field: {
                "total": sum(v[field] for v in values),
                "mean": float(np.mean([v[field] for v in values])),
                "p50": float(np.quantile([v[field] for v in values], 0.5)),
                "p95": float(np.quantile([v[field] for v in values], 0.95)),
            }
            for field in ("prompt_tokens", "generated_tokens", "model_calls", "observation_tokens")
        }
        total = [v["prompt_tokens"] + v["generated_tokens"] for v in values]
        costs[key]["total_tokens"] = {
            "total": sum(total),
            "mean": float(np.mean(total)),
            "p50": float(np.quantile(total, 0.5)),
            "p95": float(np.quantile(total, 0.95)),
        }
        stages = {
            "W": ["W"],
            "WT": ["W", "WT-raw"],
            "B2": ["blind5", "B2-raw"],
            "subgraphrag": ["subgraphrag-first", "subgraphrag-retry"],
        }.get(name, [name])
        if name == "WT-PC":
            stages = [
                "W",
                "WT-raw",
                *[
                    j["name"]
                    for j in execution["jobs"]
                    if j["family"] == family
                    and j["kind"] == "gpu"
                    and j["name"].startswith("WT-PC-")
                ],
            ]
        seconds = sum(
            read_sealed(receipt_path(directory, j))["gpu_seconds"]
            for j in execution["jobs"]
            if j["scope"] == scope and j["family"] == family and j["name"] in stages
        )
        costs[key]["measured_launcher_seconds"] = {
            "total": seconds,
            "per_question": seconds / len(records),
            "scope": (
                "launcher wall including startup/waits; not a paired controlled-latency estimate"
            ),
        }
        summary_seconds, startup_seconds, missing_summaries = 0.0, 0.0, []
        for stage in stages:
            job = next(
                (
                    j
                    for j in execution["jobs"]
                    if j["scope"] == scope and j["family"] == family and j["name"] == stage
                ),
                None,
            )
            if job is None:
                missing_summaries.append(stage)
                continue
            receipt = read_sealed(receipt_path(directory, job))
            if receipt.get("skipped_empty_requests"):
                continue
            path = Path(job["outputs"][0] + ".summary.json")
            if not path.is_file():
                missing_summaries.append(stage)
                continue
            summary = json.loads(path.read_text())
            summary_seconds += summary["seconds"]
            startup_seconds += summary.get("startup_seconds", 0.0)
        costs[key]["measured_serving_seconds"] = {
            "workload_seconds": summary_seconds if not missing_summaries else None,
            "reported_startup_seconds": startup_seconds,
            "missing_summaries": missing_summaries,
            "scope": (
                "batched generation workload and reported startups; uncontrolled cache; "
                "no per-question latency CI"
            ),
        }
    enough = len({m["cluster"] for m in members}) >= registration["analysis"]["minimum_clusters"]
    for family in FAMILIES:

        def metric(arm, endpoint, family=family):
            table = {r["id"]: r for r in results[f"primary/{family}/{arm}"]["per_question"]}
            return [table[q][endpoint] for q in execution["ids"]]

        reference = registration.get("strongest_agents", REFERENCES)[family]
        comparisons[family] = {}
        specs = [("NI", "WT", reference, "gcr_hit", -0.03), ("SUP", "WT", reference, "gcr_hit", 0)]
        if execution["gate"]["wt_pc"]["admit"]:
            for field in ("gcr_hit", "gcr_f1", "f1"):
                if metric("WT-PC", field) != metric("WT", field):
                    raise ValueError("WT-PC full-set metric invariant violated")
            specs.append(("PC-RANK", "WT-PC", "WT", "hits_at_1", 0))
        for name, arm, control, endpoint, null in specs:
            comparison = clustered_difference(
                metric(arm, endpoint),
                metric(control, endpoint),
                members,
                10000,
                SEED,
                hypothesis={"null_difference": null, "alpha": 0.025},
            )
            comparison["passes_unadjusted"] &= enough
            comparison.update(arm=arm, reference=control, endpoint=endpoint, enough_clusters=enough)
            comparisons[family][name] = comparison
        for arm in CORE_ARMS:
            if ("primary", family, arm) not in rows:
                continue
            a = {r["id"]: usage(r) for r in rows["primary", family, arm]}
            b = {r["id"]: usage(r) for r in rows["primary", family, reference]}
            ratios = {}
            for field in ("total_tokens", "generated_tokens", "model_calls"):

                def value(row, field=field):
                    return (
                        row["prompt_tokens"] + row["generated_tokens"]
                        if field == "total_tokens"
                        else row[field]
                    )

                point, draws, _ = _paired_bootstrap(
                    [value(a[q]) for q in execution["ids"]],
                    [value(b[q]) for q in execution["ids"]],
                    members,
                    10000,
                    SEED,
                )
                ratios[field] = (
                    {
                        "ratio": float(point[0] / point[1]),
                        "low": float(np.quantile(draws[:, 0] / draws[:, 1], 0.025)),
                        "high": float(np.quantile(draws[:, 0] / draws[:, 1], 0.975)),
                    }
                    if point[1] > 0 and np.all(draws[:, 1] > 0)
                    else {"ratio": None, "reason": "zero reference cost in a bootstrap draw"}
                )
            costs[f"primary/{family}/{arm}"]["paired_ratios_vs_strongest_agent"] = ratios
    primary = intersection_holm(comparisons, ["NI"], FAMILIES, alpha=0.025)
    secondary_names = ["SUP", *(["PC-RANK"] if execution["gate"]["wt_pc"]["admit"] else [])]
    secondary = intersection_holm(comparisons, secondary_names, FAMILIES, alpha=0.025)
    for row in secondary.values():
        row["gate_reached"] = primary["NI"]["passes"]
        row["passes"] &= row["gate_reached"]
    sensitivities = {}
    for method, assignments in execution.get("cluster_sensitivities", {}).items():
        sensitivity_members = [{**m, "cluster": assignments[m["id"]]} for m in members]
        enough_sensitivity = (
            len(set(assignments.values())) >= registration["analysis"]["minimum_clusters"]
        )
        by_family = {}
        for family in FAMILIES:
            by_family[family] = {}
            for name, comparison in comparisons[family].items():
                arm, control, endpoint = (
                    comparison["arm"],
                    comparison["reference"],
                    comparison["endpoint"],
                )
                value = clustered_difference(
                    metric(arm, endpoint, family),
                    metric(control, endpoint, family),
                    sensitivity_members,
                    10000,
                    SEED,
                    hypothesis={"null_difference": -0.03 if name == "NI" else 0, "alpha": 0.025},
                )
                value.update(
                    arm=arm,
                    reference=control,
                    endpoint=endpoint,
                    enough_clusters=enough_sensitivity,
                )
                value["passes_unadjusted"] &= enough_sensitivity
                by_family[family][name] = value
        sensitivities[method] = {
            "claim_bearing": False,
            "summary": cluster_summary(assignments),
            "comparisons_by_family": by_family,
        }
    cluster_sizes = Counter(execution["cluster"].values())
    descriptive = {}
    for (scope, family, arm), records in rows.items():
        if scope not in ("baseline", "dog") or arm == "WT" or arm.startswith("agent-"):
            continue
        key = f"{scope}/{family}/{arm}"
        ids = [r["id"] for r in records]
        control_key = f"{'dog' if scope == 'dog' else 'primary'}/{family}/WT"
        a = {r["id"]: r["gcr_hit"] for r in results[control_key]["per_question"]}
        b = {r["id"]: r["gcr_hit"] for r in results[key]["per_question"]}
        subset = [m for m in members if m["id"] in set(ids)]
        descriptive[key] = {
            **clustered_difference(
                [a[m["id"]] for m in subset], [b[m["id"]] for m in subset], subset
            ),
            "claim_bearing": False,
            "contrast": "WT minus published baseline, paired subpanel only",
            "selected_ids": ids,
            "power": descriptive_power(len(ids), {q: execution["cluster"][q] for q in ids}),
        }
    report = {
        "embargo_override": embargo_override,
        "registration": identity(execution["registration"]["path"]),
        "execution": identity(directory / "execution.json"),
        "population": registration["population"],
        "selected_ids": execution["ids"],
        "gate": execution["gate"],
        "gcr_comparisons": {
            "status": (
                "INCOMPLETE"
                if incomplete_gcr
                else "INFEASIBLE_BEFORE_SEAL"
                if execution["gate"].get("gcr", {}).get("admit") is False
                or execution["gate"].get("baseline_admission", {}).get("gcr", {}).get("admit")
                is False
                else "COMPLETE"
                if all(
                    any((s, f, "gcr-native") in rows for s in ("primary", "baseline"))
                    for f in FAMILIES
                )
                else "INCOMPLETE"
            ),
            "failures": incomplete_gcr or {},
            "qualification": execution["gate"].get("gcr")
            or execution["gate"].get("baseline_admission", {}).get("gcr"),
        },
        "results": results,
        "descriptive_baseline_comparisons": descriptive,
        "secondary_depth": score_depth(registration, execution, results),
        "baseline_selected_ids": execution.get("baseline_ids"),
        "competitive_claim": "Registered strongest-agent NI on the core; published-baseline "
        "contrasts secondary/descriptive on their sealed subpanel only",
        "cost": costs,
        "comparisons_by_family": comparisons,
        "primary_intersection_holm": primary,
        "gatekept_secondary_intersection_holm": secondary,
        "cluster_sensitivity": sensitivities,
        "sampled_clusters": len({m["cluster"] for m in members}),
        "cluster_summary": {
            "sizes": dict(cluster_sizes),
            "largest_component": max(cluster_sizes.values()),
            "effective_clusters_by_size": len(execution["ids"]) ** 2
            / sum(v * v for v in cluster_sizes.values()),
        },
        "population_census": {
            "empty_gold_ids": [q["id"] for q in panel if not q["answers"]],
            "empty_graph_ids": [q["id"] for q in panel if q["empty_graph"]],
            "no_answer_in_graph_ids": [q["id"] for q in panel if not q["coverage"]["present"]],
            "no_reachable_answer_ids": [
                q["id"] for q in panel if not q["coverage"]["any_reachable"]
            ],
        },
        "released_outputs": released,
        "off_intersection_omission_census": omissions,
        "physical_gpu_ledger": ledger,
    }
    write_sealed(directory / "score.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ml = sub.add_parser("project-ml-ids", help="bounded remote ID-column projection only")
    ml.add_argument("--inventory", type=Path, required=True)
    ml.add_argument("--out", type=Path, required=True)
    ids = sub.add_parser("population", help="local RoG ID projection and explicit ID intersection")
    ids.add_argument("--rog", type=Path, nargs="+", required=True)
    ids.add_argument("--ml-projection", type=Path, required=True)
    ids.add_argument("--out", type=Path, required=True)
    prior = sub.add_parser("prior", help="recompute MINTQA development paired discordance")
    prior.add_argument("--source-root", type=Path, required=True)
    prior.add_argument("--out", type=Path, required=True)
    frozen = sub.add_parser("register", help="seal prospective rules from a clean commit")
    frozen.add_argument("--config", type=Path, required=True)
    frozen.add_argument("--out", type=Path, required=True)
    clusters = sub.add_parser("cluster-audit", help="question-only population and sample audit")
    clusters.add_argument("--registration", type=Path, required=True)
    clusters.add_argument("--questions", type=Path, nargs="+", required=True)
    clusters.add_argument("--sample-count", type=int, nargs="+", required=True)
    clusters.add_argument("--prior", type=Path)
    clusters.add_argument("--out", type=Path, required=True)
    amendment = sub.add_parser("amend", help="seal a dated prospective amendment beside its parent")
    amendment.add_argument("--registration", type=Path, required=True)
    amendment.add_argument("--protocol", type=Path, required=True)
    amendment.add_argument("--audit", type=Path, required=True)
    amendment.add_argument("--record", type=Path, required=True)
    amendment.add_argument("--out", type=Path, required=True)
    for name in ("size", "prepare", "queue", "score"):
        command = sub.add_parser(name)
        command.add_argument("--registration", type=Path, required=True)
        if name == "size":
            command.add_argument("--timings", type=Path)
            command.add_argument(
                "--baseline-timings",
                type=Path,
                help="qualified matched validation prices for the baseline tier",
            )
            command.add_argument("--gate", type=Path)
            command.add_argument("--out", type=Path)
        else:
            command.add_argument("--execution", type=Path, required=True)
        if name == "queue":
            modes = command.add_mutually_exclusive_group()
            modes.add_argument("--dry-run", action="store_true")
            modes.add_argument("--run", action="store_true")
        if name == "score":
            command.add_argument(
                "--owner-override", type=Path, help="dated owner/reason record bound to both seals"
            )
    for name in ("reader-retry", "reader-merge"):
        command = sub.add_parser(name)
        for flag in ("requests", "first", "out"):
            command.add_argument("--" + flag, type=Path, required=True)
        if name == "reader-retry":
            command.add_argument("--prompts", type=Path, required=True)
        else:
            command.add_argument("--second", type=Path, required=True)
    dog = sub.add_parser("dog-requests")
    dog.add_argument("--data", type=Path, required=True)
    dog.add_argument("--out", type=Path, required=True)
    for name in ("pc-requests", "pc-merge"):
        command = sub.add_parser(name)
        command.add_argument("--execution", type=Path, required=True)
        command.add_argument("--family", choices=FAMILIES, required=True)
    args = parser.parse_args(argv)
    if args.command == "project-ml-ids":
        sources = [
            r
            for r in json.loads(args.inventory.read_text())
            if r["repository"] == "ml1996/webqsp" and "test-" in r["path"]
        ]
        if len(sources) != 2 or any(
            r["revision"] != "54a938897b1ef1e8dbd2fddb6985fd1e688172e8" for r in sources
        ):
            raise ValueError("expected exactly the pinned ml1996 test shards")
        ids = []
        for row in sources:
            reader = IdRangeReader(row["url"], row["bytes"])
            ids.extend(parquet_ids([reader]))
            row.update(
                id_projection_transferred_bytes=reader.transferred,
                whole_file_sha256_locally_verified=False,
            )
        write_json(
            args.out,
            {
                "ids": unique_ids(ids),
                "sources": sources,
                "columns": ["id"],
                "mode": "exact HTTP 206 ranges; no complete shard read or stored",
            },
        )
    elif args.command == "population":
        ml = json.loads(args.ml_projection.read_text())
        if ml["columns"] != ["id"]:
            raise ValueError("population requires an ID-only projection")
        result = population_audit(parquet_ids(args.rog), ml["ids"])
        result["id_projection"] = identity(args.ml_projection)
        result["rog_projection_sources"] = [str(p.resolve()) for p in args.rog]
        write_json(args.out, result)
    elif args.command == "prior":
        result = development_prior(args.source_root)
        result["command"] = [
            str(sys.executable),
            "-m",
            "xgraph.webqsp_test",
            *(argv if argv is not None else sys.argv[1:]),
        ]
        result["code_commit"] = git_commit(clean=False)
        result["implementation"] = identity(__file__)
        write_json(args.out, result)
    elif args.command == "cluster-audit":
        registration = read_sealed(args.registration)
        prior = json.loads(args.prior.read_text()) if args.prior else registration["prior"]
        result = audit_question_clusters(
            [row for path in args.questions for row in read_lines(path)],
            registration["population"],
            args.sample_count,
            prior,
        )
        result.update(
            registration=identity(args.registration),
            inputs=[identity(p) for p in [*args.questions, *([args.prior] if args.prior else [])]],
            code_commit=git_commit(clean=False),
            implementation=identity(__file__),
            command=[str(sys.executable), "-I", "-m", "xgraph.webqsp_test", *sys.argv[1:]],
        )
        write_json(args.out, result)
    elif args.command == "amend":
        result = amend_registration(
            args.registration, args.out, args.protocol, args.audit, args.record
        )
        print(
            json.dumps(
                {
                    "registration": str(args.out),
                    "version": result["version"],
                    "sha256": sha256_file(args.out),
                }
            )
        )
    elif args.command == "register":
        result = register(args.config, args.out)
        print(
            json.dumps(
                {
                    "registration": str(args.out),
                    "sha256": sha256_file(args.out),
                    "status": result["status"],
                }
            )
        )
    elif args.command == "size":
        require_active_registration(args.registration)
        registration = verify_registration(args.registration, metadata_only=args.timings is None)
        timing = verify_pilot(registration, args.timings) if args.timings else None
        gate = json.loads(args.gate.read_text()) if args.gate else None
        if gate is not None:
            gate_decision(gate)
        if args.baseline_timings and not args.timings:
            raise ValueError("baseline prices need the qualified core validation pilot")
        rates = matched_prices(registration, args.baseline_timings, args.timings)
        result = size_protocol(
            registration, timing, gate, **({"baseline_rates": rates} if rates else {})
        )
        if args.out:
            write_json(args.out, result)
        print(json.dumps(result, indent=2))
    elif args.command == "prepare":
        prepare_execution(args.registration, args.execution)
    elif args.command == "queue":
        if args.run:
            if not (args.execution / "execution.json").is_file():
                raise ValueError(
                    "execution seal missing; measured pilot and recorded gate are required"
                )
            run_owner(args.registration, args.execution)
        else:
            if (args.execution / "execution.json").is_file():
                registration, execution = verify_execution(args.registration, args.execution)
            else:
                registration, execution = (
                    verify_registration(args.registration, metadata_only=True),
                    None,
                )
            preview(registration, execution)
    elif args.command == "score":
        registration = verify_registration(args.registration, metadata_only=True)
        enforce_embargo(registration, args.execution, args.owner_override)
        registration, execution = verify_execution(args.registration, args.execution)
        score_execution(
            registration, execution, args.execution, embargo_override=args.owner_override
        )
    elif args.command.startswith("reader-"):
        from xgraph.method_prompts import PromptSource
        from xgraph.subgraphrag import PROMPT_NAMES, merge_reader_records, retry_requests

        requests, first = read_lines(args.requests), read_lines(args.first)
        require_complete(first, [r["id"] for r in requests])
        if args.command == "reader-retry":
            result = retry_requests(requests, first, PromptSource.load(args.prompts, PROMPT_NAMES))
        else:
            result = merge_reader_records(requests, first, read_lines(args.second))
            require_complete(result, [r["id"] for r in requests])
        write_lines(args.out, result)
    elif args.command == "dog-requests":
        from xgraph.webqsp import dog_reader_requests

        dog_reader_requests(args.data, args.data / "requests-gcr.jsonl", args.out)
    elif args.command.startswith("pc-"):
        pc_actions(args.execution, args.family, merge=args.command == "pc-merge")


if __name__ == "__main__":
    main()
