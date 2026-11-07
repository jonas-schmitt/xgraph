"""Prospective CWQ test protocol. Drafting and cost sizing never open answers."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist

from xgraph.cwq import ROOT, SOURCE_SPEC, source_manifest, verify_source
from xgraph.cwq_queue import REFERENCES
from xgraph.provenance import provenance, sha256_file
from xgraph.webqsp import _json, _read_jsonl, answer_fields, parquet_rows, score_records
from xgraph.webqsp_test import (
    git_commit,
    identity,
    positive,
    question_clusters,
    read_sealed,
    require_complete,
    select_ids,
    unique_ids,
    verify_identity,
    write_sealed,
)

FREEZE_DATE = "2026-11-23"
NOT_BEFORE = "2026-11-24T00:00:00+00:00"

ADMISSION_RULES = {
    "version": 2,
    "cluster_method": "source_family_primary_topic",
    "cluster_fallback": "source_family",
    "cluster_selection": "use the union unless either full validation or test question "
    "census fails concentration; then use source_family for both, before sampling",
    "cluster_sensitivity": "transitive_source_family; disclosed, never gating",
    "minimum_clusters": 20,
    "minimum_effective_clusters": 20,
    "maximum_cluster_share": 0.10,
    "assurance_method": "bayesian-cluster-normal-v1",
    "trials": 10000,
    "seed": 17,
    "target_joint": 0.8,
    "true_difference": 0.0,
    "NI_margin": -0.03,
    "alpha": 0.025,
    "pattern_prior_mass": 0.5,
    "mc_lower_alpha": 0.025,
    "pre_freeze_addition_cap_gpu_hours": 16,
    "pre_freeze_additions_gpu_hours": 15,
    "post_freeze_cap_gpu_hours": 114,
    "webqsp_core_cap_gpu_hours": 40,
    "decision_deadline": "2026-11-13",
}


def source_family(q):
    """An operational released-ID family, not a verified native parent annotation."""
    match = re.fullmatch(r"(WebQ(?:Test|Trn)-[0-9]+)_[0-9A-Za-z]+", q)
    return match[1] if match else q


def cluster_summary(assignments):
    """The WebQSP cluster audit's size concentration diagnostic (no outcomes)."""
    sizes = Counter(assignments.values())
    if not sizes:
        raise ValueError("empty question cluster population")
    n = len(assignments)
    return {
        "questions": n,
        "clusters": len(sizes),
        "largest_component": max(sizes.values()),
        "largest_share": max(sizes.values()) / n,
        "effective_clusters": n * n / sum(v * v for v in sizes.values()),
    }


def concentration_passes(summary):
    """Apply the registered concentration limits to a question-only summary."""
    return (
        summary["clusters"] >= ADMISSION_RULES["minimum_clusters"]
        and summary["effective_clusters"] >= ADMISSION_RULES["minimum_effective_clusters"]
        and summary["largest_share"] <= ADMISSION_RULES["maximum_cluster_share"]
    )


def _family_clusters(rows, *, primary_topic):
    parent = {r["id"]: r["id"] for r in rows}

    def root(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    seen = {}
    for row in rows:
        q = row["id"]
        keys = [("family", source_family(q))]
        if primary_topic and row["q_entity"]:
            keys.append(("topic", str(row["q_entity"][0])))
        for key in keys:
            if key in seen:
                a, b = sorted((root(q), root(seen[key])))
                parent[b] = a
            seen[key] = q
    return {q: root(q) for q in sorted(parent)}


def audit_clusters(rows, selected_ids, *, method=None):
    """Prefer the source-family/primary-topic union on the complete census.

    Source family is the prospective fallback when concentration fails. The
    broader transitive unit is disclosed as a sensitivity and never gates.
    An explicit method applies the common selection from both split censuses.
    """
    rows = list(rows)
    if any(set(r) != {"id", "question", "q_entity"} for r in rows):
        raise ValueError("CWQ cluster audit requires question-only id/question/q_entity")
    ids = unique_ids(r["id"] for r in rows)
    selected = unique_ids(selected_ids)
    if not set(selected) <= set(ids):
        raise ValueError("cluster selection outside full question census")
    preferred = _family_clusters(rows, primary_topic=True)
    preferred_summary = cluster_summary(preferred)
    if method is None:
        method = (
            ADMISSION_RULES["cluster_method"]
            if concentration_passes(preferred_summary)
            else ADMISSION_RULES["cluster_fallback"]
        )
    if method not in (ADMISSION_RULES["cluster_method"], ADMISSION_RULES["cluster_fallback"]):
        raise ValueError(f"unknown CWQ dependence unit: {method}")
    mapping = (
        preferred
        if method == "source_family_primary_topic"
        else _family_clusters(rows, primary_topic=False)
    )
    # Namespaces keep ID families distinct from literal topic labels.
    augmented = [
        {
            **r,
            "q_entity": [
                *("topic:" + str(e) for e in r["q_entity"]),
                "family:" + source_family(r["id"]),
            ],
        }
        for r in rows
    ]
    sensitivity = question_clusters(augmented)
    summary = cluster_summary({q: mapping[q] for q in selected})
    return {
        "columns": ["id", "question", "q_entity"],
        "gold_loaded": False,
        "method": method,
        "assignments": {q: mapping[q] for q in selected},
        "summary": summary,
        "census_summary": cluster_summary(mapping),
        "admissible": concentration_passes(summary),
        "preferred": {
            "method": "source_family_primary_topic",
            "census_summary": preferred_summary,
            "admissible": concentration_passes(preferred_summary),
        },
        "sensitivity": {
            "method": "transitive_source_family",
            "gating": False,
            "assignments": {q: sensitivity[q] for q in selected},
            "summary": cluster_summary({q: sensitivity[q] for q in selected}),
            "census_summary": cluster_summary(sensitivity),
        },
        "diagnostics": {
            **{
                method: cluster_summary({q: mapping[q] for q in selected})
                for method in ("transitive", "primary_topic", "exact_question")
                for mapping in [question_clusters(rows, method)]
            },
            "source_family": cluster_summary({q: source_family(q) for q in selected}),
        },
        "limits": "ID families are operational ancestry proxies. Topic/ancestry "
        "overlap is retained; no independent-KG or unseen-parent claim.",
    }


def population_audits(source, pilot_ids):
    """Choose one unit for both splits using only their complete question censuses."""
    spec = source_manifest()
    rows = {}
    for split in ("test", "validation"):
        verify_source(source, split)
        rows[split] = list(
            parquet_rows(
                source, split, sources=spec["files"], columns=["id", "question", "q_entity"]
            )
        )
        if len(unique_ids(r["id"] for r in rows[split])) != spec["populations"][split]:
            raise ValueError(f"CWQ {split} question census differs from pinned release")
    selected = {"test": [r["id"] for r in rows["test"]], "validation": pilot_ids}
    audits = {s: audit_clusters(rows[s], selected[s]) for s in rows}
    failed = sorted(s for s, a in audits.items() if not a["preferred"]["admissible"])
    method = ADMISSION_RULES["cluster_fallback" if failed else "cluster_method"]
    for split, audit in audits.items():
        if audit["method"] != method:
            audit = audits[split] = audit_clusters(rows[split], selected[split], method=method)
        audit["selection"] = {
            "method": method,
            "rule": ADMISSION_RULES["cluster_selection"],
            "fallback_triggered_by": failed,
        }
    return {"test": audits["test"], "validation_pilot": audits["validation"]}


def joint_assurance(vectors, pilot_clusters, target_clusters):
    """Seeded posterior predictive joint NI assurance at zero true advantage.

    A Bayesian bootstrap on paired cluster means integrates pilot variance
    uncertainty. Nine possible paired discordance patterns receive Jeffreys
    half-counts, so an all-agreement pilot cannot imply known zero variance.
    Fresh-target covariance is the drawn cluster-mean covariance times the sum
    of squared target cluster weights. The joint normal predictive draw uses
    the same two-family covariance, never a product of marginal powers.

    This is a normal approximation to the registered paired cluster bootstrap,
    conditional on exchangeable cluster means independent of cluster size. The
    concentration gate is mandatory; there is no finite-population correction.
    """
    import numpy as np

    ids = sorted(vectors)
    if not ids or set(ids) != set(pilot_clusters):
        raise ValueError("paired pilot vectors must cover exactly the pilot clusters")
    values = np.asarray([vectors[q] for q in ids], dtype=float)
    if values.shape != (len(ids), 2) or not np.isin(values, [-1, 0, 1]).all():
        raise ValueError("two finite paired ranked-hit differences required per pilot ID")
    summaries = {
        "pilot": cluster_summary(pilot_clusters),
        "target": cluster_summary(target_clusters),
    }
    for summary in summaries.values():
        if not concentration_passes(summary):
            return {"status": "FAIL_DEPENDENCE", "passes": False, "cluster_diagnostics": summaries}
    groups = sorted(set(pilot_clusters.values()))
    means = np.array([values[[pilot_clusters[q] == c for q in ids]].mean(axis=0) for c in groups])
    patterns = np.array([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)], dtype=float)
    points = np.concatenate([means, patterns])
    masses = np.array([1.0] * len(means) + [ADMISSION_RULES["pattern_prior_mass"]] * 9)
    moments = np.column_stack([points[:, 0] ** 2, points[:, 1] ** 2, points[:, 0] * points[:, 1]])
    trials = ADMISSION_RULES["trials"]
    rng = np.random.default_rng(ADMISSION_RULES["seed"])
    z = NormalDist().inv_cdf(1 - ADMISSION_RULES["alpha"])
    h = 1 / summaries["target"]["effective_clusters"]
    successes, marginal = 0, np.zeros(2, dtype=int)
    for start in range(0, trials, 64):
        count = min(64, trials - start)
        weights = rng.dirichlet(masses, size=count)
        center, second = weights @ points, weights @ moments
        variance = np.maximum(second[:, :2] - center**2, 0)
        covariance = second[:, 2] - center[:, 0] * center[:, 1]
        noise = rng.standard_normal((count, 2))
        sd = np.sqrt(variance)
        prediction = np.column_stack(
            [
                sd[:, 0] * noise[:, 0],
                covariance / sd[:, 0] * noise[:, 0]
                + np.sqrt(np.maximum(variance[:, 1] - covariance**2 / variance[:, 0], 0))
                * noise[:, 1],
            ]
        ) * math.sqrt(h)
        passes = prediction - z * sd * math.sqrt(h) > ADMISSION_RULES["NI_margin"]
        successes += int(passes.all(axis=1).sum())
        marginal += passes.sum(axis=0)
    probability = successes / trials
    z_mc = NormalDist().inv_cdf(1 - ADMISSION_RULES["mc_lower_alpha"])
    lower = (
        probability
        + z_mc**2 / (2 * trials)
        - z_mc * math.sqrt(probability * (1 - probability) / trials + z_mc**2 / (4 * trials**2))
    ) / (1 + z_mc**2 / trials)
    return {
        "status": "COMPLETE",
        "method": ADMISSION_RULES["assurance_method"],
        "seed": ADMISSION_RULES["seed"],
        "trials": trials,
        "true_difference": 0.0,
        "joint": probability,
        "mc_lower": lower,
        "marginal": dict(zip(REFERENCES, (marginal / trials).tolist(), strict=True)),
        "passes": lower >= ADMISSION_RULES["target_joint"],
        "pilot_covariance": np.cov(means.T, ddof=1).tolist(),
        "cluster_diagnostics": summaries,
        "limits": "posterior predictive cluster-normal approximation, not a population "
        "guarantee or exact finite-sample power of the cluster-bootstrap test; "
        "cluster means assumed exchangeable independently of size; MC bound "
        "quantifies simulation error only",
    }


def admission_decision(draft, timings, score, checks):
    """Decide the registered breadth study from complete validation evidence only."""
    if draft.get("admission_rules") != ADMISSION_RULES:
        raise ValueError("prospectively sealed CWQ admission rules required")
    if score is None or checks is None:
        raise ValueError("complete pilot scores and sealed priority/exposure checks required")
    size = size_from_cost(draft, timings)
    seal = draft["pilot"]["sha256"]
    if (
        timings["pilot_sha256"] != seal
        or score["pilot_sha256"] != seal
        or score["status"] != "COMPLETE"
        or timings["budget_seconds"] != 10800
    ):
        raise ValueError("complete scores/timings must bind the amended pilot seal")
    spent = timings["gpu_seconds"]
    if type(spent) not in (int, float) or not math.isfinite(spent) or not 0 <= spent <= 10800:
        raise ValueError("pilot timing ledger is inconsistent or over cap")
    if timings.get("budget_accounting") == "container_occupancy":
        charges = [r["gpu_seconds"] for r in timings["per_arm"]]
        reconciled = all(
            type(charge) in (int, float)
            and math.isfinite(charge)
            and 0 <= charge <= row["wall_seconds"]
            and (row.get("kind") != "cpu" or charge == 0)
            for charge, row in zip(charges, timings["per_arm"], strict=True)
        ) and math.isclose(sum(charges), spent, abs_tol=1e-6)
    else:
        reconciled = timings.get("budget_accounting") is None and (
            sum(r["wall_seconds"] for r in timings["per_arm"]) - 1e-6 <= spent
        )
    if not reconciled:
        raise ValueError("pilot timing ledger is inconsistent or over cap")
    ids = unique_ids(draft["pilot_ids"])
    if len(ids) != 128 or set(draft["pilot_clusters"]) != set(ids):
        raise ValueError("complete registered pilot IDs and clusters required")
    expected = {f"{f}/{arm}" for f, ref in REFERENCES.items() for arm in ("WT", "W", "B2", ref)}
    if set(score["scores"]) != expected:
        raise ValueError("INCOMPLETE pilot family/arm score matrix")
    hits = {}
    for key, value in score["scores"].items():
        rows = value["per_question"]
        if set(unique_ids(r["id"] for r in rows)) != set(ids):
            raise ValueError("INCOMPLETE pilot score IDs")
        hits[key] = {r["id"]: r["hits_at_1"] for r in rows}
        if any(type(v) not in (int, float) or v not in (0, 1) for v in hits[key].values()):
            raise ValueError("ranked-hit scores must be binary")
    if set(draft["clusters"]) != set(unique_ids(draft["population_ids"])):
        raise ValueError("test clusters must cover the complete prospective population")
    audits = draft["cluster_audits"]
    failed = sorted(
        "validation" if name == "validation_pilot" else name
        for name in ("test", "validation_pilot")
        if not concentration_passes(audits[name]["preferred"]["census_summary"])
    )
    method = ADMISSION_RULES["cluster_fallback" if failed else "cluster_method"]
    selection = {
        "method": method,
        "rule": ADMISSION_RULES["cluster_selection"],
        "fallback_triggered_by": failed,
    }
    for name, assignments in (
        ("test", draft["clusters"]),
        ("validation_pilot", draft["pilot_clusters"]),
    ):
        audit = audits[name]
        if (
            draft["cluster_rule"] != method
            or audit["method"] != method
            or audit["selection"] != selection
            or audit["assignments"] != assignments
        ):
            raise ValueError("registered CWQ dependence selection/assignments changed")
    selected = select_ids(draft["population_ids"], size["questions_per_arm"])
    vectors = {
        q: [hits[f"{f}/WT"][q] - hits[f"{f}/{ref}"][q] for f, ref in REFERENCES.items()]
        for q in ids
    }
    assurance = joint_assurance(
        vectors, draft["pilot_clusters"], {q: draft["clusters"][q] for q in selected}
    )
    reader = draft["published_reader"]
    baseline = (
        reader["included"] is False
        and reader["validation_retrieval"] is None
        and bool(reader["reason"])
        and not size["unpriced_stages"]
    )
    priority, exposure = checks["priority"], checks["exposure"]
    if (
        exposure["inventory_complete"] is not True
        or exposure["scope"] != "project CWQ generations and tuning inputs"
    ):
        raise ValueError("complete project CWQ ID-only exposure inventory required")
    seen = set(unique_ids(exposure["seen_ids"]))
    families = set(unique_ids(exposure["seen_source_families"]))
    overlaps = sorted(set(draft["population_ids"]) & seen)
    parent_overlap = sorted({source_family(q) for q in selected} & families)
    names = (
        "total_gpu_hours",
        "spent_gpu_hours",
        "webqsp_reserved_gpu_hours",
        "novelty_required_gpu_hours",
        "novelty_reserved_gpu_hours",
        "other_reserved_gpu_hours",
    )
    for name in names:
        value = priority[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"finite nonnegative priority budget required: {name}")
    reserved = (
        priority["spent_gpu_hours"]
        + priority["webqsp_reserved_gpu_hours"]
        + priority["novelty_reserved_gpu_hours"]
        + priority["other_reserved_gpu_hours"]
        + draft["pricing"]["cap_seconds"] / 3600
    )
    priority_passes = (
        priority["total_gpu_hours"] <= ADMISSION_RULES["post_freeze_cap_gpu_hours"]
        and reserved <= priority["total_gpu_hours"]
        and priority["webqsp_reserved_gpu_hours"] >= ADMISSION_RULES["webqsp_core_cap_gpu_hours"]
        and type(priority["webqsp_core_before"]) is int
        and priority["webqsp_core_before"] > 0
        and priority["webqsp_core_after"] == priority["webqsp_core_before"]
        and priority["novelty_admission_final"] is True
        and priority["novelty_reserved_gpu_hours"] >= priority["novelty_required_gpu_hours"]
        and priority["recorded_at"][:10] <= ADMISSION_RULES["decision_deadline"]
    )
    # Validate dates as dates rather than accepting arbitrary lexically small text.
    for evidence in (priority, exposure):
        datetime.fromisoformat(evidence["recorded_at"])
    gates = {
        "assurance": assurance["passes"],
        "baseline": baseline,
        "priority": priority_passes,
        "exposure": not overlaps
        and exposure["recorded_at"][:10] <= ADMISSION_RULES["decision_deadline"],
    }
    passes = all(gates.values())
    return {
        "status": "PASS" if passes else "FAIL",
        "admitted": passes,
        "decision_scope": "CWQ breadth admission only; dated test freeze and "
        "execution seal still required",
        "failed_gates": [k for k, v in gates.items() if not v],
        "gates": gates,
        "cost": size,
        "assurance": assurance,
        "dependence_unit": selection,
        "dependence_sensitivity": {
            "method": "transitive_source_family",
            "gating": False,
            **{name: audit["sensitivity"]["summary"] for name, audit in audits.items()},
            "assurance": joint_assurance(
                vectors,
                audits["validation_pilot"]["sensitivity"]["assignments"],
                {q: audits["test"]["sensitivity"]["assignments"][q] for q in selected},
            ),
        },
        "priority": {
            "total_reserved_gpu_hours": reserved,
            "total_gpu_hours": priority["total_gpu_hours"],
        },
        "exposure": {"overlapping_ids": overlaps, "overlapping_source_families": parent_overlap},
        "claim_boundary": "CWQ complex-question breadth on shared Freebase and "
        "WebQuestions ancestry; "
        "SubgraphRAG excluded because its pinned release has no validation retrieval. "
        "No published-reader/SOTA, independent-KG, unseen-parent or superiority claim; "
        "cannot rescue failed WebQSP transfer.",
    }


def size_from_cost(draft: dict, timings: dict) -> dict:
    """Complete unique measured stage rates only; no outcome-dependent sizing."""
    if timings.get("complete") is not True:
        raise ValueError("complete pilot timings required; incomplete matrix cannot size test")
    cap, reserve = draft["pricing"]["cap_seconds"], draft["pricing"]["reserve_seconds"]
    if not 0 <= reserve < cap <= 22 * 3600:
        raise ValueError("CWQ test budget must not exceed 22 GPU-hours")
    measured = {}
    for row in timings["per_arm"]:
        key = row["family"], row["name"]
        if key in measured:
            raise ValueError("duplicate pilot stage timing")
        if row["returncode"] != 0 or row["questions"] != draft["pilot_questions"]:
            raise ValueError("failed or changed pilot stage")
        measured[key] = positive(row["wall_seconds"], "stage wall") / row["questions"]
    needed = [tuple(k) for k in draft["pilot_stages"]]
    if len(set(needed)) != len(needed) or set(needed) != measured.keys():
        raise ValueError("complete unique pilot stage matrix required")
    rate = sum(measured[k] for k in needed)
    n = min(len(unique_ids(draft["population_ids"])), math.floor((cap - reserve) / rate))
    if n < 1:
        raise ValueError("no complete cohort fits CWQ cap")
    return {
        "status": "COST_ONLY",
        "outcomes_used": False,
        "admitted": False,
        "questions_per_arm": n,
        "seconds_per_question": rate,
        "cap_seconds": cap,
        "reserve_seconds": reserve,
        "projected_seconds": n * rate + reserve,
        "rule": (
            "same hash prefix in all cells; full population preferred; startup in pilot"
            " rate plus reserve"
        ),
        "remaining_gates": [
            "roughly 80% joint clustered assurance at zero WT advantage",
            "baseline alignment and exposure audit",
            "no higher-priority work displaced",
        ],
        "unpriced_stages": draft.get("unpriced_stages", []),
        "complete_test_cost": not draft.get("unpriced_stages"),
    }


def draft_registration(source: Path, pilot_path: Path, out: Path) -> dict:
    spec = source_manifest()
    if spec["subgraphrag"]["validation_retrieval"] is not None:
        raise ValueError("new validation release requires a prospective pilot/baseline amendment")
    paths = [*verify_source(source, "test"), *verify_source(source, "validation")]
    pilot = read_sealed(pilot_path)
    audits = population_audits(source, pilot["ids"])
    test_audit, pilot_audit = audits["test"], audits["validation_pilot"]
    clusters = test_audit["assignments"]
    ids = unique_ids(clusters)
    result = {
        "status": "DRAFT_NOT_FROZEN_NOT_ADMITTED",
        "dataset": "cwq",
        "source_spec": identity(SOURCE_SPEC),
        "revision": spec["revision"],
        "sources": [identity(p) for p in paths],
        "population": (
            "entire pinned RoG-CWQ test ID population; no answer/support/operation filter"
        ),
        "population_ids": sorted(ids),
        "population_columns": ["id", "question", "q_entity"],
        "gold_loaded": False,
        "admission_rules": ADMISSION_RULES,
        "sample": {
            "seed": 17,
            "hash": "SHA-256(UTF-8('17:' + id)), ascending bytes; ID tie-break",
            "hash_order": select_ids(ids, len(ids)),
            "count": None,
            "rule": "choose count from measured cost before freeze, never test outcomes",
        },
        "clusters": clusters,
        "cluster_rule": test_audit["method"],
        "cluster_summary": test_audit["summary"],
        "cluster_audits": audits,
        "source_parent_lineage": (
            "released ID prefix before underscore retained as operational source family; "
            "native parent annotation unavailable; shared ancestry is not independent breadth"
        ),
        "references": REFERENCES,
        "core_arms": [
            {"family": f, "arm": a} for f in REFERENCES for a in ("WT", "W", "B2", REFERENCES[f])
        ],
        "published_reader": {
            **spec["subgraphrag"],
            "top_k": 100,
            "cost": "retriever runtime unavailable, never imputed as zero",
            "included": False,
            "reason": "pinned CWQ release provides test-only retrieval; no validation arm",
            "admission": "excluded prospectively; no comparison with SubgraphRAG "
            "or published-reader claim",
        },
        "pilot": identity(pilot_path),
        "pilot_questions": 128,
        "pilot_ids": pilot["ids"],
        "pilot_clusters": pilot_audit["assignments"],
        "pilot_stages": [[j["family"], j["name"]] for j in pilot["jobs"]],
        "unpriced_stages": [],
        "pricing": {
            "cap_seconds": 22 * 3600,
            "reserve_seconds": 4 * 3600,
            "basis": (
                "complete pilot stage launcher wall, CPU transformations and startup; "
                "all attempts count"
            ),
        },
        "analysis": {
            "primary": "hits_at_1",
            "NI_margin": -0.03,
            "alpha": 0.025,
            "hypothesis": "WT minus preselected strongest agent > -0.03 in EACH family",
            "joint": "intersection-union; both one-sided component tests must pass",
            "resamples": 10000,
            "seed": 17,
            "minimum_clusters": 20,
            "bootstrap": (
                "paired CWQ dependence-unit bootstrap shared across arms; uniform "
                "question weights; no answer filtering"
            ),
            "dependence_sensitivity": {
                "method": "transitive_source_family",
                "gating": False,
                "rule": "report concentration, assurance and paired NI separately; "
                "never gates admission or primary NI",
            },
            "secondary": [
                "macro set F1",
                "exact set",
                "released RoG/GCR metrics",
                "whole-policy cost",
            ],
            "superiority": (
                "not claim-bearing until a multiplicity amendment is frozen before Nov 23"
            ),
            "missing": (
                "missing input/graph/start/retrieval retained; failed generation scores"
                " zero; incomplete matrix forbids NI"
            ),
        },
        "assurance": {
            "target_joint": 0.8,
            "primary_true_difference": 0.0,
            "status": "RULES_REGISTERED_PILOT_PENDING",
            "procedure": ADMISSION_RULES["assurance_method"],
            "implementation": "xgraph.cwq_test.joint_assurance",
            "rule": "joint simulation Monte Carlo lower bound >= target_joint "
            "at zero true advantage",
            "positive_effect_scenarios": (
                "disclosed sensitivity only; cannot determine sample count"
            ),
        },
        "admission_date": "2026-11-13",
        "freeze_date": FREEZE_DATE,
        "not_before": NOT_BEFORE,
        "incomplete_rule": (
            "ANY missing arm/family/question, invalid receipt, failed accounting or "
            "budget interruption is INCOMPLETE; no subset parity claim"
        ),
        "claim_boundary": (
            "complex-question stress test on shared Freebase and WebQuestions ancestry;"
            " SubgraphRAG excluded; no published-reader/SOTA, independent-KG "
            "or unseen-parent claim;"
            " cannot rescue failed WebQSP transfer"
        ),
        "blockers": [
            "owner-run validation pilot and cost sizing",
            "pilot-based joint clustered assurance",
            "complete project CWQ ID-only exposure inventory and priority budget snapshot",
            "models/prompts/inputs/jobs/timeouts frozen on Nov 23",
        ],
        "provenance": provenance([SOURCE_SPEC, pilot_path, *paths]),
    }
    write_sealed(out, result)
    return result


def compare_complete(
    rows: dict, panel: list[dict], clusters: dict, *, resamples=10000, sensitivity_clusters=None
) -> dict:
    """Shared exact scoring and paired clustered NI; also usable on synthetic data."""
    from xgraph.mintqa_rematch import clustered_difference

    ids = unique_ids(q["id"] for q in panel)
    if set(clusters) != set(ids):
        raise ValueError("clusters must cover the complete selected population")
    scores, comparisons = {}, {}
    for key, records in rows.items():
        if set(unique_ids(r["id"] for r in records)) != set(ids):
            raise ValueError("INCOMPLETE question matrix")
        scores[key] = score_records(records, panel, failed_zero=True)
    for family, reference in REFERENCES.items():
        for arm in ("WT", reference):
            key = family, arm
            if key not in rows:
                raise ValueError("INCOMPLETE family/arm matrix")
        a = {r["id"]: r["hits_at_1"] for r in scores[family, "WT"]["per_question"]}
        b = {r["id"]: r["hits_at_1"] for r in scores[family, reference]["per_question"]}
        members = [{"id": q, "cluster": clusters[q], "hops": 1, "weight": 1.0} for q in ids]
        comparison = clustered_difference(
            [a[q] for q in ids],
            [b[q] for q in ids],
            members,
            resamples,
            17,
            hypothesis={"null_difference": -0.03, "alpha": 0.025},
        )
        comparison.update(
            method="paired CWQ question-component bootstrap",
            reference=reference,
            endpoint="hits_at_1",
            enough_clusters=len(set(clusters.values())) >= 20,
        )
        comparison["passes_unadjusted"] &= comparison["enough_clusters"]
        comparisons[family] = comparison
    result = {
        "status": "COMPLETE",
        "comparisons": comparisons,
        "joint_NI_passes": all(r["passes_unadjusted"] for r in comparisons.values()),
        "scores": {f"{f}/{a}": value for (f, a), value in scores.items()},
    }
    if sensitivity_clusters is not None:
        sensitivity = compare_complete(rows, panel, sensitivity_clusters, resamples=resamples)
        sensitivity.pop("scores")  # Per-question scores are independent of the unit.
        result["dependence_sensitivity"] = {
            "method": "transitive_source_family",
            "gating": False,
            "summary": cluster_summary(sensitivity_clusters),
            **sensitivity,
        }
    return result


def score_registered(registration_path: Path, execution_path: Path, out: Path) -> dict:
    """All freeze, receipt and completeness checks precede ANY test-answer access."""
    registration = read_sealed(registration_path)
    if registration.get("status") != "FROZEN_ADMITTED":
        raise ValueError("CWQ test scorer requires an admitted frozen registration")
    now = datetime.now(UTC)
    frozen = datetime.fromisoformat(registration["frozen_at"])
    if frozen.date().isoformat() != FREEZE_DATE or now < datetime.fromisoformat(NOT_BEFORE):
        raise ValueError("CWQ test gold remains sealed until after the Nov 23 freeze")
    if git_commit(ROOT) != registration["commit"]:
        raise ValueError("registered scorer commit changed")
    verify_identity(registration["source_spec"])
    for pin in registration["sources"]:
        verify_identity(pin)
    execution = read_sealed(execution_path)
    if (
        execution["registration_sha256"] != sha256_file(registration_path)
        or execution["complete"] is not True
    ):
        raise ValueError("INCOMPLETE or mismatched execution receipt")
    if (
        not 0
        <= execution["reserved_seconds"]
        <= registration["pricing"]["cap_seconds"]
        <= 22 * 3600
    ):
        raise ValueError("invalid or exhausted CWQ reserved-time ledger")
    ids = select_ids(registration["population_ids"], registration["sample"]["count"])
    rows = {}
    expected = {(v["family"], v["arm"]) for v in registration["core_arms"]}
    for arm in execution["arms"]:
        key = arm["family"], arm["arm"]
        if key in rows or key not in expected:
            raise ValueError("duplicate or foreign execution arm")
        receipt = read_sealed(Path(arm["receipt"]))
        if (
            receipt["registration_sha256"] != sha256_file(registration_path)
            or receipt["returncode"] != 0
        ):
            raise ValueError("invalid CWQ arm receipt")
        verify_identity(receipt["output"])
        records = _read_jsonl(Path(receipt["output"]["path"]))
        require_complete(records, ids)
        rows[key] = records
    if set(rows) != expected:
        raise ValueError("INCOMPLETE CWQ arm matrix; test answers remain sealed")
    # This is the single real-test gold read in the CWQ tooling.
    wanted = set(ids)
    panel = [
        {"id": r["id"], **answer_fields(r, coverage=False)}
        for path in registration["sources"]
        for r in _project_gold(Path(path["path"]))
        if r["id"] in wanted
    ]
    if set(unique_ids(r["id"] for r in panel)) != wanted:
        raise ValueError("registered gold IDs do not match population")
    sensitivity = registration["cluster_audits"]["test"]["sensitivity"]["assignments"]
    result = compare_complete(
        rows,
        panel,
        {q: registration["clusters"][q] for q in ids},
        sensitivity_clusters={q: sensitivity[q] for q in ids},
    )
    result["provenance"] = provenance([registration_path, execution_path])
    _json(out, result)
    return result


def _project_gold(path):
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=16, columns=["id", "answer"], use_threads=False
    ):
        yield from batch.to_pylist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("draft")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("size")
    p.add_argument("--draft", type=Path, required=True)
    p.add_argument("--timings", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("score")
    p.add_argument("--registration", type=Path, required=True)
    p.add_argument("--execution", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "draft":
        draft_registration(args.source, args.pilot, args.out)
    elif args.command == "size":
        result = size_from_cost(read_sealed(args.draft), json.loads(args.timings.read_text()))
        result["provenance"] = provenance([args.draft, args.timings])
        _json(args.out, result)
    else:
        score_registered(args.registration, args.execution, args.out)


if __name__ == "__main__":
    main()
