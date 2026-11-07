"""Goal-first PlanBench scoring with full denominators and clustered uncertainty."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from xgraph.cost import distributions
from xgraph.planbench_pddl import check_plan


def _aggregate(rows):
    ratios = [
        r["length_over_optimal"]
        for r in rows
        if r["goal_success"] and r["length_over_optimal"] is not None
    ]
    costs = {}
    for key in ("prompt_tokens", "generated_tokens", "model_calls", "seconds"):
        known = [r[key] for r in rows if r.get(key) is not None]
        costs[key] = {
            "known": len(known),
            "total": sum(known),
            "mean": float(np.mean(known)) if known else None,
        }
    return {
        "instances": len(rows),
        "missing": sum(r["missing"] for r in rows),
        "goal_success": float(np.mean([r["goal_success"] for r in rows])),
        "plan_validity": float(np.mean([r["plan_valid"] for r in rows])),
        "completion": float(np.mean([r["complete"] for r in rows])),
        "optimal_length_known": sum(r["optimal_length"] is not None for r in rows),
        "success_length_over_optimal": float(np.mean(ratios)) if ratios else None,
        "success_lengths_compared": len(ratios),
        "costs": costs,
        "cost_distributions": distributions(rows),
    }


def _comparison(rows, seed, resamples):
    def interval(mode):
        groups = defaultdict(list)
        for row in rows:
            key = row["cluster"] if mode == "problem" else (row["base_domain"], row["size"])
            groups[key].append(row)
        if len(groups) < 2:
            return len(groups), None
        strata = defaultdict(list)
        for _key, members in sorted(groups.items(), key=lambda x: str(x[0])):
            stratum = (
                (members[0]["base_domain"], members[0]["size"]) if mode == "problem" else "all"
            )
            strata[stratum].append((sum(r["difference"] for r in members), len(members)))
        rng = np.random.default_rng(seed)
        total = np.zeros(resamples)
        count = np.zeros(resamples)
        for values in strata.values():
            values = np.asarray(values, dtype=float)
            samples = rng.integers(0, len(values), (resamples, len(values)))
            total += values[samples, 0].sum(axis=1)
            count += values[samples, 1].sum(axis=1)
        return len(groups), np.percentile(total / count, [2.5, 97.5]).tolist()

    pc, problem_ci = interval("problem")
    sc, size_ci = interval("size")
    return {
        "instances": len(rows),
        "delta": float(np.mean([r["difference"] for r in rows])),
        "problem_clusters": pc,
        "problem_cluster_ci": problem_ci,
        "size_clusters": sc,
        "size_cluster_ci": size_ci,
    }


def score(cases, records, references, arms, *, seed=16, resamples=2000):
    if not cases or not arms or resamples < 1:
        raise ValueError("scoring needs cases, arms and positive resamples")
    ids = {c.id for c in cases}
    if len(ids) != len(cases) or len(set(arms)) != len(arms):
        raise ValueError("duplicate scoring cases or arms")
    indexed = {}
    for record in records:
        key = (record["arm"], record["id"])
        if key in indexed:
            raise ValueError(f"duplicate planning output: {key}")
        if key[0] not in arms or key[1] not in ids:
            raise ValueError(f"output outside frozen scoring panel: {key}")
        indexed[key] = record
    ref = {}
    for row in references:
        if row["id"] in ref:
            raise ValueError(f"duplicate reference: {row['id']}")
        if row["id"] not in ids:
            raise ValueError(f"reference outside panel: {row['id']}")
        length = row.get("optimal_length") if row.get("status") == "optimal" else None
        if length is not None and (type(length) is not int or length < 0):
            raise ValueError("invalid proven optimal length")
        ref[row["id"]] = length
    details = []
    summaries = {}
    by_arm = {}
    for arm in arms:
        rows = []
        for case in cases:
            record = indexed.get((arm, case.id))
            missing = record is None
            record = record or {}
            plan = record.get("plan", [])
            if not isinstance(plan, list) or any(not isinstance(a, str) for a in plan):
                raise ValueError(f"plan must be a list of action strings: {case.id}")
            checked = check_plan(case.problem, "\n".join(plan))
            complete = record.get("complete") is True and record.get("termination") in {
                "model",
                "model_and_planner",
                "policy_stop",
                "goal-reached-control",
            }
            cap = record.get("budget", {}).get("max_actions")
            if cap is not None and checked.length >= cap:
                complete = False
            optimum = ref.get(case.id)
            ratio = (
                (checked.length / optimum if optimum else (1.0 if checked.length == 0 else None))
                if optimum is not None
                else None
            )
            row = {
                "id": case.id,
                "domain": case.domain,
                "cluster": case.cluster,
                "base_domain": case.cluster.split(":")[0],
                "size": len(case.problem.objects),
                "arm": arm,
                "missing": missing,
                "plan_valid": not missing and checked.valid,
                "complete": not missing and complete,
                "goal_success": not missing and complete and checked.goal_reached,
                "length": checked.length if not missing else None,
                "optimal_length": optimum,
                "length_over_optimal": ratio if checked.valid and not missing else None,
                "excess_length": checked.length - optimum
                if optimum is not None and not missing
                else None,
                "termination": record.get("termination", "missing"),
            }
            for key in ("prompt_tokens", "generated_tokens", "model_calls", "seconds"):
                value = record.get(key)
                if value is not None and (
                    not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0
                ):
                    raise ValueError(f"invalid {key} for {case.id}")
                row[key] = value
            rows.append(row)
        summary = _aggregate(rows)
        summary["by_domain"] = {
            domain: _aggregate([r for r in rows if r["domain"] == domain])
            for domain in sorted({r["domain"] for r in rows})
        }
        summary["by_size"] = {
            f"{domain}:{size}": _aggregate(
                [r for r in rows if r["domain"] == domain and r["size"] == size]
            )
            for domain, size in sorted({(r["domain"], r["size"]) for r in rows})
        }
        summaries[arm], by_arm[arm] = summary, {r["id"]: r for r in rows}
        details.extend(rows)
    comparisons = {}
    if "mask" in arms:
        for arm in arms:
            if arm == "mask" or "control" in arm or arm == "llm_planner":
                continue
            rows = [
                {
                    **r,
                    "difference": int(r["goal_success"])
                    - int(by_arm[arm][r["id"]]["goal_success"]),
                }
                for r in by_arm["mask"].values()
            ]
            result = _comparison(rows, seed, resamples)
            result["by_domain"] = {
                domain: _comparison([r for r in rows if r["domain"] == domain], seed, resamples)
                for domain in sorted({r["domain"] for r in rows})
            }
            comparisons[f"mask-{arm}"] = result
    return {
        "arms": summaries,
        "per_instance": details,
        "comparisons": comparisons,
        "seed": seed,
        "resamples": resamples,
        "intervals": "paired problem-cluster bootstrap stratified by base domain and object count; "
        "separate whole-size-cluster sensitivity; no interval for one cluster",
        "denominator": "every frozen panel instance, including missing, malformed "
        "and incomplete plans",
    }
