"""Resource accounting shared by CPU stages and the accuracy-cost frontier."""

from __future__ import annotations

import resource
import sys
import time
from collections.abc import Iterable

import numpy as np

DEDICATED_GPU_SECONDS_ESTIMATOR = "reserved GPU count times measured workload elapsed seconds"
COST_DISTRIBUTIONS = ("prompt_tokens", "generated_tokens", "model_calls")
MULTI_CALL_ARMS = frozenset(
    (
        "agent",
        "cor",
        "tog",
        "dog",
        "dog-gold",
        "gcr",
        "iterative",
        "tools",
        "walk",
        "gwalk",
        "fallback",
        "resume",
    )
)


def dedicated_gpu_seconds(report: dict) -> float:
    """Read serving's reserved interval, checking its per-question conservation.

    Historical summary-only reports use the same total. When attribution or an
    estimator is declared, it must agree; device busy time is a different metric.
    """
    summary = report["summary"]
    if summary.get("gpu_seconds_estimator", DEDICATED_GPU_SECONDS_ESTIMATOR) != (
        DEDICATED_GPU_SECONDS_ESTIMATOR
    ):
        raise ValueError("dedicated GPU-seconds require the reserved serving interval")
    total = summary["gpu_seconds"]
    if (
        isinstance(total, bool)
        or not isinstance(total, (int, float))
        or not np.isfinite(total)
        or total <= 0
    ):
        raise ValueError("dedicated GPU-seconds must be finite and positive")
    rows = report.get("results", [])
    if any("gpu_seconds" in r for r in rows):
        values = [r.get("gpu_seconds") for r in rows]
        if (
            len(rows) != summary["requests"]
            or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not np.isfinite(v)
                or v < 0
                for v in values
            )
            or not np.isclose(sum(values), total, rtol=1e-9, atol=1e-12)
        ):
            raise ValueError("dedicated GPU attribution must conserve the serving interval")
    return float(total)


def model_calls(record: dict) -> int | None:
    """All model calls for a request; None when a multi-call record does not say."""
    for key in ("model_calls", "rounds", "calls"):
        if isinstance(record.get(key), int):
            return record[key]
    if isinstance(record.get("replies"), list):
        return len(record["replies"])
    if "fallback" in record:
        return 2 if record["fallback"] else 1
    return None if record.get("arm") in MULTI_CALL_ARMS else 1


def request_cost(record: dict) -> dict:
    """Whole-request usage, already summed over calls by multi-call readers.

    token_ids is a single completion, so it substitutes for generated_tokens only
    on a single-call record. Missing usage stays unknown instead of becoming zero.
    """
    calls = model_calls(record)
    generated = record.get("generated_tokens")
    if generated is None and calls == 1 and isinstance(record.get("token_ids"), list):
        generated = len(record["token_ids"])
    return {
        "prompt_tokens": record.get("prompt_tokens"),
        "generated_tokens": generated,
        "model_calls": calls,
    }


def distributions(costs: Iterable[dict]) -> dict:
    """p50/p95 with linear interpolation, and the count of known requests per metric."""
    rows = list(costs)
    result = {}
    for key in COST_DISTRIBUTIONS:
        known = [r[key] for r in rows if r.get(key) is not None]
        p50, p95 = np.percentile(known, [50, 95]) if known else (None, None)
        result[key] = {
            "count": len(known),
            "p50": float(p50) if p50 is not None else None,
            "p95": float(p95) if p95 is not None else None,
        }
    return result


def trial_interval(values: list[float], seed: int, resamples: int = 2000) -> dict:
    """Mean and a percentile bootstrap interval across whole repeated workloads.

    One trial has no uncertainty estimate. Question observations are never used as
    independent replicates of a timing measurement.
    """
    if not values or not np.isfinite(values).all():
        raise ValueError("trial intervals need finite observations")
    observed = np.asarray(values, dtype=float)
    low = high = None
    if len(values) >= 2:
        draws = np.random.default_rng(seed).choice(observed, size=(resamples, len(values)))
        low, high = map(float, np.percentile(draws.mean(axis=1), [2.5, 97.5]))
    return {
        "mean": float(observed.mean()),
        "low": low,
        "high": high,
        "trials": len(values),
        "interval": "95% percentile bootstrap over whole trials",
        "seed": seed,
        "resamples": resamples if len(values) >= 2 else 0,
    }


def dedicated_ratio(
    trials: list[dict],
    clusters: dict[str, list[str]],
    *,
    resamples: int = 100_000,
    seed: int = 0,
    threshold: float = 1.25,
) -> dict:
    """A/P ratio of per-question means, paired over trials and whole clusters.

    Each trial supplies P and A dictionaries of attributed GPU seconds by id.
    Cluster labels repeat across all paraphrases of a case. With two dimensions
    their independently resampled multiplicities multiply (crossed bootstrap).
    Complete repeat blocks are resampled once for both arms and every question.
    Memory is bounded independently of resamples; no question is an independent
    replicate of a timing trial. This uses unweighted panel means, as calibration
    selection does; population weighting needs a separately frozen estimand.
    """
    if not trials or resamples <= 0 or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("cost ratio needs trials, positive resamples and threshold")
    qids = sorted(trials[0]["P"])
    if (
        not qids
        or set(clusters) != set(qids)
        or any(set(t[arm]) != set(qids) for t in trials for arm in ("P", "A"))
    ):
        raise ValueError("cost ratio needs the same questions in both arms, trials and clusters")
    if len({t["repeat"] for t in trials}) != len(trials):
        raise ValueError("duplicate cost ratio repeat")
    dimensions = {len(clusters[q]) for q in qids}
    if len(dimensions) != 1 or not 1 <= next(iter(dimensions)) <= 2:
        raise ValueError("cost ratio needs one or two cluster labels per question")
    values = np.array([[[t[a][q] for q in qids] for a in ("P", "A")] for t in trials])
    if not np.isfinite(values).all() or (values < 0).any() or (values[:, 0] <= 0).any():
        raise ValueError("cost ratio needs finite nonnegative costs and positive P costs")
    indexed = [
        np.unique([clusters[q][d] for q in qids], return_inverse=True)
        for d in range(next(iter(dimensions)))
    ]
    p_mean, a_mean = values.mean(axis=(0, 2))
    estimate = float(a_mean / p_mean)
    result = {
        "estimate": estimate,
        "estimator": "ratio of per-question means A/P",
        "p_mean_gpu_seconds": float(p_mean),
        "a_mean_gpu_seconds": float(a_mean),
        "low": None,
        "high": None,
        "threshold": threshold,
        "primary_pass": None,
        "questions": len(qids),
        "trials": len(trials),
        "clusters_per_dimension": [len(labels) for labels, _ in indexed],
        "interval": "95% percentile paired trial x crossed cluster bootstrap",
        "seed": seed,
        "resamples": 0,
        "empty_crossed_draws": 0,
    }
    if len(trials) < 2 or any(len(labels) < 2 for labels, _ in indexed):
        return result
    rng = np.random.default_rng(seed)
    draws = []
    while len(draws) < resamples:
        count = min(256, resamples - len(draws))
        weights = np.ones((count, len(qids)))
        for labels, inverse in indexed:
            k = len(labels)
            weights *= rng.multinomial(k, np.full(k, 1 / k), size=count)[:, inverse]
        occupied = weights.sum(axis=1) > 0
        result["empty_crossed_draws"] += int((~occupied).sum())
        weights = weights[occupied]
        repeats = rng.multinomial(
            len(trials), np.full(len(trials), 1 / len(trials)), size=len(weights)
        )
        # Shared trial/cluster weights preserve arm pairing and paraphrase dependence.
        sampled = np.einsum("br,raq,bq->ba", repeats, values, weights)
        draws.extend((sampled[:, 1] / sampled[:, 0]).tolist())
    result["low"], result["high"] = map(float, np.percentile(draws, [2.5, 97.5]))
    result["primary_pass"] = result["low"] > threshold
    result["resamples"] = resamples
    return result


def weighted_dedicated_ratio(
    trials: list[dict],
    weights: dict[str, float],
    *,
    clusters: dict[str, str] | None = None,
    strata: dict[str, str] | None = None,
    resamples: int = 100_000,
    seed: int = 17,
    threshold: float = 0.5,
    resample_questions: bool = True,
) -> dict:
    """Population ratio A/P, paired across questions and complete repeat blocks.

    P is the reference agent. Survey weights include timing-subset inclusion.
    Resample questions, or whole relation sequences within hop strata, then
    recalibrate each stratum to its frozen mass. The same question and repetition
    multiplicities apply to both arms. This is separate from CF's unweighted H2.
    Batched attribution is a sensitivity; resample_questions=False instead gives
    repeat-only uncertainty for a physically measured workload ratio.
    """
    if not trials or type(resamples) is not int or resamples <= 0:
        raise ValueError("weighted cost ratio needs trials and positive resamples")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("weighted cost ratio needs a positive threshold")
    qids = sorted(trials[0]["P"])
    if not qids or any(set(t[a]) != set(qids) for t in trials for a in ("P", "A")):
        raise ValueError("weighted cost ratio needs the same questions in all arms and trials")
    if set(weights) != set(qids) or any(
        isinstance(w, bool) or not isinstance(w, (int, float)) or not np.isfinite(w) or w <= 0
        for w in weights.values()
    ):
        raise ValueError("population weights must be finite, positive and cover exactly the panel")
    if len({t["repeat"] for t in trials}) != len(trials):
        raise ValueError("duplicate weighted cost ratio repeat")
    clusters = clusters if clusters is not None else {q: q for q in qids}
    strata = strata if strata is not None else dict.fromkeys(qids, "population")
    for name, mapping in (("clusters", clusters), ("strata", strata)):
        if set(mapping) != set(qids) or any(
            not isinstance(v, str) or not v for v in mapping.values()
        ):
            raise ValueError(f"{name} must label every panel question")
    if any(len({strata[q] for q in qids if clusters[q] == c}) != 1 for c in set(clusters.values())):
        raise ValueError("a relation cluster cannot cross population strata")
    values = np.array([[[t[a][q] for q in qids] for a in ("P", "A")] for t in trials], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values[:, 0] <= 0).any():
        raise ValueError("weighted cost ratio needs finite nonnegative costs and positive P costs")
    population = np.array([weights[q] for q in qids], dtype=float)
    population /= population.sum()
    p_mean, a_mean = np.einsum("raq,q->a", values, population) / len(trials)
    result = {
        "estimate": float(a_mean / p_mean),
        "estimator": "ratio of population-weighted per-question means A/P",
        "p_mean_gpu_seconds": float(p_mean),
        "a_mean_gpu_seconds": float(a_mean),
        "questions": len(qids),
        "trials": len(trials),
        "threshold": threshold,
        "primary_pass": None,
        "low": None,
        "high": None,
        "interval": "95% percentile paired repeat x stratified question-cluster bootstrap"
        if resample_questions
        else "95% percentile paired complete-repeat bootstrap",
        "seed": seed,
        "resamples": 0,
        "minimum_gate_repeats": 3,
    }
    if len(trials) < 2:
        return result
    groups = []
    for stratum in sorted(set(strata.values())):
        indices = np.array([i for i, q in enumerate(qids) if strata[q] == stratum])
        labels, inverse = np.unique([clusters[qids[i]] for i in indices], return_inverse=True)
        groups.append((indices, len(labels), inverse))
    rng = np.random.default_rng(seed)
    draws = []
    while len(draws) < resamples:
        count = min(256, resamples - len(draws))
        sampled_weights = np.tile(population, (count, 1))
        if resample_questions:
            for indices, k, inverse in groups:
                multiplicities = rng.multinomial(k, np.full(k, 1 / k), size=count)[:, inverse]
                part = population[indices] * multiplicities
                sampled_weights[:, indices] = (
                    part * (population[indices].sum() / part.sum(axis=1))[:, None]
                )
        repeats = rng.multinomial(len(trials), np.full(len(trials), 1 / len(trials)), size=count)
        sampled = np.einsum("br,raq,bq->ba", repeats, values, sampled_weights)
        draws.extend((sampled[:, 1] / sampled[:, 0]).tolist())
    result["low"], result["high"] = map(float, np.percentile(draws, [2.5, 97.5]))
    result["resamples"] = resamples
    if len(trials) >= 3:
        result["primary_pass"] = result["high"] <= threshold
    return result


class StageTimer:
    """Time this process's stage, including loading and writing its records.

    Peak RSS is the process high-water mark, not a delta or a sum over stages. Run
    each CLI in a fresh process for an independent peak. CPU seconds exclude children.
    """

    def __init__(self) -> None:
        self.wall = time.perf_counter()
        self.cpu = time.process_time()

    def summary(self) -> dict:
        return {
            "cost_kind": "cpu_stage",
            "seconds": time.perf_counter() - self.wall,
            "cpu_seconds": time.process_time() - self.cpu,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024**2 if sys.platform == "darwin" else 1024),
            "memory_scope": "process_high_water_mark",
        }
