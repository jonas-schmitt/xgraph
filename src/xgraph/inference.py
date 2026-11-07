"""Paired registered inference, shared by confirmation and assurance."""

from __future__ import annotations

import numpy as np

from xgraph.cost import dedicated_gpu_seconds

ENDPOINTS = ("H1a.question", "H1b.question", "H1b.any", "H2.question", "H2.gpu_seconds")
HYPOTHESES = ("H1a", "H1b", "H2")
COMPONENTS = ((0,), (1, 2), (3, 4))
NULLS = np.array([0.0, 0.0, -0.03, -0.03, 1.25])
ALPHA = 0.025


def decision_rule(pvalues: np.ndarray, effects: np.ndarray) -> dict:
    """Intersection-union across endpoints and families, then fixed sequence.

    Arrays have final axes (families, endpoints). Family prefixes are diagnostic;
    confirmatory advancement uses the maximum p-value across both families.
    All estimates survive a preceding failure.
    """
    p, effects = np.asarray(pvalues), np.asarray(effects)
    if p.shape != effects.shape or p.ndim < 2 or p.shape[-2:] != (2, len(ENDPOINTS)):
        raise ValueError("p-values and effects need matching (..., 2, 5) shapes")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be finite and between zero and one")
    family_p = np.stack([p[..., list(c)].max(axis=-1) for c in COMPONENTS], axis=-1)
    step_p = family_p.max(axis=-2)
    both = np.logical_and.accumulate(step_p <= ALPHA, axis=-1)
    return {
        "hypothesis_pvalues": family_p,
        "step_pvalues": step_p,
        "family_prefixes": np.logical_and.accumulate(family_p <= ALPHA, axis=-1),
        "passes": np.broadcast_to(both[..., None, :], family_p.shape).copy(),
        "both_families": both,
        "all_pass": both[..., -1],
        "effects": effects.copy(),
    }


def bootstrap_test(
    numerators: np.ndarray,
    denominators: np.ndarray,
    clusters: list[str],
    nulls: np.ndarray,
    *,
    resamples: int = 100_000,
    seed: int = 0,
    rng=None,
    batch_size: int = 256,
    crossed: list[str] | None = None,
    count_cache: dict | None = None,
    alpha: float = 0.025,
) -> dict:
    """Paired null-centred ratios of sums and finite-bootstrap inverted bounds.

    Every arm/family/paraphrase statistic shares each draw. Crossed draws multiply
    independent relation-sequence and event multiplicities. Occupied cells are
    aggregated only for multiplication; cells are never the resampling unit.
    Zero denominators permit multiple populations in one shared event bootstrap;
    redraw replicates lacking any known endpoint's denominator. Unknown columns
    stay NaN. Batches, duplicate denominators and bounded compact caches bound RAM.
    """
    n, d = np.asarray(numerators, dtype=float), np.asarray(denominators, dtype=float)
    if n.ndim != 3 or n.shape != d.shape or not len(n) or len(clusters) != len(n):
        raise ValueError("bootstrap needs nonempty matching (cases, families, endpoints) arrays")
    if resamples < 1 or batch_size < 1 or not 0 < alpha < 0.5:
        raise ValueError("positive resamples/batch_size and alpha in (0, .5) are required")
    if crossed is not None and len(crossed) != len(n):
        raise ValueError("crossed cluster labels must cover every case")
    if np.isinf(n).any() or np.isinf(d).any():
        raise ValueError("bootstrap inputs must be finite or NaN for unknown endpoints")
    rng = np.random.default_rng(seed) if rng is None else rng
    shape = n.shape[1:]
    n, d = n.reshape(len(n), -1), d.reshape(len(d), -1)
    known = np.isfinite(n).all(axis=0) & np.isfinite(d).all(axis=0)
    estimate, pvalues, critical, lower = (np.full(n.shape[1], np.nan) for _ in range(4))
    interval = np.full((n.shape[1], 2), np.nan)

    def result():
        return {
            "estimate": estimate.reshape(shape),
            "pvalues": pvalues.reshape(shape),
            "critical": critical.reshape(shape),
            "lower": lower.reshape(shape),
            "interval": interval.reshape(*shape, 2),
        }

    if not known.any():
        return result()
    n, d = n[:, known], d[:, known]
    if (d < 0).any() or (d.sum(axis=0) <= 0).any():
        raise ValueError("known bootstrap denominators must be nonnegative with positive totals")
    observed = n.sum(axis=0) / d.sum(axis=0)
    width = len(observed)
    if resamples * width * 8 > 512 * 1024**2:
        raise ValueError("bootstrap statistics exceed 512 MiB; reduce the study batch")
    unique_d, den_index = np.unique(d, axis=1, return_inverse=True)
    data = np.concatenate([n, unique_d], axis=1)
    _, group = np.unique(clusters, return_inverse=True)
    k = int(group.max()) + 1
    event, ne = None, 0
    if crossed is None:
        cell, row_group, row_event = group, None, None
        cells = k
    else:
        _, event = np.unique(crossed, return_inverse=True)
        ne = int(event.max()) + 1
        labels, cell = np.unique(np.column_stack([group, event]), axis=0, return_inverse=True)
        row_group, row_event = labels.T
        cells = len(labels)
    sums = np.zeros((cells, data.shape[1]))
    np.add.at(sums, cell, data)
    # Conditional redraws depend on the incidence and denominator support. Never
    # reuse corrected multiplicities for a different incidence or population mask.
    support = np.unique(unique_d > 0, axis=1)
    signature = (
        tuple(group),
        tuple(event) if event is not None else (),
        np.packbits(support, axis=0).tobytes(),
    )
    cache_keys = {
        dim: (dim, size, resamples, signature) for dim, size in (("main", k), ("crossed", ne))
    }
    batch_size = min(batch_size, max(1, 32 * 1024**2 // (8 * max(k + ne + cells, 1))))

    def counts(size, start, stop, dimension):
        key = cache_keys[dimension]
        dtype = np.uint16 if size <= np.iinfo(np.uint16).max else np.uint32
        if count_cache is not None and key not in count_cache:
            occupied = sum(v.nbytes for v in count_cache.values())
            if occupied + resamples * size * np.dtype(dtype).itemsize <= 1536 * 1024**2:
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

    def totals(primary, other):
        weights = primary if other is None else primary[:, row_group] * other[:, row_event]
        return weights @ sums

    deviations = np.empty((resamples, width))
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        primary = counts(k, start, stop, "main")
        other = counts(ne, start, stop, "crossed") if crossed is not None else None
        total = totals(primary, other)
        empty = (total[:, width:] <= 0).any(axis=1)
        while empty.any():
            redraw = int(empty.sum())
            primary[empty] = rng.multinomial(k, np.full(k, 1 / k), size=redraw)
            if other is not None:
                other[empty] = rng.multinomial(ne, np.full(ne, 1 / ne), size=redraw)
            total[empty] = totals(primary[empty], other[empty] if other is not None else None)
            empty = (total[:, width:] <= 0).any(axis=1)
        # Cache valid *paired* draws together. Updating only one crossed axis
        # would bias its conditional joint distribution in later studies.
        keys = [cache_keys["main"]] + ([cache_keys["crossed"]] if other is not None else [])
        if count_cache is not None and all(key in count_cache for key in keys):
            count_cache[keys[0]][start:stop] = primary
            if other is not None:
                count_cache[keys[1]][start:stop] = other
        deviations[start:stop] = total[:, :width] / total[:, width:][:, den_index] - observed
    null = np.broadcast_to(nulls, shape).ravel()[known]
    if not np.isfinite(null).all():
        raise ValueError("known endpoints need finite nulls")
    pvalues[known] = (1 + (deviations >= observed - null).sum(axis=0)) / (resamples + 1)
    estimate[known] = observed

    def upper_critical(dev, level):
        allowed = int(np.floor(level * (resamples + 1) - 1))
        if allowed < 0:
            return np.full(width, np.inf)
        return np.partition(dev, resamples - allowed - 1, axis=0)[resamples - allowed - 1]

    bound = upper_critical(deviations, alpha)
    critical[known], lower[known] = bound, observed - bound
    interval[known, 0] = observed - upper_critical(deviations, 0.025)
    interval[known, 1] = observed + upper_critical(-deviations, 0.025)
    return result()


def serving_pairs(
    trials: list[dict], *, p_arm: str, a_arm: str, cache: str, concurrency: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read measure_workload reports, paired by trial.repeat in one frozen cell.

    Failed infrastructure trials remain in the source ledger; only completed
    replacements can supply a repetition. Missing partners and duplicate completed
    partners fail rather than fabricating independent or zero-cost observations.
    """
    pairs = {}
    for report in trials:
        trial = report["trial"]
        if (trial["cache"], trial["concurrency"]) != (cache, concurrency):
            continue
        arm = trial["arm"]
        if arm not in (p_arm, a_arm):
            continue
        pair = pairs.setdefault(trial["repeat"], {})
        if report.get("status") != "completed":
            continue
        if arm in pair:
            raise ValueError(f"duplicate completed serving repetition {trial['repeat']}/{arm}")
        pair[arm] = report
    if not pairs:
        raise ValueError("no matched completed serving repetitions in the primary cell")
    a, p, questions = [], [], []
    for repeat, pair in sorted(pairs.items()):
        if set(pair) != {p_arm, a_arm}:
            raise ValueError(f"serving repetition {repeat}: missing matched A/P partner")
        first, second = pair[a_arm], pair[p_arm]
        if first["trial"].get("seed") != second["trial"].get("seed"):
            raise ValueError(f"serving repetition {repeat}: unmatched workload-order seeds")
        if "results" in first and "results" in second:
            ids = [[r["id"] for r in row["results"]] for row in (first, second)]
            if sorted(ids[0]) != sorted(ids[1]) or any(len(set(v)) != len(v) for v in ids):
                raise ValueError(f"serving repetition {repeat}: needs matched question ids")
        for key in ("study_path", "kv_cache_bytes", "filesystem_cache_state"):
            if key in first and key in second and first[key] != second[key]:
                raise ValueError(f"serving repetition {repeat}: mismatched {key}")
        capacity = [
            row.get("effective_server", {}).get("kv_capacity_tokens") for row in (first, second)
        ]
        if all(v is not None for v in capacity) and capacity[0] != capacity[1]:
            raise ValueError(f"serving repetition {repeat}: mismatched actual KV capacity")
        an, pn = first["summary"]["requests"], second["summary"]["requests"]
        if (
            any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in (an, pn))
            or an != pn
        ):
            raise ValueError(
                f"serving repetition {repeat}: matched workloads need equal questions "
                "with positive integer counts"
            )
        values = [dedicated_gpu_seconds(row) for row in (first, second)]
        a.append(values[0])
        p.append(values[1])
        questions.append(an)
    return np.array(a), np.array(p), np.array(questions)


def serving_inference(
    trials: list[dict], *, p_arm: str, a_arm: str, cache: str, concurrency: int, **kwargs
) -> dict:
    """H2 ratio of aggregate dedicated costs, resampling matched whole repetitions."""
    a, p, questions = serving_pairs(
        trials, p_arm=p_arm, a_arm=a_arm, cache=cache, concurrency=concurrency
    )
    result = bootstrap_test(
        a[:, None, None],
        p[:, None, None],
        list(map(str, range(len(a)))),
        np.array([1.25]),
        **kwargs,
    )
    return {
        **result,
        "repetitions": len(a),
        "questions": int(questions.sum()),
        "cost_per_question": {
            "A": float(a.sum() / questions.sum()),
            "P": float(p.sum() / questions.sum()),
        },
    }
