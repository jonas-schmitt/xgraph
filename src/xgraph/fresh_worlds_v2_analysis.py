"""Whole-block interaction inference and joint prospective pilot assurance."""

import math
from collections import Counter

import numpy as np

from xgraph.fresh_worlds_analysis import _bootstrap_differences, _matrix, _mc_lower
from xgraph.fresh_worlds_v2 import CELLS

FAMILIES = ("qwen38", "gemma31")


def block_matrix(study, scores):
    matrix, _ = _matrix(scores)
    worlds = study["worlds"]
    ids = [w["id"] for w in worlds]
    if ids != [w["id"] for w in scores["qwen38:WT"]["per_world"]]:
        raise ValueError("v2 scores do not align with the registered study")
    blocks = {}
    for i, world in enumerate(worlds):
        block = blocks.setdefault(world["block"], {})
        cell = tuple(world["cell"])
        if cell in block or cell not in CELLS:
            raise ValueError("duplicate or unknown v2 block cell")
        block[cell] = i
    if not blocks or any(set(block) != set(CELLS) for block in blocks.values()):
        raise ValueError("v2 inference requires every cell of every paired block")
    indices = np.asarray([[block[c] for c in CELLS] for block in blocks.values()])
    templates = []
    for row in indices:
        names = {worlds[i]["template"] for i in row}
        if len(names) != 1:
            raise ValueError("each v2 block must share one template")
        templates.append(names.pop())
    return matrix[indices], np.asarray(templates)


def effects(matrix):
    delta = matrix[:, :, [0, 3]] - matrix[:, :, [1, 4]]
    hard = delta[:, 3]
    return hard - delta[:, 0], delta


def _loto(values, templates):
    names = sorted(set(templates))
    if len(names) < 2:
        raise ValueError("v2 inference needs multiple templates")
    return np.asarray([values[templates != name].mean(axis=0) for name in names])


def _rare_discordance_bound(count, n, margin=0.03):
    """Conservative Chernoff bound for independent heterogeneous blocks.

    Bounding both directional discordance rates below margin suffices for
    equivalence. Jensen bounds the Poisson-binomial MGF by the binomial MGF.
    No discordance yields (1-margin)**n, rather than a zero-width claim.
    """
    rate = count / n
    if rate >= margin:
        return 1.0
    kl = (rate * math.log(rate / margin) if rate else 0.0) + (1 - rate) * math.log(
        (1 - rate) / (1 - margin)
    )
    return math.exp(-n * kl)


def _discordance_upper(count, n):
    if count == n:
        return 1.0
    low, high = count / n, 1.0 - 1e-15
    for _ in range(60):
        mid = (low + high) / 2
        if _rare_discordance_bound(count, n, mid) > 0.025:
            low = mid
        else:
            high = mid
    return high


def comparisons(study, scores, *, resamples=2000, seed=17):
    matrix, templates = block_matrix(study, scores)
    if resamples < 32:
        raise ValueError("v2 inference requires at least 32 bootstrap draws")
    interaction, delta = effects(matrix)
    rng = np.random.default_rng(seed)
    draws = _bootstrap_differences(interaction, templates, resamples, rng)
    lower = np.quantile(draws, 0.025, axis=0)
    loto = _loto(interaction, templates).min(axis=0)
    primary = {
        family: {
            "difference": float(interaction[:, j].mean()),
            "lower_975": float(lower[j]),
            "leave_one_template_out_min": float(loto[j]),
            "easy_difference": float(delta[:, 0, j].mean()),
            "hard_difference": float(delta[:, 3, j].mean()),
        }
        for j, family in enumerate(FAMILIES)
    }
    # Every secondary is a fixed block-level linear contrast. Width is nested
    # under depth; depth/confusability main effects standardize widths equally.
    plan = matrix[:, :, [0, 3]] - matrix[:, :, [2, 5]]
    depth = delta[:, 2:].mean(axis=1) - delta[:, :2].mean(axis=1)
    secondary_values = {
        "depth": depth,
        "confusability": (
            (delta[:, 1] - delta[:, 0])
            + (delta[:, 3] + delta[:, 5] - delta[:, 2] - delta[:, 4]) / 2
        )
        / 2,
        "width_at_four_hops": delta[:, 4:].mean(axis=1) - delta[:, 2:4].mean(axis=1),
        "depth_by_confusability": delta[:, 3] - delta[:, 2] - delta[:, 1] + delta[:, 0],
        "width_standardized_interaction": (delta[:, 3] + delta[:, 5]) / 2 - delta[:, 0],
        "depth_slope": depth / 3,
        "WT_plan_hard": plan[:, 3],
        "WT_plan_easy": plan[:, 0],
        "easy_equivalence": delta[:, 0],
    }
    secondaries = {}
    for name, values in secondary_values.items():
        boots = _bootstrap_differences(values, templates, resamples, rng)
        mean = values.mean(axis=0)
        errors = boots - mean
        for j, family in enumerate(FAMILIES):
            if name == "easy_equivalence":
                p = max(
                    _rare_discordance_bound(np.sum(values[:, j] == sign), len(values))
                    for sign in (-1, 1)
                )
            else:
                p = (1 + np.sum(np.abs(errors[:, j]) >= abs(mean[j]))) / (resamples + 1)
            secondaries[f"{family}:{name}"] = {
                "difference": float(mean[j]),
                "p_raw": float(p),
                "test": "discordance-rate TOST"
                if name == "easy_equivalence"
                else "centered block bootstrap",
                "ci95": (
                    [
                        -_discordance_upper(np.sum(values[:, j] == -1), len(values)),
                        _discordance_upper(np.sum(values[:, j] == 1), len(values)),
                    ]
                    if name == "easy_equivalence"
                    else list(map(float, np.quantile(boots[:, j], [0.025, 0.975])))
                ),
            }
    previous = 0.0
    for rank, key in enumerate(sorted(secondaries, key=lambda k: secondaries[k]["p_raw"])):
        adjusted = min(1.0, max(previous, (len(secondaries) - rank) * secondaries[key]["p_raw"]))
        secondaries[key]["p_holm"] = previous = adjusted
    return {
        "primary": primary,
        "primary_contrast": "D(4,confusable,1)-D(1,distinct,1)",
        "secondary": secondaries,
        "passes": bool(
            np.all(interaction.mean(axis=0) >= 0.03) and np.all(lower > 0) and np.all(loto > 0)
        ),
        "blocks": len(matrix),
        "resamples": resamples,
        "seed": seed,
        "cluster": "six-cell block; all conditions, arms and families retained",
        "multiplicity": "both-family intersection-union alpha .025; 18 secondary Holm tests",
    }


def _predictive(rows, count, mass, rng):
    """Sparse equivalent of v1's Dirichlet over every binary outcome vector.

    A Polya urn avoids allocating 2**36 states. The uniform base measure has
    zero expected interaction. Observed and new draws retain complete blocks.
    """
    urn = list(rows)
    result = []
    for _ in range(count):
        row = (
            rng.integers(0, 2, size=rows.shape[1:])
            if rng.random() < mass / (mass + len(urn))
            else urn[rng.integers(len(urn))]
        )
        urn.append(row)
        result.append(row)
    return np.asarray(result)


def assurance(study, scores, *, pilot_gpu_seconds, trials=2000, nested_resamples=2000, seed=17):
    pending = {
        "status": "pending_pilot",
        "admitted": False,
        "assurance": None,
        "confirmation_worlds": 384,
        "seed": seed,
    }
    if not scores or pilot_gpu_seconds is None:
        return pending
    if study["config"]["split"] != "development":
        raise ValueError("v2 assurance may use only development pilot outcomes")
    matrix, templates = block_matrix(study, scores)
    if len(matrix) != 16 or len(set(templates)) != 8 or set(Counter(templates).values()) != {2}:
        raise ValueError("v2 pilot requires two complete blocks per each of eight templates")
    if (
        not math.isfinite(pilot_gpu_seconds)
        or not 0 < pilot_gpu_seconds <= 7200
        or trials < 1
        or nested_resamples < 32
    ):
        raise ValueError(
            "v2 requires positive measured cost within two hours and valid simulation sizes"
        )
    projected = pilot_gpu_seconds * 4 * 1.5
    if projected > 36000:
        return {**pending, "status": "budget_no_go", "projected_gpu_seconds": projected}
    target_templates = np.repeat(sorted(set(templates)), 8)
    observed = effects(matrix)[0]
    loto_ok = bool(np.all(_loto(observed, templates) > 0))
    scenarios = []
    for mass in (0.5, 2.0):
        rng = np.random.default_rng(seed)
        passed = 0
        for _ in range(trials):
            target = np.concatenate(
                [_predictive(matrix[templates == t], 8, mass, rng) for t in sorted(set(templates))]
            )
            interaction, _ = effects(target)
            if np.any(interaction.mean(axis=0) < 0.03) or np.any(
                _loto(interaction, target_templates) <= 0
            ):
                continue
            boots = _bootstrap_differences(interaction, target_templates, nested_resamples, rng)
            passed += bool(np.all(np.quantile(boots, 0.025, axis=0) > 0))
        scenarios.append(
            {
                "prior_mass_per_template": mass,
                "assurance": passed / trials,
                "mc_lower_95": _mc_lower(passed, trials),
            }
        )
    return {
        "status": "pilot_projection",
        "admitted": loto_ok and min(s["mc_lower_95"] for s in scenarios) >= 0.8,
        "assurance": min(s["assurance"] for s in scenarios),
        "scenarios": scenarios,
        "observed_leave_one_template_out_positive": loto_ok,
        "confirmation_worlds": 384,
        "trials": trials,
        "nested_resamples": nested_resamples,
        "seed": seed,
        "pilot_gpu_seconds": pilot_gpu_seconds,
        "cost_factor": 1.5,
        "projected_gpu_seconds": projected,
        "assumption": "exchangeable blocks within templates; uniform zero-effect prior",
    }
