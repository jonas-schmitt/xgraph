"""World-clustered paired inference and pilot-based prospective assurance."""

from __future__ import annotations

import json
import math
from collections import Counter
from functools import cache
from pathlib import Path

import numpy as np

FAMILIES = ("qwen38", "gemma31")
DEPTH_ALPHA = 0.025 / 2
DEPTH_MINIMUM = 0.03
PREDICTION_INPUTS = {
    "paired_fit": (
        "paired-fit.json",
        "1ee5c1f9e8c3fb7c6b53f198157b790f7971844a5cd9cc1893cca991096d8756",
    ),
    "power_screen": (
        "power-screen.json",
        "4d3c5960d383c793a6363ecd40929ddf6f7f056dc530c4e01a7bb89f42a65954",
    ),
}


def load_prediction(*, paired_path=None, screen_path=None):
    """Verify frozen source bytes and select development coefficients only.

    paired-fit.json contains the family fits, not the common-slope fit described
    in the theory report. The latter is in power-screen.json. Both are pinned;
    no validation coefficient and no fresh-world outcome enters prediction.
    """
    from xgraph.provenance import sha256_file

    root = Path(__file__).resolve().parents[2] / "eval/fresh-worlds-diagnostics-2026-10-27"
    data, sources = {}, {}
    overrides = {"paired_fit": paired_path, "power_screen": screen_path}
    for key, (name, expected) in PREDICTION_INPUTS.items():
        path = Path(overrides[key]) if overrides[key] is not None else root / name
        if sha256_file(path) != expected:
            raise ValueError(f"frozen theory input SHA-256 mismatch: {path}")
        data[key] = json.loads(path.read_text())
        sources[key] = {
            "path": str(path),
            "sha256": expected,
            "origin": f"/home/jonas/git/xgraph/target/brainstorm/r25-scratch/theory/{name}",
        }
    fits = data["paired_fit"]["families"]
    screen = data["power_screen"]
    return {
        "model": "P(WT-only | discordant,h,f) = sigmoid(a_f + b*(h-1))",
        "shared_dev_beta": screen["shared_paired_slope"]["analysis-dev"]["beta"],
        "family_dev_beta": {
            f: fits[f]["analysis-dev"]["models"]["slope"]["beta"] for f in FAMILIES
        },
        "constant_dev_logit": {
            f: fits[f]["analysis-dev"]["models"]["constant"]["beta"][0] for f in FAMILIES
        },
        "sources": sources,
        "theory_power_screen": {
            "marginal_slope_power_at_256": {
                f: screen["results"][f]["forecast_power_256_marginal"] for f in FAMILIES
            },
            "assumptions": "optimistic normal slope screen; no multiplicity/transport uncertainty",
            "scope": "not assurance for the secondary accuracy interaction; not joint power",
        },
        "scope": "development-fitted prediction; target coefficients never refitted",
    }


def _direction_scores(delta, probabilities):
    discordant = delta != 0
    if not np.any(discordant):
        return {"log_loss": None, "brier": None, "discordant_worlds": 0}
    y = delta[discordant] == 1
    p = probabilities[discordant]
    return {
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))),
        "brier": float(np.mean((y - p) ** 2)),
        "discordant_worlds": int(sum(discordant)),
    }


def frozen_crossover_check(study, scores):
    """Predicted versus observed discordance odds by original-condition hop."""
    prediction = load_prediction()
    matrix, _, hops = _original_pairs(study, scores)
    families = {}
    for j, family in enumerate(FAMILIES):
        a = prediction["shared_dev_beta"][j]
        b = prediction["shared_dev_beta"][2]
        probabilities = 1 / (1 + np.exp(-a - b * (hops - 1)))
        constant = 1 / (1 + math.exp(-prediction["constant_dev_logit"][family]))
        delta = matrix[:, 2 * j] - matrix[:, 2 * j + 1]
        cells = {}
        for hop in sorted(set(hops)):
            values = delta[hops == hop]
            wins, losses = int(sum(values == 1)), int(sum(values == -1))
            cells[str(hop)] = {
                "worlds": len(values),
                "wt_only": wins,
                "agent_only": losses,
                "ties": int(sum(values == 0)),
                "discordant": wins + losses,
                "predicted_odds": math.exp(a + b * (hop - 1)),
                "predicted_wt_only_given_discordant": float(probabilities[hops == hop][0]),
                "observed_odds": wins / losses if losses else None,
                "odds_status": "finite" if losses else "infinite" if wins else "no_discordance",
                "observed_wt_only_given_discordant": wins / (wins + losses)
                if wins + losses
                else None,
            }
        families[family] = {
            "by_hop": cells,
            "shared_fit_odds_4_vs_2": math.exp(2 * b),
            "family_fit_odds_4_vs_2": math.exp(2 * prediction["family_dev_beta"][family][1]),
            "discordant_direction_scores": _direction_scores(delta, probabilities),
            "dev_constant_direction_scores": _direction_scores(delta, np.full(len(hops), constant)),
        }
    return {
        "prediction": prediction,
        "endpoint": "original-condition ID correctness",
        "families": families,
        "status": "descriptive predictive check",
        "scope": "conditional direction, not disagreement rate, absolute accuracy or causality",
    }


def _original_pairs(study, scores):
    worlds = study["worlds"]
    ids = [w["id"] for w in worlds]
    templates = np.asarray([w["template"] for w in worlds])
    hops = np.asarray([len(w["relations"]) for w in worlds])
    if not ids or len(set(ids)) != len(ids) or not set(hops) <= {2, 3, 4}:
        raise ValueError("unique fresh worlds at two to four hops are required")
    columns = []
    for family in FAMILIES:
        for arm in ("WT", "agent"):
            score = scores.get(f"{family}:{arm}")
            if score is None or score["missing"]:
                raise ValueError("incomplete WT/agent matrix for both families")
            rows = score["per_world"]
            if [w["id"] for w in rows] != ids or [
                w["template"] for w in rows
            ] != templates.tolist():
                raise ValueError("unpaired original-condition worlds or templates")
            values = []
            for row in rows:
                original = [c for c in row["conditions"] if c["condition"] == "original"]
                if len(original) != 1:
                    raise ValueError("one original-condition score per world is required")
                values.append(original[0]["correct"])
            columns.append(values)
    return np.asarray(columns, int).T, templates, hops


def _depth_mean(delta, hops):
    return delta[hops == 4].mean(axis=0) - delta[hops == 2].mean(axis=0)


def _depth_draws(delta, templates, hops, resamples, rng):
    return _bootstrap_differences(
        delta[hops == 4], templates[hops == 4], resamples, rng
    ) - _bootstrap_differences(delta[hops == 2], templates[hops == 2], resamples, rng)


def _depth_loto(delta, templates, hops):
    result = {}
    for name in sorted(set(templates)):
        keep = templates != name
        result[str(name)] = (
            _depth_mean(delta[keep], hops[keep]) if {2, 4} <= set(hops[keep]) else None
        )
    return result


def depth_contrast(study, scores, *, resamples=2000, seed=17):
    """Secondary original-condition four-hop minus two-hop WT/agent advantage."""
    matrix, templates, hops = _original_pairs(study, scores)
    if not {2, 4} <= set(hops) or resamples < 32:
        raise ValueError("depth contrast needs two- and four-hop worlds and >=32 resamples")
    delta = matrix[:, [0, 2]] - matrix[:, [1, 3]]
    observed = _depth_mean(delta, hops)
    draws = _depth_draws(delta, templates, hops, resamples, np.random.default_rng(seed))
    loto = _depth_loto(delta, templates, hops)
    families = {}
    for j, family in enumerate(FAMILIES):
        sensitivity = {t: float(v[j]) if v is not None else None for t, v in loto.items()}
        families[family] = {
            "difference": float(observed[j]),
            "pointwise_lower_975": float(np.quantile(draws[:, j], 0.025)),
            "adjusted_lower": float(np.quantile(draws[:, j], DEPTH_ALPHA)),
            "by_hop": {
                str(h): {
                    "difference": float(delta[hops == h, j].mean()),
                    "worlds": int(sum(hops == h)),
                }
                for h in sorted(set(hops))
            },
            "by_template": {
                str(t): float(delta[templates == t, j].mean()) for t in sorted(set(templates))
            },
            "leave_one_template_out": sensitivity,
            "leave_one_template_out_min": (
                min(sensitivity.values())
                if all(v is not None for v in sensitivity.values())
                else None
            ),
        }
    return {
        "endpoint": "original-condition ID correctness",
        "estimand": "(WT-agent at 4 hops) - (WT-agent at 2 hops)",
        "cluster": "whole world; arms and families paired; fixed template quotas",
        "multiplicity": {
            "method": "Bonferroni across two secondary family contrasts",
            "familywise_one_sided_alpha": 0.025,
            "alpha_per_family": DEPTH_ALPHA,
            "predictive_check": "descriptive, no additional significance claim",
        },
        "registered_primary_gate_changed": False,
        "minimum_useful_interaction": DEPTH_MINIMUM,
        "families": families,
        "passes": all(
            c["difference"] >= DEPTH_MINIMUM
            and c["adjusted_lower"] > 0
            and c["leave_one_template_out_min"] is not None
            and c["leave_one_template_out_min"] > 0
            for c in families.values()
        ),
        "resamples": resamples,
        "seed": seed,
        "scope": "shared-template graph transfer; cannot confirm a one-hop reversal or causality",
    }


def _mc_lower(passed, trials):
    estimate = passed / trials
    z = 1.96
    return max(
        0.0,
        (
            estimate
            + z * z / (2 * trials)
            - z * math.sqrt(estimate * (1 - estimate) / trials + z * z / (4 * trials * trials))
        )
        / (1 + z * z / trials),
    )


def depth_assurance(study, scores, *, trials=2000, nested_resamples=2000, seed=17):
    """Own secondary assurance; never borrowed from the primary six-bit model."""
    if study is None or not scores:
        return {
            "status": "pending_pilot",
            "joint_power": None,
            "marginal_power": None,
            "adequately_powered": False,
            "confirmation_worlds": 256,
            "seed": seed,
            "reason": "Complete development WT/agent original outcomes required.",
        }
    if study["config"]["split"] != "development":
        raise ValueError("secondary assurance must use development pilot outcomes")
    matrix, templates, hops = _original_pairs(study, scores)
    counts = Counter(templates)
    if len(matrix) != 64 or len(counts) != 32 or set(counts.values()) != {2}:
        raise ValueError("secondary assurance needs two pilot worlds per fixed template")
    if trials < 1 or nested_resamples < 32:
        raise ValueError("positive trials and >=32 nested resamples are required")
    names = sorted(counts)
    if any(len(set(hops[templates == t])) != 1 for t in names) or not {2, 4} <= set(hops):
        raise ValueError("templates must fix hop counts and cover two and four hops")
    target_templates = np.repeat(names, 8)
    target_hops = np.repeat([hops[templates == t][0] for t in names], 8)
    patterns = np.asarray([[(k >> j) & 1 for j in range(4)] for k in range(16)], int)
    posterior = [
        np.bincount(matrix[templates == t] @ (2 ** np.arange(4)), minlength=16) for t in names
    ]
    scenarios = []
    for prior_mass in (0.5, 2.0):
        rng = np.random.default_rng(seed)
        passed = np.zeros(2, int)
        joint = 0
        for _ in range(trials):
            target = np.concatenate(
                [
                    patterns[rng.choice(16, 8, p=rng.dirichlet(row + prior_mass / 16))]
                    for row in posterior
                ]
            )
            delta = target[:, [0, 2]] - target[:, [1, 3]]
            good = _depth_mean(delta, target_hops) >= DEPTH_MINIMUM
            if not np.any(good):
                continue
            loto = _depth_loto(delta, target_templates, target_hops)
            good &= np.all(np.asarray(list(loto.values())) > 0, axis=0)
            if not np.any(good):
                continue
            draws = _depth_draws(delta, target_templates, target_hops, nested_resamples, rng)
            good &= np.quantile(draws, DEPTH_ALPHA, axis=0) > 0
            passed += good
            joint += bool(np.all(good))
        scenarios.append(
            {
                "prior_mass_per_template": prior_mass,
                "marginal_power": {f: float(passed[j] / trials) for j, f in enumerate(FAMILIES)},
                "marginal_mc_lower_95": {
                    f: _mc_lower(passed[j], trials) for j, f in enumerate(FAMILIES)
                },
                "joint_power": joint / trials,
                "joint_mc_lower_95": _mc_lower(joint, trials),
            }
        )
    return {
        "status": "pilot_projection",
        "scenarios": scenarios,
        "joint_power": min(s["joint_power"] for s in scenarios),
        "adequately_powered": min(s["joint_mc_lower_95"] for s in scenarios) >= 0.8,
        "confirmation_worlds": 256,
        "trials": trials,
        "nested_resamples": nested_resamples,
        "seed": seed,
        "pattern": "four original-condition correctness bits; families paired",
        "test": "same adjusted interval, .03 observed effect and positive template sensitivities",
        "assumptions": "template-specific Dirichlet predictive model; no transport guarantee",
        "registered_primary_gate_changed": False,
        "incremental_gpu_hours": 0,
    }


def replay_programs(study, requests, planner_records, executor):
    """Freeze each original predicted program; execute with each request's KG version.

    The answer-reuse diagnostic freezes its original matched-execution delivery.
    Parsing and execution never consult gold, edited-condition plans or answers.
    Missing original programs and generation failures retain the ID evaluator's
    ordinary failure treatment. This is offline replay, with no model calls.
    """
    from xgraph.programs import record_path
    from xgraph.score import record_failure

    expected = {c["record_id"] for w in study["worlds"] for c in w["conditions"]}
    sources, prompts = {}, {}
    for rows, destination in ((planner_records, sources), (requests, prompts)):
        for row in rows:
            if row["id"] in destination or row["id"] not in expected:
                raise ValueError(f"duplicate or unknown record id: {row['id']}")
            destination[row["id"]] = row
    if set(prompts) != expected:
        raise ValueError("replay requires every gold-free versioned request")
    replay, reuse = [], []
    for world in study["worlds"]:
        original = next(c for c in world["conditions"] if c["name"] == "original")
        source = sources.get(original["record_id"])
        if source is None:
            continue
        path = None if record_failure(source, require_text=False) else record_path(source)
        frozen_answer = None
        delivered = []
        for condition in world["conditions"]:
            request = prompts[condition["record_id"]]
            reached = (
                executor.execute(*path, request["anchor"], request["kg_version"]) if path else None
            )
            answer = reached.answer if reached else None
            delivered.append(
                {
                    **source,
                    "id": request["id"],
                    "answer_id": answer,
                    "program_source_id": original["record_id"],
                    "kg_version": request["kg_version"],
                    "delivery": "frozen original program plus matched exact execution",
                    "additional_model_calls": 0,
                }
            )
            if condition["name"] == "original":
                frozen_answer = answer
        replay.extend(delivered)
        reuse.extend(
            {
                **row,
                "answer_id": frozen_answer,
                "delivery": "reuse original matched-execution answer",
            }
            for row in delivered
        )
    return replay, reuse


@cache
def paired_p(wins, losses):
    n = wins + losses
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n


def _bootstrap_differences(matrix, templates, resamples, rng):
    groups = [np.flatnonzero(np.asarray(templates) == name) for name in sorted(set(templates))]
    indices = np.concatenate(
        [rng.choice(group, (resamples, len(group))) for group in groups], axis=1
    )
    return matrix[indices].mean(axis=1)


def paired_contrast(wt, reference, templates, *, resamples=2000, seed=17):
    if not len(wt) == len(reference) == len(templates) or not wt or resamples < 1:
        raise ValueError("paired worlds and templates must be nonempty and aligned")
    values = np.asarray(wt, dtype=int) - np.asarray(reference, dtype=int)
    sampled = _bootstrap_differences(
        values[:, None], templates, resamples, np.random.default_rng(seed)
    )
    loto = [
        float(values[np.asarray(templates) != name].mean())
        for name in set(templates)
        if any(t != name for t in templates)
    ]
    return {
        "difference": float(values.mean()),
        "lower_975": float(np.quantile(sampled, 0.025)),
        "p_one_sided": paired_p(int(sum(values == 1)), int(sum(values == -1))),
        "worlds": len(values),
        "resamples": resamples,
        "seed": seed,
        "cluster": "world; all conditions and families paired; fixed template quotas",
        "leave_one_template_out_min": min(loto) if loto else None,
        "by_template": {
            t: float(values[np.asarray(templates) == t].mean()) for t in sorted(set(templates))
        },
    }


def _matrix(scores):
    keys = [
        f"{family}:{arm}"
        for family in ("qwen38", "gemma31")
        for arm in ("WT", "agent", "plan-exact")
    ]
    if any(key not in scores for key in keys):
        raise ValueError("assurance requires both families and all registered primary arms")
    worlds = scores[keys[0]]["per_world"]
    ids = [w["id"] for w in worlds]
    templates = [w["template"] for w in worlds]
    for key in keys:
        rows = scores[key]["per_world"]
        if (
            scores[key]["missing"]
            or [w["id"] for w in rows] != ids
            or [w["template"] for w in rows] != templates
        ):
            raise ValueError("incomplete or unpaired development pilot")
    return np.asarray(
        [[scores[key]["per_world"][i]["joint"] for key in keys] for i in range(len(ids))], int
    ), templates


def comparisons(scores, *, resamples=2000, seed=17):
    matrix, templates = _matrix(scores)
    columns = [(0, 1), (0, 2), (3, 4), (3, 5)]
    contrasts = {
        f"{family}:WT-{arm}": paired_contrast(
            matrix[:, a].tolist(), matrix[:, b].tolist(), templates, resamples=resamples, seed=seed
        )
        for (a, b), (family, arm) in zip(
            columns,
            ((f, a) for f in ("qwen38", "gemma31") for a in ("agent", "plan-exact")),
            strict=True,
        )
    }
    return {
        "contrasts": contrasts,
        "intersection_p": max(c["p_one_sided"] for c in contrasts.values()),
        "passes": all(
            c["p_one_sided"] <= 0.025 and c["lower_975"] > 0 and c["difference"] >= 0.05
            for c in contrasts.values()
        ),
        "resamples": resamples,
        "seed": seed,
    }


def assurance(scores, *, pilot_gpu_seconds, trials=2000, seed=17, nested_resamples=2000):
    if not scores or pilot_gpu_seconds is None:
        return {
            "status": "pending_pilot",
            "assurance": None,
            "confirmation_worlds": 256,
            "reason": "No fresh-world GPU pilot has run; outcomes and charged costs are required.",
            "admitted": False,
            "seed": seed,
        }
    matrix, templates = _matrix(scores)
    counts = Counter(templates)
    if len(matrix) != 64 or len(counts) != 32 or set(counts.values()) != {2}:
        raise ValueError("the pilot must contain two development worlds per fixed template")
    if (
        not math.isfinite(pilot_gpu_seconds)
        or pilot_gpu_seconds <= 0
        or trials < 1
        or nested_resamples < 32
    ):
        raise ValueError("positive measured pilot cost and simulation sizes are required")
    feasible = min(256, 32 * int(64 * 36000 / (1.5 * pilot_gpu_seconds) / 32))
    if feasible < 256:
        return {
            "status": "budget_no_go",
            "admitted": False,
            "assurance": None,
            "feasible_worlds": feasible,
            "confirmation_worlds": 256,
        }
    # Preserve the whole six-bit outcome vector: cross-arm/family dependence is
    # never factored into marginal powers. Each template has a symmetric prior
    # on all 64 possible vectors, with total mass .5 and a skeptical sensitivity 2.
    patterns = np.asarray([[(k >> j) & 1 for j in range(6)] for k in range(64)], int)
    posterior = []
    for name in sorted(counts):
        rows = matrix[np.asarray(templates) == name]
        codes = rows @ (2 ** np.arange(6))
        posterior.append(np.bincount(codes, minlength=64))
    target_templates = [t for t in sorted(counts) for _ in range(8)]
    scenarios = []
    for prior_mass in (0.5, 2.0):
        rng = np.random.default_rng(seed)
        passed = 0
        for _ in range(trials):
            target = np.concatenate(
                [
                    patterns[rng.choice(64, 8, p=rng.dirichlet(row + prior_mass / 64))]
                    for row in posterior
                ]
            )
            delta = target[:, [0, 0, 3, 3]] - target[:, [1, 2, 4, 5]]
            if np.any(delta.mean(axis=0) < 0.05):
                continue
            if any(paired_p(int(sum(col == 1)), int(sum(col == -1))) > 0.025 for col in delta.T):
                continue
            boots = _bootstrap_differences(delta, target_templates, nested_resamples, rng)
            passed += bool(np.all(np.quantile(boots, 0.025, axis=0) > 0))
        estimate = passed / trials
        z = 1.96
        center = (estimate + z * z / (2 * trials)) / (1 + z * z / trials)
        radius = (
            z
            * math.sqrt(estimate * (1 - estimate) / trials + z * z / (4 * trials * trials))
            / (1 + z * z / trials)
        )
        scenarios.append(
            {
                "prior_mass_per_template": prior_mass,
                "assurance": estimate,
                "mc_lower_95": max(0.0, center - radius),
            }
        )
    observed = comparisons(scores, resamples=max(2000, nested_resamples), seed=seed)
    loto_ok = all(c["leave_one_template_out_min"] > 0 for c in observed["contrasts"].values())
    return {
        "status": "pilot_projection",
        "admitted": min(s["mc_lower_95"] for s in scenarios) >= 0.8 and loto_ok,
        "assurance": min(s["assurance"] for s in scenarios),
        "scenarios": scenarios,
        "observed_leave_one_template_out_positive": loto_ok,
        "confirmation_worlds": 256,
        "trials": trials,
        "nested_resamples": nested_resamples,
        "seed": seed,
        "pilot_gpu_seconds": pilot_gpu_seconds,
        "cost_factor": 1.5,
        "projected_gpu_seconds": pilot_gpu_seconds * 4 * 1.5,
        "assumption": "fresh worlds exchangeable within shared templates; no transport guarantee",
    }


def _load_scores(study_dir, outputs_dir):
    from xgraph.fresh_worlds import read_jsonl, score_records

    study_path = Path(study_dir) / "study.json"
    study = json.loads(study_path.read_text())
    files, scores = [study_path], {}
    for family in FAMILIES:
        for arm in ("WT", "agent"):
            path = Path(outputs_dir) / family / f"{arm}.jsonl"
            records = read_jsonl(path) if path.exists() else []
            scores[f"{family}:{arm}"] = score_records(study, records)
            if path.exists():
                files.append(path)
    return study, scores, files


def _replay_context(study_dir, family):
    from xgraph.index import load_kg_index
    from xgraph.kgversions import load_kg_versions
    from xgraph.programs import MatchedExecutor
    from xgraph.provenance import sha256_file
    from xgraph.tokenizer import ServedTokenizer

    tokenizers = study_dir / "tokenizers.json"
    spec = json.loads(tokenizers.read_text())[family]
    token = Path(spec["path"])
    if sha256_file(token) != spec["sha256"]:
        raise ValueError("served tokenizer input changed")
    index_dir = study_dir / "indexes" / family
    versions_dir = study_dir / "kg-versions"
    index = load_kg_index(index_dir)
    # Exactly the queued one-call planner's executor configuration; no repair,
    # gold plan, endpoint selection, or intermediate-entity generation is added.
    executor = MatchedExecutor(
        index,
        ServedTokenizer.from_file(token),
        versions=load_kg_versions(versions_dir, index),
        start_rank="sitelinks",
        unquoted_starts=True,
    )
    files = [tokenizers, token]
    files.extend(
        p
        for directory in (index_dir, versions_dir)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    )
    return executor, files


def _replay_summary(study, replay, reuse):
    from xgraph.fresh_worlds import score_records

    frozen, copied = score_records(study, replay), score_records(study, reuse)

    def rates(rows):
        return {
            "worlds": len(rows),
            "joint_version_fidelity": sum(r["joint"] for r in rows) / len(rows),
            "condition_accuracy": {
                c: sum(
                    next(v["correct"] for v in r["conditions"] if v["condition"] == c) for r in rows
                )
                / len(rows)
                for c in ("original", "inner-hop-edit", "off-chain-edit")
            },
        }

    grouped = {}
    for field, key in (
        ("by_hop", lambda w: str(len(w["relations"]))),
        ("by_template", lambda w: w["template"]),
    ):
        grouped[field] = {}
        for value in sorted({key(w) for w in study["worlds"]}):
            ids = {w["id"] for w in study["worlds"] if key(w) == value}
            grouped[field][value] = {
                "frozen_program": rates([r for r in frozen["per_world"] if r["id"] in ids]),
                "reuse_answer": rates([r for r in copied["per_world"] if r["id"] in ids]),
            }
    return {
        "frozen_program": frozen,
        "reuse_answer": copied,
        **grouped,
        "joint_fidelity_gain": frozen["joint_version_fidelity"] - copied["joint_version_fidelity"],
        "additional_model_calls": 0,
        "status": "descriptive diagnostic",
        "original_answer_source": "original predicted program's matched exact execution",
        "scope": "value updating conditional on a fixed predicted program, not a planning gain",
    }


FACTORIAL_CELLS = ("s0d0", "s0d1", "s1d0", "s1d1")
FACTORIAL_DIRECTIONS = {"qwen38": 1, "gemma31": 1}
FACTORIAL_ALPHA = 0.025 / 2
FACTORIAL_MINIMUM = 0.03


def _factorial_matrix(study, scores):
    from xgraph.fresh_worlds import CONDITIONS

    worlds = study["worlds"]
    ids = [w["id"] for w in worlds]
    templates = np.asarray([w["template"] for w in worlds])
    sequences = np.asarray([">".join(map(str, w["relations"])) for w in worlds])
    hops = np.asarray([len(w["relations"]) for w in worlds])
    if not ids or len(set(ids)) != len(ids) or not set(hops) <= {2, 3, 4}:
        raise ValueError("factorial requires unique fresh worlds at two to four hops")
    matrix = np.zeros((len(worlds), 2, 4, 3), dtype=int)
    complete = True
    for j, family in enumerate(FAMILIES):
        for k, cell in enumerate(FACTORIAL_CELLS):
            score = scores.get(f"{family}:{cell}")
            if score is None:
                raise ValueError(f"incomplete factorial matrix: {family}:{cell}")
            complete &= score["missing"] == 0
            rows = score["per_world"]
            if [w["id"] for w in rows] != ids or [
                w["template"] for w in rows
            ] != templates.tolist():
                raise ValueError("unpaired factorial worlds or templates")
            for i, row in enumerate(rows):
                conditions = row["conditions"]
                if len(conditions) != 3 or {c["condition"] for c in conditions} != set(CONDITIONS):
                    raise ValueError("factorial requires all three conditions of every world")
                values = {c["condition"]: c["correct"] for c in conditions}
                if any(type(v) is not bool for v in values.values()):
                    raise ValueError("factorial correctness must be Boolean independent ID scores")
                matrix[i, j, k] = [values[c] for c in CONDITIONS]
    return matrix, templates, sequences, hops, complete


def _factorial_bands(drift):
    if set(drift) != set(FAMILIES) or any(
        not isinstance(v, int | float) or not math.isfinite(v) or not 0 <= v <= 1
        for v in drift.values()
    ):
        raise ValueError("both families need finite nonnegative per-cell numeric-drift bands")
    return 4 * np.asarray([drift[f] for f in FAMILIES])


def _factorial_delta(matrix):
    return matrix[:, :, 3] - matrix[:, :, 2] - matrix[:, :, 1] + matrix[:, :, 0]


def _factorial_loso(delta, sequences):
    return {
        str(name): delta[sequences != name].mean(axis=0) if np.any(sequences != name) else None
        for name in sorted(set(sequences))
    }


def _factorial_gate(delta, sequences, bands, *, lower=None):
    # The frozen prediction is positive in each family. Keep the sign explicit
    # so a large opposite-sign interaction cannot silently satisfy |I| admission.
    signs = np.asarray([FACTORIAL_DIRECTIONS[f] for f in FAMILIES])
    point = delta.mean(axis=0)
    good = (np.abs(point) >= FACTORIAL_MINIMUM) & (signs * point > bands)
    for value in _factorial_loso(delta, sequences).values():
        good &= False if value is None else signs * value > bands
    if lower is not None:
        good &= lower > bands
    return good


def factorial_contrasts(study, scores, *, drift, resamples=2000, seed=17):
    """Prospective availability interaction; worlds pair all cells/conditions/families."""
    matrix, templates, sequences, hops, complete = _factorial_matrix(study, scores)
    if resamples < 32:
        raise ValueError("factorial inference requires >=32 resamples")
    bands = _factorial_bands(drift)
    condition_delta = _factorial_delta(matrix)
    delta = condition_delta.mean(axis=2)
    rng = np.random.default_rng(seed)
    draws = _bootstrap_differences(condition_delta, templates, resamples, rng)
    primary_draws = draws.mean(axis=2)
    lower = np.quantile(primary_draws, FACTORIAL_ALPHA, axis=0)
    loso = _factorial_loso(delta, sequences)
    secondary = {
        "inner_minus_original": (
            condition_delta[:, :, 1].mean(axis=0) - condition_delta[:, :, 0].mean(axis=0),
            draws[:, :, 1] - draws[:, :, 0],
        ),
        "off_minus_original": (
            condition_delta[:, :, 2].mean(axis=0) - condition_delta[:, :, 0].mean(axis=0),
            draws[:, :, 2] - draws[:, :, 0],
        ),
        "depth_4_minus_2": (None, None),
    }
    if {2, 4} <= set(hops):
        secondary["depth_4_minus_2"] = (
            _depth_mean(delta, hops),
            _depth_draws(delta, templates, hops, resamples, rng),
        )
    families = {}
    for j, family in enumerate(FAMILIES):
        omitted = {s: float(v[j]) if v is not None else None for s, v in loso.items()}
        families[family] = {
            "interaction": float(delta[:, j].mean()),
            "predicted_sign": FACTORIAL_DIRECTIONS[family],
            "interval_95": np.quantile(primary_draws[:, j], [0.025, 0.975]).tolist(),
            "adjusted_lower": float(lower[j]),
            "per_cell_drift": drift[family],
            "primary_drift_band": float(bands[j]),
            "drift_adjusted_range": [
                float(delta[:, j].mean() - bands[j]),
                float(delta[:, j].mean() + bands[j]),
            ],
            "leave_one_sequence_out": omitted,
            "leave_one_sequence_out_min": (
                min(omitted.values()) if all(v is not None for v in omitted.values()) else None
            ),
            "by_sequence": {
                str(s): float(delta[sequences == s, j].mean()) for s in sorted(set(sequences))
            },
            "by_hop": {str(h): float(delta[hops == h, j].mean()) for h in sorted(set(hops))},
            "cell_accuracy": {
                c: float(matrix[:, j, k].mean()) for k, c in enumerate(FACTORIAL_CELLS)
            },
            "cell_joint_version_fidelity": {
                c: float(matrix[:, j, k].all(axis=1).mean()) for k, c in enumerate(FACTORIAL_CELLS)
            },
            "secondary": {
                name: {
                    "difference": float(point[j]) if point is not None else None,
                    "simultaneous_interval": (
                        np.quantile(boot[:, j], [0.025 / 12, 1 - 0.025 / 12]).tolist()
                        if boot is not None
                        else None
                    ),
                    "numeric_drift_band": float(2 * bands[j]),
                    "status": "defined"
                    if point is not None
                    else "undefined: missing depth stratum",
                }
                for name, (point, boot) in secondary.items()
            },
        }
    return {
        "status": "completed" if complete else "INCOMPLETE",
        "endpoint": "mean of three condition-specific independent ID correctness indicators",
        "contrast": "(s1d1-s1d0)-(s0d1-s0d0)",
        "cluster": "whole world, fixed template quotas; all cells, conditions and families paired",
        "families": families,
        "development_admitted": complete and bool(np.all(_factorial_gate(delta, sequences, bands))),
        "confirmation_passes": complete
        and bool(np.all(_factorial_gate(delta, sequences, bands, lower=lower))),
        "multiplicity": {"primary_alpha": FACTORIAL_ALPHA, "secondary_tail_alpha": 0.025 / 12},
        "worlds": len(matrix),
        "resamples": resamples,
        "seed": seed,
        "limits": "policy intervention; historical drift sensitivity, not a future bound",
    }


def _factorial_predictive(posteriors, per_template, prior_mass, rng):
    """Exact sparse Dirichlet-multinomial draws over the entire 24-bit space.

    The Pólya urn integrates parameter uncertainty. An unseen-pattern draw is
    uniform over the complement of occupied codes, then joins the urn and is
    reinforced on subsequent draws. No dense 2**24 table or independent-arm
    factorization is used, including for previously unseen patterns.
    """
    size = 1 << 24
    selected = []
    for posterior in posteriors:
        counts = dict(posterior)
        for _ in range(per_template):
            codes = list(counts)
            weights = np.asarray(
                [counts[c] + prior_mass / size for c in codes]
                + [prior_mass * (size - len(codes)) / size]
            )
            at = rng.choice(len(weights), p=weights / weights.sum())
            if at < len(codes):
                code = codes[at]
            else:
                code = int(rng.integers(size))
                while code in counts:
                    code = int(rng.integers(size))
            counts[code] = counts.get(code, 0) + 1
            selected.append(code)
    bits = (np.asarray(selected)[:, None] >> np.arange(24)) & 1
    return bits.reshape(-1, 2, 4, 3)


def factorial_assurance(
    study, scores, *, pilot_gpu_seconds, drift, trials=2000, nested_resamples=2000, seed=17
):
    """Own joint predictive assurance for the frozen factorial confirmation rule."""
    pending = {
        "status": "pending_pilot",
        "admitted": False,
        "assurance": None,
        "joint_pattern_bits": 24,
        "confirmation_worlds": 256,
        "seed": seed,
    }
    bands = _factorial_bands(drift)
    if study is None or not scores or pilot_gpu_seconds is None:
        return pending
    matrix, templates, sequences, _hops, complete = _factorial_matrix(study, scores)
    if not complete:
        return {
            **pending,
            "reason": "Incomplete pilot; missing predictions retain their denominator.",
        }
    counts = Counter(templates)
    hard = study["config"].get("design") == "mechanism-v2-hard"
    if (
        study["config"]["split"] != "development"
        or study["config"]["seed"] != 262813
        or len(matrix) != 64
        or len(counts) != (8 if hard else 32)
        or set(counts.values()) != ({8} if hard else {2})
    ):
        raise ValueError(
            "factorial assurance requires its own registered template quotas and development pilot"
        )
    if (
        not math.isfinite(pilot_gpu_seconds)
        or pilot_gpu_seconds <= 0
        or trials < 1
        or nested_resamples < 32
    ):
        raise ValueError("positive measured cost/trials and >=32 nested resamples required")
    projected = 6 * pilot_gpu_seconds
    if pilot_gpu_seconds > 7200 or projected > 36000:
        return {**pending, "status": "budget_no_go", "projected_gpu_seconds": projected}
    delta = _factorial_delta(matrix).mean(axis=2)
    if not np.all(_factorial_gate(delta, sequences, bands)):
        return {**pending, "status": "development_no_go", "projected_gpu_seconds": projected}
    names = sorted(counts)
    if any(len(set(sequences[templates == t])) != 1 for t in names):
        raise ValueError("each template must fix its relation sequence")
    per_template = 256 // len(names)
    target_templates = np.repeat(names, per_template)
    target_sequences = np.repeat([sequences[templates == t][0] for t in names], per_template)
    codes = matrix.reshape(64, 24) @ (2 ** np.arange(24))
    posteriors = [Counter(map(int, codes[templates == t])) for t in names]
    scenarios = []
    for prior_mass in (0.5, 2.0):
        rng = np.random.default_rng(seed)
        joint, marginal = 0, np.zeros(2, int)
        for _ in range(trials):
            target = _factorial_predictive(posteriors, per_template, prior_mass, rng)
            target_delta = _factorial_delta(target).mean(axis=2)
            good = _factorial_gate(target_delta, target_sequences, bands)
            if np.any(good):
                boot = _bootstrap_differences(target_delta, target_templates, nested_resamples, rng)
                good &= np.quantile(boot, FACTORIAL_ALPHA, axis=0) > bands
            marginal += good
            joint += bool(np.all(good))
        scenarios.append(
            {
                "prior_mass_per_template": prior_mass,
                "assurance": joint / trials,
                "mc_lower_95": _mc_lower(joint, trials),
                "marginal_power": {f: float(marginal[j] / trials) for j, f in enumerate(FAMILIES)},
            }
        )
    return {
        **pending,
        "status": "pilot_projection",
        "admitted": all(s["mc_lower_95"] >= 0.8 for s in scenarios),
        "assurance": min(s["assurance"] for s in scenarios),
        "scenarios": scenarios,
        "observed_development_gate_passed": True,
        "pilot_gpu_seconds": pilot_gpu_seconds,
        "projected_gpu_seconds": projected,
        "cost_factor": 1.5,
        "trials": trials,
        "nested_resamples": nested_resamples,
        "assumption": "joint 24-bit vectors exchangeable within templates; no transport guarantee",
    }


def main(argv=None):
    """Standalone diagnostics; the registered analyze command and queue stay fixed."""
    import argparse
    import sys

    from xgraph.fresh_worlds import read_jsonl, write_json
    from xgraph.provenance import provenance

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--outputs-dir", type=Path)
    parser.add_argument("--pilot-study", type=Path)
    parser.add_argument("--pilot-outputs-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spec-only", action="store_true")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--nested-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    if args.resamples < 32 or args.trials < 1 or args.nested_resamples < 32:
        parser.error("positive trials and >=32 resamples/nested resamples are required")
    if bool(args.pilot_study) != bool(args.pilot_outputs_dir):
        parser.error("--pilot-study and --pilot-outputs-dir must be supplied together")
    prediction = load_prediction()
    files = [Path(s["path"]) for s in prediction["sources"].values()]
    files.extend(
        Path(__file__).with_name(name)
        for name in ("fresh_worlds_analysis.py", "fresh_worlds.py", "programs.py")
    )
    result = {
        "status": "frozen_specification_only",
        "prediction": prediction,
        "depth_assurance": depth_assurance(None, {}),
        "registered_primary_gate_changed": False,
        "config": {
            "seed": args.seed,
            "resamples": args.resamples,
            "trials": args.trials,
            "nested_resamples": args.nested_resamples,
            "secondary_alpha_per_family": DEPTH_ALPHA,
            "minimum_useful_interaction": DEPTH_MINIMUM,
        },
    }
    if not args.spec_only:
        if args.study is None or args.outputs_dir is None:
            parser.error("--study and --outputs-dir are required unless --spec-only")
        study, scores, source_files = _load_scores(args.study, args.outputs_dir)
        files.extend(source_files)
        request_path = args.study / "requests-plan.jsonl"
        requests = read_jsonl(request_path)
        files.append(request_path)
        replays = {}
        for family in FAMILIES:
            path = args.outputs_dir / family / "plan.jsonl"
            plans = read_jsonl(path) if path.exists() else []
            executor, context_files = _replay_context(args.study, family)
            files.extend(context_files)
            if path.exists():
                files.append(path)
            replays[family] = _replay_summary(
                study, *replay_programs(study, requests, plans, executor)
            )
        paired = all(not s["missing"] for s in scores.values())
        complete = paired and all(not r["frozen_program"]["missing"] for r in replays.values())
        result.update(
            {
                "status": "completed" if complete else "INCOMPLETE",
                "replay": replays,
                "scores": scores,
                "depth_contrast": depth_contrast(
                    study, scores, resamples=args.resamples, seed=args.seed
                )
                if paired
                else None,
                "crossover": frozen_crossover_check(study, scores) if paired else None,
            }
        )
        pilot_study, pilot_scores = None, {}
        if args.pilot_study is not None:
            pilot_study, pilot_scores, source_files = _load_scores(
                args.pilot_study, args.pilot_outputs_dir
            )
            files.extend(source_files)
        elif study["config"]["split"] == "development" and len(study["worlds"]) == 64:
            pilot_study, pilot_scores = study, scores
        if pilot_scores and any(s["missing"] for s in pilot_scores.values()):
            pilot_study, pilot_scores = None, {}
        result["depth_assurance"] = depth_assurance(
            pilot_study,
            pilot_scores,
            trials=args.trials,
            nested_resamples=args.nested_resamples,
            seed=args.seed,
        )
    result["provenance"] = provenance(list(dict.fromkeys(files)))
    result["provenance"]["analysis_argv"] = [
        "-m",
        "xgraph.fresh_worlds_analysis",
        *(sys.argv[1:] if argv is None else argv),
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    return result


if __name__ == "__main__":
    main()
