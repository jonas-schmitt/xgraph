"""Stream cached repair artifacts, audit their joins, and cross-fit selective rules.

This module owns gold-dependent evaluation. xgraph.selective owns the gold-free
features and inference rule. No KG, model, tokenizer, or GPU is loaded here.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import re
import resource
from pathlib import Path

import numpy as np

from xgraph.score import record_text, replay_failure
from xgraph.selective import FEATURE_VERSION, OPERATORS, features, triggered, validate_rule

FAMILY_INPUTS = (
    "original",
    "candidates",
    "requests",
    "verdicts",
    "selected",
    "engine",
    "delivery_requests",
    "delivery_outputs",
)
THRESHOLDS = {
    "lexical_margin": [-0.5, -0.25, 0.0, 0.1, 0.25, 0.5, 1.0],
    "candidate_count": [1, 4, 8, 16, 32],
    "shortlist_size": [1, 4, 8, 16, 32],
    "plan_length": [1, 2, 3, 4],
    "original_frontier": [0, 1, 2, 4, 8, 16, 64],
    "best_alternative_frontier": [0, 1, 2, 4, 8, 16, 64],
    "best_alternative_length_delta": [-1, 0, 1],
    "alternative_frontier_min": [0, 1, 2, 4, 8, 16],
    "path_logprob": [-16, -8, -4, -2, -1, -0.5, -0.1],
}
DEFAULT_FIT = {
    "seed": 0,
    "bootstrap_seed": 0,
    "resamples": 10_000,
    "random_seeds": [0, 1, 2, 3, 4],
    "max_extra_call_fraction": 0.5,
    "minimum_retained_gain": 0.9,
    "delivery": "engine",
}
LEXICAL_SELECTION_THRESHOLDS = sorted({*THRESHOLDS["lexical_margin"], 0.75})
"""Include the brainstorm's direct-selection threshold without changing the trigger grid."""


def _lines(path: Path):
    with path.open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{number}: {error}") from error


def _index(path: Path, project) -> dict:
    result = {}
    for row in _lines(path):
        qid = row.get("id")
        if not isinstance(qid, str) or not qid or qid in result:
            raise ValueError(f"{path}: missing or duplicate question id {qid!r}")
        result[qid] = project(row)
    return result


def _same_ids(actual, expected, context):
    if set(actual) != set(expected):
        missing, extra = sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))
        raise ValueError(f"{context}: missing ids {missing[:3]}, extra ids {extra[:3]}")


def _tokens(record: dict) -> int | None:
    from xgraph.cost import request_cost

    usage = request_cost(record)
    if usage["model_calls"] != 1:
        raise ValueError(f"question {record['id']!r} needs single-call generation artifacts")
    prompt, generated = usage["prompt_tokens"], usage["generated_tokens"]
    return prompt + generated if prompt is not None and generated is not None else None


def _generation(record: dict) -> dict:
    return {
        "text": record_text(record),
        "source_text": record["text"],
        "replay_failed": replay_failure(record),
        "tokens": _tokens(record),
        "path_logprob": record.get("path_logprob"),
    }


def _metadata(questions: Path, splits: dict[str, Path]) -> dict:
    from xgraph.mintqa_bench import relation_clusters
    from xgraph.repair import hops

    asked = json.loads(questions.read_text(encoding="utf-8"))
    wanted = {q["id"]: q for q in asked}
    if not wanted or len(wanted) != len(asked):
        raise ValueError("questions must be nonempty and have unique ids")
    result = {}
    for split, path in splits.items():
        for k, row in enumerate(_lines(path)):
            qid = f"mintqa-{split}-{k}"
            if qid in wanted:
                result[qid] = {
                    "gold": row["answer"],
                    "split": split,
                    "cluster": relation_clusters({qid: row})[qid],
                    "hops": hops(wanted[qid].get("complexity", "")),
                    "question": wanted[qid]["utterance"],
                }
    _same_ids(result, wanted, "dataset gold")
    return {q["id"]: result[q["id"]] for q in asked}


def cluster_folds(clusters, seed: int) -> dict[str, int]:
    """Shuffle unique relation sequences, then alternate them between two halves."""
    labels = sorted(set(clusters))
    if len(labels) < 2:
        raise ValueError("cross-fitting needs at least two relation-sequence clusters")
    random.Random(seed).shuffle(labels)
    return {label: k % 2 for k, label in enumerate(labels)}


def _load_family(spec: dict, metadata: dict) -> list[dict]:
    from xgraph.mintqa_bench import answer_contains
    from xgraph.prompts import final_answer
    from xgraph.repair import chosen_record, merged_records, select, shortlist_scores

    missing = set(FAMILY_INPUTS) - set(spec)
    if missing:
        raise ValueError(f"repair family lacks inputs: {sorted(missing)}")
    original = _index(spec["original"], _generation)
    requests = _index(
        spec["requests"], lambda r: {k: r[k] for k in ("shown", "shortlist") if k in r}
    )
    verdicts = _index(spec["verdicts"], _generation)
    selected = _index(spec["selected"], lambda r: {"text": r["text"], "repair": r["repair"]})
    engine = _index(spec["engine"], lambda r: {"text": r["text"], "repair": r["repair"]})
    deliveries = _index(spec["delivery_requests"], lambda r: r)
    generated = _index(spec["delivery_outputs"], lambda r: r)
    for name, table in (("original", original), ("selected", selected), ("engine", engine)):
        _same_ids(table, metadata, str(spec[name]))
    _same_ids(verdicts, requests, "verifier outputs versus requests")
    _same_ids(generated, deliveries, "delivery outputs versus requests")
    rows, seen, used_deliveries = [], set(), set()
    for record in _lines(spec["candidates"]):
        qid = record.get("id")
        if qid not in metadata or qid in seen:
            raise ValueError(f"{spec['candidates']}: unexpected or duplicate question {qid!r}")
        seen.add(qid)
        meta, own = metadata[qid], original[qid]
        if record["text"] != own["source_text"] or record["question"] != meta["question"]:
            raise ValueError(f"question {qid!r}: candidates disagree with original/question")
        request = requests.get(qid)
        candidates = record["candidates"]
        if bool(request) != (len(candidates) >= 2):
            raise ValueError(f"question {qid!r}: verifier request eligibility disagrees")
        values = features(record, request["shown"] if request else None, own)
        expected = select(
            [record],
            {qid: request} if request else {},
            {qid: verdicts[qid]["text"]} if request else {},
        )[0]
        if any(expected[k] != selected[qid][k] for k in ("text", "repair")):
            raise ValueError(f"question {qid!r}: cached selected record disagrees with verifier")
        delivered = qid in deliveries
        if delivered != (expected["repair"]["chosen"] != 0):
            raise ValueError(f"question {qid!r}: delivery requests disagree with selected plan")
        if delivered:
            from xgraph.repair import plan_text

            span = plan_text(record["start"], expected["repair"]["relations"])
            if deliveries[qid]["messages"][-1]["content"] != span:
                raise ValueError(f"question {qid!r}: delivery prefill disagrees with selected plan")
            used_deliveries.add(qid)
        merged = merged_records(
            [expected],
            {qid: deliveries[qid]} if delivered else {},
            {qid: generated[qid]} if delivered else {},
        )[0]
        if any(merged[k] != engine[qid][k] for k in ("text", "repair")):
            raise ValueError(f"question {qid!r}: cached engine merge disagrees with delivery")
        shown = request["shown"] if request else list(range(len(candidates)))
        scores = shortlist_scores(record)
        lexical = max(shown, key=lambda k: (scores[k], -k)) if shown else 0
        lexical_text = record_text(chosen_record(record, lexical, "lexical"))
        lexical_engine = (
            own["text"]
            if lexical == 0
            else record_text(merged)
            if lexical == expected["repair"]["chosen"]
            else None
        )
        texts = {
            "original": own["text"],
            "executor": record_text(expected),
            "engine": record_text(merged),
            "lexical_executor": lexical_text,
            "lexical_engine": lexical_engine,
        }
        right = {
            k: (not own["replay_failed"] and answer_contains(final_answer(t), meta["gold"]))
            if t is not None
            else None
            for k, t in texts.items()
        }
        rows.append(
            {
                "id": qid,
                "cluster": meta["cluster"],
                "split": meta["split"],
                "hops": meta["hops"],
                "features": values,
                "eligible": request is not None,
                "right": right,
                "texts": texts,
                "delivery": delivered,
                "lexical_choice": lexical,
                "chosen": expected["repair"]["chosen"],
                "nonoriginal_candidate_zero": bool(
                    candidates and candidates[0]["kind"] != "original"
                ),
                "tokens": {
                    "original": own["tokens"],
                    "verifier": verdicts[qid]["tokens"] if request else 0,
                    "delivery": _tokens(generated[qid]) if delivered else 0,
                },
            }
        )
    _same_ids(seen, metadata, str(spec["candidates"]))
    _same_ids(used_deliveries, deliveries, "used delivery requests")
    order = {qid: k for k, qid in enumerate(metadata)}
    return sorted(rows, key=lambda r: order[r["id"]])


def fit_rule(
    rows: list[dict], config: dict, *, lexical_only: bool = False, lexical_selection: bool = False
) -> dict:
    """Maximize training net correctness under the call cap; break ties by cost,
    number of thresholds, then canonical rule JSON. Search space is fixed in code.
    """
    delivery = config["delivery"]
    outcome = "lexical_executor" if lexical_selection else delivery
    eligible = np.array([r["eligible"] for r in rows], dtype=bool)
    delta = np.array([int(r["right"][outcome]) - int(r["right"]["original"]) for r in rows])
    calls = np.array(
        [int(r["eligible"]) + int(delivery == "engine" and r["delivery"]) for r in rows]
    )
    if lexical_selection:
        calls[:] = 0  # executor delivery from the cached candidate needs no model call
    budget = calls.sum() * config["max_extra_call_fraction"]
    best_rule, best_key, best_gain, best_calls = {"constant": False}, None, 0, 0

    def consider(rule, mask, complexity):
        nonlocal best_rule, best_key, best_gain, best_calls
        cost = int(calls[mask].sum())
        if cost > budget:
            return
        gain = int(delta[mask].sum())
        key = (-gain, cost, complexity, json.dumps(rule, sort_keys=True))
        if best_key is None or key < best_key:
            best_rule, best_key, best_gain, best_calls = rule, key, gain, cost

    consider({"constant": False}, np.zeros(len(rows), dtype=bool), 0)
    consider({"constant": True}, eligible, 0)
    atoms = []
    for feature, thresholds in THRESHOLDS.items():
        if lexical_only and feature != "lexical_margin":
            continue
        if lexical_selection and feature == "lexical_margin":
            thresholds = LEXICAL_SELECTION_THRESHOLDS
        values = np.array(
            [r["features"][feature] if r["features"][feature] is not None else np.nan for r in rows]
        )
        for op, threshold in itertools.product((">", ">=", "<", "<="), thresholds):
            condition = {"feature": feature, "op": op, "threshold": threshold}
            # Missing values compare false for all four operators.
            mask = eligible & OPERATORS[op](values, threshold)
            rule = {"operator": "and", "conditions": [condition]}
            consider(rule, mask, 1)
            atoms.append((condition, mask))
    if not lexical_only:
        for (first, a), (second, b) in itertools.combinations(atoms, 2):
            if first["feature"] == second["feature"]:
                continue
            for join, mask in (("and", a & b), ("or", a | b)):
                consider({"operator": join, "conditions": [first, second]}, mask, 2)
    return {
        "rule": validate_rule(best_rule),
        "training_questions": len(rows),
        "training_clusters": len({r["cluster"] for r in rows}),
        "training_net_correct": best_gain,
        "training_full_net_correct": int(delta.sum()),
        "training_extra_calls": best_calls,
        "training_full_extra_calls": int(calls.sum()),
        "objective": "maximum net correct under call cap; ties: calls, thresholds, JSON",
    }


def _crossfit(
    rows: list[dict], folds: dict, config: dict, *, lexical_only=False, lexical_selection=False
):
    mask = [False] * len(rows)
    fits = {}
    for fold in (0, 1):
        training = [r for r in rows if folds[r["cluster"]] != fold]
        fit = fit_rule(
            training, config, lexical_only=lexical_only, lexical_selection=lexical_selection
        )
        fits[str(fold)] = {"trained_fold": 1 - fold, **fit}
        for k, row in enumerate(rows):
            if folds[row["cluster"]] == fold:
                mask[k] = row["eligible"] and triggered(row["features"], fit["rule"])
    return mask, fits


def _random_mask(rows, folds, reference, seed):
    chosen = set()
    for fold in (0, 1):
        eligible = sorted(r["id"] for r in rows if r["eligible"] and folds[r["cluster"]] == fold)
        count = sum(
            active and folds[r["cluster"]] == fold
            for r, active in zip(rows, reference, strict=True)
        )
        chosen.update(random.Random(f"{seed}:{fold}").sample(eligible, count))
    return [r["id"] in chosen for r in rows]


def _measure(rows, mask, delivery, config, *, lexical=False):
    from xgraph.score import paired_bootstrap

    n = len(rows)
    own = [int(r["right"]["original"]) for r in rows]
    if lexical:
        answers = [
            r["right"][f"lexical_{delivery}"] if active else r["right"]["original"]
            for r, active in zip(rows, mask, strict=True)
        ]
        missing = sum(a is None for a in answers)
        if missing:
            return {
                "status": "unavailable",
                "missing_questions": missing,
                "cached_questions": n - missing,
                "reason": "No cached engine delivery for these lexical choices; executor only.",
            }
        verified = [False] * n
        prefills = [
            active and delivery == "engine" and r["lexical_choice"] != 0
            for r, active in zip(rows, mask, strict=True)
        ]
    else:
        verified = [bool(active and r["eligible"]) for r, active in zip(rows, mask, strict=True)]
        answers = [
            int(r["right"][delivery] if active else r["right"]["original"])
            for r, active in zip(rows, verified, strict=True)
        ]
        prefills = [
            active and delivery == "engine" and r["delivery"]
            for r, active in zip(rows, verified, strict=True)
        ]
    extra = sum(verified) + sum(prefills)
    full_calls = sum(r["eligible"] + (delivery == "engine" and r["delivery"]) for r in rows)
    full_gain = sum(int(r["right"][delivery]) - int(r["right"]["original"]) for r in rows)
    net = int(sum(answers) - sum(own))
    tokens = []
    for r, verify, prefill in zip(rows, verified, prefills, strict=True):
        stages = [r["tokens"]["original"]]
        if verify:
            stages.append(r["tokens"]["verifier"])
        if prefill:
            stages.append(r["tokens"]["delivery"])
        tokens.append(sum(stages) if all(v is not None for v in stages) else None)
    bootstrap = {
        "clusters": [r["cluster"] for r in rows],
        "resamples": config["resamples"],
        "seed": config["bootstrap_seed"],
    }
    result = {
        "status": "exact_cached_replay",
        "questions": n,
        "clusters": len(set(bootstrap["clusters"])),
        "correct": int(sum(answers)),
        "accuracy": sum(answers) / n,
        "accuracy_interval": paired_bootstrap(answers, [0] * n, **bootstrap),
        "gain": paired_bootstrap(answers, own, **bootstrap),
        "gains": sum(a and not b for a, b in zip(answers, own, strict=True)),
        "losses": sum(b and not a for a, b in zip(answers, own, strict=True)),
        "retained_full_gain": net / full_gain if full_gain > 0 else None,
        "verifier_calls": sum(verified),
        "verified_share": sum(verified) / n,
        "delivery_prefills": sum(prefills),
        "delivery_prefill_share": sum(prefills) / n,
        "extra_calls": extra,
        "extra_calls_per_question": extra / n,
        "extra_call_fraction_of_full": extra / full_calls if full_calls else None,
        "tokens_known_questions": sum(t is not None for t in tokens),
        "tokens_per_question": sum(tokens) / n if all(t is not None for t in tokens) else None,
    }
    for group in ("split", "hops"):
        table = {}
        for label in sorted({r[group] for r in rows}):
            indices = [k for k, r in enumerate(rows) if r[group] == label]
            table[label] = {
                "questions": len(indices),
                "accuracy": sum(answers[k] for k in indices) / len(indices),
                "gain": sum(answers[k] - own[k] for k in indices) / len(indices),
                "verifier_calls": sum(verified[k] for k in indices),
                "delivery_prefills": sum(prefills[k] for k in indices),
            }
        result[f"by_{group}"] = table
    return result


def _evaluate_family(name, spec, metadata, folds, config, outdir):
    rows = _load_family(spec, metadata)
    print(f"{name}: audited {len(rows)} selected records and engine merges", flush=True)
    selective, fits = _crossfit(rows, folds, config)
    lexical_trigger, lexical_fits = _crossfit(rows, folds, config, lexical_only=True)
    lexical_selection, lexical_selection_fits = _crossfit(
        rows, folds, config, lexical_only=True, lexical_selection=True
    )
    frozen = fit_rule(rows, config)
    masks = {
        "path_mode": [False] * len(rows),
        "full": [r["eligible"] for r in rows],
        "selective": selective,
        "lexical_trigger": lexical_trigger,
        "lexical_only": [True] * len(rows),
        "lexical_only_fitted": lexical_selection,
    }
    masks.update(
        {
            f"random_{seed}": _random_mask(rows, folds, selective, seed)
            for seed in config["random_seeds"]
        }
    )
    # Fixed brainstorm thresholds are diagnostics, never used to select the fitted rule.
    for threshold in (0.0, 0.5, 0.75):
        masks[f"margin_gt_{threshold:g}"] = [
            r["eligible"]
            and r["features"]["lexical_margin"] is not None
            and r["features"]["lexical_margin"] > threshold
            for r in rows
        ]
        masks[f"lexical_only_gt_{threshold:g}"] = masks[f"margin_gt_{threshold:g}"]
    policies = {
        policy: {
            delivery: _measure(
                rows, mask, delivery, config, lexical=policy.startswith("lexical_only")
            )
            for delivery in ("executor", "engine")
        }
        for policy, mask in masks.items()
    }
    primary = policies["selective"][config["delivery"]]
    development_go = (
        primary["retained_full_gain"] is not None
        and primary["retained_full_gain"] >= config["minimum_retained_gain"]
        and primary["extra_call_fraction_of_full"] <= config["max_extra_call_fraction"]
    )
    for delivery in ("executor", "engine"):
        path = outdir / f"{name}-selective-{delivery}.jsonl"
        with path.open("w", encoding="utf-8") as out:
            for row, active in zip(rows, selective, strict=True):
                text = row["texts"][delivery if active else "original"]
                out.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "arm": "selective",
                            "text": text,
                            "selective": {
                                "verified": active,
                                "delivery": delivery,
                                "fold": folds[row["cluster"]],
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    with (outdir / f"{name}-features.jsonl").open("w", encoding="utf-8") as out:
        for row, active in zip(rows, selective, strict=True):
            kept = {k: v for k, v in row.items() if k != "texts"}
            kept.update(fold=folds[row["cluster"]], verified=active)
            out.write(json.dumps(kept, ensure_ascii=False) + "\n")
    result = {
        "policies": policies,
        "crossfit": fits,
        "lexical_crossfit": lexical_fits,
        "lexical_selection_crossfit": lexical_selection_fits,
        "development_go": development_go,
        "primary_delivery": config["delivery"],
        "feature_available_questions": {
            f: sum(r["features"][f] is not None for r in rows) for f in THRESHOLDS
        },
        "audit": {
            "validated_selected": len(rows),
            "validated_merges": len(rows),
            "eligible": sum(r["eligible"] for r in rows),
            "delivery_prefills": sum(r["delivery"] for r in rows),
            "nonoriginal_candidate_zero": sum(r["nonoriginal_candidate_zero"] for r in rows),
        },
    }
    x = spec.get("x_control", {})
    if set(FAMILY_INPUTS) <= set(x):
        result["x_control"], x_frozen = _evaluate_family(
            f"{name}-X", x, metadata, folds, config, outdir
        )
        result["x_control"]["status"] = "same_selector_opportunity"
        frozen["x_control"] = x_frozen
    else:
        result["x_control"] = {
            "status": "unavailable",
            "missing_inputs": sorted(set(FAMILY_INPUTS) - set(x)),
            "reason": (
                "X needs its own candidates, verifier choices and engine deliveries; "
                "P's choices cannot substitute."
            ),
        }
        if "original" in x:
            from xgraph.mintqa_bench import answer_contains
            from xgraph.prompts import final_answer

            originals = _index(x["original"], record_text)
            _same_ids(originals, metadata, "X baseline")
            result["x_control"]["baseline_accuracy"] = sum(
                answer_contains(final_answer(originals[q]), m["gold"]) for q, m in metadata.items()
            ) / len(metadata)
    print(
        f"{name}: development go={development_go}, verified={primary['verified_share']:.4f}",
        flush=True,
    )
    return result, frozen


def evaluate(manifest_path: Path, out: Path, freeze: Path) -> dict:
    """Evaluate a manifest and write a frozen all-development rule per family.

    Reported effects use only out-of-fold decisions. Bootstrap intervals condition
    on these fits, with the same paired cluster resampler as mintqa_bench.
    """
    from xgraph.cost import StageTimer
    from xgraph.provenance import provenance

    timer = StageTimer()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = DEFAULT_FIT | manifest.get("fit", {})
    if (
        config["delivery"] not in ("executor", "engine")
        or config["resamples"] < 1
        or not 0 < config["max_extra_call_fraction"] <= 1
        or not 0 < config["minimum_retained_gain"] <= 1
        or not config["random_seeds"]
        or len(set(config["random_seeds"])) != len(config["random_seeds"])
    ):
        raise ValueError(f"invalid selective fit configuration: {config}")
    if set(config) != set(DEFAULT_FIT):
        raise ValueError(f"unknown selective fit keys: {sorted(set(config) - set(DEFAULT_FIT))}")
    inputs = [manifest_path.resolve()]

    def resolved(value):
        path = Path(value)
        path = (manifest_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
        inputs.append(path)
        for suffix in (".summary.json", ".provenance.json"):
            sidecar = Path(f"{path}{suffix}")
            if sidecar.is_file():
                inputs.append(sidecar)
        return path

    def family_paths(spec):
        return {k: family_paths(v) if k == "x_control" else resolved(v) for k, v in spec.items()}

    questions = resolved(manifest["questions"])
    splits = {s: resolved(p) for s, p in manifest["splits"].items()}
    families = {name: family_paths(spec) for name, spec in manifest["families"].items()}
    if not families or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in families):
        raise ValueError("family names must be nonempty filename-safe identifiers")
    names = list(families)
    names += [f"{name}-X" for name, spec in families.items() if "x_control" in spec]
    outputs = [out, freeze]
    outputs += [
        out.parent / f"{name}-{suffix}.jsonl"
        for name in names
        for suffix in ("selective-executor", "selective-engine", "features")
    ]
    if len({p.resolve() for p in outputs}) != len(outputs) or any(
        p.resolve() in inputs for p in outputs
    ):
        raise ValueError("selective outputs must differ from each other and from inputs")
    metadata = _metadata(questions, splits)
    folds = cluster_folds((m["cluster"] for m in metadata.values()), config["seed"])
    out.parent.mkdir(parents=True, exist_ok=True)
    freeze.parent.mkdir(parents=True, exist_ok=True)
    results, rules = {}, {}
    for name, spec in families.items():
        results[name], rules[name] = _evaluate_family(
            name, spec, metadata, folds, config, out.parent
        )
        if "x_control" in rules[name]:
            rules[f"{name}-X"] = rules[name].pop("x_control")
    origin = provenance(list(dict.fromkeys(inputs)))
    report = {
        "artifact_kind": "xgraph.repair.selective",
        "version": 1,
        "feature_version": FEATURE_VERSION,
        "fit": config,
        "threshold_grid": THRESHOLDS,
        "lexical_selection_thresholds": LEXICAL_SELECTION_THRESHOLDS,
        "cluster_folds": folds,
        "families": results,
        "execution": {
            "cpu_only": True,
            "nice": os.getpriority(os.PRIO_PROCESS, 0),
            "address_space_limit_bytes": list(resource.getrlimit(resource.RLIMIT_AS)),
        },
        "provenance": origin,
        "uncertainty": (
            "95% paired relation-sequence cluster bootstrap, conditional on cross-fitted rules; "
            "no bootstrap refitting"
        ),
        "scope": (
            "Offline replay: saved verifier prompts unchanged; "
            "no serving, transfer or matched-X claim without its artifacts."
        ),
        **timer.summary(),
    }
    frozen = {
        "artifact_kind": "xgraph.selective.rules",
        "version": 1,
        "feature_version": FEATURE_VERSION,
        "families": rules,
        "fit": config,
        "threshold_grid": THRESHOLDS,
        "provenance": origin,
        "scope": (
            "All-development refit for subsequent confirmation; "
            "training accuracy is not an out-of-fold estimate."
        ),
    }
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    freeze.write_text(json.dumps(frozen, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report
