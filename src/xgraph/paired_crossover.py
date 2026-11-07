"""Frozen paired crossover fits on scored MINTQA pairs; CPU only, entirely post hoc."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

FAMILIES = ("qwen38", "gemma31")
MODES = ("constant", "slope", "status")
PANELS = ("analysis-dev", "validation")
STATUS = "POST HOC; development fit and retrospective validation transport, not confirmation"
WORKTREE = Path(__file__).resolve().parents[2]


def verified_file(path, expected):
    """Stream a frozen input hash before interpreting its contents."""
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    size = path.stat().st_size
    if digest != expected["sha256"] or size != expected["bytes"]:
        raise ValueError(f"input hash mismatch: {path}")
    return {"path": str(path), "sha256": digest, "bytes": size}


def validate_rows(rows):
    """Fail on malformed, duplicate, unpaired or inconsistent panel metadata."""
    seen, panels, clusters = set(), {}, {}
    for row in rows:
        context = f"{row.get('panel')}/{row.get('family')}/{row.get('id')}"
        if row["family"] not in FAMILIES:
            raise ValueError(f"{context}: unknown family")
        if row["panel"] not in PANELS:
            raise ValueError(f"{context}: unknown panel")
        if type(row["hops"]) is not int or not 1 <= row["hops"] <= 4:
            raise ValueError(f"{context}: hops must be integers in 1..4")
        if not math.isfinite(row["weight"]) or row["weight"] <= 0:
            raise ValueError(f"{context}: weight must be finite and positive")
        if not row["id"] or not row["cluster"] or not isinstance(row["knowledge"], str):
            raise ValueError(f"{context}: id, cluster and knowledge metadata are required")
        for arm in ("WT", "agent"):
            correct = row[arm]["correct"]
            if type(correct) not in (bool, int) or correct not in (0, 1):
                raise ValueError(f"{context}: {arm} correctness must be binary")
        if type(row["agent"]["budget_spent"]) is not bool:
            raise ValueError(f"{context}: budget_spent must be a boolean")
        key = row["panel"], row["family"], row["id"]
        if key in seen:
            raise ValueError(f"{context}: duplicate pair")
        seen.add(key)
        metadata = tuple(row[k] for k in ("hops", "cluster", "weight", "knowledge"))
        panels.setdefault((row["panel"], row["family"]), {})[row["id"]] = metadata
        cluster = row["panel"], row["cluster"]
        if clusters.setdefault(cluster, row["hops"]) != row["hops"]:
            raise ValueError(f"{context}: a relation-sequence cluster spans hops")
    for panel in PANELS:
        a, b = (panels.get((panel, f)) for f in FAMILIES)
        if not a or a != b:
            raise ValueError(f"{panel}: families must have aligned IDs and survey metadata")


def design(rows, mode, *, shared=False):
    """Only discordant pairs enter the conditional direction likelihood."""
    if mode not in MODES:
        raise ValueError(f"unknown paired model: {mode!r}")
    x, y, weights = [], [], []
    for row in rows:
        difference = row["WT"]["correct"] - row["agent"]["correct"]
        if not difference:
            continue
        features = [float(row["family"] == f) for f in FAMILIES] if shared else [1.0]
        if mode in ("slope", "status"):
            features.append(row["hops"] - 1)
        if mode == "status":
            features.extend(
                float(row["knowledge"] == value) for value in ("all_new", "mixed_old_new")
            )
        x.append(features)
        y.append(float(difference > 0))
        weights.append(row["weight"])
    if not x:
        raise ValueError("paired direction requires discordant pairs")
    return np.asarray(x), np.asarray(y), np.asarray(weights)


def fit_logistic(x, y, weights, *, max_iterations=80, diagnostics=None):
    """Survey-weighted Newton fit with the scratch analysis's fixed numerical ridge."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if (
        not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or weights.sum() <= 0
    ):
        raise ValueError("logistic design and weights must be finite with positive weight mass")
    weights = weights / weights.sum() * len(weights)
    beta = np.zeros(x.shape[1])
    ridge = 1e-9

    def loss(value):
        eta = x @ value
        return float(
            np.sum(weights * (np.logaddexp(0, eta) - y * eta)) + ridge * np.dot(value, value) / 2
        )

    old = loss(beta)
    converged = False
    for _iteration in range(max_iterations):
        p = 1 / (1 + np.exp(-np.clip(x @ beta, -35, 35)))
        gradient = x.T @ (weights * (p - y)) + ridge * beta
        hessian = x.T @ ((weights * p * (1 - p))[:, None] * x) + np.eye(len(beta)) * ridge
        step = np.linalg.solve(hessian, gradient)
        rate = 1.0
        while rate > 1e-8:
            trial = beta - rate * step
            new = loss(trial)
            if new <= old + 1e-10:
                break
            rate *= 0.5
        if np.max(np.abs(rate * step)) < 1e-8:
            converged = True
            break
        beta, old = trial, new
    if diagnostics is not None:
        p = 1 / (1 + np.exp(-np.clip(x @ beta, -35, 35)))
        score = x.T @ (weights * (p - y)) + ridge * beta
        diagnostics.append(
            {
                "converged": converged,
                "iterations": _iteration + 1,
                "score_inf_norm": float(np.max(np.abs(score))),
            }
        )
    # The scratch estimator returns its last iterate at the fixed limit. Preserve
    # that estimator for reproduction, but expose every such event in reports.
    return beta


def fit_direction(rows, mode="slope", *, shared=False, weights=None, diagnostics=None):
    x, y, supplied = design(rows, mode, shared=shared)
    if weights is not None:
        discordant = [r["WT"]["correct"] != r["agent"]["correct"] for r in rows]
        supplied = np.asarray(weights)[discordant]
    status = [] if diagnostics is not None else None
    beta = fit_logistic(x, y, supplied, diagnostics=status)
    if diagnostics is not None:
        diagnostics.append(
            {
                **status[0],
                "mode": mode,
                "family": "shared" if shared else rows[0].get("family"),
                "panel": rows[0].get("panel"),
                "resample": weights is not None,
            }
        )
    return beta


def direction_scores(rows, mode, beta, *, shared=False):
    x, y, weights = design(rows, mode, shared=shared)
    eta = x @ beta
    p = 1 / (1 + np.exp(-np.clip(eta, -35, 35)))
    return {
        "log_loss": float(np.average(np.logaddexp(0, eta) - y * eta, weights=weights)),
        "brier": float(np.average((p - y) ** 2, weights=weights)),
    }


def transport(development, validation, mode="slope", *, shared=False, diagnostics=None):
    """Fit exclusively on development, then score discordant validation pairs."""
    beta = fit_direction(development, mode, shared=shared, diagnostics=diagnostics)
    return {
        "beta": beta.tolist(),
        "validation_direction_scores": direction_scores(validation, mode, beta, shared=shared),
    }


def cluster_weights(rows, rng):
    """Resample relation sequences within hops, coupled across model families.

    Calibrate the sampled weights back to each hop's original combined mass,
    exactly as in the scratch fit. All pairs, including concordances, participate
    in calibration before the direction likelihood discards concordances.
    """
    weights = np.asarray([r["weight"] for r in rows], dtype=float)
    hops = np.asarray([r["hops"] for r in rows])
    names = sorted({(r["hops"], r["cluster"]) for r in rows})
    lookup = {key: i for i, key in enumerate(names)}
    indices = np.asarray([lookup[r["hops"], r["cluster"]] for r in rows])
    count = np.zeros(len(names))
    for hop in range(1, 5):
        ids = [i for i, key in enumerate(names) if key[0] == hop]
        chosen = rng.choice(ids, size=len(ids), replace=True)
        count += np.bincount(chosen, minlength=len(names))
    sampled = weights * count[indices]
    for hop in range(1, 5):
        mask = hops == hop
        if mask.any():
            sampled[mask] *= weights[mask].sum() / sampled[mask].sum()
    return sampled


def analyze(rows, *, bootstrap=1000, seed=25, shared_seed=26):
    """Port paired_fit.py and its shared-slope companion without arm selection."""
    validate_rows(rows)
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    rng = np.random.default_rng(seed)
    diagnostics = []
    result = {
        "model": "P(WT_only | discordant,h,f)=sigmoid(a_f+b*(h-1)); separate-family sensitivities",
        "status": STATUS,
        "families": {},
    }
    for family in FAMILIES:
        panels = {p: [r for r in rows if r["family"] == family and r["panel"] == p] for p in PANELS}
        record = {}
        for panel, sub in panels.items():
            models = {
                mode: transport(sub, panels["validation"], mode, diagnostics=diagnostics)
                for mode in MODES
            }
            coefficients, slopes, deltas = [], [], []
            for _ in range(bootstrap):
                sampled = cluster_weights(sub, rng)
                coefficients.append(
                    fit_direction(sub, "status", weights=sampled, diagnostics=diagnostics)
                )
                slopes.append(fit_direction(sub, "slope", weights=sampled, diagnostics=diagnostics))
                differences = np.asarray([r["WT"]["correct"] - r["agent"]["correct"] for r in sub])
                deltas.append(
                    [
                        np.average(differences[mask], weights=sampled[mask])
                        for hop in range(1, 5)
                        for mask in [np.asarray([r["hops"] == hop for r in sub])]
                    ]
                )
            models["status"]["bootstrap_beta_ci"] = np.quantile(
                coefficients, [0.025, 0.975], axis=0
            ).T.tolist()
            models["slope"]["bootstrap_beta_ci"] = np.quantile(
                slopes, [0.025, 0.975], axis=0
            ).T.tolist()
            values = np.asarray(slopes)
            models["slope"]["bootstrap_crossing_ci"] = np.quantile(
                1 - values[:, 0] / values[:, 1], [0.025, 0.975]
            ).tolist()
            models["slope"]["bootstrap_odds_4_vs_2_ci"] = np.quantile(
                np.exp(2 * values[:, 1]), [0.025, 0.975]
            ).tolist()
            summary = {}
            for hop in range(1, 5):
                selected = [r for r in sub if r["hops"] == hop]
                mass = sum(r["weight"] for r in selected)
                budget = [0.0, 0.0]
                for row in selected:
                    difference = row["WT"]["correct"] - row["agent"]["correct"]
                    budget[not row["agent"]["budget_spent"]] += row["weight"] * difference / mass
                summary[str(hop)] = {
                    "budget_net": budget[0],
                    "nonbudget_net": budget[1],
                    "delta_ci": np.quantile(deltas, [0.025, 0.975], axis=0)[:, hop - 1].tolist(),
                }
            record[panel] = {"models": models, "by_hop": summary}

        # Cluster folds are deterministic and depend on no outcomes.
        sub = panels["analysis-dev"]
        folds = {
            r["cluster"]: int(hashlib.sha256(r["cluster"].encode()).hexdigest(), 16) % 5
            for r in sub
        }
        cv = {}
        for mode in MODES:
            scores = []
            for k in range(5):
                train = [r for r in sub if folds[r["cluster"]] != k]
                test = [r for r in sub if folds[r["cluster"]] == k]
                fitted = transport(train, test, mode, diagnostics=diagnostics)
                _, _, weights = design(test, mode)
                scores.append((weights.sum(), fitted["validation_direction_scores"]))
            cv[mode] = {
                metric: float(sum(n * ev[metric] for n, ev in scores) / sum(n for n, _ in scores))
                for metric in ("log_loss", "brier")
            }
        record["dev_cluster_5fold"] = cv
        a, b = record["analysis-dev"]["models"]["slope"]["beta"]
        record["forecast"] = {
            "WT_only_odds_multiplication_1_to_4": float(np.exp(3 * b)),
            "WT_only_odds_multiplication_2_to_4": float(np.exp(2 * b)),
            "crossing_h": float(1 - a / b) if abs(b) > 1e-12 else None,
            "WT_only_probability_by_hop": [
                float(1 / (1 + np.exp(-(a + b * (h - 1))))) for h in range(1, 5)
            ],
        }
        result["families"][family] = record

    rng = np.random.default_rng(shared_seed)
    shared = {}
    validation = [r for r in rows if r["panel"] == "validation"]
    for panel in PANELS:
        sub = [r for r in rows if r["panel"] == panel]
        fitted = transport(sub, validation, shared=True, diagnostics=diagnostics)
        draws = [
            fit_direction(
                sub, shared=True, weights=cluster_weights(sub, rng), diagnostics=diagnostics
            )
            for _ in range(bootstrap)
        ]
        shared[panel] = {
            **fitted,
            "bootstrap_ci": np.quantile(draws, [0.025, 0.975], axis=0).T.tolist(),
            "bootstrap": (
                "same relation-sequence resample across both families; calibrated within hops"
            ),
            "interpretation": "two family intercepts and shared hop slope; descriptive only",
        }
    result["shared_paired_slope"] = shared
    groups = {}
    for row in diagnostics:
        kind = "bootstrap" if row["resample"] else "point_or_cv"
        key = f"{row['family']}/{row['panel']}/{row['mode']}/{kind}"
        group = groups.setdefault(
            key, {"fits": 0, "iteration_limit_hits": 0, "max_score_inf_norm": 0.0}
        )
        group["fits"] += 1
        group["iteration_limit_hits"] += not row["converged"]
        group["max_score_inf_norm"] = max(group["max_score_inf_norm"], row["score_inf_norm"])
    result["solver_diagnostics"] = groups
    return result


def read_pairs(path):
    """Read scored pairs as data, retaining source order for exact bootstrap reproduction."""
    rows = []
    with Path(path).open() as stream:
        for line, text in enumerate(stream, 1):
            try:
                source = json.loads(text)
                row = {
                    k: source[k]
                    for k in ("panel", "family", "id", "hops", "cluster", "weight", "knowledge")
                }
                row["WT"] = {"correct": source["WT"]["correct"]}
                row["agent"] = {k: source["agent"][k] for k in ("correct", "budget_spent")}
                rows.append(row)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line}: invalid paired row: {error}") from error
    validate_rows(rows)
    return rows


def assert_reference(actual, expected, context="fit"):
    """Check every numeric scratch output rather than transcribing its findings."""
    if isinstance(expected, dict):
        for key, value in expected.items():
            assert_reference(actual[key], value, f"{context}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"reference length mismatch: {context}")
        for i, value in enumerate(expected):
            assert_reference(actual[i], value, f"{context}[{i}]")
    elif isinstance(expected, (float, int)) and not math.isclose(
        actual, expected, abs_tol=1e-7, rel_tol=1e-7
    ):
        raise ValueError(f"reference mismatch: {context}: {actual} != {expected}")


def file_identity(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=WORKTREE / "scripts/mintqa/paired-crossover-inputs.json"
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--out", type=Path, default=WORKTREE / "target/theory")
    parser.add_argument(
        "--dry-run", action="store_true", help="save CPU argv without opening paired or raw inputs"
    )
    args = parser.parse_args(argv)
    started = time.monotonic()
    out = args.out.resolve()
    if not out.is_relative_to((WORKTREE / "target").resolve()):
        raise ValueError("analysis outputs must stay inside this worktree's target directory")
    manifest = json.loads(args.manifest.read_text())
    if manifest["format"] != "xgraph.paired_crossover_inputs.v1":
        raise ValueError("unknown paired crossover input format")
    config = manifest["configuration"]
    fixed = {"folds": 5, "ridge": 1e-9, "newton_iterations": 80, "quantiles": [0.025, 0.975]}
    if any(config[k] != value for k, value in fixed.items()):
        raise ValueError("manifest differs from the frozen estimator configuration")
    source_root = (args.source_root or Path(manifest["source_root"])).resolve()
    command = [
        "nice",
        "-n",
        "19",
        "taskset",
        "-c",
        "1-8",
        "prlimit",
        "--as=4000000000",
        str(WORKTREE / ".venv/bin/python"),
        "-I",
        "-S",
        str(WORKTREE / "scripts/mintqa/paired-crossover.py"),
        "--manifest",
        str(args.manifest.resolve()),
        "--source-root",
        str(source_root),
        "--out",
        str(out),
    ]
    out.mkdir(parents=True, exist_ok=True)
    preview = {
        "command": command,
        "manifest": file_identity(args.manifest),
        "gpu_hours": 0,
        "status": STATUS,
    }
    write_json(out / "argv.json", preview)
    (out / "regenerate.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd "
        + shlex.quote(str(WORKTREE))
        + "\nexec "
        + shlex.join(command)
        + "\n"
    )
    if args.dry_run:
        print(json.dumps(preview, indent=2))
        return

    inputs, paths = [], {}
    for role, entry in manifest["inputs"].items():
        path = source_root / entry["path"]
        inputs.append(verified_file(path, entry))
        paths[role] = path
    upstream = json.loads(paths["provenance"].read_text())
    original_root = Path(manifest["source_root"])
    for original, expected in upstream["inputs"].items():
        relative = Path(original).relative_to(original_root)
        inputs.append(verified_file(source_root / relative, expected))
    rows = read_pairs(paths["pairs"])
    result = analyze(
        rows, bootstrap=config["bootstrap"], seed=config["seed"], shared_seed=config["shared_seed"]
    )
    reference = json.loads(paths["reference"].read_text())
    shared_reference = json.loads(paths["shared_reference"].read_text())
    assert_reference(result["families"], reference["families"], "paired-fit.families")
    assert_reference(
        result["shared_paired_slope"],
        shared_reference["shared_paired_slope"],
        "shared_paired_slope",
    )
    result["configuration"] = config
    result["interpretation"] = {
        "validation_transport": (
            "only analysis-dev fits are transported; "
            "validation refits are descriptive in-sample fits"
        ),
        "forecast": (
            "conditional discordance-direction scenarios from development; "
            "no target-domain uncertainty or confirmation"
        ),
        "budget": "post-treatment descriptive partition, not the causal effect of more tool calls",
        "crossing": (
            "percentile ratio intervals can cross a zero-slope pole; "
            "crossover location is poorly identified"
        ),
        "identity": (
            "all numeric family and shared-slope scratch outputs reproduced "
            "within fixed numerical tolerance"
        ),
        "solver": (
            "fixed-iteration scratch estimator retained; iteration-limit hits and "
            "score residuals are disclosed in solver_diagnostics"
        ),
    }
    write_json(out / "paired-fit.json", result)
    code = [Path(__file__), WORKTREE / "scripts/mintqa/paired-crossover.py", WORKTREE / "uv.lock"]
    provenance = {
        **preview,
        "actual_argv": sys.argv,
        "repo_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=WORKTREE, text=True
        ).strip(),
        "repo_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=WORKTREE, text=True
            ).strip()
        ),
        "code": [file_identity(path) for path in code],
        "inputs": inputs,
        "configuration": config,
        "upstream_commits": {k: upstream[k] for k in ("git_head", "sealed_checkout_head")},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "elapsed_s": time.monotonic() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "read_scope": (
            "hash-verified scored MINTQA pairs and their recorded raw/scorer inputs; "
            "scratch Python is hashed only, never imported"
        ),
    }
    write_json(out / "provenance.json", provenance)
    dev, val = (result["shared_paired_slope"][p]["beta"][-1] for p in PANELS)
    (out / "report.md").write_text(
        f"Post hoc paired crossover analysis: shared development slope {dev:.6f}; "
        f"descriptive validation refit {val:.6f}. Every numeric scratch fit and interval "
        "reproduced within the fixed tolerance. Only development coefficients score validation "
        "transport; this is retrospective, not independent confirmation. Forecast odds are "
        "conditional on discordance. Crossover locations remain poorly identified. Budget "
        "partitions are post-treatment descriptions. Solver limit hits remain explicit "
        "in paired-fit.json. No GPU work. Input hashes, versions, "
        "configuration, seeds, command and commit are in provenance.json; rerun regenerate.sh.\n"
    )
    outputs = ["paired-fit.json", "provenance.json", "argv.json", "regenerate.sh", "report.md"]
    (out / "SHA256SUMS").write_text(
        "".join(f"{file_identity(out / name)['sha256']}  {name}\n" for name in outputs)
    )
    print(
        json.dumps(
            {
                "out": str(out),
                "shared_slope_dev": dev,
                "shared_slope_validation_post_hoc": val,
                "gpu_hours": 0,
            }
        )
    )
