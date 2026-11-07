"""Prospective WT repair visibility study. Registration never authorizes execution."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import resource
import shlex
import signal
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from xgraph import menu_control as control
from xgraph.mintqa_panels import indexed, read_jsonl, write_json, write_jsonl
from xgraph.mintqa_rematch import AuditMenus, revision_records, revision_requests
from xgraph.provenance import provenance, sha256_file

REPO = Path(__file__).resolve().parents[2]
SOURCE = Path("/home/jonas/git/xgraph")
FAMILIES = tuple(control.FAMILIES)
ARMS = ("R", "L", "S", "N")
SEED = 17
EMBARGO = "2026-11-23T23:00:00+00:00"
METRICS = ("R-N_accuracy", "R-S_accuracy", "R-L_accuracy", "R-N_f1", "support_interaction")
LIMITS = (0.0, 0.0, -0.02, -0.02, 0.0)


def select_ids(ids, size, *, seed=SEED):
    """Outcome-blind sampling: ascending SHA256('17:'+exact ID), then exact ID."""
    ids = list(ids)
    if any(not isinstance(q, str) or not q for q in ids) or len(set(ids)) != len(ids):
        raise ValueError("selection requires unique nonempty string IDs")
    if type(size) is not int or not 1 <= size <= len(ids):
        raise ValueError("selection size outside the ID population")
    return sorted(ids, key=lambda q: (hashlib.sha256(f"{seed}:{q}".encode()).digest(), q))[:size]


def _statistics(rows):
    table = {(r["family"], r["id"]): r for r in rows}
    ids = sorted({r["id"] for r in rows})
    if not ids or len(table) != len(rows) or set(table) != {(f, q) for f in FAMILIES for q in ids}:
        raise ValueError("complete paired rows for both families are required")
    clusters = sorted({str(r["cluster"]) for r in rows})
    positions = {c: i for i, c in enumerate(clusters)}
    stats = np.zeros((len(clusters), len(FAMILIES), 9))
    stratum_clusters = []
    for j, family in enumerate(FAMILIES):
        strata = [set(), set()]
        for q in ids:
            row = table[family, q]
            if row["cluster"] != table[FAMILIES[0], q]["cluster"]:
                raise ValueError("paired families must share question dependence units")
            weight = row["weight"]
            if not math.isfinite(weight) or weight <= 0 or type(row["initial_support"]) is not bool:
                raise ValueError("invalid weight or pre-intervention support stratum")
            arms = row["arms"]
            if set(arms) != set(ARMS) or any(
                not math.isfinite(a[k]) or not 0 <= a[k] <= 1
                for a in arms.values()
                for k in ("accuracy", "f1")
            ):
                raise ValueError("incomplete or invalid arm metrics")
            d = arms["R"]["accuracy"] - arms["N"]["accuracy"]
            s = row["initial_support"]
            values = [
                d,
                arms["R"]["accuracy"] - arms["S"]["accuracy"],
                arms["R"]["accuracy"] - arms["L"]["accuracy"],
                arms["R"]["f1"] - arms["N"]["f1"],
                d * (not s),
                int(not s),
                d * s,
                int(s),
                1,
            ]
            stats[positions[str(row["cluster"])]][j] += weight * np.asarray(values)
            strata[int(s)].add(str(row["cluster"]))
        stratum_clusters.append([len(s) for s in strata])
    return stats, clusters, stratum_clusters


def _effects(stats):
    with np.errstate(divide="ignore", invalid="ignore"):
        ordinary = stats[..., :4] / stats[..., 8, None]
        interaction = stats[..., 4] / stats[..., 5] - stats[..., 6] / stats[..., 7]
    return np.concatenate([ordinary, interaction[..., None]], axis=-1)


def _evaluate(stats, *, resamples, rng, minimum_clusters=20, strata=None):
    n = len(stats)
    if resamples < 20:
        raise ValueError("at least twenty cluster resamples required")
    point = _effects(stats.sum(axis=0))
    # One draw preserves all arms, endpoints, strata and families. Undefined strata
    # count as failure, including inside the percentile distribution.
    draws = rng.multinomial(n, np.full(n, 1 / n), size=resamples)
    replicates = _effects(np.einsum("bc,cfk->bfk", draws, stats))
    replicates = np.where(np.isfinite(replicates), replicates, -np.inf)
    low = np.quantile(replicates, 0.025, axis=0, method="inverted_cdf")
    deleted = _effects(stats.sum(axis=0)[None] - stats)
    sensitivity = np.all(deleted[..., :2] > 0, axis=(0, 2)) if n > 1 else [False, False]
    enough = n >= minimum_clusters and (strata is None or min(map(min, strata)) >= 5)
    passes = np.all(np.isfinite(point)) and np.all(low > np.asarray(LIMITS))
    passes = passes and np.all(point[:, :2] >= 0.01) and np.all(sensitivity) and enough
    return point, low, bool(passes), deleted, bool(enough)


def reduce_rows(rows, *, resamples=10000, seed=SEED, minimum_clusters=20):
    """The frozen intersection-union gate; all five endpoints in both families."""
    stats, clusters, strata = _statistics(rows)
    point, low, passes, deleted, enough = _evaluate(
        stats,
        resamples=resamples,
        rng=np.random.default_rng(seed),
        minimum_clusters=minimum_clusters,
        strata=strata,
    )
    families = {}
    for j, family in enumerate(FAMILIES):
        families[family] = {
            metric: {
                "point": float(point[j, k]) if np.isfinite(point[j, k]) else None,
                "lower_975": float(low[j, k]) if np.isfinite(low[j, k]) else None,
                "passes": bool(
                    np.isfinite(low[j, k])
                    and low[j, k] > LIMITS[k]
                    and (k >= 2 or point[j, k] >= 0.01)
                ),
            }
            for k, metric in enumerate(METRICS)
        }
        families[family]["initial_stratum_clusters"] = strata[j]
        families[family]["leave_cluster_out_min_accuracy"] = [
            float(v) if np.isfinite(v) else None for v in np.min(deleted[:, j, :2], axis=0)
        ]
        largest = int(np.argmax(stats[:, j, 8]))
        families[family]["exclude_largest_topic_accuracy"] = [
            float(v) if np.isfinite(v) else None for v in deleted[largest, j, :2]
        ]
    return {
        "passes": passes,
        "status": "COMPLETE" if enough else "INCOMPLETE",
        "families": families,
        "clusters": len(clusters),
        "resamples": resamples,
        "seed": seed,
        "inference": "paired whole-cluster percentile lower .975; "
        "IUT across all endpoints/families",
    }


def joint_assurance(rows, target_cluster_sizes, *, trials=1000, resamples=999, seed=SEED):
    """Pilot uncertainty plus fresh target clusters; never multiply marginal powers.

    Topic-sized target clusters share a pilot-cluster mean outcome vector across all
    arms/families. Transport is an assumption, stress-tested under two symmetric
    Dirichlet concentrations. No target outcomes enter. The .975 Wilson lower bound
    for Monte Carlo uncertainty must reach .80 under BOTH priors.
    """
    stats, _, strata = _statistics(rows)
    sizes = np.asarray(target_cluster_sizes, dtype=float)
    if (
        sizes.ndim != 1
        or len(sizes) == 0
        or np.any(~np.isfinite(sizes))
        or np.any(sizes < 1)
        or np.any(sizes != np.floor(sizes))
        or sizes.sum() > 1024
    ):
        raise ValueError("target cluster sizes must be positive integers totalling at most 1024")
    if trials < 20:
        raise ValueError("at least twenty assurance trials required")
    means = stats / stats[:, :, 8, None]
    scenarios = []
    for prior in (0.5, 1.0):
        rng, successes = np.random.default_rng(seed), 0
        for _ in range(trials):
            probability = rng.dirichlet(np.full(len(stats), prior))
            which = rng.choice(len(stats), len(sizes), p=probability)
            future = means[which] * sizes[:, None, None]
            _, _, passed, _, _ = _evaluate(
                future,
                resamples=resamples,
                rng=rng,
                strata=strata,
            )
            successes += passed and len(stats) >= 20
        p, z = successes / trials, 1.959963984540054
        lower = (
            p
            + z * z / (2 * trials)
            - z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
        )
        lower /= 1 + z * z / trials
        scenarios.append(
            {
                "prior": prior,
                "successes": successes,
                "trials": trials,
                "assurance": p,
                "mc_lower_975": max(0.0, lower),
            }
        )
    return {
        "admitted": all(s["mc_lower_975"] >= 0.8 for s in scenarios),
        "scenarios": scenarios,
        "target_cluster_sizes": sizes.astype(int).tolist(),
        "resamples": resamples,
        "seed": seed,
        "limitations": "MINTQA-to-WebQSP joint outcome transport assumed; "
        "target effects unobserved",
    }


def _cpu(out, command, phase, family=None):
    return [
        "env",
        "TOKENIZERS_PARALLELISM=false",
        "RAYON_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "nice",
        "-n",
        "10",
        "prlimit",
        "--as=4000000000",
        str(REPO / ".venv/bin/python"),
        "-I",
        "-m",
        "xgraph.repair_visibility",
        command,
        "--out",
        str(out),
        "--phase",
        phase,
        *(["--family", family] if family else []),
    ]


def generation_command(out, phase, family, arm):
    directory = out / phase
    command = control.generation_command(REPO, SOURCE, directory, family, arm)
    replacements = {
        "UV_CACHE_DIR": str(SOURCE / "target/agent-uv-cache"),
        "SKBUILD_BUILD_DIR": str(REPO / "target/repair-visibility-build/gpu/{wheel_tag}"),
        "XG_KG_INDEX": str(directory / family / "index"),
        "XG_CONTAINER_NAME": f"xgraph-repair-visibility-{phase}-{family}-{arm}",
    }
    return [
        f"{s.split('=', 1)[0]}={replacements[s.split('=', 1)[0]]}"
        if "=" in s and s.split("=", 1)[0] in replacements
        else s
        for s in command
    ]


def phase_plan(out, phase):
    from xgraph.catalog_argv import expand_launcher, flags

    stages = []
    for family in FAMILIES:
        directory = out / phase / family
        stages.append(
            {
                "name": f"{phase}/{family}/prepare",
                "kind": "cpu",
                "command": _cpu(out, "prepare", phase, family),
                "outputs": [
                    str(directory / "preparation.json"),
                    *(
                        str(directory / p)
                        for p in (
                            "first.jsonl",
                            "requests.jsonl",
                            "candidates.jsonl",
                            "initial-support.jsonl",
                            "matching.jsonl",
                            "bound-stages.json",
                            "bound-dry-run.txt",
                        )
                    ),
                    *(str(directory / f"{a}-requests.jsonl") for a in ARMS),
                ],
            }
        )
        for arm in ARMS:
            command = generation_command(out, phase, family, arm)
            # Capture flags before any sealed index is exported. The child seal
            # replaces only this future index location, never execution settings.
            probe = [f"XG_KG_INDEX={REPO}" if s.startswith("XG_KG_INDEX=") else s for s in command]
            expanded = expand_launcher(probe)
            expanded[expanded.index("--kg-index") + 1] = str(directory / "index")
            output = directory / f"{arm}-raw.jsonl"
            name = f"xgraph-repair-visibility-{phase}-{family}-{arm}"
            stages.append(
                {
                    "name": f"{phase}/{family}/{arm}",
                    "kind": "gpu",
                    "command": command,
                    "expanded_argv": expanded,
                    "expected_flags": flags(expanded),
                    "outputs": [
                        str(output),
                        str(output) + ".summary.json",
                        str(output) + ".provenance.json",
                    ],
                    "gpu_events": str(output) + ".gpu-events.jsonl",
                    "wall_timeout_seconds": 24 * 3600,
                    "forecast_seconds": (112 if phase == "pilot" else 1794),
                    "inspect_command": ["docker", "container", "inspect", name],
                    "cleanup_command": ["docker", "rm", "--force", name],
                    "headroom_command": [
                        "env",
                        f"UV_CACHE_DIR={SOURCE / 'target/agent-uv-cache'}",
                        "UV_LINK_MODE=hardlink",
                        "UV_OFFLINE=1",
                        f"TMPDIR={REPO / 'target/tmp'}",
                        "XG_GPU_IDLE=1",
                        "XG_AVAILABLE_WAIT=43200",
                        "nice",
                        "-n",
                        "10",
                        str(REPO / "scripts/gpu-headroom.sh"),
                        "60" if family == "qwen38" else "75",
                    ],
                }
            )
        stages.append(
            {
                "name": f"{phase}/{family}/finish",
                "kind": "cpu",
                "command": _cpu(out, "finish", phase, family),
                "outputs": [
                    str(directory / "finish.json"),
                    *(str(directory / f"{a}.jsonl") for a in ARMS),
                ],
            }
        )
    stages.append(
        {
            "name": f"{phase}/score",
            "kind": "cpu",
            "command": _cpu(out, "score", phase),
            "outputs": [str(out / phase / "report.json"), str(out / phase / "rows.jsonl")],
        }
    )
    if phase == "pilot":
        stages.append(
            {
                "name": "pilot/admit",
                "kind": "cpu",
                "command": _cpu(out, "admit", phase),
                "outputs": [str(out / "pilot/admission.json")],
            }
        )
    return [{"study": "repair-visibility", **s} for s in stages]


def _stride_gate(score, receipt):
    """Hash and parse the same bytes; the receipt must bind the failed score.

    Operator receipts lack an embedded creation time. Their filesystem mtime
    records write completion, conservatively after creation, and is sealed too.
    """
    score, receipt = Path(score).resolve(), Path(receipt).resolve()
    pins, payloads = [], []
    for path in (score, receipt):
        payload = path.read_bytes()
        pins.append(
            {
                "path": str(path),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        payloads.append(json.loads(payload))
    if payloads[0].get("pass") is not False:
        raise ValueError("bound stride score must record pass false")
    if {"path": str(score), "sha256": pins[0]["sha256"]} not in payloads[1].get("outputs", []):
        raise ValueError("stride receipt does not bind this score hash and path")
    return {
        "score": pins[0],
        "receipt": pins[1],
        "receipt_created_at": datetime.fromtimestamp(receipt.stat().st_mtime, UTC).isoformat(),
        "receipt_time_source": "filesystem mtime (receipt completion)",
    }


def register(
    out,
    *,
    panels=None,
    webqsp_registration=None,
    parent_registration=None,
    stride_score=None,
    stride_receipt=None,
):
    """Freeze the scientific plan and ID-only pilot before any new study data.

    The later materialization seal binds owner-exported W histories; those outputs
    are deliberately neither opened nor hashed during prospective registration.
    """
    out = Path(out).resolve()
    if (out / "registration.json").exists():
        raise ValueError("repair visibility registration is immutable")
    if (stride_score is not None or stride_receipt is not None) and (
        stride_score is None or stride_receipt is None or parent_registration is None
    ):
        raise ValueError("stride amendment requires a parent registration, score and receipt")
    panels = panels or SOURCE / "target/mintqa-rematch/panels/panels.json"
    population = json.loads(panels.read_text())["panels"]["validation"]["members"]
    ids = [r["id"] for r in population]
    selected = select_ids(ids, min(64, len(ids)))
    files = [
        p
        for root in (REPO / "src", REPO / "csrc", REPO / "scripts")
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix in {".py", ".sh", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".txt"}
    ] + [
        REPO / p
        for p in (
            "scripts/generate.sh",
            "scripts/gpu-headroom.sh",
            "scripts/mintqa/menu-control-build.txt",
            "pyproject.toml",
            "uv.lock",
            "CMakeLists.txt",
            "csrc/engine.cu",
            "docs/repair-visibility-adr.md",
        )
    ]
    # Tests may register a synthetic population before the maintained ADR is written.
    if not (REPO / "docs/repair-visibility-adr.md").exists():
        files.remove(REPO / "docs/repair-visibility-adr.md")
    registration = {
        "artifact_kind": "xgraph.repair_visibility_registration",
        "version": 1,
        "approval": "UNAPPROVED",
        "created_at": datetime.now(UTC).isoformat(),
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "source_policy": "content hashes bind implementation; "
        "clean descendant commit required at run",
        "seed": SEED,
        "families": list(FAMILIES),
        "arms": list(ARMS),
        "pilot": {
            "population": control.identity(panels),
            "population_ids": sorted(ids),
            "ids": selected,
            "members": [
                {k: r[k] for k in ("id", "cluster", "weight") if k in r}
                for r in population
                if r["id"] in selected
            ],
            "selection": "ascending SHA256 UTF-8 '17:'+exact ID, exact ID ties; first 64",
            "annotation_access": "none: selection function accepts only unique exact string IDs",
            "existing_W": {
                f: str(
                    SOURCE
                    / ".claude/worktrees/codex-final/target/final-validation"
                    / f
                    / "validation/W.jsonl"
                )
                for f in FAMILIES
            },
            "reuse": "ONLY selected sealed W first calls plus original requests and provenance, "
            "after failed Nov 13 stride gate and hash-bound owner export decision; "
            "no other sealed outcomes, no existing repairs or no-suggestion second calls",
            "development_reuse": "KG store, writable relation indexes, tokenizers, panel metadata; "
            "completed menu-control development launch events/summary/ledger "
            "for cost planning only; no development first/second call reuse",
            "maximum_gpu_hours": 2,
            "window": "2026-11-14 through 2026-11-17",
        },
        "confirmation": {
            "dataset": "separate sealed WebQSP core ID inventory, never MINTQA IDs",
            "selection": "same ID hash prefix, min(1024,population); freeze owner-supplied "
            "question-only core inventory and topic clusters before Nov 23 release",
            "maximum_questions": 1024,
            "maximum_gpu_hours": 6,
            "history": "ONLY sealed W first-call histories; no new first calls",
            "not_before": EMBARGO,
            "materialization": "child source and cohort seal before any R/L/S/N generation; "
            "W history reads require explicit owner embargo release",
        },
        "matching": {
            "candidate_edits": 4,
            "shortlist": 8,
            "endpoint_limit": 20,
            "menu_limit": 8,
            "terminal_width": 8,
            "max_tokens": 256,
            "max_model_len": 16384,
            "relation_support": "frontier",
            "committed_paths": True,
            "start_rank": "sitelinks",
            "R": "AuditMenus question IDF/alias ranking over whole writable relation vocabulary; "
            "global frequency zero; score then numeric relation ID",
            "L": "same ranking on union of outgoing relations at every capped first-W path "
            "prefix frontier; aggregate frontier frequencies break score ties",
            "S": "ascending SHA256('repair-visibility:17:'+qid+':'+relation_id), lexical alias; "
            "without question, gold, R IDs, gold exclusions or redraws",
            "count": "R/L/S min(8,local union size,writable schema size); "
            "N zero; include zero menus",
            "tokens": "exact R/L/S full user-message token equality "
            "with shared instruction/history; "
            "bounded common envelope, padding field prefix in ['', ' ', '.', 'x'] "
            "plus repeated ' x'; smallest reachable common count within max base+64; "
            "failure stops study, never drops questions; N empty padding/list",
            "held_fixed": [
                "first-W history",
                "repair candidates",
                "executed endpoints",
                "demonstrations",
                "all interface instructions",
                "KG execution constraints",
                "terminal observations",
                "delivery/acceptance",
                "generation budget",
            ],
            "neutral": "N rerun for every selected question; no historical N output reuse",
        },
        "gate": {
            "conjunction": "all conditions in BOTH families, frozen before new data",
            "minimum_accuracy_gain": 0.01,
            "positive_lower_confidence": 0.975,
            "noninferiority_margin": 0.02,
            "endpoints": list(METRICS),
            "thresholds": "R-N and R-S accuracy point >=.01 and lower >0; R-L accuracy "
            "and R-N exact label-set F1 lower >-.02; support interaction lower >0",
            "interaction": "conditional R-N accuracy benefit on initial-no-gold-support minus "
            "conditional benefit on initial-gold-support; support fixed by W alone",
            "sensitivity": "R-N and R-S accuracy positive after each whole-cluster deletion "
            "and excluding largest topic cluster, each family",
            "minimum_clusters": 20,
            "minimum_initial_stratum_clusters": 5,
            "joint_assurance": 0.8,
            "pilot_admission": "both .5/1 Dirichlet joint predictive assurance MC lower .975 >=.80 "
            "plus measured complete cost <=6h including reserve; pilot gate is "
            "descriptive, not a substitute for independent confirmation",
        },
        "analysis": {
            "unit": "paired question, clustered by MINTQA relation sequence / WebQSP transitive "
            "question-topic/exact-question components from question-only metadata",
            "weights": "pilot retains frozen parent survey weights; "
            "confirmation equal question weights",
            "accuracy": "MINTQA native casefold/NFC answer containment; WebQSP existing gcr_hit "
            "label protocol via webqsp.score_records(failed_zero=True)",
            "f1": "webqsp.set_scores exact unrendered label sets, no containment/normalization",
            "support": "capped committed full W endpoint set, not displayed top-20; MINTQA native "
            "gold compatibility / WebQSP exact unrendered label intersection",
            "bootstrap_resamples": 10000,
            "assurance_trials": 1000,
            "assurance_inner_resamples": 999,
            "multiplicity": "single intersection-union claim at one-sided alpha .025; "
            "no selecting arms",
            "missing": "failed generations retain W via WT acceptance; missing rows or undefined "
            "strata mark INCOMPLETE and fail admission; no outcome-based exclusions",
            "assurance": "Dirichlet cluster uncertainty (.5 and 1); "
            "fresh target topic-size clusters "
            "sample complete paired pilot cluster mean vectors; all gates simulated jointly",
            "transport_risk": "pilot MINTQA effects and within-topic structure "
            "may not transport to WebQSP",
        },
        "budget": {
            "pilot_seconds": 7200,
            "confirmation_seconds": 21600,
            "forecast_pilot_gpu_hours": [1.0862268050669377, 1.2356902998996007],
            "forecast_confirmation_gpu_hours": [3.1808617533935686, 5.572277670716177],
            "forecast_basis": "six successful menu-control launch events minus workload give "
            "263.963 s/launcher; 4304.549 s/6144 second calls; eight launches, "
            "1..2.5 workload multiplier, .4h/1h reserve; re-derived in evidence.json",
            "accounting": "reuse menu_control.execute_stage durable occupancy, "
            "failed-attempt ledger, "
            "headroom wait outside GPU charge, timeout cleanup, hash-bound skip receipts",
        },
        "inputs": [control.identity(p) for p in [panels, *files]],
        "stages": {p: phase_plan(out, p) for p in ("pilot", "confirmation")},
        "provenance": provenance([panels]),
    }
    if webqsp_registration is not None:
        parent = json.loads(Path(webqsp_registration).read_text())
        registration["confirmation"]["eligible_population_registration"] = control.identity(
            webqsp_registration
        )
        registration["confirmation"]["eligible_ids"] = sorted(parent["population"]["intersection"])
        registration["inputs"].append(control.identity(webqsp_registration))
    _refuse_existing_study_outputs(registration)
    if parent_registration is not None:
        parent_path = Path(parent_registration).resolve()
        if (
            sha256_file(parent_path)
            != (parent_path.parent / "registration.sha256").read_text().strip()
        ):
            raise ValueError("parent registration changed")
        parent = json.loads(parent_path.read_text())
        if parent.get("artifact_kind") != registration["artifact_kind"]:
            raise ValueError("parent is not a repair visibility registration")
        _refuse_existing_study_outputs(parent)
        scientific = (
            "seed",
            "families",
            "arms",
            "pilot",
            "confirmation",
            "matching",
            "gate",
            "analysis",
            "budget",
        )
        inherited_gate = parent.get("pilot_authorization")
        expected = {key: parent[key] for key in scientific}
        population_repin = False
        if inherited_gate is not None:
            current_gate = _stride_gate(
                stride_score or inherited_gate["score"]["path"],
                stride_receipt or inherited_gate["receipt"]["path"],
            )
            if current_gate != inherited_gate:
                raise ValueError("source-pin amendment cannot change the registered stride gate")
            registration["pilot_authorization"] = current_gate
            registration["inputs"].extend(current_gate[key] for key in ("score", "receipt"))
            # Only the already-amended authorization wording is inherited; all
            # cohort and scientific values are still independently reconstructed.
            for key in ("reuse", "window"):
                registration["pilot"][key] = parent["pilot"][key]
            pin_key = "eligible_population_registration"
            population_repin = (
                webqsp_registration is not None
                and parent["confirmation"].get(pin_key) != registration["confirmation"][pin_key]
            )
            if population_repin:
                if (
                    parent["confirmation"].get("eligible_ids")
                    != registration["confirmation"]["eligible_ids"]
                ):
                    raise ValueError(
                        "source-pin amendment cannot change the registered eligible ID list"
                    )
                expected["confirmation"] = {
                    **parent["confirmation"],
                    pin_key: registration["confirmation"][pin_key],
                }
        if any(expected[k] != registration[k] for k in scientific):
            raise ValueError("pre-data guard amendment cannot change scientific fields")
        registration["version"] = parent["version"] + 1
        registration["parent_registration"] = control.identity(parent_path)
        registration["inputs"].append(control.identity(parent_path))
        registration["amendment"] = {
            "reason": "fail closed on existing study outputs before sealing or amending",
            "scientific_fields_equal": True,
            "no_registered_study_outputs_exist": True,
        }
        archive = parent_path.parent / "v1-source.tar.gz"
        if archive.exists():
            registration["inputs"].append(control.identity(archive))
        if inherited_gate is not None:
            registration["amendment"].update(
                amended_at=registration["created_at"],
                reason="prospective implementation source-pin amendment; recorded stride "
                "authorization and all scientific fields unchanged",
                unchanged_fields=[*scientific, "pilot_authorization"],
            )
            if population_repin:
                registration["amendment"].update(
                    reason="prospective WebQSP eligible-population source-pin amendment; "
                    "eligible ID list, recorded stride authorization and all scientific "
                    "fields unchanged",
                    changed_fields=["confirmation.eligible_population_registration"],
                    eligible_ids_equal=True,
                    unchanged_fields=[k for k in scientific if k != "confirmation"]
                    + [f"confirmation.{k}" for k in parent["confirmation"] if k != pin_key]
                    + ["pilot_authorization"],
                )
        elif stride_score is not None:
            registration["pilot_authorization"] = _stride_gate(stride_score, stride_receipt)
            registration["inputs"].extend(
                registration["pilot_authorization"][key] for key in ("score", "receipt")
            )
            registration["pilot"]["reuse"] = (
                "ONLY selected sealed W first calls plus original requests and provenance, "
                "after recorded failed stride gate and hash-bound owner export decision; "
                "no other sealed outcomes, no existing repairs or no-suggestion second calls"
            )
            registration["pilot"]["window"] = (
                "after owner decision binding the recorded failed stride score and receipt"
            )
            registration["amendment"].update(
                amended_at=registration["created_at"],
                reason="Nov 13 was a proxy for stride gate decided; the development gate "
                "has failed, no further stride data will be produced, and no repair "
                "visibility data exists; replace the calendar proxy with bound evidence",
                changed_fields=["pilot.reuse", "pilot.window", "pilot_authorization"],
                unchanged_fields=[k for k in scientific if k != "pilot"],
                pilot_cohort_equal=True,
            )
    evidence = out / "evidence.json"
    if evidence.exists():
        rederived = json.loads(evidence.read_text())
        if rederived["forecast_gpu_hours"] != {
            "pilot": registration["budget"]["forecast_pilot_gpu_hours"],
            "confirmation": registration["budget"]["forecast_confirmation_gpu_hours"],
        }:
            raise ValueError("forecast differs from re-derived development receipts")
        registration["inputs"].append(control.identity(evidence))
    out.mkdir(parents=True, exist_ok=True)
    script = out / "repair-visibility.sh"
    script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd "
        + shlex.quote(str(REPO))
        + "\nexport OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        + "TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1\nexport TMPDIR="
        + shlex.quote(str(REPO / "target/tmp"))
        + '\nmkdir -p "$TMPDIR"\nexec nice -n 10 '
        + shlex.join(
            [
                str(REPO / ".venv/bin/python"),
                "-I",
                "-m",
                "xgraph.repair_visibility",
                "queue",
                "--out",
                str(out),
            ]
        )
        + ' "$@"\n'
    )
    script.chmod(0o755)
    registration["inputs"].append(control.identity(script))
    write_json(out / "registration.json", registration)
    (out / "registration.sha256").write_text(sha256_file(out / "registration.json") + "\n")


def _refuse_existing_study_outputs(registration):
    for stages in registration["stages"].values():
        for stage in stages:
            if any(Path(p).exists() for p in stage["outputs"]):
                raise ValueError(f"registration refuses existing study outputs: {stage['name']}")


def verify(out, *, require_clean=True):
    out = Path(out)
    if sha256_file(out / "registration.json") != (out / "registration.sha256").read_text().strip():
        raise ValueError("repair visibility registration changed")
    registration = json.loads((out / "registration.json").read_text())
    if require_clean:
        control.clean_commit(REPO)
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", registration["base_commit"], "HEAD"],
            cwd=REPO,
            check=True,
        )
    control.check_identities(registration["inputs"])
    return registration


def authorize(out, phase, decision, *, now=None):
    if decision is None or not Path(decision).is_file():
        raise ValueError("UNAPPROVED: a hash-bound owner decision is required")
    now = now or datetime.now(UTC)
    row = json.loads(Path(decision).read_text())
    if (
        row.get("approved") is not True
        or not row.get("owner")
        or row.get("phase") != phase
        or row.get("registration_sha256") != sha256_file(out / "registration.json")
        or row.get("stride_passes") is not False
    ):
        raise ValueError("owner decision must bind this registration and a failed stride gate")
    registration = json.loads((out / "registration.json").read_text())
    gate = registration.get("pilot_authorization")
    if (
        not gate
        or row.get("stride_score_sha256") != gate["score"]["sha256"]
        or row.get("stride_receipt_sha256") != gate["receipt"]["sha256"]
    ):
        raise ValueError("owner decision must bind the registered stride score and receipt hashes")
    if _stride_gate(gate["score"]["path"], gate["receipt"]["path"]) != gate:
        raise ValueError("registered stride score or receipt changed")
    decided = datetime.fromisoformat(row.get("decided_at", ""))
    received = datetime.fromisoformat(gate["receipt_created_at"])
    if decided.tzinfo is None or not received <= decided <= now:
        raise ValueError(
            "owner decision must be at or after stride receipt creation and not future"
        )
    if phase == "confirmation" and (
        now < datetime.fromisoformat(EMBARGO) or not row.get("embargo_released")
    ):
        raise ValueError("confirmation requires the Nov 23 embargo release")
    sources = out / phase / "sources.json"
    if not sources.exists() or row.get("sources_sha256") != sha256_file(sources):
        raise ValueError("owner decision must bind the source export seal")
    if phase == "confirmation":
        admission = out / "pilot/admission.json"
        if (
            not admission.exists()
            or row.get("admission_sha256") != sha256_file(admission)
            or not json.loads(admission.read_text()).get("admitted")
        ):
            raise ValueError(
                "confirmation requires successful frozen pilot assurance and cost admission"
            )
        admitted = json.loads(admission.read_text())
        if json.loads(sources.read_text())["cohort_inventory"] != admitted["confirmation_cohort"]:
            raise ValueError("confirmation cohort differs from the pilot assurance target")
    return row


def queue(out, *, phase="pilot", run=False, decision=None):
    if run:
        authorize(out, phase, decision)
    registration = verify(out, require_clean=run)
    stages = registration["stages"][phase]
    if not run:
        print("# UNAPPROVED; preview does not open histories, gold, GPU, vLLM or Docker")
        for stage in stages:
            print("# " + stage["name"])
            for key in ("inspect_command", "headroom_command", "command", "cleanup_command"):
                if key in stage:
                    print(
                        ("# on failure: " if key == "cleanup_command" else "")
                        + shlex.join(
                            control._stage_command(stage) if key == "command" else stage[key]
                        )
                    )
            if "expanded_argv" in stage:
                print("# expanded host generation argv: " + shlex.join(stage["expanded_argv"]))
        return
    authorize(out, phase, decision)
    os.environ["XG_REPAIR_VISIBILITY_DECISION"] = str(Path(decision).resolve())
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(
        resource.RLIMIT_AS,
        (min(4_000_000_000, hard) if hard != resource.RLIM_INFINITY else 4_000_000_000, hard),
    )

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"repair visibility owner queue received signal {signum}")

    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        _run_stages(out, phase, decision, registration, stages)
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _run_stages(out, phase, decision, registration, stages):
    (REPO / "target/tmp").mkdir(parents=True, exist_ok=True)
    with (out / f"{phase}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seal = sha256_file(out / "registration.json")
        for stage in stages:
            verify(out)
            authorize(out, phase, decision)
            _require_ledger(out, phase, stages)
            for path in stage["outputs"]:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            stage = _bound_stage(out, phase, stage)
            control.execute_stage(
                stage,
                seal,
                SOURCE / "target/queue/repair-visibility.log",
                out / phase / "gpu-ledger.jsonl",
                registration["budget"][f"{phase}_seconds"],
            )


def _require_ledger(out, phase, stages):
    ledger = out / phase / "gpu-ledger.jsonl"
    completed = [s for s in stages if s["kind"] == "gpu" and control._receipt(s).exists()]
    if completed and not ledger.exists():
        raise ValueError("completed GPU receipts require their durable lifetime ledger")
    rows = read_jsonl(ledger) if ledger.exists() else []
    starts = {r["attempt"]: r for r in rows if r["event"] == "start"}
    ended = {r["attempt"] for r in rows if r["event"] == "end"}
    if any(
        not any(r["stage"] == s["name"] and k in ended for k, r in starts.items())
        for s in completed
    ):
        raise ValueError("GPU receipt has no settled attempt in the lifetime ledger")
    return control._ledger_used(ledger)


def _authorized_sources(out, phase):
    registration = verify(out)
    authorize(out, phase, os.environ.get("XG_REPAIR_VISIBILITY_DECISION"))
    path = out / phase / "sources.json"
    sources = json.loads(path.read_text())
    if sources["registration_sha256"] != sha256_file(out / "registration.json"):
        raise ValueError("source export names a different registration")
    if phase == "pilot":
        ids = registration["pilot"]["ids"]
        if sources["population_ids"] != registration["pilot"]["population_ids"]:
            raise ValueError("pilot source export changed the frozen validation population")
        if indexed(sources["members"]) != indexed(registration["pilot"]["members"]):
            raise ValueError("pilot survey weights or relation-sequence clusters changed")
        _read_cohort(out, registration, sources["confirmation_cohort"])
    else:
        inventory = sources["cohort_inventory"]
        cohort = _read_cohort(out, registration, inventory)
        # This file contains only question metadata and must be sealed prospectively.
        frozen_at = datetime.fromisoformat(cohort["frozen_at"])
        if frozen_at.tzinfo is None or frozen_at >= datetime.fromisoformat(EMBARGO):
            raise ValueError("confirmation cohort must be sealed before the embargo release")
        if cohort["population_ids"] != sources["population_ids"]:
            raise ValueError("changed WebQSP core ID inventory")
        if not set(cohort["population_ids"]) <= set(registration["confirmation"]["eligible_ids"]):
            raise ValueError(
                "confirmation IDs are outside the independently registered WebQSP population"
            )
        ids = select_ids(cohort["population_ids"], min(1024, len(cohort["population_ids"])))
        if set(ids) & set(registration["pilot"]["population_ids"]):
            raise ValueError("confirmation panel must be separate from the MINTQA pilot")
        if cohort["ids"] != ids:
            raise ValueError("confirmation cohort is not the frozen outcome-blind ID prefix")
        if indexed(cohort["members"]) != indexed(sources["members"]):
            raise ValueError("confirmation topic dependence units changed")
    if {m["id"] for m in sources["members"]} != set(ids) or len(sources["members"]) != len(ids):
        raise ValueError("source members must cover exactly the ID-selected cohort")
    if phase == "confirmation" and any(m["weight"] != 1 for m in sources["members"]):
        raise ValueError("confirmation must use equal question weights")
    return registration, sources, ids


def _read_cohort(out, registration, pin):
    control.check_identities([pin])
    cohort = json.loads(Path(pin["path"]).read_text())
    if cohort["registration_sha256"] != sha256_file(out / "registration.json"):
        raise ValueError("cohort belongs to a different repair visibility registration")
    frozen = datetime.fromisoformat(cohort["frozen_at"])
    if frozen.tzinfo is None or frozen >= datetime.fromisoformat(EMBARGO):
        raise ValueError("cohort must be frozen before the embargo release")
    ids = select_ids(cohort["population_ids"], min(1024, len(cohort["population_ids"])))
    if cohort["ids"] != ids or set(indexed(cohort["members"])) != set(ids):
        raise ValueError("cohort is not the complete ID-selected panel")
    control.check_identities([cohort["core_execution"], cohort["question_only_input"]])
    core = json.loads(Path(cohort["core_execution"]["path"]).read_text())
    eligible = registration["confirmation"]["eligible_ids"]
    if core["ids"] != select_ids(eligible, len(core["ids"])):
        raise ValueError("core population differs from the registered eligible ID prefix")
    if set(core["ids"]) != set(cohort["population_ids"]) or any(
        r["weight"] != 1 or r["cluster"] != core["cluster"][r["id"]] for r in cohort["members"]
    ):
        raise ValueError("core inventory, topic assignments or cohort weights changed")
    return cohort


def freeze_cohort(out, questions, core_execution, *, now=None):
    """Freeze a separate WebQSP panel from a closed question-only core projection."""
    from xgraph.webqsp_test import question_clusters

    registration = verify(out)
    now = now or datetime.now(UTC)
    if now >= datetime.fromisoformat(EMBARGO):
        raise ValueError("confirmation cohort must be frozen before the embargo release")
    destination = out / "confirmation/cohort.json"
    if destination.exists():
        raise ValueError("confirmation cohort is immutable")
    rows = json.loads(Path(questions).read_text())
    if not rows or any(set(r) != {"id", "question", "q_entity"} for r in rows):
        raise ValueError("cohort needs only id/question/q_entity; gold and outcomes are forbidden")
    population = [r["id"] for r in rows]
    core = json.loads(Path(core_execution).read_text())
    if (
        core.get("artifact_kind") != "xgraph.webqsp_test_execution"
        or set(population) != set(core["ids"])
        or len(population) != len(core["ids"])
    ):
        raise ValueError("question projection must cover the complete sealed core")
    created = datetime.fromisoformat(core["created_at"])
    if created.tzinfo is None or created > now:
        raise ValueError("core inventory must be sealed before cohort selection")
    eligible = registration.get("confirmation", {}).get("eligible_ids")
    if eligible is not None and not set(population) <= set(eligible):
        raise ValueError("core inventory includes IDs outside the registered WebQSP population")
    if eligible is not None and core["ids"] != select_ids(eligible, len(core["ids"])):
        raise ValueError("core inventory is not the original ID-selected eligible prefix")
    ids = select_ids(population, min(1024, len(population)))
    if set(ids) & set(registration["pilot"]["population_ids"]):
        raise ValueError("confirmation cohort must be separate from the pilot")
    groups = {}
    for q, group in question_clusters(rows, "transitive").items():
        groups.setdefault(group, set()).add(core["cluster"][q])
    if any(len(g) != 1 for g in groups.values()):
        raise ValueError("sealed core dependence units split a shared question topic")
    clusters = core["cluster"]
    write_json(
        destination,
        {
            "registration_sha256": sha256_file(out / "registration.json"),
            "frozen_at": now.isoformat(),
            "population_ids": sorted(population),
            "ids": ids,
            "members": [{"id": q, "cluster": clusters[q], "weight": 1} for q in ids],
            "question_only_input": control.identity(questions),
            "core_execution": control.identity(core_execution),
            "seed": SEED,
            "provenance": provenance([questions, core_execution]),
        },
    )


def _load_executor(spec):
    from xgraph.index import load_kg_index
    from xgraph.programs import MatchedExecutor
    from xgraph.tokenizer import ServedTokenizer

    control.check_identities(spec["index_files"] + [spec["tokenizer"]])
    root = Path(spec["index"]).resolve()
    expected = {str(root / "kg-index.json"), *(str(p) for p in root.glob("*.npy"))}
    if {p["path"] for p in spec["index_files"]} != expected:
        raise ValueError("index seal must cover its header and every native table")
    return MatchedExecutor(
        load_kg_index(root),
        ServedTokenizer.from_file(spec["tokenizer"]["path"]),
        start_rank="sitelinks",
        unquoted_starts=True,
    )


def _bound_stage(out, phase, stage):
    if stage["kind"] != "gpu":
        return stage
    family = stage["name"].split("/")[1]
    path = out / phase / family / "preparation.json"
    prep = json.loads(path.read_text())
    control.check_identities(prep["inputs"] + prep["prepared"])
    source = json.loads((out / phase / "sources.json").read_text())
    index = str(Path(source["families"][family]["index"]).resolve())
    bound = json.loads(json.dumps(stage))
    bound["command"] = [
        f"XG_KG_INDEX={index}" if s.startswith("XG_KG_INDEX=") else s for s in bound["command"]
    ]
    bound["expected_flags"]["--kg-index"] = [index]
    args = bound["expanded_argv"]
    args[args.index("--kg-index") + 1] = index
    bound["source_seal_sha256"] = sha256_file(out / phase / "sources.json")
    return bound


def prepare(out, phase, family):
    """Owner-only export consumption. Build candidates once, never read scoring gold."""
    from xgraph.catalog_argv import flags
    from xgraph.programs import record_path
    from xgraph.repair import Repairer
    from xgraph.toolkg import KGStore

    registration, sources, ids = _authorized_sources(out, phase)
    directory = out / phase / family
    if (directory / "preparation.json").exists():
        raise ValueError("immutable preparation already exists")
    spec = sources["families"][family]
    inputs = [out / phase / "sources.json"]
    pins = [spec[k] for k in ("requests", "first", "first_provenance", "db", "relations")]
    if phase == "pilot" and spec["first"]["path"] != registration["pilot"]["existing_W"][family]:
        raise ValueError("pilot may reuse only the named sealed validation W first calls")
    control.check_identities(pins)
    before = json.loads(Path(spec["first_provenance"]["path"]).read_text())
    settings = flags(before["provenance"]["command"])
    needed = {
        "--menu": ["5"],
        "--observe-paths": ["3"],
        "--observe-width": ["8"],
        "--relations": ["frontier"],
        "--start-rank": ["sitelinks"],
    }
    if any(settings.get(k) != v for k, v in needed.items()) or any(
        k not in settings
        for k in (
            "--observe-select",
            "--observe-required",
            "--observe-no-loop",
            "--committed-paths",
            "--unquoted-starts",
            "--no-escapes",
        )
    ):
        raise ValueError("export is not the frozen W first-call recipe")
    if spec["requests"] not in before["provenance"]["inputs"]:
        raise ValueError("W first-call provenance does not bind original requests")
    wanted = set(ids)
    requests = [r for r in read_jsonl(Path(spec["requests"]["path"])) if r["id"] in wanted]
    first = [r for r in read_jsonl(Path(spec["first"]["path"])) if r["id"] in wanted]
    if set(indexed(requests)) != wanted or set(indexed(first)) != wanted:
        raise ValueError("incomplete selected W first-call export; no outcome-based substitution")
    if any(r.get("kg_version") for r in requests):
        raise ValueError("this registered study uses the frozen base KG, not KG versions")
    executor = _load_executor(spec)
    if spec["tokenizer"]["path"] != str(
        Path("/home/jonas/models") / control.FAMILIES[family] / "tokenizer.json"
    ):
        raise ValueError("source tokenizer differs from the frozen served family")
    relation_rows = [
        tuple(json.loads(s))
        for s in Path(spec["relations"]["path"]).read_text().splitlines()
        if s.strip()
    ]
    store = KGStore(Path(spec["db"]["path"]))
    repairer = Repairer(executor, store, relation_rows, edits=4)
    by_request = indexed(requests)
    candidates = []
    for row in first:
        qid, path = row["id"], record_path(row)
        request = by_request[qid]
        question = next(m["content"] for m in reversed(request["messages"]) if m["role"] == "user")
        found, empty = (
            repairer.candidates(question, *path, request.get("anchor")) if path else ([], 0)
        )
        candidates.append(
            {
                "id": qid,
                "question": question,
                "start": path[0] if path else None,
                "candidates": found,
                "empty": empty,
            }
        )
    directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(directory / "first.jsonl", first)
    write_jsonl(directory / "requests.jsonl", requests)
    write_jsonl(directory / "candidates.jsonl", candidates)
    write_jsonl(
        directory / "initial-support.jsonl",
        [control._capped_support(executor, r, by_request[r["id"]]) for r in first],
    )
    menus = AuditMenus(executor._index, executor._tokenizer)
    made = {}
    for arm in ARMS:
        made[arm] = revision_requests(
            requests,
            first,
            candidates,
            executor,
            store.label,
            observe_terminal=8,
            menus=menus,
            tokenizer=executor._tokenizer,
            visibility=arm,
        )
        write_jsonl(directory / f"{arm}-requests.jsonl", made[arm])
    matching = []
    for rows in zip(*(made[a] for a in ARMS), strict=True):
        if any(r["messages"][:-1] != rows[0]["messages"][:-1] for r in rows):
            raise ValueError("different W histories or demonstrations")
        payloads = [
            {
                k: v
                for k, v in r["revision"]["display"].items()
                if k not in {"relation_suggestions", "padding"}
            }
            for r in rows
        ]
        if any(p != payloads[0] for p in payloads):
            raise ValueError("different candidates, endpoints, question or original answer")
        token_counts = [r["revision"]["instruction_tokens"] for r in rows[:3]]
        sizes = [r["revision"]["suggestion_count"] for r in rows[:3]]
        if len(set(token_counts)) != 1 or len(set(sizes)) != 1:
            raise ValueError("visibility count/token matching failed")
        matching.append(
            {
                "id": rows[0]["id"],
                "tokens": token_counts,
                "counts": sizes,
                "common_feedback_sha256": control.digest(payloads[0]),
                "history_sha256": control.digest(rows[0]["messages"][:-1]),
            }
        )
    write_jsonl(directory / "matching.jsonl", matching)
    prepared = [control.identity(p) for p in sorted(directory.glob("*.jsonl"))]
    write_json(
        directory / "preparation.json",
        {
            "registration_sha256": sha256_file(out / "registration.json"),
            "inputs": [control.identity(p) for p in inputs]
            + pins
            + spec["index_files"]
            + [spec["tokenizer"]],
            "prepared": prepared,
            "members": sources["members"],
            "ids": ids,
            "provenance": provenance(inputs),
        },
    )
    # Preserve the exact late-bound argv before any repair generation.
    bound = [
        _bound_stage(out, phase, s)
        for s in registration["stages"][phase]
        if s["kind"] == "gpu" and s["name"].split("/")[1] == family
    ]
    write_json(directory / "bound-stages.json", bound)
    (directory / "bound-dry-run.txt").write_text(
        "\n".join(
            shlex.join(s["command"]) + "\n# expanded: " + shlex.join(s["expanded_argv"])
            for s in bound
        )
        + "\n"
    )


def finish(out, phase, family):
    registration, sources, ids = _authorized_sources(out, phase)
    directory = out / phase / family
    seal = sha256_file(out / "registration.json")
    for stage in registration["stages"][phase]:
        if stage["kind"] == "gpu" and stage["name"].split("/")[1] == family:
            control._check_receipt(_bound_stage(out, phase, stage), seal)
    executor = _load_executor(sources["families"][family])
    first = read_jsonl(directory / "first.jsonl")
    requests = indexed(read_jsonl(directory / "requests.jsonl"))
    from xgraph.mintqa_rematch import usage

    raw_arms = {arm: read_jsonl(directory / f"{arm}-raw.jsonl") for arm in ARMS}
    prompt_tokens = {
        a: {r["id"]: usage(r)["prompt_tokens"] for r in raw_arms[a]} for a in ("R", "L", "S")
    }
    if any(prompt_tokens[a] != prompt_tokens["R"] for a in ("L", "S")):
        raise ValueError("actual served R/L/S prompt token budgets differ")
    for arm in ARMS:
        raw = raw_arms[arm]
        support = [control._capped_support(executor, r, requests[r["id"]]) for r in raw]
        write_jsonl(
            directory / f"{arm}.jsonl",
            revision_records(
                first,
                raw,
                support,
                ids,
                first_source="sealed-W",
                second_source=arm,
            ),
        )
    write_json(
        directory / "finish.json",
        {
            "registration_sha256": seal,
            "provenance": provenance([directory / "first.jsonl"]),
        },
    )


def score(out, phase):
    from xgraph.mintqa_bench import answer_contains, dataset_rows
    from xgraph.prompts import final_answer
    from xgraph.score import record_outputs
    from xgraph.toolkg import KGStore
    from xgraph.webqsp import answers_of, score_records, set_scores, unrender_label

    registration, sources, ids = _authorized_sources(out, phase)
    seal = sha256_file(out / "registration.json")
    for stage in registration["stages"][phase]:
        if stage["name"].endswith(("/score", "/admit")):
            continue
        control._check_receipt(_bound_stage(out, phase, stage), seal)
    # No gold access until every matched generation and support stage has a receipt.
    control.check_identities([sources["gold"]])
    gold = indexed(json.loads(Path(sources["gold"]["path"]).read_text()))
    if not set(ids) <= set(gold):
        raise ValueError("gold export does not cover the ID-selected cohort")
    answer_paths = []
    if phase == "pilot":
        # Frozen panel questions contain id/utterance/complexity, not answers.
        # Preserve the native split/row-ID mapping and dataset answer spelling.
        control.check_identities(sources["answer_sources"])
        answer_paths = [Path(pin["path"]) for pin in sources["answer_sources"]]
        native = dataset_rows([f"{p.stem}={p}" for p in answer_paths])
        if not set(ids) <= set(native):
            raise ValueError("native answer sources do not cover the ID-selected cohort")
        gold = {q: {"id": q, "answers": [native[q]["answer"]]} for q in ids}
    rows = []
    members = {m["id"]: m for m in sources["members"]}
    for family in FAMILIES:
        directory = out / phase / family
        initial = indexed(read_jsonl(directory / "initial-support.jsonl"))
        store = KGStore(Path(sources["families"][family]["db"]["path"]))
        arms = {}
        for arm in ARMS:
            records = read_jsonl(directory / f"{arm}.jsonl")
            texts = record_outputs(records, ids)
            valid = [{**r, "text": texts[r["id"]]} for r in records]
            if phase == "confirmation":
                details = score_records(valid, [gold[q] for q in ids], failed_zero=True)[
                    "per_question"
                ]
                arms[arm] = {r["id"]: {"accuracy": r["gcr_hit"], "f1": r["f1"]} for r in details}
            else:
                arms[arm] = {
                    r["id"]: {
                        "accuracy": int(
                            answer_contains(final_answer(r["text"]), gold[r["id"]]["answers"][0])
                        ),
                        "f1": set_scores(answers_of(r), gold[r["id"]]["answers"])["f1"],
                    }
                    for r in valid
                }
        for q in ids:
            labels = [unrender_label(store.label(int(e[1:]))) for e in initial[q]["reached"]]
            compatible = (
                any(answer_contains(s, gold[q]["answers"][0]) for s in labels)
                if phase == "pilot"
                else bool(set(labels) & set(gold[q]["answers"]))
            )
            rows.append(
                {
                    "id": q,
                    "family": family,
                    **members[q],
                    "initial_support": compatible,
                    "arms": {a: arms[a][q] for a in ARMS},
                }
            )
    write_jsonl(out / phase / "rows.jsonl", rows)
    report = reduce_rows(rows)
    write_json(
        out / phase / "report.json",
        {
            **report,
            "registration_sha256": seal,
            "phase": phase,
            "claim_admitted": phase == "confirmation" and report["passes"],
            "gpu_seconds": control._ledger_used(out / phase / "gpu-ledger.jsonl"),
            "provenance": provenance(
                [
                    out / "registration.json",
                    out / phase / "sources.json",
                    Path(sources["gold"]["path"]),
                    *answer_paths,
                    out / phase / "rows.jsonl",
                ]
            ),
        },
    )


def admit(out, phase="pilot"):
    registration, sources, _ = _authorized_sources(out, phase)
    if phase != "pilot":
        raise ValueError("only pilot outcomes can admit independent confirmation")
    stage = next(s for s in registration["stages"]["pilot"] if s["name"] == "pilot/score")
    control._check_receipt(stage, sha256_file(out / "registration.json"))
    cohort_pin = sources["confirmation_cohort"]
    cohort = _read_cohort(out, registration, cohort_pin)
    target = Counter(m["cluster"] for m in cohort["members"])
    assurance = joint_assurance(read_jsonl(out / "pilot/rows.jsonl"), list(target.values()))
    workload, occupancy = 0.0, 0.0
    for stage in registration["stages"]["pilot"]:
        if stage["kind"] != "gpu":
            continue
        control._check_receipt(
            _bound_stage(out, "pilot", stage), sha256_file(out / "registration.json")
        )
        summary = json.loads(Path(stage["outputs"][0] + ".summary.json").read_text())
        if (
            summary["requests"] != len(registration["pilot"]["ids"])
            or not math.isfinite(summary["seconds"])
            or summary["seconds"] <= 0
        ):
            raise ValueError("invalid measured complete-pilot workload")
        workload += summary["seconds"] / summary["requests"] * len(cohort["ids"])
        events = read_jsonl(Path(stage["gpu_events"]))
        if len(events) != 2 or [e["event"] for e in events] != ["container_start", "container_end"]:
            raise ValueError("pilot admission requires complete single-launch cost receipts")
        occupancy += max(0.0, events[1]["monotonic"] - events[0]["monotonic"] - summary["seconds"])
    projection = 2.5 * workload + occupancy + 3600
    spent = _require_ledger(out, "pilot", registration["stages"]["pilot"])
    admitted = assurance["admitted"] and projection <= 21600 and spent <= 7200
    write_json(
        out / "pilot/admission.json",
        {
            "admitted": admitted,
            "joint_assurance": assurance,
            "projected_confirmation_seconds": projection,
            "pilot_charged_seconds": spent,
            "pricing": "2.5 times measured paired per-question workload "
            "+ measured launch overhead + 1h reserve",
            "registration_sha256": sha256_file(out / "registration.json"),
            "confirmation_cohort": cohort_pin,
            "provenance": provenance(
                [out / "pilot/report.json", out / "pilot/rows.jsonl", Path(cohort_pin["path"])]
            ),
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "register",
            "verify",
            "freeze-cohort",
            "queue",
            "prepare",
            "finish",
            "score",
            "admit",
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "confirmation"), default="pilot")
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--panels", type=Path)
    parser.add_argument("--webqsp-registration", type=Path)
    parser.add_argument("--parent-registration", type=Path)
    parser.add_argument("--stride-score", type=Path)
    parser.add_argument("--stride-receipt", type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--core-execution", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--decision", type=Path)
    args = parser.parse_args(argv)
    if args.run and args.dry_run:
        parser.error("--run and --dry-run are mutually exclusive")
    out = args.out.resolve()
    if args.command == "register":
        register(
            out,
            panels=args.panels,
            webqsp_registration=args.webqsp_registration,
            parent_registration=args.parent_registration,
            stride_score=args.stride_score,
            stride_receipt=args.stride_receipt,
        )
    elif args.command == "verify":
        verify(out)
    elif args.command == "freeze-cohort":
        if args.questions is None or args.core_execution is None:
            parser.error("freeze-cohort requires --questions and --core-execution metadata")
        freeze_cohort(out, args.questions, args.core_execution)
    elif args.command == "queue":
        queue(out, phase=args.phase, run=args.run, decision=args.decision)
    elif args.command in ("prepare", "finish"):
        if args.family is None:
            parser.error("this stage requires --family")
        {"prepare": prepare, "finish": finish}[args.command](out, args.phase, args.family)
    else:
        {"score": score, "admit": admit}[args.command](out, args.phase)


if __name__ == "__main__":
    main()
