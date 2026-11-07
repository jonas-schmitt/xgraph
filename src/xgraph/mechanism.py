"""Development-only selection by guaranteed-delivery factorial; CPU preparation/scoring.

The prepared shell queue is the only generation entry point and belongs to the
GPU owner. This module never starts a model, Docker, or a GPU process.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shlex
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from xgraph.mintqa_bench import answer_contains
from xgraph.mintqa_panels import indexed, read_jsonl, write_json, write_jsonl
from xgraph.mintqa_queue import FAMILIES, REPO, _code_digest
from xgraph.mintqa_rematch import clustered_difference, usage
from xgraph.observations import PathRelations
from xgraph.programs import record_path
from xgraph.prompts import final_answer
from xgraph.provenance import provenance, sha256_file
from xgraph.score import record_failure, spans

CELLS = {
    "paths-allfix": (0, 0),
    "s0d0": (0, 0),
    "s0d1": (0, 1),
    "s1d0": (1, 0),
    "s1d1": (1, 1),
}
FAMILY_NAMES = ("qwen38", "gemma31")
MODELS = Path("/home/jonas/models")
FRESH_DESIGN = "mechanism-v2-hard"
FRESH_CELL = (4, "confusable", 16)
FRESH_REGISTRATION = REPO / "target/mechanism-fallback-v2/registration.json"


@dataclass(frozen=True)
class FreshConfig:
    seed: int
    worlds: int = 64
    split: str = "development"
    pairs: int = 368
    design: str = FRESH_DESIGN


def generate_fresh_worlds(config, *, confirmation_seal=None):
    """Independent v2 blocks, retaining only the wide, deep, confusable cell."""
    from xgraph.fresh_worlds import _confirm_admission
    from xgraph.fresh_worlds_v2 import CELLS, ROLES, SCHEMA, world

    if (
        type(config.seed) is not int
        or type(config.worlds) is not int
        or config.worlds not in (64, 256)
        or config.split not in ("development", "confirmation")
        or config.pairs != 368
        or config.design != FRESH_DESIGN
        or (config.split == "development" and config != FreshConfig(seed=262813))
    ):
        raise ValueError("factorial requires the registered v2 wide hard population")
    if config.split == "confirmation":
        _confirm_admission(confirmation_seal, config)
    return {
        "format": 2,
        "config": asdict(config),
        "schema": SCHEMA,
        "templates": [{"id": f"v2-t{i}", "roles": roles} for i, roles in enumerate(ROLES)],
        "worlds": [
            world(config, 6 * n + 5, n % len(ROLES), FRESH_CELL) for n in range(config.worlds)
        ],
        "demonstrations": [world(config, n, n, CELLS[n], demo=True) for n in range(5)],
    }


def fresh_requests(study):
    """One common effective prompt for the prospective fresh-world availability intervention.

    Teach the union of the instructions before freezing; the generator's ordinary
    per-cell observation conversion then adds nothing, even with selection off.
    Examples use only the common language (observations, no selections or menus).
    """
    from xgraph.fresh_worlds import requests_for
    from xgraph.prompts import observed_requests
    from xgraph.syntax import SpanSyntax

    rows = [{**r, "fixed_path_demonstrations": True} for r in requests_for(study, "factorial")]
    return observed_requests(rows, 3, SpanSyntax(), select=True)


def fresh_stages(repo, study, out, models, registration, *, confirmation=False):
    """Exact owner argv, with one common request file and only the two switches varying."""
    from xgraph.fresh_worlds_queue import FAMILIES as MODEL_DIRS

    repo, study, out, models, registration = map(
        lambda p: Path(p).resolve(), (repo, study, out, models, registration)
    )
    stages = []
    for family, model_dir in MODEL_DIRS.items():
        for cell in ("s0d0", "s0d1", "s1d0", "s1d1"):
            original = generation_command(out, repo, family, cell)
            launch = original.index(str(REPO / "scripts/generate.sh"))
            controls = dict(v.split("=", 1) for v in original[1 : original.index("nice")])
            model = models / model_dir
            requests = study / "requests-factorial.jsonl"
            output = out / family / f"{cell}.jsonl"
            controls.update(
                {
                    "XG_GPU_IDLE": 1,
                    "XG_RUN_SECONDS": 36000 if confirmation else 7200,
                    "XG_SOURCE_ROOT": repo,
                    "XG_KG_INDEX": study / "indexes" / family,
                    "XG_TOKENIZER": model / "tokenizer.json",
                    "XG_KG_VERSIONS": study / "kg-versions",
                    "XG_PAGECACHE_ROOTS": str(model),
                    "UV_CACHE_DIR": repo / ".uv-cache",
                    "TMPDIR": repo / "target/tmp",
                    "CMAKE_BUILD_PARALLEL_LEVEL": 1,
                    # Data in a prospective owner argv, never executed by CPU planning.
                    "CMAKE_ARGS": "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
                }
            )
            stages.append(
                {
                    "study_type": "mechanism-fallback",
                    "family": family,
                    "name": cell,
                    "gpu": True,
                    "requests": str(requests),
                    "model": str(model),
                    "need_gib": int(controls["XG_NEED_GIB"]),
                    "max_tokens": 1024,
                    "output": str(output),
                    "argv": [
                        "env",
                        *(f"{k}={v}" for k, v in controls.items()),
                        "nice",
                        "-n",
                        "10",
                        "bash",
                        str(out / "generate-owned.sh"),
                        str(model),
                        str(requests),
                        "constrained",
                        str(output),
                        *original[launch + 5 :],
                    ],
                }
            )
    stages.append(
        {
            "family": "both",
            "name": "analysis",
            "gpu": False,
            "output": str(out / "analysis.json"),
            "input_files": [s["output"] for s in stages],
            "argv": [
                "nice",
                "-n",
                "10",
                "prlimit",
                "--as=4000000000",
                str(repo / ".venv/bin/python"),
                "-I",
                "-m",
                "xgraph.mechanism",
                "fresh-analyze",
                "--study",
                str(study),
                "--outputs-dir",
                str(out),
                "--registration",
                str(registration),
                "--budget",
                str(out / "budget.json"),
                "--out",
                str(out / "analysis.json"),
            ],
        }
    )
    return stages


FRESH_HYPOTHESIS = "frontier-selection-x-guaranteed-delivery"


def load_fresh_registration(path):
    """Read the immutable prospective registration, not historical variation outcomes."""
    from xgraph.fresh_worlds_analysis import FACTORIAL_DIRECTIONS, _factorial_bands

    path = Path(path).resolve()
    if sha256_file(path) != Path(str(path) + ".sha256").read_text().strip():
        raise ValueError("factorial registration hash changed")
    spec = json.loads(path.read_text())
    if (
        spec["format"] != 2
        or spec["hypothesis"] != FRESH_HYPOTHESIS
        or spec["directions"] != FACTORIAL_DIRECTIONS
        or spec["pilot_config"] != asdict(FreshConfig(seed=262813))
        or spec["confirmation_worlds"] != 256
        or spec["primary_cell"] != list(FRESH_CELL)
        or spec["parent_registration"]["sha256"]
        != "f70f73768f75552ab507b13ba4d45f18124acf2e8f92bca01d94a29bbfc40413"
        or sha256_file(Path(spec["parent_registration"]["path"]))
        != spec["parent_registration"]["sha256"]
        or spec["v2_generator_sha256"]
        != sha256_file(Path(__file__).with_name("fresh_worlds_v2.py"))
        or sha256_file(Path(spec["document"])) != spec["document_sha256"]
    ):
        raise ValueError("factorial registration differs from the frozen protocol")
    _factorial_bands(spec["per_cell_drift"])
    return spec


def validate_fresh_cohort(study, registration, *, seal=None):
    """Apply the own-study phase/seed/commit guard at every public boundary."""
    from collections import Counter

    from xgraph.fresh_worlds import _confirm_admission
    from xgraph.fresh_worlds_queue import _commit
    from xgraph.fresh_worlds_v2 import ROLES, question, relation

    spec = load_fresh_registration(registration)
    config = study["config"]
    if config["split"] == "confirmation":
        frozen = _confirm_admission(seal, FreshConfig(**config))
        if frozen.get("hypothesis") != FRESH_HYPOTHESIS or frozen.get(
            "registration_sha256"
        ) != sha256_file(Path(registration)):
            raise ValueError("confirmation requires its own factorial seal")
        repo = Path(__file__).resolve().parents[2]
        if frozen.get("code_commit") != _commit(repo, clean=True):
            raise ValueError("factorial confirmation code differs from the exact frozen commit")
        quota = 32
    else:
        if config != spec["pilot_config"] or seal is not None:
            raise ValueError("factorial requires its own registered development worlds")
        quota = 8
    templates = {f"v2-t{i}": roles for i, roles in enumerate(ROLES)}
    worlds = study["worlds"]
    if (
        len(worlds) != config["worlds"]
        or len({w["id"] for w in worlds}) != len(worlds)
        or Counter(w["template"] for w in worlds) != Counter(dict.fromkeys(templates, quota))
        or any(
            w.get("cell") != list(FRESH_CELL)
            or w["relations"]
            != [relation(g, r, True) for g, r in enumerate(templates[w["template"]])]
            or w["question_template"] != question(templates[w["template"]], 4)
            for w in worlds
        )
        or len({w.get("block") for w in worlds}) != len(worlds)
    ):
        raise ValueError("factorial cohort differs from the registered full template quotas")
    return spec


def prepare_fresh_requests(study_dir, registration, *, seal=None):
    """Freeze the common requests on an independently built local study."""
    from xgraph.fresh_worlds import write_json, write_jsonl

    study_dir, registration = Path(study_dir).resolve(), Path(registration).resolve()
    study = json.loads((study_dir / "study.json").read_text())
    validate_fresh_cohort(study, registration, seal=seal)
    path = study_dir / "requests-factorial.jsonl"
    if path.exists() or "mechanism_fallback" in study:
        raise ValueError("factorial requests already frozen; do not overwrite")
    import subprocess

    audit = study_dir / "validity.json"
    subprocess.run(
        [
            str(REPO / ".venv/bin/python"),
            "-I",
            "-S",
            str(REPO / "scripts/fresh-worlds-v2-audit.py"),
            "--study",
            str(study_dir),
            "--out",
            str(audit),
        ],
        check=True,
    )
    rows = fresh_requests(study)
    write_jsonl(path, rows)
    study["mechanism_fallback"] = {
        "registration_sha256": sha256_file(registration),
        "validity_sha256": sha256_file(audit),
        "audited_study_sha256": sha256_file(study_dir / "study.json"),
        "seal": str(Path(seal).resolve()) if seal is not None else None,
    }
    write_json(study_dir / "study.json", study)
    return rows


def validate_fresh_validity(study_dir, study):
    """Bind the independent pre-prompt audit to the exact graph and audit source."""
    import hashlib

    marker = study["mechanism_fallback"]
    path = Path(study_dir) / "validity.json"
    report = json.loads(path.read_text())
    audited = {key: value for key, value in study.items() if key != "mechanism_fallback"}
    study_hash = hashlib.sha256(
        (json.dumps(audited, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    ).hexdigest()
    if (
        marker.get("validity_sha256") != sha256_file(path)
        or marker.get("audited_study_sha256") != study_hash
        or report.get("status") != "passed"
        or report["provenance"]["inputs"].get(str((Path(study_dir) / "study.json").resolve()))
        != study_hash
        or report["provenance"]["inputs"].get(str(REPO / "scripts/fresh-worlds-v2-audit.py"))
        != sha256_file(REPO / "scripts/fresh-worlds-v2-audit.py")
    ):
        raise ValueError("factorial independent validity audit or graph changed")


def analyze_fresh(study_dir, outputs_dir, out, registration, budget=None):
    """Independent ID scoring and the fallback's own analysis/assurance, CPU only."""
    from xgraph.fresh_worlds import read_jsonl, score_records, write_json
    from xgraph.fresh_worlds_analysis import (
        FACTORIAL_CELLS,
        factorial_assurance,
        factorial_contrasts,
    )
    from xgraph.fresh_worlds_gold import execute

    study_dir, outputs_dir, out, registration = map(
        Path, (study_dir, outputs_dir, out, registration)
    )
    study = json.loads((study_dir / "study.json").read_text())
    marker = study.get("mechanism_fallback", {})
    if marker.get("registration_sha256") != sha256_file(registration):
        raise ValueError("study is not bound to this factorial registration")
    spec = validate_fresh_cohort(study, registration, seal=marker.get("seal"))
    validate_fresh_validity(study_dir, study)
    for world in study["worlds"]:
        for condition in world["conditions"]:
            gold = execute(world["triples"], world["start"], world["relations"], condition["edits"])
            if condition["gold"] != gold["answers"] or len(gold["answers"]) != 1:
                raise ValueError(f"independent ID gold mismatch: {condition['record_id']}")
    inputs = [
        study_dir / "study.json",
        study_dir / "requests-factorial.jsonl",
        registration,
        Path(str(registration) + ".sha256"),
        Path(spec["document"]),
        Path(spec["parent_registration"]["path"]),
        study_dir / "validity.json",
        REPO / "scripts/fresh-worlds-v2-audit.py",
    ]
    inputs += [
        Path(__file__).with_name(name)
        for name in (
            "mechanism.py",
            "fresh_worlds.py",
            "fresh_worlds_v2.py",
            "fresh_worlds_gold.py",
            "fresh_worlds_analysis.py",
            "score.py",
        )
    ]
    if marker.get("seal") is not None:
        inputs.append(Path(marker["seal"]))
    scores = {}
    for family in FAMILY_NAMES:
        for cell in FACTORIAL_CELLS:
            path = outputs_dir / family / f"{cell}.jsonl"
            scores[f"{family}:{cell}"] = score_records(
                study, read_jsonl(path) if path.exists() else []
            )
            inputs += [
                p
                for p in [
                    path,
                    *(
                        Path(str(path) + s)
                        for s in (".summary.json", ".provenance.json", ".receipt.json")
                    ),
                ]
                if p.is_file()
            ]
    seconds = None
    if budget is not None and Path(budget).exists():
        inputs.append(Path(budget))
        ledger = json.loads(Path(budget).read_text())
        attempts = ledger.get("attempts", [])
        if (
            not attempts
            or any(a.get("status") not in ("complete", "failed") for a in attempts)
            or any(
                not math.isfinite(a.get("seconds", -1)) or a.get("seconds", -1) < 0
                for a in attempts
            )
            or not math.isclose(ledger["used_seconds"], sum(a["seconds"] for a in attempts))
        ):
            raise ValueError("factorial assurance requires a closed measured lifetime ledger")
        seconds = ledger["used_seconds"]
    contrasts = factorial_contrasts(study, scores, drift=spec["per_cell_drift"])
    assurance = factorial_assurance(
        study if study["config"]["split"] == "development" else None,
        scores,
        pilot_gpu_seconds=seconds,
        drift=spec["per_cell_drift"],
    )
    origin = provenance(inputs)
    shared = {
        "hypothesis": FRESH_HYPOTHESIS,
        "registration_sha256": sha256_file(registration),
        "provenance": origin,
    }
    report = {
        **shared,
        "status": contrasts["status"],
        "config": study["config"],
        "scores": scores,
        "interaction": contrasts,
        "claim_status": "confirmed"
        if study["config"]["split"] == "confirmation" and contrasts["confirmation_passes"]
        else "development or unconfirmed",
    }
    write_json(out.with_name("assurance.json"), {**assurance, **shared})
    write_json(out, report)
    return report


def freeze_fresh_confirmation(development, assurance, models, seed, out, registration):
    from xgraph.fresh_worlds import freeze_confirmation, write_json

    load_fresh_registration(registration)
    report = json.loads(Path(assurance).read_text())
    if (
        report.get("hypothesis") != FRESH_HYPOTHESIS
        or report.get("registration_sha256") != sha256_file(Path(registration))
        or report.get("joint_pattern_bits") != 24
        or not report.get("observed_development_gate_passed")
        or len(report.get("scenarios", [])) != 2
        or any(s["mc_lower_95"] < 0.8 for s in report["scenarios"])
    ):
        raise ValueError("confirmation requires its own factorial pilot assurance")
    pilot = report.get("pilot_gpu_seconds")
    if (
        report.get("status") != "pilot_projection"
        or report.get("admitted") is not True
        or report.get("trials") != 2000
        or report.get("nested_resamples") != 2000
        or report.get("seed") != 17
        or report.get("confirmation_worlds") != 256
        or report.get("cost_factor") != 1.5
        or [s.get("prior_mass_per_template") for s in report["scenarios"]] != [0.5, 2.0]
        or any(not math.isfinite(s["mc_lower_95"]) for s in report["scenarios"])
        or not isinstance(pilot, int | float)
        or not math.isfinite(pilot)
        or not 0 < pilot <= 7200
        or report.get("projected_gpu_seconds") != 6 * pilot
        or 6 * pilot > 36000
    ):
        raise ValueError(
            "confirmation requires the registered factorial assurance settings and costs"
        )
    seal = freeze_confirmation(development, assurance, models, seed, out)
    import subprocess

    from xgraph.fresh_worlds_queue import _commit, _hash_files

    repo = Path(__file__).resolve().parents[2]
    native = subprocess.check_output(
        ["git", "ls-files", "csrc", "CMakeLists.txt", "pyproject.toml"], cwd=repo, text=True
    ).splitlines()
    seal["files"].update(_hash_files([repo / name for name in native]))
    seal.update(
        hypothesis=FRESH_HYPOTHESIS,
        registration_sha256=sha256_file(Path(registration)),
        code_commit=_commit(repo, clean=True),
    )
    write_json(out, seal)
    return seal


def sample_members(members: list[dict], size=128, seed=17) -> list[dict]:
    """Equal hop allocation, uniform sequences then one uniform question per sequence.

    Inverse subset probabilities multiply source survey weights, then weights are
    calibrated back to each development hop's mass. All input orders are canonical.
    No answers, predictions, or historical error labels participate in sampling.
    """
    indexed(members)
    grouped = defaultdict(lambda: defaultdict(list))
    for row in sorted(members, key=lambda r: r["id"]):
        if not math.isfinite(row["weight"]) or row["weight"] <= 0:
            raise ValueError(f"{row['id']}: weight must be finite and positive")
        grouped[row["hops"]][row["cluster"]].append(row)
    if not grouped or size <= 0 or size % len(grouped):
        raise ValueError("sample size must be a positive multiple of the hop strata")
    n = size // len(grouped)
    if any(len(groups) < n for groups in grouped.values()):
        raise ValueError("not enough distinct sequences for one per chosen question")
    rng, result = random.Random(seed), []
    for _, groups in sorted(grouped.items()):
        chosen = []
        for cluster in sorted(rng.sample(sorted(groups), n)):
            row = rng.choice(groups[cluster])
            probability = n / len(groups) / len(groups[cluster])
            chosen.append(
                {
                    **row,
                    "source_weight": row["weight"],
                    "subset_probability": probability,
                    "weight": row["weight"] / probability,
                }
            )
        mass = sum(r["weight"] for group in groups.values() for r in group)
        scale = mass / sum(r["weight"] for r in chosen)
        result.extend({**r, "weight": r["weight"] * scale} for r in chosen)
    return sorted(result, key=lambda r: (r["hops"], r["cluster"], r["id"]))


def generation_command(out: Path, source: Path, family: str, cell: str) -> list[str]:
    """Native launcher, independently setting both switches and resetting inherited options."""
    selection, delivery = CELLS[cell]
    model = MODELS / FAMILIES[family][0]
    environment = {
        "XG_NEED_GIB": 60 if family == "qwen38" else 75,
        "XG_AVAILABLE_WAIT": 43200,
        "XG_RUN_SECONDS": 1800,
        "XG_START_ATTEMPTS": 1,
        "XG_KG_INDEX": source / "target/mintqa-rematch/indexes" / family,
        "XG_TOKENIZER": model / "tokenizer.json",
        "XG_PATHS": 1,
        "XG_COMMITTED_PATHS": 1,
        "XG_ESCAPES": 0,
        "XG_UNQUOTED_STARTS": 1,
        "XG_START_RANK": "sitelinks",
        "XG_OPEN": 1,
        "XG_END_AFTER_ANSWER": 1,
        "XG_RELATIONS": "frontier",
        "XG_EXPRESSIONS": 0,
        "XG_STOP": "",
        "XG_KG_VERSIONS": "",
        "XG_LORA": "",
        "XG_DERIVATION": 0,
        "XG_RELEASE_MASS": "",
        "XG_RELEASE_TOP": 0,
        "XG_ESCAPE_BIAS": "",
        "XG_REPEATS": 1,
        "XG_BOUND_ESCAPES": 1,
        "XG_OBSERVE": 0 if cell == "paths-allfix" else 3,
        "XG_OBSERVE_SELECT": selection,
        "XG_OBSERVE_ANSWER": delivery,
        "XG_MENU": 0,
        "XG_PATH_DEMOS": "",
        "XG_OBSERVE_WIDTH": 0,
        "XG_OBSERVE_DEFER": 0,
        "XG_MENU_CLOSE_NEXT": 0,
        "XG_OBSERVE_REQUIRED": 0,
        "XG_OBSERVE_NO_LOOP": 0,
        "XG_OBSERVE_TERMINAL": 0,
        "XG_AUDIT": 0,
        "XG_PAGECACHE_ROOTS": f"{source / 'data'}:{model}",
        "OPENBLAS_NUM_THREADS": 1,
        "OMP_NUM_THREADS": 1,
    }
    return [
        "env",
        *(f"{key}={value}" for key, value in environment.items()),
        "nice",
        "-n",
        "10",
        str(REPO / "scripts/generate.sh"),
        str(model),
        str(out / ("requests.jsonl" if cell == "paths-allfix" else "observed-requests.jsonl")),
        "constrained",
        str(out / family / f"{cell}.jsonl"),
        "--max-tokens",
        "1024",
        "--max-model-len",
        "16384",
        "--max-num-seqs",
        "64",
        "--kv-cache-gib",
        "16" if family == "gemma31" else "8",
        "--seed",
        "17",
        *(["--ban-tokens", "<|channel>"] if family == "gemma31" else []),
    ]


def reuse_mismatches(old_argv, wanted_argv, request_bytes_match: bool) -> list[str]:
    """Compare every effective flag, apart from separately verified host locations/admission."""
    ignored = {
        "--requests",
        "--out",
        "--memory-share",
        "--provenance-only",
    }

    def options(argv):
        result, at = {}, 0
        while at < len(argv):
            flag = argv[at]
            at += 1
            if not flag.startswith("--"):
                continue
            value = True
            if at < len(argv) and not argv[at].startswith("--"):
                value, at = argv[at], at + 1
            if flag not in ignored:
                if flag in result:
                    raise ValueError(f"duplicate generation argument {flag}")
                result[flag] = value
        return result

    old, wanted = options(old_argv), options(wanted_argv)
    return [
        *(
            f"argv:{flag}"
            for flag in sorted(old.keys() | wanted.keys())
            if old.get(flag) != wanted.get(flag)
        ),
        *([] if request_bytes_match else ["request_bytes"]),
    ]


def delivery_checker(index, tokenizer):
    """Resolve an exact answer label inside retained answer rows, without anchor fallback."""
    import numpy as np

    from xgraph.oracle import MAX_REACHED

    labels = index.labels
    owners = {qid: at for at, qid in enumerate(labels.owner_ids)}

    def delivered(answer, support):
        if not answer or not support:
            return False
        tokens = tuple(tokenizer.encode(index.span_syntax.label_prefix + answer))
        lo, hi = labels.prefix_range(tokens)
        rows = np.flatnonzero(np.isin(labels.owners, [owners[qid] for qid in support]))[
            :MAX_REACHED
        ]
        return any(lo <= row < hi and labels.row_tokens(row) == tokens for row in rows)

    return delivered


def diagnose(record: dict, gold: dict, executor, label, delivered) -> dict:
    """Separate stopping, gold-suffix selection loss, and valid endpoint delivery.

    A selection error removes all gold answers reachable by the actual prefix plus
    the gold relation suffix. Earlier selections are retained; only the current
    selection changes. This is a gold-assisted diagnostic, never a policy input.
    Endpoint reachability uses canonical labels and the benchmark answer matcher.
    Delivery validity instead resolves the written label exactly in the capped
    final answer support, including aliases. It is independent of answer accuracy.
    """
    written, answer = record_path(record), final_answer(record.get("text") or "")
    costs = usage(record)
    result = {
        "id": record["id"],
        "correct": not record_failure(record) and answer_contains(answer, gold["answer"]),
        "gold_hops": len(gold["triple"]),
        "written_hops": None,
        "stopping": "unclosed",
        "truncated": record.get("finish_reason") == "length",
        "selection_decisions": 0,
        "selection_opportunities": 0,
        "selection_errors": 0,
        "selection_trace": [],
        "deliverable": False,
        "delivery_failure": False,
        "cap_events": [],
        **costs,
        "total_tokens": costs["prompt_tokens"] + costs["generated_tokens"],
        "non_observation_generated_tokens": costs["generated_tokens"] - costs["observation_tokens"],
    }
    if written is None:
        closed = spans(record.get("text") or "")
        if closed and closed[0][0] == "entity":
            result.update(written_hops=0, stopping="early")
        return result
    start, relations = written
    hops = len(relations)
    result.update(
        written_hops=hops,
        stopping="early"
        if hops < result["gold_hops"]
        else ("late" if hops > result["gold_hops"] else "matched"),
    )
    executed = executor.execute_set(start, relations)
    support = executed["reached"]
    result["cap_events"] = executed["cap_events"]
    result["deliverable"] = bool(support)
    result["delivery_failure"] = bool(support) and not delivered(answer, support)
    selections = getattr(relations, "selections", {})

    def has_gold(entities):
        return any(answer_contains(label(e), gold["answer"]) for e in entities)

    for boundary, selection in sorted(selections.items()):
        result["selection_decisions"] += 1
        if boundary > result["gold_hops"]:
            result["selection_trace"].append({"boundary": boundary, "assessable": False})
            continue
        fixed = list(relations[:boundary]) + [t[1] for t in gold["triple"][boundary:]]
        prior = {h: name for h, name in selections.items() if h < boundary}
        before = executor.execute_set(start, PathRelations(fixed, prior))
        after = executor.execute_set(start, PathRelations(fixed, {**prior, boundary: selection}))
        opportunity, retained = has_gold(before["reached"]), has_gold(after["reached"])
        result["selection_opportunities"] += int(opportunity)
        result["selection_errors"] += int(opportunity and not retained)
        result["selection_trace"].append(
            {
                "boundary": boundary,
                "selection": selection,
                "assessable": True,
                "gold_reachable_before": opportunity,
                "gold_reachable_after": retained,
                "before_cap_events": before["cap_events"],
                "after_cap_events": after["cap_events"],
            }
        )
    return result


def interaction(cells: dict, members: list[dict], resamples=10_000, seed=17) -> dict:
    """Bootstrap paired DD=(s1d1-s1d0)-(s0d1-s0d0), by sequence within hop."""
    ids = set(indexed(members))
    if any(
        cell not in cells or set(cells[cell]) != ids for cell in ("s0d0", "s0d1", "s1d0", "s1d1")
    ):
        raise ValueError("interaction requires complete, identical question ids in all four cells")
    values = [
        float(cells["s1d1"][r["id"]]["correct"])
        - float(cells["s1d0"][r["id"]]["correct"])
        - float(cells["s0d1"][r["id"]]["correct"])
        + float(cells["s0d0"][r["id"]]["correct"])
        for r in members
    ]
    return {
        **clustered_difference(values, [0.0] * len(members), members, resamples, seed),
        "contrast": "(s1d1-s1d0)-(s0d1-s0d0)",
        "units": "accuracy proportion",
        "status": "exploratory development diagnostic; unadjusted 95% interval",
    }


def render_queue(out: Path, stages: list[dict]) -> str:
    """An inert preview by default; only the GPU owner invokes --run."""
    cpu = [str(REPO / ".venv/bin/python"), "-m", "xgraph.mechanism"]
    lines = [
        "#!/usr/bin/env bash",
        "# Development only. GPU owner: invoke --run explicitly.",
        "set -euo pipefail",
        "mode=${1:---dry-run}",
        'if (($# > 1)) || [[ "$mode" != --dry-run && "$mode" != --run ]]; then',
        "    printf 'usage: mechanism.sh [--dry-run|--run]\\n' >&2; exit 2",
        "fi",
        f"cd {shlex.quote(str(REPO))}",
        "log=/home/jonas/git/xgraph/target/queue/mechanism.log",
        'if [[ "$mode" == --run ]]; then',
        f"    exec 9>{shlex.quote(str(out / 'queue.lock'))}",
        "    flock -n 9 || exit 2",
        "    " + shlex.join([*cpu, "verify", "--out", str(out)]),
        '    mkdir -p "$(dirname "$log")"',
        "fi",
    ]
    for stage in stages:
        family, cell = stage["family"], stage["cell"]
        if stage["status"] == "reused":
            lines.append(f"# exact verified reuse: {family} {cell}")
            continue
        command = shlex.join(stage["command"])
        output = out / family / f"{cell}.jsonl"
        lines.extend(
            [
                'if [[ "$mode" == --dry-run ]]; then',
                "    printf '%s\\n' " + shlex.quote(command),
                "else",
                f"    mkdir -p {shlex.quote(str(output.parent))}",
                f"    if {command} >{shlex.quote(str(output) + '.log')} 2>&1; then",
                "        status=0",
                "    else",
                "        status=$?",
                "    fi",
                f"    printf 'mechanism %s %s exit %s %s\\n' {family} {cell} "
                '"$status" "$(date -Is)" >>"$log"',
                '    ((status == 0)) || exit "$status"',
                "fi",
            ]
        )
    score = shlex.join([*cpu, "score", "--out", str(out)])
    lines.extend(
        [
            'if [[ "$mode" == --dry-run ]]; then',
            "    printf '%s\\n' " + shlex.quote(score),
            "else",
            f"    {score}",
            "fi",
            "",
        ]
    )
    return "\n".join(lines)


def _safe(path: Path) -> Path:
    path = path.resolve()
    if "codex-final" in path.parts:
        raise ValueError("codex-final is outside this development study")
    return path


def _effective_argv(command, cell):
    launcher = command.index(str(REPO / "scripts/generate.sh"))
    model, requests, arm, output = command[launcher + 1 : launcher + 5]
    environment = dict(s.split("=", 1) for s in command[:launcher] if s.startswith("XG_"))
    selection, delivery = CELLS[cell]
    return [
        "python3",
        "generate.py",
        "--model",
        model,
        "--requests",
        requests,
        "--arm",
        arm,
        "--out",
        output,
        "--kg-index",
        environment["XG_KG_INDEX"],
        "--tokenizer",
        environment["XG_TOKENIZER"],
        "--no-escapes",
        "--paths",
        *(["--observe-paths", "3"] if cell != "paths-allfix" else []),
        *(["--observe-select"] if selection else []),
        *(["--observe-answer"] if delivery else []),
        "--committed-paths",
        "--unquoted-starts",
        "--end-after-answer",
        "--relations",
        "frontier",
        "--start-rank",
        "sitelinks",
        "--open",
        *command[launcher + 5 :],
    ]


def prepare(out: Path, source: Path, size=128) -> None:
    """Freeze development selection before opening any candidate generation records."""
    from xgraph.generate import model_revision

    out, source = _safe(out), _safe(source)
    if not (out.is_relative_to(REPO / "target") or out.is_relative_to(Path("/tmp"))):
        raise ValueError("output must be in this worktree's target/ or /tmp")
    if out.exists():
        raise ValueError(f"output already exists: {out}")
    prepared = source / "target/mintqa-rematch"
    panels_path = _safe(prepared / "panels/panels.json")
    members = json.loads(panels_path.read_text())["panels"]["development"]["members"]
    chosen = sample_members(members, size, 17)
    write_json(
        out / "sample.json",
        {
            "status": "frozen before new generations",
            "seed": 17,
            "selected_at": datetime.now(UTC).isoformat(),
            "source": str(panels_path),
            "source_sha256": sha256_file(panels_path),
            "selection": "equal hop allocation; uniform sequences; "
            "one uniform question per sequence",
            "weighting": "source weight / subset probability, calibrated to source hop mass",
            "members": chosen,
        },
    )
    # Only question ids, hops, relation sequences and source weights were used above.
    ids = {r["id"] for r in chosen}
    original_requests = _safe(prepared / "panels/development/requests.jsonl")
    requests = indexed(read_jsonl(original_requests))
    if not ids <= requests.keys():
        raise ValueError("development requests do not cover the selected sample")
    write_jsonl(
        out / "requests.jsonl",
        [{**requests[r["id"]], "fixed_path_demonstrations": True} for r in chosen],
    )
    from xgraph.prompts import observed_requests
    from xgraph.syntax import SpanSyntax

    # Convert before freezing: preserving raw blind examples silently suppresses
    # the historical generator's observed-demonstration substitution.
    observed = observed_requests([requests[r["id"]] for r in chosen], 3, SpanSyntax())
    write_jsonl(
        out / "observed-requests.jsonl",
        [{**request, "fixed_path_demonstrations": True} for request in observed],
    )
    gold, inputs = {}, [panels_path, original_requests]
    for split in ("pop", "ti"):
        path = _safe(source / f"data/external/mintqa-full/questions/{split}.jsonl")
        inputs.append(path)
        with path.open() as stream:
            for number, line in enumerate(stream):
                qid = f"mintqa-{split}-{number}"
                if qid in ids:
                    row = json.loads(line)
                    gold[qid] = {key: row[key] for key in ("answer", "triple")}
    if set(gold) != ids:
        raise ValueError("development gold does not cover the selected sample")
    for row in chosen:
        chain = gold[row["id"]]["triple"]
        if len(chain) != row["hops"] or " · ".join(t[1] for t in chain) != row["cluster"]:
            raise ValueError(f"{row['id']}: gold chain differs from frozen development stratum")
    write_json(out / "gold.json", gold)
    stages, reuse, weights, historical, models = [], [], [], [], {}
    for family in FAMILY_NAMES:
        model = _safe(MODELS / FAMILIES[family][0])
        models[family] = {"path": str(model), "revision": model_revision(model)}
        revision_metadata = model / ".cache/huggingface/download/config.json.metadata"
        if revision_metadata.is_file():
            inputs.append(revision_metadata)
        inputs += sorted(model.glob("*.json")) + sorted(model.glob("*.jinja"))
        inputs += sorted((prepared / "indexes" / family).glob("*"))
        weights.extend(
            {
                "path": str(_safe(p)),
                "bytes": p.stat().st_size,
                "mtime_ns": p.stat().st_mtime_ns,
            }
            for p in sorted(model.glob("*.safetensors"))
        )
        for cell in CELLS:
            command = generation_command(out, source, family, cell)
            wanted = _effective_argv(command, cell)
            stage = {
                "family": family,
                "cell": cell,
                "command": command,
                "expected_argv": wanted,
                "status": "planned",
                "output": str(out / family / f"{cell}.jsonl"),
            }
            old_name = {
                "paths-allfix": "paths-allfix",
                "s0d0": "paths-observed-3",
                "s1d1": "paths-observed-3-select",
            }.get(cell)
            if old_name:
                candidate = _safe(prepared / family / "development" / f"{old_name}.jsonl")
                sidecar = Path(str(candidate) + ".provenance.json")
                summary = Path(str(candidate) + ".summary.json")
                if sidecar.is_file():
                    metadata = json.loads(_safe(sidecar).read_text())
                    inputs.append(sidecar)
                    previous = metadata["provenance"]["command"]
                    request_hashes = {
                        item["sha256"]
                        for item in metadata["provenance"]["inputs"]
                        if item["path"] == str(original_requests)
                    }
                    request_file = Path(wanted[wanted.index("--requests") + 1])
                    mismatches = reuse_mismatches(
                        previous, wanted, sha256_file(request_file) in request_hashes
                    )
                    reuse.append(
                        {
                            "family": family,
                            "cell": cell,
                            "candidate": str(candidate),
                            "argv": previous,
                            "expected_argv": wanted,
                            "mismatches": mismatches,
                            "status": "rejected" if mismatches else "eligible",
                            "source_commit": metadata["provenance"]["repo_commit"],
                        }
                    )
                    if not mismatches:
                        # Exact request file equality includes order and caller-owned demos.
                        rows = indexed(read_jsonl(candidate))
                        if set(rows) != ids:
                            raise ValueError(f"eligible reuse has incomplete ids: {candidate}")
                        inputs.extend([candidate, summary])
                        write_jsonl(Path(stage["output"]), [rows[r["id"]] for r in chosen])
                        stage["status"] = "reused"
                        stage["reuse_source"] = str(candidate)
                if summary.is_file():
                    inputs.append(summary)
                    saved = json.loads(_safe(summary).read_text())
                    historical.append(
                        {
                            "family": family,
                            "cell": cell,
                            "source": str(summary),
                            "requests": saved["requests"],
                            "seconds": saved["seconds"],
                            "max_tokens": saved["max_tokens"],
                        }
                    )
            stages.append(stage)
    planned = sum(s["status"] == "planned" for s in stages)
    budget = {
        "status": "forecast and enforced ceiling; no generation launched",
        "questions_per_cell": size,
        "new_cells": planned,
        "gpu_hour_limit": 6,
        "container_timeout_seconds": 1800,
        "timeout_kill_grace_seconds": 15,
        "maximum_container_hours": planned * (1800 + 15) / 3600,
        "reserved_cleanup_seconds_per_cell": 60,
        "maximum_with_cleanup_hours": planned * (1800 + 15 + 60) / 3600,
        "startup_attempts": 1,
        "historical_workload_timings": historical,
        "limits": "serial containers; model loading included; CPU build/admission waits excluded; "
        "timeouts abort the queue and never justify dropping questions",
    }
    if budget["maximum_with_cleanup_hours"] > 6:
        raise ValueError("queue exceeds six GPU-hours")
    forecasts = {}
    for family in FAMILY_NAMES:
        times = [h for h in historical if h["family"] == family]
        if times:
            per_cell = max(
                h["seconds"] / h["requests"] * size * 1024 / h["max_tokens"] for h in times
            )
            forecasts[family] = {
                "seconds_per_cell": 300 + 2 * per_cell,
                "assumption": "300s startup plus twice slowest historical workload "
                "scaled by questions "
                "and token ceiling; uncontrolled timing, not a measured serving claim",
            }
    budget["forecast_by_family"] = forecasts
    write_json(out / "budget.json", budget)
    write_json(out / "reuse.json", reuse)
    inputs.extend(
        [
            _safe(source / "data/external/mintqa-full/kg.sqlite"),
            REPO / "CMakeLists.txt",
            REPO / "scripts/gpu-headroom.sh",
        ]
    )
    inputs += [
        out / name
        for name in (
            "sample.json",
            "requests.jsonl",
            "observed-requests.jsonl",
            "gold.json",
            "budget.json",
            "reuse.json",
        )
    ]
    manifest = {
        "artifact_kind": "xgraph.mechanism.development",
        "version": 1,
        "source_root": str(source),
        "seed": 17,
        "sample_size": size,
        "code_sha256": _code_digest(),
        "model_weights_stat": weights,
        "models": models,
        "stages": stages,
        "provenance": provenance(sorted(set(map(_safe, inputs)))),
        "inference": {
            "primary": "within-family accuracy (s1d1-s1d0)-(s0d1-s0d0)",
            "bootstrap": "10000 paired relation-sequence cluster draws within hop; fixed hop mass",
            "interval": "two-sided unadjusted percentile 95%; seed 17",
            "sensitivity": "omit sport · country of origin · capital; leave one sequence out",
            "scope": "exploratory exposed development, no confirmatory or new method claim",
            "demonstrations": "the same four observed examples in the observation factorial; "
            "blind controls retain the original blind examples; "
            "native interface instructions vary with observation and selection",
            "selection_errors": "gold-assisted actual-prefix plus gold-suffix reachability loss; "
            "canonical endpoint labels, conditional on gold reachable before selection",
            "delivery_failure": "closed executable path with nonempty capped answer support but no "
            "completed answer label in that support; independent of gold correctness",
        },
    }
    write_json(out / "manifest.json", manifest)
    (out / "manifest.sha256").write_text(sha256_file(out / "manifest.json") + "\n")
    queue = out / "mechanism.sh"
    queue.write_text(render_queue(out, stages))
    queue.chmod(0o755)


def verify(out: Path, *, for_run=False) -> dict:
    """Reject altered code/inputs, or an attempted run over an existing stage."""
    out = _safe(out)
    if sha256_file(out / "manifest.json") != (out / "manifest.sha256").read_text().strip():
        raise ValueError("mechanism manifest changed")
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["code_sha256"] != _code_digest():
        raise ValueError("mechanism code changed since freeze")
    for item in manifest["provenance"]["inputs"]:
        if sha256_file(_safe(Path(item["path"]))) != item["sha256"]:
            raise ValueError(f"mechanism input changed: {item['path']}")
    for item in manifest["model_weights_stat"]:
        stat = _safe(Path(item["path"])).stat()
        if (stat.st_size, stat.st_mtime_ns) != (item["bytes"], item["mtime_ns"]):
            raise ValueError(f"model weights changed: {item['path']}")
    if for_run:
        for stage in manifest["stages"]:
            if stage["status"] == "planned" and any(
                Path(stage["output"] + suffix).exists()
                for suffix in ("", ".summary.json", ".provenance.json", ".log")
            ):
                raise ValueError(
                    f"stage already exists; preserve incomplete run: {stage['output']}"
                )
    return manifest


def score(out: Path, resamples=10_000) -> dict:
    """Score complete cells only, retaining every selected id; absent cells stay incomplete."""
    from xgraph.index import load_kg_index
    from xgraph.programs import MatchedExecutor
    from xgraph.tokenizer import ServedTokenizer
    from xgraph.toolkg import KGStore

    manifest = verify(out)
    members = json.loads((out / "sample.json").read_text())["members"]
    gold = json.loads((out / "gold.json").read_text())
    ids, source = set(indexed(members)), Path(manifest["source_root"])
    report = {"status": "INCOMPLETE", "missing_cells": [], "cells": {}, "interactions": {}}
    family_rows, executors, delivery, store, score_inputs = defaultdict(dict), {}, {}, None, []
    for stage in manifest["stages"]:
        family, cell, output = stage["family"], stage["cell"], Path(stage["output"])
        if not output.exists():
            report["missing_cells"].append(f"{family}/{cell}")
            continue
        records = indexed(read_jsonl(output))
        if set(records) != ids:
            raise ValueError(f"incomplete or extra question ids: {output}")
        if stage["status"] != "reused":
            sidecar = Path(str(output) + ".provenance.json")
            metadata = json.loads(sidecar.read_text())
            expected = stage["expected_argv"]
            request_file = Path(expected[expected.index("--requests") + 1])
            request_hash = sha256_file(request_file)
            hashes = {
                p["sha256"]
                for p in metadata["provenance"]["inputs"]
                if p["path"] == str(request_file)
            }
            mismatches = reuse_mismatches(
                metadata["provenance"]["command"], stage["expected_argv"], request_hash in hashes
            )
            if mismatches:
                raise ValueError(f"generation configuration differs: {output}: {mismatches}")
            score_inputs.append(sidecar)
        score_inputs.append(output)
        if family not in executors:
            tokenizer = ServedTokenizer.from_file(MODELS / FAMILIES[family][0] / "tokenizer.json")
            index = load_kg_index(source / "target/mintqa-rematch/indexes" / family)
            executors[family] = MatchedExecutor(
                index, tokenizer, start_rank="sitelinks", unquoted_starts=True
            )
            delivery[family] = delivery_checker(index, tokenizer)
        if store is None:
            store = KGStore(source / "data/external/mintqa-full/kg.sqlite")
        details = {
            qid: diagnose(
                records[qid],
                gold[qid],
                executors[family],
                lambda e, store=store: store.label(int(e[1:])),
                delivery[family],
            )
            for qid in ids
        }
        family_rows[family][cell] = details
        mass = sum(r["weight"] for r in members)

        def mean(field, details=details, mass=mass):
            return sum(r["weight"] * details[r["id"]][field] for r in members) / mass

        def total(field, details=details):
            return sum(r[field] for r in details.values())

        report["cells"][f"{family}/{cell}"] = {
            "questions": len(ids),
            "accuracy": mean("correct"),
            "unweighted_accuracy": total("correct") / len(ids),
            "stopping": {
                kind: {
                    "questions": sum(r["stopping"] == kind for r in details.values()),
                    "weighted_rate": sum(
                        m["weight"] * (details[m["id"]]["stopping"] == kind) for m in members
                    )
                    / mass,
                }
                for kind in ("early", "matched", "late", "unclosed")
            },
            "weighted_per_question": {
                field: mean(field)
                for field in (
                    "truncated",
                    "deliverable",
                    "delivery_failure",
                    "selection_errors",
                    "selection_decisions",
                    "selection_opportunities",
                    "prompt_tokens",
                    "generated_tokens",
                    "observation_tokens",
                    "non_observation_generated_tokens",
                    "total_tokens",
                    "model_calls",
                )
            },
            "selection_errors": total("selection_errors"),
            "selection_opportunities": total("selection_opportunities"),
            "selection_error_rate": (
                mean("selection_errors") / mean("selection_opportunities")
                if mean("selection_opportunities")
                else None
            ),
            "delivery_failures": total("delivery_failure"),
            "deliverable": total("deliverable"),
            "delivery_failure_rate": (
                mean("delivery_failure") / mean("deliverable") if mean("deliverable") else None
            ),
            "per_question": [details[r["id"]] for r in members],
        }
    for family, cells in family_rows.items():
        if not all(cell in cells for cell in ("s0d0", "s0d1", "s1d0", "s1d1")):
            continue
        result = interaction(cells, members, resamples)
        omit = [r for r in members if r["cluster"] != "sport · country of origin · capital"]
        result["without_sport"] = interaction(
            {c: {r["id"]: rows[r["id"]] for r in omit} for c, rows in cells.items()},
            omit,
            resamples,
        )
        result["sport_questions"] = len(members) - len(omit)
        result["simple_effects"] = {}
        for first, second in (
            ("s1d0", "s0d0"),
            ("s1d1", "s0d1"),
            ("s0d1", "s0d0"),
            ("s1d1", "s1d0"),
            ("s0d0", "paths-allfix"),
            ("s1d1", "paths-allfix"),
        ):
            if first in cells and second in cells:
                result["simple_effects"][f"{first}-{second}"] = clustered_difference(
                    [float(cells[first][r["id"]]["correct"]) for r in members],
                    [float(cells[second][r["id"]]["correct"]) for r in members],
                    members,
                    resamples,
                    17,
                )
        loso = []
        for cluster in sorted({r["cluster"] for r in members}):
            kept = [r for r in members if r["cluster"] != cluster]
            trimmed = {c: {r["id"]: rows[r["id"]] for r in kept} for c, rows in cells.items()}
            loso.append(interaction(trimmed, kept, resamples=1)["difference"])
        result["leave_one_sequence_out_range"] = [min(loso), max(loso)]
        report["interactions"][family] = result
    if not report["missing_cells"]:
        report["status"] = "completed exploratory development diagnostic"
    report["provenance"] = provenance([out / "manifest.json", *score_inputs])
    pending = out / "analysis.pending.json"
    pending.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    pending.replace(out / "analysis.json")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "verify",
            "score",
            "fresh-requests",
            "fresh-analyze",
            "fresh-freeze",
            "fresh-confirmation",
        ),
    )
    parser.add_argument("--out", type=Path, default=REPO / "target/queue-mechanism")
    parser.add_argument("--source-root", type=Path, default=Path("/home/jonas/git/xgraph"))
    parser.add_argument("--study", type=Path)
    parser.add_argument("--outputs-dir", type=Path)
    parser.add_argument("--budget", type=Path)
    parser.add_argument("--registration", type=Path, default=FRESH_REGISTRATION)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--assurance", type=Path)
    parser.add_argument("--models", type=Path, default=MODELS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seal", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.out, args.source_root)
    elif args.command == "verify":
        verify(args.out, for_run=True)
    elif args.command == "score":
        score(args.out)
    elif args.command == "fresh-requests":
        if args.study is None:
            parser.error("fresh-requests needs --study")
        prepare_fresh_requests(args.study, args.registration, seal=args.seal)
    elif args.command == "fresh-analyze":
        if args.study is None or args.outputs_dir is None:
            parser.error("fresh-analyze needs --study and --outputs-dir")
        analyze_fresh(args.study, args.outputs_dir, args.out, args.registration, args.budget)
    elif args.command == "fresh-freeze":
        if args.development is None or args.assurance is None or args.seed is None:
            parser.error("fresh-freeze needs --development, --assurance and --seed")
        freeze_fresh_confirmation(
            args.development, args.assurance, args.models, args.seed, args.out, args.registration
        )
    else:
        from xgraph.fresh_worlds import write_study

        load_fresh_registration(args.registration)
        if args.seal is None:
            parser.error("fresh-confirmation needs --seal")
        seal = json.loads(args.seal.read_text())
        if seal.get("hypothesis") != FRESH_HYPOTHESIS or seal.get(
            "registration_sha256"
        ) != sha256_file(args.registration):
            parser.error("fresh-confirmation requires its own factorial seal")
        write_study(
            args.out,
            FreshConfig(**seal["config"]),
            {f: Path(p["path"]) for f, p in seal["tokenizers"].items()},
            list(map(Path, seal["collision_files"])),
            list(map(Path, seal["common_files"])),
            confirmation_seal=args.seal,
        )
        prepare_fresh_requests(args.out, args.registration, seal=args.seal)


if __name__ == "__main__":
    main()
