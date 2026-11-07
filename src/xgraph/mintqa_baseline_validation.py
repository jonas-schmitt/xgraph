"""Register a gold-free MINTQA validation subpanel and owner-run ToG queue.

WT and agent inputs are local owner exports after freeze. This module never
opens another checkout's validation generations or launches their engines.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path

from xgraph.cor import linked_requests
from xgraph.mintqa_panels import indexed, pilot_members, read_jsonl, write_json, write_jsonl
from xgraph.mintqa_queue import REPO, Step, run_step
from xgraph.mintqa_rematch import (
    audit_complete_usage,
    clustered_difference,
    combined_usage,
    score_panel,
    usage,
)
from xgraph.mintqa_validation import file_identity, git_state, model_identity
from xgraph.provenance import provenance, sha256_file

FAMILIES = {
    "qwen38": ("Qwen3.8-27B-FP8", 60, 8, "agent-strong-8"),
    "gemma31": ("gemma-4-31B-it-FP8-dynamic", 75, 16, "agent-feedback-8"),
}
NOT_BEFORE = date(2026, 11, 24)
SELECTION_RULE = (
    "highest population-weighted paired accuracy among qualified published transfers; "
    "ties: lower measured workload wall time; dog-gold is excluded"
)


def validation_sample(parent, questions, requests, size=128, seed=17):
    """Use ids, hop strata, relation sequences and parent weights, without gold."""
    development = parent["panels"]["development"]["members"]
    validation = parent["panels"]["validation"]["members"]
    if {r["id"] for r in development} & {r["id"] for r in validation} or {
        r["cluster"] for r in development
    } & {r["cluster"] for r in validation}:
        raise ValueError("development and validation must be relation-disjoint")
    members = pilot_members(validation, size, seed)
    asked, prompts = indexed(questions), indexed(requests)
    if set(asked) != set(prompts) or set(asked) != set(indexed(validation)):
        raise ValueError("parent validation definitions have different ids")
    return (
        members,
        [{k: asked[r["id"]][k] for k in ("id", "utterance", "complexity")} for r in members],
        [prompts[r["id"]] for r in members],
    )


def generation_command(family, out, source, models, prompts):
    model_name, need, kv, _ = FAMILIES[family]
    model = models / model_name
    # Empty environment excludes ambient XG_*/XGRAPH_* policy changes.
    environment = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", "/home/jonas"),
        "LANG": "C.UTF-8",
        "UV_CACHE_DIR": str(REPO / ".uv-cache"),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "XG_NEED_GIB": str(need),
        "XG_AVAILABLE_WAIT": "43200",
        "XG_START_ATTEMPTS": "1",
        "XG_RUN_SECONDS": "21600",
        "XG_GPU_IDLE": "1",
        "XG_TOOLS_DB": str(source / "data/external/mintqa-full/kg.sqlite"),
        "XG_PAGECACHE_ROOTS": f"{source / 'data'}:{model}",
    }
    return [
        "env",
        "-i",
        *(f"{k}={v}" for k, v in environment.items()),
        "nice",
        "-n",
        "10",
        str(REPO / "scripts/generate.sh"),
        str(model),
        str(out / "inputs/linked.jsonl"),
        "tog",
        str(out / family / "validation/tog-depth4.jsonl"),
        "--max-tokens",
        "256",
        "--max-model-len",
        "16384",
        "--max-num-seqs",
        "8",
        "--kv-cache-gib",
        str(kv),
        "--seed",
        "17",
        "--tog-prompts",
        str(prompts),
        "--tog-depth",
        "4",
        "--tog-variant",
        "tog",
        "--tog-batch-size",
        "32",
    ]


def estimate(timings):
    result = {}
    for family in FAMILIES:
        row = timings["families"][family]["tog-depth4"]
        if row["questions"] != 64 or row["question_scheduling"] != "step_synchronous":
            raise ValueError("estimate needs the completed batched 64-question pilot")
        work, startup = row["wall_seconds"], row["startup_seconds"]
        result[family] = {
            "basis_workload_seconds": work,
            "basis_startup_seconds": startup,
            "gpu_hours": (2 * work + startup) / 3600,
            "double_workload_gpu_hours": (4 * work + startup) / 3600,
        }
    return result


def register(args):
    commit = git_state()
    if datetime.now(UTC).date() >= NOT_BEFORE:
        raise ValueError("registration must precede the 2026-11-24 evaluation window")
    out, source = args.out.resolve(), args.source.resolve()
    if not out.is_relative_to(REPO / "target"):
        raise ValueError("registration outputs must stay in this worktree's target/")
    if (out / "registration.json").exists() or any(out.glob("*/validation/*")):
        raise ValueError("registration or results already exist; choose a new output directory")
    panel_path = source / "target/mintqa-rematch/panels/panels.json"
    parent = json.loads(panel_path.read_text())
    if len(parent["panels"]["validation"]["members"]) != 2048:
        raise ValueError("the parent must be the sealed 2048-question validation definition")
    definition_path = args.definitions / "target/final-validation/freeze-manifest.json"
    definition = json.loads(definition_path.read_text())
    parent_requests = panel_path.parent / "validation/requests.jsonl"
    parent_questions = panel_path.parent / "validation/questions.json"
    frozen_inputs = {Path(r["path"]).resolve(): r["sha256"] for r in definition["inputs"]}
    if (
        Path(definition["panels"]).resolve() != panel_path.resolve()
        or definition["panel_sha256"] != sha256_file(parent_requests)
        or any(
            frozen_inputs.get(p.resolve()) != sha256_file(p)
            for p in (panel_path, parent_requests, parent_questions)
        )
    ):
        raise ValueError("sealed panel definition differs from the proposed parent")
    if any(
        definition["policy_selection"].get(f) != "WT"
        or definition["strongest_agents"].get(f) != values[3]
        for f, values in FAMILIES.items()
    ):
        raise ValueError("WT or strongest-agent declaration differs from the sealed definition")
    inputs = [
        panel_path,
        panel_path.parent / "validation/questions.json",
        panel_path.parent / "validation/requests.jsonl",
        args.linked,
        args.prompts,
        args.timings,
        args.definitions / "src/xgraph/mintqa_panels.py",
        args.definitions / "src/xgraph/mintqa_validation.py",
        args.definitions / "src/xgraph/policy_catalog.py",
        definition_path,
    ]
    members, questions, requests = validation_sample(
        parent, json.loads(inputs[1].read_text()), read_jsonl(inputs[2])
    )
    linked = linked_requests(requests, read_jsonl(args.linked))
    panels = out / "panels"
    write_json(panels / "validation/questions.json", questions)
    write_jsonl(panels / "validation/requests.jsonl", requests)
    write_jsonl(out / "inputs/linked.jsonl", linked)
    manifest = {
        "artifact_kind": "xgraph.mintqa_panels",
        "version": 1,
        "population_by_group": parent["population_by_group"],
        "subpanel": {
            "parent_panel": "validation",
            "size": 128,
            "seed": 17,
            "method": "seeded whole-cluster subset sum within hops; at most one split per hop",
            "interpretation": "conditional internal validation subpanel; descriptive only",
        },
        "panels": {"validation": {"members": members}},
    }
    write_json(panels / "panels.json", manifest)
    inputs += [
        panels / "panels.json",
        panels / "validation/questions.json",
        panels / "validation/requests.jsonl",
        out / "inputs/linked.jsonl",
        source / "data/external/mintqa-full/kg.sqlite",
        *(source / f"data/external/mintqa-full/questions/{s}.jsonl" for s in ("ti", "pop")),
    ]
    commands = {f: generation_command(f, out, source, args.models, args.prompts) for f in FAMILIES}
    for family in FAMILIES:
        selected = args.selection / family / "baseline-report.json"
        scores = json.loads(selected.read_text())["panels"]["development"]
        eligible = {name: scores[name] for name in ("tog-depth4", "tog-r", "dog", "cor-upstream")}
        if any(not r["qualification"]["usage_audited"] for r in eligible.values()):
            raise ValueError("baseline selection requires qualified complete pilot scores")
        accuracy = max(r["accuracy"] for r in eligible.values())
        tied = [name for name, r in eligible.items() if r["accuracy"] == accuracy]
        timings = json.loads(args.timings.read_text())["families"][family]
        winner = min(tied, key=lambda name: timings[name]["wall_seconds"])
        if winner != "tog-depth4":
            raise ValueError("registered ToG choice differs from the development selection")
        inputs.append(selected)
    result = {
        "artifact_kind": "xgraph.mintqa_baseline_validation",
        "version": 1,
        "code_commit": commit,
        "reference_code_commit": definition["code_commit"],
        "not_before": NOT_BEFORE.isoformat(),
        "freeze_deadline": "2026-11-23",
        "source": str(source),
        "out": str(out),
        "questions": 128,
        "seed": 17,
        "resamples": 10000,
        "selection_rule": SELECTION_RULE,
        "selected_baselines": dict.fromkeys(FAMILIES, "tog-depth4"),
        "strongest_agents": {f: values[3] for f, values in FAMILIES.items()},
        "WT_definition": {
            "first": {
                "menu": 5,
                "observe": 3,
                "width": 8,
                "demos": 5,
                "select": True,
                "required": True,
                "no_loop": True,
                "answer": True,
                "max_tokens": 1024,
                "seed": 0,
            },
            "second": {
                "policy": "two-pass-terminal",
                "terminal_width": 8,
                "max_tokens": 256,
                "seed": 17,
            },
            "gemma_channel_ban": "<|channel>",
            "first_generation": {"max_model_len": 8192, "max_num_seqs": 64, "kv_cache_gib": 8},
            "second_generation": {
                "max_model_len": 16384,
                "max_num_seqs": 64,
                "kv_cache_gib": {"qwen38": 8, "gemma31": 16},
            },
            "memory_share": "0.450",
        },
        "agent_definition": {
            "max_calls": 8,
            "matched_demos": True,
            "matched_paths": True,
            "pages": True,
            "start_rank": "sitelinks",
            "unquoted_starts": True,
            "max_tokens": 256,
            "max_model_len": 16384,
            "max_num_seqs": 64,
            "kv_cache_gib": {"qwen38": 8, "gemma31": 16},
            "seed": 17,
            "memory_share": "0.450",
            "gemma_channel_ban": "<|channel>",
        },
        "references": "owner exports of WT, WT-first, WT-second and strongest agent; "
        "only this queue's references/ directory is read; no reference regeneration",
        "analysis": {
            "comparisons": ["WT minus tog-depth4", "strongest agent minus tog-depth4"],
            "accuracy": "population-weighted native MINTQA normalized answer containment",
            "intervals": "two-sided 95% paired relation-sequence cluster bootstrap within hops; "
            "calibrated parent weights; 10000 resamples; seed 17",
            "descriptive_accuracy_intervals": True,
            "multiplicity": "none; descriptive intervals, no confirmatory claim or hypothesis",
            "failures": "all 128 questions retained; common failure policy",
            "cost": "all calls and failures; WT charges both calls even when second is rejected",
            "scope": "conditional on sampled clusters; internal validation on exposed MINTQA",
        },
        "budget": {
            "generation_gpu_hours": estimate(json.loads(args.timings.read_text())),
            "reference_incremental_gpu_hours": 0,
            "reference_cost_basis": "reuse already booked WT/agent generations; "
            "charge full per-question costs",
            "cap_gpu_hours": 24,
            "limitations": "same hop mixture and serving schedule assumed; no waits, retries, "
            "CPU preparation or new reference generations included; pilot cache/contention",
        },
        "commands": commands,
        "models": {
            f: model_identity(args.models / v[0], "size_mtime") for f, v in FAMILIES.items()
        },
        "inputs": [file_identity(p) for p in inputs],
        "provenance": provenance(inputs),
    }
    path = out / "registration.json"
    write_json(path, result)
    path.with_suffix(".json.sha256").write_text(sha256_file(path) + "\n")
    return result


def load_registration(out):
    path = out / "registration.json"
    if sha256_file(path) != path.with_suffix(".json.sha256").read_text().strip():
        raise ValueError("registration seal changed")
    result = json.loads(path.read_text())
    if git_state() != result["code_commit"]:
        raise ValueError("registered code commit changed")
    for identity in result["inputs"]:
        if file_identity(identity["path"]) != identity:
            raise ValueError(f"registered input changed: {identity['path']}")
    for model in result["models"].values():
        if model_identity(Path(model["path"]), "size_mtime") != model:
            raise ValueError("registered model inventory changed")
    return result


def reference_records(out, family, ids, expected_commit=None):
    """Require complete local exports and retain both WT calls' full costs."""
    directory = out / "references" / family
    paths = {
        name: directory / f"{name}.jsonl"
        for name in ("WT", "WT-first", "WT-second", FAMILIES[family][3])
    }
    if any(
        p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(out.resolve())
        for p in paths.values()
    ):
        raise ValueError("owner must provide regular local reference exports after freeze")
    rows = {name: read_jsonl(p) for name, p in paths.items()}
    for records in rows.values():
        audit_complete_usage(records, ids)
    first, second = indexed(rows["WT-first"]), indexed(rows["WT-second"])
    for row in rows["WT"]:
        if usage(row) != combined_usage(first[row["id"]], second[row["id"]]):
            raise ValueError("WT export must charge both raw calls, including rejected revision")
    inputs = list(paths.values())
    for name in ("WT-first", "WT-second", FAMILIES[family][3]):
        sidecar = Path(f"{paths[name]}.provenance.json")
        if sidecar.is_symlink():
            raise ValueError("reference provenance must be a regular local export")
        saved = json.loads(sidecar.read_text())
        if (
            not saved.get("provenance", {}).get("repo_commit")
            or saved["provenance"].get("repo_dirty") is not False
        ):
            raise ValueError("reference export needs clean committed generation provenance")
        if expected_commit is not None and saved["provenance"]["repo_commit"] != expected_commit:
            raise ValueError("reference generation commit differs from the sealed definition")
        if Path(saved["model"]["path"]).name != FAMILIES[family][0]:
            raise ValueError("reference model family differs")
        if name == "WT-first" and (
            (saved.get("menu"), saved.get("observe_paths"), saved.get("observe_width")) != (5, 3, 8)
            or saved.get("path_demos") != 5
            or not all(
                saved.get(k)
                for k in ("observe_select", "observe_answer", "observe_required", "observe_no_loop")
            )
        ):
            raise ValueError("WT first call is not width-eight menu observation")
        if name == "WT-second" and saved.get("observe_terminal") != 8:
            raise ValueError("WT second call is not terminal width eight")
        command = saved["provenance"].get("command", [])
        for flag, expected in (
            ("--seed", "0" if name == "WT-first" else "17"),
            ("--max-tokens", "1024" if name == "WT-first" else "256"),
        ):
            if flag not in command or command[command.index(flag) + 1 :][:1] != [expected]:
                raise ValueError(f"reference export differs in {flag}")
        if family == "gemma31" and (
            "--ban-tokens" not in command
            or command[command.index("--ban-tokens") + 1 :][:1] != ["<|channel>"]
        ):
            raise ValueError("Gemma reference export must retain its channel-token ban")
        if name == FAMILIES[family][3]:
            agent = saved.get("agent") or {}
            if (
                agent.get("calls") != 8
                or not all(agent.get(k) for k in ("matched_demos", "matched_paths", "pages"))
                or bool(agent.get("feedback")) != (family == "gemma31")
            ):
                raise ValueError("strongest-agent export differs from the registered reader")
        inputs.append(sidecar)
    return rows, inputs


def reference_snapshot(out, inputs, *, create=False):
    path = out / "reference-inputs.json"
    snapshot = [file_identity(p) for p in inputs]
    if not path.exists() and create:
        write_json(path, snapshot)
    if not path.exists() or json.loads(path.read_text()) != snapshot:
        raise ValueError("reference exports changed or have no frozen input receipt")


def score(out, registration):
    from xgraph.mintqa_bench import dataset_rows

    panels = json.loads((out / "panels/panels.json").read_text())
    members = panels["panels"]["validation"]["members"]
    ids = {r["id"] for r in members}
    questions = json.loads((out / "panels/validation/questions.json").read_text())
    data = Path(registration["source"]) / "data/external/mintqa-full/questions"
    gold_paths = [data / f"{s}.jsonl" for s in ("ti", "pop")]
    answers = {
        q: r["answer"]
        for q, r in dataset_rows(
            [f"{s}={p}" for s, p in zip(("ti", "pop"), gold_paths, strict=True)]
        ).items()
    }
    result, inputs, reference_inputs_all = (
        {},
        [out / "registration.json", out / "panels/panels.json", *gold_paths],
        [],
    )
    for family in FAMILIES:
        refs, reference_inputs = reference_records(
            out, family, ids, registration["reference_code_commit"]
        )
        reference_inputs_all += reference_inputs
        path = out / family / "validation/tog-depth4.jsonl"
        baseline = read_jsonl(path)
        audit_complete_usage(baseline, ids)
        inputs += [
            path,
            Path(f"{path}.summary.json"),
            Path(f"{path}.provenance.json"),
            *reference_inputs,
        ]
        result[family] = {}
        for name, rows in (
            ("tog-depth4", baseline),
            ("WT", refs["WT"]),
            (FAMILIES[family][3], refs[FAMILIES[family][3]]),
        ):
            item = score_panel(rows, baseline, members, questions, answers, 10000, 17)
            item["accuracy_interval"] = clustered_difference(
                [float(r["correct"]) for r in item["per_question"]],
                [0.0] * len(members),
                members,
                10000,
                17,
            )
            item["vs_baseline"] = item.pop("vs_agent")
            item["baseline_accuracy"] = item.pop("agent_accuracy")
            item["baseline_cost"] = item.pop("agent_cost")
            result[family][name] = item
    reference_snapshot(out, reference_inputs_all)
    inputs.append(out / "reference-inputs.json")
    write_json(
        out / "report.json",
        {
            "registration": registration["analysis"],
            "families": result,
            "provenance": provenance(inputs),
        },
    )


def queue(out, run=False, log=Path("/home/jonas/git/xgraph/target/queue/baseval128.log")):
    registration = load_registration(out)
    commands = registration["commands"]
    steps = [
        (
            f,
            "tog-depth4",
            Step(
                command,
                out / f / "validation/tog-depth4.jsonl",
                [out / "registration.json", out / "inputs/linked.jsonl"],
                out / "inputs/linked.jsonl",
            ),
        )
        for f, command in commands.items()
    ]
    steps.append(
        (
            "shared",
            "score",
            Step(
                [
                    "nice",
                    "-n",
                    "10",
                    "prlimit",
                    "--as=4294967296",
                    sys.executable,
                    "-I",
                    "-m",
                    "xgraph.mintqa_baseline_validation",
                    "score",
                    "--out",
                    str(out),
                ],
                out / "report.json",
                [out / "registration.json"],
            ),
        )
    )
    if not run:
        for _, _, step in steps:
            print(shlex.join(step.command))
        return
    if datetime.now(UTC).date() < date.fromisoformat(registration["not_before"]):
        raise ValueError("owner-run queue is post-freeze only: 2026-11-24 or later")
    ids = {
        r["id"]
        for r in json.loads((out / "panels/panels.json").read_text())["panels"]["validation"][
            "members"
        ]
    }
    references = []
    for family in FAMILIES:
        _, paths = reference_records(out, family, ids, registration["reference_code_commit"])
        references += paths
    reference_snapshot(out, references, create=True)
    steps[-1][2].inputs += (
        references + [out / "reference-inputs.json"] + [s.output for _, _, s in steps[:-1]]
    )
    with (out / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        log.parent.mkdir(parents=True, exist_ok=True)
        for family, stage, step in steps:
            status = 1
            try:
                load_registration(out)
                reference_snapshot(out, references)
                verdict = run_step(step)
                status = 0
            except subprocess.CalledProcessError as error:
                status = error.returncode
                raise
            finally:
                with log.open("a") as stream:
                    stream.write(
                        f"baseval128 {family} {stage} exit {status} "
                        f"{datetime.now(UTC).isoformat()}\n"
                    )
            print(f"# {verdict} {step.output}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "queue", "score"))
    parser.add_argument("--out", type=Path, default=REPO / "target/queue-baseval128")
    parser.add_argument("--source", type=Path, default=Path("/home/jonas/git/xgraph"))
    parser.add_argument("--models", type=Path, default=Path("/home/jonas/models"))
    parser.add_argument(
        "--definitions",
        type=Path,
        default=Path("/home/jonas/git/xgraph/.claude/worktrees/codex-final"),
    )
    parser.add_argument(
        "--linked",
        type=Path,
        default=Path("/home/jonas/git/xgraph/target/mintqa/requests-iterative-linked.jsonl"),
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path(
            "/home/jonas/git/xgraph/.claude/worktrees/codex-baselines2/data/external/tog-official/ToG/prompt_list.py"
        ),
    )
    parser.add_argument(
        "--timings",
        type=Path,
        default=REPO / "target/queue-basepilot2/pilots/n64-seed17/timings.json",
    )
    parser.add_argument(
        "--selection", type=Path, default=REPO / "target/basepilot2-scoring-fix/rescored"
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "register":
        register(args)
    elif args.command == "queue":
        queue(args.out.resolve(), args.run)
    else:
        registration = load_registration(args.out)
        if datetime.now(UTC).date() < NOT_BEFORE:
            raise ValueError("baseline validation scoring is post-freeze only")
        score(args.out, registration)


if __name__ == "__main__":
    main()
