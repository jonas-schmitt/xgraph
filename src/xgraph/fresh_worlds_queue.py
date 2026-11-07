"""CPU planning and the explicitly owner-run fresh-world pilot queue."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shlex
import signal
import subprocess
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from xgraph.fresh_worlds import read_jsonl, write_json, write_jsonl
from xgraph.policy_catalog import recipe
from xgraph.queue_accounting import gpu_occupancy, wait_stage

FAMILIES = {"qwen38": "Qwen3.8-27B-FP8", "gemma31": "gemma-4-31B-it-FP8-dynamic"}
CORE_STAGES = {
    "WT",
    "WT-first",
    "WT-build-second",
    "WT-second",
    "WT-merge",
    "agent",
    "plan",
    "plan-exact",
    "reader",
    "analysis",
}


def plan_queue(repo, study_dir, out, models, *, policy_arms=(), design=None):
    repo, study_dir, out, models = (Path(p).resolve() for p in (repo, study_dir, out, models))
    python = str(repo / ".venv/bin/python")
    stages = []
    v2 = design == "controlled-v2"
    extra = {}
    for declaration in policy_arms:
        name, sep, policy = declaration.partition("=")
        if not sep or not name.replace("-", "").isalnum() or name in CORE_STAGES or name in extra:
            raise ValueError("policy-arm must be a unique NAME=CATALOG_POLICY")
        recipe(policy)
        extra[name] = policy
    for family, model_dir in FAMILIES.items():
        directory = out / family
        model = models / model_dir
        index = study_dir / "indexes" / family

        def cpu(name, output, operation, *arguments, family=family):
            stages.append(
                {
                    "family": family,
                    "name": name,
                    "gpu": False,
                    "output": str(output),
                    "argv": [
                        "nice",
                        "-n",
                        "10",
                        "prlimit",
                        "--as=4294967296",
                        python,
                        "-I",
                        "-m",
                        "xgraph.fresh_worlds_queue",
                        "stage",
                        "--study",
                        str(study_dir),
                        "--family",
                        family,
                        "--operation",
                        operation,
                        "--out",
                        str(output),
                        *map(str, arguments),
                    ],
                }
            )

        def generate(
            name,
            requests,
            output,
            spec=None,
            second=False,
            arm="constrained",
            family=family,
            model=model,
            index=index,
        ):
            constrained = arm == "constrained"
            menu = bool(spec and spec["menu"] and not second)
            controls = {
                "XG_NEED_GIB": 75 if family == "gemma31" else 60,
                "XG_AVAILABLE_WAIT": 43200,
                "XG_GPU_IDLE": 1,
                "XG_START_ATTEMPTS": 1,
                "XG_KG_INDEX": index,
                "XG_TOKENIZER": model / "tokenizer.json",
                "XG_TOOLS_DB": study_dir / "kg.sqlite",
                "XG_KG_VERSIONS": study_dir / "kg-versions",
                "XG_RELATION_LABELS": study_dir / "kg/relation_labels.jsonl",
                "XG_PATHS": int(constrained),
                "XG_COMMITTED_PATHS": int(constrained),
                "XG_ESCAPES": 0,
                "XG_UNQUOTED_STARTS": 1,
                "XG_START_RANK": "sitelinks",
                "XG_OPEN": int(constrained),
                "XG_END_AFTER_ANSWER": int(constrained),
                "XG_RELATIONS": "frontier",
                "XG_EXPRESSIONS": 0,
                "XG_STOP": "",
                "XG_LORA": "",
                "XG_DERIVATION": 0,
                "XG_RELEASE_MASS": "",
                "XG_RELEASE_TOP": 0,
                "XG_ESCAPE_BIAS": "",
                "XG_REPEATS": 1,
                "XG_BOUND_ESCAPES": 1,
                "XG_MENU": spec["menu"] if menu else 0,
                "XG_OBSERVE": 3 if menu else 0,
                "XG_OBSERVE_WIDTH": spec["observe_width"] if menu else 0,
                "XG_OBSERVE_DEFER": 0,
                "XG_MENU_CLOSE_NEXT": 0,
                "XG_PATH_DEMOS": 5 if constrained and not second else "",
                "XG_OBSERVE_TERMINAL": spec["terminal_width"] if second else 0,
                "XG_AUDIT": 0,
                **{
                    f"XG_OBSERVE_{k}": int(menu)
                    for k in ("SELECT", "ANSWER", "REQUIRED", "NO_LOOP")
                },
                "XG_PAGECACHE_ROOTS": str(model),
                "OPENBLAS_NUM_THREADS": 1,
                "OMP_NUM_THREADS": 1,
                "CMAKE_BUILD_PARALLEL_LEVEL": 1,
                "CMAKE_ARGS": "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
            }
            tokens = (spec["second_max_tokens"] if second else spec["max_tokens"]) if spec else 256
            seed = (spec["second_seed"] if second else spec["seed"]) if spec else 17
            flags = [
                "--max-tokens",
                str(tokens),
                "--max-model-len",
                "16384",
                "--max-num-seqs",
                "64",
                "--kv-cache-gib",
                "16" if family == "gemma31" else "8",
                "--seed",
                str(seed),
            ]
            if family == "gemma31":
                flags += ["--ban-tokens", "<|channel>"]
            if arm == "agent":
                flags += [
                    "--max-calls",
                    str(spec["agent_budget"]),
                    "--matched-demos",
                    "--matched-paths",
                    "--pages",
                    "--start-rank",
                    "sitelinks",
                    "--unquoted-starts",
                ]
                if spec["agent_feedback"]:
                    flags += ["--agent-feedback"]
            stages.append(
                {
                    "family": family,
                    "name": name,
                    "gpu": True,
                    "requests": str(requests),
                    "model": str(model),
                    "need_gib": controls["XG_NEED_GIB"],
                    "max_tokens": tokens,
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
                        arm,
                        str(output),
                        *flags,
                    ],
                }
            )

        for name, policy in {"WT": "WT", **extra}.items():
            spec = recipe(policy, family)
            if spec["agent_budget"]:
                generate(
                    name,
                    study_dir / "requests-agent.jsonl",
                    directory / f"{name}.jsonl",
                    spec,
                    arm="agent",
                )
                continue
            first = (
                directory / f"{name}-first.jsonl" if spec["second"] else directory / f"{name}.jsonl"
            )
            generate(
                name + "-first" if spec["second"] else name,
                study_dir / ("requests-wt.jsonl" if spec["menu"] else "requests-paths.jsonl"),
                first,
                spec,
            )
            if spec["second"]:
                second_requests = directory / f"{name}-second-requests.jsonl"
                second = directory / f"{name}-second.jsonl"
                cpu(
                    name + "-build-second",
                    second_requests,
                    "build-second",
                    "--first",
                    first,
                    "--policy",
                    policy,
                )
                generate(name + "-second", second_requests, second, spec, second=True)
                cpu(
                    name + "-merge",
                    directory / f"{name}.jsonl",
                    "merge",
                    "--first",
                    first,
                    "--second",
                    second,
                )
        spec = recipe("agent-feedback-8" if family == "gemma31" else "agent-strong-8", family)
        generate(
            "agent",
            study_dir / "requests-agent.jsonl",
            directory / "agent.jsonl",
            spec,
            arm="agent",
        )
        generate(
            "plan", study_dir / "requests-plan.jsonl", directory / "plan.jsonl", arm="baseline"
        )
        cpu(
            "plan-exact",
            directory / "plan-exact.jsonl",
            "execute",
            "--first",
            directory / "plan.jsonl",
        )
        if not v2:
            generate(
                "reader",
                study_dir / "requests-reader.jsonl",
                directory / "reader.jsonl",
                arm="baseline",
            )
    # Core arms in both families keep priority over opt-in policies.
    optional = [
        s
        for s in stages
        if any(s["name"] == name or s["name"].startswith(name + "-") for name in extra)
    ]
    stages = [s for s in stages if s not in optional] + optional
    if len({s["output"] for s in stages}) != len(stages):
        raise ValueError("policy-arm output collision")
    stages.append(
        {
            "family": "both",
            "name": "analysis",
            "gpu": False,
            "output": str(out / "analysis.json"),
            "input_files": [
                str(out / f / f"{a}.jsonl")
                for f in FAMILIES
                for a in (
                    ("WT", "agent", "plan-exact") if v2 else ("WT", "agent", "plan-exact", "reader")
                )
            ],
            "argv": [
                "nice",
                "-n",
                "10",
                "prlimit",
                "--as=4294967296",
                python,
                "-I",
                "-m",
                "xgraph.fresh_worlds",
                "analyze",
                "--study",
                str(study_dir),
                "--outputs-dir",
                str(out),
                "--budget",
                str(out / "budget.json"),
                "--out",
                str(out / "analysis.json"),
            ],
        }
    )
    return stages


def _context(study_dir, family):
    from xgraph.index import load_kg_index
    from xgraph.kgversions import load_kg_versions, version_names
    from xgraph.programs import MatchedExecutor
    from xgraph.provenance import sha256_file
    from xgraph.tokenizer import ServedTokenizer
    from xgraph.toolkg import KGStore, version_stores

    tokenizers = json.loads((study_dir / "tokenizers.json").read_text())
    tokenizer_path = Path(tokenizers[family]["path"])
    if sha256_file(tokenizer_path) != tokenizers[family]["sha256"]:
        raise ValueError("served tokenizer input changed")
    tokenizer = ServedTokenizer.from_file(tokenizer_path)
    index = load_kg_index(study_dir / "indexes" / family)
    executor = MatchedExecutor(
        index,
        tokenizer,
        versions=load_kg_versions(study_dir / "kg-versions", index),
        start_rank="sitelinks",
        unquoted_starts=True,
    )
    store = KGStore(study_dir / "kg.sqlite")
    return (
        tokenizer,
        executor,
        store,
        version_stores(store, study_dir / "kg-versions", version_names(study_dir / "kg-versions")),
    )


def cpu_stage(args):
    from xgraph.path_policy import second_call_requests, select_second_call
    from xgraph.programs import record_path
    from xgraph.score import record_failure

    requests = read_jsonl(args.study / "requests-paths.jsonl")
    first = read_jsonl(args.first)
    expected = {r["id"] for r in requests}
    if len(first) != len(expected) or {r["id"] for r in first} != expected:
        raise ValueError("policy stage requires complete unique paired outputs")
    tokenizer, executor, store, stores = _context(args.study, args.family)
    if args.operation == "build-second":
        spec = recipe(args.policy, args.family)
        rows = second_call_requests(
            spec["second"],
            requests,
            first,
            executor,
            store,
            tokenizer,
            relation_rows=read_jsonl(args.study / "kg/relation_labels.jsonl"),
            stores=stores,
            terminal_width=spec["terminal_width"],
        )
    elif args.operation == "merge":
        rows = select_second_call(
            first,
            read_jsonl(args.second),
            requests,
            executor,
            first_source=str(args.first),
            second_source=str(args.second),
        )
    elif args.operation == "execute":
        rows = []
        sources = {r["id"]: r for r in first}
        for request in requests:
            source = sources[request["id"]]
            path = None if record_failure(source, require_text=False) else record_path(source)
            reached = (
                executor.execute(*path, request["anchor"], request["kg_version"]) if path else None
            )
            answer = reached.answer if reached else None
            rows.append(
                {
                    **source,
                    "answer_id": answer,
                    "delivery": "one-call relation plan plus matched exact execution",
                    "planner_source": str(args.first),
                }
            )
    else:
        raise ValueError(f"unknown CPU operation {args.operation}")
    write_jsonl(args.out, rows)


def _hash_files(paths):
    from xgraph.provenance import sha256_file

    return {str(Path(p).resolve()): sha256_file(Path(p)) for p in paths}


def _commit(repo, *, clean=False):
    if (
        clean
        and subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
    ):
        raise ValueError("fresh-world GPU queue refuses a dirty tree")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _model_inventory(model):
    model = Path(model)
    configs = [p for p in model.iterdir() if p.name.endswith((".json", ".jinja"))]
    weights = [
        {"path": str(p.resolve()), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in sorted(model.glob("*.safetensors"))
    ]
    if not weights:
        raise ValueError(f"no local model weights in {model}")
    return {
        "config_sha256": _hash_files(configs),
        "weights": weights,
        "weight_identity": "file size and nanosecond mtime; not a content SHA-256",
    }


def write_plan(repo, study_dir, out, models, *, policy_arms=(), factorial_registration=None):
    from xgraph.fresh_worlds import generator_sources
    from xgraph.provenance import sha256_file

    repo, study_dir, out, models = (Path(p).resolve() for p in (repo, study_dir, out, models))
    if out.exists():
        raise ValueError(f"queue already exists: {out}")
    study = json.loads((study_dir / "study.json").read_text())
    v2 = study["config"].get("design") == "controlled-v2"
    if study["config"].get("design") == "mechanism-v2-hard" and factorial_registration is None:
        raise ValueError("the hard fallback requires its own factorial registration and owner gate")
    confirmation = False
    if factorial_registration is not None:
        from xgraph.mechanism import fresh_requests, validate_fresh_cohort, validate_fresh_validity

        factorial_registration = Path(factorial_registration).resolve()
        marker = study.get("mechanism_fallback", {})
        confirmation = study["config"]["split"] == "confirmation"
        spec = validate_fresh_cohort(study, factorial_registration, seal=marker.get("seal"))
        validate_fresh_validity(study_dir, study)
        if (
            policy_arms
            or marker.get("registration_sha256") != sha256_file(factorial_registration)
            or read_jsonl(study_dir / "requests-factorial.jsonl") != fresh_requests(study)
        ):
            raise ValueError("factorial queue differs from the registered study/common requests")
    elif (
        study["config"]["worlds"] != (96 if v2 else 64) or study["config"]["split"] != "development"
    ):
        raise ValueError("this owner queue is the registered development pilot only")
    if v2 and policy_arms:
        raise ValueError("v2 has exactly three registered arms")
    if v2:
        from xgraph.fresh_worlds_v2 import validate_registration

        validate_registration(study_dir)
    sources = {
        name: sha256_file(Path(__file__).with_name(name)) for name in generator_sources(study)
    }
    if study.get("source_sha256") != sources:
        raise ValueError("generator source changed; rebuild development artifacts before planning")
    if factorial_registration is not None:
        from xgraph.mechanism import fresh_stages

        stages = fresh_stages(
            repo, study_dir, out, models, factorial_registration, confirmation=confirmation
        )
    else:
        stages = plan_queue(
            repo,
            study_dir,
            out,
            models,
            policy_arms=policy_arms,
            design=study["config"].get("design"),
        )
    launcher = (repo / "scripts/generate.sh").read_text()
    legacy_root = 'repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"'
    source_root = 'repo_root="${XG_SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"'
    roots = [root for root in (legacy_root, source_root) if launcher.count(root)]
    if len(roots) != 1 or launcher.count(roots[0]) != 1:
        raise ValueError("generate.sh root assignment changed; requalify the owner wrapper")
    assignment = "repo_root=" + shlex.quote(str(repo))
    if roots[0] == source_root:
        assignment += '\nrepo_root="${XG_SOURCE_ROOT:-${repo_root}}"'
    launcher = launcher.replace(roots[0], assignment)
    # Only the owned container is stopped on an external budget timeout. Never
    # kill another queue's containers. The original EXIT cleanup stays intact.
    launcher += "\n"
    cleanup_trap = "trap 'rm -rf \"${run}\"' EXIT"
    if launcher.count(cleanup_trap) != 1:
        raise ValueError("generate.sh cleanup changed")
    launcher = launcher.replace(
        cleanup_trap,
        cleanup_trap + "\nfresh_stop() {\n"
        '  if [[ -n "${container:-}" ]]; then\n'
        '    docker kill "${container}" >/dev/null 2>&1 || true\n  fi\n'
        "  exit 143\n}\ntrap fresh_stop TERM INT\n",
    )
    inputs = [p for p in study_dir.rglob("*") if p.is_file()]
    if factorial_registration is not None:
        inputs += [
            factorial_registration,
            Path(str(factorial_registration) + ".sha256"),
            Path(spec["document"]),
            Path(spec["parent_registration"]["path"]),
            Path(__file__).resolve().parents[2] / "scripts/fresh-worlds-v2-audit.py",
        ]
        if confirmation:
            inputs.append(Path(marker["seal"]))
    model_identity = {family: _model_inventory(models / name) for family, name in FAMILIES.items()}
    plan = {
        "format": 1,
        "repo": str(repo),
        "commit": _commit(repo, clean=True) if v2 else _commit(repo),
        "study": str(study_dir),
        "models": model_identity,
        "inputs": _hash_files(inputs),
        "stages": stages,
        "policy_arms": list(policy_arms),
        "gpu_budget_seconds": 7200,
        "stage_wall_timeout_seconds": 24 * 3600,
        "log": "/home/jonas/git/xgraph/target/queue/fresh-worlds-v2.log"
        if v2
        else "/home/jonas/git/xgraph/target/queue/fresh-worlds.log",
        "budget_basis": {
            "worlds": study["config"]["worlds"],
            "conditions": 3,
            "families": 2,
            "gpu_stages": sum(s["gpu"] for s in stages),
            "startup_seconds_per_stage": 180,
            "aggregate_policy_seconds_per_world_assumption": 60,
            "unmeasured": True,
            "forecast_seconds": 64 * 60 + sum(s["gpu"] for s in stages) * 180,
        },
    }
    if factorial_registration is not None:
        plan.update(
            study_type="mechanism-fallback",
            phase=study["config"]["split"],
            registration=str(factorial_registration),
            registration_sha256=sha256_file(factorial_registration),
            gpu_budget_seconds=36000 if confirmation else 7200,
            log="/home/jonas/git/xgraph/target/queue/mechanism-fallback.log",
        )
        if confirmation:
            frozen = json.loads(Path(marker["seal"]).read_text())
            pilot_path = frozen["provenance"]["inputs"][0]["path"]
            pilot = json.loads(Path(pilot_path).read_text())
            plan["budget_basis"] = {
                "worlds": 256,
                "conditions": 3,
                "families": 2,
                "gpu_stages": 8,
                "pilot_gpu_seconds": pilot["pilot_gpu_seconds"],
                "cost_factor": 1.5,
                "forecast_seconds": pilot["projected_gpu_seconds"],
                "basis": "six times the hash-bound complete measured factorial pilot charge",
            }
    if v2:
        plan["design"] = "controlled-v2"
        plan["execution_approval"] = "NOT APPROVED; dry-run preparation only"
        plan["budget_basis"] = {
            "worlds": 96,
            "conditions": 3,
            "families": 2,
            "gpu_stages": 8,
            "unmeasured": True,
            "basis": "v1 non-reader occupancy scaled by 96/64; difficulty unpriced",
        }
    out.mkdir(parents=True)
    (out / "generate-owned.sh").write_text(launcher)
    plan["owner_launcher"] = _hash_files([out / "generate-owned.sh"])
    write_json(out / "plan.json", plan)
    runner = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + "exec "
        + shlex.join(
            [
                "env",
                "USE_TORCH=0",
                "OPENBLAS_NUM_THREADS=1",
                "OMP_NUM_THREADS=1",
                "RAYON_NUM_THREADS=1",
                "TOKENIZERS_PARALLELISM=false",
                "MALLOC_ARENA_MAX=2",
                "nice",
                "-n",
                "10",
                "prlimit",
                "--as=4000000000" if factorial_registration is not None else "--as=4294967296",
                str(repo / ".venv/bin/python"),
                "-I",
                "-m",
                "xgraph.fresh_worlds_queue",
                "run",
                "--plan",
                str(out / "plan.json"),
            ]
        )
        + ' "$@"\n'
    )
    # No implicit execution: the module also requires --run explicitly.
    owner_name = (
        "mechanism-fallback.sh" if factorial_registration is not None else "fresh-worlds.sh"
    )
    (out / owner_name).write_text(runner)
    if factorial_registration is not None:
        write_json(
            out / "authorization.example.json",
            {
                "approved": False,
                "phase": plan["phase"],
                "plan_sha256": sha256_file(out / "plan.json"),
                "source_commit": plan["commit"],
                "registration_sha256": plan["registration_sha256"],
                "stride_gate": "OWNER_SUPPLIES_COMPLETED_NOV13_RECEIPT",
                "stride_gate_sha256": None,
            },
        )
    if v2:
        (out / "fresh-worlds-v2.sh").write_text(runner)
    preview = "\n".join(shlex.join(s["argv"]) for s in stages) + "\n"
    (out / "dry-run.txt").write_text(preview)
    return plan


def verify_plan(plan, out):
    from xgraph.provenance import sha256_file

    repo = Path(plan["repo"])
    if _commit(repo, clean=True) != plan["commit"]:
        raise ValueError("queue code commit changed; prepare a new plan")
    for path, expected in {**plan["inputs"], **plan["owner_launcher"]}.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"queue input changed: {path}")
    for family, identity in plan["models"].items():
        model = next(s["model"] for s in plan["stages"] if s["gpu"] and s["family"] == family)
        if _model_inventory(model) != identity:
            raise ValueError(f"served model changed: {family}")
    if json.loads((out / "plan.json").read_text()) != plan:
        raise ValueError("queue plan changed")


def _factorial_tokenizer(model):
    """Load the declared built-in tokenizer without AutoTokenizer's model dispatch.

    AutoTokenizer imports model configuration (and Torch) in the locked release.
    These are the same tokenizer classes it selects from the registered local
    checkpoint metadata, using identical from_pretrained/template methods.
    Unknown classes require a new qualification; no remote code is imported.
    """
    import importlib

    classes = {
        "Qwen2Tokenizer": ("transformers.models.qwen2.tokenization_qwen2", "Qwen2Tokenizer"),
        "GemmaTokenizer": ("transformers.models.gemma.tokenization_gemma", "GemmaTokenizer"),
        "PreTrainedTokenizerFast": ("transformers", "PreTrainedTokenizerFast"),
        "TokenizersBackend": ("transformers", "TokenizersBackend"),
    }
    name = json.loads((Path(model) / "tokenizer_config.json").read_text())["tokenizer_class"]
    if name not in classes:
        raise ValueError(f"unqualified factorial tokenizer class: {name}")
    module, attribute = classes[name]
    cls = getattr(importlib.import_module(module), attribute)
    return cls.from_pretrained(model, local_files_only=True, trust_remote_code=False)


def _admit(stage):
    # Load JSON/tokenizer/template data locally; no model or remote Python code.
    from xgraph.generate import opened_requests
    from xgraph.prompts import menu_requests, observed_requests
    from xgraph.syntax import SpanSyntax

    rows = read_jsonl(stage["requests"])
    controls = dict(
        item.split("=", 1) for item in stage["argv"][1:] if item.startswith("XG_") and "=" in item
    )
    syntax = SpanSyntax()
    if stage.get("study_type") == "mechanism-fallback":
        common = rows
        variants = [
            opened_requests(observed_requests(common, 3, syntax, select=select), syntax.opener)
            for select in (False, True)
        ]
        if variants[0] != variants[1]:
            raise ValueError("factorial effective instructions/demonstrations differ across cells")
        rows = observed_requests(rows, 3, syntax, select=controls["XG_OBSERVE_SELECT"] == "1")
    elif controls["XG_OBSERVE"] != "0":
        rows = observed_requests(
            rows, 3, syntax, select=True, width=int(controls["XG_OBSERVE_WIDTH"]), demos=5
        )
        rows = menu_requests(
            rows,
            5,
            syntax,
            observe=3,
            select=True,
            demos=5,
            width=int(controls["XG_OBSERVE_WIDTH"]),
        )
    if controls["XG_OPEN"] == "1":
        rows = opened_requests(rows, syntax.opener)
    if stage.get("study_type") == "mechanism-fallback":
        tokenizer = _factorial_tokenizer(stage["model"])
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            stage["model"], local_files_only=True, trust_remote_code=False
        )
    max_prompt = 0
    for row in rows:
        prefill = row["messages"][-1]["role"] == "assistant"
        ids = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=not prefill,
            continue_final_message=prefill,
            enable_thinking=False,
        )
        max_prompt = max(max_prompt, len(ids))
    if max_prompt + stage["max_tokens"] > 16384:
        raise ValueError(
            "complete request exceeds declared context; no evidence truncation is allowed"
        )
    return {
        "requests": len(rows),
        "max_prompt_tokens": max_prompt,
        "max_completion_tokens": stage["max_tokens"],
        "context": 16384,
    }


def _generation_artifacts(stage, admission):
    from xgraph.cost import request_cost

    output = Path(stage["output"])
    expected = [r["id"] for r in read_jsonl(stage["requests"])]
    rows = read_jsonl(output)
    if (
        len(rows) != admission["requests"]
        or len(set(expected)) != len(expected)
        or any(not isinstance(r, dict) or not isinstance(r.get("id"), str) for r in rows)
        or {r["id"] for r in rows} != set(expected)
    ):
        raise ValueError("generation output must cover the complete unique request IDs")
    costs = []
    for row in rows:
        if (
            not isinstance(row.get("text"), str)
            or "finish_reason" not in row
            or not isinstance(row["finish_reason"], str | type(None))
        ):
            raise ValueError(f"invalid generation record: {row['id']}")
        cost = request_cost(row)
        if any(type(v) is not int or v < 0 for v in cost.values()):
            raise ValueError(f"invalid generation usage: {row['id']}")
        costs.append(cost)
    artifacts = [Path(str(output) + s) for s in (".summary.json", ".provenance.json")]
    if any(not p.is_file() for p in artifacts):
        raise ValueError("stage lacks native usage or provenance")
    summary, origin = (json.loads(p.read_text()) for p in artifacts)
    if (
        not isinstance(summary, dict)
        or summary.get("artifact_kind") != "xgraph.generation"
        or summary.get("version") != 1
        or summary.get("requests") != len(rows)
        or any(
            summary.get(k) != sum(c[k] for c in costs)
            for k in ("prompt_tokens", "generated_tokens")
        )
        or not isinstance(summary.get("seconds"), int | float)
        or not math.isfinite(summary["seconds"])
        or summary["seconds"] < 0
    ):
        raise ValueError("invalid native generation summary or usage totals")
    if (
        not isinstance(origin, dict)
        or origin.get("artifact_kind") != "xgraph.generation_provenance"
        or origin.get("version") != 1
        or not isinstance(origin.get("provenance"), dict)
    ):
        raise ValueError("invalid native generation provenance")
    return _hash_files(artifacts)


def _authorize_factorial(plan, plan_path, authorization):
    from xgraph.provenance import sha256_file

    if authorization is None:
        raise ValueError(
            "factorial requires separate owner authorization after the failed Nov 13 gate"
        )
    now = datetime.now(UTC)
    if plan["phase"] == "development":
        if not datetime(2026, 11, 14, tzinfo=UTC) <= now < datetime(2026, 11, 18, tzinfo=UTC):
            raise ValueError(
                "fallback pilot is conditional on the failed Nov 13 gate; run Nov 14-17"
            )
    elif now < datetime(2026, 11, 24, tzinfo=UTC):
        raise ValueError("fallback confirmation must wait until after the Nov 23 freeze")
    approval = json.loads(Path(authorization).read_text())
    if (
        approval.get("approved") is not True
        or approval.get("phase") != plan["phase"]
        or approval.get("source_commit") != plan["commit"]
        or approval.get("plan_sha256") != sha256_file(Path(plan_path))
        or approval.get("registration_sha256") != plan["registration_sha256"]
    ):
        raise ValueError("factorial owner approval is absent or does not bind this exact plan")
    gate_path = Path(approval["stride_gate"])
    if sha256_file(gate_path) != approval.get("stride_gate_sha256"):
        raise ValueError("Nov 13 stride gate receipt hash changed")
    gate = json.loads(gate_path.read_text())
    if (
        gate.get("date") != "2026-11-13"
        or gate.get("status") != "completed"
        or gate.get("passed") is not False
    ):
        raise ValueError("fallback requires a completed failed Nov 13 variable-stride gate")
    return sha256_file(Path(authorization))


def run_queue(plan_path, *, execute=False, authorization=None):
    from xgraph.provenance import sha256_file

    out = Path(plan_path).resolve().parent
    plan = json.loads(Path(plan_path).read_text())
    if not execute:
        for stage in plan["stages"]:
            print(shlex.join(stage["argv"]))
        return
    approval_hash = None
    if plan.get("study_type") == "mechanism-fallback":
        verify_plan(plan, out)
        approval_hash = _authorize_factorial(plan, plan_path, authorization)
        (Path(plan["repo"]) / "target/tmp").mkdir(parents=True, exist_ok=True)
    lock = (out / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_plan(plan, out)
    log = Path(plan["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    seal = sha256_file(Path(plan_path))
    ledger_path = out / "budget.json"
    ledger = (
        json.loads(ledger_path.read_text())
        if ledger_path.exists()
        else {"used_seconds": 0, "attempts": []}
    )
    if ledger.get("seal", seal) != seal:
        raise ValueError("GPU budget ledger belongs to another queue plan")
    ledger["seal"] = seal
    if approval_hash is not None:
        if ledger.get("authorization_sha256", approval_hash) != approval_hash:
            raise ValueError("GPU ledger belongs to another owner authorization")
        ledger["authorization_sha256"] = approval_hash
    if any(a.get("status") == "running" for a in ledger["attempts"]):
        raise ValueError("unclosed attempt: reconcile its physical GPU cost before resuming")
    for stage in plan["stages"]:
        verify_plan(plan, out)
        if (
            approval_hash is not None
            and _authorize_factorial(plan, plan_path, authorization) != approval_hash
        ):
            raise ValueError("owner authorization changed during the queue")
        output = Path(stage["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        receipt_path = Path(str(output) + ".receipt.json")
        inputs = (
            _hash_files([stage["requests"]])
            if stage["gpu"]
            else _hash_files(
                [
                    *stage.get("input_files", []),
                    *[
                        Path(stage["argv"][stage["argv"].index(flag) + 1])
                        for flag in ("--first", "--second", "--budget")
                        if flag in stage["argv"]
                    ],
                ]
            )
        )

        def status(value, stage=stage):
            with log.open("a") as stream:
                name = (
                    "fresh-worlds-v2" if plan.get("design") == "controlled-v2" else "fresh-worlds"
                )
                stream.write(
                    f"{plan.get('study_type', name)} "
                    f"{stage['family']} {stage['name']} {value} "
                    f"{datetime.now(UTC).isoformat()}\n"
                )

        if output.exists() or receipt_path.exists():
            if not output.exists() or not receipt_path.exists():
                raise ValueError(f"partial stage must be inspected, not reused: {output}")
            receipt = json.loads(receipt_path.read_text())
            if (
                receipt["seal"] != seal
                or receipt["output_sha256"] != sha256_file(output)
                or receipt["inputs"] != inputs
                or (
                    stage["gpu"]
                    and (
                        receipt.get("gpu_events") != str(output) + ".gpu-events.jsonl"
                        or receipt.get("gpu_events") not in receipt.get("artifacts", {})
                    )
                )
                or any(sha256_file(Path(p)) != h for p, h in receipt.get("artifacts", {}).items())
            ):
                raise ValueError(f"completed stage identity changed: {output}")
            status("skip-existing")
            continue
        admission = None
        remaining = plan["gpu_budget_seconds"] - ledger["used_seconds"]
        if stage["gpu"]:
            if remaining <= 30:
                raise ValueError("fresh-world pilot GPU budget exhausted")
            admission = _admit(stage)
        attempt = {
            "stage": stage["name"],
            "family": stage["family"],
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "gpu": stage["gpu"],
        }
        # This is the physical GPU ledger. CPU post-processing must not mutate
        # the measured input while analysis reads and hashes it.
        events = Path(str(output) + ".gpu-events.jsonl") if stage["gpu"] else None
        offset = 0
        if stage["gpu"]:
            with events.open("ab") as stream:
                stream.flush()
                os.fsync(stream.fileno())
            offset = events.stat().st_size
            # Preserve the old allowance and cleanup margin within the same cap.
            attempt.update(
                reserved_seconds=remaining,
                gpu_allowance_seconds=int(remaining - 20),
                gpu_events=str(events),
                events_offset=offset,
            )
            ledger["attempts"].append(attempt)
            write_json(ledger_path, ledger)
        status("start")
        started = time.monotonic()
        command = list(stage["argv"])
        if not stage["gpu"]:
            import resource

            inherited = [
                v for v in resource.getrlimit(resource.RLIMIT_AS) if v != resource.RLIM_INFINITY
            ]
            # Preserve the sealed argv; receipts also record the executed argv.
            # A nested CPU stage must never raise its caller's address-space cap.
            for i, value in enumerate(command):
                if value.startswith("--as=") and inherited:
                    command[i] = f"--as={min(int(value[5:]), *inherited)}"
        base_env = {k: v for k, v in os.environ.items() if not k.startswith(("XG_", "XGRAPH_"))}
        container = f"xgraph-fresh-{seal[:12]}-{len(ledger['attempts'])}"
        if events:
            base_env.update(XG_GPU_EVENTS=str(events), XG_CONTAINER_NAME=container)
        timed_stage = {
            **stage,
            "gpu_seconds": int(remaining - 20) if stage["gpu"] else 0,
            "wall_timeout_seconds": stage.get(
                "wall_timeout_seconds", plan.get("stage_wall_timeout_seconds", 24 * 3600)
            ),
        }
        process, code, settled, failure = None, 1, True, None
        try:
            if stage["gpu"]:
                # Both this wait and the launcher's later locks/build/headroom
                # consume the wall bound, never the container allowance.
                status("waiting-headroom")
                subprocess.run(
                    [
                        "bash",
                        str(Path(plan["repo"]) / "scripts/gpu-headroom.sh"),
                        str(stage["need_gib"]),
                    ],
                    env={**base_env, "XG_GPU_IDLE": "1", "XG_AVAILABLE_WAIT": "43200"},
                    check=True,
                    timeout=timed_stage["wall_timeout_seconds"],
                )
            if approval_hash is not None:
                verify_plan(plan, out)
                if _authorize_factorial(plan, plan_path, authorization) != approval_hash:
                    raise ValueError("owner authorization changed while waiting for admission")
            with Path(str(output) + ".log").open(
                "a" if approval_hash is not None else "w"
            ) as stream:
                process = subprocess.Popen(
                    command,
                    cwd=plan["repo"],
                    env=base_env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                code = wait_stage(process, timed_stage, started, started, events, offset)
            if events:
                # A nonzero launcher exit still needs a closed boundary; otherwise
                # stop its descendants and confirm cleanup before charging it.
                _, launched = gpu_occupancy(events, offset, started, require_end=True)
                if code == 0 and not launched:
                    raise ValueError("stage succeeded without a GPU launch event")
        except BaseException as error:
            failure = error
            code = 124 if isinstance(error, subprocess.TimeoutExpired) else 1
            # Stop the entire launcher group before scoped container cleanup:
            # no descendant may create a container after cleanup has returned.
            if process:
                settled = False
                try:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                    if events:
                        cleanup = subprocess.run(
                            ["docker", "rm", "-f", container],
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )
                        settled = cleanup.returncode == 0 or "No such" in cleanup.stderr
                    else:
                        settled = True
                except (OSError, subprocess.SubprocessError):
                    pass  # Unknown occupancy retains the full reservation below.
        wall_seconds = time.monotonic() - started
        seconds, launched, accounting = wall_seconds, False, "measured"
        if events:
            try:
                if not settled:
                    raise ValueError("owned container cleanup unconfirmed")
                seconds, launched = gpu_occupancy(events, offset, started)
            except (ValueError, KeyError, TypeError, OSError):
                seconds, launched, accounting = remaining, None, "reserved"
        attempt.update(
            status="failed",
            seconds=seconds,
            wall_seconds=wall_seconds,
            exit=code,
            **(
                {
                    "accounting": accounting,
                    "timing_kind": "container occupancy",
                    "container_launched": launched,
                }
                if events
                else {}
            ),
        )
        if stage["gpu"]:
            ledger["used_seconds"] += seconds
            write_json(ledger_path, ledger)
        status(f"exit {code} seconds {seconds:.3f} wall_seconds {wall_seconds:.3f}")
        try:
            if failure is not None:
                raise ValueError(str(failure)) from failure
            if code or not output.is_file():
                raise ValueError(f"launcher exit {code} or missing output")
            artifacts = {}
            if stage["gpu"]:
                artifacts = _generation_artifacts(stage, admission)
                artifacts.update(_hash_files([events]))
            elif stage["name"] == "analysis":
                artifacts = _hash_files([output.with_name("assurance.json")])
            receipt = {
                "seal": seal,
                "argv": stage["argv"],
                "effective_argv": command,
                "seconds": seconds,
                "wall_seconds": wall_seconds,
                **({"gpu_events": str(events)} if events else {}),
                "gpu": stage["gpu"],
                "inputs": inputs,
                "output_sha256": sha256_file(output),
                "admission": admission,
                "artifacts": artifacts,
            }
            write_json(receipt_path, receipt)
        except (OSError, ValueError, KeyError, TypeError) as error:
            attempt["error"] = str(error)
            if stage["gpu"]:
                write_json(ledger_path, ledger)
            status(f"failed {error}")
            raise ValueError(f"stage failed; outputs retained: {stage['name']}: {error}") from error
        attempt["status"] = "complete"
        if stage["gpu"]:
            write_json(ledger_path, ledger)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("stage")
    stage.add_argument("--study", type=Path, required=True)
    stage.add_argument("--family", choices=FAMILIES, required=True)
    stage.add_argument("--operation", choices=("build-second", "merge", "execute"), required=True)
    stage.add_argument("--first", type=Path, required=True)
    stage.add_argument("--second", type=Path)
    stage.add_argument("--policy", default="WT")
    stage.add_argument("--out", type=Path, required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--study", type=Path, required=True)
    plan.add_argument("--out", type=Path, required=True)
    plan.add_argument("--models", type=Path, default=Path("/home/jonas/models"))
    plan.add_argument("--policy-arm", action="append", default=[])
    plan.add_argument("--factorial-registration", type=Path)
    run = sub.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--authorization", type=Path)
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "stage":
        cpu_stage(args)
    elif args.command == "plan":
        result = write_plan(
            Path(__file__).resolve().parents[2],
            args.study,
            args.out,
            args.models,
            policy_arms=args.policy_arm,
            factorial_registration=args.factorial_registration,
        )
        print(
            json.dumps({"stages": len(result["stages"]), "preview": str(args.out / "dry-run.txt")})
        )
    else:
        run_queue(args.plan, execute=args.run, authorization=args.authorization)


if __name__ == "__main__":
    main()
