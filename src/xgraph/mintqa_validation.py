"""Freeze, generate and score the untouched MINTQA validation panel.

Only run launches generation. Freeze and every dry run are bounded CPU work.
The seal and completed stage hashes bind all later actions to the same manifest.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import io
import json
import math
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

from xgraph import mintqa_queue as queue
from xgraph.mintqa_panels import indexed, read_jsonl, write_json
from xgraph.policy_catalog import CATALOG, GENERATION_DEFAULTS, fingerprint, recipe, selection
from xgraph.provenance import provenance, sha256_file

REPO = queue.REPO
FAMILY_ORDER = ("qwen38", "gemma31", "qwen25")
ARMS = (
    "agent-strong-8",
    "paths-allfix",
    "paths-allfix-stop",
    "menuobs",
    "menuobs-audit",
    "menuobs-two-pass",
    "audit",
    "two-pass",
)
MENU_NAMES = dict(
    zip(
        "ABCDEF",
        (
            "menuobs-4demos",
            "blind-5demos",
            "menuobs-width8",
            "menuobs-closenext",
            "menuobs-defer",
            "menuobs-width8-closenext-defer",
        ),
        strict=True,
    )
)


def registration():
    return {
        "panel": "validation",
        "questions": 2048,
        "resamples": 10_000,
        "seed": 17,
        "alpha": 0.025,
        "accuracy": "population-weighted native MINTQA normalized answer containment",
        "bootstrap": "paired relation-sequence clusters within hops, calibrated weights",
        "test": "centered paired cluster bootstrap upper tail, add-one correction",
        "bound": "percentile lower bound at alpha; strictly greater than null_difference",
        "multiplicity": "Holm across R1-R4 within each family at alpha",
        "token_ratio": "ratio of weighted prompt plus generated tokens; paired percentile 95% CI",
        "conjunction_families": ["qwen38", "gemma31"],
        "exploratory_families": ["qwen25"],
        "conjunction": "each registered comparison passes in both families; also report all four",
        "comparisons": [
            {
                "id": name,
                "arm": arm,
                "reference": reference,
                "null_difference": null,
                "alternative": "greater",
                "kind": "non-inferiority" if null else "superiority",
            }
            for name, arm, reference, null in (
                ("R1", "menuobs-audit", "agent-strong-8", -0.03),
                ("R2", "menuobs-two-pass", "agent-strong-8", -0.03),
                ("R3", "menuobs", "paths-allfix", 0.0),
                ("R4", "menuobs-audit", "audit", 0.0),
            )
        ],
        "limits": "exposed MINTQA internal validation; tokens are not controlled GPU cost",
    }


def git_state(repo=None):
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=repo or REPO, check=True, capture_output=True, text=True
        ).stdout.strip()

    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError(
            "dirty tree: commit the complete implementation before freezing or using it"
        )
    return git("rev-parse", "HEAD")


def file_identity(path, *, metadata=False):
    path = Path(path)
    try:
        stat = path.stat()
    except OSError as error:
        raise ValueError(f"frozen input changed or missing: {path}") from error
    result = {"path": str(path), "resolved": str(path.resolve()), "bytes": stat.st_size}
    if metadata:
        result["mtime_ns"] = stat.st_mtime_ns
    else:
        result["sha256"] = queue._hash(path)
    return result


def model_identity(path, weight_identity):
    files = sorted(p for p in path.rglob("*") if p.is_file() and ".cache" not in p.parts)
    weights = [p for p in files if p.suffix in (".safetensors", ".bin", ".pt")]
    if not weights or not (path / "tokenizer.json").is_file():
        raise ValueError(f"model needs weights and tokenizer.json: {path}")
    return {
        "path": str(path),
        "weight_identity": weight_identity,
        "files": [
            file_identity(p, metadata=p in weights and weight_identity == "size_mtime")
            for p in files
        ],
    }


def _safe_output(out, repo=None):
    if not (out.is_relative_to((repo or REPO) / "target") or out.is_relative_to(Path("/tmp"))):
        raise ValueError("outputs must stay in this worktree's target/ or /tmp")


def _no_validation_outputs(out, prepared, source_root):
    namespaces = {REPO / "target/mintqa-validation", source_root / "target/mintqa-validation"}
    roots = {
        out,
        REPO / "target/mintqa-rematch",
        prepared,
        *namespaces,
    }
    for root in roots:
        if root == out or root in namespaces:
            metadata = {"validation.lock"}
            if root != out:
                metadata.update({"freeze-manifest.json", "freeze-manifest.json.sha256"})
            files = [p for p in root.rglob("*") if p.is_file() and p.name not in metadata]
        else:
            files = [
                p
                for directory in root.rglob("validation")
                if directory != prepared / "panels/validation"
                and directory != REPO / "target/mintqa-rematch/panels/validation"
                for p in directory.rglob("*")
                if p.is_file()
            ]
        if files:
            raise ValueError(f"existing validation output prevents freeze: {files[0]}")


def _serialize(step):
    return {
        "command": step.command,
        "output": str(step.output),
        "inputs": list(map(str, step.inputs)),
        "requests": str(step.requests) if step.requests else None,
    }


def _step(value):
    return queue.Step(
        value["command"],
        Path(value["output"]),
        inputs=list(map(Path, value["inputs"])),
        requests=Path(value["requests"]) if value["requests"] else None,
    )


def _generation_settings(step):
    if step.requests is None:
        return None
    command = step.command
    environment = dict(x.split("=", 1) for x in command[1 : command.index("nice")] if "=" in x)
    flags = {
        name.removeprefix("--").replace("-", "_"): command[command.index(name) + 1]
        for name in (
            "--max-tokens",
            "--max-model-len",
            "--max-num-seqs",
            "--kv-cache-gib",
            "--seed",
            "--memory-share",
            "--repetition-penalty",
            "--samples",
            "--think-budget",
        )
        if name in command
    }
    return {
        "environment": environment,
        "flags": flags,
        "omitted_defaults": {k: v for k, v in GENERATION_DEFAULTS.items() if k not in command},
        "sampling": "greedy, temperature zero",
        "prefix_caching": True,
        "dtype": "auto",
        "max_path_relations": 8,
        "agent_max_tool_calls": int(command[command.index("--max-calls") + 1])
        if "--max-calls" in command
        else None,
        "ban_tokens": ["<|channel>"] if "--ban-tokens" in command else [],
        "launcher": str(REPO / "scripts/generate.sh"),
        "prompt": "base requests plus frozen code transformations; "
        "second-call feedback is generated by rematch stages",
    }


def _catalog_entries(args, family):
    selected = getattr(args, "policy_selection", {}).get(family)
    entries = [("finalist", selected)] if selected else []
    entries += [(name, name) for name in getattr(args, "secondary_cost_points", [])]
    if getattr(args, "strongest_agents", {}):
        entries.append(("agent-feedback-8", "agent-feedback-8"))
    return entries


def _catalog_first_name(args, slot, spec):
    if getattr(args, "strongest_agents", {}):
        if spec["policy"] in ("W", "WT"):
            return "W"
        if spec["policy"] in ("blind5", "B2", "BT"):
            return "blind5"
    return f"{slot}-first" if spec["second"] else slot


def _catalog_generations(args, family, steps, slot, policy):
    spec = recipe(policy, family)
    first = _catalog_first_name(args, slot, spec)
    names = [first]
    if spec["second"]:
        kind = "revision" if spec["second"].startswith("two-pass") else "audit"
        names.append(first + "-" + kind + ("-terminal" if spec["terminal_width"] else "") + "-raw")
    by_name = {s.output.stem: s for s in steps if s.requests}
    return [by_name[name] for name in names]


def generation_plan(args, family):
    """Reuse rematch's request, repair, support and acceptance policies verbatim."""
    prepared = args.source_root / "target/mintqa-rematch"
    options = argparse.Namespace(
        out=args.out,
        source_root=args.source_root,
        models=args.models,
        menu=5,
        observe_select=True,
        observe_required=True,
        observe_no_loop=True,
        observe_stop=False,
        revision=False,
        audit=False,
        source=None,
        source_arm_name=None,
        audit_arms=["audit", "two-pass"],
        audit_menu_size=8,
        audit_endpoints=12,
        audit_alternatives=4,
    )
    first = {
        s.output.stem: s
        for s in queue.arm_steps(options, family, "validation", [], prepared=prepared)
    }
    steps = [first[a] for a in ARMS[:3]]
    menu = first["paths-menu-5-observed-3"]
    old = menu.output
    menu.output = old.with_name("menuobs.jsonl")
    menu.command = [str(menu.output) if x == str(old) else x for x in menu.command]
    steps.append(menu)
    for source in ("menuobs", "paths-allfix"):
        options.source = (
            args.out / family / "validation" / f"{source}.jsonl" if source == "menuobs" else None
        )
        options.source_arm_name = "menuobs" if options.source else None
        extra = list(
            queue.arm_steps(
                options,
                family,
                "validation",
                [],
                prepared=prepared,
                audit_only=True,
                audit_local=True,
            )
        )
        # The selected first call was generated here already, so no import stage.
        if source == "menuobs":
            extra = [s for s in extra if s.output != menu.output]
        steps.extend(extra)
    catalog_steps, shared_firsts, shared_results, catalog_extra = {}, {}, {}, []
    for slot, policy in _catalog_entries(args, family):
        spec = recipe(policy, family)
        name = _catalog_first_name(args, slot, spec)
        extra = []
        if name not in shared_firsts:
            first_call = copy.deepcopy(first["agent-strong-8"] if spec["agent_budget"] else menu)
            old = str(first_call.output)
            first_call.output = first_call.output.with_name(f"{name}.jsonl")
            first_call.command = [
                str(first_call.output) if x == old else x for x in first_call.command
            ]
            extra.append(first_call)
            shared_firsts[name] = first_call
            catalog_steps[str(first_call.output)] = (spec, True)
        first_call = shared_firsts[name]
        key = (name, spec["second"], spec["terminal_width"])
        if spec["second"] and key not in shared_results:
            options.source = first_call.output
            options.source_arm_name = name
            options.audit_arms = [spec["second"]]
            options.terminal_width = spec["terminal_width"]
            extra.extend(
                s
                for s in queue.arm_steps(
                    options,
                    family,
                    "validation",
                    [],
                    prepared=prepared,
                    audit_only=True,
                    audit_local=True,
                )
                if s.output != first_call.output
            )
            final = extra[-1]
            old = str(final.output)
            final.output = final.output.with_name(f"{slot}.jsonl")
            final.command = [str(final.output) if x == old else x for x in final.command]
            for step in extra:
                if step.requests and step is not first_call:
                    catalog_steps[str(step.output)] = (spec, False)
            shared_results[key] = final
        result = shared_results.setdefault(key, first_call)
        if result.output.stem != slot:
            output = result.output.with_name(f"{slot}.jsonl")
            extra.append(
                queue.Step(
                    queue.cpu(
                        "xgraph.mintqa_rematch",
                        "source",
                        "--requests",
                        first_call.requests,
                        "--outputs",
                        result.output,
                        "--out",
                        output,
                    ),
                    output,
                    inputs=[*result.inputs, result.output, first_call.requests],
                )
            )
        catalog_extra.extend(extra)
    steps = (
        [*catalog_extra, *steps]
        if getattr(args, "strongest_agents", {})
        else [*steps, *catalog_extra]
    )
    for step in steps:
        # The sealed model inventory is checked before/after every stage. Metadata
        # mode never silently falls back to hashing thirty GB through run_step.
        step.inputs = [p for p in step.inputs if p.suffix not in (".safetensors", ".bin", ".pt")]
        step.inputs.append(args.out / "freeze-manifest.json")
        if step.requests is None:
            continue
        is_menu = step.output.stem == "menuobs"
        agent = step.output.stem == "agent-strong-8"
        choice = args.menu_first if is_menu else ""
        enabled = is_menu and choice != "B"
        controls = {
            "XG_MENU": 5 if enabled else 0,
            "XG_OBSERVE": 3 if enabled else 0,
            "XG_OBSERVE_SELECT": int(enabled),
            "XG_OBSERVE_ANSWER": int(enabled),
            "XG_OBSERVE_REQUIRED": int(enabled),
            "XG_OBSERVE_NO_LOOP": int(enabled),
            "XG_OBSERVE_WIDTH": 8 if choice in ("C", "F") else 0,
            "XG_MENU_CLOSE_NEXT": int(choice in ("D", "F")),
            "XG_OBSERVE_DEFER": 2 if choice in ("E", "F") else 0,
            "XG_PATH_DEMOS": (4 if choice == "A" else 5) if is_menu else "",
            "XG_AUDIT": 0,
            "XG_AUDIT_ENTITIES": 20,
            "XG_AUDIT_RELATIONS": 8,
            "XG_AVAILABLE_WAIT": 43200,
        }
        catalog = catalog_steps.get(str(step.output))
        stage_generation = None
        if agent:
            stage_generation = recipe("agent-strong-8", family)["generation"]["first"]
        elif args.menu_first == "menuobs":
            registered_stage = {
                "menuobs": ("M2", "first"),
                "menuobs-audit-raw": ("menuobs-audit", "second"),
                "menuobs-revision-raw": ("M2", "second"),
            }.get(step.output.stem)
            if registered_stage:
                policy, stage = registered_stage
                stage_generation = recipe(policy, family)["generation"][stage]
        if catalog:
            spec, is_first = catalog
            stage_generation = spec["generation"]["first" if is_first else "second"]
            observed = bool(is_first and spec["menu"])
            controls.update(
                XG_MENU=spec["menu"] if is_first else 0,
                XG_OBSERVE=spec["observe"] if is_first else 0,
                XG_OBSERVE_WIDTH=spec["observe_width"] if is_first else 0,
                XG_MENU_CLOSE_NEXT=0,
                XG_OBSERVE_DEFER=0,
                XG_OBSERVE_TERMINAL=0 if is_first else spec["terminal_width"],
                **{
                    f"XG_OBSERVE_{k}": int(observed)
                    for k in ("SELECT", "ANSWER", "REQUIRED", "NO_LOOP")
                },
            )
        if stage_generation:
            controls["XG_PATH_DEMOS"] = stage_generation["path_demos"] or ""
        environment = {
            x.split("=", 1)[0]: x.split("=", 1)[1]
            for x in step.command[1 : step.command.index("nice")]
        }
        # The host provenance call reads native XGRAPH_* variables too. Start
        # from an empty environment and freeze only these non-secret host paths.
        environment.update(
            {
                key: os.environ[key]
                for key in (
                    "PATH",
                    "HOME",
                    "USER",
                    "LOGNAME",
                    "LANG",
                    "LC_ALL",
                    "TMPDIR",
                    "UV_PYTHON_INSTALL_DIR",
                    "UV_PYTHON_CACHE_DIR",
                    "CUDA_HOME",
                    "CUDA_PATH",
                )
                if key in os.environ
            }
        )
        environment.update(
            UV_CACHE_DIR=str(REPO / ".uv-cache"),
            TOKENIZERS_PARALLELISM="false",
            PYTHONDONTWRITEBYTECODE="1",
            CMAKE_BUILD_PARALLEL_LEVEL="1",
        )
        environment.update({k: str(v) for k, v in controls.items()})
        if getattr(args, "strongest_agents", {}):
            environment["XG_NEED_GIB"] = str({"qwen38": 60, "gemma31": 75, "qwen25": 40}[family])
        command = step.command[step.command.index("nice") :]
        command[command.index("--max-tokens") + 1] = str(args.menu_max_tokens if is_menu else 256)
        if stage_generation:
            launcher = command.index(str(REPO / "scripts/generate.sh"))
            command[launcher + 5 :] = stage_generation["argv"]
        else:
            command += [
                "--memory-share",
                "0.30",
                "--repetition-penalty",
                "1.0",
                "--samples",
                "1",
                "--think-budget",
                "0",
            ]
        step.command = [
            "env",
            "-i",
            *(f"{k}={v}" for k, v in sorted(environment.items())),
            *command,
        ]
        # The ordinary allfix demonstrations and second-call histories come from
        # the frozen request file; agent transforms them to matched tool examples.
        if not agent and not is_menu and step.output.stem in ("paths-allfix", "paths-allfix-stop"):
            step.command[step.command.index("XG_PATH_DEMOS=")] = "XG_PATH_DEMOS=4"
    return steps


def _timings(args, family, steps):
    """Snapshot measured development workload time, naming every proxy explicitly."""
    from xgraph.catalog_argv import development_sources

    root = args.source_root
    base = root / "target/mintqa-rematch" / family / "development"
    menu_root = root / ".claude/worktrees/codex-menu/target"
    menu = (
        menu_root / "menu-smoke/development"
        if family == "qwen38"
        else menu_root / "menu-dev" / family / "development"
    ) / "paths-menu-5-observed-3.jsonl.summary.json"
    second = (
        root
        / ".claude/worktrees/codex-menu-audit/target/mintqa-menu-audit"
        / family
        / "development"
    )
    menu2 = root / f".claude/worktrees/codex-menu2/target/menu2-{args.menu_max_tokens}/development"
    sources = {
        "menuobs": (
            [menu2 / f"{MENU_NAMES[args.menu_first]}.jsonl.summary.json"]
            if args.menu_first in MENU_NAMES and family == "qwen38"
            else []
        )
        + [menu],
        "menuobs-audit-raw": [second / "menuobs-audit-raw.jsonl.summary.json"],
        "menuobs-revision-raw": [second / "menuobs-revision-raw.jsonl.summary.json"],
        "audit-raw": [
            root
            / ".claude/worktrees/codex-audit2/target/mintqa-rematch"
            / family
            / "development/audit-raw.jsonl.summary.json"
        ],
    }
    for slot, policy in _catalog_entries(args, family):
        calls = _catalog_generations(args, family, steps, slot, policy)
        for step, source in zip(calls, development_sources(policy, family), strict=True):
            sources[step.output.stem] = [root / source.replace(".provenance.json", ".summary.json")]
    result = []
    for step in steps:
        if step.requests is None:
            continue
        name = step.output.stem
        exact = sources.get(name, [base / f"{name}.jsonl.summary.json"])
        candidates = [
            *exact,
            base / "revision-raw.jsonl.summary.json",
            second / "menuobs-revision-raw.jsonl.summary.json",
            menu,
            base / "agent-strong-8.jsonl.summary.json",
        ]
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            raise ValueError(f"no measured development timing for {family}/{name}")
        data = json.loads(source.read_text())
        seconds, count = data["seconds"], data["requests"]
        if not math.isfinite(seconds) or seconds <= 0 or count <= 0:
            raise ValueError(f"invalid development timing: {source}")
        proxy = source not in exact or (
            name == "menuobs" and args.menu_first != "menuobs" and source == menu
        )
        result.append(
            {
                "generation": name,
                "source": file_identity(source),
                "seconds": seconds,
                "requests": count,
                "proxy": proxy,
                "validation_hours": seconds * 2048 / count / 3600,
            }
        )
    return {
        "generations": result,
        "workload_gpu_hours": sum(r["validation_hours"] for r in result),
        "definition": "linear workload-only GPU-hour estimate; "
        "excludes startup, CPU stages and waits; "
        "proxy timings do not establish controlled serving cost",
    }


def build_manifest(args):
    from xgraph.catalog_argv import audit_policy

    commit = git_state()
    prepared = args.source_root / "target/mintqa-rematch"
    _no_validation_outputs(args.out, prepared, args.source_root)
    panel = prepared / "panels/validation"
    panels = prepared / "panels/panels.json"
    data = args.source_root / "data/external/mintqa-full"
    inputs = [
        panels,
        panel / "requests.jsonl",
        panel / "questions.json",
        data / "kg.sqlite",
        data / "kg/relation_labels.jsonl",
        *(data / f"questions/{s}.jsonl" for s in ("ti", "pop")),
    ]
    snapshot = {str(p): file_identity(p) for p in inputs}
    members = json.loads(panels.read_text())["panels"]["validation"]["members"]
    if (
        len(members) != 2048
        or set(indexed(members)) != set(indexed(read_jsonl(panel / "requests.jsonl")))
        or set(indexed(members)) != set(indexed(json.loads((panel / "questions.json").read_text())))
    ):
        raise ValueError(
            "validation panel must contain exactly 2048 aligned unique questions and requests"
        )
    families = {}
    for family in FAMILY_ORDER:
        index = prepared / "indexes" / family
        index_manifest = index.with_name(f"{family}.manifest.json")
        snapshot[str(index_manifest)] = file_identity(index_manifest)
        inventory = json.loads(index_manifest.read_text())["files"]
        if {p.name for p in index.iterdir() if p.is_file()} != {r["name"] for r in inventory}:
            raise ValueError(f"KG index manifest mismatch: {index}")
        for item in inventory:
            path = index / item["name"]
            identity = file_identity(path)
            if identity["sha256"] != item["sha256"]:
                raise ValueError(f"KG index changed: {path}")
            snapshot[str(path)] = identity
        model = model_identity(args.models / queue.FAMILIES[family][0], args.weight_identity)
        steps = generation_plan(args, family)
        families[family] = {
            "arms": [*ARMS, *(slot for slot, _ in _catalog_entries(args, family))],
            "model": model,
            "kg_index_manifest": snapshot[str(index_manifest)],
            "menu_first": {
                "choice": args.menu_first,
                "definition": MENU_NAMES.get(args.menu_first, "paths-menu-5-observed-3"),
                "max_tokens": args.menu_max_tokens,
                "slot": "menuobs",
            },
            "prompt_demo_set": {
                "base_requests": str(panel / "requests.jsonl"),
                "allfix": "original four path demonstrations",
                "agent": "matched tool demonstrations from base requests",
                "menuobs": "four"
                if args.menu_first == "A"
                else "five including Germany highest point mountain range",
                "second_calls": "base demonstrations plus rematch execution feedback "
                "and semantic first reply",
            },
            "steps": [_serialize(s) for s in steps],
            "estimate": _timings(args, family, steps),
            "generation_settings": {
                s.output.stem: _generation_settings(s) for s in steps if s.requests is not None
            },
        }
        by_name = {s.output.stem: s for s in steps if s.requests}
        references = audit_policy(
            "agent-strong-8", family, [by_name["agent-strong-8"]], args.source_root
        )
        if args.strongest_agents:
            families[family]["agent_comparators"] = ["agent-strong-8", "agent-feedback-8"]
            references.extend(
                audit_policy(
                    "agent-feedback-8", family, [by_name["agent-feedback-8"]], args.source_root
                )
            )
        if args.menu_first == "menuobs":
            for policy, second in (
                ("M2", "menuobs-revision-raw"),
                ("menuobs-audit", "menuobs-audit-raw"),
            ):
                rows = audit_policy(
                    policy, family, [by_name["menuobs"], by_name[second]], args.source_root
                )
                references.extend(rows if policy == "M2" else rows[1:])
        families[family]["reference_argv_audit"] = references
        audit = []
        if args.policy_selection:
            spec = recipe(args.policy_selection[family], family)
            families[family].update(selected_policy=spec, policy_sha256=fingerprint(spec))
            selected_steps = _catalog_generations(args, family, steps, "finalist", spec["policy"])
            audit = audit_policy(spec["policy"], family, selected_steps, args.source_root)
            families[family]["development_argv_audit"] = audit
        for point in args.secondary_cost_points:
            audit.extend(
                audit_policy(
                    point,
                    family,
                    _catalog_generations(args, family, steps, point, point),
                    args.source_root,
                )
            )
        for row in [*references, *audit]:
            for item in row["inputs"]:
                current = file_identity(item["path"])
                if any(current[k] != item[k] for k in ("bytes", "sha256")):
                    raise ValueError(f"development input changed during freeze: {item['path']}")
                snapshot[item["path"]] = current
    result = {
        "artifact_kind": "xgraph.mintqa_validation.freeze",
        "version": 1,
        "repo": str(REPO),
        "code_commit": commit,
        "out": str(args.out),
        "source_root": str(args.source_root),
        "models": str(args.models),
        "panels": str(panels),
        "panel_sha256": snapshot[str(panel / "requests.jsonl")]["sha256"],
        "inputs": list(snapshot.values()),
        "families": families,
        "registration": registration(),
        "weight_identity_note": "size_mtime records bytes and nanosecond mtime "
        "instead of reading large weights; sha256 hashes all weights",
        "provenance": provenance([]),
    }
    if args.policy_selection:
        result["policy_selection"] = args.policy_selection
        result["version"] = 2
    if args.strongest_agents:
        result["strongest_agents"] = args.strongest_agents
    if args.secondary_cost_points:
        result["secondary_cost_points"] = {
            "role": "declared secondary accuracy and token-cost analyses; outside primary Holm",
            "comparisons": [
                {
                    "id": f"S-{name}",
                    "arm": name,
                    "reference_by_family": args.strongest_agents,
                    "null_difference": -0.03,
                    "alternative": "greater",
                    "kind": "non-inferiority",
                }
                for name in args.secondary_cost_points
            ],
        }
    if args.multiplicity == "intersection-holm":
        result["version"] = 2
        result["amendment"] = {
            "multiplicity": "intersection-holm",
            "rule": "max over required families/endpoints, then Holm across comparisons",
            "required_endpoints": ["accuracy"],
            "comparisons": [
                *registration()["comparisons"],
                *(
                    [
                        {
                            "id": "F",
                            "arm": "finalist",
                            **(
                                {"reference_by_family": args.strongest_agents}
                                if args.strongest_agents
                                else {"reference": "agent-strong-8"}
                            ),
                            "null_difference": -0.03,
                            "alternative": "greater",
                            "kind": "non-inferiority",
                        }
                    ]
                    if args.policy_selection
                    else []
                ),
            ],
        }
    if args.previous_freeze:
        result["previous_freeze"] = file_identity(args.previous_freeze)
        result["inputs"].extend(
            [result["previous_freeze"], file_identity(Path(f"{args.previous_freeze}.sha256"))]
        )
    if args.feedback_qualification:
        result["inputs"].append(file_identity(args.feedback_qualification))
    if args.nomination:
        result["nomination"] = file_identity(args.nomination)
        result["inputs"].append(result["nomination"])
    if git_state() != commit:
        raise ValueError("code changed while freezing")
    for item in result["inputs"]:
        if file_identity(item["path"]) != item:
            raise ValueError(f"input changed during freeze: {item['path']}")
    for family in result["families"].values():
        model = family["model"]
        if model_identity(Path(model["path"]), model["weight_identity"]) != model:
            raise ValueError(f"model changed during freeze: {model['path']}")
    _no_validation_outputs(args.out, prepared, args.source_root)
    return result


def load_frozen(path, *, policy_consumer=False):
    path = Path(path).resolve()
    seal = Path(f"{path}.sha256")
    if not seal.is_file() or sha256_file(path) != seal.read_text().strip():
        raise ValueError("freeze manifest mismatch or missing seal")
    manifest = json.loads(path.read_text())
    # A serving/transfer consumer verifies the producer's unchanged checkout.
    # Validation execution retains its original same-worktree restriction.
    repo = Path(manifest["repo"]).resolve() if policy_consumer else REPO
    if manifest["repo"] != str(repo) or manifest["out"] != str(path.parent):
        raise ValueError("freeze manifest checkout/output mismatch")
    _safe_output(path.parent, repo)
    if git_state(repo) != manifest["code_commit"]:
        raise ValueError("code commit changed since freeze")
    environment = provenance([])
    if any(environment[k] != manifest["provenance"][k] for k in ("python", "packages")):
        raise ValueError("frozen Python/package environment mismatch")
    for item in manifest["inputs"]:
        if file_identity(item["path"]) != item:
            raise ValueError(f"frozen input changed: {item['path']}")
    for family in manifest["families"].values():
        index_manifest = Path(family["kg_index_manifest"]["path"])
        index = index_manifest.with_name(index_manifest.name.removesuffix(".manifest.json"))
        inventory = json.loads(index_manifest.read_text())["files"]
        if {p.name for p in index.iterdir() if p.is_file()} != {r["name"] for r in inventory}:
            raise ValueError(f"frozen KG index inventory changed: {index}")
        for item in inventory:
            if file_identity(index / item["name"])["sha256"] != item["sha256"]:
                raise ValueError(f"frozen KG index manifest mismatch: {index / item['name']}")
        model = family["model"]
        if model_identity(Path(model["path"]), model["weight_identity"]) != model:
            raise ValueError(f"frozen model changed: {model['path']}")
        for spec in family["steps"]:
            if not Path(spec["output"]).resolve().is_relative_to(path.parent):
                raise ValueError("frozen output path mismatch")
    return manifest


def verify_stages(manifest, family, *, complete):
    """Never adopt orphan output, altered sidecars or a partial charged source."""
    for value in manifest["families"][family]["steps"]:
        step = _step(value)
        done = Path(f"{step.output}.done.json")
        artifacts = [
            step.output,
            *(
                Path(f"{step.output}{suffix}")
                for suffix in (".log", ".summary.json", ".provenance.json")
            ),
        ]
        if done.exists():
            with contextlib.redirect_stdout(io.StringIO()):
                queue.run_step(step, dry_run=True, cwd=REPO)
        elif complete or any(p.exists() for p in artifacts):
            raise ValueError(f"incomplete validation stage: {step.output}")


def score_step(manifest, family):
    out = Path(manifest["out"])
    dest = out / family / "report.json"
    register = manifest["registration"]
    arms = manifest["families"][family]["arms"]
    inputs = [
        out / "freeze-manifest.json",
        *(out / family / "validation" / f"{a}.jsonl" for a in arms),
    ]
    command = queue.cpu(
        "xgraph.mintqa_rematch",
        "score",
        "--panels",
        manifest["panels"],
        "--family",
        family,
        "--directory",
        out / family,
        "--data",
        Path(manifest["source_root"]) / "data/external/mintqa-full",
        "--panel",
        "validation",
        *(x for arm in arms for x in ("--arm", arm)),
        "--resamples",
        register["resamples"],
        "--seed",
        register["seed"],
        "--freeze-manifest",
        out / "freeze-manifest.json",
        "--out",
        dest,
    )
    return queue.Step(command, dest, inputs=inputs)


def validate_score_args(args):
    manifest = load_frozen(args.freeze_manifest)
    expected = score_step(manifest, args.family).command
    for flag, value in (
        ("--directory", args.directory),
        ("--panels", args.panels),
        ("--data", args.data),
        ("--resamples", args.resamples),
        ("--seed", args.seed),
        ("--out", args.out),
    ):
        if str(value) != expected[expected.index(flag) + 1]:
            raise ValueError(f"frozen score configuration mismatch: {flag}")
    arms = manifest["families"][args.family]["arms"]
    if args.panel != ["validation"] or args.arm != arms or args.agent_directory is not None:
        raise ValueError("frozen score arms, panel or agent directory mismatch")
    for family in (
        manifest["registration"]["conjunction_families"]
        if manifest.get("strongest_agents")
        else [args.family]
    ):
        verify_stages(manifest, family, complete=True)
    expected_ids = set(
        indexed(read_jsonl(Path(manifest["panels"]).parent / "validation/requests.jsonl"))
    )
    for arm in arms:
        if set(indexed(read_jsonl(args.directory / "validation" / f"{arm}.jsonl"))) != expected_ids:
            raise ValueError(f"incomplete validation arm: {arm}")
    return manifest


def intersection_holm(reports, comparisons, families, *, alpha, endpoints=("accuracy",)):
    """Intersection-union p-values precede multiplicity across claimable comparisons."""
    from xgraph.mintqa_rematch import holm_adjust

    result = {}
    for name in comparisons:
        components = []
        for family in families:
            row = reports[family][name]
            for endpoint in endpoints:
                if endpoint == "accuracy":
                    components.append(row)
                elif endpoint in row.get("endpoints", {}):
                    components.append(row["endpoints"][endpoint])
                else:
                    raise ValueError(f"missing required endpoint {family}/{name}/{endpoint}")
        p = max(c["p_value"] for c in components)
        # Validate every component, including one hidden below the maximum.
        holm_adjust([c["p_value"] for c in components])
        result[name] = {
            "p_value": p,
            "components_pass": all(c.get("passes_unadjusted", True) for c in components),
        }
    for row, p in zip(
        result.values(), holm_adjust([r["p_value"] for r in result.values()]), strict=True
    ):
        row.update(p_holm=p, passes=row["components_pass"] and p <= alpha)
    return result


def conjunction(manifest):
    out = Path(manifest["out"])
    reports, amended = {}, {}
    inputs = [out / "freeze-manifest.json"]
    for family in FAMILY_ORDER:
        step = score_step(manifest, family)
        if Path(f"{step.output}.done.json").is_file():
            verify_stages(manifest, family, complete=True)
            with contextlib.redirect_stdout(io.StringIO()):
                queue.run_step(step, dry_run=True, cwd=REPO)
            report = json.loads(step.output.read_text())
            reports[family] = report["registered_comparisons"]
            if "amended_comparisons" in report:
                amended[family] = report["amended_comparisons"]
            inputs.extend([step.output, Path(f"{step.output}.done.json")])
    required = manifest["registration"]["conjunction_families"]
    missing = [f for f in required if f not in reports]
    joint = {
        r["id"]: None if missing else all(reports[f][r["id"]]["passes"] for f in required)
        for r in manifest["registration"]["comparisons"]
    }
    result = {
        "artifact_kind": "xgraph.mintqa_validation.conjunction",
        "families": reports,
        "required_families": required,
        "pending_families": missing,
        "comparisons": joint,
        "all_pass": None if missing else all(joint.values()),
        "freeze_sha256": sha256_file(out / "freeze-manifest.json"),
        "provenance": provenance(inputs),
    }
    if "amendment" in manifest:
        result["amendment"] = manifest["amendment"]
        result["amended_comparisons"] = (
            None
            if missing
            else intersection_holm(
                amended,
                [r["id"] for r in manifest["amendment"]["comparisons"]],
                required,
                alpha=manifest["registration"]["alpha"],
                endpoints=manifest["amendment"]["required_endpoints"],
            )
        )
        result["amended_all_pass"] = (
            None
            if missing
            else all(row["passes"] for row in result["amended_comparisons"].values())
        )
    return result


def _print_estimate(manifest, family):
    estimate = manifest["families"][family]["estimate"]
    print(f"# {family}: {estimate['workload_gpu_hours']:.3f} GPU-hours; {estimate['definition']}")
    for row in estimate["generations"]:
        print(
            f"# {row['generation']}: {row['seconds']:.3f}s/{row['requests']} "
            "development questions; "
            f"proxy={row['proxy']}; {row['source']['path']}; sha256={row['source']['sha256']}"
        )


def validation_queue(manifest, *, dry_run):
    """Finish all required-family arms before the first sealed analysis starts."""
    families = manifest["registration"]["conjunction_families"]
    out = Path(manifest["out"])
    path = out / "freeze-manifest.json"
    log = Path(manifest["source_root"]) / "target/queue/validation.log"
    for family in families:
        verify_stages(manifest, family, complete=False)
        _print_estimate(manifest, family)
    print(f"# Stage exit log: {log}")
    planned = [
        (family, _step(s)) for family in families for s in manifest["families"][family]["steps"]
    ] + [(family, score_step(manifest, family)) for family in families]
    for family, step in planned:
        if dry_run:
            print(shlex.join(step.command))
            continue
        status = 1
        try:
            load_frozen(path)
            if "score" in step.command:
                for required in families:
                    verify_stages(manifest, required, complete=True)
            queue.run_step(step, cwd=REPO)
            load_frozen(path)
            status = 0
        except subprocess.CalledProcessError as error:
            status = error.returncode if error.returncode >= 0 else 128 - error.returncode
            raise
        finally:
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as stream:
                stream.write(
                    f"validation {family} {step.output.stem} exit {status} "
                    f"{datetime.now().astimezone().isoformat(timespec='seconds')}\n"
                )
    if not dry_run:
        pending = out / "conjunction.json.pending"
        write_json(pending, conjunction(manifest))
        pending.replace(out / "conjunction.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("freeze", "reseal", "run", "score", "queue"):
        cmd = commands.add_parser(action)
        cmd.add_argument("--out", type=Path, default=REPO / "target/mintqa-validation")
        cmd.add_argument("--dry-run", action="store_true")
        if action in ("freeze", "reseal"):
            cmd.add_argument("--source-root", type=Path, default=Path("/home/jonas/git/xgraph"))
            cmd.add_argument("--models", type=Path, default=Path("/home/jonas/models"))
            cmd.add_argument("--menu-first", choices=("menuobs", *MENU_NAMES), default="menuobs")
            cmd.add_argument("--menu-max-tokens", type=int, choices=(256, 1024), default=1024)
            cmd.add_argument("--policy", choices=CATALOG)
            cmd.add_argument("--family-policy", action="append", default=[])
            cmd.add_argument("--strongest-agent", action="append", default=[])
            cmd.add_argument(
                "--secondary-cost-point", choices=("W", "B2"), action="append", default=[]
            )
            cmd.add_argument("--nomination", type=Path)
            cmd.add_argument("--feedback-qualification", type=Path)
            cmd.add_argument(
                "--multiplicity", choices=("registered", "intersection-holm"), default="registered"
            )
            cmd.add_argument("--previous-freeze", type=Path, required=action == "reseal")
            cmd.add_argument(
                "--weight-identity", choices=("size_mtime", "sha256"), default="size_mtime"
            )
        elif action != "queue":
            cmd.add_argument("family", choices=FAMILY_ORDER)
    args = parser.parse_args(argv)
    args.out = args.out.resolve()
    _safe_output(args.out)
    freezing = args.action in ("freeze", "reseal")
    if freezing:
        args.policy_selection = selection(args.policy, args.family_policy, FAMILY_ORDER)
        args.strongest_agents = (
            selection("agent-strong-8", args.strongest_agent, FAMILY_ORDER)
            if args.strongest_agent
            else {}
        )
        if args.strongest_agents:
            declared = {s.partition("=")[0] for s in args.strongest_agent}
            if not {"qwen38", "gemma31"} <= declared or any(
                p not in ("agent-strong-8", "agent-feedback-8")
                for p in args.strongest_agents.values()
            ):
                parser.error(
                    "declare plain or feedback eight-call strongest agents "
                    "for both required families"
                )
            if not args.policy_selection or args.multiplicity != "intersection-holm":
                parser.error("strongest agents need a selected policy and intersection-holm")
        args.secondary_cost_points = list(dict.fromkeys(args.secondary_cost_point))
        if args.secondary_cost_points and not args.strongest_agents:
            parser.error("secondary cost points need frozen strongest agents")
        if args.policy_selection and (args.menu_first != "menuobs" or args.menu_max_tokens != 1024):
            parser.error("catalog policies have fixed first-stage settings; omit menu overrides")
        if "agent-feedback-8" in args.policy_selection.values():
            if not args.feedback_qualification:
                parser.error(
                    "feedback policy needs a prospectively declared --feedback-qualification"
                )
            qualified = json.loads(args.feedback_qualification.read_text())
            needed = {f for f, p in args.policy_selection.items() if p == "agent-feedback-8"}
            if qualified.get("qualified") is not True or not needed <= set(
                qualified.get("families", [])
            ):
                parser.error("feedback qualification must cover every frozen family")
        if args.previous_freeze:
            args.previous_freeze = args.previous_freeze.resolve()
            old = args.previous_freeze
            if old.parent == args.out:
                parser.error("re-seal needs a fresh --out directory")
            if sha256_file(old) != Path(f"{old}.sha256").read_text().strip():
                raise ValueError("previous freeze manifest mismatch")
            for path in old.parent.rglob("*"):
                if path.is_file() and path.name not in {
                    old.name,
                    f"{old.name}.sha256",
                    "validation.lock",
                }:
                    raise ValueError(f"existing validation output prevents re-seal: {path}")
        args.source_root, args.models = args.source_root.resolve(), args.models.resolve()
        if args.menu_first == "menuobs" and args.menu_max_tokens != 1024:
            parser.error("menuobs is frozen at 1024 tokens; choose an A-F ablation for 256")
        if args.out.is_relative_to(args.source_root / "target/mintqa-rematch"):
            parser.error("validation outputs must be separate from prepared inputs")
    if args.action == "queue":
        if args.dry_run:
            validation_queue(load_frozen(args.out / "freeze-manifest.json"), dry_run=True)
        else:
            with (args.out / "validation.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                validation_queue(load_frozen(args.out / "freeze-manifest.json"), dry_run=False)
        return
    if args.dry_run:
        manifest = (
            build_manifest(args) if freezing else load_frozen(args.out / "freeze-manifest.json")
        )
        if not freezing:
            verify_stages(manifest, args.family, complete=False)
        families = FAMILY_ORDER if freezing else [args.family]
        for family in families:
            _print_estimate(manifest, family)
            steps = (
                [_step(s) for s in manifest["families"][family]["steps"]]
                if args.action != "score"
                else []
            )
            if args.action != "run":
                steps.append(score_step(manifest, family))
            for step in steps:
                print(shlex.join(step.command))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "validation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.out / "freeze-manifest.json"
        if freezing:
            manifest = build_manifest(args)
            write_json(path, manifest)
            with Path(f"{path}.sha256").open("x") as seal:
                seal.write(sha256_file(path) + "\n")
            print(f"Frozen {path}; weights={args.weight_identity}")
            return
        manifest = load_frozen(path)
        verify_stages(manifest, args.family, complete=args.action == "score")
        steps = (
            [_step(s) for s in manifest["families"][args.family]["steps"]]
            if args.action == "run"
            else [score_step(manifest, args.family)]
        )
        for step in steps:
            load_frozen(path)
            queue.run_step(step, cwd=REPO)
            load_frozen(path)
        if args.action == "score":
            result = conjunction(manifest)
            pending = args.out / "conjunction.json.pending"
            write_json(pending, result)
            pending.replace(args.out / "conjunction.json")


if __name__ == "__main__":
    main()
