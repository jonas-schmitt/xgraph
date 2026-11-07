"""CPU-only comparison of catalog launchers with their development provenance."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import tempfile
from pathlib import Path

from xgraph.policy_catalog import CATALOG, GENERATION_DEFAULTS, recipe
from xgraph.provenance import provenance, sha256_file

# These are workload/artifact locations, never a blanket exemption for values
# that happen to start with '/'. Flag presence must still match exactly.
PATH_ROLES = {
    "--model": "served model location; model inventory is sealed separately",
    "--requests": "request file location for the declared evaluation panel",
    "--out": "separate evaluation destination preserves the measured development output",
    "--kg-index": "tokenizer-bound KG index location; payload inventory is sealed separately",
    "--tokenizer": "served tokenizer location; its contents are sealed separately",
    "--tools-db": "KG store location; its contents are sealed separately",
    "--relation-labels": "relation label location; its contents are sealed separately",
}


def development_sources(policy, family):
    """Name the original run, including incomplete/unstarted members of the catalog."""
    work = ".claude/worktrees"
    menu = (
        f"{work}/codex-menu/target/"
        + ("menu-smoke/development" if family == "qwen38" else f"menu-dev/{family}/development")
        + "/paths-menu-5-observed-3.jsonl"
    )
    width = (
        f"{work}/codex-menu2/target/menu2-1024/development/menuobs-width8.jsonl"
        if family == "qwen38"
        else f"{work}/codex-integrate/target/mintqa-width8/first/"
        f"{family}/development/menuobs-width8.jsonl"
    )
    blind = (
        f"{work}/codex-menu2/target/"
        + ("menu2-1024" if family == "qwen38" else f"blind5-{family}")
        + "/development/blind-5demos.jsonl"
    )
    spec = recipe(policy, family)
    if spec["agent_budget"]:
        directory = (
            "agent-qual"
            if spec["agent_feedback"]
            else "mintqa-rematch"
            if spec["agent_budget"] == 8
            else "agent-budget"
        )
        source_name = "agent-strong-8-feedback" if spec["agent_feedback"] else policy
        first = f"target/{directory}/{family}/development/{source_name}.jsonl"
    else:
        first = (
            width if policy in ("W", "WT") else blind if policy in ("blind5", "B2", "BT") else menu
        )
    result = [first + ".provenance.json"]
    if spec["second"]:
        stem = (
            "blind5"
            if policy in ("B2", "BT")
            else "menuobs-width8"
            if policy == "WT"
            else "menuobs"
        )
        if policy == "WT":
            base = f"{work}/codex-integrate/target/mintqa-width8/terminal/menuobs-width8"
        elif spec["terminal_width"]:
            base = f"{work}/codex-terminal/target/mintqa-terminal/" + (
                "blind5" if policy == "BT" else "menuobs"
            )
        else:
            base = f"{work}/codex-menu-audit/target/" + (
                "mintqa-blind5-audit" if policy == "B2" else "mintqa-menu-audit"
            )
        kind = "audit" if spec["second"].startswith("audit") else "revision"
        terminal = "-terminal" if spec["terminal_width"] else ""
        result.append(
            f"{base}/{family}/development/{stem}-{kind}{terminal}-raw.jsonl.provenance.json"
        )
    return result


def flags(argv):
    """Retain flag spelling, values and presence; only order/entrypoint are ignored."""
    argv = list(argv)
    prefix = []
    while argv and not argv[0].startswith("--"):
        prefix.append(argv.pop(0))
    if prefix:
        interpreter = Path(prefix[0]).name
        python = interpreter in ("python", "python3") or interpreter.startswith("python3.")
        script = len(prefix) == 2 and Path(prefix[1]).name == "generate.py"
        module = prefix[1:] == ["-m", "xgraph.generate"]
        if not python or not (script or module):
            raise ValueError(f"unlisted generation entrypoint arguments: {prefix}")
    result = {}
    while argv:
        key = argv.pop(0)
        if not key.startswith("--") or key in result:
            raise ValueError(f"invalid or duplicate argv flag: {key}")
        values = []
        while argv and not argv[0].startswith("--"):
            values.append(argv.pop(0))
        result[key] = values
    if "--model" not in result:
        raise ValueError("generation argv must name --model")
    return result


def path_exceptions(recorded, expanded):
    before, after = flags(recorded), flags(expanded)
    result = []
    for key, reason in PATH_ROLES.items():
        if (
            key not in before
            or key not in after
            or before[key] == after[key]
            or len(before[key]) != 1
            or len(after[key]) != 1
        ):
            continue
        original, moved = Path(before[key][0]), Path(after[key][0])
        if not original.is_absolute() or not moved.is_absolute():
            continue
        if key == "--out":
            proof = "destination only; no generated input is read from this argument"
        elif key == "--requests":
            proof = "declared panel substitution; only the file argument is compared to development"
        elif original.exists() and moved.exists() and original.samefile(moved):
            proof = "same resolved input"
        elif original.is_file() and moved.is_file() and sha256_file(original) == sha256_file(moved):
            proof = "identical input SHA-256: " + sha256_file(original)
        else:
            continue
        result.append(
            {
                "flag": key,
                "recorded": before[key],
                "recipe": after[key],
                "reason": reason,
                "proof": proof,
            }
        )
    return result


def compare_argv(recorded, expanded, exceptions=()):
    before, after = flags(recorded), flags(expanded)
    differences = {
        key: {"recorded": before.get(key), "recipe": after.get(key)}
        for key in sorted(before.keys() | after.keys())
        if before.get(key) != after.get(key)
    }
    permitted = path_exceptions(recorded, expanded)
    seen = set()
    for item in exceptions:
        key = item.get("flag")
        if item not in permitted or key in seen:
            raise ValueError(f"invalid or stale argv exception: {key}")
        seen.add(key)
        del differences[key]
    return differences


def verify_defaults(path):
    """Check omitted historical defaults without importing generation/model code."""
    found = {}
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        if not isinstance(node.args[0], ast.Constant):
            continue
        flag = node.args[0].value
        if flag in GENERATION_DEFAULTS:
            default = next((k.value for k in node.keywords if k.arg == "default"), None)
            found[flag] = ast.literal_eval(default)
    if found != GENERATION_DEFAULTS:
        raise ValueError(f"catalog omitted generation defaults changed: {found}")


def expand_launcher(command):
    """Intercept uv and fail closed before Python generation, builds, Docker or GPU."""
    command = list(command)
    launcher = next(i for i, p in enumerate(command) if p.endswith("/scripts/generate.sh"))
    if command[:2] != ["env", "-i"]:
        raise ValueError("catalog argv capture requires an isolated environment")
    with tempfile.TemporaryDirectory(prefix="xgraph-catalog-argv-") as scratch:
        root = Path(scratch)
        uv = root / "uv"
        uv.write_text(
            "#!/usr/bin/python3\nimport json, sys\n"
            'assert sys.argv[1:6] == ["run", "--no-sync", "python", "-m", "xgraph.generate"]\n'
            'assert "--provenance-only" in sys.argv\n'
            "print(json.dumps(sys.argv[6:]))\nsys.exit(77)\n"
        )
        uv.chmod(0o755)
        for name in ("docker", "nvidia-smi"):
            blocker = root / name
            blocker.write_text("#!/bin/sh\nexit 78\n")
            blocker.chmod(0o755)
        request, output = command[launcher + 2], command[launcher + 4]
        placeholder = root / "requests.jsonl"
        placeholder.write_text("{}\n")
        command[launcher + 2] = str(placeholder)
        command[launcher + 4] = str(root / "out.jsonl")
        command = [
            f"PATH={root}:{p.removeprefix('PATH=')}" if p.startswith("PATH=") else p
            for p in command
        ]
        command[2:2] = ["CUDA_VISIBLE_DEVICES=", "NVIDIA_VISIBLE_DEVICES=void"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if result.returncode != 77:
            raise ValueError(f"catalog argv interceptor did not stop safely: {result.stderr}")
        expanded = json.loads(result.stdout)
        # Restore the planned paths after using scratch files only for interception.
        expanded[expanded.index("--requests") + 1] = request
        expanded[expanded.index("--out") + 1] = output
        return expanded


def _identity(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def audit_policy(policy, family, steps, source_root):
    verify_defaults(Path(__file__).with_name("generate.py"))
    sources = development_sources(policy, family)
    if len(steps) != len(sources):
        raise ValueError("catalog generation stage count differs from development recipe")
    rows = []
    for stage, (step, source) in enumerate(zip(steps, sources, strict=True)):
        path = Path(source_root) / source
        expanded = expand_launcher(step.command)
        row = {
            "family": family,
            "policy": policy,
            "stage": "first" if stage == 0 else "second",
            "recorded_provenance": str(path),
            "status": "pending",
            "argv_matches": None,
            "expanded_argv": expanded,
            "exceptions": [],
            "inputs": [],
            "stopped_before_model_code_or_gpu": True,
        }
        if policy in ("W", "WT") and family == "gemma31":
            queue = (
                Path(source_root)
                / ".claude/worktrees/codex-integrate/target/queue-width8/width8.sh"
            )
            row["queue"] = str(queue)
            if queue.is_file():
                row["inputs"].append(_identity(queue))
        missing = [str(path)] if not path.is_file() else []
        if not missing:
            identity = _identity(path)
            saved = json.loads(path.read_text())["provenance"]
            recorded = saved["command"]
            row.update(
                recorded_argv=recorded,
                recorded_commit=saved["repo_commit"],
                recorded_sha256=identity["sha256"],
            )
            row["inputs"].append(identity)
            row["exceptions"] = path_exceptions(recorded, expanded)
            differences = compare_argv(recorded, expanded, row["exceptions"])
            if differences:
                raise ValueError(
                    f"unlisted catalog argv difference {family}/{policy}/{row['stage']}: "
                    f"{differences}"
                )
            row["argv_matches"] = True
            request = flags(recorded)["--requests"][0]
            request_identity = next((p for p in saved["inputs"] if p["path"] == request), None)
            if request_identity is None or not Path(request).is_file():
                missing.append(request)
            else:
                current = _identity(request)
                if any(current[k] != request_identity[k] for k in ("bytes", "sha256")):
                    raise ValueError(f"development request changed: {request}")
                row["inputs"].append(current)
            output = Path(str(path).removesuffix(".provenance.json"))
            for artifact in (output, Path(f"{output}.summary.json")):
                if artifact.is_file():
                    row["inputs"].append(_identity(artifact))
                else:
                    missing.append(str(artifact))
            if not missing:
                row["status"] = "matched"
        row["pending_artifacts"] = missing
        rows.append(row)
    return rows


def main(argv=None):
    from xgraph.mintqa_validation import generation_plan

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/home/jonas/git/xgraph"))
    parser.add_argument("--models", type=Path, default=Path("/home/jonas/models"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out = args.out.resolve()
    rows = []
    for family in ("qwen38", "gemma31"):
        for policy in CATALOG:
            options = argparse.Namespace(
                out=args.out.parent / "argv-preview" / policy,
                source_root=args.source_root,
                models=args.models,
                menu_first="menuobs",
                menu_max_tokens=1024,
                policy_selection={family: policy},
            )
            steps = [
                s
                for s in generation_plan(options, family)
                if s.requests and "/finalist" in str(s.output)
            ]
            rows.extend(audit_policy(policy, family, steps, args.source_root))
    report = {
        "artifact_kind": "xgraph.catalog.argv_audit",
        "results": rows,
        "normalization": "ignore entrypoint and flag order; every path difference is listed",
        "execution": "uv exits 77 before model code, builds, Docker or GPU",
        "provenance": provenance([Path(__file__), Path(__file__).with_name("policy_catalog.py")]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Catalog argv audit: {sum(r['status'] == 'matched' for r in rows)} matched, "
        f"{sum(r['status'] == 'pending' for r in rows)} pending; {args.out}"
    )


if __name__ == "__main__":
    main()
