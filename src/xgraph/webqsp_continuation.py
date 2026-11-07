"""Outcome-blind, explicitly authorized continuation of the sealed WebQSP 8a test."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shlex
import shutil
import subprocess
import warnings
from pathlib import Path

from xgraph import webqsp_test as study

ROOT = study.ROOT
PARENT_REGISTRATION_SHA256 = "76a878256c21ee4777c8ff281b3aa315ba2c82e98984a282ef9fa37924eaa46e"
PARENT_EXECUTION_SHA256 = "c9a24d5e050c803e0bc6a6d8efae473efbbf9646cb66112a33fc5c257dad31a3"
OWNER_SHA256 = "0592368b5b58cdafa6b7e7fad9e7b8274c496485895b0f2578437c0d69d8e9a8"
OWNER_BYTES = 2168
REPAIR = "2728c88"
DEFAULT_PARENT = ROOT.parent / "codex-webqsp-test/target/webqsp-test/amendment-8a"
DEFAULT_OUT = ROOT / "target/webqsp-accounting/8a-continuation"
DEFAULT_AUTHORITY = Path(
    "/home/jonas/git/xgraph/target/owner/webqsp-8a-accounting-exception-2026-10-31.md"
)
CPU_ADDRESS_BYTES = 32 * 1024**3
SCORE_ADDRESS_BYTES = 8 * 1024**3


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace_paths(value, replacements):
    if isinstance(value, dict):
        return {k: replace_paths(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [replace_paths(v, replacements) for v in value]
    if isinstance(value, str):
        for before, after in replacements:
            value = value.replace(before, after)
    return value


def normalized_science(execution):
    """Compare every parent field; permit only named path/accounting changes."""
    result = copy.deepcopy(execution)
    if "continuation" not in result:
        return result
    continuation = result.pop("continuation")
    result = replace_paths(result, [(b, a) for a, b in reversed(continuation["path_map"])])
    result["artifact_kind"] = "xgraph.webqsp_test_execution"
    result["initial_gpu_seconds"] = continuation["parent_initial_gpu_seconds"]
    return result


def science_sha256(execution):
    # JSON equality also preserves types (Python considers False equal to 0).
    return digest(
        json.dumps(normalized_science(execution), sort_keys=True, allow_nan=False).encode()
    )


def prospective_execution(draft, directory):
    child = copy.deepcopy(draft)
    child["continuation"]["draft"] = study.identity(Path(directory) / "draft.json")
    return child


def seal_sha256(value):
    return digest(
        (
            json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        ).encode()
    )


def git_object(reference):
    blob = subprocess.check_output(["git", "rev-parse", reference], cwd=ROOT, text=True).strip()
    return {"object": reference, "blob": blob}


def object_bytes(pin):
    if git_object(pin["object"]) != pin:
        raise ValueError("continuation Git object changed")
    return subprocess.check_output(["git", "cat-file", "blob", pin["blob"]], cwd=ROOT)


def source_bytes(item, path_map, repair):
    source = Path(item["path"])
    if source.as_posix().endswith("/code/src/xgraph/webqsp_test.py"):
        return object_bytes(repair)
    if source.name == "snapshot.pth":
        return replace_paths(source.read_text(), path_map).encode()
    return None


def snapshot(path, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(Path(path).read_bytes())
    return {"original": study.identity(path), "snapshot": study.identity(destination)}


def authorized_history():
    return object_bytes(
        git_object(f"{REPAIR}:eval/webqsp-accounting-2026-10-31/snapshots/ledger.json")
    )


def generation_commits(parent):
    commits = {"code": parent["commit"]}
    if "resolved_config" in parent and parent["resolved_config"].get("gcr_source_commit"):
        commits["gcr-code"] = parent["resolved_config"]["gcr_source_commit"]
    if parent.get("gate", {}).get("wt_pc", {}).get("admit"):
        commits["endpoint-code"] = parent["gate"]["wt_pc"]["source_commit"]
    return commits


def create_draft(registration_path, parent_directory, directory, authority_path):
    """Bind an immutable child draft without preparing or launching test work."""
    commit = study.git_commit()
    directory, parent_directory = Path(directory).resolve(), Path(parent_directory).resolve()
    if directory.exists():
        raise FileExistsError(f"fresh continuation destination required: {directory}")
    if not directory.is_relative_to(ROOT / "target"):
        raise ValueError("continuation destination must be in this worktree's target")
    registration_path, authority_path = Path(registration_path), Path(authority_path)
    with authority_path.open("rb") as stream:
        authority_bytes = stream.read(OWNER_BYTES)
    registration = study.read_sealed(registration_path)
    parent = study.read_sealed(parent_directory / "execution.json")
    if (
        study.identity(registration_path)["sha256"] != PARENT_REGISTRATION_SHA256
        or study.identity(parent_directory / "execution.json")["sha256"] != PARENT_EXECUTION_SHA256
        or parent["registration"] != study.identity(registration_path)
        or parent["commit"] != registration["commit"]
        or digest(authority_bytes) != OWNER_SHA256
        or authority_path.resolve() != DEFAULT_AUTHORITY.resolve()
    ):
        raise ValueError("owner authority or parent seal identity changed")
    history_bytes = authorized_history()
    ledger = study.verify_gpu_ledger(
        registration, parent, parent_directory, ledger=json.loads(history_bytes)
    )
    if (
        ledger["spent_seconds"] != 13170.26331777824
        or ledger["buckets"]["core"]["spent_seconds"] != 13170.26331777824
        or ledger["buckets"]["published_baseline_subpanel"]["spent_seconds"] != 0
        or len(ledger["attempts"]) != 1
        or ledger["attempts"][0]["stage"] != "primary-gemma31-blind5"
        or ledger["attempts"][0]["elapsed_seconds"] != 1978.2711212302092
        or list((parent_directory / "receipts").glob("*.json"))
    ):
        raise ValueError("parent history differs from the authorized orphan/charge boundary")
    destination = directory / "execution"
    path_map = [
        [str(parent_directory), str(destination)],
        [
            str(parent_directory.parents[3] / ".venv/bin/python"),
            str(destination / "code/.venv/bin/python"),
        ],
    ]
    repair = git_object(f"{REPAIR}:src/xgraph/webqsp_test.py")
    child = copy.deepcopy(parent)
    for field in ("jobs", "arms", "adapted_launchers"):
        child[field] = replace_paths(parent[field], path_map)
    child["artifact_kind"] = "xgraph.webqsp_test_continuation"
    child["initial_gpu_seconds"] = ledger["spent_seconds"]
    source_inputs, links = [], {}
    for item in parent["inputs"]:
        relative = Path(item["path"]).relative_to(parent_directory)
        if relative.parts[0] in ("code", "gcr-code", "endpoint-code"):
            data = source_bytes(item, path_map, repair)
            target = destination / relative
            expected = (
                {"path": str(target), "bytes": len(data), "sha256": digest(data)}
                if data is not None
                else {**item, "path": str(target)}
            )
            source_inputs.append({"original": item, "expected": expected})
        else:
            links[str(destination / relative.parts[0])] = str(parent_directory / relative.parts[0])
    (directory / "history").mkdir(parents=True)
    authority_snapshot = directory / "history/authority.md"
    authority_snapshot.write_bytes(authority_bytes)
    history_snapshot = directory / "history/ledger.json"
    history_snapshot.write_bytes(history_bytes)
    child["continuation"] = {
        "name": "8a-continuation",
        "implementation_commit": commit,
        "repair": repair,
        "repair_commit": git_object(f"{REPAIR}^{{commit}}"),
        "path_map": path_map,
        "source_inputs": source_inputs,
        "generation_commits": generation_commits(parent),
        "interpreters": {
            code: str((parent_directory / code / ".venv/bin/python").resolve())
            for code in generation_commits(parent)
        },
        "readonly_links": links,
        "parent_initial_gpu_seconds": parent["initial_gpu_seconds"],
        "parent_charge": {k: v["spent_seconds"] for k, v in ledger["buckets"].items()},
        "status_prefix": "webqsp-test-cont",
        "science_sha256": science_sha256(parent),
        "authority": {
            "original": {
                "path": str(authority_path.resolve()),
                "bytes": OWNER_BYTES,
                "sha256": OWNER_SHA256,
            },
            "snapshot": study.identity(authority_snapshot),
        },
        "registration": snapshot(registration_path, directory / "history/registration.json"),
        "execution": snapshot(
            parent_directory / "execution.json", directory / "history/execution.json"
        ),
        "history": {
            "original": {
                "path": str(parent_directory / "ledger.json"),
                "bytes": len(history_bytes),
                "sha256": digest(history_bytes),
            },
            "snapshot": study.identity(history_snapshot),
            "git": git_object(f"{REPAIR}:eval/webqsp-accounting-2026-10-31/snapshots/ledger.json"),
        },
    }
    if science_sha256(child) != science_sha256(parent):
        raise ValueError("continuation changed normalized science/argv")
    study.write_sealed(directory / "draft.json", child)
    with (directory / "owner-dry-run.txt").open("x") as stream:
        for job in child["jobs"]:
            stream.write(f"# {study.stage_id(job)} {job['kind']}\n")
            if "launcher_wall_cap_seconds" in job:
                stream.write(
                    f"# funded launcher wall cap seconds: {job['launcher_wall_cap_seconds']}\n"
                )
            stream.write(
                shlex.join(["env", *[f"{k}={v}" for k, v in job["env"].items()], *job["command"]])
                + "\n"
            )
    study.write_sealed(
        directory / "handoff.json",
        {
            "draft": study.identity(directory / "draft.json"),
            "argv": study.identity(directory / "owner-dry-run.txt"),
            "execution_sha256": seal_sha256(prospective_execution(child, directory)),
        },
    )
    return child


def verify_draft(directory, *, full=False):
    """Check the draft and, on owner request, every inherited input and model pin."""
    directory = Path(directory).resolve()
    handoff = study.read_sealed(directory / "handoff.json")
    for field, path in (("draft", "draft.json"), ("argv", "owner-dry-run.txt")):
        if handoff[field] != study.identity(directory / path):
            raise ValueError("reviewed draft or full argv preview changed")
    draft = study.read_sealed(directory / "draft.json")
    c = draft["continuation"]
    if study.git_commit() != c["implementation_commit"]:
        raise ValueError("continuation implementation commit changed")
    for name in ("authority", "registration", "execution", "history"):
        study.verify_identity(c[name]["snapshot"])
    authority = c["authority"]["original"]
    if (
        Path(authority["path"]).resolve() != DEFAULT_AUTHORITY.resolve()
        or authority["bytes"] != OWNER_BYTES
    ):
        raise ValueError("owner authority path or prefix length changed")
    with Path(authority["path"]).open("rb") as stream:
        if (
            digest(stream.read(authority["bytes"])) != OWNER_SHA256
            or authority["sha256"] != OWNER_SHA256
        ):
            raise ValueError("owner authority prefix changed")
    parent = json.loads(Path(c["execution"]["snapshot"]["path"]).read_text())
    registration = json.loads(Path(c["registration"]["snapshot"]["path"]).read_text())
    if (
        c["registration"]["original"]["sha256"] != PARENT_REGISTRATION_SHA256
        or c["execution"]["original"]["sha256"] != PARENT_EXECUTION_SHA256
        or c["registration"]["snapshot"]["sha256"] != PARENT_REGISTRATION_SHA256
        or c["execution"]["snapshot"]["sha256"] != PARENT_EXECUTION_SHA256
        or science_sha256(draft) != science_sha256(parent)
        or c["science_sha256"] != science_sha256(parent)
        or c["name"] != "8a-continuation"
        or c["repair"] != git_object(f"{REPAIR}:src/xgraph/webqsp_test.py")
        or c["repair_commit"] != git_object(f"{REPAIR}^{{commit}}")
        or c["history"]["git"]
        != git_object(f"{REPAIR}:eval/webqsp-accounting-2026-10-31/snapshots/ledger.json")
        or c["history"]["snapshot"]["sha256"] != digest(authorized_history())
        or c["status_prefix"] != "webqsp-test-cont"
        or c["generation_commits"] != generation_commits(parent)
    ):
        raise ValueError("continuation changed parent identities or normalized science/argv")
    original = Path(c["execution"]["original"]["path"]).parent
    destination = directory / "execution"
    expected_map = [
        [str(original), str(destination)],
        [str(original.parents[3] / ".venv/bin/python"), str(destination / "code/.venv/bin/python")],
    ]
    if c["path_map"] != expected_map:
        raise ValueError("continuation destinations changed")
    if c["interpreters"] != {
        code: str((original / code / ".venv/bin/python").resolve())
        for code in generation_commits(parent)
    }:
        raise ValueError("frozen interpreter changed")
    expected_sources, expected_links = [], {}
    for item in parent["inputs"]:
        relative = Path(item["path"]).relative_to(original)
        if relative.parts[0] in ("code", "gcr-code", "endpoint-code"):
            data = source_bytes(item, expected_map, c["repair"])
            target = destination / relative
            expected = (
                {"path": str(target), "bytes": len(data), "sha256": digest(data)}
                if data is not None
                else {**item, "path": str(target)}
            )
            expected_sources.append({"original": item, "expected": expected})
        else:
            expected_links[str(destination / relative.parts[0])] = str(original / relative.parts[0])
    if c["source_inputs"] != expected_sources or c["readonly_links"] != expected_links:
        raise ValueError("continuation source/input manifest changed")
    if handoff.get("execution_sha256") != seal_sha256(prospective_execution(draft, directory)):
        raise ValueError("reviewed final child seal identity changed")
    history = study.verify_gpu_ledger(registration, parent, directory / "history")
    charges = {k: v["spent_seconds"] for k, v in history["buckets"].items()}
    if c["parent_charge"] != charges or draft["initial_gpu_seconds"] != history["spent_seconds"]:
        raise ValueError("continuation parent charge changed")
    if full:
        # These lists are exhaustive, never just their leading entry. Do not call
        # ordinary prepare/size/select_ids or inspect a generation output here.
        for item in [
            *registration["inputs"],
            *parent["inputs"],
            *parent["adapted_launchers"],
            *[
                parent[k]
                for k in (
                    "gate_identity",
                    "timing_identity",
                    "pilot_plan_identity",
                    "baseline_timing_identity",
                )
                if parent.get(k)
            ],
        ]:
            study.verify_identity(item)
        for saved in registration["model_inventory"].values():
            if study.model_inventory(saved["path"]) != saved:
                raise ValueError("frozen model inventory changed")
        for name in ("registration", "execution", "history"):
            # Rewritable cross-worktree metadata is a diagnostic only; the saved
            # bytes and inherited frozen pins remain authoritative.
            try:
                changed = study.identity(c[name]["original"]["path"]) != c[name]["original"]
            except OSError:
                changed = True
            if changed:
                warnings.warn(
                    f"original {name} metadata drift; immutable snapshot retained", stacklevel=2
                )
        if list((original / "receipts").glob("*.json")):
            raise ValueError("parent acquired receipts outside the authorized history boundary")
    return registration, draft


def register_draft(directory):
    """Materialize the reviewed child after full owner checks; never launch a stage."""
    import fcntl

    directory = Path(directory).resolve()
    with study.Path("/tmp/xgraph-webqsp-queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registration, child = verify_draft(directory, full=True)
        c = child["continuation"]
        destination = directory / "execution"
        if destination.exists():
            raise FileExistsError(
                "continuation execution exists; preserve its seal and partial files"
            )
        size = sum(i["expected"]["bytes"] for i in c["source_inputs"])
        if size > 2 * 1024**3 and shutil.disk_usage(directory).free - size < 40 * 1024**3:
            raise ValueError("source materialization would leave less than 40 GiB free")
        destination.mkdir()
        for target, source in c["readonly_links"].items():
            Path(target).symlink_to(source, target_is_directory=True)
        for item in c["source_inputs"]:
            target = Path(item["expected"]["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            data = source_bytes(item["original"], c["path_map"], c["repair"])
            if data is None:
                shutil.copyfile(item["original"]["path"], target)
            else:
                target.write_bytes(data)
            if target.suffix == ".sh":
                target.chmod(0o755)
            study.verify_identity(item["expected"])
        original = Path(c["execution"]["original"]["path"]).parent
        for code in ("code", "gcr-code", "endpoint-code"):
            source, target = original / code, destination / code
            if not target.exists():
                continue
            if not (source / ".git").exists():
                raise ValueError("original source Git objects unavailable")
            interpreter = target / ".venv/bin/python"
            interpreter.parent.mkdir(parents=True, exist_ok=True)
            for name in ("python", "python3"):
                (interpreter.parent / name).symlink_to(c["interpreters"][code])
            if (source / ".git").exists():
                subprocess.run(["git", "init", "--quiet", str(target)], check=True)
                common = subprocess.check_output(
                    ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                    cwd=source,
                    text=True,
                ).strip()
                (target / ".git/objects/info/alternates").write_text(
                    str(Path(common) / "objects") + "\n"
                )
                commit = c["generation_commits"][code]
                subprocess.run(["git", "update-ref", "HEAD", commit], cwd=target, check=True)
                subprocess.run(["git", "read-tree", commit], cwd=target, check=True)
        c["draft"] = study.identity(directory / "draft.json")
        study.write_sealed(destination / "execution.json", child)
        if (
            study.identity(destination / "execution.json")["sha256"]
            != study.read_sealed(directory / "handoff.json")["execution_sha256"]
        ):
            raise ValueError("final child seal differs from reviewed handoff")
        study.write_json(
            destination / "ledger.json",
            {
                "spent_seconds": child["initial_gpu_seconds"],
                "active": None,
                "attempts": [],
                "history": c["history"]["snapshot"],
                "buckets": {
                    name: {"cap_seconds": cap, "spent_seconds": c["parent_charge"][name]}
                    for name, cap in study.gpu_budget_caps(registration).items()
                },
            },
        )
        verify_continuation(child["registration"]["path"], destination)
        return child


def verify_continuation(registration_path, directory):
    directory = Path(directory).resolve()
    registration, draft = verify_draft(directory.parent, full=True)
    child = study.read_sealed(directory / "execution.json")
    c = child["continuation"]
    expected = prospective_execution(draft, directory.parent)
    if (
        child != expected
        or str(registration_path) != child["registration"]["path"]
        or study.identity(directory / "execution.json")["sha256"]
        != study.read_sealed(directory.parent / "handoff.json")["execution_sha256"]
    ):
        raise ValueError("continuation changed from its reviewed immutable draft")
    for item in c["source_inputs"]:
        study.verify_identity(item["expected"])
    for target, source in c["readonly_links"].items():
        if not Path(target).is_symlink() or Path(target).resolve() != Path(source).resolve():
            raise ValueError("read-only frozen input link changed")
    for code, commit in c["generation_commits"].items():
        source = directory / code
        if (
            not (source / ".git").exists()
            or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
            != commit
        ):
            raise ValueError("original generation Git identity changed")
        for name in ("python", "python3"):
            interpreter = source / ".venv/bin" / name
            if (
                not interpreter.is_symlink()
                or str(interpreter.resolve()) != c["interpreters"][code]
            ):
                raise ValueError("continuation interpreter changed")
    study.verify_gpu_ledger(registration, child, directory)
    return registration, child


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--authority", type=Path, default=DEFAULT_AUTHORITY)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("draft", "dry-run", "register", "check", "run", "score"):
        modes.add_argument(f"--{mode}", action="store_true")
    args = parser.parse_args(argv)
    if args.draft:
        create_draft(
            args.parent / "registration.json", args.parent / "execution", args.out, args.authority
        )
    elif args.dry_run:
        verify_draft(args.out)
        print((args.out / "owner-dry-run.txt").read_text(), end="")
    elif args.register:
        register_draft(args.out)
    else:
        if args.score:
            # The embargo check precedes output, receipt and ledger access.
            draft = study.read_sealed(args.out / "draft.json")
            study.enforce_embargo(
                study.read_sealed(draft["registration"]["path"]), args.out / "execution", None
            )
        if not (args.out / "execution/execution.json").exists() and args.check:
            verify_draft(args.out, full=True)
        elif args.run:
            draft = study.read_sealed(args.out / "draft.json")
            study.run_owner(draft["registration"]["path"], args.out / "execution")
        else:
            draft = study.read_sealed(args.out / "draft.json")
            registration, execution = verify_continuation(
                draft["registration"]["path"], args.out / "execution"
            )
            if args.score:
                study.score_execution(registration, execution, args.out / "execution")
    print(
        "# continuation "
        + (
            "draft saved; no live input qualification or launch"
            if args.draft
            else "command complete"
        )
    )


if __name__ == "__main__":
    main()
