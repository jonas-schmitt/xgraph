"""Gold-free CWQ validation pilot planning and explicit GPU-owner execution."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
import re
import shlex
import subprocess
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from xgraph.cwq import MODELS, ROOT, SOURCE_SPEC, budget, source_manifest
from xgraph.provenance import provenance, sha256_file
from xgraph.webqsp import _json, _read_jsonl, verify_catalog
from xgraph.webqsp_test import (
    atomic_json,
    git_commit,
    identity,
    model_inventory,
    read_sealed,
    require_complete,
    select_ids,
    stage_id,
    unique_ids,
    verify_identity,
    write_lines,
    write_sealed,
)
from xgraph.webqsp_test import (
    execute_gpu_stage as execute_reserved_stage,
)

REFERENCES = {"qwen38": "agent-strong-8", "gemma31": "agent-feedback-8"}
LOG = Path("/home/jonas/git/xgraph/target/queue/cwq-pilot.log")
BUILD_CONSTRAINTS = ROOT / "scripts/cwq/build-constraints.txt"


def cleanup_owned_container(state, env):
    """Remove only the launcher's daemon-issued ID, then prove it absent."""
    if not (state / "launching").exists():
        return {"confirmed_absent": True, "reason": "container was never launched"}
    try:
        cid = (state / "id").read_text().strip()
    except OSError as error:
        return {"confirmed_absent": False, "error": str(error)}
    if not re.fullmatch(r"[0-9a-f]{64}", cid):
        return {"confirmed_absent": False, "error": "missing or invalid owned container ID"}
    result = {"confirmed_absent": False, "container_id": cid}
    try:
        # A failed/timed-out removal is harmless only if the daemon subsequently
        # confirms absence. Failed inspection never means 'not found'.
        with suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run(
                ["docker", "rm", "--force", cid],
                env=env,
                capture_output=True,
                text=True,
                timeout=5,
            )
        inspected = subprocess.run(
            [
                "docker",
                "container",
                "ls",
                "--all",
                "--no-trunc",
                "--filter",
                f"id={cid}",
                "--format",
                "{{.ID}}",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
        )
        result["confirmed_absent"] = inspected.returncode == 0 and not inspected.stdout.strip()
        if not result["confirmed_absent"]:
            result["error"] = (inspected.stderr or inspected.stdout or "inspection failed")[:512]
    except (OSError, subprocess.TimeoutExpired) as error:
        result["error"] = str(error)
    return result


def execute_gpu_stage(job, ledger_path, cap_seconds, log_path):
    """Bind CWQ's launcher cleanup to its reservation without changing the sealed job."""
    state = Path(str(log_path) + ".container")
    state.mkdir()
    env = {**os.environ, **job["env"]}
    return execute_reserved_stage(
        job,
        ledger_path,
        cap_seconds,
        log_path,
        launcher_env={
            "XG_CONTAINER_STATE": str(state.resolve()),
            "XG_GPU_EVENTS": str(Path(str(log_path) + ".gpu-events.jsonl").resolve()),
        },
        cleanup=lambda: cleanup_owned_container(state, env),
        occupancy_accounting=True,
    )


def pilot_plan(data: Path, out: Path, models=None) -> dict:
    """Reuse the complete transfer recipes, with an explicit CWQ population."""
    from xgraph.webqsp_catalog import catalog_plan

    data, out = Path(data).resolve(), Path(out).resolve()
    models = MODELS if models is None else models
    if set(models) != set(REFERENCES):
        raise ValueError("CWQ pilot requires both large model families")
    metadata = json.loads((data / "manifest.json").read_text())
    if (metadata.get("dataset"), metadata.get("split"), metadata.get("gold_loaded")) != (
        "cwq",
        "validation",
        False,
    ):
        raise ValueError("CWQ pilot requires gold-free CWQ validation inputs")
    panel = json.loads((data / "questions.json").read_text())
    ids = select_ids([q["id"] for q in panel], 128, 17)
    original = catalog_plan(
        data,
        out,
        models,
        arms=["W", "WT", "B2", "agent-strong-8"],
        timing_root=out / "no-timing-proxy",
    )
    jobs = []
    for old in original["jobs"]:
        if (
            old["name"].endswith(("-score", "-declared-score"))
            or old["name"] == "agent-strong-8-sets"
        ):
            continue
        job = copy.deepcopy(old)
        argv = job["command"]
        family = job["family"]
        if job["kind"] == "gpu":
            at = next(i for i, v in enumerate(argv) if v.endswith("/scripts/generate.sh"))
            job["requests"], job["outputs"] = argv[at + 2], [argv[at + 4]]
            for flag, value in (
                ("--max-model-len", "16384"),
                ("--kv-cache-gib", "16" if family == "gemma31" else "8"),
            ):
                if flag in argv:
                    argv[argv.index(flag) + 1] = value
                else:
                    argv.extend([flag, value])
            if family == "gemma31" and job["name"] == "agent-strong-8":
                job["name"] = "agent-feedback-8"
                argv[at + 4] = str(out / family / "agent-feedback-8.jsonl")
                job["outputs"] = [argv[at + 4]]
                argv.append("--agent-feedback")
            job["questions"] = 128
            job["env"].update(
                UV_CACHE_DIR=str(ROOT / "target/cwq/uv-cache"),
                UV_BUILD_CONSTRAINT=str(BUILD_CONSTRAINTS),
                UV_LINK_MODE="hardlink",
                XG_NEED_GIB="75" if family == "gemma31" else "60",
                XG_GPU_IDLE="1",
                XG_AVAILABLE_WAIT="0",
                XG_HEADROOM_WAIT="0",
                XG_START_ATTEMPTS="1",
                XG_RUN_SECONDS="7200",
                XG_START_RETRY_SLEEP="60",
                XG_PAGECACHE_ROOTS=f"{data}:{models[family]}",
            )
        else:
            argv.insert(argv.index("-m"), "-I")
            job["outputs"] = [argv[argv.index("--out") + 1]]
        for kind in ("S", "blind5", "menuobs", "agent"):
            before = str(data / f"requests-{kind}.jsonl")
            after = str(out / "pilot" / f"requests-{kind}.jsonl")
            job["command"] = [after if a == before else a for a in job["command"]]
            if job.get("requests") == before:
                job["requests"] = after
        job["scope"] = "cwq-validation"
        job["wall_timeout_seconds"] = 24 * 3600
        jobs.append(job)
    arms = [
        {
            "family": family,
            "arm": arm,
            "output": str(
                out / family / (f"{arm}-sets.jsonl" if arm in {"W", "WT", "B2"} else f"{arm}.jsonl")
            ),
        }
        for family in REFERENCES
        for arm in ("WT", "W", "B2", REFERENCES[family])
    ]
    gpu = [j for j in jobs if j["kind"] == "gpu"]
    return {
        "dataset": "cwq",
        "split": "validation",
        "gold_loaded": False,
        "data": str(data),
        "out": str(out),
        "models": {k: str(v) for k, v in models.items()},
        "ids": ids,
        "seed": 17,
        "questions": 128,
        "budget_seconds": 10800,
        "addition_budget": {
            "allocation_gpu_hours": {
                "menu_control": 4,
                "new_operator": 6,
                "fresh_worlds": 2,
                "cwq_pilot": 3,
            },
            "total_cap_gpu_hours": 16,
            "unspent_gpu_hours": 1,
            "amendment": "one unspent incremental hour added prospectively; no outputs",
        },
        "jobs": jobs,
        "arms": arms,
        "subgraphrag": {
            "top_k": 100,
            "status": "MISSING_VALIDATION_RETRIEVAL",
            "release": source_manifest()["subgraphrag"],
            "rule": "test-only release cannot supply validation retrieval; no proxy",
        },
        "estimate": {
            "measured": False,
            "low_gpu_hours": sum(128 * j["assumed_seconds_per_question"][0] + 90 for j in gpu)
            / 3600,
            "high_gpu_hours": sum(128 * j["assumed_seconds_per_question"][1] + 180 for j in gpu)
            / 3600,
            "basis": "unmeasured transfer stage envelopes plus 90-180s startup per launch; "
            "shared W charged once",
            "cap_gpu_hours": 3,
            "rule": "no cohort shrink; watchdog interruption leaves INCOMPLETE",
        },
        "transfer_overrides": [
            "16384 context for every stage; Gemma 16 GiB KV, Qwen 8 GiB KV",
            "synthetic Freebase demonstrations and complete path-set delivery",
            "Qwen plain agent-8, Gemma feedback agent-8; own agent answer retained",
        ],
        "budget_accounting": "container_occupancy",
        "timing_scope": "serial stage wall recorded separately; budget charges container "
        "occupancy including startup and retries; CPU transformations charge zero occupancy",
    }


def prepare(data: Path, out: Path) -> dict:
    commit = git_commit(ROOT)
    catalog = verify_catalog(data)
    if catalog.get("dataset") != "cwq":
        raise ValueError("CWQ pilot requires its own prepared catalog")
    if out.exists():
        raise ValueError("CWQ pilot output must be a new directory")
    plan = pilot_plan(data, out)
    out.mkdir(parents=True)
    ids = set(plan["ids"])
    pilot = out / "pilot"
    pilot.mkdir()
    panel = json.loads((data / "questions.json").read_text())
    _json(pilot / "questions.json", [q for q in panel if q["id"] in ids])
    inputs = [SOURCE_SPEC, BUILD_CONSTRAINTS, data / "catalog.json", pilot / "questions.json"]
    for kind in ("S", "blind5", "menuobs", "agent"):
        path = pilot / f"requests-{kind}.jsonl"
        rows = [q for q in _read_jsonl(data / path.name) if q["id"] in ids]
        if set(unique_ids(r["id"] for r in rows)) != ids:
            raise ValueError("pilot request IDs do not match the hash sample")
        write_lines(path, rows)
        inputs.append(path)
    plan.update(
        commit=commit,
        files=[identity(p) for p in inputs],
        model_inventory={k: model_inventory(v) for k, v in MODELS.items()},
        provenance=provenance(inputs),
    )
    write_sealed(out / "pilot.json", plan)
    _json(out / "ledger.json", {"spent_seconds": 0, "active": None, "attempts": []})
    script = out / "cwq.sh"
    script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + "exec "
        + shlex.join([str(ROOT / "scripts/cwq/pilot.sh"), str(out / "pilot.json")])
        + ' "${1:---dry-run}"\n'
    )
    script.chmod(0o755)
    return plan


def verify(plan_path: Path) -> dict:
    plan = read_sealed(plan_path)
    if git_commit(ROOT) != plan["commit"]:
        raise ValueError("CWQ pilot commit changed")
    for pin in plan["files"]:
        verify_identity(pin)
    verify_catalog(Path(plan["data"]))
    for family, path in plan["models"].items():
        if model_inventory(path) != plan["model_inventory"][family]:
            raise ValueError(f"CWQ model inventory changed: {family}")
    return plan


def preview(plan):
    for job in plan["jobs"]:
        print(f"# {stage_id(job)}; wall bound {job['wall_timeout_seconds']} seconds")
        runtime_env = {}
        if job["kind"] == "gpu":
            log = Path(plan["out"]) / job["family"] / f"{job['name']}.log"
            runtime_env = {
                "XG_CONTAINER_STATE": str(log) + ".container",
                "XG_GPU_EVENTS": str(log) + ".gpu-events.jsonl",
            }
            headroom = {**job["env"], "XG_AVAILABLE_WAIT": "43200", "XG_HEADROOM_WAIT": "900"}
            print(
                shlex.join(
                    [
                        "env",
                        *[f"{k}={v}" for k, v in headroom.items()],
                        str(ROOT / "scripts/gpu-headroom.sh"),
                        job["env"]["XG_NEED_GIB"],
                    ]
                )
            )
        print(
            shlex.join(
                [
                    "env",
                    *[f"{k}={v}" for k, v in {**job["env"], **runtime_env}.items()],
                    *job["command"],
                ]
            )
        )


def receipt_path(plan, job):
    return Path(plan["out"]) / "receipts" / (stage_id(job) + ".json")


def output_paths(job):
    paths = list(map(Path, job["outputs"]))
    if job["kind"] == "gpu":
        paths.extend(Path(str(paths[0]) + s) for s in (".summary.json", ".provenance.json"))
    return paths


def single_call_records(records):
    """Audit native records using the registered single-call stage contract."""
    for record in records:
        if (
            not isinstance(record.get("text"), str)
            or not isinstance(record.get("token_ids"), list)
            or any(type(t) is not int or t < 0 for t in record["token_ids"])
            or record.get("model_calls", 1) != 1
        ):
            raise ValueError("invalid native single-call generation record")
    return [{**r, "model_calls": 1} for r in records]


def completed(plan, job, seal):
    path = receipt_path(plan, job)
    if not path.exists():
        if any(p.exists() or p.is_symlink() for p in output_paths(job)):
            raise ValueError(f"partial/unreceipted outputs: {stage_id(job)}")
        return None
    saved = read_sealed(path)
    if saved["pilot_sha256"] != seal or saved["job"] != job or saved["returncode"] != 0:
        raise ValueError("CWQ stage receipt changed or failed")
    if [p["path"] for p in saved["outputs"]] != [str(p.resolve()) for p in output_paths(job)]:
        raise ValueError("CWQ stage receipt output inventory changed")
    for pin in saved["outputs"]:
        verify_identity(pin)
    if job["kind"] == "gpu":
        events = Path(plan["out"]) / job["family"] / f"{job['name']}.log.gpu-events.jsonl"
        if saved.get("gpu_events", {}).get("path") != str(events.resolve()):
            raise ValueError("CWQ stage receipt GPU event inventory changed")
        verify_identity(saved["gpu_events"])
    return saved


def verify_ledger(plan):
    ledger = json.loads((Path(plan["out"]) / "ledger.json").read_text())
    stages = {stage_id(j): j for j in plan["jobs"]}
    charges = [ledger["spent_seconds"], *[a["elapsed_seconds"] for a in ledger["attempts"]]]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in charges):
        raise ValueError("CWQ ledger charges must be finite and nonnegative")
    if (
        ledger["active"] is not None
        or not math.isclose(ledger["spent_seconds"], sum(charges[1:]), abs_tol=1e-6)
        or ledger["spent_seconds"] > plan["budget_seconds"]
        or any(a["stage"] not in stages for a in ledger["attempts"])
    ):
        raise ValueError("CWQ ledger is unreconciled, inconsistent or over cap")
    for stage, job in stages.items():
        path = receipt_path(plan, job)
        if not path.exists():
            continue
        receipt = read_sealed(path)
        successful = [
            a
            for a in ledger["attempts"]
            if a["stage"] == stage and a["returncode"] == 0 and not a["timeout"]
        ]
        if len(successful) != 1 or not math.isclose(
            successful[0]["elapsed_seconds"], receipt["timing"]["gpu_seconds"], abs_tol=1e-6
        ):
            raise ValueError(f"CWQ ledger does not match completed receipt: {stage}")
    return ledger


def run_owner(plan_path: Path, *, status_log=None):
    # Both stage kinds have a wall watchdog; only owned containers consume the cap.
    from xgraph.webqsp_test import execute_gpu_stage as execute_reserved_stage

    status_log = Path(os.environ.get("CWQ_STATUS_LOG", LOG) if status_log is None else status_log)
    plan = verify(plan_path)
    out, seal = Path(plan["out"]), sha256_file(plan_path)
    with (out / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_ledger(plan)
        rows = {}
        if (out / "timings.json").exists():
            previous = json.loads((out / "timings.json").read_text())
            if previous["pilot_sha256"] != seal:
                raise ValueError("CWQ timings pilot seal changed")
            rows = {(r["family"], r["name"]): r for r in previous["per_arm"]}
        status_log.parent.mkdir(parents=True, exist_ok=True)

        def status(job, message):
            with status_log.open("a") as stream:
                stream.write(
                    f"cwq-pilot {job['family']} {job['name']} {message} "
                    f"{datetime.now(UTC).isoformat()}\n"
                )

        def timings(complete=False):
            ledger = json.loads((out / "ledger.json").read_text())
            atomic_json(
                out / "timings.json",
                {
                    "complete": complete,
                    "status": "COMPLETE_CORE_PILOT" if complete else "INCOMPLETE",
                    "dataset": "cwq",
                    "split": "validation",
                    "pilot_sha256": seal,
                    "budget_seconds": plan["budget_seconds"],
                    "gpu_seconds": ledger["spent_seconds"],
                    "per_arm": [
                        rows[j["family"], j["name"]]
                        for j in plan["jobs"]
                        if (j["family"], j["name"]) in rows
                    ],
                    "subgraphrag": plan["subgraphrag"],
                    "timing_scope": plan["timing_scope"],
                    "budget_accounting": plan["budget_accounting"],
                    "provenance": plan["provenance"],
                },
            )

        timings()
        for job in plan["jobs"]:
            saved = completed(plan, job, seal)
            if saved:
                rows[job["family"], job["name"]] = saved["timing"]
                status(job, "skip-existing")
                continue
            folder = out / job["family"]
            folder.mkdir(exist_ok=True)
            log = folder / f"{job['name']}.log"
            if log.exists():
                raise ValueError(f"previous attempt preserved: {log}; reconcile before retry")
            status(job, "start")
            wall_started = time.monotonic()
            deadline = wall_started + job["wall_timeout_seconds"]
            env = {k: v for k, v in os.environ.items() if not k.startswith(("XG_", "XGRAPH_"))}
            env.update(job["env"])
            if job["kind"] == "gpu":
                # Both idle/headroom checks consume wall time, never GPU allowance.
                try:
                    with (folder / f"{job['name']}.headroom.log").open("a") as stream:
                        headroom_env = {
                            **env,
                            "XG_AVAILABLE_WAIT": "43200",
                            "XG_HEADROOM_WAIT": "900",
                        }
                        while subprocess.run(
                            [str(ROOT / "scripts/gpu-headroom.sh"), env["XG_NEED_GIB"]],
                            env=headroom_env,
                            cwd=ROOT,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            timeout=max(0.001, deadline - time.monotonic()),
                        ).returncode:
                            if time.monotonic() >= deadline:
                                raise subprocess.TimeoutExpired(
                                    job["command"], job["wall_timeout_seconds"]
                                )
                            status(job, "waiting-headroom")
                            time.sleep(min(60, max(0, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    code, elapsed = 124, 0
                else:
                    verify(plan_path)
                    code, elapsed = execute_gpu_stage(
                        {**job, "wall_timeout_seconds": max(0.001, deadline - time.monotonic())},
                        out / "ledger.json",
                        plan["budget_seconds"],
                        log,
                    )
            else:
                code, elapsed = execute_reserved_stage(
                    job,
                    out / "ledger.json",
                    plan["budget_seconds"],
                    log,
                    occupancy_accounting=True,
                )
            row = {
                "family": job["family"],
                "name": job["name"],
                "kind": job["kind"],
                "questions": 128,
                "wall_seconds": time.monotonic() - wall_started,
                "gpu_seconds": elapsed,
                "returncode": code,
            }
            rows[job["family"], job["name"]] = row
            if code:
                status(job, f"exit {code}")
                timings()
                raise RuntimeError(f"CWQ pilot INCOMPLETE: {stage_id(job)} exited {code}")
            try:
                for p in job["outputs"]:
                    records = _read_jsonl(Path(p))
                    if set(unique_ids(r["id"] for r in records)) != set(plan["ids"]):
                        raise ValueError("incomplete CWQ stage IDs")
                    if job["kind"] == "gpu":
                        # Keep native bytes and use the sealed call contract;
                        # agent rounds retain their explicit accounting.
                        native = (
                            job["command"][job["command"].index(job["requests"]) + 1]
                            == "constrained"
                        )
                        if native:
                            records = single_call_records(records)
                        require_complete(records, plan["ids"])
                for p in output_paths(job):
                    if not p.is_file():
                        raise ValueError(f"missing CWQ stage artifact: {p}")
            except (OSError, ValueError, KeyError, TypeError) as error:
                row.update(validation="FAILED", error=str(error))
                status(job, "invalid-output")
                timings()
                raise
            write_sealed(
                receipt_path(plan, job),
                {
                    "pilot_sha256": seal,
                    "job": job,
                    "outputs": [identity(p) for p in output_paths(job)],
                    "returncode": code,
                    "timing": row,
                    **(
                        {"gpu_events": identity(Path(str(log) + ".gpu-events.jsonl"))}
                        if job["kind"] == "gpu"
                        else {}
                    ),
                },
            )
            budget(
                Path(plan["data"]),
                out,
                Path(json.loads((Path(plan["data"]) / "manifest.json").read_text())["source"]),
            )
            status(job, "exit 0")
            timings()
        timings(True)


def score_pilot(plan_path: Path, out: Path) -> dict:
    from xgraph.cwq import validation_gold
    from xgraph.cwq_test import compare_complete, population_audits

    plan = verify(plan_path)
    ledger = verify_ledger(plan)
    seal = sha256_file(plan_path)
    timings = json.loads((Path(plan["out"]) / "timings.json").read_text())
    if timings.get("complete") is not True or timings["pilot_sha256"] != seal:
        raise ValueError("INCOMPLETE pilot; validation scoring requires all core stages")
    expected = []
    for job in plan["jobs"]:
        receipt = completed(plan, job, seal)
        if receipt is None:
            raise ValueError("INCOMPLETE pilot stage receipt")
        expected.append(receipt["timing"])
    if (
        timings["per_arm"] != expected
        or timings["gpu_seconds"] != ledger["spent_seconds"]
        or timings["budget_seconds"] != plan["budget_seconds"]
    ):
        raise ValueError("CWQ timings do not match stage receipts and ledger")
    rows = {}
    for arm in plan["arms"]:
        records = _read_jsonl(Path(arm["output"]))
        if arm["arm"] == "W":
            records = single_call_records(records)
        require_complete(records, plan["ids"])
        rows[arm["family"], arm["arm"]] = records
    metadata = json.loads((Path(plan["data"]) / "manifest.json").read_text())
    source = Path(metadata["source"])
    audit = population_audits(source, plan["ids"])["validation_pilot"]
    clusters = audit["assignments"]
    panel = validation_gold(source, plan["ids"])
    result = compare_complete(
        rows,
        panel,
        clusters,
        sensitivity_clusters=audit["sensitivity"]["assignments"],
    )
    result.update(
        scope="validation pilot only; not test admission or confirmatory evidence",
        subgraphrag=plan["subgraphrag"],
        pilot_sha256=seal,
        cluster_audit=audit,
        timings=identity(Path(plan["out"]) / "timings.json"),
        provenance=provenance([plan_path, SOURCE_SPEC]),
    )
    _json(out, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    for name in ("preview", "run", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--plan", type=Path, required=True)
    p = sub.add_parser("score")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.data, args.out)
    elif args.command == "preview":
        preview(read_sealed(args.plan))
    elif args.command == "verify":
        verify(args.plan)
    elif args.command == "score":
        score_pilot(args.plan, args.out)
    else:
        run_owner(args.plan)


if __name__ == "__main__":
    main()
