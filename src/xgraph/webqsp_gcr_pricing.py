"""Non-test WebQSP KG-Trie census and bounded owner-run native GCR pricing."""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import time
from collections import Counter
from pathlib import Path

from xgraph.webqsp_test import (
    FAMILIES,
    ROOT,
    atomic_json,
    baseline_price_contract,
    build_jobs,
    execute_gpu_stage,
    git_commit,
    identity,
    positive,
    read_sealed,
    select_ids,
    snapshot_code,
    unique_ids,
    verify_identity,
    write_sealed,
)

NATIVE_GCR_COMMIT = "9e4f8cb"
VALIDATION = Path(
    "/home/jonas/git/xgraph/.claude/worktrees/codex-webqsp2/target/webqsp/frozen-data/validation"
)

PUBLISHED_COMPARISON = {
    "source": "https://arxiv.org/html/2410.13080v2#S5.T1",
    "table": 1,
    "label": "GCR published WebQSP result; cited, non-paired comparison",
    "paired": False,
    "rows": [
        {"planner": "Llama-3.1-8B", "reader": "ChatGPT", "Hit": 92.6, "F1": 73.2},
        {"planner": "Llama-3.1-8B", "reader": "GPT-4o-mini", "Hit": 92.2, "F1": 74.1},
    ],
    "limits": "Published reader/checkpoints and population differ; no local delta, paired "
    "interval, significance or runtime claim; not a Qwen/Gemma result",
}


def native_plan(registration, data, out):
    """Substitute only non-test inputs/output paths in the sealed baseline recipe."""
    data, out = Path(data).resolve(), Path(out).resolve()
    spec = registration["gcr_native_pricing"]
    if (
        spec["scope"] != "validation"
        or spec["ids"] != select_ids(registration["pilot"]["cohort_ids"], 2)
        or spec["semantics"] != "hf-group"
    ):
        raise ValueError(
            "native WebQSP price must use the fixed seeded validation IDs and hf-group"
        )
    config = registration["config"]
    jobs, _ = build_jobs(config)
    jobs = [j for j in jobs if j["kind"] == "gpu" and j["name"] == "gcr-native"]
    base = Path(config["out"]) / "execution"
    for job in jobs:
        replacements = {
            str(base / "gcr-code"): str(out / "gcr-code"),
            str(base / "baseline-inputs"): str(data),
            str(base / "baseline-runs"): str(out / "runs"),
        }
        for before, after in replacements.items():
            job.update(json.loads(json.dumps(job).replace(before, after)))
        at = next(i for i, s in enumerate(job["command"]) if s.endswith("/scripts/generate.sh"))
        job["command"] = ["nice", "-n", "19", "taskset", "-c", "4-7", *job["command"][at:]]
        at = 6
        job["requests"] = str(out / "requests.jsonl")
        job["command"][at + 2] = job["requests"]
        job["wall_allowance_seconds"] = 3300
        job["env"].update(
            XG_AVAILABLE_WAIT="0",
            XG_HEADROOM_WAIT="0",
            XG_START_ATTEMPTS="1",
            XG_CONTAINER_NAME=f"webqsp-gcr-price-{job['family']}",
            XG_CONTAINER_OWNER="webqsp-amendment-6",
            XG_GPU_EVENTS=str(out / f"{job['family']}-events.jsonl"),
            TMPDIR=str(ROOT / "target/tmp"),
            UV_CACHE_DIR=str(ROOT / "target/gcr-pricing-uv-cache"),
            PIP_CACHE_DIR=str(ROOT / "target/gcr-pricing-pip-cache"),
            PYTHONDONTWRITEBYTECODE="1",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
        )
    return {
        "artifact_kind": "xgraph.webqsp_gcr_pricing",
        "version": 1,
        "commit": registration["commit"],
        "producer_commit": config["gcr_source_commit"],
        "producer_root": config.get("gcr_source_root", str(ROOT.parent / "codex-gcr2")),
        "recipe_sha256": baseline_price_contract(registration, "gcr"),
        "scope": "validation",
        "ids": spec["ids"],
        "questions": 2,
        "seed": 17,
        "semantics": "hf-group",
        "cap_seconds": 7200,
        "cleanup_seconds": 30,
        "stage_wall_seconds": 3300,
        "out": str(out),
        "data": str(data),
        "jobs": jobs,
        "inputs": [],
        "model_inventory": registration.get("model_inventory", {}),
        "log": "/home/jonas/git/xgraph/target/queue/webqsp-gcr-pricing.log",
        "limits": "One attempt per family, no hf/old fallback or score access; startup and "
        "failed launcher time charged conservatively; 2 GPU-h within existing core reserve",
    }


def headroom_argv(job):
    return [
        "env",
        "XG_GPU_IDLE=1",
        "XG_AVAILABLE_WAIT=43200",
        "XG_AVAILABLE_POLL=60",
        *(
            f"{key}={job['env'][key]}"
            for key in (
                "XG_PAGECACHE_ROOTS",
                "XG_REQUIRE_MEMFREE",
                "XG_DROP_PAGECACHE_ALWAYS",
                "XG_HEADROOM_WAIT",
            )
            if key in job["env"]
        ),
        "nice",
        "-n",
        "19",
        "taskset",
        "-c",
        "4-7",
        "bash",
        str(ROOT / "scripts/gpu-headroom.sh"),
        job["env"]["XG_NEED_GIB"],
    ]


def preview(plan):
    """Print every launch argv without probing hardware, building or opening outputs."""
    for job in plan["jobs"]:
        print(shlex.join(headroom_argv(job)))
        argv = [
            "env",
            *(f"{k}={v}" for k, v in sorted(job["env"].items())),
            "timeout",
            "--signal=TERM",
            "--kill-after=15",
            str(job["wall_allowance_seconds"] - plan["cleanup_seconds"]),
            *job["command"],
        ]
        print(shlex.join(argv))


def owner_run(path, qualification, harness_path=None):
    """Launch only from the clean sealed tree; reuse hash-bound complete attempts."""
    import fcntl

    from xgraph.webqsp_test import model_inventory

    plan = read_sealed(path)
    harness = read_sealed(harness_path) if harness_path else None
    if git_commit() != (harness["commit"] if harness else plan["commit"]):
        raise ValueError("native GCR pricing refuses a dirty or changed worktree")
    if harness:
        from xgraph.webqsp_gcr_qualification import verify_cpu

        if harness["pricing_plan"] != identity(path):
            raise ValueError("GCR execution harness changed registered pricing plan")
        verify_cpu(harness, harness_path)
    for item in plan["inputs"]:
        relocated = (
            harness.get("pricing_input_relocations", {}).get(item["path"]) if harness else None
        )
        if relocated:
            if any(item[k] != relocated[k] for k in ("sha256", "bytes")):
                raise ValueError("GCR historical input relocation changed content")
            verify_identity(relocated)
        else:
            verify_identity(item)
    for saved in plan["model_inventory"].values():
        if model_inventory(saved["path"]) != saved:
            raise ValueError("native GCR pricing model inventory changed")
    qualification = Path(qualification)
    qualified = json.loads(qualification.read_text())
    if (
        qualified.get("producer_commit") != plan["producer_commit"]
        or qualified.get("recipe_sha256") != plan["recipe_sha256"]
        or qualified.get("families") != dict.fromkeys(FAMILIES, True)
        or not qualified.get("evidence")
        or (harness and qualified.get("execution_harness") != identity(harness_path))
    ):
        raise ValueError("native GCR pricing needs matched native qualification in both families")
    for item in qualified["evidence"]:
        verify_identity(item)
    if harness:
        plan = isolated_plan(plan, harness, harness_path)
    out = Path(plan["out"])
    with (out / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_stages(plan, path, identity(qualification))


def run_stages(plan, path, qualification):
    """Use the existing process-group watchdog and lifetime ledger seam."""
    if plan.get("process_stages"):
        return run_isolated_stages(plan, path, qualification)
    out = Path(plan["out"])
    ledger_path = out / "ledger.json"
    receipt_path = out / "receipt.json"
    if receipt_path.exists():
        receipt = read_sealed(receipt_path)
        if receipt["plan"] != identity(path):
            raise ValueError("native GCR pricing receipt changed plan")
        for item in receipt["measurement"]["evidence"]:
            verify_identity(item)
        status_line(plan, "all", "skip-existing")
        return receipt["measurement"]
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        if ledger.get("plan") != identity(path) or ledger.get("active") is not None:
            raise ValueError("unreconciled native GCR launch; no further GPU work")
    else:
        ledger = {"plan": identity(path), "spent_seconds": 0, "active": None, "attempts": []}
        atomic_json(ledger_path, ledger)
    snapshot = out / "gcr-code"
    if not snapshot.exists():
        snapshot_code(plan["producer_root"], plan["producer_commit"], snapshot)
    rows, evidence, reason = [], [identity(path), qualification], None
    for job in plan["jobs"]:
        family = job["family"]
        stage_receipt = out / f"{family}-receipt.json"
        if stage_receipt.exists():
            saved = read_sealed(stage_receipt)
            if saved["job"] != job:
                raise ValueError("native GCR stage receipt changed recipe")
            for item in saved["evidence"]:
                verify_identity(item)
            status_line(plan, family, "skip-existing")
        else:
            from xgraph.webqsp_test import stage_id

            if any(a["stage"] == stage_id(job) for a in ledger["attempts"]):
                raise ValueError("unreceipted native GCR attempt; never retry it")
            if any(Path(p).exists() for p in job["outputs"]):
                raise ValueError("native GCR output without receipt; never reuse it")
            status_line(plan, family, "waiting-headroom")
            subprocess.run(headroom_argv(job), check=True)
            for p in job["outputs"]:
                Path(p).parent.mkdir(parents=True, exist_ok=True)
            status_line(plan, family, "start")
            limit = min(
                plan["cap_seconds"], ledger["spent_seconds"] + job["wall_allowance_seconds"]
            )
            status, elapsed = execute_gpu_stage(job, ledger_path, limit, out / f"{family}.log")
            ledger = json.loads(ledger_path.read_text())
            # A daemon error is not proof that the owned container was removed.
            cleanup = subprocess.run(
                ["docker", "ps", "-aq", "--filter", f"name=^/{job['env']['XG_CONTAINER_NAME']}$"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if cleanup.returncode or cleanup.stdout.strip():
                ledger["active"] = {
                    "uncertain_cleanup": family,
                    "reserved_seconds": plan["cap_seconds"],
                }
                atomic_json(ledger_path, ledger)
                raise ValueError(
                    "unconfirmed native GCR cleanup; reserve retained, no further launch"
                )
            stage_evidence = [identity(out / f"{family}.log")]
            events = Path(job["env"]["XG_GPU_EVENTS"])
            if events.exists():
                stage_evidence.append(identity(events))
            complete = False
            if status == 0:
                records = [
                    json.loads(line) for line in Path(job["outputs"][0]).read_text().splitlines()
                ]
                complete = sorted(unique_ids(r["id"] for r in records)) == sorted(plan["ids"])
                stage_evidence += [identity(p) for p in job["outputs"]]
            saved = {
                "job": job,
                "row": {
                    "family": family,
                    "name": "gcr-native",
                    "questions": plan["questions"],
                    "wall_seconds": elapsed,
                    "returncode": status if status or complete else 1,
                },
                "evidence": stage_evidence,
            }
            write_sealed(stage_receipt, saved)
            status_line(plan, family, f"exit {saved['row']['returncode']}")
        rows.append(saved["row"])
        evidence += [identity(stage_receipt), *saved["evidence"]]
        if saved["row"]["returncode"]:
            reason = f"native hf-group {family} failed or exhausted its fixed pricing allowance"
            break
    ledger = json.loads(ledger_path.read_text())
    evidence.append(identity(ledger_path))
    measurement = {
        "producer_commit": plan["producer_commit"],
        "semantics": "hf-group",
        "recipe_sha256": plan["recipe_sha256"],
        "validation_ids": plan["ids"],
        "pricing_plan": identity(path),
        "native_qualified": True,
        "status": "UNAVAILABLE" if reason else "COMPLETE",
        "reason": reason,
        "charged_seconds": ledger["spent_seconds"],
        "per_arm": rows,
        "evidence": evidence,
    }
    write_sealed(receipt_path, {"plan": identity(path), "measurement": measurement})
    group = {**measurement, "receipt": identity(receipt_path)}
    atomic_json(
        out / "baseline-timings.json",
        {"scope": "validation", "complete": True, "groups": {"gcr": group}},
    )
    status_line(plan, "all", measurement["status"])
    return measurement


def isolated_plan(plan, harness, harness_path):
    """A recorded lifecycle repair, retaining the registered scientific recipe hash."""
    return {
        **plan,
        "process_stages": True,
        "memory_recovery": harness.get("memory_recovery", False),
        "source": harness["source"],
        "load_progress_seconds": harness["load_progress_seconds"],
        "phase_wall_seconds": {"paths": 1800, "synthesis": 1500},
        "execution_harness": identity(harness_path),
    }


def run_isolated_stages(plan, path, qualification):
    from xgraph.gcr_owner import attempt_path, phase_jobs, run_phase, verify_ledger

    out = Path(plan["out"])
    receipt_path = out / "receipt.json"
    if receipt_path.exists():
        receipt = read_sealed(receipt_path)
        if receipt["plan"] != identity(path):
            raise ValueError("native GCR pricing receipt changed plan")
        for item in receipt["measurement"]["evidence"]:
            verify_identity(item)
        status_line(plan, "all", "skip-existing")
        return receipt["measurement"]
    rows, reason = [], None
    evidence = [identity(path), qualification, plan["execution_harness"]]
    for job in plan["jobs"]:
        try:
            for stage in phase_jobs(plan, job):
                run_phase(plan, path, stage)
        except ValueError:
            ledger = verify_ledger(plan, path)
            if not ledger["attempts"] or ledger["attempts"][-1]["returncode"] == 0:
                raise
            reason = f"native hf-group {job['family']} failed its fixed pricing allowance"
        ledger = verify_ledger(plan, path)
        attempts = [
            a
            for a in ledger["attempts"]
            if a["stage"].startswith(f"{job['scope']}-{job['family']}-")
        ]
        status = attempts[-1]["returncode"]
        if status == 0:
            records = [
                json.loads(line) for line in Path(job["outputs"][0]).read_text().splitlines()
            ]
            if sorted(unique_ids(r["id"] for r in records)) != sorted(plan["ids"]):
                raise ValueError("native GCR price has incomplete request coverage")
        rows.append(
            {
                "family": job["family"],
                "name": "gcr-native",
                "questions": plan["questions"],
                "wall_seconds": sum(a["elapsed_seconds"] for a in attempts),
                "returncode": status,
            }
        )
        if reason:
            break
    ledger = verify_ledger(plan, path)
    for number in range(len(ledger["attempts"])):
        receipt = attempt_path(plan, number)
        evidence += [identity(receipt), *read_sealed(receipt)["evidence"]]
    evidence.append(identity(out / "ledger.json"))
    measurement = {
        "producer_commit": plan["producer_commit"],
        "semantics": "hf-group",
        "recipe_sha256": plan["recipe_sha256"],
        "validation_ids": plan["ids"],
        "pricing_plan": identity(path),
        "native_qualified": True,
        "status": "UNAVAILABLE" if reason else "COMPLETE",
        "reason": reason,
        "charged_seconds": ledger["spent_seconds"],
        "per_arm": rows,
        "evidence": evidence,
    }
    write_sealed(receipt_path, {"plan": identity(path), "measurement": measurement})
    atomic_json(
        out / "baseline-timings.json",
        {
            "scope": "validation",
            "complete": True,
            "groups": {"gcr": {**measurement, "receipt": identity(receipt_path)}},
        },
    )
    status_line(plan, "all", measurement["status"])
    return measurement


def status_line(plan, family, status):
    from datetime import UTC, datetime

    log = Path(plan["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write(f"webqsp-gcr-pricing {family} {status} {datetime.now(UTC).isoformat()}\n")


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--native-qualification", type=Path)
    parser.add_argument("--harness-plan", type=Path)
    args = parser.parse_args(argv)
    if args.run:
        qualification = args.native_qualification or os.environ.get("XG_GCR_NATIVE_QUALIFICATION")
        if not qualification:
            parser.error("--run requires --native-qualification (or XG_GCR_NATIVE_QUALIFICATION)")
        owner_run(args.plan, qualification, args.harness_plan)
    else:
        plan = read_sealed(args.plan)
        if args.harness_plan:
            from xgraph.container_cleanup import cleanup_argvs
            from xgraph.gcr_owner import phase_jobs

            harness = read_sealed(args.harness_plan)
            plan = isolated_plan(plan, harness, args.harness_plan)
            print(f"# Native pricing lifetime cap: {plan['cap_seconds']} seconds")
            print(f"# Qualification prerequisite lifetime cap: {harness['cap_seconds']} seconds")
            for job in plan["jobs"]:
                for stage in phase_jobs(plan, job):
                    preview({**plan, "jobs": [stage]})
                    for argv in cleanup_argvs(stage["env"]["XG_CONTAINER_NAME"]):
                        print(shlex.join(argv))
        else:
            preview(plan)


def read_native_price(registration, group):
    """Accept a receipted native price or a terminal bounded attempt, never MINTQA costs."""
    spec = registration["gcr_native_pricing"]
    try:
        verify_identity(spec["plan"])
        verify_identity(group["receipt"])
        receipt = json.loads(Path(group["receipt"]["path"]).read_text())
        if receipt.get("measurement") != {k: v for k, v in group.items() if k != "receipt"}:
            raise ValueError("receipt measurement changed")
        if (
            receipt.get("plan") != spec["plan"]
            or group.get("pricing_plan") != spec["plan"]
            or group.get("producer_commit") != registration["config"]["gcr_source_commit"]
            or group.get("semantics") != "hf-group"
            or group.get("validation_ids") != spec["ids"]
            or group.get("recipe_sha256") != baseline_price_contract(registration, "gcr")
            or not group.get("evidence")
            or positive(group.get("charged_seconds"), "charge") > spec["cap_seconds"]
        ):
            raise ValueError("changed native price identity or charge")
        for item in group["evidence"]:
            verify_identity(item)
        if group.get("native_qualified") is not True:
            raise ValueError("incomplete or unqualified native measurement")
        if group.get("status") == "UNAVAILABLE" and group.get("reason"):
            rows = group.get("per_arm", [])
            if (
                not rows
                or len(rows) > 2
                or [r.get("family") for r in rows] != list(FAMILIES)[: len(rows)]
            ):
                raise ValueError("unavailable price needs an ordered bounded native attempt")
            if (
                any(
                    r.get("name") != "gcr-native"
                    or r.get("questions") != spec["questions"]
                    or type(r.get("returncode")) is not int
                    for r in rows
                )
                or any(r["returncode"] != 0 for r in rows[:-1])
                or rows[-1]["returncode"] == 0
                or sum(positive(r["wall_seconds"], "attempt wall") for r in rows)
                != group["charged_seconds"]
            ):
                raise ValueError(
                    "unavailable native price needs a terminal failed attempt and full charge"
                )
            return {"unavailable": True, "reason": group["reason"]}
        if group.get("status") != "COMPLETE":
            raise ValueError("incomplete native measurement")
        rates = {}
        for row in group["per_arm"]:
            key = f"{row['family']}/{row['name']}"
            if (
                key in rates
                or row.get("returncode") != 0
                or row.get("questions") != spec["questions"]
            ):
                raise ValueError("duplicate/failed/changed native timing")
            rates[key] = positive(row["wall_seconds"], "launcher wall") / spec["questions"]
        if set(rates) != {f"{f}/gcr-native" for f in FAMILIES}:
            raise ValueError("incomplete native stage matrix")
        if sum(rates.values()) * spec["questions"] > group["charged_seconds"]:
            raise ValueError("undercharged native attempts")
        return rates
    except (ValueError, KeyError, OSError, TypeError) as error:
        raise ValueError(f"native WebQSP GCR pricing: {error}") from error


def amortized_native_price(registration, group):
    """Project complete receipted costs on the other baseline groups' frozen basis."""
    from xgraph.webqsp_gcr_decision import describe_pricing

    raw = read_native_price(registration, group)
    spec = registration["gcr_amortization"]
    basis = len(registration["pilot"]["cohort_ids"])
    if (
        basis != 100
        or type(spec.get("basis_questions")) is not int
        or spec["basis_questions"] != basis
        or spec.get("rule") != "fixed_over_pilot_plus_workload"
        or spec.get("pricing_receipt") != group["receipt"]
    ):
        raise ValueError("GCR amortization changed the frozen 100-question basis or receipt")
    if raw.get("unavailable"):
        return {"rates": raw, "basis_questions": basis}
    directory = Path(registration["gcr_native_pricing"]["plan"]["path"]).parent
    components = describe_pricing(directory)
    rows, rates = [], {}
    for row in components["per_family"]:
        if row["questions"] != 2 or set(row["phase_workload_seconds"]) != {"paths", "synthesis"}:
            raise ValueError("GCR amortization requires both measured workloads on two questions")
        workload = sum(
            positive(v, "GCR phase workload") for v in row["phase_workload_seconds"].values()
        )
        wall = positive(row["launcher_wall_seconds"], "GCR complete launcher wall")
        fixed = wall - workload
        if not math.isfinite(fixed) or fixed < 0:
            raise ValueError("GCR amortization workload exceeds complete launcher wall")
        rate = fixed / basis + workload / row["questions"]
        rates[f"{row['family']}/gcr-native"] = rate
        rows.append(
            {
                "family": row["family"],
                "measured_questions": row["questions"],
                "launcher_wall_seconds": wall,
                "generation_workload_seconds": workload,
                "fixed_launcher_seconds": fixed,
                "workload_seconds_per_question": workload / row["questions"],
                "amortized_seconds_per_question": rate,
            }
        )
    return {
        "label": "PROSPECTIVE_ADMISSION_PROJECTION",
        "basis_questions": basis,
        "rates": rates,
        "per_family": rows,
        "inputs": components["inputs"],
        "limits": "Two-question validation workload extrapolation, not a measured 100-question "
        "run or steady-state cost. Every fixed launcher component is retained. The minimum "
        "panel of 100 charges at least one complete fixed cost per family. Actual launcher "
        "wall is bounded by the sealed funded shares; an overrun is INCOMPLETE, with no top-up.",
    }


def trie_census(data, ids, tokenizer_path):
    """Read requests and the native DiGraph store, never question/gold files."""
    from tokenizers import Tokenizer

    from xgraph.kgversions import component_ranges, version_files
    from xgraph.path_trie import TokenTrie, collect_paths
    from xgraph.toolkg import KGStore, store_for, version_stores

    data, tokenizer_path = Path(data), Path(tokenizer_path)
    ids = unique_ids(ids)
    wanted = set(ids)
    request_path = data / "requests-gcr.jsonl"
    with request_path.open() as stream:
        requests = [r for line in stream if (r := json.loads(line))["id"] in wanted]
    if set(unique_ids(r["id"] for r in requests)) != wanted:
        raise ValueError("native GCR census needs every registered non-test request")
    requests = {r["id"]: r for r in requests}
    store = KGStore(data / "kg-gcr.sqlite")
    names = {r["kg_version"] for r in requests.values()}
    versions = version_stores(store, data / "versions", names)
    bounds = component_ranges(data / "versions")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    rows = []
    try:
        for q in ids:
            started = time.monotonic()
            request = requests[q]
            read = store_for(request, store, versions)
            enumeration = collect_paths(
                read,
                request.get("anchor") or request.get("topics", []),
                hops=2,
                max_paths=100000,
                parallel_relations="preserve",
                overflow="truncate",
            )
            trie = TokenTrie.from_paths(enumeration.paths, tokenizer)
            first, stop = bounds[request["kg_version"]]
            triples = store._con.execute(
                "SELECT count(*) FROM triples WHERE s >= ? AND s < ? AND o >= ? AND o < ?",
                (first, stop, first, stop),
            ).fetchone()[0]
            rows.append(
                {
                    "id": q,
                    "triples": triples,
                    "paths": len(enumeration.paths),
                    "paths_by_hops": dict(
                        sorted(Counter(str(len(p.relations)) for p in enumeration.paths).items())
                    ),
                    "path_count_is_lower_bound": enumeration.truncated,
                    "trie_nodes": len(trie.children),
                    "trie_sequences": len(trie.sequences),
                    "trie_tokens": sum(map(len, trie.sequences)),
                    "maximum_path_tokens": max(map(len, trie.sequences), default=0),
                    "cpu_seconds": time.monotonic() - started,
                }
            )
    finally:
        for read in versions.values():
            read.store._con.close()
        store._con.close()
    files = [
        request_path,
        data / "kg-gcr.sqlite",
        tokenizer_path,
        *version_files(data / "versions"),
    ]
    return {
        "artifact_kind": "xgraph.webqsp_gcr_census",
        "per_question": rows,
        "config": {
            "hops": 2,
            "max_paths": 100000,
            "parallel_relations": "preserve",
            "graph": "released input-order DiGraph; question components",
            "seed": 17,
        },
        "inputs": [identity(p) for p in files],
        "GPU_hours_used": 0,
        "limits": "CPU structure only; no model, outputs, answer gold or measured GPU price",
    }


if __name__ == "__main__":
    main()
