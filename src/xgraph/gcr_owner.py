"""Receipted GCR processes sharing one lifetime ledger across owner resumptions."""

from __future__ import annotations

import copy
import json
import math
import subprocess
from pathlib import Path

from xgraph import webqsp_test as study


def phase_jobs(plan, job):
    """Retain all scientific arguments; add only process and handoff controls."""
    output = Path(job["outputs"][0])
    handoff = Path(str(output) + ".gcr-paths.json")
    result = []
    for phase in ("paths", "synthesis"):
        stage = copy.deepcopy(job)
        stage["name"] += "-" + phase
        stage["phase"] = phase
        stage["command"][6] = str(Path(plan["source"]) / "scripts/generate.sh")
        stage["command"] += ["--gcr-stage", phase, "--gcr-handoff", str(handoff)]
        stage["env"]["XG_CONTAINER_NAME"] += "-" + phase
        stage["env"]["XG_GPU_EVENTS"] = str(
            Path(plan["out"]) / f"{job['family']}-{phase}-events.jsonl"
        )
        stage["env"]["XG_START_ATTEMPTS"] = "1"
        if plan.get("memory_recovery"):
            roots = stage["env"].get("XG_PAGECACHE_ROOTS", plan["data"]).split(":")
            # A preceding family's reader can still occupy the unified page cache.
            for other in plan["jobs"]:
                roots += [other["command"][7], other["env"]["XG_GCR_SYNTHESIS_MODEL"]]
            stage["env"].update(
                XG_PAGECACHE_ROOTS=":".join(dict.fromkeys(p for p in roots if p)),
                XG_REQUIRE_MEMFREE="1",
                XG_DROP_PAGECACHE_ALWAYS="1",
                XG_HEADROOM_WAIT="900",
            )
        stage["load_progress_seconds"] = plan["load_progress_seconds"]
        stage["wall_allowance_seconds"] = plan["phase_wall_seconds"][phase]
        if phase == "paths":
            records = Path(str(output) + ".paths.jsonl")
            stage["outputs"] = [
                str(handoff),
                str(handoff) + ".sha256",
                str(records),
                str(records) + ".summary.json",
                str(records) + ".provenance.json",
            ]
        else:
            stage["outputs"] = [
                str(output),
                str(output) + ".summary.json",
                str(output) + ".provenance.json",
            ]
        result.append(stage)
    return result


def ledger_path(plan):
    return Path(plan.get("ledger_path", str(Path(plan["out"]) / "ledger.json")))


def attempt_path(plan, number):
    inherited = plan.get("inherited_receipts", {}).get(str(number))
    if inherited:
        study.verify_identity(inherited)
        return Path(inherited["path"])
    return Path(plan["out"]) / f"attempt-{number}.json"


def verify_ledger(plan, path, *, reconciling=False):
    """Failed attempts remain charged and bound; every new attempt needs a receipt."""
    account_path = ledger_path(plan)
    if not account_path.exists():
        study.atomic_json(
            account_path,
            {
                "plan": study.identity(path),
                "spent_seconds": 0,
                "active": None,
                "attempts": [],
            },
        )
    ledger = json.loads(account_path.read_text())
    origin = plan.get("resume_origin")
    inherited = origin["attempts"] if origin else []
    expected_plan = origin["plan"] if origin else study.identity(path)
    if ledger["active"] is not None and not reconciling:
        raise ValueError("unreconciled GCR attempt or unknown cleanup; reservation retained")
    if (
        ledger["plan"] != expected_plan
        or ledger["attempts"][: len(inherited)] != inherited
        or not 0 <= ledger["spent_seconds"] <= plan["cap_seconds"]
        or any(a["elapsed_seconds"] <= 0 for a in ledger["attempts"])
        or not math.isclose(
            ledger["spent_seconds"],
            sum(a["elapsed_seconds"] for a in ledger["attempts"]),
            abs_tol=1e-6,
        )
    ):
        raise ValueError("GCR lifetime accounting changed")
    if "cap_seconds" in ledger and ledger["cap_seconds"] != plan["cap_seconds"]:
        raise ValueError("GCR ledger cap changed")
    if (
        plan.get("cap_addendum")
        and not plan.get("preflight")
        and ledger.get("cap_addendum") != plan["cap_addendum"]
    ):
        raise ValueError("GCR ledger cap addendum changed")
    for number in plan.get("inherited_receipts", {}):
        index = int(number)
        if not 0 <= index < len(inherited):
            raise ValueError("GCR inherited receipt has no inherited charge")
        receipt = study.read_sealed(attempt_path(plan, index))
        if (
            receipt["attempt"] != inherited[index]
            or study.stage_id(receipt["job"]) != inherited[index]["stage"]
        ):
            raise ValueError("GCR inherited receipt changed its charge or stage")
        for item in receipt["evidence"]:
            study.verify_identity(item)
    jobs = {study.stage_id(s): s for j in plan["jobs"] for s in phase_jobs(plan, j)}
    for number, attempt in enumerate(ledger["attempts"][len(inherited) :], start=len(inherited)):
        if reconciling and number == len(ledger["attempts"]) - 1:
            continue
        receipt = study.read_sealed(attempt_path(plan, number))
        if (
            receipt["plan"] != study.identity(path)
            or receipt["attempt"] != attempt
            or receipt["job"] != jobs.get(attempt["stage"])
        ):
            raise ValueError("GCR attempt receipt changed plan, job or elapsed charge")
        for item in receipt["evidence"]:
            study.verify_identity(item)
    return ledger


def completed(plan, path, stage, ledger):
    attempts = [
        (i, a) for i, a in enumerate(ledger["attempts"]) if a["stage"] == study.stage_id(stage)
    ]
    successes = [(i, a) for i, a in attempts if a["returncode"] == 0]
    if len(successes) > 1 or (successes and successes[0] != attempts[-1]):
        raise ValueError("GCR stage has duplicate or superseded successful attempts")
    if successes:
        i, _ = successes[0]
        receipt = study.read_sealed(attempt_path(plan, i))
        for item in receipt["evidence"]:
            study.verify_identity(item)
        return receipt
    return None


def engine_init_oom(stage, log):
    """Recognize only an OOM proven to precede engine readiness and any request."""
    text = log.read_text()
    return (
        "CUDA error: out of memory" in text
        and any(
            s in text
            for s in ("Engine core initialization failed", "init_device", "get_memory_info")
        )
        and not any(
            s in text
            for s in (
                "[xgraph-load] ready",
                "init engine (",
                "Processed prompts:",
                "[xgraph-request]",
            )
        )
        and not any(
            Path(p).exists() for p in stage["outputs"] if not p.endswith(".provenance.json")
        )
    )


def run_phase(plan, path, stage):
    """One infrastructure retry, durably receipted and inside the stage/lifetime caps."""
    from xgraph import webqsp_gcr_pricing as pricing

    ledger = verify_ledger(plan, path)
    prior = [
        (n, a) for n, a in enumerate(ledger["attempts"]) if a["stage"] == study.stage_id(stage)
    ]
    limit = min(plan["cap_seconds"], ledger["spent_seconds"] + stage["wall_allowance_seconds"])
    pending_retry = False
    if prior:
        n, attempt = prior[-1]
        previous = study.read_sealed(attempt_path(plan, n))
        if attempt["returncode"]:
            if previous.get("automatic_retry"):
                raise ValueError("GCR model stage failed; automatic infrastructure retry exhausted")
            if previous.get("failure_class") == "engine_init_cuda_oom":
                pending_retry = True
                limit = min(plan["cap_seconds"], previous["allowance_cap_seconds"])
    for retry in (True,) if pending_retry else (False, True):
        receipt = run_attempt(plan, path, stage, limit, retry=retry)
        if receipt["attempt"]["returncode"] == 0:
            return receipt
        if retry or receipt.get("failure_class") != "engine_init_cuda_oom":
            break
        pricing.status_line(
            plan,
            f"{stage['family']}-{stage['phase']}",
            "retry engine_init_cuda_oom; cache drop and headroom charged inside launcher",
        )
    raise ValueError("GCR model stage failed; elapsed charged, completed paths retained")


def run_attempt(plan, path, stage, limit, *, retry=False):
    from xgraph import webqsp_gcr_pricing as pricing

    ledger = verify_ledger(plan, path)
    label = f"{stage['family']}-{stage['phase']}"
    receipt = completed(plan, path, stage, ledger)
    if receipt is not None:
        pricing.status_line(plan, label, "skip-existing")
        return receipt
    if any(Path(p).exists() for p in stage["outputs"]):
        raise ValueError("unreceipted GCR output; preserve it before resuming")
    if min(plan["cap_seconds"], limit) - ledger["spent_seconds"] <= 30:
        raise ValueError("GCR lifetime budget exhausted; no further launch")
    if not retry:
        pricing.status_line(plan, label, "waiting-headroom")
        subprocess.run(pricing.headroom_argv(stage), check=True)
    for output in stage["outputs"]:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    number = len(ledger["attempts"])
    log = Path(plan["out"]) / f"{label}-attempt-{number}.log"
    pricing.status_line(plan, label, "start")
    status, elapsed = study.execute_gpu_stage(
        stage,
        ledger_path(plan),
        limit,
        log,
    )
    ledger = json.loads(ledger_path(plan).read_text())
    try:
        cleanup = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"name=^/{stage['env']['XG_CONTAINER_NAME']}$"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        uncertain = cleanup.returncode != 0 or bool(cleanup.stdout.strip())
    except (subprocess.TimeoutExpired, OSError):
        uncertain = True
    if uncertain:
        ledger["active"] = {"uncertain_cleanup": label, "reserved_seconds": plan["cap_seconds"]}
        study.atomic_json(ledger_path(plan), ledger)
        pricing.status_line(plan, label, "exit cleanup unknown; reservation retained")
        raise ValueError("unknown GCR cleanup; reservation retained")
    attempt = ledger["attempts"][-1]
    if attempt["returncode"] != status or not math.isclose(
        attempt["elapsed_seconds"], elapsed, abs_tol=1e-6
    ):
        raise ValueError("GCR elapsed charge disagrees with launcher")
    return record_attempt(plan, path, stage, number, attempt, log, limit, retry=retry)


def record_attempt(plan, path, stage, number, attempt, log, limit, *, retry=False, evidence=()):
    from xgraph import webqsp_gcr_pricing as pricing

    status, elapsed = attempt["returncode"], attempt["elapsed_seconds"]
    artifacts = [log]
    events = Path(stage["env"]["XG_GPU_EVENTS"])
    if events.exists():
        artifacts.append(events)
    artifacts += [Path(p) for p in stage["outputs"] if Path(p).exists()]
    if status == 0 and not all(Path(p).is_file() for p in stage["outputs"]):
        raise ValueError("GCR success has incomplete outputs; no resumable receipt")
    failure = "engine_init_cuda_oom" if status and engine_init_oom(stage, log) else None
    if status:
        # Preserve partial output and events under immutable attempt-specific names.
        archived = Path(plan["out"]) / f"failed-attempt-{number}"
        archived.mkdir(exist_ok=True)
        preserved = [log, *sorted(archived.iterdir())]
        for artifact in artifacts[1:]:
            destination = archived / artifact.name
            artifact.rename(destination)
            preserved.append(destination)
        artifacts = preserved
    receipt = {
        "plan": study.identity(path),
        "job": stage,
        "attempt": attempt,
        "allowance_cap_seconds": limit,
        "evidence": [study.identity(p) for p in artifacts] + list(evidence),
        **({"failure_class": failure} if failure else {}),
        **({"automatic_retry": True} if retry else {}),
    }
    study.write_sealed(attempt_path(plan, number), receipt)
    pricing.status_line(
        plan, f"{stage['family']}-{stage['phase']}", f"exit {status}; elapsed {elapsed:.3f} seconds"
    )
    return receipt


def historical_cleanup_charge(plan, stage, attempt, path):
    """Use an operator-saved Docker lifecycle stream for the exact owned launch."""
    from datetime import datetime

    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    container = stage["env"]["XG_CONTAINER_NAME"]
    ids, actions, stamps = set(), {}, []
    for row in rows:
        actor = row["Actor"]
        attrs = actor["Attributes"]
        stamp = row["timeNano"]
        if (
            row["Type"] != "container"
            or attrs.get("name") != container
            or attrs.get("xgraph.gcr.owner") != "webqsp-gcr-qualification"
            or (plan.get("runtime_image") and attrs.get("image") != plan["runtime_image"])
            or type(stamp) is not int
            or (stamps and stamp < stamps[-1])
        ):
            raise ValueError("Docker cleanup evidence changed its container, image or time")
        ids.add(actor["ID"])
        stamps.append(stamp)
        if row["Action"] in ("create", "start", "die", "destroy"):
            if row["Action"] in actions:
                raise ValueError("Docker cleanup evidence has duplicate lifecycle boundaries")
            actions[row["Action"]] = stamp
    if (
        len(ids) != 1
        or list(actions) != ["create", "start", "die", "destroy"]
        or actions["create"] < datetime.fromisoformat(attempt["started_at"]).timestamp() * 1e9
    ):
        raise ValueError("Docker cleanup evidence does not prove the exact launch ended")
    # Round upward to whole seconds; never refund any previously charged time.
    elapsed = math.ceil(
        actions["destroy"] / 1e9 - datetime.fromisoformat(attempt["started_at"]).timestamp()
    )
    return max(attempt["elapsed_seconds"], elapsed)


def reconcile(path, *, lifecycle_events=None):
    """Settle a failed, already charged attempt; never restart a model or raise a cap.

    Historical Docker destruction can bound the cleanup tail. Without it or an
    existing end boundary, charge the remaining allowance conservatively.
    """
    import fcntl

    from xgraph.container_cleanup import stop_container

    path = Path(path)
    plan = study.read_sealed(path)
    account = ledger_path(plan)
    with (account.parent / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads(account.read_text())
        number = len(ledger["attempts"]) - 1
        if ledger["active"] is None and attempt_path(plan, number).exists():
            verify_ledger(plan, path)
            return study.read_sealed(attempt_path(plan, number))
        attempt = ledger["attempts"][number]
        stages = [s for j in plan["jobs"] for s in phase_jobs(plan, j)]
        stage = next(s for s in stages if study.stage_id(s) == attempt["stage"])
        label = f"{stage['family']}-{stage['phase']}"
        container = stage["env"]["XG_CONTAINER_NAME"]
        intent = Path(plan["out"]) / f"reconciliation-{number}.json"
        if intent.exists():
            saved = study.read_sealed(intent)
            if saved["plan"] != study.identity(path):
                raise ValueError("GCR reconciliation changed its plan")
            for item in saved["evidence"]:
                study.verify_identity(item)
        else:
            if (
                not attempt["returncode"]
                or not ledger["active"]
                or ledger["active"].get("uncertain_cleanup")
                not in (
                    label,
                    container,
                )
            ):
                raise ValueError("GCR reconciliation requires a failed retained reservation")
            verify_ledger(plan, path, reconciling=True)
            before = intent.with_name(f"reconciliation-{number}-ledger.json")
            study.atomic_json(before, ledger)
            events = Path(stage["env"]["XG_GPU_EVENTS"])
            data = events.read_bytes() if events.exists() else b""
            snapshot = intent.with_name(f"reconciliation-{number}-events.jsonl")
            snapshot.write_bytes(data)
            rows = [json.loads(line) for line in data.splitlines()]
            historical = (
                historical_cleanup_charge(plan, stage, attempt, lifecycle_events)
                if lifecycle_events
                else None
            )
            saved = {
                "plan": study.identity(path),
                "attempt": attempt,
                "charge_remaining_reservation": bool(
                    rows and rows[-1]["event"] == "container_start"
                ),
                "historical_elapsed_seconds": historical,
                "allowance_cap_seconds": min(
                    plan["cap_seconds"],
                    ledger["spent_seconds"]
                    - attempt["elapsed_seconds"]
                    + attempt.get("maximum_seconds", plan["cap_seconds"]),
                ),
                "evidence": [study.identity(before), study.identity(snapshot)]
                + ([study.identity(lifecycle_events)] if lifecycle_events else []),
            }
            study.write_sealed(intent, saved)
        # This may inspect/stop/remove only the exact plan-owned container.
        stop_container(container, stage["env"]["XG_GPU_EVENTS"])
        if ledger["active"] is not None:
            if saved["attempt"] != attempt:
                raise ValueError("GCR reconciliation charge changed")
            if saved["charge_remaining_reservation"]:
                extra = (
                    saved["historical_elapsed_seconds"] - attempt["elapsed_seconds"]
                    if saved.get("historical_elapsed_seconds") is not None
                    else plan["cap_seconds"] - ledger["spent_seconds"]
                )
                if extra > plan["cap_seconds"] - ledger["spent_seconds"]:
                    raise ValueError("historical cleanup exceeds the remaining GCR cap")
                attempt.update(
                    launcher_elapsed_seconds=attempt["elapsed_seconds"],
                    reservation_settlement_seconds=extra,
                    accounting="docker_lifecycle_upper_bound"
                    if saved.get("historical_elapsed_seconds") is not None
                    else "reserved",
                )
                attempt["elapsed_seconds"] += extra
                ledger["spent_seconds"] += extra
            ledger["active"] = None
            study.atomic_json(account, ledger)
        receipt = record_attempt(
            {**plan, "log": str(Path(plan["out"]) / "reconciliation.log")},
            path,
            stage,
            number,
            attempt,
            Path(plan["out"]) / f"{label}-attempt-{number}.log",
            saved["allowance_cap_seconds"],
            evidence=[study.identity(intent), *saved["evidence"]],
        )
        verify_ledger(plan, path)
        return receipt
