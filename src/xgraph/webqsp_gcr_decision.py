"""Apply amendment 6's cost rule without opening the pending core pilot."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from xgraph import webqsp_gcr_pricing as pricing
from xgraph import webqsp_test as study
from xgraph.provenance import provenance


def cohort_bound(registration, rates):
    """Prove infeasibility even at the largest possible recomputed equal share."""
    rule = registration["pricing"]
    budget = study.positive(rule["baseline_cap_seconds"], "baseline bucket")
    minimum = rule["minimum_baseline_questions"]
    expected = {"/".join(s) for s in rule["baseline_stages"]["gcr"]}
    if budget > 43200 or type(minimum) is not int or minimum < 100 or set(rates) != expected:
        raise ValueError("changed amendment-4 baseline allocation or GCR stage matrix")
    cost = sum(study.positive(v, "native rate") for v in rates.values())
    bounds = [
        {
            "groups": count,
            "share_seconds": budget / count,
            "maximum_questions": math.floor(budget / count / cost),
        }
        for count in range(len(rule["baseline_order"]), 0, -1)
    ]
    result = {
        "admit": False,
        "seconds_per_question": cost,
        "minimum_questions": minimum,
        "maximum_questions": bounds[-1]["maximum_questions"],
        "equal_share_bounds": bounds,
        "rule": "Remove the most expensive infeasible measured group first; registered "
        "order breaks ties; recompute equal shares. Whole-cluster prefixes cannot exceed "
        "the question bound. No baseline shrinks the mandatory core.",
        "limits": "The whole bucket is an upper bound, not an allocation to GCR. No ranking "
        "against unpriced groups, core size, panel IDs or other admission is inferred.",
    }
    if result["maximum_questions"] < minimum:
        result.update(
            status="INFEASIBLE_AT_BOUND",
            comparison="PUBLISHED_NON_PAIRED",
            published_comparison=pricing.PUBLISHED_COMPARISON,
        )
    else:
        result.update(status="PENDING_CORE_AND_OTHER_PRICES", comparison="PENDING")
    if registration.get("gcr_amortization"):
        ids = study.select_ids(
            registration["population"]["intersection"],
            len(registration["population"]["intersection"]),
        )
        method = registration["analysis"]["cluster_method"]
        assignments = registration["question_cluster_audit"]["methods"][method]["assignments"]
        panels, scenarios = {}, {}
        for bound in bounds:
            bound["launcher_wall_caps_seconds"] = {
                family: bound["share_seconds"] * rates[f"{family}/gcr-native"] / cost
                for family in study.FAMILIES
            }
            rows = []
            for n in range(1, len(ids) + 1):
                panel = study.cluster_subpanel(
                    ids[:n], assignments, min(n, bound["maximum_questions"])
                )
                digest = hashlib.sha256(
                    json.dumps(panel, separators=(",", ":")).encode()
                ).hexdigest()
                if digest not in panels:
                    panels[digest] = {
                        "ids": panel,
                        "cluster_prefix": list(dict.fromkeys(assignments[q] for q in panel)),
                    }
                rows.append(
                    {
                        "core_questions": n,
                        "panel_questions": len(panel),
                        "minimum_funded": len(panel) >= minimum,
                        "panel_sha256": digest,
                        "standard_error_multiplier": math.sqrt(n / len(panel)) if panel else None,
                    }
                )
            scenarios[str(bound["groups"])] = rows
        result.update(
            conditional_core_prefixes=scenarios,
            conditional_panels=panels,
            final_core_questions=None,
            final_panel_questions=None,
            pending=["core_pilot", "subgraphrag", "dog"],
            limits="Conditional GCR funding ceilings for every possible core size and equal "
            "share count. Core size is unknown; another group's price can further reduce the "
            "common panel. These are not measured core prices, admissions or sealed test IDs. "
            "SE inflation is sqrt(n_core/n_panel) at equal paired variance, before clustering.",
        )
    return result


def verify_pricing(registration, directory):
    """Read only the native pricing/qualification chain; any inconsistency blocks."""
    from xgraph import gcr_owner

    directory = Path(directory).resolve()
    path = directory / "plan.json"
    plan = study.read_sealed(path)
    receipt_path = directory / "receipt.json"
    receipt = study.read_sealed(receipt_path)
    timing_path = directory / "baseline-timings.json"
    timings = json.loads(timing_path.read_text())
    group = timings["groups"]["gcr"]
    rates = pricing.read_native_price(registration, group)
    if (
        group["status"] != "COMPLETE"
        or group["receipt"] != study.identity(receipt_path)
        or receipt["plan"] != study.identity(path)
        or plan["recipe_sha256"] != group["recipe_sha256"]
        or plan["producer_commit"] != group["producer_commit"]
        or plan["ids"] != group["validation_ids"]
        or plan["cap_seconds"] != 7200
        or plan["stage_wall_seconds"] != 3300
        or plan["cleanup_seconds"] != 30
    ):
        raise ValueError("pricing receipt changed its registered plan, recipe or limits")
    evidence = group["evidence"]
    # The process pricing producer seals these prerequisites before attempt evidence.
    qualification_ref, harness_ref = evidence[1:3]
    qualified = study.read_sealed(qualification_ref["path"])
    harness_path = Path(harness_ref["path"])
    harness = study.read_sealed(harness_path)
    if (
        harness["pricing_plan"] != study.identity(path)
        or qualified["execution_harness"] != harness_ref
        or qualified["commit"] != harness["commit"]
        or qualified["producer_commit"] != plan["producer_commit"]
        or qualified["recipe_sha256"] != plan["recipe_sha256"]
        or qualified["families"] != dict.fromkeys(study.FAMILIES, True)
        or qualified["status"] != "QUALIFIED"
        or not qualified["evidence"]
    ):
        raise ValueError("pricing qualification or execution harness binding changed")
    checked, historical = {}, {}
    source_commit = registration.get("gcr_amortization", {}).get("historical_source_commit")

    def verify(item):
        path = Path(item["path"])
        relative = str(path.relative_to(study.ROOT)) if path.is_relative_to(study.ROOT) else None
        if source_commit and relative in {
            "src/xgraph/webqsp_test.py",
            "src/xgraph/webqsp_gcr_pricing.py",
            "scripts/webqsp-amendment-6.py",
        }:
            if (
                not isinstance(source_commit, str)
                or len(source_commit) != 40
                or any(c not in "0123456789abcdef" for c in source_commit)
            ):
                raise ValueError("historical consumer source requires an exact Git commit")
            data = subprocess.check_output(
                ["git", "show", f"{source_commit}:{relative}"], cwd=study.ROOT
            )
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("historical consumer source differs from the original receipt")
            historical[item["path"]] = {
                "original": item,
                "commit": source_commit,
                "git_path": relative,
            }
            return
        study.verify_identity(item)
        checked[item["path"]] = item

    for item in [*evidence, *harness["inputs"], *qualified["evidence"]]:
        verify(item)
    relocations = []
    for item in plan["inputs"]:
        relocated = harness.get("pricing_input_relocations", {}).get(item["path"])
        if relocated:
            if any(item[k] != relocated[k] for k in ("sha256", "bytes")):
                raise ValueError("historical pricing input relocation changed content")
            relocations.append({"original": item, "preserved": relocated})
        verify(relocated or item)
    for saved in plan["model_inventory"].values():
        if study.model_inventory(saved["path"]) != saved:
            raise ValueError("pricing model inventory changed")
    sidecars = []
    for side in sorted(directory.rglob("*.sha256")):
        original = side.with_suffix("")
        if study.identity(original)["sha256"] != side.read_text().strip():
            raise ValueError(f"pricing sha256 sidecar changed: {side}")
        sidecars.append(study.identity(side))
    if not (directory / "ledger.json").is_file():
        raise ValueError("pricing ledger is missing")
    execution = pricing.isolated_plan(plan, harness, harness_path)
    ledger = gcr_owner.verify_ledger(execution, path)
    if ledger["spent_seconds"] != group["charged_seconds"]:
        raise ValueError("pricing ledger and receipt charge differ")
    stages = {
        study.stage_id(s): s for j in plan["jobs"] for s in gcr_owner.phase_jobs(execution, j)
    }
    if [a["stage"] for a in ledger["attempts"]] != list(stages):
        raise ValueError("completed pricing requires exactly the four registered process stages")
    containers = []
    for attempt in ledger["attempts"]:
        stage = stages[attempt["stage"]]
        if (
            attempt["returncode"] != 0
            or attempt["timeout"]
            or not 0 < attempt["elapsed_seconds"] <= attempt["maximum_seconds"]
            or attempt["maximum_seconds"] > stage["wall_allowance_seconds"]
            or attempt["watchdog_argv"][:3] != ["timeout", "--signal=TERM", "--kill-after=15"]
            or float(attempt["watchdog_argv"][3]) != attempt["maximum_seconds"] - 30
            or attempt["watchdog_argv"][4:] != stage["command"]
        ):
            raise ValueError("pricing attempt exceeded its allowance or changed watchdog")
        event_path = Path(stage["env"]["XG_GPU_EVENTS"])
        events = [json.loads(line) for line in event_path.read_text().splitlines()]
        container = stage["env"]["XG_CONTAINER_NAME"]
        if (
            study.identity(event_path) not in evidence
            or [e["event"] for e in events] != ["container_start", "container_end"]
            or any(e["container"] != container for e in events)
            or not 0 < events[1]["monotonic"] - events[0]["monotonic"] <= attempt["elapsed_seconds"]
        ):
            raise ValueError("pricing container cleanup is unconfirmed or changed")
        containers.append(
            {
                "container": container,
                "events": study.identity(event_path),
                "occupancy_seconds": events[1]["monotonic"] - events[0]["monotonic"],
                "absence": "success receipt follows the harness's empty docker-ps check; "
                "no live probe",
            }
        )
    for row in group["per_arm"]:
        charge = sum(
            a["elapsed_seconds"]
            for a in ledger["attempts"]
            if a["stage"].startswith(f"baseline-{row['family']}-")
        )
        if charge != row["wall_seconds"] or charge > 3300:
            raise ValueError(
                "pricing family wall differs from ledger or exceeds launcher allowance"
            )
    qledger_path = (
        Path(harness["ledger_path"])
        if "ledger_path" in harness
        else Path(harness["out"]) / "ledger.json"
    )
    if study.identity(qledger_path) not in qualified["evidence"]:
        raise ValueError("qualification charge has no bound lifetime ledger")
    qledger = json.loads(qledger_path.read_text())
    qualification_charge = qualified["protocol_qualification_gpu_seconds"]
    if (
        qledger["active"] is not None
        or qualification_charge != qledger["spent_seconds"]
        or not 0 < qualification_charge <= harness["cap_seconds"] <= 7200
        or not math.isclose(
            qualification_charge,
            sum(a["elapsed_seconds"] for a in qledger["attempts"]),
            rel_tol=0,
            abs_tol=1e-6,
        )
    ):
        raise ValueError("qualification lifetime ledger charge changed")
    return {
        "status": "VERIFIED",
        "rates": rates,
        "pricing_seconds": ledger["spent_seconds"],
        "qualification_seconds": qualification_charge,
        "qualification": qualification_ref,
        "plan": study.identity(path),
        "receipt": study.identity(receipt_path),
        "baseline_timings": study.identity(timing_path),
        "harness": harness_ref,
        "sidecars": sidecars,
        "containers": containers,
        "input_relocations": relocations,
        "checked_evidence": list(checked.values()),
        **({"historical_sources": list(historical.values())} if historical else {}),
    }


def describe_pricing(directory):
    """Describe receipted process workloads separately from the admission price."""
    directory = Path(directory)
    receipt = study.read_sealed(directory / "receipt.json")
    measurement = receipt["measurement"]
    inputs = [study.identity(directory / "receipt.json")]
    rows = []
    for arm in measurement["per_arm"]:
        family = arm["family"]
        summary_path = directory / "runs" / family / "gcr-native.jsonl.summary.json"
        files = [
            summary_path,
            *[directory / f"{family}-{phase}-events.jsonl" for phase in ("paths", "synthesis")],
        ]
        for path in files:
            item = study.identity(path)
            if item not in measurement["evidence"]:
                raise ValueError("descriptive timing has no pricing receipt binding")
            inputs.append(item)
        summary = json.loads(summary_path.read_text())
        workload = sum(summary["phase_workload_seconds"].values())
        startup = sum(summary["phase_startup_seconds"].values())
        occupancy = 0
        for path in files[1:]:
            events = [json.loads(line) for line in path.read_text().splitlines()]
            if [e["event"] for e in events] != ["container_start", "container_end"]:
                raise ValueError("descriptive timing needs complete container boundaries")
            occupancy += events[1]["monotonic"] - events[0]["monotonic"]
        outside = arm["wall_seconds"] - occupancy
        other = occupancy - startup - workload
        if min(outside, other, startup, workload) < 0:
            raise ValueError("descriptive components exceed the receipted wall")
        rows.append(
            {
                "family": family,
                "questions": arm["questions"],
                "launcher_wall_seconds": arm["wall_seconds"],
                "container_occupancy_seconds": occupancy,
                "outside_container_seconds": outside,
                "model_startup_seconds": startup,
                "generation_workload_seconds": workload,
                "teardown_and_other_inside_seconds": other,
                "phase_startup_seconds": summary["phase_startup_seconds"],
                "phase_workload_seconds": summary["phase_workload_seconds"],
                "generation_workload_seconds_per_question": workload / arm["questions"],
                "registered_launcher_seconds_per_question": arm["wall_seconds"] / arm["questions"],
            }
        )
    return {
        "label": "DESCRIPTIVE_ONLY",
        "admission_use": False,
        "per_family": rows,
        "inputs": inputs,
        "limits": "Outside-container wall includes packaging/build and final admission; their "
        "individual durations are not separately instrumented. Model startup includes loading, "
        "compile and warmup in both processes. Workload includes enumeration, scoring and "
        "generation bookkeeping, not just GPU kernels. Only two validation questions per "
        "family; no steady-state extrapolation or new runtime claim. Admission always uses "
        "complete launcher wall/2, including every component.",
    }


def record_decision(parent_path, destination):
    """Seal a decision child of amendment 6; leave the registration and queues intact."""
    parent_path, destination = Path(parent_path).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"immutable GCR decision already exists: {destination}")
    registration = study.read_sealed(parent_path)
    if registration["amendment"]["amendment"] != 6:
        raise ValueError("GCR pricing decision requires its amendment-6 parent")
    for key in ("parent", "record", "protocol", "cluster_audit"):
        if key in registration["amendment"]:
            study.verify_identity(registration["amendment"][key])
    if (Path(registration["config"]["out"]) / "execution").exists():
        raise ValueError("GCR decision cannot change a sealed execution")
    if registration.get("freeze_deadline") and datetime.now(UTC) > datetime.fromisoformat(
        registration["freeze_deadline"]
    ):
        raise ValueError("GCR decision deadline has passed")
    stage = Path(registration["gcr_native_pricing"]["plan"]["path"]).parent
    audit = verify_pricing(registration, stage)
    decision = cohort_bound(registration, audit["rates"])
    if decision["status"] != "INFEASIBLE_AT_BOUND":
        raise ValueError("GCR admission remains pending until core and other prices exist")
    initial = audit["qualification_seconds"] + audit["pricing_seconds"]
    reserve = registration["pricing"]["reserve_seconds"]
    if initial > reserve:
        raise ValueError("qualification and pricing exceed the registered core reserve")
    combined = Path(registration["config"]["baseline_timings"])
    measured = json.loads((stage / "baseline-timings.json").read_text())
    data = json.loads(combined.read_text()) if combined.exists() else measured
    if data.get("scope") != "validation" or data.get("complete") is not True:
        raise ValueError("combined baseline timings changed scope or completion")
    group = measured["groups"]["gcr"]
    if data.get("groups", {}).get("gcr", group) != group:
        raise ValueError("combined baseline timings already contain a different GCR price")
    data["groups"]["gcr"] = group
    destination.mkdir(parents=True)
    study.write_sealed(destination / "baseline-timings.json", data)
    study.write_sealed(destination / "verification.json", audit)
    study.write_sealed(destination / "descriptive-wall.json", describe_pricing(stage))
    record = {
        "artifact_kind": "xgraph.webqsp_gcr_cohort_decision",
        "amendment": 6,
        "parent": study.identity(parent_path),
        "dated_amendment": registration["amendment"],
        "decided_at": datetime.now(UTC).isoformat(),
        "seed": 17,
        "gcr": decision,
        "pending": [
            "core_pilot",
            *[g for g in registration["pricing"]["baseline_order"] if g != "gcr"],
        ],
        "execution_sealed": False,
        "verification": study.identity(destination / "verification.json"),
        "descriptive_only": study.identity(destination / "descriptive-wall.json"),
        "baseline_timing_snapshot": study.identity(destination / "baseline-timings.json"),
        "combined_baseline_timing_path": str(combined),
        "execution_core_reserve": {
            "reserve_seconds": reserve,
            "qualification_seconds": audit["qualification_seconds"],
            "pricing_seconds": audit["pricing_seconds"],
            "initial_seconds": initial,
            "remaining_seconds": reserve - initial,
            "application": "prepare adds native_pricing_charge to the owner's qualification "
            "charge; no execution seal here",
        },
        "provenance": provenance(
            [parent_path, Path(__file__), study.ROOT / "scripts/webqsp-amendment-6.py"]
        ),
        "command": [sys.executable, "-I", *sys.argv],
    }
    path = destination / "decision.json"
    study.write_sealed(path, record)
    study.atomic_json(combined, data)
    if study.native_pricing_charge(registration, combined) != audit["pricing_seconds"]:
        raise ValueError("execution reserve failed to retain the complete pricing charge")
    return path
