"""Read-only consumers of completed readiness evidence; never launch experiments.

Numerical rules come from the producer registrations. Where a registration leaves
an admission rule unspecified, report that blocker instead of inventing a gate.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

from xgraph.provenance import provenance, sha256_file

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("qwen38", "gemma31")
REFERENCES = {"qwen38": "agent-strong-8", "gemma31": "agent-feedback-8"}
CUDA_STAGES = (
    "data-directory",
    "codex-binding",
    "tokenizer-binding",
    "mquake-binding",
    "gcr-hf-prepare",
    "gcr-prepare",
    "gcr-sampling-prepare",
    "gcr-hf-anchor-index",
    "gcr-hf-anchors",
    "gcr-hf-group-anchor-index",
    "gcr-hf-group-anchors",
    "gcr-sampling-anchor-index",
    "gcr-sampling-anchors",
    "rdf-prepare",
    "rdf-requests",
    "rdf-anchors",
    "events-directory",
    "native-build",
    "gpu-stable-gate",
    "stride-case-alias-agreement",
    "stride-case-alias-no-skips",
    "anchor-agreement",
    "anchor-no-skips",
    "replay-status-agreement",
    "replay-status-no-skips",
    "f04-cuda-agreement",
    "f04-cuda-no-skips",
    "api-p0-cuda",
    "api-p0-no-skips",
    "rdf-agreement",
    "rdf-no-skips",
    "real-vllm-adapter",
    "api-p0-strict-serving",
    "api-p0-strict-check",
    "native-reader-cuda",
    "native-reader-no-skips",
    "gcr-hf",
    "gcr-hf-check",
    "gcr-hf-group",
    "gcr-hf-group-check",
    "gcr-sampling",
    "gcr-sampling-check",
    "rdf-serving",
    "rdf-serving-check",
    "occupancy-events-check",
    "feedback-adapter",
    "stella-cuda-parity",
)


def incomplete(reason):
    return {"status": "INCOMPLETE", "reason": reason}


def positive_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def serving_decision(plans, reports):
    """The sealed cost conjunction, separate from unregistered live-quality admission."""
    cells = []
    keys = ("arm", "cache", "concurrency", "repeat", "seed")
    try:
        if set(plans) != set(FAMILIES) or set(reports) != set(FAMILIES):
            return incomplete("both locked family matrices are required")
        for family in FAMILIES:
            report = reports[family]
            expected = [tuple(t[k] for k in keys) for t in plans[family]["trials"]]
            actual = [tuple(t[k] for k in keys) for t in report["trials"]]
            if (
                not expected
                or len(set(expected)) != len(expected)
                or actual != expected
                or report["status"] != "completed"
                or any(t["status"] != "completed" for t in report["trials"])
            ):
                return incomplete(f"{family}: missing, failed or changed registered repetitions")
            for concurrency in (1, 8):
                ratios = [
                    r
                    for r in report["cost_ratios"]
                    if (r["numerator"], r["denominator"], r["cache"], r["concurrency"])
                    == ("WT", REFERENCES[family], "warm", concurrency)
                ]
                if len(ratios) != 1 or ratios[0]["status"] != "completed":
                    return incomplete(f"{family}/C{concurrency}: one complete cost ratio required")
                ratio = ratios[0]["whole_trial_ratio"]
                low, high = ratio["low"], ratio["high"]
                if not positive_number(low) or not positive_number(high) or low > high:
                    return incomplete(f"{family}/C{concurrency}: invalid cost interval")
                repeats = {
                    t["repeat"]
                    for t in report["trials"]
                    if t["arm"] == "WT" and t["cache"] == "warm" and t["concurrency"] == concurrency
                }
                if len(repeats) < 3:
                    return incomplete("the producer requires at least three complete repeat pairs")
                cells.append(
                    {
                        "family": family,
                        "concurrency": concurrency,
                        "low": low,
                        "high": high,
                        "savings": high < 1,
                        "half_cost": high <= 0.5,
                    }
                )
    except (KeyError, TypeError):
        return incomplete("missing locked serving schema fields")
    return {
        "status": "PASS" if all(c["savings"] for c in cells) else "FAIL",
        "decision_scope": "registered whole-trial cost savings only",
        "half_cost": all(c["half_cost"] for c in cells),
        "cells": cells,
        "quality_assessed": False,
        "quality_requirement": "The seal requires separately scored live answers. Report their "
        "quality and the W/B2/agent frontier alongside measured cost; do not inherit validation "
        "accuracy. No binary joint live-quality/cost admission rule is registered, and this "
        "cost consumer supplies no post-hoc threshold.",
    }


def gcr_decision(ids, evidence):
    """Qualify finished HF cells independently of the rejected old/new optimization."""
    if len(ids) != 8 or len(set(ids)) != 8:
        return incomplete("the registered eight ordered benchmark ids are required")
    costs = {}
    try:
        for mode in ("hf", "hf-group"):
            cell = evidence[mode]
            timing = cell["timing"]
            if (
                cell["ids"] != ids
                or timing["variant"] != mode
                or timing["exit_status"] != 0
                or timing["cpu_launch_check"] is not False
                or cell["cpu_fixture"]
                or cell["semantics"] != mode
                or not positive_number(cell["seconds"])
                or not positive_number(timing["wall_seconds"])
            ):
                return incomplete(f"{mode}: complete native results and charged timing required")
            costs[mode] = {
                "questions": len(ids),
                "generation_seconds": cell["seconds"],
                "launcher_seconds": timing["wall_seconds"],
            }
    except (KeyError, TypeError):
        return incomplete("both hf and hf-group cells are required")
    return {
        "status": "BLOCKED_REGISTRATION",
        "benchmark_complete": True,
        "preferred_semantics": "hf-group",
        "webqsp_admitted": False,
        "costs": costs,
        "reason": "The released launcher uses hf-group. The registration's pending semantics "
        "slot still requires a prospective amendment and measured WebQSP core pricing. "
        "There is no registered numeric MINTQA-benchmark cutoff or automatic HF fallback.",
        "old_new_identity_required": False,
        "limits": "Different HF/group outputs are expected semantic changes. This benchmark "
        "does not establish published-score reproduction or transferable WebQSP runtime.",
    }


def cuda_decision(commit, expected_commit, status_lines, logs):
    """All owner stages must finish on the intended integration commit, with native tests."""
    if commit.strip() != expected_commit or not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        return incomplete("CUDA artifacts belong to a different integration commit")
    parsed = []
    for line in status_lines:
        match = re.fullmatch(r"([a-z0-9-]+) exit ([0-9]+) \S+", line.strip())
        if match is None:
            return incomplete("malformed CUDA stage status")
        parsed.append((match[1], int(match[2])))
    if any(code for _, code in parsed):
        return {"status": "FAIL", "failed_stages": [name for name, code in parsed if code]}
    if [name for name, _ in parsed] != list(CUDA_STAGES):
        return incomplete("every ordered CUDA owner stage must finish exactly once")
    for name in CUDA_STAGES:
        if not isinstance(logs.get(name), str) or not logs[name].strip():
            return incomplete(f"missing native stage log: {name}")
    for name in (
        "stride-case-alias-agreement",
        "anchor-agreement",
        "replay-status-agreement",
        "f04-cuda-agreement",
        "api-p0-cuda",
        "rdf-agreement",
        "native-reader-cuda",
        "stella-cuda-parity",
    ):
        log = logs[name]
        if not re.search(r"\b[1-9][0-9]* passed\b", log) or re.search(
            r"\b[1-9][0-9]* skipped\b", log
        ):
            return incomplete(f"{name}: required native tests did not all execute")
    gate = logs["gpu-stable-gate"]
    if "Summary: 6 passed, 0 failed" not in gate or "Stable quality gate passed." not in gate:
        return incomplete("full GPU stable gate did not pass")
    # Host vLLM skips are covered by the required native-container stage above.
    # Engine agreement and the formerly CPU-excluded CUDA modules must execute
    # in the host GPU gate itself; a passed container cannot replace those tests.
    if re.search(
        r"SKIPPED.*(?:tests/test_(?:engine|codex|drafts|policy|pressure|turtle)\.py"
        r"|CUDA (?:unavailable|not available)|no (?:CUDA|NVIDIA|GPU)|without.*GPU engine)",
        gate,
        re.IGNORECASE,
    ):
        return incomplete("required GPU agreement was skipped")
    return {
        "status": "PASS",
        "commit": expected_commit,
        "stages": list(CUDA_STAGES),
        "decision_scope": "owner CUDA agreement; permits review of main integration, "
        "not a scientific result or serving performance claim",
    }


def cwq_decision(draft, timings, score=None, checks=None):
    """Consume the prospective amendment; missing evidence is INCOMPLETE."""
    from xgraph.cwq_test import admission_decision

    try:
        return admission_decision(draft, timings, score, checks)
    except (ValueError, KeyError, TypeError) as error:
        return {"status": "INCOMPLETE", "reason": str(error), "admitted": False}


def webqsp_pilot_pin(registration, plan):
    """A repaired producer needs an explicit amendment, even when its recipes agree."""
    saved = plan.get("provenance", {})
    if saved.get("repo_dirty") is not False:
        return incomplete("validation pilot must come from a recorded clean commit")
    if saved.get("repo_commit") != registration["pilot"]["commit"]:
        return {
            "status": "BLOCKED_REGISTRATION",
            "registered_commit": registration["pilot"]["commit"],
            "measured_commit": saved.get("repo_commit"),
            "reason": "Prospectively amend the repaired pilot producer pin, preserving the "
            "original registration, repair rationale, source diff and input/recipe identities. "
            "Do not bypass verify_pilot or change the cohort using outcomes.",
        }
    return None


def project_queue(stages, as_of):
    """Scenario dates from remaining forecasts; elapsed waits never count as finished work."""
    low = high = datetime.fromisoformat(as_of)
    if low.tzinfo is None:
        raise ValueError("queue projection requires a timezone-aware audit timestamp")
    result = []
    for stage in stages:
        a, b = stage["remaining_hours"]
        gap = stage["handoff_seconds"]
        if not positive_number(a) or not positive_number(b) or a > b or gap < 0:
            raise ValueError("invalid remaining workload range or handoff")
        start = [low.isoformat(), high.isoformat()]
        low, high = low + timedelta(hours=a), high + timedelta(hours=b)
        result.append({**stage, "start": start, "end": [low.isoformat(), high.isoformat()]})
        low, high = low + timedelta(seconds=gap), high + timedelta(seconds=gap)
    return result


class Evidence:
    """Record consumed bytes and refuse changed frozen inputs or completion receipts."""

    def __init__(self):
        self.inputs = []

    def file(self, path, expected=None):
        path = Path(path).resolve()
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError(f"frozen input hash changed: {path}")
        self.inputs.append({"path": str(path), "sha256": digest, "bytes": path.stat().st_size})
        return path

    def text(self, path, expected=None):
        return self.file(path, expected).read_text()

    def json(self, path, expected=None):
        return json.loads(self.text(path, expected))

    def frozen(self, reference):
        return self.json(reference["path"], reference["sha256"])


def consume_serving(spec, evidence):
    seal = evidence.frozen(spec["seal"])
    if seal["analysis"]["savings"] != "whole_trial_ratio.high < 1":
        raise ValueError("serving savings rule differs from the audited registration")
    plans, reports = {}, {}
    for family in FAMILIES:
        name = f"locked-{family}"
        stage = next(s for s in seal["stages"] if s["name"] == name)
        receipt = evidence.json(Path(seal["out"]) / "receipts" / f"{name}.json")
        if (
            receipt["seal_sha256"] != spec["seal"]["sha256"]
            or receipt["command"] != stage["command"]
        ):
            raise ValueError(f"{name}: completion receipt changed seal or command")
        output = stage["output"]
        if output not in receipt["files"]:
            raise ValueError(f"{name}: report absent from completion receipt")
        # Check the complete receipt without parsing or printing retained completions.
        for path, expected in receipt["files"].items():
            if sha256_file(Path(path)) != expected:
                raise ValueError(f"{name}: completed artifact hash changed: {path}")
        reports[family] = evidence.json(output, receipt["files"][output])
        plan = str(Path(stage["plan"]).with_name("trials.json"))
        plans[family] = evidence.json(plan, seal["files"][plan])
    return serving_decision(plans, reports)


def consume_gcr(spec, evidence):
    manifest = evidence.frozen(spec["manifest"])
    root = Path(spec["root"])
    variants = {}
    for mode in ("hf", "hf-group"):
        path = root / f"{mode}.jsonl"
        ids, fixture = [], False
        # Stream full beam audits rather than loading the matrix into memory.
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                ids.append(row["id"])
                fixture |= row.get("status") == "cpu_launch_fixture"
        evidence.inputs.append(
            {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        )
        metadata = evidence.json(Path(f"{path}.provenance.json"))
        variants[mode] = {
            "ids": ids,
            "cpu_fixture": fixture,
            "semantics": metadata["gcr"]["beam_semantics"],
            "seconds": evidence.json(Path(f"{path}.summary.json"))["seconds"],
            "timing": evidence.json(root / f"{mode}.timing.json"),
        }
    return gcr_decision(manifest["ids"], variants)


def consume_webqsp(spec, evidence):
    from xgraph.webqsp_test import matched_prices, size_protocol, verify_pilot

    registration = evidence.frozen(spec["registration"])
    plan = evidence.json(Path(spec["timings"]).with_name("queue.json"))
    timings = evidence.json(spec["timings"])
    blocker = webqsp_pilot_pin(registration, plan)
    if blocker:
        return blocker
    verify_pilot(registration, Path(spec["timings"]))
    gate = evidence.json(spec["development_gate"]) if spec.get("development_gate") else None
    rates = None
    if spec.get("baseline_timings"):
        evidence.json(spec["baseline_timings"])
        rates = matched_prices(registration, spec["baseline_timings"], spec["timings"])
    size = size_protocol(registration, timings, gate, baseline_rates=rates)
    pending = [
        slot["name"]
        for slot in registration.get("amendment_slots", [])
        if slot.get("status") == "PENDING" and slot.get("blocks_execution", True)
    ]
    pending.extend(
        name + " development gate"
        for name, decision in size.get("optional_admission", {}).items()
        if decision["reason"] == "development gate pending"
    )
    if size.get("cohort_rule") == "two_tier":
        pending.extend(
            name + " " + decision["status"]
            for name, decision in size["baseline_admission"].items()
            if decision["status"].startswith("PENDING")
        )
        if (
            not gate
            or gate.get("native_qualification", {}).get("families")
            != {"qwen38": True, "gemma31": True}
            or not gate["native_qualification"].get("evidence")
        ):
            pending.append("native checks in both families")
        if size.get("core_power_qualified") is False:
            pending.append("core power below registered 80% transport screen")
    return {
        "status": "PASS",
        "decision_scope": "cost sizing only; not execution admission or measured assurance",
        "size": size,
        "pending_execution": pending,
        "reason": "The registered pilot pin and complete timing matrix permit sizing. "
        "Pending execution slots are listed from this registration, not inferred from the "
        "superseded amendment. Final execution sealing remains the producer's command.",
        "outcomes_used": False,
    }


def consume_cuda(spec, evidence):
    root, source = Path(spec["root"]), Path(spec["source"])
    expected = {Path(ref["path"]).resolve(): ref["sha256"] for ref in spec["frozen_inputs"]}
    recorded = {}
    for line in evidence.text(root / "input-sha256.txt").splitlines():
        digest, path = line.split(maxsplit=1)
        path = Path(path.lstrip("*"))
        recorded[(path if path.is_absolute() else source / path).resolve()] = digest
    if recorded != expected:
        raise ValueError("CUDA owner input hashes differ from the audited source")
    for path, digest in expected.items():
        evidence.file(path, digest)
    logs = {name: evidence.text(root / f"{name}.log") for name in CUDA_STAGES}
    report = evidence.json(source / "target/vllm-smoke/report.json")
    receipts = report.get("request_replay_status") if isinstance(report, dict) else None
    if (
        not isinstance(report, dict)
        or report.get("request_replay_status_checked") is not True
        or not isinstance(receipts, list)
        or not receipts
        or any(
            not isinstance(row, dict)
            or not isinstance(row.get("id"), str)
            or not row["id"]
            or row.get("replay_status") != "replayed"
            or type(row.get("prompt_tokens")) is not int
            or row["prompt_tokens"] <= 0
            or type(row.get("generated_tokens")) is not int
            or row["generated_tokens"] < 0
            for row in receipts
        )
        or len({row["id"] for row in receipts}) != len(receipts)
    ):
        return incomplete("ordinary generation's worker replay status was not confirmed")
    return cuda_decision(
        evidence.text(root / "commit.txt"),
        spec["commit"],
        evidence.text(root / "status.log").splitlines(),
        logs,
    )


def consume_cwq(spec, evidence):
    result = cwq_decision(
        evidence.frozen(spec["draft"]),
        evidence.json(spec["timings"]),
        evidence.json(spec["score"]),
        evidence.frozen(spec["checks"]),
    )
    result["provenance"] = provenance([Path(p["path"]) for p in evidence.inputs])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", choices=("serving", "webqsp", "gcr", "cuda", "cwq"))
    parser.add_argument(
        "--inputs",
        type=Path,
        required=True,
        help="hash-pinned producer registrations and finished-artifact locations",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="report within this checkout's target"
    )
    args = parser.parse_args(argv)
    if not args.out.resolve().is_relative_to((ROOT / "target").resolve()):
        print(json.dumps(incomplete("decision output must stay within this checkout's target")))
        return 2
    evidence = Evidence()
    try:
        spec = evidence.json(args.inputs)[args.gate]
        if args.gate == "cwq":
            result = consume_cwq(spec, evidence)
        else:
            result = {
                "serving": consume_serving,
                "webqsp": consume_webqsp,
                "gcr": consume_gcr,
                "cuda": consume_cuda,
            }[args.gate](spec, evidence)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        result = incomplete(str(error))
    result.update(
        gate=args.gate,
        inputs=evidence.inputs,
        decision_code_sha256=sha256_file(Path(__file__)),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "out": str(args.out)}))
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2, "BLOCKED_REGISTRATION": 3}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
