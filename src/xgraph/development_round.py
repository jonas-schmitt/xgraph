"""CPU validation and immutable bookkeeping for the exposed dev-r2 comparison."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from collections import Counter
from pathlib import Path

AGENTS = ("A-v2", "A-v2r", "A-v2r16")
TOOL_BUDGETS = {"A-v2": 8, "A-v2r": 8, "A-v2r16": 16}
QUALIFICATION_RULE = (
    "Select separately per family only after all three complete CWQ panels pass "
    "writer/accounting checks. Maximize CWQ mean ranked Hits@1 using recorded "
    "agent-answer-v2. Exact hit-count ties: lower tool budget, then lower summed "
    "prompt+generated tokens, then A-v2, A-v2r, A-v2r16. No WebQSP outcome, "
    "WT/WT-D score, F1, subgroup or historical result enters selection. Missing "
    "candidate or invalid provenance/accounting means INCOMPLETE, never selection "
    "among a surviving subset. Empty/failed/truncated answers remain zero."
)


def qualify_agents(candidates, expected_questions):
    """Apply the prospective family-local rule to complete scored CWQ candidates."""
    incomplete = {"status": "INCOMPLETE", "selected": None, "rule": QUALIFICATION_RULE}
    if set(candidates) != set(AGENTS):
        return incomplete
    for name, row in candidates.items():
        if row.get("dataset") != "cwq":
            raise ValueError("agent qualification consumes only CWQ scores")
        if not row.get("complete") or not row.get("writer_validated"):
            return incomplete
        if row.get("questions") != expected_questions:
            return incomplete
        if (
            type(row.get("hits")) is not int
            or not 0 <= row["hits"] <= expected_questions
            or type(row.get("total_tokens")) is not int
            or row["total_tokens"] < 0
        ):
            raise ValueError(f"{name}: invalid qualification hits or token accounting")
    order = sorted(
        AGENTS,
        key=lambda a: (
            -candidates[a]["hits"],
            TOOL_BUDGETS[a],
            candidates[a]["total_tokens"],
            AGENTS.index(a),
        ),
    )
    return {
        "status": "DEVELOPMENT_SELECTED",
        "selected": order[0],
        "order": order,
        "rule": QUALIFICATION_RULE,
    }


def score_panel(records, questions, *, candidate=None):
    """Use the recorded parser and failed-zero label metrics, with conserved usage."""
    from xgraph.agent_answers import V2, record_parse
    from xgraph.agent_final import VERSION
    from xgraph.mintqa_rematch import COST_FIELDS, audit_complete_usage, usage
    from xgraph.webqsp import score_records

    ids = [q["id"] for q in questions]
    if not ids:
        raise ValueError("development scoring needs a nonempty panel")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate panel IDs")
    audit_complete_usage(records, ids)
    if candidate is not None:
        if candidate not in AGENTS:
            raise ValueError(f"unknown agent candidate: {candidate}")
        for row in records:
            if row.get("answer_parser_version") != V2:
                raise ValueError(f"{row['id']}: candidate must record parser {V2}")
            parsed = record_parse(row, V2)
            if (
                row.get("answers") != parsed.answers
                or row.get("answer_parse_status") != parsed.status
            ):
                raise ValueError(f"{row['id']}: recorded parser result changed")
            controller = row.get("final_answer_controller_version")
            if controller != (None if candidate == "A-v2" else VERSION):
                raise ValueError(f"{row['id']}: wrong final-answer controller")
            calls = row.get("tool_calls")
            if type(calls) is not int or not 0 <= calls <= TOOL_BUDGETS[candidate]:
                raise ValueError(f"{row['id']}: invalid tool budget accounting")
            if len(row.get("actions", [])) != calls:
                raise ValueError(f"{row['id']}: actions do not conserve tool calls")
    score = score_records(records, questions, failed_zero=True)
    totals = {k: sum(usage(r)[k] for r in records) for k in COST_FIELDS}
    return {
        **score,
        "complete": True,
        "writer_validated": True,
        "hits": sum(int(r["hits_at_1"]) for r in score["per_question"]),
        "total_tokens": totals["prompt_tokens"] + totals["generated_tokens"],
        "usage": totals,
        "coverage": sum(
            bool(r["prediction"]) and not r["failed_output"] for r in score["per_question"]
        )
        / len(ids),
        "answer_parse_status": dict(
            Counter(r.get("answer_parse_status", "not_agent") for r in records)
        ),
        "terminal_response": dict(Counter(r.get("finish_reason", "missing") for r in records)),
        "tool_budget_reached": sum(
            r.get("tool_calls") == TOOL_BUDGETS.get(candidate) for r in records
        )
        if candidate
        else None,
        "final_answer_retries": sum(r.get("final_answer_retries", 0) for r in records),
        "revision_accepted": sum(bool(r.get("revision_accepted")) for r in records),
    }


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def verify(pin):
    if identity(pin["path"]) != pin:
        raise ValueError(f"changed artifact: {pin['path']}")


def run_job(job, root, plan_sha256):
    """Fail closed on partial attempts and hash-check completed stages on resume."""
    root = Path(root)
    key = job["key"]
    dependencies = [identity(root / "done" / f"{k}.json") for k in job["depends_on"]]
    binding = {
        "plan_sha256": plan_sha256,
        "job_sha256": digest(job),
        "inputs": [identity(p) for p in job["inputs"]],
        "dependencies": dependencies,
    }
    receipt = root / "done" / f"{key}.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["binding"] != binding:
            raise ValueError(f"{key}: changed job/input/dependency binding")
        if [p["path"] for p in saved["outputs"]] != [
            str(Path(p).resolve()) for p in job["outputs"]
        ]:
            raise ValueError(f"{key}: changed output inventory")
        for pin in saved["outputs"]:
            verify(pin)
        return "SKIP"
    attempt = root / "attempts" / f"{key}.json"
    if attempt.exists() or any(Path(p).exists() for p in job["outputs"]):
        raise ValueError(f"{key}: partial or failed attempt; automatic retry refused")
    write_once(attempt, {"binding": binding, "job": job, "started_unix": time.time()})
    log = root / "logs" / f"{key}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    for path in job["outputs"]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    rc = None

    def event(status):
        with (root / "timings.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "job": key,
                        "kind": job["kind"],
                        "status": status,
                        "unix": time.time(),
                        "wall_seconds": time.monotonic() - start,
                        "returncode": rc,
                        "boundary": "launcher wall including CPU build/admission/waits; "
                        "not GPU latency",
                        "plan_sha256": plan_sha256,
                    }
                )
                + "\n"
            )

    event("START")
    env = {**os.environ, **job["env"], "PYTHONDONTWRITEBYTECODE": "1"}
    if job["kind"] == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
        state = job["env"].get("XG_CONTAINER_STATE")
        if state:
            Path(state).mkdir(parents=True, exist_ok=False)
    try:
        with log.open("x") as stream:
            process = subprocess.Popen(
                job["command"],
                env=env,
                cwd=job.get("cwd"),
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                rc = process.wait(timeout=job["wall_timeout_seconds"])
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        if rc:
            raise RuntimeError(f"{key}: stage failed ({rc}); see {log}")
        if any(not Path(p).is_file() or not Path(p).stat().st_size for p in job["outputs"]):
            raise ValueError(f"{key}: missing expected output")
        write_once(
            receipt,
            {
                "binding": binding,
                "outputs": [identity(p) for p in job["outputs"]],
                "launcher_wall_seconds": time.monotonic() - start,
            },
        )
        event("DONE")
        return "DONE"
    except BaseException:
        event("FAILED")
        raise
