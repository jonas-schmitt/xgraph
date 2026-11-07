"""CPU planning and opt-in serial execution of the bounded WebQSP comparison."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager, suppress
from pathlib import Path

from xgraph.webqsp import _json, _read_jsonl, check_budget

LIMITATIONS = [
    "Primary: RoG/GCR label protocol on released subgraphs (gcr_hit/gcr_f1); ranked Hits@1 "
    "is secondary. Native-MID F1 is out of scope because names cannot recover MIDs.",
    "Native empty-gold omissions remain in the extended panel; "
    "primary released scores list exclusions.",
    "The engine caps live frontiers. Set delivery reports cap encounters and endpoint loss; "
    "the uncapped CPU sensitivity cannot repair online relation decisions.",
    "GCR transfer uses reconstructed prompts and joint beam, not upstream group-beam. "
    "Native-checkpoint transfer with a local synthesis model is not published-score reproduction.",
    "Native GCR uses input-order DiGraph collapse; shared GCR preserves parallel relations. "
    "Both consume the same source triples; this representation difference is explicit.",
    "GCR beam refuses banned tokens, including Gemma's channel ban. Inspect thinking/truncations.",
    "Reader budgets exceed the single-call planner's; report calls/tokens and dedicated GPU time.",
    "All arms request complete sets. Engine/matched delivery uses answer support after commitment; "
    "singleton delivery and exclusion of gold sets over the engine cap are sensitivities.",
    "No published GCR reproduction claim unless full native-checkpoint validation falls within "
    "its reported range under a documented matched protocol; "
    "a reduced pilot cannot establish this.",
    "No GPU serving fidelity or quality has been validated by this CPU preparation.",
]


def pilot_stratum(question: dict) -> str:
    """Physical gold distance (CVTs retained) crossed with released answer-set size."""
    hops = question.get("coverage", {}).get("shortest_hops", {})
    hop = str(max(hops.values())) if hops else "unreachable"
    count = len(set(question["answers"]))
    size = (
        "0"
        if not count
        else "1"
        if count == 1
        else "2-5"
        if count <= 5
        else "6-64"
        if count <= 64
        else ">64"
    )
    return f"hops={hop},answers={size}"


def pilot_subset(panel: list[dict], count: int, seed: int) -> list[dict]:
    """Seeded, input-order-independent proportional sample, keeping rare strata."""
    if not 0 < count <= len(panel):
        raise ValueError(f"pilot needs {count} questions, panel contains {len(panel)}")
    if len({q["id"] for q in panel}) != len(panel):
        raise ValueError("duplicate pilot question ids")
    groups = defaultdict(list)
    for q in panel:
        groups[pilot_stratum(q)].append(q)
    if count < len(groups):
        raise ValueError("pilot budget cannot cover every hop/answer-size stratum")
    allocated = dict.fromkeys(sorted(groups), 1)
    for _ in range(count - len(groups)):
        key = max(
            (k for k in allocated if allocated[k] < len(groups[k])),
            key=lambda k: (count * len(groups[k]) / len(panel) - allocated[k], k),
        )
        allocated[key] += 1
    selected = []
    for key, n in allocated.items():
        selected.extend(
            sorted(
                groups[key], key=lambda q: hashlib.sha256(f"{seed}:{q['id']}".encode()).digest()
            )[:n]
        )
    return sorted(selected, key=lambda q: q["id"])


def queue_plan(
    data,
    out,
    models,
    gcr_model,
    cor_prompts=None,
    *,
    pilot=False,
    pilot_questions=100,
    pilot_seed=0,
    pilot_budget_hours=8,
    project_questions=None,
    catalog=False,
    arms=None,
    pilot_timings=None,
    policy_freeze=None,
    timing_root=Path("/home/jonas/git/xgraph"),
    dog_prompts=None,
    secondary_pilot=False,
):
    """Pure command construction; preflight and execution are separate and explicit."""
    if catalog:
        from xgraph.webqsp_catalog import catalog_plan

        return catalog_plan(
            data,
            out,
            models,
            pilot=pilot,
            pilot_questions=pilot_questions,
            pilot_seed=pilot_seed,
            pilot_budget_hours=pilot_budget_hours,
            project_questions=project_questions,
            arms=arms,
            pilot_timings=pilot_timings,
            policy_freeze=policy_freeze,
            timing_root=timing_root,
            gcr_model=gcr_model,
            **({"dog_prompts": dog_prompts} if dog_prompts else {}),
            secondary_pilot=secondary_pilot,
        )
    data, out = Path(data).resolve(), Path(out).resolve()
    jobs = []
    manifest = data / "manifest.json"
    metadata = json.loads(manifest.read_text()) if manifest.is_file() else {}
    questions = metadata.get("counts", {}).get("questions", 0)
    cohort = []
    if pilot:
        if metadata.get("split") != "validation":
            raise ValueError(
                "--pilot requires a validation split manifest; test sampling is forbidden"
            )
        if set(models) != {"qwen25", "gemma31"} or cor_prompts is None:
            raise ValueError("--pilot requires both families and --cor-prompts for all arms")
        if not 0 < pilot_budget_hours <= 8:
            raise ValueError("pilot GPU budget must be positive and at most 8 hours")
        panel = json.loads((data / "questions.json").read_text())
        cohort = pilot_subset(panel, pilot_questions, pilot_seed)
        questions = len(cohort)
    total_low, total_high = 0.0, 0.0
    for family, model in models.items():
        if family not in {"qwen25", "gemma31"}:
            raise ValueError(f"unknown model family {family!r}")
        model = Path(model).resolve()
        index, tokenizer = data / f"index-{family}", model / "tokenizer.json"
        folder = out / family
        kv = "16" if family == "gemma31" else "8"
        base = {
            "XG_TOOLS_DB": str(data / "kg.sqlite"),
            "XG_KG_VERSIONS": str(data / "versions"),
            "XG_KG_INDEX": str(index),
            "XG_TOKENIZER": str(tokenizer),
            "XG_RELATION_LABELS": str(data / "kg" / "relation_labels.jsonl"),
            "XG_NEED_GIB": "75" if family == "gemma31" else "40",
            "XG_GPU_IDLE": "1",
            "XG_PATHS": "0",
            "XG_DERIVATION": "0",
            "XG_OPEN": "0",
            "XG_ESCAPES": "1",
            "XG_COMMITTED_PATHS": "0",
            "XG_BOUND_ESCAPES": "1",
            "XG_REPEATS": "1",
            "XG_START_RANK": "degree",
            "XG_UNQUOTED_STARTS": "1",
            "XG_RELATIONS": "frontier",
            "XG_END_AFTER_ANSWER": "0",
            "XG_STOP": "",
            "XG_EXPRESSIONS": "0",
            "XG_LORA": "",
            "XG_ESCAPE_BIAS": "",
            "XG_RELEASE_MASS": "",
            "XG_RELEASE_TOP": "0",
            "XG_GCR_SYNTHESIS_MODEL": "",
        }
        common = ["--seed", "0", "--max-model-len", "16384", "--kv-cache-gib", kv]

        def cpu(name, args, *, family=family):
            jobs.append(
                {
                    "kind": "cpu",
                    "family": family,
                    "name": name,
                    "env": {
                        "OPENBLAS_NUM_THREADS": "1",
                        "OMP_NUM_THREADS": "1",
                        "TOKENIZERS_PARALLELISM": "false",
                        "CUDA_VISIBLE_DEVICES": "",
                    },
                    "command": [
                        "nice",
                        "-n",
                        "10",
                        "prlimit",
                        f"--as={8000000000 if pilot else 4294967296}",
                        sys.executable,
                        "-m",
                        "xgraph.webqsp",
                        *map(str, args),
                    ],
                }
            )

        def gpu(
            name,
            arm,
            request_kind,
            extra,
            seconds,
            settings=None,
            first_model=None,
            *,
            base=base,
            folder=folder,
            model=model,
            common=common,
            family=family,
            cpu=cpu,
        ):
            nonlocal total_low, total_high
            env = {**base, **(settings or {})}
            raw = folder / f"{name}.jsonl"
            command = [
                "./scripts/generate.sh",
                str(first_model or model),
                str(data / f"requests-{request_kind}.jsonl"),
                arm,
                str(raw),
                *common,
                *extra,
            ]
            if family == "gemma31" and arm != "gcr":
                command += ["--ban-tokens", "<|channel>"]
            jobs.append(
                {
                    "kind": "gpu",
                    "family": family,
                    "name": name,
                    "env": env,
                    "command": command,
                    "assumed_seconds_per_question": seconds,
                    "questions": questions,
                }
            )
            total_low += questions * seconds[0] + 90
            total_high += questions * seconds[1] + 180
            cpu(
                name + "-score",
                [
                    "score",
                    "--outputs",
                    raw,
                    "--questions",
                    data / "questions.json",
                    "--out",
                    folder / f"{name}.score.json",
                    "--released-only",
                ],
            )
            cpu(
                name + "-full-panel-score",
                [
                    "score",
                    "--outputs",
                    raw,
                    "--questions",
                    data / "questions.json",
                    "--out",
                    folder / f"{name}.full-panel.score.json",
                ],
            )
            return raw

        for name, support in (("paths", "frontier"), ("vocabulary", "vocabulary"), ("free", None)):
            settings = {
                "XG_PATHS": "1",
                "XG_COMMITTED_PATHS": "1",
                "XG_ESCAPES": "0",
                "XG_OPEN": "1",
                "XG_END_AFTER_ANSWER": "1",
                "XG_RELATIONS": support or "frontier",
            }
            raw = gpu(
                name,
                "constrained" if support else "baseline",
                "paths",
                ["--max-tokens", "256", *([] if support else ["--open"])],
                [0.4, 2],
                settings,
            )
            sets = folder / f"{name}-sets.jsonl"
            cpu(
                name + "-sets",
                [
                    "deliver",
                    "--outputs",
                    raw,
                    "--requests",
                    data / "requests-paths.jsonl",
                    "--db",
                    data / "kg.sqlite",
                    "--index",
                    index,
                    "--tokenizer",
                    tokenizer,
                    "--kg-versions",
                    data / "versions",
                    "--out",
                    sets,
                ],
            )
            cpu(
                name + "-sets-score",
                [
                    "score",
                    "--outputs",
                    sets,
                    "--questions",
                    data / "questions.json",
                    "--out",
                    folder / f"{name}-sets.score.json",
                    "--released-only",
                ],
            )
        reader = [
            "--max-tokens",
            "512",
            "--max-calls",
            "8",
            "--start-rank",
            "degree",
            "--unquoted-starts",
        ]
        gpu("agent-feedback", "agent", "agent", [*reader, "--agent-feedback"], [2, 14])
        gpu(
            "agent-schema-turn",
            "agent",
            "agent",
            [*reader, "--agent-feedback", "--agent-schema", "turn"],
            [2, 14],
        )
        gpu("walk", "walk", "paths", [*reader, "--matched-paths", "--observe", "labels"], [1, 8])
        gpu("resume", "resume", "paths", [*reader, "--resume-max", "3"], [1, 10])
        gpu(
            "iterative",
            "iterative",
            "iterative",
            ["--max-tokens", "512", "--max-calls", "8"],
            [2, 14],
        )
        if cor_prompts is not None:
            gpu(
                "cor",
                "cor",
                "gcr",
                [
                    "--max-tokens",
                    "512",
                    "--max-calls",
                    "32",
                    "--cor-depth",
                    "4",
                    "--cor-prompts",
                    str(Path(cor_prompts).resolve()),
                ],
                [4, 24],
            )
        gcr_flags = [
            "--max-tokens",
            "512",
            "--gcr-k",
            "10",
            "--gcr-hops",
            "2",
            "--gcr-decoding",
            "beam",
            "--gcr-empty-trie",
            "open",
            "--gcr-max-paths",
            "100000",
            "--gcr-path-overflow",
            "error",
            "--gcr-batch-size",
            "1",
            "--gcr-parallel-relations",
            "preserve",
        ]
        gpu("gcr-shared", "gcr", "gcr", gcr_flags, [8, 45])
        gpu(
            "gcr-native-transfer",
            "gcr",
            "gcr",
            gcr_flags,
            [8, 45],
            {"XG_TOOLS_DB": str(data / "kg-gcr.sqlite"), "XG_GCR_SYNTHESIS_MODEL": str(model)},
            Path(gcr_model).resolve(),
        )
    pilot_info = None
    if pilot:
        gpu_jobs = [j for j in jobs if j["kind"] == "gpu"]
        per_question_high = sum(j["assumed_seconds_per_question"][1] for j in gpu_jobs)
        startups = len(gpu_jobs) * 180
        count = min(
            questions, math.floor((pilot_budget_hours * 3600 - startups) / per_question_high)
        )
        selected = pilot_subset(cohort, count, pilot_seed + 1)
        questions = len(selected)
        total_low = total_high = 0
        for job in jobs:
            if job["kind"] == "gpu":
                job["questions"] = questions
                low, high = job["assumed_seconds_per_question"]
                total_low += questions * low + 90
                total_high += questions * high + 180
            job["command"] = [
                str(out / "pilot" / Path(v).name)
                if v == str(data / "questions.json") or (str(v).startswith(str(data / "requests-")))
                else v
                for v in job["command"]
            ]
        available = Counter(pilot_stratum(q) for q in panel)
        sampled = Counter(pilot_stratum(q) for q in cohort)
        per_arm = Counter(pilot_stratum(q) for q in selected)
        pilot_info = {
            "seed": pilot_seed,
            "cohort_ids": [q["id"] for q in cohort],
            "per_arm_ids": [q["id"] for q in selected],
            "questions_per_arm": questions,
            "budget_gpu_hours": pilot_budget_hours,
            "strata": {
                k: {"available": available[k], "cohort": sampled[k], "per_arm": per_arm[k]}
                for k in sorted(available)
            },
            "rule": "same stratified subset for every arm and family; "
            "shrink uniformly using upper assumptions including startups",
        }
    return {
        "jobs": jobs,
        "primary_results": [
            {
                "family": job["family"],
                "arm": job["name"],
                "score_file": str(
                    out
                    / job["family"]
                    / (
                        job["name"]
                        + ("-sets" if job["name"] in {"paths", "vocabulary", "free"} else "")
                        + ".score.json"
                    )
                ),
            }
            for job in jobs
            if job["kind"] == "gpu"
        ],
        "questions": questions,
        "pilot": pilot_info,
        "project_questions": project_questions,
        "deadline_gate": "2026-12-10",
        "comparison_status": "set-aware released-label protocol prepared; "
        "GPU quality and fidelity unmeasured",
        "protocol": "RoG/GCR label protocol on released subgraphs",
        "primary_metrics": ["gcr_hit", "gcr_f1"],
        "gcr_published_reproduction": False,
        "limitations": LIMITATIONS,
        "estimate": {
            "measured": False,
            "questions_per_arm": questions,
            "low_gpu_hours": total_low / 3600,
            "high_gpu_hours": total_high / 3600,
            "startup_seconds_per_job": [90, 180],
            "planning_cap_gpu_hours": pilot_budget_hours if pilot else 80,
            "rule": "sum per-arm amortized seconds/question and startups; not a speed claim. "
            "Pilot validation first; do not reserve the full schedule without measured times.",
        },
    }


def preflight(data: Path, models: dict[str, Path], gcr_model: Path, *, catalog=False) -> list[Path]:
    """Read-only parity checks; no model load, CUDA import or GPU query."""
    from xgraph.kgversions import component_ranges
    from xgraph.tokenizer import ServedTokenizer

    if catalog:
        from xgraph.webqsp import verify_catalog

        saved = verify_catalog(data)
        for family, model in models.items():
            if saved["models"].get(family) != str(Path(model).resolve()):
                raise ValueError(f"{family}: catalog served model changed")
            header = json.loads((data / f"index-{family}/kg-index.json").read_text())
            if (
                header["tokenizer_fingerprint"]
                != ServedTokenizer.from_file(model / "tokenizer.json").fingerprint
            ):
                raise ValueError(f"{family}: tokenizer fingerprint mismatch")
        return [data / "catalog.json", *(Path(i["path"]) for i in saved["files"])]
    check_budget(data.parent)
    questions = json.loads((data / "questions.json").read_text())
    panel = {q["id"]: q for q in questions}
    if len(panel) != len(questions):
        raise ValueError("duplicate panel ids")
    components = component_ranges(data / "versions")
    inputs = [
        data / "manifest.json",
        data / "questions.json",
        data / "questions-released.json",
        data / "versions" / "kg-versions.json",
        data / "kg.sqlite",
        data / "kg-gcr.sqlite",
    ]
    for kind in ("paths", "agent", "iterative", "gcr"):
        path = data / f"requests-{kind}.jsonl"
        requests = _read_jsonl(path)
        if len(requests) != len(panel) or {r["id"] for r in requests} != set(panel):
            raise ValueError(f"{path}: question ids do not match the complete panel")
        for request in requests:
            if request.get("answer_mode") != "set":
                raise ValueError(f"{path}: stale singleton requests; run xgraph.webqsp requests")
            question = panel[request["id"]]
            if (
                request["kg_version"] != question["kg_version"]
                or request["anchor"] != question["anchor"]
            ):
                raise ValueError(f"{request['id']}: request graph/anchor differs from panel")
            first, stop = components[request["kg_version"]]
            if not request["anchor"] or any(
                not first <= int(e[1:]) < stop for e in request["anchor"]
            ):
                raise ValueError(f"{request['id']}: anchor outside its question component")
        inputs.append(path)
    for family, model in models.items():
        header = data / f"index-{family}" / "kg-index.json"
        tokenizer = model / "tokenizer.json"
        saved = json.loads(header.read_text())
        if saved["tokenizer_fingerprint"] != ServedTokenizer.from_file(tokenizer).fingerprint:
            raise ValueError(f"{family}: served tokenizer/index fingerprint mismatch")
        if saved["unwritable_triples"]:
            raise ValueError(
                f"{family}: unwritable triples change coverage; audit before generation"
            )
        inputs.extend([header, tokenizer, *sorted(header.parent.glob("*.npy"))])
    inputs.append(gcr_model / "tokenizer.json")
    for path in inputs:
        if not path.is_file():
            raise ValueError(f"missing queue input: {path}")
    return inputs


def write_pilot_inputs(plan: dict, data: Path, out: Path) -> list[Path]:
    """Materialize membership only; stores and tokenizer indexes remain shared."""
    info = plan["pilot"]
    panel = {q["id"]: q for q in json.loads((data / "questions.json").read_text())}
    folder = out / "pilot"
    _json(folder / "cohort.json", [panel[qid] for qid in info["cohort_ids"]])
    _json(folder / "questions.json", [panel[qid] for qid in info["per_arm_ids"]])
    paths = [folder / "cohort.json", folder / "questions.json"]
    for kind in plan.get("request_kinds", ("paths", "agent", "iterative", "gcr")):
        source = {r["id"]: r for r in _read_jsonl(data / f"requests-{kind}.jsonl")}
        path = folder / f"requests-{kind}.jsonl"
        path.write_text(
            "".join(
                json.dumps(source[qid], ensure_ascii=False) + "\n" for qid in info["per_arm_ids"]
            )
        )
        paths.append(path)
    return paths


def gpu_occupancy(path, started, container, *, require_end=False):
    """Operator a78320a's durable boundaries, with menu-control 6c7b181's snapshot order."""
    data = path.read_bytes()
    # A writer can append between the caller's clock sample and this read.
    now = time.monotonic()
    if data and not data.endswith(b"\n"):
        raise ValueError("incomplete GPU launch event")
    active, seconds, launched, previous = None, 0.0, False, started
    for line in data.splitlines():
        event = json.loads(line)
        stamp = event["monotonic"]
        if (
            type(stamp) not in (int, float)
            or not math.isfinite(stamp)
            or not previous <= stamp <= now
            or event.get("container") != container
        ):
            raise ValueError("invalid GPU launch event timestamp or container")
        previous = stamp
        if event["event"] == "container_start" and active is None:
            active, launched = stamp, True
        elif event["event"] == "container_end" and active is not None:
            seconds += stamp - active
            active = None
        else:
            raise ValueError("invalid GPU launch event sequence")
    if require_end and active is not None:
        raise ValueError("stage succeeded without a container end event")
    return seconds + (now - active if active is not None else 0), launched


def append_ledger(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def ledger_used(path):
    """An unsettled start keeps its full reservation, as in menu-control."""
    starts, completed = {}, {}
    for row in _read_jsonl(path) if path.exists() else []:
        attempt = row["attempt"]
        if row["event"] == "start" and attempt not in starts:
            seconds = starts[attempt] = row["reserved_seconds"]
        elif row["event"] == "end" and attempt in starts and attempt not in completed:
            seconds = completed[attempt] = row["seconds"]
        else:
            raise ValueError("invalid GPU ledger event sequence")
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid GPU ledger seconds")
    return sum(completed.get(attempt, seconds) for attempt, seconds in starts.items())


def wait_pilot_stage(process, job, allowance, started, events, container):
    """Reuse menu-control's independent occupancy and preparation wall watchdogs."""
    code = None
    while True:
        seconds, launched = gpu_occupancy(events, started, container)
        left = job.get("wall_timeout_seconds", 86400) - (time.monotonic() - started)
        if left <= 0 or (launched and seconds >= allowance):
            raise subprocess.TimeoutExpired(job["command"], allowance if left > 0 else 0)
        if code is not None:
            return code
        with suppress(subprocess.TimeoutExpired):
            code = process.wait(
                timeout=min(1, left, allowance - seconds) if launched else min(1, left)
            )


def execute_pilot_stage(job, env, log, ledger, budget_seconds, row):
    """Reserve before launching; settle only measured occupancy or confirmed cleanup."""
    from xgraph.provenance import sha256_file

    remaining = budget_seconds - ledger_used(ledger)
    if not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError("pilot GPU budget must be finite and positive")
    if remaining <= 30:
        raise RuntimeError("pilot GPU budget exhausted; no further GPU job launched")
    allowance = remaining - 30  # preserve the cap, reserving cleanup inside it
    events = ledger.parent / job["family"] / f"{job['name']}.gpu-events.jsonl"
    with events.open("xb") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    attempt = str(time.time_ns())
    container = f"xgraph-webqsp-{os.getpid()}-{attempt}"
    append_ledger(
        ledger,
        {
            "event": "start",
            "attempt": attempt,
            "stage": job["name"],
            "family": job["family"],
            "reserved_seconds": remaining,
            "gpu_allowance_seconds": allowance,
            "gpu_events": str(events),
            "container": container,
            "command": job["command"],
        },
    )
    started, process, settled = time.monotonic(), None, True
    row.update(
        returncode=1,
        accounting="reserved",
        gpu_seconds=remaining,
        gpu_allowance_seconds=allowance,
        wall_timeout_seconds=job.get("wall_timeout_seconds", 86400),
    )
    try:
        process = subprocess.Popen(
            job["command"],
            env={
                **env,
                "XG_GPU_EVENTS": str(events),
                "XG_CONTAINER_NAME": container,
                "XG_START_ATTEMPTS": "1",
                "XG_GPU_IDLE": "1",
                "XG_AVAILABLE_WAIT": "43200",
            },
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        row["returncode"] = wait_pilot_stage(process, job, allowance, started, events, container)
        if row["returncode"]:
            raise subprocess.CalledProcessError(row["returncode"], job["command"])
        if not gpu_occupancy(events, started, container, require_end=True)[1]:
            raise ValueError("stage succeeded without a GPU launch event")
    except BaseException as error:
        if isinstance(error, subprocess.TimeoutExpired):
            row["returncode"] = 124
        elif isinstance(error, SystemExit) and isinstance(error.code, int):
            row["returncode"] = error.code or 1
        elif isinstance(error, KeyboardInterrupt):
            row["returncode"] = 130
        else:
            row["returncode"] = row["returncode"] or 1
        row["error"] = str(error)
        settled = False
        try:
            # Stop launchers before cleanup: none may create a container afterwards.
            if process:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            settled = True
        finally:
            if process:
                stopped, settled = settled, False
                try:
                    from xgraph.container_cleanup import stop_container

                    stop_container(container, events)
                    settled = stopped
                except (OSError, ValueError, subprocess.SubprocessError) as cleanup_error:
                    log.write(f"owned container cleanup unconfirmed: {cleanup_error}\n")
        raise
    finally:
        row["wall_seconds"] = time.monotonic() - started
        try:
            if not settled:
                raise ValueError("owned container cleanup unconfirmed")
            seconds, launched = gpu_occupancy(events, started, container)
            row.update(gpu_seconds=seconds, accounting="measured", container_launched=launched)
        except (ValueError, KeyError, TypeError, OSError):
            pass  # retain the reservation when boundaries or cleanup are uncertain
        row["gpu_events"] = {"path": str(events), "sha256": sha256_file(events)}
        append_ledger(
            ledger,
            {
                "event": "end",
                "attempt": attempt,
                "seconds": row["gpu_seconds"],
                "wall_seconds": row["wall_seconds"],
                "accounting": row["accounting"],
                "returncode": row["returncode"],
                "gpu_events": row["gpu_events"],
            },
        )


@contextmanager
def pilot_termination(enabled):
    """Let the outer stage watchdog trigger the same owned cleanup as a timeout."""

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, terminate) if enabled else None
    try:
        yield
    finally:
        if enabled:
            signal.signal(signal.SIGTERM, previous)


def _read_pilot_identity(item):
    path = Path(item["path"])
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != item["sha256"]:
        raise ValueError(f"pilot restart input changed: {path}")
    return data


def _restart_ledger(plan, out):
    """Retain settled occupancy, reusing only explicitly receipted unchanged stages."""
    from xgraph.provenance import sha256_file

    restart = plan.get("pilot_restart")
    if restart is None:
        return b"", 0
    if not plan.get("pilot"):
        raise ValueError("pilot restart requires a pilot")

    read = _read_pilot_identity
    source = json.loads(read(restart["source_plan"]))
    root = Path(__file__).resolve().parents[2]

    def science(value, old_out, old_root):
        value = {
            k: v
            for k, v in value.items()
            if k not in ("provenance", "catalog_signature", "pilot_restart")
        }
        return (
            json.dumps(value, sort_keys=True)
            .replace(str(old_out), "{OUT}")
            .replace(str(old_root), "{ROOT}")
        )

    if science(source, restart["source_out"], restart["source_root"]) != science(plan, out, root):
        raise ValueError("pilot restart science or generation argv changed")
    completed = _completed_pilot_stages(plan, out, source, read)
    data = read(restart["ledger"])
    if not data or not data.endswith(b"\n"):
        raise ValueError("pilot restart needs a complete failed ledger")
    # A restart can itself fail before launch. Re-derive its inherited charge
    # against the original plan and events, retaining the exact ledger prefix.
    prior, _ = _restart_ledger(source, Path(restart["source_out"]))
    if not data.startswith(prior):
        raise ValueError("pilot restart inherited ledger prefix changed")
    rows = [json.loads(line) for line in data[len(prior) :].splitlines()]
    starts, ended, successes = {}, set(), set()
    for row in rows:
        attempt = row["attempt"]
        if row["event"] == "start" and attempt not in starts:
            reserved = row["reserved_seconds"]
            if (
                type(reserved) not in (int, float)
                or not math.isfinite(reserved)
                or not (0 < reserved <= plan["pilot"]["budget_gpu_hours"] * 3600)
            ):
                raise ValueError("invalid pilot restart reservation")
            if not any(
                row["stage"] == job["name"]
                and row["family"] == job["family"]
                and row["command"] == job["command"]
                for job in source["jobs"]
            ):
                raise ValueError("pilot restart ledger does not match its source plan")
            starts[attempt] = row
        elif row["event"] == "end" and attempt in starts and attempt not in ended:
            start = starts[attempt]
            if (
                row.get("accounting") != "measured"
                or type(row.get("returncode")) is not int
                or (row["returncode"] == 0 and (start["family"], start["stage"]) not in completed)
            ):
                raise ValueError("pilot restart requires settled failures or receipted successes")
            if (
                type(row["seconds"]) not in (int, float)
                or not 0 <= row["seconds"] <= start["reserved_seconds"]
            ):
                raise ValueError("pilot restart charge exceeds its reservation")
            event = row["gpu_events"]
            if event["path"] != start["gpu_events"] or event["path"] not in restart["events"]:
                raise ValueError("pilot restart event identity missing")
            archived = restart["events"][event["path"]]
            read(archived)
            if event["sha256"] != archived["sha256"]:
                raise ValueError("pilot restart event hash changed")
            seconds, launched = gpu_occupancy(
                Path(archived["path"]), 0, start["container"], require_end=True
            )
            if seconds != row["seconds"] or launched != (row["seconds"] > 0):
                raise ValueError("pilot restart charge differs from container events")
            if row["returncode"] == 0:
                key = start["family"], start["stage"]
                if completed[key][1]["timing"]["gpu_seconds"] != seconds or key in successes:
                    raise ValueError("pilot reuse timing differs from settled occupancy")
                successes.add(key)
            ended.add(attempt)
        else:
            raise ValueError("invalid pilot restart ledger sequence")
    if starts.keys() != ended:
        raise ValueError("pilot restart has unsettled occupancy")
    inherited = {
        (saved["job"]["family"], saved["job"]["name"])
        for item in source.get("pilot_restart", {}).get("completed", [])
        for saved in [json.loads(read(item))]
    }
    if completed.keys() != successes | inherited:
        raise ValueError("pilot reuse needs a successful occupancy for every receipt")
    # ledger_used independently checks finite, nonnegative ledger values.
    seconds = ledger_used(Path(restart["ledger"]["path"]))
    if seconds != restart["charged_seconds"] or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("pilot restart charge changed")
    if plan["pilot"]["budget_gpu_hours"] * 3600 - seconds <= 30:
        raise ValueError("pilot restart GPU budget exhausted")
    if sha256_file(Path(restart["ledger"]["path"])) != restart["ledger"]["sha256"]:
        raise ValueError("pilot restart ledger changed during validation")
    return data, seconds


def _completed_pilot_stages(plan, out, source, read):
    """Verify receipt metadata and opaque artifact hashes, without parsing model text."""
    from xgraph.provenance import sha256_file

    restart = plan["pilot_restart"]
    old_out = Path(restart["source_out"])
    root = Path(__file__).resolve().parents[2]
    completed = {}
    for item in restart.get("completed", []):
        saved = json.loads(read(item))
        job, timing = saved["job"], saved["timing"]
        key = job["family"], job["name"]

        def normalized(value, directory, repo):
            return (
                json.dumps(value, sort_keys=True)
                .replace(str(directory), "{OUT}")
                .replace(str(repo), "{ROOT}")
            )

        proof = json.loads(read(saved["unchanged_code"]))
        if (
            key in completed
            or job not in source["jobs"]
            or not any(
                normalized(job, old_out, restart["source_root"]) == normalized(j, out, root)
                for j in plan["jobs"]
            )
            or list(key) not in proof["eligible_stages"]
            or proof["source_plan"]["sha256"] != restart["source_plan"]["sha256"]
            or timing not in json.loads(read(saved["source_timings"]))["jobs"]
            or any(timing[k] != job[k] for k in ("family", "name", "kind", "questions"))
            or timing["kind"] != "gpu"
            or timing["returncode"] != 0
            or timing["accounting"] != "measured"
            or not math.isfinite(timing["gpu_seconds"])
            or timing["gpu_seconds"] <= 0
            or saved["output_records"] != job["questions"]
        ):
            raise ValueError("pilot reuse requires a complete unchanged successful stage")
        command = job["command"]
        if "--out" in command:
            output = Path(command[command.index("--out") + 1])
        else:
            (launcher,) = [
                i for i, arg in enumerate(command) if arg.endswith("/scripts/generate.sh")
            ]
            output = Path(command[launcher + 4])
        artifacts = saved["artifacts"]
        if str(output) not in [a["path"] for a in artifacts]:
            raise ValueError("pilot reuse receipt is missing its output")
        seen = set()
        for artifact in artifacts:
            path = Path(artifact["path"])
            if (
                path in seen
                or path.parent != old_out / job["family"]
                or not path.name.startswith(output.name)
                or path.is_symlink()
                or not path.is_file()
                or sha256_file(path) != artifact["sha256"]
            ):
                raise ValueError(f"pilot reuse artifact changed or escaped its stage: {path}")
            seen.add(path)
        completed[key] = item, saved
    return completed


def restart_ledger(plan, out):
    """Validate public restart metadata with a contextual error on malformed inputs."""
    try:
        return _restart_ledger(plan, out)
    except (KeyError, TypeError, AttributeError, OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid pilot restart metadata: {error}") from error


def run_queue(plan: dict, out: Path) -> None:
    """Keep launcher wall separate from the pilot's container occupancy ledger."""
    prior_ledger, prior_seconds = restart_ledger(plan, out)
    # restart_ledger has validated every receipt, its producing command, and its
    # successful occupancy against the complete inherited ledger.
    reused = {
        (saved["job"]["family"], saved["job"]["name"]): (item, saved)
        for item in plan.get("pilot_restart", {}).get("completed", [])
        for saved in [json.loads(_read_pilot_identity(item))]
    }
    out.mkdir(parents=True, exist_ok=True)
    results = []

    def save():
        gpu = [r for r in results if r["kind"] == "gpu"]
        seconds = sum(r.get("gpu_seconds", r["wall_seconds"]) for r in gpu)
        fresh_seconds = sum(
            r.get("gpu_seconds", r["wall_seconds"]) for r in gpu if "reused_receipt" not in r
        )
        projection = None
        if plan.get("project_questions") and gpu:
            projection = {
                "questions_per_arm": plan["project_questions"],
                "gpu_hours": sum(
                    r.get("gpu_seconds", r["wall_seconds"])
                    / r["questions"]
                    * plan["project_questions"]
                    for r in gpu
                )
                / 3600,
                "rule": (
                    "linear per-arm container occupancy/question including startup; "
                    if plan.get("pilot")
                    else "linear per-arm launcher wall/question including startup; "
                )
                + "exploratory, no timing confidence bound",
                "complete": len(gpu) == sum(j["kind"] == "gpu" for j in plan["jobs"])
                and all(r["returncode"] == 0 for r in gpu),
            }
        _json(
            out / "timings.json",
            {
                "jobs": results,
                "per_arm": gpu,
                "measured_gpu_wall_seconds": None
                if any(r.get("accounting") == "reserved" for r in gpu)
                else seconds,
                "charged_gpu_seconds": prior_seconds + fresh_seconds,
                **({"prior_gpu_seconds": prior_seconds} if prior_ledger else {}),
                "launcher_wall_seconds": sum(r["wall_seconds"] for r in gpu),
                "gpu_accounting": "container occupancy or retained reservation"
                if plan.get("pilot")
                else "launcher wall",
                "complete": len(results) == len(plan["jobs"])
                and all(r["returncode"] == 0 for r in results),
                "test_projection": projection,
                "pilot_budget_exceeded": bool(plan.get("pilot"))
                and prior_seconds + fresh_seconds > plan["pilot"]["budget_gpu_hours"] * 3600,
                "timing_scope": "pilot GPU charges: owned container launch through end or "
                "confirmed cleanup, including model loading; waits/builds are launcher wall only; "
                "uncertain occupancy retains reservation; CPU stages separate"
                if plan.get("pilot")
                else "serial launcher wall including admission/build/startup",
                "provenance": plan.get("provenance"),
                **({"catalog_signature": plan["catalog_signature"]} if plan.get("catalog") else {}),
            },
        )

    # generate.sh independently checks headroom and GPU idle.
    with (
        Path("/tmp/xgraph-webqsp-queue.lock").open("a") as lock,
        pilot_termination(bool(plan.get("pilot"))),
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        destinations = [
            out / "timings.json",
            out / "gpu-ledger.jsonl",
            *(out / f for f in {j["family"] for j in plan["jobs"]}),
        ]
        for path in destinations:
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"refusing existing output: {path}; use a fresh directory")
        if prior_ledger:
            with (out / "gpu-ledger.jsonl").open("xb") as stream:
                stream.write(prior_ledger)
                stream.flush()
                os.fsync(stream.fileno())
        with (out / "timings.json").open("x"):
            pass
        save()
        for job in plan["jobs"]:
            folder = out / job["family"]
            folder.mkdir(exist_ok=True)
            key = job["family"], job["name"]
            if key in reused:
                item, saved = reused[key]
                old_out = Path(plan["pilot_restart"]["source_out"])
                for artifact in saved["artifacts"]:
                    source_path = Path(artifact["path"])
                    destination = out / source_path.relative_to(old_out)
                    with source_path.open("rb") as src, destination.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
                    from xgraph.provenance import sha256_file

                    if sha256_file(destination) != artifact["sha256"]:
                        raise ValueError(f"pilot reuse artifact changed during copy: {source_path}")
                results.append({**saved["timing"], "reused_receipt": item})
                save()
                print(
                    f"webqsp {job['family']} {job['name']} skip: verified receipt "
                    f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
                    flush=True,
                )
                continue
            with (folder / f"{job['name']}.log").open("x") as log:
                started = time.monotonic()
                env = {
                    **{
                        k: v
                        for k, v in os.environ.items()
                        if not plan.get("catalog") or not k.startswith(("XG_", "XGRAPH_"))
                    },
                    **job["env"],
                }
                row = {
                    "kind": job["kind"],
                    "family": job["family"],
                    "name": job["name"],
                    "questions": job.get("questions"),
                    "returncode": 1,
                    **(
                        {"gpu_seconds": 0, "accounting": "not launched"}
                        if job["kind"] == "gpu" and plan.get("pilot")
                        else {}
                    ),
                }
                try:
                    if job["kind"] == "gpu" and plan.get("pilot"):
                        execute_pilot_stage(
                            job,
                            env,
                            log,
                            out / "gpu-ledger.jsonl",
                            plan["pilot"]["budget_gpu_hours"] * 3600,
                            row,
                        )
                    else:
                        row["returncode"] = subprocess.run(
                            job["command"],
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=False,
                        ).returncode
                finally:
                    row["wall_seconds"] = time.monotonic() - started
                    results.append(row)
                    save()
                    print(
                        f"webqsp {job['family']} {job['name']} exit {row['returncode']} "
                        f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
                        flush=True,
                    )
                if row["returncode"]:
                    raise subprocess.CalledProcessError(row["returncode"], job["command"])


def main(argv=None):
    from xgraph.generate import model_revision
    from xgraph.provenance import provenance

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--family", choices=("both", "qwen25", "qwen38", "gemma31"), default="both")
    parser.add_argument(
        "--qwen-model", type=Path, default=Path("/home/jonas/models/Qwen2.5-7B-Instruct")
    )
    parser.add_argument(
        "--gemma-model", type=Path, default=Path("/home/jonas/models/gemma-4-31B-it-FP8-dynamic")
    )
    parser.add_argument(
        "--gcr-model", type=Path, default=Path("/home/jonas/models/GCR-Meta-Llama-3.1-8B-Instruct")
    )
    parser.add_argument("--cor-prompts", type=Path)
    parser.add_argument("--dog-prompts", type=Path, help="released DoG templates.py, read as data")
    parser.add_argument(
        "--catalog", action="store_true", help="opt in to the frozen policy catalog"
    )
    parser.add_argument(
        "--arms", nargs="+", help="catalog arm allowlist; defaults to the complete catalog"
    )
    parser.add_argument(
        "--pilot-timings", type=Path, help="complete catalog pilot timings for test pricing"
    )
    parser.add_argument(
        "--policy-freeze", type=Path, help="bind --arms finalist to a sealed family mapping"
    )
    parser.add_argument("--timing-root", type=Path, default=Path("/home/jonas/git/xgraph"))
    parser.add_argument(
        "--qwen38-model", type=Path, default=Path("/home/jonas/models/Qwen3.8-27B-FP8")
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="validation-only stratified budget pilot; both families, all arms",
    )
    parser.add_argument(
        "--pilot-questions",
        type=int,
        default=100,
        help="size of the seeded validation cohort before per-arm budgeting",
    )
    parser.add_argument("--pilot-seed", type=int, default=0)
    parser.add_argument("--pilot-budget-hours", type=float, default=8)
    parser.add_argument("--pilot-restart", type=Path, help="hash-bound closed failed pilot history")
    parser.add_argument("--pilot-restart-sha256", help="expected restart metadata SHA-256")
    parser.add_argument(
        "--secondary-pilot",
        action="store_true",
        help="separate released DoG cost pilot; requires --catalog --pilot",
    )
    parser.add_argument(
        "--project-data",
        type=Path,
        help="full test data directory for timing projection; defaults to sibling test",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="default: write and print CPU-checked plan"
    )
    mode.add_argument(
        "--run", action="store_true", help="explicitly launch serial GPU and CPU stages"
    )
    args = parser.parse_args(argv)
    if bool(args.pilot_restart) != bool(args.pilot_restart_sha256) or (
        args.pilot_restart and not (args.catalog and args.pilot)
    ):
        parser.error("--pilot-restart and --pilot-restart-sha256 require a catalog pilot")
    if args.secondary_pilot and not (args.catalog and args.pilot):
        parser.error("--secondary-pilot requires --catalog --pilot")
    if args.out.exists() or args.out.is_symlink():
        parser.error(f"output directory already exists: {args.out}; use a fresh directory")
    if not args.catalog and (
        args.arms or args.pilot_timings or args.policy_freeze or args.family == "qwen38"
    ):
        parser.error("Qwen3.8, --arms, --pilot-timings and --policy-freeze require --catalog")
    models = {"qwen25": args.qwen_model, "gemma31": args.gemma_model}
    if args.catalog:
        models = {"qwen38": args.qwen38_model, "gemma31": args.gemma_model}
        if args.family == "qwen25":
            models = {"qwen25": args.qwen_model}
    if args.family != "both":
        models = {args.family: models[args.family]}
    inputs = preflight(args.data, models, args.gcr_model, catalog=args.catalog)
    if args.cor_prompts:
        if not args.cor_prompts.is_file():
            parser.error("missing CoR prompts")
        inputs.append(args.cor_prompts)
    project_manifest = (
        args.project_data or (args.data if args.secondary_pilot else args.data.parent / "test")
    ) / "manifest.json"
    project_questions = None
    if project_manifest.is_file():
        project_questions = json.loads(project_manifest.read_text())["counts"]["questions"]
        inputs.append(project_manifest)
    elif args.project_data:
        parser.error("missing --project-data manifest.json")
    plan = queue_plan(
        args.data,
        args.out,
        models,
        args.gcr_model,
        args.cor_prompts,
        pilot=args.pilot,
        pilot_questions=args.pilot_questions,
        pilot_seed=args.pilot_seed,
        pilot_budget_hours=args.pilot_budget_hours,
        project_questions=project_questions,
        catalog=args.catalog,
        arms=[a for item in args.arms for a in item.split(",")] if args.arms else None,
        pilot_timings=args.pilot_timings,
        policy_freeze=args.policy_freeze,
        timing_root=args.timing_root,
        dog_prompts=args.dog_prompts,
        secondary_pilot=args.secondary_pilot,
    )
    if plan.get("pilot"):
        plan["pilot_accounting"] = {
            "timing_kind": "owned container occupancy, including startup and cleanup",
            "events": "XG_GPU_EVENTS: durable container_start/container_end host monotonic events",
            "gpu_allowance": "remaining pilot cap less cleanup reservation",
            "cleanup_seconds": 30,
            "stage_wall_timeout_seconds": 86400,
            "uncertain": "retain full reservation for invalid events or unconfirmed cleanup",
            "wall_seconds": "separately retained launcher time including waits and builds",
        }
    if args.catalog:
        from xgraph.dog import DoGPrompts
        from xgraph.webqsp_catalog import DEFAULT_DOG_PROMPTS

        if "gcr-native" in plan["arms"]:
            from xgraph.tokenizer import ServedTokenizer

            tokenizer = args.gcr_model / "tokenizer.json"
            ServedTokenizer.from_file(tokenizer)
            inputs.extend(sorted(args.gcr_model.glob("*.json")))
            if not (args.data / "kg-gcr.sqlite").is_file():
                parser.error("native GCR requires its released DiGraph store kg-gcr.sqlite")
            inputs.append(args.data / "kg-gcr.sqlite")
        if "dog" in plan["arms"]:
            prompts = DoGPrompts.load(args.dog_prompts or DEFAULT_DOG_PROMPTS)
            inputs.append(prompts.source.path)
    plan["model_revisions"] = {name: model_revision(path) for name, path in models.items()}
    if not args.catalog or "gcr-native" in plan["arms"]:
        plan["model_revisions"]["gcr"] = model_revision(args.gcr_model)
    if args.pilot_restart:
        raw = args.pilot_restart.read_bytes()
        if hashlib.sha256(raw).hexdigest() != args.pilot_restart_sha256:
            parser.error("pilot restart metadata changed")
        plan["pilot_restart"] = json.loads(raw)
        restart_ledger(plan, args.out)
        inputs.extend(
            [
                args.pilot_restart,
                Path(plan["pilot_restart"]["ledger"]["path"]),
                Path(plan["pilot_restart"]["source_plan"]["path"]),
                *(Path(item["path"]) for item in plan["pilot_restart"]["events"].values()),
            ]
        )
    args.out.mkdir(parents=True, exist_ok=False)
    if args.pilot:
        inputs.extend(write_pilot_inputs(plan, args.data, args.out))
    if args.catalog:
        inputs.extend(
            Path(j["timing_assumption"]["source"])
            for j in plan["jobs"]
            if j["kind"] == "gpu" and j["timing_assumption"]["source"]
        )
        if args.pilot_timings:
            inputs.append(args.pilot_timings)
        if args.policy_freeze:
            inputs.extend([args.policy_freeze, Path(f"{args.policy_freeze}.sha256")])
    plan["provenance"] = provenance(list(dict.fromkeys(inputs)) if args.catalog else inputs)
    _json(args.out / "queue.json", plan)
    with (args.out / "commands.sh").open("w") as file:
        file.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        for job in plan["jobs"]:
            command = shlex.join(
                ["env", *(f"{k}={v}" for k, v in job["env"].items()), *job["command"]]
            )
            file.write(command + "\n")
            print(command)
    print(json.dumps(plan["estimate"]))
    if plan.get("pilot_restart"):
        remaining = (
            plan["pilot"]["budget_gpu_hours"] * 3600 - plan["pilot_restart"]["charged_seconds"]
        )
        print(f"# pilot remaining seconds: {remaining}; next occupancy allowance: {remaining - 30}")
    if not args.run:
        return
    if args.catalog and not args.pilot and not args.pilot_timings:
        parser.error("catalog test execution requires complete --pilot-timings")
    prior_hours = plan.get("pilot_restart", {}).get("charged_seconds", 0) / 3600
    if (
        args.catalog
        and args.pilot
        and plan["estimate"]["high_gpu_hours"] + prior_hours > args.pilot_budget_hours
    ):
        parser.error(
            "complete catalog pilot forecast exceeds its reservation; "
            "retain cohort, revise reservation"
        )
    run_queue(plan, args.out)


if __name__ == "__main__":
    main()
