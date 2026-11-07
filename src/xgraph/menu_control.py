"""Registered development-only plan-audit menu control; GPU execution is opt-in."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import resource
import shlex
import signal
import subprocess
import time
from collections import defaultdict
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from xgraph.catalog_argv import expand_launcher, flags
from xgraph.mintqa_panels import indexed, read_jsonl, write_json, write_jsonl
from xgraph.mintqa_rematch import AuditMenus, audit_requests, clustered_difference, revision_records
from xgraph.provenance import provenance, sha256_file

REPO = Path(__file__).resolve().parents[2]
FAMILIES = {"qwen38": "Qwen3.8-27B-FP8", "gemma31": "gemma-4-31B-it-FP8-dynamic"}
ARMS = ("audit", "audit-global", "audit-nomenu")
SPORT = "sport · country of origin · capital"
CPU_LIMIT = 4_000_000_000
STAGE_POLL_SECONDS = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def check_identities(items):
    for item in items:
        if identity(item["path"]) != item:
            raise ValueError(f"changed input/output: {item['path']}")


def clean_commit(root=REPO):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    if git("status", "--porcelain"):
        raise ValueError("menu control refuses dirty trees")
    return git("rev-parse", "HEAD")


def normalized_flags(options):
    """Ignore destination and request paths only; their contents are sealed separately."""
    return {
        k: v for k, v in options.items() if k not in {"--out", "--requests", "--provenance-only"}
    }


def generation_command(root, source, out, family, arm, *, results=None):
    root, source, out = map(Path, (root, source, out))
    results = out if results is None else Path(results)
    model = Path("/home/jonas/models") / FAMILIES[family]
    settings = {
        "PATH": "/home/jonas/.local/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME": str(Path.home()),
        "UV_CACHE_DIR": str(root / "target/menu-control-build/uv-cache"),
        "UV_LINK_MODE": "hardlink",
        "UV_OFFLINE": "1",
        "UV_PYTHON_DOWNLOADS": "never",
        "UV_BUILD_CONSTRAINT": str(root / "scripts/mintqa/menu-control-build.txt"),
        "TMPDIR": str(root / "target/tmp"),
        "CMAKE_ARGS": "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
        "SKBUILD_BUILD_DIR": str(root / "target/menu-control-build/native"),
        "XG_NEED_GIB": "60" if family == "qwen38" else "75",
        "XG_AVAILABLE_WAIT": "43200",
        "XG_GPU_IDLE": "1",
        "XG_START_ATTEMPTS": "1",
        "XG_START_RETRY_SLEEP": "120",
        "XG_KG_INDEX": str(source / f"target/mintqa-rematch/indexes/{family}"),
        "XG_TOKENIZER": str(model / "tokenizer.json"),
        "XG_PATHS": "1",
        "XG_COMMITTED_PATHS": "1",
        "XG_ESCAPES": "0",
        "XG_UNQUOTED_STARTS": "1",
        "XG_START_RANK": "sitelinks",
        "XG_OPEN": "1",
        "XG_END_AFTER_ANSWER": "1",
        "XG_RELATIONS": "frontier",
        "XG_OBSERVE_TERMINAL": "8",
        "XG_CONTAINER_NAME": f"xgraph-menu-control-{family}-{arm}",
        "XG_GPU_EVENTS": str(results / family / f"{arm}-raw.jsonl.gpu-events.jsonl"),
        "XG_PAGECACHE_ROOTS": f"{source / 'data/external/mintqa-full'}:{model}",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "CMAKE_BUILD_PARALLEL_LEVEL": "1",
    }
    # env -i prevents inherited display, support, STOP or sampling switches.
    return [
        "env", "-i", *(f"{k}={v}" for k, v in settings.items()), "nice", "-n", "10",
        str(root / "scripts/generate.sh"), str(model),
        str(out / family / f"{arm}-requests.jsonl"), "constrained",
        str(results / family / f"{arm}-raw.jsonl"), "--max-tokens", "256",
        "--max-model-len", "16384", "--max-num-seqs", "64", "--kv-cache-gib",
        "8" if family == "qwen38" else "16", "--seed", "17", "--memory-share", "0.450",
        *(["--ban-tokens", "<|channel>"] if family == "gemma31" else []),
    ]  # fmt: skip


def _executor(source, family):
    from xgraph.index import load_kg_index
    from xgraph.programs import MatchedExecutor
    from xgraph.tokenizer import ServedTokenizer

    index = load_kg_index(source / f"target/mintqa-rematch/indexes/{family}")
    tokenizer = ServedTokenizer.from_file(
        Path("/home/jonas/models") / FAMILIES[family] / "tokenizer.json"
    )
    return MatchedExecutor(index, tokenizer, start_rank="sitelinks", unquoted_starts=True)


def _capped_support(executor, record, request):
    """The scratch analysis's capped replay, with no expensive uncapped sensitivity."""
    from xgraph.oracle import MAX_REACHED
    from xgraph.programs import record_path
    from xgraph.prompts import final_answer

    if request.get("kg_version"):
        raise ValueError("this registered development control has no KG versions")
    index = executor._index
    path, events, reached = record_path(record), [], []
    if path:
        start = executor._read_start(index, path[0], request.get("anchor"))
        walked = executor._walk(
            index, start, path[1], request.get("anchor"), False, cap_events=events
        )
        if walked and walked[0]:
            ranks, rows_of, _ = walked
            best = min(ranks.values())
            rows = sorted({r for e, rank in ranks.items() if rank == best for r in rows_of[e]})
            if len(rows) > MAX_REACHED:
                events.append(
                    {"stage": "answer_label_rows", "before": len(rows), "kept": MAX_REACHED}
                )
            ordered = executor._order(
                index, [(int(index.labels.owners[r]), r) for r in rows[:MAX_REACHED]]
            )
            reached = [index.labels.owner_ids[e] for e in ordered]
    answer = final_answer(record.get("text", ""))
    return {
        "id": record["id"],
        "reached": reached,
        "cap_events": events,
        "answer_in_support": bool(answer and executor.named(answer) & set(reached)),
    }


def prepare(source, out, family):
    """Build all menus before registration or any new completion, without gold reads."""
    from xgraph.toolkg import KGStore

    directory = out / family
    if directory.exists():
        raise ValueError(f"prepare needs a fresh family directory: {directory}")
    panel_path = source / "target/mintqa-rematch/panels/panels.json"
    members = json.loads(panel_path.read_text())["panels"]["development"]["members"]
    request_path = source / "target/mintqa-rematch/panels/development/requests.jsonl"
    first_path = source / f"target/mintqa-rematch/{family}/development/paths-allfix.jsonl"
    requests, first = read_jsonl(request_path), read_jsonl(first_path)
    ids = {m["id"] for m in members}
    if len(ids) != 1024 or set(indexed(requests)) != ids or set(indexed(first)) != ids:
        raise ValueError("menu control requires the complete frozen development panel")
    executor = _executor(source, family)
    tokenizer = executor._tokenizer
    menus = AuditMenus(executor._index, tokenizer)
    db = source / "data/external/mintqa-full/kg.sqlite"
    store = KGStore(db)
    directory.mkdir(parents=True)
    made = {}
    for arm in ARMS:
        made[arm] = audit_requests(
            requests, first, executor, store.label, menus, tokenizer,
            show_menus=arm != "audit-nomenu", observe_terminal=8,
            menu_source={"audit": "local", "audit-global": "global"}.get(arm),
        )  # fmt: skip
        write_jsonl(directory / f"{arm}-requests.jsonl", made[arm])
    checks = []
    for local, global_menu, nomenu in zip(*(made[a] for a in ARMS), strict=True):
        if not local["messages"][:-1] == global_menu["messages"][:-1] == nomenu["messages"][:-1]:
            raise ValueError("mismatched demonstrations or first-call history")
        explanation = [r["messages"][-1]["content"].split("\n", 1)[0] for r in (local, global_menu)]
        if explanation[0] != explanation[1]:
            raise ValueError("mismatched explanation")
        sizes = [
            len(local["audit"]["display"][f"{b}_menu"]["relations"])
            for b in ("terminal", "penultimate")
        ]
        if sizes != [
            len(global_menu["audit"]["display"][f"{b}_menu"]["relations"])
            for b in ("terminal", "penultimate")
        ]:
            raise ValueError("mismatched menu sizes")
        checks.append(
            {
                "id": local["id"],
                "sizes": sizes,
                "explanation_token_delta": 0,
                "instruction_token_delta": global_menu["audit"]["instruction_tokens"]
                - local["audit"]["instruction_tokens"],
            }
        )
    write_jsonl(directory / "matching.jsonl", checks)
    write_json(directory / "members.json", members)
    model = Path("/home/jonas/models") / FAMILIES[family]
    index = source / f"target/mintqa-rematch/indexes/{family}"
    inputs = [
        panel_path,
        request_path,
        first_path,
        Path(str(first_path) + ".provenance.json"),
        db,
        index.with_suffix(".manifest.json"),
        index / "kg-index.json",
        *sorted(index.glob("*.npy")),
        *(p for p in sorted(model.iterdir()) if p.suffix in {".json", ".jinja"}),
    ]
    result = {
        "family": family,
        "source": str(source),
        "first": str(first_path),
        "requests": str(request_path),
        "inputs": [identity(p) for p in inputs],
        "prepared": [identity(p) for p in sorted(directory.iterdir())],
        "builder_sha256": sha256_file(Path(__file__).with_name("mintqa_rematch.py")),
        "matching": {
            "questions": len(checks),
            "explanation_token_delta": [0, 0],
            "instruction_token_delta_range": [
                min(r["instruction_token_delta"] for r in checks),
                max(r["instruction_token_delta"] for r in checks),
            ],
        },
        "provenance": provenance([]),
    }
    write_json(directory / "preparation.json", result)


def paired_metrics(a, b, members, *, sport=SPORT):
    """Population-weighted paired A/C/E, hop-calibrated cluster intervals and sensitivity."""
    mass = sum(m["weight"] for m in members)
    result = {}
    for key in ("A", "C", "E"):
        groups = defaultdict(lambda: [0.0, 0.0])
        for m in members:
            groups[m["cluster"]][0] += m["weight"] * (a[m["id"]][key] - b[m["id"]][key])
            groups[m["cluster"]][1] += m["weight"]
        gain = sum(v[0] for v in groups.values())
        removed_gain, removed_mass = groups.get(sport, (0, 0))
        bootstrap = clustered_difference(
            [a[m["id"]][key] for m in members],
            [b[m["id"]][key] for m in members],
            members,
            10000,
            17,
        )
        for field in ("difference", "low", "high"):
            bootstrap[field] *= 100
        bootstrap["units"] = "percentage points"
        result[key] = {
            "difference_pp": 100 * gain / mass,
            "bootstrap": bootstrap,
            "loso_min_pp": min(
                (100 * (gain - v) / (mass - w) for v, w in groups.values() if w < mass),
                default=None,
            ),
            "without_sport_pp": 100 * (gain - removed_gain) / (mass - removed_mass)
            if mass > removed_mass
            else None,
        }
    for name, wanted in (("rescues", 1), ("regressions", -1)):
        selected = [m for m in members if a[m["id"]]["A"] - b[m["id"]]["A"] == wanted]
        result[name] = {
            "ids": [m["id"] for m in selected],
            "n": len(selected),
            "population_pp": 100 * sum(m["weight"] for m in selected) / mass,
        }
    return result


def _cpu_command(out, command, family=None):
    return [
        "nice",
        "-n",
        "10",
        *([] if command == "queue" else ["prlimit", f"--as={CPU_LIMIT}"]),
        str(REPO / ".venv/bin/python"),
        "-m",
        "xgraph.menu_control",
        command,
        "--out",
        str(out),
        *(["--family", family] if family else []),
    ]


def register(source, out, *, attempt=None):
    if attempt is not None and (
        not attempt or Path(attempt).name != attempt or attempt in {".", ".."}
    ):
        raise ValueError("attempt must be a single directory name")
    results = out if attempt is None else out / "attempts" / attempt
    commit = clean_commit()
    if (out / "registration.json").exists():
        raise ValueError("registration already exists; it is immutable")
    stages, pricing, historical = [], {}, {}
    inputs = [identity(REPO / "scripts/mintqa/menu-control-build.txt")]
    for family in FAMILIES:
        prep_path = out / family / "preparation.json"
        prep = json.loads(prep_path.read_text())
        if prep["source"] != str(source) or prep["builder_sha256"] != sha256_file(
            Path(__file__).with_name("mintqa_rematch.py")
        ):
            raise ValueError("preparation code or source changed")
        check_identities(prep["inputs"] + prep["prepared"])
        inputs += prep["inputs"] + prep["prepared"] + [identity(prep_path)]
        timings = {}
        for arm in ARMS:
            command = generation_command(REPO, source, out, family, arm, results=results)
            expanded = expand_launcher(command)
            output = results / family / f"{arm}-raw.jsonl"
            name = f"xgraph-menu-control-{family}-{arm}"
            stages.append(
                {
                    "name": f"{family}/{arm}",
                    "kind": "gpu",
                    "command": command,
                    "expected_flags": flags(expanded),
                    "expanded_argv": expanded,
                    "outputs": [
                        str(output),
                        str(output) + ".summary.json",
                        str(output) + ".provenance.json",
                    ],
                    "gpu_events": str(output) + ".gpu-events.jsonl",
                    "wall_timeout_seconds": 24 * 3600,
                    "inspect_command": ["docker", "container", "inspect", name],
                    "cleanup_command": ["docker", "rm", "--force", name],
                    "headroom_command": [
                        "env",
                        f"UV_CACHE_DIR={REPO / 'target/menu-control-build/uv-cache'}",
                        "UV_LINK_MODE=hardlink",
                        "UV_OFFLINE=1",
                        f"TMPDIR={REPO / 'target/tmp'}",
                        "XG_GPU_IDLE=1",
                        "XG_AVAILABLE_WAIT=43200",
                        f"XG_PAGECACHE_ROOTS={source / 'data/external/mintqa-full'}:"
                        f"/home/jonas/models/{FAMILIES[family]}",
                        "nice",
                        "-n",
                        "10",
                        str(REPO / "scripts/gpu-headroom.sh"),
                        "60" if family == "qwen38" else "75",
                    ],
                }
            )
            if arm == "audit-global":
                continue
            base = (
                source
                / f".claude/worktrees/codex-audit2/target/mintqa-rematch/{family}/development"
            )
            old = base / f"{arm}-raw.jsonl"
            prov_path, timing_path = (
                Path(str(old) + ".provenance.json"),
                Path(str(old) + ".summary.json"),
            )
            prov, timing = json.loads(prov_path.read_text()), json.loads(timing_path.read_text())
            before, after = (
                normalized_flags(flags(prov["provenance"]["command"])),
                normalized_flags(flags(expanded)),
            )
            historical[f"{family}/{arm}"] = {
                "reuse": False,
                "recorded_argv_sha256": digest(before),
                "planned_argv_sha256": digest(after),
                "recorded_config_sha256": digest(
                    {k: v for k, v in prov.items() if k != "provenance"}
                ),
                "differences": {
                    k: {"recorded": before.get(k), "planned": after.get(k)}
                    for k in before.keys() | after.keys()
                    if before.get(k) != after.get(k)
                },
                "reason": (
                    "force bound answer delivery with terminal width eight in all arms; "
                    "local explanation also matched to global"
                ),
                "source_commit": prov["provenance"]["repo_commit"],
                "provenance": identity(prov_path),
                "timing": identity(timing_path),
            }
            if before == after:
                raise ValueError(
                    "unexpected historical match; audit request/config identity before reuse"
                )
            timings[arm] = timing["seconds"]
            inputs += [identity(prov_path), identity(timing_path)]
        # Global is priced as local; startup and workload inflation are assumptions.
        pricing[family] = {
            "historical_workload_seconds": timings,
            "workload_seconds": 2 * timings["audit"] + timings["audit-nomenu"],
        }
        for stage in stages:
            if stage["kind"] == "gpu" and stage["name"].startswith(f"{family}/"):
                arm = stage["name"].split("/")[1]
                stage["forecast_seconds"] = timings["audit" if arm == "audit-global" else arm]
        outputs = [
            str(results / family / f"{arm}{suffix}")
            for arm in ARMS
            for suffix in (".jsonl", "-support.jsonl")
        ]
        stages.append(
            {
                "name": f"{family}/finish",
                "kind": "cpu",
                "command": _cpu_command(out, "finish", family),
                "outputs": [*outputs, str(results / family / "finish.json")],
            }
        )
    stages.append(
        {
            "name": "score",
            "kind": "cpu",
            "command": _cpu_command(out, "score"),
            "outputs": [str(results / "report.json"), str(results / "rows.jsonl")],
        }
    )
    for stage in stages:
        if any(Path(p).exists() for p in stage["outputs"]):
            raise ValueError("cannot register after any output exists")
    for split in ("pop", "ti"):
        inputs.append(identity(source / f"data/external/mintqa-full/questions/{split}.jsonl"))
    unique_inputs = {r["path"]: r for r in inputs}
    estimate = (1.5 * sum(r["workload_seconds"] for r in pricing.values()) + 6 * 300) / 3600
    if estimate > 4:
        raise ValueError("projected GPU budget exceeds four hours")
    registration = {
        "artifact_kind": "xgraph.menu_control_registration",
        "version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "commit": commit,
        "source": str(source),
        "result_directory": str(results),
        "scope": "exploratory frozen MINTQA development only; no new first calls",
        "ranking": {
            "vocabulary": (
                "all writable relation IDs and aliases in the tokenizer-bound KG relation index"
            ),
            "score": (
                "AuditMenus: maximum alias IDF-weighted question stem overlap / "
                "sqrt(alias stem count)"
            ),
            "idf": (
                "one document per relation; alias stem union; "
                "log((1+relations)/(1+document_frequency))+1"
            ),
            "ties": (
                "best alias score then alias text; relation score then numeric PID; "
                "all global frequencies zero"
            ),
            "frontier_usage": (
                "displayed cardinality only; no filtering or frequency tie-break for global"
            ),
            "gold_free": True,
            "fixed_before_outputs": True,
        },
        "matching": {
            "menu_limit": 8,
            "endpoint_limit": 20,
            "explanation_token_tolerance": 5,
            "local_global_explanation_identical": True,
            "counts": (
                "same local frontier count/truncation metadata in both; explained identically"
            ),
            "payload_token_lengths": (
                "recorded, not equalized; relation labels can differ in length"
            ),
            "demonstrations": "unchanged original requests",
            "calls": "one cached first call plus one fresh second call for every question",
        },
        "decoding": {
            "relations": "frontier",
            "prompt_only": True,
            "committed_paths": True,
            "start_rank": "sitelinks",
            "unquoted_starts": True,
            "escapes": False,
            "guaranteed_answer_delivery": (
                "existing observe_terminal=8 forces bound answer prefix and EOS after a "
                "completed path; token truncation can still interrupt"
            ),
            "max_tokens": 256,
            "seed": 17,
            "terminal_width": 8,
        },
        "gate": {
            "contrast": "audit minus audit-global accuracy",
            "minimum_pp_each_family": 1.0,
            "loso_min_pp_each_family": ">0",
            "without_sport_pp_each_family": ">0",
            "conjunction": "both families; all conditions",
        },
        "analysis": {
            "contrasts": [["audit", "audit-global"], ["audit-global", "audit-nomenu"]],
            "bootstrap": (
                "paired relation-sequence clusters within hops, calibrated population weights; "
                "two-sided percentile 95%"
            ),
            "seed": 17,
            "resamples": 10000,
            "sport_sequence": SPORT,
            "loso": "leave one sequence out, renormalize remaining frozen weights; no refit",
            "C": "gold-compatible canonical label in capped committed answer support",
            "E": "C and incorrect or missing native answer; report A-without-C exceptions",
            "delivery": (
                "support/revise accepts usable second output; otherwise first answer; "
                "charge both calls"
            ),
        },
        "budget": {
            "maximum_gpu_hours": 4,
            "estimate_gpu_hours": estimate,
            "basis": pricing,
            "assumptions": (
                "1.5 times historical workload, global priced as local, "
                "300 seconds container/model startup per launch; CPU build and wait excluded"
            ),
            "stage_allowance": "min(1.5 * forecast_seconds + 300, remaining GPU seconds - 30)",
            "wall_timeout_seconds": 24 * 3600,
            "attempts": 1,
            "ledger": (
                "charge durable container_start through container_end, or confirmed cleanup; "
                "launch intent precedes Docker so failed starts are charged conservatively; "
                "missing end, invalid events or unconfirmed cleanup retain the full reservation; "
                "GPU-free corrections append original record hashes and hash-checked evidence"
            ),
        },
        "historical_reuse_audit": historical,
        "inputs": list(unique_inputs.values()),
        "stages": stages,
        "provenance": provenance([]),
    }
    script = out / "menu-control.sh"
    script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd "
        + shlex.quote(str(REPO))
        + "\nexec "
        + shlex.join(_cpu_command(out, "queue"))
        + ' "$@"\n'
    )
    script.chmod(0o755)
    registration["inputs"].append(identity(script))
    write_json(out / "registration.json", registration)
    (out / "registration.sha256").write_text(sha256_file(out / "registration.json") + "\n")


def verify(out):
    path = out / "registration.json"
    if sha256_file(path) != (out / "registration.sha256").read_text().strip():
        raise ValueError("registration changed")
    registration = json.loads(path.read_text())
    if clean_commit() != registration["commit"]:
        raise ValueError("registered commit changed")
    check_identities(registration["inputs"])
    return registration


def _receipt(stage):
    return Path(stage["outputs"][0] + ".receipt.json")


def _check_receipt(stage, seal):
    receipt = json.loads(_receipt(stage).read_text())
    if receipt["registration_sha256"] != seal or receipt["stage_sha256"] != digest(stage):
        raise ValueError("changed stage/registration receipt")
    if [r["path"] for r in receipt["outputs"]] != [
        str(Path(p).resolve()) for p in stage["outputs"]
    ]:
        raise ValueError("changed receipt output inventory")
    check_identities(receipt["outputs"])
    if "gpu_events" in stage:
        event_identity = receipt.get("gpu_events", {})
        if event_identity.get("path") != str(Path(stage["gpu_events"]).resolve()):
            raise ValueError("changed receipt GPU event inventory")
        check_identities([event_identity])


def _ledger_used(ledger):
    rows = read_jsonl(ledger) if ledger.exists() else []
    starts, completed, corrections = {}, {}, set()
    for row in rows:
        attempt, event = row["attempt"], row["event"]
        if event == "start" and attempt not in starts:
            seconds = row["reserved_seconds"]
            starts[attempt] = row
        elif event == "end" and attempt in starts and attempt not in completed:
            seconds = row["seconds"]
            completed[attempt] = row
        elif event == "correction" and attempt in completed and attempt not in corrections:
            if (
                row.get("state") != "reserved, no container started"
                or row.get("seconds") != 0
                or row.get("start_sha256") != digest(starts[attempt])
                or row.get("end_sha256") != digest(completed[attempt])
                or not row.get("evidence")
            ):
                raise ValueError("unproven GPU-free ledger correction")
            check_identities(row["evidence"])
            corrections.add(attempt)
            seconds = 0
        else:
            raise ValueError("invalid GPU ledger event sequence")
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid GPU ledger seconds")
    return sum(
        0
        if attempt in corrections
        else completed.get(attempt, {}).get("seconds", row["reserved_seconds"])
        for attempt, row in starts.items()
    )


def _append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(value + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _validate_generation(stage):
    """Check the completion contract in this stage's frozen generation flags."""
    output = Path(stage["outputs"][0])
    actual = json.loads(Path(str(output) + ".provenance.json").read_text())
    if flags(actual["provenance"]["command"]) != stage["expected_flags"]:
        raise ValueError("actual generation argv differs from registration")
    expected = identity(stage["expected_flags"]["--requests"][0])
    if expected not in actual["provenance"]["inputs"]:
        raise ValueError("generation request hash differs from registration")
    if actual["provenance"]["repo_commit"] != clean_commit() or actual["provenance"]["repo_dirty"]:
        raise ValueError("generation code differs from registration")
    terminal_width = int(stage["expected_flags"].get("--observe-terminal", ["0"])[0])
    if actual["relations"] != "frontier" or actual.get("observe_terminal", 0) != terminal_width:
        raise ValueError("generation changed KG execution or guaranteed delivery")
    asked = indexed(read_jsonl(Path(expected["path"])))
    records = read_jsonl(output)
    if set(indexed(records)) != set(asked):
        raise ValueError("incomplete generation output panel")
    for record in records:
        if not isinstance(record.get("text"), str):
            raise ValueError(f"invalid generation record text: {record['id']}")


def _gpu_seconds(path, offset, started, require_end=False):
    """Durable launch boundaries; an incomplete or invalid record fails closed."""
    if not path.exists():
        raise ValueError("GPU launch events disappeared")
    with path.open("rb") as stream:
        if os.fstat(stream.fileno()).st_size < offset:
            raise ValueError("GPU launch events were truncated")
        stream.seek(offset)
        data = stream.read()
    # A writer may append after the caller's clock sample and before this read.
    now = time.monotonic()
    if data and not data.endswith(b"\n"):
        raise ValueError("incomplete GPU launch event")
    active, seconds, launched, previous = None, 0.0, False, started
    for line in data.splitlines():
        event = json.loads(line)
        stamp = event["monotonic"]
        if not math.isfinite(stamp) or not previous <= stamp <= now:
            raise ValueError("invalid GPU launch event timestamp")
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


def _wait_stage(process, stage, allowance, started, events, offset):
    wall_bound = stage.get("wall_timeout_seconds", 3600)
    code = None
    while True:
        now = time.monotonic()
        seconds, launched = (
            _gpu_seconds(events, offset, started) if events else (now - started, True)
        )
        left = wall_bound - (now - started)
        if left <= 0 or (launched and seconds >= allowance):
            raise subprocess.TimeoutExpired(
                stage["command"], wall_bound if left <= 0 else allowance
            )
        if code is not None:
            return code
        with suppress(subprocess.TimeoutExpired):
            code = process.wait(
                timeout=min(STAGE_POLL_SECONDS, left, allowance - seconds)
                if launched
                else min(STAGE_POLL_SECONDS, left)
            )


def _stage_command(stage):
    if stage["kind"] != "gpu":
        return stage["command"]
    # Bound the CPU supervisor without imposing its virtual-address limit on
    # nvcc or CUDA. CPU stages retain their explicit hard limit.
    hard = resource.getrlimit(resource.RLIMIT_AS)[1]
    bound = "unlimited" if hard == resource.RLIM_INFINITY else str(hard)
    return ["prlimit", f"--as={bound}", *stage["command"]]


def execute_stage(stage, seal, log, ledger, budget_seconds):
    """Hash-checked resume with separate wall and conservative GPU occupancy bounds."""

    def status(message):
        _append(
            log,
            f"{stage.get('study', 'menu-control')} {stage['name']} "
            f"{message} {datetime.now(UTC).isoformat()}",
        )

    if _receipt(stage).exists():
        _check_receipt(stage, seal)
        status("skip")
        return
    if any(Path(p).exists() for p in stage["outputs"]):
        raise ValueError(f"partial output without receipt: {stage['name']}")
    remaining = budget_seconds - _ledger_used(ledger)
    if stage["kind"] == "gpu" and remaining <= 30:
        raise ValueError("GPU budget exhausted")
    if stage["kind"] == "gpu" and (
        not math.isfinite(stage["forecast_seconds"]) or stage["forecast_seconds"] <= 0
    ):
        raise ValueError("invalid registered stage forecast")
    wall_started = time.monotonic()
    if "inspect_command" in stage:
        inspection = subprocess.run(stage["inspect_command"], capture_output=True, timeout=20)
        if inspection.returncode == 0 or b"No such" not in inspection.stderr:
            raise ValueError("owned container name already exists or Docker inspection failed")
        status("waiting-headroom")
        subprocess.run(
            stage["headroom_command"], check=True, timeout=stage.get("wall_timeout_seconds", 3600)
        )
    allowance = (
        min(1.5 * stage["forecast_seconds"] + 300, remaining - 30)
        if stage["kind"] == "gpu"
        else 3600
    )
    attempt = time.time_ns()
    started = time.monotonic()
    timed_stage = {
        **stage,
        "wall_timeout_seconds": stage.get("wall_timeout_seconds", 3600) - (started - wall_started),
    }
    events = Path(stage["gpu_events"]) if "gpu_events" in stage else None
    if events:
        events.parent.mkdir(parents=True, exist_ok=True)
        with events.open("ab") as stream:
            stream.flush()
            os.fsync(stream.fileno())
    offset = events.stat().st_size if events and events.exists() else 0
    if stage["kind"] == "gpu":
        _append(
            ledger,
            json.dumps(
                {
                    "event": "start",
                    "attempt": attempt,
                    "stage": stage["name"],
                    "reserved_seconds": allowance + 30,
                    "gpu_allowance_seconds": allowance,
                    "gpu_events": str(events) if events else None,
                    "events_offset": offset,
                }
            ),
        )
    status("start")
    process = None
    settled = True
    try:
        with Path(stage["outputs"][0] + ".log").open("a") as output_log:
            process = subprocess.Popen(
                _stage_command(stage),
                cwd=REPO,
                stdout=output_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            code = _wait_stage(process, timed_stage, allowance, started, events, offset)
        if code:
            raise ValueError(f"stage failed with exit {code}: {stage['name']}")
        if events and not _gpu_seconds(events, offset, started, require_end=True)[1]:
            raise ValueError("stage succeeded without a GPU launch event")
        if "expected_flags" in stage:
            _validate_generation(stage)
        write_json(
            _receipt(stage),
            {
                "registration_sha256": seal,
                "stage_sha256": digest(stage),
                "outputs": [identity(p) for p in stage["outputs"]],
                **({"gpu_events": identity(events)} if events else {}),
            },
        )
        status("complete exit 0")
    except BaseException:
        settled = False
        try:
            # Stop launchers first: otherwise a child can create a container after cleanup.
            if process:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            settled = True
        finally:
            if "cleanup_command" in stage:
                launcher_stopped = settled
                settled = False
                cleanup = subprocess.run(stage["cleanup_command"], capture_output=True, timeout=20)
                settled = launcher_stopped and (
                    cleanup.returncode == 0 or b"No such" in cleanup.stderr
                )
            status("failed")
        raise
    finally:
        if stage["kind"] == "gpu":
            wall_seconds = time.monotonic() - started
            accounting = "measured"
            try:
                seconds, launched = (
                    _gpu_seconds(events, offset, started) if events else (wall_seconds, True)
                )
                if not settled:
                    seconds = allowance + 30
                    accounting = "reserved"
            except (ValueError, KeyError, TypeError, OSError):
                seconds, launched = allowance + 30, True
                accounting = "reserved"
            _append(
                ledger,
                json.dumps(
                    {
                        "event": "end",
                        "attempt": attempt,
                        "seconds": seconds,
                        "wall_seconds": wall_seconds,
                        "accounting": accounting,
                        "state": (
                            "GPU occupancy unknown; reservation retained"
                            if accounting == "reserved"
                            else "container launched"
                            if launched
                            else "reserved, no container started"
                        ),
                    }
                ),
            )


def queue(out, run=False):
    registration = verify(out)
    if not run:
        for stage in registration["stages"]:
            print(f"# {stage['name']}")
            for key in ("inspect_command", "headroom_command", "command", "cleanup_command"):
                if key in stage:
                    print(
                        ("# on failure: " if key == "cleanup_command" else "")
                        + shlex.join(_stage_command(stage) if key == "command" else stage[key])
                    )
            if "expanded_argv" in stage:
                print("# expanded host generation argv: " + shlex.join(stage["expanded_argv"]))
        return

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"owner queue received signal {signum}")

    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        with (out / "queue.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            (REPO / "target/tmp").mkdir(parents=True, exist_ok=True)
            seal = sha256_file(out / "registration.json")
            log = Path(registration["source"]) / "target/queue/menu-control.log"
            for stage in registration["stages"]:
                verify(out)
                execute_stage(stage, seal, log, out / "gpu-ledger.jsonl", 4 * 3600)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def finish(out, family):
    registration = verify(out)
    results = Path(registration.get("result_directory", out))
    seal = sha256_file(out / "registration.json")
    for stage in registration["stages"]:
        if stage["kind"] == "gpu" and stage["name"].startswith(family + "/"):
            _check_receipt(stage, seal)
    source = Path(registration["source"])
    requests = read_jsonl(source / "target/mintqa-rematch/panels/development/requests.jsonl")
    first_path = source / f"target/mintqa-rematch/{family}/development/paths-allfix.jsonl"
    first, asked, executor = read_jsonl(first_path), indexed(requests), _executor(source, family)
    for arm in ARMS:
        raw = read_jsonl(results / family / f"{arm}-raw.jsonl")
        support = [_capped_support(executor, r, asked[r["id"]]) for r in raw]
        write_jsonl(results / family / f"{arm}-support.jsonl", support)
        write_jsonl(
            results / family / f"{arm}.jsonl",
            revision_records(first, raw, support, list(asked), second_source=arm),
        )
    write_json(
        results / family / "finish.json",
        {"registration_sha256": seal, "provenance": provenance([first_path])},
    )


def score(out):
    from xgraph.mintqa_bench import answer_contains, dataset_rows
    from xgraph.programs import record_path
    from xgraph.prompts import final_answer
    from xgraph.score import record_outputs
    from xgraph.toolkg import KGStore

    registration = verify(out)
    results = Path(registration.get("result_directory", out))
    seal = sha256_file(out / "registration.json")
    # Complete receipts precede all gold reads. Never score an incomplete contrast.
    for stage in registration["stages"][:-1]:
        _check_receipt(stage, seal)
    source = Path(registration["source"])
    gold = dataset_rows(
        [f"{s}={source / f'data/external/mintqa-full/questions/{s}.jsonl'}" for s in ("pop", "ti")]
    )
    store = KGStore(source / "data/external/mintqa-full/kg.sqlite")
    requests = indexed(
        read_jsonl(source / "target/mintqa-rematch/panels/development/requests.jsonl")
    )
    report, detailed = {}, []
    for family in FAMILIES:
        executor = _executor(source, family)
        members = json.loads((out / family / "members.json").read_text())
        arms = {}
        for arm in ARMS:
            outputs = indexed(read_jsonl(results / family / f"{arm}.jsonl"))
            if set(outputs) != set(requests):
                raise ValueError("incomplete final output panel")
            texts = record_outputs(outputs.values(), requests)
            rows = {}
            for qid, record in outputs.items():
                support = _capped_support(executor, record, requests[qid])
                compatible = [
                    e
                    for e in support["reached"]
                    if answer_contains(store.label(int(e[1:])), gold[qid]["answer"])
                ]
                a, c = (
                    int(answer_contains(final_answer(texts[qid]), gold[qid]["answer"])),
                    int(bool(compatible)),
                )
                path = record_path(record)
                signature = (
                    [
                        sorted(
                            executor._index.relations.owner_ids[r]
                            for r in executor._rows_owned(executor._index.relations, name)
                        )
                        for name in path[1]
                    ]
                    if path
                    else None
                )
                rows[qid] = {
                    "id": qid,
                    "family": family,
                    "arm": arm,
                    "A": a,
                    "C": c,
                    "E": int(c and not a),
                    "identity_exception": bool(a and not c),
                    "support": support,
                    "relation_ids": signature,
                    "answer": final_answer(texts[qid]),
                    "revision_accepted": record["revision_accepted"],
                    "usage": record["usage"],
                }
            detailed.extend(rows.values())
            arms[arm] = rows
        contrasts = {}
        for a, b in registration["analysis"]["contrasts"]:
            result = paired_metrics(arms[a], arms[b], members)
            for kind in ("rescues", "regressions"):
                result[kind]["changed_relation_sequence"] = sum(
                    arms[a][q]["relation_ids"] != arms[b][q]["relation_ids"]
                    for q in result[kind]["ids"]
                )
            contrasts[f"{a}-{b}"] = result
        gate = contrasts["audit-audit-global"]["A"]
        mass = sum(m["weight"] for m in members)
        exceptions = [
            r["id"] for rows in arms.values() for r in rows.values() if r["identity_exception"]
        ]
        report[family] = {
            "arms_pct": {
                arm: {
                    k: 100 * sum(m["weight"] * rows[m["id"]][k] for m in members) / mass
                    for k in ("A", "C", "E")
                }
                for arm, rows in arms.items()
            },
            "contrasts": contrasts,
            "identity_exceptions": exceptions,
            "passes": bool(
                gate["difference_pp"] >= 1
                and gate["loso_min_pp"] > 0
                and gate["without_sport_pp"] > 0
            ),
        }
        del executor
    write_jsonl(results / "rows.jsonl", detailed)
    write_json(
        results / "report.json",
        {
            "status": "completed exploratory development analysis",
            "registration_sha256": seal,
            "families": report,
            "gate_passes": all(r["passes"] for r in report.values()),
            "gpu_ledger_seconds": _ledger_used(out / "gpu-ledger.jsonl"),
            "limitations": [
                "development exposed; intervals do not undo selection",
                "canonical-label compatibility is not semantic correctness",
                "fixed-plan capped replay is bookkeeping, not causal mediation",
                "shared terminal delivery across reruns; no reuse of historical accuracy contrast",
            ],
            "provenance": provenance(
                [
                    out / "registration.json",
                    *(results / f / f"{a}.jsonl" for f in FAMILIES for a in ARMS),
                ]
            ),
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "register", "verify", "queue", "finish", "score")
    )
    parser.add_argument("--source-root", type=Path, default=Path("/home/jonas/git/xgraph"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--attempt", help="fresh result directory below attempts/; register only")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.attempt is not None and args.command != "register":
        parser.error("--attempt is only valid for register")
    out, source = args.out.resolve(), args.source_root.resolve()
    if not (out.is_relative_to(REPO / "target") or out.is_relative_to(Path("/tmp"))):
        parser.error("outputs must be in this worktree's target or /tmp")
    os.nice(max(0, 10 - os.getpriority(os.PRIO_PROCESS, 0)))
    hard = resource.getrlimit(resource.RLIMIT_AS)[1]
    resource.setrlimit(
        resource.RLIMIT_AS,
        (CPU_LIMIT, hard if args.command == "queue" else CPU_LIMIT),
    )
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    if args.command == "prepare":
        if not args.family:
            parser.error("prepare requires --family")
        prepare(source, out, args.family)
    elif args.command == "register":
        register(source, out, attempt=args.attempt)
    elif args.command == "verify":
        verify(out)
    elif args.command == "queue":
        queue(out, args.run)
    elif args.command == "finish":
        if not args.family:
            parser.error("finish requires --family")
        finish(out, args.family)
    else:
        score(out)


if __name__ == "__main__":
    main()
