"""Owner-run development queue for variable-stride actions; preview is CPU-only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

from xgraph.mintqa_panels import indexed, read_jsonl, write_json, write_jsonl
from xgraph.provenance import provenance, sha256_file
from xgraph.queue_accounting import gpu_occupancy, wait_stage
from xgraph.score import replay_failure

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/home/jonas/git/xgraph")
FAMILIES = {"qwen38": "Qwen3.8-27B-FP8", "gemma31": "gemma-4-31B-it-FP8-dynamic"}
FIRST = {
    "qwen38": DATA
    / ".claude/worktrees/codex-menu2/target/menu2-1024/development/menuobs-width8.jsonl",
    "gemma31": DATA
    / ".claude/worktrees/codex-integrate/target/mintqa-width8/first"
    / "gemma31/development/menuobs-width8.jsonl",
}


def wt_directory(family):
    return DATA / (
        ".claude/worktrees/codex-integrate/target/mintqa-width8/terminal/"
        f"menuobs-width8/{family}/development"
    )


def stages(out):
    """Pure argv construction: no GPU, Docker, probes or filesystem writes."""
    out = Path(out).resolve()
    python = str(ROOT / ".venv/bin/python")
    cli = [python, "-m", "xgraph.operator_queue", "--out", str(out)]
    result = [
        {
            "name": "native-build",
            "argv": [
                "uv",
                "sync",
                "--locked",
                "--reinstall-package",
                "xgraph",
                "--config-setting",
                "cmake.define.CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
            ],
            "env": {"CMAKE_BUILD_PARALLEL_LEVEL": "1"},
            "outputs": [str(out / "native-build.log")],
        },
        {
            "name": "native-agreement",
            "argv": [
                python,
                "-m",
                "pytest",
                "-q",
                "tests/test_engine.py",
                "tests/test_vllm_adapter.py",
            ],
            "env": {},
            "gpu_seconds": 1800,
            "need_gib": 16,
            "outputs": [str(out / "native-agreement.log")],
        },
    ]
    reset = {
        "XG_START_ATTEMPTS": "1",
        "XG_HEADROOM_WAIT": "30",
        "XG_AVAILABLE_WAIT": "43200",
        "XG_GPU_IDLE": "1",
        "XG_PATHS": "1",
        "XG_COMMITTED_PATHS": "1",
        "XG_ESCAPES": "0",
        "XG_UNQUOTED_STARTS": "1",
        "XG_START_RANK": "sitelinks",
        "XG_END_AFTER_ANSWER": "1",
        "XG_RELATIONS": "frontier",
        "XG_EXPRESSIONS": "0",
        "XG_STOP": "",
        "XG_KG_VERSIONS": "",
        "XG_LORA": "",
        "XG_DERIVATION": "0",
        "XG_RELEASE_MASS": "",
        "XG_RELEASE_TOP": "0",
        "XG_ESCAPE_BIAS": "",
        "XG_REPEATS": "1",
        "XG_BOUND_ESCAPES": "1",
        "XG_OBSERVE": "0",
        "XG_OBSERVE_SELECT": "0",
        "XG_OBSERVE_ANSWER": "0",
        "XG_MENU": "0",
        "XG_PATH_DEMOS": "",
        "XG_OBSERVE_WIDTH": "0",
        "XG_OBSERVE_DEFER": "0",
        "XG_MENU_CLOSE_NEXT": "0",
        "XG_OBSERVE_REQUIRED": "0",
        "XG_OBSERVE_NO_LOOP": "0",
        "XG_OBSERVE_TERMINAL": "0",
        "XG_AUDIT": "0",
        "XG_OPEN": "0",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    for family, model_name in FAMILIES.items():
        model = Path("/home/jonas/models") / model_name
        index = DATA / f"target/mintqa-rematch/indexes/{family}"
        need = 75 if family == "gemma31" else 60
        env = {
            **reset,
            "XG_KG_INDEX": str(index),
            "XG_TOKENIZER": str(model / "tokenizer.json"),
            "XG_TOOLS_DB": str(DATA / "data/external/mintqa-full/kg.sqlite"),
            "XG_NEED_GIB": str(need),
            "XG_PAGECACHE_ROOTS": f"{DATA / 'data'}:{model}",
        }
        common = [
            "--max-model-len",
            "16384",
            "--max-num-seqs",
            "64",
            "--kv-cache-gib",
            "16" if family == "gemma31" else "8",
            "--seed",
            "2402",
            "--no-prefix-caching",
        ]
        if family == "gemma31":
            common += ["--ban-tokens", "<|channel>"]
        for arm, stride in (("new", 2), ("control", 1)):
            directory = out / family
            planner = directory / f"{arm}-planner.jsonl"
            requests = directory / f"{arm}-delivery-requests.jsonl"
            delivered = directory / f"{arm}-delivery.jsonl"
            for name, asked, kind, output, options, seconds, view in (
                (
                    "plan",
                    out / "requests.jsonl",
                    "walk",
                    planner,
                    [
                        "--matched-paths",
                        "--walk-stride",
                        str(stride),
                        "--max-calls",
                        "8",
                        "--start-rank",
                        "sitelinks",
                        "--unquoted-starts",
                        "--max-tokens",
                        "64",
                    ],
                    1800,
                    {},
                ),
                (
                    "deliver",
                    requests,
                    "constrained",
                    delivered,
                    ["--max-tokens", "256"],
                    900,
                    {"XG_OBSERVE_TERMINAL": "8"},
                ),
            ):
                if name == "deliver":
                    result.append(
                        {
                            "name": f"{family}-{arm}-requests",
                            "env": {},
                            "argv": [
                                *cli,
                                "delivery-requests",
                                "--planner",
                                str(planner),
                                "--destination",
                                str(requests),
                            ],
                            "outputs": [str(requests)],
                        }
                    )
                result.append(
                    {
                        "name": f"{family}-{arm}-{name}",
                        "env": {**env, **view},
                        "argv": [
                            str(ROOT / "scripts/generate.sh"),
                            str(model),
                            str(asked),
                            kind,
                            str(output),
                            *options,
                            *common,
                        ],
                        "gpu_seconds": seconds,
                        "need_gib": need,
                        "requests": str(asked),
                        "empty_output": str(output),
                        "outputs": [
                            str(output),
                            f"{output}.summary.json",
                            f"{output}.provenance.json",
                        ],
                    }
                )
        result.append(
            {
                "name": f"{family}-audit",
                "env": {},
                "gpu_seconds": 900,
                "need_gib": 16,
                "argv": [*cli, "audit", "--family", family],
                "outputs": [str(out / family / f"{a}.jsonl") for a in ("WT", "new", "control")]
                + [str(out / family / "audit.json")],
            }
        )
    result.append(
        {"name": "score", "env": {}, "argv": [*cli, "score"], "outputs": [str(out / "score.json")]}
    )
    for stage in result:
        # Both uv sync and generate.sh's isolated wheel build must avoid the
        # shared cache, even when the owner shell exports another UV_CACHE_DIR.
        stage["env"].update(
            UV_CACHE_DIR=str(ROOT / ".uv-cache"),
            UV_LINK_MODE="hardlink",
            TMPDIR=str(ROOT / "target/tmp"),
        )
        stage["wall_timeout_seconds"] = 24 * 3600
        if stage.get("requests"):
            stage["gpu_events"] = str(out / f"{stage['name']}.gpu-events.jsonl")
            stage["env"]["XG_GPU_EVENTS"] = stage["gpu_events"]
    return result


def preview(plan):
    for stage in plan:
        print(
            f"# new-operator {stage['name']}; GPU allowance {stage.get('gpu_seconds', 0)} seconds; "
            f"wall bound {stage.get('wall_timeout_seconds', 24 * 3600)} seconds"
        )
        print(shlex.join(["env", *[f"{k}={v}" for k, v in stage["env"].items()], *stage["argv"]]))


def command_hash(stage):
    return hashlib.sha256(json.dumps(stage, sort_keys=True).encode()).hexdigest()


def receipt(stage):
    return {
        "command_sha256": command_hash(stage),
        "outputs": [{"path": p, "sha256": sha256_file(Path(p))} for p in stage["outputs"]],
    }


def reusable(stage, saved, *, seal_sha256=None):
    return (
        (seal_sha256 is None or saved.get("manifest_sha256") == seal_sha256)
        and saved.get("command_sha256") == command_hash(stage)
        and saved.get("outputs")
        == [
            {"path": p, "sha256": sha256_file(Path(p)) if Path(p).is_file() else None}
            for p in stage["outputs"]
        ]
        and (
            not stage.get("gpu_events")
            or saved.get("gpu_events")
            == {
                "path": stage["gpu_events"],
                "sha256": sha256_file(Path(stage["gpu_events"]))
                if Path(stage["gpu_events"]).is_file()
                else None,
            }
        )
    )


def reserved_seconds(ledger):
    """Charge finished attempts' measured time; unknown attempts keep the full reservation."""
    total = 0
    for event in ledger:
        charge = event["reserved_seconds"]
        if event["status"] in {"complete", "failed"}:
            charge = event.get("elapsed_seconds", charge)
        if type(charge) not in (int, float) or not math.isfinite(charge) or charge < 0:
            raise ValueError("invalid GPU ledger charge: expected finite non-negative seconds")
        total += charge
    return total


def write_ledger(path, ledger):
    """Atomically replace mutable ledger state; sealed artifacts still use write_json."""
    import tempfile

    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(ledger, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def gate_metrics(members, arms, gold):
    """Fixed policy, relation-sequence cross-fitting; no fitting on held-out outcomes."""
    import re

    from xgraph.mintqa_bench import answer_contains
    from xgraph.prompts import final_answer
    from xgraph.score import record_failure

    ids = {m["id"] for m in members}
    if set(gold) != ids or set(arms) != {"WT", "new", "control"}:
        raise ValueError("gate needs complete gold and all three arms")
    correct, costs = {}, {}
    fields = ("prompt_tokens", "generated_tokens", "model_calls", "forced_tokens")
    for arm, rows in arms.items():
        by_id = indexed(rows)
        if set(by_id) != ids:
            raise ValueError("gate needs complete arm ids")
        for row in rows:
            if any(type(row.get(k)) is not int or row[k] < 0 for k in fields):
                raise ValueError("missing complete policy cost")
            if row["forced_tokens"] > row["generated_tokens"]:
                raise ValueError("forced cost exceeds generated tokens")
        costs[arm] = {k: sum(r[k] for r in rows) for k in fields}
        correct[arm] = {
            q: not record_failure(r) and answer_contains(final_answer(r["text"]), gold[q])
            for q, r in by_id.items()
        }

    def delta(subset, control="WT"):
        mass = sum(m["weight"] for m in subset)
        return (
            100
            * sum(
                m["weight"] * (int(correct["new"][m["id"]]) - int(correct[control][m["id"]]))
                for m in subset
            )
            / mass
            if mass
            else None
        )

    clusters = sorted(
        {m["cluster"] for m in members}, key=lambda c: hashlib.sha256(f"2402:{c}".encode()).digest()
    )
    fold_of = {c: n % 2 for n, c in enumerate(clusters)}
    folds = [delta([m for m in members if fold_of[m["cluster"]] == k]) for k in (0, 1)]
    loso = [delta([m for m in members if m["cluster"] != c]) for c in clusters]
    sport = delta([m for m in members if not re.search(r"\bsport\b", m["cluster"], re.I)])
    uplift, matched = delta(members), delta(members, "control")
    minimum = min((v for v in loso if v is not None), default=None)
    positive = lambda v: v is not None and v > 0  # noqa: E731
    return {
        "new_minus_WT_pp": uplift,
        "new_minus_control_pp": matched,
        "cross_fit_fold_pp": folds,
        "cross_fit_fit": "none; policy fixed before outputs",
        "LOSO_minimum_pp": minimum,
        "without_sport_pp": sport,
        "cost": costs,
        "pass": uplift is not None
        and uplift >= 0.5
        and positive(matched)
        and all(map(positive, folds))
        and positive(minimum)
        and positive(sport),
    }


def wt_config(path, *, first, family):
    """Match the declared WT profile, then hash every non-path generation flag.

    Extra generation flags are refused. This admits the historical omission of
    unchanged defaults, with the actual first/second configs retained separately.
    """
    from xgraph.catalog_argv import PATH_ROLES, flags

    config = json.loads(Path(f"{path}.provenance.json").read_text())
    expected = {
        "paths": True,
        "committed_paths": True,
        "escapes": False,
        "start_rank": "sitelinks",
        "unquoted_starts": True,
        "end_after_answer": True,
        "relations": "frontier",
    }
    expected.update(
        {
            "menu": 5,
            "observe_paths": 3,
            "observe_width": 8,
            "observe_select": True,
            "observe_required": True,
            "observe_no_loop": True,
            "observe_answer": True,
        }
        if first
        else {"observe_terminal": 8}
    )
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError(f"WT config does not match the preregistered profile: {path}")
    argv = flags(config["provenance"]["command"])
    wanted = {
        "--arm": ["constrained"],
        "--max-tokens": ["1024" if first else "256"],
        "--seed": ["0" if first else "17"],
        "--memory-share": ["0.450"],
    }
    if any(argv.get(k) != v for k, v in wanted.items()):
        raise ValueError(f"WT generation controls changed: {path}")
    allowed = (
        set(PATH_ROLES)
        | set(wanted)
        | {
            "--no-escapes",
            "--paths",
            "--committed-paths",
            "--unquoted-starts",
            "--end-after-answer",
            "--relations",
            "--start-rank",
            "--provenance-only",
            "--open",
            "--ban-tokens",
            "--observe-paths",
            "--menu",
            "--path-demos",
            "--observe-width",
            "--observe-select",
            "--observe-required",
            "--observe-no-loop",
            "--observe-answer",
            "--observe-terminal",
            "--max-model-len",
            "--max-num-seqs",
            "--kv-cache-gib",
            "--relation-labels",
        }
    )
    if set(argv) - allowed:
        raise ValueError(f"unknown WT flags: {set(argv) - allowed}")
    normalized = {k: v for k, v in argv.items() if k not in PATH_ROLES and k != "--provenance-only"}
    for key, default in (
        ("--max-model-len", "8192"),
        ("--max-num-seqs", "64"),
        ("--kv-cache-gib", "8"),
    ):
        normalized.setdefault(key, [default])
    desired = {
        **wanted,
        "--no-escapes": [],
        "--paths": [],
        "--committed-paths": [],
        "--unquoted-starts": [],
        "--end-after-answer": [],
        "--relations": ["frontier"],
        "--start-rank": ["sitelinks"],
        "--open": [],
        "--max-model-len": ["8192" if first else "16384"],
        "--max-num-seqs": ["64"],
        "--kv-cache-gib": ["16" if family == "gemma31" and not first else "8"],
    }
    if first:
        desired.update(
            {
                "--observe-paths": ["3"],
                "--menu": ["5"],
                "--path-demos": ["5"],
                "--observe-width": ["8"],
                "--observe-select": [],
                "--observe-required": [],
                "--observe-no-loop": [],
                "--observe-answer": [],
            }
        )
    else:
        desired["--observe-terminal"] = ["8"]
    if family == "gemma31":
        desired["--ban-tokens"] = ["<|channel>"]
    if normalized != desired:
        raise ValueError(f"WT config hash mismatch: {path}")
    return {
        "sha256": hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest(),
        "flags": normalized,
        "profile": expected,
    }


def tree_commit():
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("owner queue refuses a dirty worktree")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def cached_input_files(path):
    """Generation sidecars and CPU composites' embedded provenance stay explicit."""
    path = Path(path)
    summary, sidecar = Path(f"{path}.summary.json"), Path(f"{path}.provenance.json")
    files = [path, summary]
    if sidecar.is_file():
        return [*files, sidecar]
    record = json.loads(summary.read_text())
    if not (
        record.get("artifact_kind", "").startswith("xgraph.mintqa_rematch.")
        and isinstance(record.get("provenance"), dict)
    ):
        raise ValueError(f"cached source lacks provenance: {path}")
    return files


def prepare(out):
    """Freeze existing sample, input inventory and every owner argv after committing."""
    commit = tree_commit()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise ValueError("refusing to replace the frozen queue manifest")
    inputs = [
        out / "panel.json",
        out / "requests.jsonl",
        ROOT / "target/new-operator/gate.json",
        ROOT / "target/new-operator/selection-rule.json",
        ROOT / "target/new-operator/replay.json",
        DATA / "data/external/mintqa-full/kg.sqlite",
        DATA / "data/external/mintqa-full/kg/relation_labels.jsonl",
    ]
    contracts, weights = {}, []
    for family, name in FAMILIES.items():
        directory = wt_directory(family)
        second = directory / "menuobs-width8-revision-terminal-raw.jsonl"
        contracts[family] = {
            "first": wt_config(FIRST[family], first=True, family=family),
            "second": wt_config(second, first=False, family=family),
        }
        for p in (FIRST[family], second, directory / "menuobs-width8-two-pass-terminal.jsonl"):
            inputs += cached_input_files(p)
        inputs += [directory / "menuobs-width8-revision-terminal-requests.jsonl"]
        index = DATA / f"target/mintqa-rematch/indexes/{family}"
        inputs += sorted(p for p in index.iterdir() if p.is_file())
        model = Path("/home/jonas/models") / name
        for filename in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
            "model.safetensors.index.json",
        ):
            if (model / filename).is_file():
                inputs.append(model / filename)
        # Revision metadata plus size/mtime inventory avoids streaming checkpoints
        # through this shared-memory machine while another queue owns the GPU.
        metadata = model / ".cache/huggingface/download"
        inputs += sorted(metadata.glob("*.metadata"))
        weights.extend(
            {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted(model.glob("*.safetensors"))
        )
    proxy = out / "bin/docker"
    proxy.parent.mkdir(exist_ok=True)
    proxy.write_text(
        f"#!/bin/sh\nexec {shlex.quote(str(ROOT / '.venv/bin/python'))} "
        '-m xgraph.operator_queue docker-proxy "$@"\n'
    )
    proxy.chmod(0o755)
    inputs += [proxy, out / "preselect.py"]
    record = {
        "artifact_kind": "xgraph.new_operator_queue",
        "code_commit": commit,
        "stages": stages(out),
        "gpu_budget_seconds": 21600,
        "WT_configs": contracts,
        "weight_inventory_kind": "revision metadata hash and size/mtime; no weight payload SHA256",
        "weight_inventory": weights,
        "provenance": provenance(list(dict.fromkeys(inputs))),
    }
    write_json(out / "manifest.json", record)
    launcher = out / "new-operator.sh"
    launcher.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(ROOT))}\n"
        f"exec nice -n 10 {shlex.quote(str(ROOT / '.venv/bin/python'))} "
        f'-m xgraph.operator_queue --out {shlex.quote(str(out))} queue "${{@:---dry-run}}"\n'
    )
    launcher.chmod(0o755)


def verify(out):
    record = json.loads((out / "manifest.json").read_text())
    if tree_commit() != record["code_commit"]:
        raise ValueError("queue code commit changed")
    for entry in record["provenance"]["inputs"]:
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"queue input changed: {entry['path']}")
    for entry in record["weight_inventory"]:
        stat = Path(entry["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (entry["bytes"], entry["mtime_ns"]):
            raise ValueError(f"model weight inventory changed: {entry['path']}")
    return record


def forced_token_count(engine, tokens):
    """Replay retained generated ids; forced ids are a subset, never extra cost."""
    import numpy as np

    forced = 0
    for token in tokens:
        engine.fill_mask()
        mask = engine.mask_host()[0].view(np.uint32)
        if not int(mask[token // 32]) & (1 << (token % 32)):
            raise ValueError(f"serving mask replay disallows retained token {token}")
        forced += int(np.bitwise_count(mask).sum()) == 1
        engine.advance(np.array([token], dtype=np.int32))
    return forced


def native_counts(index, tokenizer, model, rows, questions, profile):
    """GPU-owner only: same deterministic mask state, opened spans and prefills."""
    import gc

    from xgraph.engine import Engine

    config = json.loads((model / "tokenizer_config.json").read_text())
    eos_text = config["eos_token"]
    if isinstance(eos_text, dict):
        eos_text = eos_text["content"]
    eos = set(tokenizer.encode(eos_text))
    generation = json.loads((model / "generation_config.json").read_text())
    value = generation.get("eos_token_id", [])
    eos.update([value] if isinstance(value, int) else value)
    options = {("observe" if k == "observe_paths" else k): v for k, v in profile.items()}
    options.update(eos_token_ids=sorted(eos))
    if options.get("menu"):
        options["relation_labels"] = DATA / "data/external/mintqa-full/kg/relation_labels.jsonl"
    engine = Engine(index, tokenizer, index.span_syntax, 1, **options)
    result = {}
    for row in rows:
        if replay_failure(row):
            result[row["id"]] = 0
            continue
        engine.reset()
        if options.get("menu"):
            engine.set_relation_question(0, questions[row["id"]])
        if row.get("opener"):
            engine.open_span(0)
        if row.get("prefill"):
            # Prefix ids are prompt work; verify them but never add them to the
            # generated or forced-generated counts.
            forced_token_count(engine, tokenizer.encode(row["prefill"]))
        result[row["id"]] = forced_token_count(engine, row["token_ids"])
        if engine.errors():
            raise ValueError(f"native replay error in {row['id']}")
    del engine
    gc.collect()
    return result


def answer_witnesses(row, request, executor, index, tokenizer):
    """Check each emitted Answer span against the exact capped concrete witness."""
    import re

    from xgraph.programs import record_path

    if replay_failure(row):
        return {}
    emitted = re.findall(r"\bAnswer:\s*⟦\s*([^⟧]+)⟧", row["text"])
    if not emitted:
        return {}
    parsed = record_path(row)
    if parsed is None:
        raise ValueError(f"answer has no concrete path: {row['id']}")
    support = executor.execute_set(
        *parsed, request.get("anchor"), request.get("kg_version"), witnesses=True
    )
    verified = {}
    for answer in emitted:
        entities = executor.named(answer.strip()) & set(support["reached"])
        if not entities:
            raise ValueError(f"answer outside its witnessed support: {row['id']}")
        for entity in entities:
            chain = support["witnesses"][entity]
            for subject, label, obj in zip(chain, parsed[1], chain[1:], strict=False):
                tokens = tokenizer.encode(index.span_syntax.label_prefix + label)
                lo, hi = index.relations.prefix_range(tokens)
                props = {
                    int(index.relations.owners[r])
                    for r in range(lo, hi)
                    if index.relations.row_tokens(r) == tuple(tokens)
                }
                relations, objects = index.edges_at(index._entity_at[subject])
                if not any(
                    int(r) in props and index.labels.owner_id(int(o)) == obj
                    for r, o in zip(relations, objects, strict=True)
                ):
                    raise ValueError(f"unsound witness edge: {row['id']}")
            verified[entity] = chain
    return verified


def audit_family(out, family):
    """Native token audit and complete-policy merge, before any gold is opened."""
    from xgraph.index import load_kg_index
    from xgraph.mintqa_rematch import usage
    from xgraph.programs import MatchedExecutor
    from xgraph.tokenizer import ServedTokenizer

    manifest = json.loads((out / "manifest.json").read_text())
    requests = indexed(read_jsonl(out / "requests.jsonl"))
    if any(r.get("anchor") or r.get("kg_version") for r in requests.values()):
        raise ValueError("this frozen MINTQA native replay needs unanchored, unversioned requests")
    questions = {
        q: next(m["content"] for m in reversed(r["messages"]) if m["role"] == "user")
        for q, r in requests.items()
    }
    index = load_kg_index(DATA / f"target/mintqa-rematch/indexes/{family}")
    model = Path("/home/jonas/models") / FAMILIES[family]
    tokenizer = ServedTokenizer.from_file(model / "tokenizer.json")
    executor = MatchedExecutor(index, tokenizer, start_rank="sitelinks", unquoted_starts=True)
    directory, destination = wt_directory(family), out / family
    inputs, counts, sources = [], {}, {}
    for phase, path in (
        ("first", FIRST[family]),
        ("second", directory / "menuobs-width8-revision-terminal-raw.jsonl"),
    ):
        rows = [r for r in read_jsonl(path) if r["id"] in requests]
        if {r["id"] for r in rows} != set(requests):
            raise ValueError("incomplete WT source")
        counts[phase] = native_counts(
            index,
            tokenizer,
            model,
            rows,
            questions,
            manifest["WT_configs"][family][phase]["profile"],
        )
        sources[phase] = indexed(rows)
        inputs.append(path)
    wt = [
        r
        for r in read_jsonl(directory / "menuobs-width8-two-pass-terminal.jsonl")
        if r["id"] in requests
    ]
    for row in wt:
        original = usage(row)
        if any(
            original[k] != sum(usage(sources[p][row["id"]])[k] for p in ("first", "second"))
            for k in ("prompt_tokens", "generated_tokens", "model_calls")
        ):
            raise ValueError(f"WT complete-policy usage mismatch: {row['id']}")
        row.update(usage(row))
        row["forced_tokens"] = counts["first"][row["id"]] + counts["second"][row["id"]]
        row["answer_witnesses"] = answer_witnesses(
            row, requests[row["id"]], executor, index, tokenizer
        )
    write_jsonl(destination / "WT.jsonl", wt)
    for arm in ("new", "control"):
        planning_path, delivery_path = (
            destination / f"{arm}-planner.jsonl",
            destination / f"{arm}-delivery.jsonl",
        )
        planned, delivered = read_jsonl(planning_path), indexed(read_jsonl(delivery_path))
        if {r["id"] for r in planned} != set(requests):
            raise ValueError("incomplete planning arm")
        if set(delivered) != {r["id"] for r in planned if r["plan"] is not None}:
            raise ValueError("delivery outputs must match exactly every concrete plan")
        profile = manifest["WT_configs"][family]["second"]["profile"]
        counts[arm] = native_counts(
            index, tokenizer, model, list(delivered.values()), questions, profile
        )
        merged = []
        for plan in planned:
            generated = delivered.get(plan["id"])
            row = {**plan, "policy": arm, "delivery": "engine" if generated else "unanswered"}
            if generated:
                row.update(
                    text=generated["text"],
                    token_ids=generated["token_ids"],
                    finish_reason=generated["finish_reason"],
                )
                for field in (
                    "declined",
                    "truncated",
                    "escaped",
                    "malformed",
                    "error",
                    "replay_status",
                    "replay_reason",
                    "prefill",
                ):
                    if field in generated:
                        row[field] = generated[field]
                if replay_failure(generated) and generated.get("replay_status") not in (
                    "declined",
                    "unconfirmed",
                ):
                    row["replay_status"] = (
                        "declined" if generated.get("declined") else "unconfirmed"
                    )
                    row.setdefault("replay_reason", "worker replay receipt missing or invalid")
            for field in ("prompt_tokens", "generated_tokens", "model_calls"):
                row[field] = usage(plan)[field] + (usage(generated)[field] if generated else 0)
            row["forced_tokens"] = counts[arm].get(plan["id"], 0)
            row["answer_witnesses"] = answer_witnesses(
                row, requests[row["id"]], executor, index, tokenizer
            )
            row["cost_sources"] = [str(planning_path), *([str(delivery_path)] if generated else [])]
            merged.append(row)
        write_jsonl(destination / f"{arm}.jsonl", merged)
        inputs += [planning_path, delivery_path]
    write_json(
        destination / "audit.json",
        {
            "mask_replay": "all retained ids admitted; forced generated subset counted",
            "forced_tokens_by_call": counts,
            "witnesses": "all emitted Answer spans checked",
            "provenance": provenance([*inputs, out / "manifest.json"]),
        },
    )


def score(out):
    manifest = verify(out)
    seal = sha256_file(out / "manifest.json")
    # Gold remains unread until every generation and native audit has a receipt.
    for stage in manifest["stages"]:
        if stage["name"] == "score":
            continue
        path = out / f"{stage['name']}.receipt.json"
        if not path.exists() or not reusable(stage, json.loads(path.read_text()), seal_sha256=seal):
            raise ValueError(f"cannot score before complete stage receipt: {stage['name']}")
    members = json.loads((out / "panel.json").read_text())["members"]
    ids, gold, inputs = {m["id"] for m in members}, {}, [out / "manifest.json", out / "panel.json"]
    for split in ("pop", "ti"):
        path = DATA / f"data/external/mintqa-full/questions/{split}.jsonl"
        inputs.append(path)
        with path.open() as stream:
            for n, line in enumerate(stream):
                qid = f"mintqa-{split}-{n}"
                if qid in ids:
                    gold[qid] = json.loads(line)["answer"]
    results = {}
    for family in FAMILIES:
        arms = {a: read_jsonl(out / family / f"{a}.jsonl") for a in ("WT", "new", "control")}
        results[family] = gate_metrics(members, arms, gold)
        inputs += [out / family / f"{a}.jsonl" for a in arms]
    write_json(
        out / "score.json",
        {
            "scope": "development admission only",
            "families": results,
            "pass": all(r["pass"] for r in results.values()),
            "reserved_GPU_ledger": json.loads((out / "gpu-ledger.json").read_text()),
            "provenance": provenance(inputs),
        },
    )


def guarded_docker_argv(argv, label, seconds):
    if argv[:1] != ["run"]:
        return argv
    if type(seconds) is not int or seconds < 1:
        raise ValueError("container GPU lifetime must be a positive integer")
    at = argv.index("--entrypoint")
    if argv[at + 1] != "python3" or argv[at + 3 : at + 5] != ["-m", "xgraph.generate"]:
        raise ValueError("unexpected generator container command")
    guard = (
        "import os,signal,sys,runpy; "
        "signal.signal(signal.SIGALRM,lambda *_:os._exit(124)); "
        f"signal.alarm({seconds}); sys.argv=sys.argv[2:]; "
        "runpy.run_module('xgraph.generate',run_name='__main__')"
    )
    return [
        "run",
        "--label",
        f"xgraph.new_operator={label}",
        *argv[1 : at + 3],
        "-c",
        guard,
        *argv[at + 3 :],
    ]


def docker_proxy(argv):
    real, label = os.environ["XG_OPERATOR_REAL_DOCKER"], os.environ["XG_OPERATOR_STAGE_LABEL"]
    argv = guarded_docker_argv(argv, label, int(os.environ["XG_OPERATOR_STAGE_SECONDS"]))
    os.execv(real, [real, *argv])


def cpu_limits():
    import resource

    resource.setrlimit(resource.RLIMIT_DATA, (4 * 1024**3, 4 * 1024**3))
    priority = os.getpriority(os.PRIO_PROCESS, 0)
    if priority < 10:
        os.nice(10 - priority)


def validate_stage_outputs(stage):
    """An exit status alone cannot certify complete generation artifacts."""
    from xgraph.mintqa_rematch import usage

    for filename in stage["outputs"]:
        path = Path(filename)
        if not path.is_file():
            raise ValueError(f"missing stage output: {path}")
        if path.suffix == ".json":
            if not isinstance(json.loads(path.read_text()), dict):
                raise ValueError(f"stage output must be a JSON object: {path}")
        elif path.suffix == ".jsonl":
            indexed(read_jsonl(path))
    if not stage.get("requests"):
        return
    expected = indexed(read_jsonl(Path(stage["requests"])))
    rows = indexed(read_jsonl(Path(stage["empty_output"])))
    if set(rows) != set(expected):
        raise ValueError("generation outputs must match exactly every request id")
    for row in rows.values():
        if (
            not isinstance(row.get("text"), str)
            or not isinstance(row.get("finish_reason"), str)
            or not isinstance(row.get("token_ids"), list)
            or any(type(t) is not int or t < 0 for t in row["token_ids"])
        ):
            raise ValueError(f"invalid generation record: {row['id']}")
        usage(row)
        if stage["name"].endswith("-plan"):
            if "plan" not in row:
                raise ValueError(f"missing planner plan: {row['id']}")
            plan = row["plan"]
            if plan is not None and (
                not isinstance(plan, dict)
                or not isinstance(plan.get("start"), str)
                or not plan["start"]
                or not isinstance(plan.get("relations"), list)
                or any(not isinstance(r, str) or not r for r in plan["relations"])
            ):
                raise ValueError(f"invalid planner plan: {row['id']}")


def run_queue(out, *, dry_run=True):
    import fcntl
    import signal
    import time
    from datetime import UTC, datetime

    if dry_run:
        record = json.loads((out / "manifest.json").read_text())
        ledger_path = out / "gpu-ledger.json"
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
        charged, cap = reserved_seconds(ledger), record["gpu_budget_seconds"]
        print(
            f"# new-operator GPU ledger: charged {charged:g} seconds; "
            f"remaining {cap - charged:g} of {cap:g} seconds"
        )
        preview(record["stages"])
        return
    record = verify(out)
    seal = sha256_file(out / "manifest.json")
    lock = (out / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ledger_path = out / "gpu-ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    status_log = DATA / "target/queue/new-operator.log"
    status_log.parent.mkdir(parents=True, exist_ok=True)

    def status(name, outcome):
        with status_log.open("a") as stream:
            stream.write(f"new-operator {name} {outcome} {datetime.now(UTC).isoformat()}\n")

    for stage in record["stages"]:
        tree_commit()
        saved = out / f"{stage['name']}.receipt.json"
        if saved.exists():
            if not reusable(stage, json.loads(saved.read_text()), seal_sha256=seal):
                raise ValueError(f"changed completed stage: {stage['name']}")
            status(stage["name"], "skip-existing exit 0")
            continue
        if any(Path(p).exists() for p in stage["outputs"]):
            raise ValueError(f"incomplete outputs block stage: {stage['name']}")
        env = {k: v for k, v in os.environ.items() if not k.startswith(("XG_", "XGRAPH_"))}
        env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", CMAKE_BUILD_PARALLEL_LEVEL="1")
        env.update(stage["env"])
        if "TMPDIR" in stage["env"]:
            Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        gpu = stage.get("gpu_seconds", 0)
        wall_started = time.monotonic()
        events = Path(stage["gpu_events"]) if stage.get("gpu_events") else None
        offset = 0
        if gpu:
            until = wall_started + stage.get("wall_timeout_seconds", 24 * 3600)
            while True:
                probe = subprocess.run(
                    [str(ROOT / "scripts/gpu-headroom.sh"), str(stage["need_gib"])],
                    cwd=ROOT,
                    env={
                        **env,
                        "XG_AVAILABLE_WAIT": "0",
                        "XG_HEADROOM_WAIT": "0",
                        "XG_GPU_IDLE": "1",
                    },
                    capture_output=True,
                    timeout=max(0.001, until - time.monotonic()),
                )
                if probe.returncode == 0:
                    break
                if time.monotonic() >= until:
                    raise ValueError("headroom/idle wait expired")
                status(stage["name"], "waiting for headroom and idle GPU")
                time.sleep(30)
            reservation = gpu + 30
            if reserved_seconds(ledger) + reservation > record["gpu_budget_seconds"]:
                raise ValueError(
                    "six GPU-hour budget exhausted; prior crashed attempts remain charged"
                )
            ledger.append(
                {"stage": stage["name"], "reserved_seconds": reservation, "status": "running"}
            )
            if events:
                events.parent.mkdir(parents=True, exist_ok=True)
                with events.open("ab") as stream:
                    stream.flush()
                    os.fsync(stream.fileno())
                offset = events.stat().st_size
                ledger[-1].update(gpu_events=str(events), events_offset=offset)
            write_ledger(ledger_path, ledger)
        for p in stage["outputs"]:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
        # Label only containers created by this stage. Timeout cleanup uses this
        # exact label; unrelated owner queues can never match it.
        label = command_hash(stage)
        real_docker = shutil.which("docker")
        env.update(
            XG_OPERATOR_REAL_DOCKER=real_docker or "/usr/bin/docker",
            XG_OPERATOR_STAGE_LABEL=label,
            XG_OPERATOR_STAGE_SECONDS=str(gpu),
            PATH=str(out / "bin") + os.pathsep + os.environ["PATH"],
        )
        status(stage["name"], "start")
        started = time.monotonic()
        code = 1
        cleanup_failed = False
        interrupted = None
        skipped_empty = False
        with (out / f"{stage['name']}.log").open("w") as log:
            if stage.get("requests") and not Path(stage["requests"]).read_text().strip():
                skipped_empty = True
                for p in stage["outputs"]:
                    Path(p).write_text(
                        "" if p == stage["empty_output"] else '{"skipped_empty": true}\n'
                    )
                code = 0
            else:
                process = subprocess.Popen(
                    stage["argv"],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    preexec_fn=cpu_limits if not gpu else None,
                )
                try:
                    code = wait_stage(process, stage, started, wall_started, events, offset)
                    if (
                        code == 0
                        and events
                        and not gpu_occupancy(events, offset, started, require_end=True)[1]
                    ):
                        raise ValueError("stage succeeded without a GPU launch event")
                except BaseException as error:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    if not isinstance(error, subprocess.TimeoutExpired):
                        interrupted = error
                        code = 130 if isinstance(error, (KeyboardInterrupt, SystemExit)) else 1
                    else:
                        code = 124
                finally:
                    if real_docker and gpu:
                        try:
                            found = subprocess.run(
                                [
                                    real_docker,
                                    "ps",
                                    "-aq",
                                    "--filter",
                                    f"label=xgraph.new_operator={label}",
                                ],
                                capture_output=True,
                                text=True,
                                timeout=10,
                                check=True,
                            )
                            ids = found.stdout.split()
                            if ids:
                                subprocess.run(
                                    [real_docker, "rm", "-f", *ids], check=True, timeout=10
                                )
                        except (OSError, subprocess.SubprocessError) as error:
                            log.write(f"stage container cleanup failed: {error}\n")
                            cleanup_failed = True
                            code = code or 1
            if code == 0:
                try:
                    validate_stage_outputs(stage)
                except (ValueError, OSError, KeyError, TypeError) as error:
                    log.write(f"stage output validation failed: {error}\n")
                    code = 1
        if gpu:
            now = time.monotonic()
            ledger[-1].update(
                status="complete" if code == 0 else "failed", wall_seconds=now - started
            )
            try:
                if cleanup_failed:
                    raise ValueError("container cleanup unconfirmed")
                seconds, launched = (
                    gpu_occupancy(events, offset, started) if events else (now - started, True)
                )
                ledger[-1].update(
                    elapsed_seconds=0 if skipped_empty else seconds,
                    accounting="measured",
                    timing_kind="container occupancy"
                    if events
                    else "native GPU stage elapsed time",
                    container_launched=launched if events else None,
                )
            except (ValueError, KeyError, TypeError, OSError):
                ledger[-1].update(parent_elapsed_seconds=now - started, accounting="reserved")
            write_ledger(ledger_path, ledger)
        status(stage["name"], f"exit {code}")
        if interrupted is not None:
            raise interrupted
        if code:
            raise RuntimeError(
                f"stage {stage['name']} failed; outputs retained, no automatic retry"
            )
        write_json(
            saved,
            {
                **receipt(stage),
                "manifest_sha256": seal,
                "elapsed_seconds": time.monotonic() - started,
                "timing_kind": "whole stage wall time including startup and cleanup",
                **(
                    {"gpu_events": {"path": str(events), "sha256": sha256_file(events)}}
                    if events
                    else {}
                ),
            },
        )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["docker-proxy"]:
        docker_proxy(argv[1:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "target/queue-new-operator")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("verify")
    queue = commands.add_parser("queue")
    mode = queue.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    deliver = commands.add_parser("delivery-requests")
    deliver.add_argument("--planner", type=Path, required=True)
    deliver.add_argument("--destination", type=Path, required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--family", choices=FAMILIES, required=True)
    commands.add_parser("score")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if args.command == "prepare":
        prepare(out)
    elif args.command == "verify":
        verify(out)
    elif args.command == "queue":
        import signal

        def terminated(signum, frame):
            raise SystemExit(128 + signum)

        previous = signal.signal(signal.SIGTERM, terminated)
        try:
            run_queue(out, dry_run=not args.run)
        finally:
            signal.signal(signal.SIGTERM, previous)
    elif args.command == "delivery-requests":
        from xgraph.repair import executed_delivery_requests

        requests = indexed(read_jsonl(out / "requests.jsonl"))
        write_jsonl(
            args.destination, executed_delivery_requests(read_jsonl(args.planner), requests)
        )
    elif args.command == "audit":
        audit_family(out, args.family)
    else:
        score(out)


if __name__ == "__main__":
    main()
