"""Owner-only alias queues; confirmation requires an admitted fresh design."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from xgraph.catalog_argv import expand_launcher, flags
from xgraph.index import kg_index_fingerprint
from xgraph.menu_control import check_identities, execute_stage, identity
from xgraph.mintqa_panels import read_jsonl, write_json
from xgraph.mintqa_queue import REPO, _code_digest
from xgraph.provenance import sha256_file

CPU = ["nice", "-n", "19", "taskset", "-c", "1-4", "prlimit", "--as=4000000000"]
LOG = Path("/home/jonas/git/xgraph/target/queue/case-alias.log")
OUTPUT_NAMES = (
    "W.jsonl",
    "W-revision-candidates.jsonl",
    "W-revision-terminal-requests.jsonl",
    "W-revision-terminal-raw.jsonl",
    "W-revision-terminal-support.jsonl",
    "finalist.jsonl",
)


def _git(*args):
    return subprocess.check_output([*CPU, "git", "-C", str(REPO), *args], text=True).strip()


def prepare_queue(registration, seal, out):
    """Rebase only the six sealed WT stages, with arm-symmetric owner controls."""
    source = Path(registration["source"])
    old_repo, old_out = seal["repo"], seal["out"]
    stages = [
        {
            "name": "cuda-agreement",
            "kind": "gpu",
            "forecast_seconds": 300,
            "command": [
                "nice",
                "-n",
                "19",
                "taskset",
                "-c",
                "1-4",
                str(REPO / ".venv/bin/python"),
                "-m",
                "xgraph.case_alias_queue",
                "qualify",
                "--out",
                str(out),
            ],
            "outputs": [str(out / "cuda-agreement.json")],
            "wall_timeout_seconds": 3600,
        }
    ]
    runtime = [identity(out / "requests.jsonl"), identity(out / "members.json")]
    index_fingerprints = {}
    for family in ("qwen38", "gemma31"):
        base_index = source / "target/mintqa-rematch/indexes" / family
        alias_index = REPO / "target/case-alias/indexes" / family
        baseline = json.loads((base_index / "kg-index.json").read_text())
        variant = json.loads((alias_index / "kg-index.json").read_text())
        for key in ("tokenizer_fingerprint", "span_syntax", "relations", "relation_primary_rows"):
            if baseline.get(key) != variant.get(key):
                raise ValueError(f"sealed and alias index differ in {key}; align the variant first")
        if baseline.get("labels", {}).get("owner_ids") != variant.get("labels", {}).get(
            "owner_ids"
        ):
            raise ValueError("sealed and alias entity owner positions differ")
        for name in ("tokens", "offsets", "owners"):
            filename = f"relations.{name}.npy"
            if sha256_file(base_index / filename) != sha256_file(alias_index / filename):
                raise ValueError("sealed and alias relation arrays differ; align the variant first")
        index_fingerprints[family] = {
            "WT": kg_index_fingerprint(base_index),
            "WT+alias": kg_index_fingerprint(alias_index),
            "aliases": json.loads((alias_index / "kg-index.json").read_text())[
                "initial_case_aliases"
            ],
        }
        for index in (base_index, alias_index):
            runtime.extend(
                identity(p)
                for p in sorted(index.iterdir())
                if p.name == "kg-index.json" or p.suffix == ".npy"
            )
        for arm in ("WT", "WT+alias"):
            destination = out / family / arm
            destination.mkdir(parents=True, exist_ok=True)
            index = base_index if arm == "WT" else alias_index
            for original in seal["families"][family]["steps"]:
                name = Path(original["output"]).name
                if name not in OUTPUT_NAMES:
                    continue
                output = destination / name
                command = []
                for value in original["command"]:
                    value = value.replace(old_out + f"/{family}/validation", str(destination))
                    value = value.replace(old_repo, str(REPO))
                    value = value.replace(str(base_index), str(index))
                    first = seal["families"][family]["steps"][0]
                    value = value.replace(first["requests"], str(out / "requests.jsonl"))
                    if value.startswith("PATH="):
                        value = (
                            f"PATH={REPO / '.venv/bin'}:/home/jonas/.local/bin:"
                            "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
                        )
                    if value.startswith("UV_CACHE_DIR="):
                        value = f"UV_CACHE_DIR={REPO / 'target/uv-cache'}"
                    if value.startswith("--as="):
                        value = "--as=4000000000"
                    command.append(value)
                pos = command.index("nice")
                command[pos : pos + 3] = ["nice", "-n", "19", "taskset", "-c", "1-4"]
                if "scripts/generate.sh" not in " ".join(command) and "prlimit" not in command:
                    pos = command.index("taskset") + 3
                    command[pos:pos] = ["prlimit", "--as=4000000000"]
                stage = {
                    "name": f"{family}/{arm}/{name}",
                    "command": command,
                    "kind": "cpu",
                    "outputs": [str(output)],
                }
                if original.get("requests") is not None:
                    command[command.index("--seed") + 1] = "17"
                    timing = next(
                        t
                        for t in registration["timing_sources"][family]
                        if t["generation"] == output.stem
                    )
                    forecast = (
                        timing["seconds"]
                        / timing["requests"]
                        * registration["possible_design"]["unique_questions"]
                    )
                    container = (
                        f"xgraph-case-alias-{family}-"
                        f"{arm.lower().replace('+', '-')}-{output.stem.lower()}"
                    )
                    events = str(output) + ".gpu-events.jsonl"
                    owner_env = {
                        "XG_START_ATTEMPTS": "1",
                        "XG_RUN_SECONDS": str(math.ceil(1.5 * forecast + 300)),
                        "XG_GPU_EVENTS": events,
                        "XG_CONTAINER_NAME": container,
                        "XG_GPU_IDLE": "1",
                        "XG_GPU_IDLE_WAIT": "43200",
                        "XG_AVAILABLE_WAIT": "43200",
                    }
                    for key, value in owner_env.items():
                        command[:] = [v for v in command if not v.startswith(key + "=")]
                        command.insert(2, f"{key}={value}")
                    stage.update(
                        kind="gpu",
                        forecast_seconds=forecast,
                        gpu_events=events,
                        wall_timeout_seconds=86400,
                        outputs=[
                            str(output),
                            str(output) + ".summary.json",
                            str(output) + ".provenance.json",
                        ],
                        expected_flags=flags(expand_launcher(command)),
                        inspect_command=["docker", "container", "inspect", container],
                        cleanup_command=["docker", "rm", "--force", container],
                        headroom_command=[
                            "nice",
                            "-n",
                            "19",
                            "taskset",
                            "-c",
                            "1-4",
                            "bash",
                            str(out / "headroom.sh"),
                            "60" if family == "qwen38" else "75",
                        ],
                    )
                stages.append(stage)
    runtime.extend(
        identity(source / p)
        for p in (
            "data/external/mintqa-full/kg.sqlite",
            "data/external/mintqa-full/kg/relation_labels.jsonl",
        )
    )
    registration.update(
        stages=stages,
        confirmatory_stages=[],
        runtime_inputs=runtime,
        indexes=index_fingerprints,
        code_commit=_git("rev-parse", "HEAD"),
        qualification_argv=qualification_commands(),
    )
    registration["models"] = {
        f: seal["families"][f].get("model", {}) for f in ("qwen38", "gemma31")
    }
    for model in registration["models"].values():
        runtime.extend(
            {key: row[key] for key in ("path", "bytes", "sha256")}
            for row in model.get("files", [])
            if "sha256" in row
        )
    headroom = """#!/usr/bin/env bash
set -euo pipefail
deadline=$((SECONDS + 43200))
while (( SECONDS < deadline )); do
    active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) || exit 1
    if [[ -z "$active" ]]; then
        if env XG_GPU_IDLE=1 XG_AVAILABLE_WAIT=0 XG_GPU_IDLE_WAIT=0 XG_HEADROOM_WAIT=0 \\
            nice -n 19 taskset -c 1-4 ./scripts/gpu-headroom.sh "$1"; then exit 0; fi
    fi
    sleep 60
done
exit 1
"""
    (out / "headroom.sh").write_text(headroom)
    registration["runtime_inputs"].append(identity(out / "headroom.sh"))
    stages[0]["headroom_command"] = [
        "nice",
        "-n",
        "19",
        "taskset",
        "-c",
        "1-4",
        "bash",
        str(out / "headroom.sh"),
        "60",
    ]
    python = shlex.quote(str(REPO / ".venv/bin/python"))
    queue_args = f"-m xgraph.case_alias queue --out {shlex.quote(str(out))}"
    launcher = f"""#!/usr/bin/env bash
set -euo pipefail
cd {shlex.quote(str(REPO))}
mode=${{1:---dry-run}}
if (( $# > 1 )); then exit 2; fi
case "$mode" in
  --dry-run) exec nice -n 19 taskset -c 1-4 prlimit --as=4000000000 \\
    {python} {queue_args} ;;
  --run) exec nice -n 19 taskset -c 1-4 \\
    {python} {queue_args} --run ;;
  --run-exploratory) exec nice -n 19 taskset -c 1-4 \\
    {python} {queue_args} --run-exploratory ;;
  *) exit 2 ;;
esac
"""
    (out / "case-alias.sh").write_text(launcher)
    (out / "case-alias.sh").chmod(0o755)
    registration["runtime_inputs"].append(identity(out / "case-alias.sh"))


def queue(out, *, run=False, exploratory=False):
    out = Path(out).resolve()
    path = out / "registration.json"
    seal = sha256_file(path)
    if seal != (out / "registration.sha256").read_text().strip():
        raise ValueError("registration hash changed")
    registration = json.loads(path.read_text())
    if _code_digest() != registration["code_digest"]:
        raise ValueError("registered code changed")
    if not run:
        admitted = registration.get("prospective_design", {}).get("admitted", False)
        status = "ADMITTED" if admitted else "BLOCKED: current-model exposure; n=0"
        print(f"# Fresh confirmation: {status}")
        preview_stages = registration["stages"]
        if admitted and not exploratory:
            preview_stages = registration["confirmatory_stages"]
            print("# Confirmatory argv follow; --run executes this admitted design.")
        else:
            print("# Retained exploratory only argv follow; --run-exploratory is explicit.")
        for command in registration["qualification_argv"]:
            print(f"# CUDA qualification inner argv\n{shlex.join(command)}")
        for stage in preview_stages:
            for key in ("headroom_command", "inspect_command", "command", "cleanup_command"):
                if key in stage:
                    print(f"# {stage['name']} {key}\n{shlex.join(stage[key])}")
        return
    if registration.get("draft"):
        raise ValueError("CPU draft cannot execute; register a clean commit")
    if _git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("refusing dirty tree")
    if not exploratory and not registration.get("prospective_design", {}).get("admitted"):
        raise ValueError(
            "no fresh confirmatory design: current-model outcomes cover every population ID"
        )
    stages = registration["stages"] if exploratory else registration["confirmatory_stages"]
    forecast = registration["forecast"] if exploratory else registration["confirmatory_forecast"]
    cap = min(4 if exploratory else 10, forecast["hard_cap_hours"]) * 3600
    if not stages:
        raise ValueError("empty admitted queue")
    if _git("rev-parse", "HEAD") != registration["code_commit"]:
        raise ValueError("registration commit changed; prepare again before outcomes")
    check_identities(registration["runtime_inputs"])
    for model in registration["models"].values():
        for row in model.get("files", []):
            if "sha256" not in row:
                stat = Path(row["path"]).stat()
                if stat.st_size != row["bytes"] or stat.st_mtime_ns != row["mtime_ns"]:
                    raise ValueError("sealed model weight metadata changed")
    with (out / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for stage in stages:
            with LOG.open("a") as log:
                log.write(f"case-alias {stage['name']} start {datetime.now(UTC).isoformat()}\n")
            try:
                if (
                    stage["name"] == "cuda-agreement"
                    and not Path(stage["outputs"][0] + ".receipt.json").exists()
                ):
                    subprocess.run(
                        stage["headroom_command"],
                        check=True,
                        cwd=REPO,
                    )
                execute_stage(stage, seal, LOG, out / "gpu-ledger.jsonl", cap)
            except BaseException:
                with LOG.open("a") as log:
                    log.write(f"case-alias {stage['name']} exit 1\n")
                raise
            with LOG.open("a") as log:
                log.write(f"case-alias {stage['name']} exit 0\n")


def qualification_commands():
    return [
        [
            *CPU,
            "env",
            "CMAKE_BUILD_PARALLEL_LEVEL=1",
            f"UV_CACHE_DIR={REPO / 'target/uv-cache'}",
            "CMAKE_ARGS=-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
            "uv",
            "sync",
            "--locked",
            "--reinstall-package",
            "xgraph",
        ],
        [
            "env",
            "PYTEST_ADDOPTS=",
            "CUDA_VISIBLE_DEVICES=0",
            "nice",
            "-n",
            "19",
            "taskset",
            "-c",
            "1-4",
            str(REPO / ".venv/bin/python"),
            "-m",
            "pytest",
            "-q",
            "tests/test_engine.py",
            "tests/test_vllm_adapter.py",
        ],
    ]


def qualify(out):
    """Executed only by the GPU owner, after the rest of the queue is idle."""
    commands = qualification_commands()
    for command in commands:
        subprocess.run(command, cwd=REPO, check=True)
    write_json(
        Path(out) / "cuda-agreement.json", {"code_digest": _code_digest(), "commands": commands}
    )


def score(out):
    """Separate CPU scoring; the owner inference queue never dispatches this action."""
    from xgraph.menu_control import _check_receipt
    from xgraph.mintqa_bench import answer_contains, dataset_rows, relation_clusters
    from xgraph.mintqa_rematch import clustered_difference
    from xgraph.prompts import final_answer
    from xgraph.score import record_failure

    out = Path(out)
    registration = json.loads((out / "registration.json").read_text())
    source = Path(registration["source"]) / "data/external/mintqa-full/questions"
    seal = sha256_file(out / "registration.json")
    if seal != (out / "registration.sha256").read_text().strip():
        raise ValueError("registration hash changed")
    for stage in registration["stages"]:
        _check_receipt(stage, seal)
    gold = dataset_rows([f"ti={source / 'ti.jsonl'}", f"pop={source / 'pop.jsonl'}"])
    clusters = relation_clusters(gold)
    members = json.loads((out / "members.json").read_text())
    result = {"status": "exploratory exposed-data result; cannot confirm", "families": {}}
    for family in ("qwen38", "gemma31"):
        outcomes = {}
        for arm in ("WT", "WT+alias"):
            rows = read_jsonl(out / family / arm / "finalist.jsonl")
            expected = {m["id"] for panel in members.values() for m in panel}
            if len(rows) != len(expected) or {r["id"] for r in rows} != expected:
                raise ValueError("incomplete or duplicate exploratory outputs")
            outcomes[arm] = {
                r["id"]: not record_failure(r)
                and answer_contains(final_answer(r["text"]), gold[r["id"]]["answer"])
                for r in rows
            }
        result["families"][family] = {}
        for cohort, panel in members.items():
            if any(r["cluster"] != clusters[r["id"]] for r in panel):
                raise ValueError("cluster metadata changed")
            a = [float(outcomes["WT+alias"][r["id"]]) for r in panel]
            b = [float(outcomes["WT"][r["id"]]) for r in panel]
            result["families"][family][cohort] = clustered_difference(a, b, panel, seed=17)
    write_json(out / "exploratory-report.json", result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("qualify",))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    qualify(args.out)


if __name__ == "__main__":
    main()
