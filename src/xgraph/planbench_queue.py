"""Serial, reservation-guarded GPU pilot queue. Dry run is the default.

The queue records rate assumptions rather than presenting an unmeasured runtime
as evidence. Execution holds the shared launch lock, requires an idle GPU, uses
an offline pinned image and enforces a wall-time reservation cap for both jobs.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import time
from collections import Counter
from importlib import metadata
from pathlib import Path

from xgraph.planbench_backend import model_metadata_paths
from xgraph.planbench_data import DOMAINS
from xgraph.provenance import provenance, sha256_file

IMAGE = (
    "vllm/vllm-openai:v0.31.0@sha256:"
    "3f7dd5b777d34d1724456ce71f87385dca288c3bb23029ab27dee358f5d2b971"
)
ARMS = ["free", "resume", "feedback", "menu", "mask", "llm_planner"]
PLANNER = "target/planbench/tools/downward-1eef26b2cbf599a1894606aa898d9d49e1034cb9"
MODELS = [
    ("qwen25", "Qwen2.5-7B-Instruct", 8, 40, 0.25, 40, 2000),
    ("gemma31", "gemma-4-31B-it-FP8-dynamic", 16, 70, 0.45, 20, 1000),
]


def plan_queue(repo, panel_path, reference_path, out, *, count=100, seed=16, models_root=None):
    repo, out = Path(repo).resolve(), Path(out).resolve()
    panel_path, reference_path = Path(panel_path).resolve(), Path(reference_path).resolve()
    panel = json.loads(panel_path.read_text())
    reference = json.loads(reference_path.read_text())
    if Counter(r["domain"] for r in panel["cases"]) != {d: count for d in DOMAINS}:
        raise ValueError("queue requires the declared number of instances in each domain")
    if panel["seed"] != seed or reference["panel_sha256"] != sha256_file(panel_path):
        raise ValueError("queue seed or reference panel hash mismatch")
    if {r["id"] for r in reference["references"]} != {r["id"] for r in panel["cases"]}:
        raise ValueError("queue references do not cover the complete frozen panel")
    blocks = {r["cluster"] for r in panel["cases"] if r["domain"] == "blocksworld"}
    mystery = {r["cluster"] for r in panel["cases"] if r["domain"] == "mystery_blocksworld"}
    if blocks != mystery:
        raise ValueError("queue must keep Mystery renamings paired")
    models_root = Path(models_root or "/home/jonas/models").resolve()
    instances = len(panel["cases"])
    jobs, nominal, worst = [], 0.0, 0.0
    for family, name, kv, need, share, decode_rate, prefill_rate in MODELS:
        model, output = models_root / name, out / family
        command = [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--gpus",
            "all",
            "--ipc",
            "host",
            "--network",
            "none",
            "--name",
            f"xgraph-planbench-{family}-{os.getpid()}",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--volume",
            f"{out / 'site'}:/site:ro",
            "--volume",
            f"{repo / 'data/external/planbench'}:/data/planbench:ro",
            "--volume",
            f"{model}:/model:ro",
            "--volume",
            f"{repo / PLANNER}:/planner:ro",
            "--volume",
            f"{out / 'inputs'}:/inputs:ro",
            "--volume",
            f"{output}:/out",
            "--volume",
            f"{out / 'cache'}:/cache",
            "--env",
            "PYTHONPATH=/site",
            "--env",
            "VLLM_USE_V2_MODEL_RUNNER=1",
            "--env",
            "HF_HUB_OFFLINE=1",
            "--env",
            "TRANSFORMERS_OFFLINE=1",
            "--env",
            "HF_HOME=/cache/hf",
            "--env",
            "VLLM_CACHE_ROOT=/cache/vllm",
            "--env",
            "TRITON_CACHE_DIR=/cache/triton",
            "--env",
            "TORCHINDUCTOR_CACHE_DIR=/cache/torch",
            "--env",
            "OMP_NUM_THREADS=1",
            "--env",
            "OPENBLAS_NUM_THREADS=1",
            "--entrypoint",
            "nice",
            IMAGE,
            "-n",
            "10",
            "python3",
            "-m",
            "xgraph.planbench",
            "run",
            "--model",
            "/model",
            "--data",
            "/data/planbench",
            "--panel",
            "/inputs/panel.json",
            "--references",
            "/inputs/references.json",
            "--fast-downward",
            "/planner/fast-downward.py",
            "--run-manifest",
            f"/inputs/{family}.json",
            "--out",
            "/out/records.jsonl",
            "--kv-cache-gib",
            str(kv),
            "--memory-share",
            str(share),
            "--max-actions",
            "64",
            "--max-calls",
            "65",
            "--generated-tokens",
            "2048",
            "--prompt-tokens",
            "262144",
            "--context-tokens",
            "16384",
            "--seed",
            str(seed),
            "--arms",
            *ARMS,
        ]
        episodes = instances * len(ARMS)
        # Nominal scenario: average six calls and 256 output tokens per episode,
        # 2048 prompt tokens per call, plus half an hour per family for startup.
        nominal_job = (
            episodes * 256 / decode_rate + episodes * 6 * 2048 / prefill_rate
        ) / 3600 + 0.5
        # Algebraic token-budget ceiling under the SAME assumed rates. The three
        # one-call interfaces cannot consume the aggregate interactive prompt cap.
        max_prompt = instances * (3 * 16384 + 3 * 262144)
        worst_job = (episodes * 2048 / decode_rate + max_prompt / prefill_rate) / 3600 + 0.5
        nominal += nominal_job
        worst += worst_job
        jobs.append(
            {
                "family": family,
                "model": str(model),
                "kv_cache_gib": kv,
                "need_gib": need,
                "command": command,
                "output": str(output / "records.jsonl"),
                "episodes": episodes,
                "assumed_decode_tokens_s": decode_rate,
                "assumed_prefill_tokens_s": prefill_rate,
                "nominal_gpu_hours": nominal_job,
            }
        )
    return {
        "dry_run": True,
        "image": IMAGE,
        "instances_per_model": instances,
        "count_per_domain": count,
        "panel_sha256": sha256_file(panel_path),
        "references_sha256": sha256_file(reference_path),
        "jobs": jobs,
        "estimate": {
            "measured": False,
            "nominal_gpu_hours": nominal,
            "token_budget_scenario_gpu_hours": worst,
            "assumptions": "six calls/episode, 2048 prompt tokens/call, 256 output tokens/episode; "
            "half-hour startup per family; rates recorded per job",
            "limitations": "host mask construction, synchronization and CPU planner pauses "
            "are unmeasured; the hard reservation cap can leave incomplete jobs",
        },
    }


def _snapshot(repo, site):
    site.mkdir(parents=True)
    shutil.copytree(
        repo / "src/xgraph", site / "xgraph", ignore=shutil.ignore_patterns("__pycache__")
    )
    for package in ("xgraph", "pyperplan"):
        dist = metadata.distribution(package)
        for entry in dist.files or ():
            if ".dist-info" in str(entry).split("/")[0]:
                source, destination = Path(dist.locate_file(entry)), site / str(entry)
                if source.is_file():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
        if package == "pyperplan":
            import pyperplan

            shutil.copytree(
                Path(pyperplan.__file__).parent,
                site / "pyperplan",
                ignore=shutil.ignore_patterns("__pycache__"),
            )


def _execute(queue, args, repo):
    out = args.out.resolve()
    if out.exists():
        raise ValueError("GPU queue needs a fresh output directory")
    common = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=repo, text=True
        ).strip()
    )
    lock = common.parent / "target/generate/launch.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        subprocess.run(["docker", "image", "inspect", IMAGE], check=True, stdout=subprocess.DEVNULL)
        _snapshot(repo, out / "site")
        (out / "inputs").mkdir()
        (out / "cache").mkdir()
        shutil.copy2(args.panel, out / "inputs/panel.json")
        shutil.copy2(args.references, out / "inputs/references.json")
        started = time.monotonic()
        queue["dry_run"] = False
        queue["gpu_lock"] = str(lock)
        queue["execution"] = []
        for job in queue["jobs"]:
            remaining = args.max_gpu_hours * 3600 - (time.monotonic() - started)
            if remaining <= 0:
                queue["status"] = "reservation_exhausted"
                break
            model = Path(job["model"])
            env = dict(
                os.environ, XG_GPU_IDLE="1", XG_AVAILABLE_WAIT="0", XG_PAGECACHE_ROOTS=str(model)
            )
            subprocess.run(
                [str(repo / "scripts/gpu-headroom.sh"), str(job["need_gib"])], check=True, env=env
            )
            # Hash weight shards only when execution is explicitly requested and
            # the GPU has been reserved. No model or weight hash pass runs in a dry run.
            weights = sorted([*model.glob("*.safetensors"), *model.glob("pytorch_model*.bin")])
            if not weights:
                raise ValueError(f"no frozen model weights found under {model}")
            inputs = [
                args.panel,
                args.references,
                repo / "data/external/planbench/manifest.json",
                *sorted((out / "site/xgraph").glob("planbench*.py")),
                *model_metadata_paths(model),
                *weights,
                repo / PLANNER / "install.json",
                repo / PLANNER / "builds/release/bin/downward",
            ]
            prov = provenance(inputs)
            run_manifest = {
                "job": job,
                "provenance": prov,
                "model_weight_hashes": {
                    Path(p["path"]).name: p["sha256"]
                    for p in prov["inputs"]
                    if Path(p["path"]) in weights
                },
            }
            (out / "inputs" / f"{job['family']}.json").write_text(
                json.dumps(run_manifest, indent=2) + "\n"
            )
            subprocess.run(
                [str(repo / "scripts/gpu-headroom.sh"), str(job["need_gib"])], check=True, env=env
            )
            output = Path(job["output"])
            output.parent.mkdir()
            before = time.monotonic()
            name = job["command"][job["command"].index("--name") + 1]
            try:
                with (output.parent / "container.log").open("w") as log:
                    result = subprocess.run(
                        job["command"],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=max(1, args.max_gpu_hours * 3600 - (time.monotonic() - started)),
                    )
                    status = f"exit_{result.returncode}"
            except subprocess.TimeoutExpired:
                status = "reservation_exhausted"
            finally:
                # Only this queue's named container; never a shared server.
                subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            queue["execution"].append(
                {
                    "family": job["family"],
                    "status": status,
                    "reserved_gpu_seconds": time.monotonic() - before,
                }
            )
            (out / "execution.json").write_text(json.dumps(queue, indent=2) + "\n")
            if status != "exit_0":
                queue["status"] = status
                break
        else:
            queue["status"] = "completed"
        queue["reservation_seconds"] = time.monotonic() - started
        (out / "execution.json").write_text(json.dumps(queue, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--panel", type=Path, default=Path("target/planbench/panel.json"))
    parser.add_argument("--references", type=Path, default=Path("target/planbench/references.json"))
    parser.add_argument("--models-root", type=Path, default=Path("/home/jonas/models"))
    parser.add_argument("--out", type=Path, default=Path("target/planbench/gpu-pilot"))
    parser.add_argument("--queue-json", type=Path, default=Path("target/planbench/queue.json"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--max-gpu-hours", type=float, default=64)
    args = parser.parse_args()
    if not 0 < args.max_gpu_hours <= 64:
        raise ValueError("reservation cap must be positive and at most 64 GPU-hours")
    repo = Path(__file__).resolve().parents[2]
    queue = plan_queue(
        repo,
        args.panel,
        args.references,
        args.out,
        count=args.count,
        seed=args.seed,
        models_root=args.models_root,
    )
    inputs = [args.panel, args.references, repo / "data/external/planbench/manifest.json"]
    inputs.append(repo / PLANNER / "install.json")
    for job in queue["jobs"]:
        inputs.extend(model_metadata_paths(Path(job["model"])))
    queue["provenance"] = provenance(inputs)
    queue["reservation_cap_gpu_hours"] = args.max_gpu_hours
    # Commands are always reviewable before execution; no Docker or GPU probe in dry mode.
    for job in queue["jobs"]:
        print(shlex.join(job["command"]))
    print(json.dumps(queue["estimate"], indent=2))
    if args.execute:
        _execute(queue, args, repo)
    args.queue_json.parent.mkdir(parents=True, exist_ok=True)
    args.queue_json.write_text(json.dumps(queue, indent=2) + "\n")


if __name__ == "__main__":
    main()
