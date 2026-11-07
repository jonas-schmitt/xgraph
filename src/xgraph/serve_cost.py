"""Controlled serving study: frozen panels, serial server cells and trace replay.

The CLI is scripts/serve_cost_trials.py (or scripts/serve-cost.sh for execution).
Importing this module and planning a study use neither Docker nor a GPU.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
import re
import resource
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

from xgraph.cost import (
    dedicated_gpu_seconds,
    dedicated_ratio,
    distributions,
    model_calls,
    trial_interval,
    weighted_dedicated_ratio,
)
from xgraph.frontier import read_records
from xgraph.provenance import provenance, sha256_file
from xgraph.serve_arms import freeze_arms, replay_arm
from xgraph.serving import SAMPLING_FIELDS, _one, measure_workload
from xgraph.syntax import SpanSyntax

IMAGE = (
    "vllm/vllm-openai:v0.31.0@sha256:"
    "3f7dd5b777d34d1724456ce71f87385dca288c3bb23029ab27dee358f5d2b971"
)


def calibration_settings(profile: str) -> dict:
    """Explicit support contracts, checked against calibration's dry commands.

    A server is owned per arm/cell; these settings never read ambient XG_* values.
    P's delivery uses xgraph_stop=0 in its STOP-capable server, whose begin clears
    the preceding request's STOP state. Other delivery servers have STOP off.
    """
    if profile not in (
        "paths",
        "paths-no-stop",
        "delivery",
        "masked-agent",
        "vocabulary",
        "masked-agent-vocabulary",
        "masked-agent-successor",
    ):
        raise ValueError(f"unknown calibration support profile {profile!r}")
    return {
        "opened": profile in ("paths", "paths-no-stop", "vocabulary"),
        "engine_settings": {
            "XGRAPH_PATHS": "1",
            "XGRAPH_COMMITTED_PATHS": "1",
            "XGRAPH_ESCAPES": "0",
            "XGRAPH_UNQUOTED_STARTS": "1",
            "XGRAPH_START_RANK": "sitelinks",
            "XGRAPH_RELATIONS": (
                profile.removeprefix("masked-agent-")
                if profile.startswith("masked-agent-")
                else "vocabulary"
                if profile == "vocabulary"
                else "frontier"
            ),
            "XGRAPH_STOP": "residual-goal" if profile == "paths" else "",
            "XGRAPH_END_AFTER_ANSWER": "0" if profile.startswith("masked-agent") else "1",
            "XGRAPH_EXPRESSIONS": "0",
            "XGRAPH_DERIVATION": "0",
        },
    }


def write_json(path: Path, value: dict) -> None:
    """Replace a report atomically, including when a trial is interrupted."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_runtime(study: dict, out: Path, wheel: Path | None) -> dict:
    """Archive dirty sources and freeze one GPU-built wheel for all server cells."""
    started = time.perf_counter()
    runtime = out.resolve() / "runtime"
    runtime.mkdir()
    repo = Path(__file__).resolve().parents[2]
    sources = sorted(
        {
            p
            for folder in ("src", "csrc", "scripts")
            for p in (repo / folder).rglob("*")
            if p.is_file() and p.suffix in (".py", ".cu", ".h", ".sh")
        }
        | {repo / p for p in ("pyproject.toml", "uv.lock", "CMakeLists.txt")}
    )
    snapshot = runtime / "source.zip"
    with zipfile.ZipFile(snapshot, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            archive.write(source, source.relative_to(repo))
    if wheel:
        built = runtime / wheel.name
        shutil.copyfile(wheel, built)
    else:
        subprocess.run(
            ["uv", "build", "--offline", "--wheel", "--out-dir", str(runtime)], cwd=repo, check=True
        )
        wheels = list(runtime.glob("xgraph-*.whl"))
        if len(wheels) != 1:
            raise ValueError(f"expected exactly one xgraph wheel in {runtime}")
        built = wheels[0]
    site = runtime / "site"
    with zipfile.ZipFile(built) as archive:
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in archive.namelist()):
            raise ValueError("wheel contains a path outside its installation directory")
        if not any(
            Path(name).name.startswith("_engine.") and name.endswith(".so")
            for name in archive.namelist()
        ):
            raise ValueError(
                "the serving wheel contains no native engine; build it on the GPU host"
            )
        for source in sources:
            if source.suffix == ".py" and source.is_relative_to(repo / "src/xgraph"):
                name = str(source.relative_to(repo / "src"))
                if name not in archive.namelist() or archive.read(name) != source.read_bytes():
                    raise ValueError(
                        f"serving wheel Python sources differ from this checkout: {name}"
                    )
        archive.extractall(site)
    return {
        "site": str(site),
        "wheel": str(built),
        "wheel_sha256": sha256_file(built),
        "source_snapshot": str(snapshot),
        "source_sha256": sha256_file(snapshot),
        "source_files": {str(p): sha256_file(p) for p in sources},
        "preparation_seconds": time.perf_counter() - started,
    }


def check_sources(study: dict) -> None:
    """Refuse a mixed-source study if another session edits the host runner mid-run."""
    for filename, expected in study.get("runtime", {}).get("source_files", {}).items():
        if sha256_file(Path(filename)) != expected:
            raise ValueError(f"source changed during the study: {filename}")
    for filename, expected in study.get("input_identity", {}).items():
        stat = Path(filename).stat()
        actual = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        if actual != expected:
            raise ValueError(f"frozen input changed during the study: {filename}")


def _capture(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=30).stdout


def _get(url: str, *, timeout: float = 10) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def trial_wall_bound(study: dict, trial: dict) -> int:
    """Four times the registered high forecast, with startup and a wall floor."""
    from xgraph.serve_mintqa import reservation_estimate

    if study.get("timing_panel") or study["arms"][trial["arm"]].get("policy"):
        forecast = reservation_estimate(study, [trial])["double_service_gpu_hours"] * 3600
    else:
        arm = study["arms"][trial["arm"]]
        forecast = study["request_timeout"] * len(arm["requests"])
    return math.ceil(max(1800, study["startup_timeout"] + 120, 4 * forecast))


def supervise_trial(command, trial, *, wall_seconds, grace_seconds=10, server=None, env=None):
    """Enforce the wall bound in a different process from all native client calls.

    The child owns a process group so a forked monitor cannot retain log pipes or
    outlive a failed trial. Only the preassigned owned server is cleaned up.
    """
    out = Path(trial["output"])
    started = time.perf_counter()
    failure = None
    with out.with_suffix(".log").open("a", buffering=1) as log:
        log.write(f"trial start wall_bound_seconds={wall_seconds} command={command!r}\n")
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        try:
            code = process.wait(timeout=wall_seconds)
            if code:
                raise RuntimeError(f"trial worker exit {code}; see {out.with_suffix('.log')}")
        except BaseException as error:
            failure = error
            timed_out = isinstance(error, subprocess.TimeoutExpired)
            log.write(f"trial failed: {'wall bound exceeded' if timed_out else str(error)}\n")
            log.flush()
            if process.poll() is None:
                if server is not None:
                    with suppress(ProcessLookupError):
                        process.send_signal(signal.SIGUSR1)
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=grace_seconds)
            # Also reap forked descendants after their direct parent has exited.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            teardown_error = None
            if server is not None:
                server.launched = True  # also handles an interrupted docker run
                try:
                    server.stop()
                except Exception as cleanup_error:
                    teardown_error = str(cleanup_error)
            retained = json.loads(out.read_text()) if out.is_file() else {}
            retained.update(
                artifact_kind="xgraph.controlled_serving",
                version=1,
                status="failed",
                trial=trial.copy(),
                error="trial wall bound exceeded" if timed_out else str(error),
                driver_seconds=time.perf_counter() - started,
                watchdog={
                    "timed_out": timed_out,
                    "wall_bound_seconds": wall_seconds,
                    "worker_pid": process.pid,
                    "worker_returncode": process.returncode,
                    "owned_server": server.name if server else None,
                    "teardown_error": teardown_error,
                },
            )
            write_json(out, retained)
            if timed_out:
                raise TimeoutError(f"trial wall bound exceeded ({wall_seconds}s)") from error
            raise
        finally:
            if failure is None:
                log.write(f"trial worker exit 0 seconds={time.perf_counter() - started:.3f}\n")


def supervise_cell(study: dict, trial: dict, kv_cache_bytes: int, site: Path) -> dict:
    """Exec a fresh interpreter per cell; never fork a live reader/monitor pool."""
    server = VLLMServer(study, trial, kv_cache_bytes, site)
    trial["server_name"] = server.name
    specification = Path(trial["output"]).with_suffix(".worker.json")
    write_json(
        specification,
        {
            "study": str(Path(trial["output"]).parent / "study.json"),
            "runtime": str(Path(trial["output"]).parent / "runtime.json"),
            "trial": trial,
            "kv_cache_bytes": kv_cache_bytes,
            "site": str(site),
            "actual_kv_capacity_tokens": study.get("actual_kv_capacity_tokens"),
            "traceback_seconds": study.get("traceback_seconds", 300),
        },
    )
    from xgraph.serve_mintqa import ServingBudget

    budget = (
        ServingBudget(Path(os.environ["XG_SERVING_BUDGET"]), os.environ["XG_SERVING_SEAL_SHA256"])
        if os.environ.get("XG_SERVING_BUDGET")
        else None
    )
    attempt = budget.begin(trial["output"], trial["wall_bound_seconds"]) if budget else None
    failed = True
    try:
        supervise_trial(
            [
                sys.executable,
                "-I",
                "-u",
                str(Path(__file__).resolve().parents[2] / "scripts/serve_cost_trials.py"),
                "--cell-spec",
                str(specification),
            ],
            trial,
            wall_seconds=attempt["wall_seconds"] if attempt else trial["wall_bound_seconds"],
            server=server,
        )
        failed = False
    finally:
        if budget:
            report = (
                json.loads(Path(trial["output"]).read_text())
                if Path(trial["output"]).is_file()
                else {}
            )
            confirmed = ("teardown_seconds" in report and not report.get("teardown_error")) or (
                "watchdog" in report and report["watchdog"].get("teardown_error") is None
            )
            budget.finish(
                attempt,
                cleanup_confirmed=confirmed,
                failed=failed or report.get("status") != "completed",
            )
    return json.loads(Path(trial["output"]).read_text())


def score_tokens(url: str, qid: str, prompt: list[int], params, timeout: float) -> dict:
    """Pinned vLLM raw one-token scoring, including the entire requested support.

    Scoring has no visible-content TTFT. The request's complete latency includes
    prefill and decode; raw responses are retained independently of frozen scores.
    v0.31.0's completion protocol requires logprobs even with explicit token ids.
    """
    started = time.perf_counter()
    tokens = getattr(params, "logprob_token_ids", None)
    body = {
        "model": "/model",
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "ignore_eos": True,
        "add_special_tokens": False,
        "logprobs": getattr(params, "logprobs", 0),
        "return_tokens_as_token_ids": True,
        **({"logprob_token_ids": tokens} if tokens is not None else {}),
    }
    http = urllib.request.Request(
        url.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(http, timeout=timeout) as response:
        raw = json.loads(response.read())
    usage = raw.get("usage", {})
    if usage.get("prompt_tokens") != len(prompt) or usage.get("completion_tokens") != 1:
        raise ValueError(f"gcr {qid}: raw scoring token usage differs or is unknown")
    choices = raw.get("choices", [])
    if len(choices) != 1 or choices[0].get("finish_reason") == "abort":
        raise ValueError(f"gcr {qid}: raw scoring aborted")
    rows = (choices[0].get("logprobs") or {}).get("top_logprobs")
    if not rows or (tokens is not None and any(f"token_id:{t}" not in rows[0] for t in tokens)):
        raise ValueError(f"gcr {qid}: raw scoring omits requested token logprobs")
    return {
        "id": qid,
        "latency": time.perf_counter() - started,
        "ttft": None,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": 1,
        "finish_reason": choices[0].get("finish_reason"),
        "kind": "beam_step",
        "request": body,
        "response": raw,
    }


def _configuration(value) -> dict:
    """Flatten the JSON server_info representation, retaining effective cache settings."""
    found = {}
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, dict):
                found.update(_configuration(child))
            else:
                found[key] = child
    return found


class VLLMServer:
    """One fresh, named container per trial; never reuse or stop another server."""

    def __init__(self, study: dict, trial: dict, kv_cache_bytes: int, site: Path):
        self.study, self.trial, self.kv_cache_bytes, self.site = study, trial, kv_cache_bytes, site
        self.name = trial.get("server_name") or "xgraph-cost-" + uuid.uuid4().hex[:16]
        self.url = f"http://127.0.0.1:{study['port']}"
        self.launched = False
        self.last_owned_pids: set[int] = set()
        self.log_path = Path(trial["output"]).with_suffix(".server.log")

    def command(self) -> list[str]:
        s = self.study
        home = self.site.parent / "container-home"
        command = [
            "docker",
            "run",
            "--pull=never",
            "-d",
            "--name",
            self.name,
            "--label",
            f"xgraph.serving-output={Path(self.trial['output']).resolve().parent}",
            "--gpus",
            "all",
            "--ipc",
            "host",
            "--publish",
            f"127.0.0.1:{s['port']}:8000",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--volume",
            f"{s['model']}:/model:ro",
            "--volume",
            f"{self.site}:/site:ro",
            "--volume",
            f"{home}:/home/xgraph",
            "--env",
            "HOME=/home/xgraph",
            "--env",
            "PYTHONPATH=/site",
            "--env",
            "VLLM_USE_V2_MODEL_RUNNER=1",
            "--env",
            "VLLM_SERVER_DEV_MODE=1",
        ]
        arm = s["arms"][self.trial["arm"]]
        constrained = arm["opt_in"]
        if constrained:
            for source, target in ((s["kg_index"], "/index"), (s["tokenizer"], "/tokenizer.json")):
                command.extend(["--volume", f"{source}:{target}:ro"])
            settings = {
                **arm.get("engine_settings", s["engine_settings"]),
                "XGRAPH_KG_INDEX": "/index",
                "XGRAPH_TOKENIZER": "/tokenizer.json",
            }
            if s.get("kg_versions"):
                command.extend(["--volume", f"{s['kg_versions']}:/kg-versions:ro"])
                settings["XGRAPH_KG_VERSIONS"] = "/kg-versions"
            if arm.get("live_mintqa") and int(settings.get("XGRAPH_MENU", "0")):
                labels = arm["options"]["relation_labels"]
                command.extend(["--volume", f"{labels}:/relation-labels.jsonl:ro"])
                settings["XGRAPH_RELATION_LABELS"] = "/relation-labels.jsonl"
            for key, value in sorted(settings.items()):
                command.extend(["--env", f"{key}={value}"])
        command.extend(
            [
                "--entrypoint",
                "vllm",
                s["image"],
                "serve",
                "/model",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--generation-config",
                "vllm",
                "--seed",
                str(self.trial["seed"]),
                "--kv-cache-memory-bytes",
                str(self.kv_cache_bytes),
                "--gpu-memory-utilization",
                "0.45",
                "--max-model-len",
                str(s["max_model_len"]),
                "--max-num-seqs",
                str(s["max_num_seqs"]),
                "--limit-mm-per-prompt",
                '{"image":0,"video":0}',
                "--enable-prefix-caching"
                if self.trial["cache"] != "off"
                else "--no-enable-prefix-caching",
            ]
        )
        if arm.get("live_mintqa"):
            command.extend(["--enable-per-request-metrics", "--enable-prompt-tokens-details"])
        if constrained:
            command.extend(["--logits-processors", "xgraph.vllm:XGraphLogitsProcessor"])
        elif arm.get("kind") == "gcr":
            width = max(r["record"]["config"]["k"] for r in arm["requests"])
            command.extend(
                [
                    "--logits-processors",
                    "xgraph.path_trie:GCRLogitsProcessor",
                    "--logprobs-mode",
                    "raw_logprobs",
                    "--max-logprobs",
                    str(2 * width),
                ]
            )
        return command

    def start(self) -> dict:
        # A ready listener on this port must never be mistaken for our own server.
        try:
            _get(self.url + "/health", timeout=1)
        except (OSError, urllib.error.URLError):
            pass
        else:
            raise RuntimeError(f"the serving port is already occupied: {self.url}")
        (self.site.parent / "container-home").mkdir(exist_ok=True)
        self.launched = True  # stop also handles an interrupted `docker run`
        _capture(self.command())
        deadline = time.monotonic() + self.study["startup_timeout"]
        while True:
            if (
                _capture(["docker", "inspect", "--format", "{{.State.Running}}", self.name]).strip()
                != "true"
            ):
                raise RuntimeError(f"{self.name} exited during startup; see {self.log_path}")
            try:
                _get(self.url + "/health", timeout=2)
                break
            except (OSError, urllib.error.URLError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"server startup exceeded {self.study['startup_timeout']} seconds"
                    ) from None
                time.sleep(1)
        try:
            raw = json.loads(_get(self.url + "/server_info?config_format=json"))
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            raise RuntimeError(
                "vLLM /server_info is missing after /health succeeded; "
                "launch the serving container with --env VLLM_SERVER_DEV_MODE=1 "
                f"to enable effective-configuration checks; see {self.log_path}"
            ) from error
        config = _configuration(raw)
        # Server logs go to either Docker stream; capture both for the KV token capacity.
        logs = subprocess.run(
            ["docker", "logs", self.name], capture_output=True, text=True, check=True, timeout=30
        )
        self.log_path.write_text(logs.stdout + logs.stderr)
        capacities = re.findall(r"GPU KV cache size:\s*([\d,]+) tokens", self.log_path.read_text())
        if not capacities:
            raise RuntimeError("server logs do not report actual GPU KV token capacity")
        effective = {
            key: config.get(key)
            for key in (
                "kv_cache_memory_bytes",
                "enable_prefix_caching",
                "max_model_len",
                "max_num_seqs",
            )
        }
        effective.update(kv_capacity_tokens=int(capacities[-1].replace(",", "")), server_info=raw)
        environment = json.loads(
            _capture(["docker", "inspect", "--format", "{{json .Config.Env}}", self.name])
        )
        effective["environment"] = dict(
            value.split("=", 1) for value in environment if value.startswith(("VLLM_", "XGRAPH_"))
        )
        effective["engine_settings"] = dict(
            value.split("=", 1) for value in environment if value.startswith("XGRAPH_")
        )
        return effective

    def owned_pids(self) -> set[int]:
        if not self.launched:
            return self.last_owned_pids
        # The container may have exited; retain ownership during CUDA cleanup.
        with suppress(subprocess.CalledProcessError):
            self.last_owned_pids |= {
                int(line.strip())
                for line in _capture(["docker", "top", self.name, "-eo", "pid"]).splitlines()
                if line.strip().isdigit()
            }
        return self.last_owned_pids

    def metrics(self) -> dict:
        try:
            raw = _get(self.url + "/metrics").decode()
        except (OSError, urllib.error.URLError):
            return {"available": False}
        counters: dict[str, float] = {}
        numeric: dict[str, float] = {}
        for line in raw.splitlines():
            if line.startswith("vllm:"):
                name, _, number = line.rpartition(" ")
                try:
                    key = name.partition("{")[0]
                    if not math.isfinite(float(number)):
                        continue
                    numeric[key] = numeric.get(key, 0) + float(number)
                    if "prefix_cache" in key:
                        counters[key] = counters.get(key, 0) + float(number)
                except ValueError:
                    continue
        return {
            "available": True,
            "prefix_cache_counters": counters,
            "engine_counters": {
                k: v for k, v in numeric.items() if k.endswith(("_sum", "_count", "_total"))
            },
            "prometheus_text": raw,
        }

    def stop(self) -> None:
        if not self.launched:
            return
        try:
            logs = subprocess.run(
                ["docker", "logs", self.name], capture_output=True, text=True, timeout=30
            )
            if logs.returncode == 0:
                self.log_path.write_text(logs.stdout + logs.stderr)
        finally:
            result = subprocess.run(
                ["docker", "stop", "--time", "30", self.name],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode and "No such container" not in result.stderr:
                raise RuntimeError(
                    f"could not stop owned timing server {self.name}: {result.stderr}"
                )
            removed = subprocess.run(
                ["docker", "rm", self.name], capture_output=True, text=True, timeout=30
            )
            if removed.returncode and "No such container" not in removed.stderr:
                raise RuntimeError(
                    f"could not remove owned timing container {self.name}: {removed.stderr}"
                )
            self.launched = False


class ResourceMonitor:
    """Sample host availability, GPU processes and server process resources by phase.

    MemAvailable is a unified-memory proxy, not GPU allocation or process RSS.
    Unobserved short-lived processes and activity between samples remain a limitation.
    """

    def __init__(self, server: VLLMServer, interval: float):
        self.server, self.interval = server, interval
        self.samples: list[dict] = []
        self.errors: list[str] = []
        self.foreign: set[int] = set()
        self.current_phase = "startup"
        self.finished = threading.Event()
        self.mutex = threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.started = time.perf_counter()
        self.smi = os.environ.get("XG_NVIDIA_SMI", "nvidia-smi")

    def start(self) -> None:
        self.hardware = (
            _capture(
                [
                    self.smi,
                    "--query-gpu=name,uuid,driver_version,memory.total",
                    "--format=csv,noheader,nounits",
                ]
            )
            .strip()
            .splitlines()
        )
        if len(self.hardware) != 1:
            raise ValueError("this runner requires exactly one visible GPU")
        self._sample()
        self.thread.start()

    def phase(self, name: str) -> None:
        self._sample()
        with self.mutex:
            self.current_phase = name
        self._sample()

    def _sample(self) -> None:
        with self.mutex:
            try:
                meminfo = Path(os.environ.get("XG_MEMINFO", "/proc/meminfo")).read_text()
                available = int(re.search(r"^MemAvailable:\s+(\d+)", meminfo, re.M)[1]) * 1024
                raw = _capture(
                    [
                        self.smi,
                        "--query-compute-apps=pid,process_name,used_gpu_memory",
                        "--format=csv,noheader,nounits",
                    ]
                )
                owned = self.server.owned_pids()
                processes = []
                for row in csv.reader(io.StringIO(raw)):
                    if not row:
                        continue
                    pid, name, memory = (part.strip() for part in row)
                    pid = int(pid)
                    if pid not in owned:
                        self.foreign.add(pid)
                    processes.append(
                        {
                            "pid": pid,
                            "name": name,
                            "owned": pid in owned,
                            "gpu_memory_bytes": int(memory) * 1024**2 if memory.isdigit() else None,
                        }
                    )
                rss, cpu = 0, {}
                for pid in owned:
                    try:
                        status = Path(f"/proc/{pid}/status").read_text()
                        match = re.search(r"^VmRSS:\s+(\d+)", status, re.M)
                        rss += int(match[1]) * 1024 if match else 0
                        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                        # utime/stime are fields 14/15; fields after comm start at field 3.
                        cpu[str(pid)] = (int(stat[11]) + int(stat[12])) / os.sysconf("SC_CLK_TCK")
                    except FileNotFoundError:
                        continue
                self.samples.append(
                    {
                        "seconds": time.perf_counter() - self.started,
                        "phase": self.current_phase,
                        "host_available_bytes": available,
                        "server_rss_bytes": rss,
                        "server_cpu_seconds_by_pid": cpu,
                        "gpu_processes": processes,
                    }
                )
            except Exception as error:
                self.errors.append(str(error))

    def _loop(self):
        while not self.finished.wait(self.interval):
            self._sample()

    def stop(self) -> dict:
        self.finished.set()
        if self.thread.is_alive():
            self.thread.join()
        self._sample()
        before = self.samples[0]["host_available_bytes"] if self.samples else None
        lowest = min((s["host_available_bytes"] for s in self.samples), default=None)
        measured = [s for s in self.samples if s["phase"] == "measured"]
        cpu = {}
        for sample in measured:
            for pid, seconds in sample["server_cpu_seconds_by_pid"].items():
                cpu.setdefault(pid, []).append(seconds)
        gpu_memory = [
            sum(p["gpu_memory_bytes"] for p in s["gpu_processes"] if p["owned"])
            for s in self.samples
            if s["gpu_processes"]
            and any(p["owned"] for p in s["gpu_processes"])
            and all(p["gpu_memory_bytes"] is not None for p in s["gpu_processes"] if p["owned"])
        ]
        return {
            "samples": self.samples,
            "sample_seconds": self.interval,
            "hardware": self.hardware,
            "gpu_count": len(self.hardware),
            "foreign_gpu_processes": sorted(self.foreign),
            "errors": self.errors,
            "host_available_bytes_before": before,
            "host_available_bytes_lowest": lowest,
            "host_available_drop_bytes": max(0, before - lowest) if before is not None else None,
            "host_memory_scope": (
                "system MemAvailable proxy; includes page cache and unrelated host use"
            ),
            "server_peak_rss_bytes": max(
                (s["server_rss_bytes"] for s in self.samples), default=None
            ),
            "server_rss_scope": (
                "sum of container process RSS; shared pages may be counted more than once"
            ),
            "server_cpu_seconds": sum(max(v) - min(v) for v in cpu.values())
            if len(measured) >= 2
            else None,
            "server_cpu_scope": (
                "sampled /proc CPU ticks for container processes during measured phase"
            ),
            "gpu_peak_process_memory_bytes": max(gpu_memory, default=None),
            "gpu_memory_scope": (
                "sampled nvidia-smi owned compute-process memory; null when unavailable"
            ),
        }


def _validate_effective(effective: dict, study: dict, trial: dict, kv_cache_bytes: int) -> None:
    expected = {
        "kv_cache_memory_bytes": kv_cache_bytes,
        "enable_prefix_caching": trial["cache"] != "off",
        "max_model_len": study["max_model_len"],
        "max_num_seqs": study["max_num_seqs"],
    }
    for key, value in expected.items():
        if effective.get(key) != value or type(effective.get(key)) is not type(value):
            raise ValueError(f"effective server {key}={effective.get(key)!r}, requested {value!r}")
    if (
        not isinstance(effective.get("kv_capacity_tokens"), int)
        or effective["kv_capacity_tokens"] <= 0
    ):
        raise ValueError("server must report positive actual KV token capacity")
    actual = study.get("actual_kv_capacity_tokens")
    if actual is not None and effective["kv_capacity_tokens"] != actual:
        raise ValueError("actual KV capacity differs from earlier cells with equal requested bytes")
    arm = study.get("arms", {}).get(trial["arm"], {})
    if (arm.get("confirmation") or arm.get("live_mintqa")) and arm.get("opt_in"):
        support = effective.get("engine_settings", {})
        for key, value in arm["engine_settings"].items():
            if support.get(key) != value:
                raise ValueError(
                    f"effective engine support {key}={support.get(key)!r}, requested {value!r}"
                )


class _TimedStore:
    def __init__(self, store):
        self._wrapped = store
        self.calls = 0
        self.seconds = 0.0

    def __getattr__(self, name):
        attribute = getattr(self._wrapped, name)
        if not callable(attribute):
            return attribute

        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return attribute(*args, **kwargs)
            finally:
                self.calls += 1
                self.seconds += time.perf_counter() - started

        return timed


class _SQLCounter(threading.local):
    statements = 0

    def trace(self, statement):
        self.statements += 1


class _ReaderConnection:
    """Give each question thread its own read connection and SQLite statement cache.

    A Python trace callback runs under SQLite's connection mutex. Another thread
    using that connection can wait for that mutex while holding the GIL, preventing
    the callback from returning. A thread-local counter alone cannot prevent this.
    """

    def __init__(self, connection, counter):
        self.database = connection.execute("PRAGMA database_list").fetchone()[2]
        self.views = [
            row[0]
            for row in connection.execute("SELECT sql FROM sqlite_temp_master WHERE type='view'")
        ]
        self.counter = counter
        self.local = threading.local()
        self.connections = [connection]
        self.mutex = threading.Lock()
        self.local.connection = connection
        connection.set_trace_callback(counter.trace)

    def __getattr__(self, name):
        connection = getattr(self.local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(
                f"file:{self.database}?mode=ro", uri=True, check_same_thread=False
            )
            connection.execute("PRAGMA cache_size = -64")
            for view in self.views:
                # sqlite_temp_master stores CREATE VIEW without the TEMP keyword.
                connection.execute(view.replace("CREATE VIEW", "CREATE TEMP VIEW", 1))
            connection.set_trace_callback(self.counter.trace)
            self.local.connection = connection
            with self.mutex:
                self.connections.append(connection)
        return getattr(connection, name)

    def close(self):
        # Called only after the question/scoring pools have joined.
        for connection in self.connections:
            connection.close()


class _SharedTimedStore(_TimedStore):
    """An executor shared by repair and readers, with question-thread counters."""

    def __init__(self, store):
        self.local = threading.local()
        super().__init__(store)

    @property
    def calls(self):
        return getattr(self.local, "calls", 0)

    @calls.setter
    def calls(self, value):
        self.local.calls = value

    @property
    def seconds(self):
        return getattr(self.local, "seconds", 0.0)

    @seconds.setter
    def seconds(self, value):
        self.local.seconds = value


def _close_reader(context: dict) -> None:
    """Close owned connections once, including the question-component stores."""
    if context.get("scoring_pool") is not None:
        context["scoring_pool"].shutdown(wait=True)
    stores = [context["store"], *(v.store for v in (context["versions"] or {}).values())]
    for connection in {id(store._con): store._con for store in stores}.values():
        connection.close()


def prepare_reader(arm: dict) -> dict:
    """Load shared reader resources once per server cell, outside steady-state timing."""
    from xgraph.toolkg import KGStore, version_stores

    context = {
        "store": KGStore(arm["tools_db"]),
        "versions": None,
        "names": None,
        "tokenizer": None,
        "executor": None,
        "cor_prompts": None,
        "scoring_pool": ThreadPoolExecutor(max_workers=32) if arm["kind"] == "gcr" else None,
        "sql": _SQLCounter(),
    }
    try:
        options = arm.get("options", {})
        if arm["kind"] == "cor" and options.get("cor_prompts"):
            from xgraph.cor import CorPrompts

            context["cor_prompts"] = CorPrompts.load(options["cor_prompts"])
        named = {r["kg_version"] for r in arm.get("requests", []) if r.get("kg_version")}
        if named:
            if not arm.get("kg_versions"):
                raise ValueError("versioned reader replay needs --kg-versions")
            context["versions"] = version_stores(context["store"], arm["kg_versions"], named)
        stores = [context["store"], *(v.store for v in (context["versions"] or {}).values())]
        for store in {id(s): s for s in stores}.values():
            store._con = _ReaderConnection(store._con, context["sql"])
        if options.get("max_prompt_tokens") is not None or arm["kind"] == "gcr":
            from transformers import AutoTokenizer

            context["tokenizer"] = AutoTokenizer.from_pretrained(
                arm["model"], local_files_only=True, trust_remote_code=False
            )
        if options.get("relation_labels"):
            from xgraph.agent import relation_names_of

            context["names"] = relation_names_of(
                [
                    tuple(json.loads(line))
                    for line in Path(options["relation_labels"]).read_text().splitlines()
                    if line.strip()
                ]
            )
        if (
            options.get("matched_paths")
            or options.get("feedback")
            or arm["kind"] in ("walk", "resume", "execute", "repair")
        ):
            from xgraph.index import load_kg_index
            from xgraph.kgversions import load_kg_versions
            from xgraph.programs import MatchedExecutor
            from xgraph.tokenizer import ServedTokenizer

            index = load_kg_index(arm["kg_index"])
            context["executor"] = MatchedExecutor(
                index,
                ServedTokenizer.from_file(arm["tokenizer"]),
                versions=load_kg_versions(arm["kg_versions"], index) if named else None,
                start_rank=options.get("start_rank", "degree"),
                unquoted_starts=options.get("unquoted_starts", not options.get("feedback", False)),
                strip_quotes=options.get("strip_quotes", False),
            )
            context["executor"] = _SharedTimedStore(context["executor"])
            context["executor_timing"] = context["executor"]
        if arm["kind"] == "repair" or options.get("revision"):
            from xgraph.repair import Repairer

            rows = [
                tuple(json.loads(line))
                for line in Path(options["relation_labels"]).read_text().splitlines()
                if line.strip()
            ]
            context["repairer"] = Repairer(
                context["executor"], context["store"], rows, edits=4, stores=context["versions"]
            )
        if arm.get("live_mintqa"):
            from xgraph.mintqa_rematch import AuditMenus

            context["served_tokenizer"] = context["executor"]._tokenizer
            context["audit_menus"] = AuditMenus(
                context["executor"]._index, context["served_tokenizer"]
            )
        if options.get("cpu_stop"):
            from xgraph.stopping import RelationTerms

            context["stop_terms"] = RelationTerms.from_index(index, context["executor"]._tokenizer)
        return context
    except BaseException:
        _close_reader(context)
        raise


def replay_reader(
    url: str, request: dict, arm: dict, *, timeout: float = 600, context: dict | None = None
) -> dict:
    """Issue every recorded model call, executing the reader's actual CPU/KG logic.

    New server completions are measured, while frozen replies drive the next call.
    This is trajectory replay, not a fresh adaptive reader or an accuracy experiment.
    Sparse agent action logs cannot reconstruct the model's assistant messages.
    """
    started = time.perf_counter()
    record = request["record"]
    kind = arm["kind"]
    options = arm.get("options", {})
    turns = record.get("turns")
    if kind == "resume" and isinstance(turns, list):
        turns = [{**t, "text": t["reply"]} for t in turns]
    if turns is None and kind in ("iterative", "walk"):
        turns = [{"text": reply} for reply in record.get("replies", [])]
        if kind == "iterative" and "start" in record:
            # The start reply is never carried into subsequent conversations.
            turns.insert(
                0, {"text": f"Start: ⟦ {record['start']}⟧" if record["start"] else "Unknown"}
            )
    if (
        not isinstance(turns, list)
        or (not turns and model_calls(record) != 0)
        or any(not isinstance(t.get("text"), str) for t in turns)
    ):
        raise ValueError(
            f"{kind} {request['id']}: need complete turns with text; actions alone are insufficient"
        )
    calls: list[dict] = []
    attempts = 0
    owns_context = context is None
    if context is None:
        context = prepare_reader({**arm, "requests": [request]})
    sql = context.get("sql")
    if sql is not None:
        sql.statements = 0
    facts = []

    class ReplayLLM:
        def get_tokenizer(self):
            if context["tokenizer"] is None:
                raise ValueError("reader prompt limits need a prepared tokenizer")
            return context["tokenizer"]

        def chat(self, conversations, sampling, **kwargs):
            nonlocal attempts
            base = len(calls)
            if base + len(conversations) > len(turns):
                raise ValueError(f"{kind} {request['id']}: reconstructed extra/unrecorded call")
            attempts += len(conversations)
            parameters = sampling if isinstance(sampling, list) else [sampling] * len(conversations)

            def issue(job):
                index, messages, params = job
                saved = turns[base + index]
                if "messages" in saved and saved["messages"] != messages:
                    raise ValueError(
                        f"{kind} {request['id']}: reconstructed prompt differs "
                        "from recorded messages"
                    )
                maximum = getattr(params, "max_tokens", arm["max_tokens"])
                settings = request.get("sampling", {})
                if kind == "gcr" or (kind == "cor" and record.get("cor_mode") == "upstream"):
                    settings = {
                        **settings,
                        **{k: getattr(params, k) for k in SAMPLING_FIELDS if hasattr(params, k)},
                    }
                call_started = time.perf_counter() - started
                result = _one(
                    url,
                    "/model",
                    {
                        "id": request["id"],
                        "messages": messages,
                        "seed": getattr(params, "seed", request.get("seed", 0)),
                        "sampling": settings,
                    },
                    maximum,
                    False,
                    timeout=timeout,
                    extra_args=getattr(params, "extra_args", None),
                    structured_outputs=getattr(params, "structured_outputs", None),
                )
                return {
                    **result,
                    "max_tokens": maximum,
                    "call": base + index,
                    "started_seconds": call_started,
                }

            jobs = list(zip(range(len(conversations)), conversations, parameters, strict=True))
            pool = context.get("scoring_pool")
            measured = (
                list(pool.map(issue, jobs)) if pool and len(jobs) > 1 else [issue(j) for j in jobs]
            )
            calls.extend(measured)
            outputs = []
            for saved, result in zip(turns[base : base + len(measured)], measured, strict=True):
                if (
                    saved.get("prompt_tokens") is not None
                    and saved["prompt_tokens"] != result["prompt_tokens"]
                ):
                    raise ValueError(
                        f"{kind} {request['id']}: reconstructed prompt token count "
                        "differs from trace"
                    )
                outputs.append(
                    SimpleNamespace(
                        prompt_token_ids=[0] * result["prompt_tokens"],
                        outputs=[
                            SimpleNamespace(
                                text=saved["text"].removeprefix(saved.get("prefill") or ""),
                                token_ids=saved.get(
                                    "token_ids",
                                    [0]
                                    * saved.get("generated_tokens", result["completion_tokens"]),
                                ),
                                finish_reason=saved.get("finish_reason", "stop"),
                            )
                        ],
                    )
                )
            return outputs

        def generate(self, prompts, sampling, **kwargs):
            nonlocal attempts
            base = len(calls)
            if base + len(prompts) > len(turns):
                raise ValueError(f"gcr {request['id']}: unrecorded scoring call")
            attempts += len(prompts)

            def issue(job):
                index, prompt, params = job
                saved = turns[base + index]
                if saved.get("kind") != "beam_step" or not saved.get("token_logprobs"):
                    raise ValueError("GCR beam replay needs every raw scoring turn and logprobs")
                call_started = time.perf_counter() - started
                result = score_tokens(
                    url, request["id"], prompt["prompt_token_ids"], params, timeout
                )
                if saved.get("prompt_tokens") != result["prompt_tokens"]:
                    raise ValueError("GCR beam reconstructed prompt token count differs from trace")
                return {**result, "started_seconds": call_started, "call": base + index}

            jobs = list(zip(range(len(prompts)), prompts, sampling, strict=True))
            pool = context.get("scoring_pool")
            if pool is None:
                with ThreadPoolExecutor(max_workers=min(32, len(jobs))) as temporary:
                    measured = list(temporary.map(issue, jobs))
            else:
                measured = list(pool.map(issue, jobs))
            calls.extend(measured)
            return [
                SimpleNamespace(
                    prompt_token_ids=prompt["prompt_token_ids"],
                    outputs=[
                        SimpleNamespace(
                            text=saved["text"],
                            token_ids=saved["token_ids"],
                            finish_reason=saved["finish_reason"],
                            logprobs=[
                                {
                                    int(t): SimpleNamespace(logprob=float(v))
                                    for t, v in saved["token_logprobs"].items()
                                }
                            ],
                        )
                    ],
                )
                for prompt, saved in zip(prompts, turns[base : base + len(prompts)], strict=True)
            ]

    llm = ReplayLLM()
    read = _TimedStore(context["store"])
    versions = {
        name: _TimedStore(value) for name, value in (context["versions"] or {}).items()
    } or None
    names = context["names"]
    executor = context.get("executor_timing")
    if executor is not None:
        executor.calls, executor.seconds = 0, 0.0
    elif context["executor"] is not None:
        executor = _TimedStore(context["executor"])
    asked = {k: v for k, v in request.items() if k != "record"}
    if kind == "cor" and record.get("cor_mode") == "upstream":
        asked["seed"] = record.get("seed", asked.get("seed", 0))
    sampling = SimpleNamespace(max_tokens=arm["max_tokens"])
    try:
        if kind == "cor":
            from xgraph.cor import run_cor

            prompts = context.get("cor_prompts")
            if record.get("cor_mode") == "upstream" and (
                prompts is None or record.get("cor_prompts", {}).get("sha256") != prompts.sha256
            ):
                raise ValueError(
                    f"cor {request['id']}: local prompt file hash differs from the record"
                )
            (result,) = run_cor(
                llm,
                [asked],
                read,
                sampling,
                names=names,
                versions=versions,
                mode=record.get("cor_mode", "reconstructed"),
                prompts=prompts,
                **{
                    k: v
                    for k, v in options.items()
                    if k in ("max_calls", "max_depth", "max_prompt_tokens")
                },
            )
        elif kind == "agent":
            from xgraph.agent import demonstrations, run_agent

            if options.get("matched_demos") or options.get("feedback"):
                asked["demonstrations"] = demonstrations(asked["messages"])
            from xgraph.generate import engine_arguments

            def round_params(r, prefill):
                return SimpleNamespace(
                    max_tokens=arm["max_tokens"],
                    extra_args=engine_arguments(
                        "constrained" if options.get("mask") else "agent",
                        anchor=r.get("anchor"),
                        kg_version=r.get("kg_version"),
                        prefill=prefill if options.get("mask") else None,
                    ),
                )

            (result,) = run_agent(
                llm,
                [asked],
                read,
                sampling,
                names=names,
                versions=versions,
                executor=executor,
                round_params=round_params
                if options.get("mask") or options.get("force_path") or options.get("schema")
                else None,
                schema_params=(lambda **kwargs: kwargs) if options.get("schema") else None,
                **{
                    k: v
                    for k, v in options.items()
                    if k in ("max_calls", "pages", "feedback", "mask", "force_path", "schema")
                },
            )
        elif kind == "iterative":
            from xgraph.iterative import START_TOKENS, Facts, iterate

            def tracked_facts(db, version):
                value = Facts(db, version)
                if sql is not None:
                    value._con.set_trace_callback(sql.trace)
                    if version is None:
                        value.read._con.set_trace_callback(sql.trace)
                timed = _TimedStore(value)
                facts.append((timed, version is None))
                return timed

            if record.get("turns") is None and options.get("max_prompt_tokens") is not None:
                raise ValueError(
                    "iterative clipping needs per-turn usage; supply turns or "
                    "omit max_prompt_tokens and verify totals"
                )
            (result,) = iterate(
                llm,
                [asked],
                arm["tools_db"],
                sampling,
                versions=versions,
                facts_factory=tracked_facts,
                start_params=SimpleNamespace(
                    max_tokens=options.get("start_max_tokens", START_TOKENS)
                ),
                **{
                    k: v
                    for k, v in options.items()
                    if k in ("max_lookups", "cap", "max_prompt_tokens", "written_start")
                },
            )
        elif kind == "walk":
            from xgraph.walk import walk

            (result,) = walk(
                llm,
                [asked],
                read,
                sampling,
                max_hops=options.get("max_calls", 8),
                versions=versions,
                observe="labels",
                max_options=None,
                min_hops=options.get("min_hops", 0),
                executor=executor,
            )
        elif kind == "resume":
            from xgraph.resume import resume

            (result,) = resume(
                llm,
                [asked],
                read,
                executor,
                sampling,
                resume_max=options.get("resume_max", 3),
                max_hops=options.get("max_calls", 8),
                versions=versions,
                max_prompt_tokens=options.get("max_prompt_tokens"),
            )
        elif kind == "gcr":
            from xgraph.gcr import GCRConfig, run_gcr

            config = GCRConfig(**record["config"])
            if config.decoding != "beam":
                raise ValueError("confirmation GCR timing requires beam decoding")
            (result,) = run_gcr(
                llm, [asked], read, SimpleNamespace, config=config, versions=versions
            )
            for original, rebuilt in zip(turns, result["turns"], strict=True):
                for field in ("prefix_sha256", "token_logprobs"):
                    a, b = original.get(field), rebuilt.get(field)
                    if field == "token_logprobs" and a is not None:
                        a = {int(k): v for k, v in a.items()}
                    if a != b:
                        raise ValueError(f"GCR beam reconstructed {field} differs from trace")
        else:
            raise ValueError(f"unknown reader kind {kind!r}")
        if len(calls) != len(turns) or model_calls(record) != len(calls):
            raise ValueError(f"{kind} {request['id']}: reconstructed call count differs from trace")
        if (
            record.get("prompt_tokens") is None
            or sum(c["prompt_tokens"] for c in calls) != record["prompt_tokens"]
        ):
            raise ValueError(
                f"{kind} {request['id']}: whole-trajectory prompt token count differs or is unknown"
            )
        if result["text"] != record["text"]:
            raise ValueError(
                f"{kind} {request['id']}: reconstructed final delivery differs from trace"
            )
        latency = time.perf_counter() - started
        model_wall, end = 0.0, 0.0
        for call in sorted(calls, key=lambda c: c["started_seconds"]):
            beginning = call["started_seconds"]
            ending = min(latency, beginning + call["latency"])
            model_wall += max(0.0, ending - max(end, beginning))
            end = max(end, ending)
        return {
            "id": request["id"],
            "text": result["text"],
            "trajectory_kind": "frozen_reader_replay",
            "calls": calls,
            "model_calls": attempts,
            "tool_calls": result.get(
                "tool_calls",
                len(result["lookups"])
                if "lookups" in result
                else max(0, len(result.get("path", [])) - 1)
                if kind == "walk"
                else 0,
            ),
            "kg_store_calls": read.calls
            + sum(v.calls for v in (versions or {}).values())
            + sum(f.calls for f, _ in facts),
            "kg_store_seconds": sum(f.seconds for f, _ in facts)
            if facts
            else read.seconds + sum(v.seconds for v in (versions or {}).values()),
            "kg_sql_statements": sql.statements if sql else None,
            "executor_calls": executor.calls if executor else 0,
            "executor_seconds": executor.seconds if executor else 0,
            **(
                {
                    "executed_record": {
                        k: v
                        for k, v in result.items()
                        if k in ("id", "arm", "text", "plan", "reached", "finish_reason")
                    }
                }
                if kind in ("walk", "resume")
                else {}
            ),
            "retries": result.get("retries", result.get("resume_rounds", 0)),
            "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
            "completion_tokens": sum(c["completion_tokens"] for c in calls),
            "finish_reason": calls[-1]["finish_reason"] if calls else result.get("finish_reason"),
            "truncated": result.get("finish_reason") == "length"
            or any(c["finish_reason"] == "length" and c.get("kind") != "beam_step" for c in calls),
            "ttft": calls[0]["ttft"] + calls[0]["started_seconds"]
            if calls and calls[0]["ttft"] is not None
            else None,
            "latency": latency,
            "reader_cpu_and_tool_wall_seconds": latency - model_wall,
        }
    except Exception as error:
        error.result = {
            "calls": calls,
            "model_calls": attempts,
            "prompt_tokens": sum(c["prompt_tokens"] for c in calls)
            if len(calls) == attempts
            else None,
            "completion_tokens": sum(c["completion_tokens"] for c in calls)
            if len(calls) == attempts
            else None,
        }
        raise
    finally:
        for value, owns_read in facts:
            if owns_read:
                value.read._con.close()
        if owns_context:
            _close_reader(context)


def run_cell(study: dict, trial: dict, kv_cache_bytes: int, site: Path) -> dict:
    """Run and retain one complete cell, including failed/invalid/interrupted trials."""
    report = {
        "artifact_kind": "xgraph.controlled_serving",
        "version": 1,
        "status": "running",
        "trial": trial.copy(),
        "kv_cache_bytes": kv_cache_bytes,
        "filesystem_cache_state": study.get("filesystem_cache_state", "uncontrolled"),
        "study_path": str(Path(trial["output"]).parent / "study.json"),
    }
    out = Path(trial["output"])
    server = VLLMServer(study, trial, kv_cache_bytes, site)
    if hasattr(server, "command"):
        report["launch_command"] = server.command()
    monitor = ResourceMonitor(server, study["sample_seconds"])
    started = time.perf_counter()
    raised = None
    reader = None
    phase = "startup"

    def progress(count, total, result):
        with out.with_suffix(".partial.jsonl").open("a") as partial:
            partial.write(json.dumps({"phase": phase, "result": result}, ensure_ascii=False) + "\n")
        every = study.get("heartbeat_requests", 8)
        if count % every == 0 or count == total:
            print(f"trial progress phase={phase} completed={count}/{total}", flush=True)

    try:
        print("trial phase startup", flush=True)
        monitor.start()
        report["effective_server"] = server.start()
        _validate_effective(report["effective_server"], study, trial, kv_cache_bytes)
        report["startup_seconds"] = time.perf_counter() - started
        arm = study["arms"][trial["arm"]]
        reader_started = time.perf_counter()
        print("trial phase reader preparation", flush=True)
        if arm.get("kind"):
            reader = prepare_reader(arm)
        report["reader_preparation_seconds"] = time.perf_counter() - reader_started
        ordered = arm["requests"].copy()
        random.Random(trial["seed"]).shuffle(ordered)
        ordered = [{"seed": trial["seed"], **r} for r in ordered]

        def question(request):
            if arm.get("live_mintqa"):
                from xgraph.serve_mintqa import run_policy

                return run_policy(
                    server.url, request, arm, timeout=study["request_timeout"], context=reader
                )
            if arm.get("confirmation") and (
                arm["kind"] in ("paths", "execute", "repair") or arm.get("delivery") == "engine"
            ):
                return replay_arm(
                    server.url, request, arm, timeout=study["request_timeout"], context=reader
                )
            if arm.get("kind"):
                return replay_reader(
                    server.url, request, arm, timeout=study["request_timeout"], context=reader
                )
            if not arm["opt_in"]:
                request = {k: v for k, v in request.items() if k != "prefill_replay"}
            opener = None if request["messages"][-1]["role"] == "assistant" else arm["opener"]
            support = arm.get("engine_settings", study.get("engine_settings", {}))
            extra = None
            if arm["opt_in"] and support.get("XGRAPH_STOP"):
                question_text = next(
                    (m["content"] for m in reversed(request["messages"]) if m["role"] == "user"),
                    None,
                )
                if not isinstance(question_text, str) or not question_text.strip():
                    raise ValueError(f"request {request['id']}: STOP needs a user question")
                extra = {"xgraph_question": question_text}
            try:
                result = _one(
                    server.url,
                    "/model",
                    request,
                    arm["max_tokens"],
                    arm["opt_in"],
                    opener,
                    timeout=study["request_timeout"],
                    extra_args=extra,
                )
            except Exception as error:
                error.result = {"model_calls": 1, "tool_calls": 0}
                raise
            return {**result, "model_calls": 1, "tool_calls": 0}

        report["metrics_at_ready"] = server.metrics()
        warmup = (
            arm.get("warmup_requests", arm["requests"][: study["warmup_questions"]])
            if trial["cache"] == "warm"
            else []
        )
        warm_started = time.perf_counter()
        report["warmup"] = {
            "questions": len(warmup),
            "sha256": hashlib.sha256(json.dumps(warmup, sort_keys=True).encode()).hexdigest(),
        }
        monitor.phase("warmup")
        phase = "warmup"
        print(f"trial phase warmup requests={len(warmup)}", flush=True)
        if warmup:
            warm_results, warm_summary = measure_workload(
                warmup, concurrency=trial["concurrency"], question=question, progress=progress
            )
            report["warmup"].update(
                results=warm_results,
                summary=warm_summary,
                seconds=time.perf_counter() - warm_started,
            )
            if warm_summary["failures"]:
                raise RuntimeError("warm-up failed")
        report["warmup"]["seconds"] = time.perf_counter() - warm_started
        report["metrics_before"] = server.metrics()
        monitor.phase("measured")
        phase = "measured"
        print(f"trial phase measured requests={len(ordered)}", flush=True)
        report["results"], report["summary"] = measure_workload(
            ordered,
            concurrency=trial["concurrency"],
            question=question,
            gpu_count=study["gpu_count"],
            progress=progress,
        )
        monitor.phase("teardown")
        report["metrics_after"] = server.metrics()
        before = report["metrics_before"].get("engine_counters", {})
        after = report["metrics_after"].get("engine_counters", {})
        report["measured_engine_counter_deltas"] = {
            k: v - before.get(k, 0) for k, v in after.items()
        }
        report["summary"]["client_peak_rss_mib"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        )
        report["summary"]["client_memory_scope"] = (
            "load generator worker process high-water mark; one trial"
        )
        report["summary"]["distributions"] = distributions(
            {
                "prompt_tokens": r.get("prompt_tokens"),
                "generated_tokens": r.get("completion_tokens"),
                "model_calls": r.get("model_calls"),
            }
            for r in report["results"]
        )
        if arm.get("live_mintqa"):
            for key in (
                "observation_tokens",
                "cached_prompt_tokens",
                "uncached_prompt_tokens",
                "engine_prefill_first_token_seconds",
                "engine_decode_seconds",
                "engine_queue_seconds",
                "client_thread_cpu_seconds",
            ):
                values = [r.get(key) for r in report["results"]]
                report["summary"][key] = sum(values) if all(v is not None for v in values) else None
        report["status"] = "failed" if report["summary"]["failures"] else "completed"
    except BaseException as error:
        if hasattr(error, "results"):
            report["results"] = error.results
        report.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=str(error),
        )
        raised = error
    finally:
        teardown = time.perf_counter()
        if reader is not None:
            _close_reader(reader)
        try:
            server.stop()
            if hasattr(server, "log_path") and server.log_path.is_file():
                report["prefill_declines"] = server.log_path.read_text().count(
                    "declined the replayed prefill"
                )
                if report["prefill_declines"]:
                    report.update(
                        status="invalid", error="engine declined replayed prefills; left unmasked"
                    )
        except Exception as error:
            report.update(status="failed", teardown_error=str(error))
        try:
            report["resources"] = monitor.stop()
            if report["resources"]["foreign_gpu_processes"] or report["resources"]["errors"]:
                report["status"] = "invalid"
        except Exception as error:
            report.update(status="invalid", monitor_error=str(error))
        report["teardown_seconds"] = time.perf_counter() - teardown
        report["driver_seconds"] = time.perf_counter() - started
        if "summary" in report:
            report["protocol_overhead_seconds"] = max(
                0,
                report["driver_seconds"]
                - sum(
                    (
                        report.get("startup_seconds", 0),
                        report.get("reader_preparation_seconds", 0),
                        report.get("warmup", {}).get("seconds", 0),
                        report["summary"]["wall_seconds"],
                        report["teardown_seconds"],
                    )
                ),
            )
        write_json(out, report)
    if raised:
        raise raised
    return report


def study_report(study: dict, plan: dict) -> dict:
    """Aggregate valid repeated trials by cell; pair the engine delta within repeats."""
    reports = []
    for trial in plan["trials"]:
        path = Path(trial["output"])
        if path.exists():
            report = json.loads(path.read_text())
            if study.get("quality_answers") and "summary" in report:
                from xgraph.serve_mintqa import score_live_trial

                score_live_trial(report, study["quality_answers"])
            if report.get("status") == "completed" and trial["status"] == "completed":
                reports.append(report)
    capacities = {r["effective_server"]["kv_capacity_tokens"] for r in reports}
    if len(capacities) > 1:
        raise ValueError(
            f"equal requested bytes gave unequal actual KV capacities: {sorted(capacities)}"
        )
    groups: dict[tuple, list[dict]] = {}
    for report in reports:
        t = report["trial"]
        groups.setdefault((t["arm"], t["cache"], t["concurrency"]), []).append(report)
    cells = []
    per_question = {
        "serving_seconds": "wall_seconds",
        "gpu_seconds": "gpu_seconds",
        "prompt_tokens": "prompt_tokens",
        "generated_tokens": "completion_tokens",
        "model_calls": "model_calls",
        "tool_calls": "tool_calls",
        "observation_tokens": "observation_tokens",
        "cached_prompt_tokens": "cached_prompt_tokens",
        "uncached_prompt_tokens": "uncached_prompt_tokens",
        "engine_prefill_first_token_seconds": "engine_prefill_first_token_seconds",
        "engine_decode_seconds": "engine_decode_seconds",
        "engine_queue_seconds": "engine_queue_seconds",
        "client_thread_cpu_seconds": "client_thread_cpu_seconds",
    }
    for (arm, cache, concurrency), runs in sorted(groups.items()):
        metrics = {}
        for key, source in per_question.items():
            values = [
                r["summary"][source] / r["summary"]["requests"]
                for r in runs
                if r["summary"].get(source) is not None
            ]
            if len(values) == len(runs):
                metrics[key] = trial_interval(values, plan["seed"])
        if study.get("cost_weights"):
            weights = study["cost_weights"]
            metrics["population_gpu_seconds"] = trial_interval(
                [
                    sum(weights[r["id"]] * r["gpu_seconds"] for r in run["results"])
                    / sum(weights.values())
                    for run in runs
                ],
                plan["seed"],
            )
        for key in (
            "latency_p50",
            "latency_p95",
            "latency_p99",
            "ttft_p50",
            "ttft_p95",
            "ttft_p99",
            "completed_questions_per_second",
            "correct_questions_per_second",
            "reserved_gpu_seconds",
            "accuracy",
            "output_tokens_per_second",
        ):
            values = [r["summary"][key] for r in runs if r["summary"].get(key) is not None]
            if len(values) == len(runs):
                metrics[key] = trial_interval(values, plan["seed"])
        for key in ("startup_seconds", "reader_preparation_seconds", "teardown_seconds"):
            values = [r[key] for r in runs if r.get(key) is not None]
            if len(values) == len(runs):
                metrics[key] = trial_interval(values, plan["seed"])
        for key in (
            "gpu_peak_process_memory_bytes",
            "server_peak_rss_bytes",
            "host_available_drop_bytes",
            "server_cpu_seconds",
        ):
            values = [
                r["resources"][key] for r in runs if r.get("resources", {}).get(key) is not None
            ]
            if len(values) == len(runs):
                metrics[key] = trial_interval(values, plan["seed"])
        expected = sum(
            1
            for t in plan["trials"]
            if (t["arm"], t["cache"], t["concurrency"]) == (arm, cache, concurrency)
        )
        cells.append(
            {
                "arm": arm,
                "cache": cache,
                "concurrency": concurrency,
                "status": "completed" if len(runs) == expected else "incomplete",
                "metrics": metrics,
                "trial_outputs": [r["trial"]["output"] for r in runs],
                "question_costs": question_costs(runs, study["panel_ids"]),
                "quality_by_trial": [
                    {"trial": r["trial"], **r["quality"]} for r in runs if "quality" in r
                ],
            }
        )
        if study.get("timing_panel"):
            from xgraph.serve_mintqa import amortized_cost

            cells[-1]["amortization_by_trial"] = [
                {
                    "repeat": run["trial"]["repeat"],
                    "costs": amortized_cost(
                        run, study["amortize_questions"], gpu_count=study["gpu_count"]
                    ),
                }
                for run in runs
            ]
            for row in cells[-1]["question_costs"]:
                row["population_weight"] = study["cost_weights"][row["id"]]
    overhead = []
    paths, free = study["arms"].get("paths"), study["arms"].get("unconstrained")
    if paths and free:
        if any(paths.get(k) != free.get(k) for k in ("requests", "max_tokens", "opener")):
            raise ValueError("engine delta needs identical prompts, opener and token budgets")
        paired = {}
        for report in reports:
            t = report["trial"]
            if t["arm"] in ("paths", "unconstrained"):
                paired.setdefault((t["cache"], t["concurrency"], t["repeat"]), {})[t["arm"]] = (
                    report
                )
        conditions = {}
        for (cache, concurrency, repeat), pair in paired.items():
            if len(pair) != 2:
                continue
            a, b = pair["paths"], pair["unconstrained"]
            if a["trial"]["seed"] != b["trial"]["seed"]:
                raise ValueError("engine delta needs matched request-order seeds")
            differences = {
                key: a["summary"][key] - b["summary"][key]
                for key in a["summary"]
                if key
                in (
                    "wall_seconds",
                    "gpu_seconds",
                    "latency_p50",
                    "latency_p95",
                    "latency_p99",
                    "ttft_p50",
                    "ttft_p95",
                    "ttft_p99",
                )
                and a["summary"].get(key) is not None
                and b["summary"].get(key) is not None
            }
            conditions.setdefault((cache, concurrency), []).append(
                {
                    "repeat": repeat,
                    "paths": a["trial"]["output"],
                    "unconstrained": b["trial"]["output"],
                    **differences,
                }
            )
        for (cache, concurrency), pairs in sorted(conditions.items()):
            keys = set.intersection(*(set(p) for p in pairs)) - {"repeat", "paths", "unconstrained"}
            overhead.append(
                {
                    "cache": cache,
                    "concurrency": concurrency,
                    "pairs": pairs,
                    "kind": "matched_prompt_serving_delta",
                    "scope": (
                        "constrained minus no processor; includes changed output histories, "
                        "not kernel attribution"
                    ),
                    **{
                        k: trial_interval([p[k] for p in pairs], plan["seed"]) for k in sorted(keys)
                    },
                }
            )
    return {
        "artifact_kind": "xgraph.serving_study_report",
        "version": 1,
        "status": plan["status"],
        "panel_ids": study["panel_ids"],
        "study": study,
        "cells": cells,
        "engine_overhead": overhead,
        "cost_ratios": cost_ratios(study, plan, reports),
        "trials": plan["trials"],
        "accuracy_scope": "live policy outputs; score retained text separately against frozen gold"
        if study.get("timing_panel")
        else "timing replay; use a matched frozen quality evaluation for accuracy",
    }


def question_costs(reports: list[dict], ids: list[str]) -> list[dict]:
    """Mean per-question attribution over complete workloads, usable by agreement.py."""
    import numpy as np

    rows = [{r["id"]: r for r in report.get("results", [])} for report in reports]
    if not rows or any(set(r) != set(ids) for r in rows):
        return []
    fields = (
        "gpu_seconds",
        "prompt_tokens",
        "completion_tokens",
        "model_calls",
        "tool_calls",
        "latency",
        "client_queue_seconds",
        "service_seconds",
        "observation_tokens",
        "cached_prompt_tokens",
        "uncached_prompt_tokens",
        "engine_prefill_first_token_seconds",
        "engine_decode_seconds",
        "engine_queue_seconds",
        "client_thread_cpu_seconds",
        "kg_store_calls",
        "retries",
    )
    result = []
    for qid in ids:
        mean = {
            key: float(np.mean([r[qid][key] for r in rows]))
            if all(r[qid].get(key) is not None for r in rows)
            else None
            for key in fields
        }
        result.append(
            {
                "id": qid,
                **mean,
                "generated_tokens": mean["completion_tokens"],
                "trials": len(reports),
            }
        )
    return result


def cost_ratios(study: dict, plan: dict, reports: list[dict]) -> list[dict]:
    """Never drop a failed/missing repeat block from an H2 cost comparison."""
    results = []
    available = {
        (r["trial"]["arm"], r["trial"]["cache"], r["trial"]["concurrency"], r["trial"]["repeat"]): r
        for r in reports
    }
    for numerator, denominator in study.get("cost_ratios", []):
        conditions = sorted(
            {
                (t["cache"], t["concurrency"])
                for t in plan["trials"]
                if t["arm"] in (numerator, denominator)
            }
        )
        for cache, concurrency in conditions:
            repeats = sorted(
                {
                    t["repeat"]
                    for t in plan["trials"]
                    if t["cache"] == cache
                    and t["concurrency"] == concurrency
                    and t["arm"] in (numerator, denominator)
                }
            )
            pairs = []
            for repeat in repeats:
                pair = [
                    available.get((a, cache, concurrency, repeat)) for a in (numerator, denominator)
                ]
                if any(r is None for r in pair):
                    continue
                if pair[0]["trial"]["seed"] != pair[1]["trial"]["seed"]:
                    raise ValueError("A/P comparison needs paired request-order seeds")
                for report in pair:
                    dedicated_gpu_seconds(report)
                pairs.append(
                    {
                        "repeat": repeat,
                        **{
                            role: {r["id"]: r.get("gpu_seconds") for r in report.get("results", [])}
                            for role, report in zip(("A", "P"), pair, strict=True)
                        },
                    }
                )
            result = {
                "numerator": numerator,
                "denominator": denominator,
                "cache": cache,
                "concurrency": concurrency,
                "primary_cell": concurrency == 1
                and (
                    cache == study["cost_primary_cache"]
                    if "cost_primary_cache" in study
                    else cache in ("cold", "warm")
                    if study.get("cost_weights")
                    else cache == "off"
                ),
                "status": "incomplete",
                "low": None,
                "high": None,
                "primary_pass": None,
                "savings_pass": None,
                "half_pass": None,
            }
            if len(pairs) == len(repeats) and pairs:
                if study.get("cost_weights"):
                    result.update(
                        weighted_dedicated_ratio(
                            pairs,
                            study["cost_weights"],
                            clusters={q: labels[0] for q, labels in study["cost_clusters"].items()},
                            strata=study["cost_strata"],
                            seed=plan["seed"],
                            resamples=study.get("cost_resamples", 100_000),
                        ),
                        status="completed",
                    )
                    result.update(
                        scope="serial population cost"
                        if concurrency == 1
                        else "batched population allocation sensitivity",
                        physical_workload_ratio=weighted_dedicated_ratio(
                            pairs,
                            dict.fromkeys(study["cost_weights"], 1),
                            seed=plan["seed"],
                            resamples=study.get("cost_resamples", 100_000),
                            resample_questions=False,
                        ),
                    )
                    whole = result["physical_workload_ratio"]
                    result["whole_trial_ratio"] = whole
                    if len(pairs) >= 3 and whole["high"] is not None:
                        result["savings_pass"] = whole["high"] < 1
                        result["half_pass"] = whole["high"] <= 0.5
                    if not result["primary_cell"]:
                        result["primary_pass"] = None
                else:
                    result.update(
                        dedicated_ratio(
                            pairs,
                            study["cost_clusters"],
                            seed=plan["seed"],
                            resamples=study.get("cost_resamples", 100_000),
                        ),
                        status="completed",
                    )
                    if not result["primary_cell"]:
                        result["sensitivity_bound_exceeds_threshold"] = result["primary_pass"]
                        result["primary_pass"] = None
            results.append(result)
    return results


def export_costs(report: dict, out: Path) -> None:
    """Export cell means for calibration accounting, never as timing replicates."""
    for cell in report["cells"]:
        if cell["status"] != "completed" or not cell["question_costs"]:
            continue
        name = cell["arm"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"arm cannot be used as an export filename: {name!r}")
        path = out / f"costs-{name}-{cell['cache']}-c{cell['concurrency']}.jsonl"
        rows = [
            {
                **r,
                "arm": name,
                "rounds": int(r["model_calls"]) if r["model_calls"] is not None else None,
            }
            for r in cell["question_costs"]
        ]
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        summary = {
            "artifact_kind": "xgraph.controlled_question_costs",
            "version": 1,
            "cost_kind": "controlled_serving",
            "cache": cell["cache"],
            "concurrency": cell["concurrency"],
            "questions": len(rows),
            "seconds": cell["metrics"]["serving_seconds"]["mean"] * len(rows),
            "scope": "per-question means; use paired trial reports for timing inference",
            "trials": cell["trial_outputs"],
            "provenance": provenance([Path(p) for p in cell["trial_outputs"]]),
        }
        write_json(Path(str(path) + ".summary.json"), summary)
        write_json(Path(str(path) + ".provenance.json"), summary["provenance"])


def add_arguments(parser) -> None:
    """The built-in runner's frozen configuration; the external-runner API stays usable."""
    parser.add_argument("--model", type=Path, help="use the built-in vLLM/Docker runner")
    parser.add_argument(
        "--mintqa-panels", type=Path, help="live policies on frozen development members"
    )
    parser.add_argument(
        "--policy-freeze", type=Path, help="sealed validation policy for --arm finalist"
    )
    parser.add_argument("--policy-family", choices=("qwen38", "gemma31", "qwen25"))
    parser.add_argument("--timing-questions", type=int, default=128)
    parser.add_argument(
        "--timing-selection",
        choices=("population-matched", "id-hash"),
        default="population-matched",
    )
    parser.add_argument(
        "--descriptive", action="store_true", help="W/B2 timing without agent ratios"
    )
    parser.add_argument(
        "--quality-split", action="append", default=[], help="post-timing gold as split=file"
    )
    parser.add_argument("--amortize-questions", type=int, nargs="+", default=[128, 10000, 1000000])
    parser.add_argument("--panel", type=Path, default=Path("target/mintqa/panel-1024.json"))
    parser.add_argument(
        "--paths-requests", type=Path, default=Path("target/mintqa/requests-paths.jsonl")
    )
    parser.add_argument(
        "--retrieval-requests",
        type=Path,
        default=Path("target/mintqa/requests-rag-linked-matched.jsonl"),
    )
    parser.add_argument("--kg-index", type=Path, default=os.environ.get("XG_KG_INDEX"))
    parser.add_argument("--tokenizer", type=Path, default=os.environ.get("XG_TOKENIZER"))
    parser.add_argument("--kg-versions", type=Path)
    parser.add_argument("--tools-db", type=Path)
    parser.add_argument("--relation-labels", type=Path)
    parser.add_argument("--arm-config", type=Path, help="named confirmation arm recipes (JSON)")
    parser.add_argument(
        "--cost-ratio", action="append", default=[], help="frozen numerator:denominator names"
    )
    parser.add_argument(
        "--cost-clusters", type=Path, help="question id to [sequence,event] or [event] (JSON)"
    )
    parser.add_argument("--cost-resamples", type=int, default=100_000)
    parser.add_argument(
        "--reader", action="append", default=[], help="agent|iterative|cor=records.jsonl"
    )
    parser.add_argument(
        "--reader-config", type=Path, help="per-reader requests and reconstruction options"
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--ban-tokens", nargs="+", default=[], help="exact token strings; bias -100"
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--warmup-questions", type=int, default=32)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--sample-seconds", type=float, default=1)
    parser.add_argument("--traceback-seconds", type=float, default=300)
    parser.add_argument("--heartbeat-requests", type=int, default=8)
    parser.add_argument("--need-gib", type=int, default=int(os.environ.get("XG_NEED_GIB", "55")))
    parser.add_argument(
        "--wheel", type=Path, help="use an already GPU-built wheel instead of uv build"
    )
    parser.add_argument("--engine-config", type=Path, help="JSON overrides for XGRAPH_* settings")
    parser.add_argument(
        "--sampling-config", type=Path, help="frozen JSON sampling settings for every arm"
    )


def panel_records(path: Path, ids: list[str]) -> list[dict]:
    """Require the complete frozen panel; never silently time a different subset."""
    rows = {r["id"]: r for r in read_records(path)}
    missing = set(ids) - rows.keys()
    if missing:
        raise ValueError(f"{path}: missing {len(missing)} panel ids, e.g. {sorted(missing)[:3]}")
    return [rows[q] for q in ids]


def freeze_study(args) -> dict:
    """Materialize paired workloads and provenance before the first GPU action."""
    freeze_started = time.perf_counter()
    if args.arm_config and (args.reader or args.reader_config or args.engine_config):
        raise ValueError("--arm-config cannot mix legacy readers or engine support overrides")
    if args.kg_index is None:
        raise ValueError("the built-in runner needs --kg-index or XG_KG_INDEX")
    if not args.model.is_dir() or not args.kg_index.is_dir():
        raise ValueError("--model and --kg-index must be existing directories")
    if (
        args.max_tokens <= 0
        or args.max_model_len <= args.max_tokens
        or args.max_num_seqs < max(args.concurrency)
        or args.warmup_questions <= 0
        or any(n <= 0 for n in args.amortize_questions)
        or min(
            args.startup_timeout,
            args.request_timeout,
            args.sample_seconds,
            args.traceback_seconds,
            args.heartbeat_requests,
            args.need_gib,
        )
        <= 0
    ):
        raise ValueError("invalid token, sequence, warm-up, timeout or memory budget")
    timing = None
    if args.mintqa_panels:
        from xgraph.serve_mintqa import timing_panel

        manifest = json.loads(args.mintqa_panels.read_text())
        matches = [
            item
            for item in manifest.get("outputs", [])
            if Path(item["path"]).resolve() == args.paths_requests.resolve()
        ]
        if matches and sha256_file(args.paths_requests) != matches[0]["sha256"]:
            raise ValueError("frozen development requests differ from panels.json")
        if args.run and not matches:
            raise ValueError("live execution needs development request provenance in panels.json")
        timing = timing_panel(
            manifest["panels"]["development"]["members"],
            args.timing_questions,
            args.seed,
            selection=args.timing_selection,
        )
        remaining = sorted(
            {r["id"] for r in manifest["panels"]["development"]["members"]} - set(timing["ids"])
        )
        if len(remaining) < args.warmup_questions:
            raise ValueError("live MINTQA needs enough development questions for disjoint warm-up")
        if args.timing_selection == "id-hash":
            timing["warmup_ids"] = sorted(sorted(
                remaining, key=lambda q: (hashlib.sha256(f"{args.seed}:{q}".encode()).digest(), q)
            )[:args.warmup_questions])  # fmt: skip
        else:
            timing["warmup_ids"] = sorted(random.Random(args.seed + 1).sample(
                remaining, args.warmup_questions))  # fmt: skip
        panel = timing
    else:
        panel = json.loads(args.panel.read_text())
    ids = panel.get("ids") if isinstance(panel, dict) else panel
    if (
        not isinstance(ids, list)
        or not ids
        or any(not isinstance(q, str) for q in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("--panel needs a nonempty unique list of string ids (or {ids: [...]})")
    inputs = [args.mintqa_panels or args.panel, args.paths_requests]
    quality_answers = {}
    for spec in args.quality_split:
        split, sep, filename = spec.partition("=")
        if not sep or split not in ("ti", "pop") or not timing:
            raise ValueError("--quality-split needs live MINTQA and ti=file or pop=file")
        path = Path(filename)
        inputs.append(path)
        with path.open() as lines:
            for i, line in enumerate(lines):
                qid = f"mintqa-{split}-{i}"
                if qid in ids:
                    if qid in quality_answers:
                        raise ValueError(f"duplicate quality question: {qid}")
                    quality_answers[qid] = json.loads(line)["answer"]
    if args.quality_split and set(quality_answers) != set(ids):
        raise ValueError("quality sources must cover every selected development question")
    if not args.arm_config and not timing:
        inputs.append(args.retrieval_requests)
    tokenizer = args.tokenizer or args.model / "tokenizer.json"
    inputs.extend(
        p
        for folder in (args.model, args.kg_index)
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    )
    inputs.append(tokenizer)
    header = json.loads((args.kg_index / "kg-index.json").read_text())
    opener = SpanSyntax(**header.get("span_syntax", {})).opener
    paths = panel_records(args.paths_requests, ids)
    sampling = json.loads(args.sampling_config.read_text()) if args.sampling_config else {}
    if not isinstance(sampling, dict) or sampling.keys() - SAMPLING_FIELDS:
        raise ValueError("--sampling-config contains unsupported sampling fields")
    if args.sampling_config:
        inputs.append(args.sampling_config)
    if args.ban_tokens:
        from tokenizers import Tokenizer

        served = Tokenizer.from_file(str(tokenizer))
        bias = dict(sampling.get("logit_bias", {}))
        for token in args.ban_tokens:
            token_id = served.token_to_id(token)
            if token_id is None:
                raise ValueError(f"--ban-tokens: {token!r} is not a token of this model")
            bias[str(token_id)] = -100.0
        sampling["logit_bias"] = bias
    settings = {
        "XGRAPH_PATHS": "1",
        "XGRAPH_COMMITTED_PATHS": "1",
        "XGRAPH_ESCAPES": "0",
        "XGRAPH_UNQUOTED_STARTS": "1",
        "XGRAPH_START_RANK": "degree",
    }
    if args.engine_config:
        overrides = json.loads(args.engine_config.read_text())
        if not isinstance(overrides, dict):
            raise ValueError("--engine-config needs a JSON object")
        allowed = {
            "XGRAPH_PATHS",
            "XGRAPH_COMMITTED_PATHS",
            "XGRAPH_ESCAPES",
            "XGRAPH_UNQUOTED_STARTS",
            "XGRAPH_START_RANK",
            "XGRAPH_BOUND_ESCAPES",
            "XGRAPH_REPEATS",
            "XGRAPH_DERIVATION",
            "XGRAPH_ESCAPE_BIAS",
            "XGRAPH_RELEASE_MASS",
            "XGRAPH_RELEASE_TOP",
            "XGRAPH_RELATIONS",
            "XGRAPH_STOP",
            "XGRAPH_END_AFTER_ANSWER",
            "XGRAPH_EXPRESSIONS",
        }
        if any(
            k not in allowed
            or k in ("XGRAPH_KG_INDEX", "XGRAPH_TOKENIZER", "XGRAPH_KG_VERSIONS")
            or not isinstance(v, str)
            for k, v in overrides.items()
        ):
            raise ValueError("--engine-config needs supported XGRAPH_* string values")
        settings.update(overrides)
        inputs.append(args.engine_config)
    arms = (
        {}
        if args.arm_config or timing
        else {
            "paths": {
                "requests": paths,
                "opt_in": True,
                "opener": opener,
                "max_tokens": args.max_tokens,
            },
            "unconstrained": {
                "requests": paths,
                "opt_in": False,
                "opener": opener,
                "max_tokens": args.max_tokens,
            },
            "retrieval": {
                "requests": panel_records(args.retrieval_requests, ids),
                "opt_in": False,
                "opener": None,
                "max_tokens": args.max_tokens,
            },
        }
    )
    support_audit = None
    if timing:
        from xgraph.serve_mintqa import freeze_arms as freeze_mintqa

        arms = freeze_mintqa(args, timing, tokenizer, header, inputs)
        support_audit = {"policy": "native live rematch; exact serving commands retained"}
    elif args.arm_config:
        arms, support_audit = freeze_arms(args, ids, tokenizer, header, inputs)
    availability = {
        kind: {"status": "unavailable", "reason": "no --reader records supplied"}
        for kind in ("agent", "iterative", "cor")
    }
    if args.arm_config or timing:
        availability = {
            name: {
                "status": "available",
                "kind": arm["kind"],
                "trajectory_kind": "live_mintqa_policy" if timing else "frozen_confirmation_replay",
            }
            for name, arm in arms.items()
        }
    for optional in (args.kg_versions, args.tools_db, args.reader_config, args.wheel):
        if optional:
            inputs.extend(
                sorted(p for p in optional.rglob("*") if p.is_file())
                if optional.is_dir()
                else [optional]
            )
    configs = json.loads(args.reader_config.read_text()) if args.reader_config else {}
    if not isinstance(configs, dict) or any(not isinstance(v, dict) for v in configs.values()):
        raise ValueError("--reader-config needs an object of per-reader configuration objects")
    for spec in args.reader:
        kind, sep, filename = spec.partition("=")
        if not sep or kind not in availability or availability[kind]["status"] != "unavailable":
            raise ValueError(
                f"--reader needs a unique agent|iterative|cor=records.jsonl, got {spec!r}"
            )
        if not args.tools_db or kind not in configs or "requests" not in configs[kind]:
            raise ValueError(
                f"--reader {kind} needs --tools-db and --reader-config "
                "with its requests and options"
            )
        config = configs[kind].copy()
        allowed_options = {
            "agent": {
                "requests",
                "max_tokens",
                "max_calls",
                "pages",
                "matched_demos",
                "matched_paths",
                "feedback",
                "relation_labels",
                "start_rank",
                "unquoted_starts",
            },
            "cor": {
                "requests",
                "max_tokens",
                "max_calls",
                "max_depth",
                "max_prompt_tokens",
                "relation_labels",
                "cor_prompts",
            },
            "iterative": {
                "requests",
                "max_tokens",
                "max_lookups",
                "cap",
                "max_prompt_tokens",
                "written_start",
                "start_max_tokens",
            },
        }
        if unknown := config.keys() - allowed_options[kind]:
            raise ValueError(f"unknown {kind} reader options: {sorted(unknown)}")
        path = Path(filename).resolve()
        source_requests = Path(config.pop("requests")).resolve()
        records = panel_records(path, ids)
        asked = panel_records(source_requests, ids)
        if any(
            a.get("kg_version") != r.get("kg_version") for a, r in zip(asked, records, strict=True)
        ):
            raise ValueError(f"{path}: reader records and requests select different KG versions")
        inputs.extend([path, source_requests])
        for suffix in (".summary.json", ".provenance.json"):
            if Path(str(path) + suffix).exists():
                inputs.append(Path(str(path) + suffix))
        if config.get("relation_labels"):
            config["relation_labels"] = str(Path(config["relation_labels"]).resolve())
            inputs.append(Path(config["relation_labels"]))
        if config.get("cor_prompts"):
            config["cor_prompts"] = str(Path(config["cor_prompts"]).resolve())
            inputs.append(Path(config["cor_prompts"]))
        for r in records:
            turns = r.get("turns") if kind != "iterative" else r.get("turns", r.get("replies"))
            if (
                not isinstance(turns, list)
                or (not turns and model_calls(r) != 0)
                or model_calls(r) is None
                or r.get("prompt_tokens") is None
            ):
                raise ValueError(
                    f"{path}: {r['id']} lacks full {kind} replies/usage; "
                    "actions alone are insufficient"
                )
            if r.get("arm", kind) != kind:
                raise ValueError(f"{path}: {r['id']} is not a {kind} record")
            if (
                kind == "iterative"
                and config.get("max_prompt_tokens") is not None
                and (
                    "turns" not in r
                    or any(type(t.get("generated_tokens")) is not int for t in r["turns"])
                )
            ):
                raise ValueError(f"{path}: iterative clipping needs original per-turn token usage")
        if kind != "iterative":
            from xgraph.generate import walk_requests

            converted = walk_requests(asked) if all("messages" in r for r in asked) else asked
            asked = [
                {**original, **converted_row}
                for original, converted_row in zip(asked, converted, strict=True)
            ]
        arms[kind] = {
            "kind": kind,
            "requests": [{**a, "record": r} for a, r in zip(asked, records, strict=True)],
            "max_tokens": config.pop("max_tokens", args.max_tokens),
            "opt_in": False,
            "opener": None,
            "options": config,
            "model": str(args.model.resolve()),
            "tools_db": str(args.tools_db.resolve()),
            "kg_index": str(args.kg_index.resolve()),
            "tokenizer": str(tokenizer.resolve()),
            "kg_versions": str(args.kg_versions.resolve()) if args.kg_versions else None,
        }
        availability[kind] = {
            "status": "available",
            "records": str(path),
            "trajectory_kind": "frozen_reader_replay",
        }
    for name, arm in arms.items():
        if type(arm["max_tokens"]) is not int or not 0 < arm["max_tokens"] < args.max_model_len:
            raise ValueError(f"{name}: max_tokens must be positive and below max_model_len")
        for key in ("requests", "warmup_requests"):
            if key in arm:
                arm[key] = [{**r, "sampling": {**r.get("sampling", {}), **sampling}}
                            for r in arm[key]]  # fmt: skip
        if arm.get("confirmation") and any(
            r["sampling"] != arm["expected_sampling"] for r in arm["requests"]
        ):
            raise ValueError(f"{name}: sampling overrides differ from frozen calibration sources")
    # Live MINTQA has already resolved and deduplicated the frozen nomination.
    selected = list(arms) if timing else args.arm or list(arms)
    if set(selected) - arms.keys():
        raise ValueError(f"unknown or unavailable arms: {sorted(set(selected) - arms.keys())}")
    ratios = []
    clusters = None
    if args.cost_ratio:
        if not args.cost_clusters or args.cost_resamples <= 0:
            raise ValueError("--cost-ratio needs --cost-clusters and positive --cost-resamples")
        clusters = json.loads(args.cost_clusters.read_text())
        if set(clusters) != set(ids) or any(
            not isinstance(c, list)
            or not 1 <= len(c) <= 2
            or any(not isinstance(v, str) or not v for v in c)
            for c in clusters.values()
        ):
            raise ValueError(
                "cost clusters need one or two nonempty labels for every panel question"
            )
        inputs.append(args.cost_clusters)
        for spec in args.cost_ratio:
            a, sep, p = spec.partition(":")
            if not sep or a == p or a not in selected or p not in selected or [a, p] in ratios:
                raise ValueError(f"invalid or duplicate --cost-ratio {spec!r}")
            ratios.append([a, p])
    if timing:
        reference = next(iter(arms.values())).get("cost_reference", "agent-strong-8")
        ratios = [] if args.descriptive else [[a, reference] for a in selected if a != reference]
        clusters = {r["id"]: [r["cluster"]] for r in timing["members"]}
    from xgraph.generate import model_revision

    inputs = list(dict.fromkeys(inputs))

    def identities():
        result = {}
        for path in inputs:
            stat = path.stat()
            result[str(path)] = [
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            ]
        return result

    identity = identities() if timing else {}
    inputs_provenance = provenance(inputs)
    if timing:
        if identities() != identity:
            raise ValueError("frozen MINTQA inputs changed while hashing")
        index_manifest = args.kg_index.with_suffix(".manifest.json")
        if index_manifest.is_file():
            saved = json.loads(index_manifest.read_text())
            hashes = {
                str(Path(r["path"]).resolve()): r["sha256"] for r in inputs_provenance["inputs"]
            }
            if saved["settings"].get("unique_aliases") is not True or any(
                hashes.get(str((args.kg_index / row["name"]).resolve())) != row["sha256"]
                for row in saved["files"]
            ):
                raise ValueError(
                    "MINTQA rematch index differs from its frozen unique-alias manifest"
                )
            support_audit["index_manifest_verified"] = True
        elif args.run:
            raise ValueError("MINTQA execution needs the frozen rematch index manifest")
        else:
            support_audit["index_manifest_verified"] = False
    return {
        "artifact_kind": "xgraph.serving_study",
        "version": 1,
        "panel_ids": ids,
        "amortize_questions": args.amortize_questions,
        "timing_panel": timing,
        "descriptive": args.descriptive,
        "quality_answers": quality_answers,
        **({"cost_primary_cache": "warm"} if timing else {}),
        "cost_weights": {r["id"]: r["weight"] for r in timing["members"]} if timing else None,
        "cost_strata": {r["id"]: str(r["hops"]) for r in timing["members"]} if timing else None,
        "arms": {name: arms[name] for name in selected},
        "reader_availability": availability,
        "model": str(args.model.resolve()),
        "model_revision": model_revision(args.model),
        "tokenizer": str(tokenizer.resolve()),
        "tokenizer_fingerprint": header.get("tokenizer_fingerprint"),
        "banned_tokens": args.ban_tokens,
        "kg_index": str(args.kg_index.resolve()),
        "kg_versions": str(args.kg_versions.resolve()) if args.kg_versions else None,
        "engine_settings": settings,
        "support_audit": support_audit,
        "cost_ratios": ratios,
        "cost_clusters": clusters,
        "cost_resamples": args.cost_resamples,
        "image": IMAGE,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "warmup_questions": args.warmup_questions,
        "port": args.port,
        "startup_timeout": args.startup_timeout,
        "request_timeout": args.request_timeout,
        "sample_seconds": args.sample_seconds,
        "traceback_seconds": args.traceback_seconds,
        "heartbeat_requests": args.heartbeat_requests,
        "trial_watchdog": {
            "forecast": "four times high reservation forecast per trial",
            "minimum_seconds": 1800,
            "startup_allowance_seconds": args.startup_timeout + 120,
            "termination_grace_seconds": 10,
        },
        "need_gib": args.need_gib,
        "gpu_count": 1,
        "filesystem_cache_state": "uncontrolled; no OS page-cache eviction",
        "kg_store_cache_state": (
            "KG tables loaded before timing; shared KGStore/version resources per cell; "
            "iterative Facts helper caches per question; no explicit eviction"
        ),
        "preparation": {
            "freeze_and_hash_seconds": time.perf_counter() - freeze_started,
            "index_build_seconds": None,
            "store_build_seconds": None,
            "scope": "existing index/store artifacts; their construction is outside this study",
        },
        "provenance": inputs_provenance,
        "input_identity": identity,
    }
