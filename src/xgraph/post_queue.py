"""Seal and launch the GPU owner's post-validation serving and transfer queue."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import time
from contextlib import suppress
from datetime import datetime
from pathlib import Path

from xgraph.gpu_idle import executing_pids
from xgraph.provenance import provenance, sha256_file

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("/home/jonas/git/xgraph")
FINAL = SOURCE / ".claude/worktrees/codex-final"
WEBQSP = SOURCE / ".claude/worktrees/codex-webqsp2/target/webqsp/frozen-data/validation"
FAMILIES = ("qwen38", "gemma31")
TRANSFER_ARMS = ("W", "WT", "B2", "agent-strong-8", "agent-feedback-8", "dog", "gcr-native")
# Keep admission and retry policy explicit in execution and the saved preview.
# A failed attempt is retained for accounting before the owner relaunches.
OWNER_ENV = {"XG_GPU_IDLE": "1", "XG_AVAILABLE_WAIT": "43200", "XG_START_ATTEMPTS": "1"}


def queue_stages(out, policy_freeze, webqsp_data, source_root, *, reduced=False):
    out, policy_freeze = Path(out), Path(policy_freeze)
    stages = []
    for plan in ("locked", "descriptive") if reduced else ("screen", "locked", "cold"):
        for family in FAMILIES:
            name = f"{plan}-{family}"
            destination = out / "serving" / name
            command = [
                "bash",
                str(ROOT / "scripts/mintqa/serve-cost.sh"),
                family,
                str(destination),
                "--run",
                "--plan",
                plan,
                "--policy-freeze",
                str(policy_freeze),
                "--source-root",
                str(source_root),
            ]
            if plan == "screen":
                command += ["--two-call", "WT"]
            stages.append(
                {
                    "name": name,
                    "kind": "serving",
                    "command": command,
                    "output": str(destination / "report.json"),
                }
            )
    for family in FAMILIES:
        destination = out / "overhead" / family
        stages.append(
            {
                "name": f"overhead-{family}",
                "kind": "overhead",
                "command": [
                    str(ROOT / ".venv/bin/python"),
                    "-m",
                    "xgraph.post_queue",
                    "overhead",
                    "--screen",
                    str(out / "serving" / f"{'locked' if reduced else 'screen'}-{family}"),
                    "--out",
                    str(destination / "engine.json"),
                ],
                "output": str(destination / "engine.json"),
            }
        )
    if reduced:
        return stages
    destination = out / "webqsp-pilot"
    stages.append(
        {
            "name": "webqsp-pilot",
            "kind": "webqsp",
            "command": [
                "bash",
                str(ROOT / "scripts/webqsp-queue.sh"),
                "--catalog",
                "--pilot",
                "--family",
                "both",
                "--data",
                str(webqsp_data),
                "--out",
                str(destination),
                "--policy-freeze",
                str(policy_freeze),
                "--pilot-questions",
                "100",
                "--pilot-seed",
                "17",
                "--arms",
                *TRANSFER_ARMS,
                "--pilot-budget-hours",
                "16",
                "--run",
            ],
            "output": str(destination / "timings.json"),
        }
    )
    return stages


def idle_snapshot(validation_log, *, proc_root=Path("/proc")):
    """Fail closed on an unreadable marker, Docker query or process inventory."""
    try:
        if "validationq exit" not in Path(validation_log).read_text():
            return False
        result = subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True, timeout=30)
        if result.returncode or result.stdout.strip():
            return False
        return not executing_pids(SOURCE, proc_root=proc_root)
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        return False


def wait_for_idle(validation_log, *, quiet_seconds=600, poll_seconds=30):
    """All three conditions must hold continuously for the complete quiet interval."""
    quiet_since = None
    while True:
        ready, now = idle_snapshot(validation_log), time.monotonic()
        if ready:
            if quiet_since is None:
                quiet_since = now
            if now - quiet_since >= quiet_seconds:
                return
        else:
            quiet_since = None
        print(
            "post waiting: validationq exit + no containers + no headroom waiter; "
            f"quiet={0 if quiet_since is None else int(now - quiet_since)}s/{quiet_seconds}s",
            flush=True,
        )
        time.sleep(min(poll_seconds, 30))


def run_stage(command, *, wall_seconds, **kwargs):
    """Bound preparation, overhead replay and transfer as well as serving cells."""
    process = subprocess.Popen(command, start_new_session=True, **kwargs)
    try:
        return subprocess.CompletedProcess(command, process.wait(timeout=wall_seconds))
    except subprocess.TimeoutExpired:
        print(
            f"post stage wall bound exceeded ({wall_seconds}s)", file=kwargs["stdout"], flush=True
        )
        with suppress(ProcessLookupError):
            process.terminate()  # a healthy supervisor cleans up its worker/server
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=150)
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        return subprocess.CompletedProcess(command, 124)


def select_stages(stages, names=None):
    """Select complete registered stages without rewriting any stage argv."""
    if names is None:
        return stages
    unknown = set(names) - {stage["name"] for stage in stages}
    if not names or unknown or len(names) != len(set(names)):
        raise ValueError(f"invalid or duplicate post stages: {names}; unknown: {sorted(unknown)}")
    return [stage for stage in stages if stage["name"] in names]


def run_queue(seal, *, dry_run=False, stage_names=None):
    stages = select_stages(seal["stages"], stage_names)
    if dry_run:
        print(
            "# Wait 600 continuous seconds: validation.log has 'validationq exit', "
            "docker ps -q is empty, and no repository/worktree scripts/gpu-headroom.sh executes."
        )
        for stage in stages:
            print(f"# stage wall bound seconds: {stage.get('wall_bound_seconds', 86400)}")
            print(f"# post {stage['name']}")
            print(
                shlex.join(["env", *(f"{k}={v}" for k, v in OWNER_ENV.items()), *stage["command"]])
            )
        return
    out = Path(seal["out"])
    out.mkdir(parents=True, exist_ok=True)
    with (out / "post.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget_env = {}
        budget_path = out / "serving-budget.json"
        if "lifetime_budget_seconds" in seal:
            from xgraph.serve_mintqa import SERVING_CAP_SECONDS, ServingBudget

            if seal["lifetime_budget_seconds"] != SERVING_CAP_SECONDS:
                raise ValueError("serving lifetime cap changed")
            if not budget_path.exists() and (
                (out / "receipts").exists()
                or any(
                    Path(s["output"]).parent.exists() or (out / f"{s['name']}.log").exists()
                    for s in seal["stages"]
                )
            ):
                raise ValueError("serving budget missing beside prior execution artifacts")
            budget = ServingBudget(budget_path, seal["seal_sha256"])
            if budget.data["status"] == "INCOMPLETE":
                raise RuntimeError("serving study INCOMPLETE; no top-up or resume")
            if any(a["status"] == "running" for a in budget.data["attempts"]):
                raise RuntimeError("unclosed serving budget attempt: INCOMPLETE")
            budget.write()
            budget_env = {
                "XG_SERVING_BUDGET": str(budget_path),
                "XG_SERVING_SEAL_SHA256": seal["seal_sha256"],
            }
        wait_for_idle(Path(seal["validation_log"]))
        for stage in stages:
            code = 1
            try:
                if stage_complete(stage, seal):
                    print(f"post skip {stage['name']}: verified existing outputs", flush=True)
                    code = 0
                    continue
                attempt = None
                if budget_env and stage["kind"] == "overhead":
                    budget = ServingBudget(budget_path, seal["seal_sha256"])
                    attempt = budget.begin(stage["name"], stage["wall_bound_seconds"])
                result = None
                try:
                    with (out / f"{stage['name']}.log").open("a") as log:
                        result = run_stage(
                            stage["command"],
                            wall_seconds=(
                                attempt["wall_seconds"]
                                if attempt
                                else stage.get("wall_bound_seconds", 86400)
                            ),
                            cwd=ROOT,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            env={
                                **os.environ,
                                **OWNER_ENV,
                                **budget_env,
                                "UV_CACHE_DIR": seal.get("uv_cache", str(ROOT / ".uv-cache")),
                            },
                        )
                finally:
                    if attempt:
                        budget.finish(
                            attempt,
                            cleanup_confirmed=result is not None,
                            failed=result is None or bool(result.returncode),
                        )
                code = result.returncode
                if code:
                    raise SystemExit(code)
                code = 1
                if not stage_complete(stage, seal):
                    raise ValueError(
                        f"{stage['name']}: no complete output after successful command"
                    )
                code = 0
            finally:
                path = Path(seal["post_log"])
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a") as log:
                    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
                    log.write(f"post {stage['name']} exit {code} {timestamp}\n")


def stage_complete(stage, seal):
    """Reuse only complete, recipe-matched, unaltered outputs; never overwrite partials."""
    output = Path(stage["output"])
    receipt = Path(seal["out"]) / "receipts" / f"{stage['name']}.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["seal_sha256"] != seal["seal_sha256"] or saved["command"] != stage["command"]:
            raise ValueError(f"{stage['name']}: existing receipt changed seal or command")
        for path, digest in saved["files"].items():
            if not Path(path).is_file() or sha256_file(Path(path)) != digest:
                raise ValueError(f"{stage['name']}: completed output changed: {path}")
        return True
    if not output.exists():
        if output.parent.exists():
            raise FileExistsError(f"{stage['name']}: partial output directory {output.parent}")
        return False
    result = json.loads(output.read_text())
    if stage["kind"] == "serving":
        trials = json.loads((output.parent / "trials.json").read_text())
        study = json.loads((output.parent / "study.json").read_text())
        expected = json.loads(Path(stage["plan"]).read_text())
        planned_trials = json.loads(Path(stage["plan"]).with_name("trials.json").read_text())
        keys = ("arm", "cache", "concurrency", "repeat", "seed")
        if [tuple(t[k] for k in keys) for t in trials["trials"]] != [
            tuple(t[k] for k in keys) for t in planned_trials["trials"]
        ]:
            raise ValueError(f"{stage['name']}: serving trial matrix changed or incomplete")
        if (
            result.get("status") != "completed"
            or trials.get("status") != "completed"
            or any(t["status"] != "completed" for t in trials["trials"])
        ):
            raise ValueError(f"{stage['name']}: incomplete serving trial")
        for key, value in expected.items():
            if key in ("provenance", "preparation", "input_identity"):
                continue  # output path/time and filesystem metadata are recorded anew
            if study.get(key) != value:
                raise ValueError(f"{stage['name']}: serving plan changed: {key}")
        for trial in trials["trials"]:
            cell = json.loads(Path(trial["output"]).read_text())
            if cell["status"] != "completed":
                raise ValueError(f"{stage['name']}: incomplete retained serving cell")
    elif stage["kind"] == "webqsp":
        if result.get("complete") is not True:
            raise ValueError(f"{stage['name']}: incomplete WebQSP timing output")
        if stage.get("plan"):
            expected = json.loads(Path(stage["plan"]).read_text())
            actual = json.loads((output.parent / "queue.json").read_text())
            for key in (
                "arms",
                "catalog_signature",
                "frozen_policies",
                "pilot",
                "questions",
                "pilot_restart",
            ):
                if actual.get(key) != expected.get(key):
                    raise ValueError(f"{stage['name']}: WebQSP plan changed: {key}")
            if (
                result.get("catalog_signature") != expected["catalog_signature"]
                or [(j["family"], j["name"], j["kind"]) for j in result["jobs"]]
                != [(j["family"], j["name"], j["kind"]) for j in expected["jobs"]]
                or any(j["returncode"] for j in result["jobs"])
            ):
                raise ValueError(f"{stage['name']}: incomplete or changed WebQSP timed stages")
    else:
        cells = {(r["stage"], r["batch"], r["mask_control"]) for r in result.get("results", [])}
        expected = {
            (s, b, c) for s in ("first", "second") for b in (1, 8) for c in ("engine", "noop")
        }
        if (
            result.get("profile") != "WT"
            or cells != expected
            or any(r["errors"] for r in result["results"])
        ):
            raise ValueError(f"{stage['name']}: incomplete WT engine timing output")
    files = {
        str(p): sha256_file(p)
        for p in sorted(output.parent.rglob("*"))
        if p.is_file()
        and p.suffix in (".json", ".jsonl")
        and "runtime" not in p.relative_to(output.parent).parts
    }
    # Reused prices retain their original event paths outside the new output.
    # Bind those metadata files in the completion receipt as well.
    for row in result.get("per_arm", []):
        if "reused_receipt" in row:
            for item in (row["reused_receipt"], row["gpu_events"]):
                path = Path(item["path"])
                if sha256_file(path) != item["sha256"]:
                    raise ValueError(f"pilot reused metadata changed: {path}")
                files[str(path)] = item["sha256"]
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(
        json.dumps(
            {
                "seal_sha256": seal["seal_sha256"],
                "command": stage["command"],
                "files": files,
            },
            indent=1,
        )
        + "\n"
    )
    return True


def overhead(screen, out):
    """Replay one retained WT trial from the locked or historical screen panel."""
    study = json.loads((screen / "study.json").read_text())
    plan = json.loads((screen / "trials.json").read_text())
    trial = next(
        t
        for t in plan["trials"]
        if t["repeat"] == 0
        and t["cache"] == "warm"
        and t["concurrency"] == 8
        and study["arms"][t["arm"]]["policy"] == "WT"
    )
    model = Path(study["model"])
    config = json.loads((model / "config.json").read_text())
    vocab = config.get("vocab_size", config.get("text_config", {}).get("vocab_size"))
    if not isinstance(vocab, int) or vocab <= 0:
        raise ValueError(f"model config has no logits vocabulary width: {model}")
    subprocess.run(
        ["bash", str(ROOT / "scripts/gpu-headroom.sh"), "16"],
        check=True,
        env={**os.environ, "XG_GPU_IDLE": "1"},
    )
    subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            "-m",
            "xgraph.bench",
            "--wt-trial",
            trial["output"],
            "--index",
            study["kg_index"],
            "--tokenizer",
            study["tokenizer"],
            "--batch",
            "1",
            "8",
            "--repeats",
            "3",
            "--vocab",
            str(vocab),
            "--out",
            str(out),
        ],
        check=True,
    )


def verify_seal(path):
    """Check the immutable consumer seal without accessing Docker or the GPU."""
    from xgraph.mintqa_validation import git_state
    from xgraph.policy_catalog import frozen_recipe

    digest = sha256_file(path)
    if Path(f"{path}.sha256").read_text().strip() != digest:
        raise ValueError("post serving seal changed")
    seal = json.loads(path.read_text())
    if seal["repo"] != str(ROOT) or seal["code_commit"] != git_state():
        raise ValueError("post serving checkout or clean code commit changed")
    current = provenance([])
    if any(current[k] != seal["provenance"][k] for k in ("python", "packages")):
        raise ValueError("post serving Python/package environment changed")
    for filename, expected in seal["files"].items():
        if sha256_file(Path(filename)) != expected:
            raise ValueError(f"post serving frozen input changed: {filename}")
    for family in FAMILIES:
        if frozen_recipe(seal["policy_freeze"], family) != seal["frozen_policies"][family]:
            raise ValueError(f"post serving validation recipe changed: {family}")
    return {**seal, "seal_sha256": digest}


def write_owner_wrapper(path, launcher, *, commit, log):
    """An installable wrapper keeps the sealed launcher in its producer checkout."""
    label = path.stem
    path.write_text(
        "#!/usr/bin/env bash\n# GPU owner installs this file; default preview is CPU only.\n"
        "set -euo pipefail\n"
        "mode=${1:---dry-run}\n"
        'if (($# > 1)) || [[ "$mode" != --dry-run && "$mode" != --run ]]; then\n'
        f"  printf 'usage: {label}.sh [--dry-run|--run]\\n' >&2; exit 2\nfi\n"
        f"cd {shlex.quote(str(ROOT))}\n"
        f"log={shlex.quote(str(log))}\n"
        'if [[ "$mode" == --run ]]; then\n'
        '  mkdir -p "$(dirname "$log")"\n'
        "  trap 'status=$?; trap - EXIT; "
        f'printf "{label} exit %s %s\\n" "$status" "$(date -Is)" >>"$log"; '
        'exit "$status"' + "' EXIT\nfi\n"
        'tree_status="$(git status --porcelain --untracked-files=all)"\n'
        'if [[ -n "$tree_status" ]]; then\n'
        "  printf 'dirty producer tree\\n' >&2; exit 2\nfi\n"
        f'if [[ "$(git rev-parse HEAD)" != {shlex.quote(commit)} ]]; then\n'
        "  printf 'producer commit changed\\n' >&2; exit 2\nfi\n"
        f"if ! printf '%s  %s\\n' {sha256_file(launcher)} {shlex.quote(str(launcher))} "
        "| sha256sum --check --status; then\n"
        "  printf 'launcher hash changed\\n' >&2; exit 2\nfi\n"
        f"export TMPDIR={shlex.quote(str(ROOT / 'target/tmp'))} UV_LINK_MODE=hardlink\n"
        'if [[ "$mode" == --run ]]; then\n'
        '  cuda_compiler="$(command -v nvcc)" || { printf "nvcc is required\\n" >&2; exit 2; }\n'
        "  unset CUDA_VISIBLE_DEVICES SKBUILD_BUILD_DIR\n"
        '  export CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=${cuda_compiler}"\n'
        '  mkdir -p "$TMPDIR"\nfi\n'
        f'nice -n 10 {shlex.quote(str(launcher))} "$mode"\n'
    )
    path.chmod(0o755)


def freeze(args):
    """Materialize CPU plans and their prices before writing the consumer seal."""
    from xgraph.mintqa_validation import git_state
    from xgraph.policy_catalog import frozen_recipe
    from xgraph.serve_mintqa import REDUCED_DESIGN, SERVING_CAP_SECONDS

    commit = git_state()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / "target") or (out / "serving-seal.json").exists():
        raise ValueError("freeze needs an unsealed output under this worktree's target/")
    owner_wrapper = getattr(args, "owner_wrapper", None)
    if owner_wrapper and (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*\.sh", owner_wrapper)
        or owner_wrapper == "post.sh"
    ):
        raise ValueError("owner wrapper must be a simple .sh filename other than post.sh")
    frozen = {f: frozen_recipe(args.policy_freeze, f) for f in FAMILIES}
    if any(v["recipe"]["policy"] != "WT" for v in frozen.values()) or {
        f: v["strongest_agent"] for f, v in frozen.items()
    } != {"qwen38": "agent-strong-8", "gemma31": "agent-feedback-8"}:
        raise ValueError("post queue needs the shared WT nomination and fixed strongest agents")
    amendment = getattr(args, "serving_amendment", None)
    amendment_files = {}
    if amendment:
        amendment = amendment.resolve()
        registered = json.loads(amendment.read_text())
        if (
            Path(f"{amendment}.sha256").read_text().strip() != sha256_file(amendment)
            or registered.get("artifact_kind") != "xgraph.serving_amendment"
            or registered.get("version") != 1
            or registered.get("design") != REDUCED_DESIGN
        ):
            raise ValueError("serving amendment hash or registered design changed")
        if registered.get("execution_out") != str(out):
            raise ValueError("serving amendment execution destination changed; no budget reset")
        amendment_files = registered["files"]
        for filename, digest in amendment_files.items():
            if sha256_file(Path(filename)) != digest:
                raise ValueError(f"serving amendment input changed: {filename}")
    stages = queue_stages(
        out,
        args.policy_freeze.resolve(),
        args.webqsp_data.resolve(),
        args.source_root.resolve(),
        reduced=bool(amendment),
    )
    stages = select_stages(stages, getattr(args, "stage", None))
    if amendment and [s["name"] for s in stages] != [
        f"{p}-{f}" for p in ("locked", "descriptive", "overhead") for f in FAMILIES
    ]:
        raise ValueError("serving amendment requires its intact registered stage matrix")
    restart = getattr(args, "pilot_restart", None)
    if restart:
        if not any(stage["kind"] == "webqsp" for stage in stages):
            raise ValueError("pilot restart requires the WebQSP pilot stage")
        for stage in stages:
            if stage["kind"] == "webqsp":
                stage["command"] += [
                    "--pilot-restart",
                    str(restart.resolve()),
                    "--pilot-restart-sha256",
                    sha256_file(restart),
                ]
    out.mkdir(parents=True, exist_ok=True)
    # Native wheels build offline at owner launch. Reuse the populated cache
    # that installed the locked environment, rather than forcing an empty one.
    cache = str(
        Path(
            os.environ.get(
                "UV_CACHE_DIR",
                Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "uv",
            )
        ).resolve()
    )
    files, prices = dict(amendment_files), {}
    for stage in stages:
        if stage["kind"] == "overhead":
            stage["wall_bound_seconds"] = 1800
            stage["reservation_gpu_seconds"] = 300
            continue
        preview = out / "plans" / stage["name"]
        command = stage["command"].copy()
        if stage["kind"] == "serving":
            command[3:5] = [str(preview), "--dry-run"]
            stage["plan"] = str(preview / "study.json")
        else:
            command[command.index("--out") + 1] = str(preview)
            command[command.index("--run")] = "--dry-run"
            stage["plan"] = str(preview / "queue.json")
        with (out / f"plan-{stage['name']}.log").open("x") as log:
            subprocess.run(
                command,
                check=True,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "UV_CACHE_DIR": cache},
            )
        study = json.loads(Path(stage["plan"]).read_text())
        files.update({r["path"]: r["sha256"] for r in study["provenance"]["inputs"]})
        if stage["kind"] == "serving":
            prices[stage["name"]] = json.loads((preview / "gpu-hours.json").read_text())
            trials = json.loads((preview / "trials.json").read_text())["trials"]
            stage["wall_bound_seconds"] = 1800 + sum(t["wall_bound_seconds"] for t in trials)
        else:
            prices[stage["name"]] = study["estimate"]
            stage["wall_bound_seconds"] = max(
                1800, round(4 * study["estimate"]["high_gpu_hours"] * 3600)
            )
    serving = sum(p["gpu_hours"] for k, p in prices.items() if k != "webqsp-pilot")
    double = sum(p["double_service_gpu_hours"] for k, p in prices.items() if k != "webqsp-pilot")
    web = prices.get("webqsp-pilot", {"low_gpu_hours": 0, "high_gpu_hours": 0})
    replay_hours = sum(s.get("reservation_gpu_seconds", 0) for s in stages) / 3600
    prices["total"] = {
        "measured": False,
        "serving_gpu_hours": serving,
        "double_service_gpu_hours": double,
        "overhead_reservation_gpu_hours": replay_hours,
        "low_gpu_hours": serving + web["low_gpu_hours"] + replay_hours,
        "high_gpu_hours": double + web["high_gpu_hours"] + replay_hours,
        "excluded": "CPU preparation, external waits, validation queue; no monetary rate supplied",
    }
    if amendment and prices["total"]["high_gpu_hours"] * 3600 > SERVING_CAP_SECONDS:
        raise ValueError("registered serving forecast exceeds the 24-hour lifetime cap")
    if web["high_gpu_hours"] > 16:
        raise ValueError(
            "full 100-question transfer forecast exceeds its 16-hour pilot reservation"
        )
    script = out / "post.sh"
    script.write_text(
        "#!/usr/bin/env bash\n# GPU owner launches by absolute path; --dry-run is CPU only.\n"
        "set -euo pipefail\n"
        "if [[ ${BASH_SOURCE[0]} != /* ]]; then\n"
        '  printf "launch by absolute path\\n" >&2; exit 2\nfi\n'
        f"cd {shlex.quote(str(ROOT))}\n"
        "export XG_GPU_IDLE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1\n"
        "mode=${1:---dry-run}\n"
        'if (($# > 1)) || [[ "$mode" != --dry-run && "$mode" != --run ]]; then\n'
        "  printf 'usage: post.sh [--dry-run|--run]\\n' >&2; exit 2\nfi\n"
        f"exec nice -n 10 {shlex.quote(str(ROOT / '.venv/bin/python'))} -m xgraph.post_queue run "
        f'--seal {shlex.quote(str(out / "serving-seal.json"))} "$mode"\n'
    )
    script.chmod(0o755)
    post_log = getattr(args, "post_log", None) or args.source_root / "target/queue/post.log"
    wrapper = out / owner_wrapper if owner_wrapper else None
    if wrapper:
        write_owner_wrapper(wrapper, script, commit=commit, log=post_log)
    pinned = [
        *sorted((ROOT / "src/xgraph").rglob("*.py")),
        *sorted((ROOT / "csrc").glob("*")),
        ROOT / "scripts/mintqa/serve-cost.sh",
        ROOT / "scripts/serve_cost_trials.py",
        ROOT / "scripts/webqsp-queue.sh",
        ROOT / "scripts/generate.sh",
        ROOT / "scripts/gpu-headroom.sh",
        ROOT / "uv.lock",
        ROOT / "CMakeLists.txt",
        ROOT / "pyproject.toml",
        script,
        *([wrapper] if wrapper else []),
        args.nomination,
        args.policy_freeze,
        Path(f"{args.policy_freeze}.sha256"),
        *([amendment, Path(f"{amendment}.sha256")] if amendment else []),
        *sorted((out / "plans").rglob("*")),
    ]
    files.update({str(p.resolve()): sha256_file(p) for p in pinned if p.is_file()})
    seal = {
        "artifact_kind": "xgraph.post_validation_serving_seal",
        "version": 1,
        "repo": str(ROOT),
        "out": str(out),
        "code_commit": commit,
        "uv_cache": cache,
        "policy_freeze": str(args.policy_freeze.resolve()),
        "frozen_policies": frozen,
        "validation_log": str(args.source_root / "target/queue/validation.log"),
        "post_log": str(post_log),
        "stages": stages,
        "files": files,
        "prices": prices,
        **(
            {"serving_amendment": str(amendment), "lifetime_budget_seconds": SERVING_CAP_SECONDS}
            if amendment
            else {}
        ),
        "analysis": {
            "scope": (
                "development only; locked WT/agent primary, W/B2 descriptive; warm only"
                if amendment
                else "development only; screen descriptive, locked primary, cold sensitivity"
            ),
            "primary": "WT/reference complete reserved workload GPU-seconds per admitted question, "
            "warm C1 and C8 separately in each family; fixed timing workload",
            "interval": "95% percentile paired complete-repeat bootstrap; 100000 draws, seed 17; "
            "all questions and both arms stay in each repetition block",
            "savings": "whole_trial_ratio.high < 1",
            "half": "whole_trial_ratio.high <= 0.5",
            "population_sensitivity": "survey-weighted ratio; paired repeat x cluster bootstrap",
            "failure": "retain missing/failed repetitions; incomplete cells make no claim",
            "cost_scope": "all calls, forced displays, retries, unsuccessful revisions, CPU/tool "
            "waits and final answer delivery inside the reserved interval; startup, warmup, "
            "index load and teardown reported separately with amortization",
            "quality": "score retained live answers separately; no inherited historical quality",
            "report": "final-answer latency; completed and correct questions per second; "
            "reserved GPU-seconds; setup amortization and sampled host/device memory",
            "budget": "24 GPU-hour lifetime cap, failures included; INCOMPLETE on exhaustion; "
            "no top-up; conservative worker-through-cleanup reservations"
            if amendment
            else None,
            "transfer": "100 shared validation questions, 7 policies, both families; sets; "
            "DoG uses labelled BM25 timing proxy, GCR native planner plus family reader",
            "transfer_accounting": "GPU allowances charge durable owned container occupancy; "
            "launcher waits/builds have a separate wall bound; uncertain intervals keep "
            "their reservation; serving trial forecast-derived wall bounds are unchanged",
        },
        "provenance": provenance([]),
    }
    if git_state() != commit:
        raise ValueError("code changed during post queue freeze")
    path = out / "serving-seal.json"
    path.write_text(json.dumps(seal, indent=1, ensure_ascii=False) + "\n")
    Path(f"{path}.sha256").write_text(sha256_file(path) + "\n")
    (out / "prices.json").write_text(json.dumps(prices, indent=1) + "\n")
    print(json.dumps(prices["total"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser(
        "freeze", help="CPU plans, immutable serving seal and owner script"
    )
    prepare.add_argument("--out", type=Path, default=ROOT / "target/queue-post")
    prepare.add_argument(
        "--policy-freeze", type=Path, default=FINAL / "target/final-validation/freeze-manifest.json"
    )
    prepare.add_argument(
        "--nomination", type=Path, default=SOURCE / "target/brainstorm/nomination-2026-10-25.md"
    )
    prepare.add_argument("--webqsp-data", type=Path, default=WEBQSP)
    prepare.add_argument("--source-root", type=Path, default=SOURCE)
    prepare.add_argument("--stage", action="append", help="include only these named stages")
    prepare.add_argument("--post-log", type=Path, help="owner's append-only per-stage status log")
    prepare.add_argument("--owner-wrapper", help="installable .sh wrapper filename under --out")
    prepare.add_argument(
        "--serving-amendment", type=Path, help="hash-bound reduced serving registration"
    )
    prepare.add_argument(
        "--pilot-restart", type=Path, help="closed failed pilot history for full replay"
    )
    run = commands.add_parser("run", help="GPU owner only, unless --dry-run")
    run.add_argument("--seal", type=Path, required=True)
    modes = run.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--run", action="store_true", help="GPU owner only")
    run.add_argument("--stage", action="append", help="run only these sealed stages, in seal order")
    replay = commands.add_parser("overhead", help="GPU owner only: short WT engine replay")
    replay.add_argument("--screen", type=Path, required=True)
    replay.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "freeze":
        freeze(args)
    elif args.action == "overhead":
        overhead(args.screen, args.out)
    else:
        run_queue(verify_seal(args.seal), dry_run=args.dry_run, stage_names=args.stage)


if __name__ == "__main__":
    main()
