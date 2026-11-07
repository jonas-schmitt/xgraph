"""Owner queue for the exact registered WebQSP engine, without benchmark generation."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from xgraph import webqsp_test as study
from xgraph.container_cleanup import cleanup_argvs, stop_container

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "target/webqsp-native-q"
REGISTRATION = ROOT.parent / "codex-webqsp-test/target/webqsp-test/amendment-8a/registration.json"
LOG = Path("/home/jonas/git/xgraph/target/queue/webqsp-native-q.log")
CAP = 1800
IMAGE = (
    "vllm/vllm-openai:v0.31.0@"
    "sha256:3f7dd5b777d34d1724456ce71f87385dca288c3bb23029ab27dee358f5d2b971"
)


def runtime_metadata():
    """Record the host dependency versions and build-tool identities without GPU discovery."""
    tools = [
        sys.executable,
        "/usr/local/cuda/bin/nvcc",
        *[shutil.which(name) for name in ("uv", "cmake", "ninja")],
    ]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: metadata.version(name)
            for name in ("xgraph", "numpy", "tokenizers", "torch", "pytest")
        },
        "executables": [study.identity(p) for p in tools if p and Path(p).is_file()],
        "compiler_present": Path("/usr/local/cuda/bin/nvcc").is_file(),
    }


def container_name(plan):
    return (
        "webqsp-native-"
        + plan["source_commit"][:12]
        + "-"
        + hashlib.sha256(plan["out"].encode()).hexdigest()[:12]
    )


def container_tag(plan):
    return hashlib.sha256((plan["source_commit"] + plan["out"]).encode()).hexdigest()


def inspect_argv(plan):
    return ["docker", "inspect", "--format", "{{json .Config.Labels}}", container_name(plan)]


def container_query(plan):
    result = subprocess.run(
        cleanup_argvs(container_name(plan))[0], capture_output=True, text=True, timeout=5
    )
    if result.returncode:
        raise ValueError("native container ownership cannot be inspected")
    return result.stdout.strip()


def cleanup_owned(plan):
    if container_query(plan):
        inspected = subprocess.run(inspect_argv(plan), capture_output=True, text=True, timeout=5)
        if inspected.returncode:
            raise ValueError("native container ownership cannot be inspected")
        labels = json.loads(inspected.stdout)
        if labels.get("xgraph.owner") != "webqsp-native-q" or labels.get(
            "xgraph.plan"
        ) != container_tag(plan):
            raise ValueError("refusing to remove a container without this native plan's ownership")
        stop_container(container_name(plan), Path(plan["out"]) / "cleanup-events.jsonl")
    return {"confirmed_absent": True}


READERS = [
    "tests/test_path_trie.py::test_processor_masks_every_prefix_and_releases_hypotheses",
    "tests/test_dog.py::test_dog_processor_masks_complete_triples_in_isolated_reordered_slots[cuda]",
    "tests/test_dog.py::test_dog_host_beams_use_native_cuda_requested_raw_logprobs",
    "tests/test_dog.py::test_dog_native_cuda_tokens_stop_only_after_complete_strings",
]


def git(*args):
    return subprocess.check_output(
        ["git", "--no-optional-locks", *args], cwd=ROOT, text=True
    ).strip()


def clean(commit):
    if git("status", "--porcelain", "--untracked-files=all") or git("rev-parse", "HEAD") != commit:
        raise ValueError("webqsp-native-q refuses a dirty or changed owner tree")


def driver(site, source, arguments):
    # scikit-build's editable finder otherwise overrides even an explicit sys.path.
    return (
        "import sys; "
        "sys.meta_path[:]=[f for f in sys.meta_path "
        "if not type(f).__module__.startswith('_editable')]; "
        f"sys.path[:0]=[{str(site)!r},{str(source)!r}]; "
        "import xgraph; "
        f"assert xgraph.__file__.startswith({str(site)!r}), xgraph.__file__; "
        "import pytest; "
        f"raise SystemExit(pytest.main({arguments!r}))"
    )


def stages(plan):
    """Return the bounded build, agreement and native-reader commands."""
    out = Path(plan["out"])
    source, site = out / "source", out / "site"
    temp = ROOT / "target/tmp/webqsp-native-q"
    env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "XG_START_ATTEMPTS": "1",
        "XG_GPU_IDLE": "1",
        "XG_AVAILABLE_WAIT": "0",
        "XG_HEADROOM_WAIT": "0",
        "TMPDIR": str(temp),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTEST_ADDOPTS": "",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "CMAKE_BUILD_PARALLEL_LEVEL": "1",
        "CMAKE_ARGS": "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc "
        "-DCMAKE_CUDA_ARCHITECTURES=121-real",
        "SKBUILD_BUILD_DIR": str(out / "cuda-build"),
        "UV_CACHE_DIR": str(ROOT / "target/webqsp-8a-gate/uv-cache"),
        "UV_LINK_MODE": "hardlink",
        "UV_OFFLINE": "1",
        "UV_PYTHON_DOWNLOADS": "never",
    }
    host_args = [
        "-q",
        "-p",
        "no:cacheprovider",
        f"--rootdir={source}",
        f"--junitxml={out / 'engine-adapter.xml'}",
        f"--basetemp={temp / 'engine-adapter'}",
        str(source / "tests/test_engine.py"),
        str(source / "tests/test_vllm_adapter.py"),
    ]
    reader_args = [
        "-q",
        "-p",
        "no:cacheprovider",
        "--junitxml=/out/native-readers.xml",
        "--basetemp=/owner-tmp/readers",
        *READERS,
    ]
    # Import pytest from the host tools without adding their Torch/vLLM to image imports.
    reader_driver = (
        "import sys; sys.path.insert(0, '/host-tools'); import pytest; sys.path.pop(0); "
        "sys.path[:0]=['/site','/source']; import xgraph; "
        "assert xgraph.__file__.startswith('/site/'), xgraph.__file__; "
        f"raise SystemExit(pytest.main({reader_args!r}))"
    )
    commands = [
        (
            "native-build",
            600,
            [
                "uv",
                "build",
                "--wheel",
                "--verbose",
                "--offline",
                "--python",
                str(ROOT / ".venv/bin/python"),
                "--out-dir",
                str(out / "wheels"),
                "--config-setting",
                "cmake.define.CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
                "--config-setting",
                "cmake.define.CMAKE_CUDA_ARCHITECTURES=121-real",
                str(source),
            ],
        ),
        (
            "engine-adapter",
            600,
            [str(ROOT / ".venv/bin/python"), "-I", "-c", driver(site, source, host_args)],
        ),
        (
            "native-readers",
            180,
            [
                "docker",
                "run",
                "--rm",
                "--pull=never",
                "--name",
                container_name(plan),
                "--label",
                "xgraph.owner=webqsp-native-q",
                "--label",
                f"xgraph.plan={container_tag(plan)}",
                "--gpus",
                "all",
                "--network",
                "none",
                "--memory=4g",
                "--memory-swap=4g",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--volume",
                f"{source}:/source:ro",
                "--volume",
                f"{site}:/site:ro",
                "--volume",
                f"{out}:/out",
                "--volume",
                f"{temp}:/owner-tmp",
                "--volume",
                f"{ROOT / '.venv/lib/python3.12/site-packages'}:/host-tools:ro",
                "--workdir",
                "/source",
                "--env",
                "HOME=/out/home",
                "--env",
                "TMPDIR=/owner-tmp",
                "--env",
                "PYTHONDONTWRITEBYTECODE=1",
                "--env",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
                "--env",
                "OPENBLAS_NUM_THREADS=1",
                "--env",
                "OMP_NUM_THREADS=1",
                "--entrypoint",
                "python3",
                IMAGE,
                "-c",
                reader_driver,
            ],
        ),
    ]
    return [
        {
            "name": name,
            "family": "shared",
            "scope": "webqsp-native",
            "kind": "gpu",
            "launcher_wall_cap_seconds": seconds,
            "env": env.copy(),
            "command": [
                "nice",
                "-n",
                "10",
                str(ROOT / ".venv/bin/python"),
                "-I",
                str(ROOT / "scripts/webqsp-native-memory.py"),
                "--report",
                str(out / (name + "-resources.json")),
                "--",
                *command,
            ],
        }
        for name, seconds, command in commands
    ]


def headroom():
    return [
        "env",
        "XG_GPU_IDLE=1",
        "XG_AVAILABLE_WAIT=43200",
        "XG_AVAILABLE_POLL=30",
        "XG_NEED_GIB=8",
        "XG_HEADROOM_WAIT=0",
        f"XG_PAGECACHE_ROOTS={OUT}",
        f"UV_CACHE_DIR={ROOT / 'target/webqsp-8a-gate/uv-cache'}",
        f"UV_PROJECT_ENVIRONMENT={ROOT / '.venv'}",
        "UV_LINK_MODE=hardlink",
        "UV_OFFLINE=1",
        "UV_PYTHON_DOWNLOADS=never",
        "PYTHONDONTWRITEBYTECODE=1",
        f"TMPDIR={ROOT / 'target/tmp'}",
        "OPENBLAS_NUM_THREADS=1",
        "OMP_NUM_THREADS=1",
        "bash",
        str(ROOT / "scripts/gpu-headroom.sh"),
        "8",
    ]


def preview(plan):
    rows = [f"# Conservative lifetime cap {CAP}s; all build/test/cleanup attempts charged."]
    for job in plan["stages"]:
        if job["name"] == "native-readers":
            rows.append("# before launch: " + shlex.join(cleanup_argvs(container_name(plan))[0]))
        rows += [
            shlex.join(headroom()),
            shlex.join(
                ["env", *[f"{k}={v}" for k, v in sorted(job["env"].items())], *job["command"]]
            ),
        ]
    for command in [inspect_argv(plan), *cleanup_argvs(container_name(plan))]:
        rows.append("# conditional cleanup: " + shlex.join(command))
    return "\n".join(rows) + "\n"


def junit(path, expected):
    passed, skipped, seen = set(), set(), set()
    for case in ET.parse(path).getroot().iter("testcase"):
        parts = case.attrib["classname"].split(".")
        at = max(i for i, part in enumerate(parts) if part.startswith("test_"))
        node = (
            "/".join(parts[: at + 1]) + ".py::" + "::".join([*parts[at + 1 :], case.attrib["name"]])
        )
        if node in seen or case.find("failure") is not None or case.find("error") is not None:
            raise ValueError(f"native agreement duplicate or failed case: {node}")
        seen.add(node)
        skip = case.find("skipped")
        if skip is None:
            passed.add(node)
        elif (
            node
            in {
                f"tests/test_engine.py::test_engine_end_after_answer_drafts_and_slot_reuse[True-{r}]"
                for r in ("vocabulary", "successor")
            }
            and skip.get("message") == "STOP requires frontier relation support"
        ):
            skipped.add(node)
        else:
            raise ValueError(f"native agreement skipped: {node}")
    if not passed or seen != set(expected):
        raise ValueError("native agreement JUnit differs from the complete collected inventory")
    return {"passed": sorted(passed), "unsupported": sorted(skipped)}


def prepare(registration_path, out, *, check=False):
    from xgraph.webqsp_gate import DECISION, check_gcr

    registration = study.read_sealed(registration_path)
    decision = json.loads(DECISION.read_text())
    if decision["registration"] != study.identity(registration_path):
        raise ValueError("native queue requires the reviewed amendment-8a decision registration")
    qualified = check_gcr(decision["native_gcr"], registration)
    qualification_charge = qualified["protocol_qualification_gpu_seconds"]
    pricing_charge = study.native_pricing_charge(
        registration, registration["config"]["baseline_timings"]
    )
    group = json.loads(Path(registration["config"]["baseline_timings"]).read_text())["groups"][
        "gcr"
    ]
    if qualification_charge + CAP + pricing_charge > registration["pricing"]["reserve_seconds"]:
        raise ValueError("native queue cap does not fit the remaining registered core reserve")
    commit = git("rev-parse", "HEAD")
    if not check:
        clean(commit)
    environment = runtime_metadata()
    if not check and not environment["compiler_present"]:
        raise ValueError("native owner preparation needs /usr/local/cuda/bin/nvcc")
    out = Path(out).resolve()
    if not check and out != OUT.resolve():
        raise ValueError(
            "native qualification has one fixed lifetime destination; use --check elsewhere"
        )
    out.mkdir(parents=True, exist_ok=False)
    source = out / "source"
    source.mkdir()
    archive = subprocess.check_output(["git", "archive", registration["commit"]], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(source, filter="data")
    plan = {
        "artifact_kind": "xgraph.webqsp_native_plan",
        "commit": commit,
        "source_commit": registration["commit"],
        "registration": study.identity(registration_path),
        "out": str(out),
        "cap_seconds": CAP,
        "check_only": check,
        "seed": 17,
        "prior_qualification_gpu_seconds": qualification_charge,
        "native_pricing_gpu_seconds": pricing_charge,
        "environment": environment,
        "log": str(LOG),
        "created_at": datetime.now(UTC).isoformat(),
        "inputs": [study.identity(p) for p in sorted(source.rglob("*")) if p.is_file()]
        + [
            study.identity(__file__),
            study.identity(ROOT / "scripts/webqsp-native-q.py"),
            study.identity(ROOT / "scripts/webqsp-native-memory.py"),
            study.identity(ROOT / "scripts/gpu-headroom.sh"),
            decision["native_gcr"],
            group["receipt"],
        ],
    }
    plan["stages"] = stages(plan)
    inventories = {}
    for name, selectors in (
        ("engine-adapter", ["tests/test_engine.py", "tests/test_vllm_adapter.py"]),
        ("native-readers", READERS),
    ):
        args = ["--collect-only", "-q", "-p", "no:cacheprovider", *selectors]
        command = [str(ROOT / ".venv/bin/python"), "-I", "-c", driver(source / "src", source, args)]
        result = subprocess.run(
            command,
            cwd=source,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "TMPDIR": str(ROOT / "target/tmp"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "RAYON_NUM_THREADS": "1",
            },
            capture_output=True,
            text=True,
            check=True,
            timeout=180,
        )
        (out / f"{name}-collection.txt").write_text(result.stdout + result.stderr)
        inventories[name] = [
            line for line in result.stdout.splitlines() if line.startswith("tests/")
        ]
        if not inventories[name] or len(inventories[name]) != len(set(inventories[name])):
            raise ValueError("empty or duplicate native collection")
    plan["inventories"] = inventories
    study.write_sealed(out / "plan.json", plan)
    (out / "dry-run.txt").write_text(preview(plan))
    wrapper = out / "webqsp-native-q.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nexec nice -n 10 "
        + shlex.join(
            [
                str(ROOT / ".venv/bin/python"),
                "-I",
                str(ROOT / "scripts/webqsp-native-q.py"),
                "--plan",
                str(out / "plan.json"),
            ]
        )
        + ' "$@"\n'
    )
    wrapper.chmod(0o755)
    return plan


def verify_plan(path):
    plan = study.read_sealed(path)
    if Path(plan["out"]).resolve() != OUT.resolve():
        raise ValueError("native qualification cannot change its lifetime destination")
    if plan["cap_seconds"] != CAP or plan["check_only"] or plan["stages"] != stages(plan):
        raise ValueError("native queue plan is changed or only a CPU preview")
    if plan["environment"] != runtime_metadata():
        raise ValueError("native queue dependency versions or build executables changed")
    for item in [plan["registration"], *plan["inputs"]]:
        study.verify_identity(item)
    for key in ("prior_qualification_gpu_seconds", "native_pricing_gpu_seconds"):
        study.positive(plan[key], key)
    registration = study.read_sealed(plan["registration"]["path"])
    total = plan["prior_qualification_gpu_seconds"] + CAP + plan["native_pricing_gpu_seconds"]
    if total > registration["pricing"]["reserve_seconds"]:
        raise ValueError("native lifetime cap exceeds the registered remaining reserve")
    return plan


def verify_stage(plan, path, job, ledger):
    receipt_path = Path(plan["out"]) / "receipts" / (job["name"] + ".json")
    receipt = study.read_sealed(receipt_path)
    attempts = [a for a in ledger["attempts"] if a["stage"] == study.stage_id(job)]
    if (
        receipt["plan"] != study.identity(path)
        or receipt["job"] != job
        or len(attempts) != 1
        or receipt["attempt"] != attempts[0]
        or attempts[0]["returncode"] != 0
        or attempts[0]["timeout"]
        or not 0 <= attempts[0]["elapsed_seconds"] <= job["launcher_wall_cap_seconds"]
    ):
        raise ValueError("native stage receipt or ledger changed")
    for item in receipt["evidence"]:
        study.verify_identity(item)
    resource_path = Path(plan["out"]) / (job["name"] + "-resources.json")
    resource = json.loads(resource_path.read_text())
    at = job["command"].index("--")
    if (
        study.identity(resource_path) not in receipt["evidence"]
        or resource.get("command") != job["command"][at + 1 :]
        or resource.get("returncode") != 0
        or resource.get("limit_violation") is not None
        or resource.get("per_process_rss_limit_bytes") != 4 * 1024**3
    ):
        raise ValueError("native resource receipt is missing, failed or changed")
    if job["name"] != "native-build":
        junit(Path(plan["out"]) / (job["name"] + ".xml"), plan["inventories"][job["name"]])
    return study.identity(receipt_path)


def verify_qualification(item, registration):
    study.verify_identity(item)
    qualified = study.read_sealed(item["path"])
    plan_ref = qualified["plan"]
    study.verify_identity(plan_ref)
    plan = verify_plan(plan_ref["path"])
    if (
        qualified.get("status") != "QUALIFIED"
        or plan["source_commit"] != registration["commit"]
        or json.loads(Path(plan["registration"]["path"]).read_text()) != registration
    ):
        raise ValueError("native qualification does not cover the registered source")
    study.verify_identity(qualified["ledger"])
    ledger = json.loads(Path(qualified["ledger"]["path"]).read_text())
    spent = sum(a["elapsed_seconds"] for a in ledger["attempts"])
    if (
        ledger["active"] is not None
        or len(ledger["attempts"]) != len(plan["stages"])
        or spent != ledger["spent_seconds"]
        or not 0 <= spent <= CAP
        or qualified["protocol_qualification_gpu_seconds"] != spent
    ):
        raise ValueError("native qualification has unsettled or changed lifetime accounting")
    evidence = [verify_stage(plan, plan_ref["path"], j, ledger) for j in plan["stages"]]
    if qualified["evidence"] != evidence:
        raise ValueError("native qualification stage evidence changed")
    return qualified


def run(path, *, skip_existing=False):
    plan = verify_plan(path)
    clean(plan["commit"])
    out = Path(plan["out"])
    (ROOT / "target/tmp/webqsp-native-q").mkdir(parents=True, exist_ok=True)
    with (out / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_path = out / "ledger.json"
        if not ledger_path.exists():
            study.write_json(ledger_path, {"spent_seconds": 0, "active": None, "attempts": []})
        ledger = json.loads(ledger_path.read_text())
        if ledger["active"] is not None:
            raise ValueError("unsettled native reservation; preserve it for owner reconciliation")
        evidence = []
        for job in plan["stages"]:
            name = job["name"]
            receipt = out / "receipts" / (name + ".json")
            if receipt.exists():
                if not skip_existing:
                    raise ValueError("existing native receipt; select --skip-existing")
                evidence.append(verify_stage(plan, path, job, ledger))
                status(plan, name, "skip-existing")
                continue
            if any(a["stage"] == study.stage_id(job) for a in ledger["attempts"]):
                raise ValueError("unreceipted native attempt; no implicit retry")
            # Wait before reserving GPU seconds. The launch itself checks idle again.
            status(plan, name, "headroom")
            admission_env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith(("XG_", "XGRAPH_", "UV_", "PYTEST_"))
                and k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX")
            }
            try:
                subprocess.run(headroom(), cwd=ROOT, env=admission_env, check=True, timeout=43300)
            except (OSError, subprocess.SubprocessError) as error:
                status(plan, name, f"headroom failed: {type(error).__name__}")
                raise
            clean(plan["commit"])
            status(plan, name, "start")
            cleanup = None
            if name == "native-readers":
                if container_query(plan):
                    status(plan, name, "refused: pre-existing container")
                    raise ValueError("pre-existing native container; refuse launch without cleanup")

                def cleanup():
                    return cleanup_owned(plan)

            code, _ = study.execute_gpu_stage(
                job, ledger_path, CAP, out / (name + ".log"), cleanup=cleanup
            )
            ledger = json.loads(ledger_path.read_text())
            status(plan, name, f"exit {code}")
            if code:
                raise ValueError(f"native stage failed: {name}; elapsed time remains charged")
            artifacts = [out / (name + ".log"), out / (name + "-resources.json")]
            if name == "native-build":
                wheels = list((out / "wheels").glob("xgraph-*.whl"))
                if len(wheels) != 1 or (out / "site").exists():
                    raise ValueError("native build needs one wheel and a fresh extraction")
                with zipfile.ZipFile(wheels[0]) as wheel:
                    wheel.extractall(out / "site")
                if len(list((out / "site/xgraph").glob("_engine*.so"))) != 1:
                    raise ValueError("native wheel has no unique CUDA engine")
                for original in (out / "source/src/xgraph").rglob("*.py"):
                    installed = (
                        out / "site/xgraph" / original.relative_to(out / "source/src/xgraph")
                    )
                    if not installed.is_file() or installed.read_bytes() != original.read_bytes():
                        raise ValueError(f"native wheel changed the registered source: {original}")
                artifacts += [
                    *wheels,
                    *[p for p in sorted((out / "site").rglob("*")) if p.is_file()],
                ]
            else:
                xml = out / (name + ".xml")
                junit(xml, plan["inventories"][name])
                artifacts.append(xml)
            study.write_sealed(
                receipt,
                {
                    "plan": study.identity(path),
                    "job": job,
                    "attempt": ledger["attempts"][-1],
                    "evidence": [study.identity(p) for p in artifacts],
                },
            )
            evidence.append(verify_stage(plan, path, job, ledger))
        result = {
            "artifact_kind": "xgraph.webqsp_native_qualification",
            "status": "QUALIFIED",
            "plan": study.identity(path),
            "ledger": study.identity(ledger_path),
            "evidence": evidence,
            "protocol_qualification_gpu_seconds": ledger["spent_seconds"],
            "limits": "Registered engine/adapter and tensor agreement; no model accuracy or price.",
        }
        destination = out / "qualification.json"
        if destination.exists():
            if study.read_sealed(destination) != result:
                raise ValueError("native qualification changed")
        else:
            study.write_sealed(destination, result)
        return result


def status(plan, stage, value):
    log = Path(plan["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write(f"webqsp-native-q {stage} {value} {datetime.now(UTC).isoformat()}\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, default=REGISTRATION)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--plan", type=Path, default=OUT / "plan.json")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)
    if args.check and not args.prepare:
        parser.error("--check is only valid with --prepare")
    if args.prepare:
        prepare(args.registration, args.out, check=args.check)
    elif args.run:
        run(args.plan, skip_existing=args.skip_existing)
    else:
        print(preview(study.read_sealed(args.plan)), end="")
