"""Prospective serving contamination recovery; no launch without owner bindings."""

import argparse
import copy
import ctypes
import fcntl
import hashlib
import importlib.machinery
import json
import math
import os
import random
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import time
import zipfile
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "eval/serving-contamination-recovery-2026-11-01/protocol.json"
PROTOCOL_SHA256 = "58a44158e63361bbc841e94172db236e0c3a546c15ff2d3e3864de7d4efb06bc"
RUNTIME_BOUNDARY_SHA256 = "56e4d3c1ad17d6bac34b2fc091b12115faa091e81ec85c74594342a7c2170f8b"
OS_BACKGROUND_SHA256 = "a84e961d6c737e54e28388f6d8c42220d35590cbbbc1a75b197e072a1f39319d"
OS_CONTEXT_SHA256 = "40be3505b14c4f2a2a4bfef481a47795597dc936f32ab3580b843bdc992a014b"
PARENT_LINEAGE_SHA256 = "5c13546a1cd373b3e0e880cbc829b3053ece29e24bc6b84d5dee19536569ba15"
BOUNDED_STORAGE_SHA256 = "b6c984996e10a6663e88e26e6d8cdbb2b64fd2b31d3c0d22c76aa7c284342b64"
CONTROL_ENVELOPE_SHA256 = "47103d4414698fca8a62ac3511553eaf23e906d3232bb4ef3beae50c6d367d72"
ENGINEERING_AMENDMENT_SHA256 = "a2a817b4468c697d6215be480a0388bc802008f55eccde5c7dc39f5343e7a182"
ENGINEERING_AMENDMENT_PATH = (
    "eval/serving-contamination-recovery-2026-11-01/decisions/"
    "engineering-amendment-implementation-2026-11-01.json"
)
DECISION_SHA256 = {
    "runtime-transport-boundary-2026-11-01.json": RUNTIME_BOUNDARY_SHA256,
    "os-background-identity-refinement-2026-11-01.json": OS_BACKGROUND_SHA256,
    "os-context-lifecycle-refinement-2026-11-01.json": OS_CONTEXT_SHA256,
    "parent-runtime-lineage-binding-2026-11-01.json": PARENT_LINEAGE_SHA256,
    "bounded-evidence-storage-decision-2026-11-01.json": BOUNDED_STORAGE_SHA256,
    "control-envelope-decision-2026-11-01.json": CONTROL_ENVELOPE_SHA256,
}
CLAIM_NAMESPACE = "refs/xgraph/serving-contamination-recovery-2026-11-01"
CLAIM_COMMON_DIRECTORY = Path("/home/jonas/git/xgraph/.git")
CLEANUP_SECONDS = 180
NATIVE_BUDGET_KIND = "xgraph.serving_recovery_native_budget"
CONTROL_ALLOWANCE_FIELDS = ("relay_setup_seconds", "post_native_seconds")
STABLE_CHECKS = (
    "Maintenance tests",
    "Environment matches uv.lock",
    "Historical HF beam oracle environment",
    "Formatting",
    "Lint",
    "Documentation (warnings are errors)",
    "Test suite",
)
STAGES = (
    "locked-qwen38",
    "locked-gemma31",
    "descriptive-qwen38",
    "descriptive-gemma31",
    "overhead-qwen38",
    "overhead-gemma31",
)
TIMING_MARKER = Path("/home/jonas/git/xgraph/target/gate/TIMING")
TIMING_LOCK = Path("/tmp/xgraph-serving-cost.lock")
_SOURCE_BYTES_SHA256 = None


def verify_controller(review, *, argv_sha256):
    """Require the reviewed clean producer and all source qualification predicates."""
    protocol = load_protocol()
    decisions = decision_bindings()
    if (
        review.get("approved") is not True
        or review.get("protocol_sha256") != PROTOCOL_SHA256
        or review.get("canonical_child") != protocol["canonical_child"]
        or review.get("argv_sha256") != argv_sha256
        or re.fullmatch(r"[0-9a-f]{64}", argv_sha256) is None
        or review.get("decisions") != decisions
    ):
        raise ValueError(
            "owner review, decisions, destination, protocol or exact argv binding changed"
        )
    commit = review.get("controller_commit", "")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("review requires a full controller commit identity")
    if Path(review["controller_root"]).resolve() != ROOT:
        raise ValueError("reviewed controller source root changed")
    git = ClaimStore(ROOT).git
    common = Path(
        git("rev-parse", "--path-format=absolute", "--git-common-dir").stdout.decode().strip()
    ).resolve()
    if common != CLAIM_COMMON_DIRECTORY.resolve():
        raise ValueError("controller is outside the global worktree-shared claim repository")
    kind = git("cat-file", "-t", commit, check=False)
    if kind.returncode or kind.stdout.strip() != b"commit":
        raise ValueError("reviewed controller identity must be a commit object")
    if git("rev-parse", "HEAD").stdout.strip().decode() != commit:
        raise ValueError("reviewed controller commit changed")
    if git("status", "--porcelain", "--untracked-files=normal").stdout:
        raise ValueError("reviewed controller source is dirty")
    source = Path(__file__).resolve()
    if source != ROOT / "src/xgraph/serving_recovery.py":
        raise ValueError("controller imported outside reviewed source")
    committed = git("show", f"{commit}:src/xgraph/serving_recovery.py").stdout
    if source.read_bytes() != committed:
        raise ValueError("controller source differs from reviewed commit")
    for entry in decisions.values():
        if git("show", f"{commit}:{entry['path']}").stdout != (ROOT / entry["path"]).read_bytes():
            raise ValueError("adopted decision source differs from reviewed commit")
    monitor = ROOT / "src/xgraph/serving_recovery_monitor.py"
    monitor_pin = metadata_identity(monitor)
    frozen_monitor = git("show", f"{commit}:src/xgraph/serving_recovery_monitor.py", check=False)
    if frozen_monitor.returncode or monitor.read_bytes() != frozen_monitor.stdout:
        raise ValueError("monitor source differs from the reviewed controller commit")
    gate = review.get("full_gate", {})
    if (
        gate.get("source_commit") != commit
        or type(gate.get("returncode")) is not int
        or gate["returncode"] != 0
        or gate.get("passed_checks") != list(STABLE_CHECKS)
    ):
        raise ValueError("complete stable gate for the reviewed controller is required")
    return {
        "commit": commit,
        "module": str(source),
        "sha256": hashlib.sha256(committed).hexdigest(),
        "protocol_sha256": PROTOCOL_SHA256,
        "argv_sha256": argv_sha256,
        "decisions": decisions,
        "claim_common_directory": str(common),
        "monitor": {
            "module": str(monitor),
            "bytes": monitor_pin["bytes"],
            "sha256": monitor_pin["sha256"],
        },
    }


def endpoint_eligibility(schedule, completed):
    """Fixed completeness dependencies; never consult an endpoint's value."""
    if [stage["name"] for stage in schedule] != list(STAGES):
        raise ValueError("endpoint matrix stage order or coverage changed")
    for stage, count in zip(schedule, (20, 20, 12, 12, 1, 1), strict=True):
        name = stage["name"]
        expected_ids = (
            [name]
            if name.startswith("overhead")
            else [f"{name}/{number:04d}" for number in range(count)]
        )
        if [job["id"] for job in stage["jobs"]] != expected_ids:
            raise ValueError(f"{name}: endpoint matrix coverage or canonical job IDs changed")
    if len(completed) != len(set(completed)):
        raise ValueError("duplicate completed IDs cannot qualify endpoint coverage")
    completed = set(completed)
    expected = {job["id"] for stage in schedule for job in stage["jobs"]}
    if completed - expected:
        raise ValueError("foreign or old-phase receipt cannot qualify recovery")
    complete = {
        stage["name"]: all(job["id"] in completed for job in stage["jobs"]) for stage in schedule
    }
    primary = {family: complete[f"locked-{family}"] for family in ("qwen38", "gemma31")}
    return {
        "primary_family": primary,
        "joint_primary": all(primary.values()),
        "descriptive": {
            family: complete[f"descriptive-{family}"] for family in ("qwen38", "gemma31")
        },
        "overhead": {
            family: complete[f"overhead-{family}"] and primary[family]
            for family in ("qwen38", "gemma31")
        },
        "overall": "COMPLETE" if completed == expected else "INCOMPLETE",
    }


def load_protocol():
    data = PROTOCOL.read_bytes()
    if hashlib.sha256(data).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("adopted serving recovery protocol changed")
    return json.loads(data)


def decision_bindings():
    """Verify exact adopted decision bytes before any parent/control input access."""
    result = {}
    for name, expected in DECISION_SHA256.items():
        relative = Path("eval/serving-contamination-recovery-2026-11-01/decisions") / name
        pin = metadata_identity(ROOT / relative)
        if pin["sha256"] != expected:
            raise ValueError(f"adopted serving recovery decision changed: {name}")
        result[name] = {**pin, "path": str(relative)}
    return result


def parent_lineage_binding():
    """Load source-only authority; do not open its original runtime inputs here."""
    path = PROTOCOL.parent / "decisions" / "parent-runtime-lineage-binding-2026-11-01.json"
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != PARENT_LINEAGE_SHA256:
        raise ValueError("adopted parent runtime lineage decision changed")
    record = json.loads(data)
    if record.get("protocol_sha256") != PROTOCOL_SHA256 or set(
        record.get("stage_bindings", {})
    ) != set(STAGES):
        raise ValueError("parent runtime lineage authority has incomplete stage bindings")
    for stage in STAGES:
        entry = record["stage_bindings"][stage]
        original = "locked-" + stage.rsplit("-", 1)[1]
        if (
            entry.get("original_stage") != original
            or entry.get("basis")
            != (
                "actual_original_sidecar" if stage == original else "approved_common_source_fixture"
            )
            or entry.get("new_runtime_must_be_fresh_qualified") is not True
        ):
            raise ValueError(
                "parent runtime lineage does not explicitly authorize this stage basis"
            )
    return record


def lineage_registration(child):
    record = parent_lineage_binding()
    originals = {"audit": ("metadata", record["source_audit"])}
    for stage in STAGES[:2]:
        entry = record["stage_bindings"][stage]
        originals[stage + "-runtime"] = ("metadata", entry["parent_runtime_sidecar"])
        originals[stage + "-source"] = ("source-archive", entry["parent_source_archive"])
    directory = Path(child).parent / "lineage"
    return {
        "decision_sha256": PARENT_LINEAGE_SHA256,
        "stage_bindings": copy.deepcopy(record["stage_bindings"]),
        "inputs": {
            name: {
                "kind": kind,
                "original": copy.deepcopy(pin),
                "snapshot": {
                    **pin,
                    "path": str(
                        directory / (name + (".zip" if kind == "source-archive" else ".json"))
                    ),
                },
            }
            for name, (kind, pin) in originals.items()
        },
    }


def lineage_payloads(lineage, *, snapshots=False):
    """Read only the five exact adopted source/metadata identities."""
    result = {}
    for name, item in lineage["inputs"].items():
        pin = item["snapshot" if snapshots else "original"]
        maximum = 64 * 1024**2 if item["kind"] == "source-archive" else 2 * 1024**2
        if artifact_identity(pin["path"], maximum=maximum) != pin:
            raise ValueError("approved parent runtime lineage source metadata changed")
        data = Path(pin["path"]).read_bytes()
        if len(data) != pin["bytes"] or hashlib.sha256(data).hexdigest() != pin["sha256"]:
            raise ValueError("parent runtime lineage changed before retention")
        result[name] = data
    return result


def verify_registered_runtime_lineage(registration, source):
    """Check retained source fixtures without opening an original study or model."""
    lineage = registration["parent_lineage"]
    validations = {}
    for stage in STAGES[:2]:
        sidecar = lineage["inputs"][stage + "-runtime"]
        archive = lineage["inputs"][stage + "-source"]
        recorded = json.loads(Path(sidecar["snapshot"]["path"]).read_bytes())
        if set(recorded) != {
            "site",
            "wheel",
            "wheel_sha256",
            "source_snapshot",
            "source_sha256",
            "source_files",
            "preparation_seconds",
        }:
            raise ValueError(
                "parent runtime sidecar differs from the frozen seven-key source schema"
            )
        validation = verify_parent_runtime_lineage(recorded, source, approved_archive=archive)
        entry = lineage["stage_bindings"][stage]
        if (
            sidecar["original"] != entry["parent_runtime_sidecar"]
            or archive["original"] != entry["parent_source_archive"]
            or validation["relative_source_map_sha256"] != entry["relative_source_map_sha256"]
        ):
            raise ValueError("parent runtime sidecar/archive/map differs from its adopted binding")
        validations[stage] = {
            "sidecar": copy.deepcopy(sidecar),
            "archive": copy.deepcopy(archive),
            "validation": validation,
        }
    return {
        "decision_sha256": PARENT_LINEAGE_SHA256,
        "source_audit": copy.deepcopy(lineage["inputs"]["audit"]),
        "historical_dependency_bytes_qualified": False,
        "old_runtime_reused": False,
        "stages": {
            stage: {
                "basis": entry["basis"],
                "original_stage": entry["original_stage"],
                **copy.deepcopy(validations[entry["original_stage"]]),
            }
            for stage, entry in lineage["stage_bindings"].items()
        },
    }


def load_controller_monitor(source_proof):
    """Load the reviewed new monitor by file, independent of frozen xgraph imports."""
    pin = source_proof["monitor"]
    path = ROOT / "src/xgraph/serving_recovery_monitor.py"
    expected = {"path": str(path), "bytes": pin["bytes"], "sha256": pin["sha256"]}
    if pin["module"] != str(path) or metadata_identity(path) != expected:
        raise ValueError("controller monitor source binding changed")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != pin["sha256"]:
        raise ValueError("controller monitor source changed before compilation")
    git = ClaimStore(ROOT).git
    if (
        git("rev-parse", "HEAD").stdout.decode().strip() != source_proof["commit"]
        or git("status", "--porcelain", "--untracked-files=normal").stdout
        or Path(
            git("rev-parse", "--path-format=absolute", "--git-common-dir").stdout.decode().strip()
        ).resolve()
        != CLAIM_COMMON_DIRECTORY.resolve()
        or git("show", f"{source_proof['commit']}:src/xgraph/serving_recovery_monitor.py").stdout
        != data
    ):
        raise ValueError("controller monitor source/commit/global claim binding changed")
    alias = (
        "_xgraph_serving_recovery_monitor_"
        + pin["sha256"][:16]
        + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    )
    module = ModuleType(alias)
    module.__file__, module.__package__ = str(path), ""
    sys.modules[alias] = module
    exec(compile(data, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


def metadata_inputs(protocol, *, raw=False):
    """Read only the seven exact adopted control-metadata identities."""
    result = {}
    for pin in protocol["parent_metadata"]:
        path = Path(pin["path"])
        if path.is_symlink() or path.resolve() != path:
            raise ValueError(f"parent metadata is a symbolic link: {path}")
        if path.stat().st_size != pin["bytes"]:
            raise ValueError(f"adopted parent metadata size changed: {path}")
        data = path.read_bytes()
        if len(data) != pin["bytes"] or hashlib.sha256(data).hexdigest() != pin["sha256"]:
            raise ValueError(f"adopted parent metadata changed: {path}")
        result[str(path)] = data if raw else json.loads(data)
    return result


NATIVE_ATTESTATION_SCRIPT = (
    "import importlib, importlib.metadata, importlib.util, json, platform, sys\n"
    """from pathlib import Path
names = ('serve_cost', 'serve_mintqa', 'post_queue', 'cost', 'serving')
modules = {name:str(Path(importlib.import_module('xgraph.'+name).__file__).resolve())
           for name in names}
packages = {name:importlib.metadata.version(name) for name in ('xgraph','numpy','tokenizers')}
spec = importlib.util.find_spec('xgraph._engine')
engine = None if spec is None else {'path':spec.origin, 'loader':type(spec.loader).__name__}
native_binary = None
if sys.argv[1] == 'load-native':
    native_binary = str(Path(importlib.import_module('xgraph._engine').__file__).resolve())
print(json.dumps({'executable':sys.executable, 'python':platform.python_version(),
                  'modules':modules, 'packages':packages, 'engine_resolution':engine,
                  'native_binary':native_binary,
                  'native_binary_imported':native_binary is not None}))
"""
)


SOURCE_RELAY = """import hashlib, sys
from pathlib import Path
from types import ModuleType
source = Path(sys.argv[1])
data = source.read_bytes()
if hashlib.sha256(data).hexdigest() != sys.argv[2]:
    raise ValueError('reviewed serving recovery controller changed')
control = ModuleType('serving_recovery_control')
control.__file__, control.__package__ = str(source), ''
sys.modules[control.__name__] = control
exec(compile(data, str(source), 'exec', dont_inherit=True), control.__dict__)
control.native_cell(Path(sys.argv[3]))
"""

SOURCE_CONTROLLER = """import hashlib, sys
from pathlib import Path
from types import ModuleType
source = Path(sys.argv[1])
data = source.read_bytes()
if hashlib.sha256(data).hexdigest() != sys.argv[2]:
    raise ValueError('reviewed serving recovery controller changed')
control = ModuleType('serving_recovery_control')
control.__file__, control.__package__ = str(source), ''
sys.modules[control.__name__] = control
exec(compile(data, str(source), 'exec', dont_inherit=True), control.__dict__)
control._SOURCE_BYTES_SHA256 = sys.argv[2]
control.main(sys.argv[3:])
"""


def prospective_argv(schedule, child, source_proof):
    """Exact future outer argv; the relay has no unguarded worker entry point."""
    child = Path(child)
    return [
        {
            "id": job["id"],
            "profile": native_profile(job),
            "argv": [
                str(child / "frozen-code/.venv/bin/python"),
                "-I",
                "-u",
                "-c",
                SOURCE_RELAY,
                source_proof["module"],
                source_proof["sha256"],
                str(child / "specs" / (job["id"].replace("/", "-") + ".json")),
            ],
        }
        for stage in schedule
        for job in stage["jobs"]
    ]


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def immutable_json(path, value):
    """Create a new sealed metadata file without replacing prior artifacts."""
    path = Path(path)
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
    with path.with_suffix(path.suffix + ".sha256").open("x") as stream:
        stream.write(hashlib.sha256(data).hexdigest() + "\n")


def read_seal_digest(path):
    """Read exactly one regular, no-follow, lowercase ASCII SHA-256 sidecar."""
    path = Path(path)
    try:
        before = path.lstat()
        if path.resolve() != path or not stat.S_ISREG(before.st_mode) or before.st_size != 65:
            raise ValueError(f"seal sidecar must be a regular 65-byte file: {path}")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or opened.st_size != 65:
                raise ValueError(f"seal sidecar changed before open: {path}")
            data = stream.read(66)
    except OSError as error:
        raise ValueError(f"seal sidecar could not be read safely: {path}") from error
    if re.fullmatch(rb"[0-9a-f]{64}\n", data) is None:
        raise ValueError(f"seal sidecar is not lowercase ASCII hex plus newline: {path}")
    return data[:-1].decode("ascii")


def load_registered_controller(registration_path):
    """Authenticate original source custody before compiling at its actual filename.

    This does not replace the original metadata/native verifiers. Callers must
    subsequently run those unchanged verifiers on the returned legacy module.
    """
    path = Path(registration_path)
    try:
        claims = ClaimStore(ROOT)
        common = Path(
            claims.git("rev-parse", "--path-format=absolute", "--git-common-dir")
            .stdout.decode()
            .strip()
        ).resolve()
        if common != CLAIM_COMMON_DIRECTORY.resolve():
            raise ValueError("original registration needs the global claim repository")
        expected = Path(load_protocol()["canonical_child"]).parent / "registration.json"
        if path != expected or path.resolve() != path:
            raise ValueError("original registration canonical path changed")
        pin = metadata_identity(path)
        if read_seal_digest(path.with_suffix(".json.sha256")) != pin["sha256"]:
            raise ValueError("original registration seal changed")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != pin["sha256"]:
            raise ValueError("original registration changed before decoding")
        registration = json.loads(data)
        if not isinstance(registration, dict):
            raise ValueError("original registration must be a JSON object")
        draft = {key: value for key, value in registration.items() if key != "claim"}
        record = _registration_claim(draft)
        # A detached self-consistent claim is insufficient: verify the exact
        # typed record at the one original global registration reference.
        if (
            registration["claim"]["record"] != record
            or claims.verify("registration", record) != registration["claim"]["commit"]
        ):
            raise ValueError("original registration global claim changed")
        proof = registration["source"]
        source = Path(proof["module"])
        if (
            re.fullmatch(r"[0-9a-f]{40}", proof["commit"]) is None
            or re.fullmatch(r"[0-9a-f]{64}", proof["sha256"]) is None
            or proof["protocol_sha256"] != PROTOCOL_SHA256
            or Path(proof["claim_common_directory"]).resolve() != common
            or not source.is_absolute()
            or source.resolve() != source
            or source.parts[-3:] != ("src", "xgraph", "serving_recovery.py")
            or not stat.S_ISREG(source.lstat().st_mode)
        ):
            raise ValueError("original registered controller source identity changed")
        original_root = source.parents[2]
        git = ClaimStore(original_root).git
        original_common = Path(
            git("rev-parse", "--path-format=absolute", "--git-common-dir").stdout.decode().strip()
        ).resolve()
        if (
            original_common != common
            or git("cat-file", "-t", proof["commit"]).stdout.strip() != b"commit"
            or git("rev-parse", "HEAD").stdout.decode().strip() != proof["commit"]
            or git("status", "--porcelain", "--untracked-files=normal").stdout
        ):
            raise ValueError("original controller commit/clean source/global custody changed")
        code = source.read_bytes()
        monitor = original_root / "src/xgraph/serving_recovery_monitor.py"
        if (
            metadata_identity(source)["sha256"] != proof["sha256"]
            or hashlib.sha256(code).hexdigest() != proof["sha256"]
            or git("show", f"{proof['commit']}:src/xgraph/serving_recovery.py").stdout != code
            or proof["monitor"]["module"] != str(monitor)
            or metadata_identity(monitor)
            != {
                "path": str(monitor),
                "bytes": proof["monitor"]["bytes"],
                "sha256": proof["monitor"]["sha256"],
            }
            or git("show", f"{proof['commit']}:src/xgraph/serving_recovery_monitor.py").stdout
            != monitor.read_bytes()
        ):
            raise ValueError("original controller/monitor differs from its registered Git blob")
        alias = (
            "_registered_serving_controller_"
            + proof["sha256"]
            + hashlib.sha256(str(source).encode()).hexdigest()[:16]
        )
        legacy = ModuleType(alias)
        legacy.__file__, legacy.__package__ = str(source), ""
        sys.modules[alias] = legacy
        exec(compile(code, str(source), "exec", dont_inherit=True), legacy.__dict__)
        if (
            original_root != legacy.ROOT
            or legacy.CLAIM_COMMON_DIRECTORY.resolve() != common
            or legacy.CLAIM_NAMESPACE != CLAIM_NAMESPACE
            or legacy.PROTOCOL_SHA256 != PROTOCOL_SHA256
        ):
            raise ValueError("original controller authority/domain changed; no rebasing")
        return legacy, registration
    except (OSError, KeyError, TypeError, IndexError, subprocess.CalledProcessError) as error:
        raise ValueError(f"original registered controller custody unavailable: {path}") from error


def metadata_draft(protocol, payloads, source_proof, review_pin):
    """Construct a control-only draft; never follow embedded study/data paths."""
    pins = protocol["parent_metadata"]
    if len(pins) != 7:
        raise ValueError("recovery needs the seven adopted parent metadata identities")
    seal = json.loads(payloads[pins[0]["path"]])
    history = json.loads(payloads[pins[1]["path"]])
    parent_charge = protocol["cap"]["includes_parent_seconds"]
    charged = sum(
        nonnegative(attempt["charged_gpu_seconds"], "parent attempt charge")
        for attempt in history["attempts"]
    )
    if history["charged_gpu_seconds"] != parent_charge or charged != parent_charge:
        raise ValueError("parent history did not conserve every adopted charge")
    plans = {Path(pin["path"]).parent.name: json.loads(payloads[pin["path"]]) for pin in pins[3:]}
    schedule = build_schedule(protocol, seal, plans)
    child = Path(protocol["canonical_child"])
    return {
        "artifact_kind": "xgraph.serving_recovery_registration",
        "protocol_sha256": PROTOCOL_SHA256,
        "canonical_child": str(child),
        "cap_seconds": protocol["cap"]["cumulative_gpu_seconds"],
        "parent_charged_gpu_seconds": parent_charge,
        "parent_source_commit": protocol["parent_source_commit"],
        "parent_metadata": [
            {
                "original": pin,
                "snapshot": {**pin, "path": str(child.parent / "metadata" / f"{i:02d}.json")},
            }
            for i, pin in enumerate(pins)
        ],
        "decisions": {
            name: {
                "source": pin,
                "snapshot": {**pin, "path": str(child.parent / "decisions" / name)},
            }
            for name, pin in source_proof["decisions"].items()
        },
        "source": source_proof,
        "owner_review": review_pin,
        "schedule": schedule,
        "argv": prospective_argv(schedule, child, source_proof),
        "native_preparation": None,
        "parent_lineage": lineage_registration(child),
        "launch_enabled": False,
    }


def metadata_identity(path):
    path = Path(path)
    if path.is_symlink() or path.resolve() != path:
        raise ValueError(f"metadata identity path is a symbolic link: {path}")
    if path.stat().st_size > 2 * 1024**2:
        raise ValueError(f"unexpected metadata size: {path}")
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _registration_claim(draft):
    return {
        "canonical_child": draft["canonical_child"],
        "binding_sha256": json_sha256(draft),
        "protocol_sha256": PROTOCOL_SHA256,
        "controller_commit": draft["source"]["commit"],
    }


def register_metadata(review_path, *, argv_sha256):
    """Create prospective metadata once after the committed source/argv owner review.

    This does not prepare an environment or qualify native execution. The public
    run path stays closed until that separate execution contract is implemented.
    """
    review_path = Path(review_path)
    protocol = load_protocol()
    expected_review = Path(protocol["canonical_child"]).parent / "owner-review.json"
    if review_path != expected_review or review_path.resolve() != expected_review:
        raise ValueError("canonical owner review path changed")
    review_pin = metadata_identity(review_path)
    review = json.loads(review_path.read_text())
    source_proof = verify_controller(review, argv_sha256=argv_sha256)
    destination = Path(protocol["canonical_child"]).parent / "registration.json"
    claims = ClaimStore(ROOT)
    if (
        destination.exists()
        or destination.is_symlink()
        or claims.git(
            "rev-parse", "--verify", claims.reference("registration"), check=False
        ).returncode
        == 0
    ):
        raise FileExistsError("serving recovery registration already claimed or present; no reset")
    payloads = metadata_inputs(protocol, raw=True)
    draft = metadata_draft(protocol, payloads, source_proof, review_pin)
    source_payloads = lineage_payloads(draft["parent_lineage"])
    if json_sha256(draft["argv"]) != argv_sha256:
        raise ValueError("actual prospective argv differs from the reviewed exact argv")
    draft["provenance"] = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": [sys.executable, *sys.argv],
        "seed": 17,
        "controller_commit": source_proof["commit"],
    }
    record = _registration_claim(draft)
    claim = claims.create("registration", record)
    directory = destination.parent / "metadata"
    directory.mkdir(parents=True)
    for item in draft["parent_metadata"]:
        with Path(item["snapshot"]["path"]).open("xb") as stream:
            stream.write(payloads[item["original"]["path"]])
    (destination.parent / "decisions").mkdir()
    for item in draft["decisions"].values():
        data = (ROOT / item["source"]["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != item["source"]["sha256"]:
            raise ValueError("adopted decision changed while archiving registration")
        with Path(item["snapshot"]["path"]).open("xb") as stream:
            stream.write(data)
    (destination.parent / "lineage").mkdir()
    for name, item in draft["parent_lineage"]["inputs"].items():
        with Path(item["snapshot"]["path"]).open("xb") as stream:
            stream.write(source_payloads[name])
    registration = {**draft, "claim": {"commit": claim, "record": record}}
    immutable_json(destination, registration)
    return registration


def verify_metadata_registration(path):
    protocol = load_protocol()
    path = Path(path)
    expected = Path(protocol["canonical_child"]).parent / "registration.json"
    if path != expected or path.is_symlink():
        raise ValueError("canonical serving recovery registration path changed")
    pin = metadata_identity(path)
    if read_seal_digest(path.with_suffix(path.suffix + ".sha256")) != pin["sha256"]:
        raise ValueError("serving recovery registration seal changed")
    registration = json.loads(path.read_text())
    draft = {key: value for key, value in registration.items() if key != "claim"}
    record = _registration_claim(draft)
    if (
        registration["claim"]["record"] != record
        or ClaimStore(ROOT).verify("registration", record) != registration["claim"]["commit"]
    ):
        raise ValueError("serving recovery immutable registration claim changed")
    review_pin = registration["owner_review"]
    if metadata_identity(review_pin["path"]) != review_pin:
        raise ValueError("reviewed owner metadata changed")
    argv_sha = json_sha256(registration["argv"])
    proof = verify_controller(
        json.loads(Path(review_pin["path"]).read_text()), argv_sha256=argv_sha
    )
    if proof != registration["source"]:
        raise ValueError("registered controller source proof changed")
    payloads = {}
    if [item["original"] for item in registration["parent_metadata"]] != protocol[
        "parent_metadata"
    ]:
        raise ValueError("registered parent metadata differs from adopted identities")
    for item in registration["parent_metadata"]:
        if metadata_identity(item["snapshot"]["path"]) != item["snapshot"]:
            raise ValueError("immutable parent metadata snapshot changed")
        if {key: item["snapshot"][key] for key in ("bytes", "sha256")} != {
            key: item["original"][key] for key in ("bytes", "sha256")
        }:
            raise ValueError("parent snapshot bytes differ from adopted history")
        payloads[item["original"]["path"]] = Path(item["snapshot"]["path"]).read_bytes()
    expected_draft = metadata_draft(protocol, payloads, proof, review_pin)
    if any(registration[key] != value for key, value in expected_draft.items()):
        raise ValueError("registered normalized schedule, science or cumulative history changed")
    for item in registration["decisions"].values():
        if metadata_identity(item["snapshot"]["path"]) != item["snapshot"]:
            raise ValueError("immutable adopted decision snapshot changed")
    lineage_payloads(registration["parent_lineage"], snapshots=True)
    return registration


def execution_preparation_request(registration_path, review_path):
    """Bind one preparation attempt without reading any study or runtime payload."""
    registration_path, review_path = Path(registration_path), Path(review_path)
    registration = verify_metadata_registration(registration_path)
    child = Path(registration["canonical_child"])
    if (
        review_path != child.parent / "preparation-review.json"
        or review_path.resolve() != review_path
    ):
        raise ValueError("canonical execution preparation review path changed")
    review_pin = metadata_identity(review_path)
    data = review_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != review_pin["sha256"]:
        raise ValueError("execution preparation review changed while read")
    review = json.loads(data)
    registration_pin = metadata_identity(registration_path)
    if (
        review.get("approved") is not True
        or review.get("purpose") != "serving_recovery_preparation"
        or review.get("launch_enabled") is not False
        or review.get("protocol_sha256") != PROTOCOL_SHA256
        or review.get("runtime_transport_boundary_sha256") != RUNTIME_BOUNDARY_SHA256
        or review.get("decisions") != registration["source"]["decisions"]
        or review.get("canonical_child") != str(child)
        or review.get("controller_commit") != registration["source"]["commit"]
        or review.get("registration") != registration_pin
    ):
        raise ValueError(
            "approved preparation review, source or immutable registration binding changed"
        )
    return {
        "artifact_kind": "xgraph.serving_recovery_preparation_request",
        "canonical_child": str(child),
        "protocol_sha256": PROTOCOL_SHA256,
        "runtime_transport_boundary_sha256": RUNTIME_BOUNDARY_SHA256,
        "controller": registration["source"],
        "registration": registration_pin,
        "registration_claim": registration["claim"]["commit"],
        "owner_review": review_pin,
        "parent_source_commit": registration["parent_source_commit"],
        "parent_charged_gpu_seconds": registration["parent_charged_gpu_seconds"],
        "cap_seconds": registration["cap_seconds"],
        "argv_sha256": json_sha256(registration["argv"]),
        "launch_enabled": False,
        "decisions": registration["decisions"],
        "parent_lineage": registration["parent_lineage"],
    }


def begin_execution_preparation(registration_path, review_path):
    """Claim the canonical execution before creating any native preparation artifacts."""
    request = execution_preparation_request(registration_path, review_path)
    child = Path(request["canonical_child"])
    claims = ClaimStore(ROOT)
    if (
        child.resolve() != child
        or child.exists()
        or child.is_symlink()
        or claims.git(
            "rev-parse", "--verify", claims.reference("execution"), check=False
        ).returncode
        == 0
    ):
        raise FileExistsError(
            "canonical serving recovery execution already present/claimed; no reset"
        )
    claim = claims.create("execution", request)
    child.mkdir()
    state = {
        "status": "PREPARING_NOT_LAUNCH_READY",
        "request": request,
        "claim": claim,
        "created_at": datetime.now(UTC).isoformat(),
    }
    immutable_json(child / "preparation.json", state)
    return state


def verify_execution_preparation(registration_path):
    registration = verify_metadata_registration(registration_path)
    child = Path(registration["canonical_child"])
    path = child / "preparation.json"
    pin = metadata_identity(path)
    if read_seal_digest(path.with_suffix(".json.sha256")) != pin["sha256"]:
        raise ValueError("execution preparation metadata seal changed")
    state = json.loads(path.read_bytes())
    request = execution_preparation_request(
        registration_path, child.parent / "preparation-review.json"
    )
    if (
        state.get("status") != "PREPARING_NOT_LAUNCH_READY"
        or state.get("request") != request
        or ClaimStore(ROOT).verify("execution", request) != state.get("claim")
    ):
        raise ValueError("persistent execution preparation claim or request changed")
    return state


def prepare_native_execution(registration_path, review_path):
    """Perform the approved source-only fresh build and opaque execution sealing.

    This API is never called during development tests with real parent inputs.
    It creates no timing lease and admits no study worker or model operation.
    """
    request = execution_preparation_request(registration_path, review_path)
    review = json.loads(Path(review_path).read_bytes())
    options = review.get("native_preparation")
    required = {
        "source_root",
        "base_environment",
        "cuda_compiler",
        "wall_seconds",
        "cpu_affinity",
        "address_space_limit",
    }
    if (
        not isinstance(options, dict)
        or set(options) != required
        or any(
            not isinstance(options[key], str) or Path(options[key]).resolve() != Path(options[key])
            for key in ("source_root", "base_environment")
        )
        or not isinstance(options["cuda_compiler"], dict)
        or artifact_identity(options["cuda_compiler"]["path"], maximum=128 * 1024**2)
        != options["cuda_compiler"]
        or nonnegative(options["wall_seconds"], "approved native build wall") <= 0
        or not isinstance(options["cpu_affinity"], list)
        or not options["cpu_affinity"]
        or any(type(value) is not int or value < 0 for value in options["cpu_affinity"])
        or not isinstance(options["address_space_limit"], list)
        or len(options["address_space_limit"]) != 2
        or any(type(value) is not int or value < -1 for value in options["address_space_limit"])
    ):
        raise ValueError(
            "native preparation source/environment/compiler/resource profile is UNBOUND"
        )
    begin_execution_preparation(registration_path, review_path)
    child = Path(request["canonical_child"])
    destination = child / "frozen-code"
    source = snapshot_native_source(
        Path(options["source_root"]), request["parent_source_commit"], destination
    )
    environment = snapshot_native_environment(
        Path(options["base_environment"]), destination, source
    )
    verify_private_environment(destination, environment)
    # Verify literal parent sidecar/archive lineage before building; opaque study
    # bytes are copied only by the final seal and are never decoded here.
    registration = verify_metadata_registration(registration_path)
    verify_registered_runtime_lineage(registration, source)
    output, temporary = destination / "target/build", child / "preparation-tmp"
    output.mkdir(parents=True)
    temporary.mkdir()
    profile = native_profile(registration["schedule"][0]["jobs"][0])
    env = native_runtime_environment(
        profile,
        source=destination,
        temporary=temporary,
        cuda_compiler=options["cuda_compiler"]["path"],
        parent=os.environ,
    )
    argv = native_build_argv(destination, output)
    started = time.monotonic()
    try:
        with (child / "native-build.log").open("x") as log:
            result = supervise_overhead_stage(
                argv,
                cwd=destination,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                wall_seconds=options["wall_seconds"],
                preexec_fn=lambda: worker_resource_setup(
                    {
                        "nice": 19,
                        "cpu_affinity": options["cpu_affinity"],
                        "address_space_limit": options["address_space_limit"],
                    }
                ),
            )
        build = {
            "argv": argv,
            "returncode": result.returncode,
            "source_sha256": json_sha256(source),
            "environment_sha256": json_sha256(environment),
            "wheel_override": None,
            "compiler": options["cuda_compiler"],
            "elapsed_seconds": time.monotonic() - started,
        }
        immutable_json(child / "native-build.json", build)
        if result.returncode:
            raise RuntimeError("approved fresh native build failed; no alternate wheel or retry")
        runtime = json.loads((output / "preparation.json").read_bytes())
        wheel = verify_native_wheel(destination, source, runtime)
        enabled = enable_native_engine(destination, source, environment, wheel, build)
        attestation = attest_native_environment(destination, source, enabled, load_native=True)
        cwd = destination / "target/native-launch-cwd"
        cwd.mkdir()
        attest_native_module_launch(destination, source, enabled, cwd=cwd)
        return seal_native_execution(
            registration_path,
            {
                "source": source,
                "environment": enabled,
                "wheel": wheel,
                "build": build,
                "attestation": attestation,
            },
        )
    except BaseException as error:
        failure = child / "preparation-failed.json"
        if not failure.exists():
            immutable_json(
                failure,
                {
                    "artifact_kind": "xgraph.serving_recovery_preparation_failure",
                    "kind": type(error).__name__,
                    "launch_enabled": False,
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
        raise


def verify_prepared_native(child, prepared, parent_commit):
    """Verify actual private source/runtime proofs before sealing or using them."""
    child = Path(child)
    if set(prepared) != {"source", "environment", "wheel", "build", "attestation"}:
        raise ValueError("native preparation proof schema changed")
    source, environment, wheel, build, attested = (
        prepared[key] for key in ("source", "environment", "wheel", "build", "attestation")
    )
    destination = child / "frozen-code"
    if (
        source.get("root") != str(destination)
        or source.get("commit") != parent_commit
        or environment.get("native_engine", {}).get("wheel") != wheel
        or environment["native_engine"].get("build") != build
        or build.get("returncode") != 0
        or type(build.get("returncode")) is not int
        or build.get("source_sha256") != json_sha256(source)
        or build.get("environment_sha256")
        != environment["native_engine"].get("base_environment_sha256")
        or build.get("wheel_override") is not None
        or build.get("argv") != native_build_argv(destination, destination / "target/build")
    ):
        raise ValueError("native private source, fresh build or environment lineage changed")
    verify_native_source(destination, source)
    verify_private_environment(destination, environment)
    if (
        verify_native_wheel(destination, source, wheel["runtime"]) != wheel
        or attest_native_environment(destination, source, environment, load_native=True) != attested
        or attested.get("native_binary_imported") is not True
    ):
        raise ValueError("native wheel or actual isolated binary-import proof changed")
    return prepared


def seal_native_execution(registration_path, prepared):
    """Seal metadata/native preparation while keeping every original study opaque."""
    registration_path = Path(registration_path)
    registration = verify_metadata_registration(registration_path)
    preparation = verify_execution_preparation(registration_path)
    child = Path(registration["canonical_child"])
    execution_path = child / "execution.json"
    if (
        execution_path.exists()
        or execution_path.is_symlink()
        or execution_path.with_suffix(".json.sha256").exists()
    ):
        raise FileExistsError("prepared execution already sealed; no reset or reseal")
    verify_prepared_native(child, prepared, registration["parent_source_commit"])
    parent_runtime_lineage = verify_registered_runtime_lineage(registration, prepared["source"])
    parent_seal = json.loads(
        Path(registration["parent_metadata"][0]["snapshot"]["path"]).read_bytes()
    )
    original_plans = {
        Path(item["original"]["path"]).parent.name: item["snapshot"]
        for item in registration["parent_metadata"][3:]
    }
    stages = []
    for stage in registration["schedule"][:4]:
        original = stage["plan"]
        pin = artifact_identity(original, maximum=128 * 1024**2)
        if pin["sha256"] != parent_seal["files"].get(original):
            raise ValueError("opaque original study differs from the immutable parent seal")
        directory = Path(stage["output"]).parent
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / "original-study.json"
        data = Path(original).read_bytes()
        if hashlib.sha256(data).hexdigest() != pin["sha256"]:
            raise ValueError("opaque original study changed before copying")
        with destination.open("xb") as stream:
            stream.write(data)
        preview_original = str(Path(original).with_name("launch-commands.json"))
        preview_pin = metadata_identity(preview_original)
        if preview_pin["sha256"] != parent_seal["files"].get(preview_original):
            raise ValueError("parent launch preview differs from its immutable seal")
        verify_launch_preview(stage, preview_pin)
        preview = copy_pinned_artifact(preview_pin, directory / "launch-commands.json")
        stage_wheel, runtime_origin = mirror_stage_runtime(
            stage["name"], prepared["source"], prepared["wheel"]
        )
        runtime_path = directory / "runtime.json"
        immutable_json(runtime_path, stage_wheel["runtime"])
        plan = prepare_relocated_trial_plan(
            stage,
            original_plans[stage["name"]],
            controller_commit=registration["source"]["commit"],
            native_commit=registration["parent_source_commit"],
        )
        stages.append(
            {
                "name": stage["name"],
                "original_path": original,
                "original_study": artifact_identity(destination, maximum=128 * 1024**2),
                "runtime_json": metadata_identity(runtime_path),
                "trials": plan,
                "runtime_wheel": stage_wheel,
                "runtime_origin": runtime_origin,
                "launch_commands": preview,
                "capacity_basis": "first-native-observed-calibration",
                "server_names": {
                    job["id"]: "xgraph-recovery-" + secrets.token_hex(16) for job in stage["jobs"]
                },
            }
        )
    result = {
        "artifact_kind": "xgraph.serving_recovery_execution",
        "version": 1,
        "status": "PREPARED_REQUIRES_OWNER_RELEASE",
        "launch_enabled": False,
        "canonical_child": str(child),
        "registration": metadata_identity(registration_path),
        "preparation": metadata_identity(child / "preparation.json"),
        "preparation_claim": preparation["claim"],
        "controller": registration["source"],
        "decisions": registration["decisions"],
        "native": copy.deepcopy(prepared),
        "stages": stages,
        "parent_runtime_lineage": parent_runtime_lineage,
        "cap_seconds": registration["cap_seconds"],
        "parent_charged_gpu_seconds": registration["parent_charged_gpu_seconds"],
        "schedule_sha256": json_sha256(registration["schedule"]),
        "argv_sha256": json_sha256(registration["argv"]),
        "created_at": datetime.now(UTC).isoformat(),
    }
    immutable_json(execution_path, result)
    return result


def verify_launch_preview(stage, pin):
    """Inspect only sealed Docker control commands, never a study or model report."""
    if metadata_identity(pin["path"]) != pin:
        raise ValueError("sealed launch preview bytes changed")
    data = Path(pin["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != pin["sha256"]:
        raise ValueError("sealed launch preview changed before decoding")
    preview = json.loads(data)
    if (
        set(preview) != {"executed", "commands"}
        or preview["executed"] is not False
        or not isinstance(preview["commands"], list)
        or len(preview["commands"]) != len(stage["jobs"])
    ):
        raise ValueError("launch preview executed flag or planned matrix changed")
    for row, job in zip(preview["commands"], stage["jobs"], strict=True):
        if (
            not isinstance(row, dict)
            or set(row) != {"trial", "command"}
            or row["trial"] != job["original_trial"]
            or not isinstance(row["command"], list)
            or any(not isinstance(token, str) or not token for token in row["command"])
            or row["command"][:4] != ["docker", "run", "--pull=never", "-d"]
        ):
            raise ValueError("sealed launch preview trial or command matrix changed")
    return preview


def relocated_server_command(stage, job, retained):
    """Apply only the four declared ephemeral/path substitutions to a Docker preview."""
    preview = verify_launch_preview(stage, retained["launch_commands"])
    number = next(i for i, planned in enumerate(stage["jobs"]) if planned["id"] == job["id"])
    command = list(preview["commands"][number]["command"])
    old_directory = Path(job["original_trial"]["output"]).parent
    new_directory = Path(job["output"]).parent
    site = retained["runtime_wheel"]["runtime"]["site"]
    replacements = {
        "xgraph.serving-output=" + str(old_directory): "xgraph.serving-output="
        + str(new_directory),
        str(Path(stage["plan"]).parent / "runtime/site") + ":/site:ro": site + ":/site:ro",
        str(Path(stage["plan"]).parent / "runtime/container-home") + ":/home/xgraph": str(
            Path(site).parent / "container-home"
        )
        + ":/home/xgraph",
    }
    if command.count("--name") != 1:
        raise ValueError("sealed Docker preview does not have one server name")
    command[command.index("--name") + 1] = retained["server_names"][job["id"]]
    for original, replacement in replacements.items():
        if command.count(original) != 1:
            raise ValueError("sealed Docker preview has ambiguous site/HOME/output scope")
        command[command.index(original)] = replacement
    if command.count("--entrypoint") != 1:
        raise ValueError("sealed Docker preview has no unique entrypoint/image")
    entry = command.index("--entrypoint")
    if command[entry + 1] != "vllm" or entry + 2 >= len(command):
        raise ValueError("sealed Docker preview entrypoint changed")
    return command


def copy_pinned_artifact(original, destination):
    """Copy a registered regular source/runtime artifact without following links."""
    destination = Path(destination)
    if destination.resolve() != destination:
        raise ValueError("runtime artifact copy destination is not canonical")
    if artifact_identity(original["path"]) != original:
        raise ValueError("runtime artifact changed before a qualified stage mirror")
    digest, size = hashlib.sha256(), 0
    descriptor = os.open(original["path"], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source, destination.open("xb") as target:
        while data := source.read(1024**2):
            size += len(data)
            if size > original["bytes"]:
                raise ValueError("runtime artifact grew during mirroring")
            digest.update(data)
            target.write(data)
        target.flush()
        os.fsync(target.fileno())
    if size != original["bytes"] or digest.hexdigest() != original["sha256"]:
        raise ValueError("runtime artifact changed during mirroring; partial retained")
    return {**copy.deepcopy(original), "path": str(destination)}


def mirror_stage_runtime(stage, source, wheel):
    """Share fresh build bytes while retaining each original stage's HOME scope."""
    if stage not in STAGES[:4]:
        raise ValueError("unknown stage runtime mirror")
    root = Path(source["root"])
    if verify_native_wheel(root, source, wheel["runtime"]) != wheel:
        raise ValueError("qualified fresh build changed before stage mirroring")
    directory = root / "target/stages" / stage / "runtime"
    directory.mkdir(parents=True)
    started = time.monotonic()
    built = copy_pinned_artifact(wheel["wheel"], directory / Path(wheel["wheel"]["path"]).name)
    snapshot = copy_pinned_artifact(wheel["source_snapshot"], directory / "source.zip")
    site = directory / "site"
    site.mkdir()
    for name, pin in wheel["files"].items():
        target = site / name
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_pinned_artifact(pin, target)
    elapsed = time.monotonic() - started
    runtime = {
        **copy.deepcopy(wheel["runtime"]),
        "wheel": built["path"],
        "site": str(site),
        "source_snapshot": snapshot["path"],
        "preparation_seconds": elapsed,
    }
    proof = verify_native_wheel(root, source, runtime)
    return proof, {
        "kind": "byte-identical mirror of qualified fresh build",
        "build_wheel": copy.deepcopy(wheel["wheel"]),
        "copy_seconds": elapsed,
        "source_preparation_seconds_counted_at_build_only": wheel["runtime"]["preparation_seconds"],
        "container_home": str(directory / "container-home"),
    }


def verify_native_execution(registration_path):
    """Verify the whole sealed preparation, without decoding an original study."""
    registration_path = Path(registration_path)
    registration = verify_metadata_registration(registration_path)
    preparation = verify_execution_preparation(registration_path)
    child = Path(registration["canonical_child"])
    path = child / "execution.json"
    pin = metadata_identity(path)
    if read_seal_digest(path.with_suffix(".json.sha256")) != pin["sha256"]:
        raise ValueError("native execution seal changed")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != pin["sha256"]:
        raise ValueError("native execution changed before decoding")
    execution = json.loads(data)
    expected = {
        "artifact_kind": "xgraph.serving_recovery_execution",
        "version": 1,
        "status": "PREPARED_REQUIRES_OWNER_RELEASE",
        "launch_enabled": False,
        "canonical_child": str(child),
        "registration": metadata_identity(registration_path),
        "preparation": metadata_identity(child / "preparation.json"),
        "preparation_claim": preparation["claim"],
        "controller": registration["source"],
        "decisions": registration["decisions"],
        "cap_seconds": registration["cap_seconds"],
        "parent_charged_gpu_seconds": registration["parent_charged_gpu_seconds"],
        "schedule_sha256": json_sha256(registration["schedule"]),
        "argv_sha256": json_sha256(registration["argv"]),
    }
    if set(execution) != {
        *expected,
        "native",
        "stages",
        "created_at",
        "parent_runtime_lineage",
    } or any(
        type(execution.get(key)) is not type(value) or execution[key] != value
        for key, value in expected.items()
    ):
        raise ValueError("native execution registration/source/destination binding changed")
    verify_prepared_native(child, execution["native"], registration["parent_source_commit"])
    if execution["parent_runtime_lineage"] != verify_registered_runtime_lineage(
        registration, execution["native"]["source"]
    ):
        raise ValueError("native execution parent runtime source lineage binding changed")
    if [item["name"] for item in execution["stages"]] != list(STAGES[:4]):
        raise ValueError("native execution stage order changed")
    seal = json.loads(Path(registration["parent_metadata"][0]["snapshot"]["path"]).read_bytes())
    names = set()
    for stage, retained in zip(registration["schedule"][:4], execution["stages"], strict=True):
        directory = Path(stage["output"]).parent
        original, runtime, trials = (
            retained[key] for key in ("original_study", "runtime_json", "trials")
        )
        if (
            set(retained)
            != {
                "name",
                "original_path",
                "original_study",
                "runtime_json",
                "trials",
                "runtime_wheel",
                "runtime_origin",
                "launch_commands",
                "capacity_basis",
                "server_names",
            }
            or retained["original_path"] != stage["plan"]
            or original.get("path") != str(directory / "original-study.json")
            or original.get("sha256") != seal["files"].get(stage["plan"])
            or artifact_identity(original["path"], maximum=128 * 1024**2) != original
            or runtime.get("path") != str(directory / "runtime.json")
            or metadata_identity(runtime["path"]) != runtime
            or read_seal_digest(Path(runtime["path"]).with_suffix(".json.sha256"))
            != runtime["sha256"]
            or json.loads(Path(runtime["path"]).read_bytes())
            != retained["runtime_wheel"]["runtime"]
            or trials.get("path") != str(directory / "trials.json")
            or metadata_identity(trials["path"]) != trials
            or read_seal_digest(Path(trials["path"]).with_suffix(".json.sha256"))
            != trials["sha256"]
            or retained["capacity_basis"] != "first-native-observed-calibration"
            or set(retained["server_names"]) != {job["id"] for job in stage["jobs"]}
        ):
            raise ValueError("native execution opaque study/runtime/plan identity changed")
        preview = retained["launch_commands"]
        if preview.get("path") != str(directory / "launch-commands.json") or preview.get(
            "sha256"
        ) != seal["files"].get(str(Path(stage["plan"]).with_name("launch-commands.json"))):
            raise ValueError("native launch preview parent path/seal binding changed")
        verify_launch_preview(stage, preview)
        mirror, origin = retained["runtime_wheel"], retained["runtime_origin"]
        global_wheel = execution["native"]["wheel"]
        expected_directory = (
            Path(execution["native"]["source"]["root"])
            / "target/stages"
            / stage["name"]
            / "runtime"
        )
        if (
            mirror["wheel"]["path"]
            != str(expected_directory / Path(global_wheel["wheel"]["path"]).name)
            or mirror["runtime"]["site"] != str(expected_directory / "site")
            or mirror["wheel"]["sha256"] != global_wheel["wheel"]["sha256"]
            or mirror["engine"]["sha256"] != global_wheel["engine"]["sha256"]
            or mirror["source_snapshot"]["sha256"] != global_wheel["source_snapshot"]["sha256"]
            or mirror["runtime"]["source_files"] != global_wheel["runtime"]["source_files"]
            or origin
            != {
                "kind": "byte-identical mirror of qualified fresh build",
                "build_wheel": global_wheel["wheel"],
                "copy_seconds": mirror["runtime"]["preparation_seconds"],
                "source_preparation_seconds_counted_at_build_only": global_wheel["runtime"][
                    "preparation_seconds"
                ],
                "container_home": str(expected_directory / "container-home"),
            }
            or verify_native_wheel(
                Path(execution["native"]["source"]["root"]),
                execution["native"]["source"],
                mirror["runtime"],
            )
            != mirror
        ):
            raise ValueError("native execution stage mirror differs from its qualified fresh build")
        plan = json.loads(Path(trials["path"]).read_bytes())
        if (
            plan.get("trials") != [job["trial"] for job in stage["jobs"]]
            or plan.get("seed") != 17
            or any(plan.get("kv_cache_bytes") != job["kv_cache_bytes"] for job in stage["jobs"])
        ):
            raise ValueError("native execution planned matrix differs from registered science")
        for name in retained["server_names"].values():
            if (
                not isinstance(name, str)
                or re.fullmatch(r"xgraph-recovery-[0-9a-f]{32}", name) is None
                or name in names
            ):
                raise ValueError("native execution server identity is duplicated or invalid")
            names.add(name)
    return execution


def engineering_amendment_request(registration_path, review_path):
    """Describe one source successor without replacing any original sealed proof.

    The implementation decision is proposal provenance only. Adoption additionally
    needs this exact reviewed, gated source and a create-once typed claim; native
    launch still requires the separate owner and production resource reviews.
    """
    registration_path, review_path = Path(registration_path), Path(review_path)
    child = Path(load_protocol()["canonical_child"])
    if (
        review_path != child.parent / "engineering-source-review.json"
        or review_path.resolve() != review_path
    ):
        raise ValueError("canonical engineering source review path changed")
    review_pin = metadata_identity(review_path)
    data = review_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != review_pin["sha256"]:
        raise ValueError("engineering source review changed before decoding")
    review = json.loads(data)
    if (
        not isinstance(review, dict)
        or review.get("purpose") != "serving_recovery_engineering_amendment"
        or review.get("comparison_values_used") is not False
        or review.get("implementation_decision_sha256") != ENGINEERING_AMENDMENT_SHA256
    ):
        raise ValueError("explicit comparison-blind engineering source review is required")
    source = verify_controller(review, argv_sha256=review.get("argv_sha256", ""))
    decision = metadata_identity(ROOT / ENGINEERING_AMENDMENT_PATH)
    if (
        decision["sha256"] != ENGINEERING_AMENDMENT_SHA256
        or ClaimStore(ROOT).git("show", f"{source['commit']}:{ENGINEERING_AMENDMENT_PATH}").stdout
        != (ROOT / ENGINEERING_AMENDMENT_PATH).read_bytes()
    ):
        raise ValueError("committed engineering implementation decision changed")
    # Source paths in a self-consistent JSON file are not code authority. The
    # loader first authenticates the original typed global registration claim.
    legacy, retained = load_registered_controller(registration_path)
    registration = legacy.verify_metadata_registration(registration_path)
    execution = legacy.verify_native_execution(registration_path)
    registration_pin = metadata_identity(registration_path)
    execution_pin = metadata_identity(child / "execution.json")
    if (
        registration != retained
        or review.get("registration") != registration_pin
        or review.get("execution") != execution_pin
        or registration["canonical_child"] != str(child)
        or source["claim_common_directory"] != registration["source"]["claim_common_directory"]
        or source["decisions"] != registration["source"]["decisions"]
    ):
        raise ValueError("engineering review original seals, decisions or authority changed")
    effective = prospective_argv(registration["schedule"], child, source)
    original = registration["argv"]
    if len(effective) != len(original) or any(
        {key: value for key, value in old.items() if key != "argv"}
        != {key: value for key, value in new.items() if key != "argv"}
        or len(old["argv"]) != len(new["argv"])
        or old["argv"][5:7] != [registration["source"]["module"], registration["source"]["sha256"]]
        or new["argv"][5:7] != [source["module"], source["sha256"]]
        or any(
            left != right
            for i, (left, right) in enumerate(zip(old["argv"], new["argv"], strict=True))
            if i not in (5, 6)
        )
        for old, new in zip(original, effective, strict=True)
    ):
        raise ValueError("engineering amendment changes scientific argv or profile fields")
    if json_sha256(effective) != review["argv_sha256"]:
        raise ValueError("effective controller argv differs from the reviewed exact argv")
    return {
        "artifact_kind": "xgraph.serving_recovery_engineering_amendment_request",
        "version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "canonical_child": str(child),
        "implementation_decision": decision,
        "source_review": review_pin,
        "original": {
            "registration": registration_pin,
            "registration_claim": registration["claim"],
            "preparation": metadata_identity(child / "preparation.json"),
            "preparation_claim": execution["preparation_claim"],
            "execution": execution_pin,
            "source": registration["source"],
            "native_sha256": json_sha256(execution["native"]),
            "schedule_sha256": json_sha256(registration["schedule"]),
            "argv_sha256": json_sha256(original),
            "decisions": registration["decisions"],
            "parent_lineage": registration["parent_lineage"],
        },
        "effective_source": source,
        "effective_argv": effective,
        "effective_argv_sha256": json_sha256(effective),
        "cap_seconds": registration["cap_seconds"],
        "parent_charged_gpu_seconds": registration["parent_charged_gpu_seconds"],
        "control_changes": {
            "physical_archive_bytes": 2147483648,
            "control_allowances": {"relay_setup_seconds": 120, "post_native_seconds": 1800},
            "state_proxy_bytes": 67108864,
            "frame_plaintext_bytes": 1048576,
            "canonical_archive_bytes": 17179869184,
            "frames": 16384,
        },
    }


def adopt_engineering_amendment(registration_path, review_path):
    """Append one amendment before the original Native lifetime has begun."""
    request = engineering_amendment_request(registration_path, review_path)
    child = Path(request["canonical_child"])
    path = child.parent / "engineering-amendment.json"
    claims = ClaimStore(ROOT)
    if any(
        candidate.exists() or candidate.is_symlink()
        for candidate in (path, path.with_suffix(".json.sha256"))
    ):
        raise FileExistsError("engineering amendment already present; no replacement")
    if request["effective_source"]["sha256"] != _SOURCE_BYTES_SHA256:
        raise ValueError("engineering adoption requires the reviewed source-bytes bootstrap")
    prior = (
        "launch.json",
        "budget.json",
        "run-status.json",
        "runtime-tmp",
        "specs",
        "coordination.json",
        "coordination-release.json",
        "quiet-admission.json",
        "handoffs",
        "dispatch",
        "receipts",
        "coverage",
        "native-admissions",
    )
    if any((child / name).exists() or (child / name).is_symlink() for name in prior):
        raise FileExistsError(
            "engineering adoption refuses prior Native launch/budget/attempt state"
        )
    registration = json.loads(Path(registration_path).read_bytes())
    prepared_names = {
        "original-study.json",
        "launch-commands.json",
        "runtime.json",
        "runtime.json.sha256",
        "trials.json",
        "trials.json.sha256",
    }
    for stage in registration["schedule"]:
        directory = Path(stage["output"]).parent
        allowed = set() if stage["kind"] == "overhead" else prepared_names
        if directory.exists() and (
            directory.is_symlink()
            or any(entry.name not in allowed for entry in directory.iterdir())
        ):
            raise FileExistsError(
                "engineering adoption refuses prior Native stage/attempt artifacts"
            )
    # ClaimStore atomically verifies that the shared original launch and budget
    # references are absent. Partial publication never rolls that claim back.
    commit = claims.create("engineering-amendment", request)
    if engineering_amendment_request(registration_path, review_path) != request:
        raise ValueError("engineering amendment changed after claim; retain the consumed claim")
    adopted = {
        "artifact_kind": "xgraph.serving_recovery_engineering_amendment",
        "version": 1,
        "request": request,
        "claim": {"commit": commit, "record": request},
    }
    immutable_json(path, adopted)
    return adopted


def verify_engineering_amendment(registration_path, pin):
    """Recheck the sealed amendment and original proofs, including after launch."""
    child = Path(load_protocol()["canonical_child"])
    path = child.parent / "engineering-amendment.json"
    if (
        not isinstance(pin, dict)
        or pin.get("path") != str(path)
        or metadata_identity(path) != pin
        or read_seal_digest(path.with_suffix(".json.sha256")) != pin["sha256"]
    ):
        raise ValueError("engineering amendment identity/seal changed")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != pin["sha256"]:
        raise ValueError("engineering amendment identity changed before decoding")
    adopted = json.loads(data)
    request = engineering_amendment_request(
        registration_path, child.parent / "engineering-source-review.json"
    )
    commit = ClaimStore(ROOT).verify("engineering-amendment", request)
    if adopted != {
        "artifact_kind": "xgraph.serving_recovery_engineering_amendment",
        "version": 1,
        "request": request,
        "claim": {"commit": commit, "record": request},
    }:
        raise ValueError("engineering amendment typed claim or source/original proof changed")
    originals = {}
    for name in ("registration", "execution"):
        original_pin = request["original"][name]
        original_data = Path(original_pin["path"]).read_bytes()
        if (
            len(original_data) != original_pin["bytes"]
            or hashlib.sha256(original_data).hexdigest() != original_pin["sha256"]
        ):
            raise ValueError("engineering amendment original metadata changed after verification")
        originals[name] = json.loads(original_data)
    return {"pin": pin, "request": request, "claim": adopted["claim"], **originals}


def verify_owner_release(registration_path, review_path):
    """Bind explicit owner-reviewed terminal controls; never rerun old checkers."""
    registration_path, review_path = Path(registration_path), Path(review_path)
    child = Path(load_protocol()["canonical_child"])
    if review_path != child.parent / "launch-review.json" or review_path.resolve() != review_path:
        raise ValueError("canonical owner launch review path changed")
    pin = metadata_identity(review_path)
    data = review_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != pin["sha256"]:
        raise ValueError("owner launch review changed before decoding")
    review = json.loads(data)
    if not isinstance(review, dict):
        raise ValueError("owner launch review must be a JSON object")
    amendment = None
    if "engineering_amendment" in review:
        amendment = verify_engineering_amendment(registration_path, review["engineering_amendment"])
        registration, execution = amendment["registration"], amendment["execution"]
        source = amendment["request"]["effective_source"]
        argv = amendment["request"]["effective_argv"]
    else:
        registration = verify_metadata_registration(registration_path)
        source, argv = registration["source"], registration["argv"]
    required = {
        "artifact_kind",
        "approved",
        "purpose",
        "protocol_sha256",
        "decisions",
        "controller_commit",
        "canonical_child",
        "execution",
        "registration",
        "argv_sha256",
        "cap_seconds",
        "parent_charged_gpu_seconds",
        "prerequisites",
        "comparison_blind",
        "native_owner_profile",
    }
    if amendment is not None:
        required.add("engineering_amendment")
    expected = {
        "artifact_kind": "xgraph.serving_recovery_launch_review",
        "approved": True,
        "purpose": "serving_recovery_launch",
        "protocol_sha256": PROTOCOL_SHA256,
        "decisions": registration["source"]["decisions"],
        "controller_commit": source["commit"],
        "canonical_child": str(child),
        "registration": metadata_identity(registration_path),
        "argv_sha256": json_sha256(argv),
        "cap_seconds": registration["cap_seconds"],
        "parent_charged_gpu_seconds": registration["parent_charged_gpu_seconds"],
        "comparison_blind": True,
    }
    if (
        not required <= review.keys()
        or set(review) - required - {"created_at"}
        or any(
            type(review.get(key)) is not type(value) or review[key] != value
            for key, value in expected.items()
        )
    ):
        raise ValueError(
            "owner review approval/source/argv/decisions/cap binding is incomplete or changed"
        )
    prerequisites = review["prerequisites"]
    roles = ("selected_webqsp", "corrected_assay")
    if not isinstance(prerequisites, dict) or set(prerequisites) != set(roles):
        raise ValueError("owner launch review needs both fixed terminal prerequisites")
    retained = {}
    for role in roles:
        original = prerequisites[role]
        path = child.parent / "prerequisites" / (role + ".json")
        if (
            not isinstance(original, dict)
            or original.get("path") != str(path)
            or metadata_identity(path) != original
            or read_seal_digest(path.with_suffix(".json.sha256")) != original["sha256"]
        ):
            raise ValueError("owner prerequisite fixed path/identity/seal changed")
        item = json.loads(path.read_bytes())
        if (
            item.get("artifact_kind") != "xgraph.serving_recovery_prerequisite"
            or item.get("role") != role
            or item.get("terminal_reviewed") is not True
            or item.get("disposition") not in ("processing_complete", "terminal_incomplete")
            or item.get("comparison_value_used") is not False
            or not isinstance(item.get("source_commit"), str)
            or re.fullmatch(r"[0-9a-f]{40}", item["source_commit"]) is None
        ):
            raise ValueError(
                "owner prerequisite terminal review or comparison-blind disposition is invalid"
            )
        retained[role] = item
    profile = review["native_owner_profile"]
    if (
        not isinstance(profile, dict)
        or not isinstance(profile.get("cpu_affinity"), list)
        or not profile["cpu_affinity"]
        or len(set(profile["cpu_affinity"])) != len(profile["cpu_affinity"])
        or any(type(value) is not int or value < 0 for value in profile["cpu_affinity"])
        or not isinstance(profile.get("address_space_limit"), list)
        or len(profile["address_space_limit"]) != 2
        or any(type(value) is not int or value < -1 for value in profile["address_space_limit"])
    ):
        raise ValueError("owner native launch affinity/address-space profile is incomplete")
    execution_path = child / "execution.json"
    if (
        review.get("execution", {}).get("path") != str(execution_path)
        or metadata_identity(execution_path) != review["execution"]
    ):
        raise ValueError("owner review execution identity changed")
    if amendment is None:
        execution = verify_native_execution(registration_path)
    if execution["argv_sha256"] != json_sha256(registration["argv"]):
        raise ValueError("owner review execution argv changed")
    release = {
        "release": pin,
        "review": review,
        "registration": registration,
        "execution": execution,
        "prerequisites": retained,
        "scope": "explicit source-qualified owner review of pinned terminal control metadata",
    }
    if amendment is not None:
        release.update(
            engineering_amendment=amendment, effective_source=source, effective_argv=argv
        )
    return release


def native_launch_resources(release):
    """Require prospective measured control/storage allocations, without guessing defaults."""
    registration, execution = release["registration"], release["execution"]
    amendment = None
    if "engineering_amendment" in release:
        amendment = verify_engineering_amendment(
            release["review"]["registration"]["path"], release["review"]["engineering_amendment"]
        )
        if (
            amendment != release["engineering_amendment"]
            or amendment["registration"] != registration
            or amendment["execution"] != execution
            or amendment["request"]["effective_source"] != _effective_source(release)
            or amendment["request"]["effective_argv"] != _effective_argv(release)
        ):
            raise ValueError("resource qualification engineering amendment/source binding changed")
    source = _effective_source(release)
    child = Path(registration["canonical_child"])
    pin = release["review"]["native_owner_profile"].get("resource_qualification")
    path = child.parent / "qualifications/native-control.json"
    if (
        not isinstance(pin, dict)
        or pin.get("path") != str(path)
        or metadata_identity(path) != pin
        or read_seal_digest(path.with_suffix(".json.sha256")) != pin["sha256"]
    ):
        raise ValueError(
            "UNBOUND production resource qualification: fixed path/bytes/seal required"
        )
    qualified = json.loads(path.read_bytes())
    if (
        qualified.get("artifact_kind") != "xgraph.serving_recovery_resource_qualification"
        or qualified.get("approved") is not True
        or qualified.get("production_values_bound") is not True
        or qualified.get("comparison_values_used") is not False
        or qualified.get("controller") != source
        or qualified.get("decisions") != execution["decisions"]
        or qualified.get("execution") != release["review"]["execution"]
        or not isinstance(qualified.get("source_only_measurement"), dict)
        or not qualified["source_only_measurement"]
    ):
        raise ValueError("production control/source qualification is incomplete or changed")
    if amendment is not None and qualified.get("engineering_amendment") != _engineering_seal(
        release
    ):
        raise ValueError("production qualification lacks the adopted engineering amendment")
    values = qualified.get("values")
    keys = {
        "control_allowances",
        "storage_limits",
        "cuda_compiler",
        "disk",
        "monitor_policy",
        "helpers",
        "scheduler",
    }
    if not isinstance(values, dict) or set(values) != keys:
        raise ValueError("production control allocation schema is incomplete")
    allowances = values["control_allowances"]
    if (
        not isinstance(allowances, dict)
        or set(allowances) != set(CONTROL_ALLOWANCE_FIELDS)
        or any(
            nonnegative(value, "qualified control allowance") <= 0 for value in allowances.values()
        )
    ):
        raise ValueError(
            "UNBOUND native setup/finalization allowances; positive qualification required"
        )
    limits = values["storage_limits"]
    ceilings = {
        "state_proxy_bytes": 67108864,
        "frame_plaintext_bytes": 8388608,
        "physical_archive_bytes": 1073741824,
        "canonical_archive_bytes": 17179869184,
        "frames": 65536,
    }
    if amendment is not None:
        ceilings["physical_archive_bytes"] = 2147483648
    if (
        not isinstance(limits, dict)
        or set(limits) != set(ceilings)
        or any(
            type(value) is not int or not 0 < value <= ceilings[key]
            for key, value in limits.items()
        )
    ):
        raise ValueError("qualified storage allocations exceed the adopted ceilings")
    if amendment is not None:
        changes = amendment["request"]["control_changes"]
        if allowances != changes["control_allowances"] or limits != {
            key: value for key, value in changes.items() if key != "control_allowances"
        }:
            raise ValueError(
                "qualified storage/control allowances differ from the engineering amendment"
            )
    disk = values["disk"]
    if (
        not isinstance(disk, dict)
        or set(disk)
        != {
            "native_and_log_bytes_per_trial",
            "remaining_preparation_bytes",
            "failure_and_receipt_bytes",
        }
        or any(type(value) is not int or value <= 0 for value in disk.values())
    ):
        raise ValueError("whole-block native/storage/receipt allocations are unqualified")
    policy = values["monitor_policy"]
    if (
        not isinstance(policy, dict)
        or policy.get("decision_sha256") != OS_BACKGROUND_SHA256
        or policy.get("context_decision_sha256") != OS_CONTEXT_SHA256
        or not isinstance(policy.get("roles"), list)
        or any(
            not isinstance(policy.get(key), str) or not policy[key]
            for key in ("boot_id", "pid_namespace", "gpu_uuid")
        )
    ):
        raise ValueError("qualified monitor policy lacks the host and adopted OS decisions")
    jobs = [job for stage in registration["schedule"] for job in stage["jobs"]]
    if (
        not isinstance(values["helpers"], dict)
        or set(values["helpers"]) != {job["id"] for job in jobs}
        or any(not isinstance(value, list) for value in values["helpers"].values())
    ):
        raise ValueError("qualified finite helper recipes do not cover the exact fixed schedule")
    required_helpers = native_metadata_helper_argvs(
        {
            "source_proof": source,
            "source": execution["native"]["source"],
            "environment": execution["native"]["environment"],
            **(
                {
                    "registered_source_proof": registration["source"],
                    "engineering_amendment": _engineering_seal(release),
                }
                if amendment is not None
                else {}
            ),
        }
    )
    for helpers in values["helpers"].values():
        declared = [
            helper.get("argv_sha256")
            for helper in helpers
            if isinstance(helper, dict)
            and helper.get("parent") == "worker"
            and helper.get("session", "inherit") == "inherit"
        ]
        if len(helpers) > 128 or any(
            commandline_sha256(argv) not in declared for argv in required_helpers
        ):
            raise ValueError(
                "qualified helpers omit an exact worker-child metadata/attestation recipe"
            )
    compiler = values["cuda_compiler"]
    if (
        not isinstance(compiler, dict)
        or "NOTFOUND" in str(compiler.get("path"))
        or artifact_identity(compiler["path"], maximum=128 * 1024**2) != compiler
    ):
        raise ValueError("qualified native compiler source identity changed")
    scheduler = values["scheduler"]
    if (
        not isinstance(scheduler, dict)
        or set(scheduler) != {"pid", "startticks", "script", "order", "mapping"}
        or any(
            type(scheduler.get(key)) is not int or scheduler[key] <= 0
            for key in ("pid", "startticks")
        )
        or any(
            not isinstance(scheduler.get(key), dict)
            or metadata_identity(scheduler[key]["path"]) != scheduler[key]
            for key in ("script", "order", "mapping")
        )
    ):
        raise ValueError(
            "qualified foreground scheduler/source/order/mapping binding is incomplete"
        )
    measurement = qualified["source_only_measurement"]
    capacity = measurement.get("archive_capacity")
    bounds = {
        "max_frame_plaintext_bytes": limits["frame_plaintext_bytes"],
        "max_state_proxy_bytes": limits["state_proxy_bytes"],
        "canonical_bytes": limits["canonical_archive_bytes"],
        "physical_bytes": limits["physical_archive_bytes"],
        "max_frames": limits["frames"],
    }
    longest = max(job["wall_seconds"] for job in jobs)
    minimum_frames = math.ceil(longest + CLEANUP_SECONDS + sum(allowances.values())) + 3
    if (
        not isinstance(capacity, dict)
        or capacity.get("qualified_before_outcomes") is not True
        or capacity.get("controller_source_sha256") != json_sha256(source)
        or capacity.get("maximum_native_seconds") != longest
        or capacity.get("nominal_interval_seconds") != 1
        or any(
            type(capacity.get(key)) is not int or not 0 < capacity[key] <= ceiling
            for key, ceiling in bounds.items()
        )
        or capacity["max_frames"] < minimum_frames
    ):
        raise ValueError(
            "source-qualified archive shape/capacity/longest-watchdog proof is UNBOUND"
        )
    return {
        **copy.deepcopy(values),
        "qualification": pin,
        "measurement": copy.deepcopy(measurement),
    }


def _effective_source(release):
    """Select source from an already verified release, without changing its registration."""
    if "engineering_amendment" in release:
        if "effective_source" not in release:
            raise ValueError("verified engineering release lacks its effective controller")
        return release["effective_source"]
    return release["registration"]["source"]


def _effective_argv(release):
    if "engineering_amendment" in release:
        if "effective_argv" not in release:
            raise ValueError("verified engineering release lacks its effective relay argv")
        return release["effective_argv"]
    return release["registration"]["argv"]


def _engineering_seal(release):
    amendment = release["engineering_amendment"]
    return {"seal": copy.deepcopy(amendment["pin"]), "claim": amendment["claim"]["commit"]}


def native_budget_binding(release, resources):
    registration, native = release["registration"], release["execution"]["native"]
    binding = {
        "artifact_kind": NATIVE_BUDGET_KIND,
        "cap_seconds": registration["cap_seconds"],
        "require_receipts": True,
        "registration": release["review"]["registration"],
        "execution": release["review"]["execution"],
        "owner_release": release["release"],
        "controller_source_sha256": json_sha256(_effective_source(release)),
        "native_source_sha256": json_sha256(native["source"]),
        "native_environment_sha256": json_sha256(native["environment"]),
        "schedule_sha256": json_sha256(registration["schedule"]),
        "control_envelope_decision_sha256": CONTROL_ENVELOPE_SHA256,
        "control_allowances": resources["control_allowances"],
        "allowance_qualification": resources["qualification"],
    }
    if "engineering_amendment" in release:
        binding.update(
            original_controller_source_sha256=json_sha256(registration["source"]),
            engineering_amendment=_engineering_seal(release),
        )
    return binding


def native_launch_record(release, controller):
    record = {
        "protocol_sha256": PROTOCOL_SHA256,
        "decisions": release["execution"]["decisions"],
        "registration": release["review"]["registration"],
        "execution": release["review"]["execution"],
        "owner_release": release["release"],
        "controller": copy.deepcopy(controller),
        "argv_sha256": release["execution"]["argv_sha256"],
        "resource_qualification": release["review"]["native_owner_profile"][
            "resource_qualification"
        ],
    }
    if "engineering_amendment" in release:
        record.update(
            registered_argv_sha256=record["argv_sha256"],
            argv_sha256=json_sha256(_effective_argv(release)),
            effective_source=copy.deepcopy(_effective_source(release)),
            engineering_amendment=_engineering_seal(release),
        )
    return record


def commandline_sha256(argv):
    return hashlib.sha256(("\0".join(argv) + "\0").encode()).hexdigest()


def process_commandline_sha256(pid=None):
    path = Path("/proc") / str(os.getpid() if pid is None else pid) / "cmdline"
    with path.open("rb") as stream:
        data = stream.read(128 * 1024 + 1)
    if not data or len(data) > 128 * 1024 or not data.endswith(b"\0"):
        raise ValueError("live process command identity is unavailable or oversized")
    return hashlib.sha256(data).hexdigest()


def expected_server_control(command, job, capacity):
    def value(flag):
        if command.count(flag) != 1:
            raise ValueError("sealed server command has ambiguous " + flag)
        return command[command.index(flag) + 1]

    environment = [command[i + 1] for i, token in enumerate(command) if token == "--env"]
    settings = dict(token.split("=", 1) for token in environment if token.startswith("XGRAPH_"))
    cache = "--enable-prefix-caching" in command
    if cache != (job["trial"]["cache"] != "off"):
        raise ValueError("sealed server command prefix-cache contract changed")
    return {
        "kv_cache_memory_bytes": int(value("--kv-cache-memory-bytes")),
        "enable_prefix_caching": cache,
        "max_model_len": int(value("--max-model-len")),
        "max_num_seqs": int(value("--max-num-seqs")),
        "kv_capacity_tokens": capacity,
        "engine_settings": settings,
    }


def native_context(release, resources, budget, job):
    """Derive only registered metadata; study decoding belongs to the admitted relay."""
    registration, execution = release["registration"], release["execution"]
    child = Path(registration["canonical_child"])
    planned = next(
        (
            item
            for stage in registration["schedule"]
            for item in stage["jobs"]
            if item["id"] == job["id"]
        ),
        None,
    )
    if planned != job:
        raise ValueError("native context job differs from the immutable fixed schedule")
    native = execution["native"]
    source, environment = native["source"], native["environment"]
    destination = Path(source["root"])
    profile = native_profile(job)
    temporary, cwd = child / "runtime-tmp", destination / "target/native-launch-cwd"
    temporary.mkdir(exist_ok=True)
    cwd.mkdir(exist_ok=True)
    if any(cwd.iterdir()):
        raise ValueError("native module cwd is not the attested empty directory")
    owner_profile = release["review"]["native_owner_profile"]
    address_space = (
        [profile["address_space_bytes"]] * 2
        if profile["address_space_bytes"] is not None
        else owner_profile["address_space_limit"]
    )
    argv = next(item["argv"] for item in _effective_argv(release) if item["id"] == job["id"])
    intent = {
        "kind": "overhead" if job["kind"] == "overhead" else "serving",
        "job_id": job["id"],
        "output": job["output"],
        "worker_argv_sha256": commandline_sha256(argv),
        "helpers": copy.deepcopy(resources["helpers"][job["id"]]),
    }
    policy = {
        **copy.deepcopy(resources["monitor_policy"]),
        "execution_id": str(child),
        "source_sha256": json_sha256(_effective_source(release)),
        "storage_limits": resources["storage_limits"],
        "decisions": {
            "protocol": PROTOCOL_SHA256,
            "runtime_transport": RUNTIME_BOUNDARY_SHA256,
            "os_background": OS_BACKGROUND_SHA256,
            "os_context": OS_CONTEXT_SHA256,
            "parent_lineage": PARENT_LINEAGE_SHA256,
            "storage": BOUNDED_STORAGE_SHA256,
            "control_envelope": CONTROL_ENVELOPE_SHA256,
        },
    }
    context = {
        "job": job,
        "source": source,
        "environment": environment,
        "wheel": native["wheel"],
        "source_proof": _effective_source(release),
        "argv": argv,
        "cwd": str(cwd),
        "profile": {
            "nice": profile["nice"],
            "cpu_affinity": owner_profile["cpu_affinity"],
            "address_space_limit": address_space,
        },
        "runtime_environment": native_runtime_environment(
            profile,
            source=destination,
            temporary=temporary,
            cuda_compiler=resources["cuda_compiler"]["path"],
            parent=os.environ,
        ),
        "policy": policy,
        "intent": intent,
        "controller": budget.controller_identity,
        "control_allowances": resources["control_allowances"],
        "phase_tracking": True,
        "actual_kv_capacity_tokens": budget.data["calibrations"]
        .get(job["stage"], {})
        .get("kv_capacity_tokens"),
        "server_name": None,
        "expected_effective": None,
        "expected_overhead": None,
        "native_packages": native["attestation"]["packages"],
        "setup_deadline": budget.data["attempts"][budget.data["active"]]["started_monotonic"]
        + resources["control_allowances"]["relay_setup_seconds"],
    }
    if "engineering_amendment" in release:
        policy["decisions"]["storage_amendment"] = ENGINEERING_AMENDMENT_SHA256
        context.update(
            registered_source_proof=registration["source"],
            engineering_amendment=_engineering_seal(release),
        )
    if job["kind"] == "overhead":
        primary_stage = next(
            stage
            for stage in registration["schedule"]
            if stage["name"] == "locked-" + job["stage"].removeprefix("overhead-")
        )
        primary = next(
            item
            for item in primary_stage["jobs"]
            if (
                item["trial"]["repeat"],
                item["trial"]["arm"],
                item["trial"]["cache"],
                item["trial"]["concurrency"],
            )
            == (0, "finalist", "warm", 8)
        )
        completed = {
            item["id"] for item in budget.data["attempts"] if item["status"] == "completed"
        }
        if not {item["id"] for item in primary_stage["jobs"]} <= completed:
            raise ValueError("overhead depends on all twenty new qualified family primary cells")
        screen = Path(primary["output"]).parent
        context.update(
            primary_job=primary,
            primary_source=artifact_identity(primary["output"]),
            primary_study=artifact_identity(screen / "study.json", maximum=128 * 1024**2),
            primary_plan=metadata_identity(screen / "trials.json"),
        )
    else:
        stage = next(stage for stage in registration["schedule"] if stage["name"] == job["stage"])
        retained = next(item for item in execution["stages"] if item["name"] == job["stage"])
        command = relocated_server_command(stage, job, retained)
        image = command[command.index("--entrypoint") + 2]
        intent["server"] = {
            "name": retained["server_names"][job["id"]],
            "image": image,
            "labels": {"xgraph.serving-output": str(Path(job["output"]).parent)},
        }
        basis = "locked-" + job["stage"].rsplit("-", 1)[1]
        lineage = registration["parent_lineage"]["inputs"]
        context.update(
            wheel=retained["runtime_wheel"],
            server_command=command,
            server_name=intent["server"]["name"],
            original_study=retained["original_study"],
            runtime_json=retained["runtime_json"],
            parent_runtime=json.loads(
                Path(lineage[basis + "-runtime"]["snapshot"]["path"]).read_bytes()
            ),
            parent_runtime_archive=lineage[basis + "-source"],
            expected_effective=expected_server_control(
                command, job, context["actual_kv_capacity_tokens"]
            ),
        )
    return context


def native_context_binding(context):
    return {
        key: copy.deepcopy(value)
        for key, value in context.items()
        if key not in ("runtime_environment", "specification", "handoff_nonce")
    }


class NativeExecutor:
    """One owner-qualified dispatch route; no resume or alternative model launcher."""

    def __init__(
        self, release, resources, budget, launch, *, clock=time.monotonic, sleep=time.sleep
    ):
        self.release, self.resources, self.budget, self.launch = release, resources, budget, launch
        self.clock, self.sleep = clock, sleep
        self.collector = load_controller_monitor(_effective_source(release))
        self.lease = None
        (budget.path.parent / "runtime-tmp").mkdir(exist_ok=True)
        self.last_cleanup = {"confirmed_absent": False, "errors": ["no supervised owned interval"]}

    def prepare_context(self, job, attempt):
        self.budget.verify()
        context = native_context(self.release, self.resources, self.budget, job)
        path = self.budget.path.parent / "specs" / (job["id"].replace("/", "-") + ".json")
        path.parent.mkdir(exist_ok=True)
        nonce = secrets.token_hex(32)
        spec = {
            "artifact_kind": "xgraph.serving_recovery_native_specification",
            "version": 2 if "engineering_amendment" in self.release else 1,
            "registration": self.release["review"]["registration"],
            "owner_release": self.release["release"],
            "execution": self.release["review"]["execution"],
            "job_id": job["id"],
            "controller": self.budget.controller_identity,
            "launch": self.launch,
            "attempt": attempt,
            "context": native_context_binding(context),
            "nonce": nonce,
        }
        if "engineering_amendment" in self.release:
            spec["engineering_amendment"] = _engineering_seal(self.release)
        immutable_json(path, spec)
        context.update(specification=metadata_identity(path), handoff_nonce=nonce)
        return context

    def admit_block(self, jobs):
        self.budget.verify()
        disk = self.resources["disk"]
        required = (
            len(jobs)
            * (
                self.resources["storage_limits"]["physical_archive_bytes"]
                + disk["native_and_log_bytes_per_trial"]
            )
            + disk["remaining_preparation_bytes"]
            + disk["failure_and_receipt_bytes"]
            + 20 * 1024**3
        )
        if shutil.disk_usage(self.budget.path.parent).free < required:
            self.budget.data["status"] = "INCOMPLETE"
            self.budget.data["block"] = None
            self.budget.write()
            raise RuntimeError(
                "whole fixed block lacks qualified archive/staging/failure space plus reserve"
            )

    def admit_job(self, job):
        # Original serving headroom remains before its lifetime charge; overhead
        # still executes its own original headroom inside its native stage.
        fresh = verify_owner_release(
            self.release["review"]["registration"]["path"], self.release["release"]["path"]
        )
        if fresh != self.release or native_launch_resources(fresh) != self.resources:
            raise ValueError("native owner/source/resource qualification changed before a stage")
        native = self.release["execution"]["native"]
        destination = Path(native["source"]["root"])
        profile = native_profile(job)
        environment = native_runtime_environment(
            profile,
            source=destination,
            temporary=self.budget.path.parent / "runtime-tmp",
            cuda_compiler=self.resources["cuda_compiler"]["path"],
            parent=os.environ,
        )
        subprocess.run(
            ["bash", str(destination / "scripts/gpu-headroom.sh"), str(profile["need_gib"])],
            cwd=destination,
            env=environment,
            timeout=43200,
            check=True,
        )

    def _coordinate(self, context, _monitor, first, second):
        capacity = self.resources["measurement"]["archive_capacity"]
        projection = _monitor.retention_estimate(
            max(13476, context["job"]["wall_seconds"])
            + CLEANUP_SECONDS
            + sum(context["control_allowances"].values()),
            interval_s=1,
        )
        if (
            projection["observed_max_frame_json_bytes"] > capacity["max_frame_plaintext_bytes"]
            or projection["retained_json_bytes"] > capacity["max_state_proxy_bytes"]
            or projection["interval_samples_including_boundaries"] > capacity["max_frames"]
        ):
            raise ValueError(
                "current baseline exceeds the prospectively qualified storage shape/capacity"
            )
        cap_path = (
            self.budget.path.parent
            / "coverage"
            / context["job"]["id"].replace("/", "-")
            / "capacity-admission.json"
        )
        immutable_json(
            cap_path,
            {
                "resource_qualification": self.resources["qualification"],
                "qualified_capacity": capacity,
                "current_projection": projection,
                "projection_status_changed": False,
                "scope": (
                    "selected qualified allocations and baseline checks; no completion guarantee"
                ),
            },
        )
        if self.lease is not None:
            return
        path = self.budget.path.parent / "quiet-admission.json"
        immutable_json(
            path,
            {
                "artifact_kind": "xgraph.serving_recovery_quiet_admission",
                "policy_sha256": json_sha256(context["policy"]),
                "intent_sha256": json_sha256(context["intent"]),
                "samples": [first, second],
                "controller": context["controller"],
            },
        )
        request = {
            "execution": self.release["review"]["execution"],
            "controller_commit": _effective_source(self.release)["commit"],
            "controller_source_sha256": _effective_source(self.release)["sha256"],
            "decisions": decision_bindings(),
            "scheduler": self.resources["scheduler"],
            "quiet_qualification": {
                "qualified": True,
                "at": second["at"],
                "os_background_refinement_sha256": OS_BACKGROUND_SHA256,
                "boot_id": second["boot_id"],
                "pid_namespace": second["pid_namespace"],
                "archive": artifact_identity(path, maximum=128 * 1024**2),
            },
        }
        if "engineering_amendment" in self.release:
            request["engineering_amendment"] = _engineering_seal(self.release)
        self.lease = CoordinationLease(request, clock=self.clock)
        self.lease.acquire()

    def execute(self, job, attempt):
        self.last_cleanup = {
            "confirmed_absent": False,
            "errors": ["current attempt has no retained cleanup proof"],
        }
        context = self.prepare_context(job, attempt)
        directory = self.budget.path.parent / "coverage" / job["id"].replace("/", "-")
        directory.mkdir(parents=True)
        policy, intent = context["policy"], context["intent"]
        bindings = {
            "execution_id": policy["execution_id"],
            "trial_id": intent["job_id"],
            "source_sha256": policy["source_sha256"],
            "policy_sha256": json_sha256(policy),
            "intent_sha256": hashlib.sha256(
                json.dumps(intent, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest(),
            "decisions": policy["decisions"],
        }
        writer = self.collector.ArchiveWriter(
            directory, limits=policy["storage_limits"], bindings=bindings
        )
        monitor = self.collector.Monitor(policy, intent, frame_sink=writer)
        # ABI/import shape is checked before native GO; it consumes setup S.
        try:
            attest_native_module_launch(
                context["source"]["root"],
                context["source"],
                context["environment"],
                cwd=context["cwd"],
            )
            result = supervise_native_attempt(
                context,
                self.budget,
                attempt,
                monitor=monitor,
                writer=writer,
                clock=self.clock,
                sleep=self.sleep,
                admitted=lambda monitor, first, second: self._coordinate(
                    context, monitor, first, second
                ),
            )
            closed = self.budget.data["attempts"][-1]
            receipt = json.loads(Path(closed["receipt"]["path"]).read_bytes())
            self.last_cleanup = receipt["cleanup"]
            return result
        except BaseException:
            self.last_cleanup = {
                "confirmed_absent": False,
                "errors": ["current attempt cleanup is unconfirmed"],
            }
            raise
        finally:
            writer.close()

    def close(self):
        if self.lease is not None:
            if self.lease.descriptor is None:
                return False
            try:
                self.lease.release(self.last_cleanup)
            finally:
                self.lease.close()
        return True


def run_native_recovery(registration_path, review_path, *, clock=time.monotonic):
    """Run the one fully bound prospective execution, preserving every prior charge."""
    release = verify_owner_release(registration_path, review_path)
    if _effective_source(release)["sha256"] != _SOURCE_BYTES_SHA256:
        raise ValueError("actual recovery requires the reviewed source-bytes controller bootstrap")
    resources = native_launch_resources(release)
    source = _effective_source(release)
    if (
        str(Path(__file__).resolve()) != source["module"]
        or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != source["sha256"]
    ):
        raise ValueError("native relay actual effective source identity changed")
    controller = controller_process_identity()
    policy = resources["monitor_policy"]
    if any(controller[key] != policy[key] for key in ("boot_id", "pid_namespace")):
        raise ValueError("owner-qualified monitor policy belongs to a different host clock domain")
    foreground = verify_scheduler_parent(resources["scheduler"])
    child = Path(release["registration"]["canonical_child"])
    claims = ClaimStore(ROOT)
    record = native_launch_record(release, controller)
    launch = {"commit": claims.create("launch", record), "record": record}
    budget = RecoveryBudget(
        child / "budget.json",
        native_budget_binding(release, resources),
        {"charged_gpu_seconds": release["registration"]["parent_charged_gpu_seconds"]},
        clock=clock,
        claims=claims,
        controller_identity=controller,
        create=True,
    )
    immutable_json(
        child / "launch.json",
        {
            "artifact_kind": "xgraph.serving_recovery_launch",
            "claim": launch,
            "foreground": foreground,
            "controller": controller,
            "owner_release": release["release"],
            "resource_qualification": resources["qualification"],
        },
    )
    executor, failure, endpoints = None, None, None
    released, release_error = False, None
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(_number, _frame):
        raise KeyboardInterrupt("serving recovery owner termination")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        executor = NativeExecutor(release, resources, budget, launch, clock=clock)
        endpoints = run_qualified_schedule(
            release["registration"]["schedule"],
            budget,
            executor.execute,
            admit_block=executor.admit_block,
            admit_job=executor.admit_job,
        )
    except BaseException as error:
        failure = {
            "kind": type(error).__name__,
            "message": "owned execution stopped; no retry or resume",
        }
        if budget.data["status"] == "ACTIVE" and budget.data["active"] is None:
            budget.data["status"], budget.data["block"] = "INCOMPLETE", None
            budget.write()
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        if executor is not None:
            try:
                released = executor.close() is not False
            except (OSError, ValueError) as error:
                release_error = {
                    "kind": type(error).__name__,
                    "message": "coordination retained for owner reconciliation",
                }
        if endpoints is None:
            completed = [
                item["id"] for item in budget.data["attempts"] if item["status"] == "completed"
            ]
            endpoints = endpoint_eligibility(release["registration"]["schedule"], completed)
        result = {
            "artifact_kind": "xgraph.serving_recovery_run_status",
            "endpoints": endpoints,
            "budget_status": budget.data["status"],
            "parent_charged_gpu_seconds": budget.data["parent_charged_gpu_seconds"],
            "charged_gpu_seconds": budget.data["charged_gpu_seconds"],
            "cap_seconds": budget.binding["cap_seconds"],
            "launch": launch,
            "failure": failure,
            "coordination_released": released,
            "coordination_error": release_error,
            "scientific_estimates_computed": False,
        }
        immutable_json(child / "run-status.json", result)
    return result


def controller_process_identity(pid=None):
    """Read the host boot/namespace/PID-start identity; no process action."""
    pid = os.getpid() if pid is None else pid
    path = Path("/proc") / str(pid)
    text = (path / "stat").read_text()
    fields = text.rsplit(")", 1)[1].split()
    identity = {
        "pid": pid,
        "startticks": int(fields[19]),
        "pgid": int(fields[2]),
        "sid": int(fields[3]),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "pid_namespace": os.readlink(path / "ns/pid"),
    }
    process_identity(identity)
    return identity


def verify_scheduler_parent(scheduler):
    """Corroborate a foreground master queue; this grants no process permission."""
    own = controller_process_identity()
    expected = controller_process_identity(scheduler["pid"])
    if process_identity(expected) != process_identity(scheduler) or any(
        expected[key] != own[key] for key in ("boot_id", "pid_namespace")
    ):
        raise ValueError("approved scheduler is not the same live host process identity")
    chain, pid = [], own["pid"]
    for _ in range(32):
        identity = controller_process_identity(pid)
        path = Path("/proc") / str(pid) / "stat"
        parent = int(path.read_text().rsplit(")", 1)[1].split()[1])
        if controller_process_identity(pid) != identity:
            raise ValueError("scheduler ancestry raced a process replacement")
        chain.append(identity)
        if pid == scheduler["pid"]:
            return {
                "scheduler": copy.deepcopy(scheduler),
                "chain": chain,
                "scope": "foreground ancestry only; monitor independently classifies every process",
            }
        if parent <= 0 or parent == pid:
            break
        pid = parent
    raise ValueError("serving must be a foreground child of the bound master scheduler")


def signal_bound_container(monitor, snapshot, server, *, signal_number, deadline, clock):
    """Signal one previously observed container by full CID, never a name lookup."""
    if signal_number not in (signal.SIGTERM, signal.SIGKILL):
        raise ValueError("owned container cleanup supports only TERM or KILL")
    binding = snapshot.get("container_binding")
    if (
        not isinstance(binding, dict)
        or not isinstance(server, dict)
        or re.fullmatch(r"[0-9a-f]{64}", binding.get("cid", "")) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", binding.get("image", "")) is None
        or binding.get("boot_id") != snapshot.get("boot_id")
        or binding.get("name") != server.get("name")
        or binding.get("image_ref") != server.get("image")
        or (server.get("image_id") is not None and server["image_id"] != binding.get("image"))
        or not isinstance(server.get("labels"), dict)
        or not server["labels"]
    ):
        raise ValueError("owned container needs its prior CID/image and source intent")
    if clock() >= deadline:
        raise TimeoutError("owned cleanup deadline exhausted before container inspect")
    shape = (
        '{"Id":{{json .Id}},"Name":{{json .Name}},"Image":{{json .Image}},'
        '"Config":{"Image":{{json .Config.Image}},"Labels":{{json .Config.Labels}}},'
        '"State":{"Running":{{json .State.Running}},"Pid":{{json .State.Pid}}}}'
    )
    inspect = ("docker", "inspect", "--type", "container", "--format", shape, binding["cid"])
    observed = json.loads(monitor.query(inspect))
    receipt = {
        "cid": binding["cid"],
        "signal": signal_number,
        "inspect_command": list(inspect),
        "observed": observed,
        "cleanup_confirmed": False,
    }
    if observed is None:
        return {**receipt, "status": "observed-absent"}
    if (
        not isinstance(observed, dict)
        or observed.get("Id") != binding["cid"]
        or observed.get("Name") != "/" + server["name"]
        or observed.get("Image") != binding["image"]
        or not isinstance(observed.get("Config"), dict)
        or observed["Config"].get("Image") != server["image"]
        or not isinstance(observed["Config"].get("Labels"), dict)
        or any(observed["Config"]["Labels"].get(k) != v for k, v in server["labels"].items())
        or not isinstance(observed.get("State"), dict)
        or type(observed["State"].get("Running")) is not bool
        or type(observed["State"].get("Pid")) is not int
        or observed["State"]["Pid"] != (binding["pid"] if observed["State"]["Running"] else 0)
    ):
        raise ValueError("fresh container inspect differs from its bound identity/source intent")
    if not observed["State"]["Running"]:
        return {**receipt, "status": "observed-exited"}
    remaining = deadline - clock()
    if remaining <= 0:
        raise TimeoutError("owned cleanup deadline exhausted before container signal")
    command = [
        "docker",
        "kill",
        "--signal",
        signal.Signals(signal_number).name.removeprefix("SIG"),
        binding["cid"],
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=min(2, remaining), check=False
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {
            **receipt,
            "status": "signal-unconfirmed",
            "command": command,
            "error": {"kind": type(error).__name__},
        }
    return {
        **receipt,
        "status": "signal-acknowledged" if result.returncode == 0 else "signal-failed",
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout[:4096],
        "stderr": result.stderr[:4096],
    }


def stop_proven_owned(
    monitor, *, signal_number, server=None, deadline=None, clock=time.monotonic, known_snapshot=None
):
    """Signal proved host identities by PID handles and bound containers by CID.

    A successful signal never proves cleanup. The caller must retain this
    receipt, obtain fresh monitor cleanup evidence, and stop on uncertainty.
    Unknown processes and the scheduler are never inferred to be owned.
    """
    if signal_number not in (signal.SIGTERM, signal.SIGKILL):
        raise ValueError("owned cleanup supports only TERM or KILL")
    retained = {
        "signaled": [],
        "already_absent": [],
        "errors": [],
        "cleanup_confirmed": False,
        "snapshot_at": None,
        "container_actions": [],
        "signal": signal_number,
    }
    try:
        snapshot = monitor.sample()
    except (OSError, ValueError) as error:
        retained["errors"].append({"field": "fresh_inventory", "kind": type(error).__name__})
        if known_snapshot is None:
            return retained
        # A stale inventory never grants a host PID signal. Its previously bound
        # immutable CID may only be used after a fresh trusted Docker inspect.
        snapshot = {**known_snapshot, "processes": []}
    retained["snapshot_at"] = snapshot.get("at")
    server = server if server is not None else getattr(monitor, "_intent", {}).get("server")
    if snapshot.get("container_binding") is not None:
        try:
            action = signal_bound_container(
                monitor,
                snapshot,
                server,
                signal_number=signal_number,
                deadline=clock() + 4 if deadline is None else deadline,
                clock=clock,
            )
            retained["container_actions"].append(action)
            if action.get("returncode", 0) != 0 or "error" in action:
                retained["errors"].append({"field": "container_signal", "kind": "DockerExit"})
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
            retained["container_actions"].append(
                {
                    "cid": snapshot["container_binding"].get("cid"),
                    "signal": signal_number,
                    "status": "refused",
                    "cleanup_confirmed": False,
                    "error": {"kind": type(error).__name__},
                }
            )
            retained["errors"].append({"field": "container_signal", "kind": type(error).__name__})
    for process in snapshot["processes"]:
        if not process.get("classification", "").startswith("owned-"):
            continue
        proof = process.get("ownership", {})
        if proof.get("origin") == "container":
            # Its inner PID namespace is not the host namespace. Only the bound
            # full-CID Docker action above can stop this resource.
            continue
        try:
            if deadline is not None and clock() >= deadline:
                raise TimeoutError("owned cleanup deadline exhausted before host signal")
            identity = process_identity(process)
            if (
                proof.get("origin") not in ("worker", "helper", "container")
                or process_identity(proof) != identity
                or proof.get("boot_id") != snapshot["boot_id"]
                or any(
                    type(process.get(key)) is not int
                    or process[key] <= 0
                    or proof.get(key) != process[key]
                    for key in ("pgid", "sid")
                )
                or type(process.get("present_at_end")) is not bool
            ):
                raise ValueError("owned cleanup identity proof is incomplete")
            if not process["present_at_end"]:
                continue
            descriptor = open_pid_handle(identity[0])
            try:
                observed = controller_process_identity(identity[0])
                if (
                    any(
                        observed[key] != process[key]
                        for key in ("pid", "startticks", "pgid", "sid")
                    )
                    or observed["boot_id"] != snapshot["boot_id"]
                    or observed["pid_namespace"] != snapshot["pid_namespace"]
                ):
                    raise ValueError("owned cleanup PID identity changed before signaling")
                signal_pid_handle(descriptor, signal_number)
                retained["signaled"].append(
                    {"pid": identity[0], "startticks": identity[1], "signal": signal_number}
                )
            finally:
                os.close(descriptor)
        except ProcessLookupError:
            retained["already_absent"].append(
                {"pid": process["pid"], "startticks": process["startticks"]}
            )
        except (ValueError, OSError, KeyError, TypeError) as error:
            retained["errors"].append({"pid": process.get("pid"), "kind": type(error).__name__})
    return retained


def open_pid_handle(pid):
    """Use the Linux PID handle API even when this Python lacks its wrapper."""
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    libc = ctypes.CDLL(None, use_errno=True)
    function = libc.pidfd_open
    function.argtypes, function.restype = [ctypes.c_int, ctypes.c_uint], ctypes.c_int
    descriptor = function(pid, 0)
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return descriptor


def signal_pid_handle(descriptor, number):
    if hasattr(signal, "pidfd_send_signal"):
        return signal.pidfd_send_signal(descriptor, number)
    libc = ctypes.CDLL(None, use_errno=True)
    function = libc.pidfd_send_signal
    function.argtypes, function.restype = (
        [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint],
        ctypes.c_int,
    )
    if function(descriptor, number, None, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return None


def supervise_overhead_stage(command, *, wall_seconds, **kwargs):
    """Equivalent frozen stage watchdog, retaining cleanup on interruption.

    The unchanged post_queue overhead/bench command runs in this newly created
    session. This supervisor owns its direct process object and birth identity;
    it never discovers ownership from an arbitrary PID or a process name.
    Successful exit still requires the external inventory's cleanup proof.
    """
    nonnegative(wall_seconds, "overhead original wall bound")
    if wall_seconds <= 0:
        raise ValueError("overhead stage wall bound must be positive")
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
    process, birth, descriptor = None, None, None
    extra_setup = kwargs.pop("preexec_fn", None)

    def child_setup():
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
        if extra_setup is not None:
            extra_setup()

    try:
        try:
            process = subprocess.Popen(
                command, start_new_session=True, preexec_fn=child_setup, **kwargs
            )
            descriptor = open_pid_handle(process.pid)
            birth = controller_process_identity(process.pid)
            if birth["pid"] != birth["pgid"] or birth["pid"] != birth["sid"]:
                raise ValueError("owned overhead stage did not create its independent session")
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)
        return subprocess.CompletedProcess(command, process.wait(timeout=wall_seconds))
    except BaseException as original:
        if process is not None:
            if birth is not None:
                try:
                    current = controller_process_identity(process.pid)
                except FileNotFoundError:
                    current = None
                if current is not None and current != birth:
                    raise ValueError(
                        "owned overhead session birth changed; no foreign signal"
                    ) from original
                if current is not None:
                    with suppress(ProcessLookupError):
                        os.killpg(birth["pgid"], signal.SIGTERM)
            elif descriptor is not None:
                # A handle obtained directly at our spawn is safe to terminate,
                # but unavailable group identity never grants a group signal.
                with suppress(ProcessLookupError):
                    signal_pid_handle(descriptor, signal.SIGTERM)
            try:
                process.wait(timeout=150)
            except subprocess.TimeoutExpired:
                if birth is not None:
                    current = controller_process_identity(process.pid)
                    if current != birth:
                        raise ValueError(
                            "owned overhead process identity changed before kill"
                        ) from original
                    with suppress(ProcessLookupError):
                        os.killpg(birth["pgid"], signal.SIGKILL)
                elif descriptor is not None:
                    signal_pid_handle(descriptor, signal.SIGKILL)
                process.wait()
            if birth is not None:
                # Like the frozen supervisor, reap a surviving forked group
                # even when its direct parent already exited after TERM.
                try:
                    current = controller_process_identity(process.pid)
                except FileNotFoundError:
                    current = None
                if current is not None and current != birth:
                    raise ValueError(
                        "overhead group leader was replaced; no foreign group signal"
                    ) from original
                with suppress(ProcessLookupError):
                    os.killpg(birth["pgid"], signal.SIGKILL)
        if isinstance(original, subprocess.TimeoutExpired):
            return subprocess.CompletedProcess(command, 124)
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


class CoordinationLease:
    """Own one TIMING marker and the frozen host flock, without granting isolation.

    The bound executor must separately qualify the live monitor and predecessor
    predicates. This class only verifies and retains the coordination identities;
    it never changes a scheduler route or terminates a process.
    """

    def __init__(self, request, *, clock=time.monotonic):
        self.request = copy.deepcopy(request)
        self.clock = clock
        self.descriptor = None
        self.receipt = None
        self.receipt_pin = None
        self.marker_bytes = None

    def acquire(self):
        child = Path(load_protocol()["canonical_child"])
        receipt_path = child / "coordination.json"
        if (
            receipt_path.exists()
            or receipt_path.is_symlink()
            or receipt_path.with_suffix(".json.sha256").exists()
        ):
            raise FileExistsError("serving coordination receipt is create-once")
        request = self.request
        execution = request["execution"]
        if (
            Path(execution["path"]) != child / "execution.json"
            or metadata_identity(execution["path"]) != execution
            or re.fullmatch(r"[0-9a-f]{40}", request.get("controller_commit", "")) is None
            or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            != request.get("controller_source_sha256")
            or request.get("decisions") != decision_bindings()
        ):
            raise ValueError("coordination controller or canonical execution binding changed")
        if "engineering_amendment" in request:
            amended = verify_engineering_amendment(
                child.parent / "registration.json", request["engineering_amendment"]["seal"]
            )
            if (
                request["engineering_amendment"]
                != {"seal": amended["pin"], "claim": amended["claim"]["commit"]}
                or request["controller_commit"] != amended["request"]["effective_source"]["commit"]
                or request["controller_source_sha256"]
                != amended["request"]["effective_source"]["sha256"]
            ):
                raise ValueError("coordination engineering amendment/effective source changed")
        identity = controller_process_identity()
        scheduler = request["scheduler"]
        observed = controller_process_identity(scheduler["pid"])
        if (
            process_identity(observed) != process_identity(scheduler)
            or observed["boot_id"] != identity["boot_id"]
            or observed["pid_namespace"] != identity["pid_namespace"]
            or any(
                metadata_identity(scheduler[key]["path"]) != scheduler[key]
                for key in ("script", "order", "mapping")
            )
        ):
            raise ValueError("scheduler process/source/order/mapping binding changed")
        quiet = request["quiet_qualification"]
        age = self.clock() - nonnegative(quiet["at"], "quiet qualification timestamp")
        if (
            quiet.get("qualified") is not True
            or age < 0
            or age > 5
            or quiet.get("os_background_refinement_sha256") != OS_BACKGROUND_SHA256
            or quiet.get("boot_id") != identity["boot_id"]
            or quiet.get("pid_namespace") != identity["pid_namespace"]
            or artifact_identity(quiet["archive"]["path"], maximum=128 * 1024**2)
            != quiet["archive"]
        ):
            raise ValueError("quiet qualification/archive is stale, incomplete or changed")
        if TIMING_MARKER.exists() or TIMING_MARKER.is_symlink():
            raise FileExistsError("existing TIMING marker cannot be borrowed or overwritten")
        descriptor = os.open(TIMING_LOCK, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("frozen timing flock is not a regular file")
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            marker = os.open(
                TIMING_MARKER, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            with os.fdopen(marker, "wb") as stream:
                info = os.fstat(stream.fileno())
                nonce = secrets.token_hex(32)
                lock_info = os.fstat(descriptor)
                receipt = {
                    "artifact_kind": "xgraph.serving_recovery_coordination",
                    "status": "HELD",
                    "protocol_sha256": PROTOCOL_SHA256,
                    "controller_commit": request["controller_commit"],
                    "controller_source_sha256": request["controller_source_sha256"],
                    "execution": execution,
                    "controller_process": identity,
                    "scheduler": scheduler,
                    "decisions": request["decisions"],
                    "quiet_qualification": quiet,
                    "created_at": datetime.now(UTC).isoformat(),
                    "marker": {
                        "path": str(TIMING_MARKER),
                        "device": info.st_dev,
                        "inode": info.st_ino,
                        "owner_nonce": nonce,
                    },
                    "host_flock": {
                        "path": str(TIMING_LOCK),
                        "device": lock_info.st_dev,
                        "inode": lock_info.st_ino,
                    },
                    "isolation_qualification": False,
                }
                if "engineering_amendment" in request:
                    receipt["engineering_amendment"] = copy.deepcopy(
                        request["engineering_amendment"]
                    )
                immutable_json(receipt_path, receipt)
                receipt_pin = metadata_identity(receipt_path)
                marker_bytes = (
                    json.dumps(
                        {"receipt_sha256": receipt_pin["sha256"], "owner_nonce": nonce},
                        sort_keys=True,
                    )
                    + "\n"
                ).encode()
                stream.write(marker_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            self.descriptor, self.receipt = descriptor, copy.deepcopy(receipt)
            self.receipt_pin, self.marker_bytes = receipt_pin, marker_bytes
            return copy.deepcopy(receipt)
        except BaseException:
            # A partially created own marker is deliberately retained. Its
            # presence blocks other coordinated work until owner reconciliation.
            os.close(descriptor)
            raise

    def release(self, cleanup):
        if cleanup.get("confirmed_absent") is not True or cleanup.get("errors") != []:
            raise ValueError("unknown owned cleanup retains TIMING marker and receipt")
        if self.descriptor is None or self.receipt is None:
            raise ValueError("coordination ownership is not live")
        marker = self.receipt["marker"]
        expected_identity = marker["device"], marker["inode"]
        if (
            metadata_identity(self.receipt_pin["path"]) != self.receipt_pin
            or read_seal_digest(Path(self.receipt_pin["path"]).with_suffix(".json.sha256"))
            != self.receipt_pin["sha256"]
            or controller_process_identity() != self.receipt["controller_process"]
        ):
            raise ValueError("coordination receipt/controller identity changed")
        try:
            descriptor = os.open(TIMING_MARKER, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as stream:
                info = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(info.st_mode)
                    or (info.st_dev, info.st_ino) != expected_identity
                    or stream.read(len(self.marker_bytes) + 1) != self.marker_bytes
                ):
                    raise ValueError("TIMING marker ownership/identity changed")
            current = TIMING_MARKER.lstat()
            if (current.st_dev, current.st_ino) != expected_identity:
                raise ValueError("TIMING marker identity changed before release")
        except OSError as error:
            raise ValueError("TIMING marker ownership could not be verified") from error
        TIMING_MARKER.unlink()
        immutable_json(
            Path(self.receipt_pin["path"]).parent / "coordination-release.json",
            {
                "status": "RELEASED_AFTER_CONFIRMED_OWNED_CLEANUP",
                "coordination": self.receipt_pin,
                "marker": marker,
                "cleanup": copy.deepcopy(cleanup),
                "released_at": datetime.now(UTC).isoformat(),
            },
        )
        self.close()
        return True

    def close(self):
        """Drop only this process's flock descriptor; never infer or remove a marker."""
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None


def check_inventory_sample(snapshot):
    if (
        snapshot.get("complete") is not True
        or snapshot.get("ownership_complete") is not True
        or snapshot.get("errors") != []
    ):
        raise ValueError("live inventory is incomplete or contaminated; stop owned work")


@contextmanager
def control_deadline(deadline, *, clock=time.monotonic):
    """Bound post-native source/stream verification without changing a worker watchdog."""
    remaining = nonnegative(deadline, "control deadline") - clock()
    if remaining <= 0:
        raise TimeoutError("qualified post-native control allowance exhausted")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer != (0.0, 0.0):
        raise ValueError("native controller cannot borrow another owner's alarm")

    def expired(_number, _frame):
        raise TimeoutError("qualified post-native control allowance exhausted")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def worker_resource_setup(profile):
    """Restore the explicitly reviewed native launch profile, never the CPU-test profile."""
    import resource

    os.setpriority(os.PRIO_PROCESS, 0, profile["nice"])
    os.sched_setaffinity(0, profile["cpu_affinity"])
    resource.setrlimit(resource.RLIMIT_AS, tuple(profile["address_space_limit"]))


def signal_bound_worker(identity, number):
    """The pre-native bound relay remains stoppable when storage/inventory itself fails."""
    try:
        handle = open_pid_handle(identity["pid"])
        try:
            if controller_process_identity(identity["pid"]) != identity:
                raise ValueError("owned relay identity changed; refuse a numeric PID signal")
            signal_pid_handle(handle, number)
        finally:
            os.close(handle)
    except ProcessLookupError:
        return {"absent": True, "identity": identity}
    return {"identity": identity, "signal": number}


def cleanup_owned_attempt(monitor, process, identity, *, server, deadline, clock, sleep):
    """TERM, bounded grace, KILL and fresh absence proof within one cleanup envelope."""
    actions = []
    cleanup = {"confirmed_absent": False, "errors": ["cleanup not yet observed"], "owned": []}
    prior = getattr(monitor, "_samples", ())
    known = copy.deepcopy(prior[-1]) if prior else None

    def observe():
        # Reap our direct child before a scan can see it as an unreaped zombie.
        if clock() >= deadline:
            raise TimeoutError("owned cleanup deadline exhausted before observation")
        try:
            process.poll()
            result = monitor.confirm_cleanup()
        except (OSError, ValueError, KeyError, TypeError) as error:
            if clock() >= deadline:
                raise
            actions.append(
                {"scope": "observation", "at": clock(), "error": {"kind": type(error).__name__}}
            )
            result = {
                "confirmed_absent": False,
                "errors": ["cleanup observation unavailable"],
                "owned": [],
            }
        if clock() >= deadline:
            raise TimeoutError("owned cleanup deadline exhausted during observation")
        return result

    def absent():
        return cleanup.get("confirmed_absent") is True and cleanup.get("errors") == []

    def send(number):
        try:
            resource = stop_proven_owned(
                monitor,
                signal_number=number,
                server=server,
                deadline=deadline,
                clock=clock,
                known_snapshot=known,
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            resource = {"errors": [{"kind": type(error).__name__}], "signaled": []}
        actions.append({"scope": "resources", "signal": number, "at": clock(), "receipt": resource})
        if clock() >= deadline:
            raise TimeoutError("owned cleanup deadline exhausted during resource signal")
        # Avoid a second TERM interrupting the relay's own frozen teardown when
        # the fresh host inventory already signaled this exact worker.
        through_resources = identity is not None and any(
            item.get("pid") == identity["pid"]
            and item.get("startticks") == identity["startticks"]
            and item.get("signal") == number
            for item in resource.get("signaled", [])
        )
        if identity is not None and process.poll() is None:
            retained = {"scope": "relay", "signal": number, "at": clock()}
            if through_resources:
                retained["receipt"] = {"identity": identity, "status": "signaled-through-resources"}
            else:
                try:
                    if clock() >= deadline:
                        raise TimeoutError("owned cleanup deadline exhausted before relay signal")
                    retained["receipt"] = signal_bound_worker(identity, number)
                except (OSError, ValueError) as error:
                    retained["error"] = {"kind": type(error).__name__}
            actions.append(retained)
            if clock() >= deadline:
                raise TimeoutError("owned cleanup deadline exhausted during relay signal")

    try:
        with control_deadline(deadline, clock=clock):
            cleanup = observe()
            if absent():
                return cleanup, actions
            send(signal.SIGTERM)
            # Keep the KILL/reap/observation tail inside the same180 seconds;
            # neither this grace nor a native watchdog is restarted.
            grace_end = min(clock() + 150, deadline - (CLEANUP_SECONDS - 150))
            cleanup = observe()
            while not absent() and clock() < grace_end:
                sleep(min(1, grace_end - clock()))
                cleanup = observe()
            if not absent():
                send(signal.SIGKILL)
                cleanup = observe()
                while not absent() and clock() < deadline:
                    sleep(min(1, deadline - clock()))
                    cleanup = observe()
            if process.poll() is not None:
                process.wait(timeout=max(0.001, min(1, deadline - clock())))
            cleanup = observe()
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        cleanup = {
            "confirmed_absent": False,
            "errors": ["cleanup observation unavailable"],
            "owned": [],
            "failure": {"kind": type(error).__name__},
        }
        actions.append({"scope": "cleanup", "at": clock(), "error": {"kind": type(error).__name__}})
    return cleanup, actions


def supervise_native_attempt(
    context,
    budget,
    attempt,
    *,
    monitor,
    writer,
    clock=time.monotonic,
    sleep=time.sleep,
    process_factory=subprocess.Popen,
    admitted=None,
):
    """Compose the real process, full-lifetime observer, durable archive and settlement.

    The caller must have verified the owner release and persistent launch claim.
    The private relay independently rechecks both and waits for the bound GO.
    No signal or successful exit substitutes for independently observed cleanup.
    """
    job = context["job"]
    child, output = budget.path.parent, Path(job["output"])
    key = job["id"].replace("/", "-")
    directory = child / "coverage" / key
    process = identity = None
    start = native_started = native_ended = None
    cleanup = {"confirmed_absent": False, "errors": ["owned interval not established"], "owned": []}
    archive, returncode, failure = None, None, None
    control = None
    allowances = context["control_allowances"]
    setup_deadline = attempt["started_monotonic"] + allowances["relay_setup_seconds"]
    post_deadline = None
    envelope = {
        "artifact_kind": "xgraph.serving_recovery_dispatch",
        "job_id": job["id"],
        "attempt_started_monotonic": attempt["started_monotonic"],
        "controller": copy.deepcopy(context["controller"]),
        "specification": copy.deepcopy(context["specification"]),
        "source_sha256": json_sha256(context["source_proof"]),
        "worker": None,
        "spawned_monotonic": None,
        "go_monotonic": None,
        "coverage_start": None,
        "coverage_end": None,
        "control_allowances": copy.deepcopy(allowances),
        "cleanup_actions": [],
    }
    try:
        first = monitor.sample()
        check_inventory_sample(first)
        sleep(max(0, first["at"] + 1 - clock()))
        second = monitor.sample()
        check_inventory_sample(second)
        baseline = monitor.freeze_baseline([first, second])
        start = baseline["at"]
        envelope["coverage_start"] = start
        if admitted is not None:
            admitted(monitor, first, second)
        if clock() >= setup_deadline:
            raise TimeoutError("qualified relay setup allowance exhausted before spawn")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.with_suffix(".outer.log").open("x") as log:
            process = process_factory(
                context["argv"],
                cwd=context["cwd"],
                env=context["runtime_environment"],
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=lambda: worker_resource_setup(context["profile"]),
            )
            identity = controller_process_identity(process.pid)
            if (
                identity["pid"] != identity["pgid"]
                or identity["pid"] != identity["sid"]
                or any(
                    identity[key] != context["controller"][key]
                    for key in ("boot_id", "pid_namespace")
                )
            ):
                raise ValueError("native relay did not start in its bound independent host session")
            envelope.update(worker=identity, spawned_monotonic=clock())
            context["bound_worker"] = identity
            bound = {key: identity[key] for key in ("pid", "startticks", "pgid", "sid")}
            monitor.bind_worker(
                {
                    **bound,
                    "job_id": job["id"],
                    "argv_sha256": context["intent"]["worker_argv_sha256"],
                }
            )
            handoff = {
                "artifact_kind": "xgraph.serving_recovery_native_handoff",
                "job_id": job["id"],
                "controller": context["controller"],
                "worker": identity,
                "specification": context["specification"],
                "nonce": context["handoff_nonce"],
                "go_monotonic": clock(),
            }
            handoff_path = child / "handoffs" / (key + ".json")
            handoff_path.parent.mkdir(exist_ok=True)
            immutable_json(handoff_path, handoff)
            envelope["handoff"] = metadata_identity(handoff_path)
            if clock() >= setup_deadline:
                raise TimeoutError("qualified relay setup allowance exhausted before GO")
            process.stdin.write(
                (
                    json.dumps(
                        {
                            "nonce": context["handoff_nonce"],
                            "handoff_sha256": envelope["handoff"]["sha256"],
                        }
                    )
                    + "\n"
                ).encode()
            )
            process.stdin.flush()
            process.stdin.close()
            envelope["go_monotonic"] = handoff["go_monotonic"]
            previous_sample = second["at"]
            next_sample = previous_sample + 1
            while True:
                returncode = process.poll()
                if returncode is not None:
                    break
                if context.get("phase_tracking") is True:
                    phase = native_phase_record(context, "start", required=False)
                    if phase is not None:
                        native_started = phase["at"]
                        if native_started > setup_deadline:
                            raise TimeoutError("native phase began after qualified setup allowance")
                    ended = native_phase_record(context, "end", required=False)
                    if ended is not None:
                        native_ended = ended["at"]
                        post_deadline = native_ended + allowances["post_native_seconds"]
                if native_started is None:
                    deadline = setup_deadline
                elif post_deadline is not None:
                    deadline = post_deadline
                else:
                    deadline = setup_deadline + job["wall_seconds"] + CLEANUP_SECONDS
                if clock() >= deadline:
                    raise TimeoutError("native setup/watchdog/cleanup envelope exhausted")
                sleep(max(0, min(next_sample, deadline) - clock()))
                snapshot = monitor.sample()
                check_inventory_sample(snapshot)
                if snapshot["at"] - previous_sample > 5:
                    raise ValueError("native inventory missed its maximum sampling gap")
                previous_sample = snapshot["at"]
                next_sample = max(next_sample + 1, previous_sample)
            process.wait()
        if context.get("phase_tracking") is True:
            start_phase = native_phase_record(context, "start")
            end_phase = native_phase_record(context, "end")
            native_started, native_ended = start_phase["at"], end_phase["at"]
            envelope.update(native_start=start_phase["identity"], native_end=end_phase["identity"])
            post_deadline = native_ended + allowances["post_native_seconds"]
            if (
                not envelope["go_monotonic"] <= native_started <= setup_deadline
                or not native_started
                <= native_ended
                <= setup_deadline + job["wall_seconds"] + CLEANUP_SECONDS
            ):
                raise ValueError(
                    "native phase boundaries exceed the unchanged watchdog/cleanup envelope"
                )
        cleanup = monitor.confirm_cleanup()
        envelope["coverage_end"] = cleanup.get("at")
        if cleanup.get("confirmed_absent") is not True or cleanup.get("errors") != []:
            raise ValueError("owned cleanup is unknown; stop and retain reservation")
        if post_deadline is None:
            post_deadline = clock() + allowances["post_native_seconds"]
        with control_deadline(post_deadline, clock=clock):
            monitor.finish_archive()
            archive = retain_inventory_archive(directory, monitor.evidence(start, cleanup["at"]))
    except BaseException as error:
        failure = {"kind": type(error).__name__, "message": "native orchestration/control failed"}
        if process is not None and cleanup.get("confirmed_absent") is not True:
            cleanup_deadline = min(
                clock() + CLEANUP_SECONDS,
                (native_started if native_started is not None else setup_deadline)
                + job["wall_seconds"]
                + CLEANUP_SECONDS,
            )
            envelope["cleanup_deadline"] = cleanup_deadline
            cleanup, envelope["cleanup_actions"] = cleanup_owned_attempt(
                monitor,
                process,
                identity,
                server=context.get("intent", {}).get("server"),
                deadline=cleanup_deadline,
                clock=clock,
                sleep=sleep,
            )
        else:
            with suppress(OSError, ValueError):
                cleanup = monitor.confirm_cleanup()
        envelope["coverage_end"] = cleanup.get("at")
        try:
            storage = monitor.storage_failure_metadata()
        except (OSError, ValueError):
            storage = {"qualification": False, "error": "storage progress unavailable"}
        archive = {"failed": True, "failure": failure, "storage_failure": None}
        try:
            directory.mkdir(parents=True, exist_ok=True)
            storage_path = directory / "storage-failure.json"
            immutable_json(storage_path, storage)
            archive["storage_failure"] = artifact_identity(storage_path, maximum=128 * 1024**2)
        except (OSError, ValueError) as storage_error:
            # The terminal accounting receipt stays small even if an unacked
            # frame is large or its separate retention write cannot complete.
            archive["storage_retention_error"] = {"kind": type(storage_error).__name__}
    finally:
        if process is not None and process.stdin is not None:
            # A failed pre-GO binding must also release our own input pipe;
            # EOF refuses native work even when process identity was unreadable.
            with suppress(OSError, ValueError):
                process.stdin.close()
        envelope.update(
            native_started_monotonic=native_started,
            native_ended_monotonic=native_ended,
            driver_returncode=returncode,
            failure=failure,
        )
    try:
        dispatch_path = child / "dispatch" / (key + ".json")
        dispatch_path.parent.mkdir(exist_ok=True)
        immutable_json(dispatch_path, envelope)
        control_path = output.with_suffix(".control.json")
        if control_path.exists() and not control_path.is_symlink():
            control = artifact_identity(control_path, maximum=128 * 1024**2)
        return complete_controlled_attempt(
            budget,
            attempt,
            job,
            control=control,
            archive=archive,
            cleanup=cleanup,
            source_proof=context["source_proof"],
            expected_policy=context["policy"],
            expected_intent=context["intent"],
            driver_returncode=returncode,
            server_name=context.get("server_name"),
            expected_effective=context.get("expected_effective"),
            expected_overhead=context.get("expected_overhead"),
            envelope=metadata_identity(dispatch_path),
            dispatch_failure=failure,
            post_deadline=post_deadline,
            clock=clock,
        )
    finally:
        writer.close()


def native_cell(specification):
    """Enter the frozen supervisor only through the execution/admission verifier."""
    context = verify_native_launch(specification)
    source, environment, wheel = (context[key] for key in ("source", "environment", "wheel"))
    destination = Path(source["root"])
    verify_native_source(destination, source)
    verify_private_environment(destination, environment)
    if (
        sys.executable != environment["python"]
        or any(key in os.environ for key in ("XG_SERVING_BUDGET", "XG_SERVING_SEAL_SHA256"))
        or verify_native_wheel(destination, source, wheel["runtime"]) != wheel
    ):
        raise ValueError(
            "native private interpreter, wheel or inherited old-budget profile changed"
        )
    attested = attest_native_environment(destination, source, environment, load_native=True)
    if attested["native_binary_imported"] is not True:
        raise ValueError("native private binary import was not established")
    job = context["job"]
    role = "post_queue" if job["kind"] == "overhead" else "serve_cost"
    native = importlib.import_module("xgraph." + role)
    if Path(native.__file__).resolve() != destination / "src/xgraph" / (role + ".py"):
        raise ValueError("native supervisor imported outside the private frozen source")

    def interrupted(_number, _frame):
        raise KeyboardInterrupt("owned relay termination; frozen supervisor cleanup required")

    if job["kind"] == "overhead":
        output = Path(job["output"])
        primary = context["primary_job"]
        command = native_overhead_command(job, primary, source)
        pin = context["primary_source"]
        if Path(pin["path"]) != Path(primary["output"]) or artifact_identity(pin["path"]) != pin:
            raise ValueError("native overhead new qualified primary source identity changed")
        selection = verify_overhead_selection(context)
        if any(
            path.exists() or path.is_symlink()
            for path in (
                output,
                output.with_suffix(".control.json"),
                output.with_suffix(".control.json.sha256"),
                output.with_suffix(".driver.log"),
            )
        ):
            raise FileExistsError("native overhead orphan/report/receipt reuse is forbidden")
        requested = (
            overhead_input_contract(context) if context.get("phase_tracking") is True else None
        )
        previous, code, report = signal.getsignal(signal.SIGTERM), None, None
        original_error = None
        phase_started = phase_ended = False
        try:
            signal.signal(signal.SIGTERM, interrupted)
            if context.get("phase_tracking") is True:
                record_native_phase(context, "start", requested=requested)
                phase_started = True
            with output.with_suffix(".driver.log").open("x") as stream:
                result = supervise_overhead_stage(
                    command,
                    wall_seconds=job["wall_seconds"],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            code = result.returncode
            if phase_started:
                record_native_phase(context, "end")
                phase_ended = True
            if type(code) is not int or code != 0:
                raise RuntimeError("native frozen overhead stage did not exit successfully")
            report = json.loads(output.read_bytes())
            verify_native_source(destination, source)
        except BaseException as error:
            original_error = type(error).__name__
            raise
        finally:
            signal.signal(signal.SIGTERM, previous)
            integrity_errors = []
            if phase_started and not phase_ended:
                try:
                    record_native_phase(context, "end")
                except (OSError, ValueError) as error:
                    integrity_errors.append({"field": "native_phase", "kind": type(error).__name__})
            report = retained_native_report(
                output, report, integrity_errors, skip_decode=original_error == "JSONDecodeError"
            )
            report_pin = retained_native_pin(output, integrity_errors)
            try:
                control = project_overhead_control(report) if report is not None else {}
            except (KeyError, TypeError, AttributeError):
                integrity_errors.append({"field": "control", "kind": "invalid-overhead-schema"})
                control = {}
            retained = {
                "job_id": job["id"],
                "control": control,
                "returncode": code,
                "primary_job_id": primary["id"],
                "primary_source": pin,
                "selection": selection,
                "requested_overhead": requested,
                "command": command,
                "wall_seconds": job["wall_seconds"],
                "report": report_pin,
                "integrity_errors": integrity_errors,
                "original_error_kind": original_error,
                "source_sha256": json_sha256(source),
                "environment_sha256": json_sha256(environment),
                "native_binary_import": attested,
                "native_phases_required": False,
                "native_headroom_within_charged_stage": True,
                "outer_cleanup_qualification": False,
            }
            retain_native_control(output.with_suffix(".control.json"), retained, original_error)
        return {
            "control": artifact_identity(output.with_suffix(".control.json")),
            "report": retained["report"],
        }
    if job["kind"] not in ("primary", "descriptive"):
        raise ValueError("native serving relay cannot execute serverless overhead")
    output, stage = Path(job["output"]), Path(job["output"]).parent
    if any(
        path.exists() or path.is_symlink()
        for path in (
            output,
            output.with_suffix(".worker.json"),
            output.with_suffix(".partial.jsonl"),
            output.with_suffix(".server.log"),
            output.with_suffix(".control.json"),
        )
    ):
        raise FileExistsError("native serving orphan/report/specification reuse is forbidden")
    original, runtime_pin = context["original_study"], context["runtime_json"]
    if (
        Path(original["path"]) != stage / "original-study.json"
        or artifact_identity(original["path"], maximum=128 * 1024**2) != original
        or Path(runtime_pin["path"]) != stage / "runtime.json"
        or artifact_identity(runtime_pin["path"], maximum=2 * 1024**2) != runtime_pin
        or json_sha256(json.loads(Path(runtime_pin["path"]).read_bytes()))
        != json_sha256(wheel["runtime"])
    ):
        raise ValueError("native opaque study/runtime identity or canonical path changed")
    derived, study, recipe = transport_runtime_study(
        Path(original["path"]).read_bytes(),
        wheel["runtime"],
        source_proof=source,
        original_source=source["source_origin"],
        actual_kv_capacity_tokens=context["actual_kv_capacity_tokens"],
        parent_runtime=context["parent_runtime"],
        parent_runtime_archive=context["parent_runtime_archive"],
    )
    study_path = stage / "study.json"
    if study_path.exists() or study_path.is_symlink():
        if (
            artifact_identity(study_path, maximum=128 * 1024**2)["sha256"]
            != recipe["derived_sha256"]
        ):
            raise ValueError("native derived study changed between fresh cells")
    else:
        with study_path.open("xb") as stream:
            stream.write(derived)
    trial = {
        **copy.deepcopy(job["trial"]),
        "status": "running",
        "server_name": context["server_name"],
    }
    if context.get("phase_tracking") is True:
        actual_command = native.VLLMServer(
            study, trial, job["kv_cache_bytes"], Path(wheel["runtime"]["site"])
        ).command()
        if actual_command != context["server_command"]:
            raise ValueError(
                "frozen actual Docker factory differs from the sealed relocated preview"
            )
    previous = signal.getsignal(signal.SIGTERM)
    report, original_error = None, None
    phase_started = phase_ended = False
    try:
        signal.signal(signal.SIGTERM, interrupted)
        native.check_sources(study)
        if context.get("phase_tracking") is True:
            record_native_phase(context, "start", requested=context["expected_effective"])
            phase_started = True
        report = native.supervise_cell(
            study, trial, job["kv_cache_bytes"], Path(wheel["runtime"]["site"])
        )
        if phase_started:
            record_native_phase(context, "end")
            phase_ended = True
        native.check_sources(study)
        verify_native_source(destination, source)
    except BaseException as error:
        original_error = type(error).__name__
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        integrity_errors = []
        if phase_started and not phase_ended:
            try:
                record_native_phase(context, "end")
            except (OSError, ValueError) as error:
                integrity_errors.append({"field": "native_phase", "kind": type(error).__name__})
        report = retained_native_report(
            output, report, integrity_errors, skip_decode=original_error == "JSONDecodeError"
        )
        try:
            control = (
                project_native_control(report)
                if report is not None
                else {"status": "failed", "error": True}
            )
        except (KeyError, TypeError, AttributeError):
            integrity_errors.append({"field": "control", "kind": "invalid-native-schema"})
            control = {"status": "failed", "error": True}
        worker = output.with_suffix(".worker.json")
        retained = {
            "job_id": job["id"],
            "control": control,
            "original_study": original,
            "derived_study": retained_native_pin(
                study_path, integrity_errors, maximum=128 * 1024**2
            ),
            "transport": recipe,
            "worker_specification": retained_native_pin(
                worker, integrity_errors, maximum=2 * 1024**2
            ),
            "report": retained_native_pin(output, integrity_errors),
            "integrity_errors": integrity_errors,
            "original_error_kind": original_error,
            "source_sha256": json_sha256(source),
            "environment_sha256": json_sha256(environment),
            "native_binary_import": attested,
            "outer_cleanup_qualification": False,
        }
        retain_native_control(output.with_suffix(".control.json"), retained, original_error)
    return {
        "control": artifact_identity(output.with_suffix(".control.json")),
        "report": retained["report"],
    }


def verify_native_launch(specification):
    """Authenticate the private child independently before any study decode or model call."""
    child = Path(load_protocol()["canonical_child"])
    specification = Path(specification)
    allowed = {
        child
        / "specs"
        / (stage + ("" if stage.startswith("overhead-") else f"-{number:04d}") + ".json")
        for stage, count in zip(STAGES, (20, 20, 12, 12, 1, 1), strict=True)
        for number in range(count)
    }
    if (
        specification not in allowed
        or specification.resolve() != specification
        or not specification.is_file()
    ):
        raise ValueError(
            "unbound native serving recovery; canonical qualified specification required"
        )
    release = verify_owner_release(
        child.parent / "registration.json", child.parent / "launch-review.json"
    )
    resources = native_launch_resources(release)
    source = _effective_source(release)
    if (
        str(Path(__file__).resolve()) != source["module"]
        or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != source["sha256"]
    ):
        raise ValueError("native relay actual effective source identity changed")
    pin = metadata_identity(specification)
    if read_seal_digest(specification.with_suffix(".json.sha256")) != pin["sha256"]:
        raise ValueError("native child specification seal changed")
    spec = json.loads(specification.read_bytes())
    controller = spec.get("controller")
    process_identity(controller)
    if (
        os.getppid() != controller["pid"]
        or controller_process_identity(controller["pid"]) != controller
    ):
        raise ValueError(
            "native handoff parent/controller is not the approved live process identity"
        )
    record = native_launch_record(release, controller)
    claim = ClaimStore(ROOT).verify("launch", record)
    if spec.get("launch") != {"commit": claim, "record": record}:
        raise ValueError("native child does not bind the persistent one-use launch claim")
    budget = RecoveryBudget(
        child / "budget.json",
        native_budget_binding(release, resources),
        {"charged_gpu_seconds": release["registration"]["parent_charged_gpu_seconds"]},
        clock=time.monotonic,
        claims=ClaimStore(ROOT),
        controller_identity=controller,
        create=False,
    )
    active = budget.data["active"]
    if active is None or budget.data["attempts"][active] != spec.get("attempt"):
        raise ValueError("native child has no same running persistent budget attempt")
    attempt = budget.data["attempts"][active]
    job = next(
        (
            job
            for stage in release["registration"]["schedule"]
            for job in stage["jobs"]
            if job["id"] == spec.get("job_id")
        ),
        None,
    )
    if job is None or any(attempt.get(key) != value for key, value in job.items()):
        raise ValueError("native child job differs from its exact registered fresh matrix")
    context = native_context(release, resources, budget, job)
    expected = {
        "artifact_kind": "xgraph.serving_recovery_native_specification",
        "version": 2 if "engineering_amendment" in release else 1,
        "registration": release["review"]["registration"],
        "owner_release": release["release"],
        "execution": release["review"]["execution"],
        "job_id": job["id"],
        "controller": controller,
        "launch": {"commit": claim, "record": record},
        "attempt": attempt,
        "context": native_context_binding(context),
    }
    if "engineering_amendment" in release:
        expected["engineering_amendment"] = _engineering_seal(release)
    if (
        set(spec) != {*expected, "nonce"}
        or any(spec.get(key) != value for key, value in expected.items())
        or not isinstance(spec["nonce"], str)
        or re.fullmatch(r"[0-9a-f]{64}", spec["nonce"]) is None
    ):
        raise ValueError("native child source/runtime/science/profile binding changed")
    worker = controller_process_identity()
    if (
        worker["pid"] != worker["pgid"]
        or worker["pid"] != worker["sid"]
        or any(worker[key] != controller[key] for key in ("boot_id", "pid_namespace"))
        or process_commandline_sha256() != context["intent"]["worker_argv_sha256"]
    ):
        raise ValueError("native relay worker/argv/session identity changed")
    # Read only the fixed parent-authored GO token; an ordinary direct invocation
    # cannot turn an old specification into a fresh worker admission.
    line = sys.stdin.readline(4097)
    if not line.endswith("\n") or len(line) > 4096:
        raise ValueError("native relay is unbound without its complete parent GO token")
    token = json.loads(line)
    hpath = child / "handoffs" / (job["id"].replace("/", "-") + ".json")
    handoff_pin = metadata_identity(hpath)
    if (
        token != {"nonce": spec["nonce"], "handoff_sha256": handoff_pin["sha256"]}
        or read_seal_digest(hpath.with_suffix(".json.sha256")) != handoff_pin["sha256"]
    ):
        raise ValueError("native child GO/handoff identity changed")
    handoff = json.loads(hpath.read_bytes())
    if (
        set(handoff)
        != {
            "artifact_kind",
            "job_id",
            "controller",
            "worker",
            "specification",
            "nonce",
            "go_monotonic",
        }
        or handoff["artifact_kind"] != "xgraph.serving_recovery_native_handoff"
        or handoff["job_id"] != job["id"]
        or handoff["controller"] != controller
        or handoff["worker"] != worker
        or handoff["specification"] != pin
        or handoff["nonce"] != spec["nonce"]
        or nonnegative(handoff["go_monotonic"], "native GO time") < attempt["started_monotonic"]
    ):
        raise ValueError("native handoff does not bind this exact live worker and attempt")
    verify_child_native_profile(context)
    if time.monotonic() >= context["setup_deadline"]:
        raise TimeoutError("qualified relay setup allowance exhausted before admitting native work")
    admission = child / "native-admissions" / (job["id"].replace("/", "-") + ".json")
    admission.parent.mkdir(exist_ok=True)
    immutable_json(admission, {"handoff": handoff_pin, "worker": worker, "specification": pin})
    context.update(specification=pin, handoff_nonce=spec["nonce"])
    return context


def verify_child_native_profile(context):
    import resource

    profile = context["profile"]
    if (
        sys.executable != context["environment"]["python"]
        or sorted(os.sched_getaffinity(0)) != sorted(profile["cpu_affinity"])
        or list(resource.getrlimit(resource.RLIMIT_AS)) != profile["address_space_limit"]
        or os.getpriority(os.PRIO_PROCESS, 0) != profile["nice"]
        or Path.cwd() != Path(context["cwd"])
        or any(
            key in os.environ
            for key in ("PYTHONPATH", "PYTHONHOME", "XG_SERVING_BUDGET", "XG_SERVING_SEAL_SHA256")
        )
        or any(
            os.environ.get(key) != value
            for key, value in native_profile(context["job"])["environment"].items()
        )
        or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
    ):
        raise ValueError(
            "native child does not have the reviewed interpreter/resource/environment profile"
        )


def record_native_phase(context, phase, *, requested=None):
    if phase not in ("start", "end"):
        raise ValueError("native phase must be start or end")
    path = Path(context["job"]["output"]).with_suffix(".native-" + phase + ".json")
    sidecar = path.with_suffix(".json.sha256")
    if any(candidate.exists() or candidate.is_symlink() for candidate in (path, sidecar)):
        raise FileExistsError("native phase marker already exists; no reuse or overwrite")
    worker = controller_process_identity()
    at = time.monotonic()
    if phase == "start" and at >= nonnegative(
        context.get("setup_deadline"), "native setup deadline"
    ):
        raise TimeoutError("qualified relay setup allowance exhausted before native start")
    temporary = path.parent / (".native-phase-" + secrets.token_hex(16))
    temporary.mkdir(mode=0o700)
    prepared = temporary / "record.json"
    immutable_json(
        prepared,
        {
            "artifact_kind": "xgraph.serving_recovery_native_phase",
            "phase": phase,
            "job_id": context["job"]["id"],
            "specification": context["specification"],
            "worker": worker,
            "at": at,
            "requested_sha256": json_sha256(requested) if requested is not None else None,
        },
    )
    # Publish the completed seal first and the complete data last. The optional
    # parent poll sees either no marker or a fully written, sealed marker.
    os.link(prepared.with_suffix(".json.sha256"), sidecar)
    os.link(prepared, path)
    prepared.with_suffix(".json.sha256").unlink()
    prepared.unlink()
    temporary.rmdir()


def native_phase_record(context, phase, *, required=True):
    path = Path(context["job"]["output"]).with_suffix(".native-" + phase + ".json")
    if not path.exists() and not path.is_symlink() and not required:
        return None
    pin = metadata_identity(path)
    sidecar = path.with_suffix(".json.sha256")
    if not required:
        try:
            seal_info = sidecar.lstat()
        except FileNotFoundError:
            return None
        if stat.S_ISREG(seal_info.st_mode) and seal_info.st_size < 65:
            return None
    if read_seal_digest(sidecar) != pin["sha256"]:
        raise ValueError("native phase marker seal changed")
    record = json.loads(path.read_bytes())
    if (
        record.get("artifact_kind") != "xgraph.serving_recovery_native_phase"
        or record.get("phase") != phase
        or record.get("job_id") != context["job"]["id"]
        or record.get("specification") != context["specification"]
        or (
            context.get("bound_worker") is not None
            and record.get("worker") != context["bound_worker"]
        )
    ):
        raise ValueError("native phase marker job/specification changed")
    process_identity(record["worker"])
    nonnegative(record["at"], "native phase marker")
    return {**record, "identity": pin}


def verify_overhead_selection(context):
    """Bind the exact unchanged frozen selector to the new qualified WT source.

    Only admitted native execution calls this: study bytes remain opaque during
    development and preparation. No answer field is accessed by this predicate.
    """
    job, primary = context["job"], context["primary_job"]
    screen = Path(primary["output"]).parent
    pins = {}
    for field, name, maximum in (
        ("primary_study", "study.json", 128 * 1024**2),
        ("primary_plan", "trials.json", 2 * 1024**2),
    ):
        pin = context.get(field)
        if (
            not isinstance(pin, dict)
            or pin.get("path") != str(screen / name)
            or artifact_identity(pin["path"], maximum=maximum) != pin
        ):
            raise ValueError("native overhead primary plan/study identity or fixed path changed")
        pins[field] = pin
    study = json.loads((screen / "study.json").read_bytes())
    plan = json.loads((screen / "trials.json").read_bytes())
    # This is the literal first-match predicate from frozen post_queue.overhead.
    selected = next(
        (
            trial
            for trial in plan["trials"]
            if trial["repeat"] == 0
            and trial["cache"] == "warm"
            and trial["concurrency"] == 8
            and study["arms"][trial["arm"]]["policy"] == "WT"
        ),
        None,
    )
    if selected != primary["trial"] or selected is None or selected["output"] != primary["output"]:
        raise ValueError(
            "frozen overhead selector does not select the bound new primary plan source"
        )
    command = job["command"]
    if command.count("--screen") != 1 or command[command.index("--screen") + 1] != str(screen):
        raise ValueError("frozen overhead selector screen differs from the new primary plan")
    return {
        **copy.deepcopy(pins),
        "primary_source": copy.deepcopy(context["primary_source"]),
        "selected_trial_sha256": json_sha256(selected),
        "selector": "frozen-first-repeat0-warm-C8-WT",
    }


def retained_native_report(path, report, errors, *, skip_decode=False):
    """Best-effort operational projection; malformed output cannot replace the failure."""
    if report is None and Path(path).is_file():
        if skip_decode:
            errors.append({"field": "report", "kind": "JSONDecodeError"})
            return None
        try:
            report = json.loads(Path(path).read_bytes())
        except (OSError, ValueError) as error:
            errors.append({"field": "report", "kind": type(error).__name__})
            return None
    if report is not None and not isinstance(report, dict):
        errors.append({"field": "report", "kind": "not-a-JSON-object"})
        return None
    return report


def retained_native_pin(path, errors, *, maximum=512 * 1024**2):
    if not Path(path).exists() and not Path(path).is_symlink():
        return None
    try:
        return artifact_identity(path, maximum=maximum)
    except (OSError, ValueError) as error:
        errors.append({"field": str(path), "kind": type(error).__name__})
        return None


def retain_native_control(path, retained, original_error):
    try:
        immutable_json(path, retained)
    except (OSError, ValueError):
        if original_error is None:
            raise
        os.write(2, b"Native control receipt write failed; original failure preserved.\n")


def project_native_control(report):
    """Select operational fields only; never follow results, quality or latency values."""
    result = {
        key: copy.deepcopy(report.get(key))
        for key in (
            "artifact_kind",
            "version",
            "status",
            "trial",
            "kv_cache_bytes",
            "study_path",
            "driver_seconds",
            "prefill_declines",
        )
    }
    result.update(
        {
            key: True if report.get(key) is not None else None
            for key in ("error", "teardown_error", "monitor_error", "watchdog")
        }
    )
    resources = report.get("resources", {})
    result["resources"] = {
        key: copy.deepcopy(resources.get(key))
        for key in ("errors", "foreign_gpu_processes", "samples")
    }
    effective = report.get("effective_server", {})
    result["effective_server"] = {
        key: copy.deepcopy(effective.get(key))
        for key in (
            "kv_cache_memory_bytes",
            "enable_prefix_caching",
            "max_model_len",
            "max_num_seqs",
            "kv_capacity_tokens",
            "engine_settings",
        )
    }
    return result


def native_overhead_command(job, primary, source):
    """Relocate only the frozen overhead interpreter; retain its exact bare argv."""
    output = Path(job["output"])
    family = job["stage"].removeprefix("overhead-")
    child = output.parents[2]
    trial = primary["trial"]
    screen = child / "serving" / ("locked-" + family)
    if (
        family not in ("qwen38", "gemma31")
        or job["id"] != job["stage"]
        or job["kind"] != "overhead"
        or job["wall_seconds"] != 1800
        or output != child / "overhead" / family / "engine.json"
        or primary["kind"] != "primary"
        or primary["stage"] != "locked-" + family
        or not primary["id"].startswith(primary["stage"] + "/")
        or Path(primary["output"]).parent != screen
        or trial["output"] != primary["output"]
        or (trial["repeat"], trial["arm"], trial["cache"], trial["concurrency"], trial["seed"])
        != (0, "finalist", "warm", 8, 17)
    ):
        raise ValueError("native overhead requires the fixed new primary repeat0/warm/C8 WT source")
    original = [
        str(Path(source["source_origin"]) / ".venv/bin/python"),
        "-m",
        "xgraph.post_queue",
        "overhead",
        "--screen",
        str(screen),
        "--out",
        str(output),
    ]
    if job["command"] != original:
        raise ValueError("native frozen overhead stage argv changed outside path relocation")
    return [str(Path(source["root"]) / ".venv/bin/python"), *original[1:]]


def overhead_input_contract(context):
    """Pre-bind the frozen bench metadata recipe; never inspect a generated answer.

    This runs only inside the admitted private relay. Heavy input hashes are
    computed once by the unchanged bench producer, within its charged stage.
    """
    study_path = Path(context["primary_study"]["path"])
    study = json.loads(study_path.read_bytes())
    source = context["source"]
    root = Path(source["root"])
    primary = context["primary_job"]
    arm = study["arms"][primary["trial"]["arm"]]
    index, tokenizer = Path(study["kg_index"]), Path(study["tokenizer"])
    model = Path(study["model"])
    config = json.loads((model / "config.json").read_bytes())
    vocab = config.get("vocab_size", config.get("text_config", {}).get("vocab_size"))
    if type(vocab) is not int or vocab <= 0:
        raise ValueError("frozen overhead model configuration has no positive vocabulary width")
    inputs = [
        Path(primary["output"]),
        study_path,
        tokenizer,
        index / "kg-index.json",
        root / "src/xgraph/bench.py",
    ]
    inputs.extend(
        path
        for name in ("config.json", "generation_config.json")
        if (path := Path(arm.get("model", tokenizer.parent)) / name).is_file()
    )
    inputs.append(Path(arm["options"]["relation_labels"]))
    inputs.extend(sorted(index.glob("*")))
    if (root / "csrc/engine.cu").is_file():
        inputs.append(root / "csrc/engine.cu")
    inputs = list(dict.fromkeys(path for path in inputs if path.is_file()))
    critical = [context["primary_source"], context["primary_study"]]
    for entry in source["files"]:
        if entry["path"] in ("src/xgraph/bench.py", "csrc/engine.cu"):
            critical.append(
                {
                    "path": str(root / entry["path"]),
                    "bytes": entry["bytes"],
                    "sha256": entry["sha256"],
                }
            )
    return {
        "policy_sha256": arm["policy_sha256"],
        "sample_ids": sorted(study["panel_ids"])[:8],
        "input_paths": [str(path) for path in inputs],
        "input_sizes": [path.stat().st_size for path in inputs],
        "critical_inputs": critical,
        "repo_commit": source["commit"],
        "packages": context["native_packages"],
        "command": [
            Path(context["environment"]["python"]).name,
            str(root / "src/xgraph/bench.py"),
            "--wt-trial",
            primary["output"],
            "--index",
            str(index),
            "--tokenizer",
            str(tokenizer),
            "--batch",
            "1",
            "8",
            "--repeats",
            "3",
            "--vocab",
            str(vocab),
            "--out",
            context["job"]["output"],
        ],
        "source_sha256": json_sha256(source),
        "provenance_assurance": (
            "unchanged qualified producer hashes with independently bound critical inputs"
        ),
    }


def verify_overhead_provenance(provenance, expected):
    if "provenance_assurance" not in expected:
        if any(
            provenance.get(key) != expected[key]
            for key in ("inputs", "command", "repo_commit", "packages")
        ):
            raise ValueError(
                "overhead provenance does not bind new primary/study/source/input identities"
            )
        return
    rows = provenance.get("inputs")
    if (
        expected["provenance_assurance"]
        != "unchanged qualified producer hashes with independently bound critical inputs"
        or not isinstance(rows, list)
        or len(rows) != len(expected["input_paths"])
        or any(
            not isinstance(row, dict)
            or set(row) != {"path", "bytes", "sha256"}
            or row["path"] != path
            or type(row["bytes"]) is not int
            or row["bytes"] != size
            or not isinstance(row["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None
            for row, path, size in zip(
                rows, expected["input_paths"], expected["input_sizes"], strict=True
            )
        )
        or any(pin not in rows for pin in expected["critical_inputs"])
        or any(
            provenance.get(key) != expected[key] for key in ("command", "repo_commit", "packages")
        )
    ):
        raise ValueError(
            "frozen overhead provenance paths/source/new primary/command binding changed"
        )


def project_overhead_control(report):
    result = {
        key: copy.deepcopy(report.get(key))
        for key in (
            "artifact_kind",
            "version",
            "profile",
            "engine",
            "policy_sha256",
            "sample_ids",
            "provenance",
        )
    }
    result["results"] = [
        {
            key: copy.deepcopy(row.get(key))
            for key in ("stage", "batch", "mask_control", "errors", "repeats")
        }
        for row in report.get("results", [])
    ]
    return result


def transport_runtime_study(
    original,
    runtime,
    *,
    source_proof,
    original_source,
    actual_kv_capacity_tokens,
    parent_runtime=None,
    parent_runtime_archive=None,
):
    """Mechanical transport for an admitted runtime only; never preparation.

    Study fields outside runtime/source identity bookkeeping are copied without
    inspecting their semantics. The original bytes and the derived recipe stay
    retained. No answer or performance value chooses a branch here.
    """
    destination, original_source = Path(source_proof["root"]), Path(original_source)
    verify_native_source(destination, source_proof)
    if str(original_source) != source_proof["source_origin"]:
        raise ValueError("runtime transport original source binding changed")
    if actual_kv_capacity_tokens is not None and (
        type(actual_kv_capacity_tokens) is not int or actual_kv_capacity_tokens <= 0
    ):
        raise ValueError("runtime transport needs a positive new-cell KV capacity or null")
    study = json.loads(original)
    # The frozen CLI writes runtime.json separately after serializing study.json.
    # Never infer a recorded runtime from a field absent from the sealed study.
    lineage = verify_parent_runtime_lineage(
        parent_runtime, source_proof, approved_archive=parent_runtime_archive
    )
    derived = copy.deepcopy(study)
    registered = {entry["path"] for entry in source_proof["files"]}
    identities, replacements = {}, []
    for name, identity in study.get("input_identity", {}).items():
        path = Path(name)
        if (
            not path.is_absolute()
            or not isinstance(identity, list)
            or len(identity) != 5
            or any(type(value) is not int or value < 0 for value in identity)
        ):
            raise ValueError("runtime transport input identity is invalid")
        if path.is_relative_to(original_source):
            relative = str(path.relative_to(original_source))
            source_path = relative in NATIVE_SOURCE_TOP or relative.startswith(
                NATIVE_SOURCE_PREFIXES
            )
            if source_path:
                if relative not in registered:
                    raise ValueError(
                        "runtime transport cannot rewrite an unregistered source input"
                    )
                relocated = destination / relative
                info = relocated.stat()
                current = [
                    info.st_dev,
                    info.st_ino,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                ]
                identities[str(relocated)] = current
                replacements.append(
                    {"original": name, "derived": str(relocated), "identity": current}
                )
                continue
        identities[name] = copy.deepcopy(identity)
    if "input_identity" in study:
        derived["input_identity"] = identities
    derived["runtime"] = copy.deepcopy(runtime)
    # The original CLI learns this only from the current stage's first fresh
    # cell. The new worker specification supplies that same stage-local value.
    derived.pop("actual_kv_capacity_tokens", None)
    changed = sorted(
        key
        for key in set(study) | set(derived)
        if key not in study or key not in derived or study[key] != derived[key]
    )
    if set(changed) - {"runtime", "input_identity", "actual_kv_capacity_tokens"}:
        raise ValueError("runtime transport changed a non-bookkeeping study field")
    data = (json.dumps(derived, indent=2, allow_nan=False) + "\n").encode()
    memory = copy.deepcopy(derived)
    if actual_kv_capacity_tokens is not None:
        memory["actual_kv_capacity_tokens"] = actual_kv_capacity_tokens
    return (
        data,
        memory,
        {
            "boundary_sha256": RUNTIME_BOUNDARY_SHA256,
            "original_sha256": hashlib.sha256(original).hexdigest(),
            "derived_sha256": hashlib.sha256(data).hexdigest(),
            "source_sha256": json_sha256(source_proof),
            "runtime_sha256": json_sha256(runtime),
            "parent_runtime_lineage": lineage,
            "changed_top_level_keys": changed,
            "source_identity_replacements": replacements,
            "new_stage_actual_kv_capacity_tokens": actual_kv_capacity_tokens,
            "science_fields_changed": False,
        },
    )


def verify_parent_runtime_lineage(recorded, source_proof, *, approved_archive=None):
    """Require the parent's recorded archive/map, not a fresh re-stat assertion.

    The original opaque study authenticates this recorded metadata. Only the
    admitted relay decodes it, before any source bookkeeping substitution.
    """
    if not isinstance(recorded, dict):
        raise ValueError("NOT_READY: parent runtime source lineage is absent")
    origin = Path(source_proof["source_origin"])
    expected = {
        entry["path"]: entry["sha256"]
        for entry in source_proof["files"]
        if (
            (
                entry["path"].startswith(("src/", "csrc/", "scripts/"))
                and Path(entry["path"]).suffix in (".py", ".cu", ".h", ".sh")
            )
            or entry["path"] in ("pyproject.toml", "uv.lock", "CMakeLists.txt")
        )
    }
    original_map = {str(origin / relative): digest for relative, digest in expected.items()}
    if not expected or recorded.get("source_files") != original_map:
        raise ValueError(
            "NOT_READY: parent recorded source files differ from frozen source lineage"
        )
    try:
        if approved_archive is None:
            archive = recorded["source_snapshot"]
            original_pin = None
        else:
            original_pin = approved_archive["original"]
            archive = approved_archive["snapshot"]["path"]
            if (
                original_pin["path"] != recorded["source_snapshot"]
                or original_pin["sha256"] != recorded["source_sha256"]
                or any(
                    approved_archive["snapshot"][key] != original_pin[key]
                    for key in ("sha256", "bytes")
                )
            ):
                raise ValueError("parent source archive approved original/snapshot binding changed")
        pin = artifact_identity(archive, maximum=64 * 1024**2)
        if approved_archive is not None and pin != approved_archive["snapshot"]:
            raise ValueError("retained approved parent source archive changed")
        if pin["sha256"] != recorded["source_sha256"]:
            raise ValueError("parent source archive digest changed")
        with zipfile.ZipFile(archive) as stream:
            entries = stream.infolist()
            if len(entries) != len(expected) or {entry.filename for entry in entries} != set(
                expected
            ):
                raise ValueError(
                    "parent source archive contains missing/extra/duplicate source entries"
                )
            total = 0
            for entry in entries:
                total += entry.file_size
                if (
                    entry.is_dir()
                    or entry.file_size > 16 * 1024**2
                    or total > 64 * 1024**2
                    or stat.S_ISLNK(entry.external_attr >> 16)
                ):
                    raise ValueError("parent source archive entry is not bounded regular source")
                if hashlib.sha256(stream.read(entry)).hexdigest() != expected[entry.filename]:
                    raise ValueError("parent archived source bytes differ from the frozen source")
    except (KeyError, TypeError, OSError, zipfile.BadZipFile) as error:
        raise ValueError(
            "NOT_READY: parent runtime source archive lineage is unavailable"
        ) from error
    return {
        "source_snapshot": pin,
        "original_source_archive": original_pin,
        "source_files_sha256": json_sha256(original_map),
        "relative_source_map_sha256": hashlib.sha256(
            json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "relative_source_map_encoding": (
            "json.dumps(relative_map, sort_keys=True, separators=(',', ':')); UTF-8; no newline"
        ),
        "source_file_count": len(expected),
        "source_origin": str(origin),
        "normalization": "declared original source root to exact private frozen source only",
    }


def native_profile(job):
    """Project the source-wired owner/family wrappers, not the CPU test context.

    Two outer nice increments already saturate at 19 from ordinary nonnegative
    launch priority. The family wrapper adds the seven-GiB AS ceiling; bare
    overhead does not. Historical inherited overhead AS was not recorded.
    """
    if job["stage"] not in STAGES or job["kind"] not in ("primary", "descriptive", "overhead"):
        raise ValueError("unknown original native stage/profile")
    family = job["stage"].rsplit("-", 1)[1]
    overhead = job["kind"] == "overhead"
    if overhead != job["stage"].startswith("overhead"):
        raise ValueError("native stage/profile kind changed")
    environment = {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "XG_GPU_IDLE": "1",
        "XG_AVAILABLE_WAIT": "43200",
        "XG_START_ATTEMPTS": "1",
        "XG_GPU_IDLE_WAIT": "0",
    }
    if not overhead:
        model = "Qwen3.8-27B-FP8" if family == "qwen38" else "gemma-4-31B-it-FP8-dynamic"
        environment["XG_PAGECACHE_ROOTS"] = (
            f"/home/jonas/git/xgraph/data:/home/jonas/models/{model}"
        )
    return {
        "nice": 19,
        "address_space_bytes": None if overhead else 7 * 1024**3,
        "environment": environment,
        "need_gib": 16 if overhead else 60 if family == "qwen38" else 75,
        "headroom_before_worker_charge": not overhead,
        "native_headroom_within_charge": overhead,
        "external_admission_before_charge": True,
        "overhead_inherited_as": "not established by original source; bind owner launch context",
        "admission_wait": "external coordinated wait, then unchanged zero-wait native check",
    }


def native_runtime_environment(profile, *, source, temporary, cuda_compiler, parent):
    """Keep frozen admission settings; remove CPU-gate and old-budget overrides."""
    if not Path(cuda_compiler).is_absolute() or "NOTFOUND" in str(cuda_compiler):
        raise ValueError("qualified absolute CUDA compiler path required")
    removed = {
        "CUDA_VISIBLE_DEVICES",
        "CMAKE_ARGS",
        "SKBUILD_CMAKE_ARGS",
        "SKBUILD_BUILD_DIR",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTEST_ADDOPTS",
        "UV_PROJECT_ENVIRONMENT",
        "VIRTUAL_ENV",
        "MKL_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
        "MALLOC_ARENA_MAX",
    }
    environment = {
        key: value
        for key, value in parent.items()
        if key not in removed and not key.startswith(("XG_", "XGRAPH_", "GIT_"))
    }
    environment.update(profile["environment"])
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        TMPDIR=str(temporary),
        CMAKE_ARGS=f"-DCMAKE_CUDA_COMPILER={cuda_compiler}",
        UV_PROJECT_ENVIRONMENT=str(Path(source) / ".venv"),
        VIRTUAL_ENV=str(Path(source) / ".venv"),
        UV_LINK_MODE="hardlink",
        UV_OFFLINE="1",
        UV_CACHE_DIR="/home/jonas/git/xgraph/target/agent-uv-cache",
    )
    return environment


NATIVE_SOURCE_PREFIXES = ("src/", "csrc/", "scripts/", "cmake/")
NATIVE_SOURCE_TOP = {
    ".gitignore",
    "CMakeLists.txt",
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "uv.lock",
}


def native_source_entries(repository, commit):
    """Read source Git identities, without opening eval or historical run blobs."""
    git = ClaimStore(repository).git
    kind = git("cat-file", "-t", commit, check=False)
    if (
        re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or kind.returncode
        or kind.stdout.strip() != b"commit"
    ):
        raise ValueError("native source requires a full commit object identity")
    entries, omitted = [], []
    for line in git("ls-tree", "-r", "-z", "--full-tree", commit).stdout.split(b"\0"):
        if not line:
            continue
        declaration, raw_path = line.split(b"\t", 1)
        mode, kind, blob = declaration.decode().split()
        relative = raw_path.decode()
        if relative not in NATIVE_SOURCE_TOP and not relative.startswith(NATIVE_SOURCE_PREFIXES):
            omitted.append(relative)
            continue
        if mode not in ("100644", "100755") or kind != "blob":
            raise ValueError(
                f"native source must be regular, not symbolic or a submodule: {relative}"
            )
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"unsafe native source path: {relative}")
        entries.append({"path": relative, "mode": mode, "blob": blob})
    if not any(entry["path"] == "src/xgraph/serve_cost.py" for entry in entries):
        raise ValueError("frozen native source has no serving supervisor")
    return entries, omitted


def snapshot_native_source(source, commit, destination):
    """Materialize exact source blobs in a private sparse checkout, never outputs."""
    source, destination = Path(source).resolve(), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"fresh private native source required: {destination}")
    if destination.resolve() != destination:
        raise ValueError("private native source destination contains a symbolic link")
    entries, omitted = native_source_entries(source, commit)
    original = ClaimStore(source)
    if original.git("rev-parse", "HEAD").stdout.decode().strip() != commit:
        raise ValueError("original native source HEAD differs from frozen commit")
    payloads = [
        (entry, original.git("cat-file", "blob", entry["blob"]).stdout) for entry in entries
    ]
    if sum(len(data) for _, data in payloads) > 128 * 1024**2:
        raise ValueError("unexpected private native source size")
    destination.mkdir()
    files = []
    for entry, data in payloads:
        path = destination / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
        path.chmod(0o755 if entry["mode"] == "100755" else 0o644)
        files.append({**entry, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    ClaimStore(destination).git("init", "--quiet")
    common = (
        original.git("rev-parse", "--path-format=absolute", "--git-common-dir")
        .stdout.decode()
        .strip()
    )
    (destination / ".git/objects/info/alternates").write_text(str(Path(common) / "objects") + "\n")
    private = ClaimStore(destination)
    private.git("update-ref", "HEAD", commit)
    private.git("read-tree", commit)
    if omitted:
        private.git(
            "update-index",
            "--skip-worktree",
            "-z",
            "--stdin",
            data=b"\0".join(path.encode() for path in omitted) + b"\0",
        )
    proof = {
        "root": str(destination),
        "commit": commit,
        "tree": private.git("rev-parse", f"{commit}^{{tree}}").stdout.decode().strip(),
        "source_origin": str(source),
        "source_only_sparse_checkout": True,
        "files": files,
    }
    return verify_native_source(destination, proof)


def verify_native_source(destination, proof):
    destination = Path(destination)
    if str(destination) != proof["root"] or destination.resolve() != destination:
        raise ValueError("private native source root changed")
    git = ClaimStore(destination).git
    if git("rev-parse", "HEAD").stdout.decode().strip() != proof["commit"]:
        raise ValueError("private native source commit changed")
    entries, _omitted = native_source_entries(destination, proof["commit"])
    if [
        {key: entry[key] for key in ("path", "mode", "blob")} for entry in proof["files"]
    ] != entries:
        raise ValueError("private native source manifest changed")
    if git("rev-parse", f"{proof['commit']}^{{tree}}").stdout.decode().strip() != proof["tree"]:
        raise ValueError("private native source tree changed")
    actual = {
        path.name
        for path in destination.iterdir()
        if path.name in NATIVE_SOURCE_TOP and path.is_file()
    }
    for prefix in NATIVE_SOURCE_PREFIXES:
        folder = destination / prefix.rstrip("/")
        if folder.exists():
            if folder.resolve() != folder:
                raise ValueError("native source inventory contains a symbolic directory")
            actual.update(
                str(path.relative_to(destination))
                for path in folder.rglob("*")
                if path.is_file() or path.is_symlink()
            )
    if actual != {entry["path"] for entry in proof["files"]}:
        raise ValueError("private native source inventory has an unregistered addition or deletion")
    originals = ClaimStore(destination).objects(*(entry["blob"] for entry in proof["files"]))
    for entry, original in zip(proof["files"], originals, strict=True):
        path = destination / entry["path"]
        if path.resolve() != path or not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"private native source changed: {path}")
        data = path.read_bytes()
        if (
            original["oid"] != entry["blob"]
            or original["kind"] != "blob"
            or data != original["data"]
            or hashlib.sha256(data).hexdigest() != entry["sha256"]
        ):
            raise ValueError(f"private native source bytes changed: {path}")
    return proof


def snapshot_native_environment(base, destination, source_proof):
    """Reuse the existing inert dependency-path snapshot, never an editable hook."""
    from xgraph.webqsp_test import snapshot_environment

    destination, base = Path(destination), Path(base).resolve()
    verify_native_source(destination, source_proof)
    if (destination / ".venv").exists() or (destination / ".venv").is_symlink():
        raise FileExistsError("private native environment already exists")
    original_configuration = metadata_identity(base / "pyvenv.cfg")
    interpreter = (base / "bin/python").resolve()
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    dependencies = base / "lib" / version / "site-packages"
    if not interpreter.is_file() or not dependencies.is_dir():
        raise ValueError("frozen base interpreter or dependencies are unavailable")
    snapshot_environment(base, destination)
    site = destination / ".venv/lib" / version / "site-packages"
    proof = {
        "base": str(base),
        "python": str(destination / ".venv/bin/python"),
        "resolved_python": str(interpreter),
        "dependencies": str(dependencies),
        "original_configuration": original_configuration,
        "configuration": metadata_identity(destination / ".venv/pyvenv.cfg"),
        "pth": metadata_identity(site / "snapshot.pth"),
        "source_commit": source_proof["commit"],
    }
    proof["inventory"] = private_environment_inventory(destination)
    return proof


def private_environment_inventory(destination):
    """Inventory only owned environment metadata, never linked dependencies."""
    root = Path(destination) / ".venv"
    if root.resolve() != root or not root.is_dir():
        raise ValueError("private environment layout is not a regular owned directory")
    result, pending = {}, [root]
    while pending:
        directory = pending.pop()
        for path in sorted(directory.iterdir()):
            relative = str(path.relative_to(root))
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                result[relative] = {"kind": "link", "target": os.readlink(path)}
            elif stat.S_ISDIR(mode):
                result[relative] = {"kind": "directory"}
                pending.append(path)
            elif stat.S_ISREG(mode):
                result[relative] = {"kind": "file", **metadata_identity(path)}
            else:
                raise ValueError(f"unexpected private environment layout entry: {path}")
    return result


NATIVE_BUILD_SCRIPT = """import json, sys
from pathlib import Path
from xgraph.serve_cost import prepare_runtime
result = prepare_runtime({}, Path(sys.argv[1]), None)
with Path(sys.argv[2]).open('x') as stream:
    json.dump(result, stream, sort_keys=True, allow_nan=False)
    stream.write('\\n')
"""


def native_build_argv(destination, output):
    """Describe the frozen fresh-wheel build; this function never launches it."""
    destination, output = Path(destination), Path(output)
    if output.resolve() != output or not output.is_relative_to(destination / "target"):
        raise ValueError("private native build output must be within its owned target")
    return [
        str(destination / ".venv/bin/python"),
        "-I",
        "-B",
        "-c",
        NATIVE_BUILD_SCRIPT,
        str(output),
        str(output / "preparation.json"),
    ]


def artifact_identity(path, *, maximum=512 * 1024**2):
    """Hash one bounded regular generated artifact without following links."""
    path = Path(path)
    try:
        before = path.lstat()
        if path.resolve() != path or not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise ValueError(f"native runtime artifact must be bounded and regular: {path}")
        digest = hashlib.sha256()
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or opened.st_size != before.st_size:
                raise ValueError(f"native runtime artifact changed before open: {path}")
            size = 0
            while data := stream.read(1024**2):
                size += len(data)
                if size > maximum:
                    raise ValueError(f"native runtime artifact grew beyond its bound: {path}")
                digest.update(data)
        if size != before.st_size or path.stat().st_mtime_ns != before.st_mtime_ns:
            raise ValueError(f"native runtime artifact changed during hashing: {path}")
    except OSError as error:
        raise ValueError(f"native runtime artifact is unavailable: {path}") from error
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def verify_native_wheel(destination, source_proof, runtime):
    """Bind wheel/source/site bytes, without loading a native binary or claiming ABI fidelity."""
    destination = Path(destination)
    verify_native_source(destination, source_proof)
    if set(runtime) != {
        "site",
        "wheel",
        "wheel_sha256",
        "source_snapshot",
        "source_sha256",
        "source_files",
        "preparation_seconds",
    }:
        raise ValueError("native runtime source/build result schema changed")
    nonnegative(runtime["preparation_seconds"], "native runtime preparation seconds")
    wheel, site = Path(runtime["wheel"]), Path(runtime["site"])
    if (
        not wheel.is_relative_to(destination / "target")
        or site != wheel.parent / "site"
        or site.resolve() != site
        or not site.is_dir()
    ):
        raise ValueError("private native wheel/runtime destination changed")
    pin = artifact_identity(wheel)
    if pin["sha256"] != runtime["wheel_sha256"]:
        raise ValueError("native wheel hash changed")
    selected_sources = {
        entry["path"]: entry
        for entry in source_proof["files"]
        if (
            (
                Path(entry["path"]).parts[0] in ("src", "csrc", "scripts")
                and Path(entry["path"]).suffix in (".py", ".cu", ".h", ".sh")
            )
            or entry["path"] in ("pyproject.toml", "uv.lock", "CMakeLists.txt")
        )
    }
    expected_map = {
        str(destination / name): entry["sha256"] for name, entry in selected_sources.items()
    }
    snapshot = Path(runtime["source_snapshot"])
    if snapshot != wheel.parent / "source.zip" or runtime["source_files"] != expected_map:
        raise ValueError("native runtime source snapshot/map differs from private frozen source")
    source_pin = artifact_identity(snapshot, maximum=128 * 1024**2)
    if source_pin["sha256"] != runtime["source_sha256"]:
        raise ValueError("native runtime source archive hash changed")
    with zipfile.ZipFile(snapshot) as archive:
        names = archive.namelist()
        if (
            len(names) != len(set(names))
            or set(names) != set(selected_sources)
            or sum(entry.file_size for entry in archive.infolist()) > 128 * 1024**2
        ):
            raise ValueError("native runtime source archive coverage changed")
        for entry in archive.infolist():
            if (
                stat.S_ISLNK(entry.external_attr >> 16)
                or entry.is_dir()
                or hashlib.sha256(archive.read(entry)).hexdigest()
                != selected_sources[entry.filename]["sha256"]
            ):
                raise ValueError("native runtime source archive bytes differ from frozen source")
    expected_python = {
        entry["path"].removeprefix("src/"): entry
        for entry in source_proof["files"]
        if entry["path"].startswith("src/xgraph/") and entry["path"].endswith(".py")
    }
    entries, engines, seen = {}, [], set()
    with zipfile.ZipFile(wheel) as archive:
        if sum(entry.file_size for entry in archive.infolist()) > 512 * 1024**2:
            raise ValueError("native wheel uncompressed size exceeds the bound")
        for entry in archive.infolist():
            name, relative = entry.filename, Path(entry.filename)
            mode = entry.external_attr >> 16
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in name
                or str(relative) != name.rstrip("/")
                or name in seen
                or stat.S_ISLNK(mode)
            ):
                raise ValueError("native wheel has duplicate, symbolic or unsafe paths")
            seen.add(name)
            if entry.is_dir():
                continue
            if relative.suffix in (".pth", ".pyc") or relative.name in (
                "sitecustomize.py",
                "usercustomize.py",
            ):
                raise ValueError("native wheel contains an unapproved startup hook")
            data = archive.read(entry)
            if relative.suffix == ".py":
                expected = expected_python.get(name)
                if expected is None or hashlib.sha256(data).hexdigest() != expected["sha256"]:
                    raise ValueError("native wheel Python differs from frozen source")
            if relative.suffix == ".so":
                if name not in {
                    "xgraph/_engine" + suffix for suffix in importlib.machinery.EXTENSION_SUFFIXES
                } or not data.startswith(b"\x7fELF"):
                    raise ValueError("native wheel contains an unapproved native engine")
                engines.append(name)
            extracted = site / relative
            extracted_pin = artifact_identity(extracted)
            if (
                extracted_pin["bytes"] != len(data)
                or extracted_pin["sha256"] != hashlib.sha256(data).hexdigest()
            ):
                raise ValueError("extracted native runtime bytes differ from the wheel")
            entries[name] = extracted_pin
    if set(expected_python) - entries.keys() or len(engines) != 1:
        raise ValueError("native wheel source coverage or unique native engine is missing")
    actual = set()
    pending = [site]
    while pending:
        for path in pending.pop().iterdir():
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                pending.append(path)
            elif stat.S_ISREG(mode):
                actual.add(str(path.relative_to(site)))
            else:
                raise ValueError("native runtime site contains an unapproved symbolic entry")
    if actual != set(entries):
        raise ValueError("native runtime site inventory differs from the wheel")
    return {
        "runtime": copy.deepcopy(runtime),
        "wheel": pin,
        "source_snapshot": source_pin,
        "files": entries,
        "engine": entries[engines[0]],
        "source_sha256": json_sha256(source_proof),
        "native_abi_qualification": False,
        "native_gpu_qualification": False,
    }


def enable_native_engine(destination, source_proof, environment_proof, wheel_proof, build):
    """Add only the content-bound wheel engine to the frozen package search path.

    A later bound execution builder must produce and retain the actual fresh
    build receipt. This helper checks that receipt's source/environment/argv
    contract; it does not launch a build or certify a producer-reported success.
    """
    destination = Path(destination)
    verify_native_source(destination, source_proof)
    verify_private_environment(destination, environment_proof)
    if verify_native_wheel(destination, source_proof, wheel_proof["runtime"]) != wheel_proof:
        raise ValueError("native wheel proof changed")
    output = Path(wheel_proof["wheel"]["path"]).parent.parent
    if (
        build.get("argv") != native_build_argv(destination, output)
        or type(build.get("returncode")) is not int
        or build["returncode"] != 0
        or build.get("source_sha256") != json_sha256(source_proof)
        or build.get("environment_sha256") != json_sha256(environment_proof)
        or build.get("wheel_override") is not None
    ):
        raise ValueError("native fresh build lineage, argv or source/environment changed")
    proof = copy.deepcopy(environment_proof)
    proof["native_engine"] = {
        "wheel": copy.deepcopy(wheel_proof),
        "build": copy.deepcopy(build),
        "base_environment_sha256": json_sha256(environment_proof),
    }
    path = Path(proof["pth"]["path"])
    path.write_bytes(
        path.read_bytes()
        + (
            "import xgraph; xgraph.__path__.append("
            + repr(str(Path(wheel_proof["engine"]["path"]).parent))
            + ")\n"
        ).encode()
    )
    proof["pth"] = metadata_identity(path)
    proof["inventory"] = private_environment_inventory(destination)
    return proof


def verify_private_environment(destination, proof):
    destination = Path(destination)
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site = f"lib/{version}/site-packages"
    directories = {"bin", "lib", f"lib/{version}", site}
    links = {"bin/python", "bin/python3", f"bin/{version}"}
    files = {"pyvenv.cfg", f"{site}/snapshot.pth"}
    inventory = private_environment_inventory(destination)
    dependencies = Path(proof["dependencies"])
    if dependencies.resolve() != dependencies or not dependencies.is_dir():
        raise ValueError("private environment dependency path changed")
    for name in ("sitecustomize", "usercustomize"):
        for suffix in ("", ".py", ".pyc", *importlib.machinery.EXTENSION_SUFFIXES):
            path = dependencies / (name + suffix)
            if path.exists() or path.is_symlink():
                raise ValueError("inert dependency path supplies unapproved startup customization")
    if set(inventory) != directories | links | files:
        raise ValueError("private environment startup/layout inventory changed")
    if inventory != proof["inventory"]:
        raise ValueError("private environment startup inventory bytes or links changed")
    for relative in directories:
        if inventory[relative] != {"kind": "directory"}:
            raise ValueError("private environment directory layout changed")
    for relative in links:
        if inventory[relative] != {"kind": "link", "target": proof["resolved_python"]}:
            raise ValueError("private environment interpreter aliases changed")
    expected = (
        "import sys; sys.dont_write_bytecode = True\n"
        + str(destination / "src")
        + "\n"
        + proof["dependencies"]
        + "\n"
    ).encode()
    if "native_engine" in proof:
        native = proof["native_engine"]
        wheel = native["wheel"]
        build = native["build"]
        if (
            build.get("environment_sha256") != native["base_environment_sha256"]
            or build.get("wheel_override") is not None
            or type(build.get("returncode")) is not int
            or build["returncode"] != 0
            or build.get("argv")
            != native_build_argv(destination, Path(wheel["wheel"]["path"]).parent.parent)
        ):
            raise ValueError("native engine build lineage changed")
        for pin in (wheel["wheel"], *wheel["files"].values()):
            if artifact_identity(pin["path"]) != pin:
                raise ValueError("native wheel/runtime engine bytes changed")
        expected += (
            "import xgraph; xgraph.__path__.append("
            + repr(str(Path(wheel["engine"]["path"]).parent))
            + ")\n"
        ).encode()
    if Path(proof["pth"]["path"]).read_bytes() != expected:
        raise ValueError("private environment startup recipe changed")
    return proof


def attest_native_module_launch(destination, source_proof, environment_proof, *, cwd):
    """Attest the frozen bare -m import shape without changing its argv."""
    destination, cwd = Path(destination), Path(cwd)
    verify_native_source(destination, source_proof)
    verify_private_environment(destination, environment_proof)
    if (
        cwd != destination / "target/native-launch-cwd"
        or cwd.resolve() != cwd
        or not cwd.is_dir()
        or any(cwd.iterdir())
    ):
        raise ValueError("native bare module working directory must be the fixed empty private cwd")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", "PYTHONHOME") and not key.startswith("GIT_")
    }
    environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="")
    script = """import importlib, json, os, site, sys
from pathlib import Path
names = ('post_queue', 'bench', 'engine')
modules = {name:str(Path(importlib.import_module('xgraph.'+name).__file__).resolve())
           for name in names}
print(json.dumps({'modules':modules,
                  'executable':sys.executable, 'cwd':str(Path.cwd()),
                  'isolated_flag':bool(sys.flags.isolated),
                  'user_site_enabled':site.ENABLE_USER_SITE,
                  'PYTHONPATH_present':'PYTHONPATH' in os.environ,
                  'PYTHONHOME_present':'PYTHONHOME' in os.environ}))
"""
    result = subprocess.run(
        [environment_proof["python"], "-B", "-c", script],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    proof = json.loads(result.stdout)
    if (
        proof.get("modules")
        != {
            name: str(destination / "src/xgraph" / (name + ".py"))
            for name in ("post_queue", "bench", "engine")
        }
        or proof.get("executable") != environment_proof["python"]
        or proof.get("cwd") != str(cwd)
        or proof.get("isolated_flag") is not False
        or proof.get("user_site_enabled") is not False
        or proof.get("PYTHONPATH_present") is not False
        or proof.get("PYTHONHOME_present") is not False
    ):
        raise ValueError(
            "native bare module import/source environment differs from the private proof"
        )
    return proof


def native_attestation_argv(environment_proof, *, load_native):
    return [
        str(environment_proof["python"]),
        "-I",
        "-B",
        "-c",
        NATIVE_ATTESTATION_SCRIPT,
        "load-native" if load_native else "resolve-only",
    ]


def native_metadata_helper_argvs(context):
    """Enumerate fixed verification argv; qualification must declare, never grant, them."""
    controller = context["source_proof"]
    commit, native_commit = controller["commit"], context["source"]["commit"]
    arguments = [
        ("cat-file", "--batch"),
        ("rev-parse", "--path-format=absolute", "--git-common-dir"),
        ("cat-file", "-t", commit),
        ("cat-file", "-t", native_commit),
        ("rev-parse", "HEAD"),
        ("status", "--porcelain", "--untracked-files=normal"),
        ("show", f"{commit}:src/xgraph/serving_recovery.py"),
        ("show", f"{commit}:src/xgraph/serving_recovery_monitor.py"),
        ("ls-tree", "-r", "-z", "--full-tree", native_commit),
        ("rev-parse", f"{native_commit}^{{tree}}"),
        *(("show", f"{commit}:{entry['path']}") for entry in controller["decisions"].values()),
    ]
    if "engineering_amendment" in context:
        original = context["registered_source_proof"]
        old = original["commit"]
        arguments.extend(
            [
                ("cat-file", "-t", old),
                ("show", f"{old}:src/xgraph/serving_recovery.py"),
                ("show", f"{old}:src/xgraph/serving_recovery_monitor.py"),
                *(("show", f"{old}:{entry['path']}") for entry in original["decisions"].values()),
                ("show", f"{commit}:{ENGINEERING_AMENDMENT_PATH}"),
            ]
        )
    commands = {tuple(ClaimStore.argv(*items)) for items in arguments}
    commands.add(tuple(native_attestation_argv(context["environment"], load_native=True)))
    return [list(argv) for argv in sorted(commands)]


def attest_native_environment(destination, source_proof, environment_proof, *, load_native=False):
    """Isolated source/dependency attestation; no model, Docker or CUDA call."""
    destination = Path(destination)
    verify_native_source(destination, source_proof)
    verify_private_environment(destination, environment_proof)
    if type(load_native) is not bool or (load_native and "native_engine" not in environment_proof):
        raise ValueError("native ABI loading requires the content-bound wheel/environment")
    for key in ("original_configuration", "configuration", "pth"):
        pin = environment_proof[key]
        if metadata_identity(pin["path"]) != pin:
            raise ValueError(f"private native environment metadata changed: {key}")
    python = Path(environment_proof["python"])
    if (
        python != destination / ".venv/bin/python"
        or str(python.resolve()) != environment_proof["resolved_python"]
        or environment_proof["source_commit"] != source_proof["commit"]
    ):
        raise ValueError("private native interpreter or source binding changed")
    if (
        "native_engine" in environment_proof
        and verify_native_wheel(
            destination, source_proof, environment_proof["native_engine"]["wheel"]["runtime"]
        )
        != environment_proof["native_engine"]["wheel"]
    ):
        raise ValueError("native wheel source/runtime proof changed")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", "PYTHONHOME") and not key.startswith("GIT_")
    }
    environment.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        native_attestation_argv(environment_proof, load_native=load_native),
        env=environment,
        cwd=destination,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        raise ValueError(f"isolated native import attestation failed: {result.stderr[-2000:]}")
    attested = json.loads(result.stdout)
    if attested["executable"] != str(python) or any(
        path != str(destination / "src/xgraph" / f"{name}.py")
        for name, path in attested["modules"].items()
    ):
        raise ValueError("native modules imported outside the private frozen source")
    if "native_engine" in environment_proof and attested["engine_resolution"] != {
        "path": environment_proof["native_engine"]["wheel"]["engine"]["path"],
        "loader": "ExtensionFileLoader",
    }:
        raise ValueError("native engine resolved outside the content-bound frozen wheel")
    if (
        load_native
        and attested["native_binary"]
        != environment_proof["native_engine"]["wheel"]["engine"]["path"]
    ):
        raise ValueError("native ABI binary imported outside the bound frozen wheel")
    return {
        **attested,
        "source_commit": source_proof["commit"],
        "worker_script": str(destination / "scripts/serve_cost_trials.py"),
        "native_abi_qualification": False,
        "native_gpu_qualification": False,
    }


class ClaimStore:
    """Atomic create-once commit claims in Git's worktree-shared namespace."""

    def __init__(self, repo):
        self.repo = Path(repo).resolve()

    @staticmethod
    def argv(*arguments):
        return [
            "git",
            "-c",
            "user.name=xgraph owner",
            "-c",
            "user.email=owner@xgraph.invalid",
            "-c",
            "commit.gpgSign=false",
            *arguments,
        ]

    def git(self, *arguments, data=None, check=True):
        environment = {
            key: value for key, value in os.environ.items() if not key.startswith("GIT_")
        }
        environment.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
        return subprocess.run(
            self.argv(*arguments),
            cwd=self.repo,
            env=environment,
            input=data,
            capture_output=True,
            check=check,
        )

    def objects(self, *expressions):
        """Read exact Git objects through one fixed, declarable helper argv."""
        if (
            not expressions
            or len(expressions) > 4096
            or any(
                not isinstance(expression, str)
                or not expression
                or len(expression) > 8192
                or "\n" in expression
                or "\r" in expression
                or "\0" in expression
                for expression in expressions
            )
        ):
            raise ValueError("bounded Git object expressions are required")
        raw = self.git("cat-file", "--batch", data=("\n".join(expressions) + "\n").encode()).stdout
        if len(raw) > 64 * 1024**2:
            raise ValueError("Git source/control object batch exceeds its metadata bound")
        result, offset = [], 0
        for _expression in expressions:
            boundary = raw.find(b"\n", offset)
            if boundary < 0:
                raise ValueError("Git object batch header is incomplete")
            header = raw[offset:boundary]
            if header.endswith(b" missing"):
                raise ValueError("required Git source/control object is missing")
            fields = header.split()
            if (
                len(fields) != 3
                or re.fullmatch(rb"[0-9a-f]{40}", fields[0]) is None
                or fields[1] not in (b"blob", b"tree", b"commit", b"tag")
                or not fields[2].isdigit()
            ):
                raise ValueError("Git object batch type/identity/size is invalid")
            size = int(fields[2])
            start, end = boundary + 1, boundary + 1 + size
            data = raw[start:end]
            if len(data) != size or raw[end : end + 1] != b"\n":
                raise ValueError("Git object batch body is incomplete")
            if (
                hashlib.sha1(fields[1] + b" " + fields[2] + b"\0" + data).hexdigest().encode()
                != fields[0]
            ):
                raise ValueError("Git object batch content identity changed")
            result.append({"oid": fields[0].decode(), "kind": fields[1].decode(), "data": data})
            offset = end + 1
        if offset != len(raw):
            raise ValueError("Git object batch has unexpected trailing data")
        return result

    def reference(self, phase):
        if phase not in ("registration", "execution", "launch", "engineering-amendment"):
            raise ValueError(f"unknown serving recovery claim phase: {phase}")
        return f"{CLAIM_NAMESPACE}/{phase}"

    def create(self, phase, record):
        reference = self.reference(phase)
        if self.git("rev-parse", "--verify", reference, check=False).returncode == 0:
            raise FileExistsError(f"serving recovery {phase} already claimed; no reset")
        payload = {
            "artifact_kind": "xgraph.serving_recovery_claim",
            "phase": phase,
            "record": record,
        }
        data = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        blob = self.git("hash-object", "-w", "--stdin", data=data).stdout.strip().decode()
        tree = (
            self.git("mktree", data=f"100644 blob {blob}\tclaim.json\n".encode())
            .stdout.strip()
            .decode()
        )
        commit = (
            self.git("commit-tree", tree, data=b"Bind one prospective serving recovery\n")
            .stdout.strip()
            .decode()
        )
        if phase == "engineering-amendment":
            transaction = "\n".join(
                [
                    "start",
                    f"verify {CLAIM_NAMESPACE}/launch {'0' * 40}",
                    f"verify {CLAIM_NAMESPACE}/budget {'0' * 40}",
                    f"create {reference} {commit}",
                    "prepare",
                    "commit",
                    "",
                ]
            ).encode()
            result = self.git("update-ref", "--stdin", data=transaction, check=False)
        else:
            result = self.git("update-ref", reference, commit, "0" * 40, check=False)
        if result.returncode:
            reason = "already claimed"
            if phase == "engineering-amendment":
                reason += " or launch/budget present"
            raise FileExistsError(f"serving recovery {phase} {reason}; no reset")
        return commit

    def verify(self, phase, record):
        reference = self.reference(phase)
        try:
            head = self.objects(reference)[0]
        except ValueError as error:
            raise ValueError(f"serving recovery {phase} claim missing or invalid") from error
        if head["kind"] != "commit":
            raise ValueError("serving recovery claim must be a commit object")
        commit = head["oid"]
        retained = self.objects(f"{commit}:claim.json")[0]
        if retained["kind"] != "blob":
            raise ValueError("serving recovery claim metadata must be a blob object")
        saved = json.loads(retained["data"])
        if saved != {
            "artifact_kind": "xgraph.serving_recovery_claim",
            "phase": phase,
            "record": record,
        }:
            raise ValueError(f"serving recovery {phase} claim binding changed")
        return commit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--register", action="store_true")
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--amend-engineering", action="store_true")
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--owner-review", type=Path)
    parser.add_argument("--preparation-review", type=Path)
    parser.add_argument("--engineering-review", type=Path)
    parser.add_argument("--argv-sha256")
    args = parser.parse_args(argv)
    protocol = load_protocol()
    if args.amend_engineering:
        if args.registration is None or args.engineering_review is None:
            raise ValueError(
                "unbound engineering amendment: exact registration and source review required"
            )
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != _SOURCE_BYTES_SHA256:
            raise ValueError(
                "engineering writes require the reviewed source-bytes controller bootstrap"
            )
        adopted = adopt_engineering_amendment(args.registration, args.engineering_review)
        path = Path(adopted["request"]["canonical_child"]).parent / "engineering-amendment.json"
        print(
            json.dumps(
                {
                    "canonical_child": adopted["request"]["canonical_child"],
                    "status": "ENGINEERING_AMENDED_NOT_LAUNCH_READY",
                    "engineering_amendment": metadata_identity(path),
                    "gpu_execution": False,
                }
            )
        )
        return
    if args.run:
        if args.registration is None or args.owner_review is None:
            raise ValueError(
                "unbound serving recovery: exact registration and owner review required"
            )
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != _SOURCE_BYTES_SHA256:
            raise ValueError(
                "unbound serving recovery: reviewed source-bytes owner launcher required"
            )
        print(json.dumps(run_native_recovery(args.registration, args.owner_review)))
        return
    if args.register or args.prepare:
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != _SOURCE_BYTES_SHA256:
            raise ValueError(
                "metadata writes require the reviewed source-bytes controller bootstrap"
            )
        if args.register:
            if args.owner_review is None or args.argv_sha256 is None:
                raise ValueError(
                    "unbound registration: exact owner review and argv digest required"
                )
            result = register_metadata(args.owner_review, argv_sha256=args.argv_sha256)
        else:
            if args.registration is None or args.preparation_review is None:
                raise ValueError(
                    "unbound preparation: exact registration and preparation review required"
                )
            result = prepare_native_execution(args.registration, args.preparation_review)
        print(
            json.dumps(
                {
                    "canonical_child": result["canonical_child"],
                    "status": result.get("status", "REGISTERED_NOT_LAUNCH_READY"),
                    "gpu_execution": False,
                }
            )
        )
        return
    print(
        json.dumps(
            {
                "status": "UNBOUND_NOT_LAUNCH_READY",
                "id": protocol["id"],
                "cumulative_gpu_seconds": protocol["cap"]["cumulative_gpu_seconds"],
                "canonical_child": protocol["canonical_child"],
                "parent_metadata_opened": False,
                "gpu_execution": False,
            }
        )
    )


def build_schedule(protocol, parent_seal, plans):
    """Relocate only orchestration paths; retain every original planned cell.

    Study JSON is opaque here because it embeds answers. It is pinned by the
    parent seal, copied byte for byte during later authorized preparation, and
    decoded only by the original worker during authorized execution.
    """
    if [stage["name"] for stage in parent_seal["stages"]] != list(STAGES):
        raise ValueError("original stage order or coverage changed")
    root = Path(protocol["canonical_child"])
    parent = Path(parent_seal["out"])
    if not root.is_absolute() or root == parent or root.is_relative_to(parent):
        raise ValueError("fresh canonical child destination required")
    schedule = []
    for stage in parent_seal["stages"]:
        name = stage["name"]
        primary, overhead = name.startswith("locked"), name.startswith("overhead")
        expected_kind = "overhead" if overhead else "serving"
        if stage["kind"] != expected_kind:
            raise ValueError(f"{name}: original stage kind changed")
        relocated = copy.deepcopy(stage)
        relocated["output"] = str(root / Path(stage["output"]).relative_to(parent))
        relocated["command"] = [
            str(root / Path(token).relative_to(parent))
            if token.startswith(str(parent) + "/")
            else token
            for token in stage["command"]
        ]
        jobs = []
        if overhead:
            jobs.append(
                {
                    "id": name,
                    "stage": name,
                    "kind": "overhead",
                    "output": relocated["output"],
                    "command": relocated["command"],
                    "wall_seconds": stage["wall_bound_seconds"],
                    "block": name,
                }
            )
        else:
            plan = plans[name]
            agent = "agent-feedback-8" if name.endswith("gemma31") else "agent-strong-8"
            arms = ["finalist", agent] if primary else ["W", "B2"]
            expected, rng = [], random.Random(17)
            for repeat in range(5 if primary else 3):
                cells = [(arm, concurrency) for arm in arms for concurrency in (1, 8)]
                rng.shuffle(cells)
                expected.extend(
                    (repeat, arm, "warm", concurrency, 17 + repeat) for arm, concurrency in cells
                )
            keys = ("repeat", "arm", "cache", "concurrency", "seed")
            actual = [tuple(trial[key] for key in keys) for trial in plan["trials"]]
            if plan["seed"] != 17 or actual != expected:
                raise ValueError(f"{name}: original matrix, paired seeds, cache or order changed")
            for number, original in enumerate(plan["trials"]):
                if original["status"] != "planned":
                    raise ValueError(f"{name}: original timing reuse is forbidden")
                trial = copy.deepcopy(original)
                trial["output"] = str(Path(relocated["output"]).parent / f"trial-{number:04d}.json")
                jobs.append(
                    {
                        "id": f"{name}/{number:04d}",
                        "stage": name,
                        "kind": "primary" if primary else "descriptive",
                        "trial": trial,
                        "original_trial": copy.deepcopy(original),
                        "output": trial["output"],
                        "wall_seconds": original["wall_bound_seconds"],
                        "kv_cache_bytes": plan["kv_cache_bytes"],
                        "opaque_study": stage["plan"],
                        "block": f"{name}/{trial['repeat']}",
                    }
                )
        relocated["jobs"] = jobs
        schedule.append(relocated)
    return schedule


def prepare_relocated_trial_plan(stage, parent_pin, *, controller_commit, native_commit):
    """Retain a new immutable planned matrix; never copy original output paths."""
    name = stage["name"]
    count = 20 if name.startswith("locked-") else 12
    if (
        name not in STAGES[:4]
        or len(stage["jobs"]) != count
        or [job["id"] for job in stage["jobs"]] != [f"{name}/{i:04d}" for i in range(count)]
        or any(
            re.fullmatch(r"[0-9a-f]{40}", value) is None
            for value in (controller_commit, native_commit)
        )
        or metadata_identity(parent_pin["path"]) != parent_pin
    ):
        raise ValueError(
            "relocated trial plan needs the fixed stage, parent metadata and source commits"
        )
    original = json.loads(Path(parent_pin["path"]).read_bytes())
    jobs = stage["jobs"]
    if (
        original.get("trials") != [job["original_trial"] for job in jobs]
        or original.get("seed") != 17
        or any(original.get("kv_cache_bytes") != job["kv_cache_bytes"] for job in jobs)
        or any(
            job["trial"] != {**job["original_trial"], "output": job["output"]}
            or job["trial"].get("status") != "planned"
            for job in jobs
        )
    ):
        raise ValueError("relocated trial plan changed original science, order or fresh status")
    destination = Path(stage["output"]).parent / "trials.json"
    if any(Path(job["output"]).parent != destination.parent for job in jobs):
        raise ValueError("relocated trial plan has a foreign output destination")
    plan = {
        **copy.deepcopy(original),
        "trials": [copy.deepcopy(job["trial"]) for job in jobs],
        "parent_provenance": copy.deepcopy(original.get("provenance")),
        "provenance": {
            "controller_commit": controller_commit,
            "native_commit": native_commit,
            "seed": 17,
            "stage": name,
            "parent_plan": copy.deepcopy(parent_pin),
            "command": [sys.executable, *sys.argv],
            "created_at": datetime.now(UTC).isoformat(),
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    immutable_json(destination, plan)
    return metadata_identity(destination)


def write_json(path, value):
    """Replace owned metadata, refusing a substituted symbolic link."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"metadata path is a symbolic link: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def nonnegative(value, context):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}: expected a finite nonnegative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{context}: expected a finite nonnegative number")
    return value


def process_identity(process):
    try:
        identity = process["pid"], process["startticks"]
    except (KeyError, TypeError) as error:
        raise ValueError("incomplete process identity") from error
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in identity
    ):
        raise ValueError(f"invalid process identity: {identity}")
    return identity


def permitted_identity(process, policy):
    """Retain measured early-boot zero only under the adopted exact OS proof."""
    if process.get("startticks") != 0 or type(process.get("startticks")) is not int:
        return process_identity(process)
    pid = process.get("pid")
    proof = process.get("early_boot_proof")
    if (
        type(pid) is not int
        or pid <= 0
        or not isinstance(policy, dict)
        or policy.get("context_decision_sha256") != OS_CONTEXT_SHA256
        or not isinstance(proof, dict)
    ):
        raise ValueError("measured early-boot process identity proof is missing")
    role = next((item for item in policy["roles"] if item["id"] == proof.get("role")), None)
    if (
        role is None
        or role.get("early_boot") is not True
        or proof
        != {
            "decision_sha256": OS_CONTEXT_SHA256,
            "role": role["id"],
            "measured": True,
            "boot_id": policy["boot_id"],
            "pid_namespace": policy["pid_namespace"],
            "pid": pid,
            "startticks": 0,
        }
        or not (
            (
                role.get("kernel") is True
                and {"pid": pid, "startticks": 0} in role.get("identities", [])
            )
            or (
                role.get("identity_basis") == "boot-manager"
                and role.get("corroboration", {}).get("pid") == pid
                and role["corroboration"].get("startticks") == 0
            )
        )
    ):
        raise ValueError("early-boot identity is not the exact predeclared measured OS proof")
    return pid, 0


def qualify_outer_coverage(samples, *, start, end, permitted=(), owned=(), policy=None):
    """Qualify only the observed coordinated interval, never inferred absence.

    Inputs are retained operational inventory projections, not latency or quality
    rows. PID reuse cannot inherit permission. Native foreign-GPU rejection is
    stricter than any OS background permission and remains in force.
    """
    nonnegative(start, "coverage start")
    nonnegative(end, "coverage end")
    if end < start or len(samples) < 2:
        raise ValueError("coverage boundary samples missing")
    times = [nonnegative(sample["at"], "inventory timestamp") for sample in samples]
    if any(b <= a for a, b in pairwise(times)):
        raise ValueError("inventory coverage timestamps are not strictly ordered")
    before = [i for i, value in enumerate(times) if value <= start]
    after = [i for i, value in enumerate(times) if value >= end]
    if not before or not after:
        raise ValueError("coverage boundary samples missing")
    selected = samples[before[-1] : after[0] + 1]
    gaps = [b["at"] - a["at"] for a, b in pairwise(selected)]
    if not gaps or max(gaps) > 5:
        raise ValueError("inventory coverage gap exceeds five seconds")
    allowed = {permitted_identity(p, policy) for p in permitted}
    own = {process_identity(p) for p in owned}
    for sample in selected:
        if (
            not {"at", "complete", "errors", "processes", "races"} <= set(sample)
            or sample["complete"] is not True
            or not all(
                isinstance(sample[field], list) for field in ("errors", "processes", "races")
            )
            or sample["errors"]
        ):
            raise ValueError("incomplete or failed resource inventory coverage")
        for race in sample.get("races", []):
            if race.get("classification") != "owned-exit" or process_identity(race) not in own:
                raise ValueError("unclassified process identity race in inventory coverage")
        for process in sample["processes"]:
            identity = permitted_identity(process, policy)
            if type(process.get("active")) is not bool or type(process.get("gpu")) is not bool:
                raise ValueError("unknown activity or GPU classification in process inventory")
            if process.get("gpu") and identity not in own:
                raise ValueError(
                    "foreign GPU compute identity; background permissions do not apply"
                )
            if process["active"] and identity not in allowed | own:
                raise ValueError("unclassified relevant active process identity in inventory")
    return {
        "qualified": True,
        "target_sample_seconds": 1,
        "maximum_allowed_gap_seconds": 5,
        "max_gap_seconds": max(gaps),
        "sample_count": len(selected),
        "start": start,
        "end": end,
        "scope": (
            "observed outer process inventories; unsampled/graphics-only activity not proved absent"
        ),
    }


def write_inventory_archive(directory, evidence):
    """Retain immutable metadata plus one JSONL frame at a time, without a giant JSON string."""
    directory = Path(directory)
    if directory.resolve() != directory:
        raise ValueError("inventory archive directory must be canonical")
    directory.mkdir(parents=True)
    count = 0
    with (directory / "frames.jsonl").open("xb") as stream:
        for frame in evidence["samples"]:
            stream.write((json.dumps(frame, sort_keys=True, allow_nan=False) + "\n").encode())
            count += 1
        stream.flush()
        os.fsync(stream.fileno())
    if type(evidence.get("frame_count")) is not int or count != evidence["frame_count"]:
        raise ValueError("inventory archive frame count is incomplete; partial artifacts retained")
    header = {
        "artifact_kind": "xgraph.serving_recovery_inventory_archive",
        "version": 1,
        "evidence": {
            key: copy.deepcopy(value) for key, value in evidence.items() if key != "samples"
        },
    }
    immutable_json(directory / "metadata.json", header)
    return {
        "metadata": artifact_identity(directory / "metadata.json", maximum=128 * 1024**2),
        "frames": artifact_identity(directory / "frames.jsonl", maximum=2 * 1024**3),
    }


def retain_inventory_archive(directory, evidence):
    """Seal a finished monitor stream without materializing or re-encoding its frames."""
    directory = Path(directory)
    if (
        directory.resolve() != directory
        or not isinstance(evidence, dict)
        or "samples" in evidence
        or not isinstance(evidence.get("archive"), dict)
    ):
        raise ValueError("durable inventory needs a finished stream and canonical directory")
    stream = evidence["archive"]
    if (
        stream.get("closed") is not True
        or stream.get("failed") is not False
        or stream.get("writer_identity", {}).get("directory") != str(directory)
        or stream["writer_identity"].get("archive_name") != "frames.xgm.gz"
    ):
        raise ValueError("durable inventory stream is unclosed or belongs to another trial")
    immutable_json(
        directory / "metadata.json",
        {
            "artifact_kind": "xgraph.serving_recovery_inventory_archive",
            "version": 2,
            "evidence": copy.deepcopy(evidence),
        },
    )
    return {
        "metadata": artifact_identity(directory / "metadata.json", maximum=128 * 1024**2),
        "stream": copy.deepcopy(stream),
    }


def read_inventory_archive(archive, *, directory, source_proof=None):
    """Verify fixed paths before opening, then expose a bounded single-pass frame stream."""
    directory = Path(directory)
    if isinstance(archive, dict) and set(archive) == {"metadata", "stream"}:
        if source_proof is None:
            raise ValueError("durable archive replay requires the source-bound reader")
        meta = archive["metadata"]
        if (
            directory.resolve() != directory
            or meta.get("path") != str(directory / "metadata.json")
            or artifact_identity(meta["path"], maximum=128 * 1024**2) != meta
            or read_seal_digest(directory / "metadata.json.sha256") != meta["sha256"]
        ):
            raise ValueError("durable archive canonical metadata path/bytes/seal changed")
        data = Path(meta["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != meta["sha256"]:
            raise ValueError("durable archive metadata changed before decoding")
        header = json.loads(data)
        if (
            set(header) != {"artifact_kind", "version", "evidence"}
            or header["artifact_kind"] != "xgraph.serving_recovery_inventory_archive"
            or type(header["version"]) is not int
            or header["version"] != 2
            or not isinstance(header["evidence"], dict)
            or "samples" in header["evidence"]
            or header["evidence"].get("archive") != archive["stream"]
        ):
            raise ValueError("durable archive evidence/header binding changed")
        collector = load_controller_monitor(source_proof)
        reader = collector.ArchiveReader(archive["stream"], directory=directory)
        return {**header["evidence"], "samples": reader.iter_frames()}
    if (
        set(archive) != {"metadata", "frames"}
        or directory.resolve() != directory
        or archive["metadata"].get("path") != str(directory / "metadata.json")
        or archive["frames"].get("path") != str(directory / "frames.jsonl")
    ):
        raise ValueError("inventory archive canonical paths changed")
    meta, frames = archive["metadata"], archive["frames"]
    if (
        artifact_identity(meta["path"], maximum=128 * 1024**2) != meta
        or read_seal_digest(directory / "metadata.json.sha256") != meta["sha256"]
        or artifact_identity(frames["path"], maximum=2 * 1024**3) != frames
    ):
        raise ValueError("inventory archive bytes or metadata seal changed")
    data = Path(meta["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != meta["sha256"]:
        raise ValueError("inventory archive metadata changed before decoding")
    header = json.loads(data)
    if (
        header.get("artifact_kind") != "xgraph.serving_recovery_inventory_archive"
        or type(header.get("version")) is not int
        or header["version"] != 1
        or not isinstance(header.get("evidence"), dict)
        or "samples" in header["evidence"]
    ):
        raise ValueError("inventory archive metadata schema changed")

    def rows():
        descriptor = os.open(frames["path"], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        digest, size = hashlib.sha256(), 0
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size != frames["bytes"]:
                raise ValueError("inventory archive frame file changed before replay")
            while line := stream.readline(32 * 1024**2 + 1):
                size += len(line)
                if len(line) > 32 * 1024**2 or not line.endswith(b"\n") or size > frames["bytes"]:
                    raise ValueError("inventory archive frame is incomplete or oversized")
                digest.update(line)
                yield json.loads(line)
        if size != frames["bytes"] or digest.hexdigest() != frames["sha256"]:
            raise ValueError("inventory archive frame digest changed during replay")

    return {**header["evidence"], "samples": rows()}


def qualify_archived_coverage(evidence, *, source_proof, expected_policy, expected_intent):
    """Exhaust the source-bound lossless decoder with only two full scans in RAM."""
    required = {
        "encoding",
        "samples",
        "baseline",
        "identity_registry",
        "frame_count",
        "last_frame_sha256",
        "interval",
        "owned",
        "permitted",
        "errors",
        "violations",
    }
    if (
        not required <= evidence.keys()
        or evidence["encoding"] != "inventory-delta-v1"
        or evidence["errors"] != []
        or evidence["violations"] != []
        or type(evidence["frame_count"]) is not int
        or evidence["frame_count"] < 2
        or re.fullmatch(r"[0-9a-f]{64}", evidence["last_frame_sha256"]) is None
    ):
        raise ValueError(
            "retained inventory archive schema, errors or complete anchors are invalid"
        )
    baseline = evidence["baseline"]
    if (
        expected_policy.get("decision_sha256") != OS_BACKGROUND_SHA256
        or expected_policy.get("context_decision_sha256") != OS_CONTEXT_SHA256
        or json_sha256(baseline.get("policy")) != json_sha256(expected_policy)
        or json_sha256(baseline.get("intent")) != json_sha256(expected_intent)
    ):
        raise ValueError("archive baseline differs from the reviewed policy/intent/decision")
    start = nonnegative(evidence["interval"]["start"], "archive interval start")
    end = nonnegative(evidence["interval"]["end"], "archive interval end")
    if end < start:
        raise ValueError("archive interval boundaries are reversed")
    collector = load_controller_monitor(source_proof)
    previous, count, maximum_gap, before, after = None, 0, 0, False, False
    # Never break once end is covered: final registry/count/chain checks run at
    # iterator exhaustion and can invalidate an otherwise clean-looking prefix.
    for snapshot in collector.iter_reconstructed_evidence(evidence):
        at = nonnegative(snapshot["at"], "archive inventory timestamp")
        if (
            any(
                snapshot.get(key) != expected_policy[key]
                for key in ("boot_id", "pid_namespace", "gpu_uuid")
            )
            or snapshot.get("ownership_complete") is not True
            or not nonnegative(snapshot["scan_start"], "archive scan start") <= at
            or nonnegative(snapshot["scan_end"], "archive scan end") != at
        ):
            raise ValueError("archive host/ownership or scan boundary binding changed")
        if previous is not None:
            pair = qualify_outer_coverage(
                [previous, snapshot],
                start=previous["at"],
                end=at,
                permitted=evidence["permitted"],
                owned=evidence["owned"],
                policy=expected_policy,
            )
            maximum_gap = max(maximum_gap, pair["max_gap_seconds"])
        before, after = before or at <= start, after or at >= end
        previous, count = snapshot, count + 1
    if count != evidence["frame_count"] or not before or not after:
        raise ValueError("archive inventory count or full interval boundaries are incomplete")
    return {
        "qualified": True,
        "archive_fully_exhausted": True,
        "sample_count": count,
        "max_gap_seconds": maximum_gap,
        "maximum_allowed_gap_seconds": 5,
        "target_sample_seconds": 1,
        "start": start,
        "end": end,
        "policy_sha256": json_sha256(expected_policy),
        "intent_sha256": json_sha256(expected_intent),
        "collector_sha256": source_proof["monitor"]["sha256"],
        "encoding": evidence["encoding"],
        "scope": "sampled coordinated compute inventories; no proof of zero interference",
    }


def qualify_coverage(
    samples, *, start, end, native_samples, native_duration_seconds=None, permitted=(), owned=()
):
    """Add the original serving worker's own clock and four phase boundaries."""
    outer = qualify_outer_coverage(samples, start=start, end=end, permitted=permitted, owned=owned)
    return {
        **outer,
        **qualify_native_boundaries(
            native_samples,
            native_duration_seconds=native_duration_seconds,
            outer_duration_seconds=end - start,
        ),
    }


def qualify_native_boundaries(native_samples, *, native_duration_seconds, outer_duration_seconds):
    """Keep the frozen native clock separate from the outer supervisor clock."""
    phases, phase_times = [], []
    for sample in native_samples:
        phase = sample.get("phase")
        if not phases or phases[-1] != phase:
            phases.append(phase)
        phase_times.append(nonnegative(sample["seconds"], "native coverage timestamp"))
    if phases != ["startup", "warmup", "measured", "teardown"]:
        raise ValueError("native phase boundary coverage missing or reordered")
    if native_duration_seconds is None:
        raise ValueError("native duration boundary binding is missing")
    duration = nonnegative(native_duration_seconds, "native duration")
    if (
        phase_times[0] > 5
        or abs(phase_times[-1] - duration) > 5
        or duration > outer_duration_seconds + 5
    ):
        raise ValueError("native boundary samples do not cover bound native driver duration")
    if any(b < a or b - a > 5 for a, b in pairwise(phase_times)):
        raise ValueError("native phase coverage gap exceeds five seconds")
    return {
        "native_phases": phases,
        "native_duration_seconds": duration,
        "scope": (
            "observed process inventories and frozen worker GPU checks; "
            "transient and graphics-only contexts not proved absent"
        ),
    }


def _outer_evidence(
    evidence, cleanup, *, source_proof=None, expected_policy=None, expected_intent=None
):
    if cleanup.get("confirmed_absent") is not True or cleanup.get("errors") != []:
        raise ValueError("owned cleanup is not confirmed without errors")
    if evidence.get("errors") != [] or evidence.get("violations") != []:
        raise ValueError("external control/coverage errors or forbidden activity")
    if evidence.get("encoding") is not None:
        if source_proof is None or expected_policy is None or expected_intent is None:
            raise ValueError("encoded evidence requires reviewed collector/policy/intent bindings")
        return qualify_archived_coverage(
            evidence,
            source_proof=source_proof,
            expected_policy=expected_policy,
            expected_intent=expected_intent,
        )
    return qualify_outer_coverage(
        evidence["samples"],
        start=evidence["start"],
        end=evidence["end"],
        permitted=evidence["permitted"],
        owned=evidence["owned"],
        policy=expected_policy,
    )


def qualify_native_control(
    job,
    control,
    evidence,
    cleanup,
    *,
    server_name,
    expected_effective,
    source_proof=None,
    expected_policy=None,
    expected_intent=None,
):
    """Check only operational fields; no answer, quality or latency value is a predicate."""
    outer = _outer_evidence(
        evidence,
        cleanup,
        source_proof=source_proof,
        expected_policy=expected_policy,
        expected_intent=expected_intent,
    )
    expected_trial = {**job["trial"], "status": "running", "server_name": server_name}
    if (
        job["kind"] not in ("primary", "descriptive")
        or not server_name
        or control.get("artifact_kind") != "xgraph.controlled_serving"
        or type(control.get("version")) is not int
        or control["version"] != 1
        or control.get("status") != "completed"
        or control.get("trial") != expected_trial
        or control.get("study_path") != str(Path(job["output"]).parent / "study.json")
        or type(control.get("kv_cache_bytes")) is not int
        or control["kv_cache_bytes"] != job["kv_cache_bytes"]
        or type(control.get("prefill_declines")) is not int
        or control["prefill_declines"] != 0
        or any(
            control.get(key) is not None
            for key in ("error", "teardown_error", "monitor_error", "watchdog")
        )
    ):
        raise ValueError("native control, trial/server binding or successful status changed")
    expected_effective = copy.deepcopy(expected_effective)
    calibration = None
    if expected_effective.get("kv_capacity_tokens") is None:
        capacity = control.get("effective_server", {}).get("kv_capacity_tokens")
        if (
            job["id"] != job["stage"] + "/0000"
            or job["trial"]["repeat"] != 0
            or job["trial"]["seed"] != 17
            or type(capacity) is not int
            or capacity <= 0
        ):
            raise ValueError("first native stage capacity calibration is missing or out of order")
        expected_effective["kv_capacity_tokens"] = capacity
        calibration = {
            "basis": "first-native-observed-calibration",
            "stage": job["stage"],
            "job_id": job["id"],
            "kv_capacity_tokens": capacity,
        }
    required = {
        "kv_cache_memory_bytes",
        "enable_prefix_caching",
        "max_model_len",
        "max_num_seqs",
        "kv_capacity_tokens",
        "engine_settings",
    }
    if (
        set(expected_effective) != required
        or expected_effective["kv_cache_memory_bytes"] != job["kv_cache_bytes"]
        or type(expected_effective["enable_prefix_caching"]) is not bool
        or expected_effective["enable_prefix_caching"] != (job["trial"]["cache"] != "off")
        or any(
            type(expected_effective[key]) is not int or expected_effective[key] <= 0
            for key in (
                "kv_cache_memory_bytes",
                "max_model_len",
                "max_num_seqs",
                "kv_capacity_tokens",
            )
        )
        or not isinstance(expected_effective["engine_settings"], dict)
    ):
        raise ValueError("native expected effective-server contract is incomplete")
    actual = control.get("effective_server", {})
    if any(
        type(actual.get(key)) is not type(value) or actual[key] != value
        for key, value in expected_effective.items()
    ):
        raise ValueError(
            "native effective server differs from bound requested configuration/capacity"
        )
    resources = control.get("resources", {})
    if (
        resources.get("errors") != []
        or resources.get("foreign_gpu_processes") != []
        or not isinstance(resources.get("samples"), list)
    ):
        raise ValueError("native resource monitor errors or foreign GPU identities")
    coverage = {
        **outer,
        **qualify_native_boundaries(
            resources["samples"],
            native_duration_seconds=control.get("driver_seconds"),
            outer_duration_seconds=outer["end"] - outer["start"],
        ),
    }
    return {
        "qualified": True,
        "job_id": job["id"],
        "scope": "native serving control and observed outer inventory",
        "coverage": coverage,
        "effective_server": copy.deepcopy(expected_effective),
        "capacity_calibration": calibration,
        "native_foreign_gpu_rejection_preserved": True,
    }


WT_OVERHEAD_SETTINGS = {
    "paths": True,
    "committed_paths": True,
    "escapes": False,
    "start_rank": "sitelinks",
    "unquoted_starts": True,
    "end_after_answer": True,
    "menu": 5,
    "observe": 3,
    "observe_width": 8,
    "observe_terminal": 8,
    "observe_select": True,
    "observe_answer": True,
    "observe_required": True,
    "observe_no_loop": True,
    "request_path_options": True,
}


def qualify_overhead_control(
    report,
    evidence,
    cleanup,
    *,
    expected,
    source_proof=None,
    expected_policy=None,
    expected_intent=None,
):
    """Qualify bare bench replay with outer coverage, never invented native phases."""
    outer = _outer_evidence(
        evidence,
        cleanup,
        source_proof=source_proof,
        expected_policy=expected_policy,
        expected_intent=expected_intent,
    )
    rows = report.get("results", [])
    cells = [(row.get("stage"), row.get("batch"), row.get("mask_control")) for row in rows]
    matrix = {
        (stage, batch, control)
        for stage in ("first", "second")
        for batch in (1, 8)
        for control in ("engine", "noop")
    }
    if (
        report.get("artifact_kind") != "xgraph.engine_bench"
        or type(report.get("version")) is not int
        or report["version"] != 1
        or report.get("profile") != "WT"
        or report.get("engine") != WT_OVERHEAD_SETTINGS
        or len(cells) != 8
        or set(cells) != matrix
        or any(
            type(row.get("errors")) is not int
            or row["errors"] != 0
            or type(row.get("repeats")) is not int
            or row["repeats"] != 3
            for row in rows
        )
    ):
        raise ValueError(
            "overhead report/matrix, duplicate cell, errors or fixed replay settings changed"
        )
    if (
        re.fullmatch(r"[0-9a-f]{64}", expected.get("policy_sha256", "")) is None
        or len(expected["sample_ids"]) != 8
        or len(set(expected["sample_ids"])) != 8
        or report.get("sample_ids") != expected["sample_ids"]
        or report.get("policy_sha256") != expected["policy_sha256"]
    ):
        raise ValueError("overhead fixed policy or deterministic sample binding changed")
    provenance = report.get("provenance", {})
    verify_overhead_provenance(provenance, expected)
    coverage = outer
    return {
        "qualified": True,
        "scope": "serverless overhead and observed outer inventory",
        "coverage": coverage,
        "native_phases_required": False,
    }


def complete_controlled_attempt(
    budget,
    attempt,
    job,
    *,
    control,
    archive,
    cleanup,
    source_proof,
    expected_policy,
    expected_intent,
    driver_returncode,
    server_name=None,
    expected_effective=None,
    expected_overhead=None,
    envelope=None,
    dispatch_failure=None,
    post_deadline=None,
    clock=time.monotonic,
):
    """Bind full-archive/control qualification before any qualified budget close."""
    budget.verify()
    active = budget.data["active"]
    if (
        active is None
        or budget.data["attempts"][active] != attempt
        or json_sha256({key: attempt.get(key) for key in job}) != json_sha256(job)
        or budget.binding.get("require_receipts") is not True
        or budget.binding.get("controller_source_sha256") != json_sha256(source_proof)
    ):
        raise ValueError("controlled completion needs the same active attempt and bound controller")
    output = Path(job["output"])
    if output.parents[2] != budget.path.parent:
        raise ValueError("controlled completion canonical execution destination changed")
    receipt_path = budget.path.parent / "receipts" / (job["id"].replace("/", "-") + ".json")
    if (
        receipt_path.exists()
        or receipt_path.is_symlink()
        or receipt_path.with_suffix(".json.sha256").exists()
    ):
        raise FileExistsError("orphan/previous receipt cannot be reused or overwritten")
    qualification, failure, qualified = None, None, False
    evidence = None
    dispatch = None
    guard = None
    try:
        if envelope is not None:
            dispatch = verify_dispatch_envelope(budget, attempt, job, envelope, cleanup)
        elif budget.binding.get("artifact_kind") == NATIVE_BUDGET_KIND:
            raise ValueError("native completion needs its independently retained dispatch envelope")
        if dispatch_failure is not None:
            raise ValueError("native dispatch/control envelope failed")
        if post_deadline is not None:
            guard = control_deadline(post_deadline, clock=clock)
            guard.__enter__()
        if type(driver_returncode) is not int or driver_returncode != 0:
            raise ValueError("owned native driver did not exit successfully")
        if (
            not isinstance(control, dict)
            or control.get("path") != str(output.with_suffix(".control.json"))
            or artifact_identity(control["path"], maximum=128 * 1024**2) != control
            or read_seal_digest(output.with_suffix(".control.json.sha256")) != control["sha256"]
        ):
            raise ValueError("native control receipt path, bytes or seal changed")
        data = Path(control["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != control["sha256"]:
            raise ValueError("native control receipt changed before decoding")
        retained = json.loads(data)
        if (
            retained.get("job_id") != job["id"]
            or retained.get("integrity_errors") != []
            or retained.get("original_error_kind") is not None
            or retained.get("source_sha256") != budget.binding["native_source_sha256"]
            or retained.get("environment_sha256") != budget.binding["native_environment_sha256"]
            or not isinstance(retained.get("report"), dict)
            or retained["report"].get("path") != str(output)
            or artifact_identity(output) != retained["report"]
            or expected_intent.get("job_id") != job["id"]
            or expected_intent.get("output") != str(output)
        ):
            raise ValueError("native report/source/runtime/job binding changed")
        evidence = read_inventory_archive(
            archive,
            directory=budget.path.parent / "coverage" / job["id"].replace("/", "-"),
            source_proof=source_proof,
        )
        interval_start = (
            dispatch["coverage_start"] if dispatch is not None else attempt["started_monotonic"]
        )
        if (
            evidence["interval"]["start"] != interval_start
            or cleanup.get("at") != evidence["interval"]["end"]
            or cleanup.get("owned") != evidence["owned"]
            or not any(item.get("origin") == "worker" for item in evidence["owned"])
        ):
            raise ValueError("outer archive does not bind the full owned attempt/cleanup interval")
        common = {
            "source_proof": source_proof,
            "expected_policy": expected_policy,
            "expected_intent": expected_intent,
        }
        if job["kind"] == "overhead":
            if (
                retained.get("returncode") != 0
                or retained.get("native_phases_required") is not False
                or retained.get("native_headroom_within_charged_stage") is not True
            ):
                raise ValueError("frozen serverless overhead stage/control contract changed")
            if (
                expected_overhead is None
                and budget.binding.get("artifact_kind") == NATIVE_BUDGET_KIND
            ):
                expected_overhead = retained.get("requested_overhead")
                phase_pin = dispatch.get("native_start") if dispatch is not None else None
                if (
                    not isinstance(expected_overhead, dict)
                    or expected_overhead.get("source_sha256")
                    != budget.binding["native_source_sha256"]
                    or not isinstance(phase_pin, dict)
                    or phase_pin.get("path") != str(output.with_suffix(".native-start.json"))
                    or metadata_identity(phase_pin["path"]) != phase_pin
                    or read_seal_digest(Path(phase_pin["path"]).with_suffix(".json.sha256"))
                    != phase_pin["sha256"]
                    or json.loads(Path(phase_pin["path"]).read_bytes()).get("requested_sha256")
                    != json_sha256(expected_overhead)
                ):
                    raise ValueError(
                        "overhead request/provenance recipe was not bound before its native stage"
                    )
            qualification = qualify_overhead_control(
                retained["control"], evidence, cleanup, expected=expected_overhead, **common
            )
        else:
            qualification = qualify_native_control(
                job,
                retained["control"],
                evidence,
                cleanup,
                server_name=server_name,
                expected_effective=expected_effective,
                **common,
            )
        qualified = qualification["qualified"] is True
    except (ValueError, OSError, KeyError, TypeError) as error:
        # Retain only the error class in the controller receipt. Original native
        # report/log bytes remain separately pinned, without copying payloads.
        failure = {"kind": type(error).__name__, "message": "operational qualification failed"}
    finally:
        if guard is not None:
            guard.__exit__(None, None, None)
        if evidence is not None and hasattr(evidence["samples"], "close"):
            evidence["samples"].close()
    receipt_path.parent.mkdir(exist_ok=True)
    if post_deadline is not None and clock() > post_deadline:
        qualified = False
        failure = failure or {
            "kind": "TimeoutError",
            "message": "post-native control allowance exhausted before spending cutoff",
        }
        if qualification is not None:
            qualification = {
                "qualified": False,
                "completed_control_checks": qualification,
                "reason": "post-native allowance exhausted",
            }
    confirmed = cleanup.get("confirmed_absent") is True and cleanup.get("errors") == []
    if budget.binding.get("artifact_kind") == NATIVE_BUDGET_KIND:
        # A storage failure can still settle actual wall time only from the
        # separately retained clock/worker/cleanup proof. No archive substitute.
        confirmed = confirmed and dispatch is not None and dispatch.get("worker") is not None
    settlement = budget.prepare_finish(
        attempt,
        cleanup_confirmed=confirmed,
        qualified=qualified,
        post_deadline=post_deadline,
    )
    if qualified and not settlement["control_qualified"]:
        failure = failure or {
            "kind": "TimeoutError",
            "message": "post-native control allowance exhausted at spending cutoff",
        }
        if qualification is not None:
            qualification = {
                "qualified": False,
                "completed_control_checks": qualification,
                "reason": "post-native allowance exhausted",
            }
    qualified = settlement["control_qualified"]
    receipt = {
        "artifact_kind": "xgraph.serving_recovery_trial_receipt",
        "job_id": job["id"],
        "binding_sha256": json_sha256(budget.binding),
        "control_qualified": qualified,
        "control": control,
        "archive": archive,
        "cleanup": copy.deepcopy(cleanup),
        "settlement": settlement,
        "qualification": qualification,
        "failure": failure,
        "driver_returncode": driver_returncode,
        "dispatch": envelope,
        "created_at": datetime.now(UTC).isoformat(),
    }
    immutable_json(receipt_path, receipt)
    pin = metadata_identity(receipt_path)
    budget.finish(attempt, cleanup_confirmed=confirmed, qualified=qualified, receipt=pin)
    return {
        "receipt": pin,
        "qualified": budget.data["attempts"][active]["qualified"],
        "attempt_status": budget.data["attempts"][active]["status"],
    }


def verify_dispatch_envelope(budget, attempt, job, pin, cleanup):
    path = budget.path.parent / "dispatch" / (job["id"].replace("/", "-") + ".json")
    if (
        not isinstance(pin, dict)
        or pin.get("path") != str(path)
        or metadata_identity(path) != pin
        or read_seal_digest(path.with_suffix(".json.sha256")) != pin["sha256"]
    ):
        raise ValueError("native dispatch envelope canonical identity/seal changed")
    value = json.loads(path.read_bytes())
    if (
        value.get("artifact_kind") != "xgraph.serving_recovery_dispatch"
        or value.get("job_id") != job["id"]
        or value.get("attempt_started_monotonic") != attempt["started_monotonic"]
        or value.get("controller") != budget.controller_identity
        or value.get("source_sha256") != budget.binding["controller_source_sha256"]
        or value.get("control_allowances") != budget.control_allowances()
    ):
        raise ValueError("native dispatch clock/source/controller/allowance binding changed")
    worker = value.get("worker")
    if worker is None:
        if value.get("spawned_monotonic") is not None or value.get("go_monotonic") is not None:
            raise ValueError("unstarted native dispatch has a false worker boundary")
        return value
    process_identity(worker)
    if (
        worker.get("boot_id") != budget.controller_identity["boot_id"]
        or worker.get("pid_namespace") != budget.controller_identity["pid_namespace"]
        or worker.get("pid") != worker.get("pgid")
        or worker.get("pid") != worker.get("sid")
    ):
        raise ValueError("native dispatch worker is not the bound host/session identity")
    start, spawned, go, end = (
        nonnegative(value.get(key), "dispatch " + key)
        for key in ("coverage_start", "spawned_monotonic", "go_monotonic", "coverage_end")
    )
    if not attempt["started_monotonic"] <= start <= spawned <= go <= end:
        raise ValueError("native dispatch omitted the pre-start or cleanup observation boundary")
    if cleanup.get("at") != end or not any(
        item.get("origin") == "worker"
        and all(
            item.get(key) == worker[key] for key in ("pid", "startticks", "pgid", "sid", "boot_id")
        )
        for item in cleanup.get("owned", [])
    ):
        raise ValueError("native dispatch cleanup does not bind its own worker identity")
    return value


def run_qualified_schedule(schedule, budget, execute, *, admit_block=None, admit_job=None):
    """Run the fixed new matrix once; executor must close a qualified receipt.

    This is the orchestration seam, not a second launch interface. The native
    caller must first bind the owner release, global launch claim and lease.
    """
    endpoint_eligibility(schedule, [])
    budget.verify()
    if (
        budget.data["status"] != "ACTIVE"
        or budget.data["attempts"]
        or budget.data["block"] is not None
        or budget.data["active"] is not None
    ):
        raise RuntimeError("recovery schedule is closed or not fresh; no retry/resume")
    if budget.binding.get("schedule_sha256") != json_sha256(schedule):
        raise ValueError("fixed schedule differs from the bound execution")
    stopped = False
    for stage in schedule:
        if stopped:
            break
        blocks = (
            [stage["jobs"][i : i + 4] for i in range(0, len(stage["jobs"]), 4)]
            if stage["name"].startswith("locked-")
            else [[job] for job in stage["jobs"]]
        )
        for block in blocks:
            try:
                if admit_block is not None:
                    admit_block(copy.deepcopy(block))
                budget.reserve_block(block)
            except RuntimeError:
                if budget.data["status"] != "INCOMPLETE":
                    raise
                stopped = True
                break
            for job in block:
                if (
                    job["kind"] != "overhead"
                    and not job["id"].endswith("/0000")
                    and job["stage"] not in budget.data["calibrations"]
                ):
                    raise ValueError("later cell has no authenticated first-native stage capacity")
                if admit_job is not None:
                    admit_job(copy.deepcopy(job))
                attempt = budget.begin(job)
                try:
                    result = execute(copy.deepcopy(job), attempt)
                    budget.verify()
                    retained = budget.data["attempts"][-1]
                    if (
                        retained["id"] != job["id"]
                        or retained.get("receipt") != result.get("receipt")
                        or retained["status"] == "running"
                        or type(result.get("qualified")) is not bool
                        or result["qualified"] != retained.get("qualified")
                    ):
                        raise ValueError(
                            "executor returned without its bound closed attempt receipt"
                        )
                    if not retained["qualified"]:
                        stopped = True
                        break
                    if job["kind"] != "overhead" and job["id"].endswith("/0000"):
                        budget.record_stage_capacity(job)
                except BaseException as error:
                    # Durable running checkpoint remains if best-effort
                    # terminal accounting fails; never proceed.
                    with suppress(BaseException):
                        budget.fail_unclosed(attempt, type(error).__name__)
                    raise
            if stopped:
                break
    completed = [item["id"] for item in budget.data["attempts"] if item["status"] == "completed"]
    result = endpoint_eligibility(schedule, completed)
    if result["overall"] == "COMPLETE":
        budget.data["status"] = "COMPLETE"
        budget.write()
    return result


class BudgetJournal:
    """One serialized spending checkpoint outside the disposable output tree.

    Git objects retain every accepted transition. The JSON ledger is a required
    projection, never a source from which new credit can be reconstructed.
    """

    def __init__(self, claims, path):
        self.claims, self.path = claims, Path(path)
        if self.path.resolve() != self.path:
            raise ValueError("budget journal needs the canonical ledger destination")
        self.reference = f"{CLAIM_NAMESPACE}/budget"

    def read(self):
        try:
            head = self.claims.objects(self.reference)[0]
        except ValueError as error:
            raise ValueError(
                "persistent budget checkpoint missing or invalid; explicit creation required"
            ) from error
        if head["kind"] != "commit":
            raise ValueError("budget checkpoint must be a commit object")
        revision = head["oid"]
        retained = self.claims.objects(f"{revision}:budget.json")[0]
        if retained["kind"] != "blob":
            raise ValueError("budget checkpoint metadata must be a blob object")
        saved = json.loads(retained["data"])
        if (
            saved.get("artifact_kind") != "xgraph.serving_recovery_budget_checkpoint"
            or saved.get("path") != str(self.path)
            or not isinstance(saved.get("state"), dict)
        ):
            raise ValueError("persistent budget journal path/type changed")
        return revision, saved["state"]

    def advance(self, state, previous):
        saved = {
            "artifact_kind": "xgraph.serving_recovery_budget_checkpoint",
            "path": str(self.path),
            "state": state,
        }
        data = (json.dumps(saved, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        git = self.claims.git
        blob = git("hash-object", "-w", "--stdin", data=data).stdout.strip().decode()
        tree = (
            git("mktree", data=f"100644 blob {blob}\tbudget.json\n".encode())
            .stdout.strip()
            .decode()
        )
        parents = [] if previous is None else ["-p", previous]
        revision = (
            git(
                "commit-tree",
                tree,
                *parents,
                data=b"Retain one serving recovery spending transition\n",
            )
            .stdout.strip()
            .decode()
        )
        result = git("update-ref", self.reference, revision, previous or "0" * 40, check=False)
        if result.returncode:
            raise RuntimeError("stale/already-created persistent budget checkpoint; no reset")
        return revision


class RecoveryBudget:
    """Cumulative parent charges and complete-block admission."""

    def __init__(self, path, binding, history, *, clock, claims, controller_identity, create=False):
        self.path = Path(path)
        self.binding = copy.deepcopy(binding)
        self.history = copy.deepcopy(history)
        self.clock = clock
        self.controller_identity = copy.deepcopy(controller_identity)
        process_identity(self.controller_identity)
        if not all(
            isinstance(self.controller_identity.get(key), str) and self.controller_identity[key]
            for key in ("boot_id", "pid_namespace")
        ):
            raise ValueError("budget clock domain needs boot and process namespace identities")
        if type(create) is not bool:
            raise ValueError("budget creation flag must be boolean")
        self.journal = BudgetJournal(claims, self.path)
        self.failed_write = False
        if self.path.is_symlink():
            raise ValueError(f"budget path is a symbolic link: {self.path}")
        if create:
            if self.path.exists():
                raise FileExistsError("budget already exists; no reset or implicit open")
            self.data = {
                "binding": copy.deepcopy(binding),
                "clock_domain": copy.deepcopy(self.controller_identity),
                "parent_charged_gpu_seconds": nonnegative(
                    history["charged_gpu_seconds"], "parent charge"
                ),
                "charged_gpu_seconds": history["charged_gpu_seconds"],
                "status": "ACTIVE",
                "block": None,
                "active": None,
                "attempts": [],
                "calibrations": {},
            }
            self._validate()
            self.revision = self.journal.advance(self.data, None)
            self.persisted = copy.deepcopy(self.data)
            try:
                with self.path.open("x") as stream:
                    stream.write(
                        json.dumps(self.data, sort_keys=True, indent=2, allow_nan=False) + "\n"
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
            except BaseException:
                self.failed_write = True
                raise
        else:
            if not self.path.exists():
                raise ValueError("budget projection missing; no reset/recreation from history")
            self.revision, self.persisted = self.journal.read()
            self.data = json.loads(self.path.read_bytes())
            self.verify()

    def verify(self):
        self._validate()
        self._verify_persisted()

    def _verify_persisted(self):
        if self.failed_write:
            raise RuntimeError(
                "budget write failed; owner reconciliation required, no continuation"
            )
        revision, state = self.journal.read()
        if revision != self.revision or state != self.persisted:
            raise ValueError("stale persistent budget checkpoint; no overwrite or continuation")
        if (
            self.path.is_symlink()
            or not self.path.is_file()
            or json.loads(self.path.read_bytes()) != self.persisted
        ):
            raise ValueError("budget projection differs from persisted journal checkpoint")

    def _validate(self):
        if self.data["binding"] != self.binding:
            raise ValueError("budget binding changed")
        if self.data.get("clock_domain") != self.controller_identity:
            raise ValueError("budget clock domain changed; boot/process reconciliation required")
        self.control_allowances()
        if self.data["parent_charged_gpu_seconds"] != self.history["charged_gpu_seconds"]:
            raise ValueError("budget parent charge changed")
        charged = nonnegative(self.history["charged_gpu_seconds"], "parent charge")
        seen = set()
        for attempt in self.data["attempts"]:
            if attempt["id"] in seen:
                raise ValueError("budget repeated a scheduled attempt")
            seen.add(attempt["id"])
            charged += nonnegative(attempt["charged_gpu_seconds"], attempt["id"])
        if (
            any(attempt["status"] == "failed" for attempt in self.data["attempts"])
            and self.data["status"] != "INCOMPLETE"
        ):
            raise ValueError("failed recovery must remain INCOMPLETE; no reactivation")
        if charged != self.data["charged_gpu_seconds"]:
            raise ValueError("budget cumulative charge changed")
        if self.data["status"] not in ("ACTIVE", "INCOMPLETE", "COMPLETE"):
            raise ValueError("budget status changed")
        nonnegative(self.binding["cap_seconds"], "cumulative cap")
        if (
            "require_receipts" in self.binding
            and type(self.binding["require_receipts"]) is not bool
        ):
            raise ValueError("budget receipt requirement is not boolean")
        attempts, active, block = self.data["attempts"], self.data["active"], self.data["block"]
        calibrations = self.data.get("calibrations")
        if not isinstance(calibrations, dict):
            raise ValueError("authenticated stage capacity calibrations are missing")
        for stage, calibration in calibrations.items():
            if (
                stage not in STAGES[:4]
                or calibration.get("stage") != stage
                or calibration.get("basis") != "first-native-observed-calibration"
                or calibration.get("job_id") != stage + "/0000"
                or type(calibration.get("kv_capacity_tokens")) is not int
                or calibration["kv_capacity_tokens"] <= 0
            ):
                raise ValueError("stage capacity calibration identity is invalid")
            attempt = next((item for item in attempts if item["id"] == calibration["job_id"]), None)
            if (
                attempt is None
                or attempt["status"] != "completed"
                or attempt.get("receipt") != calibration.get("receipt")
            ):
                raise ValueError(
                    "stage capacity calibration lacks its completed first-cell receipt"
                )
            receipt = self.verify_receipt(attempt, calibration["receipt"], qualified=True)
            observed = json.loads(Path(receipt["path"]).read_bytes()).get("qualification", {})
            expected = {
                key: calibration[key] for key in ("basis", "stage", "job_id", "kv_capacity_tokens")
            }
            if (
                observed.get("capacity_calibration") != expected
                or observed.get("effective_server", {}).get("kv_capacity_tokens")
                != calibration["kv_capacity_tokens"]
            ):
                raise ValueError("stage capacity differs from the first native observed receipt")
        running = [i for i, attempt in enumerate(attempts) if attempt["status"] == "running"]
        if active is not None and (type(active) is not int or not 0 <= active < len(attempts)):
            raise ValueError("budget active attempt index changed")
        if running:
            if running != [active] or self.data["status"] != "ACTIVE" or block is None:
                raise ValueError("running/active/block budget state is inconsistent")
        elif active is not None:
            attempt = attempts[active]
            if not (
                self.data["status"] == "INCOMPLETE"
                and attempt["status"] == "failed"
                and attempt.get("cleanup_confirmed") is False
            ):
                raise ValueError("unclosed active attempt state changed")
        for attempt in attempts:
            reserved = nonnegative(attempt["reserved_gpu_seconds"], "attempt reservation")
            if reserved != self.reservation(attempt):
                raise ValueError("original attempt wall reservation changed")
            if "settlement" in attempt:
                self.verify_settlement(attempt, attempt["settlement"])
            if attempt["status"] == "running":
                if attempt["charged_gpu_seconds"] != reserved:
                    raise ValueError("active attempt reservation changed")
            elif attempt["status"] in ("completed", "failed"):
                if type(attempt.get("cleanup_confirmed")) is not bool:
                    raise ValueError("attempt cleanup qualification is not boolean")
                if type(attempt.get("qualified")) is not bool:
                    raise ValueError("attempt qualification is not boolean")
                elapsed = nonnegative(attempt["elapsed_seconds"], "attempt elapsed")
                expected = elapsed if attempt["cleanup_confirmed"] else max(elapsed, reserved)
                if attempt["charged_gpu_seconds"] != expected:
                    raise ValueError("attempt actual/reserved charge changed")
                if (attempt["status"] == "completed") != attempt["qualified"]:
                    raise ValueError("attempt completion qualification changed")
                if attempt["status"] == "completed" and (
                    not attempt["cleanup_confirmed"] or elapsed > reserved
                ):
                    raise ValueError(
                        "completed attempt needs confirmed cleanup within its full bound"
                    )
                settlement = attempt.get("settlement")
                if settlement is None or any(
                    attempt[key] != settlement[key]
                    for key in (
                        "elapsed_seconds",
                        "charged_gpu_seconds",
                        "cleanup_confirmed",
                        "qualified",
                    )
                ):
                    raise ValueError("closed attempt differs from its persisted settlement cutoff")
                if attempt.get("receipt") is not None:
                    self.verify_receipt(attempt, attempt["receipt"], qualified=attempt["qualified"])
                elif attempt["qualified"] and self.binding.get("require_receipts") is True:
                    raise ValueError("completed production attempt is missing its bound receipt")
            else:
                raise ValueError("unknown attempt status in recovery budget")
        if block is not None:
            jobs, next_index = block["jobs"], block["next"]
            if (
                self.data["status"] != "ACTIVE"
                or not jobs
                or len({job["id"] for job in jobs}) != len(jobs)
                or type(next_index) is not int
                or not 0 <= next_index < len(jobs)
            ):
                raise ValueError("admitted block state is inconsistent")
            by_id = {attempt["id"]: attempt for attempt in attempts}
            if any(
                by_id.get(job["id"], {}).get("status") != "completed" for job in jobs[:next_index]
            ):
                raise ValueError("admitted block completion order changed")
            if running and attempts[active]["id"] != jobs[next_index]["id"]:
                raise ValueError("active attempt differs from admitted block order")
            pending = jobs[next_index + bool(running) :]
            required = sum(self.reservation(job) for job in pending)
            if (
                block["reserved_gpu_seconds"] != required
                or charged + required > self.binding["cap_seconds"]
            ):
                raise ValueError("whole-block reservation or cumulative funding changed")

    def write(self):
        self._validate()
        self._verify_persisted()
        before, after = self.persisted, self.data
        if any(
            after["calibrations"].get(key) != value for key, value in before["calibrations"].items()
        ):
            raise ValueError("first native observed capacity cannot be rewritten")
        if (
            any(
                after[key] != before[key]
                for key in ("binding", "clock_domain", "parent_charged_gpu_seconds")
            )
            or (before["status"] != "ACTIVE" and after != before)
            or not len(before["attempts"]) <= len(after["attempts"]) <= len(before["attempts"]) + 1
        ):
            raise ValueError("immutable spending history or terminal transition changed")
        for original, current in zip(before["attempts"], after["attempts"], strict=False):
            if original["status"] != "running":
                if current != original:
                    raise ValueError("settled spending attempt is immutable")
                continue
            mutable = {
                "settlement",
                "status",
                "elapsed_seconds",
                "charged_gpu_seconds",
                "cleanup_confirmed",
                "qualified",
                "receipt",
            }
            if (
                {key: value for key, value in current.items() if key not in mutable}
                != {key: value for key, value in original.items() if key not in mutable}
                or (
                    "settlement" in original and current.get("settlement") != original["settlement"]
                )
                or (current["status"] != "running" and "settlement" not in original)
            ):
                raise ValueError("running spending transition changed its job or persisted cutoff")
        if (
            len(after["attempts"]) > len(before["attempts"])
            and after["attempts"][-1]["status"] != "running"
        ):
            raise ValueError("new spending transition must begin with the full running reservation")
        try:
            self.revision = self.journal.advance(self.data, self.revision)
            self.persisted = copy.deepcopy(self.data)
            write_json(self.path, self.data)
        except BaseException:
            self.failed_write = True
            raise

    def verify_receipt(self, attempt, receipt, *, qualified):
        path = self.path.parent / "receipts" / (attempt["id"].replace("/", "-") + ".json")
        if not isinstance(receipt, dict) or receipt.get("path") != str(path):
            raise ValueError("attempt receipt canonical path/binding changed")
        try:
            if (
                metadata_identity(path) != receipt
                or read_seal_digest(path.with_suffix(".json.sha256")) != receipt["sha256"]
            ):
                raise ValueError("attempt receipt bytes or seal changed")
            retained = json.loads(path.read_bytes())
        except (OSError, KeyError) as error:
            raise ValueError("attempt receipt is unavailable or incomplete") from error
        if (
            retained.get("job_id") != attempt["id"]
            or retained.get("binding_sha256") != json_sha256(self.binding)
            or type(retained.get("control_qualified")) is not bool
            or (qualified and retained["control_qualified"] is not True)
        ):
            raise ValueError("attempt receipt qualification/job/execution binding changed")
        if self.binding.get("require_receipts") is True and (
            attempt.get("settlement") is None or retained.get("settlement") != attempt["settlement"]
        ):
            raise ValueError("attempt receipt lacks the persisted spending settlement/cutoff")
        return receipt

    def record_stage_capacity(self, job):
        """Learn only the first fixed fresh cell's already-qualified capacity."""
        self.verify()
        stage = job["stage"]
        if (
            job["id"] != stage + "/0000"
            or stage in self.data["calibrations"]
            or job["kind"] not in ("primary", "descriptive")
        ):
            raise ValueError("capacity calibration is not the unique first fixed fresh cell")
        attempt = next((item for item in self.data["attempts"] if item["id"] == job["id"]), None)
        if attempt is None or attempt["status"] != "completed":
            raise ValueError("capacity requires the qualified first-cell receipt")
        receipt = self.verify_receipt(attempt, attempt["receipt"], qualified=True)
        observation = (
            json.loads(Path(receipt["path"]).read_bytes())
            .get("qualification", {})
            .get("capacity_calibration")
        )
        if not isinstance(observation, dict):
            raise ValueError("first-cell native capacity observation is unavailable")
        self.data["calibrations"][stage] = {
            **copy.deepcopy(observation),
            "receipt": copy.deepcopy(receipt),
        }
        self.write()

    def fail_unclosed(self, attempt, error_kind):
        """Retain a live controller failure immediately, without parsing outputs."""
        self.verify()
        if self.data["active"] is None:
            self.data["status"] = "INCOMPLETE"
            self.data["block"] = None
            self.write()
            return
        settlement = self.prepare_finish(attempt, cleanup_confirmed=False, qualified=False)
        path = self.path.parent / "receipts" / (attempt["id"].replace("/", "-") + ".json")
        path.parent.mkdir(exist_ok=True)
        immutable_json(
            path,
            {
                "artifact_kind": "xgraph.serving_recovery_trial_receipt",
                "job_id": attempt["id"],
                "binding_sha256": json_sha256(self.binding),
                "control_qualified": False,
                "settlement": settlement,
                "failure": {
                    "kind": error_kind,
                    "message": "live controller failure; cleanup unconfirmed",
                },
                "control": None,
                "archive": None,
                "cleanup": {"confirmed_absent": False},
            },
        )
        self.finish(
            attempt, cleanup_confirmed=False, qualified=False, receipt=metadata_identity(path)
        )

    def reserve_block(self, jobs):
        self.verify()
        if self.data["status"] != "ACTIVE":
            raise RuntimeError(f"budget {self.data['status']}; no retry or reset")
        if self.data["active"] is not None:
            raise RuntimeError("unclosed active attempt; no relaunch")
        if self.data["block"] is not None:
            raise RuntimeError("a whole block is already admitted")
        if not jobs or len({job["id"] for job in jobs}) != len(jobs):
            raise ValueError("admission needs a nonempty unique ordered block")
        required = sum(self.reservation(job) for job in jobs)
        remaining = self.binding["cap_seconds"] - self.data["charged_gpu_seconds"]
        if remaining < required:
            self.data["status"] = "INCOMPLETE"
            self.write()
            raise RuntimeError(
                f"unfunded whole block: need {required:g} seconds; have {remaining:g}"
            )
        self.data["block"] = {
            "jobs": copy.deepcopy(jobs),
            "next": 0,
            "reserved_gpu_seconds": required,
        }
        self.write()
        return required

    def begin(self, job):
        self.verify()
        if self.data["status"] != "ACTIVE":
            raise RuntimeError(f"budget {self.data['status']}; no retry or reset")
        if self.data["active"] is not None:
            raise RuntimeError("unclosed active attempt; no relaunch")
        block = self.data["block"]
        if block is None or job != block["jobs"][block["next"]]:
            raise ValueError("scheduled cell differs from the admitted block order")
        if any(attempt["id"] == job["id"] for attempt in self.data["attempts"]):
            raise ValueError("scheduled attempt already exists; no retry")
        reserved = self.reservation(job)
        attempt = {
            **copy.deepcopy(job),
            "status": "running",
            "started_monotonic": self.clock(),
            "reserved_gpu_seconds": reserved,
            "charged_gpu_seconds": reserved,
        }
        block["reserved_gpu_seconds"] -= reserved
        self.data["active"] = len(self.data["attempts"])
        self.data["attempts"].append(attempt)
        self.data["charged_gpu_seconds"] += reserved
        self.write()
        return copy.deepcopy(attempt)

    def control_allowances(self):
        kind = self.binding.get("artifact_kind")
        if kind is None:
            if "control_allowances" in self.binding:
                raise ValueError("control allowances require the typed native budget binding")
            return {key: 0 for key in CONTROL_ALLOWANCE_FIELDS}
        if kind != NATIVE_BUDGET_KIND:
            raise ValueError("unknown recovery budget binding type")
        selected = self.binding.get("control_allowances")
        if (
            not isinstance(selected, dict)
            or set(selected) != set(CONTROL_ALLOWANCE_FIELDS)
            or self.binding.get("control_envelope_decision_sha256") != CONTROL_ENVELOPE_SHA256
        ):
            raise ValueError("native control envelope allowances are UNBOUND")
        for key, value in selected.items():
            if nonnegative(value, key) <= 0:
                raise ValueError(
                    "native setup/finalization allowances must be positive and qualified"
                )
        if not isinstance(self.binding.get("allowance_qualification"), dict):
            raise ValueError(
                "native control allowances lack their pre-outcome qualification binding"
            )
        return copy.deepcopy(selected)

    def reservation(self, job):
        allowances = self.control_allowances()
        return (
            nonnegative(job["wall_seconds"], job["id"]) + CLEANUP_SECONDS + sum(allowances.values())
        )

    def active_attempt(self, handle):
        self.verify()
        active = self.data["active"]
        if (
            active is None
            or {
                key: value
                for key, value in self.data["attempts"][active].items()
                if key != "settlement"
            }
            != handle
            or handle["status"] != "running"
        ):
            raise RuntimeError("no matching active attempt; cannot charge twice")
        return active, self.data["attempts"][active]

    def verify_settlement(self, attempt, settlement):
        if (
            not isinstance(settlement, dict)
            or settlement.get("clock_domain") != self.controller_identity
            or settlement.get("started_monotonic") != attempt["started_monotonic"]
            or any(
                type(settlement.get(key)) is not bool
                for key in ("cleanup_confirmed", "control_qualified", "qualified")
            )
        ):
            raise ValueError("settlement clock domain or typed cutoff flags changed")
        ended = nonnegative(settlement.get("ended_monotonic"), "settlement cutoff")
        elapsed = nonnegative(ended - attempt["started_monotonic"], "settlement elapsed")
        if "post_deadline" not in settlement:
            raise ValueError("settlement post-native deadline is missing")
        post_deadline = settlement["post_deadline"]
        within_post = post_deadline is None or ended <= nonnegative(
            post_deadline, "settlement post-native deadline"
        )
        if settlement["control_qualified"] and not within_post:
            raise ValueError("settlement qualification exceeded its post-native deadline")
        expected_charge = (
            elapsed
            if settlement["cleanup_confirmed"]
            else max(elapsed, attempt["reserved_gpu_seconds"])
        )
        expected_qualified = (
            settlement["control_qualified"]
            and settlement["cleanup_confirmed"]
            and elapsed <= attempt["reserved_gpu_seconds"]
        )
        if (
            settlement.get("elapsed_seconds") != elapsed
            or settlement.get("charged_gpu_seconds") != expected_charge
            or settlement["qualified"] != expected_qualified
        ):
            raise ValueError("settlement elapsed/reservation/qualification changed")

    def prepare_finish(self, handle, *, cleanup_confirmed, qualified, post_deadline=None):
        """Persist the cost cutoff before writing a receipt or refunding reservation."""
        if type(cleanup_confirmed) is not bool or type(qualified) is not bool:
            raise ValueError("unknown finish flags: cleanup and qualification must be boolean")
        if post_deadline is not None:
            post_deadline = nonnegative(post_deadline, "settlement post-native deadline")
        _active, attempt = self.active_attempt(handle)
        if "settlement" in attempt:
            raise RuntimeError(
                "spending cutoff already persisted; no alternate settlement or retry"
            )
        ended = nonnegative(self.clock(), "settlement cutoff")
        elapsed = nonnegative(ended - attempt["started_monotonic"], "settlement elapsed")
        qualified = qualified and (post_deadline is None or ended <= post_deadline)
        settlement = {
            "clock_domain": copy.deepcopy(self.controller_identity),
            "started_monotonic": attempt["started_monotonic"],
            "ended_monotonic": ended,
            "post_deadline": post_deadline,
            "elapsed_seconds": elapsed,
            "charged_gpu_seconds": elapsed
            if cleanup_confirmed
            else max(elapsed, attempt["reserved_gpu_seconds"]),
            "cleanup_confirmed": cleanup_confirmed,
            "control_qualified": qualified,
            "qualified": qualified
            and cleanup_confirmed
            and elapsed <= attempt["reserved_gpu_seconds"],
        }
        attempt["settlement"] = settlement
        self.write()
        return copy.deepcopy(settlement)

    def finish(self, attempt, *, cleanup_confirmed, qualified, receipt=None):
        if type(cleanup_confirmed) is not bool or type(qualified) is not bool:
            raise ValueError("unknown finish flags: cleanup and qualification must be boolean")
        active, retained = self.active_attempt(attempt)
        if "settlement" not in retained:
            if self.binding.get("require_receipts") is True:
                raise ValueError(
                    "production receipt requires the persisted spending settlement/cutoff"
                )
            self.prepare_finish(attempt, cleanup_confirmed=cleanup_confirmed, qualified=qualified)
            active, retained = self.active_attempt(attempt)
        settlement = retained["settlement"]
        if (
            settlement["cleanup_confirmed"] != cleanup_confirmed
            or settlement["control_qualified"] != qualified
        ):
            raise ValueError("finish flags differ from the persisted spending settlement")
        attempt = retained
        elapsed, charge, qualified = (
            settlement[key] for key in ("elapsed_seconds", "charged_gpu_seconds", "qualified")
        )
        if receipt is not None:
            self.verify_receipt(attempt, receipt, qualified=qualified)
        elif qualified and self.binding.get("require_receipts") is True:
            raise ValueError("production attempt cannot complete before its bound receipt")
        attempt.update(
            status="completed" if qualified else "failed",
            elapsed_seconds=elapsed,
            charged_gpu_seconds=charge,
            cleanup_confirmed=cleanup_confirmed,
            qualified=qualified,
        )
        if receipt is not None:
            attempt["receipt"] = copy.deepcopy(receipt)
        charged = self.data["parent_charged_gpu_seconds"]
        for item in self.data["attempts"]:
            charged += item["charged_gpu_seconds"]
        self.data["charged_gpu_seconds"] = charged
        self.data["active"] = None if cleanup_confirmed else active
        block = self.data["block"]
        if not qualified or charged > self.binding["cap_seconds"]:
            self.data["status"] = "INCOMPLETE"
            self.data["block"] = None
        else:
            block["next"] += 1
            if block["next"] == len(block["jobs"]):
                self.data["block"] = None
        self.write()


if __name__ == "__main__":
    main()
