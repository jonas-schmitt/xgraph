"""Reproducible CPU preparation and opt-in GPU queue for a matched MINTQA rematch.

Dry runs print commands without creating files or touching the GPU. All generated
artifacts live under this checkout (or /tmp); the input checkout is read-only.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from xgraph.mintqa_panels import indexed, pilot_members, read_jsonl, write_json, write_jsonl
from xgraph.observations import MAX_OBSERVE
from xgraph.provenance import provenance, sha256_file

REPO = Path(__file__).resolve().parents[2]
FAMILIES = {
    "qwen38": ("Qwen3.8-27B-FP8", "qwen3.8", 55),
    "qwen25": ("Qwen2.5-7B-Instruct", "qwen2.5", 40),
    "gemma31": ("gemma-4-31B-it-FP8-dynamic", "gemma4", 75),
    # Twice BF16 weight envelopes (10/16/24 GiB) for unified-memory loading,
    # plus 8 GiB KV and 8 GiB runtime headroom; no simultaneous model loads.
    "gemma-e2b": ("gemma-4-E2B-it", "gemma4", 36),
    "gemma-e4b": ("gemma-4-E4B-it", "gemma4", 48),
    "gemma12": ("gemma-4-12B-it", "gemma4", 64),
}
DEFAULT_FAMILIES = ("qwen38", "qwen25", "gemma31")
SHARED_GEMMA = ("gemma-e2b", "gemma-e4b", "gemma12")
ARMS = ("paths-allfix", "paths-allfix-stop", "agent-strong-8", "vocabulary", "x", "cascade")
PUBLISHED_ARMS = ("tog-depth4", "tog-r", "dog", "cor-upstream")
CPU_LIMIT = 7 * 1024**3
_HASH_CACHE = {}


def _hash(path: Path):
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if key not in _HASH_CACHE:
        _HASH_CACHE[key] = sha256_file(path)
    return _HASH_CACHE[key]


def _code_digest():
    paths = sorted((REPO / "src/xgraph").glob("*.py"))
    paths += sorted((REPO / "scripts/mintqa").glob("*"))
    paths += [REPO / "scripts/generate.sh", REPO / "pyproject.toml", REPO / "uv.lock"]
    paths += sorted((REPO / "csrc").glob("**/*"))
    return hashlib.sha256(
        json.dumps([(str(p.relative_to(REPO)), _hash(p)) for p in paths if p.is_file()]).encode()
    ).hexdigest()


@dataclass
class Step:
    command: list[str]
    output: Path
    inputs: list[Path] = field(default_factory=list)
    requests: Path | None = None  # A generation, including a possibly empty routed subset.
    refresh: bool = False  # Only derived score reports may replace an older complete report.


def run_step(step: Step, dry_run=False, cwd=REPO):
    """Run or exactly reuse one stage; an orphan output, sidecar or log is partial."""
    out = step.output
    done = Path(f"{out}.done.json")
    log = Path(f"{out}.log")
    artifacts = [out, Path(f"{out}.summary.json"), Path(f"{out}.provenance.json")]
    signature = None
    if not dry_run or done.exists():
        signature = {
            "command": step.command,
            "code_sha256": _code_digest(),
            "inputs": [{"path": str(p), "sha256": _hash(p)} for p in step.inputs],
        }
    if done.exists():
        saved = json.loads(done.read_text())
        if saved["signature"] == signature:
            for item in saved["outputs"]:
                path = Path(item["path"])
                if not path.is_file() or _hash(path) != item["sha256"]:
                    raise ValueError(f"completed queue artifacts changed: {path}")
            print(f"# reuse {out}", flush=True)
            return "reused"
        if not step.refresh:
            raise ValueError(
                f"queue settings, code or inputs changed: {out}; choose a new directory"
            )
    elif any(p.exists() for p in [*artifacts, log]):
        raise ValueError(f"partial queue stage: {out}; preserve artifacts before retrying")
    print(shlex.join(step.command), flush=True)
    if dry_run:
        return "planned"
    out.parent.mkdir(parents=True, exist_ok=True)
    actual = out.with_name(out.name + ".pending") if step.refresh else out
    if actual != out and actual.exists():
        raise ValueError(f"partial derived report: {actual}")
    command = [str(actual) if arg == str(out) else arg for arg in step.command]
    if step.requests is not None and not read_jsonl(step.requests):
        write_jsonl(out, [])
        write_json(artifacts[1], {"requests": 0, "model_calls": 0})
        write_json(artifacts[2], provenance([step.requests]))
    else:
        with log.open("w" if step.refresh else "x") as stream:
            subprocess.run(command, cwd=cwd, check=True, stdout=stream, stderr=subprocess.STDOUT)
        if actual != out:
            actual.replace(out)
    if not out.is_file():
        raise ValueError(f"stage returned without its output: {out}")
    if step.requests is not None:
        for sidecar in artifacts[1:]:
            if not sidecar.is_file() or not json.loads(sidecar.read_text()):
                raise ValueError(f"generation lacks a complete sidecar: {sidecar}")
        expected = set(indexed(read_jsonl(step.requests)))
        if set(indexed(read_jsonl(out))) != expected:
            raise ValueError(f"generation output ids differ from requests: {out}")
    if out.suffix == ".json":
        metadata = json.loads(out.read_text())
        described = []
        if metadata.get("artifact_kind") == "xgraph.mintqa_panels":
            described = metadata["outputs"]
        elif metadata.get("artifact_kind") == "xgraph.mintqa_index":
            directory = out.with_name(out.name.removesuffix(".manifest.json"))
            described = [
                {"path": str(directory / f["name"]), "sha256": f["sha256"]}
                for f in metadata["files"]
            ]
        for item in described:
            path = Path(item["path"])
            if _hash(path) != item["sha256"]:
                raise ValueError(f"artifact differs from its manifest: {path}")
            artifacts.append(path)
    result = {
        "signature": signature,
        "outputs": [{"path": str(p), "sha256": _hash(p)} for p in artifacts if p.is_file()],
        "provenance": provenance([]),
    }
    if signature["code_sha256"] != _code_digest() or signature["inputs"] != [
        {"path": str(p), "sha256": _hash(p)} for p in step.inputs
    ]:
        raise ValueError(f"queue code or inputs changed during stage: {out}")
    if done.exists() and step.refresh:
        pending = done.with_name(done.name + ".pending")
        write_json(pending, result)
        pending.replace(done)
    else:
        write_json(done, result)
    return "completed"


def check_observe_support(
    launcher: Path, observed: list[int], *, required: bool = False, no_loop: bool = False
):
    if not observed:
        return
    code = "\n".join(
        line for line in launcher.read_text().splitlines() if not line.lstrip().startswith("#")
    )
    if any(marker not in code for marker in ("XG_OBSERVE", "XGRAPH_OBSERVE", "--observe-paths")):
        raise ValueError(
            "observed mode needs a launcher mapping XG_OBSERVE to XGRAPH_OBSERVE "
            "and --observe-paths; dry runs remain available"
        )
    for enabled, variable, flag in (
        (required, "OBSERVE_REQUIRED", "--observe-required"),
        (no_loop, "OBSERVE_NO_LOOP", "--observe-no-loop"),
    ):
        if enabled and any(
            marker not in code for marker in (f"XG_{variable}", f"XGRAPH_{variable}", flag)
        ):
            raise ValueError(f"observed v3 needs a launcher forwarding {variable} and {flag}")


def cpu(module, *args):
    return [
        "env",
        "OPENBLAS_NUM_THREADS=1",
        "OMP_NUM_THREADS=1",
        "TOKENIZERS_PARALLELISM=false",
        "nice",
        "-n",
        "10",
        "prlimit",
        f"--as={CPU_LIMIT}",
        sys.executable,
        "-m",
        module,
        *map(str, args),
    ]


def family_index(args, family, prepared):
    if family in SHARED_GEMMA:
        return args.source_root / "target/mintqa-rematch/indexes/gemma31"
    return prepared / "indexes" / family


def gemma_check_step(args, family, index):
    model = args.models / FAMILIES[family][0]
    reference = args.models / FAMILIES["gemma31"][0]
    output = args.out / family / "gemma-compatibility.json"
    return Step(
        cpu(
            "xgraph.mintqa_scale",
            "qualify",
            family,
            "--models",
            args.models,
            "--index",
            index,
            "--out",
            output,
        ),
        output,
        inputs=list(
            dict.fromkeys(
                [
                    index.with_suffix(".manifest.json"),
                    *sorted(index.glob("*")),
                    *(
                        p
                        for directory in (model, reference)
                        for p in sorted(directory.glob("*"))
                        if p.suffix in (".json", ".jinja")
                    ),
                ]
            )
        ),
    )


def embedding_preparation(module, *args, device="cpu"):
    if device == "cpu":
        return cpu(module, *args)
    headroom = Path(__file__).resolve().parents[2] / "scripts/gpu-headroom.sh"
    # CUDA reserves a large virtual address range; the CPU address-space cap is
    # unsuitable. Model activations are bounded by embedding batch/bucket limits.
    return [
        "bash",
        "-c",
        f'{shlex.quote(str(headroom))} 8 && exec "$@"',
        "stella-preparation",
        "env",
        "OPENBLAS_NUM_THREADS=1",
        "OMP_NUM_THREADS=1",
        "TOKENIZERS_PARALLELISM=false",
        "nice",
        "-n",
        "10",
        sys.executable,
        "-m",
        module,
        *map(str, args),
    ]


def build_steps(args, families):
    data = args.source_root / "data/external/mintqa-full"
    source = args.source_root / "target/mintqa"
    yield Step(
        cpu(
            "xgraph.mintqa_panels",
            "--source",
            source,
            "--data",
            data,
            "--out",
            args.out / "panels",
            "--development",
            args.development,
            "--validation",
            args.validation,
            "--seed",
            args.seed,
        ),
        args.out / "panels/panels.json",
        inputs=[
            source / "questions.json",
            source / "requests-paths.jsonl",
            data / "questions/ti.jsonl",
            data / "questions/pop.jsonl",
        ],
    )
    for family in families:
        if family in SHARED_GEMMA:
            yield gemma_check_step(args, family, family_index(args, family, args.out))
            continue
        model, token_family, _ = FAMILIES[family]
        tokenizer = args.models / model / "tokenizer.json"
        yield Step(
            cpu(
                "xgraph.mintqa_indexes",
                "--kg",
                data / "kg",
                "--tokenizer",
                tokenizer,
                "--out",
                args.out / "indexes" / family,
                "--reuse",
                data / f"index-{token_family}-ua",
            ),
            args.out / "indexes" / f"{family}.manifest.json",
            inputs=[tokenizer, *sorted((data / "kg").glob("*"))],
        )


def arm_steps(
    args,
    family,
    panel,
    observed,
    *,
    prepared=None,
    audit_only=False,
    audit_local=False,
    scale=False,
):
    prepared = args.out if prepared is None else prepared
    model_name, _, need = FAMILIES[family]
    model = args.models / model_name
    tokenizer = model / "tokenizer.json"
    index = family_index(args, family, prepared)
    data = args.source_root / "data/external/mintqa-full"
    inputs = prepared / "panels" / panel
    directory = args.out / family / panel
    requests = inputs / "requests.jsonl"
    common_inputs = [
        prepared / "panels/panels.json",
        inputs / "questions.json",
        index.with_suffix(".manifest.json"),
        data / "kg.sqlite",
        tokenizer,
    ]
    common_inputs += sorted(model.glob("*.json")) + sorted(model.glob("*.jinja"))
    common_inputs += sorted(model.glob("*.safetensors"))
    if family in SHARED_GEMMA or (scale and family == "gemma31"):
        common_inputs.append(args.out / family / "gemma-compatibility.json")
    common_inputs = list(dict.fromkeys(common_inputs))

    def process(name, command, sources):
        return Step(command, directory / f"{name}.jsonl", inputs=[*common_inputs, *sources])

    def generate(
        name,
        arm="constrained",
        request=requests,
        *,
        support="frontier",
        stop="",
        opened=1,
        observe=0,
        select=False,
        required=False,
        no_loop=False,
        audit=False,
        menus=8,
        menu=0,
        terminal=0,
        tokens=256,
        seed=17,
        smoke_recipe=False,
    ):
        constrained = arm == "constrained"
        environment = {
            "XG_NEED_GIB": need,
            "XG_KG_INDEX": index,
            "XG_TOKENIZER": tokenizer,
            "XG_TOOLS_DB": data / "kg.sqlite",
            "XG_RELATION_LABELS": data / "kg/relation_labels.jsonl",
            "XG_PATHS": int(constrained),
            "XG_COMMITTED_PATHS": int(constrained),
            "XG_ESCAPES": 0,
            "XG_START_RANK": "sitelinks",
            "XG_UNQUOTED_STARTS": 1,
            "XG_END_AFTER_ANSWER": int(constrained),
            "XG_OPEN": opened if constrained else 0,
            "XG_RELATIONS": support,
            "XG_STOP": stop,
            "XG_OBSERVE": observe,
            "XG_EXPRESSIONS": 0,
            "XG_KG_VERSIONS": "",
            "XG_LORA": "",
            "XG_DERIVATION": 0,
            "XG_RELEASE_MASS": "",
            "XG_RELEASE_TOP": 0,
            "XG_ESCAPE_BIAS": "",
            "XG_REPEATS": 1,
            "XG_BOUND_ESCAPES": 1,
            "XG_PAGECACHE_ROOTS": f"{data}:{model}",
            "OPENBLAS_NUM_THREADS": 1,
            "OMP_NUM_THREADS": 1,
        }
        if terminal or os.environ.get("XG_OBSERVE_TERMINAL") not in (None, "", "0"):
            environment["XG_OBSERVE_TERMINAL"] = terminal
        if terminal:
            environment.update(
                XG_MENU=0,
                XG_AUDIT=0,
                XG_OBSERVE_SELECT=0,
                XG_OBSERVE_ANSWER=0,
                XG_OBSERVE_REQUIRED=0,
                XG_OBSERVE_NO_LOOP=0,
                XG_OBSERVE_WIDTH=0,
                XG_OBSERVE_DEFER=0,
                XG_MENU_CLOSE_NEXT=0,
            )
        if audit or os.environ.get("XG_AUDIT") == "1":
            environment.update(
                XG_AUDIT=int(audit),
                XG_AUDIT_ENTITIES=20,
                XG_AUDIT_RELATIONS=menus,
                XG_OBSERVE_SELECT=int(select),
                XG_OBSERVE_ANSWER=int(select),
                XG_OBSERVE_REQUIRED=int(required),
                XG_OBSERVE_NO_LOOP=int(no_loop),
            )
        if args.menu or os.environ.get("XG_MENU") not in (None, "", "0"):
            environment.update(
                XG_MENU=menu,
                XG_AUDIT=int(audit),
                XG_OBSERVE_SELECT=int(select),
                XG_OBSERVE_ANSWER=int(select),
                XG_OBSERVE_REQUIRED=int(required),
                XG_OBSERVE_NO_LOOP=int(no_loop),
            )
        if (
            args.observe_select
            or args.observe_required
            or args.observe_no_loop
            or any(
                os.environ.get(name) == "1" for name in ("XG_OBSERVE_SELECT", "XG_OBSERVE_ANSWER")
            )
        ):
            # Preserve old commands when off; explicitly isolate controls in a v2 sweep.
            environment["XG_OBSERVE_SELECT"] = int(select)
            environment["XG_OBSERVE_ANSWER"] = int(select)
        if (
            args.observe_required
            or args.observe_no_loop
            or any(
                os.environ.get(name) == "1"
                for name in ("XG_OBSERVE_REQUIRED", "XG_OBSERVE_NO_LOOP")
            )
        ):
            environment["XG_OBSERVE_REQUIRED"] = int(required)
            environment["XG_OBSERVE_NO_LOOP"] = int(no_loop)
        output = directory / f"{name}.jsonl"
        command = [
            "env",
            *(f"{k}={v}" for k, v in environment.items()),
            "nice",
            "-n",
            "10",
            str(REPO / "scripts/generate.sh"),
            str(model),
            str(request),
            arm,
            str(output),
            "--max-tokens",
            str(tokens),
        ]
        if not smoke_recipe:
            command += ["--max-model-len", "16384", "--max-num-seqs", "64"]
        command += [
            "--kv-cache-gib",
            "16" if family == "gemma31" else "8",
            "--seed",
            str(seed),
        ]
        if FAMILIES[family][1] == "gemma4":
            command += ["--ban-tokens", "<|channel>"]
        if arm == "agent":
            command += [
                "--max-calls",
                "8",
                "--matched-demos",
                "--matched-paths",
                "--pages",
                "--start-rank",
                "sitelinks",
                "--unquoted-starts",
            ]
        return Step(command, output, inputs=[*common_inputs, request], requests=request)

    def policy(name, command, *, source=None, second=None, support=None, candidates=None, extra=()):
        dest = directory / f"{name}.jsonl"
        options = [command, "--out", dest, "--requests", requests]
        sources = [requests]
        for flag, value in (
            ("--outputs", source),
            ("--second", second),
            ("--support", support),
            ("--candidates", candidates),
        ):
            if value is not None:
                path = value if isinstance(value, Path) else directory / f"{value}.jsonl"
                options += [flag, path]
                sources.append(path)
        options += list(extra)
        return process(name, cpu("xgraph.mintqa_rematch", *options), sources)

    match = [
        "--index",
        index,
        "--tokenizer",
        tokenizer,
        "--start-rank",
        "sitelinks",
        "--unquoted-starts",
    ]
    support_args = ["--index", index, "--tokenizer", tokenizer]

    def repair_candidates(first, name):
        return process(
            name,
            cpu(
                "xgraph.repair",
                "candidates",
                "--outputs",
                first,
                "--questions",
                inputs / "questions.json",
                "--requests",
                requests,
                "--db",
                data / "kg.sqlite",
                *match,
                "--relations",
                data / "kg/relation_labels.jsonl",
                "--edits",
                "4",
                "--out",
                directory / f"{name}.jsonl",
            ),
            [first, requests, data / "kg/relation_labels.jsonl"],
        )

    def audit_stages(first=None, prefix=""):
        if first is None:
            first = (
                (prepared if audit_only and not audit_local else args.out)
                / family
                / panel
                / "paths-allfix.jsonl"
            )
        candidates = first.with_name("revision-candidates.jsonl")
        if args.source is not None:
            prefix = f"{args.source_arm_name}-"
            yield policy(args.source_arm_name, "source", source=args.source)
            first = directory / f"{args.source_arm_name}.jsonl"
            candidates = directory / f"{prefix}revision-candidates.jsonl"
            if {"two-pass", "two-pass-terminal", "audit-witness"} & set(args.audit_arms):
                yield repair_candidates(first, f"{prefix}revision-candidates")
        elif audit_local and {"two-pass", "two-pass-terminal", "audit-witness"} & set(
            args.audit_arms
        ):
            yield repair_candidates(first, "revision-candidates")
        for arm in args.audit_arms:
            terminal = arm in {"audit-terminal", "two-pass-terminal"}
            revision = arm in {"two-pass", "two-pass-terminal"}
            stem = prefix + (
                "revision-terminal"
                if arm == "two-pass-terminal"
                else "revision"
                if revision
                else arm
            )
            witness = arm == "audit-witness"
            yield policy(
                f"{stem}-requests",
                "revision" if revision else "audit-witness" if witness else "audit",
                source=first,
                candidates=candidates if witness or revision else None,
                extra=[
                    *support_args,
                    "--db",
                    data / "kg.sqlite",
                    *([] if revision else ["--menu-size", args.audit_menu_size]),
                    *(["--no-menus"] if arm == "audit-nomenu" else []),
                    *(["--observe-terminal", args.terminal_width] if terminal else []),
                    *(
                        [
                            "--endpoints",
                            args.audit_endpoints,
                            "--alternatives",
                            args.audit_alternatives,
                        ]
                        if witness
                        else []
                    ),
                ],
            )
            audit_request = directory / f"{stem}-requests.jsonl"
            yield generate(
                f"{stem}-raw",
                request=audit_request,
                terminal=args.terminal_width if terminal else 0,
            )
            yield process(
                f"{stem}-support",
                cpu(
                    "xgraph.mintqa_rematch",
                    "support",
                    "--requests",
                    audit_request,
                    "--outputs",
                    directory / f"{stem}-raw.jsonl",
                    *support_args,
                    "--out",
                    directory / f"{stem}-support.jsonl",
                ),
                [directory / f"{stem}-raw.jsonl", audit_request],
            )
            yield policy(
                prefix + arm,
                "revise",
                source=first,
                second=f"{stem}-raw",
                support=f"{stem}-support",
                extra=[
                    "--second-source",
                    f"{stem}-raw",
                    *(["--first-source", prefix.removesuffix("-")] if prefix else []),
                ],
            )

    if audit_only:
        yield from audit_stages()
        return
    if scale:
        yield generate("paths-allfix")
        yield generate("paths-allfix-stop", stop="residual-goal")
        yield generate("agent-strong-8", "agent")
        yield generate(
            "menuobs",
            menu=5,
            observe=3,
            select=True,
            required=True,
            no_loop=True,
            tokens=1024,
            seed=0,
            smoke_recipe=True,
        )
        yield from audit_stages(directory / "menuobs.jsonl", "menuobs-")
        return
    yield generate("paths-allfix")
    yield generate("paths-allfix-stop", stop="residual-goal")
    yield generate("vocabulary-raw", support="vocabulary")
    yield generate("agent-strong-8", "agent")
    if args.menu:
        yield generate(f"paths-menu-{args.menu}", menu=args.menu)
        yield generate(
            f"paths-menu-{args.menu}-observed-3",
            menu=args.menu,
            observe=3,
            select=True,
            required=True,
            no_loop=True,
        )
    for k in observed:
        yield generate(f"paths-observed-{k}", observe=k)
        if args.observe_select:
            yield generate(f"paths-observed-{k}-select", observe=k, select=True)
        if args.observe_required:
            yield generate(f"paths-observed-{k}-required", observe=k, select=True, required=True)
        if args.observe_no_loop:
            yield generate(
                f"paths-observed-{k}-select-noloop", observe=k, select=True, no_loop=True
            )
            if args.observe_required:
                yield generate(
                    f"paths-observed-{k}-required-noloop",
                    observe=k,
                    select=True,
                    required=True,
                    no_loop=True,
                )
        if args.observe_stop:
            yield generate(f"paths-observed-{k}-stop", observe=k, stop="residual-goal")
            if args.observe_select:
                yield generate(
                    f"paths-observed-{k}-select-stop", observe=k, select=True, stop="residual-goal"
                )
    yield policy("requests-free-opened", "open")
    yield generate("free-opened-raw", "baseline", directory / "requests-free-opened.jsonl")
    yield policy("free-opened", "restore", source="free-opened-raw")
    for planner, stem, repair in (
        ("free-opened", "x", True),
        ("vocabulary-raw", "vocabulary", False),
    ):
        src = directory / f"{planner}.jsonl"
        executed = directory / f"{stem}-executed.jsonl"
        options = [
            "execute",
            "--outputs",
            src,
            "--db",
            data / "kg.sqlite",
            "--matched",
            *match,
            "--out",
            executed,
        ]
        if repair:
            options += ["--repair", "--strip-quotes"]
        yield process(f"{stem}-executed", cpu("xgraph.programs", *options), [src])
        delivery = directory / f"{stem}-delivery-requests.jsonl"
        yield process(
            f"{stem}-delivery-requests",
            cpu(
                "xgraph.repair",
                "deliver",
                "--executed",
                executed,
                "--requests",
                requests,
                "--out",
                delivery,
            ),
            [executed, requests],
        )
        yield generate(f"{stem}-delivery", request=delivery, opened=0)
        generated = directory / f"{stem}-delivery.jsonl"
        merged = directory / f"{stem}-merged.jsonl"
        yield process(
            f"{stem}-merged",
            cpu(
                "xgraph.repair",
                "merge",
                "--executed",
                executed,
                "--delivered",
                delivery,
                "--generated",
                generated,
                "--out",
                merged,
            ),
            [executed, delivery, generated],
        )
        yield policy(
            stem, "account", source=f"{stem}-merged", extra=["--source", src, "--source", generated]
        )
    yield policy("paths-support", "support", source="paths-allfix", extra=support_args)
    yield policy("cascade-requests", "route", source="paths-allfix", support="paths-support")
    yield generate("cascade-agent", "agent", directory / "cascade-requests.jsonl")
    yield policy(
        "cascade", "cascade", source="paths-allfix", second="cascade-agent", support="paths-support"
    )
    if args.revision or (args.audit and "audit-witness" in args.audit_arms):
        yield repair_candidates(directory / "paths-allfix.jsonl", "revision-candidates")
    if args.revision:
        yield policy(
            "revision-requests",
            "revision",
            source="paths-allfix",
            candidates="revision-candidates",
            extra=[*support_args, "--db", data / "kg.sqlite"],
        )
        yield generate("revision-raw", request=directory / "revision-requests.jsonl")
        revision_requests_path = directory / "revision-requests.jsonl"
        yield process(
            "revision-support",
            cpu(
                "xgraph.mintqa_rematch",
                "support",
                "--outputs",
                directory / "revision-raw.jsonl",
                "--requests",
                revision_requests_path,
                *support_args,
                "--out",
                directory / "revision-support.jsonl",
            ),
            [directory / "revision-raw.jsonl", revision_requests_path],
        )
        yield policy(
            "two-pass",
            "revise",
            source="paths-allfix",
            second="revision-raw",
            support="revision-support",
        )
    if args.audit:
        yield generate("paths-audit", audit=True)
        yield generate("paths-audit-nomenu", audit=True, menus=0)
        yield from audit_stages()


def score_steps(args, families, panels, observed):
    arms = [
        *ARMS,
        *(["two-pass"] if args.revision else []),
        *(["paths-audit", "paths-audit-nomenu"] if args.audit else []),
        *([f"paths-menu-{args.menu}", f"paths-menu-{args.menu}-observed-3"] if args.menu else []),
        *(f"paths-observed-{k}" for k in observed),
        *(f"paths-observed-{k}-stop" for k in observed if args.observe_stop),
        *(f"paths-observed-{k}-select" for k in observed if args.observe_select),
        *(f"paths-observed-{k}-required" for k in observed if args.observe_required),
        *(f"paths-observed-{k}-select-noloop" for k in observed if args.observe_no_loop),
        *(
            f"paths-observed-{k}-required-noloop"
            for k in observed
            if args.observe_required and args.observe_no_loop
        ),
        *(
            f"paths-observed-{k}-select-stop"
            for k in observed
            if args.observe_select and args.observe_stop
        ),
        *(args.audit_arms if args.audit else []),
    ]
    for family in families:
        out = args.out / family / "report.json"
        command = cpu(
            "xgraph.mintqa_rematch",
            "score",
            "--panels",
            args.out / "panels/panels.json",
            "--family",
            family,
            "--directory",
            args.out / family,
            "--data",
            args.source_root / "data/external/mintqa-full",
            "--out",
            out,
            "--resamples",
            args.resamples,
            "--seed",
            args.seed,
        )
        for panel in panels:
            command += ["--panel", panel]
        for arm in arms:
            command += ["--arm", arm]
        sources = [args.out / "panels/panels.json"]
        sources += [args.out / "panels" / p / "questions.json" for p in panels]
        sources += [args.out / family / p / f"{a}.jsonl" for p in panels for a in arms]
        sources += [
            args.source_root / f"data/external/mintqa-full/questions/{s}.jsonl"
            for s in ("ti", "pop")
        ]
        yield Step(command, out, inputs=sources, refresh=True)


def baseline_budget(args, families):
    """Review forecasts scaled to the frozen panel; never presented as measurements."""
    manifest = args.prepared_root / "panels/panels.json"
    questions = (
        len(json.loads(manifest.read_text())["panels"]["development"]["members"])
        if manifest.is_file()
        else args.development
    )
    questions = args.pilot if args.pilot is not None else questions
    return {
        "questions": questions,
        "gpu_jobs": len(families) * len(args.baseline_arms),
        "workload_hours_low": None,
        "workload_hours_high": None,
        "kind": "unpriced: new batching and retrieval require qualified serving timings",
        "excludes": ["CPU retrieval/indexing", "model startup", "serving qualification"],
    }


def baseline_steps(args, families):
    """New readers only; frozen inputs and cached agent references remain read-only."""
    prepared = args.prepared_root
    panels = args.out / "panels" if args.pilot is not None else prepared / "panels"
    panel = panels / "development"
    data = args.source_root / "data/external/mintqa-full"
    linked = args.out / "inputs/linked.jsonl"
    retrieved = args.out / "inputs/dog.jsonl"
    oracle = args.out / "inputs/dog-gold.jsonl"
    if args.baseline_stage in ("prepare", "all"):
        if args.pilot is not None:
            yield Step(
                cpu(
                    "xgraph.mintqa_panels",
                    "--frozen",
                    prepared / "panels/panels.json",
                    "--pilot",
                    args.pilot,
                    "--pilot-seed",
                    args.pilot_seed,
                    "--out",
                    panels,
                ),
                panels / "panels.json",
                inputs=[
                    prepared / "panels/panels.json",
                    prepared / "panels/development/questions.json",
                    prepared / "panels/development/requests.jsonl",
                ],
            )
        yield Step(
            cpu(
                "xgraph.cor",
                "requests",
                "--requests",
                panel / "requests.jsonl",
                "--linked-requests",
                args.linked_requests,
                "--out",
                linked,
            ),
            linked,
            inputs=[panel / "requests.jsonl", args.linked_requests],
        )
        if any(a.startswith("dog") for a in args.baseline_arms):
            yield Step(
                embedding_preparation(
                    "xgraph.dog",
                    "requests",
                    "--requests",
                    panel / "requests.jsonl",
                    "--linked-requests",
                    linked,
                    "--db",
                    data / "kg.sqlite",
                    "--top-k",
                    "120",
                    "--hops",
                    "4",
                    "--candidate-limit",
                    args.dog_candidate_limit,
                    "--candidate-overflow",
                    args.dog_candidate_overflow,
                    "--retriever",
                    args.dog_retriever,
                    *(
                        ["--bm25-prefilter", args.dog_bm25_prefilter]
                        if args.dog_retriever == "bm25-stella"
                        else []
                    ),
                    *(
                        [
                            "--embedding-model",
                            args.dog_embedding_model,
                            "--embedding-device",
                            args.dog_embedding_device,
                            "--embedding-dtype",
                            args.dog_embedding_dtype,
                            "--embedding-batch-size",
                            args.dog_embedding_batch_size,
                            *(
                                ["--embedding-bucket-size", args.dog_embedding_bucket_size]
                                if args.dog_embedding_bucket_size is not None
                                else []
                            ),
                        ]
                        if args.dog_retriever != "bm25"
                        else []
                    ),
                    "--out",
                    retrieved,
                    device=args.dog_embedding_device if args.dog_retriever != "bm25" else "cpu",
                ),
                retrieved,
                inputs=[
                    panel / "requests.jsonl",
                    linked,
                    data / "kg.sqlite",
                    *(
                        sorted(p for p in args.dog_embedding_model.rglob("*") if p.is_file())
                        if args.dog_embedding_model is not None
                        else []
                    ),
                ],
            )
        if "dog-gold" in args.baseline_arms:
            yield Step(
                cpu(
                    "xgraph.dog",
                    "gold",
                    "--requests",
                    retrieved,
                    "--gold-chains",
                    args.dog_gold_chains,
                    "--db",
                    data / "kg.sqlite",
                    "--out",
                    oracle,
                ),
                oracle,
                inputs=[
                    retrieved,
                    Path(f"{retrieved}.summary.json"),
                    args.dog_gold_chains,
                    data / "kg.sqlite",
                ],
            )
    for family in families:
        model_name, _, need = FAMILIES[family]
        model = args.models / model_name
        directory = args.out / family / "development"
        common = [
            panels / "panels.json",
            *sorted(model.glob("*.json")),
            *sorted(model.glob("*.jinja")),
            *sorted(model.glob("*.safetensors")),
        ]
        if args.baseline_stage in ("run", "all"):
            for name in args.baseline_arms:
                arm = (
                    "tog" if name.startswith("tog") else "dog" if name.startswith("dog") else "cor"
                )
                request = oracle if name == "dog-gold" else retrieved if arm == "dog" else linked
                prompt = (
                    args.tog_prompts
                    if arm == "tog"
                    else args.dog_prompts
                    if arm == "dog"
                    else args.cor_prompts
                )
                output = directory / f"{name}.jsonl"
                environment = {
                    "XG_NEED_GIB": os.environ.get("XG_NEED_GIB", need),
                    "XG_AVAILABLE_WAIT": os.environ.get("XG_AVAILABLE_WAIT", "43200"),
                    "XG_TOOLS_DB": data / "kg.sqlite",
                    "XG_PAGECACHE_ROOTS": f"{data}:{model}",
                    "XG_KG_VERSIONS": "",
                    "XG_LORA": "",
                    "XG_OPEN": 0,
                    "XG_PATHS": 0,
                    "XG_AUDIT": 0,
                    "XG_MENU": 0,
                    "OPENBLAS_NUM_THREADS": 1,
                    "OMP_NUM_THREADS": 1,
                }
                command = [
                    "env",
                    *(f"{k}={v}" for k, v in environment.items()),
                    "nice",
                    "-n",
                    "10",
                    str(REPO / "scripts/generate.sh"),
                    str(model),
                    str(request),
                    "dog-gold" if name == "dog-gold" else arm,
                    str(output),
                    "--max-tokens",
                    "256" if arm == "tog" else "512",
                    "--max-model-len",
                    "16384",
                    "--max-num-seqs",
                    "8",
                    "--kv-cache-gib",
                    "8",
                    "--seed",
                    str(args.seed),
                ]
                if arm == "tog":
                    command += [
                        "--tog-prompts",
                        str(prompt),
                        "--tog-depth",
                        "3" if name == "tog" else "4",
                        "--tog-batch-size",
                        str(args.tog_batch_size),
                        "--tog-variant",
                        "tog-r" if name == "tog-r" else "tog",
                    ]
                elif arm == "dog":
                    command += [
                        "--dog-prompts",
                        str(prompt),
                        "--dog-beam",
                        "2" if name == "dog-beam2" else "1",
                        "--dog-steps",
                        "5",
                        "--dog-batch-size",
                        str(args.dog_batch_size),
                    ]
                else:
                    command += [
                        "--cor-prompts",
                        str(prompt),
                        "--cor-mode",
                        "upstream",
                        "--cor-depth",
                        "4",
                        "--cor-temperature-exploration",
                        "0.01",
                        "--cor-temperature-reasoning",
                        "0.01",
                    ]
                yield Step(
                    command,
                    output,
                    inputs=[
                        *common,
                        request,
                        Path(str(request) + ".summary.json"),
                        prompt,
                        *([data / "kg.sqlite"] if arm != "dog" else []),
                    ],
                    requests=request,
                )
        if args.baseline_stage in ("score", "all"):
            out = args.out / family / "baseline-report.json"
            reference = prepared / family
            sources = [
                panels / "panels.json",
                panel / "questions.json",
                reference / "development/agent-strong-8.jsonl",
                *(directory / f"{a}.jsonl" for a in args.baseline_arms),
                *(data / f"questions/{s}.jsonl" for s in ("ti", "pop")),
            ]
            yield Step(
                cpu(
                    "xgraph.mintqa_rematch",
                    "score",
                    "--panels",
                    panels / "panels.json",
                    "--family",
                    family,
                    "--directory",
                    args.out / family,
                    "--agent-directory",
                    reference,
                    "--data",
                    data,
                    "--panel",
                    "development",
                    "--require-complete",
                    *(arg for arm in args.baseline_arms for arg in ("--arm", arm)),
                    "--resamples",
                    args.resamples,
                    "--seed",
                    args.seed,
                    "--out",
                    out,
                ),
                out,
                inputs=sources,
                refresh=True,
            )
    if args.pilot is not None and args.baseline_stage in ("score", "all"):
        out = args.out / "timings.json"
        sources = [panels / "panels.json"]
        sources += [
            args.out / family / "development" / f"{arm}.jsonl{suffix}"
            for family in families
            for arm in args.baseline_arms
            for suffix in ("", ".summary.json", ".provenance.json")
        ]
        yield Step(
            cpu(
                "xgraph.mintqa_rematch",
                "timings",
                "--panels",
                panels / "panels.json",
                "--directory",
                args.out,
                *(arg for family in families for arg in ("--family", family)),
                *(arg for arm in args.baseline_arms for arg in ("--arm", arm)),
                "--out",
                out,
            ),
            out,
            inputs=sources,
            refresh=True,
        )


def check_baseline_inputs(args, families):
    panel = args.prepared_root / "panels/development"
    manifest = json.loads((args.prepared_root / "panels/panels.json").read_text())
    expected = set(indexed(manifest["panels"]["development"]["members"]))
    if set(indexed(read_jsonl(panel / "requests.jsonl"))) != expected:
        raise ValueError("frozen development requests differ from panel membership")
    if set(indexed(json.loads((panel / "questions.json").read_text()))) != expected:
        raise ValueError("frozen development questions differ from panel membership")
    from xgraph.mintqa_rematch import audit_complete_usage

    for family in families:
        path = args.prepared_root / family / "development/agent-strong-8.jsonl"
        audit_complete_usage(read_jsonl(path), expected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "run", "score", "audit", "baselines"))
    parser.add_argument("family", choices=(*FAMILIES, "all"), nargs="?", default="all")
    parser.add_argument(
        "panel", choices=("development", "validation", "both"), nargs="?", default="development"
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=os.environ.get("XG_MINTQA_DRY") == "1"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(os.environ.get("XG_MINTQA_SOURCE", "/home/jonas/git/xgraph")),
    )
    parser.add_argument("--source", type=Path, help="audit a cached first-call JSONL")
    parser.add_argument("--source-arm-name", help="name the imported first call and its policies")
    parser.add_argument(
        "--models",
        type=Path,
        default=Path(os.environ.get("XG_MINTQA_MODELS", "/home/jonas/models")),
    )
    parser.add_argument(
        "--observe",
        type=int,
        action="append",
        default=[],
        help="add paths-observed-<k> with k frontier entities (repeatable)",
    )
    parser.add_argument(
        "--observe-stop",
        action="store_true",
        help="also add paths-observed-<k>-stop with residual-goal STOP for every --observe",
    )
    parser.add_argument("--no-revision", dest="revision", action="store_false")
    parser.add_argument("--audit-menu-size", type=int, default=8)
    parser.add_argument(
        "--terminal-width",
        type=int,
        default=8,
        help="terminal second-call endpoint display limit (1..64)",
    )
    parser.add_argument(
        "--audit-arms",
        nargs="+",
        help=(
            "audit, audit-nomenu, audit-witness; cached audit also accepts two-pass, "
            "audit-terminal and two-pass-terminal "
            "(commas or spaces)"
        ),
    )
    parser.add_argument(
        "--audit-endpoints", type=int, default=12, help="witness audit endpoint limit"
    )
    parser.add_argument(
        "--audit-alternatives", type=int, default=4, help="witness audit alternative limit"
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        default=os.environ.get("XG_AUDIT") == "1",
        help="add paths-audit and paths-audit-nomenu; existing arms keep their settings",
    )
    parser.add_argument(
        "--menu",
        type=int,
        default=int(os.environ.get("XG_MENU") or "0"),
        help="add paths-menu-k and paths-menu-k-observed-3 (k=1..16)",
    )
    parser.add_argument(
        "--observe-select",
        action="store_true",
        default=os.environ.get("XG_OBSERVE_SELECT") == "1",
        help="also add observed selection arms with guaranteed answer delivery",
    )
    parser.add_argument("--development", type=int, default=1024)
    parser.add_argument(
        "--observe-required",
        action="store_true",
        default=os.environ.get("XG_OBSERVE_REQUIRED") == "1",
        help="add required-selection arms; existing arms keep their settings",
    )
    parser.add_argument(
        "--observe-no-loop",
        action="store_true",
        default=os.environ.get("XG_OBSERVE_NO_LOOP") == "1",
        help="add guarded selection arms, and guarded required arms when requested",
    )
    parser.add_argument("--validation", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--pilot", type=int, help="development pilot size, balanced across hops")
    parser.add_argument("--pilot-seed", type=int, help="pilot subset seed (default 17)")
    parser.add_argument(
        "--baseline-stage", choices=("prepare", "run", "score", "all"), default="all"
    )
    parser.add_argument(
        "--baseline-arms",
        choices=(*PUBLISHED_ARMS, "tog", "dog-beam2", "dog-gold"),
        nargs="+",
        default=list(PUBLISHED_ARMS),
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        help="read-only root holding the frozen panels and agent outputs",
    )
    parser.add_argument("--linked-requests", type=Path)
    parser.add_argument("--tog-prompts", type=Path)
    parser.add_argument("--dog-prompts", type=Path)
    parser.add_argument(
        "--dog-gold-chains", type=Path, help="mapped chains for the oracle-retrieval upper bound"
    )
    parser.add_argument("--cor-prompts", type=Path)
    parser.add_argument("--tog-batch-size", type=int, default=32)
    parser.add_argument("--dog-batch-size", type=int, default=32)
    parser.add_argument(
        "--dog-retriever", choices=("bm25", "stella", "bm25-stella"), default="bm25-stella"
    )
    parser.add_argument("--dog-bm25-prefilter", type=int, default=2000)
    parser.add_argument("--dog-embedding-model", type=Path)
    parser.add_argument("--dog-embedding-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--dog-embedding-dtype", choices=("float32", "bfloat16", "float16"), default="float32"
    )
    parser.add_argument("--dog-embedding-batch-size", type=int, default=32)
    parser.add_argument("--dog-embedding-bucket-size", type=int)
    parser.add_argument("--dog-candidate-overflow", choices=("error", "truncate"), default="error")
    parser.add_argument("--dog-candidate-limit", type=int, default=0)
    args = parser.parse_args(argv)
    if (args.pilot is not None or args.pilot_seed is not None) and args.action != "baselines":
        parser.error("--pilot and --pilot-seed require baselines")
    if args.pilot_seed is not None and args.pilot is None:
        parser.error("--pilot-seed requires --pilot")
    if args.pilot_seed is None:
        args.pilot_seed = 17
    if args.out is None:
        args.out = Path(
            os.environ.get(
                "XG_MINTQA_DIR",
                REPO
                / (
                    "target/mintqa-baselines"
                    if args.action == "baselines"
                    else "target/mintqa-rematch"
                ),
            )
        )
    if not 0 <= args.menu <= 16:
        parser.error("--menu must be in [1, 16], or zero to disable")
    args.out, args.source_root, args.models = (
        p.resolve() for p in (args.out, args.source_root, args.models)
    )
    if not (args.out.is_relative_to(REPO) or args.out.is_relative_to(Path("/tmp"))):
        parser.error("outputs must be in this checkout or /tmp; the input checkout is read-only")
    if args.audit_menu_size < 1:
        parser.error("--audit-menu-size must be positive")
    if not 1 <= args.terminal_width <= 64:
        parser.error("--terminal-width must be in [1, 64]")
    if not 1 <= args.audit_endpoints <= 20 or not 0 <= args.audit_alternatives <= 8:
        parser.error("witness audit needs endpoints in [1, 20] and alternatives in [0, 8]")
    if args.audit_arms is not None:
        args.audit = True
        args.audit_arms = [arm for item in args.audit_arms for arm in item.split(",")]
        allowed = {"audit", "audit-nomenu", "audit-witness"}
        if args.action == "audit":
            allowed.update(("two-pass", "audit-terminal", "two-pass-terminal"))
        if set(args.audit_arms) - allowed:
            parser.error(f"--audit-arms must be chosen from {', '.join(sorted(allowed))}")
        if len(set(args.audit_arms)) != len(args.audit_arms):
            parser.error("--audit-arms must be distinct")
    else:
        args.audit_arms = ["audit", "audit-nomenu"]
    if args.source is not None or args.source_arm_name is not None:
        if args.action != "audit" or args.source is None or not args.source_arm_name:
            parser.error("--source and --source-arm-name must be supplied together for audit")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.source_arm_name) is None:
            parser.error(
                "--source-arm-name must be a filename component beginning with a letter or digit"
            )
        args.source = args.source.resolve()
    observed = args.observe or (
        [int(os.environ["XG_OBSERVE"])] if os.environ.get("XG_OBSERVE", "0") != "0" else []
    )
    if any(not 1 <= k <= MAX_OBSERVE for k in observed) or len(set(observed)) != len(observed):
        parser.error(f"observe limits must be distinct integers from 1 through {MAX_OBSERVE}")
    if args.observe_stop and not observed:
        parser.error("--observe-stop needs --observe or a positive XG_OBSERVE")
    if args.observe_select and not observed:
        parser.error("--observe-select needs --observe or a positive XG_OBSERVE")
    if (args.observe_required or args.observe_no_loop) and not observed:
        parser.error(
            "--observe-required and --observe-no-loop need --observe or a positive XG_OBSERVE"
        )
    families = list(DEFAULT_FAMILIES) if args.family == "all" else [args.family]
    panels = ["development", "validation"] if args.panel == "both" else [args.panel]
    if args.action == "baselines":
        if args.panel != "development":
            parser.error("published baseline selection is restricted to frozen development")
        if (
            len(set(args.baseline_arms)) != len(args.baseline_arms)
            or args.dog_candidate_limit < 0
            or args.tog_batch_size < 1
            or args.dog_batch_size < 1
            or args.dog_bm25_prefilter < 1
        ):
            parser.error(
                "baseline arms must be distinct, candidate limit nonnegative and batch positive"
            )
        if args.dog_retriever == "bm25-stella" and args.dog_embedding_model is None:
            args.dog_embedding_model = args.models / "stella_en_400M_v5"
        if args.dog_retriever != "bm25" and args.dog_embedding_model is None:
            parser.error(f"--dog-retriever {args.dog_retriever} needs --dog-embedding-model")
        if args.dog_retriever == "bm25-stella" and args.dog_embedding_dtype != "float32":
            parser.error("BM25-Stella needs --dog-embedding-dtype float32")
        if args.dog_retriever == "bm25" and args.dog_embedding_model is not None:
            parser.error("--dog-embedding-model needs --dog-retriever stella")
        if args.dog_embedding_model is not None:
            args.dog_embedding_model = args.dog_embedding_model.resolve()
        args.prepared_root = (
            args.prepared_root or args.source_root / "target/mintqa-rematch"
        ).resolve()
        if args.pilot is not None:
            args.out = (args.out / f"pilots/n{args.pilot}-seed{args.pilot_seed}").resolve()
            if not (args.out.is_relative_to(REPO) or args.out.is_relative_to(Path("/tmp"))):
                parser.error("pilot output must be in this checkout or /tmp")
            members = json.loads((args.prepared_root / "panels/panels.json").read_text())["panels"][
                "development"
            ]["members"]
            try:
                selected = pilot_members(members, args.pilot, args.pilot_seed)
            except ValueError as error:
                parser.error(str(error))
            counts = Counter(r["cluster"] for r in selected)
            full_counts = Counter(r["cluster"] for r in members)
            print(
                "# Pilot plan: "
                + json.dumps(
                    {
                        "questions": len(selected),
                        "seed": args.pilot_seed,
                        "by_hop": dict(sorted(Counter(str(r["hops"]) for r in selected).items())),
                        "split_clusters": sum(n < full_counts[c] for c, n in counts.items()),
                        "ids": [r["id"] for r in selected],
                        "out": str(args.out),
                        "comparison": "agent-strong-8 on same ids; paired pilot cluster bootstrap",
                        "projection": "timings.json: serial or batched workload plus startup",
                    }
                ),
                flush=True,
            )
        if args.out.is_relative_to(args.prepared_root) or args.prepared_root.is_relative_to(
            args.out
        ):
            parser.error("baseline output must be separate from the read-only prepared inputs")
        args.linked_requests = (
            args.linked_requests
            or args.source_root / "target/mintqa/requests-iterative-linked.jsonl"
        ).resolve()
        args.tog_prompts = (
            args.tog_prompts or REPO / "data/external/tog-official/ToG/prompt_list.py"
        ).resolve()
        args.dog_prompts = (
            args.dog_prompts or args.source_root / "data/external/dog-2wiki/release/templates.py"
        ).resolve()
        args.dog_gold_chains = (
            args.dog_gold_chains or args.source_root / "target/mintqa/chains.json"
        ).resolve()
        args.cor_prompts = (
            args.cor_prompts
            or args.source_root
            / "data/external/cor-upstream/repo/chain_of_relations/methods/cor/prompt.yml"
        ).resolve()
        if not args.dry_run:
            check_baseline_inputs(args, families)
        print(
            "# GPU-hour estimate (unmeasured): " + json.dumps(baseline_budget(args, families)),
            flush=True,
        )
        if "dog-gold" in args.baseline_arms:
            from xgraph.dog import ORACLE_INTERPRETATION

            print(f"# dog-gold: {ORACLE_INTERPRETATION}", flush=True)
        steps = list(baseline_steps(args, families))
    elif args.action == "audit":
        if args.family == "all" or args.panel != "development":
            parser.error("the cached audit diagnostic needs one family and the development panel")
        prepared = args.source_root / "target/mintqa-rematch"
        if args.out.is_relative_to(prepared) or prepared.is_relative_to(args.out):
            parser.error("cached audit outputs must be separate from the read-only prepared inputs")
        steps = list(
            arm_steps(args, args.family, "development", [], prepared=prepared, audit_only=True)
        )
        agent_directory = prepared / args.family
        arms = args.audit_arms
        prefix = ""
        if args.source is not None:
            prefix = f"{args.source_arm_name}-"
            arms = [args.source_arm_name, *(prefix + arm for arm in arms)]
        output = args.out / args.family / f"{prefix}audit-report.json"
        steps.append(
            Step(
                cpu(
                    "xgraph.mintqa_rematch",
                    "score",
                    "--panels",
                    prepared / "panels/panels.json",
                    "--family",
                    args.family,
                    "--directory",
                    args.out / args.family,
                    "--agent-directory",
                    agent_directory,
                    "--data",
                    args.source_root / "data/external/mintqa-full",
                    "--panel",
                    "development",
                    *(arg for arm in arms for arg in ("--arm", arm)),
                    "--seed",
                    args.seed,
                    "--resamples",
                    args.resamples,
                    "--out",
                    output,
                ),
                output,
                inputs=[
                    prepared / "panels/panels.json",
                    prepared / "panels/development/questions.json",
                    agent_directory / "development/agent-strong-8.jsonl",
                    *(args.out / args.family / f"development/{arm}.jsonl" for arm in arms),
                    *(
                        args.source_root / f"data/external/mintqa-full/questions/{s}.jsonl"
                        for s in ("ti", "pop")
                    ),
                ],
                refresh=True,
            )
        )
    elif args.action == "build":
        steps = list(build_steps(args, families))
    elif args.action == "run":
        if not args.dry_run:
            if args.audit:
                launcher = (REPO / "scripts/generate.sh").read_text()
                if any(s not in launcher for s in ("XG_AUDIT", "XGRAPH_AUDIT", "--audit")):
                    raise ValueError(
                        "audit needs a launcher forwarding XG_AUDIT to XGRAPH_AUDIT and --audit"
                    )
            if args.menu:
                launcher = (REPO / "scripts/generate.sh").read_text()
                if any(s not in launcher for s in ("XG_MENU", "XGRAPH_MENU", "--menu")):
                    raise ValueError(
                        "menu needs a launcher forwarding XG_MENU to XGRAPH_MENU and --menu"
                    )
            check_observe_support(
                REPO / "scripts/generate.sh",
                observed,
                required=args.observe_required,
                no_loop=args.observe_no_loop,
            )
            from xgraph.mintqa_indexes import build_unique_index
            from xgraph.mintqa_panels import build_panels

            if not (args.out / "panels/panels.json").is_file():
                parser.error("build frozen panels and indexes before running the GPU queue")
            build_panels(
                args.source_root / "target/mintqa",
                args.source_root / "data/external/mintqa-full",
                args.out / "panels",
                args.development,
                args.validation,
                args.seed,
            )
            for family in families:
                if family in SHARED_GEMMA:
                    continue  # The read-only qualification stage verifies the shared index.
                model = args.models / FAMILIES[family][0]
                if not (args.out / "indexes" / f"{family}.manifest.json").is_file():
                    parser.error(f"build the {family} unique-alias index first")
                build_unique_index(
                    args.source_root / "data/external/mintqa-full/kg",
                    model / "tokenizer.json",
                    args.out / "indexes" / family,
                )
        steps = [step for f in families for p in panels for step in arm_steps(args, f, p, observed)]
    else:
        steps = list(score_steps(args, families, panels, observed))
    if args.action in ("run", "audit"):
        steps = [
            *(
                gemma_check_step(args, f, family_index(args, f, args.out))
                for f in families
                if f in SHARED_GEMMA
            ),
            *steps,
        ]
    if args.dry_run:
        for step in steps:
            run_step(step, dry_run=True)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for step in steps:
            run_step(step)


if __name__ == "__main__":
    main()
