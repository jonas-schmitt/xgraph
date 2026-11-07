"""Evidence-producing qualification of the frozen native WebQSP GCR recipe."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shlex
import struct
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

PREFIX = ["nice", "-n", "19", "taskset", "-c", "4-7"]
WORKER = "scripts/gcr-qualification-check.py"
REFERENCE_NAME = "webqsp-gcr-qualification-reference"


def runtime_argv(plan):
    """Recheck the reference in the exact serving image, with CUDA hidden."""
    from xgraph.webqsp_test import ROOT

    out = Path(plan["out"])
    driver = (
        "import json,sys,torch,pytest; "
        "open('/out/runtime-versions.json','x').write(json.dumps({'torch':torch.__version__})); "
        "sys.prefix='/reference'; "
        "raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','tests/test_gcr.py',"
        "'-k','hf_group','--basetemp','/qualification-tmp/gcr-runtime',"
        "'--junitxml','/out/runtime.xml']))"
    )
    return [
        *PREFIX,
        "timeout",
        "--signal=TERM",
        "--kill-after=15",
        "300",
        "docker",
        "run",
        "--rm",
        "--name",
        REFERENCE_NAME,
        "--label",
        "xgraph.owner=webqsp-gcr-qualification",
        "--memory=4g",
        "--memory-swap=4g",
        "--cpuset-cpus=4-7",
        "--network=none",
        "--read-only",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--workdir",
        "/native",
        "-v",
        f"{out / 'gcr-code'}:/native:ro",
        "-v",
        f"{Path(plan['producer_root']) / '.venv/gcr-hf-oracle'}:/reference/gcr-hf-oracle:ro",
        "-v",
        f"{out}:/out",
        "-v",
        f"{ROOT / 'target/tmp'}:/qualification-tmp",
        "-e",
        "CUDA_VISIBLE_DEVICES=",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "PYTHONPATH=/native/src:/native",
        "-e",
        "OMP_NUM_THREADS=1",
        "-e",
        "OPENBLAS_NUM_THREADS=1",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "HF_HOME=/qualification-tmp/hf",
        "-e",
        "TMPDIR=/qualification-tmp",
        "--entrypoint",
        "/usr/bin/nice",
        plan["runtime_image"],
        "-n",
        "19",
        "taskset",
        "-c",
        "4-7",
        "python3",
        "-c",
        driver,
    ]


def verify_runtime(plan, path):
    from xgraph.webqsp_test import identity, read_sealed, verify_identity

    out = Path(plan["out"])
    receipt = read_sealed(out / "runtime.json")
    if receipt["plan"] != identity(path) or receipt["status"] != "HF_REFERENCE_PASS":
        raise ValueError("native GCR serving runtime reference is unqualified")
    for item in receipt["evidence"]:
        verify_identity(item)
    check_junit(out / "runtime.xml")
    return identity(out / "runtime.json")


def runtime_check(plan, path):
    """Owner-only CPU container, with no GPU device and no checkpoint load."""
    from xgraph import webqsp_gcr_pricing as pricing
    from xgraph.webqsp_test import identity, write_sealed

    out = Path(plan["out"])
    if (out / "runtime.json").exists():
        verify_runtime(plan, path)
        pricing.status_line(plan, "reference", "skip-existing")
        return
    attempt = out / "runtime-attempt.json"
    if attempt.exists():
        raise ValueError("unreceipted native runtime reference attempt; never retry")
    argv = runtime_argv(plan)
    write_sealed(attempt, {"plan": identity(path), "command": argv, "GPU_seconds": 0})
    pricing.status_line(plan, "reference", "start; CPU-only container")
    with (out / "runtime.log").open("x") as stream:
        result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, timeout=330)
    cleanup = subprocess.run(runtime_cleanup_argv(), capture_output=True, text=True, timeout=15)
    if result.returncode or cleanup.returncode or cleanup.stdout.strip():
        pricing.status_line(plan, "reference", "exit failed or cleanup unknown; no GPU launch")
        raise ValueError("native GCR serving runtime reference failed or cleanup unknown")
    check_junit(out / "runtime.xml")
    versions = json.loads((out / "runtime-versions.json").read_text())
    write_sealed(
        out / "runtime.json",
        {
            "plan": identity(path),
            "status": "HF_REFERENCE_PASS",
            "versions": versions,
            "evidence": [
                identity(out / name)
                for name in (
                    "runtime-attempt.json",
                    "runtime.log",
                    "runtime.xml",
                    "runtime-versions.json",
                )
            ],
        },
    )
    pricing.status_line(plan, "reference", "exit 0; HF reference verified")


def runtime_cleanup_argv():
    return ["docker", "ps", "-aq", "--filter", f"name=^/{REFERENCE_NAME}$"]


def check_provenance(provenance, plan, job):
    """Require the trained planner and the selected family's actual synthesis reader."""
    try:
        matches = (
            provenance["provenance"]["repo_commit"] == plan["producer_commit"]
            and Path(provenance["model"]["path"]).resolve() == Path(job["command"][7]).resolve()
            and Path(provenance["gcr"]["synthesis_model"]["path"]).resolve()
            == Path(job["env"]["XG_GCR_SYNTHESIS_MODEL"]).resolve()
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"native GCR model provenance missing: {error}") from error
    if not matches:
        raise ValueError("native GCR model provenance changed producer, planner or reader")


def reader_witness(plan, job, record, messages):
    """Find receipted prompts and serving format through the sealed resume chain."""

    # This also runs in the frozen producer, before the consumer modules existed.
    def identity(path):
        path = Path(path).resolve()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}

    def verify_identity(item):
        if identity(item["path"]) != item:
            raise ValueError(f"frozen input changed: {item['path']}")

    def read_sealed(path):
        if identity(path)["sha256"] != Path(str(path) + ".sha256").read_text().strip():
            raise ValueError(f"native GCR stage receipt seal changed: {path}")
        return json.loads(Path(path).read_text())

    def stage_id(stage):
        return f"{stage['scope']}-{stage['family']}-{stage['name']}"

    out = Path(plan["out"])
    ledger = json.loads(Path(plan.get("ledger_path", out / "ledger.json")).read_text())
    handoff = Path(job["outputs"][0] + ".gcr-paths.json")
    receipts, evidence = {}, []
    infrastructure = {
        "XG_CONTAINER_NAME",
        "XG_GPU_EVENTS",
        "XG_PAGECACHE_ROOTS",
        "XG_START_ATTEMPTS",
        "XG_REQUIRE_MEMFREE",
        "XG_DROP_PAGECACHE_ALWAYS",
        "XG_HEADROOM_WAIT",
    }
    for phase in ("paths", "synthesis"):
        name = stage_id({**job, "name": job["name"] + "-" + phase})
        attempts = [(i, a) for i, a in enumerate(ledger["attempts"]) if a["stage"] == name]
        successes = [(i, a) for i, a in attempts if a["returncode"] == 0]
        if len(successes) != 1 or successes[0] != attempts[-1]:
            raise ValueError("native GCR reader requires one completed receipted stage")
        number, attempt = successes[0]
        receipt_id = plan.get("inherited_receipts", {}).get(str(number))
        if receipt_id is None:
            receipt_id = identity(out / f"attempt-{number}.json")
        verify_identity(receipt_id)
        receipt = read_sealed(receipt_id["path"])
        actual = receipt["job"]
        command = [*job["command"][7:], "--gcr-stage", phase, "--gcr-handoff", str(handoff)]
        if (
            receipt["attempt"] != attempt
            or stage_id(actual) != name
            or actual["command"][7:] != command
            or {k: v for k, v in actual["env"].items() if k not in infrastructure}
            != {k: v for k, v in job["env"].items() if k not in infrastructure}
        ):
            raise ValueError("native GCR reader stage changed the registered argv or environment")
        for item in receipt["evidence"]:
            verify_identity(item)
        receipts[phase] = receipt
        evidence += [receipt_id, *receipt["evidence"]]
    if identity(handoff) not in receipts["paths"]["evidence"]:
        raise ValueError("native GCR reader has no receipted path handoff")
    saved = json.loads(handoff.read_text())
    rows = saved["states"]
    if len(rows) != 1 or rows[0]["state"]["samples"] != record["samples"]:
        raise ValueError("native GCR reader changed the saved path samples")
    for name in ("gcr.py", "gcr_stages.py", "generate.py"):
        source = str(Path(plan["source"]) / "src/xgraph" / name)
        registered = [i["sha256"] for i in plan["inputs"] if i["path"] == source]
        # Installed container paths and relocated snapshots name the same bytes.
        used = [
            sha for p, sha in saved["binding"]["inputs"].items() if p.endswith("/xgraph/" + name)
        ]
        if len(registered) != 1 or used != registered:
            raise ValueError("native GCR reader changed the registered stage source hash")

    def digest(value):
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

    prompt_hash = digest(rows[0]["messages"])
    if prompt_hash != digest(messages):
        raise ValueError("native GCR reader prompt differs from the registered recipe")
    logs = [i for i in receipts["synthesis"]["evidence"] if i["path"].endswith(".log")]
    if len(logs) != 1:
        raise ValueError("native GCR reader has no unique receipted synthesis log")
    formats = set(
        re.findall(
            r"Detected the chat template content format to be '(string|openai)'",
            Path(logs[0]["path"]).read_text(),
        )
    )
    if len(formats) != 1:
        raise ValueError("native GCR reader has no unambiguous serving content format")
    return {"messages_sha256": prompt_hash, "content_format": formats.pop(), "evidence": evidence}


def qualification_plan(pricing, out):
    """Keep the frozen recipe, with one non-test request per synthesis family."""
    plan = copy.deepcopy(pricing)
    out = Path(out).resolve()
    plan.update(
        artifact_kind="xgraph.webqsp_gcr_qualification",
        out=str(out),
        questions=1,
        ids=pricing["ids"][:1],
        cap_seconds=5400,
        stage_wall_seconds=2400,
        log="/home/jonas/git/xgraph/target/queue/gcr-native-qualification.log",
        limits="CPU evidence plus one validation request per family; no accuracy or price claim. "
        "All attempts charged to the core qualification reserve, never the baseline bucket.",
    )
    plan["jobs"] = json.loads(json.dumps(plan["jobs"]).replace(pricing["out"], str(out)))
    for job in plan["jobs"]:
        family = job["family"]
        job["wall_allowance_seconds"] = 2400
        job["env"].update(
            XG_CONTAINER_NAME=f"gcr-native-qualification-{family}",
            XG_CONTAINER_OWNER="webqsp-gcr-qualification",
        )
    return plan


def check_record(record, paths, encode, synthesis_prompt_tokens, expected_config):
    """Check model evidence without answer gold or a correctness score."""

    def require(condition, reason):
        if not condition:
            raise ValueError(f"native GCR evidence: {reason}")

    def close(left, right):
        return math.isfinite(left) and math.isclose(left, right, abs_tol=2e-4, rel_tol=1e-6)

    def f32(value):
        return struct.unpack("f", struct.pack("f", value))[0]

    try:
        require(record["arm"] == "gcr" and record["status"] == "ok", "incomplete episode")
        require(
            not any(record[k] for k in ("empty_trie", "context_limited", "budget_spent")),
            "no constrained path or exhausted budget",
        )
        require(
            all(record["config"].get(k) == v for k, v in expected_config.items()),
            "changed configuration",
        )
        beam = record["beam_search"]
        expected = {
            "semantics": "hf-group",
            "num_beam_groups": 10,
            "group_size": 1,
            "diversity_penalty": 1,
            "early_stopping": False,
            "length_penalty": 1,
            "num_return_sequences": 10,
            "returned_sequences": 10,
            "scores_include_diversity_penalty": True,
        }
        require(all(beam.get(k) == v for k, v in expected.items()), "changed group beam")
        samples, turns = record["samples"], record["turns"]
        require(len(samples) == 10, "missing returned sequences")
        require(record["paths"] == [s["path"] for s in samples], "path delivery changed")
        require(
            record["hypotheses"] == [s["hypothesis"] for s in samples],
            "hypothesis delivery changed",
        )
        syntheses = [t for t in turns if t["stage"] == "synthesis"]
        require(
            len(syntheses) == 1 and syntheses[0]["prompt_tokens"] == synthesis_prompt_tokens,
            "missing synthesis or changed reader prompt",
        )
        steps = [t for t in turns if t.get("kind") == "beam_step"]
        require(
            len(steps) == beam["scoring_requests"] and len(turns) == len(steps) + 1,
            "unaccounted model calls",
        )
        require(record["calls"] == record["rounds"] == len(turns), "incomplete call usage")
        require(
            record["prompt_tokens"] == sum(t["prompt_tokens"] for t in turns)
            and record["generated_tokens"] == sum(len(t["token_ids"]) for t in turns)
            and all(t["generated_tokens"] == len(t["token_ids"]) for t in turns),
            "incomplete token usage",
        )
        for stage in ("start", "paths", "synthesis"):
            selected = [t for t in turns if t["stage"] == stage]
            require(
                record["stages"][stage]["calls"] == len(selected)
                and all(
                    record["stages"][stage][key] == sum(t[key] for t in selected)
                    for key in ("prompt_tokens", "generated_tokens")
                ),
                "incomplete stage usage",
            )
        waves = {}
        for step in steps:
            depth = step["prompt_tokens"] - beam.get("prompt_tokens", 0)
            wave = waves.setdefault(depth, {False: [], True: []})
            wave[bool(step.get("tie_fallback"))].append(step)
        if all(len(w[False]) == beam["num_beam_groups"] for w in waves.values()):
            # hf-group keeps one live slot per group, in the same order at
            # every depth. A full-row tie query replaces that group's first
            # query; another group's identical prefix is a different call.
            score_groups = [{} for _ in range(beam["num_beam_groups"])]
            for wave in waves.values():
                for group, step in enumerate(wave[False]):
                    prefix = step["prefix_sha256"]
                    ties = [t for t in wave[True] if t["prefix_sha256"] == prefix]
                    if ties:
                        peers = [t for t in wave[False] if t["prefix_sha256"] == prefix]
                        require(len(ties) == len(peers), "ambiguous group tie witnesses")
                        step = ties[next(i for i, peer in enumerate(peers) if peer is step)]
                    score_groups[group][prefix] = step["token_logprobs"]
        else:
            # Deduplicated/chunked witnesses are sufficient only when repeated
            # raw scores agree. Never overwrite a conflicting model call.
            scores = {}
            for step in steps:
                row = scores.setdefault(step["prefix_sha256"], {})
                require(
                    all(k not in row or row[k] == v for k, v in step["token_logprobs"].items()),
                    "ambiguous repeated-prefix score witnesses",
                )
                row.update(step["token_logprobs"])
            score_groups = [scores]

        def sums(sample, path, scores):
            raw, path_score = 0.0, None
            sequence = sample["sequence_token_ids"]
            for at, token in enumerate(sequence):
                prefix = hashlib.sha256(json.dumps(sequence[:at]).encode()).hexdigest()
                row = scores.get(prefix, {})
                if at == len(sequence) - 1 and sample["finish_reason"] == "stop":
                    # HF writes the canonical EOS but retains the selected
                    # candidate's score, witnessed in this group's own row.
                    candidates = [
                        f32(raw + row[str(eos)]) for eos in beam["eos_token_ids"] if str(eos) in row
                    ]
                    matching = [v for v in candidates if close(sample["joint_logprob"], v)]
                    if not matching:
                        return None
                    raw = matching[0]
                elif str(token) not in row:
                    return None
                else:
                    raw = f32(raw + row[str(token)])
                if at + 1 == len(path):
                    path_score = raw
            return raw, path_score

        for sample in samples:
            require(sample["path"] in paths and sample["valid_path"], "foreign or invalid path")
            path = encode(sample["path"])
            sequence = sample["sequence_token_ids"]
            require(
                path == sample["path_token_ids"] and sequence[: len(path)] == path,
                "wrong path tokenization",
            )
            limit = record["config"].get("max_tokens", 512)
            require(0 < len(sequence) <= limit, "changed token bound")
            require(
                (sample["finish_reason"] == "stop" and sequence[-1] in beam["eos_token_ids"])
                or (sample["finish_reason"] == "length" and len(sequence) == limit),
                "invalid sequence termination",
            )
            witnesses = [sums(sample, path, scores) for scores in score_groups]
            require(
                any(
                    witness is not None
                    and close(sample["joint_logprob"], witness[0])
                    and close(sample["path_logprob"], witness[1])
                    for witness in witnesses
                )
                and close(sample["beam_score"], sample["search_logprob"] / len(sequence)),
                "raw-logprob witness or group score changed",
            )
        return {"complete_paths": len(samples), "calls": len(turns), "synthesis_calls": 1}
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError(f"native GCR evidence missing or malformed: {error}") from error


def check_junit(path):
    """A skipped oracle or an empty test selection is not qualification."""
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    if not cases or any(
        c.find(k) is not None for c in cases for k in ("skipped", "failure", "error")
    ):
        raise ValueError("native GCR CPU checks failed, skipped or empty")
    return [c.attrib["name"] for c in cases]


def worker_argv(plan, family, *, record=False):
    from xgraph.webqsp_test import ROOT

    out = Path(plan["out"])
    argv = [
        *PREFIX,
        str(out / "gcr-code/.venv/bin/python"),
        "-I",
        str(ROOT / WORKER),
        "--source",
        str(out / "gcr-code"),
        "--plan",
        str(out / "plan.json"),
        "--family",
        family,
        "--out",
        str(out / f"{family}-{'model-check' if record else 'prompts'}.json"),
    ]
    if record:
        job = next(j for j in plan["jobs"] if j["family"] == family)
        argv += ["--record", job["outputs"][0]]
    return argv


def cpu_environment(out):
    from xgraph.webqsp_test import ROOT

    temporary = ROOT / "target/tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    return {
        **{k: v for k, v in os.environ.items() if not k.startswith(("XG_", "XGRAPH_"))},
        "CUDA_VISIBLE_DEVICES": "",
        "TMPDIR": str(temporary),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(Path(out) / "hf-cache"),
        "UV_CACHE_DIR": str(ROOT / "target/uv-cache"),
    }


def cpu_checks(plan, plan_path):
    """Reuse the pinned port's independent HF reference, never its GPU benchmark verdict."""
    from xgraph.webqsp_test import ROOT, identity, write_sealed

    out = Path(plan["out"])
    source = out / "gcr-code"
    oracle = Path(plan["producer_root"]) / ".venv/gcr-hf-oracle"
    if not (oracle / "transformers").is_dir():
        raise ValueError("missing read-only Transformers 4.44 oracle dependencies")
    (source / ".venv/gcr-hf-oracle").symlink_to(oracle, target_is_directory=True)
    # Pin the executable reference, including native extension modules, as well as versions.
    oracle_files = [
        identity(p) for p in sorted(oracle.rglob("*")) if p.is_file() and p.suffix != ".pyc"
    ]
    write_sealed(out / "oracle-inputs.json", oracle_files)
    commands, checks = [], []
    for name, selection in (
        ("beam", ["tests/test_gcr.py", "-k", "hf_group"]),
        ("trie", ["tests/test_path_trie.py", "-k", "not cuda"]),
    ):
        junit = out / f"{name}.xml"
        argv = [
            *PREFIX,
            str(source / ".venv/bin/python"),
            "-I",
            "-m",
            "pytest",
            "-q",
            *selection,
            "--basetemp",
            str(ROOT / f"target/tmp/gcr-qualification-{name}"),
            "--junitxml",
            str(junit),
        ]
        commands.append(argv)
        with (out / f"{name}.log").open("x") as log:
            subprocess.run(
                argv,
                cwd=source,
                env=cpu_environment(out),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=900,
            )
        checks += check_junit(junit)
    required = (
        "test_hf_group_matches_released_transformers",
        "test_hf_group_diversity_can_promote",
    )
    if any(not any(c.startswith(name) for c in checks) for name in required):
        raise ValueError("native GCR CPU oracle selection missed a required seam")
    for job in plan["jobs"]:
        argv = worker_argv(plan, job["family"])
        commands.append(argv)
        subprocess.run(argv, env=cpu_environment(out), check=True, timeout=180)
    evidence = [
        identity(out / name)
        for name in (
            "beam.xml",
            "beam.log",
            "trie.xml",
            "trie.log",
            "oracle-inputs.json",
            *(f"{j['family']}-prompts.json" for j in plan["jobs"]),
        )
    ]
    report = {
        "artifact_kind": "xgraph.gcr_cpu_qualification",
        "status": "CPU_PASS_GPU_PENDING",
        "producer_commit": plan["producer_commit"],
        "recipe_sha256": plan["recipe_sha256"],
        "commit": plan["commit"],
        "seed": 0,
        "selection_seed": 17,
        "commands": commands,
        "checks": checks,
        "evidence": evidence,
        "GPU_hours_used": 0,
        "plan": identity(plan_path),
        "limits": "Fixed scores and prompt rendering only; no model serving, accuracy or price.",
    }
    write_sealed(out / "cpu.json", report)
    return report


def prepare(pricing_path, out):
    from xgraph import webqsp_test as study

    pricing_path, out = Path(pricing_path).resolve(), Path(out).resolve()
    original = study.read_sealed(pricing_path)
    commit = study.git_commit()
    if (
        original.get("artifact_kind") != "xgraph.webqsp_gcr_pricing"
        or original["scope"] != "validation"
    ):
        raise ValueError("qualification requires the frozen non-test pricing plan")
    if not all(q.startswith("WebQTrn-") for q in original["ids"]):
        raise ValueError("qualification refuses test requests")
    for item in original["inputs"]:
        study.verify_identity(item)
    out.mkdir(parents=True, exist_ok=False)
    plan = qualification_plan(original, out)
    plan.update(commit=commit, pricing_plan=study.identity(pricing_path))
    requests = {}
    with Path(original["jobs"][0]["requests"]).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in plan["ids"]:
                if row["id"] in requests:
                    raise ValueError("duplicate qualification request")
                requests[row["id"]] = row
    if set(requests) != set(plan["ids"]):
        raise ValueError("missing fixed qualification request")
    (out / "requests.jsonl").write_text(
        "".join(json.dumps(requests[q]) + "\n" for q in plan["ids"])
    )
    study.snapshot_code(plan["producer_root"], plan["producer_commit"], out / "gcr-code")
    image = re.search(
        r'^image="([^"\n]+@sha256:[a-f0-9]{64})"$',
        (out / "gcr-code/scripts/generate.sh").read_text(),
        re.MULTILINE,
    )
    if image is None:
        raise ValueError("native producer does not pin its serving image by digest")
    plan["runtime_image"] = image.group(1)
    # Preserve the clean original pricing consumer: no registration or recipe re-pin.
    frozen = out / "pricing-code"
    study.snapshot_code(study.ROOT, original["commit"], frozen)
    launcher = subprocess.check_output(
        ["git", "show", f"{original['commit']}:scripts/generate.sh"], cwd=study.ROOT
    )
    (frozen / "scripts/generate.sh").write_bytes(launcher)
    if study.git_commit(frozen) != original["commit"]:
        raise ValueError("frozen pricing consumer is not clean")
    plan["inputs"] += [
        study.identity(p)
        for p in (
            pricing_path,
            out / "requests.jsonl",
            study.ROOT / WORKER,
            Path(__file__),
            *(
                out / "gcr-code" / p
                for p in (
                    "CMakeLists.txt",
                    "pyproject.toml",
                    "uv.lock",
                )
            ),
            *(
                p
                for directory in ("src", "csrc", "scripts", "tests")
                for p in sorted((out / "gcr-code" / directory).rglob("*"))
                if p.is_file() and "__pycache__" not in p.parts
            ),
        )
    ]
    path = out / "plan.json"
    study.write_sealed(path, plan)
    cpu_checks(plan, path)
    command = [
        *PREFIX,
        str(study.ROOT / ".venv/bin/python"),
        "-I",
        "-m",
        "xgraph.webqsp_gcr_qualification",
        "--plan",
        str(path),
    ]
    shell = "#!/usr/bin/env bash\nset -euo pipefail\n"
    shell += f"cd {shlex.quote(str(study.ROOT))}\n"
    shell += 'mode="${1:---dry-run}"\n'
    shell += 'if (($# > 1)) || [[ "$mode" != --dry-run && "$mode" != --run ]]; then exit 2; fi\n'
    (out / "queue.sh").write_text(shell + f'exec {shlex.join(command)} "$mode"\n')
    pricing_command = [
        *PREFIX,
        str(frozen / ".venv/bin/python"),
        "-I",
        "-m",
        "xgraph.webqsp_gcr_pricing",
        "--plan",
        str(pricing_path),
        "--native-qualification",
        str(out / "qualification.json"),
    ]
    verify = shlex.join([*command, "--verify"])
    (out / "pricing-queue.sh").write_text(
        shell
        + f'if [[ "$mode" == --run ]]; then {verify}; '
        + f"else printf '%s\\n' {shlex.quote(verify)}; fi\n"
        + f'exec {shlex.join(pricing_command)} "$mode"\n'
    )
    for name in ("queue.sh", "pricing-queue.sh"):
        (out / name).chmod(0o755)
    with (out / "dry-run.txt").open("x") as stream:
        import contextlib

        with contextlib.redirect_stdout(stream):
            preview(plan)
    with (out / "pricing-dry-run.txt").open("x") as stream:
        subprocess.run(
            [str(out / "pricing-queue.sh"), "--dry-run"],
            stdout=stream,
            env=cpu_environment(out),
            check=True,
        )
    return plan


def prepare_resume(previous_path, out, *, activate=True, cap_addendum=None):
    """Hold the original lifetime ledger lock throughout continuation preparation."""
    import fcntl

    from xgraph.gcr_owner import ledger_path
    from xgraph.webqsp_test import read_sealed

    previous_path = Path(previous_path).resolve()
    account = ledger_path(read_sealed(previous_path))
    with (account.parent / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _prepare_resume(previous_path, out, activate=activate, cap_addendum=cap_addendum)


def _prepare_resume(previous_path, out, *, activate, cap_addendum):
    """Prospective lifecycle repair; retain the existing ledger and failed evidence."""
    import contextlib
    import shutil

    from xgraph import webqsp_test as study
    from xgraph.gcr_owner import attempt_path, ledger_path, verify_ledger

    previous_path, out = Path(previous_path).resolve(), Path(out).resolve()
    previous = study.read_sealed(previous_path)
    pricing_path = Path(previous["pricing_plan"]["path"])
    study.verify_identity(previous["pricing_plan"])
    original = study.read_sealed(pricing_path)
    commit = study.git_commit(clean=activate)
    account_path = ledger_path(previous)
    inherited = json.loads(account_path.read_text())
    expected_plan = previous.get("resume_origin", {}).get("plan", study.identity(previous_path))
    if (
        inherited["plan"] != expected_plan
        or inherited["active"] is not None
        or not 0 <= inherited["spent_seconds"] < previous["cap_seconds"]
        or not math.isclose(
            inherited["spent_seconds"],
            sum(a["elapsed_seconds"] for a in inherited["attempts"]),
            abs_tol=1e-6,
        )
    ):
        raise ValueError("resume requires reconciled original lifetime accounting")
    cap = previous["cap_seconds"]
    cap_evidence = []
    if cap_addendum:
        cap_addendum = Path(cap_addendum).resolve()
        spec = json.loads(cap_addendum.read_text())
        if (
            spec.get("pre_seal") is not True
            or spec.get("outcome_blind") is not True
            or spec.get("previous_plan") != study.identity(previous_path)
            or spec.get("previous_ledger") != study.identity(account_path)
            or spec.get("previous_cap_seconds") != cap
            or spec.get("core_reserve_seconds") != 14400
            or spec.get("pricing_cap_seconds") != 7200
            or not 0 < spec.get("increment_seconds", 0) <= 1800
            or spec.get("cap_seconds") != cap + spec["increment_seconds"]
            or spec["cap_seconds"] + spec["pricing_cap_seconds"] > spec["core_reserve_seconds"]
        ):
            raise ValueError("qualification cap addendum exceeds or changes the fixed core reserve")
        cap = spec["cap_seconds"]
        cap_evidence = [*spec.get("evidence", [])]
        cap_evidence += [spec[key] for key in ("document", "parent") if key in spec]
        for item in cap_evidence:
            study.verify_identity(item)
    receipts = {}
    if previous.get("process_stages"):
        verify_ledger(previous, previous_path)
        for number in range(len(inherited["attempts"])):
            receipt = attempt_path(previous, number)
            if receipt.exists():
                receipts[str(number)] = study.identity(receipt)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(account_path, out / "inherited-ledger.json")
    shutil.copyfile(previous_path.parent / "requests.jsonl", out / "requests.jsonl")
    plan = qualification_plan(original, out)
    evidence = [study.identity(out / "inherited-ledger.json"), study.identity(previous_path)]
    for receipt in receipts.values():
        evidence += [receipt, *study.read_sealed(receipt["path"])["evidence"]]
    for name in ("gemma31.log", "gemma31-events.jsonl"):
        if (previous_path.parent / name).exists():
            evidence.append(study.identity(previous_path.parent / name))
    plan.update(
        commit=commit,
        preflight=not activate,
        pricing_plan=study.identity(pricing_path),
        process_stages=True,
        memory_recovery=True,
        cap_seconds=cap,
        inherited_receipts=receipts,
        source=str(out / "gcr-code"),
        load_progress_seconds=300,
        phase_wall_seconds={"paths": 1200, "synthesis": 1800},
        ledger_path=str(account_path),
        resume_origin={
            "plan": inherited["plan"],
            "attempts": inherited["attempts"],
            "evidence": evidence,
        },
    )
    if previous.get("process_stages"):
        # Keep handoff/output destinations: their bindings include the request paths.
        plan["jobs"] = copy.deepcopy(previous["jobs"])
    if cap_addendum:
        plan["cap_addendum"] = study.identity(cap_addendum)
        evidence += [plan["cap_addendum"], *cap_evidence]
    elif previous.get("cap_addendum"):
        plan["cap_addendum"] = previous["cap_addendum"]
    # The input bytes stay frozen even where the current consumer has been repaired.
    relocations = {}
    for item in original["inputs"]:
        try:
            study.verify_identity(item)
        except ValueError:
            relative = Path(item["path"]).relative_to(study.ROOT)
            if relative != Path("src/xgraph/webqsp_gcr_pricing.py"):
                raise
            frozen = out / "registered-inputs" / relative
            frozen.parent.mkdir(parents=True)
            frozen.write_bytes(
                subprocess.check_output(
                    ["git", "show", f"{original['commit']}:{relative}"],
                    cwd=study.ROOT,
                )
            )
            saved = study.identity(frozen)
            if any(saved[k] != item[k] for k in ("bytes", "sha256")):
                raise ValueError(
                    "registered pricing source no longer matches its Git snapshot"
                ) from None
            relocations[item["path"]] = saved
    plan["pricing_input_relocations"] = relocations
    source = Path(plan["source"])
    study.snapshot_code(plan["producer_root"], plan["producer_commit"], source)
    patch = study.ROOT / "scripts/gcr-process-handoff.patch"
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=source, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=source, check=True)
    shutil.copyfile(Path(__file__).with_name("gcr_stages.py"), source / "src/xgraph/gcr_stages.py")
    shutil.copyfile(study.ROOT / "scripts/gpu-headroom.sh", source / "scripts/gpu-headroom.sh")
    if receipts:
        for name in ("gcr_stages.py", "gcr.py", "generate.py"):
            if (Path(previous["source"]) / "src/xgraph" / name).read_bytes() != (
                source / "src/xgraph" / name
            ).read_bytes():
                raise ValueError(
                    "completed GCR handoff cannot be reused after a stage source change"
                )
    image = re.search(
        r'^image="([^"\n]+@sha256:[a-f0-9]{64})"$',
        (source / "scripts/generate.sh").read_text(),
        re.MULTILINE,
    )
    if image is None:
        raise ValueError("native producer lost its digest-pinned image")
    plan["runtime_image"] = image.group(1)
    plan["inputs"] = [relocations.get(i["path"], i) for i in original["inputs"]]
    plan["inputs"] += [
        *evidence,
        study.identity(pricing_path),
        study.identity(out / "requests.jsonl"),
        *(study.identity(j["requests"]) for j in plan["jobs"]),
        study.identity(patch),
        study.identity(study.ROOT / WORKER),
        *[
            study.identity(Path(__file__).with_name(name))
            for name in (
                "webqsp_gcr_qualification.py",
                "webqsp_gcr_pricing.py",
                "gcr_owner.py",
                "gcr_stages.py",
                "load_progress.py",
                "container_cleanup.py",
                "webqsp_test.py",
            )
        ],
        *[
            study.identity(p)
            for directory in ("src", "csrc", "scripts", "tests")
            for p in sorted((source / directory).rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts
        ],
        *[
            study.identity(source / name)
            for name in ("CMakeLists.txt", "pyproject.toml", "uv.lock")
        ],
    ]
    path = out / "plan.json"
    study.write_sealed(path, plan)
    cpu_checks(plan, path)
    if previous.get("process_stages"):
        # The same image/reference was already qualified; copy no model outputs.
        prior_runtime = verify_runtime(previous, previous_path)
        if previous["runtime_image"] != plan["runtime_image"]:
            raise ValueError("GCR serving runtime image changed on resume")
        shutil.copyfile(previous_path.parent / "runtime.xml", out / "runtime.xml")
        runtime = study.read_sealed(prior_runtime["path"])
        study.write_sealed(
            out / "runtime.json",
            {
                **runtime,
                "plan": study.identity(path),
                "evidence": [
                    prior_runtime,
                    *runtime["evidence"],
                    study.identity(out / "runtime.xml"),
                ],
            },
        )
    command = [
        *PREFIX,
        str(study.ROOT / ".venv/bin/python"),
        "-I",
        "-m",
        "xgraph.webqsp_gcr_qualification",
        "--plan",
        str(path),
    ]
    shell = f"#!/usr/bin/env bash\nset -euo pipefail\ncd {shlex.quote(str(study.ROOT))}\n"
    shell += 'mode="${1:---dry-run}"\n'
    shell += 'if (($# > 1)) || [[ "$mode" != --dry-run && "$mode" != --run ]]; then exit 2; fi\n'
    (out / "queue.sh").write_text(shell + f'exec {shlex.join(command)} "$mode"\n')
    verify = shlex.join([*command, "--verify"])
    price_command = [
        *PREFIX,
        str(study.ROOT / ".venv/bin/python"),
        "-I",
        "-m",
        "xgraph.webqsp_gcr_pricing",
        "--plan",
        str(pricing_path),
        "--native-qualification",
        str(out / "qualification.json"),
        "--harness-plan",
        str(path),
    ]
    (out / "pricing-queue.sh").write_text(
        shell
        + f'if [[ "$mode" == --run ]]; then {verify}; '
        + f"else printf '%s\\n' {shlex.quote(verify)}; fi\n"
        + f'exec {shlex.join(price_command)} "$mode"\n'
    )
    for name in ("queue.sh", "pricing-queue.sh"):
        (out / name).chmod(0o755)
        with (out / ("dry-run.txt" if name == "queue.sh" else "pricing-dry-run.txt")).open(
            "x"
        ) as stream:
            if name == "queue.sh":
                with contextlib.redirect_stdout(stream):
                    preview(plan)
            else:
                subprocess.run(
                    [str(out / name), "--dry-run"],
                    stdout=stream,
                    env=cpu_environment(out),
                    check=True,
                )
    if activate:
        if cap_addendum:
            inherited.update(cap_seconds=cap, cap_addendum=plan["cap_addendum"])
            study.atomic_json(account_path, inherited)
        for name in ("queue.sh", "pricing-queue.sh"):
            shutil.copyfile(previous_path.parent / name, out / ("original-" + name))
            (previous_path.parent / name).write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n"
                + f'exec bash {shlex.quote(str(out / name))} "$@"\n'
            )
        for name in ("dry-run.txt", "pricing-dry-run.txt"):
            if (previous_path.parent / name).exists():
                shutil.copyfile(previous_path.parent / name, out / ("original-" + name))
            shutil.copyfile(out / name, previous_path.parent / name)
    return plan


def verify_cpu(plan, path):
    from xgraph.webqsp_test import identity, read_sealed, verify_identity

    for item in plan["inputs"]:
        verify_identity(item)
    out = Path(plan["out"])
    cpu = read_sealed(out / "cpu.json")
    if (
        cpu["status"] != "CPU_PASS_GPU_PENDING"
        or cpu["plan"] != identity(path)
        or any(cpu[k] != plan[k] for k in ("commit", "producer_commit", "recipe_sha256"))
    ):
        raise ValueError("native GCR CPU evidence does not match plan")
    for item in [*cpu["evidence"], *read_sealed(out / "oracle-inputs.json")]:
        verify_identity(item)
    for name in ("beam", "trie"):
        check_junit(out / f"{name}.xml")
    return identity(out / "cpu.json")


def preview(plan):
    from xgraph.container_cleanup import cleanup_argvs
    from xgraph.gcr_owner import phase_jobs
    from xgraph.webqsp_gcr_pricing import preview as price_preview

    print(
        f"# Qualification lifetime cap: {plan['cap_seconds']} seconds, including inherited charges"
    )
    print("# One engine-init OOM retry per stage: identical argv, bounded by the unspent allowance")
    print(shlex.join(runtime_argv(plan)))
    print(shlex.join(runtime_cleanup_argv()))
    for job in plan["jobs"]:
        for stage in phase_jobs(plan, job) if plan.get("process_stages") else [job]:
            price_preview({**plan, "jobs": [stage]})
            for argv in cleanup_argvs(stage["env"]["XG_CONTAINER_NAME"]):
                print(shlex.join(argv))
        print(shlex.join(worker_argv(plan, job["family"], record=True)))


def cleanup_argv(job):
    return ["docker", "ps", "-aq", "--filter", f"name=^/{job['env']['XG_CONTAINER_NAME']}$"]


def finalize(path):
    """No record is emitted until all raw model evidence has been rechecked."""
    from xgraph import webqsp_test as study

    path = Path(path)
    plan = study.read_sealed(path)
    if plan.get("process_stages"):
        return finalize_isolated(path, plan)
    out = Path(plan["out"])
    evidence = [study.identity(path), verify_cpu(plan, path)]
    ledger_path = out / "ledger.json"
    if not ledger_path.exists():
        raise ValueError("native GCR qualification is GPU_PENDING")
    evidence.append(verify_runtime(plan, path))
    ledger = json.loads(ledger_path.read_text())
    if (
        ledger["plan"] != study.identity(path)
        or ledger["active"] is not None
        or not 0 < ledger["spent_seconds"] <= plan["cap_seconds"]
        or len(ledger["attempts"]) != len(plan["jobs"])
        or [a["stage"] for a in ledger["attempts"]] != [study.stage_id(j) for j in plan["jobs"]]
        or any(a["returncode"] or a["elapsed_seconds"] <= 0 for a in ledger["attempts"])
        or not math.isclose(
            ledger["spent_seconds"],
            sum(a["elapsed_seconds"] for a in ledger["attempts"]),
            abs_tol=1e-6,
        )
    ):
        raise ValueError("native GCR qualification has incomplete GPU accounting")
    for job in plan["jobs"]:
        receipt_path = out / f"{job['family']}-receipt.json"
        if not receipt_path.exists():
            raise ValueError("native GCR qualification is GPU_PENDING")
        receipt = study.read_sealed(receipt_path)
        if (
            receipt["plan"] != study.identity(path)
            or receipt["job"] != job
            or receipt["returncode"]
        ):
            raise ValueError("native GCR qualification has a failed or changed stage")
        attempt = next(a for a in ledger["attempts"] if a["stage"] == study.stage_id(job))
        if not math.isclose(receipt["elapsed_seconds"], attempt["elapsed_seconds"], abs_tol=1e-6):
            raise ValueError("native GCR stage elapsed disagrees with its ledger")
        for item in receipt["evidence"]:
            study.verify_identity(item)
        # Re-derive the check from the raw output; a saved boolean is not evidence.
        subprocess.run(
            worker_argv(plan, job["family"], record=True),
            env=cpu_environment(out),
            check=True,
            timeout=180,
        )
        evidence += [study.identity(receipt_path), *receipt["evidence"]]
    evidence.append(study.identity(ledger_path))
    record = {
        "artifact_kind": "xgraph.gcr_native_qualification",
        "status": "QUALIFIED",
        "producer_commit": plan["producer_commit"],
        "recipe_sha256": plan["recipe_sha256"],
        "families": dict.fromkeys(study.FAMILIES, True),
        "evidence": evidence,
        "commit": plan["commit"],
        "scope": "validation",
        "ids": plan["ids"],
        "protocol_qualification_gpu_seconds": ledger["spent_seconds"],
        "charge_basis": "conservative launcher wall, all attempts; core qualification reserve",
        "limits": "Native planner transfer and reader plumbing; no published-score reproduction, "
        "accuracy gate, controlled cost or WebQSP price. Pricing is a separate bounded queue.",
    }
    record_path = out / "qualification.json"
    if record_path.exists():
        if study.read_sealed(record_path) != record:
            raise ValueError("native GCR qualification record changed")
    else:
        study.write_sealed(record_path, record)
    return record


def owner_run(path):
    import fcntl

    from xgraph import webqsp_gcr_pricing as pricing
    from xgraph import webqsp_test as study

    path = Path(path)
    plan = study.read_sealed(path)
    if plan.get("preflight"):
        raise ValueError("a CPU preflight cannot authorize an owner run")
    if study.git_commit() != plan["commit"]:
        raise ValueError("native GCR qualification refuses a dirty or changed tree")
    verify_cpu(plan, path)
    for saved in plan["model_inventory"].values():
        if study.model_inventory(saved["path"]) != saved:
            raise ValueError("native GCR model inventory changed")
    if plan.get("process_stages"):
        from xgraph.gcr_owner import ledger_path, phase_jobs, run_phase, verify_ledger

        with (ledger_path(plan).parent / "queue.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            verify_ledger(plan, path)
            runtime_check(plan, path)
            for job in plan["jobs"]:
                for stage in phase_jobs(plan, job):
                    run_phase(plan, path, stage)
            record = finalize(path)
            pricing.status_line(plan, "all", "QUALIFIED")
            return record
    out = Path(plan["out"])
    with (out / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_path = out / "ledger.json"
        if not ledger_path.exists():
            study.atomic_json(
                ledger_path,
                {"plan": study.identity(path), "spent_seconds": 0, "active": None, "attempts": []},
            )
        ledger = json.loads(ledger_path.read_text())
        if ledger["plan"] != study.identity(path) or ledger["active"] is not None:
            raise ValueError("unreconciled native GCR qualification attempt")
        if (
            not 0 <= ledger["spent_seconds"] <= plan["cap_seconds"]
            or [a["stage"] for a in ledger["attempts"]]
            != [study.stage_id(j) for j in plan["jobs"][: len(ledger["attempts"])]]
            or not math.isclose(
                ledger["spent_seconds"],
                sum(a["elapsed_seconds"] for a in ledger["attempts"]),
                abs_tol=1e-6,
            )
        ):
            raise ValueError("native GCR qualification has changed GPU accounting")
        runtime_check(plan, path)
        for job in plan["jobs"]:
            family = job["family"]
            receipt_path = out / f"{family}-receipt.json"
            if receipt_path.exists():
                receipt = study.read_sealed(receipt_path)
                if (
                    receipt["plan"] != study.identity(path)
                    or receipt["job"] != job
                    or receipt["returncode"]
                ):
                    raise ValueError("failed or changed qualification stage; never retry")
                attempts = [a for a in ledger["attempts"] if a["stage"] == study.stage_id(job)]
                if len(attempts) != 1 or not math.isclose(
                    receipt["elapsed_seconds"], attempts[0]["elapsed_seconds"], abs_tol=1e-6
                ):
                    raise ValueError("native GCR stage elapsed disagrees with its ledger")
                for item in receipt["evidence"]:
                    study.verify_identity(item)
                pricing.status_line(plan, family, "skip-existing")
                continue
            if any(a["stage"] == study.stage_id(job) for a in ledger["attempts"]) or any(
                Path(p).exists() for p in job["outputs"]
            ):
                raise ValueError("unreceipted qualification attempt/output; never retry")
            pricing.status_line(plan, family, "waiting-headroom")
            subprocess.run(pricing.headroom_argv(job), check=True)
            for output in job["outputs"]:
                Path(output).parent.mkdir(parents=True, exist_ok=True)
            pricing.status_line(plan, family, "start")
            status, elapsed = study.execute_gpu_stage(
                job,
                ledger_path,
                min(plan["cap_seconds"], ledger["spent_seconds"] + job["wall_allowance_seconds"]),
                out / f"{family}.log",
            )
            ledger = json.loads(ledger_path.read_text())
            try:
                cleanup = subprocess.run(
                    cleanup_argv(job), capture_output=True, text=True, timeout=15
                )
                uncertain = cleanup.returncode != 0 or bool(cleanup.stdout.strip())
            except (subprocess.TimeoutExpired, OSError):
                uncertain = True
            if uncertain:
                ledger["active"] = {
                    "uncertain_cleanup": family,
                    "reserved_seconds": plan["cap_seconds"],
                }
                study.atomic_json(ledger_path, ledger)
                pricing.status_line(plan, family, "exit cleanup unknown; reservation retained")
                raise ValueError("unknown native GCR cleanup; full reservation retained")
            if status:
                pricing.status_line(plan, family, f"exit {status}; no qualification")
                raise ValueError("native GCR model stage failed; no qualification record")
            subprocess.run(
                worker_argv(plan, family, record=True),
                env=cpu_environment(out),
                check=True,
                timeout=180,
            )
            output = Path(job["outputs"][0])
            evidence = [
                study.identity(p)
                for p in (
                    output,
                    Path(str(output) + ".summary.json"),
                    Path(str(output) + ".provenance.json"),
                    out / f"{family}.log",
                    Path(job["env"]["XG_GPU_EVENTS"]),
                    out / f"{family}-model-check.json",
                )
            ]
            study.write_sealed(
                receipt_path,
                {
                    "plan": study.identity(path),
                    "job": job,
                    "returncode": status,
                    "elapsed_seconds": elapsed,
                    "evidence": evidence,
                },
            )
            pricing.status_line(plan, family, "exit 0; evidence verified")
        record = finalize(path)
        pricing.status_line(plan, "all", "QUALIFIED")
        return record


def finalize_isolated(path, plan):
    from xgraph import webqsp_test as study
    from xgraph.gcr_owner import attempt_path, completed, ledger_path, phase_jobs, verify_ledger

    out = Path(plan["out"])
    if not ledger_path(plan).exists():
        raise ValueError("native GCR qualification is GPU_PENDING")
    evidence = [study.identity(path), verify_cpu(plan, path), verify_runtime(plan, path)]
    ledger = verify_ledger(plan, path)
    for job in plan["jobs"]:
        for stage in phase_jobs(plan, job):
            if completed(plan, path, stage, ledger) is None:
                raise ValueError("native GCR qualification is GPU_PENDING")
        subprocess.run(
            worker_argv(plan, job["family"], record=True),
            env=cpu_environment(out),
            check=True,
            timeout=180,
        )
        evidence.append(study.identity(out / f"{job['family']}-model-check.json"))
    inherited = len(plan.get("resume_origin", {}).get("attempts", []))
    for number in range(inherited, len(ledger["attempts"])):
        receipt_path = attempt_path(plan, number)
        evidence += [study.identity(receipt_path), *study.read_sealed(receipt_path)["evidence"]]
    evidence += [
        *plan.get("resume_origin", {}).get("evidence", []),
        study.identity(ledger_path(plan)),
    ]
    record = {
        "artifact_kind": "xgraph.gcr_native_qualification",
        "status": "QUALIFIED",
        "producer_commit": plan["producer_commit"],
        "recipe_sha256": plan["recipe_sha256"],
        "families": dict.fromkeys(study.FAMILIES, True),
        "evidence": evidence,
        "commit": plan["commit"],
        "scope": "validation",
        "ids": plan["ids"],
        "protocol_qualification_gpu_seconds": ledger["spent_seconds"],
        "charge_basis": "lifetime launcher wall including inherited failed attempts",
        "execution_harness": study.identity(path),
        "limits": "Native model plumbing only; no accuracy or price claim. Pricing is separate.",
    }
    record_path = out / "qualification.json"
    if record_path.exists():
        if study.read_sealed(record_path) != record:
            raise ValueError("native GCR qualification record changed")
    else:
        study.write_sealed(record_path, record)
    return record


def main(argv=None):
    import argparse

    from xgraph.webqsp_test import git_commit, read_sealed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--pricing-plan", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--cleanup-events",
        type=Path,
        help="operator-saved Docker events for retrospective cleanup accounting",
    )
    parser.add_argument(
        "--cap-addendum",
        type=Path,
        help="pre-seal qualification cap record within the existing reserve",
    )
    parser.add_argument(
        "--check", action="store_true", help="prepare resume evidence without activation"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument(
        "--reconcile",
        action="store_true",
        help="stop/remove a retained failed container and settle its charge; no model launch",
    )
    args = parser.parse_args(argv)
    if args.prepare:
        if args.resume_from:
            if not args.out:
                parser.error("--prepare --resume-from requires --out")
            prepare_resume(
                args.resume_from, args.out, activate=not args.check, cap_addendum=args.cap_addendum
            )
            return
        if not args.pricing_plan or not args.out:
            parser.error("--prepare requires --pricing-plan and --out")
        prepare(args.pricing_plan, args.out)
    elif not args.plan:
        parser.error("--plan is required")
    elif args.run:
        owner_run(args.plan)
    elif args.reconcile:
        from xgraph.gcr_owner import reconcile

        reconcile(args.plan, lifecycle_events=args.cleanup_events)
    elif args.verify:
        if git_commit() != read_sealed(args.plan)["commit"]:
            raise ValueError("native GCR qualification refuses a dirty or changed tree")
        finalize(args.plan)
    else:
        preview(read_sealed(args.plan))


if __name__ == "__main__":
    main()
