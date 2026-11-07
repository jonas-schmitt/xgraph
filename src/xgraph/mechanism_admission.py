"""Hash-bound owner admission and separately authorized mechanism variations.

Dry runs print every possible launcher argv without probing the GPU or writing.
The exact Gemma anchor is reused; a missing or damaged anchor stops admission.
"""

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

from xgraph import mechanism_repro as repro
from xgraph.mintqa_queue import REPO, _code_digest
from xgraph.provenance import sha256_file

HERE = REPO / "target/mechanism-amendment"
FROZEN = REPO / "target/queue-mechanism-repro"
ADMISSION = REPO / "target/queue-mechanism-admission"
VARIATIONS = REPO / "target/queue-mechanism-variations"
BASE = "d9ca1ed3d214069625263ee6df1c8a9ccaeee597"
RUN_SECONDS = 3 * 2956 // 2 + 600
FAMILIES = ("qwen38", "gemma31")


def launch_command(stage, output, owner_path=None):
    """Keep generation controls; inherit the owner's compiler/cache environment."""
    command = list(stage["command"])
    if "-i" in command:
        command.remove("-i")
    command[command.index(stage["output"])] = str(output)
    for index, value in enumerate(command):
        if value.startswith("PATH="):
            command[index] = "PATH=" + (os.environ["PATH"] if owner_path is None else owner_path)
        if value.startswith("XG_RUN_SECONDS="):
            command[index] = f"XG_RUN_SECONDS={RUN_SECONDS}"
        if value.startswith("XG_START_ATTEMPTS="):
            command[index] = "XG_START_ATTEMPTS=1"
    index = command.index("nice")
    command[index : index + 3] = ["nice", "-n", "19", "taskset", "-c", "8-10"]
    return command


def require_variations(proof, authorization, amendment_hash, plan_hash, receipt_hash, commit):
    """A prepared plan and a successful admission never authorize generation."""
    statuses = {"exact-reproduction", "near-reproduction", "noise-band-reproduction"}
    if (
        proof.get("status") != "admitted"
        or set(proof.get("families", {})) != set(FAMILIES)
        or any(proof["families"][f].get("status") not in statuses for f in FAMILIES)
    ):
        raise ValueError("variations require admission of both families")
    if not authorization or authorization.get("status") != "separately-authorized":
        raise ValueError("variations require a separate owner launch authorization")
    expected = {
        "amendment_sha256": amendment_hash,
        "plan_sha256": plan_hash,
        "admission_sha256": receipt_hash,
        "commit": commit,
    }
    if any(authorization.get(key) != value for key, value in expected.items()):
        raise ValueError("separate launch authorization hash or commit differs")


def _json(path):
    return json.loads(Path(path).read_text())


def preflight():
    """Verify before any queue/log/attempt write; retain the old failed gate."""
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip():
        raise ValueError("refusing dirty worktree")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    parent = subprocess.check_output(["git", "rev-parse", "HEAD^"], cwd=REPO, text=True).strip()
    if parent != BASE:
        raise ValueError("queue requires the clean commit containing the amendment above its base")
    amendment = _json(HERE / "amendment.json")
    plan = _json(VARIATIONS / "manifest.json")
    if _code_digest() != plan["code_sha256"]:
        raise ValueError("source differs from the frozen owner queue")
    for path in (HERE / "amendment.json", VARIATIONS / "manifest.json"):
        if sha256_file(path) != path.with_suffix(".sha256").read_text().strip():
            raise ValueError(f"frozen hash changed: {path}")
        if (
            subprocess.check_output(["git", "show", f"HEAD:{path.relative_to(REPO)}"], cwd=REPO)
            != path.read_bytes()
        ):
            raise ValueError(f"queue record is not committed: {path}")
    if amendment["variations_authorized"] is not False:
        raise ValueError("the amendment cannot authorize variations")
    registration = _json(REPO / amendment["registration"])
    noise = _json(REPO / amendment["noise"])
    if registration["fixed_candidate_secondary_rule"] != amendment["fixed_rule"]:
        raise ValueError("registered fixed rule changed")
    for family in FAMILIES:
        if repro.amended_rules(registration, noise, family) != amendment["rules_by_family"][family]:
            raise ValueError("derived amendment thresholds changed")
    for item in amendment["inputs"]:
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"frozen input changed: {item['path']}")
    for item in plan["weights_stat"]:
        stat = Path(item["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (item["bytes"], item["mtime_ns"]):
            raise ValueError(f"model weight inventory changed: {item['path']}")
    # Never reuse the confounded first factorial or write into the old queue.
    original = _json(FROZEN / "manifest.json")
    for stage in (s for s in original["stages"] if s["cell"] != "s0d0"):
        if any(
            Path(stage["output"] + suffix).exists()
            for suffix in ("", ".log", ".summary.json", ".provenance.json")
        ):
            raise ValueError("unexpected variation artifacts in the original reproduction queue")
    return amendment, {**plan, "commit": head}, head


def admission_proof(amendment, plan, head):
    """Recompute historical-versus-anchor admission, never against the repeat."""
    gold = _json(plan["gold"])
    old_messages = repro.read(plan["historical_effective_requests"])
    review = _json(HERE / "oracle-review.json")
    prompts = _json(HERE / "prompt-identity.json")
    if review["status"] != "passed" or prompts["status"] != "passed":
        raise ValueError("oracle review and prompt identity must pass")
    result = {
        "status": "admitted",
        "families": {},
        "commit": head,
        "amendment_sha256": sha256_file(HERE / "amendment.json"),
        "plan_sha256": sha256_file(VARIATIONS / "manifest.json"),
        "variations_authorized": False,
    }
    ids = plan["selected_ids"]
    for family in FAMILIES:
        stage = next(s for s in plan["stages"] if s["family"] == family and s["cell"] == "s0d0")
        if not repro.check_stage(plan, stage):
            raise ValueError("registered anchor missing; no automatic replacement or noise repeat")
        old = repro.read(plan["historical_outputs"][family])
        current = repro.read(stage["output"])
        messages = repro.effective_messages(plan, stage)
        classified = repro.classify_anchor(
            {q: old[q] for q in ids},
            {q: current[q] for q in ids},
            {q: gold[q] for q in ids},
            {q: old_messages[q]["messages"] for q in ids},
            {q: messages[q] for q in ids},
            amendment["rules_by_family"][family],
        )
        if classified["status"] == "failed-reproduction":
            raise ValueError(f"historical anchor failed admission: {family}")
        result["families"][family] = classified
    expected_pairs = {
        ("historical-anchor", d["id"]) for d in result["families"]["qwen38"]["differences"]
    }
    original = repro.read(
        next(s["output"] for s in plan["stages"] if s["family"] == "qwen38" and s["cell"] == "s0d0")
    )
    repeat = repro.read(REPO / amendment["repeat_output"])
    expected_pairs |= {
        ("anchor-repeat", q) for q in ids if original[q]["text"] != repeat[q]["text"]
    }
    if {(r["pair"], r["id"]) for r in review["pairs"]} != expected_pairs or len(
        review["pairs"]
    ) != len(expected_pairs):
        raise ValueError("oracle review does not cover every historical/repeat text difference")
    for pair in review["pairs"]:
        for stream in pair["replay"]["streams"].values():
            divergence = stream["divergence"]
            if (
                stream["rejections"]
                or stream["mask_differences"]
                or not divergence
                or divergence["forced_payload"]
                or not divergence["both_choices_legal_in_both_oracles"]
            ):
                raise ValueError("oracle rejected a token or forced payload agreement")
    result["reproduction_status"] = (
        "noise-band-reproduction"
        if any(f["status"] == "noise-band-reproduction" for f in result["families"].values())
        else "near-reproduction"
        if any(f["status"] == "near-reproduction" for f in result["families"].values())
        else "exact-reproduction"
    )
    return result


def _status(log, stage, event, code=0):
    stamp = subprocess.check_output(["date", "-Is"], text=True).strip()
    with log.open("a") as stream:
        stream.write(f"mechanism-amendment {stage} {event} exit {code} {stamp}\n")


def preview(phase):
    plan = _json(VARIATIONS / "manifest.json")
    print("UNRUN: clean committed amendment; frozen inputs; no GPU probes in dry-run.")
    print("CPU preflight and both-family historical-anchor admission; exact Gemma anchor reused.")
    for stage in plan["stages"]:
        if phase == "admission" and stage["cell"] != "s0d0":
            continue
        if phase == "variations" and stage["cell"] == "s0d0":
            continue
        if stage["cell"] == "s0d0":
            print(
                f"# REUSE ONLY {stage['family']} anchor: {stage['output']}; "
                "missing/partial refuses."
            )
            print("# Original generation argv (reference only): " + shlex.join(stage["command"]))
        else:
            print(
                shlex.join(launch_command(stage, stage["output"], "${PATH}")).replace(
                    "'PATH=${PATH}'", 'PATH="$PATH"'
                )
            )
    print("Headroom: XG_GPU_IDLE=1; XG_AVAILABLE_WAIT=43200; XG_START_ATTEMPTS=1.")
    print(
        f"Per-stage container deadline {RUN_SECONDS}s; "
        "build/headroom outside it; partial attempts refuse."
    )
    print("Gemma KV=16 GiB; Qwen KV=8 GiB; 16384 context; 256 tokens; full 1024 order; seed 17.")
    print(
        "No Gemma noise-band extension is registered or inferred; its exact anchor needs no repeat."
    )
    print("Status append path: /home/jonas/git/xgraph/target/queue/mechanism-" + phase + ".log")
    if phase == "variations":
        print(
            "Requires current admission.json and a separate --authorization "
            "bound to its SHA-256, amendment, plan and commit."
        )
        print(
            "CPU score: python -I -m xgraph.mechanism_admission "
            "--phase variations --run --authorization <owner-file>"
        )


def run(phase, authorization):
    amendment, plan, head = preflight()
    proof = admission_proof(amendment, plan, head)
    receipt = ADMISSION / "admission.json"
    if phase == "variations":
        if not receipt.is_file() or _json(receipt) != proof:
            raise ValueError("both-family admission receipt missing or stale")
        require_variations(
            proof,
            _json(authorization) if authorization else None,
            sha256_file(HERE / "amendment.json"),
            sha256_file(VARIATIONS / "manifest.json"),
            sha256_file(receipt),
            head,
        )
    # Only the owner reaches these writes; authorization is checked before them.
    log = Path(f"/home/jonas/git/xgraph/target/queue/mechanism-{phase}.log")
    with (HERE / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _status(log, phase, "start")
        final_code = 0
        try:
            if phase == "admission":
                if receipt.exists() and _json(receipt) != proof:
                    raise ValueError("existing admission differs; refusing overwrite")
                if not receipt.exists():
                    with receipt.open("x") as stream:
                        stream.write(json.dumps(proof, indent=2) + "\n")
                for family in FAMILIES:
                    _status(log, family + "-anchor", "skipped-verified")
                _status(log, "both-family-admission", "verified")
                return
            for stage in (s for s in plan["stages"] if s["cell"] != "s0d0"):
                label = stage["family"] + "-" + stage["cell"]
                if repro.check_stage(plan, stage):
                    _status(log, label, "skipped-verified")
                    continue
                output = Path(stage["output"])
                attempt = Path(str(output) + ".attempt.json")
                command = launch_command(stage, output)
                output.parent.mkdir(exist_ok=True)
                with attempt.open("x") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "commit": head,
                                "argv": command,
                                "authorization_sha256": sha256_file(Path(authorization)),
                                "admission_sha256": sha256_file(receipt),
                                "inherited_environment_sha256": hashlib.sha256(
                                    json.dumps(dict(os.environ), sort_keys=True).encode()
                                ).hexdigest(),
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                _status(log, label, "start")
                with Path(str(output) + ".log").open("x") as stream:
                    completed = subprocess.run(
                        command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=False
                    )
                _status(log, label, "finished", completed.returncode)
                if completed.returncode:
                    raise ValueError(
                        f"{label} failed with exit {completed.returncode}; no automatic retry"
                    )
                preflight()
                if not repro.check_stage(plan, stage):
                    raise ValueError(f"{label} incomplete after launcher")
                _status(log, label, "verified")
            proof["status"] = proof["reproduction_status"]
            (VARIATIONS / "anchor-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
            repro.score(VARIATIONS, plan, proof=proof)
            _status(log, "score", "verified")
        except Exception:
            final_code = 2
            _status(log, phase, "failed", 2)
            raise
        finally:
            _status(log, phase, "end", final_code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("admission", "variations"), required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--run", action="store_true")
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    if args.run:
        run(args.phase, args.authorization)
    else:
        preview(args.phase)


if __name__ == "__main__":
    main()
