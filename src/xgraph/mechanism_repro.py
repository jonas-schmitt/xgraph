"""Frozen exact/near reproduction admission and scoring for the GPU owner's queue.

``python -I -m xgraph.mechanism_repro verify|stage|anchors|score --out <queue>``
checks a prepared queue without starting a model. The hash-bound gate must exist
before any GPU stage. Effective request messages include the answer prefill;
tokenizer/chat-template files and generation configuration are frozen inputs.
"""

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

from xgraph.mechanism import interaction
from xgraph.mintqa_queue import REPO, _code_digest
from xgraph.mintqa_rematch import clustered_difference
from xgraph.prompts import final_answer
from xgraph.provenance import provenance, sha256_file
from xgraph.score import normalize, record_failure

GATE_RULES = {
    "questions_per_family": 128,
    "primary": {"status": "exact-reproduction", "minimum_identical_texts": 128},
    "secondary": {
        "status": "near-reproduction",
        "minimum_identical_texts": 120,
        "maximum_accuracy_difference_questions": 2,
    },
    "scorer": "failure-filtered final_answer; NFC/casefold/strip exact label equality; "
    "historical and anchor rescored identically, without survey weights",
    "prompt_proof": "byte-identical effective request messages for every selected id, "
    "including the answer prefill; differing texts must have differing generation token ids",
    "admission": "both families must pass primary or secondary before any variation",
    "numeric_drift_band": "plus/minus absolute population-weighted anchor-minus-historical "
    "accuracy difference under the same scorer, attached to every contrast; absolute "
    "contrasts at or below the band are not identified; sensitivity only, not a confidence bound",
    "historical_selected_check": "descriptive text comparison, not an additional admission gate",
}


def amended_rules(registration, noise, family):
    """Apply the prospective Qwen-only function; preserve the original gate."""
    fixed = registration["fixed_candidate_secondary_rule"]
    if (
        fixed["minimum_identical_texts"] != "max(0, I - 2)"
        or fixed["maximum_accuracy_difference_questions"] != "max(2, abs(delta))"
    ):
        raise ValueError("noise amendment differs from the registered fixed rule")
    if family not in ("qwen38", "gemma31") or noise["n"] != 128:
        raise ValueError("noise amendment requires the registered family and panel")
    rules = copy.deepcopy(GATE_RULES)
    if family == "qwen38":
        rules["secondary"] = {
            "status": "noise-band-reproduction",
            "minimum_identical_texts": max(0, noise["identical_texts"] - 2),
            "maximum_accuracy_difference_questions": max(
                2, abs(noise["signed_accuracy_questions"])
            ),
        }
    return rules


def message_bytes(messages):
    """Stable UTF-8 representation preserving every effective message's content bytes."""
    return json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def correct(record, answer):
    """Strict label exact match, applied identically to historical and new records."""
    if record_failure(record):
        return False
    label = final_answer(record["text"])
    return label is not None and normalize(label) == normalize(answer)


def freeze_gate(root, manifest):
    """Register the fixed fallback only in a queue with no generation artifacts yet."""
    if len(manifest["selected_ids"]) != 128 or len(set(manifest["selected_ids"])) != 128:
        raise ValueError("gate registration requires the frozen 128 distinct question ids")
    for stage in manifest["stages"]:
        path = Path(stage["output"])
        if any(
            Path(str(path) + suffix).exists()
            for suffix in ("", ".log", ".summary.json", ".provenance.json")
        ):
            raise ValueError("cannot register the gate after a GPU stage has started")
    if any(
        (root / name).exists()
        for name in ("anchor-proof.json", "analysis.json", "gate.json", "gate.sha256")
    ):
        raise ValueError("gate or results already exist; registration is immutable")
    inputs = [
        root / "requests.jsonl",
        Path(manifest["gold"]),
        Path(manifest["historical_effective_requests"]),
    ]
    gate = {
        "status": "pre-registered before any queue GPU run",
        "rules": GATE_RULES,
        "selected_ids_sha256": hashlib.sha256(message_bytes(manifest["selected_ids"])).hexdigest(),
        "provenance": provenance(inputs),
    }
    with (root / "gate.json").open("x") as file:
        file.write(json.dumps(gate, indent=2) + "\n")
    digest = sha256_file(root / "gate.json")
    with (root / "gate.sha256").open("x") as file:
        file.write(digest + "\n")
    return digest


def load_gate(root, manifest):
    digest = sha256_file(root / "gate.json")
    if digest != manifest["gate_sha256"] or digest != (root / "gate.sha256").read_text().strip():
        raise ValueError("frozen gate hash changed")
    gate = json.loads((root / "gate.json").read_text())
    if gate["rules"] != GATE_RULES:
        raise ValueError("gate rules differ from the pre-registered classifier")
    if len(manifest["selected_ids"]) != 128 or len(set(manifest["selected_ids"])) != 128:
        raise ValueError("gate requires the frozen 128 distinct question ids")
    if (
        gate["selected_ids_sha256"]
        != hashlib.sha256(message_bytes(manifest["selected_ids"])).hexdigest()
    ):
        raise ValueError("gate question ids changed")
    return gate


def effective_messages(manifest, stage):
    """Reconstruct generation's effective messages under its frozen clean source."""
    from xgraph.generate import opened_requests
    from xgraph.index import saved_span_syntax
    from xgraph.prompts import observed_requests

    flags = stage["expected_flags"]
    syntax = saved_span_syntax(Path(flags["--kg-index"]))
    requests = list(read(flags["--requests"]).values())
    requests = observed_requests(
        requests, int(flags["--observe-paths"]), syntax, select="--observe-select" in flags
    )
    if "--open" in flags:
        requests = opened_requests(requests, syntax.opener)
    return {row["id"]: row["messages"] for row in requests}


def classify_anchor(historical, current, gold, old_messages, new_messages, rules):
    """Classify one complete family; prompt proof covers even identical answers."""
    ids = list(historical)
    if len(ids) != rules["questions_per_family"] or any(
        set(rows) != set(ids) for rows in (current, gold, old_messages, new_messages)
    ):
        raise ValueError("anchor classifier requires the same complete frozen question ids")
    prompt_mismatches, differences, prompt_hashes = [], [], {}
    generation_proof = True
    historical_correct = anchor_correct = 0
    for qid in ids:
        a, b = historical[qid], current[qid]
        before, after = message_bytes(old_messages[qid]), message_bytes(new_messages[qid])
        prompt_hashes[qid] = {
            "historical": hashlib.sha256(before).hexdigest(),
            "anchor": hashlib.sha256(after).hexdigest(),
        }
        if before != after:
            prompt_mismatches.append(qid)
        historical_correct += correct(a, gold[qid]["answer"])
        anchor_correct += correct(b, gold[qid]["answer"])
        if a["text"] == b["text"]:
            continue
        tokens_a, tokens_b = a.get("token_ids"), b.get("token_ids")
        has_difference = (
            isinstance(tokens_a, list) and isinstance(tokens_b, list) and tokens_a != tokens_b
        )
        generation_proof &= has_difference
        prefix = None
        if has_difference:
            prefix = next(
                (i for i, (x, y) in enumerate(zip(tokens_a, tokens_b, strict=False)) if x != y),
                min(len(tokens_a), len(tokens_b)),
            )
        differences.append(
            {
                "id": qid,
                "first_token_difference": prefix,
                "effective_messages_identical": before == after,
                "historical_effective_messages_sha256": hashlib.sha256(before).hexdigest(),
                "anchor_effective_messages_sha256": hashlib.sha256(after).hexdigest(),
                "historical_text": a["text"],
                "new_text": b["text"],
            }
        )
    identical = len(ids) - len(differences)
    accuracy_difference = anchor_correct - historical_correct
    if not prompt_mismatches and identical >= rules["primary"]["minimum_identical_texts"]:
        status = rules["primary"]["status"]
    elif (
        not prompt_mismatches
        and generation_proof
        and identical >= rules["secondary"]["minimum_identical_texts"]
        and abs(accuracy_difference) <= rules["secondary"]["maximum_accuracy_difference_questions"]
    ):
        status = rules["secondary"]["status"]
    else:
        status = "failed-reproduction"
    return {
        "status": status,
        "questions": len(ids),
        "identical_texts": identical,
        "historical_correct": historical_correct,
        "anchor_correct": anchor_correct,
        "accuracy_difference_questions": accuracy_difference,
        "historical_accuracy": historical_correct / len(ids),
        "anchor_accuracy": anchor_correct / len(ids),
        "prompt_mismatches": prompt_mismatches,
        "effective_messages_sha256": prompt_hashes,
        "generation_token_proof": generation_proof,
        "differences": differences,
    }


def read(path):
    with Path(path).open() as file:
        rows = [json.loads(line) for line in file]
    indexed = {r["id"]: r for r in rows}
    if len(rows) != len(indexed):
        raise ValueError(f"duplicate ids: {path}")
    return indexed


def options(command):
    result = {}
    at = 0
    while at < len(command):
        key = command[at]
        at += 1
        if not key.startswith("--"):
            continue
        value = True
        if at < len(command) and not command[at].startswith("--"):
            value = command[at]
            at += 1
        result[key] = value
    return result


def verify(root):
    path = root / "manifest.json"
    if sha256_file(path) != (root / "manifest.sha256").read_text().strip():
        raise ValueError("manifest hash changed")
    m = json.loads(path.read_text())
    load_gate(root, m)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip():
        raise ValueError("refusing dirty worktree")
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        != m["commit"]
    ):
        raise ValueError("commit differs from frozen owner queue")
    if _code_digest() != m["code_sha256"]:
        raise ValueError("source code changed")
    for item in m["inputs"]:
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"input changed: {item['path']}")
    for item in m["weights_stat"]:
        stat = Path(item["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (item["bytes"], item["mtime_ns"]):
            raise ValueError(f"model weights changed: {item['path']}")
    historical = read(m["historical_effective_requests"])
    for stage in (s for s in m["stages"] if s["cell"] == "s0d0"):
        messages = effective_messages(m, stage)
        if any(
            message_bytes(historical[q]["messages"]) != message_bytes(messages[q])
            for q in m["selected_ids"]
        ):
            raise ValueError(f"effective request messages differ: {stage['family']}")
    return m


def check_stage(m, stage):
    path = Path(stage["output"])
    files = [path, Path(str(path) + ".summary.json"), Path(str(path) + ".provenance.json")]
    if not any(p.exists() for p in files):
        if Path(str(path) + ".log").exists():
            raise ValueError(f"partial stage log exists: {path}")
        return False
    if not all(p.is_file() for p in files):
        raise ValueError(f"partial stage exists: {path}")
    rows = read(path)
    if set(rows) != set(m["full_ids"]):
        raise ValueError(f"incomplete or extra ids: {path}")
    meta = json.loads(files[2].read_text())
    summary = json.loads(files[1].read_text())
    expected = stage["expected_flags"]
    actual = options(meta["provenance"]["command"])
    actual.pop("--provenance-only", None)
    if actual != expected:
        raise ValueError(f"argv differs: {path}: {actual} != {expected}")
    if (
        meta["provenance"]["repo_commit"] != stage.get("source_commit", m["commit"])
        or meta["provenance"]["repo_dirty"]
    ):
        raise ValueError(f"generation source identity differs: {path}")
    request = Path(expected["--requests"])
    if not any(
        i["path"] == str(request) and i["sha256"] == sha256_file(request)
        for i in meta["provenance"]["inputs"]
    ):
        raise ValueError(f"generation request hash differs: {path}")
    historical = m["historical_summaries"][stage["family"]]
    if (
        summary["engine_args"] != historical["engine_args"]
        or summary["vllm_version"] != historical["vllm_version"]
    ):
        raise ValueError(f"native serving configuration differs: {path}")
    if summary["max_tokens"] != 256 or summary["requests"] != len(m["full_ids"]):
        raise ValueError(f"budget or batch differs: {path}")
    if [r["id"] for r in rows.values()] != m["full_ids"]:
        raise ValueError(f"output order differs from historical panel: {path}")
    return True


def anchors(root, m):
    gate = load_gate(root, m)
    gold = json.loads(Path(m["gold"]).read_text())
    old_messages = read(m["historical_effective_requests"])
    proof = {
        "families": {},
        "gate_rules": gate["rules"],
    }
    inputs = [
        root / "manifest.json",
        root / "requests.jsonl",
        Path(m["gold"]),
        Path(m["historical_effective_requests"]),
    ]
    if (root / "gate.json").exists():
        inputs.append(root / "gate.json")
    for family in ("qwen38", "gemma31"):
        stage = next(s for s in m["stages"] if s["family"] == family and s["cell"] == "s0d0")
        if not check_stage(m, stage):
            raise ValueError("anchor missing")
        current = read(stage["output"])
        historical = read(m["historical_outputs"][family])
        messages = effective_messages(m, stage)
        ids = m["selected_ids"]
        proof["families"][family] = {
            **classify_anchor(
                {q: historical[q] for q in ids},
                {q: current[q] for q in ids},
                {q: gold[q] for q in ids},
                {q: old_messages[q]["messages"] for q in ids},
                {q: messages[q] for q in ids},
                gate["rules"],
            ),
            "historical_sha256": sha256_file(Path(m["historical_outputs"][family])),
            "new_sha256": sha256_file(Path(stage["output"])),
        }
        inputs.extend([Path(m["historical_outputs"][family]), Path(stage["output"])])
    statuses = {f["status"] for f in proof["families"].values()}
    proof["status"] = (
        "failed-reproduction"
        if "failed-reproduction" in statuses
        else ("near-reproduction" if "near-reproduction" in statuses else "exact-reproduction")
    )
    proof["limits"] = (
        "Same full panel and order, prefix caching, token budget, effective request messages, "
        "seed, input identities and serving args. Effective messages are reconstructed from "
        "the historical source, not recovered server traces. Historical wheel hashes, cache "
        "contents and scheduler traces were not retained; a mismatch cannot be attributed "
        "solely to nondeterminism."
    )
    proof["provenance"] = provenance(inputs)
    (root / "anchor-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    if proof["status"] == "failed-reproduction":
        raise ValueError(
            "historical reproduction failed; inspect anchor-proof.json before further GPU work"
        )
    return proof


def drift_contrast(contrast, band):
    """Attach the registered numerical sensitivity separately from sampling uncertainty."""
    difference = contrast["difference"]
    return {
        **contrast,
        "numeric_drift_band": [-band, band],
        "drift_adjusted_range": [difference - band, difference + band],
        "identification": "not identified"
        if abs(difference) <= band
        else "exceeds numeric-drift band",
    }


def score(root, m, *, proof=None):
    # The amended owner queue supplies its separately verified admission proof
    # and writes only its own analysis. The original CLI still uses the old gate.
    if proof is None:
        proof = anchors(root, m)
    members = json.loads(Path(m["sample"]).read_text())["members"]
    gold = json.loads(Path(m["gold"]).read_text())
    mass = sum(r["weight"] for r in members)
    report = {
        "status": "completed exploratory development factorial",
        "cells": {},
        "interactions": {},
        "historical_selected_reproduction": {},
        "reproduction_status": proof["status"],
        "anchor_families": proof["families"],
        "scorer": GATE_RULES["scorer"],
        "numeric_drift": {},
        "contrasts": {},
        "limits": "Exploratory development factorial. Numeric-drift bands are anchor-based "
        "sensitivities, not bounds on unobserved variation-cell drift or confidence intervals. "
        "Historical accuracy is rescored with exact label matching; earlier containment "
        "numbers are a different metric. Selection changes its historical demonstrations.",
    }
    for family in ("qwen38", "gemma31"):
        cells = {}
        for stage in [s for s in m["stages"] if s["family"] == family]:
            if not check_stage(m, stage):
                raise ValueError("factorial incomplete")
            rows = read(stage["output"])
            if stage["cell"] == "s1d1":
                previous = read(m["historical_selected_outputs"][family])
                changed = [
                    qid for qid in m["selected_ids"] if rows[qid]["text"] != previous[qid]["text"]
                ]
                report["historical_selected_reproduction"][family] = {
                    "identical_texts": len(m["selected_ids"]) - len(changed),
                    "questions": len(m["selected_ids"]),
                    "different_ids": changed,
                }
            details = {
                q["id"]: {"correct": correct(rows[q["id"]], gold[q["id"]]["answer"])}
                for q in members
            }
            cells[stage["cell"]] = details
            report["cells"][family + "/" + stage["cell"]] = {
                "questions": len(members),
                "accuracy": sum(q["weight"] * details[q["id"]]["correct"] for q in members) / mass,
                "per_question": details,
            }
        historical = read(m["historical_outputs"][family])
        historical_accuracy = (
            sum(
                q["weight"] * correct(historical[q["id"]], gold[q["id"]]["answer"]) for q in members
            )
            / mass
        )
        anchor_accuracy = report["cells"][family + "/s0d0"]["accuracy"]
        delta = anchor_accuracy - historical_accuracy
        band = abs(delta)
        report["numeric_drift"][family] = {
            "historical_weighted_accuracy": historical_accuracy,
            "anchor_weighted_accuracy": anchor_accuracy,
            "weighted_accuracy_difference": delta,
            "numeric_drift_band": [-band, band],
            "units": "accuracy proportion",
        }
        report["contrasts"][family] = {}
        for second, first in combinations(("s0d0", "s0d1", "s1d0", "s1d1"), 2):
            contrast = clustered_difference(
                [float(cells[first][q["id"]]["correct"]) for q in members],
                [float(cells[second][q["id"]]["correct"]) for q in members],
                members,
                10_000,
                17,
            )
            report["contrasts"][family][f"{first}-{second}"] = drift_contrast(
                {**contrast, "units": "accuracy proportion"}, band
            )
        result = drift_contrast(interaction(cells, members), band)
        report["interactions"][family] = result
        report["contrasts"][family]["interaction"] = result
    report["provenance"] = provenance(
        [
            root / "manifest.json",
            root / "anchor-proof.json",
            root / "gate.json",
            Path(m["sample"]),
            Path(m["gold"]),
            *[Path(s["output"]) for s in m["stages"]],
            *[Path(p) for p in m["historical_selected_outputs"].values()],
            *[Path(p) for p in m["historical_outputs"].values()],
        ]
    )
    (root / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "stage", "anchors", "score"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--family")
    parser.add_argument("--cell")
    args = parser.parse_args()
    m = verify(args.out)
    if args.action == "stage":
        stage = next(
            s for s in m["stages"] if s["family"] == args.family and s["cell"] == args.cell
        )
        raise SystemExit(0 if check_stage(m, stage) else 1)
    if args.action == "anchors":
        anchors(args.out, m)
    if args.action == "score":
        score(args.out, m)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
