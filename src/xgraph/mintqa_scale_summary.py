"""CPU-only development table from explicit scale/rematch output paths."""

from __future__ import annotations

import json
import math
from pathlib import Path

from xgraph.mintqa_bench import answer_contains, dataset_rows
from xgraph.mintqa_panels import indexed, read_jsonl, write_json
from xgraph.mintqa_queue import _code_digest
from xgraph.mintqa_rematch import usage
from xgraph.mintqa_scale import SCALE_ARMS, SCALE_FAMILIES
from xgraph.prompts import final_answer
from xgraph.provenance import provenance
from xgraph.score import record_outputs

SUMMARY_FAMILIES = (*SCALE_FAMILIES, "qwen25", "qwen38")


def parse_sources(specs):
    result = {}
    for spec in specs:
        name, equals, path = spec.partition("=")
        family, colon, arm = name.partition(":")
        if (
            not equals
            or not colon
            or not path
            or family not in SUMMARY_FAMILIES
            or arm not in SCALE_ARMS
        ):
            raise ValueError(f"--source needs <family>:<arm>=<JSONL path>: {spec}")
        if (family, arm) in result:
            raise ValueError(f"duplicate --source for {family}:{arm}")
        result[family, arm] = Path(path).resolve()
    return result


def summarize(out: Path, source_root: Path, sources: dict) -> dict:
    prepared = source_root / "target/mintqa-rematch"
    manifest_path = prepared / "panels/panels.json"
    questions_path = prepared / "panels/development/questions.json"
    splits = [source_root / f"data/external/mintqa-full/questions/{s}.jsonl" for s in ("ti", "pop")]
    members = json.loads(manifest_path.read_text())["panels"]["development"]["members"]
    ids = [row["id"] for row in members]
    if (
        not members
        or len(set(ids)) != len(ids)
        or set(indexed(json.loads(questions_path.read_text()))) != set(ids)
    ):
        raise ValueError("summary requires unique, aligned development panel ids")
    if any(not math.isfinite(row["weight"]) or row["weight"] <= 0 for row in members):
        raise ValueError("summary requires finite positive panel weights")
    mass = sum(row["weight"] for row in members)
    native = dataset_rows([f"{name}={p}" for name, p in zip(("ti", "pop"), splits, strict=True)])
    inputs = [manifest_path, questions_path, *splits]
    rows = []
    for family in SUMMARY_FAMILIES:
        for arm in SCALE_ARMS:
            local = out / family / f"development/{arm}.jsonl"
            cached = prepared / family / f"development/{arm}.jsonl"
            path = sources.get(
                (family, arm), local if local.exists() or not cached.exists() else cached
            )
            row = {
                "family": family,
                "arm": arm,
                "path": str(path),
                "status": "missing",
                "accuracy": None,
                "weighted_tokens": None,
                "weighted_generated_tokens": None,
                "weighted_observation_tokens": None,
            }
            rows.append(row)
            if not path.is_file():
                continue
            inputs.append(path)
            sidecar = Path(f"{path}.provenance.json")
            if sidecar.is_file():
                inputs.append(sidecar)
            records = indexed(read_jsonl(path))
            row["missing_ids"] = sorted(set(ids) - records.keys())
            row["extra_ids"] = sorted(records.keys() - set(ids))
            if row["missing_ids"] or row["extra_ids"]:
                row["status"] = "incomplete"
                continue
            texts = record_outputs(list(records.values()), ids)
            row["accuracy"] = (
                sum(
                    member["weight"]
                    * answer_contains(
                        final_answer(texts[member["id"]]), native[member["id"]]["answer"]
                    )
                    for member in members
                )
                / mass
            )
            # Older records may omit observation counts, but never invent total usage.
            missing_usage = [
                qid
                for qid, r in records.items()
                if "usage" not in r
                and not ("prompt_tokens" in r and ("generated_tokens" in r or "token_ids" in r))
            ]
            costs = {}
            for qid, record in records.items():
                if qid in missing_usage:
                    continue
                try:
                    costs[qid] = usage(record)
                except KeyError:
                    missing_usage.append(qid)
            if missing_usage:
                row.update(status="missing_usage", missing_usage_ids=missing_usage)
                continue
            for field in ("prompt_tokens", "generated_tokens", "observation_tokens"):
                row[f"weighted_{field}"] = (
                    sum(member["weight"] * costs[member["id"]][field] for member in members) / mass
                )
            row["weighted_tokens"] = (
                row["weighted_prompt_tokens"] + row["weighted_generated_tokens"]
            )
            row["status"] = "complete"
    return {
        "artifact_kind": "xgraph.mintqa_scale_summary",
        "version": 1,
        "panel": "development",
        "questions": len(ids),
        "weight_sum": mass,
        "rows": rows,
        "missing_inputs": [row for row in rows if row["status"] != "complete"],
        "code_sha256": _code_digest(),
        "provenance": provenance(list(dict.fromkeys(inputs))),
    }


def summary_markdown(report):
    lines = [
        "Development panel; population-weighted accuracy and tokens/question.",
        "Tokens include all calls",
        "and forced steps. These are resource proxies, not measured GPU time.",
        "",
        "| Family | Arm | Accuracy (%) | Tokens/q | Status |",
        "|---|---|---:|---:|---|",
    ]
    for row in report["rows"]:
        accuracy = "—" if row["accuracy"] is None else f"{100 * row['accuracy']:.2f}"
        tokens = "—" if row["weighted_tokens"] is None else f"{row['weighted_tokens']:.2f}"
        lines.append(
            f"| {row['family']} | {row['arm']} | {accuracy} | {tokens} | {row['status']} |"
        )
    lines += ["", "Source paths, missing ids, hashes and provenance: `summary.json`.", ""]
    return "\n".join(lines)


def write_summary(out, source_root, sources):
    report = summarize(out, source_root, sources)
    table = summary_markdown(report)
    # Derived summaries may refresh; generation records are never changed.
    for name, value in (
        ("summary.json", report),
        ("summary.md", table),
        ("summary.md.provenance.json", report["provenance"]),
    ):
        path = out / name
        pending = path.with_name(path.name + ".pending")
        if isinstance(value, str):
            pending.parent.mkdir(parents=True, exist_ok=True)
            with pending.open("x") as stream:
                stream.write(value)
        else:
            write_json(pending, value)
        pending.replace(path)
    print(table)
