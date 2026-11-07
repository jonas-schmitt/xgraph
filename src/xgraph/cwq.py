"""ComplexWebQuestions released-subgraph adapter; all preparation is gold-free.

Only the validation scorer here opens answers. Test answers belong exclusively
to the registered scorer in xgraph.cwq_test, after its dated freeze.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from xgraph.provenance import provenance, sha256_file
from xgraph.webqsp import (
    _json,
    _line,
    answer_fields,
    catalog_request_fingerprint,
    catalog_requests,
    convert,
    data_bytes,
    fetch_files,
    parquet_rows,
    score_records,
    shard_paths,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SPEC = ROOT / "scripts/cwq/sources.json"
GRAPH_COLUMNS = ["id", "question", "q_entity", "graph"]
DISK_CAP = 20 * 1024**3
RAM_CAP = 16 * 1024**3
PROCESS_CAP = 4 * 1024**3
MODELS = {
    "qwen38": Path("/home/jonas/models/Qwen3.8-27B-FP8"),
    "gemma31": Path("/home/jonas/models/gemma-4-31B-it-FP8-dynamic"),
}


def source_manifest() -> dict:
    return json.loads(SOURCE_SPEC.read_text())


def source_specs(spec=None) -> dict:
    spec = source_manifest() if spec is None else spec
    return {n: (v["url"], v["bytes"], v["sha256"]) for n, v in spec["files"].items()}


def budget(*roots: Path, reserve=0) -> int:
    used = data_bytes(*roots)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if used + reserve > DISK_CAP or rss > min(PROCESS_CAP, RAM_CAP):
        raise ValueError(f"CWQ resource ceiling exceeded: disk={used}+{reserve}, RSS={rss}")
    return used


def fetch(out: Path) -> dict:
    """New directory only. All shards stay opaque bytes until projected as data."""
    spec = source_manifest()
    if out.name != "rog-cwq-" + spec["revision"]:
        raise ValueError("download directory must be named rog-cwq-<pinned revision>")
    if out.exists():
        raise ValueError(f"download requires its own new directory: {out}")
    size = sum(r["bytes"] for r in spec["files"].values())
    out.parent.mkdir(parents=True, exist_ok=True)
    if size > DISK_CAP or shutil.disk_usage(out.parent).free < size + 1024**3:
        raise ValueError("insufficient disk for pinned CWQ fetch")
    out.mkdir()
    fetch_files(out, source_specs(spec))
    report = {**spec, "gold_loaded": False, "provenance": provenance([SOURCE_SPEC])}
    _json(out / "sources.json", report)
    return report


def verify_source(source: Path, split: str, *, spec=None) -> list[Path]:
    spec = source_manifest() if spec is None else spec
    paths = shard_paths(source, split, sources=spec["files"])
    for path in paths:
        pin = spec["files"][path.name]
        if path.stat().st_size != pin["bytes"] or sha256_file(path) != pin["sha256"]:
            raise ValueError(f"CWQ source size/SHA-256 changed: {path}")
    return paths


def convert_split(source: Path, out: Path, split: str) -> dict:
    spec = source_manifest()
    paths = verify_source(source, split)
    budget(source, out, reserve=1024**3)
    report = convert(
        parquet_rows(source, split, sources=spec["files"], columns=GRAPH_COLUMNS),
        out,
        dataset="cwq",
        gold=False,
    )
    if report["counts"]["questions"] != spec["populations"][split]:
        raise ValueError("CWQ split question population changed")
    report.update(
        artifact_kind="xgraph.cwq",
        split=split,
        revision=spec["revision"],
        graph_columns=GRAPH_COLUMNS,
        native_annotations=None,
        source=str(source.resolve()),
        source_manifest_sha256=sha256_file(SOURCE_SPEC),
        disk_bytes=budget(source, out),
        provenance=provenance([SOURCE_SPEC, *paths]),
    )
    _json(out / "manifest.json", report)
    return report


def build_family_index(data: Path, family: str, model: Path) -> dict:
    from xgraph.index import add_sitelinks, load_kg_index
    from xgraph.syntax import SpanSyntax
    from xgraph.tokenizer import ServedTokenizer
    from xgraph.wikidata import build_index

    manifest = json.loads((data / "manifest.json").read_text())
    if manifest.get("dataset") != "cwq" or manifest.get("gold_loaded") is not False:
        raise ValueError("CWQ indexes require gold-free CWQ components")
    source = Path(manifest["source"])
    budget(source, data, reserve=1024**3)
    out = data / f"index-{family}"
    if out.exists():
        raise ValueError(f"index output exists: {out}")
    started = time.monotonic()
    tokenizer = model / "tokenizer.json"
    served = ServedTokenizer.from_file(tokenizer)
    build_index(data / "kg", served, SpanSyntax(), out)
    add_sitelinks(out, np.zeros(len(load_kg_index(out).labels.owner_ids), np.int32))
    report = {
        "family": family,
        "seconds": time.monotonic() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "disk_bytes": budget(source, data),
        "fingerprint": served.fingerprint,
        "provenance": provenance([data / "manifest.json", tokenizer]),
    }
    _json(data / f"index-{family}.report.json", report)
    return report


def prepare_requests(data: Path) -> dict:
    """Use the frozen Freebase transfer demonstrations and component anchors."""
    from xgraph.toolkg import KGStore

    manifest = json.loads((data / "manifest.json").read_text())
    if manifest.get("dataset") != "cwq" or manifest.get("gold_loaded") is not False:
        raise ValueError("CWQ requests require gold-free CWQ components")
    panel = json.loads((data / "questions.json").read_text())
    store = KGStore(data / "kg.sqlite")
    for kind in ("S", "blind5", "menuobs", "agent"):
        with (data / f"requests-{kind}.jsonl").open("w") as file:
            for question in panel:
                starts = [store.label(int(e[1:])) for e in question["anchor"]]
                _line(
                    file,
                    catalog_requests(
                        question,
                        starts,
                        demos=4 if kind in {"S", "agent"} else 5,
                        observed=kind == "menuobs",
                    ),
                )
    store._con.close()
    paths = sorted(p for p in data.rglob("*") if p.is_file() and p.name != "catalog.json")
    report = {
        "artifact_kind": "xgraph.cwq.catalog",
        "dataset": "cwq",
        "gold_loaded": False,
        "models": {f: str(p) for f, p in MODELS.items()},
        "request_builder_sha256": catalog_request_fingerprint(),
        "start_rank": "equal zero sitelinks, then degree, then first label row",
        "files": [{"path": str(p.resolve()), "sha256": sha256_file(p)} for p in paths],
        "provenance": provenance([data / "manifest.json"]),
    }
    _json(data / "catalog.json", report)
    return report


def score_validation(records: list[dict], panel: list[dict]) -> dict:
    result = score_records(records, panel, failed_zero=True)
    result.update(
        dataset="cwq",
        split="validation",
        primary_metrics=["hits_at_1"],
        secondary_metrics=["f1", "exact_set", "gcr_hit", "gcr_f1"],
    )
    return result


def prepare_cpu(source: Path, out: Path) -> dict:
    """Sequential bounded workers; stop a worker if the combined disk cap is crossed."""
    if out.exists():
        raise ValueError("CPU preparation requires a new output directory")
    logs = out.with_name(out.name + ".preparation")
    logs.mkdir(parents=True, exist_ok=False)
    commands = [
        ["convert", "--source", str(source), "--split", "validation", "--out", str(out)],
        ["store", "--data", str(out)],
        *[["index", "--data", str(out), "--family", f] for f in MODELS],
        ["requests", "--data", str(out)],
    ]
    stages = []
    for number, command in enumerate(commands):
        budget(source, out, logs)
        argv = [
            "nice",
            "-n",
            "10",
            "prlimit",
            f"--as={PROCESS_CAP}",
            sys.executable,
            "-I",
            "-m",
            "xgraph.cwq",
            *command,
        ]
        started = time.monotonic()
        with (logs / f"{number}-{command[0]}.log").open("x") as stream:
            process = subprocess.Popen(
                argv,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={
                    **os.environ,
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "CUDA_VISIBLE_DEVICES": "",
                    "TOKENIZERS_PARALLELISM": "false",
                },
            )
            try:
                while process.poll() is None:
                    budget(source, out, logs)
                    time.sleep(1)
            except BaseException:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
        stages.append(
            {
                "argv": argv,
                "seconds": time.monotonic() - started,
                "returncode": process.returncode,
                "disk_bytes": budget(source, out, logs),
            }
        )
        _json(logs / "stages.json", {"stages": stages, "provenance": provenance([SOURCE_SPEC])})
        if process.returncode:
            raise RuntimeError(f"CWQ CPU preparation stopped: {command}")
    return {"stages": stages, "disk_bytes": budget(source, out, logs)}


def validation_gold(source: Path, ids: list[str]) -> list[dict]:
    from xgraph.webqsp_test import unique_ids

    wanted = set(unique_ids(ids))
    spec = source_manifest()
    verify_source(source, "validation")
    panel = [
        {"id": row["id"], **answer_fields(row, coverage=False)}
        for row in parquet_rows(
            source, "validation", sources=spec["files"], columns=["id", "answer"]
        )
        if row["id"] in wanted
    ]
    if set(unique_ids(q["id"] for q in panel)) != wanted:
        raise ValueError("validation gold does not cover the declared IDs")
    return panel


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("convert")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", choices=("validation", "test"), required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("index")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--family", choices=MODELS, required=True)
    p.add_argument("--model", type=Path)
    for name in ("store", "requests"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, required=True)
    p = sub.add_parser("score-validation")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "fetch":
        report = fetch(args.out)
    elif args.command == "prepare":
        report = prepare_cpu(args.source, args.out)
    elif args.command == "convert":
        report = convert_split(args.source, args.out, args.split)
    elif args.command == "index":
        report = build_family_index(args.data, args.family, args.model or MODELS[args.family])
    elif args.command == "store":
        from xgraph.toolkg import build_store

        source = Path(json.loads((args.data / "manifest.json").read_text())["source"])
        budget(source, args.data, reserve=1024**3)
        if (args.data / "kg.sqlite").exists():
            raise ValueError("CWQ store already exists")
        started = time.monotonic()
        build_store(args.data / "kg", args.data / "kg.sqlite")
        report = {
            "seconds": time.monotonic() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "disk_bytes": budget(source, args.data),
            "provenance": provenance([args.data / "manifest.json"]),
        }
        _json(args.data / "store.report.json", report)
    elif args.command == "requests":
        report = prepare_requests(args.data)
    else:
        from xgraph.webqsp import _read_jsonl

        ids = [q["id"] for q in _read_jsonl(args.requests)]
        panel = validation_gold(args.source, ids)
        report = score_validation(_read_jsonl(args.outputs), panel)
        report["provenance"] = provenance([args.requests, args.outputs, SOURCE_SPEC])
        _json(args.out, report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                in {
                    "artifact_kind",
                    "split",
                    "counts",
                    "seconds",
                    "peak_rss_mib",
                    "disk_bytes",
                    "revision",
                }
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
