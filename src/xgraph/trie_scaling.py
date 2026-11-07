"""Bounded CPU evidence for explicit GCR tries and indexed xgraph frontiers.

Samples are deterministic DFS prefixes, never inference inputs or unbiased
estimates. Full trie sizes remain unknown when only a prefix was constructed.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from pathlib import Path

from xgraph.gcr import prepare_pilot
from xgraph.index import load_kg_index
from xgraph.mintqa_panels import indexed, read_jsonl, write_json
from xgraph.oracle import MAX_REACHED
from xgraph.path_trie import TokenTrie, collect_paths, trie_arguments
from xgraph.programs import MatchedExecutor, record_path
from xgraph.provenance import provenance
from xgraph.tokenizer import ServedTokenizer
from xgraph.toolkg import KGStore


def _python_bytes(trie: TokenTrie) -> int:
    """Unique reachable Python object sizes, including sequence storage and dicts.

    This excludes allocator fragmentation, enumeration lists, temporary encoding,
    serialization, worker copies, native allocations and any model weights.
    """
    pending, seen, size = [trie, vars(trie)], set(), 0
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        size += sys.getsizeof(item)
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset)):
            pending.extend(item)
    return size


def measure_trie(store, starts, tokenizer, *, depth, walks, limit):
    """Measure the production trie on all walks or an explicitly bounded prefix."""
    if type(limit) is not int or limit < 1 or type(walks) is not int or walks < 0:
        raise ValueError("walk count must be nonnegative and sample limit positive")
    started = time.perf_counter()
    paths = collect_paths(store, starts, hops=depth, max_paths=limit, overflow="truncate")
    if len(paths.paths) != min(walks, limit) or paths.truncated != (walks > limit):
        raise ValueError(f"depth {depth}: walk count disagrees with exact admission: {walks}")
    trie = TokenTrie.from_paths(paths.paths, tokenizer, empty="closed")
    construction_seconds = time.perf_counter() - started
    size = _python_bytes(trie)
    payload_bytes = len(json.dumps(trie_arguments("scaling", trie)).encode())
    exact = not paths.truncated
    return {
        "depth": depth,
        "exact_graph_walks": walks,
        "method": "exact" if exact else "stable_dfs_prefix",
        "retained_graph_walks": len(paths.paths),
        "token_paths": len(trie.sequences),
        "measured_trie_nodes": len(trie.children),
        "measured_trie_python_bytes": size,
        "measured_payload_json_bytes": payload_bytes,
        "full_trie_nodes": len(trie.children) if exact else None,
        "full_trie_python_bytes": size if exact else None,
        "full_payload_json_bytes": payload_bytes if exact else None,
        "full_trie_nodes_lower_bound": len(trie.children),
        "construction_seconds": construction_seconds,
        "seconds": time.perf_counter() - started,
    }


def replay_frontiers(executor, request, record):
    """Replay a saved model plan through the engine-matched CPU path executor.

    Every prefix retains all live start senses, before answer commitment, under
    the engine's frontier cap and ordering. No gold path is substituted.
    """
    result = {"id": request["id"], "entity_cap": MAX_REACHED, "steps": []}
    parsed = record_path(record)
    if parsed is None:
        return {**result, "status": "no_parsed_path"}
    start, relations = parsed
    for hop in range(len(relations) + 1):
        entities = executor.frontier(
            start, relations[:hop], request.get("anchor"), request.get("kg_version")
        )
        result["steps"].append(
            {"hop": hop, "entities": len(entities), "entity_ids": sorted(entities)}
        )
    return {**result, "status": "replayed", "start": start, "relations": list(relations)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("panels", "requests", "reference-panels", "reference-requests", "db", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path("/home/jonas/models"))
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--exact-limit", type=int, default=20000)
    parser.add_argument("--prefix-paths", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=1650)
    args = parser.parse_args(argv)
    if args.exact_limit < 1 or not 1 <= args.prefix_paths <= args.exact_limit:
        parser.error("require 1 <= prefix-paths <= exact-limit")
    if not 0 < args.max_seconds <= 1750:
        parser.error("max-seconds must be in (0, 1750]")
    started = time.perf_counter()
    panels = json.loads(args.panels.read_text())
    requests = read_jsonl(args.requests)
    store = KGStore(args.db)
    admission = prepare_pilot(
        panels,
        json.loads(args.reference_panels.read_text()),
        requests,
        read_jsonl(args.reference_requests),
        store,
        max_paths=args.exact_limit,
        depths=(1, 2, 3, 4),
    )
    by_id = indexed(requests)
    inputs = [
        args.panels,
        args.requests,
        args.reference_panels,
        args.reference_requests,
        args.db,
        Path(__file__),
        Path(__file__).with_name("gcr.py"),
        Path(__file__).with_name("path_trie.py"),
        Path(__file__).with_name("programs.py"),
        Path(__file__).with_name("index.py"),
        Path(__file__).with_name("tokenizer.py"),
    ]
    report = {
        "artifact_kind": "xgraph.trie_scaling",
        "model_calls": 0,
        "seed": 17,
        "pilot": panels.get("pilot"),
        "ids": admission["ids"],
        "walk_counts": admission,
        "limits": {"exact_walks": args.exact_limit, "sample_walks": args.prefix_paths},
        "method": {
            "walks": "exact endpoint-multiplicity DP; directed walks, cycles, parallel relations",
            "tries": "production full-string tokenization, canonical labels, both PATH markers",
            "nodes": "children list length, including root; duplicate token paths collapse",
            "bytes": "unique reachable Python object sizes; separate JSON payload bytes",
            "payload": "UTF-8 default json.dumps(trie_arguments('scaling', trie)); no framing",
            "samples": "stable DFS prefixes; node lower bounds only, no extrapolation of full heap",
            "index": "actual saved KG index file sizes; one shared index, not per question",
            "frontiers": "CPU MatchedExecutor replay of saved development paths-allfix plans",
            "frontier_steps": "start and each relation; live frontier before answer commitment",
        },
        "tokenizers": {},
        "frontiers": {},
    }
    model_names = {
        "qwen38": "Qwen3.8-27B-FP8",
        "gemma31": "gemma-4-31B-it-FP8-dynamic",
        "native": "GCR-Meta-Llama-3.1-8B-Instruct",
    }
    for family, model_name in model_names.items():
        tokenizer_file = args.models / model_name / "tokenizer.json"
        tokenizer = ServedTokenizer.from_file(tokenizer_file)
        inputs.append(tokenizer_file)
        measurements = []
        for row in admission["per_question"]:
            request = by_id[row["id"]]
            for depth in range(1, 5):
                if time.perf_counter() - started > args.max_seconds:
                    raise TimeoutError("bounded scaling deadline exceeded; no complete report")
                walks = sum(row["paths_by_hop"][str(h)] for h in range(1, depth + 1))
                limit = args.exact_limit if walks <= args.exact_limit else args.prefix_paths
                measurements.append(
                    {
                        "id": row["id"],
                        **measure_trie(
                            store,
                            request.get("anchor") or request["topics"],
                            tokenizer,
                            depth=depth,
                            walks=walks,
                            limit=limit,
                        ),
                    }
                )
        report["tokenizers"][family] = {
            "tokenizer": str(tokenizer_file),
            "fingerprint": tokenizer.fingerprint,
            "measurements": measurements,
        }
        print(f"measured {family}: {len(measurements)} question/depth tries", flush=True)
        if family != "native":
            directory = args.prepared / "indexes" / family
            files = sorted(p for p in directory.iterdir() if p.is_file())
            trace_file = args.prepared / family / "development/paths-allfix.jsonl"
            profile_file = Path(str(trace_file) + ".provenance.json")
            request_file = args.prepared / "panels/development/requests.jsonl"
            trace, trace_requests = (
                indexed(read_jsonl(trace_file)),
                indexed(read_jsonl(request_file)),
            )
            profile = json.loads(profile_file.read_text())
            if not (
                profile["paths"]
                and profile["committed_paths"]
                and profile["relations"] == "frontier"
                and not profile["kg_versions"]
            ):
                raise ValueError(f"{family}: unsupported development trace profile")
            index = load_kg_index(directory)
            executor = MatchedExecutor(
                index,
                tokenizer,
                start_rank=profile["start_rank"],
                unquoted_starts=profile["unquoted_starts"],
            )
            replay = []
            for qid in admission["ids"]:
                if trace_requests[qid]["messages"] != by_id[qid]["messages"]:
                    raise ValueError(f"{family}/{qid}: development question messages differ")
                replay.append(replay_frontiers(executor, trace_requests[qid], trace[qid]))
            report["frontiers"][family] = {
                "index": str(directory),
                "shared_index_disk_bytes": sum(p.stat().st_size for p in files),
                "index_files": [{"path": str(p), "bytes": p.stat().st_size} for p in files],
                "trace": str(trace_file),
                "profile": {
                    k: profile[k] for k in ("start_rank", "unquoted_starts", "committed_paths")
                },
                "questions": replay,
            }
            inputs.extend([*files, trace_file, profile_file, request_file])
            del executor, index, trace, trace_requests
        del tokenizer
        gc.collect()
    report["provenance"] = provenance(list(dict.fromkeys(inputs)))
    report["seconds"] = time.perf_counter() - started
    report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    report["limitations"] = [
        "sampled heap/payload sizes describe prefixes only; full trie bytes are null",
        "Python bytes exclude allocator overhead, temporaries, native memory and worker copies",
        "index disk bytes exclude runtime GPU tables, masks and per-sequence state",
        "frontier sizes are capped at MAX_REACHED; equality to the cap need not mean truncation",
        "frontiers replay model-selected plans, not every relation chain",
        "Different path languages: sizes do not establish matched quality or serving cost",
    ]
    if args.out.resolve() in {p.resolve() for p in inputs}:
        parser.error("--out must differ from input paths")
    write_json(args.out, report)
    print(json.dumps({"seconds": report["seconds"], "peak_rss_bytes": report["peak_rss_bytes"]}))


if __name__ == "__main__":
    main()
