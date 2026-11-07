"""Engine benchmark: real token streams replayed through the vLLM path, timed per step.

Streams (the token ids of constrained outputs, as xgraph.generate writes them) are
assigned to `batch` slots in turn and replayed one decoding step at a time as the
vLLM adapter drives the engine: `sync` advances every slot over its newest token,
read from a token history on the GPU, then `apply_mask` writes the masks into a
[batch, vocab] float32 logits tensor. CUDA events time both per step, after a short
warm-up. Timings are only meaningful on an idle GPU (XG_GPU_IDLE=1
scripts/gpu-headroom.sh).

    python -m xgraph.bench --index data/wikidata/gate-index \\
        --tokenizer data/tokenizers/qwen3.8/tokenizer.json \\
        --outputs target/gate/wwq-constrained.jsonl --batch 1 16 64 256 \\
        --vocab 248320 --out target/bench/engine.json

With `--kg` instead of `--outputs`, the streams are texts stating triples of a KG
directory (`kg_texts`), so that indexes of KGs no model was constrained to, such as
subsets of the gate KG (`python -m xgraph.wikidata subset`), can be measured. With
`--kg-versions <version file> --kg-version <name>`, every slot reads that KG version
(xgraph.kgversions), so that lookups through a version's tables are timed.

The engine is the fact-span engine with escapes by default. `--paths`, `--committed-paths`
and `--no-escapes` build the one path-mode evaluations serve (XGRAPH_PATHS=1,
XGRAPH_COMMITTED_PATHS=1, XGRAPH_ESCAPES=0, see xgraph.vllm), and `--open` starts every
slot inside a span, as a forced opener does (outputs of `python -m xgraph.generate --open`
hold the tokens after the opener); the report records the configuration (`engine`).
Requests' anchors and replayed prefills are not replayed: their streams start unbound.

`--compare-stop --questions <questions.json>` replays the same streams with STOP off
and with residual-goal STOP. Questions (id/utterance) are joined to output records by id.
Use outputs generated with STOP so both engines admit the streams; a replay error
rejects the comparison. `--stop residual-goal` runs only the STOP configuration.
`--start-rank` and `--unquoted-starts` match the evaluated start settings. Admission
(reset, begin, version, STOP compilation/copy, open) is measured separately as wall time,
including device completion; CUDA step times exclude it and engine construction.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

WARMUP = 3


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank quantile: no extrapolation beyond observed timings."""
    if not values or not 0 < fraction <= 1:
        raise ValueError("percentile needs samples and a fraction in (0, 1]")
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def replenished_schedule(
    lengths: list[int], *, batch: int, steps: int
) -> list[list[tuple[int, int, int]]]:
    """Fixed-duration replay, replacing each completed request on the next step.

    Entries are (slot, monotonically increasing request number, token offset).
    Request numbers index streams modulo their count. A replacement is admitted
    from a fresh state, including its pinned KG version. Every row stays active;
    requests still open at the horizon are reported as censored by the caller.
    """
    if not lengths or min([*lengths, batch, steps]) <= 0:
        raise ValueError("stream lengths, batch and steps must be positive")
    requests, positions, next_request = list(range(batch)), [0] * batch, batch
    result = []
    for _ in range(steps):
        row = []
        for slot in range(batch):
            if positions[slot] == lengths[requests[slot] % len(lengths)]:
                requests[slot], positions[slot] = next_request, 0
                next_request += 1
            row.append((slot, requests[slot], positions[slot]))
            positions[slot] += 1
        result.append(row)
    return result


def _step_times(
    engine,
    streams: list[list[int]],
    batch: int,
    vocab: int,
    version=None,
    opened=False,
    *,
    questions: list[str] | None = None,
    path_options: dict | None = None,
    relation_questions: list[str] | None = None,
    noop_mask: bool = False,
):
    """One replay: sync/mask event times after warm-up, and admission wall ms."""
    import torch

    width = engine.max_sequences  # vLLM keeps history and lengths for every slot
    rows = [streams[i % len(streams)] for i in range(batch)]
    steps = max(len(s) for s in rows)
    history = torch.zeros((width, steps), dtype=torch.int32, device="cuda")
    for slot, tokens in enumerate(rows):
        history[slot, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
    lengths = torch.zeros(width, dtype=torch.int32, device="cuda")
    lengths[:batch] = torch.tensor([len(t) for t in rows], dtype=torch.int32)
    slots = torch.arange(batch, dtype=torch.int32, device="cuda")
    prompt_len = torch.zeros(width, dtype=torch.int32, device="cuda")
    logits = torch.zeros((batch, vocab), dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()  # exclude history/logits allocation and copies from admission
    setup_started = time.perf_counter()
    engine.reset()
    engine.begin(slots)
    for slot in range(batch):  # after begin, which returns slots to the base KG
        engine.set_version(slot, version)
        if path_options is not None:
            engine.set_path_options(slot, path_options)
        if relation_questions is not None:
            engine.set_relation_question(slot, relation_questions[slot % len(streams)])
        if questions is not None:
            engine.set_stop(slot, questions[slot % len(streams)])
        if opened:  # after begin, which returns slots to free text
            engine.open_span(slot)
    torch.cuda.synchronize()
    setup_ms = (time.perf_counter() - setup_started) * 1000
    sync_ms, mask_ms = [], []
    for step in range(steps):
        total_len = torch.clamp(lengths, max=step + 1)
        events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        events[0].record()
        engine.sync(slots, history, prompt_len, total_len)
        events[1].record()
        if not noop_mask:
            engine.apply_mask(logits, slots)
        events[2].record()
        torch.cuda.synchronize()
        if step >= WARMUP or steps <= WARMUP:
            sync_ms.append(events[0].elapsed_time(events[1]))
            mask_ms.append(events[1].elapsed_time(events[2]))
    return sync_ms, mask_ms, setup_ms


def replay(
    engine,
    streams: list[list[int]],
    *,
    batch: int,
    vocab: int,
    repeats: int = 1,
    version: str | None = None,
    opened: bool = False,
    questions: list[str] | None = None,
    path_options: dict | None = None,
    relation_questions: list[str] | None = None,
    noop_mask: bool = False,
) -> dict:
    """Replay the streams in `batch` slots, `repeats` times; per-step ms. Every slot reads
    the KG version (None: the base KG), and with opened starts inside a span.
    With questions, install STOP for the question corresponding to each assigned stream
    once per replay, before opening spans. Admission wall time is separate from steps.

    `*_ms_per_step` keeps each step's fastest time over the repeats (interference only
    adds time, so the minimum estimates a quiet GPU); `*_ms_median` and `*_ms_p90` take
    every repeat's every step. `active_share` is the mean share of slots still writing
    over the timed steps: finished slots stay in the batch, as in vLLM until they are
    replaced, and a low share flatters the per-step times.
    """
    import statistics

    if not streams or any(not s for s in streams):
        raise ValueError("engine replay needs nonempty token streams")
    if batch <= 0 or vocab <= 0 or repeats <= 0:
        raise ValueError("batch, vocab and repeats must be positive")
    if questions is not None and len(questions) != len(streams):
        raise ValueError("STOP replay needs one question per token stream")
    if relation_questions is not None and len(relation_questions) != len(streams):
        raise ValueError("policy replay needs one relation question per token stream")
    runs = [
        _step_times(
            engine,
            streams,
            batch,
            vocab,
            version,
            opened,
            questions=questions,
            path_options=path_options,
            relation_questions=relation_questions,
            noop_mask=noop_mask,
        )
        for _ in range(repeats)
    ]
    sync = [min(r[0][i] for r in runs) for i in range(len(runs[0][0]))]
    mask = [min(r[1][i] for r in runs) for i in range(len(runs[0][1]))]
    every_sync = [ms for r in runs for ms in r[0]]
    every_mask = [ms for r in runs for ms in r[1]]
    lengths = [len(streams[i % len(streams)]) for i in range(batch)]
    steps = max(lengths)
    timed = range(WARMUP, steps) if steps > WARMUP else range(steps)
    active = [sum(n > step for n in lengths) / batch for step in timed]

    def p90(values: list[float]) -> float:
        return statistics.quantiles(values, n=10)[-1] if len(values) > 1 else values[0]

    return {
        "batch": batch,
        "assigned_streams": min(batch, len(streams)),
        "kg_version": version,
        "steps": steps,
        "repeats": repeats,
        "setup_ms_median": statistics.median(r[2] for r in runs),
        "setup_ms_per_request": statistics.median(r[2] for r in runs) / batch,
        "setup_ms_p95": percentile([r[2] for r in runs], 0.95),
        "sync_ms_per_step": sum(sync) / len(sync),
        "mask_ms_per_step": sum(mask) / len(mask),
        "sync_ms_median": statistics.median(every_sync),
        "mask_ms_median": statistics.median(every_mask),
        "sync_ms_p90": p90(every_sync),
        "mask_ms_p90": p90(every_mask),
        "sync_ms_p95": percentile(every_sync, 0.95),
        "mask_ms_p95": percentile(every_mask, 0.95),
        "raw_repetitions": [
            {"repetition": i, "sync_ms": r[0], "mask_ms": r[1], "setup_ms": r[2]}
            for i, r in enumerate(runs)
        ],
        "active_share": sum(active) / len(active),
        "errors": engine.errors(batch),
        "mask_control": "noop" if noop_mask else "engine",
    }


def wt_overhead(args):
    """Replay both WT calls with the served capabilities and question admission.

    The control omits only apply_mask; history advancement and admission stay
    identical. It measures the event/callback floor, not unconstrained LLM cost.
    """
    from xgraph.engine import Engine
    from xgraph.index import load_kg_index
    from xgraph.provenance import provenance
    from xgraph.tokenizer import ServedTokenizer

    trial = json.loads(args.wt_trial.read_text())
    study_path = Path(trial["study_path"])
    study = json.loads(study_path.read_text())
    arm = study["arms"][trial["trial"]["arm"]]
    if trial["status"] != "completed" or arm.get("catalog_policy", {}).get("policy") != "WT":
        raise ValueError("WT overhead needs a completed WT live-policy trial")
    rows = sorted(trial["results"], key=lambda r: r["id"])
    if len(rows) != len(study["panel_ids"]) or {r["id"] for r in rows} != set(study["panel_ids"]):
        raise ValueError("WT overhead needs the complete development timing panel")
    rows = rows[:8]  # short deterministic replay, identical in both controls
    tokenizer = ServedTokenizer.from_file(args.tokenizer)
    index = load_kg_index(args.index)
    inputs = [
        args.wt_trial,
        study_path,
        args.tokenizer,
        args.index / "kg-index.json",
        Path(__file__),
    ]
    eos = set()
    for name in ("config.json", "generation_config.json"):
        path = Path(arm.get("model", args.tokenizer.parent)) / name
        if path.is_file():
            config = json.loads(path.read_text())
            value = config.get("eos_token_id", config.get("text_config", {}).get("eos_token_id"))
            if value is not None:
                eos.update([value] if isinstance(value, int) else value)
            inputs.append(path)
    labels = Path(arm["options"]["relation_labels"])
    inputs.append(labels)
    settings = {
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
    results = []
    for stage in ("first", "second"):
        streams, questions = [], []
        for row in rows:
            matches = [c for c in row["calls"] if c["stage"] == stage]
            if len(matches) != 1:
                raise ValueError(f"{row['id']}: WT needs exactly one {stage} call")
            call = matches[0]
            if (
                call["status"] != "completed"
                or not call.get("token_ids")
                or call.get("opener") != index.span_syntax.opener
                or call["engine_arguments"]["xgraph_path_options"] != arm[f"{stage}_path_options"]
            ):
                raise ValueError(
                    f"{row['id']}: WT {stage} call profile or exact token stream changed"
                )
            streams.append(call["token_ids"])
            questions.append(call["engine_arguments"]["xgraph_question"])
        engine = Engine(
            index,
            tokenizer,
            index.span_syntax,
            max(args.batch),
            eos_token_ids=sorted(eos),
            relation_labels=labels,
            **settings,
        )
        for batch in args.batch:
            for noop in (False, True):
                result = replay(
                    engine,
                    streams,
                    batch=batch,
                    vocab=args.vocab,
                    repeats=args.repeats,
                    opened=True,
                    relation_questions=questions,
                    path_options=arm[f"{stage}_path_options"],
                    noop_mask=noop,
                )
                if result["errors"]:
                    raise ValueError("WT overhead replay rejected a generated token stream")
                results.append(
                    {**result, "stage": stage, "mask_control": "noop" if noop else "engine"}
                )
        del engine
    inputs.extend(sorted(args.index.glob("*")))
    native = Path(__file__).resolve().parents[2] / "csrc/engine.cu"
    if native.is_file():
        inputs.append(native)
    return {
        "artifact_kind": "xgraph.engine_bench",
        "version": 1,
        "profile": "WT",
        "engine": settings,
        "policy_sha256": arm.get("policy_sha256"),
        "sample_ids": [r["id"] for r in rows],
        "results": results,
        "control": "identical admission and sync; omit apply_mask only; no model generation",
        "provenance": provenance(list(dict.fromkeys(p for p in inputs if p.is_file()))),
    }


def kg_texts(kg: Path, n: int, seed: int, facts: int = 8) -> list[str]:
    """n texts, each stating `facts` triples of a KG directory drawn with the seed: a fact
    span and an entity span of its object per triple, with canonical labels. Streams for
    KGs that no model output was constrained to, such as subsets of the gate KG."""
    import numpy as np

    from xgraph.syntax import SpanSyntax

    syntax = SpanSyntax()
    triples = np.load(Path(kg) / "triples.npy", mmap_mode="r")
    relations: dict[int, str] = {}
    for line in (Path(kg) / "relation_labels.jsonl").open(encoding="utf-8"):
        prop, label, kind = json.loads(line)
        if kind == "label":
            relations.setdefault(int(prop[1:]), label)
    rng = np.random.default_rng(seed)
    drawn = np.asarray(triples[np.sort(rng.choice(len(triples), size=4 * n * facts))])
    drawn = drawn[np.isin(drawn[:, 1], list(relations))]
    wanted = set(drawn[:, 0].tolist()) | set(drawn[:, 2].tolist())
    names: dict[int, str] = {}
    for line in (Path(kg) / "entity_labels.jsonl").open(encoding="utf-8"):
        item, label, kind = json.loads(line)
        if kind == "label" and int(item[1:]) in wanted:
            names.setdefault(int(item[1:]), label)
    usable = [(s, r, o) for s, r, o in drawn.tolist() if s in names and o in names]
    usable = [usable[i] for i in rng.permutation(len(usable))]
    if len(usable) < n * facts:
        raise ValueError(f"{kg} has too few labelled triples for {n} texts of {facts} facts")

    def span(*labels: str) -> str:
        inner = syntax.separator.join(syntax.label_prefix + label for label in labels)
        return syntax.opener + inner + syntax.closer

    texts = []
    for i in range(n):
        parts = []
        for s, r, o in usable[i * facts : (i + 1) * facts]:
            parts.append(f"{span(names[s], relations[r], names[o])}, so {span(names[o])}.")
        texts.append(" ".join(parts))
    return texts


def main(argv: list[str] | None = None) -> None:
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "assay":
        from xgraph.mask_assay import main as assay_main

        assay_main(argv[1:])
        return

    from xgraph.engine import Engine
    from xgraph.index import load_kg_index
    from xgraph.provenance import provenance
    from xgraph.tokenizer import ServedTokenizer

    parser = argparse.ArgumentParser(prog="python -m xgraph.bench", description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="KG index directory")
    parser.add_argument("--tokenizer", type=Path, required=True, help="served tokenizer.json")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--outputs", type=Path, help="constrained outputs (JSONL)")
    source.add_argument("--kg", type=Path, help="KG directory: texts of its own triples")
    source.add_argument(
        "--wt-trial", type=Path, help="completed WT trial: both profiles and noop mask"
    )
    parser.add_argument("--texts", type=int, default=256, help="with --kg: how many texts")
    parser.add_argument("--facts", type=int, default=8, help="with --kg: triples per text")
    parser.add_argument("--seed", type=int, default=0, help="with --kg: the draw's seed")
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 16, 64, 256])
    parser.add_argument("--vocab", type=int, required=True, help="the model's logits width")
    parser.add_argument("--repeats", type=int, default=1, help="keep each step's fastest time")
    parser.add_argument("--out", type=Path, required=True, help="results (JSON)")
    parser.add_argument("--kg-versions", type=Path, help="a KG version file (xgraph.kgversions)")
    parser.add_argument("--kg-version", help="with --kg-versions: the version every slot reads")
    parser.add_argument("--paths", action="store_true", help="path mode (XGRAPH_PATHS=1)")
    parser.add_argument(
        "--committed-paths", action="store_true", help="with --paths: committed paths"
    )
    parser.add_argument(
        "--no-escapes", action="store_true", help="no escape marker (XGRAPH_ESCAPES=0)"
    )
    parser.add_argument(
        "--open", action="store_true", help="every slot starts inside a span (a forced opener)"
    )
    parser.add_argument("--start-rank", choices=("degree", "sitelinks"), default="degree")
    parser.add_argument("--unquoted-starts", action="store_true")
    stop = parser.add_mutually_exclusive_group()
    stop.add_argument("--stop", choices=("residual-goal",), help="enable online STOP")
    stop.add_argument(
        "--compare-stop", action="store_true", help="time STOP off and on using the same streams"
    )
    parser.add_argument("--questions", type=Path, help="STOP: id/utterance questions (JSON)")
    args = parser.parse_args(argv)
    if bool(args.kg_versions) != bool(args.kg_version):
        parser.error("--kg-versions and --kg-version go together")
    if args.committed_paths and not args.paths:
        parser.error("--committed-paths needs --paths")
    if args.stop or args.compare_stop:
        if not (args.paths and args.committed_paths and args.questions and args.outputs):
            parser.error("STOP needs --paths --committed-paths --questions and --outputs")
    elif args.questions:
        parser.error("--questions needs --stop or --compare-stop")
    if min([*args.batch, args.vocab, args.repeats]) <= 0:
        parser.error("--batch, --vocab and --repeats must be positive")
    if args.wt_trial:
        if args.stop or args.compare_stop or args.questions or args.kg_versions:
            parser.error("WT overhead cannot mix STOP or KG version overrides")
        report = wt_overhead(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")
        return
    settings = {
        "paths": args.paths,
        "committed_paths": args.committed_paths,
        "escapes": not args.no_escapes,
        "start_rank": args.start_rank,
        "unquoted_starts": args.unquoted_starts,
    }
    tokenizer = ServedTokenizer.from_file(args.tokenizer)
    index = load_kg_index(args.index)
    versions = None
    if args.kg_versions:
        from xgraph.kgversions import load_kg_versions

        versions = load_kg_versions(args.kg_versions, index)
    questions = None
    if args.outputs:
        with args.outputs.open() as source:
            records = [json.loads(line) for line in source if line.strip()]
        streams = [r["token_ids"] for r in records]
        inputs = [args.outputs]
        if args.questions:
            by_id = {q["id"]: q["utterance"] for q in json.loads(args.questions.read_text())}
            questions = []
            for record in records:
                question = by_id.get(record["id"])
                if not isinstance(question, str) or not question.strip():
                    raise ValueError(f"record {record['id']!r}: STOP needs its question text")
                questions.append(question)
            inputs.append(args.questions)
    else:
        streams = tokenizer.encode_batch(kg_texts(args.kg, args.texts, args.seed, args.facts))
        inputs = [args.kg / "manifest.json"] if (args.kg / "manifest.json").exists() else []
    if not streams or any(not s for s in streams):
        parser.error("the source must contain nonempty token streams")
    rules = [None, "residual-goal"] if args.compare_stop else [args.stop]
    results = []
    for rule in rules:
        engine = Engine(
            index,
            tokenizer,
            index.span_syntax,
            max(args.batch),
            versions=versions,
            stop=rule,
            **settings,
        )
        for batch in args.batch:
            result = replay(
                engine,
                streams,
                batch=batch,
                vocab=args.vocab,
                repeats=args.repeats,
                version=args.kg_version,
                opened=args.open,
                questions=questions if rule else None,
            )
            if args.compare_stop and result["errors"]:
                raise ValueError(
                    f"STOP comparison: {result['errors']} invalid streams with stop={rule!r}; "
                    "use STOP-enabled outputs admitted by both configurations"
                )
            results.append({**result, "stop": rule})
        del engine  # release the first engine's KG allocation before constructing the next
    inputs.extend([
        args.index / "kg-index.json", args.tokenizer, Path(__file__),
        Path(__file__).with_name("stopping.py"), Path(__file__).with_name("engine.py"),
    ])  # fmt: skip
    native_source = Path(__file__).resolve().parents[2] / "csrc/engine.cu"
    if native_source.is_file():  # a source checkout, rather than an installed wheel alone
        inputs.append(native_source)
    report = {
        "artifact_kind": "xgraph.engine_bench",
        "version": 1,
        "engine": {**settings, "open": args.open},
        "results": results,
        "stop_rules": rules,
        "questions": str(args.questions) if args.questions else None,
        "streams": {"kg": str(args.kg), "texts": args.texts, "facts": args.facts, "seed": args.seed}
        if args.kg
        else {"outputs": str(args.outputs)},
        "kg_versions": str(args.kg_versions) if args.kg_versions else None,
        "provenance": provenance(inputs),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    for r in results:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
