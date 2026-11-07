"""Bounded RoG-WebQSP preparation and scoring. See docs/reproducing.md, WebQSP.

Released strings are node identities, not recoverable Freebase MIDs. Never label
their set scores as native MID scores. Native scoring requires explicit answer_ids.
"""

from __future__ import annotations

import json
import re
import resource
import shutil
import time
import urllib.request
from collections import Counter, deque
from collections.abc import Iterable
from contextlib import ExitStack
from pathlib import Path

import numpy as np

ROG_REVISION = "c0632533135a06f8c5d536b420deec5fcb5c58f3"
GCR_REVISION = "9518e8e69197d5e577a6e414060fb898771617b9"
NATIVE_REVISION = "ab8ada1aafc0f785f5b55a74fb28c6d71b18da74"
MAX_DATA_BYTES = 15_000_000_000
_ROG = f"https://huggingface.co/datasets/rmanluo/RoG-webqsp/resolve/{ROG_REVISION}/"
_GCR = f"https://raw.githubusercontent.com/RManLuo/graph-constrained-reasoning/{GCR_REVISION}/"
_NATIVE = f"https://raw.githubusercontent.com/LHRLAB/ChatKBQA/{NATIVE_REVISION}/"
SOURCES = {
    "test-00000-of-00002-9ee8d68f7d951e1f.parquet": (
        _ROG + "data/test-00000-of-00002-9ee8d68f7d951e1f.parquet",
        90854720,
        "3e756abdcdf944e72fb7408ff670ff33d030c1a81413338478904e8b07c64235",
    ),
    "test-00001-of-00002-773a7b8213e159f5.parquet": (
        _ROG + "data/test-00001-of-00002-773a7b8213e159f5.parquet",
        93236638,
        "0f7b81b759a0c0a7e675f567b7b378687c849d5aeb75cd50409de80b1c5b0a83",
    ),
    "validation-00000-of-00001-6ee6adc5b154643a.parquet": (
        _ROG + "data/validation-00000-of-00001-6ee6adc5b154643a.parquet",
        24340437,
        "8355b852e5aae75197d4220031f88a6b7627dfd457383edf29c7f9b9a92a272d",
    ),
    "RoG-README.md": (
        _ROG + "README.md",
        900,
        "51e97c57b9856c302ad3cc1090cf405ef135172feeea316bf1ccd63510ae1250",
    ),
    "native-test.json": (
        _NATIVE + "data/WebQSP/origin/WebQSP.test.json",
        5140169,
        "856f50eb1ede43b799b10ddd0a980e72c88475b1a2259491af57e5db410ded52",
    ),
    "native-evaluate.py": (
        _NATIVE + "generation/webqsp_evaluate_offcial.py",
        5960,
        "bd54742c810e5fb2c219d7f381a63281f766cfa2ce1d79629b8504cd18c5eba3",
    ),
    "gcr-qa_utils.py": (
        _GCR + "src/utils/qa_utils.py",
        23741,
        "2e89e3e44502e676eaabc227c3894c26c106c62d978e2a2aa7e77dd81bf9552e",
    ),
    "gcr-graph_utils.py": (
        _GCR + "src/utils/graph_utils.py",
        5926,
        "92923f04a456328000f194dda87d1b473b2e706feffe2b1eec7b903830ecb837",
    ),
    "gcr-README.md": (
        _GCR + "README.md",
        4633,
        "388ac5ca0b87347f82f41d40dcb729344c6449d0a9c6644cf8d67c41c674dc35",
    ),
    "gcr-LICENSE": (
        _GCR + "LICENSE",
        1077,
        "d981694a61972fe3a9e082f4638c283d06a152ecf018ffe4ae2eaaeb6d524c74",
    ),
}


def data_bytes(*roots: Path) -> int:
    """Logical bytes of data artifacts, counting each resolved path once."""
    files = {p.resolve() for root in roots if root.exists() for p in root.rglob("*") if p.is_file()}
    total = 0
    for path in files:
        try:
            total += path.stat().st_size
        except FileNotFoundError:
            # A running worker can finish and unlink a temporary file between
            # enumeration and this snapshot. It no longer consumes disk.
            continue
    return total


def check_budget(*roots: Path, reserve: int = 0) -> None:
    used = data_bytes(*roots)
    if used + reserve > MAX_DATA_BYTES:
        raise ValueError(f"WebQSP data budget exceeded: {used} + {reserve} > {MAX_DATA_BYTES}")


def fetch(out: Path, cache: Path | None = None) -> dict:
    """Only pinned test/validation shards and small audited references, no remote code import."""
    from xgraph.provenance import provenance

    out.mkdir(parents=True, exist_ok=True)
    check_budget(out, reserve=sum(spec[1] for spec in SOURCES.values()))
    if shutil.disk_usage(out).free < sum(spec[1] for spec in SOURCES.values()) + 1_000_000_000:
        raise ValueError("insufficient free disk for bounded WebQSP fetch")
    fetch_files(out, SOURCES, cache=cache)
    report = {
        "rog_revision": ROG_REVISION,
        "gcr_revision": GCR_REVISION,
        "native_annotation_mirror_revision": NATIVE_REVISION,
        "licenses": {
            "RoG_data": "NOT DECLARED in pinned dataset card; evaluation only, no redistribution",
            "native_annotations": "WebQSP data via a pinned mirror; upstream data terms unresolved",
            "GCR_code": "MIT; gcr-LICENSE retained; does not license RoG/WebQSP data",
        },
        "files": {
            name: {"url": url, "bytes": n, "sha256": sha} for name, (url, n, sha) in SOURCES.items()
        },
        "provenance": provenance([out / name for name in SOURCES]),
    }
    _json(out / "sources.json", report)
    return report


def fetch_files(out: Path, sources: dict, *, cache: Path | None = None) -> None:
    """Stream pinned bytes as data. Callers own directory and resource admission."""
    from xgraph.provenance import sha256_file

    for name, (url, size, digest) in sources.items():
        target = out / name
        if target.is_file() and target.stat().st_size == size and sha256_file(target) == digest:
            continue
        part = target.with_suffix(target.suffix + ".part")
        candidate = cache / name if cache is not None else None
        if candidate is not None and candidate.is_file():
            shutil.copyfile(candidate, part)
        else:
            with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as file:
                copied = 0
                while block := response.read(1 << 20):
                    copied += len(block)
                    if copied > size:
                        raise ValueError(f"{url}: download exceeds pinned size {size}")
                    file.write(block)
        if part.stat().st_size != size or sha256_file(part) != digest:
            raise ValueError(f"{name}: pinned size/SHA-256 mismatch in {part}")
        part.replace(target)


def shard_paths(source: Path, split: str, *, sources=None) -> list[Path]:
    """A complete pinned split; never silently prepare a partial shard population."""
    sources = SOURCES if sources is None else sources
    paths = [
        Path(source) / name
        for name in sorted(sources)
        if name.startswith(split + "-") and name.endswith(".parquet")
    ]
    if not paths:
        raise ValueError(f"no pinned shards for split {split!r}")
    if missing := [str(p) for p in paths if not p.is_file()]:
        raise ValueError(f"missing released shard(s): {missing}")
    return paths


def parquet_rows(source: Path, split: str, *, sources=None, columns=None):
    """Arrow batches of one question; no datasets cache or full-split materialization."""
    import pyarrow.parquet as pq

    for path in shard_paths(source, split, sources=sources):
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=1, columns=columns, use_threads=False
        ):
            yield from batch.to_pylist()


def set_scores(prediction: list[str], gold: list[str]) -> dict:
    """Exact set metrics, with prediction order used only for ranked Hits@1."""
    predicted, wanted = set(prediction), set(gold)
    correct = len(predicted & wanted)
    precision = correct / len(predicted) if predicted else 1.0
    recall = correct / len(wanted) if wanted else 1.0
    f1 = 2 * correct / (len(predicted) + len(wanted)) if predicted or wanted else 1.0
    return {
        "hits_at_1": int(bool(prediction) and prediction[0] in wanted),
        "any_hit": int(correct > 0),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_set": int(predicted == wanted),
    }


def gcr_scores(prediction: str | list[str], gold: list[str]) -> dict:
    """GCR qa_utils.py eval_result protocol, distinct from exact set metrics.

    Frequencies rank duplicate lines; containment uses the released normalization.
    Recall matches against the joined lines, precision against each individual line.
    """
    from xgraph.twowiki import rog_normalize

    lines = prediction.split("\n") if isinstance(prediction, str) else prediction
    counts = Counter(p for p in lines if p.strip())
    lines = sorted(counts, key=lambda p: -counts[p])  # stable ties, like upstream
    match = lambda p, a: rog_normalize(a) in rog_normalize(p)  # noqa: E731
    hit = int(any(match(" ".join(lines), a) for a in gold))
    if not lines or not gold:
        return {"gcr_hit": hit, "gcr_f1": 0, "gcr_precision": 0, "gcr_recall": 0}
    precision = sum(any(match(p, a) for a in gold) for p in lines) / len(lines)
    recall = sum(match(" ".join(lines), a) for a in gold) / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"gcr_hit": hit, "gcr_f1": f1, "gcr_precision": precision, "gcr_recall": recall}


def _by_id(rows: list[dict], key: str = "id") -> dict[str, dict]:
    found = {}
    for row in rows:
        qid = row[key]
        if qid in found:
            raise ValueError(f"duplicate question/output id {qid!r}")
        found[qid] = row
    return found


def _checked_outputs(records: list[dict], ids: set[str]) -> dict[str, dict]:
    outputs = _by_id(records)
    if extra := outputs.keys() - ids:
        raise ValueError(f"unknown output ids: {sorted(extra)}")
    return outputs


def answers_of(record: dict) -> list[str]:
    """Explicit sets, released prediction lines, or the last Answer line's spans/JSON.

    Free rationale and intermediate paths never become predictions. Commas inside
    names are preserved. Unknown and escaped spans count as an abstention.
    """
    from xgraph.score import replay_failure

    if replay_failure(record):
        return []
    from xgraph.agent_answers import recorded_answers

    versioned = recorded_answers(record)
    if versioned is not None:
        values = versioned
    elif "answers" in record or "prediction" in record:
        values = record.get("answers", record.get("prediction"))
        if isinstance(values, str):
            values = values.split("\n")
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError("answers/prediction must be a string or list of strings")
    else:
        from xgraph.prompts import parse_answer_set

        values = parse_answer_set(record.get("text", ""))
    return list(
        dict.fromkeys(
            unrender_label(v.strip())
            for v in values
            if v.strip() and v.strip().casefold() != "unknown" and not v.strip().startswith("⁇")
        )
    )


def _average(rows: list[dict], metrics: list[str]) -> dict:
    return {key: sum(row[key] for row in rows) / len(rows) if rows else None for key in metrics}


def native_scores(records: list[dict], questions: list[dict]) -> dict:
    """Native evaluator's good/complete eligibility and best-parse set F1.

    The released script calls any-hit 'Hits@1'. We expose that as any_hit and
    additionally report strict ranked hits_at_1. Identity comes only from answer_ids.
    """
    native = _by_id(questions, "QuestionId")
    outputs = _checked_outputs(records, set(native))
    details, skipped = [], []
    for qid, question in native.items():
        parses = question["Parses"]
        if not any(
            p["AnnotatorComment"].get("QuestionQuality") == "Good"
            and p["AnnotatorComment"].get("ParseQuality") == "Complete"
            for p in parses
        ):
            skipped.append(qid)
            continue
        record = outputs.get(qid)
        if record is not None and (
            not isinstance(record.get("answer_ids"), list)
            or any(not isinstance(v, str) for v in record["answer_ids"])
        ):
            raise ValueError(
                f"{qid}: native scoring needs explicit answer_ids; labels are not MIDs"
            )
        predictions = record["answer_ids"] if record is not None else []
        scores = [
            set_scores(predictions, [a["AnswerArgument"] for a in p["Answers"]]) for p in parses
        ]
        best = dict(max(scores, key=lambda s: s["f1"]))
        # The native script returns hit=1 for a parse with no labelled answers.
        best["any_hit"] = max(
            s["any_hit"] if p["Answers"] else 1 for s, p in zip(scores, parses, strict=True)
        )
        best["hits_at_1"] = max(s["hits_at_1"] for s in scores)
        if record is None:
            best = dict.fromkeys(best, 0.0)
        details.append({"id": qid, **best})
    return {
        "protocol": "native_mid_best_parse",
        "input_questions": len(native),
        "questions": len(details),
        "official_ineligible_ids": skipped,
        "missing_outputs": sum(row["id"] not in outputs for row in details),
        **_average(details, list(set_scores([], []))),
        "per_question": details,
    }


def score_records(
    records: list[dict],
    panel: list[dict],
    *,
    released_only: bool = False,
    failed_zero: bool = False,
) -> dict:
    """All declared questions, missing predictions scored zero, no coverage filtering."""
    from xgraph.oracle import MAX_REACHED
    from xgraph.score import record_failure, replay_failure

    questions = _by_id(panel)
    outputs = _checked_outputs(records, set(questions))
    excluded = [q["id"] for q in panel if released_only and q.get("missing_subgraph")]
    questions = {qid: q for qid, q in questions.items() if qid not in excluded}
    outputs = {qid: row for qid, row in outputs.items() if qid not in excluded}
    details, singletons = [], []
    for qid, question in questions.items():
        predicted = answers_of(outputs[qid]) if qid in outputs else []
        source = outputs.get(qid, {})
        missing = qid not in outputs or source.get("delivery", {}).get("missing_output", False)
        failed = replay_failure(source) or (
            failed_zero
            and (
                record_failure(source, require_text=False)
                or question.get("missing_subgraph", False)
                or source.get("delivery", {}).get("no_path", False)
            )
        )
        score = set_scores(predicted, question["answers"])
        score.update(gcr_scores(predicted, question["answers"]))
        if missing or failed:
            score = dict.fromkeys(score, 0.0)
        events = list(source.get("delivery", {}).get("cap_events", []))
        events.extend(source.get("cap_events", []))
        if source.get("trie", {}).get("truncated"):
            events.append({"stage": "gcr_paths", "truncated": True})
        gold_count = len(set(question["answers"]))
        details.append(
            {
                "id": qid,
                "prediction": predicted,
                **score,
                "gold_count": gold_count,
                "gold_over_cap": gold_count > MAX_REACHED,
                "cap_encountered": bool(events)
                or source.get("delivery", {}).get("cap_encountered", False),
                "cap_events": events,
                "missing_output": missing,
                "truncated": source.get("truncated", False)
                or source.get("finish_reason") == "length",
                "budget_spent": source.get("budget_spent", False),
                **({"failed_output": bool(failed)} if failed_zero else {}),
            }
        )
        raw = source.get("delivery", {}).get("raw_answer")
        singleton = (answers_of({"answers": raw}) if raw is not None else predicted)[:1]
        single_score = {
            **set_scores(singleton, question["answers"]),
            **gcr_scores(singleton, question["answers"]),
        }
        if missing or failed:
            single_score = dict.fromkeys(single_score, 0.0)
        singletons.append({"id": qid, "prediction": singleton, **single_score})
    metrics = [*set_scores([], []), *gcr_scores([], [])]
    bounded = [row for row in details if not row["gold_over_cap"]]
    return {
        "protocol": "RoG/GCR label protocol on released subgraphs",
        "panel": "released_subgraphs"
        if released_only
        else "declared_panel_including_native_omissions",
        "primary_metrics": ["gcr_hit", "gcr_f1"],
        "secondary_metrics": ["hits_at_1", "f1", "precision", "recall", "exact_set"],
        "questions": len(questions),
        "excluded_missing_subgraph_ids": excluded,
        "coverage_counts": {
            key: sum(bool(q.get("coverage", {}).get(key)) for q in questions.values())
            for key in ("present", "reachable_2", "reachable_8", "any_reachable", "all_reachable")
        },
        "coverage_denominator": len(questions),
        "missing_outputs": sum(row["missing_output"] for row in details),
        **({"failed_outputs": sum(row["failed_output"] for row in details)} if failed_zero else {}),
        **_average(details, metrics),
        "caps": {
            "engine_frontier_cap": MAX_REACHED,
            "encountered_ids": [r["id"] for r in details if r["cap_encountered"]],
            "gold_over_cap_ids": [r["id"] for r in details if r["gold_over_cap"]],
        },
        "secondary": {
            "without_gold_over_cap": {
                "questions": len(bounded),
                "excluded_ids": [r["id"] for r in details if r["gold_over_cap"]],
                "missing_outputs": sum(r["missing_output"] for r in bounded),
                **_average(bounded, metrics),
            },
            "singleton": {
                "questions": len(singletons),
                **_average(singletons, metrics),
                "per_question": singletons,
            },
        },
        "per_question": details,
    }


def render_label(text: str) -> str:
    """Preserve names, escaping reserved span/trie syntax reversibly as literal text."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"empty or non-string Freebase node/relation: {text!r}")
    text = text.strip().replace("\\", "\\\\")
    for char in "⟦⟧⁇·|<>\n\r":
        text = text.replace(char, f"\\u{ord(char):04x}")
    if text.startswith(("earlier:", "later:", "same:")):
        text = "\\u" + f"{ord(text[0]):04x}" + text[1:]
    return text


def unrender_label(text: str) -> str:
    """Undo the bridge's displayed syntax escapes, leaving ordinary Unicode alone."""
    return re.sub(r"\\\\|\\u([0-9a-fA-F]{4})", lambda m: chr(int(m[1], 16)) if m[1] else "\\", text)


def graph_coverage(row: dict) -> dict:
    """Exact directed reachability on the supplied graph; CVT edges are physical hops."""
    adjacency: dict[str, set[str]] = {}
    nodes = set()
    for s, _, o in row["graph"]:
        nodes.update((s, o))
        adjacency.setdefault(s, set()).add(o)
    distance = {s: 0 for s in row["q_entity"] if s in nodes}
    pending = deque(distance)
    while pending:
        node = pending.popleft()
        for other in adjacency.get(node, ()):
            if other not in distance:
                distance[other] = distance[node] + 1
                pending.append(other)
    answers = list(dict.fromkeys(row["answer"]))
    reachable = [a for a in answers if a in distance]
    return {
        "gold_count": len(answers),
        "present": [a for a in answers if a in nodes],
        "reachable_2": [a for a in answers if distance.get(a, float("inf")) <= 2],
        "reachable_8": [a for a in answers if distance.get(a, float("inf")) <= 8],
        "reachable": reachable,
        "any_reachable": bool(reachable),
        "all_reachable": bool(answers) and len(reachable) == len(answers),
        "shortest_hops": {a: distance[a] for a in reachable},
    }


PATH_PROMPT = """Answer using the supplied Freebase question entities and exact relation ids.
Write a relation path as ⟦ start entity · relation.id · next.relation.id⟧, then
Answer: ⟦ label⟧. A path contains only its starting entity and relation ids.
CVT/mediator nodes remain in the graph: traversing a mediator requires both relations
and counts as two physical hops. Follow at most eight physical hops. The KG executes
your path. No inverse edges have been added; use the directed relations present.
For example, a synthetic film's actors follow
⟦ Example Film · film.film.starring · film.performance.actor⟧.
The complete reached answer set is delivered separately by the matched CPU executor.
Do not infer that a question has only one answer."""


def _requests(question: dict, starts: list[str], kind: str) -> dict:
    from xgraph.iterative import INSTRUCTIONS
    from xgraph.prompts import SET_FORMAT, SET_INSTRUCTION

    instructions = PATH_PROMPT
    if kind == "iterative":
        instructions = INSTRUCTIONS.replace("Wikidata", "Freebase").replace(
            'End with the line "Answer: ⟦ label⟧", naming one entity.', SET_FORMAT
        )
    elif kind == "gcr":
        instructions = "Answer from the supplied Freebase question entities."
    instructions += "\n" + SET_INSTRUCTION
    if kind in ("agent", "gcr"):
        instructions += "\n" + SET_FORMAT
    asked = question["utterance"]
    if kind != "gcr":
        asked += "\nQuestion entities: " + json.dumps(starts, ensure_ascii=False)
    return {
        "id": question["id"],
        "answer_mode": "set",
        "question": question["utterance"],
        "anchor": question["anchor"],
        "topics": question["anchor"],
        "kg_version": question["kg_version"],
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": asked},
        ],
    }


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def _line(file, value) -> None:
    file.write(json.dumps(value, ensure_ascii=False) + "\n")


def _finish_triples(raw: Path, count: int) -> None:
    target = np.lib.format.open_memmap(
        raw.with_suffix(".npy"),
        mode="w+",
        dtype=np.int32,
        shape=(count, 3),
    )
    if count:
        source = np.memmap(raw, mode="r", dtype=np.int32, shape=(count, 3))
        for at in range(0, count, 100_000):
            target[at : at + 100_000] = source[at : at + 100_000]
        del source
    target.flush()
    del target
    raw.unlink()


def convert(
    rows: Iterable[dict],
    out: str | Path,
    *,
    native: list[dict] | None = None,
    dataset: str = "webqsp",
    gold: bool = True,
) -> dict:
    """Stream question components into the existing numeric KG-directory format.

    Internal Q/P ids are compatibility numbers, never Wikidata identities. The
    identities table retains each question's released node string. Labels supplied
    by RoG are used as-is except reversible syntax escaping; opaque MIDs stay named
    by their own MID so CVT edges remain writable. Gold-only entities are never added.
    """
    from xgraph.kgversions import write_kg_versions
    from xgraph.oracle import MAX_REACHED

    if dataset not in {"webqsp", "cwq"}:
        raise ValueError(f"unknown released-subgraph dataset {dataset!r}")
    if native is not None and (dataset != "webqsp" or not gold):
        raise ValueError("native WebQSP annotations require WebQSP gold conversion")
    if dataset == "cwq" and gold:
        raise ValueError("CWQ preparation is gold-free; answers require the separate scorer")
    started = time.monotonic()
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"build output must be empty: {out}")
    kg, gcr_kg = out / "kg", out / "kg-gcr"
    kg.mkdir(parents=True, exist_ok=True)
    gcr_kg.mkdir()
    counts: Counter[str] = Counter()
    relations: dict[str, int] = {}
    components, questions, seen = {}, [], set()
    next_entity = 1
    native_by_id = _by_id(native, "QuestionId") if native is not None else {}

    def with_missing():
        yield from rows
        if native is not None:
            for qid, q in native_by_id.items():
                if qid not in seen:
                    if any(p["Answers"] for p in q["Parses"]):
                        raise ValueError(f"{qid}: missing nonempty-gold native question in RoG")
                    yield {
                        "id": qid,
                        "question": q["RawQuestion"],
                        "answer": [],
                        "q_entity": [f"Unavailable graph for {qid}"],
                        "graph": [],
                        "missing_subgraph": True,
                    }

    with ExitStack() as stack:
        entities = stack.enter_context((kg / "entity_labels.jsonl").open("w", encoding="utf-8"))
        identities = stack.enter_context((kg / "identities.jsonl").open("w", encoding="utf-8"))
        raw = stack.enter_context((kg / "triples.raw").open("wb"))
        gcr_raw = stack.enter_context((gcr_kg / "triples.raw").open("wb"))
        requests = {
            kind: stack.enter_context((out / f"requests-{kind}.jsonl").open("w", encoding="utf-8"))
            for kind in ("paths", "agent", "gcr", "iterative")
        }
        for row in with_missing():
            qid = row["id"]
            if qid in seen:
                raise ValueError(f"duplicate RoG question id {qid!r}")
            if native is not None and qid not in native_by_id:
                raise ValueError(f"unknown native question id {qid!r}")
            seen.add(qid)
            graph = []
            for triple in row["graph"]:
                if (
                    not isinstance(triple, (list, tuple))
                    or len(triple) != 3
                    or any(not isinstance(v, str) or not v.strip() for v in triple)
                ):
                    raise ValueError(f"{qid}: invalid triple {triple!r}")
                graph.append(tuple(v.strip() for v in triple))
            topics = list(dict.fromkeys(s.strip() for s in row["q_entity"]))
            if not topics:
                # Isolated sentinel prevents an empty anchor from opening the union KG.
                topics = [f"Unavailable start for {qid}"]
                counts["missing_starts"] += 1
            nodes = sorted({v for s, _, o in graph for v in (s, o)} | set(topics))
            mapping = {node: next_entity + i for i, node in enumerate(nodes)}
            first, stop = next_entity, next_entity + len(nodes)
            if stop >= np.iinfo(np.int32).max:
                raise ValueError("WebQSP component ids exceed int32")
            next_entity = stop
            version = f"q{len(questions)}"
            components[version] = [first, stop]
            for node, number in mapping.items():
                rendered = render_label(node)
                _line(entities, [f"Q{number}", rendered, "label"])
                _line(identities, [f"Q{number}", qid, node, rendered])
                counts["escaped_labels"] += rendered != node
                counts["opaque_mid_nodes"] += bool(re.fullmatch(r"[mg]\.[\w]+", node))
            triples, collapsed = set(), {}
            for s, relation, o in graph:
                render_label(relation)  # fail early on blank labels
                prop = relations.setdefault(relation, len(relations) + 1)
                triple = (mapping[s], prop, mapping[o])
                triples.add(triple)
                collapsed[(mapping[s], mapping[o])] = prop  # upstream DiGraph last wins
            np.asarray(sorted(triples), dtype=np.int32).reshape(-1, 3).tofile(raw)
            gcr_triples = sorted((s, p, o) for (s, o), p in collapsed.items())
            np.asarray(gcr_triples, dtype=np.int32).reshape(-1, 3).tofile(gcr_raw)
            question = {
                "id": qid,
                "utterance": row["question"],
                "sparql": "# question entities: " + " ".join(f"wd:Q{mapping[s]}" for s in topics),
                "anchor": [f"Q{mapping[s]}" for s in topics],
                "kg_version": version,
                "missing_subgraph": bool(row.get("missing_subgraph")),
                "complexity": "Freebase directed physical hops; CVTs retained",
            }
            if gold:
                question.update(answer_fields({**row, "graph": graph, "q_entity": topics}))
            questions.append(question)
            for kind, file in requests.items():
                _line(file, _requests(question, [render_label(s) for s in topics], kind))
            counts.update(
                questions=1,
                entities=len(nodes),
                triples=len(triples),
                source_triples=len(graph),
                gcr_triples=len(gcr_triples),
                parallel_relations_lost=len(triples) - len(gcr_triples),
                missing_subgraphs=bool(row.get("missing_subgraph")),
            )
            if gold:
                counts.update(
                    multiple_answers=len(question["answers"]) > 1,
                    gold_sets_over_engine_cap=len(question["answers"]) > MAX_REACHED,
                )
            if counts["questions"] % 100 == 0:
                check_budget(out, reserve=100_000_000)
    with (kg / "relation_labels.jsonl").open("w", encoding="utf-8") as file:
        for label, number in relations.items():
            _line(file, [f"P{number}", render_label(label), "label"])
    _finish_triples(kg / "triples.raw", counts["triples"])
    _finish_triples(gcr_kg / "triples.raw", counts["gcr_triples"])
    for name in ("entity_labels.jsonl", "relation_labels.jsonl"):
        (gcr_kg / name).symlink_to(Path("..") / "kg" / name)
    write_kg_versions(
        out / "versions",
        {name: np.zeros((0, 3), np.int32) for name in components},
        components=components,
    )
    _json(out / "questions.json", questions)
    _json(out / "questions-released.json", [q for q in questions if not q["missing_subgraph"]])
    if native is not None:
        _json(out / "gold-native.json", {"Questions": native})
    counts["relations"] = len(relations)
    report = {
        "artifact_kind": "xgraph.webqsp",
        "dataset": dataset,
        "gold_loaded": gold,
        "counts": dict(counts),
        "cvt_policy": "retained; two physical edges through a CVT count as two hops",
        "inverse_policy": "only supplied directed edges; no inverse augmentation",
        "identity": "question-local released strings; native MIDs cannot be reconstructed",
        "gcr_graph": "same source; exact input-order DiGraph last-edge collapse in kg-gcr",
        "seconds": time.monotonic() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    if gold:
        report.update(
            coverage_counts={
                key: sum(bool(q["coverage"][key]) for q in questions)
                for key in (
                    "present",
                    "reachable_2",
                    "reachable_8",
                    "any_reachable",
                    "all_reachable",
                )
            },
            coverage_denominator=len(questions),
        )
    _json(out / "manifest.json", report)
    return report


def answer_fields(row: dict, *, coverage: bool = True) -> dict:
    """Scoring-only extraction, kept separate from question/graph preparation."""
    result = {"answers": list(dict.fromkeys(row["answer"]))}
    if coverage:
        result["coverage"] = graph_coverage(row)
    return result


def deliver_records(
    records: list[dict],
    requests: list[dict],
    executor,
    store,
    versions,
    *,
    complete_path_set: bool = False,
) -> list[dict]:
    """Whole matched frontier as a set, with an uncapped CPU sensitivity beside it.

    This is a distinct CPU delivery arm; raw generated answers remain separately
    scoreable. It exposes cap loss rather than pretending to repair online traversal.
    """
    from xgraph.programs import record_path, set_execution_fields
    from xgraph.score import record_failure
    from xgraph.toolkg import store_for

    asked = _by_id(requests)
    outputs = _checked_outputs(records, set(asked))
    delivered = []
    for qid, request in asked.items():
        source = outputs.get(qid, {"id": qid, "text": ""})
        replay = source
        terminal = None
        if complete_path_set:
            written = record_path(source)
            if written and not record_failure(source, require_text=False):
                start, relations = written
                # Validate the original trace first, including audited edits. Keep
                # intermediate selections: expanding them would change the plan.
                executor.execute_set(
                    start, relations, request.get("anchor"), request.get("kg_version")
                )
                selections = dict(getattr(relations, "selections", {}))
                terminal = selections.pop(len(relations), None) if relations else None
                replay = {
                    **source,
                    "plan": {
                        "start": start,
                        "relations": list(relations),
                        "selections": selections,
                        "defers": list(getattr(relations, "defers", ())),
                        "decisions": getattr(relations, "decisions", {}),
                    },
                }
            elif record_failure(source, require_text=False):
                replay = {"id": qid, "text": ""}
        fields = set_execution_fields(
            replay, request, executor, store_for(request, store, versions)
        )
        if complete_path_set:
            fields["delivery"].update(rule="committed_path_set", terminal_selection=terminal)
        for key in ("answers", "uncapped_answers"):
            fields[key] = [unrender_label(v) for v in fields[key]]
        fields["delivery"]["missing_output"] = qid not in outputs
        delivered.append({**source, "id": qid, "kg_version": request["kg_version"], **fields})
    return delivered


def reference_audit(source: Path, predictions: Path | None = None) -> dict:
    """Compare audited pure metric functions from SHA-verified reference scripts.

    Only these functions are evaluated, in an isolated namespace: the source file's
    imports, CLI and IO never execute. Missing released predictions stay explicit.
    """
    import ast
    import random
    import string

    from xgraph.provenance import sha256_file

    def reference(name, functions):
        path = source / name
        if sha256_file(path) != SOURCES[name][2]:
            raise ValueError(f"reference script SHA-256 mismatch: {path}")
        parsed = ast.parse(path.read_text())
        selected = [
            n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in functions
        ]
        if {n.name for n in selected} != set(functions):
            raise ValueError(f"missing reference metric functions: {path}")
        namespace = {"re": re, "string": string}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
        return namespace

    gcr = reference(
        "gcr-qa_utils.py", ["normalize", "match", "eval_f1", "eval_hit", "extract_topk_prediction"]
    )
    native = reference("native-evaluate.py", ["FindInList", "CalculatePRF1"])
    rng = random.Random(0)
    pool = ["Alice", "Bob", "New York", "York", "The Beatles", "Beatles", "wrong", "m.a"]
    for _ in range(300):
        predicted = rng.choices(pool, k=rng.randrange(6))
        gold = rng.sample(pool, k=rng.randrange(1, 5))
        ranked = gcr["extract_topk_prediction"](predicted)
        expected = gcr["eval_f1"](ranked, gold)
        actual = gcr_scores(predicted, gold)
        assert np.allclose(
            expected, [actual["gcr_f1"], actual["gcr_precision"], actual["gcr_recall"]]
        )
        assert actual["gcr_hit"] == gcr["eval_hit"](" ".join(ranked), gold)
        # Native list scorer expects unique entity ids; our set contract enforces that.
        unique = list(dict.fromkeys(predicted))
        expected_native = native["CalculatePRF1"](
            [{"AnswerArgument": a} for a in gold],
            unique,
        )
        actual_native = set_scores(unique, gold)
        assert np.allclose(
            expected_native[:3], [actual_native[k] for k in ("precision", "recall", "f1")]
        )
    count = 0
    if predictions is not None:
        for row in _read_jsonl(predictions):
            gold = row.get("ground_truth", row.get("answer"))
            if gold is None:
                raise ValueError("released prediction needs ground_truth or answer")
            ranked = gcr["extract_topk_prediction"](row["prediction"])
            expected = gcr["eval_f1"](ranked, gold)
            actual = gcr_scores(row["prediction"], gold)
            assert np.allclose(
                expected, [actual[k] for k in ("gcr_f1", "gcr_precision", "gcr_recall")]
            )
            count += 1
    return {
        "seed": 0,
        "synthetic_cases": 300,
        "metric_function_parity": True,
        "released_prediction_records": count,
        "released_predictions_available": predictions is not None,
        "prediction_limitation": None
        if predictions
        else (
            "No predictions in the pinned GCR repository tree or linked from its README. "
            "Only reference-function parity on synthetic examples was checked."
        ),
    }


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


# Synthetic teaching examples, fixed before any WebQSP generation. No benchmark
# question, gold answer, or model output was consulted to choose these examples.
CATALOG_DEMOS = [
    ("Who directed Example Film?", "Example Film", ["film.film.directed_by"],
     ["Example Director"]),
    ("Who acted in Example Film?", "Example Film",
     ["film.film.starring", "film.performance.actor"], ["Example Actor", "Other Actor"]),
    ("Where was the director of Example Film born?", "Example Film",
     ["film.film.directed_by", "people.person.place_of_birth"], ["Example City"]),
    ("What is the capital of the country of Example City?", "Example City",
     ["location.location.containedby", "location.country.capital"], ["Example Capital"]),
    ("Which country contains the birthplace of the director of Example Sequel?",
     "Example Sequel", ["film.film.directed_by", "people.person.place_of_birth",
                        "location.location.containedby"], ["Example Republic"]),
]  # fmt: skip


def catalog_requests(question, starts, *, demos=4, observed=False):
    """Freebase demonstrations owned by this benchmark, never replaced by Wikidata demos."""
    request = _requests(question, starts, "paths")
    messages = [request["messages"][0]]
    if observed:
        from xgraph.prompts import OBSERVATION_INSTRUCTIONS

        messages[0] = {
            **messages[0],
            "content": messages[0]["content"]
            + "\n"
            + OBSERVATION_INSTRUCTIONS.format(k=3)
            + '\nThe graph suggests up to 5 relations in " ⟦?" menus. '
            'At a multi-entity boundary choose a label with " ⟦= label⟧". '
            "Selections narrow the path. All terminal endpoints of the committed "
            "path are delivered as a set; the final label only ranks that set.",
        }
    for asked, start, relations, answers in CATALOG_DEMOS[:demos]:
        # Teaching observations are synthetic, not claims about a served KG.
        body = start
        for hop, relation in enumerate(relations):
            if observed:
                body += f" ·→ {start if hop == 0 else 'Example Node ' + str(hop)}"
                body += f" ⟦? {relation}⟧"
            body += " · " + relation
        if observed:
            body += " ·→ " + " ; ".join(answers[:3]) + " ⟦? ∅⟧"
            if len(answers) > 1:
                body += f" ⟦= {answers[0]} ·→ {answers[0]} ⟦? ∅⟧"
        messages.extend(
            [
                {"role": "user", "content": asked},
                {"role": "assistant", "content": f"⟦ {body}⟧.\nAnswer: ⟦ {answers[0]}⟧"},
            ]
        )
    return {**request, "messages": [*messages, request["messages"][-1]], "webqsp_demos": demos}


def catalog_request_fingerprint() -> str:
    import hashlib
    import inspect

    return hashlib.sha256(
        (
            json.dumps(CATALOG_DEMOS, ensure_ascii=False)
            + PATH_PROMPT
            + inspect.getsource(_requests)
            + inspect.getsource(catalog_requests)
        ).encode()
    ).hexdigest()


def verify_catalog(data: Path) -> dict:
    """Refuse changed input, tokenizer, or index bytes, including reused symlink targets."""
    from xgraph.provenance import sha256_file

    saved = json.loads((data / "catalog.json").read_text())
    if saved.get("request_builder_sha256") != catalog_request_fingerprint():
        raise ValueError(
            "catalog request definitions changed; preserve inputs and prepare a new directory"
        )
    for item in saved["files"]:
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"catalog input hash changed: {path}")
    return saved


DOG_REVISION = "de0df080ab82f4dbd7615471f2ad33fed5fbf78a"
DOG_WEBQSP_BLOB = "77753315af00397cf7b4cbb42529e3715f43ef50"


def dog_panel(rows: Iterable[dict]) -> tuple[list[dict], dict]:
    """DoG's separately labelled retrieved top-120, any-answer-present population.

    Input order is retrieval order. Never substitute the first 120 RoG triples for
    DoG's retrieved triples. No inference-time gold is written into requests.
    """
    selected, excluded, seen = [], [], set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError(f"duplicate DoG id: {row['id']}")
        seen.add(row["id"])
        graph = row["graph"][:120]
        names = {v for s, _, o in graph for v in (s, o)}
        answers = row["ground_truth"] if "ground_truth" in row else row["answer"]
        if isinstance(answers, str):
            answers = [answers]
        if any(a in names for a in answers):
            selected.append({**row, "answer": answers, "graph": graph})
        else:
            excluded.append(row["id"])
    return selected, {
        "population": "dog_top120_answer_present",
        "secondary_population": True,
        "top_triples": 120,
        "questions": len(selected),
        "excluded_ids": excluded,
        "filter": "at least one gold answer is a subject or object after top-120",
    }


def prepare_catalog(data: Path, out: Path, models: dict[str, Path]) -> dict:
    """Local read-only bridge and tokenizer-bound indexes, with exact array reuse.

    Rebuild each legacy index once to verify its arrays against this KG/tokenizer.
    Keep only references to verified arrays. New metadata/zero sitelinks are local;
    equal counts make the sitelink ordering exactly the existing degree ordering.
    """
    from xgraph.index import add_sitelinks, load_kg_index
    from xgraph.provenance import provenance, sha256_file
    from xgraph.syntax import SpanSyntax
    from xgraph.tokenizer import ServedTokenizer
    from xgraph.toolkg import KGStore
    from xgraph.wikidata import build_index

    started = time.monotonic()
    data, out = Path(data).resolve(), Path(out).absolute()
    model_paths = {f: str(Path(m).resolve()) for f, m in models.items()}
    if (out / "catalog.json").is_file():
        saved = verify_catalog(out)
        if saved["source"] != str(data) or saved["models"] != model_paths:
            raise ValueError("catalog source/models changed")
        return saved
    if data == out or (out.exists() and any(out.iterdir())):
        raise ValueError(f"catalog output must be empty and separate from inputs: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out).free < 2_000_000_000:
        raise ValueError("insufficient free disk for bounded catalog preparation")
    inputs = [data / "manifest.json", data / "questions.json", data / "kg.sqlite"]
    inputs += sorted(p for p in (data / "kg").iterdir() if p.is_file())
    inputs += sorted(p for p in (data / "versions").iterdir() if p.is_file())
    before = {str(p): sha256_file(p) for p in inputs}
    for name in (
        "kg",
        "kg.sqlite",
        "kg-gcr",
        "kg-gcr.sqlite",
        "versions",
        "manifest.json",
        "questions.json",
        "questions-released.json",
        "gold-native.json",
    ):
        if (data / name).exists():
            (out / name).symlink_to(data / name, target_is_directory=(data / name).is_dir())
    indexes = {}
    for family, model in models.items():
        if family not in {"qwen25", "qwen38", "gemma31"}:
            raise ValueError(f"unknown model family {family!r}")
        tokenizer = Path(model).resolve() / "tokenizer.json"
        served = ServedTokenizer.from_file(tokenizer)
        target, candidate = out / f"index-{family}", data / f"index-{family}"
        if candidate.exists():
            header = json.loads((candidate / "kg-index.json").read_text())
            if header["tokenizer_fingerprint"] != served.fingerprint:
                raise ValueError(f"{family}: tokenizer fingerprint mismatch")
        build_index(data / "kg", served, SpanSyntax(), target)
        reused = None
        if candidate.exists():
            fresh = json.loads((target / "kg-index.json").read_text())
            # Legacy indexes predate canonical relation rows; all original metadata
            # and arrays must match. Canonical rows come from the fresh build.
            comparable = {k: v for k, v in fresh.items() if k != "relation_primary_rows"}
            if header != comparable and header != fresh:
                raise ValueError(f"{candidate}: index metadata differs from fresh build")
            for path in sorted(candidate.glob("*.npy")):
                local = target / path.name
                if not local.is_file() or sha256_file(path) != sha256_file(local):
                    raise ValueError(f"index array hash differs from fresh build: {path}")
                path = path.resolve()
                before[str(path)] = sha256_file(path)
                local.unlink()  # Only the duplicate created above.
                local.symlink_to(path)
            before[str(candidate / "kg-index.json")] = sha256_file(candidate / "kg-index.json")
            reused = str(candidate)
        add_sitelinks(target, np.zeros(len(load_kg_index(target).labels.owner_ids), dtype=np.int32))
        before[str(tokenizer)] = sha256_file(tokenizer)
        indexes[family] = {"reused_from": reused, "fingerprint": served.fingerprint}
    store = KGStore(data / "kg.sqlite")
    panel = json.loads((data / "questions.json").read_text())
    for kind in ("S", "blind5", "menuobs", "agent", "gcr"):
        with (out / f"requests-{kind}.jsonl").open("w", encoding="utf-8") as file:
            for question in panel:
                starts = [store.label(int(e[1:])) for e in question["anchor"]]
                request = catalog_requests(
                    question,
                    starts,
                    demos=4 if kind in {"S", "agent"} else 5,
                    observed=kind == "menuobs",
                )
                if kind == "gcr":
                    request = _requests(question, starts, "gcr")
                _line(file, request)
    _json(
        out / "demonstrations.json",
        {
            "chosen_before_outputs": True,
            "origin": "synthetic Freebase examples; fifth mirrors MINTQA's nested composition",
            "examples": CATALOG_DEMOS,
        },
    )
    for path, digest in before.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"catalog input hash changed during build: {path}")
    local_files = [p for p in out.rglob("*") if p.is_file()]
    before.update({str(p): sha256_file(p) for p in local_files})
    report = {
        "artifact_kind": "xgraph.webqsp.catalog",
        "source": str(data),
        "models": model_paths,
        "request_builder_sha256": catalog_request_fingerprint(),
        "indexes": indexes,
        "start_rank": "equal zero sitelinks, then degree, then first label row",
        "files": [{"path": p, "sha256": h} for p, h in sorted(before.items())],
        "seconds": time.monotonic() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "provenance": provenance(inputs),
    }
    _json(out / "catalog.json", report)
    return report


def dog_reader_requests(data: Path, requests: Path, out: Path, *, validation_proxy=False):
    """Use the entire released component, or an explicit validation-only BM25 proxy."""
    import hashlib

    from xgraph.dog import DoGGraph, RetrievalConfig, retrieve_subgraph
    from xgraph.kgversions import component_ranges
    from xgraph.provenance import provenance
    from xgraph.toolkg import KGStore

    metadata = json.loads((data / "manifest.json").read_text())
    if validation_proxy:
        if metadata.get("split") != "validation" or metadata.get("secondary_population"):
            raise ValueError("DoG timing proxy requires validation questions")
    elif not (
        metadata.get("secondary_population")
        and metadata.get("population") == "dog_top120_answer_present"
        and metadata.get("top_triples") == 120
    ):
        raise ValueError("DoG reader requires the released top-120 population")
    rows = _read_jsonl(requests)
    _by_id(rows)
    bounds = component_ranges(data / "versions")
    store = KGStore(data / "kg.sqlite")
    inputs = [
        requests,
        data / "manifest.json",
        data / "kg.sqlite",
        *sorted((data / "versions").glob("*.json")),
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as written:
        for request in rows:
            first, stop = bounds[request["kg_version"]]
            if request.get("answer_mode") != "set" or any(
                not first <= int(e[1:]) < stop for e in request["anchor"]
            ):
                raise ValueError(
                    f"{request['id']}: DoG needs set delivery and component-local anchors"
                )
            read = store.component(first, stop)
            try:
                if validation_proxy:
                    retrieved = retrieve_subgraph(
                        read,
                        request["anchor"],
                        request["question"],
                        config=RetrievalConfig(hops=2, candidate_limit=0),
                    )
                    graph = retrieved.pop("graph")
                    retrieved["population_role"] = "validation_bm25_timing_proxy_only"
                else:
                    graph = [
                        {
                            "ids": list(triple),
                            "labels": [
                                read.label(triple[0]),
                                read.relation_label(triple[1]),
                                read.label(triple[2]),
                            ],
                        }
                        for triple in read._con.execute(
                            "SELECT DISTINCT s, p, o FROM triples ORDER BY s, p, o"
                        )
                    ]
                    if len(graph) > 120:
                        raise ValueError(f"{request['id']}: released DoG graph exceeds top-120")
                    DoGGraph(graph)
                    retrieved = {
                        "ranker": "released_top120",
                        "model_calls": 0,
                        "retained_triples": len(graph),
                        "graph_sha256": hashlib.sha256(
                            json.dumps(graph, sort_keys=True).encode()
                        ).hexdigest(),
                        "deviation": "all released component triples; "
                        "no retrieval, reranking or answer filtering",
                    }
                _line(
                    written,
                    {
                        k: request[k]
                        for k in ("id", "question", "anchor", "topics", "kg_version", "answer_mode")
                    }
                    | {"dog_graph": graph, "retrieval": retrieved},
                )
            finally:
                read._con.close()
    report = {
        "records": len(rows),
        "model_calls": 0,
        "validation_proxy": validation_proxy,
        "provenance": provenance(inputs),
    }
    _json(Path(str(out) + ".summary.json"), report)
    return report


def main(argv: list[str] | None = None) -> None:
    import argparse

    from xgraph.provenance import provenance

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("fetch")
    download.add_argument("--out", type=Path, default=Path("data/external/webqsp/source"))
    download.add_argument("--cache", type=Path)
    build = sub.add_parser("build")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--split", choices=("validation", "test"), required=True)
    build.add_argument("--out", type=Path, required=True)
    requests = sub.add_parser(
        "requests", help="refresh set-aware requests without rebuilding the KG"
    )
    requests.add_argument("--data", type=Path, required=True)
    index = sub.add_parser("index")
    index.add_argument("--data", type=Path, required=True)
    index.add_argument("--tokenizer", type=Path, required=True)
    index.add_argument("--out", type=Path, required=True)
    prepare = sub.add_parser("prepare-catalog", help="verify read-only reuse into a local catalog")
    prepare.add_argument("--data", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--models", type=Path, default=Path("/home/jonas/models"))
    prepare.add_argument(
        "--families",
        nargs="+",
        default=["qwen25", "qwen38", "gemma31"],
        choices=("qwen25", "qwen38", "gemma31"),
    )
    dog = sub.add_parser("dog-panel", help="import a local pinned DoG retrieved graph release")
    dog.add_argument("--source", type=Path, required=True)
    dog.add_argument("--out", type=Path, required=True)
    dog.add_argument("--reference-questions", type=Path, required=True)
    dog_requests = sub.add_parser(
        "dog-requests", help="prepare released-graph reader inputs without an embedding model"
    )
    for flag in ("data", "requests", "out"):
        dog_requests.add_argument("--" + flag, type=Path, required=True)
    dog_requests.add_argument("--validation-proxy", action="store_true")
    for name in ("second-requests", "select-second"):
        second = sub.add_parser(name)
        for flag in ("data", "index", "tokenizer", "requests", "outputs", "out"):
            second.add_argument("--" + flag, type=Path, required=True)
        if name == "second-requests":
            second.add_argument(
                "--mode",
                choices=("audit", "two-pass", "audit-terminal", "two-pass-terminal"),
                required=True,
            )
            second.add_argument("--terminal-width", type=int, default=8)
        else:
            second.add_argument("--second", type=Path, required=True)
            second.add_argument("--first-source", required=True)
            second.add_argument("--second-source", required=True)
    account = sub.add_parser("account", help="charge each generation source once")
    account.add_argument("--outputs", type=Path, required=True)
    account.add_argument("--source", type=Path, action="append", required=True)
    account.add_argument("--out", type=Path, required=True)
    score = sub.add_parser("score")
    score.add_argument("--outputs", type=Path, required=True)
    score.add_argument("--questions", type=Path, required=True)
    score.add_argument("--protocol", choices=("released", "native"), default="released")
    score.add_argument(
        "--released-only",
        action="store_true",
        help="secondary panel: explicitly exclude native omissions from RoG",
    )
    score.add_argument("--out", type=Path, required=True)
    score.add_argument(
        "--population",
        choices=(
            "rog_released_subgraphs",
            "rog_declared_including_native_omissions",
            "dog_top120_answer_present",
        ),
    )
    score.add_argument("--secondary-population", action="store_true")
    score.add_argument(
        "--failed-zero",
        action="store_true",
        help="catalog: failed/truncated records score zero even after replay",
    )
    deliver = sub.add_parser("deliver")
    for flag in ("outputs", "requests", "db", "index", "tokenizer", "kg-versions", "out"):
        deliver.add_argument("--" + flag, type=Path, required=True)
    deliver.add_argument("--complete-path-set", action="store_true")
    deliver.add_argument("--start-rank", choices=("degree", "sitelinks"), default="degree")
    audit = sub.add_parser("audit")
    audit.add_argument("--source", type=Path, required=True)
    audit.add_argument("--predictions", type=Path)
    audit.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    inputs = []
    if args.command == "fetch":
        report = fetch(args.out, args.cache)
    elif args.command == "dog-requests":
        report = dog_reader_requests(
            args.data, args.requests, args.out, validation_proxy=args.validation_proxy
        )
    elif args.command == "requests":
        from xgraph.toolkg import KGStore

        inputs = [args.data / "questions.json", args.data / "kg.sqlite"]
        panel = json.loads(inputs[0].read_text())
        store = KGStore(inputs[1])
        for kind in ("paths", "agent", "iterative", "gcr"):
            with (args.data / f"requests-{kind}.jsonl").open("w", encoding="utf-8") as file:
                for question in panel:
                    starts = [store.label(int(e[1:])) for e in question["anchor"]]
                    _line(file, _requests(question, starts, kind))
        report = {"questions": len(panel), "answer_mode": "set", "provenance": provenance(inputs)}
        _json(args.data / "requests.provenance.json", report)
    elif args.command == "build":
        import pyarrow

        inputs = [args.source / name for name in SOURCES if name.startswith(args.split + "-")]
        native = None
        if args.split == "test":
            inputs.append(args.source / "native-test.json")
            native = json.loads(inputs[-1].read_text())["Questions"]
        for path in inputs:
            from xgraph.provenance import sha256_file

            if sha256_file(path) != SOURCES[path.name][2]:
                raise ValueError(f"source SHA-256 mismatch: {path}")
        check_budget(args.source.parent, args.out, reserve=2_000_000_000)
        report = convert(parquet_rows(args.source, args.split), args.out, native=native)
        report.update(split=args.split, pyarrow=pyarrow.__version__, provenance=provenance(inputs))
        check_budget(args.source.parent, args.out)
        _json(args.out / "manifest.json", report)
    elif args.command == "index":
        from xgraph.index import load_kg_index
        from xgraph.syntax import SpanSyntax
        from xgraph.tokenizer import ServedTokenizer
        from xgraph.wikidata import build_index

        if args.out.exists() and any(args.out.iterdir()):
            raise ValueError(f"index output must be empty: {args.out}")
        check_budget(args.data.parent, args.out, reserve=2_000_000_000)
        started = time.monotonic()
        build_index(
            args.data / "kg", ServedTokenizer.from_file(args.tokenizer), SpanSyntax(), args.out
        )
        saved = load_kg_index(args.out)
        inputs = [args.data / "manifest.json", args.tokenizer]
        report = {
            "seconds": time.monotonic() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "unwritable_triples": saved.unwritable_triples,
            "excluded_labels": len(saved.labels.excluded),
            "fingerprint": saved.tokenizer_fingerprint,
            "provenance": provenance(inputs),
        }
        _json(args.out / "webqsp-index.json", report)
        check_budget(args.data.parent, args.out)
    elif args.command in {"second-requests", "select-second"}:
        from xgraph.index import load_kg_index
        from xgraph.kgversions import load_kg_versions
        from xgraph.path_policy import second_call_requests, select_second_call
        from xgraph.programs import MatchedExecutor
        from xgraph.tokenizer import ServedTokenizer
        from xgraph.toolkg import KGStore, version_stores

        requests, records = _read_jsonl(args.requests), _read_jsonl(args.outputs)
        _checked_outputs(records, {r["id"] for r in requests})
        saved = load_kg_index(args.index)
        tok = ServedTokenizer.from_file(args.tokenizer)
        executor = MatchedExecutor(
            saved,
            tok,
            versions=load_kg_versions(args.data / "versions", saved),
            start_rank="sitelinks",
            unquoted_starts=True,
        )
        store = KGStore(args.data / "kg.sqlite")
        inputs = [
            args.requests,
            args.outputs,
            args.tokenizer,
            args.data / "kg.sqlite",
            args.data / "versions/kg-versions.json",
            *sorted(args.index.glob("*")),
        ]
        if args.command == "second-requests":
            relation_labels = args.data / "kg/relation_labels.jsonl"
            inputs.append(relation_labels)
            rows = second_call_requests(
                args.mode,
                requests,
                records,
                executor,
                store,
                tok,
                relation_rows=[json.loads(s) for s in relation_labels.read_text().splitlines()],
                stores=version_stores(
                    store, args.data / "versions", {r["kg_version"] for r in requests}
                ),
                terminal_width=args.terminal_width,
            )
        else:
            inputs.append(args.second)
            rows = select_second_call(
                records,
                _read_jsonl(args.second),
                requests,
                executor,
                first_source=args.first_source,
                second_source=args.second_source,
            )
        with args.out.open("w", encoding="utf-8") as file:
            for row in rows:
                _line(file, row)
        report = {"records": len(rows), "provenance": provenance(inputs)}
        _json(Path(str(args.out) + ".summary.json"), report)
    elif args.command == "account":
        from xgraph.mintqa_rematch import combined_usage

        inputs = [args.outputs, *args.source]
        records = _read_jsonl(args.outputs)
        sources = [_by_id(_read_jsonl(p)) for p in args.source]
        if {r["id"] for r in records} != set(sources[0]):
            raise ValueError("account needs first-call usage for the complete panel")
        rows = [
            {
                **row,
                "usage": combined_usage(*(s.get(row["id"]) for s in sources)),
                "cost_sources": list(map(str, args.source)),
            }
            for row in records
        ]
        with args.out.open("w", encoding="utf-8") as file:
            for row in rows:
                _line(file, row)
        report = {"records": len(rows), "provenance": provenance(inputs)}
        _json(Path(str(args.out) + ".summary.json"), report)
    elif args.command == "dog-panel":
        import hashlib

        from xgraph.provenance import sha256_file
        from xgraph.toolkg import build_store

        if not args.source.is_file():
            raise ValueError(
                f"missing local DoG retrieved graph file: {args.source}; no download attempted"
            )
        raw = args.source.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if len(raw) != 19_304_913 or blob != DOG_WEBQSP_BLOB:
            raise ValueError("DoG source differs from pinned retrieved top-120 release")
        selected, condition = dog_panel(_read_jsonl(args.source))
        reference = _by_id(json.loads(args.reference_questions.read_text()))
        for row in selected:
            if row["id"] not in reference or set(row["answer"]) != set(
                reference[row["id"]]["answers"]
            ):
                raise ValueError(f"{row['id']}: DoG gold/id differs from RoG reference")
        if len(selected) != 1542:
            raise ValueError(f"DoG published panel needs 1542 questions, found {len(selected)}")
        report = convert(selected, args.out)
        build_store(args.out / "kg", args.out / "kg.sqlite")
        build_store(args.out / "kg-gcr", args.out / "kg-gcr.sqlite")
        inputs = [args.source, args.reference_questions]
        report.update(
            condition,
            split="test",
            dog_revision=DOG_REVISION,
            source_sha256=sha256_file(args.source),
            provenance=provenance(inputs),
        )
        _json(args.out / "manifest.json", report)
    elif args.command == "prepare-catalog":
        from xgraph.mintqa_queue import FAMILIES

        report = prepare_catalog(
            args.data, args.out, {f: args.models / FAMILIES[f][0] for f in args.families}
        )
    elif args.command == "score":
        inputs = [args.outputs, args.questions]
        questions = json.loads(args.questions.read_text())
        records = _read_jsonl(args.outputs)
        if args.protocol == "native":
            if args.released_only:
                parser.error("--released-only is a released-string scoring option")
            if args.population or args.secondary_population:
                parser.error("--population is a released-label scoring option")
            report = native_scores(records, questions["Questions"])
        else:
            report = score_records(
                records, questions, released_only=args.released_only, failed_zero=args.failed_zero
            )
            if args.secondary_population and not args.population:
                parser.error("--secondary-population requires --population")
            if args.population:
                report.update(
                    population=args.population,
                    secondary_population=args.secondary_population
                    or args.population == "dog_top120_answer_present",
                )
        report["provenance"] = provenance(inputs)
        _json(args.out, report)
    elif args.command == "deliver":
        from xgraph.index import load_kg_index
        from xgraph.kgversions import load_kg_versions
        from xgraph.programs import MatchedExecutor
        from xgraph.tokenizer import ServedTokenizer
        from xgraph.toolkg import KGStore, version_stores

        started = time.monotonic()
        saved = load_kg_index(args.index)
        requests = _read_jsonl(args.requests)
        records = _read_jsonl(args.outputs)
        executor = MatchedExecutor(
            saved,
            ServedTokenizer.from_file(args.tokenizer),
            versions=load_kg_versions(args.kg_versions, saved),
            start_rank=args.start_rank,
            unquoted_starts=True,
        )
        store = KGStore(args.db)
        versions = version_stores(store, args.kg_versions, {r["kg_version"] for r in requests})
        delivered = deliver_records(
            records, requests, executor, store, versions, complete_path_set=args.complete_path_set
        )
        with args.out.open("w", encoding="utf-8") as file:
            for row in delivered:
                _line(file, row)
        inputs = [
            args.outputs,
            args.requests,
            args.db,
            args.index / "kg-index.json",
            args.tokenizer,
            args.kg_versions / "kg-versions.json",
        ]
        report = {
            "questions": len(delivered),
            "cap_encountered": sum(r["delivery"]["cap_encountered"] for r in delivered),
            "endpoint_loss": sum(r["delivery"]["endpoint_loss"] > 0 for r in delivered),
            "seconds": time.monotonic() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "provenance": provenance(inputs),
        }
        _json(args.out.with_suffix(args.out.suffix + ".summary.json"), report)
    else:
        inputs = [args.source / "gcr-qa_utils.py", args.source / "native-evaluate.py"]
        if args.predictions:
            inputs.append(args.predictions)
        report = {
            **reference_audit(args.source, args.predictions),
            "provenance": provenance(inputs),
        }
        _json(args.out, report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("provenance", "per_question", "files")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
