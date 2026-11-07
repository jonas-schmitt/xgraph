"""Judging WWQ fact spans against Wikipedia (`docs/quality-gate.md`, run setup).

For each fact span `⟦ subject · relation · object⟧` of an answer, the candidate
articles are those of the entities whose English label or alias equals the subject
or object text (titles from the gate KG's sitelinks), and the articles titled with
that text itself. Their passages are pooled, BM25 ranks them with the span's three
segments as the query, and the top five per span, merged per answer, go to the judge
with all of that answer's fact spans. Both arms go through the same procedure, and
the judge never sees the KG.

    python -m xgraph.gate wwq-facts --outputs target/run/constrained.jsonl \\
        --kg data/wikidata/gate-kg --wikipedia data/wikipedia/enwiki-20261004.db \\
        --cache target/run/judge.jsonl --out target/run/constrained.facts.json
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from xgraph.score import normalize, record_text, spans
from xgraph.wikipedia import bm25_top

TOP = 5


def fact_spans(text: str) -> list[tuple[str, str, str]]:
    return [segments for kind, segments in spans(text) if kind == "fact" and len(segments) == 3]


def claim(span: tuple[str, str, str]) -> str:
    """A fact span as the judge reads it: its triple without the span delimiters."""
    subject, relation, obj = span
    return f"{subject} — {relation} — {obj}"


def entity_titles(texts: set[str], kg: str | Path) -> dict[str, list[str]]:
    """Article titles of the entities whose label or alias, normalized, is one of the texts."""
    kg = Path(kg)
    items: dict[str, set[str]] = {}
    with (kg / "entity_labels.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            item, label, _ = json.loads(line)
            key = normalize(label)
            if key in texts:
                items.setdefault(item, set()).add(key)
    titles: dict[str, list[str]] = {}
    with (kg / "sitelinks.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            item, title = json.loads(line)
            for key in items.get(item, ()):
                titles.setdefault(key, []).append(title)
    return {key: sorted(found) for key, found in titles.items()}


def _span_passages(span, db, titles: dict[str, list[str]]) -> list[str]:
    pages: dict[str, None] = {}
    for text in (span[0], span[2]):
        for title in [*titles.get(normalize(text), []), text]:
            pages[title] = None
    pool = [p for title in pages for p in db.passages(title)]
    return bm25_top(" ".join(span), pool, TOP)


def judge_fact_spans(
    records: list[dict], judge, db, titles: dict[str, list[str]], workers: int = 8
) -> dict[str, list[dict]]:
    """Per answer, each fact span with the judge's decision; the distinct spans of an
    answer go to the judge once, and a repeated span carries its verdict each time."""

    def one(record: dict) -> tuple[str, list[dict]]:
        found = fact_spans(record_text(record))
        if not found:
            return record["id"], []
        distinct = list(dict.fromkeys(found))  # a repeated span is judged once
        passages: dict[str, None] = {}
        for span in distinct:
            for passage in _span_passages(span, db, titles):
                passages[passage] = None
        topic = ", ".join(dict.fromkeys(span[0] for span in distinct))
        decisions = judge.supported(topic, [claim(s) for s in distinct], list(passages))
        verdict = dict(zip(distinct, decisions, strict=True))
        return record["id"], [{"span": list(s), "supported": verdict[s]} for s in found]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(one, records))


GAMMA = 10


def _topic_passages(topic: str, facts: list[str], db) -> list[str]:
    """FActScore's retrieval: per fact, the top passages of the topic's article, merged."""
    article = db.passages(topic)
    merged: dict[str, None] = {}
    for fact in facts:
        for passage in bm25_top(f"{topic} {fact.strip()}", article, TOP):
            merged[passage] = None
    return list(merged)


def factscore(records: list[dict], judge, db, workers: int = 8) -> dict[str, dict]:
    """Per biography (id = topic): atomic facts, decisions, and FActScore with gamma = 10.

    A biography without facts scores 0, so that an arm cannot gain by not answering.
    """
    from xgraph.prompts import strip_spans

    def one(record: dict) -> tuple[str, dict]:
        topic = record["id"]
        text = strip_spans(record_text(record))
        facts = judge.atomic_facts(topic, text) if text.strip() else []
        if not facts:
            return topic, {"facts": [], "supported": [], "score": 0.0}
        decisions = judge.supported(topic, facts, _topic_passages(topic, facts, db))
        share = sum(decisions) / len(facts)
        penalty = 1.0 if len(facts) > GAMMA else math.exp(1 - GAMMA / len(facts))
        return topic, {"facts": facts, "supported": decisions, "score": share * penalty}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(one, records))


def validate_judge(archive: str | Path, judge, db, workers: int = 8) -> dict[str, dict]:
    """The judge on FActScore's human-labeled facts (IR excluded), per labeled model.

    Reports fact-level agreement and FActScore (without the length penalty, on the
    human atomic facts) from the human labels and from the judge's decisions.
    """
    report: dict[str, dict] = {}
    with zipfile.ZipFile(archive) as z:
        names = sorted(
            n for n in z.namelist() if n.startswith("data/labeled/") and n.endswith(".jsonl")
        )
        for name in names:
            topics = []
            for line in z.read(name).decode("utf-8").splitlines():
                row = json.loads(line)
                facts = [
                    (fact["text"], fact["label"] == "S")
                    for sentence in row.get("annotations") or []
                    for fact in sentence.get("human-atomic-facts") or []
                    if fact["label"] in ("S", "NS")
                ]
                if facts:
                    topics.append((row["topic"], facts))

            def one(item: tuple[str, list[tuple[str, bool]]]) -> tuple[list[bool], list[bool]]:
                topic, facts = item
                texts = [text for text, _ in facts]
                decided = judge.supported(topic, texts, _topic_passages(topic, texts, db))
                return [label for _, label in facts], decided

            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(one, topics))
            pairs = [
                (h, j) for human, decided in results for h, j in zip(human, decided, strict=True)
            ]
            report[Path(name).stem] = {
                "topics": len(results),
                "facts": len(pairs),
                "agreement": sum(h == j for h, j in pairs) / len(pairs) if pairs else None,
                "human_factscore": _mean([sum(h) / len(h) for h, _ in results]),
                "judge_factscore": _mean([sum(d) / len(d) for _, d in results]),
            }
    return report


MAX_NAMES = 6  # names per gold entity in the judge's prompt
MAX_GOLD = 10  # gold entities per question in the judge's prompt


def judged_answers(
    outputs: dict[str, str],
    questions: dict[str, str],
    gold: dict,
    names: dict[str, list[str]],
    judge,
    workers: int = 8,
) -> dict[str, dict]:
    """Answer correctness that does not depend on surface form: an answer whose label
    is a gold entity's label or alias is correct; otherwise the judge decides whether it
    names a gold entity (`Judge.same_entity`). Questions without an answer are wrong."""
    from xgraph.prompts import final_answer

    def one(question: str) -> tuple[str, dict]:
        entities = [e for e in gold["answers"][question]["entities"] if e in names]
        answer = final_answer(outputs[question])
        if answer is None:
            return question, {"correct": False, "by": "no answer"}
        if any(normalize(answer) == normalize(n) for e in entities for n in names[e]):
            return question, {"correct": True, "by": "label"}
        if not entities:
            return question, {"correct": False, "by": "no gold name"}
        listed = [names[e][:MAX_NAMES] for e in entities[:MAX_GOLD]]
        same = judge.same_entity(questions[question], answer, listed)
        return question, {"correct": same, "by": "judge"}

    scored = [q for q, a in gold["answers"].items() if a["entities"]]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(one, scored))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main(argv: list[str] | None = None) -> None:
    from xgraph.judge import Judge
    from xgraph.provenance import provenance
    from xgraph.wikipedia import PassageDB

    parser = argparse.ArgumentParser(prog="python -m xgraph.gate", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    facts = commands.add_parser("wwq-facts", help="judge the fact spans of WWQ answers")
    facts.add_argument("--outputs", type=Path, required=True, help="xgraph.generate output")
    facts.add_argument("--kg", type=Path, required=True, help="gate KG directory")
    facts.add_argument("--wikipedia", type=Path, required=True, help="passage database")
    facts.add_argument("--cache", type=Path, required=True, help="judge decision cache")
    facts.add_argument("--out", type=Path, required=True, help="decisions (JSON)")
    facts.add_argument("--workers", type=int, default=8, help="judge calls at a time")
    bios = commands.add_parser("factscore", help="FActScore of biographies")
    bios.add_argument("--outputs", type=Path, required=True, help="xgraph.generate output")
    bios.add_argument("--wikipedia", type=Path, required=True, help="passage database")
    bios.add_argument("--cache", type=Path, required=True, help="judge decision cache")
    bios.add_argument("--out", type=Path, required=True, help="scores (JSON)")
    bios.add_argument("--workers", type=int, default=8, help="judge calls at a time")
    check = commands.add_parser("validate-judge", help="the judge on FActScore's human labels")
    check.add_argument("--archive", type=Path, required=True, help="FActScore data.zip")
    check.add_argument("--wikipedia", type=Path, required=True, help="passage database")
    check.add_argument("--cache", type=Path, required=True, help="judge decision cache")
    check.add_argument("--out", type=Path, required=True, help="report (JSON)")
    check.add_argument("--workers", type=int, default=8, help="judge calls at a time")
    answers = commands.add_parser(
        "wwq-judged", help="answer accuracy by the judge, independent of surface form"
    )
    answers.add_argument("--outputs", type=Path, required=True, help="xgraph.generate output")
    answers.add_argument("--questions", type=Path, required=True, help="WWQ-format questions")
    answers.add_argument("--gold", type=Path, required=True, help="gold answer file (JSON)")
    answers.add_argument("--labels", type=Path, required=True, help="gold labels (JSON lines)")
    answers.add_argument("--cache", type=Path, required=True, help="judge decision cache")
    answers.add_argument("--out", type=Path, required=True, help="decisions (JSON)")
    answers.add_argument("--workers", type=int, default=8, help="judge calls at a time")
    args = parser.parse_args(argv)

    judge = Judge(args.cache)
    if args.command == "wwq-judged":
        gold = json.loads(args.gold.read_text())
        names: dict[str, list[str]] = {}
        for line in args.labels.read_text(encoding="utf-8").splitlines():
            item, label, kind = json.loads(line)
            listed = names.setdefault(item, [])
            listed.insert(0, label) if kind == "label" else listed.append(label)
        questions = {q["id"]: q["utterance"] for q in json.loads(args.questions.read_text())}
        outputs = {
            r["id"]: record_text(r)
            for r in map(json.loads, args.outputs.read_text().splitlines())
            if r
        }
        decided = judged_answers(outputs, questions, gold, names, judge, args.workers)
        report = {
            "artifact_kind": "xgraph.wwq_judged_answers",
            "version": 1,
            "accuracy": sum(d["correct"] for d in decided.values()) / len(decided),
            "decisions": decided,
            "provenance": provenance([args.outputs, args.questions, args.gold, args.labels]),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
        return
    db = PassageDB(args.wikipedia)
    if args.command == "validate-judge":
        result = {"by_model": validate_judge(args.archive, judge, db, args.workers)}
        inputs = [args.archive, args.wikipedia]
    elif args.command == "factscore":
        records = [json.loads(line) for line in args.outputs.read_text().splitlines() if line]
        scores = factscore(records, judge, db, args.workers)
        result = {"scores": scores, "mean": _mean([s["score"] for s in scores.values()])}
        inputs = [args.outputs, args.wikipedia]
    if args.command != "wwq-facts":
        report = {"artifact_kind": f"xgraph.{args.command}", "version": 1, **result}
        report["provenance"] = provenance(inputs)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
        return

    records = [json.loads(line) for line in args.outputs.read_text().splitlines() if line]
    texts = {
        normalize(t) for r in records for s in fact_spans(record_text(r)) for t in (s[0], s[2])
    }
    titles = entity_titles(texts, args.kg)
    decisions = judge_fact_spans(records, judge, db, titles, args.workers)
    report = {
        "artifact_kind": "xgraph.wwq_fact_spans",
        "version": 1,
        "decisions": decisions,
        "provenance": provenance([args.outputs, args.wikipedia]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
