"""Grounded answers: answers backed by a fact span the KG holds.

An answer is grounded when its final entity span names the subject or the object of a
fact span earlier in the same output, and the KG holds that fact: some triple whose
subject, relation, and object have these texts as labels or aliases (NFC, case
folded). Under the constraint every closed fact span holds by construction; for an
unconstrained arm this is a check after the fact, the post-hoc verification baseline.
Reported per arm: coverage (grounded answers over questions), accuracy among grounded
answers and among the other answers, and the share of fact spans the KG holds. Matching
by label, a held fact may belong to a homonym; at the entity level an answer is grounded
only through a held triple whose answer side is one of the question's gold entities
(`grounded_entity`, `per_question_entity`).

    python -m xgraph.grounding --outputs target/gate/wwq-constrained.jsonl \\
        --gold eval/wwq-gold-wdqs-2026-10-12.json --labels data/wikidata/gold-labels.jsonl \\
        --kg data/wikidata/gate-kg --out target/gate/wwq-constrained.grounding.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from xgraph.prompts import final_answer
from xgraph.score import normalize, record_text, spans

CHUNK = 1 << 22


def _fact_spans(text: str) -> list[tuple[str, str, str]]:
    before = text.rsplit("\nAnswer:", 1)[0]
    return [s for kind, s in spans(before) if kind == "fact" and len(s) == 3]


def grounded(
    outputs: dict[str, str],
    correct: dict[str, bool],
    kg: str | Path,
    gold_entities: dict[str, set[int]] | None = None,
) -> dict:
    """With gold_entities, an answer is also checked at the entity level: grounded
    through a held triple whose answer side is a gold entity, not a homonym."""
    kg = Path(kg)
    facts = {q: _fact_spans(t) for q, t in outputs.items()}
    entity_texts = {normalize(x) for fs in facts.values() for s, _, o in fs for x in (s, o)}
    relation_texts = {normalize(r) for fs in facts.values() for _, r, _ in fs}
    items: dict[str, set[int]] = {}
    with (kg / "entity_labels.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            item, label, _ = json.loads(line)
            if (key := normalize(label)) in entity_texts:
                items.setdefault(key, set()).add(int(item[1:]))
    properties: dict[str, set[int]] = {}
    with (kg / "relation_labels.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            prop, label, _ = json.loads(line)
            if (key := normalize(label)) in relation_texts:
                properties.setdefault(key, set()).add(int(prop[1:]))
    gold_numbers = set().union(*gold_entities.values()) if gold_entities else set()
    subjects = np.array(sorted(set().union(*items.values()) | gold_numbers), dtype=np.int64)
    held_triples: set[tuple[int, int, int]] = set()
    if len(subjects):
        triples = np.load(kg / "triples.npy", mmap_mode="r")
        for at in range(0, len(triples), CHUNK):
            chunk = np.asarray(triples[at : at + CHUNK], dtype=np.int64)
            held_triples.update(map(tuple, chunk[np.isin(chunk[:, 0], subjects)].tolist()))

    def holds(s: str, r: str, o: str) -> bool:
        return any(
            (a, p, b) in held_triples
            for a in items.get(normalize(s), ())
            for p in properties.get(normalize(r), ())
            for b in items.get(normalize(o), ())
        )

    def holds_for(gold: set[int], s: str, r: str, o: str, answer: str) -> bool:
        """The fact holds with a gold entity on the answer's side."""
        relations = properties.get(normalize(r), ())
        if answer == normalize(o) and any(
            (a, p, b) in held_triples for a in items.get(normalize(s), ()) for p in relations
            for b in gold
        ):  # fmt: skip
            return True
        return answer == normalize(s) and any(
            (a, p, b) in held_triples for a in gold for p in relations
            for b in items.get(normalize(o), ())
        )  # fmt: skip

    per_question: dict[str, bool] = {}
    per_question_entity: dict[str, bool] = {}
    span_count = held_count = 0
    for question, text in outputs.items():
        held = [f for f in facts[question] if holds(*f)]
        span_count += len(facts[question])
        held_count += len(held)
        answer = final_answer(text)
        per_question[question] = answer is not None and any(
            normalize(answer) in (normalize(s), normalize(o)) for s, _, o in held
        )
        if gold_entities is not None:
            gold = gold_entities.get(question, set())
            per_question_entity[question] = answer is not None and any(
                holds_for(gold, *f, normalize(answer)) for f in held
            )
    answered = {q for q, t in outputs.items() if final_answer(t) is not None}
    inside = [q for q in outputs if per_question[q]]
    outside = [q for q in answered if not per_question[q]]

    def accuracy(questions: list[str]) -> float | None:
        return sum(correct[q] for q in questions) / len(questions) if questions else None

    return {
        "questions": len(outputs),
        "fact_spans": span_count,
        "fact_spans_held": held_count,
        "grounded": len(inside),
        "coverage": len(inside) / len(outputs) if outputs else 0.0,
        "accuracy_grounded": accuracy(inside),
        "accuracy_ungrounded_answered": accuracy(outside),
        "per_question": per_question,
        **(
            {
                "grounded_entity": sum(per_question_entity.values()),
                "per_question_entity": per_question_entity,
            }
            if gold_entities is not None
            else {}
        ),
    }


def main(argv: list[str] | None = None) -> None:
    from xgraph.provenance import provenance
    from xgraph.score import load_labels, wwq_accuracy

    parser = argparse.ArgumentParser(prog="python -m xgraph.grounding", description=__doc__)
    parser.add_argument("--outputs", type=Path, required=True, help="xgraph.generate output")
    parser.add_argument("--gold", type=Path, required=True, help="gold answer file (JSON)")
    parser.add_argument("--labels", type=Path, required=True, help="gold labels (JSON lines)")
    parser.add_argument("--kg", type=Path, required=True, help="gate KG directory")
    parser.add_argument("--out", type=Path, required=True, help="result (JSON)")
    args = parser.parse_args(argv)
    gold = json.loads(args.gold.read_text())
    labels = load_labels(
        [args.labels], {e for a in gold["answers"].values() for e in a["entities"]}
    )
    texts = {
        r["id"]: record_text(r) for r in map(json.loads, args.outputs.read_text().splitlines()) if r
    }
    correct = wwq_accuracy(texts, gold, labels)["correct"]
    gold_entities = {q: {int(e[1:]) for e in a["entities"]} for q, a in gold["answers"].items()}
    result = grounded({q: texts[q] for q in correct}, correct, args.kg, gold_entities)
    report = {
        "artifact_kind": "xgraph.grounded_answers",
        "version": 1,
        **result,
        "provenance": provenance([args.outputs, args.gold, args.labels]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if not k.startswith("per_question")}, indent=1)
    )


if __name__ == "__main__":
    main()
