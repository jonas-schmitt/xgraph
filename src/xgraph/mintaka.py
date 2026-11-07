"""Mintaka (amazon-science/mintaka, CC BY 4.0) as a fresh evaluation set.

Mintaka's questions are crowdsourced in natural language, with Wikidata entities for
the question and the answer. Its entity-answer test questions are written in the
formats the WWQ pipeline reads, so that prompts, generation, scoring, the judge, and
the retrieval arm work unchanged: questions with an `id`, an `utterance`, a `sparql`
field that only names the question's entities (`wd:Q...`, which the retrieval arm and
the coverage measurement read) and the Mintaka complexity type, and a gold answer
file like `python -m xgraph.wwq gold` writes.

    python -m xgraph.mintaka --test data/mintaka/test.json \\
        --questions target/mintaka/questions.json --gold target/mintaka/gold.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_mintaka(path: str | Path) -> tuple[list[dict], dict]:
    """The entity-answer questions and their gold answers."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    questions, answers = [], {}
    for r in records:
        answer = r["answer"]
        if answer["answerType"] != "entity" or not answer["answer"]:
            continue
        entities = [a["name"] for a in answer["answer"] if str(a["name"]).startswith("Q")]
        if not entities:
            continue
        named = " ".join(
            f"wd:{e['name']}"
            for e in r.get("questionEntity") or []
            if e.get("entityType") == "entity" and str(e["name"]).startswith("Q")
        )
        questions.append(
            {
                "id": r["id"],
                "utterance": r["question"],
                "sparql": f"# question entities: {named}",
                "complexity": r["complexityType"],
            }
        )
        answers[r["id"]] = {"entities": entities, "literals": []}
    return questions, {"answers": answers, "failed": {}}


def by_complexity(correct: dict[str, bool], questions: list[dict]) -> dict[str, dict]:
    """Answer accuracy per Mintaka complexity type."""
    groups: dict[str, list[bool]] = {}
    for q in questions:
        if q["id"] in correct:
            groups.setdefault(q["complexity"], []).append(correct[q["id"]])
    return {
        kind: {"questions": len(hits), "accuracy": sum(hits) / len(hits)}
        for kind, hits in sorted(groups.items())
    }


def gold_in_kg(gold: dict, kg: str | Path) -> dict[str, int]:
    """How many questions have a gold answer entity among the KG's labelled items: the
    questions a constrained answer span can answer at all."""
    wanted = {e for a in gold["answers"].values() for e in a["entities"]}
    present: set[str] = set()
    with (Path(kg) / "entity_labels.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            item = line[2 : line.index('"', 2)]
            if item in wanted:
                present.add(item)
    return {
        "questions": len(gold["answers"]),
        "with_an_answer_in_kg": sum(
            any(e in present for e in a["entities"]) for a in gold["answers"].values()
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m xgraph.mintaka", description=__doc__)
    parser.add_argument("--test", type=Path, required=True, help="a Mintaka split (mintaka_*.json)")
    parser.add_argument("--questions", type=Path, required=True, help="questions (JSON)")
    parser.add_argument("--gold", type=Path, required=True, help="gold answer file (JSON)")
    parser.add_argument("--sample", type=int, help="keep this many questions, seeded")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    questions, gold = load_mintaka(args.test)
    if args.sample is not None:
        import random

        kept = set(random.Random(args.seed).sample([q["id"] for q in questions], args.sample))
        questions = [q for q in questions if q["id"] in kept]  # in the split's order
        gold = {**gold, "answers": {k: v for k, v in gold["answers"].items() if k in kept}}
    for path, content in ((args.questions, questions), (args.gold, gold)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, ensure_ascii=False, indent=1) + "\n")
    print(f"{len(questions)} entity-answer questions")


if __name__ == "__main__":
    main()
