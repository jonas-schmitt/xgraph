"""Templated questions from KG triples, for evaluating graph use at scale.

WWQ's dev split holds few single-pattern questions whose entities the train split does
not name, too few to measure whether a model adopts edited facts. Here each question
asks for one relation of one subject ("What is the place of birth of Ada Lovelace?"),
drawn with a seed from the (subject, relation) pairs of the KG with a template and
exactly one object, spread evenly over the templates, leaving out excluded subjects.
They are written as WWQ questions (`utterance`, and a single-pattern `sparql`) with a
gold answer file, so that prompts, probes (`python -m xgraph.probes`), generation,
and scoring work unchanged.

    python -m xgraph.synthetic --kg data/wikidata/gate-kg --n 2000 --seed 0 \\
        --exclude data/wwq/train.json --questions target/synthetic/questions.json \\
        --gold target/synthetic/gold.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TEMPLATES = {
    19: "What is the place of birth of {}?",
    20: "Where did {} die?",
    26: "Who is the spouse of {}?",
    27: "What country is {} a citizen of?",
    36: "What is the capital of {}?",
    50: "Who wrote {}?",
    57: "Who directed {}?",
    69: "Where was {} educated?",
    86: "Who composed {}?",
    108: "Who employs {}?",
    112: "Who founded {}?",
    136: "What genre is {}?",
    159: "Where is the headquarters of {}?",
    175: "Who performed {}?",
    264: "What record label is {} on?",
    1037: "Who is the director or manager of {}?",
}


def questions_from_kg(
    kg: str | Path, n: int, seed: int, exclude: set[int]
) -> tuple[list[dict], dict]:
    kg = Path(kg)
    triples = np.load(kg / "triples.npy")
    triples = triples[np.isin(triples[:, 1], list(TEMPLATES))].astype(np.int64)
    key = triples[:, 0] * (1 << 20) + triples[:, 1]
    _, first, counts = np.unique(key, return_index=True, return_counts=True)
    single = triples[first[counts == 1]]
    single = single[~np.isin(single[:, 0], sorted(exclude))]
    names: dict[int, str] = {}
    wanted = set(single[:, 0].tolist()) | set(single[:, 2].tolist())
    with (kg / "entity_labels.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            item, label, kind = json.loads(line)
            if kind == "label" and (number := int(item[1:])) in wanted:
                names.setdefault(number, label)
    single = single[[s in names and o in names for s, _, o in single.tolist()]]
    rng = np.random.default_rng(seed)
    per_relation = max(1, n // len(TEMPLATES))
    chosen = []
    for relation in sorted(TEMPLATES):
        rows = single[single[:, 1] == relation]
        picks = rng.choice(len(rows), size=min(per_relation, len(rows)), replace=False)
        chosen.extend(rows[sorted(picks)].tolist())
    questions, answers = [], {}
    for s, p, o in chosen:
        qid = f"syn-Q{s}-P{p}"
        questions.append(
            {
                "id": qid,
                "utterance": TEMPLATES[p].format(names[s]),
                "sparql": f"SELECT ?x WHERE {{ wd:Q{s} wdt:P{p} ?x. }}",
            }
        )
        answers[qid] = {"entities": [f"Q{o}"], "literals": []}
    return questions, {"answers": answers, "failed": {}}


def main(argv: list[str] | None = None) -> None:
    from xgraph.wwq import NAMED

    parser = argparse.ArgumentParser(prog="python -m xgraph.synthetic", description=__doc__)
    parser.add_argument("--kg", type=Path, required=True, help="gate KG directory")
    parser.add_argument("--n", type=int, required=True, help="questions, spread over templates")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--exclude", type=Path, help="a WWQ split whose named entities to leave out"
    )
    parser.add_argument("--questions", type=Path, required=True, help="questions (JSON)")
    parser.add_argument("--gold", type=Path, required=True, help="gold answer file (JSON)")
    args = parser.parse_args(argv)
    exclude: set[int] = set()
    if args.exclude:
        for q in json.loads(args.exclude.read_text()):
            exclude |= {int(x) for x in NAMED.findall(q["sparql"])}
    questions, gold = questions_from_kg(args.kg, args.n, args.seed, exclude)
    for path, content in ((args.questions, questions), (args.gold, gold)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, ensure_ascii=False, indent=1) + "\n")
    print(f"{len(questions)} questions")


if __name__ == "__main__":
    main()
