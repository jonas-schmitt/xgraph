"""Voting over sampled answers: each question keeps the answer most of its samples give.

`python -m xgraph.generate --samples k` writes each request k times as `<id>#<j>`, each
sampled with its own seed; the vote groups them by question, counts their final
answers (`Answer: ⟦ label⟧`, xgraph.prompts.final_answer) after normalization
(xgraph.score.normalize), and keeps the record of the earliest sample giving the most
frequent one. Under path mode every sample's answer is bound to what its path reached,
so the vote is over KG-executed answers. A sample without an answer does not vote; a
question none of whose samples answers keeps its first sample.

    python -m xgraph.vote --outputs target/run/paths-sampled.jsonl \\
        --out target/run/paths-voted.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def vote(records: list[dict]) -> list[dict]:
    """One record per question (the id before "#"), in first-seen order, with the votes
    each normalized answer got and the number of samples."""
    from xgraph.prompts import final_answer
    from xgraph.score import normalize, record_text

    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record["id"].rsplit("#", 1)[0], []).append(record)
    voted = []
    for qid, samples in groups.items():
        answers = [final_answer(record_text(r)) for r in samples]
        votes = Counter(normalize(a) for a in answers if a)
        winner = {**samples[0], "text": record_text(samples[0])}
        if votes:
            best = max(votes.values())
            winner = next(
                r
                for r, a in zip(samples, answers, strict=True)
                if a and votes[normalize(a)] == best
            )
        voted.append({**winner, "id": qid, "votes": dict(votes), "samples": len(samples)})
    return voted


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m xgraph.vote", description=__doc__)
    parser.add_argument("--outputs", type=Path, required=True, help="sampled records (JSONL)")
    parser.add_argument("--out", type=Path, required=True, help="one record per question")
    args = parser.parse_args(argv)
    records = [json.loads(line) for line in args.outputs.read_text().splitlines() if line]
    voted = vote(records)
    with args.out.open("w", encoding="utf-8") as out:
        for record in voted:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"samples": len(records), "questions": len(voted)}))


if __name__ == "__main__":
    main()
