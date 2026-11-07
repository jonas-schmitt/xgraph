"""Plan-length routing: path mode first, another arm for questions its plan calls one-hop.

Path mode answers by executing the relations it writes, which wins on multi-hop questions
and loses one-hop questions to plain one-hop retrieval. The router reads only path mode's
own output: where its first path span has at most `max_relations` relations, or where it
wrote no path span at all (it declined to plan), the question takes the other arm's
answer instead; every other question keeps path mode's. No gold start, chain, hop count
or answer enters the decision. A question the other arm did not answer keeps path mode's
answer.

Each routed record carries `routed` ("paths" or "other"). The summary counts the routed
questions and the tokens both arms spent: a routed question pays for both calls.

    python -m xgraph.route --paths target/mintqa/qwen25-paths-committed-all.jsonl \\
        --other target/mintqa/qwen25-rag-linked-all.jsonl \\
        --out target/mintqa/qwen25-routed.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from xgraph.cost import StageTimer
from xgraph.planning import plan_of


def _relations(text: str) -> int | None:
    """The number of relations in the first path span, None without one."""
    plan = plan_of(text, {})
    return None if plan is None else len(plan)


def route(paths: list[dict], other: list[dict], max_relations: int = 1) -> list[dict]:
    """Path mode's records, each replaced by the other arm's record for the same id where
    path mode wrote no path span or one of at most max_relations relations."""
    by_id = {r["id"]: r for r in other}
    routed = []
    for record in paths:
        relations = _relations(record["text"])
        short = relations is None or relations <= max_relations
        if short and record["id"] in by_id:
            routed.append({**by_id[record["id"]], "routed": "other"})
        else:
            routed.append({**record, "routed": "paths"})
    return routed


def main(argv: list[str] | None = None) -> None:
    from xgraph.provenance import provenance

    parser = argparse.ArgumentParser(prog="python -m xgraph.route", description=__doc__)
    parser.add_argument("--paths", type=Path, required=True, help="path-mode records (JSONL)")
    parser.add_argument("--other", type=Path, required=True, help="the other arm's records")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--max-relations",
        type=int,
        default=1,
        help="route plans of at most this many relations (0: only a missing path)",
    )
    args = parser.parse_args(argv)
    timer = StageTimer()

    def load(path: Path) -> list[dict]:
        with path.open(encoding="utf-8") as lines:
            return [json.loads(line) for line in lines if line.strip()]

    paths, other = load(args.paths), load(args.other)
    routed = route(paths, other, args.max_relations)
    with args.out.open("w", encoding="utf-8") as out:
        for record in routed:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    to_other = {r["id"] for r in routed if r["routed"] == "other"}
    tokens = sum(r.get("prompt_tokens", 0) for r in paths) + sum(
        r.get("prompt_tokens", 0) for r in other if r["id"] in to_other
    )
    summary = {
        "artifact_kind": "xgraph.route",
        "version": 1,
        "questions": len(routed),
        "routed_to_other": len(to_other),
        "max_relations": args.max_relations,
        "prompt_tokens": tokens,
        "provenance": provenance([args.paths, args.other]),
        **timer.summary(),
    }
    Path(f"{args.out}.summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}))


if __name__ == "__main__":
    main()
