"""Recall-aware long-form scores, beside FActScore's precision.

A text can raise its share of supported facts by stating fewer of them. Two scores
count what it gets right instead: supported facts per text, and F1@K (Wei et al.,
2024, "Long-form factuality in large language models"): the harmonic mean of the
text's precision (supported facts over facts) and its recall min(supported / K, 1),
with K the median number of facts the control arm states, and 0 for a text without a
supported fact. Both are compared between two arms of `python -m xgraph.gate
factscore` with the paired bootstrap over topics.

    python -m xgraph.longform --treated target/gate/bio-constrained.factscore.json \\
        --control target/gate/bio-baseline.factscore.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from xgraph.score import paired_bootstrap


def f1_at_k(supported: list[bool], k: float) -> float:
    hits = sum(supported)
    if hits == 0:
        return 0.0
    precision = hits / len(supported)
    recall = min(hits / k, 1.0)
    return 2 * precision * recall / (precision + recall)


def _arm(scores: dict, topics: list[str], k: float) -> dict:
    return {
        "facts_per_text": statistics.mean(len(scores[t]["facts"]) for t in topics),
        "supported_per_text": statistics.mean(sum(scores[t]["supported"]) for t in topics),
        "f1_at_k": statistics.mean(f1_at_k(scores[t]["supported"], k) for t in topics),
    }


def compare(treated: dict, control: dict, k: float | None = None) -> dict:
    """Both arms' scores over their common topics, and the paired differences."""
    topics = sorted(set(treated) & set(control))
    if k is None:
        k = statistics.median(len(control[t]["facts"]) for t in topics)
    return {
        "topics": len(topics),
        "k": k,
        "treated": _arm(treated, topics, k),
        "control": _arm(control, topics, k),
        "supported_difference": paired_bootstrap(
            [sum(treated[t]["supported"]) for t in topics],
            [sum(control[t]["supported"]) for t in topics],
        ),
        "f1_difference": paired_bootstrap(
            [f1_at_k(treated[t]["supported"], k) for t in topics],
            [f1_at_k(control[t]["supported"], k) for t in topics],
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m xgraph.longform", description=__doc__)
    parser.add_argument("--treated", type=Path, required=True, help="FActScore result (JSON)")
    parser.add_argument("--control", type=Path, required=True, help="FActScore result (JSON)")
    parser.add_argument("--k", type=float, help="F1@K's K (default: the control's median)")
    parser.add_argument("--out", type=Path, help="result (JSON)")
    args = parser.parse_args(argv)
    result = compare(
        json.loads(args.treated.read_text())["scores"],
        json.loads(args.control.read_text())["scores"],
        args.k,
    )
    from xgraph.provenance import provenance

    record = {
        "artifact_kind": "xgraph.longform_comparison",
        "version": 1,
        **result,
        "provenance": provenance([args.treated, args.control]),
    }
    text = json.dumps(record, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
