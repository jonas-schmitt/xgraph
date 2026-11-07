"""KG index statistics: how wide the span masks are, and where a label has one way on.

The measures follow the feasibility study's CoDEx-L table. The entity-mode ones
(label_index_stats) are computed over distinct labels, as token sequences under
the served tokenizer; the fact-mode ones (fact_stats) over the writable triples.
Both count only label tokens: delimiters are left out.

    python -m xgraph.stats --codex data/codex \\
        --tokenizer data/tokenizers/qwen3.8/tokenizer.json \\
        --out target/reports/codex-l-label-index.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from xgraph.index import KGIndex, LabelIndex

DEFINITIONS = {
    "tokens_per_label": "label length in tokens, over distinct labels (p95 by linear "
    "interpolation)",
    "first_token_choices": "distinct first tokens of all labels: the allowed label tokens "
    "right after the span opener",
    "first_token_share_of_vocabulary": "first_token_choices divided by the served "
    "tokenizer's vocabulary size",
    "choices_one_token_in": "for each distinct first token, how many distinct second tokens "
    "follow it",
    "mean_choices_two_tokens_in": "the same, for each distinct two-token prefix",
    "mean_choices_three_or_more_tokens_in": "the same, for each distinct prefix of three or "
    "more tokens",
    "single_continuation_share": "share of label token positions, over distinct labels, at "
    "which the prefix typed so far has exactly one label continuation and is not a complete "
    "label; the escape marker stays allowed, so the token is not forced, but a KG draft could "
    "propose it",
    "shared_labels": "labels carried by more than one entity, and how many entities carry them",
}

FACT_DEFINITIONS = {
    "subjects": "entities with at least one writable triple",
    "out_degree": "for each such subject, its distinct (relation, object) pairs",
    "objects_per_pair": "for each subject and relation, its distinct objects",
    "single_object_pair_share": "share of (subject, relation) pairs with exactly one object",
    "single_continuation_object_share": "share of object label token positions, over every "
    "pair and each label of its objects, at which the pair's object labels typed so far have "
    "exactly one label continuation and no complete label; not forced, since the escape marker "
    "stays allowed, but a KG draft could propose it",
}


def _mean(values: list[int]) -> float | None:
    return statistics.fmean(values) if values else None


def label_index_stats(index: LabelIndex, vocab_size: int | None = None) -> dict:
    """The measures in DEFINITIONS; a measure with nothing to measure is None."""
    labels = sorted({index.row_tokens(r) for r in range(index.num_rows)})
    entities_of: dict[tuple[int, ...], set[str]] = defaultdict(set)
    for row in range(index.num_rows):
        entities_of[index.row_tokens(row)].add(index.owner_id(row))
    continuations: dict[tuple[int, ...], set[int]] = defaultdict(set)
    for label in labels:
        for depth, token in enumerate(label):
            continuations[label[:depth]].add(token)
    complete = set(labels)

    def choices(depths) -> list[int]:
        return [len(c) for prefix, c in continuations.items() if len(prefix) in depths]

    lengths = [len(label) for label in labels]
    one_in = choices({1})
    deeper = choices(range(3, max(lengths, default=0) + 1))
    positions = [label[:depth] for label in labels for depth in range(len(label))]
    single = sum(1 for p in positions if len(continuations[p]) == 1 and p not in complete)
    shared = [entities for entities in entities_of.values() if len(entities) > 1]
    first = len(continuations[()])
    return {
        "rows": index.num_rows,
        "distinct_labels": len(labels),
        "excluded_labels": len(index.excluded),
        "tokens_per_label": {
            "mean": statistics.fmean(lengths),
            "median": statistics.median(lengths),
            "p95": float(np.percentile(lengths, 95)),
            "max": max(lengths),
        }
        if lengths
        else None,
        "first_token_choices": first,
        "first_token_share_of_vocabulary": first / vocab_size if vocab_size else None,
        "choices_one_token_in": {"mean": statistics.fmean(one_in), "max": max(one_in)}
        if one_in
        else None,
        "mean_choices_two_tokens_in": _mean(choices({2})),
        "mean_choices_three_or_more_tokens_in": _mean(deeper),
        "single_continuation_share": single / len(positions) if positions else None,
        "shared_labels": {"labels": len(shared), "entities": sum(len(e) for e in shared)},
    }


def _spread(values: list[int]) -> dict | None:
    if not values:
        return None
    return {
        "median": float(np.median(values)),
        "p99": float(np.percentile(values, 99)),
        "max": max(values),
    }


def fact_stats(index: KGIndex) -> dict:
    """The fact-mode measures in FACT_DEFINITIONS, over the writable triples."""
    labels = index.labels
    degrees: list[int] = []
    per_pair: list[int] = []
    positions = single = 0
    for entity in range(len(labels.owner_ids)):
        start, stop = index.adjacency_offsets[entity], index.adjacency_offsets[entity + 1]
        if start == stop:
            continue
        objects: dict[int, set[int]] = defaultdict(set)  # relation -> object rows
        for edge in range(start, stop):
            objects[int(index.edge_relations[edge])].add(int(index.edge_objects[edge]))
        degrees.append(sum(len({labels.owners[r] for r in rows}) for rows in objects.values()))
        for rows in objects.values():
            per_pair.append(len({labels.owners[r] for r in rows}))
            written = [labels.row_tokens(r) for r in rows]
            continuations: dict[tuple[int, ...], set[int]] = defaultdict(set)
            for label in written:
                for depth, token in enumerate(label):
                    continuations[label[:depth]].add(token)
            complete = set(written)
            for label in written:
                for depth in range(len(label)):
                    prefix = label[:depth]
                    positions += 1
                    single += len(continuations[prefix]) == 1 and prefix not in complete
    return {
        "subjects": len(degrees),
        "out_degree": _spread(degrees),
        "objects_per_pair": _spread(per_pair),
        "single_object_pair_share": (
            sum(n == 1 for n in per_pair) / len(per_pair) if per_pair else None
        ),
        "single_continuation_object_share": single / positions if positions else None,
    }


def main(argv: list[str] | None = None) -> None:
    from xgraph.codex import load_codex
    from xgraph.index import build_kg_index
    from xgraph.provenance import provenance
    from xgraph.syntax import SpanSyntax
    from xgraph.tokenizer import ServedTokenizer

    parser = argparse.ArgumentParser(prog="python -m xgraph.stats", description=__doc__)
    parser.add_argument("--codex", type=Path, required=True, help="CoDEx data directory")
    parser.add_argument("--size", default="l", choices=("s", "m", "l"))
    parser.add_argument("--tokenizer", type=Path, required=True, help="served tokenizer.json")
    parser.add_argument("--out", type=Path, required=True, help="report file (JSON)")
    parser.add_argument(
        "--label-prefix",
        default=SpanSyntax().label_prefix,
        help="text written before each label (default: a space)",
    )
    args = parser.parse_args(argv)
    syntax = SpanSyntax(label_prefix=args.label_prefix)

    tokenizer = ServedTokenizer.from_file(args.tokenizer)
    index = build_kg_index(load_codex(args.codex, args.size), tokenizer, syntax)
    triples = sorted((args.codex / "triples" / f"codex-{args.size}").glob("*.txt"))
    relations = args.codex / "relations" / "en" / "relations.json"
    report = {
        "artifact_kind": "xgraph.kg_index_stats",
        "version": 2,
        "kg": f"codex-{args.size}",
        "span_syntax": dataclasses.asdict(syntax),
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "stats": label_index_stats(index.labels, tokenizer.vocab_size),
        "fact_stats": fact_stats(index),
        "definitions": DEFINITIONS | FACT_DEFINITIONS,
        "provenance": provenance(
            [args.codex / "entities" / "en" / "entities.json", relations, *triples, args.tokenizer]
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"stats": report["stats"], "fact_stats": report["fact_stats"]}, indent=2))


if __name__ == "__main__":
    main()
