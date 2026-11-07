"""Compare generation runs written in different span formats (docs/span-format-adr.md).

Each run is scored in its own format: WWQ answer accuracy by label match, answers
that parse, spans per answer by kind, answers cut at the token limit, and tokens per
answer; every run after the first is compared with the first by the paired bootstrap
over questions.

    python -m xgraph.compare --gold eval/wwq-dev-gold-wdqs-2026-10-13.json \\
        --labels data/wikidata/dev-gold-labels.jsonl --out target/formats.json \\
        plain:plain:target/dev/formats/baseline-plain.jsonl \\
        brackets:brackets:target/dev/formats/baseline-brackets.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from xgraph.prompts import final_answer
from xgraph.score import load_labels, paired_bootstrap, record_text, spans, wwq_accuracy
from xgraph.syntax import preset


def _syntax(style: str):
    return None if style == "plain" else preset(style)


def compare(
    runs: list[tuple[str, str, Path]], gold: dict, labels: dict, resamples: int = 10_000
) -> dict:
    """runs: (name, style, outputs) with style a syntax preset or plain; first is the reference."""
    scored, correct = {}, {}
    for name, style, path in runs:
        syntax = _syntax(style)
        records = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
        texts = {r["id"]: record_text(r) for r in records}
        correct[name] = wwq_accuracy(texts, gold, labels, syntax)["correct"]
        kinds = Counter(
            k for r in records for k, _ in (spans(record_text(r), syntax) if syntax else [])
        )
        n = len(records)
        scored[name] = {
            "style": style,
            "answers": n,
            "accuracy": sum(correct[name].values()) / len(correct[name]),
            "parsed": sum(final_answer(record_text(r), syntax) is not None for r in records),
            "fact_spans_per_answer": kinds["fact"] / n,
            "entity_spans_per_answer": kinds["entity"] / n,
            "escaped_spans": kinds["escaped"],
            "cut_at_limit": sum(r["finish_reason"] == "length" for r in records),
            "tokens_per_answer": sum(len(r["token_ids"]) for r in records) / n,
        }
    reference = runs[0][0]
    questions = sorted(correct[reference])
    against = {
        name: paired_bootstrap(
            [float(correct[name][q]) for q in questions],
            [float(correct[reference][q]) for q in questions],
            resamples=resamples,
            seed=0,
        )
        for name, _, _ in runs[1:]
    }
    return {"reference": reference, "runs": scored, "against_reference": against}


def main(argv: list[str] | None = None) -> None:
    from xgraph.provenance import provenance

    parser = argparse.ArgumentParser(prog="python -m xgraph.compare", description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="WWQ gold answer file")
    parser.add_argument("--labels", type=Path, required=True, help="gold entity labels")
    parser.add_argument("--out", type=Path, required=True, help="report (JSON)")
    parser.add_argument("runs", nargs="+", help="name:style:outputs.jsonl, reference first")
    args = parser.parse_args(argv)
    runs = [(n, s, Path(p)) for n, s, p in (r.split(":", 2) for r in args.runs)]
    gold = json.loads(args.gold.read_text())
    wanted = {q for a in gold["answers"].values() for q in a["entities"]}
    report = compare(runs, gold, load_labels([args.labels], wanted))
    report["artifact_kind"] = "xgraph.format_comparison"
    report["version"] = 1
    report["provenance"] = provenance([args.gold, args.labels, *(p for _, _, p in runs)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    for name, run in report["runs"].items():
        print(name, round(run["accuracy"], 3), report["against_reference"].get(name, ""))


if __name__ == "__main__":
    main()
