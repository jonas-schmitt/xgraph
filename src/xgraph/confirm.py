"""The confirmatory test's analysis, as pre-registered in docs/confirmatory-test.md.

Four arms: A prompted, B prompted and constrained, C fine-tuned, D fine-tuned and
constrained. Five tests, each a paired bootstrap over questions (10,000 resamples, seed
0) whose 99% interval must exclude zero in the hypothesized direction (Bonferroni over
five tests):

- H1: answer accuracy, B > A;
- H2: answer accuracy, D > C;
- H3, prompted: fact-span error rate (pooled), B < A;
- H3, fine-tuned: fact-span error rate (pooled), D < C;
- H4: answer accuracy, D > A.

Answer accuracy is xgraph.score.wwq_accuracy over every question with a gold entity;
the fact-span error rate pools the judge's decisions (`python -m xgraph.gate wwq-facts`).

    python -m xgraph.confirm --gold target/mintaka/gold.json \\
        --labels data/wikidata/mintaka-gold-labels.jsonl \\
        --arm A target/mintaka/A.jsonl target/mintaka/A.facts.json ... \\
        --out eval/mintaka-<date>/confirmatory.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from xgraph.score import (
    load_labels,
    paired_bootstrap,
    paired_ratio_bootstrap,
    record_text,
    wwq_accuracy,
)

LEVEL = 0.99
RESAMPLES = 10_000
SEED = 0
TESTS = (
    ("H1", "accuracy", "B", "A"),
    ("H2", "accuracy", "D", "C"),
    ("H3-prompted", "fact_error", "B", "A"),
    ("H3-fine-tuned", "fact_error", "D", "C"),
    ("H4", "accuracy", "D", "A"),
)


def confirm(
    outputs: dict[str, dict[str, str]],
    decisions: dict[str, dict[str, list[dict]]],
    gold: dict,
    labels: dict[str, set[str]],
) -> dict:
    """The five tests' differences, 99% intervals and verdicts, and each arm's scores."""
    correct = {arm: wwq_accuracy(texts, gold, labels)["correct"] for arm, texts in outputs.items()}
    questions = sorted(next(iter(correct.values())))

    def errors(arm: str) -> list[tuple[float, float]]:
        found = decisions[arm]
        return [
            (sum(not d["supported"] for d in found.get(q, [])), len(found.get(q, [])))
            for q in questions
        ]

    tests = []
    for name, measure, treated, control in TESTS:
        if measure == "accuracy":
            result = paired_bootstrap(
                [float(correct[treated][q]) for q in questions],
                [float(correct[control][q]) for q in questions],
                resamples=RESAMPLES,
                seed=SEED,
                level=LEVEL,
            )
            confirmed = result["low"] > 0
        else:
            result = paired_ratio_bootstrap(
                errors(treated), errors(control), resamples=RESAMPLES, seed=SEED, level=LEVEL
            )
            confirmed = result["high"] < 0
        tests.append(
            {
                "test": name,
                "measure": measure,
                "treated": treated,
                "control": control,
                "difference": result["difference"],
                "low": result["low"],
                "high": result["high"],
                "level": LEVEL,
                "confirmed": bool(confirmed),
            }
        )
    accuracy = {arm: sum(c[q] for q in questions) / len(questions) for arm, c in correct.items()}
    fact_error = {}
    for arm in decisions:
        pairs = errors(arm)
        total = sum(n for _, n in pairs)
        fact_error[arm] = sum(u for u, _ in pairs) / total if total else None
    return {
        "questions": len(questions),
        "accuracy": accuracy,
        "fact_error": fact_error,
        "tests": tests,
    }


def main(argv: list[str] | None = None) -> None:
    from xgraph.provenance import provenance

    parser = argparse.ArgumentParser(prog="python -m xgraph.confirm", description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="gold answer file (JSON)")
    parser.add_argument("--labels", type=Path, required=True, help="gold labels (JSON lines)")
    parser.add_argument(
        "--arm",
        nargs=3,
        action="append",
        required=True,
        metavar=("NAME", "OUTPUTS", "FACTS"),
        help="an arm (A, B, C or D), its outputs (JSONL) and its judged facts (JSON)",
    )
    parser.add_argument("--out", type=Path, required=True, help="result (JSON)")
    args = parser.parse_args(argv)
    arms = {name: (Path(o), Path(f)) for name, o, f in args.arm}
    if sorted(arms) != ["A", "B", "C", "D"]:
        parser.error("give the four arms A, B, C and D")
    gold = json.loads(args.gold.read_text())
    wanted = {e for a in gold["answers"].values() for e in a["entities"]}
    labels = load_labels([args.labels], wanted)
    outputs = {
        arm: {r["id"]: record_text(r) for r in map(json.loads, o.read_text().splitlines()) if r}
        for arm, (o, _) in arms.items()
    }
    decisions = {arm: json.loads(f.read_text())["decisions"] for arm, (_, f) in arms.items()}
    result = confirm(outputs, decisions, gold, labels)
    inputs = [args.gold, args.labels, *(p for pair in arms.values() for p in pair)]
    report = {
        "artifact_kind": "xgraph.confirmatory_test",
        "version": 1,
        **result,
        "provenance": provenance(inputs),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    for t in result["tests"]:
        verdict = "confirmed" if t["confirmed"] else "not confirmed"
        print(
            f"{t['test']}: {t['treated']} - {t['control']} {t['measure']} "
            f"{t['difference']:+.4f} [{t['low']:+.4f}, {t['high']:+.4f}] {verdict}"
        )


if __name__ == "__main__":
    main()
