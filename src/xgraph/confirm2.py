"""The second confirmatory test's analysis, as pre-registered in docs/confirmatory-test-2.md.

Per model family, arms A (prompted), F (the fallback arm with literal values and the
no-repeat rule), and L (F's constrained pass). Two tests, each a paired bootstrap over
questions (10,000 resamples, seed 0) at the 99% level (Bonferroni over four tests, two
per model):

- H1, no accuracy cost: semantic accuracy, F - A, with the interval's low end above
  -1.5 points (non-inferiority);
- H2, fewer unsupported facts: fact-span error rate (pooled) of the delivered answers,
  F - A < 0 (amended before F's outputs existed; L's rate was tested before).

Reported, not tested: the share of answers both semantically correct and grounded at
the entity level (xgraph.grounding's per_question_entity), for F and for A; L's
fact-span error rate (the mechanism). Every input must hold a judgment for every
question: a missing judgment is an error, not an answer without facts.

    python -m xgraph.confirm2 --model qwen3.8-27b \\
        --semantic A target/c2/A.judged.json --semantic F target/c2/F.judged.json \\
        --grounding A target/c2/A.grounding.json --grounding F target/c2/F.grounding.json \\
        --facts A target/c2/A.facts.json --facts F target/c2/F.facts.json \\
        --out eval/mintaka-train-<date>/qwen.json

`python -m xgraph.confirm2 constrained-pass` writes L's outputs from F's records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from xgraph.score import paired_bootstrap, paired_ratio_bootstrap

LEVEL = 0.99
RESAMPLES = 10_000
SEED = 0
MARGIN = 0.015
TESTS = (
    ("H1", "semantic_accuracy", "F", "A"),
    ("H2", "fact_error", "F", "A"),
)


def constrained_pass(records: list[dict]) -> list[dict]:
    """L's records: each fallback record's constrained text, other records as they are."""
    return [
        {
            **r,
            "text": r["constrained_text"],
            "token_ids": r.get("constrained_token_ids"),
            "finish_reason": r.get("constrained_finish_reason"),
        }
        if r.get("fallback")
        else dict(r)
        for r in records
    ]


def confirm2(
    semantic: dict[str, dict[str, bool]],
    grounded: dict[str, dict[str, bool]],
    decisions: dict[str, dict[str, list[dict]]],
) -> dict:
    """The two tests' differences, 99% intervals and verdicts, and the arms' scores."""
    questions = sorted(semantic["A"])
    for name, table in (("semantic", semantic), ("grounded", grounded), ("facts", decisions)):
        for arm, judged in table.items():
            missing = set(questions) - set(judged)
            if missing:
                raise ValueError(f"{name} of arm {arm}: {len(missing)} questions missing")
    verified = {
        arm: {q: bool(semantic[arm][q] and grounded[arm][q]) for q in questions}
        for arm in ("A", "F")
    }

    def errors(arm: str) -> list[tuple[float, float]]:
        found = decisions[arm]
        return [(sum(not d["supported"] for d in found[q]), len(found[q])) for q in questions]

    tests = []
    for name, measure, treated, control in TESTS:
        if measure == "fact_error":
            result = paired_ratio_bootstrap(
                errors(treated), errors(control), resamples=RESAMPLES, seed=SEED, level=LEVEL
            )
            confirmed = result["high"] < 0
        else:
            result = paired_bootstrap(
                [float(semantic[treated][q]) for q in questions],
                [float(semantic[control][q]) for q in questions],
                resamples=RESAMPLES,
                seed=SEED,
                level=LEVEL,
            )
            confirmed = result["low"] > -MARGIN
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
                "margin": MARGIN if name == "H1" else 0.0,
                "confirmed": bool(confirmed),
            }
        )
    fact_error = {}
    for arm in decisions:
        pairs = errors(arm)
        total = sum(n for _, n in pairs)
        fact_error[arm] = sum(u for u, _ in pairs) / total if total else None
    return {
        "questions": len(questions),
        "semantic_accuracy": {
            arm: sum(semantic[arm][q] for q in questions) / len(questions) for arm in semantic
        },
        "verified_correct": {arm: sum(verified[arm].values()) / len(questions) for arm in verified},
        "fact_error": fact_error,
        "tests": tests,
    }


def _pairs(values: list[list[str]]) -> dict[str, Path]:
    return {arm: Path(path) for arm, path in values}


def main(argv: list[str] | None = None) -> None:
    from xgraph.provenance import provenance

    if argv is None:
        import sys

        argv = sys.argv[1:]
    if argv[:1] == ["constrained-pass"]:
        parser = argparse.ArgumentParser(prog="python -m xgraph.confirm2 constrained-pass")
        parser.add_argument("--fallback", type=Path, required=True, help="F's records (JSONL)")
        parser.add_argument("--out", type=Path, required=True, help="L's records (JSONL)")
        args = parser.parse_args(argv[1:])
        records = [json.loads(line) for line in args.fallback.read_text().splitlines() if line]
        with args.out.open("w", encoding="utf-8") as out:
            for record in constrained_pass(records):
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        return
    parser = argparse.ArgumentParser(prog="python -m xgraph.confirm2", description=__doc__)
    parser.add_argument("--model", required=True, help="the model family's name")
    for name, what in (
        ("semantic", "python -m xgraph.gate wwq-judged, for A and F"),
        ("grounding", "python -m xgraph.grounding, for A and F"),
        ("facts", "python -m xgraph.gate wwq-facts, for A and F (and L, reported)"),
    ):
        parser.add_argument(
            f"--{name}", nargs=2, action="append", required=True, metavar=("ARM", "PATH"), help=what
        )
    parser.add_argument("--out", type=Path, required=True, help="result (JSON)")
    args = parser.parse_args(argv)
    semantic_paths, grounding_paths = _pairs(args.semantic), _pairs(args.grounding)
    facts_paths = _pairs(args.facts)
    if sorted(semantic_paths) != ["A", "F"] or sorted(grounding_paths) != ["A", "F"]:
        parser.error("give --semantic and --grounding for arms A and F")
    if not {"A", "F"} <= set(facts_paths) <= {"A", "F", "L"}:
        parser.error("give --facts for arms A and F (and optionally L)")
    semantic = {
        arm: {q: d["correct"] for q, d in json.loads(p.read_text())["decisions"].items()}
        for arm, p in semantic_paths.items()
    }
    grounded = {  # at the entity level: through a gold entity's own fact
        arm: json.loads(p.read_text())["per_question_entity"] for arm, p in grounding_paths.items()
    }
    decisions = {arm: json.loads(p.read_text())["decisions"] for arm, p in facts_paths.items()}
    result = confirm2(semantic, grounded, decisions)
    inputs = [*semantic_paths.values(), *grounding_paths.values(), *facts_paths.values()]
    report = {
        "artifact_kind": "xgraph.confirmatory_test_2",
        "version": 1,
        "model": args.model,
        **result,
        "provenance": provenance(inputs),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    for t in result["tests"]:
        print(
            f"{args.model} {t['test']}: {t['difference']:+.4f} "
            f"[{t['low']:+.4f}, {t['high']:+.4f}] confirmed={t['confirmed']}"
        )


if __name__ == "__main__":
    main()
