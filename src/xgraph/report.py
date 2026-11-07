"""The quality gate's report: scores of both arms and the pre-registered pass rule.

The gate passes when (1) the constrained arm's error rate is lower than the
baseline's on both the WWQ fact-span error rate and the FActScore error rate, each
with a 95% interval of the paired difference that excludes zero, and (2) WWQ answer
accuracy is non-inferior: the lower bound of the paired difference's interval lies
above -2 percentage points (`docs/quality-gate.md`).

    python -m xgraph.report --gate target/gate --gold eval/wwq-gold-wdqs-2026-10-12.json \\
        --labels data/wikidata/gold-labels.jsonl --out target/gate/report.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from xgraph.prompts import final_answer
from xgraph.score import (
    load_labels,
    names_gold,
    paired_bootstrap,
    paired_ratio_bootstrap,
    record_text,
    spans,
    wwq_accuracy,
)

NON_INFERIORITY = -0.02
SEED = 0


def gate_verdict(*, accuracy: dict, fact_errors: dict, factscore_errors: dict) -> dict:
    """Paired differences are constrained minus baseline."""
    fewer = fact_errors["high"] < 0 and factscore_errors["high"] < 0
    holds = accuracy["low"] > NON_INFERIORITY
    return {"fewer_errors": fewer, "accuracy_not_lower": holds, "passed": fewer and holds}


def span_counts(texts: list[str]) -> dict:
    kinds = Counter(kind for text in texts for kind, _ in spans(text))
    total = sum(kinds.values())
    return {
        "answers": len(texts),
        "entity": kinds["entity"],
        "fact": kinds["fact"],
        "escaped": kinds["escaped"],
        "spans_per_answer": total / len(texts) if texts else 0.0,
        "unresolved_rate": kinds["escaped"] / total if total else 0.0,
    }


def missing_facts(outputs: dict[str, str], gold: dict, labels: dict[str, set[str]]) -> dict:
    """Final answers after the linking facts were removed: correct, grounded but wrong, or
    escaped (an escaped answer span, or none)."""
    counts = {"correct": 0, "grounded_wrong": 0, "escaped": 0}
    for question, text in outputs.items():
        answer = final_answer(text)
        if answer is None:
            counts["escaped"] += 1
        elif names_gold(answer, gold["answers"][question]["entities"], labels):
            counts["correct"] += 1
        else:
            counts["grounded_wrong"] += 1
    return counts


def changed_facts(
    outputs: dict[str, str], swaps: dict, gold: dict, labels: dict[str, set[str]]
) -> dict:
    """Final answers after the gold triple moved: the KG's new answer, the original, other."""
    counts = {"kg": 0, "original": 0, "other": 0}
    for question, text in outputs.items():
        answer = final_answer(text)
        new = f"Q{swaps[question][3]}"
        if names_gold(answer, [new], labels):
            counts["kg"] += 1
        elif names_gold(answer, gold["answers"][question]["entities"], labels):
            counts["original"] += 1
        else:
            counts["other"] += 1
    return counts


def entity_accuracy(outputs: dict[str, str], gold: dict, entity_labels: Path) -> dict:
    """Answers whose label exactly one KG entity has, and that entity gold.

    Answers whose label several KG entities share are counted apart; they never count
    as entity-correct. Questions without a gold entity are excluded.
    """
    scored = {q: a for q, a in gold["answers"].items() if a["entities"]}
    answers = {q: final_answer(outputs[q]) for q in scored}
    texts = {a for a in answers.values() if a}
    owners: dict[str, set[str]] = {}
    with entity_labels.open(encoding="utf-8") as lines:
        for line in lines:
            item, label, _ = json.loads(line)
            if label in texts:
                owners.setdefault(label, set()).add(item)
    correct = shared = 0
    for question, answer in answers.items():
        found = owners.get(answer or "", set())
        if len(found) > 1:
            shared += 1
        elif found and next(iter(found)) in scored[question]["entities"]:
            correct += 1
    return {
        "scored": len(scored),
        "correct": correct,
        "shared_label": shared,
        "accuracy": correct / len(scored) if scored else 0.0,
    }


def _outputs(path: Path) -> dict[str, str]:
    return {r["id"]: record_text(r) for r in map(json.loads, path.read_text().splitlines()) if r}


def build_report(gate: Path, gold: dict, labels_path: Path) -> dict:
    arms = ("baseline", "constrained")
    texts = {arm: _outputs(gate / f"wwq-{arm}.jsonl") for arm in arms}
    wanted = {q for a in gold["answers"].values() for q in a["entities"]}
    labels = load_labels([labels_path], wanted)
    accuracy = {arm: wwq_accuracy(texts[arm], gold, labels) for arm in arms}
    scored = sorted(accuracy["baseline"]["correct"])
    correct = {arm: [float(accuracy[arm]["correct"][q]) for q in scored] for arm in arms}

    facts = {
        arm: json.loads((gate / f"wwq-{arm}.facts.json").read_text())["decisions"] for arm in arms
    }
    questions = sorted(set(facts["baseline"]) & set(facts["constrained"]))
    pooled = {
        arm: [
            (sum(not d["supported"] for d in facts[arm][q]), len(facts[arm][q])) for q in questions
        ]
        for arm in arms
    }
    bios = {
        arm: json.loads((gate / f"bio-{arm}.factscore.json").read_text())["scores"] for arm in arms
    }
    topics = sorted(set(bios["baseline"]) & set(bios["constrained"]))
    bio_errors = {arm: [1.0 - bios[arm][t]["score"] for t in topics] for arm in arms}

    accuracy_ci = paired_bootstrap(correct["constrained"], correct["baseline"], seed=SEED)
    fact_ci = paired_ratio_bootstrap(pooled["constrained"], pooled["baseline"], seed=SEED)
    bio_ci = paired_bootstrap(bio_errors["constrained"], bio_errors["baseline"], seed=SEED)
    return {
        "artifact_kind": "xgraph.gate_report",
        "version": 1,
        "verdict": gate_verdict(accuracy=accuracy_ci, fact_errors=fact_ci, factscore_errors=bio_ci),
        "wwq_accuracy": {
            **{arm: sum(correct[arm]) / len(scored) for arm in arms},
            "paired": accuracy_ci,
            "scored": len(scored),
            "excluded": len(accuracy["baseline"]["excluded"]),
        },
        "wwq_fact_span_error_rate": fact_ci,
        "factscore_error_rate": {
            **{arm: sum(bio_errors[arm]) / len(topics) for arm in arms},
            "paired": bio_ci,
            "biographies": len(topics),
        },
        "spans": {arm: span_counts(list(texts[arm].values())) for arm in arms},
    }


def probe_report(probes: Path, outputs: Path, gold: dict, gold_labels: Path, kg: Path) -> dict:
    """The robustness probes' counts per arm (scripts/gate-probes.sh outputs)."""
    swaps = json.loads((probes / "probes.json").read_text())["changed_facts"]
    wanted = {q for a in gold["answers"].values() for q in a["entities"]}
    new = {f"Q{s[3]}" for s in swaps.values()}
    labels = load_labels([gold_labels], wanted)
    labels.update(load_labels([kg / "entity_labels.jsonl"], new))
    result: dict = {"missing_facts": {}, "changed_facts": {}}
    for arm in ("wwq-constrained", "rag-constrained"):
        path = outputs / f"missing-{arm}.jsonl"
        if path.exists():
            result["missing_facts"][arm] = missing_facts(_outputs(path), gold, labels)
    for arm in ("wwq-baseline", "wwq-constrained", "rag-baseline", "rag-constrained"):
        path = outputs / f"changed-{arm}.jsonl"
        if arm == "wwq-baseline":  # the gate's own baseline run, on the probe's questions
            path = outputs.parent / "gate" / "wwq-baseline.jsonl"
        if path.exists():
            texts = {q: s for q, s in _outputs(path).items() if q in swaps}
            result["changed_facts"][arm] = changed_facts(texts, swaps, gold, labels)
    return result


def reported(gate: Path, gold: dict, gold_labels: Path, kg: Path) -> dict:
    """Reported but not gated: entity-level accuracy, and the retrieval arms' scores."""
    wanted = {q for a in gold["answers"].values() for q in a["entities"]}
    labels = load_labels([gold_labels], wanted)
    result: dict = {"entity_level_accuracy": {}, "retrieval_arms": {}}
    for name in ("wwq-baseline", "wwq-constrained", "wwq-rag-baseline", "wwq-rag-constrained"):
        path = gate / f"{name}.jsonl"
        if not path.exists():
            continue
        texts = _outputs(path)
        result["entity_level_accuracy"][name] = entity_accuracy(
            texts, gold, kg / "entity_labels.jsonl"
        )
        if "rag" in name:
            correct = wwq_accuracy(texts, gold, labels)["correct"]
            arm: dict = {"accuracy": sum(correct.values()) / len(correct)}
            facts = gate / f"{name}.facts.json"
            if facts.exists():
                decisions = json.loads(facts.read_text())["decisions"].values()
                spans_judged = [d for answer in decisions for d in answer]
                arm["fact_span_error_rate"] = sum(not d["supported"] for d in spans_judged) / max(
                    len(spans_judged), 1
                )
            arm["spans"] = span_counts(list(texts.values()))
            result["retrieval_arms"][name] = arm
    return result


def main(argv: list[str] | None = None) -> None:
    from xgraph.provenance import provenance

    parser = argparse.ArgumentParser(prog="python -m xgraph.report", description=__doc__)
    parser.add_argument("--gate", type=Path, required=True, help="gate output directory")
    parser.add_argument("--gold", type=Path, required=True, help="gold answer file")
    parser.add_argument("--labels", type=Path, required=True, help="gold entity labels")
    parser.add_argument("--out", type=Path, required=True, help="report (JSON)")
    parser.add_argument("--probes", type=Path, help="python -m xgraph.probes output directory")
    parser.add_argument("--probe-outputs", type=Path, help="scripts/gate-probes.sh outputs")
    parser.add_argument("--kg", type=Path, help="gate KG directory, for the probes' labels")
    args = parser.parse_args(argv)
    gold = json.loads(args.gold.read_text())
    report = build_report(args.gate, gold, args.labels)
    if args.kg:
        report["reported"] = reported(args.gate, gold, args.labels, args.kg)
    if args.probes:
        report["probes"] = probe_report(args.probes, args.probe_outputs, gold, args.labels, args.kg)
    report["provenance"] = provenance([args.gold, args.labels, *sorted(args.gate.glob("*.json*"))])
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["verdict"]))


if __name__ == "__main__":
    main()
