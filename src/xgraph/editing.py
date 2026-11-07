"""Scoring answers against an edited or updated KG.

For MQuAKE (python -m xgraph.mquake) and real updates (python -m xgraph.updates) each
question has a new answer (the edited or updated KG's) and an old one (the original
KG's, or the value before the update): an answer names the new one, the old one, or
neither (by label or alias, xgraph.score). Reported overall, per group (the questions'
complexity field: hops, update or control), and for MQuAKE per case (`cases`), a case
being one question asked as paraphrases: `new_any`, the share of cases any of whose
questions gets the new answer (MQuAKE's measure); `new_all`, every question;
`new_majority`, more than half; `histogram`, cases counted by questions right, keyed
"<right>/<asked>"; and `voted`, the share whose answer voted over its questions (as
xgraph.vote does, a tie going to the earliest question given) names the case's new
answer. A question whose gold failed (world gold, `python -m xgraph.world_gold`, when
its chain stops in the KG) counts as missed and adds no gold to its case's vote.

Without --gold-old, as for a benchmark with one answer per question, only the new
answer is scored and `old` is left out.

With --compare, `compare` gives both arms' new_any and its paired difference with a
bootstrap interval over cases (difference, low, high), and the same for the share of
questions (`per_question`), for `new_all`, and for `voted`. With --cluster, a JSON object
mapping each case id (or each question id, a case's questions in one cluster) to a
cluster id, every interval resamples whole clusters instead of cases, both arms together,
and `compare.clusters` counts them: cases in one cluster, such as questions sharing a
relation sequence, are not independent. `python -m xgraph.editing clusters` writes that
file for MQuAKE, the asked cases keyed by their edited chain's relation sequence
(xgraph.mquake.relation_sequence), and `python -m xgraph.editing panel` draws a seeded
panel of cases from it that spans every cluster, by the square root of its size (a
calibration panel), whose case list --cases reads. With --interaction <first cases>
<second cases>, two JSON lists of case ids such as recall strata (python -m xgraph.recall),
`compare.interaction` gives the arm by stratum interaction: the difference a minus b on
the first stratum minus that on the second, for the case measure (`case_measure`) and the
share of questions (`per_question`), each stratum's a, b and difference beside it, with a
paired bootstrap interval drawing cases (or clusters, with --cluster) from both strata
together. Listed cases that are not compared are left out; `cases` counts those that are.

    python -m xgraph.editing --outputs target/mqr-cf-3k/derive.jsonl \\
        --questions target/mqr-cf-3k/questions.json --gold-new target/mqr-cf-3k/gold-new.json \\
        --gold-old target/mqr-cf-3k/gold-old.json --labels target/mqr-cf-3k/labels.jsonl \\
        --out target/mqr-cf-3k/derive.editing.json
    python -m xgraph.editing clusters --data data/mquake/MQuAKE-Remastered-CF-3k.json \\
        --questions target/mqr-cf-3k/questions.json --out target/mqr-cf-3k/clusters.json
    python -m xgraph.editing panel --clusters target/mqr-cf-3k/clusters.json --size 512 \\
        --seed 0 --out target/mqr-cf-3k/panel-cases.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Collection
from pathlib import Path


def case_members(questions: list[dict]) -> dict[str, list[str]]:
    """Each MQuAKE case's question ids, in the order the questions are given."""
    from xgraph.mquake import case_id

    members: dict[str, list[str]] = {}
    for qid in dict.fromkeys(q["id"] for q in questions):
        members.setdefault(case_id(qid), []).append(qid)
    return members


def voted_cases(
    texts: dict[str, str],
    members: dict[str, list[str]],
    gold_new: dict,
    labels: dict[str, set[str]],
) -> dict[str, bool]:
    """Whether each case's answer voted over its questions' final answers (xgraph.vote:
    normalized, unanswered questions do not vote, a tie goes to the earliest question)
    names the case's new answer: any gold entity of its questions with a gold answer."""
    from xgraph.score import wwq_accuracy
    from xgraph.vote import vote

    records = [  # vote groups by the id before the last "#"
        {"id": f"{case}#{j}", "text": texts.get(qid, "")}
        for case, qids in members.items()
        for j, qid in enumerate(qids)
    ]
    winners = {r["id"]: r["text"] for r in vote(records)}
    answers = {}
    for case, qids in members.items():
        gold = [gold_new["answers"][q] for q in qids if q in gold_new["answers"]]
        entities = sorted({e for g in gold for e in g["entities"]})
        answers[case] = {"entities": entities, "literals": []}
    found = wwq_accuracy(winners, {"answers": answers, "failed": {}}, labels)["correct"]
    return {case: found.get(case, False) for case in members}


def case_panel(
    right: dict[str, bool], members: dict[str, list[str]], voted: dict[str, bool]
) -> dict:
    """Shares of cases all or more than half of whose questions get the new answer, cases
    counted by questions right ("<right>/<asked>"), and the share whose vote is right."""
    counts = [(sum(right[q] for q in qids), len(qids)) for qids in members.values()]
    histogram = {f"{r}/{k}": 0 for k in sorted({k for _, k in counts}) for r in range(k + 1)}
    for r, k in counts:
        histogram[f"{r}/{k}"] += 1
    n = len(counts)
    return {
        "new_all": sum(r == k for r, k in counts) / n if n else 0.0,
        "new_majority": sum(2 * r > k for r, k in counts) / n if n else 0.0,
        "histogram": histogram,
        "voted": sum(voted[c] for c in members) / n if n else 0.0,
    }


def edit_scores(
    texts: dict[str, str],
    questions: list[dict],
    gold_new: dict,
    gold_old: dict | None,
    labels: dict[str, set[str]],
) -> dict:
    """New-answer scores, and old-answer ones when gold_old is given."""
    from xgraph.mquake import by_case
    from xgraph.score import wwq_accuracy

    asked = {q["id"]: q for q in questions}
    outputs = {qid: texts.get(qid, "") for qid in asked}

    def only_asked(gold: dict) -> dict:
        answers = {q: a for q, a in gold["answers"].items() if q in asked}
        failed = {q: f for q, f in gold.get("failed", {}).items() if q in asked}
        return {**gold, "answers": answers, "failed": failed}

    def correct(gold: dict) -> dict[str, bool]:  # a question whose gold failed counts as missed
        found = wwq_accuracy(outputs, only_asked(gold), labels)["correct"]
        return {qid: found.get(qid, False) for qid in asked}

    new = correct(gold_new)
    old = correct(gold_old) if gold_old is not None else None
    groups: dict[str, list[str]] = {}
    for qid, question in asked.items():
        groups.setdefault(question.get("complexity", "all"), []).append(qid)

    def share(flags: dict[str, bool], ids: list[str]) -> float:
        return sum(flags[q] for q in ids) / len(ids) if ids else 0.0

    ids = sorted(asked)

    def shares(members: list[str]) -> dict[str, float]:
        found = {"new": share(new, members)}
        return found if old is None else {**found, "old": share(old, members)}

    result = {
        "questions": len(ids),
        **shares(ids),
        "by_group": {
            group: {"questions": len(members), **shares(members)}
            for group, members in sorted(groups.items())
        },
    }
    if all(qid.startswith("mquake-") for qid in ids):
        cases = by_case({qid: new[qid] for qid in ids})
        members = case_members(questions)
        voted = voted_cases(outputs, members, only_asked(gold_new), labels)
        result["cases"] = {
            "cases": len(cases),
            "new_any": share(cases, sorted(cases)),
            **case_panel(new, members, voted),
        }
    return result


def case_clusters(
    clusters: dict[str, str | int], members: dict[str, list[str]], ids: list[str]
) -> list[str | int]:
    """Each case's cluster, by its case id or else by its questions' ids, which must then
    all be given and all name the same cluster."""
    found = []
    for case in ids:
        if case in clusters:
            label = clusters[case]
        else:
            named = {q: clusters[q] for q in members[case] if q in clusters}
            if len(named) != len(members[case]) or len(set(named.values())) != 1:
                raise ValueError(
                    f"case {case} has no cluster: neither it nor all of its questions "
                    f"{members[case]} are in the cluster file, in one cluster (given: {named})"
                )
            label = next(iter(named.values()))
        if not isinstance(label, (str, int)):
            raise ValueError(f"case {case}'s cluster {label!r} is not a string or a number")
        found.append(label)
    return found


def compare_cases(
    a: dict[str, str],
    b: dict[str, str],
    questions: list[dict],
    gold_new: dict,
    gold_old: dict | None,
    labels: dict[str, set[str]],
    resamples: int = 10_000,
    clusters: dict[str, str | int] | None = None,
    interaction: tuple[Collection[str], Collection[str]] | None = None,
) -> dict:
    """Two arms' MQuAKE case measure (a case counts when any of its questions gets the new
    answer) and their difference, a minus b, with a paired bootstrap interval over cases:
    a case's questions are paraphrases, so the case is the unit. per_question gives the
    share of questions with the new answer and its difference, also resampled by case;
    new_all (every question of a case gets the new answer) and voted (the case's answer
    voted over its questions, case_panel) the same as the case measure. With clusters
    (case or question id to cluster id, case_clusters), every interval resamples whole
    clusters of cases instead. With interaction (two disjoint sets of case ids), the arm
    by stratum interaction (strata_interaction) is given too."""
    from xgraph.mquake import by_case
    from xgraph.score import paired_bootstrap, paired_ratio_bootstrap, wwq_accuracy

    asked = {q["id"] for q in questions}
    answers = {**gold_new, "answers": {q: v for q, v in gold_new["answers"].items() if q in asked}}

    def correct(texts: dict[str, str]) -> dict[str, bool]:
        found = wwq_accuracy({q: texts.get(q, "") for q in asked}, answers, labels)["correct"]
        return {q: found.get(q, False) for q in asked}

    right_a, right_b = correct(a), correct(b)
    first, second = by_case(right_a), by_case(right_b)
    ids = sorted(first)
    members = case_members(questions)
    drawn = case_clusters(clusters, members, ids) if clusters is not None else None
    paired = paired_bootstrap(
        [float(first[c]) for c in ids],
        [float(second[c]) for c in ids],
        resamples=resamples,
        clusters=drawn,
    )

    def counts(right: dict[str, bool]) -> list[tuple[float, float]]:
        return [(sum(right[q] for q in members[c]), len(members[c])) for c in ids]

    shares = paired_ratio_bootstrap(
        counts(right_a), counts(right_b), resamples=resamples, clusters=drawn
    )

    def compared(first: list[float], second: list[float]) -> dict[str, float]:
        paired = paired_bootstrap(first, second, resamples=resamples, clusters=drawn)
        return {"a": sum(first) / len(ids), "b": sum(second) / len(ids), **paired}

    def every(right: dict[str, bool]) -> list[float]:
        return [float(all(right[q] for q in members[c])) for c in ids]

    def voted(texts: dict[str, str]) -> list[float]:
        flags = voted_cases(texts, members, answers, labels)
        return [float(flags[c]) for c in ids]

    found = {
        "cases": len(ids),
        **({"clusters": len(set(drawn))} if drawn is not None else {}),
        "a": sum(first[c] for c in ids) / len(ids),
        "b": sum(second[c] for c in ids) / len(ids),
        **paired,
        "per_question": {
            "a": shares["treated"],
            "b": shares["control"],
            "difference": shares["difference"],
            "low": shares["low"],
            "high": shares["high"],
        },
        "new_all": compared(every(right_a), every(right_b)),
        "voted": compared(voted(a), voted(b)),
    }
    if interaction is not None:
        found["interaction"] = strata_interaction(
            {c: (float(first[c]), float(second[c])) for c in ids},
            dict(zip(ids, zip(counts(right_a), counts(right_b), strict=True), strict=True)),
            interaction,
            dict(zip(ids, drawn, strict=True)) if drawn is not None else None,
            resamples,
        )
    return found


def strata_interaction(
    measure: dict[str, tuple[float, float]],
    shares: dict[str, tuple[tuple[float, float], tuple[float, float]]],
    strata: tuple[Collection[str], Collection[str]],
    clusters: dict[str, str | int] | None,
    resamples: int,
) -> dict:
    """The arm by stratum interaction of the compared cases: measure holds each case's
    value per arm (a, b), shares its (questions right, questions) per arm; strata are two
    disjoint sets of case ids, of which the compared ones count."""
    from xgraph.score import interaction_bootstrap

    overlap = sorted(set(strata[0]) & set(strata[1]))
    if overlap:
        raise ValueError(f"the strata share cases {overlap[:5]}: they must be disjoint")
    ids = sorted(measure)
    first = [c in strata[0] for c in ids]
    second = [c in strata[1] for c in ids]
    if not any(first) or not any(second):
        raise ValueError(
            f"a stratum has no compared case: {sum(first)} and {sum(second)} of {len(ids)}"
        )
    drawn = [clusters[c] for c in ids] if clusters is not None else None

    def bootstrap(a: list, b: list) -> dict:
        found = interaction_bootstrap(a, b, first, second, resamples=resamples, clusters=drawn)
        for name in ("first", "second"):
            arms = found[name]
            found[name] = {
                "a": arms["treated"],
                "b": arms["control"],
                "difference": arms["difference"],
            }
        return found

    return {
        "cases": {"first": sum(first), "second": sum(second)},
        "case_measure": bootstrap(
            [(measure[c][0], 1.0) for c in ids], [(measure[c][1], 1.0) for c in ids]
        ),
        "per_question": bootstrap([shares[c][0] for c in ids], [shares[c][1] for c in ids]),
    }


def update_events(cases: dict[str, dict]) -> dict[str, str]:
    """Each case id mapped to its update event: the connected component of cases joined by
    the (subject, relation) pairs their edits rewrite, named by its smallest pair."""
    parent: dict[tuple, tuple] = {}

    def root(node: tuple) -> tuple:
        while parent.setdefault(node, node) != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for case_id, case in cases.items():
        for s, r, _ in case["orig"]["edit_triples"]:
            parent[root(("pair", s, r))] = root(("case", case_id))
    pairs: dict[tuple, list[tuple[str, str]]] = {}
    for node in list(parent):
        if node[0] == "pair":
            pairs.setdefault(root(node), []).append((node[1], node[2]))
    return {c: ":".join(min(pairs[root(("case", c))])) for c in cases}


def write_clusters(argv: list[str]) -> None:
    """The clusters command: the asked MQuAKE cases keyed by their relation sequence, or
    with --by event by their update event."""
    from xgraph.mquake import relation_sequence

    parser = argparse.ArgumentParser(
        prog="python -m xgraph.editing clusters",
        description="Write a cluster file for --cluster: each asked MQuAKE case's id mapped "
        "to its edited chain's relation sequence, or with --by event to its update event: "
        "asked cases that edit a shared (subject, relation) pair, directly or through other "
        "asked cases, form one event, named by its smallest edited pair (Q1:P6).",
    )
    parser.add_argument("--by", choices=("sequence", "event"), default="sequence")
    parser.add_argument("--data", type=Path, required=True, help="MQuAKE(-Remastered) JSON")
    parser.add_argument("--questions", type=Path, required=True, help="the asked questions")
    parser.add_argument("--out", type=Path, required=True, help="the cluster file (JSON)")
    args = parser.parse_args(argv)
    by_id = {f"mquake-{c['case_id']}": c for c in json.loads(args.data.read_text())}
    members = case_members(json.loads(args.questions.read_text()))
    missing = [c for c in members if c not in by_id]
    if missing:
        raise SystemExit(f"{args.questions}: cases {missing[:5]} are not in {args.data}")
    if args.by == "event":
        clusters = update_events({c: by_id[c] for c in members})
    else:
        clusters = {c: relation_sequence(by_id[c]) for c in members}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(clusters, indent=1) + "\n")
    print(json.dumps({"cases": len(clusters), "clusters": len(set(clusters.values()))}))


def panel_allocation(sizes: dict[str, int], size: int) -> dict[str, int]:
    """Cases per cluster for a panel of `size` (see write_panel): n = min(N, max(1,
    floor(k sqrt N))) with the largest k that keeps the total within size, the slots left
    one each to the clusters whose k sqrt N lies closest below its next case (ties by
    cluster name)."""
    if size < len(sizes):
        raise ValueError(f"a panel of {size} cases cannot span {len(sizes)} clusters")
    if size > sum(sizes.values()):
        raise ValueError(f"a panel of {size} cases is larger than the {sum(sizes.values())} cases")

    def counts(k: float) -> dict[str, int]:
        return {c: min(n, max(1, math.floor(k * math.sqrt(n)))) for c, n in sizes.items()}

    low, high = 0.0, float(size)
    for _ in range(200):
        middle = (low + high) / 2
        low, high = (middle, high) if sum(counts(middle).values()) <= size else (low, middle)
    allocation = counts(low)
    open_ = [c for c in sorted(sizes) if allocation[c] < sizes[c]]
    open_.sort(key=lambda c: -(low * math.sqrt(sizes[c]) - allocation[c]))
    for cluster in open_[: size - sum(allocation.values())]:
        allocation[cluster] += 1
    return allocation


def write_panel(argv: list[str]) -> None:
    """The panel command: a seeded sample of cases stratified by cluster (see its help)."""
    from xgraph.provenance import provenance

    parser = argparse.ArgumentParser(
        prog="python -m xgraph.editing panel",
        description="Write a panel of --size cases from a cluster file (python -m "
        "xgraph.editing clusters), stratified by cluster so that it spans every cluster: each "
        "cluster of N cases gives min(N, max(1, floor(k sqrt N))) of them, k the largest scale "
        "that keeps the total within --size, and the slots left go one each to the clusters "
        "nearest their next case; within a cluster the cases are drawn with a generator seeded "
        "by --seed and the cluster, so one cluster's draw does not depend on the others. Writes "
        "the case ids (a JSON list, in case order, as --cases reads it) to --out and, to "
        "<out>.summary.json, each cluster's [cases, drawn] and each drawn case's weight N/n, "
        "the cases of the cluster file it stands for.",
    )
    parser.add_argument("--clusters", type=Path, required=True, help="case id to cluster (JSON)")
    parser.add_argument("--size", type=int, required=True, help="cases in the panel")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True, help="the panel's case ids (JSON)")
    args = parser.parse_args(argv)
    clusters = json.loads(args.clusters.read_text(encoding="utf-8"))
    members: dict[str, list[str]] = {}
    for case, cluster in clusters.items():
        members.setdefault(cluster, []).append(case)

    def order(case: str) -> tuple:
        number = case.rsplit("-", 1)[-1]
        return (int(number), case) if number.isdigit() else (sys.maxsize, case)

    try:
        allocation = panel_allocation({c: len(m) for c, m in members.items()}, args.size)
    except ValueError as error:
        raise SystemExit(f"{args.clusters}: {error}") from None
    drawn, weights = [], {}
    for cluster in sorted(members):
        cases = sorted(members[cluster], key=order)
        chosen = random.Random(f"{args.seed}:{cluster}").sample(cases, allocation[cluster])
        drawn += chosen
        weights.update({case: len(cases) / allocation[cluster] for case in chosen})
    drawn.sort(key=order)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(drawn, indent=1) + "\n")
    summary = {
        "artifact_kind": "xgraph.editing_panel",
        "version": 1,
        "size": args.size,
        "seed": args.seed,
        "cases": len(clusters),
        "clusters": len(members),
        "allocation": {c: [len(members[c]), allocation[c]] for c in sorted(members)},
        "weights": {case: weights[case] for case in drawn},
        "provenance": provenance([args.clusters]),
    }
    Path(f"{args.out}.summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    repeated = sum(n > 1 for n in allocation.values())  # clusters with more than one case
    print(json.dumps({"cases": len(drawn), "clusters": len(members), "repeated": repeated}))


def main(argv: list[str] | None = None) -> None:
    from xgraph.provenance import provenance
    from xgraph.score import load_labels, record_outputs

    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["clusters"]:
        write_clusters(argv[1:])
        return
    if argv[:1] == ["panel"]:
        write_panel(argv[1:])
        return
    parser = argparse.ArgumentParser(prog="python -m xgraph.editing", description=__doc__)
    parser.add_argument("--outputs", type=Path, required=True, help="generated answers (JSONL)")
    parser.add_argument("--questions", type=Path, required=True, help="questions (JSON)")
    parser.add_argument("--gold-new", type=Path, required=True)
    parser.add_argument(
        "--gold-old", type=Path, help="the old answers; without it only new ones are scored"
    )
    parser.add_argument("--labels", type=Path, required=True, help="gold labels (JSON lines)")
    parser.add_argument(
        "--cases", type=Path, help="score only these MQuAKE cases (a JSON list of case ids)"
    )
    parser.add_argument("--out", type=Path, required=True, help="result (JSON)")
    parser.add_argument(
        "--compare", type=Path, help="another arm's answers: the case measure's paired difference"
    )
    parser.add_argument(
        "--cluster",
        type=Path,
        help="with --compare: a JSON object mapping case (or question) ids to cluster ids; "
        "intervals resample whole clusters",
    )
    parser.add_argument(
        "--interaction",
        type=Path,
        nargs=2,
        metavar=("FIRST", "SECOND"),
        help="with --compare: two strata, JSON lists of case ids; the arm by stratum "
        "interaction, a minus b on the first minus a minus b on the second",
    )
    args = parser.parse_args(argv)
    if args.cluster and not args.compare:
        parser.error("--cluster resamples the comparison: it needs --compare")
    if args.interaction and not args.compare:
        parser.error("--interaction compares two arms: it needs --compare")
    questions = json.loads(args.questions.read_text())
    if args.cases:
        kept = set(json.loads(args.cases.read_text()))
        questions = [q for q in questions if q["id"].rsplit("-", 1)[0] in kept]
    gold_new = json.loads(args.gold_new.read_text())
    gold_old = json.loads(args.gold_old.read_text()) if args.gold_old else None
    golds = [g for g in (gold_new, gold_old) if g is not None]
    wanted = {e for g in golds for a in g["answers"].values() for e in a["entities"]}
    labels = load_labels([args.labels], wanted)
    qids = [q["id"] for q in questions]

    def outputs(path):
        return record_outputs(
            (json.loads(line) for line in path.read_text().splitlines() if line.strip()), qids
        )

    texts = outputs(args.outputs)
    result = {"artifact_kind": "xgraph.editing_scores", "version": 1}
    result.update(edit_scores(texts, questions, gold_new, gold_old, labels))
    if args.compare:
        other = outputs(args.compare)
        clusters = json.loads(args.cluster.read_text()) if args.cluster else None
        strata = None
        if args.interaction:
            strata = tuple(set(json.loads(path.read_text())) for path in args.interaction)
        result["compare"] = {
            "against": str(args.compare),
            **({"cluster_file": str(args.cluster)} if args.cluster else {}),
            **compare_cases(
                texts,
                other,
                questions,
                gold_new,
                gold_old,
                labels,
                clusters=clusters,
                interaction=strata,
            ),
        }
        if args.interaction:
            first_cases, second_cases = map(str, args.interaction)
            result["compare"]["interaction"] = {
                "first_cases": first_cases,
                "second_cases": second_cases,
                **result["compare"]["interaction"],
            }
    inputs = [args.outputs, args.questions, args.gold_new, args.labels]
    inputs += [args.gold_old] if args.gold_old else []
    inputs += [args.compare] if args.compare else []
    inputs += [args.cluster] if args.cluster else []
    inputs += list(args.interaction) if args.interaction else []
    result["provenance"] = provenance(inputs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("provenance", "by_group")}))


if __name__ == "__main__":
    main()
