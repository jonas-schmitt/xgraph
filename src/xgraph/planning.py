"""Planning: the relations a path-mode answer writes, against the gold chain's relations.

In path mode the model chooses relations and the KG supplies every entity, so a wrong
answer comes from a wrong plan (relations or length), a wrong start, or the KG. This
module compares each answer's first path span, its relations read as property ids by
label or alias (xgraph.wikidata.relation_names), with the gold chain's properties, and
reports how often the plan is exact, too long, too short, or the same length with other
relations, and how often the answer is right given an exact plan and given another one;
overall and per group (the questions' complexity field). A walk's record (xgraph.walk)
writes no path span; its plan is the relations of the path it walked (its `path`). Nor
does a KG agent's (xgraph.agent): its plan is the relations of its last whole-path call,
or of the relations it followed one at a time (walked_paths).

The CPU-only `exact` command needs only outputs, the question panel, gold chains and
relation labels. It streams records, keeps missing outputs in the denominator, and
writes exact-plan counts and rates overall, by question group and per question, with
the agent's recorded call/failure/retry metrics. Exactness scores relation sequences,
independently of starts, executability and answer accuracy. Unreadable whole-path calls
do not replace a readable plan; string relations and whitespace are read as the tool
reads them. The existing `score` command also scores answer accuracy given a plan.

Gold chains are JSON mapping a question id to its chain of (subject, property, object)
ids: MINTQA's chains.json (python -m xgraph.mintqa_bench), or MQuAKE's edited chains
written by the `chains` command. Correctness comes from the questions' gold answers, as
xgraph.score scores them.

    python -m xgraph.planning chains --data data/mquake/MQuAKE-Remastered-CF-3k.json \\
        --questions target/mqr-cf-3k-allw/questions.json --out target/mqr-cf-3k-allw/chains.json
    python -m xgraph.planning score --outputs target/mintqa/qwen25-paths-committed.jsonl \\
        --chains target/mintqa/chains.json \\
        --relations data/external/mintqa-full/kg/relation_labels.jsonl \\
        --questions target/mintqa/questions-representable.json \\
        --gold target/mintqa/gold.json --labels target/mintqa/labels.jsonl \\
        --out target/mintqa/qwen25-paths-committed.planning.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path

from xgraph.score import record_text

_PATH = re.compile(r"⟦([^⟧]*·[^⟧]*)⟧")


def plan_of(text: str, names: dict[str, str]) -> list[str] | None:
    """The property ids of the first path span's relations (unknown names as "?name"),
    or None without a path span."""
    from xgraph.audit import semantic_text
    from xgraph.observations import path_segments

    for found in _PATH.finditer(semantic_text(text)):
        relations = path_segments(found.group(1))[1:]
        if relations:
            return [names.get(r, f"?{r}") for r in relations]
    return None


def walked_paths(records: list[dict]) -> dict[str, list[str]]:
    """The paths of records that write no path span, by question id: a walk's (xgraph.walk:
    the start's label, then the relations followed), and a KG agent's (xgraph.agent: the
    start and relations of its last readable execute_path call, else the first followed
    entity and the relations of its follow calls in order, else none). A string relation
    list and whitespace are read as the tool reads them; malformed calls are ignored."""
    paths = {r["id"]: r["path"] for r in records if isinstance(r.get("path"), list)}
    for r in records:
        if "actions" in r and r["id"] not in paths:
            paths[r["id"]] = _agent_path(r["actions"])
    return paths


def _agent_path(actions: list[dict]) -> list[str]:
    for action in reversed(actions):
        if action.get("tool") != "execute_path":
            continue
        args = action.get("arguments") or {}
        if "start" not in args or "relations" not in args:
            continue
        # Invalid feedback calls have no executed path; retain the last readable one.
        if action.get("failed") and action.get("result", {}).get("failed_hop", 0) is None:
            continue
        relations = args["relations"] or []
        if isinstance(relations, str):
            relations = [r for r in re.split(r"[·,]", relations) if r.strip()]
        if isinstance(relations, (list, tuple)):
            return [str(args["start"]).strip(), *(str(r).strip() for r in relations)]
    follows = [a.get("arguments") or {} for a in actions if a.get("tool") == "follow"]
    if follows:
        return [follows[0].get("entity", ""), *(f.get("relation", "") for f in follows)]
    return []


def _tally(rows: list[tuple[list[str] | None, list[str], bool]]) -> dict:
    with_path = [(p, g, r) for p, g, r in rows if p is not None]
    exact = [r for p, g, r in with_path if p == g]
    other = [r for p, g, r in with_path if p != g]
    return {
        "questions": len(rows),
        "with_path": len(with_path),
        "exact_plan": len(exact),
        "too_long": sum(len(p) > len(g) for p, g, _ in with_path),
        "too_short": sum(len(p) < len(g) for p, g, _ in with_path),
        "same_length_other": sum(len(p) == len(g) and p != g for p, g, _ in with_path),
        "right_given_exact_plan": sum(exact) / len(exact) if exact else 0.0,
        "right_given_other_plan": sum(other) / len(other) if other else 0.0,
    }


def plan_scores(
    texts: dict[str, str],
    chains: dict[str, list[list[str]]],
    names: dict[str, str],
    right: dict[str, bool],
    groups: dict[str, str],
    paths: dict[str, list[str]] | None = None,
) -> dict:
    """Plan tallies (see _tally) over the questions with a gold chain, overall and per
    group. A question with a walked path (walked_paths) has that path's relations as its
    plan, none when the walk found no start, instead of its text's path span."""
    rows: dict[str, tuple[list[str] | None, list[str], bool]] = {}
    for qid, chain in chains.items():
        gold = [p for _, p, _ in chain]
        if paths is not None and qid in paths:
            walked = paths[qid]
            plan = [names.get(r, f"?{r}") for r in walked[1:]] if walked else None
        else:
            plan = plan_of(texts.get(qid, ""), names)
        rows[qid] = (plan, gold, right.get(qid, False))
    by_group: dict[str, list] = {}
    for qid, row in rows.items():
        by_group.setdefault(groups.get(qid, ""), []).append(row)
    return {
        "all": _tally(list(rows.values())),
        "by_group": {name: _tally(members) for name, members in sorted(by_group.items())},
    }


def _names(path: Path) -> dict[str, str]:
    """Each relation label or alias (relation_names' policy) to its property id."""
    from xgraph.wikidata import relation_names

    rows = [tuple(json.loads(line)) for line in path.read_text().splitlines() if line]
    names: dict[str, str] = {}
    for prop, labels in relation_names(rows).items():
        for label in labels:
            names.setdefault(label, prop)
    return names


def exact_plan_scores(
    records: Iterable[dict],
    chains: dict[str, list[list[str]]],
    names: dict[str, str],
    groups: dict[str, str] | None = None,
) -> dict:
    """Final relation-plan exactness, with every supplied gold chain in the denominator.

    Records can be streamed: retain only plans and metrics, never full agent turns or
    tool observations. Gold answers, entity labels, a KG and a model are unnecessary.
    Starts and answer delivery are not scored; malformed whole-path calls do not
    replace the agent's last readable plan (walked_paths).
    """
    rows = {
        qid: {
            "plan": None,
            "gold_plan": [p for _, p, _ in chain],
            "exact_plan": False,
            "has_output": False,
            "metrics": {},
        }
        for qid, chain in chains.items()
    }
    for record in records:
        qid = record["id"]
        if qid not in rows:
            continue
        if rows[qid]["has_output"]:
            raise ValueError(f"duplicate output for question {qid!r}")
        from xgraph.score import replay_failure

        paths = {} if replay_failure(record) else walked_paths([record])
        if qid in paths:
            path = paths[qid]
            plan = [names.get(r, f"?{r}") for r in path[1:]] if path else None
        else:
            plan = plan_of(record_text(record), names)
        metrics = {
            k: record[k]
            for k in (
                "tool_calls",
                "failures",
                "retries",
                "retry_changed_plan",
                "answer_refusals",
                "budget_spent",
            )
            if k in record
        }
        rows[qid].update(
            plan=plan, exact_plan=plan == rows[qid]["gold_plan"], has_output=True, metrics=metrics
        )

    def tally(members: list[dict]) -> dict:
        counts = _tally([(r["plan"], r["gold_plan"], False) for r in members])
        return {k: v for k, v in counts.items() if not k.startswith("right_given_")} | {
            "exact_plan_rate": counts["exact_plan"] / len(members) if members else 0.0
        }

    by_group: dict[str, list[dict]] = {}
    for qid, row in rows.items():
        by_group.setdefault((groups or {}).get(qid, ""), []).append(row)
    return {
        "all": tally(list(rows.values())),
        "by_group": {g: tally(members) for g, members in sorted(by_group.items())},
        "per_question": rows,
    }


def main(argv: list[str] | None = None) -> None:
    from xgraph.provenance import provenance

    parser = argparse.ArgumentParser(prog="python -m xgraph.planning", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    chains = commands.add_parser("chains", help="MQuAKE questions' edited chains")
    chains.add_argument("--data", type=Path, required=True, help="MQuAKE-Remastered JSON")
    chains.add_argument("--questions", type=Path, required=True)
    chains.add_argument("--out", type=Path, required=True)
    score = commands.add_parser("score", help="plans against gold chains")
    score.add_argument("--outputs", type=Path, required=True, help="path-mode records (JSONL)")
    score.add_argument("--chains", type=Path, required=True, help="question id to gold chain")
    score.add_argument("--relations", type=Path, required=True, help="relation_labels.jsonl")
    score.add_argument("--questions", type=Path, required=True)
    score.add_argument("--gold", type=Path, required=True, help="gold answers (WWQ format)")
    score.add_argument("--labels", type=Path, required=True, help="gold answers' label rows")
    score.add_argument("--out", type=Path, required=True)
    exact = commands.add_parser("exact", help="CPU-only final relation-plan exactness")
    exact.add_argument("--outputs", type=Path, required=True, help="generation records (JSONL)")
    exact.add_argument("--chains", type=Path, required=True, help="question id to gold chain")
    exact.add_argument("--relations", type=Path, required=True, help="relation_labels.jsonl")
    exact.add_argument("--questions", type=Path, required=True, help="the complete question panel")
    exact.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    if args.command == "chains":
        from xgraph.mquake import case_id

        by_case = {
            f"mquake-{c['case_id']}": c["orig"]["new_triples"]
            for c in json.loads(args.data.read_text(encoding="utf-8"))
        }
        written = {q["id"]: by_case[case_id(q["id"])] for q in questions}
        args.out.write_text(json.dumps(written) + "\n")
        print(json.dumps({"questions": len(written)}))
        return
    if args.command == "exact":
        chains = json.loads(args.chains.read_text(encoding="utf-8"))
        ids = [q["id"] for q in questions]
        if len(set(ids)) != len(ids):
            parser.error("the question panel contains a duplicate question id")
        if missing := set(ids) - chains.keys():
            parser.error(f"no gold chain for {len(missing)} panel questions: {sorted(missing)[:5]}")
        selected = {q["id"]: chains[q["id"]] for q in questions}
        groups = {q["id"]: q.get("complexity", "") for q in questions}
        with args.outputs.open(encoding="utf-8") as records:
            result = {
                "artifact_kind": "xgraph.planning.exact",
                "version": 1,
                **exact_plan_scores(
                    (json.loads(line) for line in records if line.strip()),
                    selected,
                    _names(args.relations),
                    groups,
                ),
                "provenance": provenance(
                    [args.outputs, args.chains, args.relations, args.questions]
                ),
            }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result["all"]))
        return
    from xgraph.score import load_labels, wwq_accuracy

    rows = [json.loads(line) for line in args.outputs.read_text().splitlines() if line]
    texts = {r["id"]: record_text(r) for r in rows}
    asked = {q["id"] for q in questions}
    gold = json.loads(args.gold.read_text())
    gold = {  # the questions asked: a question without an output is wrong
        "answers": {q: a for q, a in gold["answers"].items() if q in asked},
        "failed": {q: f for q, f in gold.get("failed", {}).items() if q in asked},
    }
    wanted = {e for a in gold["answers"].values() for e in a["entities"]}
    answered = {q: texts.get(q, "") for q in gold["answers"]}
    right = wwq_accuracy(answered, gold, load_labels([args.labels], wanted))["correct"]
    chain_of = {
        qid: chain
        for qid, chain in json.loads(args.chains.read_text()).items()
        if qid in asked and qid in gold["answers"]
    }
    groups = {q["id"]: q.get("complexity", "") for q in questions}
    result = {
        "artifact_kind": "xgraph.planning",
        "version": 1,
        **plan_scores(texts, chain_of, _names(args.relations), right, groups, walked_paths(rows)),
        "provenance": provenance(
            [args.outputs, args.chains, args.relations, args.questions, args.gold]
        ),
    }
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["all"]))


if __name__ == "__main__":
    main()
