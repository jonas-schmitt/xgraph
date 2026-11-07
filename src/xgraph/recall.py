"""Which unedited hops of an MQuAKE case's edited chain the served model recalls.

MQuAKE keeps only chains whose every fact its model could recall, the regime in which an
edit bank or edits in the prompt suffice: the unedited hops come from the model's memory.
This probe asks each unedited hop of each asked case's edited chain on its own, closed
book (the dataset's single-hop question, with the few-shot prompt of the prompted arm),
and scores the answer against the dataset's answer and aliases and against the KG's
current objects of the hop's subject and property (a hop the model knows as it is today,
when the dataset's answer has since changed, counts as recalled). Cases fall into strata
by where their unrecalled hops are: none ("recalled"), one before the first edit, so that
the edited subject itself is not known ("unrecalled-before"), or only after it
("unrecalled-after"); cases whose chain is all edits have no unedited hop ("no-hops").

    python -m xgraph.recall requests --data data/mquake/MQuAKE-Remastered-CF-3k.json \\
        --questions target/mqr-cf-3k-worlds/questions.json \\
        --out target/mqr-cf-3k-worlds/recall-requests.jsonl
    scripts/generate.sh <model> target/mqr-cf-3k-worlds/recall-requests.jsonl baseline \\
        target/mqr-cf-3k-worlds/recall.jsonl --max-tokens 512
    python -m xgraph.recall score --data data/mquake/MQuAKE-Remastered-CF-3k.json \\
        --questions target/mqr-cf-3k-worlds/questions.json \\
        --outputs target/mqr-cf-3k-worlds/recall.jsonl --db data/wikidata/gate-kg.sqlite \\
        --kg data/wikidata/gate-kg --out target/mqr-cf-3k-worlds/recall/strata.json

The strata file holds every hop's result and each case's stratum; `<stratum>-cases.json`
files beside it list each stratum's case ids for `python -m xgraph.editing --cases`.

Each hop's result also has a `lenient` key: recalled as above, or a near miss whose answer
and a dataset or KG name contain one another as whole words after normalization ("Kingdom
of France" for "France"). It does not change the strata; with --lenient, the strata file
also holds the strata it gives (`lenient_strata`, `lenient_cases`), and
`<stratum>-lenient-cases.json` files list them.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from xgraph.score import record_text

STRATA = ("recalled", "unrecalled-before", "unrecalled-after", "no-hops")


def unedited_hops(case: dict) -> list[dict]:
    """The hops of the case's edited chain that are not edits, with the dataset's single-hop
    question and answers, and whether they come before or after the first edited hop."""
    chain = case["orig"]["new_triples"]
    edits = {tuple(t) for t in case["orig"]["edit_triples"]}
    first = next((i for i, t in enumerate(chain) if tuple(t) in edits), len(chain))
    hops = []
    for i, (triple, hop) in enumerate(zip(chain, case["new_single_hops"], strict=True)):
        if tuple(triple) in edits:
            continue
        hops.append(
            {
                "id": f"recall-{case['case_id']}-{i}",
                "case": case["case_id"],
                "hop": i,
                "position": "before" if i < first else "after",
                "question": hop["question"],
                "answers": [hop["answer"], *hop["answer_alias"]],
                "triple": list(triple),
            }
        )
    return hops


def near_miss(said: str, name: str) -> bool:
    """Whether one of two normalized names contains the other as a run of whole words."""

    def contains(longer: tuple[str, ...], shorter: tuple[str, ...]) -> bool:
        n = len(shorter)
        return n > 0 and any(longer[i : i + n] == shorter for i in range(len(longer) - n + 1))

    first, second = tuple(re.findall(r"\w+", said)), tuple(re.findall(r"\w+", name))
    return contains(first, second) or contains(second, first)


def recalled(hop: dict, text: str, kg: dict[tuple[str, str], set[str]]) -> dict[str, bool]:
    """Whether the output's final answer names the dataset's answer (or an alias), whether
    it names a current KG object of the hop's subject and property (kg: normalized labels
    and aliases of those objects), and whether it does either leniently: either, or it and
    one of those names contain one another as whole words (near_miss)."""
    from xgraph.prompts import final_answer
    from xgraph.score import normalize

    answer = final_answer(text)
    if not answer:
        return {"dataset": False, "kg": False, "lenient": False}
    said = normalize(answer)
    subject, prop, _ = hop["triple"]
    gold = {normalize(a) for a in hop["answers"]}
    current = kg.get((subject, prop), set())
    found = {"dataset": said in gold, "kg": said in current}
    near = any(near_miss(said, name) for name in gold | current)
    return {**found, "lenient": found["dataset"] or found["kg"] or near}


def case_strata(
    hops: list[dict], results: dict[str, dict[str, bool]], lenient: bool = False
) -> dict[int, str]:
    """Each case's stratum (STRATA): a hop counts as recalled by the dataset or the KG,
    or with lenient by its lenient key; a hop without a result counts as unrecalled."""
    keys = ("lenient",) if lenient else ("dataset", "kg")
    by_case: dict[int, list[dict]] = {}
    for hop in hops:
        by_case.setdefault(hop["case"], []).append(hop)
    strata = {}
    for case, mine in by_case.items():
        missed = [h for h in mine if not any(results.get(h["id"], {}).get(k) for k in keys)]
        if any(h["position"] == "before" for h in missed):
            strata[case] = "unrecalled-before"
        elif missed:
            strata[case] = "unrecalled-after"
        else:
            strata[case] = "recalled"
    return strata


def _asked(questions: Path) -> list[int]:
    from xgraph.gwalk import asked_case_ids

    return asked_case_ids(json.loads(questions.read_text(encoding="utf-8")))


def _cases(data: Path, questions: Path) -> list[dict]:
    by_id = {c["case_id"]: c for c in json.loads(data.read_text(encoding="utf-8"))}
    return [by_id[c] for c in _asked(questions)]


def _kg_names(db: Path, kg: Path, hops: list[dict]) -> dict[tuple[str, str], set[str]]:
    """Normalized labels and aliases (kg/entity_labels.jsonl) of the KG objects (the store
    db) of every hop's subject and property."""
    import sqlite3

    from xgraph.score import normalize

    objects: dict[tuple[str, str], set[str]] = {}
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as connection:
        for hop in hops:
            subject, prop, _ = hop["triple"]
            rows = connection.execute(
                "SELECT o FROM triples WHERE s = ? AND p = ?", (int(subject[1:]), int(prop[1:]))
            ).fetchall()
            objects[(subject, prop)] = {f"Q{o}" for (o,) in rows}
    wanted = set().union(*objects.values()) if objects else set()
    names: dict[str, set[str]] = {}
    with (kg / "entity_labels.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            item, name, _ = json.loads(line)
            if item in wanted:
                names.setdefault(item, set()).add(normalize(name))
    return {
        key: set().union(*(names.get(o, set()) for o in found)) if found else set()
        for key, found in objects.items()
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m xgraph.recall", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    requests = commands.add_parser("requests", help="one request per unedited hop")
    score = commands.add_parser("score", help="each hop's recall and each case's stratum")
    for command in (requests, score):
        command.add_argument("--data", type=Path, required=True, help="MQuAKE-Remastered JSON")
        command.add_argument("--questions", type=Path, required=True, help="the asked questions")
        command.add_argument("--out", type=Path, required=True)
    score.add_argument("--outputs", type=Path, required=True, help="the probe's outputs")
    score.add_argument("--db", type=Path, required=True, help="the KG store (SQLite)")
    score.add_argument("--kg", type=Path, required=True, help="the KG (its entity labels)")
    score.add_argument(
        "--lenient",
        action="store_true",
        help="also the strata of lenient recall: <stratum>-lenient-cases.json",
    )
    args = parser.parse_args(argv)

    hops = [hop for case in _cases(args.data, args.questions) for hop in unedited_hops(case)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "requests":
        from xgraph.prompts import wwq_messages

        with args.out.open("w", encoding="utf-8") as out:
            for hop in hops:
                request = {"id": hop["id"], "messages": wwq_messages(hop["question"])}
                out.write(json.dumps(request, ensure_ascii=False) + "\n")
        print(json.dumps({"hops": len(hops)}))
        return

    from xgraph.provenance import provenance

    rows = (json.loads(line) for line in args.outputs.read_text().splitlines() if line)
    texts = {r["id"]: record_text(r) for r in rows}
    kg = _kg_names(args.db, args.kg, hops)
    results = {h["id"]: recalled(h, texts[h["id"]], kg) for h in hops if h["id"] in texts}
    asked = _asked(args.questions)

    def stratified(lenient: bool) -> dict[int, str]:
        found = case_strata(hops, results, lenient)
        for case in asked:
            found.setdefault(case, "no-hops")
        return found

    def counted(found: dict[int, str]) -> dict[str, int]:
        return {name: sum(s == name for s in found.values()) for name in STRATA}

    strata = stratified(False)
    written = {"": strata}  # file suffix: strata
    lenient = {}
    if args.lenient:
        written["-lenient"] = found = stratified(True)
        lenient = {
            "lenient_strata": counted(found),
            "lenient_cases": {str(c): s for c, s in sorted(found.items())},
        }
    result = {
        "artifact_kind": "xgraph.recall",
        "version": 1,
        "hops": len(hops),
        "answered": len(results),
        "recalled": {
            "dataset": sum(r["dataset"] for r in results.values()),
            "kg": sum(r["kg"] for r in results.values()),
            "either": sum(r["dataset"] or r["kg"] for r in results.values()),
            "lenient": sum(r["lenient"] for r in results.values()),
        },
        "strata": counted(strata),
        "cases": {str(c): s for c, s in sorted(strata.items())},
        **lenient,
        "results": results,
        "provenance": provenance([args.data, args.questions, args.outputs, args.db]),
        # entity_labels.jsonl is read too; the KG directory is named, not hashed (large)
        "kg": str(args.kg),
    }
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    for suffix, found in written.items():
        for name in STRATA:
            ids = [f"mquake-{c}" for c, s in sorted(found.items()) if s == name]
            (args.out.parent / f"{name}{suffix}-cases.json").write_text(json.dumps(ids) + "\n")
    shown = ("hops", "answered", "recalled", "strata", "lenient_strata")
    print(json.dumps({k: result[k] for k in shown if k in result}))


if __name__ == "__main__":
    main()
