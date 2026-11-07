"""World gold: MQuAKE's edited relation chain executed in each case's world KG.

MQuAKE-Remastered's gold new answer follows the case's edited chain as the dataset
recorded it; an unedited hop may have changed in Wikidata since (a head of government
replaced), so a method that reads the current KG can give the answer the KG entails
and be scored wrong. World gold runs the chain's relations from its first subject in
the case's interference-free world (`python -m xgraph.mquake --worlds`, the KG stores
`python -m xgraph.toolkg build` wrote per world) and takes every entity it reaches. It
is a secondary measure beside the dataset's gold: it scores every arm the same way, and
a multi-valued hop makes it more lenient than the dataset's single answer.

    python -m xgraph.world_gold --data data/mquake/MQuAKE-Remastered-CF-3k.json \\
        --questions target/mqr-cf-3k-worlds/questions.json \\
        --worlds target/mqr-cf-3k-worlds/worlds.json --db data/wikidata/mqrw-kg-w{}.sqlite \\
        --kg-labels data/wikidata/gate-kg/entity_labels.jsonl \\
        --out-gold target/mqr-cf-3k-worlds/gold-world.json \\
        --out-labels target/mqr-cf-3k-worlds/labels-world.jsonl \\
        --gold-new target/mqr-cf-3k-worlds/gold-new.json \\
        --out-agree target/mqr-cf-3k-worlds/agree-cases.json

With --gold-new and --out-agree it also writes the cases each of whose questions both
golds answer with the same entities: the cases where scoring does not depend on drift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def world_gold(
    cases: list[dict], questions: list[dict], world_of: dict[str, int], stores: list[Any]
) -> tuple[dict, dict[str, str]]:
    """Gold answers ({"answers", "failed"}, as xgraph.score reads them) for the questions,
    and the labels of the entities they name. stores[k].objects_of(items, prop) gives
    the labelled objects of the items under the property in world k."""
    by_case = {f"mquake-{c['case_id']}": c for c in cases}
    answers: dict[str, dict] = {}
    failed: dict[str, str] = {}
    found: dict[str, str] = {}
    reached: dict[str, dict[int, str] | str] = {}
    for question in questions:
        case = question["id"].rsplit("-", 1)[0]
        if case not in reached:
            chain = by_case[case]["orig"]["new_triples"]
            store = stores[world_of[case]]
            frontier: dict[int, str] = {int(chain[0][0][1:]): ""}
            for hop, (_, prop, _) in enumerate(chain):
                frontier = store.objects_of(sorted(frontier), int(prop[1:]))
                if not frontier:
                    reached[case] = f"the chain stops after {hop} of {len(chain)} relations"
                    break
            else:
                reached[case] = frontier
        result = reached[case]
        if isinstance(result, str):
            failed[question["id"]] = result
            continue
        entities = sorted(f"Q{item}" for item in result)
        answers[question["id"]] = {"entities": entities, "literals": []}
        found.update((f"Q{item}", label) for item, label in result.items())
    return {"answers": answers, "failed": failed}, found


def agreeing_cases(world: dict, dataset: dict) -> list[str]:
    """The cases each of whose questions names the same entities in both golds."""
    agree: dict[str, bool] = {}
    for qid, answer in dataset["answers"].items():
        mine = world["answers"].get(qid)
        same = mine is not None and set(mine["entities"]) == set(answer["entities"])
        case = qid.rsplit("-", 1)[0]
        agree[case] = agree.get(case, True) and same
    return sorted(case for case, same in agree.items() if same)


def main(argv: list[str] | None = None) -> None:
    from xgraph.provenance import provenance
    from xgraph.toolkg import KGStore

    parser = argparse.ArgumentParser(prog="python -m xgraph.world_gold", description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="MQuAKE-Remastered-CF-3k.json")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--worlds", type=Path, required=True, help="xgraph.mquake's worlds.json")
    parser.add_argument("--db", required=True, help="world k's KG store, {} for k")
    parser.add_argument(
        "--kg-labels", type=Path, required=True, help="the KG's entity_labels.jsonl (aliases)"
    )
    parser.add_argument("--out-gold", type=Path, required=True)
    parser.add_argument("--out-labels", type=Path, required=True)
    parser.add_argument(
        "--gold-new", type=Path, help="the dataset's gold (xgraph.mquake's gold-new.json)"
    )
    parser.add_argument(
        "--out-agree", type=Path, help="the cases whose questions both golds answer alike"
    )
    args = parser.parse_args(argv)
    cases = json.loads(args.data.read_text())
    questions = json.loads(args.questions.read_text())
    world_of = json.loads(args.worlds.read_text())
    stores = [KGStore(args.db.format(k)) for k in range(max(world_of.values()) + 1)]
    gold, found = world_gold(cases, questions, world_of, stores)
    args.out_gold.write_text(json.dumps(gold) + "\n")
    with args.out_labels.open("w", encoding="utf-8") as out:
        for item, label in sorted(found.items()):
            out.write(json.dumps([item, label, "label"], ensure_ascii=False) + "\n")
        with args.kg_labels.open(encoding="utf-8") as lines:
            for line in lines:
                if json.loads(line)[0] in found:
                    out.write(line)
    agree = None
    if args.gold_new and args.out_agree:
        agree = agreeing_cases(gold, json.loads(args.gold_new.read_text()))
        args.out_agree.write_text(json.dumps(agree) + "\n")
    manifest = {
        "artifact_kind": "xgraph.world_gold",
        "version": 1,
        "questions": len(questions),
        "answered": len(gold["answers"]),
        "failed": len(gold["failed"]),
        "agreeing_cases": None if agree is None else len(agree),
        "provenance": provenance(
            [args.data, args.questions, args.worlds, *([args.gold_new] if args.gold_new else [])]
        ),
    }
    Path(f"{args.out_gold}.manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "provenance"}))


if __name__ == "__main__":
    main()
