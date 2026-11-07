"""MQuAKE: multi-hop questions whose answers change when the KG is edited.

MQuAKE-CF-3k (Zhong et al., EMNLP 2023; the v2 file of the authors' repository) holds
3,000 cases, each a chain of 2 to 4 Wikidata triples, one or more of them edited, with
multi-hop questions whose answer follows the edited chain. Method pilot 4
(docs/research-plan.md) applies every case's edits to the gate KG at once (an edit
replaces all objects of its subject and relation; where two cases edit the same pair
differently, the first case wins) and asks whether a model answers with the new answer
when its spans are constrained to the edited KG, given the edited facts in the prompt,
or both.

The questions are written in the WWQ format of the gate's prompts (the chain's first
subject as the question entity), with gold files for the new answer and the original
one and a label file from the dataset's answers and aliases plus the KG's labels.

    python -m xgraph.mquake --data data/mquake/MQuAKE-CF-3k-v2.json \\
        --kg data/wikidata/gate-kg --cases 1000 --seed 0 \\
        --out-kg data/wikidata/mquake-kg --out target/mquake

requests-edits.jsonl asks each question with its case's edited facts in the prompt (the
in-context editing arm). With --worlds, the asked cases are also split into
interference-free worlds (no case's edit rewrites a hop of another's edited chain within
one world), each its own edited KG <out-kg>-w<k>, so that every case can be asked in a
world whose KG holds its edited chain; worlds.json maps each case to its world. With
--edit-labels, an edited object or subject the KG has no label for gets the name its edit
gives it, in the edited KGs' own entity_labels.jsonl. With --dated, triples from
python -m xgraph.dated first replace the KG's objects of their pairs (the KG as Wikidata
held it at a date), then the cases' edits apply. With --worlds --kg-versions <dir>, the
worlds are also written as KG versions w0, w1, ... of the KG (xgraph.kgversions; the dated
triples included in each), so that one KG index and one server serve every world: a
request reads its world's version by its kg_version field (python -m xgraph.kgversions
tag). --no-world-kgs then leaves the worlds' own KG directories out. KG versions add no
labels, so with --edit-labels the KG's own triples with the edits' names are written as
<kg-versions>/base-kg, the KG whose KG index the versions are to be read over.
--exclude <MQuAKE JSON> leaves the cases that file holds unasked, matched by original
chain or by a shared question (releases number their cases on their own; their edits
still apply): a confirmatory sample from CF-6334 can avoid every CF-3k case.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def _number(item: str) -> int:
    return int(item[1:])


def edit_triples(cases: list[dict]) -> tuple[np.ndarray, int]:
    """Every case's edits as (subject, property, object) numbers, the first edit of a
    (subject, property) pair winning, and how many later edits conflicted with it."""
    chosen: dict[tuple[int, int], int] = {}
    conflicts = 0
    for case in cases:
        for s, p, o in case["orig"]["edit_triples"]:
            key, obj = (_number(s), _number(p)), _number(o)
            if key not in chosen:
                chosen[key] = obj
            elif chosen[key] != obj:
                conflicts += 1
    edits = np.array([[s, p, o] for (s, p), o in chosen.items()], dtype=np.int32)
    return edits.reshape(-1, 3), conflicts


def _conflicts(case: dict, edit: list[str]) -> bool:
    """An edit that rewrites a hop of the case's edited chain to another object."""
    s, p, o = edit
    return any(cs == s and cp == p and co != o for cs, cp, co in case["orig"]["new_triples"])


def worlds(asked: list[dict], cases: list[dict]) -> list[dict]:
    """Edited KG worlds in which the asked cases' edits do not interfere: the asked
    cases split into groups no two of whose cases conflict (one's edit rewriting a hop
    of the other's edited chain), and for each group every case's edits except those
    that conflict with a group member, the members' own edits first (MQuAKE-Remastered
    masks conflicting edits the same way). Groups are colored greedily, the cases with
    the most conflicts first. Each world: {"cases": [case ids], "edits": (n, 3) array}."""

    def hops(group: list[dict]) -> dict[tuple[str, str], set[str]]:
        found: dict[tuple[str, str], set[str]] = {}
        for c in group:
            for s, p, o in c["orig"]["new_triples"]:
                found.setdefault((s, p), set()).add(o)
        return found

    def clashes(chain: dict[tuple[str, str], set[str]], edit: list[str]) -> bool:
        return bool(chain.get((edit[0], edit[1]), set()) - {edit[2]})

    ids = [c["case_id"] for c in asked]
    editors: dict[tuple[str, str], list[dict]] = {}
    for b in asked:
        for e in b["orig"]["edit_triples"]:
            editors.setdefault((e[0], e[1]), []).append(b)
    clash: dict[int, set[int]] = {i: set() for i in ids}
    for a in asked:
        chain = hops([a])
        for key in chain:
            for b in editors.get(key, []):
                if b is not a and any(clashes(chain, e) for e in b["orig"]["edit_triples"]):
                    clash[a["case_id"]].add(b["case_id"])
                    clash[b["case_id"]].add(a["case_id"])
    color: dict[int, int] = {}
    for i in sorted(ids, key=lambda i: (-len(clash[i]), i)):
        used = {color[j] for j in clash[i] if j in color}
        color[i] = next(k for k in range(len(ids) + 1) if k not in used)
    found = []
    for k in range(max(color.values(), default=-1) + 1):
        members = [c for c in asked if color[c["case_id"]] == k]
        chain = hops(members)
        others = [c for c in cases if color.get(c["case_id"]) != k]
        kept = [
            {**c, "orig": {**c["orig"], "edit_triples": [
                e for e in c["orig"]["edit_triples"] if not clashes(chain, e)
            ]}}
            for c in others
        ]  # fmt: skip
        edits, _ = edit_triples(members + kept)
        found.append({"cases": sorted(c["case_id"] for c in members), "edits": edits})
    return found


def excluded_cases(cases: list[dict], others: list[dict]) -> set[int]:
    """The ids of the cases another release holds, matched by content, not by id
    (releases number their cases each on their own): the same original chain, or a
    question in common."""
    chains = {tuple(map(tuple, c["orig"]["triples"])) for c in others}
    asked = {q for c in others for q in c["questions"]}
    return {
        c["case_id"]
        for c in cases
        if tuple(map(tuple, c["orig"]["triples"])) in chains
        or any(q in asked for q in c["questions"])
    }


def edit_labels(cases: list[dict], labelled: set[str]) -> list[list[str]]:
    """Label rows for the edited objects and subjects the KG has no label for, named as
    the edits name them (an edit's requested rewrite and its edit triple are aligned):
    without a label row an entity cannot be on a path's frontier or name an answer."""
    rows: dict[str, list[str]] = {}
    for case in cases:
        pairs = zip(case["requested_rewrite"], case["orig"]["edit_triples"], strict=True)
        for rewrite, (s, _, o) in pairs:
            for item, name in ((o, rewrite["target_new"]["str"]), (s, rewrite["subject"])):
                if item not in labelled and item not in rows:
                    rows[item] = [item, name, "label"]
    return list(rows.values())


def apply_edits(triples: np.ndarray, edits: np.ndarray) -> np.ndarray:
    """The triples with every object of an edited (subject, property) pair replaced."""
    key = triples[:, 0].astype(np.int64) << 16 | triples[:, 1].astype(np.int64)
    edited = edits[:, 0].astype(np.int64) << 16 | edits[:, 1].astype(np.int64)
    kept = triples[~np.isin(key, edited)]
    return np.concatenate([kept, edits.astype(np.int32)])


def version_edits(edits: np.ndarray, dated: np.ndarray | None) -> np.ndarray:
    """A world's edits as a KG version of the undated KG: the world's edits, then the
    dated triples of the pairs the world does not edit, so that apply_edits of the KG with
    them gives apply_edits of the dated KG with the world's edits."""
    if dated is None or not len(dated):
        return edits
    edited = {(s, p) for s, p, _ in edits.tolist()}
    kept = np.array([(s, p) not in edited for s, p, _ in dated.tolist()], dtype=bool)
    return np.concatenate([edits, dated[kept].astype(np.int32)]).reshape(-1, 3)


def questions(
    cases: list[dict], per_case: int = 1
) -> tuple[list[dict], dict, dict, list[list[str]]]:
    """WWQ-format questions (each case's first question, or its first per_case ones,
    with ids mquake-<case>-<k>), gold files for the new and the original answers, and
    label rows for the answers from the dataset."""
    items, new, old, labels = [], {}, {}, []
    for case in cases:
        chain, new_chain = case["orig"]["triples"], case["orig"]["new_triples"]
        new_item, old_item = new_chain[-1][2], chain[-1][2]
        asked = case["questions"][:per_case]
        for k, utterance in enumerate(asked):
            qid = f"mquake-{case['case_id']}" + (f"-{k}" if per_case > 1 else "")
            items.append(
                {
                    "id": qid,
                    "utterance": utterance,
                    "sparql": f"# question entities: wd:{chain[0][0]}",
                    "complexity": f"{len(chain)}-hop",
                }
            )
            new[qid] = {"entities": [new_item], "literals": []}
            old[qid] = {"entities": [old_item], "literals": []}
        for item, name, aliases in (
            (new_item, case["new_answer"], case["new_answer_alias"]),
            (old_item, case["answer"], case["answer_alias"]),
        ):
            labels.append([item, name, "label"])
            labels.extend([item, alias, "alias"] for alias in aliases)
    return items, {"answers": new, "failed": {}}, {"answers": old, "failed": {}}, labels


def edit_requests(cases: list[dict], per_case: int = 1) -> list[dict]:
    """The in-context editing arm: each question asked with its case's edited facts
    (labelled, from the edited chain) as Wikidata facts in the prompt, unconstrained."""
    from xgraph.rag import rag_messages

    requests = []
    for case in cases:
        chain, labelled = case["orig"]["new_triples"], case["orig"]["new_triples_labeled"]
        edited = [tuple(t) for t in case["orig"]["edit_triples"]]
        facts = [tuple(labelled[i]) for i, t in enumerate(chain) if tuple(t) in edited]
        for k, utterance in enumerate(case["questions"][:per_case]):
            qid = f"mquake-{case['case_id']}" + (f"-{k}" if per_case > 1 else "")
            requests.append({"id": qid, "messages": rag_messages(utterance, facts)})
    return requests


def covered_cases(cases: list[dict], triples: np.ndarray) -> list[str]:
    """The cases whose whole edited chain (new_triples) the edited KG holds; the others
    ask about facts that changed since MQuAKE was built, so their gold no longer holds."""
    needed = {tuple(_number(x) for x in t) for c in cases for t in c["orig"]["new_triples"]}
    subjects = np.array(sorted({t[0] for t in needed}), dtype=np.int64)
    rows = triples[np.isin(triples[:, 0], subjects)] if len(subjects) else triples[:0]
    held = {tuple(int(x) for x in row) for row in rows}
    return [
        f"mquake-{c['case_id']}"
        for c in cases
        if all(tuple(_number(x) for x in t) in held for t in c["orig"]["new_triples"])
    ]


def case_id(qid: str) -> str:
    """The case a question belongs to: its id without the paraphrase index."""
    return qid.rsplit("-", 1)[0] if qid.count("-") > 1 else qid


def relation_sequence(case: dict) -> str:
    """The relations of the case's edited chain in order, joined by ">" (P26>P108): cases
    sharing it ask the same kind of question about different entities."""
    return ">".join(p for _, p, _ in case["orig"]["new_triples"])


def by_case(correct: dict[str, bool]) -> dict[str, bool]:
    """MQuAKE's measure: a case is answered when any of its questions is."""
    cases: dict[str, bool] = {}
    for qid, right in correct.items():
        case = case_id(qid)
        cases[case] = cases.get(case, False) or right
    return cases


def main(argv: list[str] | None = None) -> None:
    from xgraph.probes import write_variant
    from xgraph.provenance import provenance

    parser = argparse.ArgumentParser(prog="python -m xgraph.mquake", description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="MQuAKE-CF-3k-v2.json")
    parser.add_argument("--kg", type=Path, required=True, help="gate KG directory")
    parser.add_argument("--cases", type=int, default=1000, help="cases to ask about")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-case", type=int, default=1, help="questions asked per case (1-3)")
    parser.add_argument("--out-kg", type=Path, required=True, help="the edited KG directory")
    parser.add_argument("--out", type=Path, required=True, help="questions, gold, labels")
    parser.add_argument(
        "--worlds",
        action="store_true",
        help="also write interference-free worlds: <out-kg>-w<k> and worlds.json",
    )
    parser.add_argument(
        "--edit-labels",
        action="store_true",
        help="give edited objects and subjects the KG has no label for the edits' names "
        "(edit_labels): the edited KGs get their own entity_labels.jsonl",
    )
    parser.add_argument(
        "--dated",
        type=Path,
        help="triples (.npy, python -m xgraph.dated) replacing the KG's objects of their "
        "pairs before the cases' edits: the KG as Wikidata held it at a date",
    )
    parser.add_argument(
        "--kg-versions",
        type=Path,
        help="with --worlds: also write every world's edits as KG versions w<k> of the KG "
        "(a version file, xgraph.kgversions)",
    )
    parser.add_argument(
        "--no-world-kgs",
        action="store_true",
        help="with --kg-versions: leave out the worlds' own KG directories <out-kg>-w<k>",
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        help="an MQuAKE JSON file: its cases, matched by original chain or a shared question, "
        "are not asked (their edits still apply)",
    )
    args = parser.parse_args(argv)
    if args.kg_versions and not args.worlds:
        parser.error("--kg-versions writes the worlds' edits: it needs --worlds")
    if args.no_world_kgs and not args.kg_versions:
        parser.error("--no-world-kgs needs --kg-versions, which then holds the worlds")
    cases = json.loads(args.data.read_text())
    edits, conflicts = edit_triples(cases)
    base = triples = np.load(args.kg / "triples.npy")
    dated = np.load(args.dated) if args.dated else None
    if dated is not None:
        triples = apply_edits(triples, dated)
    edited = apply_edits(triples, edits)
    extra = None
    if args.edit_labels:
        named = {x for c in cases for s, _, o in c["orig"]["edit_triples"] for x in (s, o)}
        labelled = set()
        with (args.kg / "entity_labels.jsonl").open(encoding="utf-8") as lines:
            for line in lines:
                item = json.loads(line)[0]
                if item in named:
                    labelled.add(item)
        extra = edit_labels(cases, labelled)
    write_variant(args.kg, edited, args.out_kg, extra_labels=extra)
    base_kg = None
    if args.kg_versions and extra:  # versions add no labels: the base KG index must hold them
        base_kg = args.kg_versions / "base-kg"
        write_variant(args.kg, base, base_kg, extra_labels=extra)
    covered = covered_cases(cases, edited)
    excluded = set()
    if args.exclude:
        excluded = excluded_cases(cases, json.loads(args.exclude.read_text()))
    pool = [c for c in cases if c["case_id"] not in excluded]
    asked = sorted(random.Random(args.seed).sample(pool, args.cases), key=lambda c: c["case_id"])
    items, new, old, labels = questions(asked, args.per_case)
    asked_ids = {f"mquake-{c['case_id']}" for c in asked}
    covered_asked = [c for c in covered if c in asked_ids]
    wanted = {row[0] for row in labels}
    with (args.kg / "entity_labels.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            row = json.loads(line)
            if row[0] in wanted:
                labels.append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "questions.json").write_text(json.dumps(items, indent=1) + "\n")
    (args.out / "gold-new.json").write_text(json.dumps(new) + "\n")
    (args.out / "gold-old.json").write_text(json.dumps(old) + "\n")
    with (args.out / "labels.jsonl").open("w", encoding="utf-8") as out:
        for row in labels:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "artifact_kind": "xgraph.mquake",
        "version": 1,
        "cases": len(cases),
        "asked": len(items),
        "edits": len(edits),
        "conflicting_edits": conflicts,
        "kg_triples": len(triples),
        "edited_kg_triples": len(edited),
        "covered_asked_cases": len(covered_asked),
        "edit_labels": len(extra) if extra is not None else None,
        "dated_triples": len(dated) if dated is not None else None,
        "excluded_cases": len(excluded),
        "base_kg": str(base_kg) if base_kg else None,
        "provenance": provenance(
            [args.data, args.kg / "triples.npy"]
            + ([args.dated] if args.dated else [])
            + ([args.exclude] if args.exclude else [])
        ),
    }
    (args.out / "covered-cases.json").write_text(json.dumps(covered_asked) + "\n")
    with (args.out / "requests-edits.jsonl").open("w", encoding="utf-8") as out:
        for request in edit_requests(asked, args.per_case):
            out.write(json.dumps(request, ensure_ascii=False) + "\n")
    if args.worlds:  # each asked case in a world without edits that conflict with it
        placed, held, versions = {}, [], {}
        for k, world in enumerate(worlds(asked, cases)):
            out_kg = args.out_kg.with_name(f"{args.out_kg.name}-w{k}")
            world_triples = apply_edits(triples, world["edits"])
            versions[f"w{k}"] = version_edits(world["edits"], dated)
            if not args.no_world_kgs:
                write_variant(args.kg, world_triples, out_kg, extra_labels=extra)
            members = [c for c in asked if c["case_id"] in set(world["cases"])]
            held += covered_cases(members, world_triples)
            placed.update({f"mquake-{c}": k for c in world["cases"]})
        (args.out / "worlds.json").write_text(json.dumps(placed) + "\n")
        (args.out / "covered-world-cases.json").write_text(json.dumps(sorted(held)) + "\n")
        manifest["worlds"] = max(placed.values()) + 1
        manifest["covered_world_cases"] = len(held)
        if args.kg_versions:  # one version per world, over the KG itself
            from xgraph.kgversions import write_kg_versions

            write_kg_versions(
                args.kg_versions,
                versions,
                base_kg=str(args.kg),
                base_triples=len(base),
                provenance=manifest["provenance"],
            )
            manifest["kg_versions"] = str(args.kg_versions)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "provenance"}))


if __name__ == "__main__":
    main()
