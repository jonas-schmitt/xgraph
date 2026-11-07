"""Robustness probes of the quality gate (`docs/quality-gate.md`, run setup).

- Missing facts: for a seeded sample of questions whose gold answer is one triple
  away from an entity the gold query names, every KG triple linking those named
  entities and gold entities, in either direction, is removed.
- Changed facts: for a seeded sample of questions whose gold query has a single
  triple pattern, each gold triple (named entity, property, gold entity) in the KG
  is replaced by one whose object is another entity of the gold entity's class
  (instance of, P31), chosen with the seed among those not gold and not linked.
- Ambiguous names: the questions whose named entity or gold entity has a label or
  alias that another KG entity also has.

With --exclude, only questions that name no entity of another split take part (held-out
entities, for evaluating models trained on that split). A variant KG directory links the
original label files and holds its own triples, so `python -m xgraph.wikidata index`
builds its KG index as usual.

    python -m xgraph.probes --questions data/wwq/test.json \\
        --gold eval/wwq-gold-wdqs-2026-10-12.json --kg data/wikidata/gate-kg \\
        --out data/wikidata/probes
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

from xgraph.wwq import NAMED, Question, load_wwq

INSTANCE_OF = 31
_PATTERN = re.compile(r"\bwdt:P(\d+)\b")


def single_pattern(sparql: str) -> bool:
    """One named entity, one direct property, and nothing else that constrains."""
    body = sparql.upper()
    if any(word in body for word in ("FILTER", "OPTIONAL", "UNION", "MINUS", " P:", "PS:", "PQ:")):
        return False
    return len(NAMED.findall(sparql)) == 1 and len(_PATTERN.findall(sparql)) == 1


def _named(question: Question) -> set[int]:
    return {int(n) for n in NAMED.findall(question.sparql)}


def _gold(gold: dict, question: Question) -> set[int]:
    answers = gold["answers"].get(question.id) or {"entities": []}
    return {int(e[1:]) for e in answers["entities"]}


def _links(triples: np.ndarray, named: set[int], answers: set[int]) -> np.ndarray:
    """Rows linking a named entity and a gold entity, in either direction."""
    s, o = triples[:, 0], triples[:, 2]
    a, b = np.array(sorted(named)), np.array(sorted(answers))
    return (np.isin(s, a) & np.isin(o, b)) | (np.isin(s, b) & np.isin(o, a))


def remove_facts(
    triples: np.ndarray, questions: list[Question], gold: dict, n: int, seed: int
) -> tuple[np.ndarray, list[str]]:
    """The triples without the sampled questions' links, and the sampled question ids."""
    eligible = [
        q
        for q in questions
        if _named(q) and _gold(gold, q) and _links(triples, _named(q), _gold(gold, q)).any()
    ]
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(eligible), size=min(n, len(eligible)), replace=False))
    drop = np.zeros(len(triples), dtype=bool)
    for i in chosen:
        drop |= _links(triples, _named(eligible[i]), _gold(gold, eligible[i]))
    return triples[~drop], [eligible[i].id for i in chosen]


def change_facts(
    triples: np.ndarray,
    questions: list[Question],
    gold: dict,
    n: int,
    seed: int,
    plausible: bool = False,
) -> tuple[np.ndarray, dict[str, tuple[int, int, int, int]]]:
    """The triples with the sampled questions' gold triples moved to a same-class entity.

    With plausible, the new object must also be an object of the same property elsewhere
    in the KG (a country for "country"), not only share a class with the old one.
    Returns the new triples and, per question, (subject, property, old, new object).
    """
    objects_of: dict[int, set[int]] = {}
    rng = np.random.default_rng(seed)
    classes = triples[triples[:, 1] == INSTANCE_OF]
    eligible = []
    for q in questions:
        if not single_pattern(q.sparql) or not _gold(gold, q):
            continue
        (subject,) = _named(q)
        (prop,) = (int(p) for p in _PATTERN.findall(q.sparql))
        rows = triples[(triples[:, 0] == subject) & (triples[:, 1] == prop)]
        hits = [int(o) for o in rows[:, 2] if int(o) in _gold(gold, q)]
        if hits:
            eligible.append((q, subject, prop, hits))
    order = rng.permutation(len(eligible))
    swaps: dict[str, tuple[int, int, int, int]] = {}
    drop = np.zeros(len(triples), dtype=bool)
    added = []
    for i in order:
        if len(swaps) == n:
            break
        q, subject, prop, hits = eligible[i]
        old = hits[0]
        kinds = classes[classes[:, 0] == old][:, 2]
        if not len(kinds):
            continue
        linked = set(triples[(triples[:, 0] == subject) & (triples[:, 1] == prop)][:, 2].tolist())
        same = classes[np.isin(classes[:, 2], kinds)][:, 0]
        pool = set(same.tolist())
        if plausible:
            if prop not in objects_of:
                objects_of[prop] = set(triples[triples[:, 1] == prop][:, 2].tolist())
            pool &= objects_of[prop]
        candidates = sorted(pool - _gold(gold, q) - linked - {subject})
        if not candidates:
            continue
        new = int(candidates[rng.integers(len(candidates))])
        drop |= (
            (triples[:, 0] == subject)
            & (triples[:, 1] == prop)
            & np.isin(triples[:, 2], sorted(_gold(gold, q)))
        )
        added.append((subject, prop, new))
        swaps[q.id] = (subject, prop, old, new)
    changed = np.concatenate([triples[~drop], np.array(added, dtype=triples.dtype).reshape(-1, 3)])
    return changed, swaps


def write_variant(
    kg: str | Path,
    triples: np.ndarray,
    out: str | Path,
    extra_labels: list[list[str]] | None = None,
) -> None:
    """A KG directory with these triples and the KG's labels (linked); with extra_labels,
    its entity labels are a copy of the KG's followed by those rows."""
    kg, out = Path(kg).resolve(), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("entity_labels.jsonl", "relation_labels.jsonl", "sitelinks.jsonl"):
        target = out / name
        target.unlink(missing_ok=True)
        if name == "entity_labels.jsonl" and extra_labels:
            shutil.copyfile(kg / name, target)
            with target.open("a", encoding="utf-8") as labels:
                for row in extra_labels:
                    labels.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            target.symlink_to(kg / name)
    np.save(out / "triples.npy", triples.astype(np.int32))


def ambiguous_questions(questions: list[Question], gold: dict, kg: str | Path) -> list[str]:
    """Questions with a named or gold entity whose label or alias another entity shares."""
    from xgraph.score import normalize

    involved = {q.id: _named(q) | _gold(gold, q) for q in questions}
    wanted = {f"Q{i}" for items in involved.values() for i in items}
    path = Path(kg) / "entity_labels.jsonl"
    their: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            item, label, _ = json.loads(line)
            if item in wanted:
                their.setdefault(item, set()).add(normalize(label))
    texts = set().union(*their.values()) if their else set()
    owners: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            item, label, _ = json.loads(line)
            key = normalize(label)
            if key in texts:
                owners.setdefault(key, set()).add(item)
    shared = {key for key, items in owners.items() if len(items) > 1}
    return [
        q.id for q in questions if any(their.get(f"Q{i}", set()) & shared for i in involved[q.id])
    ]


def held_out(questions: list[Question], other: list[Question]) -> list[Question]:
    """The questions that name no entity any question of the other split names."""
    seen = set().union(*(_named(q) for q in other)) if other else set()
    return [q for q in questions if not _named(q) & seen]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m xgraph.probes", description=__doc__)
    parser.add_argument("--questions", type=Path, required=True, help="a WWQ split (.json)")
    parser.add_argument("--gold", type=Path, required=True, help="gold answer file")
    parser.add_argument("--kg", type=Path, required=True, help="gate KG directory")
    parser.add_argument("--out", type=Path, required=True, help="directory for the variants")
    parser.add_argument("--missing", type=int, default=200, help="questions to remove facts of")
    parser.add_argument("--changed", type=int, default=100, help="questions to change facts of")
    parser.add_argument(
        "--exclude", type=Path, help="a split whose named entities the questions must not name"
    )
    parser.add_argument(
        "--plausible", action="store_true", help="changed objects already objects of the property"
    )
    args = parser.parse_args(argv)
    questions = load_wwq(args.questions)
    if args.exclude:
        questions = held_out(questions, load_wwq(args.exclude))
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    triples = np.load(args.kg / "triples.npy")
    kept, missing = remove_facts(triples, questions, gold, n=args.missing, seed=1)
    write_variant(args.kg, kept, args.out / "missing-facts-kg")
    del kept
    changed, swaps = change_facts(
        triples, questions, gold, n=args.changed, seed=2, plausible=args.plausible
    )
    write_variant(args.kg, changed, args.out / "changed-facts-kg")
    report = {
        "artifact_kind": "xgraph.probes",
        "version": 1,
        "missing_facts": missing,
        "changed_facts": {k: list(v) for k, v in swaps.items()},
        "ambiguous_names": ambiguous_questions(questions, gold, args.kg),
    }
    (args.out / "probes.json").write_text(json.dumps(report, indent=1) + "\n")
    print({k: len(v) for k, v in report.items() if isinstance(v, (list, dict))})


if __name__ == "__main__":
    main()
