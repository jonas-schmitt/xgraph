"""The overlap registry: what each confirmation case shares with data seen before it.

A confirmation set is fresh only in the ways it is shown to be. The overlap registry
records, for every asked MQuAKE case (by its id in the confirmation questions,
`mquake-<case>`), what it shares with each named reference set, so that a claim on a
stratum names the overlaps the stratum excludes. It is written before any confirmation
output is generated.

**References** reduce to the same parts; a part a reference cannot know is null, not
false:

- `--mquake NAME=FILE`: a MQuAKE release (development data such as CF-3k). Questions,
  original chains (`orig.triples`), edited chains (`orig.new_triples`), entities and
  triples of both, edit triples and edited (subject, relation) pairs, and relation
  sequences (xgraph.mquake.relation_sequence).
- `--chains NAME=FILE`: QID chains by question, as `python -m xgraph.mintqa_bench
  questions` writes `chains.json` (MINTQA's representable questions). Chains, their
  entities, triples, and relation sequences.
- `--mintqa-rows NAME=SPLIT=FILE` (repeatable, one name gathering several splits) with
  `--mintqa-relations` (the MINTQA KG's relation_labels.jsonl): every MINTQA question as
  the dataset writes it, representable or not. Questions, and relation sequences with
  each relation named by its label (xgraph.expressions.Relations; a name that names no
  relation stops the build).
- `--training NAME=FILE`: a path-mode training corpus (`python -m xgraph.pathdata`):
  each example's question, and its walk as a chain, entities, triples, and relation
  sequence.

Per case and reference: `question` (one of the case's questions, normalized by
question_key, among the reference's), `chain` (the case's original chain is one of the
reference's chains), `edited_chain` (its edited chain is one of the reference's edited
chains), `entities` (the shared entity ids of its chains and edits), `triples` (the
shared triples of its original and edited chains), `edit_triples` (its edit triples the
reference also edits), `edited_pairs` (its edited pairs the reference also edits), and
`relation_sequence` (the reference holds its relation sequence).

**Demonstrations** are what the prompts show before the question:

- `--demos NAME=FILE`: a request file (JSON lines with `messages`). Every distinct
  prefix before a request's last user turn (the system prompt and the demonstration
  turns) is read; the last user turn, the case's own question and context, is not.
  Spans (xgraph.score.spans) give entity labels (entity spans, and the first and, for a
  span of three parts whose middle names a relation, the last part of every other span),
  facts (three parts, the middle naming a relation through `--relations`, a KG's
  relation_labels.jsonl), and relation sequences: a span whose parts after the first all
  name relations (a path), and the relations of each reply's facts in order. A span of
  three parts that names two relations counts both as a path and as a fact.
- `--demo-text NAME=FILE`: prompts as free text (such as the GWalk prompts). A question
  or entity label counts when it occurs in the text (both normalized by question_key,
  at word boundaries), and relation sequences come from paths written
  `start->(relation)->...->answer`; facts are not read (null).

Per case and demonstration source: `question`, `entities` (case entity ids with a label
among the demonstrations' labels, or occurring in their text), `triples` (case triples
whose subject label, relation, and object label a demonstrated fact writes), and
`relation_sequence`. A case entity's labels are those its MQuAKE file gives it, and
with `--entity-labels` (a KG's entity_labels.jsonl) also its labels and aliases there.

**Strata** (`--stratum NAME=COND,COND,...`) are the cases that share none of the named
overlaps: a condition is `<part>:<reference>` or `<part>:demo:<source>`, met when the
part is false or empty. The confirmatory test's composition stratum, for instance, is
`relation_sequence:cf3k,relation_sequence:mintqa`; a condition on a part the reference
leaves null stops the build. Each stratum gives its cases, their relation sequences,
and its case ids. `summary` counts, per reference and source, the cases with each
overlap.

    python -m xgraph.overlap registry --data data/mquake/MQuAKE-Remastered-CF-6334.json \\
        --questions target/confirm/mqr-cf6334/questions.json \\
        --mquake cf3k=data/mquake/MQuAKE-Remastered-CF-3k.json \\
        --chains mintqa=target/mintqa/chains.json \\
        --training train=target/pathdata/train.jsonl \\
        --relations data/wikidata/gate-kg-v3/relation_labels.jsonl \\
        --demos paths=target/confirm/mqr-cf6334/requests-paths.jsonl \\
        --stratum unseen=relation_sequence:cf3k,relation_sequence:mintqa \\
        --out target/confirm/overlap/cf6334.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from xgraph.expressions import Relations
from xgraph.mquake import case_id, relation_sequence
from xgraph.score import normalize, spans

Triple = tuple[str, str, str]
Chain = tuple[Triple, ...]

PARTS = (
    "question",
    "chain",
    "edited_chain",
    "entities",
    "triples",
    "edit_triples",
    "edited_pairs",
    "relation_sequence",
)
"""A case's overlaps with a reference, in the registry's order."""

DEMO_PARTS = ("question", "entities", "triples", "relation_sequence")
"""A case's overlaps with a demonstration source."""

ARROW_PATH = re.compile(r"[^\"\n>]*?((?:->\([^()\n]+\))+)->answer")
"""A relation path written as GWalk's prompts write one: `start->(r1)->(r2)->answer`."""


def question_key(text: str) -> str:
    """A question or label as compared: normalized (xgraph.score.normalize), every run of
    characters other than letters and digits one space."""
    return re.sub(r"[\W_]+", " ", normalize(text)).strip()


@dataclass
class Reference:
    """What a reference set holds; a part it cannot know is None."""

    questions: set[str] | None = None
    chains: set[Chain] | None = None
    edited_chains: set[Chain] | None = None
    entities: set[str] | None = None
    triples: set[Triple] | None = None
    edit_triples: set[Triple] | None = None
    edited_pairs: set[tuple[str, str]] | None = None
    sequences: set[str] | None = None


@dataclass
class Demonstrations:
    """What prompt demonstrations show: questions, entity labels, facts as (subject label,
    relation, object label), and relation sequences; with `text` set, questions and
    labels are looked for in it, and facts is None."""

    questions: set[str] = field(default_factory=set)
    labels: set[str] = field(default_factory=set)
    facts: set[Triple] | None = field(default_factory=set)
    sequences: set[str] = field(default_factory=set)
    text: str | None = None


def _chain(triples: list[list[str]]) -> Chain:
    return tuple(tuple(t) for t in triples)


def _sequence(chain: Chain) -> str:
    return ">".join(p for _, p, _ in chain)


def _entities(triples) -> set[str]:
    return {x for s, _, o in triples for x in (s, o)}


def mquake_reference(cases: list[dict]) -> Reference:
    """A MQuAKE release as a reference."""
    ref = Reference(set(), set(), set(), set(), set(), set(), set(), set())
    for case in cases:
        orig, new = _chain(case["orig"]["triples"]), _chain(case["orig"]["new_triples"])
        edits = _chain(case["orig"]["edit_triples"])
        ref.questions.update(question_key(q) for q in case["questions"])
        ref.chains.add(orig)
        ref.edited_chains.add(new)
        ref.triples.update(orig + new)
        ref.entities.update(_entities(orig + new + edits))
        ref.edit_triples.update(edits)
        ref.edited_pairs.update((s, p) for s, p, _ in edits)
        ref.sequences.add(relation_sequence(case))
    return ref


def chains_reference(chains: dict[str, list[list[str]]]) -> Reference:
    """QID chains by question (MINTQA's chains.json) as a reference."""
    ref = Reference(chains=set(), entities=set(), triples=set(), sequences=set())
    for chain in map(_chain, chains.values()):
        ref.chains.add(chain)
        ref.triples.update(chain)
        ref.entities.update(_entities(chain))
        ref.sequences.add(_sequence(chain))
    return ref


def mintqa_rows_reference(rows: dict[str, dict], relations: Relations) -> Reference:
    """MINTQA's questions as the dataset writes them (xgraph.mintqa_bench.dataset_rows):
    their texts and relation sequences, each relation named by its label; a name that
    names several relations gives every sequence it can stand for."""
    ref = Reference(questions=set(), sequences=set())
    for qid, row in rows.items():
        ref.questions.add(question_key(row["question"]))
        named = []
        for _, name, _ in row["triple"]:
            props = relations.props(name)
            if not props:
                raise ValueError(f"{qid}: no relation is named {name!r}")
            named.append(sorted(f"P{p}" for p in props))
        ref.sequences.update(">".join(seq) for seq in itertools.product(*named))
    return ref


def training_reference(path: str | Path) -> Reference:
    """A path-mode training corpus (JSON lines with `question` and `walk`), read line by
    line, as a reference."""
    ref = Reference(questions=set(), chains=set(), entities=set(), triples=set(), sequences=set())
    with Path(path).open(encoding="utf-8") as lines:
        for n, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            example = json.loads(line)
            walk = example.get("walk")
            if not walk or len(walk["entities"]) != len(walk["relations"]) + 1:
                raise ValueError(f"{path}:{n}: no walk of n relations over n + 1 entities")
            ents, rels = walk["entities"], walk["relations"]
            chain = tuple((ents[k], r, ents[k + 1]) for k, r in enumerate(rels))
            ref.questions.add(question_key(example["question"]))
            ref.chains.add(chain)
            ref.triples.update(chain)
            ref.entities.update(ents)
            ref.sequences.add(_sequence(chain))
    return ref


def _relation_ids(name: str, relations: Relations) -> list[str]:
    return sorted(f"P{p}" for p in relations.props(name))


def _read_spans(text: str, relations: Relations, demos: Demonstrations) -> None:
    """The labels, facts, and relation sequences of one message's spans."""
    chained = []
    for kind, parts in spans(text):
        if kind == "escaped":
            continue
        demos.labels.add(question_key(parts[0]))
        if kind == "entity":
            continue
        named = [_relation_ids(p, relations) for p in parts[1:]]
        if all(named):  # a path: every part after its start names a relation
            demos.sequences.update(">".join(s) for s in itertools.product(*named))
        if len(parts) == 3 and named[0]:  # a fact, or a path that reads as one
            demos.labels.add(question_key(parts[2]))
            facts = {(question_key(parts[0]), p, question_key(parts[2])) for p in named[0]}
            demos.facts.update(facts)
            if not named[1]:
                chained.append(named[0])
    if chained:
        demos.sequences.update(">".join(s) for s in itertools.product(*chained))


def request_demonstrations(path: str | Path, relations: Relations) -> Demonstrations:
    """The demonstrations of a request file: every distinct prefix before a request's
    last user turn, its spans read for labels, facts, and relation sequences."""
    demos = Demonstrations()
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as lines:
        for n, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            last = max((k for k, m in enumerate(messages) if m["role"] == "user"), default=None)
            if last is None:
                raise ValueError(f"{path}:{n}: a request without a user turn")
            prefix = messages[:last]
            key = json.dumps(prefix, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            for message in prefix:
                if message["role"] == "user":
                    demos.questions.add(question_key(message["content"]))
                _read_spans(message["content"], relations, demos)
    return demos


def text_demonstrations(text: str, relations: Relations) -> Demonstrations:
    """Prompts as free text: questions and labels are looked for in it, and relation
    paths written `start->(r1)->...->answer` read from it."""
    demos = Demonstrations(facts=None, text=f" {question_key(text)} ")
    for found in ARROW_PATH.finditer(text):
        names = re.findall(r"->\(([^()\n]+)\)", found.group(1))
        named = [_relation_ids(name.strip(), relations) for name in names]
        if all(named):
            demos.sequences.update(">".join(s) for s in itertools.product(*named))
    return demos


def _sorted(items) -> list:
    return sorted(list(x) if isinstance(x, tuple) else x for x in items)


def _overlap(case: dict, questions: set[str], ref: Reference) -> dict:
    orig, new = _chain(case["orig"]["triples"]), _chain(case["orig"]["new_triples"])
    edits = _chain(case["orig"]["edit_triples"])

    def shared(mine, theirs):
        return None if theirs is None else _sorted(set(mine) & theirs)

    def held(mine, theirs):
        return None if theirs is None else mine in theirs

    return {
        "question": None if ref.questions is None else bool(questions & ref.questions),
        "chain": held(orig, ref.chains),
        "edited_chain": held(new, ref.edited_chains),
        "entities": shared(_entities(orig + new + edits), ref.entities),
        "triples": shared(orig + new, ref.triples),
        "edit_triples": shared(edits, ref.edit_triples),
        "edited_pairs": shared({(s, p) for s, p, _ in edits}, ref.edited_pairs),
        "relation_sequence": held(relation_sequence(case), ref.sequences),
    }


def _case_labels(
    case: dict, extra: dict[str, set[str]]
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Each entity of the case's chains with its labels: the MQuAKE file's, and those
    and extra's (free text is searched for the first only: KG aliases can be as short
    and common as any word)."""
    own: dict[str, set[str]] = {}
    for ids, named in (("triples", "triples_labeled"), ("new_triples", "new_triples_labeled")):
        for (s, _, o), (sl, _, ol) in zip(case["orig"][ids], case["orig"][named], strict=True):
            own.setdefault(s, set()).add(question_key(sl))
            own.setdefault(o, set()).add(question_key(ol))
    every = {i: names | {question_key(n) for n in extra.get(i, ())} for i, names in own.items()}
    return own, every


def _demo_overlap(case: dict, questions: set[str], demos: Demonstrations, case_labels) -> dict:
    triples = set(_chain(case["orig"]["triples"]) + _chain(case["orig"]["new_triples"]))
    own, labels = case_labels
    if demos.text is not None:
        shown = lambda key: bool(key) and f" {key} " in demos.text  # noqa: E731
        return {
            "question": any(shown(q) for q in questions),
            "entities": sorted(i for i, names in own.items() if any(map(shown, names))),
            "triples": None,
            "relation_sequence": relation_sequence(case) in demos.sequences,
        }
    facts = demos.facts
    return {
        "question": bool(questions & demos.questions),
        "entities": sorted(i for i, names in labels.items() if names & demos.labels),
        "triples": _sorted(
            (s, p, o)
            for s, p, o in triples
            if any((a, p, b) in facts for a in labels[s] for b in labels[o])
        ),
        "relation_sequence": relation_sequence(case) in demos.sequences,
    }


def registry(
    cases: list[dict],
    asked: list[dict],
    references: dict[str, Reference],
    demonstrations: dict[str, Demonstrations],
    entity_labels: dict[str, set[str]] | None = None,
) -> dict[str, dict]:
    """Each asked case (by the case ids of the confirmation questions) with its relation
    sequence and its overlaps with every reference and demonstration source."""
    utterances: dict[str, set[str]] = {}
    for question in asked:
        utterances.setdefault(case_id(question["id"]), set()).add(
            question_key(question["utterance"])
        )
    by_id = {f"mquake-{c['case_id']}": c for c in cases}
    missing = sorted(set(utterances) - set(by_id))
    if missing:
        raise ValueError(f"asked cases the MQuAKE file does not hold: {missing[:5]}")
    records = {}
    for qid in sorted(utterances, key=lambda q: int(q.rsplit("-", 1)[1])):
        case = by_id[qid]
        questions = utterances[qid] | {question_key(q) for q in case["questions"]}
        labels = _case_labels(case, entity_labels or {})
        records[qid] = {
            "relation_sequence": relation_sequence(case),
            "references": {n: _overlap(case, questions, r) for n, r in references.items()},
            "demonstrations": {
                n: _demo_overlap(case, questions, d, labels) for n, d in demonstrations.items()
            },
        }
    return records


def _counts(values: list) -> int | None:
    return None if values and values[0] is None else sum(bool(v) for v in values)


def summary(records: dict[str, dict]) -> dict:
    """The cases and relation sequences, and per reference and source the cases with
    each overlap (None where it is not determined)."""
    first = next(iter(records.values()), {"references": {}, "demonstrations": {}})

    def count(group: str, name: str, parts) -> dict:
        return {p: _counts([r[group][name][p] for r in records.values()]) for p in parts}

    return {
        "cases": len(records),
        "sequences": len({r["relation_sequence"] for r in records.values()}),
        "references": {n: count("references", n, PARTS) for n in first["references"]},
        "demonstrations": {
            n: count("demonstrations", n, DEMO_PARTS) for n in first["demonstrations"]
        },
    }


def strata(records: dict[str, dict], specs: dict[str, list[str]]) -> dict[str, dict]:
    """Each named stratum: the cases that share none of its conditions' overlaps
    (`<part>:<reference>` or `<part>:demo:<source>`)."""
    first = next(iter(records.values()), {"references": {}, "demonstrations": {}})
    found = {}
    for name, conditions in specs.items():
        tests = []
        for condition in conditions:
            part, _, where = condition.partition(":")
            group, source = ("demonstrations", where[5:]) if where.startswith("demo:") else (
                "references", where)  # fmt: skip
            if part not in (DEMO_PARTS if group == "demonstrations" else PARTS):
                raise ValueError(f"stratum {name}: unknown overlap {part!r} in {condition!r}")
            if source not in first[group]:
                raise ValueError(f"stratum {name}: unknown reference {source!r} in {condition!r}")
            if first[group][source][part] is None:
                raise ValueError(f"stratum {name}: {source} does not determine {part}")
            tests.append((group, source, part))
        ids = [
            qid
            for qid, record in records.items()
            if not any(record[g][s][p] for g, s, p in tests)
        ]  # fmt: skip
        found[name] = {
            "conditions": conditions,
            "cases": len(ids),
            "sequences": len({records[q]["relation_sequence"] for q in ids}),
            "ids": ids,
        }
    return found


def _named(spec: str, flag: str) -> tuple[str, str]:
    name, sep, value = spec.partition("=")
    if not sep or not name or not value:
        raise SystemExit(f"{flag} takes NAME=VALUE, not {spec!r}")
    return name, value


def main(argv: list[str] | None = None) -> None:
    from xgraph.mintqa_bench import dataset_rows
    from xgraph.provenance import provenance
    from xgraph.score import load_labels

    parser = argparse.ArgumentParser(prog="python -m xgraph.overlap", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("registry", help="write the overlap registry of asked cases")
    build.add_argument("--data", type=Path, required=True, help="the cases' MQuAKE JSON")
    build.add_argument("--questions", type=Path, required=True, help="the asked questions.json")
    build.add_argument("--mquake", action="append", default=[], help="NAME=MQuAKE JSON")
    build.add_argument("--chains", action="append", default=[], help="NAME=chains.json")
    build.add_argument("--mintqa-rows", action="append", default=[], help="NAME=SPLIT=FILE")
    build.add_argument("--mintqa-relations", type=Path, help="the MINTQA KG's relation labels")
    build.add_argument("--training", action="append", default=[], help="NAME=corpus JSONL")
    build.add_argument("--demos", action="append", default=[], help="NAME=request JSONL")
    build.add_argument("--demo-text", action="append", default=[], help="NAME=prompt text")
    build.add_argument("--relations", type=Path, help="a KG's relation_labels.jsonl")
    build.add_argument("--entity-labels", type=Path, help="a KG's entity_labels.jsonl")
    build.add_argument("--stratum", action="append", default=[], help="NAME=COND,COND,...")
    build.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if (args.demos or args.demo_text) and not args.relations:
        parser.error("--demos and --demo-text read relation names: they need --relations")
    if args.mintqa_rows and not args.mintqa_relations:
        parser.error("--mintqa-rows names relations by label: it needs --mintqa-relations")
    inputs = [args.data, args.questions]
    references: dict[str, Reference] = {}

    def add(name: str, ref: Reference) -> None:
        if name in references:
            parser.error(f"two references are named {name!r}")
        references[name] = ref

    for spec in args.mquake:
        name, path = _named(spec, "--mquake")
        inputs.append(Path(path))
        add(name, mquake_reference(json.loads(Path(path).read_text(encoding="utf-8"))))
    for spec in args.chains:
        name, path = _named(spec, "--chains")
        inputs.append(Path(path))
        add(name, chains_reference(json.loads(Path(path).read_text(encoding="utf-8"))))
    splits: dict[str, list[str]] = {}
    for spec in args.mintqa_rows:
        name, split = _named(spec, "--mintqa-rows")
        splits.setdefault(name, []).append(split)
        inputs.append(Path(_named(split, "--mintqa-rows NAME=")[1]))
    if splits:
        inputs.append(args.mintqa_relations)
        mintqa_relations = Relations.read(args.mintqa_relations)
        for name, specs in splits.items():
            add(name, mintqa_rows_reference(dataset_rows(specs), mintqa_relations))
    for spec in args.training:
        name, path = _named(spec, "--training")
        inputs.append(Path(path))
        add(name, training_reference(path))
    demonstrations: dict[str, Demonstrations] = {}
    if args.relations:
        inputs.append(args.relations)
        relations = Relations.read(args.relations)
        for spec in args.demos:
            name, path = _named(spec, "--demos")
            inputs.append(Path(path))
            demonstrations[name] = request_demonstrations(path, relations)
        for spec in args.demo_text:
            name, path = _named(spec, "--demo-text")
            inputs.append(Path(path))
            text = Path(path).read_text(encoding="utf-8")
            demonstrations[name] = text_demonstrations(text, relations)
    if len(demonstrations) != len(args.demos) + len(args.demo_text):
        parser.error("two demonstration sources share a name")
    cases = json.loads(args.data.read_text(encoding="utf-8"))
    asked = json.loads(args.questions.read_text(encoding="utf-8"))
    extra = None
    if args.entity_labels:
        inputs.append(args.entity_labels)
        asked_ids = {case_id(q["id"]) for q in asked}
        wanted = {
            x
            for c in cases
            if f"mquake-{c['case_id']}" in asked_ids
            for key in ("triples", "new_triples")
            for s, _, o in c["orig"][key]
            for x in (s, o)
        }
        extra = load_labels([args.entity_labels], wanted)
    records = registry(cases, asked, references, demonstrations, extra)
    specs = {}
    for spec in args.stratum:
        name, conditions = _named(spec, "--stratum")
        specs[name] = conditions.split(",")
    result = {
        "artifact_kind": "xgraph.overlap",
        "version": 1,
        "references": sorted(references),
        "demonstrations": sorted(demonstrations),
        "summary": summary(records),
        "strata": strata(records, specs),
        "cases": records,
        "provenance": provenance(inputs),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    shown = {k: v for k, v in result["summary"].items()}
    shown["strata"] = {n: {"cases": s["cases"], "sequences": s["sequences"]}
                       for n, s in result["strata"].items()}  # fmt: skip
    print(json.dumps(shown))


if __name__ == "__main__":
    main()
