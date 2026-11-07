"""Naturalized questions for path-writing examples, written by a model and checked.

Path-writing examples (xgraph.pathdata) ask templated questions (`What is the place of
birth of the author of The Hobbit?`), so that a model trained on them alone is untested
on natural questions. This module has a model rewrite them, in two steps around a
generation run (xgraph.generate, the baseline arm):

1. `requests` writes one chat request per example, in xgraph.generate's request format,
   asking for a natural question from the start entity's label and the relation labels
   in order, read from the example's reply. The request names nothing the path reaches:
   no intermediate entity and no answer.
2. `mix` reads the generations back (`--rewrites`), checks each rewrite
   (`rewrite_problem`), and writes the training data: the examples in their order, a
   share of them (`--share`) with an accepted rewrite as the question
   (`question_source` naturalized, the templated one kept as `templated_question`), the
   reply unchanged; `<out>.manifest.json` counts the rewrites accepted and rejected by
   reason, with the settings and provenance.

**The check** is lexical and cheap. A rewrite is rejected for its form (blank, more
than one line, longer than MAX_QUESTION characters, span syntax, or cut off at the
token limit); when it does not name the start by its label (`no start`); when it names
an entity the path reaches, by any of its labels (`leak`: matched as whole words,
ignoring case, or as written for labels of up to three characters such as `UK`; labels
within the start's label, or within a label or alias of the path's relations, do not
count); when a relation of the path is named neither by a word of its label or aliases
(words compared by their first five letters, function words left out) nor by one of them
whole (`dropped hop`); or when a relation's label or alias that holds a word other than
function words is written more often than the path holds the relation (`repeated hop`,
longer ones matched first).

**Reconstruction.** A rewrite can also be read back by a model: with
`--reconstruction-requests`, `mix` writes, for every rewrite the lexical check accepts,
a request that asks for the relations the question follows, in order, from the path's
relation labels shuffled with DISTRACTORS others of the file's relations. With their
generations as `--reconstructions`, a rewrite is accepted only when its reconstruction
is the path's relation labels in order (`reconstruction` otherwise).

    python -m xgraph.naturalize requests --examples target/pathdata/train-balanced.jsonl \\
        --out target/pathdata/naturalize-requests.jsonl
    python -m xgraph.naturalize mix --examples target/pathdata/train-balanced.jsonl \\
        --kg data/wikidata/gate-kg --rewrites <generations.jsonl> \\
        --reconstruction-requests target/pathdata/reconstruct-requests.jsonl
    python -m xgraph.naturalize mix --examples target/pathdata/train-balanced.jsonl \\
        --kg data/wikidata/gate-kg --rewrites <generations.jsonl> \\
        --reconstructions <their generations.jsonl> --share 0.5 --seed 0 --out <mixed.jsonl>

xgraph.pathdata's `examples` takes a question writer in-process (`ask`); this is its
batched counterpart, for a model served by vLLM.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from xgraph.pathdata import PREPOSITIONS
from xgraph.syntax import SpanSyntax

REWRITE_INSTRUCTIONS = (
    "You turn a relation path into a natural English question. The path starts at an "
    "entity and follows relations outwards, one after the other; the question asks for "
    "the entity the last relation reaches. Write one fluent question a person might ask: "
    "name the start entity exactly as given, and ask for every relation in order, each "
    "once, in your own words. Name no other entity, do not guess the answer, and do not "
    "answer. Reply with the question alone.\n\n"
    "For example, for the start entity The Hobbit and the relations 1. author, 2. place "
    "of birth: Where was the author of The Hobbit born?"
)
RECONSTRUCT_INSTRUCTIONS = (
    "You read a question about an entity and list the relations it follows from that "
    "entity to the answer, in order. Use only labels from the given list, a label more "
    "than once if the question follows it more than once. Reply with the labels "
    "separated by ' · ' and nothing else."
)
MAX_QUESTION = 300
"""The longest rewrite, in characters, that is kept."""
DISTRACTORS = 3
"""Relation labels off the path offered in a reconstruction request."""
STEM = 5
FUNCTION_WORDS = PREPOSITIONS | {
    "the", "a", "an", "is", "was", "are", "were", "be", "what", "which", "who", "whom",
    "whose", "where", "when", "and", "or", "name", "that", "this", "its", "his", "her",
    "their", "does", "did", "has", "had", "have",
}  # fmt: skip
_WORD = re.compile(r"\w+")
_SYNTAX = SpanSyntax()


def _jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            if line.strip():
                yield json.loads(line)


def path_labels(example: dict, syntax: SpanSyntax = _SYNTAX) -> tuple[str, list[str]]:
    """The start's label and the relation labels, read from the example's path span."""
    reply = example["messages"][-1]["content"]
    first = reply.split("\n", 1)[0]
    opening = syntax.opener + syntax.label_prefix
    if not first.startswith(opening) or syntax.closer not in first:
        raise ValueError(f"example {example['id']!r} has no path span: {first!r}")
    start, *relations = first[len(opening) : first.index(syntax.closer)].split(
        syntax.separator + syntax.label_prefix
    )
    if len(relations) != len(example["walk"]["relations"]):
        raise ValueError(f"example {example['id']!r}: its path span and walk differ")
    return start, relations


def rewrite_request(example: dict) -> dict:
    """The request for a natural question about the example's path."""
    start, relations = path_labels(example)
    listed = "\n".join(f"{i}. {label}" for i, label in enumerate(relations, 1))
    content = f"Start entity: {start}\nRelations, from the start entity outwards:\n{listed}"
    return {
        "id": example["id"],
        "messages": [
            {"role": "system", "content": REWRITE_INSTRUCTIONS},
            {"role": "user", "content": content},
        ],
    }


@dataclass(frozen=True)
class Names:
    """The labels the check reads: every label of the entities on the examples' walks,
    and each relation's aliases."""

    entities: dict[int, list[str]]
    relation_aliases: dict[int, list[str]]


def load_names(kg: str | Path, examples: Iterable[dict]) -> Names:
    """The labels of every entity the examples walk and the relations' aliases, from a
    KG directory (the entity labels streamed)."""
    from xgraph.pathdata import _relations

    kg = Path(kg)
    wanted = {int(e[1:]) for example in examples for e in example["walk"]["entities"]}
    entities: dict[int, list[str]] = {}
    with (kg / "entity_labels.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            item, label, _ = json.loads(line)
            if (number := int(item[1:])) in wanted:
                entities.setdefault(number, []).append(label)
    _, aliases = _relations(kg)
    return Names(entities, aliases)


def _named(label: str, text: str) -> int:
    """How often the label is written in the text as whole words: as written when it has
    up to three characters, ignoring case otherwise."""
    flags = 0 if len(label) <= 3 else re.IGNORECASE
    return len(re.findall(rf"(?<!\w){re.escape(label)}(?!\w)", text, flags))


def _stems(text: str) -> set[str]:
    return {
        w[:STEM] for w in _WORD.findall(text.lower()) if len(w) >= 3 and w not in FUNCTION_WORDS
    }


def rewrite_problem(question: str, example: dict, names: Names) -> str | None:
    """Why the lexical check rejects a rewrite of the example's question, or None."""
    if (
        not question
        or "\n" in question
        or len(question) > MAX_QUESTION
        or any(d.strip() in question for d in _SYNTAX.delimiters)
    ):
        return "form"
    start, labels = path_labels(example)
    rest = re.sub(re.escape(start), " ", question, flags=re.IGNORECASE)
    if rest == question:
        return "no start"
    relations = [int(p[1:]) for p in example["walk"]["relations"]]
    phrases = {
        r: [label, *names.relation_aliases.get(r, [])]
        for r, label in zip(relations, labels, strict=True)
    }
    spoken = " | ".join(p.lower() for options in phrases.values() for p in options)
    for entity in example["walk"]["entities"][1:]:
        for label in names.entities.get(int(entity[1:]), []):
            if len(label) < 2 or label.lower() in start.lower() or label.lower() in spoken:
                continue
            if _named(label, rest):
                return "leak"
    words = _stems(rest)
    for options in phrases.values():
        if not words & _stems(" ".join(options)) and not any(_named(p, rest) for p in options):
            return "dropped hop"
    written: Counter[int] = Counter()
    remaining = rest
    ranked = sorted(
        ((p, r) for r, ps in phrases.items() for p in ps if _stems(p)), key=lambda pr: -len(pr[0])
    )
    for phrase, relation in ranked:
        pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
        flags = 0 if len(phrase) <= 3 else re.IGNORECASE
        written[relation] += len(re.findall(pattern, remaining, flags))
        remaining = re.sub(pattern, " ", remaining, flags=flags)
    held = Counter(relations)
    if any(written[r] > held[r] for r in held):
        return "repeated hop"
    return None


def reconstruction_request(
    example: dict, question: str, distractors: Sequence[str], rng: np.random.Generator
) -> dict:
    """The request to read the relations back from a rewrite, choosing from the path's
    relation labels and the distractors, shuffled."""
    start, labels = path_labels(example)
    offered = list(dict.fromkeys([*labels, *distractors]))
    offered = [offered[i] for i in rng.permutation(len(offered))]
    content = f"Question: {question}\nStart entity: {start}\nRelation labels: " + "; ".join(offered)
    return {
        "id": example["id"],
        "messages": [
            {"role": "system", "content": RECONSTRUCT_INSTRUCTIONS},
            {"role": "user", "content": content},
        ],
    }


def reconstructed(text: str, example: dict) -> bool:
    """Whether a reconstruction is the example's relation labels in order."""
    parts = [p.strip().strip(".").strip().lower() for p in re.split(r"[·\n;]", text)]
    return [p for p in parts if p] == [label.lower() for label in path_labels(example)[1]]


def _generations(path: Path) -> dict[str, dict]:
    return {record["id"]: record for record in _jsonl(path)}


def check(
    examples: Sequence[dict],
    rewrites: dict[str, dict],
    names: Names,
    reconstructions: dict[str, dict] | None = None,
) -> tuple[dict[str, str], Counter[str], int]:
    """The accepted rewrites by example id, the rejected ones counted by reason, and the
    examples without a rewrite."""
    accepted: dict[str, str] = {}
    rejected: Counter[str] = Counter()
    missing = 0
    for example in examples:
        record = rewrites.get(example["id"])
        if record is None:
            missing += 1
            continue
        question = record["text"].strip().strip('"').strip()
        problem = "form" if record.get("finish_reason") == "length" else None
        problem = problem or rewrite_problem(question, example, names)
        if problem is None and reconstructions is not None:
            back = reconstructions.get(example["id"])
            if back is None or not reconstructed(back["text"], example):
                problem = "reconstruction"
        if problem is None:
            accepted[example["id"]] = question
        else:
            rejected[problem] += 1
    return accepted, rejected, missing


def mix(examples: Sequence[dict], accepted: dict[str, str], share: float, seed: int) -> list[dict]:
    """The examples in their order, round(share * len(examples)) of them (or every one
    with an accepted rewrite, if fewer) asking the rewrite instead, drawn with the seed."""
    from xgraph.prompts import wwq_messages

    if not 0.0 <= share <= 1.0:
        raise ValueError(f"share must be in [0, 1], got {share}")
    candidates = [e["id"] for e in examples if e["id"] in accepted]
    rng = np.random.default_rng(seed)
    wanted = min(round(share * len(examples)), len(candidates))
    chosen = {candidates[i] for i in rng.permutation(len(candidates))[:wanted]}
    mixed = []
    for example in examples:
        if example["id"] not in chosen:
            mixed.append(example)
            continue
        question = accepted[example["id"]]
        mixed.append(
            {
                **example,
                "question": question,
                "question_source": "naturalized",
                "templated_question": example["question"],
                "messages": [
                    *wwq_messages(question, path_spans=True, shots=False),
                    example["messages"][-1],
                ],
            }
        )
    return mixed


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    from xgraph.provenance import provenance

    parser = argparse.ArgumentParser(prog="python -m xgraph.naturalize", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    asked = commands.add_parser("requests", help="rewrite requests (xgraph.generate's format)")
    asked.add_argument("--examples", type=Path, required=True, help="path-writing examples")
    asked.add_argument("--out", type=Path, required=True, help="requests (JSONL)")
    mixed = commands.add_parser("mix", help="check the rewrites and mix them in")
    mixed.add_argument("--examples", type=Path, required=True, help="path-writing examples")
    mixed.add_argument("--kg", type=Path, required=True, help="the examples' KG directory")
    mixed.add_argument("--rewrites", type=Path, required=True, help="generations (JSONL)")
    mixed.add_argument(
        "--reconstruction-requests", type=Path, help="write reconstruction requests here"
    )
    mixed.add_argument(
        "--reconstructions", type=Path, help="their generations: accept only rewrites read back"
    )
    mixed.add_argument("--share", type=float, default=0.5, help="share of naturalized questions")
    mixed.add_argument("--seed", type=int, default=0)
    mixed.add_argument("--out", type=Path, help="the mixed examples (JSONL)")
    args = parser.parse_args(argv)

    examples = list(_jsonl(args.examples))
    if args.command == "requests":
        _write_jsonl(args.out, (rewrite_request(e) for e in examples))
        print(f"{len(examples)} rewrite requests")
        return

    if not (args.out or args.reconstruction_requests):
        parser.error("mix needs --out, --reconstruction-requests, or both")
    names = load_names(args.kg, examples)
    rewrites = _generations(args.rewrites)
    if args.reconstruction_requests:
        lexical, _, _ = check(examples, rewrites, names)
        pool = sorted({p for e in examples for p in e["walk"]["relations"]})
        labels = {p: path_labels(e)[1][i] for e in examples
                  for i, p in enumerate(e["walk"]["relations"])}  # fmt: skip
        rng = np.random.default_rng(args.seed)
        requests = []
        for example in examples:
            if example["id"] not in lexical:
                continue
            off_path = [p for p in pool if p not in example["walk"]["relations"]]
            drawn = rng.permutation(len(off_path))[:DISTRACTORS]
            distractors = [labels[off_path[i]] for i in drawn]
            requests.append(
                reconstruction_request(example, lexical[example["id"]], distractors, rng)
            )
        _write_jsonl(args.reconstruction_requests, requests)
        print(f"{len(requests)} reconstruction requests")
    if not args.out:
        return
    backs = _generations(args.reconstructions) if args.reconstructions else None
    accepted, rejected, missing = check(examples, rewrites, names, backs)
    written = mix(examples, accepted, args.share, args.seed)
    _write_jsonl(args.out, written)
    inputs = [args.examples, args.rewrites, *([args.reconstructions] if backs else [])]
    inputs += [args.kg / "entity_labels.jsonl", args.kg / "relation_labels.jsonl"]
    manifest = {
        "artifact_kind": "xgraph.naturalized_examples",
        "version": 1,
        "examples": len(written),
        "naturalized": sum(e.get("question_source") == "naturalized" for e in written),
        "accepted": len(accepted),
        "rejected": dict(rejected.most_common()),
        "missing": missing,
        "settings": {
            "share": args.share,
            "seed": args.seed,
            "reconstruction": backs is not None,
            "max_question": MAX_QUESTION,
            "distractors": DISTRACTORS,
        },
        "provenance": provenance(inputs),
    }
    args.out.with_name(args.out.name + ".manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n"
    )
    print(f"{manifest['naturalized']} of {len(written)} questions naturalized")


if __name__ == "__main__":
    main()
