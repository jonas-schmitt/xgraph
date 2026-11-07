"""The retrieval arms (`docs/quality-gate.md`, run setup): KG triples in the prompt.

The topic entities of a question are the entities its gold SPARQL query names, or with
`--linked`, the entities linked from its text (`xgraph.linking`), as a deployed system
would find them. Their
one-hop triples in the gate KG, outgoing then incoming, each ordered by relation
label, are capped at 100 per question and listed before the question as
`subject · relation · object` lines, with canonical English labels. The few-shot
instructions and examples are the baseline's. With `--matched-demos` (matched
demonstrations, `matched_messages`), they are path mode's instead: its example questions
with as many hops each (1, 2, 3, and 3), each asked after a few one-hop facts of the
entity it starts from (DEMO_FACTS) and answered as a chain of fact spans, under the
baseline's instructions without the sentence that invites escapes, as path mode's prompt
has none. Iterative retrieval (xgraph.iterative) lists facts for more hops.

    python -m xgraph.rag --test data/wwq/test.json --kg data/wikidata/gate-kg \\
        --out target/run/rag-requests.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from xgraph.prompts import CHAIN_EXAMPLES, wwq_messages
from xgraph.wwq import NAMED, load_wwq

CAP = 100
CHUNK = 1 << 22


def one_hop(
    questions: dict[str, set[int]], kg: str | Path, cap: int = CAP
) -> dict[str, list[tuple[str, str, str]]]:
    """Per question, the one-hop triples of its topic entities, as label triples."""
    kg = Path(kg)
    topics = np.array(sorted(set().union(*questions.values())), dtype=np.int64)
    triples = np.load(kg / "triples.npy", mmap_mode="r")
    found = []
    for at in range(0, len(triples), CHUNK):
        chunk = np.asarray(triples[at : at + CHUNK], dtype=np.int64)
        hit = np.isin(chunk[:, 0], topics) | np.isin(chunk[:, 2], topics)
        found.append(chunk[hit])
    rows = np.concatenate(found) if found else np.zeros((0, 3), dtype=np.int64)

    items = {int(x) for x in rows[:, [0, 2]].ravel()} | {int(t) for t in topics}
    names = _labels(kg / "entity_labels.jsonl", {f"Q{i}" for i in items})
    relations = _labels(kg / "relation_labels.jsonl", {f"P{int(p)}" for p in rows[:, 1]})

    def label_triple(s: int, p: int, o: int) -> tuple[str, str, str] | None:
        s_, p_, o_ = names.get(f"Q{s}"), relations.get(f"P{p}"), names.get(f"Q{o}")
        return (s_, p_, o_) if s_ and p_ and o_ else None

    result = {}
    for question, entities in questions.items():
        mine = np.array(sorted(entities), dtype=np.int64)
        outgoing = rows[np.isin(rows[:, 0], mine)]
        incoming = rows[np.isin(rows[:, 2], mine) & ~np.isin(rows[:, 0], mine)]
        listed: list[tuple[str, str, str]] = []
        for part in (outgoing, incoming):
            labelled = {t for s, p, o in part.tolist() if (t := label_triple(s, p, o))}
            listed.extend(sorted(labelled, key=lambda t: (t[1], t[0], t[2])))
        result[question] = listed[:cap]
    return result


def _labels(path: Path, wanted: set[str]) -> dict[str, str]:
    """The canonical English label of each wanted id that has one."""
    labels: dict[str, str] = {}
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            item, label, kind = json.loads(line)
            if kind == "label" and item in wanted:
                labels.setdefault(item, label)
    return labels


def facts_block(triples: list[tuple[str, str, str]]) -> str:
    lines = "\n".join(f"- {s} · {r} · {o}" for s, r, o in triples) or "(none)"
    return f"Facts from Wikidata:\n{lines}"


def rag_messages(
    question: str, triples: list[tuple[str, str, str]], matched: bool = False
) -> list[dict]:
    """The baseline's instructions and examples, then the facts and the question; with
    matched, the matched demonstrations (matched_messages) instead."""
    if matched:
        return matched_messages(question, triples)
    messages = wwq_messages(question)
    messages[-1] = {"role": "user", "content": f"{facts_block(triples)}\n\nQuestion: {question}"}
    return messages


MATCHED_INSTRUCTIONS = """\
Answer the question with a short rationale, then the answer on a last line of its own.

Write every entity as a span with its English Wikidata label, like ⟦ Marie Curie⟧, \
and every fact you use as a span of subject, Wikidata property, and object, like \
⟦ Marie Curie · employer · University of Paris⟧. Use the exact labels Wikidata uses. \
When the question takes several steps, write each step as a fact span whose subject is \
the object of the step before, and answer with the object of the last step.

End with the line "Answer: ⟦ label⟧", naming one entity."""
"""The retrieval prompt for matched demonstrations: the baseline's instructions without
the sentence that invites escapes (path mode's prompt has none), and with the chain
prompt's sentence on multi-hop answers."""

DEMO_FACTS: dict[str, list[tuple[str, str, str]]] = {
    "Steve Jobs": [
        ("Steve Jobs", "country of citizenship", "United States of America"),
        ("Steve Jobs", "educated at", "Reed College"),
        ("Steve Jobs", "employer", "Apple Inc."),
        ("Steve Jobs", "place of birth", "San Francisco"),
        ("Steve Jobs", "spouse", "Laurene Powell Jobs"),
        ("Apple Inc.", "founded by", "Steve Jobs"),
        ("NeXT", "founded by", "Steve Jobs"),
    ],
    "Mount Kilimanjaro": [
        ("Mount Kilimanjaro", "country", "Tanzania"),
        ("Mount Kilimanjaro", "instance of", "stratovolcano"),
        ("Mount Kilimanjaro", "located in protected area", "Kilimanjaro National Park"),
        ("Kibo", "part of", "Mount Kilimanjaro"),
    ],
    "Tanzania": [
        ("Tanzania", "capital", "Dodoma"),
        ("Tanzania", "continent", "Africa"),
        ("Tanzania", "currency", "Tanzanian shilling"),
        ("Tanzania", "official language", "Swahili"),
        ("Tanzania", "shares border with", "Kenya"),
        ("Mount Kilimanjaro", "country", "Tanzania"),
    ],
    "The Hobbit": [
        ("The Hobbit", "author", "J. R. R. Tolkien"),
        ("The Hobbit", "genre", "high fantasy"),
        ("The Hobbit", "instance of", "literary work"),
        ("The Hobbit", "publisher", "George Allen & Unwin"),
        ("The Lord of the Rings", "follows", "The Hobbit"),
    ],
    "J. R. R. Tolkien": [
        ("J. R. R. Tolkien", "country of citizenship", "United Kingdom"),
        ("J. R. R. Tolkien", "educated at", "Exeter College"),
        ("J. R. R. Tolkien", "employer", "University of Oxford"),
        ("J. R. R. Tolkien", "notable work", "The Lord of the Rings"),
        ("J. R. R. Tolkien", "place of birth", "Bloemfontein"),
        ("J. R. R. Tolkien", "place of death", "Bournemouth"),
        ("The Hobbit", "author", "J. R. R. Tolkien"),
    ],
    "Bloemfontein": [
        ("Bloemfontein", "country", "South Africa"),
        ("Bloemfontein", "instance of", "city"),
        (
            "Bloemfontein",
            "located in the administrative territorial entity",
            "Mangaung Metropolitan Municipality",
        ),
        ("J. R. R. Tolkien", "place of birth", "Bloemfontein"),
    ],
    "Harry Potter and the Philosopher's Stone": [
        (
            "Harry Potter and the Philosopher's Stone",
            "author",
            "J. K. Rowling",
        ),
        (
            "Harry Potter and the Philosopher's Stone",
            "followed by",
            "Harry Potter and the Chamber of Secrets",
        ),
        ("Harry Potter and the Philosopher's Stone", "genre", "fantasy literature"),
        ("Harry Potter and the Philosopher's Stone", "publisher", "Bloomsbury Publishing"),
        (
            "Harry Potter and the Chamber of Secrets",
            "follows",
            "Harry Potter and the Philosopher's Stone",
        ),
    ],
    "J. K. Rowling": [
        ("J. K. Rowling", "country of citizenship", "United Kingdom"),
        ("J. K. Rowling", "educated at", "University of Exeter"),
        ("J. K. Rowling", "notable work", "Harry Potter"),
        ("J. K. Rowling", "place of birth", "Yate"),
        ("Harry Potter and the Philosopher's Stone", "author", "J. K. Rowling"),
    ],
    "United Kingdom": [
        ("United Kingdom", "capital", "London"),
        ("United Kingdom", "continent", "Europe"),
        ("United Kingdom", "currency", "pound sterling"),
        ("United Kingdom", "official language", "English"),
        ("J. K. Rowling", "country of citizenship", "United Kingdom"),
    ],
}
"""The demonstrations' retrieved facts, per entity, listed as one_hop lists them
(outgoing, then incoming, each by relation label): a few of the entity's Wikidata facts."""

MATCHED_STARTS = [
    "Steve Jobs",
    "Mount Kilimanjaro",
    "The Hobbit",
    "Harry Potter and the Philosopher's Stone",
]
"""The entity each of path mode's examples (prompts.PATH_EXAMPLES) starts from."""


def matched_messages(question: str, triples: list[tuple[str, str, str]]) -> list[dict]:
    """One-hop retrieval with matched demonstrations: path mode's examples, the same
    questions with as many hops each (prompts.CHAIN_EXAMPLES, answered as a chain of
    fact spans), each asked after the one-hop facts of the entity it starts from as the
    question is asked after its own, under MATCHED_INSTRUCTIONS."""
    messages = [{"role": "system", "content": MATCHED_INSTRUCTIONS}]
    for example, start in zip(CHAIN_EXAMPLES, MATCHED_STARTS, strict=True):
        shown = f"{facts_block(DEMO_FACTS[start])}\n\nQuestion: {example['question']}"
        messages.append({"role": "user", "content": shown})
        messages.append({"role": "assistant", "content": example["response"]})
    messages.append({"role": "user", "content": f"{facts_block(triples)}\n\nQuestion: {question}"})
    return messages


UPDATES = (
    "Facts updated in Wikidata after your training; they replace what you remember, and "
    "for everything else use your own knowledge:"
)


def edit_messages(question: str, triples: list[tuple[str, str, str]]) -> list[dict]:
    """The in-context editing prompt: rag_messages' instructions and examples, with the
    facts introduced as updates over the model's own knowledge (as MeLLo and IKE frame
    edits) rather than as the facts to answer from."""
    lines = "\n".join(f"- {s} · {r} · {o}" for s, r, o in triples) or "(none)"
    messages = wwq_messages(question)
    messages[-1] = {"role": "user", "content": f"{UPDATES}\n{lines}\n\nQuestion: {question}"}
    return messages


COPY = (
    "Facts from the knowledge graph. They may differ from what you remember; where they do, "
    "the knowledge graph is right. Answer from these facts alone, and copy the answer's name "
    "exactly as the facts write it:"
)


def copy_messages(question: str, triples: list[tuple[str, str, str]]) -> list[dict]:
    """The copying prompt (xgraph.interventions' read-copy arm): rag_messages' instructions
    and examples, with the facts framed as edit_messages frames them, but as the only
    source of the answer, whose name is to be copied from them."""
    lines = "\n".join(f"- {s} · {r} · {o}" for s, r, o in triples) or "(none)"
    messages = wwq_messages(question)
    messages[-1] = {"role": "user", "content": f"{COPY}\n{lines}\n\nQuestion: {question}"}
    return messages


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m xgraph.rag", description=__doc__)
    parser.add_argument("--test", type=Path, required=True, help="a WWQ split (.json)")
    parser.add_argument("--kg", type=Path, required=True, help="gate KG directory")
    parser.add_argument("--out", type=Path, required=True, help="requests (JSONL)")
    parser.add_argument("--cap", type=int, default=CAP)
    parser.add_argument(
        "--linked",
        action="store_true",
        help="topic entities linked from the question text (xgraph.linking), not the gold query's",
    )
    parser.add_argument(
        "--matched-demos",
        action="store_true",
        help="path mode's example questions and hops as demonstrations (matched_messages)",
    )
    args = parser.parse_args(argv)
    questions = load_wwq(args.test)
    if args.linked:
        from xgraph.linking import link

        topics = {
            k: set(v) for k, v in link({q.id: q.utterance for q in questions}, args.kg).items()
        }
    else:
        topics = {q.id: {int(n) for n in NAMED.findall(q.sparql)} for q in questions}
    found = one_hop(topics, args.kg, args.cap)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for q in questions:
            messages = rag_messages(q.utterance, found[q.id], args.matched_demos)
            request = {"id": q.id, "messages": messages}
            out.write(json.dumps(request, ensure_ascii=False) + "\n")
    print(f"{len(questions)} requests, {sum(map(len, found.values()))} facts listed")


if __name__ == "__main__":
    main()
