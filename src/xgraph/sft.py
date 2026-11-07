"""Fine-tuning data: WWQ train questions answered with real KG facts in the span format.

The prompted model loses most answers by picking the wrong relation or object although
the gold answer is one KG triple away, or by looping under the mask
(docs/post-gate-findings.md). Fine-tuning on the span format is the feasibility
study's remedy. Each example is a WWQ train question (never dev or test) with the
gate's instructions as the system message and, as the reply, the gate KG's triples
that link an entity the gold query names to a gold answer, in either direction, as
fact spans with canonical English labels (at most three, ordered by relation label;
those with a property the gold query uses, when any link, since those answer the
question),
then the answer as an entity span. Where the KG holds no linking triple, the reply
names the topic and the answer as entity spans only, so that the model does not learn
to invent facts.

    python -m xgraph.wwq gold --test data/wwq/train.json --out target/sft/wwq-train-gold.json
    python -m xgraph.sft wwq --train data/wwq/train.json --gold target/sft/wwq-train-gold.json \\
        --kg data/wikidata/gate-kg --out target/sft/train.jsonl

Fine-tuning on short WWQ replies alone shortens biographies; `people` and `replay`
build long-form replay examples: biographies the prompted constrained model wrote of
people sampled from the KG (none of them FActScore topics), kept when they finished
without a loop.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from xgraph.prompts import INSTRUCTIONS
from xgraph.syntax import SpanSyntax
from xgraph.wwq import NAMED

MAX_FACTS = 3
QUERY_PROPERTY = re.compile(r"\bwdt:P(\d+)\b")
CHUNK = 1 << 22

_S = SpanSyntax()


def _span(*segments: str) -> str:
    joined = (_S.separator + _S.label_prefix).join(segments)
    return f"{_S.opener}{_S.label_prefix}{joined}{_S.closer}"


def target(facts: list[tuple[str, str, str]], answer: str, topic: str | None = None) -> str:
    """The reply: the linking facts as fact spans (or the topic alone), then the answer."""
    if facts:
        body = " and ".join(_span(*fact) for fact in facts) + "."
    else:
        body = f"{_span(topic)}: {_span(answer)}." if topic else f"{_span(answer)}."
    return f"{body}\nAnswer: {_span(answer)}"


def _labels(path: Path, wanted: set[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            item, label, kind = json.loads(line)
            if kind == "label" and item in wanted and _S.closer not in label:
                found.setdefault(item, label)
    return found


def examples(train: str | Path, gold_file: dict, kg: str | Path) -> list[dict]:
    """One chat example per train question with a gold entity the KG can name.

    gold_file is a gold answer file of the train split (python -m xgraph.wwq gold).
    """
    kg = Path(kg)
    questions = json.loads(Path(train).read_text(encoding="utf-8"))
    answers = gold_file["answers"]
    parsed = []
    for q in questions:
        entities = answers.get(q["id"], {}).get("entities", [])
        gold = sorted({int(e[1:]) for e in entities})
        named = sorted({int(n) for n in NAMED.findall(q["sparql"])})
        if gold:
            parsed.append((q, named, gold))

    pairs = {(n << 32) | g for _, named, gold in parsed for n in named for g in gold}
    codes = np.array(sorted(pairs), dtype=np.int64)
    links: dict[int, list[tuple[int, int, int]]] = {}
    triples = np.load(kg / "triples.npy", mmap_mode="r")
    for at in range(0, len(triples), CHUNK):
        chunk = np.asarray(triples[at : at + CHUNK], dtype=np.int64)
        for a, b in ((0, 2), (2, 0)):  # named -> gold, then gold -> named
            hit = np.isin((chunk[:, a] << 32) | chunk[:, b], codes)
            for s, p, o in chunk[hit].tolist():
                links.setdefault((min(s, o) << 32) | max(s, o), []).append((s, p, o))

    items = {f"Q{i}" for _, named, gold in parsed for i in (*named, *gold)}
    items |= {f"Q{x}" for found in links.values() for s, _, o in found for x in (s, o)}
    names = _labels(kg / "entity_labels.jsonl", items)
    props = {f"P{p}" for found in links.values() for _, p, _ in found}
    relations = _labels(kg / "relation_labels.jsonl", props)

    out = []
    for q, named, gold in parsed:
        answer = next((names[f"Q{g}"] for g in gold if f"Q{g}" in names), None)
        if answer is None:
            continue
        facts: dict[tuple[str, str, str], int] = {}
        for n in named:
            for g in gold:
                for s, p, o in links.get((min(n, g) << 32) | max(n, g), []):
                    labelled = (names.get(f"Q{s}"), relations.get(f"P{p}"), names.get(f"Q{o}"))
                    if all(labelled):
                        facts[labelled] = p
        # The gold query's properties say which facts answer the question; other linking
        # facts are used only when none of those link the entities.
        asked = {int(p) for p in QUERY_PROPERTY.findall(q["sparql"])}
        answering = [f for f, p in facts.items() if p in asked] or list(facts)
        chosen = sorted(answering, key=lambda f: (f[1], f[0], f[2]))[:MAX_FACTS]
        topic = next((names[f"Q{n}"] for n in named if f"Q{n}" in names), None)
        reply = target(chosen, answer, topic)
        out.append(
            {
                "id": q["id"],
                "facts": len(chosen),
                "messages": [
                    {"role": "system", "content": INSTRUCTIONS},
                    {"role": "user", "content": q["utterance"]},
                    {"role": "assistant", "content": reply},
                ],
            }
        )
    return out


def counterfactual(
    train: str | Path, gold_file: dict, kg: str | Path, n: int, seed: int, out: str | Path
) -> tuple[list[dict], dict[str, tuple[int, int, int, int]]]:
    """Examples from two KG worlds: for n train questions drawn with the seed, the gold
    triple is moved to another entity of the gold entity's class that is already an object
    of the same property (xgraph.probes, plausible changes) and the
    example states the changed fact and answers with its object; the other questions keep
    their examples from the original KG. The changed KG is written to `out`. Each example
    records its world, "changed" or "original"."""
    from xgraph.probes import change_facts, write_variant
    from xgraph.wwq import load_wwq

    triples = np.load(Path(kg) / "triples.npy")
    changed, swaps = change_facts(
        triples, load_wwq(train), gold_file, n=n, seed=seed, plausible=True
    )
    write_variant(kg, changed, out)
    del triples, changed
    moved = {
        "answers": {
            **gold_file["answers"],
            **{q: {"entities": [f"Q{new}"], "literals": []} for q, (*_, new) in swaps.items()},
        },
        "failed": gold_file.get("failed", {}),
    }
    built = [{**e, "world": "changed"} for e in examples(train, moved, out) if e["id"] in swaps]
    built += [
        {**e, "world": "original"} for e in examples(train, gold_file, kg) if e["id"] not in swaps
    ]
    return built, swaps


HUMAN = 5  # instance of (P31) human (Q5)


def sample_people(kg: str | Path, n: int, seed: int, exclude: set[str]) -> list[str]:
    """Labels of n people (instances of human with a label) drawn with the seed."""
    kg = Path(kg)
    triples = np.load(kg / "triples.npy", mmap_mode="r")
    people: set[int] = set()
    for at in range(0, len(triples), CHUNK):
        chunk = np.asarray(triples[at : at + CHUNK])
        people.update(chunk[(chunk[:, 1] == 31) & (chunk[:, 2] == HUMAN)][:, 0].tolist())
    names = _labels(kg / "entity_labels.jsonl", {f"Q{i}" for i in people})
    pool = sorted(label for label in set(names.values()) if label not in exclude)
    rng = np.random.default_rng(seed)
    return [pool[i] for i in sorted(rng.choice(len(pool), size=min(n, len(pool)), replace=False))]


def replay(requests: list[dict], outputs: list[dict]) -> list[dict]:
    """Chat examples from a model's own finished outputs: their prompt and reply.

    Outputs cut at the token limit, or repeating one span four or more times, are left
    out, so that the examples keep long-form behaviour without its loops.
    """
    by_id = {r["id"]: r for r in requests}
    kept = []
    for out in outputs:
        spans_written = re.findall(re.escape(_S.opener) + ".*?" + re.escape(_S.closer), out["text"])
        looping = any(spans_written.count(s) >= 4 for s in set(spans_written))
        if out["finish_reason"] != "stop" or looping:
            continue
        messages = [*by_id[out["id"]]["messages"], {"role": "assistant", "content": out["text"]}]
        kept.append({"id": out["id"], "facts": None, "messages": messages})
    return kept


def main(argv: list[str] | None = None) -> None:
    from xgraph.prompts import bio_messages

    parser = argparse.ArgumentParser(prog="python -m xgraph.sft", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    wwq = commands.add_parser("wwq", help="examples from WWQ train questions and the KG")
    wwq.add_argument("--train", type=Path, required=True, help="WWQ train.json")
    wwq.add_argument("--gold", type=Path, required=True, help="the train split's gold answers")
    wwq.add_argument("--kg", type=Path, required=True, help="gate KG directory")
    wwq.add_argument("--out", type=Path, required=True, help="examples (JSONL)")
    cf = commands.add_parser("counterfactual", help="WWQ examples from original and changed KGs")
    cf.add_argument("--train", type=Path, required=True, help="WWQ train.json")
    cf.add_argument("--gold", type=Path, required=True, help="the train split's gold answers")
    cf.add_argument("--kg", type=Path, required=True, help="gate KG directory")
    cf.add_argument("--changed", type=int, required=True, help="questions to change facts of")
    cf.add_argument("--seed", type=int, default=0)
    cf.add_argument("--changed-kg", type=Path, required=True, help="changed KG directory")
    cf.add_argument("--out", type=Path, required=True, help="examples (JSONL)")
    people = commands.add_parser("people", help="biography requests for sampled KG people")
    people.add_argument("--kg", type=Path, required=True, help="gate KG directory")
    people.add_argument("--exclude", type=Path, required=True, help="FActScore data.zip")
    people.add_argument("--n", type=int, default=500)
    people.add_argument("--seed", type=int, default=3)
    people.add_argument("--out", type=Path, required=True, help="requests (JSONL)")
    again = commands.add_parser("replay", help="examples from a model's finished outputs")
    again.add_argument("--requests", type=Path, required=True, help="the requests (JSONL)")
    again.add_argument("--outputs", type=Path, required=True, help="their outputs (JSONL)")
    again.add_argument("--out", type=Path, required=True, help="examples (JSONL)")
    args = parser.parse_args(argv)

    def lines(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    if args.command == "wwq":
        built = examples(args.train, json.loads(args.gold.read_text()), args.kg)
        note = f"{sum(e['facts'] > 0 for e in built)} with linking facts"
    elif args.command == "counterfactual":
        built, swaps = counterfactual(
            args.train,
            json.loads(args.gold.read_text()),
            args.kg,
            args.changed,
            args.seed,
            args.changed_kg,
        )
        (args.changed_kg / "swaps.json").write_text(
            json.dumps({q: list(s) for q, s in swaps.items()}, indent=1) + "\n"
        )
        note = f"{len(swaps)} answering from the changed KG"
    elif args.command == "people":
        import zipfile

        with zipfile.ZipFile(args.exclude) as z:
            excluded = {
                name
                for part in ("unlabeled", "labeled")
                for name in z.read(f"data/{part}/prompt_entities.txt").decode().splitlines()
            }
        chosen = sample_people(args.kg, args.n, args.seed, excluded)
        built = [{"id": name, "messages": bio_messages(name)} for name in chosen]
        note = "biography requests"
    else:
        built = replay(lines(args.requests), lines(args.outputs))
        note = "replayed"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for example in built:
            out.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"{len(built)} examples, {note}")


if __name__ == "__main__":
    main()
