"""Real updates: facts that changed after a cutoff, as single- and multi-hop questions.

Wikidata records when many relations' current values began (the start-time qualifier,
P580): a country's head of government, a team's head coach, a company's chief executive.
A value that began after a model's training cutoff is one the model cannot know, while
the gate KG (built from a later dump) holds it. This module fetches such statements from
the Wikidata Query Service, pairs each subject's latest value with the value it replaced
(the predecessor whose end time is latest), keeps the updates the KG holds, and writes
WWQ-format questions about them: one hop ("Who is the head of government of X?") and two
hops through a stable fact pointing at the updated subject ("... of the country Y is a
citizen of?", with Y the best-connected such entity in the KG), and two hops with the
update first, asking a fact about the new value ("Where was the current head of
government of X born?"). Updates whose old value the KG still holds are left out, as
the new answer would be ambiguous. Unchanged controls ask the same templates about
subjects whose value did not change. Gold files give the new (current) and the old
answer.

    python -m xgraph.updates fetch --since 2024-01-01 --out data/updates
    python -m xgraph.updates build --raw data/updates --db data/wikidata/gate-kg.sqlite \\
        --kg data/wikidata/gate-kg --since 2024-01-01 --until 2026-10-07 --out target/updates
    python -m xgraph.updates sample --questions target/updates/questions.json --cap 150 \\
        --out target/updates/eval-questions.json
    python -m xgraph.updates oracle --questions target/updates/eval-questions.json \\
        --db data/wikidata/gate-kg.sqlite --out target/updates/requests-oracle.jsonl

`oracle` writes the update-in-prompt arm: each question with the fact it turns on (the
subject's current value under the question's property) in the prompt.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

PROPERTIES = {
    "P6": ("head of government", [("P27", "the country {x} is a citizen of"),
                                  ("P17", "the country where {x} is")]),
    "P35": ("head of state", [("P27", "the country {x} is a citizen of"),
                              ("P17", "the country where {x} is")]),
    "P286": ("head coach", [("P466", "the team based at {x}", True)]),
    "P488": ("chairperson", [("P749", "the parent organization of {x}")]),
    "P169": ("chief executive officer", [("P176", "the company that makes {x}")]),
    "P1037": ("director", [("P749", "the parent organization of {x}")]),
}  # fmt: skip
"""Updated property: its question noun, and the first hops that lead to its subject: a
property and a phrase, the hop (x, property, subject), or with True (subject, property,
x) read backwards (a stadium's occupant). x must have that property only once."""

AFTER = [
    ("P19", "Where was the current {noun} of {name} born?"),
    ("P102", "Which political party is the current {noun} of {name} a member of?"),
    ("P69", "Where was the current {noun} of {name} educated?"),
    ("P27", "Which country is the current {noun} of {name} a citizen of?"),
]
"""Updated-first-hop chains: a fact about the updated value, asked through the update.
Asked when the KG holds the fact for both the new and the old value and the answers
differ, so the old value's answer is the old answer."""

QUERY = (
    "SELECT ?s ?o ?start ?old ?oldEnd WHERE {{ ?s p:{p} ?st . ?st ps:{p} ?o ; pq:P580 ?start . "
    'FILTER(?start >= "{since}"^^xsd:dateTime) OPTIONAL {{ ?s p:{p} ?ost . ?ost ps:{p} ?old ; '
    "pq:P582 ?oldEnd . FILTER(?old != ?o) }} }}"
)
ENDPOINT = "https://query.wikidata.org/sparql"
AGENT = "xgraph-research/0.1 (research use; constrained decoding over Wikidata)"
HUMAN = 5
"""Q5 (human): a citizenship hop must go through a person (instance of, P31)."""
COUNTRIES = {6256, 3624078}
"""Country and sovereign state: a hop phrased "the country ..." needs a subject of these."""


@dataclass(frozen=True)
class Update:
    subject: str
    prop: str
    new: str
    old: str
    start: str


def _item(cell: dict) -> str:
    return cell["value"].rsplit("/", 1)[-1]


def _is_item(name: str) -> bool:
    """A Wikidata item id, not a blank node (an unknown or some value)."""
    return name[:1] == "Q" and name[1:].isdigit()


def updates_from(
    bindings: list[dict], prop: str, since: str, until: str | None = None
) -> list[Update]:
    """Each subject's latest value starting on or after since (and, given until, on or
    before it: a later start is a planned succession or an error), with the predecessor
    whose end time is latest; subjects without a known predecessor are left out."""
    latest: dict[str, tuple[str, str]] = {}
    for row in bindings:
        start = row["start"]["value"][:10]
        if start < since or (until is not None and start > until):
            continue
        subject, new = _item(row["s"]), _item(row["o"])
        if not (_is_item(subject) and _is_item(new)):
            continue
        if subject not in latest or start > latest[subject][1]:
            latest[subject] = (new, start)
    olds: dict[str, tuple[str, str]] = {}
    for row in bindings:
        subject = _item(row["s"])
        if "old" not in row or subject not in latest or _item(row["o"]) != latest[subject][0]:
            continue
        old, end = _item(row["old"]), row["oldEnd"]["value"][:10]
        if (
            _is_item(old)
            and old != latest[subject][0]
            and (subject not in olds or end > olds[subject][1])
        ):
            olds[subject] = (old, end)
    return [
        Update(subject, prop, new, olds[subject][0], start)
        for subject, (new, start) in sorted(latest.items())
        if subject in olds
    ]


def _number(item: str) -> int:
    return int(item[1:])


def _first_hop(store, subject: int, prop: str, backwards: bool = False) -> int | None:
    """The best-connected entity x with a fact (x, prop, subject), or with backwards
    (subject, prop, x), whose prop has the one value that leads to the subject."""
    from xgraph.score import normalize

    number = _number(prop)
    name = normalize(store.label(subject) or "")
    if backwards:
        found = [x for x in store.objects_of([subject], number) if
                 len(store.subjects_of(x, number)) == 1]  # fmt: skip
    else:
        found = [x for x in store.subjects_of(subject, number)
                 if len(store.objects_of([x], number)) == 1]  # fmt: skip
    if prop in ("P27", "P17") and not COUNTRIES & set(store.objects_of([subject], 31)):
        return None  # "the country ..." only for a country
    found = [x for x in found if normalize(store.label(x) or "") != name]  # no "Panama in Panama"
    if prop == "P27":  # only people have a country of citizenship
        found = [x for x in found if HUMAN in store.objects_of([x], 31)]
    found.sort(key=lambda i: (-store.degree(i), i))
    return found[0] if found else None


def questions(updates: list[Update], controls: list[tuple[str, str, str]], store):
    """WWQ-format questions (one and two hops) and gold files for the new and old answers.
    controls are (subject, property, current value) triples that did not change."""
    items, new, old = [], {}, {}

    def add(kind, subject, prop, current, previous, start):
        noun, hops = PROPERTIES[prop]
        s = _number(subject)
        name = store.label(s)
        if name is None or store.label(_number(current)) is None:
            return
        values = store.objects_of([s], _number(prop))
        if _number(current) not in values:
            return  # the KG does not hold the current value
        if previous != current and _number(previous) in values:
            return  # the KG still holds the old value too: the new answer is ambiguous
        if kind == "control" and len(values) > 1:
            return  # two current values: no one answer
        asked = [("1", 1, f"Who is the {noun} of {name}?", subject)]
        for first, phrase, *backwards in hops:
            x = _first_hop(store, s, first, bool(backwards))
            if x is not None and store.label(x):
                text = phrase.format(x=store.label(x))
                key = "2" if len(asked) == 1 else f"2{chr(ord('a') + len(asked) - 1)}"
                asked.append((key, 2, f"Who is the {noun} of {text}?", f"Q{x}"))
        for key, hops_count, utterance, entity in asked:
            qid = f"{kind}-{prop}-{subject}-{key}"
            items.append(
                {
                    "id": qid,
                    "utterance": utterance,
                    "sparql": f"# question entities: wd:{entity}",
                    "complexity": f"{hops_count}-hop {kind}",
                    "start": start,
                }
            )
            new[qid] = {"entities": [current], "literals": []}
            old[qid] = {"entities": [previous], "literals": []}
        for after, template in AFTER:  # the update first, then a fact about its value
            now = sorted(store.objects_of([_number(current)], _number(after)))
            before = sorted(store.objects_of([_number(previous)], _number(after)))
            if not now or not before or (kind == "update" and set(now) & set(before)):
                continue
            if not all(store.label(e) for e in now):
                continue
            qid = f"{kind}-{prop}-{subject}-{after}"
            items.append(
                {
                    "id": qid,
                    "utterance": template.format(noun=noun, name=name),
                    "sparql": f"# question entities: wd:{subject}",
                    "complexity": f"2-hop {kind}-first",
                    "start": start,
                }
            )
            new[qid] = {"entities": [f"Q{e}" for e in now], "literals": []}
            old[qid] = {"entities": [f"Q{e}" for e in before], "literals": []}

    for u in updates:
        add("update", u.subject, u.prop, u.new, u.old, u.start)
    for subject, prop, current in controls:
        add("control", subject, prop, current, current, None)
    return items, {"answers": new, "failed": {}}, {"answers": old, "failed": {}}


def oracle_requests(questions: list[dict], store) -> list[dict]:
    """The update-in-prompt oracle: each question with the fact it turns on in the
    prompt, the subject's current value under the question's property (its id names
    both: update-P6-Q31-..., control-P6-Q31-...), unconstrained, as the MQuAKE oracle
    arm gives a case's edits, framed as updates (xgraph.rag.edit_messages). A question about
    a fact of the new value (update-first) gets the update, not that fact."""
    from xgraph.rag import edit_messages

    requests = []
    for question in questions:
        _, prop, subject = question["id"].split("-")[:3]
        item, number = _number(subject), _number(prop)
        values = store.objects_of([item], number)
        relation = store.relations_of([item]).get(number, prop)
        facts = [
            [store.label(item) or subject, relation, label] for _, label in sorted(values.items())
        ]
        messages = edit_messages(question["utterance"], [tuple(f) for f in facts])
        requests.append({"id": question["id"], "messages": messages, "facts": facts})
    return requests


def gold_labels(entity_labels: Path, *golds: dict) -> list[list[str]]:
    """The label rows (a KG's entity_labels.jsonl) of every gold entity, for
    xgraph.score.load_labels."""
    wanted = {e for gold in golds for a in gold["answers"].values() for e in a["entities"]}
    rows = []
    with Path(entity_labels).open(encoding="utf-8") as lines:
        for line in lines:
            row = json.loads(line)
            if row[0] in wanted:
                rows.append(row)
    return rows


def fetch(props: list[str], since: str, out: Path) -> None:
    """Query the Wikidata Query Service for each property's statements started since."""
    import urllib.parse
    import urllib.request

    out.mkdir(parents=True, exist_ok=True)
    for prop in props:
        query = QUERY.format(p=prop, since=since)
        url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query})
        request = urllib.request.Request(
            url, headers={"Accept": "application/sparql-results+json", "User-Agent": AGENT}
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            (out / f"{prop}.json").write_bytes(response.read())
        time.sleep(2)
    manifest = {"since": since, "properties": props, "fetched": time.strftime("%Y-%m-%d"),
                "query": QUERY, "endpoint": ENDPOINT}  # fmt: skip
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


def _controls(store, prop: str, updated: set[str], count: int, seed: int):
    """Subjects with a current value for prop that did not change: (subject, prop, value)."""
    pool = sorted(
        {(s, o) for s, o in store.pairs(_number(prop), 200_000) if f"Q{s}" not in updated}
    )
    picked = random.Random(seed).sample(pool, min(count, len(pool)))
    return [(f"Q{s}", prop, f"Q{o}") for s, o in sorted(picked)]


def sample(items: list[dict], cap: int, seed: int) -> list[dict]:
    """At most cap questions of each group (complexity) and property, seeded, in id order:
    an evaluation set no one property (head coaches) or control group dominates."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for item in items:
        groups.setdefault((item["complexity"], item["id"].split("-")[1]), []).append(item)
    rng = random.Random(seed)
    picked = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda i: i["id"])
        picked += rng.sample(members, min(cap, len(members)))
    return sorted(picked, key=lambda i: i["id"])


def main(argv: list[str] | None = None) -> None:
    from xgraph.provenance import provenance
    from xgraph.toolkg import KGStore

    parser = argparse.ArgumentParser(prog="python -m xgraph.updates", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("fetch", help="query the Wikidata Query Service")
    get.add_argument("--since", required=True, help="earliest start time (YYYY-MM-DD)")
    get.add_argument("--out", type=Path, required=True)
    make = commands.add_parser("build", help="questions about the updates the KG holds")
    make.add_argument("--raw", type=Path, required=True, help="fetched results")
    make.add_argument("--db", type=Path, required=True, help="KG store (xgraph.toolkg)")
    make.add_argument("--kg", type=Path, required=True, help="the KG directory it was built from")
    make.add_argument("--since", required=True)
    make.add_argument("--until", help="latest start time (YYYY-MM-DD), e.g. the dump's date")
    make.add_argument("--seed", type=int, default=0)
    make.add_argument("--out", type=Path, required=True)
    pick = commands.add_parser("sample", help="an evaluation set, capped per group and property")
    pick.add_argument("--questions", type=Path, required=True)
    pick.add_argument("--cap", type=int, required=True)
    pick.add_argument("--seed", type=int, default=0)
    pick.add_argument("--out", type=Path, required=True)
    tell = commands.add_parser("oracle", help="requests with the fact each question turns on")
    tell.add_argument("--questions", type=Path, required=True)
    tell.add_argument("--db", type=Path, required=True, help="KG store (xgraph.toolkg)")
    tell.add_argument("--out", type=Path, required=True, help="requests (JSONL)")
    args = parser.parse_args(argv)
    if args.command == "oracle":
        told = oracle_requests(json.loads(args.questions.read_text()), KGStore(args.db))
        with args.out.open("w", encoding="utf-8") as out:
            for request in told:
                out.write(json.dumps(request, ensure_ascii=False) + "\n")
        print(len(told), "requests")
        return
    if args.command == "fetch":
        fetch(sorted(PROPERTIES), args.since, args.out)
        return
    if args.command == "sample":
        picked = sample(json.loads(args.questions.read_text()), args.cap, args.seed)
        args.out.write_text(json.dumps(picked, indent=1, ensure_ascii=False) + "\n")
        print(len(picked), "questions")
        return
    store = KGStore(args.db)
    updates: list[Update] = []
    controls = []
    for prop in sorted(PROPERTIES):
        bindings = json.loads((args.raw / f"{prop}.json").read_text())["results"]["bindings"]
        found = updates_from(bindings, prop, args.since, args.until)
        updates += found
        updated = {u.subject for u in found} | {_item(b["s"]) for b in bindings}
        controls += _controls(store, prop, updated, len(found), args.seed)
    items, new, old = questions(updates, controls, store)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "questions.json").write_text(json.dumps(items, indent=1, ensure_ascii=False) + "\n")
    (args.out / "gold-new.json").write_text(json.dumps(new) + "\n")
    (args.out / "gold-old.json").write_text(json.dumps(old) + "\n")
    with (args.out / "labels.jsonl").open("w", encoding="utf-8") as out:
        for row in gold_labels(args.kg / "entity_labels.jsonl", new, old):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for item in items:
        counts[item["complexity"]] = counts.get(item["complexity"], 0) + 1
    manifest = {"artifact_kind": "xgraph.updates", "version": 2, "since": args.since,
                "until": args.until,
                "updates": len(updates), "questions": counts,
                "provenance": provenance([*(args.raw / f"{p}.json" for p in sorted(PROPERTIES)),
                                          args.db])}  # fmt: skip
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "provenance"}))


if __name__ == "__main__":
    main()
