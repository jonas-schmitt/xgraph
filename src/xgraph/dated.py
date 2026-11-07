"""The KG's hops on MQuAKE's chains as Wikidata held them at a date.

MQuAKE-Remastered's gold follows each chain as Wikidata stood when the dataset was built;
some unedited hops have changed since (a head of state replaced), so a KG of today
answers them differently. This module asks the Wikidata Query Service for every
statement, with its rank and start and end time qualifiers (P580, P582), on the
(subject, property) pairs of every case's original and edited chain, and keeps for each
pair the values valid at a date: not deprecated, started on or before it (or without a
start), and not ended by it. A pair with no dated statement, or none valid at the date,
is left as the KG has it. The result is a list of triples that replace the KG's objects
of those pairs (xgraph.mquake --dated applies it before the cases' edits, as an edit
replaces every object of its subject and property).

    python -m xgraph.dated fetch --data data/mquake/MQuAKE-Remastered-CF-3k.json \\
        --out data/wikidata/mqr-dated
    python -m xgraph.dated triples --fetched data/wikidata/mqr-dated --date 2023-04-01 \\
        --out data/wikidata/mqr-dated/2023-04-01.npy
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

QUERY = (
    "SELECT ?s ?o ?rank ?start ?end WHERE {{ VALUES ?s {{ {subjects} }} ?s p:{p} ?st . "
    "?st ps:{p} ?o ; wikibase:rank ?rank . OPTIONAL {{ ?st pq:P580 ?start }} "
    "OPTIONAL {{ ?st pq:P582 ?end }} }}"
)
BATCH = 200
"""Subjects per query."""


def _item(cell: dict) -> str:
    return cell["value"].rsplit("/", 1)[-1]


def chain_pairs(cases: list[dict]) -> dict[str, set[str]]:
    """Per property, the subjects of every case's original and edited chain hops."""
    pairs: dict[str, set[str]] = {}
    for case in cases:
        for s, p, _ in case["orig"]["triples"] + case["orig"]["new_triples"]:
            pairs.setdefault(p, set()).add(s)
    return pairs


def as_of(rows: list[dict], date: str) -> dict[str, set[str]]:
    """Per subject with a dated statement (a start or end time), its values valid at the
    date (YYYY-MM-DD): not deprecated, started on or before it, not ended by it."""
    dated: set[str] = set()
    valid: dict[str, set[str]] = {}
    for r in rows:
        subject, value = _item(r["s"]), _item(r["o"])
        start = r["start"]["value"][:10] if "start" in r else None
        end = r["end"]["value"][:10] if "end" in r else None
        if start or end:
            dated.add(subject)
        valid.setdefault(subject, set())
        if r["rank"]["value"].endswith("DeprecatedRank"):
            continue
        current = (start is None or start <= date) and (end is None or end > date)
        if current and value[:1] == "Q" and value[1:].isdigit():
            valid[subject].add(value)
    return {s: valid[s] for s in valid if s in dated}


def dated_triples(fetched: dict[str, dict[str, set[str]]]) -> np.ndarray:
    """(subject, property, object) rows replacing each dated pair's objects; a pair with
    no value valid at the date has no row, so the KG keeps its objects."""
    rows = [
        [int(s[1:]), int(p[1:]), int(o[1:])]
        for p in sorted(fetched, key=lambda p: int(p[1:]))
        for s, objects in fetched[p].items()
        for o in sorted(objects, key=lambda o: int(o[1:]))
    ]
    return np.array(rows, dtype=np.int32).reshape(-1, 3)


def fetch(pairs: dict[str, set[str]], out: Path) -> None:
    """Every statement of the pairs, a query per property and batch of subjects."""
    import urllib.parse
    import urllib.request

    from xgraph.updates import AGENT, ENDPOINT

    out.mkdir(parents=True, exist_ok=True)
    for prop in sorted(pairs, key=lambda p: int(p[1:])):
        subjects = sorted(pairs[prop], key=lambda s: int(s[1:]))
        bindings = []
        for at in range(0, len(subjects), BATCH):
            values = " ".join(f"wd:{s}" for s in subjects[at : at + BATCH])
            query = QUERY.format(p=prop, subjects=values)
            request = urllib.request.Request(
                ENDPOINT,
                data=urllib.parse.urlencode({"query": query}).encode(),
                headers={"Accept": "application/sparql-results+json", "User-Agent": AGENT},
            )
            with urllib.request.urlopen(request, timeout=300) as response:
                bindings += json.loads(response.read())["results"]["bindings"]
            time.sleep(1)
        (out / f"{prop}.json").write_text(json.dumps(bindings) + "\n")
    manifest = {
        "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoint": ENDPOINT,
        "query": QUERY,
        "properties": {p: len(s) for p, s in sorted(pairs.items())},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m xgraph.dated", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetching = commands.add_parser("fetch", help="the chains' statements from Wikidata")
    fetching.add_argument("--data", type=Path, required=True, help="MQuAKE-Remastered JSON")
    fetching.add_argument("--out", type=Path, required=True, help="a directory")
    dating = commands.add_parser("triples", help="the values valid at a date, as triples")
    dating.add_argument("--fetched", type=Path, required=True, help="fetch's directory")
    dating.add_argument("--date", required=True, help="YYYY-MM-DD")
    dating.add_argument("--out", type=Path, required=True, help="the triples (.npy)")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        fetch(chain_pairs(json.loads(args.data.read_text(encoding="utf-8"))), args.out)
        return
    fetched = {}
    for path in sorted(args.fetched.glob("P*.json")):
        fetched[path.stem] = as_of(json.loads(path.read_text(encoding="utf-8")), args.date)
    triples = dated_triples(fetched)
    np.save(args.out, triples)
    summary = {
        "date": args.date,
        "dated_pairs": sum(len(v) for v in fetched.values()),
        "pairs_with_values": sum(bool(o) for v in fetched.values() for o in v.values()),
        "triples": len(triples),
    }
    Path(f"{args.out}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
