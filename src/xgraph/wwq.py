"""WikiWebQuestions (WWQ) and its gold answers.

The quality gate (`docs/quality-gate.md`) takes WWQ's gold answers from one run of
its gold SPARQL queries against the live Wikidata Query Service, stored as a file
with the date and provenance. A query is sent at a polite rate, identified by the
project, and retried when the service asks for a pause (HTTP 429 with Retry-After)
or fails transiently; a query that still fails is recorded, not fatal.

    python -m xgraph.wwq gold --test data/wwq/test.json \\
        --out eval/wwq-gold-wdqs-2026-10-12.json

`coverage` then measures, before any scored run, how many questions the gate KG
can answer at all (`python -m xgraph.wwq coverage`).
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "xgraph/0.0 (research; https://github.com/jonas-schmitt/xgraph)"
ENTITY = "http://www.wikidata.org/entity/"
RETRY_STATUS = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class Question:
    id: str
    utterance: str
    sparql: str


def load_wwq(path: str | Path) -> list[Question]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Question(r["id"], r["utterance"], r["sparql"]) for r in records]


def answers(result: dict) -> tuple[list[str], list[str]]:
    """The entity QIDs and the other values of a SPARQL JSON result, each once, in order."""
    entities: dict[str, None] = {}
    literals: dict[str, None] = {}
    for row in result["results"]["bindings"]:
        for value in row.values():
            text = value["value"]
            local = text[len(ENTITY) :]
            if value["type"] == "uri" and text.startswith(ENTITY) and local[:1] == "Q":
                entities[local] = None
            else:
                literals[text] = None
    return list(entities), list(literals)


def query_wdqs(
    sparql: str,
    *,
    endpoint: str = ENDPOINT,
    opener: Callable = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
    retries: int = 5,
    timeout: float = 120.0,
) -> dict:
    """Run one SPARQL query, pausing and retrying when the service asks or fails briefly."""
    url = endpoint + "?" + urllib.parse.urlencode({"query": sparql, "format": "json"})
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"}
    )
    for attempt in range(retries + 1):
        try:
            with opener(request, timeout=timeout) as response:
                # The service can return control characters inside string values.
                return json.loads(response.read().decode("utf-8"), strict=False)
        except urllib.error.HTTPError as error:
            if error.code not in RETRY_STATUS or attempt == retries:
                raise
            pause = error.headers.get("Retry-After") if error.headers else None
            sleep(float(pause) if pause else 2.0 ** (attempt + 1))
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries:
                raise
            sleep(2.0 ** (attempt + 1))
    raise AssertionError("unreachable")


def collect_gold(
    questions: Iterable[Question],
    *,
    query: Callable[[str], dict] = query_wdqs,
    delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    previous: dict | None = None,
) -> dict:
    """Gold answers for every question: entities and other values, plus the failures.

    With a previous result, its answers are kept and only the other questions are
    queried.
    """
    kept = previous["answers"] if previous else {}
    gold: dict[str, dict[str, list[str]]] = {}
    failed: dict[str, str] = {}
    asked = 0
    for question in questions:
        if question.id in kept:
            gold[question.id] = kept[question.id]
            continue
        if asked:
            sleep(delay)
        asked += 1
        try:
            entities, literals = answers(query(question.sparql))
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as error:
            failed[question.id] = f"{type(error).__name__}: {error}"
            continue
        gold[question.id] = {"entities": entities, "literals": literals}
    return {"answers": gold, "failed": failed}


NAMED = re.compile(r"\bwd:Q(\d+)\b")
CHUNK = 1 << 22
"""Triples read per step by coverage."""


def coverage(questions: Iterable[Question], gold: dict, kg: str | Path) -> dict:
    """How many questions the gate KG can answer, from its streamed files.

    A question is scored when its gold answers hold an entity. Its answer is in the KG
    when a gold entity has an English label there, and one hop away when a KG triple
    links a gold entity, in either direction, to an entity its gold query names.
    """
    kg = Path(kg)
    answered, failed = gold["answers"], gold["failed"]
    scored: dict[str, tuple[set[int], set[int]]] = {}
    counts = {"questions": 0, "failed": 0, "no_entity": 0}
    for question in questions:
        counts["questions"] += 1
        if question.id in failed or question.id not in answered:
            counts["failed"] += 1
        elif not answered[question.id]["entities"]:
            counts["no_entity"] += 1
        else:
            entities = {int(q[1:]) for q in answered[question.id]["entities"]}
            named = {int(n) for n in NAMED.findall(question.sparql)}
            scored[question.id] = (entities, named)

    wanted = set().union(*(entities for entities, _ in scored.values()))
    labelled: set[int] = set()
    with (kg / "entity_labels.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            item = int(json.loads(line)[0][1:])
            if item in wanted:
                labelled.add(item)

    pairs = {(n << 32) | e for entities, named in scored.values() for e in entities for n in named}
    pair_codes = np.array(sorted(pairs), dtype=np.int64)
    linked: set[int] = set()
    triples = np.load(kg / "triples.npy", mmap_mode="r")
    for at in range(0, len(triples), CHUNK):
        chunk = np.asarray(triples[at : at + CHUNK], dtype=np.int64)
        for named, entity in ((chunk[:, 0], chunk[:, 2]), (chunk[:, 2], chunk[:, 0])):
            codes = (named << 32) | entity
            linked.update(codes[np.isin(codes, pair_codes)].tolist())

    def one_hop(entities: set[int], named: set[int]) -> bool:
        return any((n << 32) | e in linked for e in entities for n in named)

    not_in_kg = [q for q, (entities, _) in scored.items() if not entities & labelled]
    not_one_hop = [q for q, (entities, named) in scored.items() if not one_hop(entities, named)]
    counts["scored"] = len(scored)
    counts["answer_in_kg"] = len(scored) - len(not_in_kg)
    counts["answer_one_hop"] = len(scored) - len(not_one_hop)
    return {"counts": counts, "answer_not_in_kg": not_in_kg, "answer_not_one_hop": not_one_hop}


def main(argv: list[str] | None = None) -> None:
    from xgraph.provenance import provenance

    parser = argparse.ArgumentParser(prog="python -m xgraph.wwq", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gold = commands.add_parser("gold", help="collect gold answers from the query service")
    gold.add_argument("--test", type=Path, required=True, help="WWQ test.json")
    gold.add_argument("--out", type=Path, required=True, help="gold answer file (JSON)")
    gold.add_argument("--delay", type=float, default=1.0, help="seconds between queries")
    gold.add_argument(
        "--resume", type=Path, help="earlier gold file: keep its answers, query the rest"
    )
    cover = commands.add_parser("coverage", help="measure the gate KG's coverage of the gold")
    cover.add_argument("--test", type=Path, required=True, help="WWQ test.json")
    cover.add_argument("--gold", type=Path, required=True, help="gold answer file")
    cover.add_argument("--kg", type=Path, required=True, help="gate KG directory")
    cover.add_argument("--out", type=Path, required=True, help="report file (JSON)")
    args = parser.parse_args(argv)
    if args.command == "coverage":
        gold_file = json.loads(args.gold.read_text(encoding="utf-8"))
        report = {
            "artifact_kind": "xgraph.wwq_coverage",
            "version": 1,
            **coverage(load_wwq(args.test), gold_file, args.kg),
            "provenance": provenance(
                [args.test, args.gold, *sorted(p for p in args.kg.iterdir() if p.is_file())]
            ),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(report["counts"], indent=1))
        return
    # Provenance first, so that it names the code that runs the queries.
    record = provenance([args.test] + ([args.resume] if args.resume else []))
    previous = json.loads(args.resume.read_text(encoding="utf-8")) if args.resume else None
    started = (
        previous["queried_from"] if previous else datetime.now(UTC).isoformat(timespec="seconds")
    )
    result = collect_gold(load_wwq(args.test), delay=args.delay, previous=previous)
    report = {
        "artifact_kind": "xgraph.wwq_gold",
        "version": 1,
        "endpoint": ENDPOINT,
        "queried_from": started,
        "queried_to": datetime.now(UTC).isoformat(timespec="seconds"),
        "resumed_from": str(args.resume) if args.resume else None,
        **result,
        "provenance": record,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(result['answers'])} answered, {len(result['failed'])} failed")


if __name__ == "__main__":
    main()
