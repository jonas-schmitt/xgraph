"""Turtle output (`docs/turtle-output-adr.md`): the triples of closed fact spans as RDF.

The model writes spans over labels; a closed fact span resolves to the KG's triples
whose subject, relation and object carry the labels written, and xgraph emits those
triples as Turtle next to the text, without changing what the model writes or the
mask. Entity spans name entities without stating a fact, so they emit nothing.

    python -m xgraph.turtle --index data/wikidata/gate-index \\
        --tokenizer data/tokenizers/qwen3.8/tokenizer.json \\
        --outputs target/gate/wwq-constrained.jsonl --out target/gate/wwq-constrained.ttl
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

PREFIXES = {
    "wd": "http://www.wikidata.org/entity/",
    "wdt": "http://www.wikidata.org/prop/direct/",
}


def span_triples(spans) -> Iterable[tuple[str, str, str]]:
    """Resolved facts only; entity results and path endpoints assert no RDF edge."""
    for span in spans:
        if span.resolved:
            yield from span.triples


def turtle(triples: Iterable[tuple[str, str, str]]) -> str:
    """Prefixed Turtle statements, each triple once, in first-seen order."""
    head = "".join(f"@prefix {name}: <{iri}> .\n" for name, iri in PREFIXES.items())
    seen = dict.fromkeys(triples)
    body = "".join(f"wd:{s} wdt:{p} wd:{o} .\n" for s, p, o in seen)
    return head + ("\n" + body if body else "")


def closed_triples(engine, index, tokens: list[int]) -> list[tuple[str, str, str]]:
    """The triples of every fact span the token stream closes, replayed on the engine."""
    from xgraph.engine import PHASES, closed_span

    free = PHASES.index("free text")
    engine.reset()
    found: list[tuple[str, str, str]] = []
    phase = free
    for token in tokens:
        engine.advance(np.array([token], dtype=np.int32))
        states = engine.states_host(1)
        now = int(states["phase"][0])
        if phase != free and now == free:
            found.extend(span_triples([closed_span(index, states, 0)]))
        phase = now
    return found


def main(argv: list[str] | None = None) -> None:
    from xgraph.engine import Engine
    from xgraph.index import load_kg_index
    from xgraph.syntax import SpanSyntax
    from xgraph.tokenizer import ServedTokenizer

    parser = argparse.ArgumentParser(prog="python -m xgraph.turtle", description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="KG index directory")
    parser.add_argument("--tokenizer", type=Path, required=True, help="served tokenizer.json")
    parser.add_argument("--outputs", type=Path, required=True, help="xgraph.generate output")
    parser.add_argument("--out", type=Path, required=True, help="Turtle file")
    args = parser.parse_args(argv)
    index = load_kg_index(args.index)
    engine = Engine(index, ServedTokenizer.from_file(args.tokenizer), SpanSyntax(), 1)
    triples: list[tuple[str, str, str]] = []
    for line in args.outputs.read_text().splitlines():
        if line:
            triples.extend(closed_triples(engine, index, json.loads(line)["token_ids"]))
    args.out.write_text(turtle(triples), encoding="utf-8")
    print(f"{len(dict.fromkeys(triples))} triples")


if __name__ == "__main__":
    main()
