"""CPU: import RDF, pin string-ID replacements/deletions, and export resolved facts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdflib import Graph, Namespace
from toy_kg import served_tokenizer, walk

from xgraph.index import load_kg_index
from xgraph.kgversions import load_kg_versions, read_kg_versions, write_kg_edits
from xgraph.oracle import Oracle
from xgraph.rdf import rdflib_edits, to_rdflib, write_rdflib
from xgraph.syntax import SpanSyntax
from xgraph.wikidata import build_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("target/examples/rdf"))
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    graph = Graph().parse(
        data="""
        @prefix ex: <https://example.org/> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        ex:atlas rdfs:label "Atlas" ; ex:creator ex:ada .
        ex:ada rdfs:label "Ada" ; ex:bornIn ex:cedar .
        ex:cedar rdfs:label "Cedar" . ex:delta rdfs:label "Delta" .
        ex:creator rdfs:label "creator" . ex:bornIn rdfs:label "born in" .
    """,
        format="turtle",
    )
    ex = Namespace("https://example.org/")
    tokenizer, syntax = served_tokenizer(args.tokenizer), SpanSyntax()
    write_rdflib(graph, args.out / "kg")
    build_index(args.out / "kg", tokenizer, syntax, args.out / "index")
    index = load_kg_index(args.out / "index")
    edits = {
        "updated": rdflib_edits(
            graph, removed=[(ex.ada, ex.bornIn, ex.cedar)], added=[(ex.ada, ex.bornIn, ex.delta)]
        ),
        "deleted": rdflib_edits(graph, removed=[(ex.ada, ex.bornIn, ex.cedar)]),
    }
    write_kg_edits(args.out / "versions", edits, index=index)
    versions = load_kg_versions(args.out / "versions", index)
    oracle = Oracle(index, tokenizer, syntax, versions=read_kg_versions(args.out / "versions"))
    fact = walk(
        oracle,
        tokenizer,
        oracle.initial_state("updated"),
        syntax.opener,
        " Ada",
        syntax.separator,
        " born in",
        syntax.separator,
        " Delta",
        syntax.closer,
    ).last_span
    exported = sorted(tuple(map(str, row)) for row in to_rdflib(spans=[fact]))
    print(
        json.dumps(
            {
                "base": index.object_rows(str(ex.ada), str(ex.bornIn)),
                "updated": list(fact.triples),
                "exported": exported,
                "deleted_rows": versions.view(index, "deleted").object_rows(
                    str(ex.ada), str(ex.bornIn)
                ),
                "loaded_edits": read_kg_versions(args.out / "versions"),
            }
        )
    )


if __name__ == "__main__":
    main()
