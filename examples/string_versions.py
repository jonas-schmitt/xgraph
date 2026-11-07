"""CPU: persist application-ID edits and interleave pinned base/edited readers."""

import argparse
import json
from pathlib import Path

from toy_kg import graph, served_tokenizer, walk

from xgraph import Oracle, SpanSyntax, build_kg_index
from xgraph.kgversions import load_kg_versions, read_kg_versions, write_kg_edits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("target/examples/string-versions"))
    args = parser.parse_args()
    tokenizer, syntax = served_tokenizer(), SpanSyntax()
    index = build_kg_index(graph(), tokenizer, syntax)
    edits = {
        "updated": [("person:ada", "rel:born-in", "place:delta")],
        "deleted": [("person:ada", "rel:born-in", None)],
    }
    write_kg_edits(args.out, edits, index=index)
    versions = load_kg_versions(args.out, index, strict=True)
    oracle = Oracle(
        index,
        tokenizer,
        syntax,
        paths=True,
        committed_paths=True,
        versions=read_kg_versions(args.out),
    )
    pieces = (
        syntax.opener,
        " Atlas",
        syntax.separator,
        " creator",
        syntax.separator,
        " born in",
        syntax.closer,
    )
    base = oracle.initial_state()
    updated = oracle.initial_state("updated")
    base = walk(oracle, tokenizer, base, *pieces)
    updated = walk(oracle, tokenizer, updated, *pieces)
    base_again = walk(oracle, tokenizer, oracle.initial_state(), *pieces)
    deleted = versions.view(index, "deleted").object_rows("person:ada", "rel:born-in")
    print(
        json.dumps(
            {
                "base": base.last_span.entity_ids,
                "updated": updated.last_span.entity_ids,
                "base_again": base_again.last_span.entity_ids,
                "deleted": deleted,
                "loaded_edits": read_kg_versions(args.out),
            }
        )
    )


if __name__ == "__main__":
    main()
