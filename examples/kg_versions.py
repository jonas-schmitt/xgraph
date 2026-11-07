"""CPU: pin base and edited readers over one index, and save/load an edit set.

Uses application-assigned Q/P numbers because the existing version file format
stores numeric triples. The in-memory version API also accepts arbitrary string IDs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from toy_kg import graph, rejected, served_tokenizer, walk

from xgraph.index import build_kg_index
from xgraph.kgversions import (
    build_kg_versions,
    load_kg_versions,
    read_kg_versions,
    write_kg_versions,
)
from xgraph.oracle import Oracle
from xgraph.syntax import SpanSyntax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--out", type=Path, default=Path("target/examples/versions"))
    args = parser.parse_args()
    tokenizer, syntax = served_tokenizer(args.tokenizer), SpanSyntax()
    index = build_kg_index(graph(numeric=True), tokenizer, syntax)
    # This replaces ALL objects of (Q2, P2), keeping every other pair unchanged.
    edits = {"updated": [("Q2", "P2", "Q4")]}
    versions = build_kg_versions(index, edits)
    assert versions.stats[0].unwritable_edits == 0
    write_kg_versions(args.out, {"updated": np.array([[2, 2, 4]], dtype=np.int32)}, index=index)
    loaded = load_kg_versions(args.out, index, strict=True)
    assert loaded.names == versions.names

    oracle = Oracle(
        index,
        tokenizer,
        syntax,
        paths=True,
        committed_paths=True,
        versions=read_kg_versions(args.out),
    )
    # Each immutable state pins its version. Interleave both readers safely.
    base = oracle.initial_state()
    updated = oracle.initial_state(version="updated")
    pieces = (
        syntax.opener,
        " Atlas",
        syntax.separator,
        " creator",
        syntax.separator,
        " born in",
        syntax.closer,
    )
    base = walk(oracle, tokenizer, base, *pieces)
    updated = walk(oracle, tokenizer, updated, *pieces)
    base_again = walk(oracle, tokenizer, oracle.initial_state(), *pieces)
    print(
        json.dumps(
            {
                "base": base.last_span.entity_ids,
                "edited": updated.last_span.entity_ids,
                "base_after_edit": base_again.last_span.entity_ids,
                "rejected_old_answer": rejected(
                    oracle, tokenizer, updated, syntax.opener, " Cedar", syntax.closer
                ),
                "loaded_edits": read_kg_versions(args.out),
                "unwritable_edits": versions.stats[0].unwritable_edits,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
