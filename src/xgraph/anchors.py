"""CPU pre-validation of every anchor in a generation request set.

Unknown IDs in mixed anchors are errors here, although the runtime ignores them.
This stricter input check leaves engine and oracle masking semantics unchanged.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from xgraph.index import KGIndex, kg_index_fingerprint, load_kg_index


def validate_anchors(
    index: KGIndex, requests: Iterable[dict], *, require_anchors: bool = False
) -> dict:
    """List all empty, malformed, unknown and unwritable request anchors on CPU."""
    # Retain only IDs and anchors; no copied KG arrays or model/prompt tensors.
    rows = [
        {key: row[key] for key in ("id", "anchor", "_error") if key in row}
        if isinstance(row, dict)
        else {"_error": "request is not an object"}
        for row in requests
    ]
    wanted = {
        entity
        for row in rows
        if isinstance(row.get("anchor"), list)
        for entity in row["anchor"]
        if isinstance(entity, str)
    }
    positions = {entity: n for n, entity in enumerate(index.labels.owner_ids) if entity in wanted}
    writable = np.zeros(len(index.labels.owner_ids), dtype=bool)
    writable[index.labels.owners] = True
    issues = []
    anchored = 0
    for line, row in enumerate(rows, 1):

        def issue(reason, ids, *, line=line, row=row):
            issues.append({"line": line, "id": row.get("id"), "reason": reason, "ids": ids})

        if "_error" in row:
            issue(row["_error"], [])
            continue
        anchor = row.get("anchor")
        if anchor is None:
            if require_anchors:
                issue("missing anchor", [])
            continue
        anchored += 1
        if not isinstance(anchor, list) or any(not isinstance(e, str) or not e for e in anchor):
            issue("anchor must be a list of nonempty entity ID strings", anchor)
            continue
        if not anchor:
            issue("empty anchor", [])
            continue
        unique = list(dict.fromkeys(anchor))
        unknown = [e for e in unique if e not in positions]
        unwritable = [e for e in unique if e in positions and not writable[positions[e]]]
        if unknown:
            issue("unknown anchor IDs", unknown)
        if unwritable:
            issue("unwritable anchor IDs", unwritable)
    return {
        "valid": not issues,
        "requests": len(rows),
        "anchored_requests": anchored,
        "require_anchors": require_anchors,
        "issues": issues,
    }


def _requests(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            # Keep physical line numbers, including blank/malformed requests.
            if not line.strip():
                yield {"_error": "empty request line"}
                continue
            try:
                yield json.loads(line)
            except ValueError as error:
                yield {"_error": f"invalid JSON: {error}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kg-index", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--require-anchors", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    from xgraph.provenance import provenance

    report = validate_anchors(
        load_kg_index(args.kg_index), _requests(args.requests), require_anchors=args.require_anchors
    )
    report["provenance"] = provenance([args.requests, args.kg_index / "kg-index.json"])
    report["kg_index_fingerprint"] = kg_index_fingerprint(args.kg_index)
    report["seed"] = None  # Deterministic validation, without sampling.
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
