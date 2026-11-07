"""Loader for the CoDEx knowledge graphs (github.com/tsafavi/codex).

A CoDEx directory, as `scripts/fetch-data.sh` lays it out, holds the English
entity and relation labels in `entities/en/entities.json` and
`relations/en/relations.json`, and each size's triples, one
tab-separated `subject relation object` per line, in
`triples/codex-<size>/{train,valid,test}.txt`. CoDEx has one label per entity and
no aliases.
"""

from __future__ import annotations

import json
from pathlib import Path

from xgraph.kg import Entity, KnowledgeGraph, Relation

SIZES = ("s", "m", "l")


class CodexFormatError(ValueError):
    """A CoDEx file is missing or inconsistent."""


def _labels(path: Path, ids: set[str], kind: str, kinds: str, size: str) -> dict[str, str]:
    """The label of each id from a CoDEx label file, validated."""
    if not path.is_file():
        raise CodexFormatError(f"missing CoDEx {kind} file {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, dict):
        raise CodexFormatError(f"{path} must hold a JSON object keyed by {kind} id")
    missing = sorted(ids - records.keys())
    if missing:
        raise CodexFormatError(
            f"{path} has no label for {len(missing)} {kinds} in codex-{size}, "
            f"such as {', '.join(missing[:5])}"
        )
    labels = {}
    for item in ids:
        record = records[item]
        label = record.get("label") if isinstance(record, dict) else None
        if not isinstance(label, str):
            raise CodexFormatError(f"{path}: {kind} {item} has no string label")
        labels[item] = label
    return labels


def load_codex(root: Path, size: str = "l") -> KnowledgeGraph:
    """The entities, relations, and triples of one CoDEx size, sorted, with their labels."""
    if size not in SIZES:
        raise ValueError(f"CoDEx size must be one of {SIZES}, got {size!r}")
    root = Path(root)
    triples: set[tuple[str, str, str]] = set()
    for split in ("train", "valid", "test"):
        path = root / "triples" / f"codex-{size}" / f"{split}.txt"
        if not path.is_file():
            raise CodexFormatError(f"missing CoDEx triples file {path}")
        with path.open(encoding="utf-8") as lines:
            for number, line in enumerate(lines, start=1):
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 3:
                    raise CodexFormatError(f"{path}:{number}: expected 3 tab-separated fields")
                triples.add((fields[0], fields[1], fields[2]))

    entity_ids = {s for s, _, _ in triples} | {o for _, _, o in triples}
    relation_ids = {r for _, r, _ in triples}
    entity_labels = _labels(
        root / "entities" / "en" / "entities.json", entity_ids, "entity", "entities", size
    )
    relation_labels = _labels(
        root / "relations" / "en" / "relations.json", relation_ids, "relation", "relations", size
    )
    return KnowledgeGraph(
        entities=tuple(Entity(i, (entity_labels[i],)) for i in sorted(entity_ids)),
        relations=tuple(Relation(i, (relation_labels[i],)) for i in sorted(relation_ids)),
        triples=tuple(sorted(triples)),
    )
