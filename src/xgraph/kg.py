"""Knowledge graph data model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Entity:
    """A KG node with an application-owned ID and writable labels.

    Args:
        id: Stable nonempty string, such as ``item:atlas`` or an RDF IRI.
        labels: Canonical label first, followed by aliases. Shared labels do
            not merge different IDs. Empty labels remain valid source data
            but may leave this entity unwritable in a KG index.
    """

    id: str
    labels: tuple[str, ...]


class KGFormatError(ValueError):
    """A knowledge graph is inconsistent, such as a triple naming an unknown entity."""


@dataclass(frozen=True)
class Relation:
    """A directed edge type with a stable string ID and labels, canonical first.

    Relation labels are what a model writes; IDs identify edges and version
    edits. Several relations can share a label, in which case execution follows
    their union. No inverse relation or inference rule is added automatically.
    """

    id: str
    labels: tuple[str, ...]


@dataclass(frozen=True)
class KnowledgeGraph:
    """An in-memory source KG, independent of a tokenizer or model.

    Args:
        entities: Entity records with unique IDs, including isolated nodes.
        relations: Relation records with unique IDs.
        triples: ``(subject_id, relation_id, object_id)`` tuples. Each ID must
            be declared in the appropriate table. Literals need entity IDs and
            rendered labels too; use the optional RDF adapter to project RDF.

    Construction stores these records. :func:`validate` checks their structure;
    :func:`xgraph.index.build_kg_index` calls it before compiling. Duplicate
    triples are accepted and index edges are deduplicated. Use a KG directory
    and the directory builder when this in-memory representation is too large.
    """

    entities: tuple[Entity, ...]
    relations: tuple[Relation, ...] = ()
    triples: tuple[tuple[str, str, str], ...] = ()


def validate(
    graph: KnowledgeGraph | str | Path | np.ndarray,
    *,
    entity_ids=None,
    relation_ids=None,
) -> None:
    """Validate a source KG, KG directory, or int32 (n, 3) triples array.

    Arrays require declared numeric entity_ids and relation_ids. Errors identify
    the source file/table, one-based row and offending ID where available.
    Directories are streamed and arrays memory-mapped with pickle disabled.
    Empty/excluded labels and repeated label surfaces or triples remain valid.
    A directory alias must have a canonical label row declaring its owner.
    """
    if isinstance(graph, (str, Path)):
        _validate_directory(Path(graph))
        return
    if isinstance(graph, np.ndarray):
        known = []
        for kind, ids in (("entities", entity_ids), ("relations", relation_ids)):
            if ids is None:
                raise KGFormatError(f"triples: array validation requires declared {kind} IDs")
            try:
                declared_ids = iter(ids)
            except TypeError as error:
                raise KGFormatError(f"{kind}: expected a sequence of integer IDs") from error
            seen = set()
            for row, number in enumerate(declared_ids, 1):
                if isinstance(number, (bool, np.bool_)) or not isinstance(
                    number, (int, np.integer)
                ):
                    raise KGFormatError(f"{kind} row {row}, ID {number!r}: expected an integer")
                if not 0 <= number <= np.iinfo(np.int32).max:
                    raise KGFormatError(f"{kind} row {row}, ID {number}: outside int32 ID range")
                if number in seen:
                    raise KGFormatError(f"{kind} row {row}, ID {number}: duplicate ID")
                seen.add(int(number))
            known.append(np.array(sorted(seen), dtype=np.int64))
        _validate_triples(graph, "triples", *known)
        return
    if not isinstance(graph, KnowledgeGraph):
        raise KGFormatError("expected a KnowledgeGraph, KG directory, or triples array")
    for kind in ("entities", "relations", "triples"):
        if not isinstance(getattr(graph, kind), (tuple, list)):
            raise KGFormatError(f"{kind}: expected a sequence of source records")
    declared = {}
    for kind in ("entities", "relations"):
        seen = {}
        for row, item in enumerate(getattr(graph, kind), 1):
            if not isinstance(item, Entity if kind == "entities" else Relation):
                raise KGFormatError(
                    f"{kind} row {row}: expected an Entity or Relation, got {item!r}"
                )
            context = f"{kind} row {row}, ID {item.id!r}"
            if not isinstance(item.id, str) or not item.id:
                raise KGFormatError(f"{context}: ID must be a nonempty string")
            if item.id in seen:
                raise KGFormatError(f"{context}: duplicate ID (first row {seen[item.id]})")
            seen[item.id] = row
            if not isinstance(item.labels, (tuple, list)):
                raise KGFormatError(f"{context}: labels must be a sequence of strings")
            for at, label in enumerate(item.labels, 1):
                if not isinstance(label, str):
                    raise KGFormatError(f"{context}, label {at}: expected a string, got {label!r}")
        declared[kind] = seen
    for row, triple in enumerate(graph.triples, 1):
        if not isinstance(triple, (tuple, list)) or len(triple) != 3:
            raise KGFormatError(f"triples row {row}: expected three IDs, got {triple!r}")
        for name, kind in zip(triple, ("entities", "relations", "entities"), strict=True):
            context = f"triples row {row}, ID {name!r}"
            if not isinstance(name, str):
                raise KGFormatError(f"{context}: ID must be a string")
            if name not in declared[kind]:
                raise KGFormatError(f"{context}: triple names unknown {name} in {kind}")


def _numeric_id(value, prefix: str, context: str) -> int:
    if (
        not isinstance(value, str)
        or not value.startswith(prefix)
        or not value[1:].isascii()
        or not value[1:].isdigit()
        or len(value[1:]) > 10
        or str(int(value[1:])) != value[1:]
        or int(value[1:]) > np.iinfo(np.int32).max
    ):
        raise KGFormatError(f"{context}, ID {value!r}: expected {prefix}<int32 number> ID")
    return int(value[1:])


def _directory_ids(path: Path, prefix: str) -> np.ndarray:
    canonical, pending = {}, {}
    try:
        with path.open(encoding="utf-8") as lines:
            for row, line in enumerate(lines, 1):
                context = f"{path} row {row}"
                try:
                    entry = json.loads(line)
                except ValueError as error:
                    raise KGFormatError(f"{context}: malformed JSON: {error}") from error
                if not isinstance(entry, list) or len(entry) != 3:
                    raise KGFormatError(f"{context}: expected three fields [ID, label, kind]")
                owner, label, kind = entry
                number = _numeric_id(owner, prefix, context)
                context += f", ID {owner!r}"
                if not isinstance(label, str):
                    raise KGFormatError(f"{context}: label must be a string")
                if kind not in ("label", "alias"):
                    raise KGFormatError(f"{context}: kind must be 'label' or 'alias'")
                if kind == "label":
                    if number in canonical:
                        raise KGFormatError(
                            f"{context}: duplicate ID (first canonical row {canonical[number]})"
                        )
                    canonical[number] = row
                    pending.pop(number, None)
                elif number not in canonical:
                    pending.setdefault(number, row)
    except (OSError, UnicodeError) as error:
        raise KGFormatError(f"{path}: cannot read label file: {error}") from error
    if pending:
        number, row = next(iter(pending.items()))
        raise KGFormatError(
            f"{path} row {row}, ID {prefix}{number}: dangling alias has no canonical label row"
        )
    return np.array(sorted(canonical), dtype=np.int64)


def _source_array(path: Path, columns: int, dtype) -> np.ndarray:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, EOFError) as error:
        raise KGFormatError(f"{path}: cannot read numeric array: {error}") from error
    _array_layout(array, str(path), columns, dtype)
    return array


def _array_layout(array, source, columns, dtype):
    if not isinstance(array, np.ndarray) or array.ndim != 2 or array.shape[1] != columns:
        raise KGFormatError(
            f"{source}: expected shape (n, {columns}), got {getattr(array, 'shape', None)}"
        )
    if array.dtype != np.dtype(dtype):
        raise KGFormatError(f"{source}: dtype must be {np.dtype(dtype)}, got {array.dtype}")


def _unknown(numbers, known):
    at = np.searchsorted(known, numbers)
    if not len(known):
        return np.ones(len(numbers), dtype=bool)
    return (at == len(known)) | (known[np.minimum(at, len(known) - 1)] != numbers)


def _validate_triples(array, source, entities, relations):
    _array_layout(array, source, 3, np.int32)
    for at in range(0, len(array), 65536):
        chunk = array[at : at + 65536]
        for col, known, prefix in ((0, entities, "Q"), (1, relations, "P"), (2, entities, "Q")):
            bad = np.flatnonzero(_unknown(chunk[:, col], known))
            if len(bad):
                row = int(bad[0])
                raise KGFormatError(
                    f"{source} row {at + row + 1}, ID {prefix}{chunk[row, col]}: unknown ID"
                )


def _validate_directory(directory: Path):
    entities = _directory_ids(directory / "entity_labels.jsonl", "Q")
    relations = _directory_ids(directory / "relation_labels.jsonl", "P")
    path = directory / "triples.npy"
    _validate_triples(_source_array(path, 3, np.int32), str(path), entities, relations)
    for name, columns, dtype in (
        ("sitelink_counts.npy", 2, np.int32),
        ("date_values.npy", 4, np.int64),
    ):
        path = directory / name
        if not path.exists():
            continue
        array = _source_array(path, columns, dtype)
        previous = -1
        for at in range(0, len(array), 65536):
            chunk = array[at : at + 65536]
            numbers = chunk[:, 0]
            bad = _unknown(numbers, entities)
            bad |= numbers <= np.concatenate(([previous], numbers[:-1]))
            if name == "sitelink_counts.npy":
                bad |= chunk[:, 1] < 0
            else:
                bad |= (
                    (chunk[:, 2] < 0) | (chunk[:, 2] > 12) | (chunk[:, 3] < 0) | (chunk[:, 3] > 31)
                )
            rows = np.flatnonzero(bad)
            if len(rows):
                row = int(rows[0])
                raise KGFormatError(
                    f"{path} row {at + row + 1}, ID Q{numbers[row]}: dangling, duplicate, "
                    "unsorted or invalid metadata row"
                )
            previous = int(numbers[-1])


def main(argv: list[str] | None = None) -> None:
    """Validate source files or explicitly verify a saved KG index's complete identity."""
    import argparse

    from xgraph.index import KGIndexFormatError, load_kg_index

    parser = argparse.ArgumentParser(prog="python -m xgraph.kg")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate", help="validate source KG files or verify a saved index")
    source = check.add_mutually_exclusive_group(required=True)
    source.add_argument("--kg", type=Path)
    source.add_argument("--index", type=Path)
    args = parser.parse_args(argv)
    result = {"valid": True}
    try:
        if args.index:
            index = load_kg_index(args.index, verify=True)
            result.update(
                base_fingerprint=index.base_fingerprint,
                fingerprint_status=index.fingerprint_status,
                integrity_verified=index.fingerprint_status == "verified",
            )
        else:
            validate(args.kg)
    except (KGFormatError, KGIndexFormatError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
