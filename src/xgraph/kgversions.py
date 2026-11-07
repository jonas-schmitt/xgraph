"""KG versions: named sets of edits over one KG index, chosen per sequence.

An edit is a triple (subject, relation, object) by id. A version groups its edits by
(subject, relation) and replaces every object the base KG index holds for each such pair
with the listed objects, as xgraph.mquake.apply_edits does to triples, so that a sequence
reading the version sees what the KG index rebuilt from the edited triples would hold
(docs/kg-versions-adr.md). The label index and the relation index stay the base's: an
edit whose subject or object has no label row, or whose relation has no relation index
row, is an unwritable edit, which adds no edge but still replaces its pair's objects.
An object of None explicitly deletes a pair without an unwritable added triple.

For the engine, every subject a version edits gets its whole versioned edge list, in the
base adjacency's layout: the base edges of its unedited relations and one edge per label
row of each listed object, sorted by relation and then object row. All versions share one
set of tables (`native_tables`); version k (counted from 1, 0 being the base KG) owns the
edited subjects version_subjects[version_offsets[k - 1]:version_offsets[k]], sorted, and
the edges of its i-th edited subject are version_edge_*[version_adjacency[i]:
version_adjacency[i + 1]].

A version file is a directory: kg-versions.json (the format, and each version's name and
edit file) and one .npy array of shape (n, 3) and dtype int32 per version, its edits as
Wikidata numbers (subject item, property, object item), as xgraph.mquake writes edits;
item n is entity "Q<n>" and property n relation "P<n>". The vLLM adapter loads one from
XGRAPH_KG_VERSIONS (xgraph.vllm); python -m xgraph.mquake --worlds --kg-versions <dir>
writes MQuAKE's worlds as versions w0, w1, ..., and

    python -m xgraph.kgversions tag --requests <requests.jsonl> \\
        --worlds <worlds.json> --out <tagged.jsonl>

gives each MQuAKE request its world's version in its kg_version field, which
xgraph.generate passes on to the engine, and

    python -m xgraph.kgversions conditions --questions <questions.json> \\
        --worlds <worlds.json> --out <conditions.json>

writes each MQuAKE question as an interventions build's question with one condition, its
world's version, whose record keeps the question's id (write_conditions), for GWalk
reading one KG store through the worlds' versions (python -m xgraph.gwalk --conditions).

write_kg_edits writes the string-id format instead: JSONL edit files (an object
may be null for pair deletion), checked file hashes and a content pin to the base
index. Both formats share the readers and the engine's version tables.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from xgraph.index import IndexMismatch, KGIndex

Edit = tuple[str, str, str | None]
"""Replace a pair's objects; None explicitly deletes the pair without an added edge."""


@dataclass(frozen=True)
class VersionStats:
    """What registering one version found."""

    name: str
    edits: int
    """Edits listed, duplicates included."""
    pairs: int
    """(subject, relation) pairs replaced: those whose subject and relation have labels."""
    unwritable_edits: int
    """Edits whose subject, relation, or non-None object has no label, adding no edge."""
    subjects: int
    """Subjects whose edge lists the version replaces."""
    edges: int
    """Adjacency entries of those subjects in the version."""


@dataclass(frozen=True)
class KGVersions:
    """The KG versions of one KG index, as tables the engine copies to the GPU."""

    names: tuple[str, ...]
    stats: tuple[VersionStats, ...]
    offsets: np.ndarray
    """int64 [versions + 1]: version k's edited subjects are subjects[offsets[k-1]:offsets[k]]."""
    subjects: np.ndarray
    """int32: edited subjects as entity positions (in labels.owner_ids), sorted per version."""
    adjacency: np.ndarray
    """int64 [len(subjects) + 1]: each edited subject's edges in edge_relations, edge_objects."""
    edge_relations: np.ndarray
    """int32: relation positions, sorted within a subject."""
    edge_objects: np.ndarray
    """int32: object label index rows, sorted within a subject and relation."""
    index_shape: tuple[str, int, int, int]
    """The tokenizer fingerprint, entity, relation, and label row counts of the KG index the
    versions were built for."""
    base_fingerprint: str
    """The exact base identity, including compiled tables and source provenance when available."""
    fingerprint_status: str = "verified"
    """'legacy' for edits loaded without a recorded base identity."""

    def number(self, name: str | None) -> int:
        """The engine's number of a version: 0 for the base KG (None), then 1, 2, ..."""
        if name is None:
            return 0
        if name not in self.names:
            raise ValueError(f"no KG version {name!r}; the registered ones are {list(self.names)}")
        return self.names.index(name) + 1

    def matches(self, index: KGIndex) -> bool:
        """Compare recorded base identities without rescanning table contents."""
        return self.base_fingerprint == index.base_fingerprint

    def check_base(self, index: KGIndex) -> None:
        """Refuse applying these tables to any other base identity."""
        if not self.matches(index):
            raise IndexMismatch(
                "the KG versions were built for another KG index: base fingerprint "
                f"{self.base_fingerprint} does not match {index.base_fingerprint}"
            )

    def native_tables(self) -> dict[str, np.ndarray]:
        return {
            "version_offsets": self.offsets,
            "version_subjects": self.subjects,
            "version_adjacency": self.adjacency,
            "version_edge_relations": self.edge_relations,
            "version_edge_objects": self.edge_objects,
        }

    def view(self, index: KGIndex, name: str | None) -> KGIndex | VersionView:
        """The KG index as a sequence reading this version sees it (the index for None)."""
        number = self.number(name)
        self.check_base(index)
        return index if number == 0 else VersionView(index, self, number)


def native_tables(versions: KGVersions | None) -> dict[str, np.ndarray]:
    """The engine's version tables: none registered when versions is None."""
    if versions is not None:
        return versions.native_tables()
    return {
        "version_offsets": np.zeros(1, dtype=np.int64),
        "version_subjects": np.zeros(0, dtype=np.int32),
        "version_adjacency": np.zeros(1, dtype=np.int64),
        "version_edge_relations": np.zeros(0, dtype=np.int32),
        "version_edge_objects": np.zeros(0, dtype=np.int32),
    }


class VersionView:
    """A KG index's relations_of and object_rows as one version holds them, for reading
    the engine's closed spans (xgraph.engine.closed_span) and states."""

    def __init__(self, index: KGIndex, versions: KGVersions, number: int) -> None:
        versions.check_base(index)
        self.labels, self.relations = index.labels, index.relations
        self._index = index
        self._versions = versions
        start, stop = int(versions.offsets[number - 1]), int(versions.offsets[number])
        self._first = start
        self._subjects = versions.subjects[start:stop]

    def relations_of(self, subject: str) -> list[str]:
        relations, _ = self._edges(subject)
        return [self.relations.owner_ids[r] for r in np.unique(relations)]

    def object_rows(
        self, subject: str, relation: str, lo: int = 0, hi: int | None = None
    ) -> list[int]:
        relations, objects = self._edges(subject)
        position = self._index._position(self._index._relation_at, relation, "relation")
        first = int(np.searchsorted(relations, position, side="left"))
        last = int(np.searchsorted(relations, position, side="right"))
        objects = objects[first:last]
        hi = self.labels.num_rows if hi is None else hi
        a, b = np.searchsorted(objects, lo, side="left"), np.searchsorted(objects, hi, side="left")
        return [int(row) for row in objects[a:b]]

    def edges_at(self, entity: int) -> tuple[np.ndarray, np.ndarray]:
        """The adjacency entries of the entity at this position as the version holds them:
        their relation positions and object rows (KGIndex.edges_at)."""
        at = int(np.searchsorted(self._subjects, entity))
        if at < len(self._subjects) and self._subjects[at] == entity:
            v = self._versions
            start, stop = int(v.adjacency[self._first + at]), int(v.adjacency[self._first + at + 1])
            return v.edge_relations[start:stop], v.edge_objects[start:stop]
        return self._index.edges_at(entity)

    def _edges(self, subject: str) -> tuple[np.ndarray, np.ndarray]:
        return self.edges_at(self._index._position(self._index._entity_at, subject, "entity"))


def _shape(index: KGIndex) -> tuple[str, int, int, int]:
    return (
        index.tokenizer_fingerprint,
        len(index.labels.owner_ids),
        len(index.relations.owner_ids),
        index.labels.num_rows,
    )


def build_kg_versions(index: KGIndex, versions: Mapping[str, Iterable[Edit]]) -> KGVersions:
    """The tables of these versions (edits by id, such as ("Q1", "P26", "Q2")) over the index.

    Raises ValueError for an empty version name. Edits naming ids the index
    does not hold, or holds without labels, are unwritable edits (VersionStats).

    Edits are grouped by subject/relation ID pair. Their objects replace all
    base objects for that pair; unmentioned pairs inherit the base. ``None``
    explicitly deletes a pair. An unwritable replacement can still remove its
    old objects, so inspect ``stats`` before serving. Versions share label
    tables and cannot add new entity/relation labels.

    The result binds to the index's recorded base fingerprint. Compilation
    runs on CPU; constructing an engine later uploads these tables to the GPU.
    """
    if isinstance(versions, VersionEdits):
        versions.check_base(index)
    named = {
        name: [(str(s), str(r), None if o is None else str(o)) for s, r, o in edits]
        for name, edits in versions.items()
    }
    for name in named:
        if not isinstance(name, str) or not name:
            raise ValueError(f"a KG version needs a non-empty name, got {name!r}")
    wanted = {x for edits in named.values() for s, _, o in edits for x in (s, o)}
    entity_at = {e: n for n, e in enumerate(index.labels.owner_ids) if e in wanted}
    relation_at = {r: n for n, r in enumerate(index.relations.owner_ids)}
    owners = np.asarray(index.labels.owners)
    known = np.array(sorted(entity_at.values()), dtype=np.int64)
    found = np.flatnonzero(np.isin(owners, known)) if len(known) else np.zeros(0, np.int64)
    rows_of: dict[int, list[int]] = {}
    for row in found.tolist():
        rows_of.setdefault(int(owners[row]), []).append(row)
    writable = set(np.unique(np.asarray(index.relations.owners)).tolist())

    offsets, subjects, adjacency = [0], [], [0]
    relations_out: list[np.ndarray] = []
    objects_out: list[np.ndarray] = []
    stats = []
    for name, edits in named.items():
        replaced: dict[int, dict[int, set[int]]] = {}  # subject: relation: object rows
        unwritable = 0
        for s, r, o in edits:
            subject, relation = entity_at.get(s), relation_at.get(r)
            if subject is None or subject not in rows_of or relation not in writable:
                unwritable += 1  # no base edge of the pair, and none to add
                continue
            rows = rows_of.get(entity_at.get(o, -1), [])
            unwritable += o is not None and not rows
            replaced.setdefault(subject, {}).setdefault(relation, set()).update(rows)
        edges = 0
        for subject in sorted(replaced):
            start = int(index.adjacency_offsets[subject])
            stop = int(index.adjacency_offsets[subject + 1])
            base_relations = np.asarray(index.edge_relations[start:stop], dtype=np.int64)
            base_objects = np.asarray(index.edge_objects[start:stop], dtype=np.int64)
            keep = ~np.isin(base_relations, list(replaced[subject]))
            added = [(r, row) for r, rows in replaced[subject].items() for row in rows]
            pairs = np.concatenate(
                [
                    np.stack([base_relations[keep], base_objects[keep]], axis=1),
                    np.array(added, dtype=np.int64).reshape(-1, 2),
                ]
            )
            pairs = np.unique(pairs, axis=0)  # sorted by relation, then object row
            relations_out.append(pairs[:, 0].astype(np.int32))
            objects_out.append(pairs[:, 1].astype(np.int32))
            subjects.append(subject)
            adjacency.append(adjacency[-1] + len(pairs))
            edges += len(pairs)
        offsets.append(len(subjects))
        stats.append(
            VersionStats(
                name=name,
                edits=len(edits),
                pairs=sum(len(r) for r in replaced.values()),
                unwritable_edits=unwritable,
                subjects=len(replaced),
                edges=edges,
            )
        )

    def join(parts: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)

    return KGVersions(
        names=tuple(named),
        stats=tuple(stats),
        offsets=np.array(offsets, dtype=np.int64),
        subjects=np.array(subjects, dtype=np.int32),
        adjacency=np.array(adjacency, dtype=np.int64),
        edge_relations=join(relations_out),
        edge_objects=join(objects_out),
        index_shape=_shape(index),
        base_fingerprint=index.base_fingerprint,
        fingerprint_status=(
            versions.fingerprint_status
            if isinstance(versions, VersionEdits)
            else index.fingerprint_status
        ),
    )


VERSIONS_FILE = "kg-versions.json"
VERSIONS_FORMAT = 1
"""The version file layout this module writes and reads."""
ID_VERSIONS_FORMAT = 2
"""Explicit string ids, pair deletion, and a pin to the base index's content."""
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


class KGVersionsFormatError(ValueError):
    """A version file is missing parts or was written in another format."""


class VersionEdits(dict):
    """Edit mapping retaining a persisted base fingerprint for every consuming reader.

    Unbound legacy files have fingerprint_status='legacy'. New bound files are
    checked by build_kg_versions, load_kg_versions and Oracle before use.
    """

    def __init__(
        self,
        *,
        base_fingerprint: str | None,
        source: str,
        base_index_identity: str | None = None,
    ):
        super().__init__()
        self.base_fingerprint = base_fingerprint
        self.fingerprint_status = "legacy" if base_fingerprint is None else "verified"
        self.source = source
        self.base_index_identity = base_index_identity

    def check_base(self, index: KGIndex) -> None:
        if (
            self.base_fingerprint is None
            and self.base_index_identity is not None
            and self.base_index_identity != _identity(index)
        ):
            raise KGVersionsFormatError(
                f"{self.source}: versions are pinned to a different base KG index"
            )
        if self.base_fingerprint is not None and self.base_fingerprint != index.base_fingerprint:
            raise IndexMismatch(
                f"{self.source}: base fingerprint {self.base_fingerprint} does not match "
                f"KG index {index.base_fingerprint}"
            )


def _edit_array(edits, source):
    edits = np.asarray(edits)
    if edits.ndim != 2 or edits.shape[1] != 3:
        raise KGVersionsFormatError(f"{source}: edits must be (n, 3), got {edits.shape}")
    if not np.issubdtype(edits.dtype, np.integer):
        raise KGVersionsFormatError(f"{source}: edits dtype must be integer, got {edits.dtype}")
    for at in range(0, len(edits), 65536):
        chunk = edits[at : at + 65536]
        bad = np.argwhere((chunk < 0) | (chunk > np.iinfo(np.int32).max))
        if len(bad):
            row, col = map(int, bad[0])
            raise KGVersionsFormatError(
                f"{source} row {at + row + 1}, ID {chunk[row, col]}: outside int32 ID range"
            )
    return edits


def write_kg_versions(
    directory: str | Path,
    versions: Mapping[str, np.ndarray],
    *,
    index: KGIndex | None = None,
    **metadata: object,
) -> None:
    """Write a version file: each version's edits, an (n, 3) array of Wikidata numbers
    (subject item, property, object item), as <name>.npy, and kg-versions.json listing
    them in order, with any metadata (such as provenance) beside them.

    Supply index to pin the edits to its exact base fingerprint. Omitting it
    retains the legacy unbound format, which strict readers refuse.
    """
    if set(metadata) & {"base_fingerprint", "format", "versions"}:
        raise ValueError("reserved KG-version metadata; supply index to bind the base fingerprint")
    if index is not None and index.fingerprint_status == "legacy":
        raise ValueError(
            "binding persisted versions requires a complete base fingerprint; legacy index supplied"
        )
    directory = Path(directory)
    listed = []
    for name, edits in versions.items():
        if not _NAME.fullmatch(name):
            raise ValueError(
                f"a KG version name is letters, digits, '_', '.' and '-', got {name!r}"
            )
        edits = _edit_array(edits, f"KG version {name!r}")
        listed.append({"name": name, "edits": f"{name}.npy", "count": len(edits)})
    directory.mkdir(parents=True, exist_ok=True)
    for entry in listed:
        np.save(directory / entry["edits"], np.asarray(versions[entry["name"]], dtype=np.int32))
    header = {"format": VERSIONS_FORMAT, "versions": listed, **metadata}
    if index is not None:
        header["base_fingerprint"] = index.base_fingerprint
    (directory / VERSIONS_FILE).write_text(json.dumps(header, indent=1) + "\n", encoding="utf-8")


def _identity(index: KGIndex) -> str:
    """Pin the complete saved index contract, independent of array memory layout."""
    import hashlib
    from dataclasses import asdict

    digest = hashlib.sha256()
    metadata = {
        "tokenizer": index.tokenizer_fingerprint,
        "syntax": asdict(index.span_syntax),
        "unwritable_triples": index.unwritable_triples,
        "expressions": index.expressions,
        "verdicts": index.verdicts,
        "initial_case_aliases": index.initial_case_aliases,
    }
    arrays = {}
    for part in ("labels", "relations"):
        rows = getattr(index, part)
        metadata[part] = [rows.owner_ids, [asdict(e) for e in rows.excluded]]
        arrays.update(
            {f"{part}.{name}": getattr(rows, name) for name in ("tokens", "offsets", "owners")}
        )
    for name in (
        "adjacency_offsets",
        "edge_relations",
        "edge_objects",
        "sitelinks",
        "date_entities",
        "date_keys",
        "relation_primary_rows",
    ):
        arrays[name] = getattr(index, name)
    digest.update(json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode())
    for name, array in arrays.items():
        digest.update(name.encode())
        if array is None:
            digest.update(b"null")
            continue
        digest.update(f"{array.dtype.str}:{array.shape}".encode())
        for first in range(0, len(array), 1 << 18):
            digest.update(array[first : first + (1 << 18)].tobytes(order="C"))
    return digest.hexdigest()


def write_kg_edits(
    directory: str | Path,
    versions: Mapping[str, Iterable[Edit]],
    *,
    index: KGIndex,
    **metadata: object,
) -> None:
    """Write string-id edits, pinned to index, with None for pair deletion.

    Requires a complete base fingerprint. These files work with read_kg_versions,
    load_kg_versions and Oracle's versions without scanning the base arrays.

    ``versions`` maps names to application-ID edit triples. Each pair supplies
    either replacement objects or an explicit deletion, never both. The writer
    validates edits, writes UTF-8 JSONL files and records their SHA-256 hashes
    with the exact base identity. Additional metadata is recorded in the header;
    use it for source provenance without replacing reserved identity fields.
    Save separate experiments in separate directories.
    """
    from xgraph.provenance import sha256_file

    if set(metadata) & {"base_fingerprint", "base_index_identity", "format", "versions"}:
        raise ValueError("reserved KG-version metadata; supply index to bind the base fingerprint")
    if index.fingerprint_status == "legacy":
        raise ValueError(
            "binding persisted versions requires a complete base fingerprint; legacy index supplied"
        )
    named = {name: list(edits) for name, edits in versions.items()}
    for name, edits in named.items():
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError(f"invalid KG version name {name!r}")
        for edit in edits:
            if (
                not isinstance(edit, (tuple, list))
                or len(edit) != 3
                or any(not isinstance(x, str) or not x for x in edit[:2])
                or (edit[2] is not None and (not isinstance(edit[2], str) or not edit[2]))
            ):
                raise ValueError(
                    f"KG version {name!r}: expected (subject, relation, object or None), "
                    f"got {edit!r}"
                )
        deleted = {(s, r) for s, r, o in edits if o is None}
        if any((s, r) in deleted for s, r, o in edits if o is not None):
            raise ValueError(f"KG version {name!r}: pair has both deletion and replacement objects")
    build_kg_versions(index, named)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    listed = []
    for name, edits in named.items():
        filename = f"{name}.jsonl"
        (directory / filename).write_text(
            "".join(json.dumps(edit, ensure_ascii=False) + "\n" for edit in edits), encoding="utf-8"
        )
        listed.append(
            {
                "name": name,
                "edits": filename,
                "count": len(edits),
                "sha256": sha256_file(directory / filename),
            }
        )
    header = {
        **metadata,
        "format": ID_VERSIONS_FORMAT,
        "base_fingerprint": index.base_fingerprint,
        "versions": listed,
    }
    (directory / VERSIONS_FILE).write_text(json.dumps(header, indent=1) + "\n", encoding="utf-8")


def _header(directory: str | Path) -> dict:
    path = Path(directory) / VERSIONS_FILE
    if not path.is_file():
        raise KGVersionsFormatError(f"{directory} holds no {VERSIONS_FILE}")
    header = json.loads(path.read_text(encoding="utf-8"))
    if header.get("format") not in (VERSIONS_FORMAT, ID_VERSIONS_FORMAT):
        raise KGVersionsFormatError(
            f"{path} is of format {header.get('format')}; this xgraph reads formats "
            f"{VERSIONS_FORMAT} and {ID_VERSIONS_FORMAT}"
        )
    return header


def version_names(directory: str | Path) -> list[str]:
    """The names a version file registers, in order, read from its kg-versions.json alone."""
    return [entry["name"] for entry in _header(directory)["versions"]]


def component_ranges(directory: str | Path) -> dict[str, list[int]]:
    """Optional question-component bounds for store readers of disconnected KGs.

    The engine uses anchors within these components; the bounds scope SQL readers,
    including label lookup. They are not edge overlays or an engine access control.
    """
    header = _header(directory)
    components = header.get("components", {})
    names = {entry["name"] for entry in header["versions"]}
    if not isinstance(components, dict):
        raise KGVersionsFormatError("KG components must map version names to entity ranges")
    for name, bounds in components.items():
        if (
            name not in names
            or not isinstance(bounds, list)
            or len(bounds) != 2
            or any(type(n) is not int for n in bounds)
            or not 0 <= bounds[0] < bounds[1]
        ):
            raise KGVersionsFormatError(f"invalid KG component {name!r}: {bounds!r}")
    return components


def version_files(directory: str | Path) -> list[Path]:
    """The files of a version file, kg-versions.json first, for provenance."""
    directory = Path(directory)
    entries = _header(directory)["versions"]
    return [directory / VERSIONS_FILE, *(directory / entry["edits"] for entry in entries)]


def read_kg_versions(
    directory: str | Path, *, index: KGIndex | None = None, strict: bool = False
) -> VersionEdits:
    """Read named edit sets as subject/relation/object ID tuples.

    Supports string-ID JSONL and legacy numeric arrays, with ``None`` for an
    explicit pair deletion in the string format. JSONL files are hash-checked.
    The dict-compatible return value retains its base binding for oracle and
    compiled-version admission. Supplying ``index`` checks that binding now;
    ``strict=True`` refuses files without a complete recorded base identity.
    Identity binding does not authenticate headers or revalidate source truth.
    """
    directory = Path(directory)
    header = _header(directory)
    fingerprint = header.get("base_fingerprint")
    if fingerprint is not None and (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(c not in "0123456789abcdef" for c in fingerprint)
    ):
        raise KGVersionsFormatError(f"{directory}/{VERSIONS_FILE}: malformed base fingerprint")
    if fingerprint is None and strict:
        raise KGVersionsFormatError(
            f"{directory}: legacy versions have no base fingerprint; strict mode refuses them"
        )
    legacy_identity = None
    if header["format"] == ID_VERSIONS_FORMAT and fingerprint is None:
        legacy_identity = header.get("base_index_identity")
        if (
            not isinstance(legacy_identity, str)
            or len(legacy_identity) != 64
            or any(c not in "0123456789abcdef" for c in legacy_identity)
        ):
            raise KGVersionsFormatError(f"{directory}: missing or malformed string-ID base pin")
    versions = VersionEdits(
        base_fingerprint=fingerprint,
        source=str(directory / VERSIONS_FILE),
        base_index_identity=legacy_identity,
    )
    if index is not None:
        versions.check_base(index)
        if strict and index.fingerprint_status == "legacy":
            raise KGVersionsFormatError(
                "legacy base index: strict mode requires a recorded complete fingerprint"
            )
    for entry in header["versions"]:
        file = directory / entry["edits"]
        if not file.is_file():
            raise KGVersionsFormatError(f"{directory} misses {file.name}")
        if header["format"] == ID_VERSIONS_FORMAT:
            from xgraph.provenance import sha256_file

            if sha256_file(file) != entry.get("sha256"):
                raise KGVersionsFormatError(
                    f"{file}: KG edits SHA-256 does not match the version file"
                )
            try:
                edits = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()]
                if any(
                    not isinstance(e, list)
                    or len(e) != 3
                    or any(not isinstance(x, str) or not x for x in e[:2])
                    or (e[2] is not None and (not isinstance(e[2], str) or not e[2]))
                    for e in edits
                ):
                    raise ValueError("invalid edit")
            except (ValueError, TypeError) as exc:
                raise KGVersionsFormatError(f"{file}: invalid string-id KG edits") from exc
            versions[entry["name"]] = [tuple(e) for e in edits]
            continue
        try:
            edits = _edit_array(np.load(file, mmap_mode="r", allow_pickle=False), str(file))
        except (OSError, ValueError, EOFError) as error:
            raise KGVersionsFormatError(f"{file}: invalid edits array: {error}") from error
        versions[entry["name"]] = [(f"Q{s}", f"P{p}", f"Q{o}") for s, p, o in edits.tolist()]
    return versions


def load_kg_versions(directory: str | Path, index: KGIndex, *, strict: bool = False) -> KGVersions:
    """Read, check and compile a version directory for this exact base index.

    Returns CPU-built ``KGVersions`` for engine construction or a version view.
    ``strict=True`` refuses legacy files or indexes lacking complete recorded
    identities. This call does not initialize CUDA or modify the base arrays.
    """
    return build_kg_versions(index, read_kg_versions(directory, index=index, strict=strict))


def write_conditions(questions: Path, worlds: dict, worlds_path: Path, out: Path) -> None:
    """The conditions command: each MQuAKE question (mquake-<case>-<k>) as a question of an
    interventions build (python -m xgraph.interventions, conditions.json) with one
    condition, named and read as its case's world's KG version (w<k>), whose record_id is
    the question's id, so that python -m xgraph.gwalk --conditions asks every question in
    its world's KG version and python -m xgraph.editing scores its records."""
    from xgraph.mquake import case_id
    from xgraph.provenance import provenance

    rows = []
    for question in json.loads(questions.read_text(encoding="utf-8")):
        world = worlds.get(case_id(question["id"]))
        if world is None:
            raise SystemExit(f"{worlds_path} places no world for question {question['id']!r}")
        version = f"w{world}"
        condition = {"name": version, "version": version, "record_id": question["id"]}
        rows.append({"id": question["id"], "utterance": question["utterance"],
                     "conditions": [condition]})  # fmt: skip
    built = {
        "artifact_kind": "xgraph.kgversions_conditions",
        "version": 1,
        "counts": {
            "questions": len(rows),
            "versions": len({r["conditions"][0]["version"] for r in rows}),
        },
        "questions": rows,
        "provenance": provenance([questions, worlds_path]),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(built, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(built["counts"]))


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m xgraph.kgversions", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    tag = commands.add_parser(
        "tag",
        help="give MQuAKE requests their world's KG version (w<k>) as their kg_version",
    )
    tag.add_argument("--requests", type=Path, required=True, help="requests (JSONL)")
    tag.add_argument(
        "--worlds", type=Path, required=True, help="worlds.json of python -m xgraph.mquake"
    )
    tag.add_argument("--out", type=Path, required=True, help="the tagged requests (JSONL)")
    asked = commands.add_parser(
        "conditions",
        help="MQuAKE questions as an interventions build's conditions.json, each question one "
        "condition in its world's KG version (w<k>), its record keeping the question's id, "
        "for python -m xgraph.gwalk --conditions",
    )
    asked.add_argument("--questions", type=Path, required=True, help="questions (WWQ format)")
    asked.add_argument(
        "--worlds", type=Path, required=True, help="worlds.json of python -m xgraph.mquake"
    )
    asked.add_argument("--out", type=Path, required=True, help="conditions.json")
    args = parser.parse_args(argv)

    from xgraph.mquake import case_id

    worlds = json.loads(args.worlds.read_text(encoding="utf-8"))
    if args.command == "conditions":
        write_conditions(args.questions, worlds, args.worlds, args.out)
        return
    tagged = []
    for line in args.requests.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        request = json.loads(line)
        world = worlds.get(case_id(request["id"]))
        if world is None:
            raise SystemExit(f"{args.worlds} places no world for request {request['id']!r}")
        tagged.append({**request, "kg_version": f"w{world}"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for request in tagged:
            out.write(json.dumps(request, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
