"""KG index build: the label index, the relation index, and the adjacency.

The label index holds one row per (entity, label) pair: the label's tokens under
the served tokenizer. Rows are sorted lexicographically by token sequence, a row
that ends sorting before every row that continues it, so the rows sharing a typed
token prefix form one contiguous row range and, inside it, each column is sorted.
"""

from __future__ import annotations

import bisect
import dataclasses
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from xgraph.expressions import PRECISIONS, Date
from xgraph.kg import Entity, KGFormatError, KnowledgeGraph, Relation, validate
from xgraph.syntax import SpanSyntax
from xgraph.tokenizer import ServedTokenizer

END = -1
"""The column value past a row's last token; it sorts before every token id."""


class UnknownId(LookupError):
    """A query names an entity or relation the KG index does not hold."""


class IndexMismatch(ValueError):
    """The KG index, tokenizer, or KG-version base identity differs."""


@dataclass(frozen=True)
class ExcludedLabel:
    """A label the index leaves out, with the reason."""

    owner_id: str
    """The entity or relation the label belongs to."""
    label: str
    reason: str


@dataclass(frozen=True)
class LabelIndex:
    """Sorted tokenized label rows, retaining every distinct label owner.

    A token prefix selects a contiguous half-open row range. Each row identifies
    an entity or relation through ``owner_id``; several rows can belong to the
    same owner, and the same token sequence can have several owners. ``excluded``
    records source labels that cannot be written in the configured span syntax.
    Arrays are immutable in a shared index and can be memory-mapped from disk.
    """

    tokens: np.ndarray
    """int32: the tokens of all rows, concatenated in row order."""
    offsets: np.ndarray
    """int64, one longer than the row count: row r is tokens[offsets[r]:offsets[r + 1]]."""
    owners: np.ndarray
    """int32: for each row, the position of its owner (the entity or relation it names)
    in owner_ids."""
    owner_ids: tuple[str, ...]
    tokenizer_fingerprint: str
    excluded: tuple[ExcludedLabel, ...]

    @property
    def num_rows(self) -> int:
        """Number of tokenized label rows, including distinct alias owners."""
        return len(self.owners)

    def row_tokens(self, row: int) -> tuple[int, ...]:
        """Return one row's prefixed label tokens in the served vocabulary."""
        return tuple(int(t) for t in self.tokens[self.offsets[row] : self.offsets[row + 1]])

    def owner_id(self, row: int) -> str:
        """The id of the entity or relation the row names."""
        return self.owner_ids[self.owners[row]]

    def token_at(self, row: int, depth: int) -> int:
        """The row's token at this depth, or END past its last token."""
        start, stop = self.offsets[row], self.offsets[row + 1]
        return int(self.tokens[start + depth]) if start + depth < stop else END

    def prefix_range(self, prefix: tuple[int, ...] | list[int]) -> tuple[int, int]:
        """The row range [lo, hi) of rows that start with these tokens."""
        lo, hi = 0, self.num_rows
        for depth, token in enumerate(prefix):
            column = range(lo, hi)
            key = lambda row, depth=depth: self.token_at(row, depth)  # noqa: E731
            lo, hi = (
                lo + bisect.bisect_left(column, token, key=key),
                lo + bisect.bisect_right(column, token, key=key),
            )
        return lo, hi


def build_label_index(
    kg: KnowledgeGraph,
    tokenizer: ServedTokenizer,
    syntax: SpanSyntax,
    *,
    initial_case_aliases: bool = False,
) -> LabelIndex:
    """Tokenize exact, nonempty entity label encodings and sort; see label_rows.

    Non-round-tripping labels and encodings with unknown bytes are excluded and
    reported, including when the tokenizer intentionally normalizes the text.
    """
    entities = case_alias_entities(kg.entities)[0] if initial_case_aliases else kg.entities
    return label_rows(entities, tokenizer, syntax)


def initial_case_alias(label: str) -> str | None:
    """initial-case-v1: toggle one cased initial code point, preserving the rest.

    Do not skip punctuation/whitespace or expand a character into several code
    points. In particular, this does not title-case words or normalize names.
    """
    if not label:
        return None
    first = label[0]
    alternate = first.upper() if first.islower() else first.lower()
    if len(alternate) != 1 or alternate == first:
        return None
    return alternate + label[1:]


def case_alias_entities(entities):
    """Add only canonical-label variants, deduplicated by exact owner/surface."""
    expanded, added = [], []
    for entity in entities:
        alias = initial_case_alias(entity.labels[0]) if entity.labels else None
        if alias is not None and alias not in entity.labels:
            added.append((entity.id, alias))
            entity = dataclasses.replace(entity, labels=(*entity.labels, alias))
        expanded.append(entity)
    return tuple(expanded), added


def case_alias_statistics(entities, added, excluded):
    """Count admitted alias-owner pairs and those sharing an exact surface.

    A collision counts each added pair whose surface has another owner in the
    final label index, including collisions between generated aliases.
    """
    excluded = {(e.owner_id, e.label) for e in excluded}
    owners = {}
    for entity in entities:
        for label in entity.labels:
            if (entity.id, label) not in excluded:
                owners.setdefault(label, set()).add(entity.id)
    admitted = [(owner, label) for owner, label in added if (owner, label) not in excluded]
    return {
        "rule": "initial-case-v1",
        "aliases": len(admitted),
        "collisions": sum(len(owners[label]) > 1 for _, label in admitted),
    }


def _label_encoding_error(label, tokens, tokenizer, syntax):
    """exact-v1 admission of a prefixed label, shared by every index builder."""
    if tokenizer.special_ids.intersection(tokens):
        return "label contains a special token"
    if not tokens:
        return "empty label encoding"
    pieces = [tokenizer.token_bytes(token) for token in tokens]
    if None in pieces:
        return "label contains a token with unknown bytes"
    if b"".join(pieces) != (syntax.label_prefix + label).encode("utf-8"):
        return "label bytes do not round-trip"
    return None


def label_rows(
    items: Sequence[Entity | Relation], tokenizer: ServedTokenizer, syntax: SpanSyntax
) -> LabelIndex:
    """A label index over the labels of these entities or relations.

    Each row holds the label as a span writes it, after the label prefix. Blank
    labels, labels whose written form holds span syntax (SpanSyntax.excludes), and
    labels that hold the text of a special token (which the tokenizer turns into that
    token, and which a span never allows) are excluded and reported. A label repeated for the
    same item becomes one row; a label shared by several items becomes one row per
    item. exact-v1 also excludes empty encodings, unknown token bytes and any
    encoding whose concatenated token bytes differ from the UTF-8 label prefix
    plus source label. Tokenizer normalization is not a label equivalence policy.
    """
    pairs: list[tuple[str, int]] = []
    excluded: list[ExcludedLabel] = []
    seen: set[tuple[int, str]] = set()
    for item_index, item in enumerate(items):
        for label in item.labels:
            if (item_index, label) in seen:
                continue
            seen.add((item_index, label))
            if not label.strip():
                excluded.append(ExcludedLabel(item.id, label, "blank label"))
            elif syntax.excludes(label):
                excluded.append(ExcludedLabel(item.id, label, "label contains span syntax"))
            else:
                pairs.append((label, item_index))

    rows = []
    for (label, item_index), tokens in zip(
        pairs,
        tokenizer.encode_batch([syntax.label_prefix + label for label, _ in pairs]),
        strict=True,
    ):
        reason = _label_encoding_error(label, tokens, tokenizer, syntax)
        if reason:
            excluded.append(ExcludedLabel(items[item_index].id, label, reason))
        else:
            rows.append((tuple(tokens), item_index))
    rows.sort()
    lengths = np.array([len(tokens) for tokens, _ in rows], dtype=np.int64)
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    tokens = np.fromiter((t for row, _ in rows for t in row), dtype=np.int32, count=offsets[-1])
    return LabelIndex(
        tokens=tokens,
        offsets=offsets,
        owners=np.array([e for _, e in rows], dtype=np.int32),
        owner_ids=tuple(item.id for item in items),
        tokenizer_fingerprint=tokenizer.fingerprint,
        excluded=tuple(excluded),
    )


@dataclass(frozen=True)
class KGIndex:
    """One source KG compiled for a served tokenizer and span syntax.

    Use :func:`build_kg_index` or :func:`load_kg_index` to obtain this object.
    Entity labels, relation labels and adjacency share integer array positions;
    public ID-based queries translate application IDs for you. Object aliases
    expand adjacency by label row. Inspect exclusions and unwritable triples
    before serving, and keep arrays and recorded identity immutable in use.
    """

    labels: LabelIndex
    """The label index of the entities."""
    relations: LabelIndex
    """The relation index: relation labels as written, in the same layout."""
    adjacency_offsets: np.ndarray
    """int64, one longer than the entity count: subject e's edges are
    edge_relations[adjacency_offsets[e]:adjacency_offsets[e + 1]], likewise edge_objects."""
    edge_relations: np.ndarray
    """int32: the edge's relation, as a position in relations.owner_ids."""
    edge_objects: np.ndarray
    """int32: the label index row of the edge's object. Edges of one subject are sorted by
    relation and then object row, so the objects of one relation form one sorted object slice."""
    unwritable_triples: int
    """Triples left out because their subject, relation, or object has no label to write."""
    span_syntax: SpanSyntax = field(default_factory=SpanSyntax)
    """The span syntax the index was built for: labels are tokenized after its label prefix,
    and labels containing its delimiters are excluded."""
    sitelinks: np.ndarray | None = None
    """int32, one per entity (in labels.owner_ids order): its sitelink count, the sites
    that link to it (0 if unknown); None for an index built without sitelink counts. The
    sitelink rank orders an ambiguous label's entities by it."""
    date_entities: np.ndarray | None = None
    """int32, sorted: the positions (in labels.owner_ids) of the entities that are dates,
    such as merged dates (xgraph.literals); None for an index built without dates."""
    date_keys: np.ndarray | None = None
    """int32, one per date entity: its date packed into a key (pack_date)."""
    verdicts: dict[str, str] | None = None
    """The value comparison's verdict entities by verdict ("yes", "no"; VERDICTS), for an
    index built with them; None otherwise."""
    expressions: bool = True
    """Whether the build excluded the labels that hold expression syntax (the operand
    separator, or an operator at the start; SpanSyntax.excludes); False for an index saved
    before it did, whose labels may hold it."""
    relation_primary_rows: np.ndarray | None = None
    """int32, one per relation: its canonical label row, or -1 if excluded. None
    means a legacy index that did not retain canonical-label identity."""
    initial_case_aliases: dict | None = None
    """Opt-in entity surface rule and admitted alias/collision counts; absent by default."""
    source_graph_hash: str | None = None
    """SHA-256 of all source entities, relations, labels and triples; absent on legacy indexes."""
    build_config: dict | None = None
    """The builder and its options, independent of file locations."""
    fingerprint_status: str = "legacy"
    """'verified' for a complete recorded identity, not a load-time integrity check."""
    _recorded_base_fingerprint: InitVar[str | None] = None
    base_fingerprint: str = field(init=False)
    """Recorded complete identity, or the legacy tokenizer/entity/relation/row-count identity."""

    def __post_init__(self, _recorded_base_fingerprint: str | None) -> None:
        if _recorded_base_fingerprint is not None:
            identity = _recorded_base_fingerprint
        elif self.fingerprint_status == "verified":
            identity = self.compute_base_fingerprint()
        else:
            # Preserve the admission rule used before complete identities were recorded.
            shape = (
                self.tokenizer_fingerprint,
                len(self.labels.owner_ids),
                len(self.relations.owner_ids),
                self.labels.num_rows,
            )
            identity = "legacy:" + hashlib.sha256(json.dumps(shape).encode()).hexdigest()
        object.__setattr__(self, "base_fingerprint", identity)

    def compute_base_fingerprint(self) -> str:
        """Explicit full pass over source identity, configuration and compiled tables.

        Builders compute this once; loaders use the recorded identity unless asked
        to verify. Index tables and metadata must remain immutable during use.
        dataclasses.replace recomputes identity for changed modern indexes.
        """
        digest = hashlib.sha256()
        _hash_record(digest, ["xgraph.base-index", 1, self.source_graph_hash, self.build_config])
        _hash_record(
            digest,
            [
                self.tokenizer_fingerprint,
                dataclasses.asdict(self.span_syntax),
                self.unwritable_triples,
                self.expressions,
                self.verdicts,
                self.initial_case_aliases,
            ],
        )
        for part in _ARRAYS:
            rows = getattr(self, part)
            _hash_record(digest, [part, rows.tokenizer_fingerprint])
            for owner in rows.owner_ids:
                _hash_record(digest, ["owner", owner])
            for excluded in rows.excluded:
                _hash_record(digest, ["excluded", dataclasses.asdict(excluded)])
            for name in _ARRAYS[part]:
                _hash_array(digest, f"{part}.{name}", getattr(rows, name))
        for name in (*_ADJACENCY, SITELINKS, *DATES, "relation_primary_rows"):
            _hash_array(digest, name, getattr(self, name))
        return digest.hexdigest()

    @property
    def tokenizer_fingerprint(self) -> str:
        return self.labels.tokenizer_fingerprint

    def relations_of(self, subject: str) -> list[str]:
        """The relations of the subject's triples, by relation position."""
        start, stop = self._edges(subject)
        return [self.relations.owner_ids[r] for r in np.unique(self.edge_relations[start:stop])]

    def object_rows(
        self, subject: str, relation: str, lo: int = 0, hi: int | None = None
    ) -> list[int]:
        """The object slice of (subject, relation), restricted to the row range [lo, hi)."""
        start, stop = self._edges(subject)
        position = self._position(self._relation_at, relation, "relation")
        relations = self.edge_relations[start:stop]
        first = start + int(np.searchsorted(relations, position, side="left"))
        last = start + int(np.searchsorted(relations, position, side="right"))
        objects = self.edge_objects[first:last]
        hi = self.labels.num_rows if hi is None else hi
        a, b = np.searchsorted(objects, lo, side="left"), np.searchsorted(objects, hi, side="left")
        return [int(row) for row in objects[a:b]]

    @cached_property
    def _entity_at(self) -> dict[str, int]:
        return {entity: i for i, entity in enumerate(self.labels.owner_ids)}

    @cached_property
    def _relation_at(self) -> dict[str, int]:
        return {relation: i for i, relation in enumerate(self.relations.owner_ids)}

    @staticmethod
    def _position(positions: dict[str, int], item: str, kind: str) -> int:
        if item not in positions:
            raise UnknownId(f"the KG index holds no {kind} {item!r}")
        return positions[item]

    def edges_at(self, entity: int) -> tuple[np.ndarray, np.ndarray]:
        """The adjacency entries of the entity at this position (in labels.owner_ids): their
        relation positions and object rows."""
        start, stop = int(self.adjacency_offsets[entity]), int(self.adjacency_offsets[entity + 1])
        return self.edge_relations[start:stop], self.edge_objects[start:stop]

    def _edges(self, subject: str) -> tuple[int, int]:
        entity = self._position(self._entity_at, subject, "entity")
        return int(self.adjacency_offsets[entity]), int(self.adjacency_offsets[entity + 1])

    def date_of(self, entity: str) -> Date | None:
        """The entity's date, from the date table; None if it is no date or the index has
        no dates."""
        position = self._position(self._entity_at, entity, "entity")
        if self.date_entities is None:
            return None
        at = int(np.searchsorted(self.date_entities, position))
        if at == len(self.date_entities) or self.date_entities[at] != position:
            return None
        return unpack_date(int(self.date_keys[at]))


def build_kg_index(
    kg: KnowledgeGraph,
    tokenizer: ServedTokenizer,
    syntax: SpanSyntax,
    sitelinks: Mapping[str, int] | None = None,
    dates: Mapping[str, Date] | None = None,
    verdicts: bool = False,
    *,
    initial_case_aliases: bool = False,
) -> KGIndex:
    """The label index, the relation index, and the adjacency of the KG's triples; with
    sitelinks (sitelink counts by entity id), each entity's count, 0 for one not named;
    with dates (the dates of the entities that are dates, by entity id), the date table
    (date_table); with verdicts, the verdict entities (VERDICTS), after the KG's.

    A triple naming an unknown entity or relation is a KGFormatError. A triple whose
    subject, relation, or object has no label left to write is counted and left out.
    Each object entity contributes one edge per label row, so that an object slice
    is in label index order.

    Args:
        kg: In-memory source graph with unique entity/relation IDs and labels.
        tokenizer: The served model's tokenizer, or a local demonstration BPE
            tokenizer for CPU examples. No model weights are needed.
        syntax: Delimiters and label prefix; reuse this syntax when decoding.
        sitelinks: Optional nonnegative counts by entity ID, zero for missing
            IDs. Empty counts supply neutral zeros for witness diagnostics.
        dates: Optional date values by entity ID for comparison expressions.
        verdicts: Add the reserved comparison verdict entities.
        initial_case_aliases: Opt in to the initial-case entity surface rule.

    Returns:
        KGIndex: Tokenizer-bound compiled tables with a recorded base identity.

    The build materializes source records and expanded adjacency. For larger
    sources use the bounded KG directory builder. A successful build can still
    exclude labels and triples; it does not validate real-world truth. Every
    entity and relation row must exactly reconstruct its prefixed source label
    from known token bytes (exact-v1); failures are reported in ``excluded``.
    The versioned rule is recorded in ``build_config["label_encoding"]``. Loading
    existing indexes preserves their original rows and admission policy.
    """
    validate(kg)
    source_hash = _source_graph_hash(kg)
    if verdicts:
        taken = sorted(set(VERDICTS.values()) & {e.id for e in kg.entities})
        if taken:
            raise KGFormatError(f"the KG holds {taken}, the ids of the verdict entities")
        kg = dataclasses.replace(
            kg, entities=(*kg.entities, *(Entity(i, (v,)) for v, i in VERDICTS.items()))
        )
    aliases = None
    if initial_case_aliases:
        entities, added = case_alias_entities(kg.entities)
        labels = label_rows(entities, tokenizer, syntax)
        aliases = case_alias_statistics(entities, added, labels.excluded)
    else:
        labels = build_label_index(kg, tokenizer, syntax)
    relations = label_rows(kg.relations, tokenizer, syntax)
    entity_at = {entity.id: i for i, entity in enumerate(kg.entities)}
    relation_at = {relation.id: i for i, relation in enumerate(kg.relations)}
    rows_of: list[list[int]] = [[] for _ in kg.entities]
    for row, entity in enumerate(labels.owners):
        rows_of[entity].append(row)
    writable_relations = set(int(r) for r in relations.owners)

    edges: set[tuple[int, int, int]] = set()
    unwritable = 0
    for subject, relation, obj in kg.triples:
        s, r, o = entity_at[subject], relation_at[relation], entity_at[obj]
        if not rows_of[s] or not rows_of[o] or r not in writable_relations:
            unwritable += 1
            continue
        edges.update((s, r, row) for row in rows_of[o])

    ordered = np.array(sorted(edges), dtype=np.int64).reshape(-1, 3)
    offsets = np.zeros(len(kg.entities) + 1, dtype=np.int64)
    np.cumsum(np.bincount(ordered[:, 0], minlength=len(kg.entities)), out=offsets[1:])
    counts = None
    if sitelinks is not None:
        counts = sitelink_array([sitelinks.get(e.id, 0) for e in kg.entities], len(kg.entities))
    table = (None, None)
    if dates is not None:
        unknown = sorted(set(dates) - set(entity_at))
        if unknown:
            raise KGFormatError(f"dates name entities the KG lacks: {unknown[:5]}")
        table = date_table({entity_at[e]: date for e, date in dates.items()})
    return KGIndex(
        labels=labels,
        relations=relations,
        adjacency_offsets=offsets,
        edge_relations=ordered[:, 1].astype(np.int32),
        edge_objects=ordered[:, 2].astype(np.int32),
        unwritable_triples=unwritable,
        span_syntax=syntax,
        sitelinks=counts,
        verdicts=dict(VERDICTS) if verdicts else None,
        date_entities=table[0],
        date_keys=table[1],
        relation_primary_rows=primary_relation_rows(
            relations, tokenizer, syntax, {r.id: r.labels[0] for r in kg.relations if r.labels}
        ),
        initial_case_aliases=aliases,
        source_graph_hash=source_hash,
        build_config={
            "builder": "in-memory",
            "label_encoding": "exact-v1",
            "initial_case_aliases": initial_case_aliases,
            "verdicts": verdicts,
            "sitelinks": sitelinks is not None,
            "dates": dates is not None,
        },
        fingerprint_status="verified",
    )


def _hash_record(digest, record):
    data = json.dumps(record, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)


def _hash_array(digest, name, array):
    _hash_record(digest, [name, None if array is None else [array.dtype.str, array.shape]])
    if array is not None:
        for at in range(0, len(array), 65536):
            digest.update(np.ascontiguousarray(array[at : at + 65536]).tobytes())


def _source_graph_hash(graph: KnowledgeGraph | Path) -> str:
    """Stream every source record, including labels/triples the build excludes.

    Source order and numeric-array serialization are part of this conservative
    identity. A source directory's manifest/provenance is not graph content.
    """
    digest = hashlib.sha256()
    if isinstance(graph, KnowledgeGraph):
        for kind in ("entities", "relations"):
            _hash_record(digest, kind)
            for item in getattr(graph, kind):
                _hash_record(digest, [item.id, item.labels])
        _hash_record(digest, "triples")
        for triple in graph.triples:
            _hash_record(digest, triple)
    else:
        from xgraph.provenance import sha256_file

        for name in (
            "entity_labels.jsonl",
            "relation_labels.jsonl",
            "triples.npy",
            "sitelink_counts.npy",
            "date_values.npy",
        ):
            path = graph / name
            _hash_record(digest, [name, sha256_file(path) if path.exists() else None])
    return digest.hexdigest()


def primary_relation_rows(relations, tokenizer, syntax, canonical):
    """Resolve canonical text to existing writable rows without changing row order."""
    result = np.full(len(relations.owner_ids), -1, dtype=np.int32)
    excluded = {(e.owner_id, e.label) for e in relations.excluded}
    tokens = {
        owner: tuple(tokenizer.encode(syntax.label_prefix + canonical[r]))
        for owner, r in enumerate(relations.owner_ids)
        if r in canonical
        and canonical[r].strip()
        and not syntax.excludes(canonical[r])
        and (r, canonical[r]) not in excluded
    }
    for row, owner in enumerate(relations.owners):
        owner = int(owner)
        if result[owner] < 0 and relations.row_tokens(row) == tokens.get(owner):
            result[owner] = row
    for owner, relation in enumerate(relations.owner_ids):
        if (
            relation in canonical
            and result[owner] < 0
            and (relation, canonical[relation]) not in excluded
        ):
            raise IndexMismatch(
                f"canonical label {canonical[relation]!r} for {relation} is neither indexed "
                "nor excluded; supply relation labels from the same KG as the index"
            )
    return result


def sitelink_array(counts: Sequence[int] | np.ndarray, entities: int) -> np.ndarray:
    """Sitelink counts as a KG index holds them: int32, one per entity, none negative."""
    array = np.asarray(counts, dtype=np.int64).reshape(-1)
    if len(array) != entities:
        raise ValueError(
            f"a KG index needs one sitelink count per entity ({entities}), got {len(array)}"
        )
    if len(array) and int(array.min()) < 0:
        raise ValueError(f"sitelink counts cannot be negative, got {int(array.min())}")
    if len(array) and int(array.max()) > np.iinfo(np.int32).max:
        raise ValueError(f"a sitelink count of {int(array.max())} does not fit in int32")
    return array.astype(np.int32)


DATE_YEAR_BITS = 21
"""A packed date key (pack_date) is int32: the year in its top DATE_YEAR_BITS bits (signed),
then the month (4 bits, 0 if unstated), the day (5 bits, 0 if unstated), and the precision
(2 bits: 0 year, 1 month, 2 day). Keys sort by year, month, day, and precision."""
DATE_YEARS = (-(1 << (DATE_YEAR_BITS - 1)), (1 << (DATE_YEAR_BITS - 1)) - 1)
"""The years a key can hold; a date outside them (a deep-time date) cannot be compared."""


def pack_date(date: Date) -> int:
    """The date as a key (DATE_YEAR_BITS), every field it keeps and its precision."""
    if not DATE_YEARS[0] <= date.year <= DATE_YEARS[1]:
        raise ValueError(f"the year of {date} does not fit a date key {DATE_YEARS}")
    month, day = date.month or 0, date.day or 0
    if not (0 <= month <= 12 and 0 <= day <= 31):
        raise ValueError(f"{date!r} has no valid month and day for a date key")
    return (date.year << 11) | (month << 7) | (day << 2) | PRECISIONS.index(date.precision)


def unpack_date(key: int) -> Date:
    """The date a key (pack_date) holds."""
    month, day = (key >> 7) & 15, (key >> 2) & 31
    return Date(key >> 11, month or None, day or None, PRECISIONS[key & 3])


def _bound(keys: np.ndarray, first: bool) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.int64)
    year, month, day, precision = keys >> 11, (keys >> 7) & 15, (keys >> 2) & 31, keys & 3
    edge_month, edge_day = (1, 1) if first else (12, 31)
    month = np.where((precision == 0) | (month == 0), edge_month, month)
    day = np.where(precision == 2, day, edge_day)
    return ((year << 11) | (month << 7) | (day << 2)).astype(np.int32)


def date_lo(keys: np.ndarray) -> np.ndarray:
    """The first day each key's date covers at its precision (Date.lo), packed as a key
    of day precision bits 0, so that bounds compare as (year, month, day) tuples."""
    return _bound(keys, True)


def date_hi(keys: np.ndarray) -> np.ndarray:
    """The last day each key's date covers (Date.hi), packed as date_lo packs; a date is
    before another (Date.before) when its date_hi is below the other's date_lo."""
    return _bound(keys, False)


def date_table(dates: Mapping[int, Date]) -> tuple[np.ndarray, np.ndarray]:
    """The date table of dates by entity position: the positions, sorted, and their keys
    (int32 both). A date whose year does not fit a key (DATE_YEARS) is left out."""
    kept = sorted((e, d) for e, d in dates.items() if DATE_YEARS[0] <= d.year <= DATE_YEARS[1])
    entities = np.array([e for e, _ in kept], dtype=np.int32)
    keys = np.array([pack_date(d) for _, d in kept], dtype=np.int32)
    return entities, keys


VERDICT_NUMBERS = {"no": 1_499_999_998, "yes": 1_499_999_999}
"""The item numbers of the verdict entities: below the literal values' (xgraph.literals
FIRST_ID) and above every Wikidata item's."""
VERDICTS = {verdict: f"Q{number}" for verdict, number in VERDICT_NUMBERS.items()}
"""The value comparison's verdicts as entities, each labelled with its verdict, which an
index built with verdicts holds so that a comparison's verdict binds the answer span."""

START_RANKS = ("degree", "sitelinks")
"""How best-connected order ranks the entities a label names: by degree (then first row),
or by sitelink count, then degree and first row (the sitelink rank)."""


def check_start_rank(start_rank: str, index: KGIndex) -> str:
    """The start rank, refused when unknown or when the index lacks what it ranks by."""
    if start_rank not in START_RANKS:
        raise ValueError(f"start_rank must be one of {START_RANKS}, got {start_rank!r}")
    if start_rank == "sitelinks" and index.sitelinks is None:
        raise ValueError(
            "the sitelink rank needs a KG index with sitelink counts: write them with "
            "python -m xgraph.wikidata sitelink-counts and give them to the index with "
            "python -m xgraph.wikidata index-sitelinks (or rebuild it)"
        )
    if start_rank == "sitelinks":
        sitelink_array(index.sitelinks, len(index.labels.owner_ids))  # one per entity
    return start_rank


def check_expressions(index: KGIndex) -> None:
    """Refuse an index that expressions cannot run on: one without the date table, one built
    for a span syntax without operators, or one saved before expression syntax was excluded
    from its labels."""
    if not index.span_syntax.writes_expressions:
        raise ValueError(
            "expressions need a KG index built for a span syntax with operators and an "
            f"operand separator, not {index.span_syntax}"
        )
    if not index.expressions:
        raise ValueError(
            "the KG index was saved before labels holding expression syntax were excluded: "
            "rebuild it (python -m xgraph.wikidata index)"
        )
    if index.date_entities is None:
        raise ValueError(
            "expressions need a KG index with the date table: merge dates into the KG "
            "(python -m xgraph.literals merge-dates) and rebuild its index"
        )


class KGIndexFormatError(ValueError):
    """A saved KG index is missing files or was written in another format."""


INDEX_FORMAT = 1
"""The version of the saved KG index layout; a loader refuses any other."""

_ARRAYS = {
    "labels": ("tokens", "offsets", "owners"),
    "relations": ("tokens", "offsets", "owners"),
}
_ADJACENCY = ("adjacency_offsets", "edge_relations", "edge_objects")
SITELINKS = "sitelinks"
"""The optional sitelink counts: sitelinks.npy, named by kg-index.json's "sitelinks" field.
An index saved without them, by an xgraph from before they existed too, loads with none,
and an xgraph from before them loads an index that has them, ignoring them."""
DATES = ("date_entities", "date_keys")
"""The optional date table: date_entities.npy and date_keys.npy, named by kg-index.json's
"dates" field; like the sitelink counts, an index without it loads with none, and an
xgraph from before it ignores it. The header's "expressions" field says whether the build
excluded expression syntax from the labels (KGIndex.expressions); an index without it
loads as one that did not."""


def save_kg_index(index: KGIndex, directory: str | Path) -> None:
    """Write a KG index: kg-index.json (format, tokenizer fingerprint, ids, exclusions) and
    one .npy file per array, so that loading can memory-map the arrays.

    Serialize all arrays and metadata to distinct staged files before replacing
    any destination, with the header replaced last. This preserves mapped inputs
    when saving to the same directory or through filesystem and buffer aliases.
    Serialization failure leaves existing destination files untouched. Replacement
    is per file; concurrent readers should use separately named index directories.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".kg-index-", dir=directory) as temporary:
        staged = Path(temporary)
        _write_kg_index(index, staged)
        for path in sorted(staged.iterdir(), key=lambda p: p.name == "kg-index.json"):
            os.replace(path, directory / path.name)


def _write_kg_index(index: KGIndex, directory: Path) -> None:
    """Serialize one complete payload before the caller replaces destination files."""
    header: dict = {
        "format": INDEX_FORMAT,
        "tokenizer_fingerprint": index.tokenizer_fingerprint,
        "unwritable_triples": index.unwritable_triples,
        "span_syntax": dataclasses.asdict(index.span_syntax),
    }
    for part in _ARRAYS:
        rows: LabelIndex = getattr(index, part)
        header[part] = {
            "owner_ids": list(rows.owner_ids),
            "excluded": [[e.owner_id, e.label, e.reason] for e in rows.excluded],
        }
        for name in _ARRAYS[part]:
            np.save(directory / f"{part}.{name}.npy", getattr(rows, name))
    for name in _ADJACENCY:
        np.save(directory / f"{name}.npy", getattr(index, name))
    if index.sitelinks is not None:
        np.save(directory / f"{SITELINKS}.npy", index.sitelinks)
        header[SITELINKS] = f"{SITELINKS}.npy"
    if index.date_entities is not None:
        for name in DATES:
            np.save(directory / f"{name}.npy", getattr(index, name))
        header["dates"] = {name: f"{name}.npy" for name in DATES}
    header["expressions"] = index.expressions
    if index.relation_primary_rows is not None:
        np.save(directory / "relation_primary_rows.npy", index.relation_primary_rows)
        header["relation_primary_rows"] = "relation_primary_rows.npy"
    if index.verdicts is not None:
        header["verdicts"] = index.verdicts
    if index.initial_case_aliases is not None:
        header["initial_case_aliases"] = index.initial_case_aliases
    if index.source_graph_hash is not None:
        header["source_graph_hash"] = index.source_graph_hash
        header["build_config"] = index.build_config
    if index.fingerprint_status == "verified":
        header["base_fingerprint"] = index.base_fingerprint
    (directory / "kg-index.json").write_text(json.dumps(header) + "\n", encoding="utf-8")


def add_sitelinks(directory: str | Path, counts: Sequence[int] | np.ndarray) -> None:
    """Give a saved KG index sitelink counts (one per entity, in owner_ids order) without
    rebuilding it, replacing any it has."""
    directory = Path(directory)
    header = _header(directory)
    array = sitelink_array(counts, len(header["labels"]["owner_ids"]))
    if "base_fingerprint" in header:
        index = load_kg_index(directory)
        config = {**index.build_config, "sitelinks_update": True}
        updated = dataclasses.replace(index, sitelinks=array, build_config=config)
        header["build_config"] = config
        header["base_fingerprint"] = updated.base_fingerprint
    np.save(directory / f"{SITELINKS}.npy", array)
    header[SITELINKS] = f"{SITELINKS}.npy"
    (directory / "kg-index.json").write_text(json.dumps(header) + "\n", encoding="utf-8")


def _header(directory: Path) -> dict:
    path = directory / "kg-index.json"
    if not path.is_file():
        raise KGIndexFormatError(f"{directory} holds no kg-index.json")
    header = json.loads(path.read_text(encoding="utf-8"))
    if header.get("format") != INDEX_FORMAT:
        raise KGIndexFormatError(
            f"{directory} holds a KG index of format {header.get('format')}; "
            f"this xgraph reads format {INDEX_FORMAT}: rebuild the index"
        )
    return header


def saved_span_syntax(directory: str | Path) -> SpanSyntax:
    """The span syntax a saved KG index was built for, read from its header alone."""
    return SpanSyntax(**_header(Path(directory)).get("span_syntax", {}))


def load_kg_index(
    directory: str | Path,
    *,
    verify: bool = False,
    strict: bool = False,
    expected_fingerprint: str | None = None,
) -> KGIndex:
    """Memory-map an index and bind its recorded identity without scanning arrays.

    verify=True recomputes the full fingerprint and checks it against the record.
    strict=True (also XGRAPH_STRICT_FINGERPRINT=1) verifies and refuses legacy
    indexes. Legacy default admission retains the previous shape/tokenizer rule;
    verification can compute their content hash but has no original record to check.
    expected_fingerprint pins the recorded base identity, not file integrity.
    """
    strict = strict or os.environ.get("XGRAPH_STRICT_FINGERPRINT", "0") == "1"
    directory = Path(directory)
    header = _header(directory)
    recorded = header.get("base_fingerprint")
    if recorded is not None and (
        not isinstance(recorded, str)
        or len(recorded) != 64
        or any(c not in "0123456789abcdef" for c in recorded)
        or not isinstance(header.get("source_graph_hash"), str)
        or not isinstance(header.get("build_config"), dict)
    ):
        raise KGIndexFormatError(f"{directory}/kg-index.json: malformed base fingerprint metadata")
    if recorded is None and strict:
        raise KGIndexFormatError(
            f"{directory}: legacy index has no recorded base fingerprint; strict mode refuses it"
        )

    def array(name: str) -> np.ndarray:
        file = directory / f"{name}.npy"
        if not file.is_file():
            raise KGIndexFormatError(f"{directory} misses {file.name}")
        return np.load(file, mmap_mode="r")

    parts = {}
    for part, names in _ARRAYS.items():
        parts[part] = LabelIndex(
            **{name: array(f"{part}.{name}") for name in names},
            owner_ids=tuple(header[part]["owner_ids"]),
            tokenizer_fingerprint=header["tokenizer_fingerprint"],
            excluded=tuple(ExcludedLabel(*e) for e in header[part]["excluded"]),
        )
    index = KGIndex(
        **parts,
        **{name: array(name) for name in _ADJACENCY},
        unwritable_triples=header["unwritable_triples"],
        # An index saved before the syntax was recorded was built for the default one.
        span_syntax=SpanSyntax(**header.get("span_syntax", {})),
        sitelinks=array(SITELINKS) if header.get(SITELINKS) else None,
        **{name: array(name) if header.get("dates") else None for name in DATES},
        expressions=bool(header.get("expressions", False)),
        verdicts=header.get("verdicts"),
        relation_primary_rows=array("relation_primary_rows")
        if header.get("relation_primary_rows")
        else None,
        initial_case_aliases=header.get("initial_case_aliases"),
        source_graph_hash=header.get("source_graph_hash"),
        build_config=header.get("build_config"),
        fingerprint_status="verified" if recorded is not None else "legacy",
        _recorded_base_fingerprint=recorded,
    )
    if verify or strict:
        computed = index.compute_base_fingerprint()
        if recorded is not None and computed != recorded:
            raise KGIndexFormatError(
                f"{directory}: base fingerprint mismatch: recorded {recorded}, "
                f"computed {computed}; rebuild from the pinned KG"
            )
    if expected_fingerprint is not None and index.base_fingerprint != expected_fingerprint:
        raise IndexMismatch(
            f"{directory}: base fingerprint {index.base_fingerprint} does not match "
            f"expected {expected_fingerprint}"
        )
    return index


def kg_index_fingerprint(directory: str | Path) -> str:
    """SHA-256 of sorted saved-index filenames and their file hashes.

    Reports and temporary files are excluded. Existing indexes are read without
    modification; enabling a build option never changes the default serialization.
    """
    import hashlib

    from xgraph.provenance import sha256_file

    directory = Path(directory)
    header = _header(directory)
    names = [
        "kg-index.json",
        *(f"{p}.{n}.npy" for p, ns in _ARRAYS.items() for n in ns),
        *(f"{n}.npy" for n in _ADJACENCY),
    ]
    for key in (SITELINKS, "relation_primary_rows"):
        if header.get(key):
            names.append(header[key])
    if header.get("dates"):
        names.extend(header["dates"].values())
    files = {name: sha256_file(directory / name) for name in sorted(names)}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
