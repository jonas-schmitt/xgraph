"""Experimental surface-to-identity semantic oracle; no serving or engine masks."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType

from xgraph.kg import KGFormatError, KnowledgeGraph
from xgraph.syntax import SpanSyntax

NORMALIZATION = "nfc-ascii-v1"
ROOT_OPEN = "⟦ "
BIND_MARKER = " ⟦# "
ROOT_END = "⟧"
MAX_ROOT_BYTES = 4096
_QID = re.compile(r"Q[1-9][0-9]*\Z")
_ASCII_CASE = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
_SYNTAX = SpanSyntax()


@dataclass(frozen=True)
class BoundRoot:
    """One selected entity; no homonym rank is consulted after this decision."""

    surface: str
    qid: str
    version: str | None
    component: frozenset[str]
    graph_fingerprint: str


@dataclass(frozen=True)
class IdentityExecution:
    endpoints: frozenset[str]
    witnesses: tuple[tuple[str, ...], ...]


def normalize_surface(surface: str, mode: str = NORMALIZATION) -> str:
    """NFC, ASCII case folding, NFC; punctuation and whitespace stay exact."""
    if mode == "exact-v1":
        return surface
    if mode != NORMALIZATION:
        raise ValueError(f"unknown surface normalization {mode!r}")
    return unicodedata.normalize(
        "NFC", unicodedata.normalize("NFC", surface).translate(_ASCII_CASE)
    )


def _writable(surface: str, special_tokens: tuple[str, ...] = ()) -> bool:
    return (
        bool(surface.strip())
        and not _SYNTAX.excludes(surface)
        and not any(token in surface for token in special_tokens)
    )


def root_prefix(surface: str, qid: str) -> str:
    """Serialize a root prefix; the outer path span remains open after the ID closer."""
    if not _writable(surface) or not _QID.fullmatch(qid):
        raise ValueError(f"invalid root surface or QID: {surface!r}, {qid!r}")
    text = f"{ROOT_OPEN}{surface}{BIND_MARKER}{qid}{ROOT_END}"
    if len(text.encode("utf-8")) > MAX_ROOT_BYTES:
        raise ValueError("root prefix exceeds the byte limit")
    return text


class IdentityOracle:
    """A small reference recognizer, isolated from the production span language."""

    def __init__(
        self,
        kg: KnowledgeGraph,
        *,
        versions: Mapping[str, Iterable[tuple[str, str, str]]] | None = None,
        normalization: str = NORMALIZATION,
        special_tokens: Iterable[str] = (),
        max_candidates: int = 64,
    ) -> None:
        normalize_surface("", normalization)
        if type(max_candidates) is not int or max_candidates < 1:
            raise ValueError("max_candidates must be a positive integer")
        self.kg = kg
        self.normalization = normalization
        self.special_tokens = tuple(special_tokens)
        if any(not token for token in self.special_tokens):
            raise ValueError("special token text must not be empty")
        self.max_candidates = max_candidates
        self.versions = MappingProxyType(
            {name: tuple(tuple(edit) for edit in edits) for name, edits in (versions or {}).items()}
        )
        if any(not isinstance(name, str) or not name for name in self.versions):
            raise ValueError("a KG version needs a nonempty name")
        self.entities = frozenset(e.id for e in kg.entities)
        if len(self.entities) != len(kg.entities) or any(
            not _QID.fullmatch(e) for e in self.entities
        ):
            raise KGFormatError("entity IDs must be distinct canonical QIDs")
        relation_ids = {r.id for r in kg.relations}
        if len(relation_ids) != len(kg.relations):
            raise KGFormatError("relation IDs must be distinct")
        for s, r, o in kg.triples:
            if s not in self.entities or o not in self.entities or r not in relation_ids:
                raise KGFormatError(f"triple names an unknown entity or relation: {(s, r, o)!r}")
        owners: dict[str, set[str]] = {}
        for entity in kg.entities:
            for label in entity.labels:
                if _writable(label, self.special_tokens):
                    owners.setdefault(normalize_surface(label, normalization), set()).add(entity.id)
        self._owners = {key: frozenset(ids) for key, ids in owners.items()}
        self._writable_entities = frozenset(e for ids in self._owners.values() for e in ids)
        self.graph_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "kg": asdict(kg),
                    "versions": dict(self.versions),
                    "normalization": normalization,
                    "unicode_version": unicodedata.unidata_version,
                    "special_tokens": self.special_tokens,
                    "max_candidates": max_candidates,
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()

    def _scope(self, ids: Iterable[str] | None) -> frozenset[str]:
        scope = self.entities if ids is None else frozenset(ids)
        if not scope <= self.entities:
            raise ValueError(f"scope names unknown entity IDs: {sorted(scope - self.entities)!r}")
        return scope

    def candidates(
        self,
        surface: str,
        *,
        component: Iterable[str] | None = None,
        anchor: Iterable[str] | None = None,
    ) -> frozenset[str]:
        scope, anchored = self._scope(component), self._scope(anchor)
        if not _writable(surface, self.special_tokens):
            return frozenset()
        return self._owners.get(normalize_surface(surface, self.normalization), frozenset()) & (
            scope & anchored
        )

    def bind(
        self,
        surface: str,
        qid: str,
        *,
        version: str | None = None,
        component: Iterable[str] | None = None,
        anchor: Iterable[str] | None = None,
    ) -> BoundRoot:
        if version is not None and version not in self.versions:
            raise ValueError(f"unknown KG version {version!r}")
        scope = self._scope(component)
        candidates = self.candidates(surface, component=scope, anchor=anchor)
        if len(candidates) > self.max_candidates:
            raise ValueError(
                f"identity-set overflow: {len(candidates)} candidates; binding refused"
            )
        for candidate in candidates:
            root_prefix(surface, candidate)
        if qid not in candidates:
            raise ValueError(f"{qid!r} is not a candidate for surface {surface!r}")
        return BoundRoot(surface, qid, version, scope, self.graph_fingerprint)

    def binding_suffixes(
        self,
        surface: str,
        *,
        component: Iterable[str] | None = None,
        anchor: Iterable[str] | None = None,
    ) -> tuple[bytes, ...]:
        """The finite ID language after a complete surface and binding marker."""
        candidates = self.candidates(surface, component=component, anchor=anchor)
        if len(candidates) > self.max_candidates:
            raise ValueError(
                f"identity-set overflow: {len(candidates)} candidates; binding refused"
            )
        for candidate in candidates:
            root_prefix(surface, candidate)
        return tuple((qid + ROOT_END).encode() for qid in sorted(candidates))

    def binding_mask(
        self,
        surface: str,
        prefix: bytes,
        vocabulary: Mapping[int, bytes | None],
        *,
        component: Iterable[str] | None = None,
        anchor: Iterable[str] | None = None,
    ) -> frozenset[int]:
        """Scalar ID-field mask only; this is not a production surface-token mask."""
        suffixes = self.binding_suffixes(surface, component=component, anchor=anchor)
        return frozenset(
            token
            for token, piece in vocabulary.items()
            if piece and any(suffix.startswith(prefix + piece) for suffix in suffixes)
        )

    def read_root(
        self,
        token_bytes: Iterable[bytes],
        *,
        version: str | None = None,
        component: Iterable[str] | None = None,
        anchor: Iterable[str] | None = None,
    ) -> BoundRoot:
        """Recognize a complete root prefix across arbitrary UTF-8 token boundaries."""
        raw = bytearray()
        for piece in token_bytes:
            raw.extend(piece)
            if len(raw) > MAX_ROOT_BYTES:
                raise ValueError("root prefix exceeds the byte limit")
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError("root prefix is not valid UTF-8") from error
        if (
            not text.startswith(ROOT_OPEN)
            or not text.endswith(ROOT_END)
            or text.count(BIND_MARKER) != 1
        ):
            raise ValueError("expected a surface and an explicit QID binding")
        surface, qid = text[len(ROOT_OPEN) : -len(ROOT_END)].split(BIND_MARKER)
        return self.bind(surface, qid, version=version, component=component, anchor=anchor)

    def triples(self, version: str | None = None) -> frozenset[tuple[str, str, str]]:
        """Apply replacements before dropping unwritable edits, as KG versions do."""
        if version is not None and version not in self.versions:
            raise ValueError(f"unknown KG version {version!r}")
        edits = self.versions[version] if version is not None else ()
        pairs = {(s, r) for s, r, _ in edits}
        writable = self._writable_entities
        relations = {
            r.id
            for r in self.kg.relations
            if any(_writable(s, self.special_tokens) for s in r.labels)
        }
        return frozenset(
            (s, r, o)
            for s, r, o in (*[t for t in self.kg.triples if t[:2] not in pairs], *edits)
            if s in writable and o in writable and r in relations
        )

    def execute(self, root: BoundRoot, relations: tuple[str, ...]) -> IdentityExecution:
        """Uncapped adjacency-list replay for small graphs, with one witness per endpoint."""
        if root.graph_fingerprint != self.graph_fingerprint:
            raise ValueError("bound root belongs to another graph fingerprint")
        self.bind(root.surface, root.qid, version=root.version, component=root.component)
        witnesses = {root.qid: (root.qid,)}
        triples = sorted(self.triples(root.version))
        for relation in relations:
            reached: dict[str, tuple[str, ...]] = {}
            for subject, edge_relation, obj in triples:
                if subject in witnesses and edge_relation == relation and obj in root.component:
                    reached.setdefault(obj, (*witnesses[subject], relation, obj))
            witnesses = reached
        return IdentityExecution(
            frozenset(witnesses), tuple(witnesses[e] for e in sorted(witnesses))
        )
