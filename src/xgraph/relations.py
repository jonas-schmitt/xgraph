"""Relation support without an entity frontier: vocabulary and KG-wide successors.

Successor tables are immutable engine-setup data, shared by every slot. They are built
from the writable index adjacency, with one table for the base KG and each registered
version. They are not saved into the tokenizer-bound KG index format.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from xgraph.index import KGIndex
from xgraph.kgversions import KGVersions

RELATIONS = ("frontier", "vocabulary", "successor")
"""Native mode codes, in order; frontier preserves the original path semantics."""

_CHUNK = 1_000_000
"""Bound edge gathers and expanded relation joins independently of KG size."""


def check_relations(mode: str, *, paths: bool, expressions: bool, stop: str | None) -> None:
    if mode not in RELATIONS:
        raise ValueError(f"unknown relations mode {mode!r}; expected one of {RELATIONS}")
    if mode != "frontier" and (not paths or expressions or stop):
        raise ValueError("weaker relations need path mode without expressions or STOP")


def relation_successors(index: KGIndex, versions: KGVersions | None = None) -> np.ndarray:
    """uint32 [1 + versions, R, ceil(R/32)]: bit r2 of row r1 witnesses an object of
    r1 that is a subject of r2. R counts relation ids, including ids without writable
    labels (their rows/bits stay zero). Table v costs 4*R*ceil(R/32) bytes. Padding bits
    are zero. Aliases contribute no extra witnesses; unwritable edits remove old ones.

    Join incoming edge counts to unique outgoing relations by intermediate entity.
    Counts include object label rows, so aliases weight a witness but cannot change
    its presence. Versions subtract and add exactly the same weights, including
    pair replacements and unwritable edits. No Python walk over base entities or
    edges is needed, and no optional sparse-matrix dependency is required.

    Beyond the result, keep one R*R count array, outgoing CSR runs, incoming counts
    for edited entities only, and bounded chunks of edges and expanded joins.
    """
    if versions is not None:
        versions.check_base(index)
    n = len(index.relations.owner_ids)
    tables = np.zeros(
        (1 + (len(versions.names) if versions else 0), n, (n + 31) // 32), dtype=np.uint32
    )
    if not n:
        return tables
    outgoing = _outgoing(index.adjacency_offsets, index.edge_relations)
    edited = np.zeros(len(index.labels.owner_ids), dtype=bool)
    if versions is not None:
        edited[versions.subjects] = True
    productive = (np.diff(outgoing[0]) > 0) | edited
    # A fixed pair has at most one witness per adjacency entry, even with aliases.
    dtype = np.int32 if len(index.edge_relations) <= np.iinfo(np.int32).max else np.int64
    counts = np.zeros(n * n, dtype=dtype)
    incoming_keys, incoming_weights = [], []
    for start in range(0, len(index.edge_relations), _CHUNK):
        stop = start + _CHUNK
        objects = index.labels.owners[index.edge_objects[start:stop]]
        keep = productive[objects]
        keys, weights = np.unique(
            objects[keep].astype(np.int64) * n + index.edge_relations[start:stop][keep],
            return_counts=True,
        )
        for pairs, values in _join(keys, weights, *outgoing, n):
            np.add.at(counts, pairs, values)
        keep = edited[keys // n]
        incoming_keys.append(keys[keep])
        incoming_weights.append(weights[keep])
    incoming = _sum_keys(incoming_keys, incoming_weights)
    packed = np.packbits(counts.reshape(n, n) > 0, axis=1, bitorder="little")
    # Explicit byte order preserves uint32 bit numbering on either host endianness.
    padded = np.zeros((n, tables.shape[2] * 4), dtype=np.uint8)
    padded[:, : packed.shape[1]] = packed
    tables[0] = padded.view("<u4")
    if versions is not None:
        for v in range(1, len(tables)):
            tables[v] = tables[0]
            keys, delta = _version_delta(index, versions, v, outgoing, incoming, n)
            live = counts[keys].astype(np.int64) + delta > 0
            rows, relations = keys // n, keys % n
            bits = np.uint32(1) << (relations & 31).astype(np.uint32)
            np.bitwise_and.at(tables[v], (rows, relations >> 5), ~bits)
            np.bitwise_or.at(tables[v], (rows[live], relations[live] >> 5), bits[live])
    return tables


def _outgoing(offsets: np.ndarray, relations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique outgoing relations per subject, preserving empty subjects and CSR order."""
    starts = np.ones(len(relations), dtype=bool)
    np.not_equal(relations[1:], relations[:-1], out=starts[1:])
    starts[offsets[:-1][offsets[1:] > offsets[:-1]]] = True
    runs = np.flatnonzero(starts)
    return np.searchsorted(runs, offsets), relations[runs]


def _ranges(offsets: np.ndarray, rows: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Expand selected CSR rows in bounded chunks: parent positions and entry positions."""
    lengths = offsets[rows + 1] - offsets[rows]
    ends = np.cumsum(lengths)
    total = int(ends[-1]) if len(ends) else 0
    for start in range(0, total, _CHUNK):
        positions = np.arange(start, min(start + _CHUNK, total))
        parents = np.searchsorted(ends, positions, side="right")
        entries = offsets[rows[parents]] + positions - (ends[parents] - lengths[parents])
        yield parents, entries


def _join(
    keys: np.ndarray,
    weights: np.ndarray,
    offsets: np.ndarray,
    relations: np.ndarray,
    n: int,
    *,
    entities: np.ndarray | None = None,
    signs: np.ndarray | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Weighted incoming (entity, r1) keys joined to outgoing r2, yielding (r1, r2)."""
    objects = keys // n
    if entities is not None:
        objects = np.searchsorted(entities, objects)
    for parents, entries in _ranges(offsets, objects):
        values = weights[parents]
        if signs is not None:
            values = values * signs[entries]
        yield (keys[parents] % n) * n + relations[entries], values


def _sum_keys(keys: list[np.ndarray], weights: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Sum signed weights of packed keys; omit exact cancellations."""
    if not keys or not any(len(k) for k in keys):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    joined = np.concatenate(keys)
    order = np.argsort(joined)
    joined = joined[order]
    values = np.concatenate(weights)[order]
    starts = np.r_[0, np.flatnonzero(joined[1:] != joined[:-1]) + 1]
    sums = np.add.reduceat(values, starts)
    keep = sums != 0
    return joined[starts[keep]], sums[keep]


def _version_delta(
    index: KGIndex,
    versions: KGVersions,
    v: int,
    outgoing: tuple[np.ndarray, np.ndarray],
    incoming: tuple[np.ndarray, np.ndarray],
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """For incoming label-edge counts I and unique outgoing indicators O, W = I.T O.

    Wv - W0 = (Iv - I0).T O0 + Iv.T (Ov - O0). The first term reads only edited
    subjects' base/effective adjacency. The second reads incoming counts only for
    edited entities whose outgoing relation set changed. Using Iv in the second
    term accounts for edits to both hops, without double counting. A bit clears
    only when the complete base count plus both terms is zero.
    """
    lo, hi = versions.offsets[v - 1 : v + 1]
    subjects = versions.subjects[lo:hi]
    keys, weights = [], []
    for _, entries in _ranges(index.adjacency_offsets, subjects):
        objects = index.labels.owners[index.edge_objects[entries]]
        k, w = np.unique(
            objects.astype(np.int64) * n + index.edge_relations[entries], return_counts=True
        )
        keys.append(k)
        weights.append(-w)
    first, last = versions.adjacency[lo], versions.adjacency[hi]
    for start in range(int(first), int(last), _CHUNK):
        stop = min(start + _CHUNK, int(last))
        objects = index.labels.owners[versions.edge_objects[start:stop]]
        k, w = np.unique(
            objects.astype(np.int64) * n + versions.edge_relations[start:stop], return_counts=True
        )
        keys.append(k)
        weights.append(w)
    edge_keys, edge_delta = _sum_keys(keys, weights)
    pair_keys, pair_weights = [], []
    for k, w in _join(edge_keys, edge_delta, *outgoing, n):
        pair_keys.append(k)
        pair_weights.append(w)

    # Unique outgoing indicators before/after, indexed by actual subject positions.
    old_keys = []
    for parents, entries in _ranges(outgoing[0], subjects):
        old_keys.append(subjects[parents].astype(np.int64) * n + outgoing[1][entries])
    offsets, relations = _outgoing(
        versions.adjacency[lo : hi + 1] - first, versions.edge_relations[first:last]
    )
    new_keys = np.repeat(subjects.astype(np.int64), np.diff(offsets)) * n + relations
    out_keys, out_delta = _sum_keys(
        [*old_keys, new_keys],
        [
            *(np.full(len(k), -1, dtype=np.int64) for k in old_keys),
            np.ones(len(new_keys), dtype=np.int64),
        ],
    )
    entities, starts = np.unique(out_keys // n, return_index=True)
    # Effective incoming counts, including new edges to previously leaf entities.
    have = np.isin(incoming[0] // n, entities)
    changed = np.isin(edge_keys // n, entities)
    effective_keys, effective_weights = _sum_keys(
        [incoming[0][have], edge_keys[changed]], [incoming[1][have], edge_delta[changed]]
    )
    for k, w in _join(
        effective_keys,
        effective_weights,
        np.r_[starts, len(out_keys)],
        out_keys % n,
        n,
        entities=entities,
        signs=out_delta,
    ):
        pair_keys.append(k)
        pair_weights.append(w)
    return _sum_keys(pair_keys, pair_weights)
