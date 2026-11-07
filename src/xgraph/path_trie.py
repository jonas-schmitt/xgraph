"""GCR's bounded entity-relation paths and token trie.

The serialization is ``<PATH> entity -> relation -> entity </PATH>``:
canonical labels, every intermediate entity, and one space inside each marker.
Unlike xgraph path mode, relations do not hide intermediate entities. See the GCR
section of docs/reproducing.md for upstream details that could not be verified offline.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from heapq import nsmallest
from itertools import groupby
from typing import Any

from xgraph.toolkg import KGStore, VersionedStore

try:
    from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor as _Base
except ImportError:
    _Base = object  # type: ignore[assignment,misc]

OPENER = "<PATH>"
CLOSER = "</PATH>"
SEPARATOR = " -> "
MAX_PATHS = 100_000
REQUEST = "gcr"


@dataclass(frozen=True)
class KGPath:
    entities: tuple[int, ...]
    relations: tuple[int, ...]
    labels: tuple[str, ...]


def serialize_path(path: KGPath) -> str:
    if (
        not path.relations
        or len(path.entities) != len(path.relations) + 1
        or len(path.labels) != len(path.relations) * 2 + 1
    ):
        raise ValueError("GCR serialization needs a nonempty path with alternating labels")
    for label in path.labels:
        if not label.strip() or any(s in label for s in (OPENER, CLOSER, SEPARATOR, "\n", "\r")):
            raise ValueError(f"unwritable GCR path label {label!r} in {path.entities}")
    return OPENER + " " + SEPARATOR.join(path.labels) + " " + CLOSER


def enumerate_paths(
    store: KGStore | VersionedStore,
    starts: list[str],
    hops: int = 2,
    max_paths: int = MAX_PATHS,
    parallel_relations: str = "preserve",
) -> list[KGPath]:
    """All labelled forward walks of lengths 1..hops, including cycles.

    Canonical labels only, duplicate starts/triples removed, stable id order. An
    absent label cannot be serialized. Exceeding max_paths raises, never truncates.
    ``collapse`` retains the greatest relation id for each (subject, object), a
    deterministic reconstruction of upstream DiGraph's last-edge-wins behaviour.
    """
    return collect_paths(store, starts, hops, max_paths, parallel_relations).paths


@dataclass(frozen=True)
class PathEnumeration:
    paths: list[KGPath]
    truncated: bool


def collect_paths(
    store: KGStore | VersionedStore,
    starts: list[str],
    hops: int = 2,
    max_paths: int = MAX_PATHS,
    parallel_relations: str = "preserve",
    overflow: str = "error",
) -> PathEnumeration:
    """Retain the first max_paths walks in stable depth-first id order.

    Probe just one extra path to distinguish an exact fit from truncation. Walks
    and adjacency are streamed; no stack or cache grows with a hub's fanout.
    ``truncate`` is opt-in; the historical enumeration API still raises on overflow.
    """
    for name, value in (("hops", hops), ("max_paths", max_paths)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if parallel_relations not in ("preserve", "collapse"):
        raise ValueError(f"unknown parallel_relations {parallel_relations!r}")
    if overflow not in ("error", "truncate"):
        raise ValueError(f"unknown GCR path overflow policy {overflow!r}")
    if not isinstance(starts, list) or any(
        not isinstance(s, str) or not re.fullmatch(r"Q\d+", s) for s in starts
    ):
        raise ValueError(f"starts must be a list of QIDs, got {starts!r}")
    roots = (
        KGPath((s,), (), (label,))
        for s in sorted({int(s[1:]) for s in starts})
        if (label := store.label(s)) is not None
    )

    def children(path: KGPath) -> Iterator[KGPath]:
        edges = store.outgoing_edges(path.entities[-1], by_object=parallel_relations == "collapse")
        if parallel_relations == "collapse":
            # The last edge in each object group has the greatest relation id.
            def collapsed() -> Iterator[tuple]:
                for _, group in groupby(edges, key=lambda edge: edge[1]):
                    yield max(group)

            # Only this bounded prefix can be visited before the global path cap.
            ordered = nsmallest(max_paths + 1, collapsed())
        else:
            ordered = edges
        for prop, obj, relation, label in ordered:
            yield KGPath(
                (*path.entities, obj), (*path.relations, prop), (*path.labels, relation, label)
            )

    paths: list[KGPath] = []
    stack = [iter(roots)]
    while stack:
        path = next(stack[-1], None)
        if path is None:
            stack.pop()
            continue
        if path.relations:
            serialize_path(path)  # Reject ambiguous syntax rather than silently exclude it.
            if len(paths) >= max_paths:
                if overflow == "error":
                    raise ValueError(f"GCR paths exceed max_paths={max_paths} for starts {starts}")
                return PathEnumeration(paths, truncated=True)
            paths.append(path)
        if len(path.relations) < hops:
            stack.append(iter(children(path)))
    return PathEnumeration(paths, truncated=False)


def encode(tokenizer: Any, text: str) -> list[int]:
    """No implicit BOS/EOS, for ServedTokenizer, tokenizers, or a served HF tokenizer."""
    from xgraph.tokenizer import ServedTokenizer

    if isinstance(tokenizer, ServedTokenizer):
        return tokenizer.encode(text)
    result = tokenizer.encode(text, add_special_tokens=False)
    return list(result.ids if hasattr(result, "ids") else result)


class EmptyTrieError(ValueError):
    """The explicit fail-closed policy refuses generation over an empty trie."""


class TokenTrie:
    """A full first path, including both markers, then an unconstrained hypothesis.

    A None allowed set means free generation. An invalid nonempty prefix raises:
    a decoder must never silently fail open on a broken path. The only fail-open
    condition is an initially empty trie with empty='open', as upstream permits.
    """

    def __init__(self, sequences: Iterable[Iterable[int]], empty: str = "open") -> None:
        if empty not in ("open", "closed"):
            raise ValueError(f"empty must be 'open' or 'closed', got {empty!r}")
        self.empty = empty
        checked = []
        for sequence in sequences:
            if not isinstance(sequence, (list, tuple)):
                raise ValueError("GCR trie sequences must be lists or tuples of token ids")
            if not sequence or any(type(t) is not int or t < 0 for t in sequence):
                raise ValueError("GCR trie sequences must contain nonnegative integer token ids")
            checked.append(tuple(sequence))
        self.sequences = tuple(sorted(set(checked)))
        self.children: list[dict[int, int]] = [{}]
        self.terminal: set[int] = set()
        for sequence in self.sequences:
            node = 0
            for token in sequence:
                if node in self.terminal:
                    raise ValueError("a GCR path token sequence is a proper prefix of another")
                if token not in self.children[node]:
                    self.children[node][token] = len(self.children)
                    self.children.append({})
                node = self.children[node][token]
            if self.children[node]:
                raise ValueError("a GCR path token sequence is a proper prefix of another")
            self.terminal.add(node)

    @classmethod
    def from_paths(cls, paths: list[KGPath], tokenizer: Any, empty: str = "open") -> TokenTrie:
        # Tokenize complete strings: BPE merges may cross label/separator boundaries.
        return cls((encode(tokenizer, serialize_path(p)) for p in paths), empty)

    def allowed(self, prefix: Iterable[int]) -> frozenset[int] | None:
        if not self.sequences:
            if self.empty == "closed":
                raise EmptyTrieError("GCR empty trie with fail-closed policy")
            return None
        node = 0
        for token in prefix:
            if node in self.terminal:
                return None  # Everything after </PATH> is hypothesis text.
            if token not in self.children[node]:
                raise ValueError(f"disallowed GCR path token {token} at trie node {node}")
            node = self.children[node][token]
        return None if node in self.terminal else frozenset(self.children[node])


def trie_arguments(request_id: str, trie: TokenTrie) -> dict:
    """Serializable per-request trie; no process-local registry shared with vLLM workers."""
    return {
        REQUEST: {
            "request": request_id,
            "sequences": [list(s) for s in trie.sequences],
            "empty": trie.empty,
        }
    }


def _request_trie(params: Any) -> tuple[str, TokenTrie] | None:
    payload = (params.extra_args or {}).get(REQUEST)
    if payload is None:
        return None
    if isinstance(payload, str):
        import json

        def unique_fields(pairs):
            result = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError(f"gcr repeats {key!r}")
                result[key] = item
            return result

        try:
            payload = json.loads(payload, object_pairs_hook=unique_fields)
        except ValueError as error:
            raise ValueError(f"gcr must be a JSON object: {error}") from error
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("request"), str)
        or not payload["request"]
        or not isinstance(payload.get("sequences"), list)
        or any(not isinstance(s, list) for s in payload["sequences"])
    ):
        raise ValueError("gcr needs a request name and a list of token sequences")
    trie = TokenTrie(payload["sequences"], payload.get("empty", "open"))
    trie.allowed([])  # Refuse fail-closed empty tries before vLLM can sample all -inf.
    return payload["request"], trie


class GCRLogitsProcessor(_Base):
    """vLLM Model Runner V2 adapter for per-request CPU tries.

    Full generated histories are replayed, excluding prompts. This handles preemption,
    batch reordering and slot reuse without a mutable decode cursor. It synchronizes
    token histories to the CPU; it is a comparator, not a GPU trie implementation.
    One constrained path starts immediately; after its closer the mask is removed.
    Speculative decoding is explicitly unsupported.
    """

    def __init__(self, vllm_config: Any, req_states: Any) -> None:
        if getattr(vllm_config, "speculative_config", None) is not None:
            raise ValueError("GCR does not support speculative decoding")
        self._req_states = req_states
        self._slots: dict[int, str] = {}
        self._tries: dict[str, TokenTrie] = {}

    @classmethod
    def validate_params(cls, sampling_params: Any) -> None:
        _request_trie(sampling_params)

    def add_request(self, req_idx: int, sampling_params: Any) -> bool:
        request = _request_trie(sampling_params)
        if request is not None:
            key, trie = request
            if key in self._tries and (
                self._tries[key].sequences != trie.sequences or self._tries[key].empty != trie.empty
            ):
                raise ValueError(f"GCR request {key!r} already has a different trie")
        self._slots.pop(req_idx, None)
        if request is not None:
            self._slots[req_idx] = key
            self._tries[key] = trie
        active = set(self._slots.values())
        self._tries = {k: t for k, t in self._tries.items() if k in active}
        return request is not None

    def apply_staged_writes(self) -> None:
        pass  # Only immutable CPU tries, so there are no GPU writes to stage.

    def apply(self, logits: Any, ctx: Any) -> Any:
        if len(ctx.expanded_idx_mapping) != len(ctx.idx_mapping):
            raise ValueError("GCR does not support speculative decoding rows")
        state = self._req_states
        for row, slot in enumerate(ctx.expanded_idx_mapping.tolist()):
            if slot not in self._slots:
                continue
            start = int(state.prompt_len.gpu[slot])
            stop = int(state.total_len.gpu[slot])
            prefix = state.all_token_ids.gpu[slot, start:stop].tolist()
            allowed = self._tries[self._slots[slot]].allowed(prefix)
            if allowed is not None:
                ids = sorted(allowed)
                if not ids or ids[-1] >= logits.shape[1]:
                    raise ValueError(f"GCR request {self._slots[slot]!r} has invalid token ids")
                values = logits[row, ids].clone()
                logits[row].fill_(float("-inf"))
                logits[row, ids] = values
        return logits
