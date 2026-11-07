"""The GPU engine: advance and mask for all sequences of a decoding step, in GPU memory.

The engine keeps one constraint state per sequence on the GPU and runs two kernels
per decoding step: advance (by the sampled tokens, which may stay on the GPU) and
mask (into a device buffer of int32 words, bit i of word w standing for token
32w + i). Its semantics are the oracle's (`xgraph.oracle`); this module builds the
tables the kernels read:

- for free text and escaped spans, a byte-level automaton for the span opener and
  the span closer: for each partially matched prefix and each token, the next
  prefix, a rejection, or completion exactly at the token's end; plus the mask
  of each prefix;
- inside a span, for each byte position of the closer, the separator, and the
  escape marker, the tokens that continue it and where they end;
- the root mask: the first tokens of all label index rows;
- with unquoted starts, which rows an unbound subject may name (see quoted_rows);
- with expressions, the same per-position token lists for each operator (after the label
  prefix) and the operand separator, and the date table's entities with their dates'
  interval bounds.

Observed paths (observe=k) mirror the oracle's language and canonical tokenization.
At the observation marker the GPU selects at most k rows of the capped, versioned
frontier in best-connected order. Per-sequence memory retains these rows; the mask
reads their indexed tokens and precompiled punctuation to allow exactly one payload
token per step. No per-token host read, tokenization or vLLM round trip is needed.
Draft rows copy observation state and memory along with the existing frontier.
Release and escape bias cannot override an observation in progress. The zero mode
uploads no observation tables, allocates no row list and keeps the old public state
fields, masks and constructor call. See docs/observed-paths-adr.md.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from xgraph.index import (
    VERDICTS,
    IndexMismatch,
    KGIndex,
    LabelIndex,
    check_expressions,
    check_start_rank,
    date_hi,
    date_lo,
)
from xgraph.kgversions import KGVersions, VersionView
from xgraph.kgversions import native_tables as version_tables
from xgraph.oracle import COMPARISONS
from xgraph.relations import RELATIONS, relation_successors
from xgraph.syntax import QUOTATION_MARKS, SpanSyntax
from xgraph.tokenizer import ServedTokenizer

try:
    from xgraph import _engine
except ImportError:  # built without a CUDA compiler
    _engine = None

REJECT = -1
COMPLETE = -2
MAX_DELIMITER_BYTES = 127
"""Partial delimiter matches are stored as int8 byte counts."""
PHASES = ("free text", "span", "escaped", "error")
"""The engine's phase codes, in order; "error" marks a sequence given a token outside its mask."""
PHASE_ERROR = PHASES.index("error")
SEGMENTS = ("subject", "relation", "object")
CLOSED = (
    "none",
    "unresolved",
    "entity span",
    "fact span",
    "path span",
    "comparison",
    "relation plan",
)
"""What a sequence's last_kind field says about the span it closed most recently."""


class EngineUnavailable(RuntimeError):
    """xgraph was built without its GPU engine."""


def mask_words(vocab_size: int) -> int:
    return (vocab_size + 31) // 32


def pack_mask(allowed: np.ndarray, words: int) -> np.ndarray:
    """A boolean array over token ids as uint32 words, bit i of word w for token 32w + i."""
    bits = np.zeros(words * 32, dtype=bool)
    bits[: len(allowed)] = allowed
    return np.packbits(bits, bitorder="little").view("<u4").astype(np.uint32)


def _failure(delimiter: bytes) -> list[int]:
    """KMP failure function: for each prefix, its longest proper prefix that is also a suffix."""
    failure = [0] * len(delimiter)
    k = 0
    for i in range(1, len(delimiter)):
        while k > 0 and delimiter[i] != delimiter[k]:
            k = failure[k - 1]
        if delimiter[i] == delimiter[k]:
            k += 1
        failure[i] = k
    return failure


def delimiter_automaton(
    delimiter: bytes,
    token_bytes: list[bytes | None],
    special_ids: frozenset[int],
    specials_allowed: bool,
    words: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Next-prefix codes [len(delimiter), vocab] (int8) and masks [len(delimiter), words].

    A token completing the delimiter before its own end is rejected. Special
    tokens reset the prefix when allowed; unused ids are always rejected.
    """
    failure = _failure(delimiter)
    states = len(delimiter)
    codes = np.full((states, len(token_bytes)), REJECT, dtype=np.int8)
    for token, data in enumerate(token_bytes):
        if data is None:
            if token in special_ids and specials_allowed:
                codes[:, token] = 0
            continue
        for start in range(states):
            matched = start
            for at, byte in enumerate(data):
                while matched > 0 and delimiter[matched] != byte:
                    matched = failure[matched - 1]
                if delimiter[matched] == byte:
                    matched += 1
                if matched == states:
                    codes[start, token] = COMPLETE if at == len(data) - 1 else REJECT
                    break
            else:
                codes[start, token] = matched
    masks = np.stack([pack_mask(codes[s] != REJECT, words) for s in range(states)])
    return codes, masks


def delimiter_tokens(
    delimiter: bytes, token_bytes: list[bytes | None]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each byte position, the tokens continuing the delimiter from there (CSR)."""
    by_position: list[list[tuple[int, int]]] = [[] for _ in delimiter]
    for token, data in enumerate(token_bytes):
        if not data:
            continue
        for pos in range(len(delimiter)):
            if delimiter.startswith(data, pos):
                by_position[pos].append((token, pos + len(data)))
    offsets = np.zeros(len(delimiter) + 1, dtype=np.int32)
    np.cumsum([len(entries) for entries in by_position], out=offsets[1:])
    flat = [entry for entries in by_position for entry in entries]
    tokens = np.array([t for t, _ in flat], dtype=np.int32)
    next_pos = np.array([p for _, p in flat], dtype=np.int32)
    return offsets, tokens, next_pos


def root_mask(
    index: LabelIndex, vocab_size: int, words: int, kept: np.ndarray | None = None
) -> np.ndarray:
    """The first tokens of the label index rows (only the kept ones, if given)."""
    first = np.zeros(vocab_size, dtype=bool)
    rows = index.offsets[1:] > index.offsets[:-1]
    if kept is not None:
        rows &= kept
    first[index.tokens[index.offsets[:-1][rows]]] = True
    return pack_mask(first, words)


def quoted_rows(
    labels: LabelIndex, token_bytes: list[bytes | None], syntax: SpanSyntax
) -> np.ndarray:
    """For each label index row, whether its label starts with a quotation mark
    (SpanSyntax.quoted), which unquoted starts keep an unbound subject from naming.

    A row is decided by its first bytes, the label prefix and the longest mark; rows are
    grouped by their first k tokens, k = 1, 2, ..., so that each distinct start is decoded
    once, until every row is decided."""
    offsets = labels.offsets.astype(np.int64)
    tokens = np.asarray(labels.tokens)
    lengths = np.diff(offsets)
    quoted = np.zeros(len(lengths), dtype=bool)
    need = len(syntax.label_prefix.encode("utf-8")) + max(
        len(mark.encode("utf-8")) for mark in QUOTATION_MARKS
    )
    rows = np.flatnonzero(lengths > 0)
    k = 1
    while len(rows):
        starts = np.stack([tokens[offsets[rows] + j] for j in range(k)], axis=1)
        distinct, inverse = np.unique(starts, axis=0, return_inverse=True)
        written = [b"".join(token_bytes[t] or b"" for t in start) for start in distinct.tolist()]
        decided_start = np.array([len(w) >= need for w in written], dtype=bool)
        quoted_start = np.array([syntax.quoted(w) for w in written], dtype=bool)
        inverse = inverse.reshape(-1)
        decided = decided_start[inverse] | (lengths[rows] == k)
        quoted[rows[decided]] = quoted_start[inverse][decided]
        rows = rows[~decided]
        k += 1
    return quoted


def _shared_prefix(a: bytes, b: bytes) -> int:
    """How many leading bytes a and b share."""
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def unbound_rows(
    index: KGIndex, token_bytes: list[bytes | None], syntax: SpanSyntax, unquoted_starts: bool
) -> np.ndarray | None:
    """For each label index row, whether an unbound subject may name it: not a row of a
    verdict entity (KGIndex.verdicts), nor with unquoted starts a quoted row (quoted_rows);
    None when every row may be named."""
    if not unquoted_starts and not index.verdicts:
        return None
    kept = np.ones(index.labels.num_rows, dtype=bool)
    if unquoted_starts:
        kept &= ~quoted_rows(index.labels, token_bytes, syntax)
    if index.verdicts:
        verdicts = [index.labels.owner_ids.index(e) for e in index.verdicts.values()]
        kept &= ~np.isin(np.asarray(index.labels.owners), verdicts)
    return kept


def relation_rows(index: KGIndex) -> tuple[np.ndarray, np.ndarray]:
    """For each relation, its relation index rows (CSR): offsets and rows."""
    owners = index.relations.owners
    order = np.argsort(owners, kind="stable").astype(np.int32)
    offsets = np.zeros(len(index.relations.owner_ids) + 1, dtype=np.int32)
    np.cumsum(np.bincount(owners, minlength=len(index.relations.owner_ids)), out=offsets[1:])
    return offsets, order


WIDE_ROWS = 4096
MAX_WIDE_DEPTH = 4  # below 8: a key is the range start * 8 + depth, here and on the GPU
"""Label prefixes of one to MAX_WIDE_DEPTH tokens with at least this many rows get a precomputed
mask of their next tokens, so that the GPU does not scan their rows on every step."""


def wide_masks(
    labels: LabelIndex, words: int, wide: int = WIDE_ROWS, kept: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Keys (row range start * 8 + depth) and next-token masks of the wide label prefixes.

    At depth d (1 to MAX_WIDE_DEPTH), a prefix's rows are those whose first d tokens equal it, a
    contiguous range in label index order; its mask holds the tokens at depth d of
    those rows (labels that end there add none), or only of the kept ones, if given: only
    unbound subjects read these masks, and with unquoted starts they keep the unquoted rows.
    """
    if not 0 < MAX_WIDE_DEPTH < 8:
        raise ValueError(f"MAX_WIDE_DEPTH must be 1 to 7, got {MAX_WIDE_DEPTH}")
    offsets = labels.offsets.astype(np.int64)
    tokens = np.asarray(labels.tokens)
    lengths = np.diff(offsets)
    if not len(lengths):
        return np.zeros(0, dtype=np.int64), np.zeros((0, words), dtype=np.uint32)

    def at(depth: int) -> np.ndarray:
        found = np.full(len(lengths), -1, dtype=np.int64)
        has = lengths > depth
        found[has] = tokens[offsets[:-1][has] + depth]
        return found

    columns = [at(d) for d in range(MAX_WIDE_DEPTH + 1)]
    keys, masks = [], []
    for depth in range(1, MAX_WIDE_DEPTH + 1):
        prefix = np.stack(columns[:depth], axis=1)
        valid = (prefix >= 0).all(axis=1)
        change = np.ones(len(prefix), dtype=bool)
        change[1:] = (prefix[1:] != prefix[:-1]).any(axis=1)
        starts = np.flatnonzero(change)
        stops = np.append(starts[1:], len(prefix))
        for lo, hi in zip(starts.tolist(), stops.tolist(), strict=True):
            if hi - lo < wide or not valid[lo]:
                continue
            following = columns[depth][lo:hi]
            if kept is not None:
                following = following[kept[lo:hi]]
            following = np.unique(following)
            following = following[following >= 0]
            mask = np.zeros(words, dtype=np.uint32)
            np.bitwise_or.at(mask, following >> 5, (1 << (following & 31)).astype(np.uint32))
            keys.append(lo * 8 + depth)
            masks.append(mask)
    order = np.argsort(np.array(keys, dtype=np.int64))
    return (
        np.array(keys, dtype=np.int64)[order],
        np.array(masks, dtype=np.uint32).reshape(-1, words)[order],
    )


def entity_rows(labels: LabelIndex) -> dict[str, np.ndarray]:
    """Each entity's label index rows, in row order: offsets [entities + 1] into rows."""
    owners = np.asarray(labels.owners, dtype=np.int64)
    count = len(labels.owner_ids)
    order = np.argsort(owners, kind="stable").astype(np.int32)
    offsets = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(np.bincount(owners, minlength=count), out=offsets[1:])
    return {"entity_row_offsets": offsets, "entity_rows": order}


def native_tables(
    index: KGIndex,
    tokenizer: ServedTokenizer,
    syntax: SpanSyntax,
    escapes: bool = True,
    versions: KGVersions | None = None,
    unquoted_starts: bool = False,
    expressions: bool = False,
    stop: bool = False,
    relations: str = "frontier",
    eos_token_ids: Iterable[int] = (),
    observe: int = 0,
    observe_select: bool = False,
    observe_answer: bool = False,
    observe_no_loop: bool = False,
    audit: bool = False,
    menu: int = 0,
    relation_labels=None,
    relation_display=None,
    menu_close_next: bool = False,
    observe_defer: int = 0,
    observe_terminal: int = 0,
    request_path_options: bool = False,
    terminal_order: str = "identity",
    terminal_seed: int = 17,
) -> dict:
    """The named arrays the native engine copies to the GPU.

    Without escapes, the escape marker's automaton lists no token at any position, so the
    marker can never start and every span must name the KG. The KG versions' tables
    (xgraph.kgversions) are empty without versions, and entity_sitelinks (the index's
    sitelink counts) without sitelink counts. With unquoted starts, unquoted_rows counts
    the rows before each row that an unbound subject may name (those that do not start
    with a quotation mark), and the root mask and the wide masks hold only their tokens;
    without, unquoted_rows is empty. With expressions, operator<k>_* list the tokens of each
    operator (after the label prefix), operator_lcp the leading bytes each pair of them
    shares, operand_* the operand separator's, and date_entities, date_lo, and date_hi the
    date table with its dates' interval bounds; without, there are no operators and no dates.
    With stop, stop_label_offsets and stop_label_words hold normalized raw words of
    each start label, shared by all requests; otherwise both are empty.
    relation_successors is a packed bitmap per KG version only with successor support;
    frontier and vocabulary support hold an empty array.
    """
    opener, separator, closer, escape = (
        d.encode("utf-8") for d in (syntax.opener, syntax.separator, syntax.closer, syntax.escape)
    )
    vocab = tokenizer.vocab_size
    words = mask_words(vocab)
    token_bytes = [tokenizer.token_bytes(t) for t in range(vocab)]
    special = tokenizer.special_ids
    free_next, free_mask = delimiter_automaton(opener, token_bytes, special, True, words)
    esc_next, esc_mask = delimiter_automaton(closer, token_bytes, special, False, words)
    relation_row_offsets, relation_row_list = relation_rows(index)
    kept = unbound_rows(index, token_bytes, syntax, unquoted_starts)
    unquoted = np.zeros(0 if kept is None else len(kept) + 1, dtype=np.int32)
    if kept is not None:
        np.cumsum(kept, out=unquoted[1:])
    tables = {
        "row_tokens": index.labels.tokens,
        "row_offsets": index.labels.offsets,
        "row_entities": index.labels.owners,
        "relation_tokens": index.relations.tokens,
        "relation_offsets": index.relations.offsets,
        "relation_owners": index.relations.owners,
        "relation_row_offsets": relation_row_offsets,
        "relation_rows": relation_row_list,
        "adjacency_offsets": index.adjacency_offsets,
        "edge_relations": index.edge_relations,
        "edge_objects": index.edge_objects,
        "root_mask": root_mask(index.labels, vocab, words, kept),
        "unquoted_rows": unquoted,
        **entity_rows(index.labels),
        "wide_keys": (wide := wide_masks(index.labels, words, WIDE_ROWS, kept))[0],
        "wide_masks": wide[1],
        "free_next": free_next,
        "free_mask": free_mask,
        "eos_mask": pack_mask(np.isin(np.arange(vocab), list(eos_token_ids)), words),
        "esc_next": esc_next,
        "esc_mask": esc_mask,
        **version_tables(versions),
        "entity_sitelinks": (
            np.zeros(0, dtype=np.int32) if index.sitelinks is None else index.sitelinks
        ),
        "relation_successors": (
            relation_successors(index, versions)
            if relations == "successor"
            else np.zeros(0, dtype=np.uint32)
        ),
    }
    if observe_no_loop:
        # Equal complete labels form contiguous groups, including shared aliases.
        groups = np.empty((index.relations.num_rows, 2), dtype=np.int32)
        lo = 0
        while lo < len(groups):
            hi = lo + 1
            while hi < len(groups) and index.relations.row_tokens(hi) == index.relations.row_tokens(
                lo
            ):
                hi += 1
            groups[lo:hi] = (lo, hi)
            lo = hi
        tables["loop_label_groups"] = groups.ravel()
    if observe:
        from xgraph.observations import observation_pieces

        pieces = observation_pieces(tokenizer, syntax)
        tables["observation_tokens"] = np.array([t for p in pieces for t in p], dtype=np.int32)
        tables["observation_offsets"] = np.array(
            [0, *np.cumsum([len(p) for p in pieces])], dtype=np.int32
        )
    if audit or menu:
        from xgraph.audit import RelationScores

        display = (
            relation_display
            if relation_display is not None
            else RelationScores(index, tokenizer, relation_labels)
        )
        tables["audit_primary_rows"] = np.asarray(display.primary, dtype=np.int32)
        tables["audit_default_alias_ranks"] = display.default_alias_ranks
    if menu:
        from xgraph.menus import menu_marker, menu_pieces
        from xgraph.observations import canonical_tokens

        pieces = menu_pieces(tokenizer, syntax, len(index.relations.owner_ids))
        tables["audit_tokens"] = np.array([t for p in pieces for t in p], dtype=np.int32)
        tables["audit_offsets"] = np.array(
            [0, *np.cumsum([len(p) for p in pieces])], dtype=np.int32
        )
        if not observe:
            marker = canonical_tokens(tokenizer, [menu_marker(syntax)])
            tables["observation_tokens"] = np.array(marker, dtype=np.int32)
            tables["observation_offsets"] = np.array([0, len(marker)], dtype=np.int32)
        elif request_path_options:
            marker = canonical_tokens(tokenizer, [menu_marker(syntax)])
            tables["request_menu_tokens"] = np.array(marker, dtype=np.int32)
            tables["request_menu_offsets"] = np.array([0, len(marker)], dtype=np.int32)
    if observe_select or observe_answer:
        from xgraph.observations import answer_tokens, selection_tokens

        if observe_select:
            tables["selection_tokens"] = np.array(
                selection_tokens(tokenizer, syntax), dtype=np.int32
            )
        if observe_answer:
            tables["answer_tokens"] = np.array(answer_tokens(tokenizer, syntax), dtype=np.int32)
    if audit:
        from xgraph.audit import audit_commands, audit_pieces
        from xgraph.observations import answer_tokens

        for name, pieces in (
            ("audit", audit_pieces(tokenizer, syntax, len(index.relations.owner_ids))),
            ("audit_command", audit_commands(tokenizer, syntax)),
        ):
            tables[name + "_tokens"] = np.array([t for p in pieces for t in p], dtype=np.int32)
            tables[name + "_offsets"] = np.array(
                [0, *np.cumsum([len(p) for p in pieces])], dtype=np.int32
            )
        tables["answer_tokens"] = np.array(answer_tokens(tokenizer, syntax), dtype=np.int32)
    if observe_terminal:
        from xgraph.audit import terminal_pieces
        from xgraph.observations import answer_tokens

        pieces = terminal_pieces(tokenizer, syntax)
        prefix = "request_terminal" if request_path_options and menu else "audit"
        tables[prefix + "_tokens"] = np.array([t for p in pieces for t in p], dtype=np.int32)
        tables[prefix + "_offsets"] = np.array(
            [0, *np.cumsum([len(p) for p in pieces])], dtype=np.int32
        )
        tables["answer_tokens"] = np.array(answer_tokens(tokenizer, syntax), dtype=np.int32)
        tables["terminal_order_config"] = np.array(
            [("identity", "reversed", "random").index(terminal_order), terminal_seed],
            dtype=np.uint32,
        )
    if menu_close_next:
        from xgraph.observations import decision_pieces

        pieces = decision_pieces(tokenizer, syntax)
        tables["decision_tokens"] = np.array([t for p in pieces for t in p], dtype=np.int32)
        tables["decision_offsets"] = np.array(
            [0, *np.cumsum([len(p) for p in pieces])], dtype=np.int32
        )
    if observe_defer:
        from xgraph.observations import defer_tokens

        tables["defer_tokens"] = np.array(defer_tokens(tokenizer, syntax), dtype=np.int32)
    for name, delimiter in (("closer", closer), ("separator", separator), ("escape", escape)):
        offsets, tokens, next_pos = delimiter_tokens(delimiter, token_bytes)
        tables[f"{name}_offsets"], tables[f"{name}_tokens"], tables[f"{name}_next"] = (
            offsets,
            tokens,
            next_pos,
        )
    operators = [
        (syntax.label_prefix + op).encode("utf-8")
        # A value comparison (the third operator) needs the verdict entities to bind.
        for op in (syntax.operators if index.verdicts else syntax.operators[:2])
        if expressions
    ]
    for k, operator in enumerate(operators):
        offsets, tokens, next_pos = delimiter_tokens(operator, token_bytes)
        tables[f"operator{k}_offsets"], tables[f"operator{k}_tokens"] = offsets, tokens
        tables[f"operator{k}_next"] = next_pos
    tables["operator_lcp"] = np.array(
        [[_shared_prefix(a, b) for b in operators] for a in operators],
        dtype=np.int32,
    ).reshape(-1)
    offsets, tokens, next_pos = delimiter_tokens(
        syntax.operand_separator.encode("utf-8"), token_bytes
    )
    tables["operand_offsets"], tables["operand_tokens"], tables["operand_next"] = (
        offsets,
        tokens,
        next_pos,
    )
    keys = index.date_keys if expressions else None
    empty = np.zeros(0, dtype=np.int32)
    tables["date_entities"] = empty if keys is None else index.date_entities
    tables["date_lo"] = empty if keys is None else date_lo(keys)
    tables["date_hi"] = empty if keys is None else date_hi(keys)
    tables["date_keys"] = empty if keys is None else keys
    verdicts = [] if not index.verdicts else [index.verdicts["yes"], index.verdicts["no"]]
    positions = {e: n for n, e in enumerate(index.labels.owner_ids) if e in verdicts}
    tables["verdict_entities"] = np.array([positions[e] for e in verdicts], dtype=np.int32)
    # STOP removes exact raw words of the written start, not all words with their
    # stems. Normalize label words once on the host, shared by every request.
    tables["stop_label_offsets"] = np.zeros(0, dtype=np.int64)
    tables["stop_label_words"] = np.zeros(0, dtype=np.uint8)
    if stop:
        import re

        offsets, written = [0], bytearray()
        for row in range(index.labels.num_rows):
            raw = b"".join(token_bytes[t] for t in index.labels.row_tokens(row))
            name = raw.decode("utf-8", "replace").removeprefix(syntax.label_prefix)
            written.extend(
                b"\0".join(w.encode() for w in re.findall(r"\w+", name.casefold())) + b"\0"
            )
            offsets.append(len(written))
        tables["stop_label_offsets"] = np.array(offsets, dtype=np.int64)
        tables["stop_label_words"] = np.frombuffer(written, dtype=np.uint8)
    if not escapes:
        tables["escape_offsets"] = np.zeros(len(escape) + 1, dtype=np.int32)
        tables["escape_tokens"] = np.zeros(0, dtype=np.int32)
        tables["escape_next"] = np.zeros(0, dtype=np.int32)
    return {name: np.ascontiguousarray(array) for name, array in tables.items()}


class Engine:
    """Shared KG tables and independent constraint slots resident on the GPU.

    Args:
        index: Tokenizer-bound KG index; tables are copied to device memory.
        tokenizer: The exact served tokenizer recorded by the index.
        syntax: Span syntax used at index build time.
        max_sequences: Number of allocated request slots.
        versions: Optional compiled KG versions from ``build_kg_versions``.

    Other options select the same span language and execution policies as the
    CPU oracle, with additional logit-based release and escape-bias policies.
    Construction raises :class:`EngineUnavailable` without a native CUDA build.
    Importing this class alone does not allocate device memory.

    Call :meth:`begin` before assigning a slot's anchor/version and decoding.
    :meth:`sync` consumes newly committed history, then :meth:`apply_mask`
    masks float32 GPU logits in place. Every reused slot needs a fresh begin.
    Tensor arguments use DLPack; their device, shape, lifetime and stream must
    satisfy each method's contract. Host-copy methods are diagnostic operations
    that wait for the device. See the integration guide for the full lifecycle.
    """

    def __init__(
        self,
        index: KGIndex,
        tokenizer: ServedTokenizer,
        syntax: SpanSyntax,
        max_sequences: int,
        *,
        escapes: bool = True,
        repeats: bool = True,
        release_mass: float = 0.0,
        release_top: bool = False,
        derivation: bool = False,
        escape_bias: float = 0.0,
        paths: bool = False,
        bound_escapes: bool = True,
        committed_paths: bool = False,
        versions: KGVersions | None = None,
        start_rank: str = "degree",
        unquoted_starts: bool = False,
        expressions: bool = False,
        stop: str | None = None,
        relations: str = "frontier",
        end_after_answer: bool = False,
        eos_token_ids: Iterable[int] = (),
        observe: int = 0,
        observe_width: int = 0,
        observe_defer: int = 0,
        menu_close_next: bool = False,
        observe_select: bool = False,
        observe_answer: bool = False,
        observe_required: bool = False,
        observe_no_loop: bool = False,
        audit: bool = False,
        audit_entities: int = 20,
        audit_relations: int = 8,
        menu: int = 0,
        relation_labels=None,
        observe_terminal: int = 0,
        terminal_order: str = "identity",
        terminal_seed: int = 17,
        request_path_options: bool = False,
    ) -> None:
        """With escapes False the escape marker is never allowed: every span names the KG.
        With repeats False a fact span cannot close with the labels of a fact span its
        sequence closed before (among the first MAX_FACTS), as in the oracle.

        Release, in apply_mask only: inside a span, a row whose model puts less than
        release_mass of its probability (over the served vocabulary, at the logits' own
        temperature) on the allowed set, or with release_top one whose own top token is
        disallowed, may write only the escape marker (its next tokens, once started).
        Release is per step: the next step is masked as usual unless it is released too.

        With derivation True spans are bound by entity, as in the oracle: after a fact
        span, spans may only name the object rows it reached (the first MAX_REACHED), and
        a fact span's subject is the best-connected entity it may name.

        escape_bias (apply_mask only) is added to the logits of the tokens that commit to
        the escape marker inside a span (they start it and cannot continue the label): a
        negative bias makes escaping costlier without forbidding it.

        With paths True a span is a path of relations (implies derivation mode's binding),
        as in the oracle; with bound_escapes False a bound subject cannot escape, and with
        committed_paths True a path resolves to what its best start entity reaches.

        versions are KG versions of this index (xgraph.kgversions) that a slot can read
        instead of the base KG (set_version).

        start_rank "sitelinks" puts the entities with more sitelinks first in best-connected
        order (the sitelink rank), as in the oracle; it needs an index with sitelink counts.

        With unquoted_starts True an unbound subject cannot name a label that starts with a
        quotation mark (syntax.QUOTATION_MARKS), as in the oracle.

        With expressions True (committed paths, escapes, and an index with the date table:
        index.check_expressions) a span may be a date comparison, as in the oracle.

        With stop="residual-goal" (committed paths), set_stop installs each request's
        question-derived STOP relations. At qualifying path boundaries only the closer
        delimiter is admitted; longer labels remain writable. Off by default.

        relations (path mode) selects frontier execution (the default), the global
        vocabulary, or KG-version relation successors. The latter two execute no path
        and bind no answer. STOP and expressions require frontier support.

        With end_after_answer True (path mode), a normally closed answer span admits
        only the supplied served-model eos_token_ids, as in the oracle.

        With audit True, the blind path closes into a canonical terminal readback,
        one masked decision and a bound answer with EOS. set_audit compiles each
        request's relation ranks once; all subsequent execution stays on the GPU.
        audit_entities and audit_relations bound only the display, not support."""
        if _engine is None:
            raise EngineUnavailable(
                "xgraph was built without its GPU engine: no CUDA compiler was found at build time"
            )
        if index.tokenizer_fingerprint != tokenizer.fingerprint:
            raise IndexMismatch(
                f"the KG index was built for tokenizer {index.tokenizer_fingerprint[:12]}, "
                f"not the served tokenizer {tokenizer.fingerprint[:12]}"
            )
        if index.span_syntax != syntax:
            raise IndexMismatch(
                f"the KG index was built for span syntax {index.span_syntax}, not {syntax}"
            )
        if not (escapes or repeats):
            raise ValueError("repeats can only be switched off with escapes, the way out")
        if committed_paths and not paths:
            raise ValueError("committed_paths needs path mode (paths=True)")
        if end_after_answer and not paths:
            raise ValueError("end_after_answer needs path mode (paths=True)")
        from xgraph.menus import check_menu

        check_menu(menu, paths=paths, relations=relations, expressions=expressions, audit=audit)
        self._menu = menu
        from xgraph.host_policy import normalize_host_policy

        policy = normalize_host_policy(
            tokenizer,
            paths=paths,
            relations=relations,
            expressions=expressions,
            stop=stop,
            end_after_answer=end_after_answer,
            eos_token_ids=eos_token_ids,
            observe=observe,
            observe_width=observe_width,
            observe_defer=observe_defer,
            menu_close_next=menu_close_next,
            observe_select=observe_select,
            observe_answer=observe_answer,
            observe_required=observe_required,
            observe_no_loop=observe_no_loop,
            audit=audit,
            audit_entities=audit_entities,
            audit_relations=audit_relations,
            menu=menu,
            observe_terminal=observe_terminal,
            terminal_order=terminal_order,
            terminal_seed=terminal_seed,
            request_path_options=request_path_options,
        )
        end_after_answer, observe_select, observe_answer = (
            policy.end_after_answer,
            policy.observe_select,
            policy.observe_answer,
        )
        eos = policy.eos_token_ids
        self._observe_terminal = observe_terminal
        from xgraph.audit import RelationScores

        if (audit or observe_terminal) and index.sitelinks is None:
            raise ValueError("audit/terminal display needs an index with sitelink counts")
        self._audit = audit
        self._audit_terms = (
            RelationScores(index, tokenizer, relation_labels) if audit or menu else None
        )
        self._observe_no_loop = observe_no_loop
        self._observe = observe
        self._observe_defer, self._menu_close_next = observe_defer, menu_close_next
        self._observe_select, self._observe_answer = observe_select, observe_answer
        self._request_path_options = request_path_options
        self._path_capabilities = {
            "menu": menu, "observe": observe, "observe_select": observe_select,
            "observe_answer": observe_answer, "observe_required": observe_required,
            "observe_no_loop": observe_no_loop, "observe_width": observe_width,
            "observe_defer": observe_defer, "menu_close_next": menu_close_next,
            "observe_terminal": observe_terminal,
        }  # fmt: skip
        if request_path_options and (not paths or relations != "frontier" or audit or expressions):
            raise ValueError("request_path_options needs frontier paths with audit/expressions off")
        if stop not in (None, "residual-goal"):
            raise ValueError(f"unknown STOP rule {stop!r}; expected residual-goal")
        if stop and not committed_paths:
            raise ValueError("residual-goal STOP needs committed paths (committed_paths=True)")
        sitelinks_rank = check_start_rank(start_rank, index) == "sitelinks"
        if expressions and not committed_paths:
            raise ValueError("expressions need committed paths (committed_paths=True)")
        if expressions and not escapes:
            raise ValueError("expressions need escapes, the way out of an operand or relation")
        if expressions:
            check_expressions(index)
            if len(syntax.operators) > len(COMPARISONS):
                raise ValueError(f"expressions know the operators of {COMPARISONS}, by position")
        if not escapes and (release_mass > 0 or release_top):
            raise ValueError("release needs escapes: it releases to the escape marker")
        if not 0.0 <= release_mass < 1.0:
            raise ValueError(f"release_mass must be in [0, 1), got {release_mass}")
        if max_sequences < 1:
            raise ValueError(f"max_sequences must be at least 1, got {max_sequences}")
        written = {
            name: getattr(syntax, name)
            for name in ("opener", "separator", "closer", "escape", "operand_separator")
        }
        written.update({f"operator {op!r}": syntax.label_prefix + op for op in syntax.operators})
        for name, text in written.items():
            delimiter = text.encode("utf-8")
            if len(delimiter) > MAX_DELIMITER_BYTES:
                raise ValueError(
                    f"the span {name} is {len(delimiter)} bytes in UTF-8; "
                    f"the engine supports at most {MAX_DELIMITER_BYTES}"
                )
        if versions is not None:
            versions.check_base(index)
        self._index = index
        self._versions = versions
        self._entity_index: dict[str, int] | None = None  # built on the first anchor
        self._stop_terms = None
        if stop:
            from xgraph.stopping import RelationTerms

            self._stop_terms = RelationTerms.from_index(index, tokenizer)
        self._native = _engine.Engine(
            native_tables(
                index,
                tokenizer,
                syntax,
                escapes,
                versions,
                unquoted_starts,
                expressions,
                bool(stop),
                relations,
                eos,
                observe,
                observe_select,
                observe_answer,
                observe_no_loop,
                audit,
                menu,
                relation_labels,
                self._audit_terms,
                menu_close_next,
                observe_defer,
                observe_terminal,
                **({"request_path_options": True} if request_path_options else {}),
                terminal_order=terminal_order,
                terminal_seed=terminal_seed,
            ),
            tokenizer.vocab_size,
            mask_words(tokenizer.vocab_size),
            max_sequences,
            repeats,
            math.log(release_mass) if release_mass > 0 else -math.inf,
            release_top,
            derivation or paths,
            escape_bias,
            paths,
            bound_escapes,
            committed_paths,
            sitelinks_rank,
            unquoted_starts or bool(index.verdicts),  # unbound subjects skip some rows
            expressions,
            end_after_answer,
            RELATIONS.index(relations),
            **({"observe": observe} if observe else {}),
            **({"observe_width": observe_width} if observe_width else {}),
            **({"observe_defer": observe_defer} if observe_defer else {}),
            **({"menu_close_next": True} if menu_close_next else {}),
            **({"observe_select": True} if observe_select else {}),
            **({"observe_answer": True} if observe_answer else {}),
            **({"observe_terminal": observe_terminal} if observe_terminal else {}),
            **({"observe_required": True} if observe_required else {}),
            **({"observe_no_loop": True} if observe_no_loop else {}),
            **({"menu": menu} if menu else {}),
            **({"request_path_options": True} if request_path_options else {}),
            **(
                {
                    "audit": True,
                    "audit_entities": audit_entities,
                    "audit_relations": audit_relations,
                }
                if audit
                else {}
            ),
        )

    def set_path_options(self, slot: int, options: dict, stream: int = 0) -> None:
        """Install a complete view after begin, before opening/replaying the request.

        Values cannot exceed the constructor's allocated capabilities. begin and
        reset restore constructor defaults; drafts read the slot's immutable view.
        The underlying path language, KG tables and end-after-answer stay fixed.
        """
        from xgraph.observations import PATH_OPTION_FIELDS, checked_path_options

        if not self._request_path_options:
            raise ValueError("set_path_options needs request_path_options=True")
        options = checked_path_options(options, capabilities=self._path_capabilities)
        self._native.set_path_options(
            slot, np.array([options[k] for k in PATH_OPTION_FIELDS], dtype=np.int32), stream
        )

    def set_audit(self, slot: int, question: str, stream: int = 0) -> None:
        """Compile the request's lexical relation scores once, after begin."""
        if not (self._audit or self._menu):
            raise ValueError("audit is not enabled")
        self.set_relation_question(slot, question, stream)

    def set_relation_question(
        self, slot: int, question: str, stream: int = 0, *, ranked=None
    ) -> None:
        """Shared question ranking for terminal audits and per-step relation menus."""
        if self._audit_terms is None:
            raise ValueError("audit or menu is not enabled")
        self.set_relation_scores(
            slot, self._audit_terms.for_question(question) if ranked is None else ranked, stream
        )
        self._native.set_alias_ranks(slot, self._audit_terms.alias_ranks(question), stream)

    def set_relation_scores(self, slot: int, ranked, stream: int = 0) -> None:
        """Install up to 64 (relation id, rank) pairs. Lower ranks precede others.

        Equal ranks use frontier frequency then index id. begin/reset clears the
        list; speculative rows read the immutable real slot's list.
        """
        from xgraph.audit import checked_relation_ranks

        if not (self._audit or self._menu):
            raise ValueError("audit is not enabled")
        ids = {r: n for n, r in enumerate(self._index.relations.owner_ids)}
        ranked = checked_relation_ranks(list(ranked), ids)
        packed = np.array([(ids[r], rank) for r, rank in ranked], dtype=np.int32).ravel()
        self._native.set_relation_scores(slot, packed, stream)

    @property
    def max_sequences(self) -> int:
        return self._native.max_sequences

    def reset(self, stream: int = 0) -> None:
        """Clear every constraint state and request binding, and deactivate all slots.

        Call begin before sync or apply_mask can constrain a new request.
        """
        self._native.reset(stream)

    def anchor(self, slot: int, entities: list[str], stream: int = 0) -> None:
        """Derivation mode: start the slot's spans from these entities (ids such as "Q42"),
        by any of their labels, as if a fact span had reached them. Call after begin."""
        labels = self._index.labels
        if self._entity_index is None:
            self._entity_index = {e: n for n, e in enumerate(labels.owner_ids)}
            self._entity_rows = entity_rows(labels)
        offsets, rows = self._entity_rows["entity_row_offsets"], self._entity_rows["entity_rows"]
        found = sorted(
            int(r)
            for e in dict.fromkeys(entities)
            if (n := self._entity_index.get(e)) is not None
            for r in rows[offsets[n] : offsets[n + 1]]
        )
        if not found:
            raise ValueError(f"anchor has no writable label rows: {entities!r}")
        self._native.set_reached(slot, np.array(found, dtype=np.int32), stream)

    def set_version(self, slot: int, version: str | None, stream: int = 0) -> None:
        """Let the slot read this KG version (None: the base KG) until its next begin.
        Call after begin."""
        if self._versions is not None:
            number = self._versions.number(version)
        elif version is not None:
            raise ValueError(f"no KG version {version!r}; the engine has none registered")
        else:
            number = 0
        self._native.set_version(slot, number, stream)

    def set_stop(self, slot: int, question: str, stream: int = 0) -> None:
        """Compile this request's STOP relations from its question, after begin and
        before sync/open_span. The next begin clears them; drafts read them in place."""
        if self._stop_terms is None:
            raise ValueError("set_stop needs stop='residual-goal'")
        goal = self._stop_terms.for_question(question)
        relations = np.zeros((len(self._index.relations.owner_ids), 3), dtype=np.uint32)
        for relation in goal.head:
            relations[relation, 0] = 1
        for relation, mask in goal.coverage.items():
            relations[relation, 1] = mask & 0xFFFFFFFF
            relations[relation, 2] = mask >> 32
        words = b"".join(w.encode("utf-8") for w in goal.words)
        offsets = np.array(
            [0, *np.cumsum([len(w.encode("utf-8")) for w in goal.words])], dtype=np.int32
        )
        self._native.set_stop(
            slot,
            relations.view(np.int32).reshape(-1),
            np.frombuffer(words, dtype=np.uint8),
            offsets,
            stream,
        )

    def open_span(self, slot: int, stream: int = 0) -> None:
        """Start the slot inside a span, as if the span opener had just been written: a
        request whose prompt ends with the opener (a forced opener), whose first token then
        continues the span. Call after begin; composes with anchor and set_version."""
        self._native.open_span(slot, stream)

    def view(self, version: str | None) -> KGIndex | VersionView:
        """The KG index as a slot reading this KG version sees it, for closed_span."""
        if version is None:
            return self._index
        if self._versions is None:
            raise ValueError(f"no KG version {version!r}; the engine has none registered")
        return self._versions.view(self._index, version)

    def advance(self, tokens, stream: int = 0) -> None:
        """Advance sequences 0..len(tokens)-1 by these tokens (1-D int32, on the GPU or host)."""
        self._native.advance(tokens, stream)

    def fill_mask(self, sequences: int | None = None, stream: int = 0) -> None:
        """Write the masks of sequences 0..sequences-1 (default: all) into the mask buffer."""
        self._native.fill_mask(self._count(sequences), stream)

    def begin(self, slots, stream: int = 0) -> None:
        """Start constraining these slots (int32 on the GPU) from the initial state.

        Each will follow its token history from the prompt length on (see sync).
        """
        self._native.begin(slots, True, stream)

    def end(self, slots, stream: int = 0) -> None:
        """Stop constraining these slots; apply_mask leaves their rows alone."""
        self._native.begin(slots, False, stream)

    def sync(self, slots, history, prompt_len, total_len, stream: int = 0) -> None:
        """Advance the active slots over their token histories, read in place on the GPU.

        history is [slots, max length] int32, one row per slot; prompt_len and total_len
        are int32 per slot. Each slot follows its history from where it stopped (from
        prompt_len the first time) up to total_len, so the prompt is never advanced over.
        """
        self._native.sync(slots, history, prompt_len, total_len, stream)

    def apply_mask(
        self, logits, row_slots, stream: int = 0, *, local_pos=None, input_ids=None
    ) -> None:
        """Set every logit its slot's mask disallows to -inf, in place.

        logits is [rows, columns] float32 on the GPU, row i belonging to slot
        row_slots[i]; columns past the served vocabulary are always disallowed, and
        rows of inactive slots are left alone.

        Under speculative decoding a request owns consecutive rows, one per draft
        position: local_pos[i] is row i's offset among its request's rows, and
        input_ids[i] the token fed at row i (the last committed token at offset 0,
        then the drafts). Row i is masked as if the drafts of its request's earlier
        rows had been committed; the slot's state does not move.
        """
        if (local_pos is None) != (input_ids is None):
            raise ValueError("local_pos and input_ids go together")
        if local_pos is None:
            local_pos = input_ids = row_slots[:0]
        self._native.apply_mask(logits, row_slots, local_pos, input_ids, stream)

    @property
    def mask(self):
        """The mask buffer on the GPU, [max_sequences, words] int32; readable through DLPack."""
        return self._native.mask()

    def mask_host(self, sequences: int | None = None) -> np.ndarray:
        """A host copy of the masks of sequences 0..sequences-1, after waiting for the GPU."""
        return self._native.mask_host(self._count(sequences))

    def states_host(self, sequences: int | None = None) -> dict[str, np.ndarray]:
        """A host copy of the constraint states, one array per field, after waiting for the GPU."""
        states = self._native.states_host(self._count(sequences))
        return {
            name: states[:, i]
            for i, name in enumerate(_engine.STATE_FIELDS)
            if (self._observe or self._menu or not name.startswith("observe"))
            and (self._observe_select or not name.startswith("select"))
            and (
                self._observe_answer
                or self._audit
                or self._observe_terminal
                or name != "answer_pos"
            )
            and (self._observe_no_loop or not name.startswith("loop"))
            and (
                self._audit or self._menu or self._observe_terminal or not name.startswith("audit")
            )
            and (self._observe_defer or not name.startswith("defer"))
            and (self._menu_close_next or not name.startswith("decision"))
        }

    def errors(self, sequences: int | None = None) -> int:
        """How many sequences were advanced by a token outside their mask."""
        return int(np.count_nonzero(self.states_host(sequences)["phase"] == PHASE_ERROR))

    def _count(self, sequences: int | None) -> int:
        return self.max_sequences if sequences is None else sequences


@dataclass(frozen=True)
class ClosedSpanRecord:
    kind: str
    """One of the engine's CLOSED kinds: none, unresolved, entity span, fact span, path span,
    comparison, relation plan (unexecuted and unbound)."""
    entity_ids: tuple[str, ...] = ()
    """An entity span's entities; a comparison's result (the operand it returns, or both)."""
    triples: tuple[tuple[str, str, str], ...] = ()
    operator: str = ""
    """A comparison's comparison (oracle.COMPARISONS)."""
    operands: tuple[str, ...] = ()
    """A comparison's operands' start entities, A's then B's."""

    @property
    def resolved(self) -> bool:
        """The oracle's export contract: only executed, grounded spans resolve."""
        return self.kind in ("entity span", "fact span", "path span", "comparison")


def closed_span(
    index: KGIndex | VersionView, states: dict[str, np.ndarray], seq: int
) -> ClosedSpanRecord:
    """Resolve a sequence's last closed span from the row ranges its state records. For a
    sequence that reads a KG version, index is the version's view (Engine.view)."""
    kind = CLOSED[int(states["last_kind"][seq])]
    lo, hi = int(states["last_lo"][seq]), int(states["last_hi"][seq])
    if kind == "comparison":  # lo, hi: the operands' start rows; subj_lo: the result
        operands = (index.labels.owner_id(lo), index.labels.owner_id(hi))
        result = int(states["last_subj_lo"][seq])
        if result >= 3:  # a value comparison's verdict: 3 yes, 4 no
            chosen = (VERDICTS["yes" if result == 3 else "no"],)
        else:  # a date comparison's operand: 0 A's, 1 B's, 2 both
            chosen = operands if result == 2 else operands[result : result + 1]
        operator = COMPARISONS[int(states["last_subj_hi"][seq]) - 1]
        return ClosedSpanRecord(kind, tuple(sorted(set(chosen))), (), operator, operands)
    if kind == "entity span":
        unbound = "last_subj_lo" in states and int(states["last_subj_lo"][seq]) == -1
        excluded = VERDICTS.values() if unbound else ()
        return ClosedSpanRecord(
            kind,
            tuple(
                sorted(
                    index.labels.owner_id(r)
                    for r in range(lo, hi)
                    if index.labels.owner_id(r) not in excluded
                )
            ),
        )
    if kind != "fact span":
        return ClosedSpanRecord(kind)
    subjects = {
        index.labels.owner_id(r)
        for r in range(int(states["last_subj_lo"][seq]), int(states["last_subj_hi"][seq]))
    }
    relations = {
        index.relations.owner_id(r)
        for r in range(int(states["last_rel_lo"][seq]), int(states["last_rel_hi"][seq]))
    }
    triples = {
        (s, r, index.labels.owner_id(row))
        for s in subjects
        for r in relations
        for row in index.object_rows(s, r, lo, hi)
    }
    return ClosedSpanRecord(kind, (), tuple(sorted(triples)))
