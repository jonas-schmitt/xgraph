"""The oracle: a scalar CPU implementation of advance and mask, written for clarity.

Tests check the engine against it step by step; it never serves requests. It
follows one sequence at a time and recomputes everything from the label rows, the
KG index's edges taken as plain data, and the token bytes on each step, without
the engine's ranges, tables, or kernels.

Span syntax is matched on bytes, so a delimiter may be split across tokens in any
way, but it must end exactly at a token boundary:

- In free text, a token that ends the span opener opens a span; a token that
  completes the opener before its own end is not allowed. Special tokens (such as
  end of text) are allowed and break any partial opener.
- A span starts in its subject segment. Inside a segment, branches run side by
  side, because one token can serve more than one of them: the label branch (the
  label tokens typed so far, which must start some candidate row), the closer
  branch, the separator branch, and the escape branch. A token is allowed if it
  keeps at least one branch alive. Closer and separator start only right after a
  complete label, the escape marker at any token boundary while the label branch
  is alive.
- Subject segment: the candidates are all entity labels. The closer ends an
  entity span, resolved to every entity with the complete label. The separator,
  allowed when one of those entities has a triple, starts the relation segment
  with them as the candidate subjects.
- Relation segment: the candidates are the labels of the relations the candidate
  subjects have. There is no closer; the separator starts the object segment with
  the relations of the complete label as the candidate relations.
- Object segment: the candidates are the labels of the objects of the candidate
  subjects and relations. The closer ends a fact span, resolved to every triple of
  a candidate subject and relation whose object has the complete label.
- If a closer, a separator, and an escape marker complete on the same token, the
  closer wins, then the separator.
- After an escape, any text is allowed until the span closer, which closes the
  span unresolved. Special tokens are not allowed in a span.
- Without repeats, a fact span whose three labels are those of a fact span the
  sequence closed before (among its first MAX_FACTS) cannot close; the escape
  remains, so this needs escapes.
- In derivation mode, spans are bound by entity. After a fact span, every span (entity
  spans included) may only name the entities that fact span reached (objects of its
  subject under its relations, written with the span's object label), by any of their
  labels: the first MAX_REACHED of those entities' label index rows. A fact span's
  subject is one entity: the best-connected (most adjacency entries, one per triple and
  object label row, then the first label index row) among the rows it may name with the
  complete label; after a fact span, an entity span names one entity the same way.
- In path mode (which binds spans as derivation mode does), a span is a path: a subject,
  then relations only, each one the current entities (the frontier) have. The first
  frontier is every entity the subject label may name that has triples (the first
  MAX_REACHED by first label index row), not one sense: the relations decide which senses
  go on; after a relation, the separator moves on to its objects (the first
  MAX_REACHED entities in row order, and only if some of them have triples) and the
  closer ends the path, which resolves to that last frontier and binds the spans after
  it to those entities. After MAX_PATH_RELATIONS relations the path must close. The span
  right after a path is its answer, an entity span: it takes no separator, so it cannot
  go on as another path (a later span may).
- With committed_paths True (path mode), a path commits to one start entity: it still
  offers the relations of its whole frontier, but resolves (and binds the spans after it)
  to exactly what the best start reaches, the first start in best-connected order (as a
  fact span's subject is chosen) from which the whole path is non-empty. Each frontier
  entity carries the rank of the best start that reaches it (the least over the entities
  it was reached from), and the closer keeps the entities with the least rank present.
  That rank is the best start with a non-empty path, and every entity it reaches carries
  it, since a better start reaching one would have given it a lesser rank. The first
  frontier then keeps its first MAX_REACHED starts by rank, and each next frontier its
  first MAX_REACHED entities by rank and then first label index row, so that the best
  starts' entities survive the cap first. An anchored path starting from one entity is
  unchanged.
- With relations="vocabulary" (path mode), relation segments admit every indexed relation
  label. No path is executed and no frontier is stored. A complete relation can close or,
  before MAX_PATH_RELATIONS, take another separator. The start keeps path mode's entity
  admission, including anchors and unquoted starts. A closed plan keeps its labels but is
  unresolved, with no entity result or answer binding. The next span is still an entity
  answer span, and free text outside spans is unrestricted.
- With relations="successor", the first relation admits the vocabulary. Each later
  relation must succeed an admitted id of the preceding label: some entity in the KG
  version is an object of that id and a subject of the next. Ambiguous labels retain
  every compatible id, never an incompatible sense. This schema walk stores relation
  support only, with no entity frontier or answer binding. A separator needs at least
  one successor; the closer needs only a complete label. STOP and expressions require
  relations="frontier" (the default, unchanged execution semantics).
- With start_rank "sitelinks" (the sitelink rank), best-connected order puts the entity
  with the most sitelinks (the KG index's sitelink counts) first, then the one with the
  most adjacency entries, then the first label index row. It is one order for every choice
  above: derivation mode's subject and its bound entity span, and a committed path's
  starts. A KG version keeps the counts and breaks their ties by its own degree. An index
  without sitelink counts refuses it.
- With bound_escapes False, a subject segment the binding restricts (derivation or path
  mode) cannot escape: the answer after a path names what the KG reached, while a path's
  relations and unbound spans keep the escape.
- With unquoted_starts True, a subject segment the binding does not restrict (every
  subject outside derivation and path mode; there, one no anchor, fact span, or path has
  bound) cannot name a label that starts with a quotation mark (syntax.QUOTATION_MARKS):
  its rows are no candidates there, whatever tokens they start with. A model copying the
  opening quote of a title its question quotes is then not steered into another label
  that starts with one. Objects and bound subjects name such labels as before.
- A request can start inside a span (a forced opener): its prompt ends with the span
  opener, which the state never reads, and opened_state gives the state after the opener.
- A state may read a KG version instead of the base KG (docs/kg-versions-adr.md): edits
  (subject, relation, object) that replace every object of their (subject, relation)
  pair, as xgraph.mquake.apply_edits does to triples. Every KG read above (the relations
  and objects of a subject, whether it has triples, its degree in best-connected order,
  the frontier's steps) then reads the version, which equals the KG index rebuilt from the
  edited triples: an edit whose subject, relation, or object has no label left writes no
  edge, but still replaces its pair's objects.
- With expressions True (committed paths, escapes, and an index with the date table and
  expression syntax excluded from its labels: index.check_expressions), a span may be a date
  comparison, `⟦ earlier: A | B · relation · relation⟧` (or `later:`). An operator
  (syntax.operators after the label prefix: the first compares by the earlier date, the
  second by the later) is a branch at the start of an unbound subject segment, beside the
  label and escape branches, and must end at a token boundary. Operand A is then an unbound
  subject whose complete label takes the operand separator (when one of its entities has
  triples) and neither the closer nor the separator; operand B's complete label takes the
  separator (likewise) and neither the closer nor the operand separator. Each operand's
  starts are ranked as a committed path's, and the two frontiers advance in lock step: the
  relation segment offers the relations that some entity of each frontier has, each
  operand walks every relation with the written label, the separator needs both next
  frontiers to have an entity with triples, and the closer needs both to have a date (an
  entity of the date table). The comparison commits each operand to its best start that
  reaches a date (the least rank among the frontier's dated entities, as
  xgraph.expressions.Executor commits with dated=True) and compares the earliest date that
  start reaches (by interval bounds, Date.lo then Date.hi) as Date.before does: the
  operand whose date is earlier (later) is the result. When neither date is before the
  other (equal or overlapping intervals) the comparison is undetermined and both operands'
  start entities are the result. The result binds the span after it, its answer, as a path
  binds its answer: an entity span naming the result's entities by any of their labels.
- A value comparison, `⟦ same: A | B · relation⟧` (the third operator), needs an index with
  the verdict entities (index.VERDICTS; without them the operator is not offered). Its
  operands and relations are written as a date comparison's; each operand commits to its
  best start that reaches any value (the least rank in its frontier, as
  xgraph.expressions.Executor commits with dated=False), and the closer needs both to reach
  values of one kind: entities on both sides, or dates (date entities) only on both. Its
  verdict is yes when the two reaches share an entity or an equal date (the same stored
  date, as the executor compares them; overlapping dates are no shared date), and no
  otherwise; the verdict's entity binds the answer. The verdict entities are never
  candidates of an unbound subject.
- With stop="residual-goal" (committed paths), each request supplies question text to
  initial_state or anchored_state. At a complete path relation label, the host-compiled
  STOP relations (stopping.RelationTerms) force the closer among delimiter branches when
  the last relation names the answer head and no next-frontier relation covers a question
  stem unused by the written start and consumed relations. A longer relation label stays
  writable; comparisons keep their expression rules. The rule is off by default.
- With end_after_answer True (path mode), closing the entity answer span immediately
  after a path, relation plan or comparison leaves only the supplied eos_token_ids
  allowed. Earlier masks are unchanged. An escaped answer does not trigger completion.
  EOS keeps the completed state; begin a new state for another answer. Off by default.
- With observe=k (1..64, paths with frontier support and expressions off), every start
  and relation takes the canonical observation marker " ·→" before its usual separator
  or closer. A longer label remains writable until the marker commits the shorter
  label. Its payload is forced one token at a time: up to k labels of the capped live
  frontier in best-connected order, joined by " ;", then " (+N more)" if needed, or
  "∅" with the label prefix for an empty frontier. Each entity uses its first indexed
  label row, not necessarily its canonical human label. Punctuation pieces are encoded
  independently; indexed labels retain their tokenization context. This canonical
  concatenation round-trips to exact bytes but need not be BPE of the whole text.
  Observations do not consume a relation, change a frontier or commit a start sense;
  committed paths still choose a sense only on closing. Starts show all capped live
  senses, answers have no observation. STOP and the hop cap act after the observation;
  escapes remain available outside its forced payload. Versioned degrees and frontiers
  determine both content and display rank. A terminal start shows empty; an admitted
  relation always reaches something under frontier support. The marker is the span
  separator plus "→", so existing indexed labels already exclude it; literal arrows
  in labels remain valid. observe=0 leaves existing states
  and masks unchanged. See docs/observed-paths-adr.md.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from itertools import pairwise
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xgraph.stopping import StopRelations

from xgraph.expressions import Date
from xgraph.index import (
    IndexMismatch,
    KGIndex,
    check_expressions,
    check_start_rank,
    unpack_date,
)
from xgraph.syntax import SpanSyntax
from xgraph.tokenizer import ServedTokenizer

MAX_FACTS = 64
"""Fact spans a sequence remembers for the no-repeat rule (the engine's kMaxFacts)."""

MAX_REACHED = 64
"""Object rows a derivation carries from one fact span to the next (the engine's kMaxReached)."""
MAX_PATH_RELATIONS = 8
"""Path mode: the most relations a path names; after them it must close, so that a model
looping (country · country · ...) still reaches an answer."""
COMPARISONS = ("earlier", "later", "same")
"""Expressions: the comparison each of SpanSyntax.operators writes, by position."""


class Phase(Enum):
    """The part of the output a sequence is in."""

    FREE = "free text"
    SPAN = "span"
    ESCAPED = "escaped"


class Segment(Enum):
    SUBJECT = "subject"
    RELATION = "relation"
    OBJECT = "object"


class TokenNotAllowed(ValueError):
    """Advance was given a token outside the allowed set."""


@dataclass(frozen=True)
class ClosedSpan:
    """The most recently closed span, expressed as stable KG IDs.

    Entity and path spans expose endpoints in ``entity_ids``; fact spans expose
    matching ID triples in ``triples``. An escaped span has ``resolved=False``.
    Endpoints are not a full path witness and a resolved result does not assert
    real-world truth or question relevance. A token-limited open span has not
    produced a new closed result.
    """

    labels: tuple[tuple[int, ...], ...]
    """The token labels of the segments written, in order; empty after an escape.
    An unexecuted relation plan retains its labels with resolved=False."""
    entity_ids: tuple[str, ...]
    """Entity-span owners or executed path/comparison endpoints; empty for facts."""
    triples: tuple[tuple[str, str, str], ...]
    """A fact span: every matching (subject, relation, object), sorted; otherwise empty."""
    resolved: bool
    operator: str = ""
    """A date comparison: its comparison (COMPARISONS); its entity_ids are its result, the
    operand it returns, or both when undetermined. Empty for any other span."""
    operands: tuple[str, ...] = ()
    """A date comparison: the start entity each operand committed to, A's then B's."""
    path: bool = False
    """A selected start closed without a relation is also an executed path."""


UNRESOLVED = ClosedSpan((), (), (), resolved=False)


@dataclass(frozen=True)
class OracleState:
    """Immutable per-sequence decoding state, pinned to one KG version.

    Obtain states from :meth:`Oracle.initial_state` or
    :meth:`Oracle.anchored_state` and replace them with the return value of
    :meth:`Oracle.advance`. Do not assemble or edit internal branch fields.
    ``last_span`` retains the previous result until another span closes; inspect
    it at closing boundaries rather than treating every token as a new result.
    """

    phase: Phase
    tail: bytes = b""
    """Free text and escaped spans: the latest bytes, which may start a delimiter."""
    segment: Segment = Segment.SUBJECT
    subjects: frozenset[str] = frozenset()
    """Relation and object segments: the candidate subjects."""
    relations: frozenset[str] = frozenset()
    """Object segment: the candidate relations; successor mode: the preceding label's
    compatible relation ids (empty before the first relation)."""
    written: tuple[tuple[int, ...], ...] = ()
    """The token labels of the segments completed so far."""
    typed: tuple[int, ...] | None = ()
    """Span: the label tokens typed so far, or None once the label branch is dead."""
    closing: bytes | None = None
    """Span: the closer bytes typed so far, or None when the closer branch is not alive."""
    closing_label: tuple[int, ...] = ()
    """Span: the complete label the closer branch started after."""
    separating: bytes | None = None
    """Span: the separator bytes typed so far, or None when that branch is not alive."""
    separating_label: tuple[int, ...] = ()
    """Span: the complete label the separator branch started after."""
    escaping: bytes | None = None
    """Span: the escape-marker bytes typed so far, or None when that branch is not alive."""
    last_span: ClosedSpan | None = None
    """The span closed most recently, kept until the next one closes."""
    facts: tuple[tuple[tuple[int, ...], ...], ...] = ()
    """The token labels of the fact spans closed so far, the first MAX_FACTS of them."""
    reached: frozenset[int] | None = None
    """Derivation mode: the object rows the last fact span reached; None before one."""
    ranks: tuple[tuple[str, int], ...] = ()
    """Path mode, relation segment: each frontier entity (the subjects) with the rank of the
    best start entity that reaches it, in best-connected order from 0."""
    version: str | None = None
    """The KG version the sequence reads, or None for the base KG."""
    operating: bytes | None = None
    """Expressions, span: the operator bytes typed so far (after the label prefix's), or None
    when that branch is not alive."""
    operator: int = 0
    """Expressions: the operator the span started with, its position in the syntax's
    operators plus one; 0 for a span without one."""
    starts: tuple[tuple[str, ...], ...] = ()
    """Expressions: the ranked starts of each operand written so far, best first, so that
    an operand's start of rank k is starts[operand][k]."""
    stop: StopRelations | None = None
    """Per-request STOP relations, retained across spans."""
    ranks_b: tuple[tuple[str, int], ...] = ()
    """Expressions, relation segment: operand B's frontier, as ranks holds operand A's."""
    answer_complete: bool = False
    """End after answer: a normally closed path-mode answer admits only EOS."""
    observing: tuple[int, ...] | None = None
    """Canonical observation marker prefix, beside the label branch until it commits."""
    observing_label: tuple[int, ...] = ()
    observation: tuple[int, ...] = ()
    observation_offset: int = 0
    observed: bool = False
    """The current complete label has been observed and cannot be extended."""
    selecting: bool = False
    selected: bool = False
    selection: tuple[int, ...] | None = None
    selection_segment: Segment = Segment.SUBJECT
    selection_label: tuple[int, ...] = ()
    selection_frontier: tuple[tuple[str, int], ...] = ()
    answer_pos: int = 0
    """One-based cursor in the forced canonical answer prefix; zero is inactive."""
    loop_history: tuple[tuple[frozenset[str], frozenset[str]], ...] = ()
    """At most MAX_PATH_RELATIONS consumed (entity set, relation ids) entries per path."""
    audit_phase: int = 0
    audit_payload: tuple[int, ...] = ()
    audit_offset: int = 0
    audit_plan: tuple[tuple[int, ...], ...] = ()
    audit_frontier: tuple[tuple[str, int], ...] = ()
    audit_penultimate: tuple[tuple[str, int], ...] = ()
    audit_choices: tuple[int, ...] = ()
    audit_command: tuple[int, ...] = ()
    audit_action: int = -1
    audit_scores: tuple[tuple[str, int], ...] = ()
    audit_alias_ranks: tuple[int, ...] = ()
    defer_count: int = 0
    defer_pos: int = 0
    decision_phase: int = 0
    decision_pos: int = 0
    decision_choices: int = 0


@dataclass(frozen=True)
class _Graph:
    """The KG one sequence reads: every (subject, relation, object row) edge, the subjects
    with triples, and each subject's degree (its adjacency entries)."""

    edges: frozenset[tuple[str, str, int]]
    has_triples: frozenset[str]
    degree: dict[str, int]
    outgoing: dict[str, list[tuple[str, str, int]]]

    @classmethod
    def of(cls, edges: set[tuple[str, str, int]]) -> _Graph:
        outgoing: dict[str, list[tuple[str, str, int]]] = {}
        for edge in edges:
            outgoing.setdefault(edge[0], []).append(edge)
        degree = {subject: len(rows) for subject, rows in outgoing.items()}
        return cls(frozenset(edges), frozenset(degree), degree, outgoing)

    def from_subjects(self, subjects):
        """Query the plain reference edges without scanning unrelated subjects."""
        return (edge for subject in subjects for edge in self.outgoing.get(subject, ()))


class Oracle:
    """CPU reference for the span language and KG execution policy.

    Args:
        index: Compiled KG index. Its tokenizer fingerprint and span syntax
            must match the next two arguments.
        tokenizer: Tokenizer used to build the index and encode output tokens.
        syntax: Delimiters and label prefix used at index build time.
        paths: Execute a start label followed by relation labels. The next
            span is an entity answer bound to the reached frontier.
        committed_paths: Resolve a path to the best-ranked surviving start's
            endpoints. Requires ``paths=True``.
        versions: Optional mapping of names to replacement ID triples or pair
            deletions. A state selects one version before decoding.
        escapes: Permit an escape to close a span unresolved.
        bound_escapes: Permit an anchored start or bound answer to escape.

    Remaining keyword arguments configure the policies described in the span
    modes guide. Defaults preserve ordinary entity/fact spans with escapes.
    A single oracle shares reference tables among independent immutable states.
    :meth:`allowed` scans the vocabulary; use this implementation for small
    examples and correctness checks, and :class:`xgraph.engine.Engine` for GPU
    sampling. Construction does not initialize CUDA.
    """

    def __init__(
        self,
        index: KGIndex,
        tokenizer: ServedTokenizer,
        syntax: SpanSyntax,
        *,
        escapes: bool = True,
        repeats: bool = True,
        derivation: bool = False,
        paths: bool = False,
        bound_escapes: bool = True,
        committed_paths: bool = False,
        versions: Mapping[str, Iterable[tuple[str, str, str | None]]] | None = None,
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
    ) -> None:
        """With escapes False the escape marker is never allowed: every span names the KG.
        With repeats False a fact span cannot close with the labels of an earlier one.
        With derivation True spans are bound by entity, with paths True spans are paths,
        and with committed_paths True a path resolves to one start entity's reach (see the
        module docstring). versions names KG versions, each a list of edits (subject,
        relation, object) by id, that a state can read instead of the base KG. start_rank
        "sitelinks" puts the entities with more sitelinks first in best-connected order
        (the sitelink rank), which needs an index with sitelink counts. With unquoted_starts
        True an unbound subject cannot name a label that starts with a quotation mark. With
        expressions True a span may be a date comparison (the module docstring)."""
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
        from xgraph.menus import check_menu, menu_marker, menu_pieces

        check_menu(menu, paths=paths, relations=relations, expressions=expressions, audit=audit)
        self._menu = menu
        self._menu_pieces = (
            menu_pieces(tokenizer, syntax, len(index.relations.owner_ids)) if menu else []
        )
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
        )
        end_after_answer, observe_select, observe_answer = (
            policy.end_after_answer,
            policy.observe_select,
            policy.observe_answer,
        )
        self._end_after_answer = end_after_answer
        self._eos_token_ids = policy.eos_token_ids
        self._relations = relations
        self._observe_terminal = observe_terminal
        self._terminal_order, self._terminal_seed = terminal_order, terminal_seed
        from xgraph.audit import RelationScores, audit_commands, audit_pieces, terminal_pieces
        from xgraph.observations import answer_tokens, observation_pieces, selection_tokens

        if (audit or observe_terminal) and index.sitelinks is None:
            raise ValueError("audit/terminal display needs an index with sitelink counts")
        self._audit, self._audit_entities, self._audit_relations = (
            audit,
            audit_entities,
            audit_relations,
        )
        self._audit_pieces = (
            terminal_pieces(tokenizer, syntax)
            if observe_terminal
            else audit_pieces(tokenizer, syntax, len(index.relations.owner_ids))
            if audit
            else []
        )
        self._audit_commands = audit_commands(tokenizer, syntax) if audit else []
        self._audit_terms = (
            RelationScores(index, tokenizer, relation_labels) if audit or menu else None
        )
        self._audit_sitelinks = (
            dict(zip(index.labels.owner_ids, index.sitelinks, strict=True))
            if audit or observe_terminal
            else {}
        )
        self._observe_required, self._observe_no_loop = observe_required, observe_no_loop
        self._observe = observe
        self._observe_width = observe_width
        self._observe_defer, self._menu_close_next = observe_defer, menu_close_next
        from xgraph.observations import decision_pieces, defer_tokens

        self._decision_pieces = decision_pieces(tokenizer, syntax) if menu_close_next else []
        self._defer_tokens = defer_tokens(tokenizer, syntax) if observe_defer else ()
        self._observe_select = observe_select
        self._observe_answer = observe_answer
        self._selection_tokens = selection_tokens(tokenizer, syntax) if observe_select else ()
        self._answer_tokens = (
            answer_tokens(tokenizer, syntax) if observe_answer or audit or observe_terminal else ()
        )
        self._observation_pieces = observation_pieces(tokenizer, syntax) if observe else []
        if menu and not observe:
            self._observation_pieces = [tuple(tokenizer.encode(menu_marker(syntax)))]
        if expressions and not committed_paths:
            raise ValueError("expressions need committed paths (committed_paths=True)")
        if expressions and not escapes:
            raise ValueError("expressions need escapes, the way out of an operand or relation")
        if expressions and len(syntax.operators) > len(COMPARISONS):
            raise ValueError(f"expressions know the operators of {COMPARISONS}, by position")
        if stop not in (None, "residual-goal"):
            raise ValueError(f"unknown STOP rule {stop!r}; expected residual-goal")
        if stop and not committed_paths:
            raise ValueError("residual-goal STOP needs committed paths (committed_paths=True)")
        self._stop_terms = None
        if stop:
            from xgraph.stopping import RelationTerms

            self._stop_terms = RelationTerms.from_index(index, tokenizer)
        self._relation_numbers = {r: n for n, r in enumerate(index.relations.owner_ids)}
        labels, relations = index.labels, index.relations
        self._expressions = expressions
        self._dates: dict[str, Date] = {}  # expressions: each date entity's date
        if expressions:
            check_expressions(index)
            self._dates = {
                labels.owner_ids[int(e)]: unpack_date(int(k))
                for e, k in zip(index.date_entities, index.date_keys, strict=True)
            }
        # A value comparison (the third operator) needs the verdict entities to bind.
        offered = syntax.operators if index.verdicts else syntax.operators[:2]
        self._operators = [(syntax.label_prefix + op).encode("utf-8") for op in offered]
        self._verdicts = dict(index.verdicts or {})
        self._operand_separator = syntax.operand_separator.encode("utf-8")
        self._sitelinks: dict[str, int] = {}  # empty: best-connected order by degree alone
        if check_start_rank(start_rank, index) == "sitelinks":
            self._sitelinks = {
                e: int(n) for e, n in zip(labels.owner_ids, index.sitelinks, strict=True)
            }
        self._entity_rows = [
            (labels.row_tokens(r), labels.owner_id(r)) for r in range(labels.num_rows)
        ]
        self._first_row: dict[str, int] = {}  # each entity's first label index row
        for row, (_, owner) in enumerate(self._entity_rows):
            self._first_row.setdefault(owner, row)
        self._relation_rows = [
            (relations.row_tokens(r), relations.owner_id(r)) for r in range(relations.num_rows)
        ]
        for kind, rows in (("entity", self._entity_rows), ("relation", self._relation_rows)):
            for row, (left, right) in enumerate(pairwise(rows), 1):
                if left[0] > right[0]:
                    raise IndexMismatch(
                        f"{kind} label rows are not sorted as token tuples at rows "
                        f"{row - 1} and {row}"
                    )
        # Every (subject, relation, object row) the adjacency holds, as plain data.
        edges: set[tuple[str, str, int]] = set()
        for entity, subject in enumerate(labels.owner_ids):
            start, stop = index.adjacency_offsets[entity], index.adjacency_offsets[entity + 1]
            for edge in range(start, stop):
                relation = relations.owner_ids[index.edge_relations[edge]]
                edges.add((subject, relation, int(index.edge_objects[edge])))
        self._graphs: dict[str | None, _Graph] = {None: _Graph.of(edges)}
        # A version replaces every edge of an edited (subject, relation) pair by an edge to
        # each label row of each listed object; an edit whose subject, relation, or object
        # has no label left writes none, as the triple would in a rebuilt index.
        rows_of: dict[str, list[int]] = {}
        for row, (_, owner) in enumerate(self._entity_rows):
            rows_of.setdefault(owner, []).append(row)
        # The first row is already stored; retain only additional alias rows.
        self._other_rows = {e: tuple(rows[1:]) for e, rows in rows_of.items() if len(rows) > 1}
        writable = {owner for _, owner in self._relation_rows}
        from xgraph.kgversions import VersionEdits

        if isinstance(versions, VersionEdits):
            versions.check_base(index)
        for name, edits in (versions or {}).items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"a KG version needs a non-empty name, got {name!r}")
            edits = list(edits)
            pairs = {(s, r) for s, r, _ in edits}
            kept = {(s, r, o) for s, r, o in edges if (s, r) not in pairs}
            added = {
                (s, r, row)
                for s, r, o in edits
                if s in rows_of and r in writable
                for row in rows_of.get(o, ())
            }
            self._graphs[name] = _Graph.of(kept | added)
        # Independent reference construction from plain edge sets, not the engine's
        # packed host tables. Aliases of object entities must not multiply witnesses.
        self._successors: dict[str | None, dict[str, set[str]]] = {}
        if self._relations == "successor":
            for name, graph in self._graphs.items():
                outgoing: dict[str, set[str]] = {}
                for s, r, _ in graph.edges:
                    outgoing.setdefault(s, set()).add(r)
                pairs: dict[str, set[str]] = {}
                for _, r, o in graph.edges:
                    pairs.setdefault(r, set()).update(outgoing.get(self._entity_rows[o][1], ()))
                self._successors[name] = pairs
        self._derivation = derivation or paths
        self._paths = paths
        self._bound_escapes = bound_escapes
        self._committed_paths = committed_paths
        self._bytes = [tokenizer.token_bytes(t) for t in range(tokenizer.vocab_size)]
        self._special = tokenizer.special_ids
        self._opener = syntax.opener.encode("utf-8")
        self._closer = syntax.closer.encode("utf-8")
        self._separator = syntax.separator.encode("utf-8")
        self._label_prefix = syntax.label_prefix
        self._escape = syntax.escape.encode("utf-8")
        self._escapes = escapes
        self._repeats = repeats
        # The label index rows an unbound subject may not name: with unquoted starts those
        # that start with a quotation mark, and always the verdict entities'.
        verdict_ids = set(self._verdicts.values())
        self._quoted_rows = frozenset(
            row
            for row, (tokens, owner) in enumerate(self._entity_rows)
            if owner in verdict_ids
            or (unquoted_starts and syntax.quoted(b"".join(self._bytes[t] for t in tokens)))
        )

    def initial_state(
        self, version: str | None = None, *, question: str | None = None
    ) -> OracleState:
        """Start in free text, pinning ``version`` (``None`` selects the base).

        Unknown version names raise ``ValueError``. Supply ``question`` when
        using question-dependent STOP, menus or audits.
        """
        stop = self._stop_terms.for_question(question) if self._stop_terms is not None else None
        scores = self._audit_terms.for_question(question) if self._audit_terms and question else ()
        priorities = (
            tuple(self._audit_terms.alias_ranks(question)) if self._audit_terms and question else ()
        )
        return OracleState(
            Phase.FREE,
            version=self._version(version),
            stop=stop,
            audit_scores=scores,
            audit_alias_ranks=priorities,
        )

    def anchored_state(
        self, entities: list[str], version: str | None = None, *, question: str | None = None
    ) -> OracleState:
        """Derivation mode's initial state for a question about these entities: as if a
        fact span had reached them, so spans start from them (by any of their labels)."""
        rows = self._rows_for(entities)
        if not rows:
            raise ValueError(f"anchor has no writable label rows: {entities!r}")
        initial = self.initial_state(question=question)
        return OracleState(
            Phase.FREE,
            reached=frozenset(rows[:MAX_REACHED]),
            version=self._version(version),
            stop=initial.stop,
            audit_scores=initial.audit_scores,
            audit_alias_ranks=initial.audit_alias_ranks,
        )

    def opened_state(self, state: OracleState) -> OracleState:
        """The state right after the span opener, from a state in free text (such as
        initial_state or anchored_state): where a request whose prompt ends with the
        opener starts (a forced opener), keeping the state's memory."""
        if state.answer_complete:
            raise ValueError("the answer is complete; only EOS may follow")
        if state.phase is not Phase.FREE:
            raise ValueError(f"a span opens from free text, not from {state.phase.value}")
        return OracleState(
            Phase.SPAN,
            last_span=state.last_span,
            facts=state.facts,
            reached=state.reached,
            version=state.version,
            stop=state.stop,
            audit_scores=state.audit_scores,
            audit_alias_ranks=state.audit_alias_ranks,
        )

    def _version(self, version: str | None) -> str | None:
        if version not in self._graphs:
            names = sorted(name for name in self._graphs if name is not None)
            raise ValueError(f"the oracle has no KG version {version!r}; it has {names}")
        return version

    def _graph(self, state: OracleState) -> _Graph:
        return self._graphs[state.version]

    def _rows_for(self, entities) -> list[int]:
        """All rows of these entities, in the original global label index order."""
        return sorted(
            row
            for entity in set(entities)
            if entity in self._first_row
            for row in (self._first_row[entity], *self._other_rows.get(entity, ()))
        )

    def allowed(self, state: OracleState) -> frozenset[int]:
        """Return every token ID that :meth:`advance` accepts in this state.

        This scans the served vocabulary. A sampling loop masks all other IDs
        before sampling; free text can still be subject to delimiter boundaries.
        """
        return frozenset(t for t in range(len(self._bytes)) if self._step(state, t) is not None)

    def advance(self, state: OracleState, token: int) -> OracleState:
        """Consume one admitted token and return a new state.

        Raises:
            TokenNotAllowed: The token is outside this state's allowed set.

        The input state is unchanged, including on rejection. Consume generated
        tokens once; prompts are not ordinary generated history. For a prompt
        ending in the opener, start with :meth:`opened_state` instead.
        """
        following = self._step(state, token)
        if following is None:
            raise TokenNotAllowed(f"token {token} is not allowed in {state.phase.value}")
        return following

    def _step(self, state: OracleState, token: int) -> OracleState | None:
        if not 0 <= token < len(self._bytes):
            return None
        if state.audit_phase == 6:
            if token != state.audit_payload[state.audit_offset]:
                return None
            offset = state.audit_offset + 1
            return replace(
                state,
                audit_offset=offset,
                audit_phase=0 if offset == len(state.audit_payload) else 6,
            )
        if state.audit_phase:
            return self._audit_step(state, token)
        if state.decision_phase:
            return self._decision_step(state, token)
        if state.answer_complete:
            return state if token in self._eos_token_ids else None
        if state.answer_pos:
            if token != self._answer_tokens[state.answer_pos - 1]:
                return None
            if state.answer_pos == len(self._answer_tokens):
                return self.opened_state(replace(state, answer_pos=0))
            return replace(state, answer_pos=state.answer_pos + 1)
        if state.observation_offset < len(state.observation):
            if token != state.observation[state.observation_offset]:
                return None
            offset = state.observation_offset + 1
            result = replace(
                state,
                observation_offset=offset,
            )
            return self._after_observation(result) if offset == len(state.observation) else result
        data = self._bytes[token]
        if state.phase is Phase.FREE and token in self._special:
            return replace(state, tail=b"")
        if data is None:
            return None
        if state.phase is Phase.FREE:
            return self._free_step(state, data)
        if state.phase is Phase.SPAN:
            return self._span_step(state, token, data)
        return self._escaped_step(state, data)

    def _free_step(self, state: OracleState, data: bytes) -> OracleState | None:
        text = state.tail + data
        at = text.find(self._opener)
        if at < 0:
            return replace(state, tail=_suffix(text, len(self._opener) - 1))
        if at + len(self._opener) != len(text):
            return None
        return self.opened_state(state)

    def _escaped_step(self, state: OracleState, data: bytes) -> OracleState | None:
        text = state.tail + data
        at = text.find(self._closer)
        if at < 0:
            return replace(state, tail=_suffix(text, len(self._closer) - 1))
        if at + len(self._closer) != len(text):
            return None
        return OracleState(
            Phase.FREE,
            last_span=UNRESOLVED,
            facts=state.facts,
            reached=state.reached,
            version=state.version,
            stop=state.stop,
            audit_scores=state.audit_scores,
            audit_alias_ranks=state.audit_alias_ranks,
        )

    def _candidates(self, state: OracleState, prefix=(), *, exact=False):
        """Query plain reference rows, preserving their global label index order."""
        if prefix is None:
            return iter(())
        if (
            state.segment is Segment.SUBJECT
            and not state.selecting
            and (not self._derivation or state.reached is None)
        ):
            # The plain rows retain the index's lexicographic token order. Search
            # them independently of the engine's packed prefix-range queries.
            def key(item):
                return item[0] if exact else item[0][: len(prefix)]

            lo = bisect_left(self._entity_rows, prefix, key=key)
            hi = bisect_right(self._entity_rows, prefix, key=key)
            return (
                (*self._entity_rows[row], row)
                for row in range(lo, hi)
                if row not in self._quoted_rows
            )
        return (
            candidate
            for candidate in self._segment_candidates(state)
            if (candidate[0] == prefix if exact else candidate[0][: len(prefix)] == prefix)
        )

    def _segment_candidates(self, state: OracleState) -> list[tuple[tuple[int, ...], str, int]]:
        """All candidates of a bound segment, before matching the typed tokens."""
        if state.segment is Segment.SUBJECT:
            if state.selecting:
                frontier = dict(state.selection_frontier)
                return [(*self._entity_rows[row], row) for row in self._rows_for(frontier)]
            return [(*self._entity_rows[row], row) for row in sorted(state.reached)]
        if state.segment is Segment.RELATION:
            if self._relations != "frontier":
                have = (
                    {r for _, r in self._relation_rows}
                    if self._relations == "vocabulary" or not state.relations
                    else set().union(
                        *(self._successors[state.version].get(r, set()) for r in state.relations)
                    )
                )
                return [
                    (tokens, r, row)
                    for row, (tokens, r) in enumerate(self._relation_rows)
                    if r in have
                ]
            have = {r for _, r, _ in self._graph(state).from_subjects(state.subjects)}
            blocked = self._loop_blocked(state, state.subjects) if not state.observed else set()
            bad_labels = {tokens for tokens, r in self._relation_rows if r in blocked & have}
            if state.operator:  # a comparison: the relations both frontiers have
                b = dict(state.ranks_b)
                have &= {r for _, r, _ in self._graph(state).from_subjects(b)}
            return [
                (tokens, r, row)
                for row, (tokens, r) in enumerate(self._relation_rows)
                if r in have and tokens not in bad_labels
            ]
        objects = {
            o
            for s, r, o in self._graph(state).from_subjects(state.subjects)
            if s in state.subjects and r in state.relations
        }
        return [(*self._entity_rows[row], row) for row in sorted(objects)]

    def _span_step(self, state: OracleState, token: int, data: bytes) -> OracleState | None:
        defer_alive = False
        if state.selecting and state.defer_pos:
            if token == self._defer_tokens[state.defer_pos - 1]:
                if state.defer_pos == len(self._defer_tokens):
                    result = replace(
                        state,
                        selecting=False,
                        selected=True,
                        observed=True,
                        segment=state.selection_segment,
                        typed=state.selection_label,
                        defer_count=state.defer_count + 1,
                        defer_pos=0,
                        closing=None,
                        separating=None,
                        escaping=None,
                        observing=None,
                    )
                    return self._after_observation(result) if self._menu_close_next else result
                state = replace(state, defer_pos=state.defer_pos + 1)
                defer_alive = True
            else:
                state = replace(state, defer_pos=0)
        required = self._requires_selection(state)
        if required:
            pos = len(state.selection or ())
            if token != self._selection_tokens[pos]:
                return None
        label: tuple[int, ...] | None = None
        complete: list[tuple[str, int]] = []
        if state.typed is not None:
            complete = [
                (item, row) for _, item, row in self._candidates(state, state.typed, exact=True)
            ]
            if (
                not state.observed
                and next(self._candidates(state, (*state.typed, token)), None) is not None
            ):
                label = (*state.typed, token)

        observes = (
            (self._observe or self._menu)
            and not state.observed
            and (state.selecting or not self._answers_path(state))
        )
        selection, selection_label = None, state.selection_label
        if self._observe_select and not state.selecting and not state.selected:
            prefix = state.selection
            if (
                prefix is None
                and state.observed
                and complete
                and (required or not self._residual_goal_stop(state))
                and self._observed_frontier(state, state.typed)
            ):
                prefix, selection_label = (), state.typed
            if prefix is not None and token == self._selection_tokens[len(prefix)]:
                selection = (*prefix, token)
                if selection == self._selection_tokens:
                    return replace(
                        state,
                        segment=Segment.SUBJECT,
                        typed=(),
                        observed=False,
                        selecting=True,
                        defer_pos=1 if required and state.defer_count < self._observe_defer else 0,
                        selection=None,
                        selection_label=selection_label,
                        selection_segment=state.segment,
                        selection_frontier=tuple(
                            self._observed_frontier(state, selection_label).items()
                        ),
                        observation=(),
                        observation_offset=0,
                        closing=None,
                        separating=None,
                        escaping=None,
                        observing=None,
                    )
        observing, observing_label = None, state.observing_label
        if observes:
            marker = self._observation_pieces[0]
            prefix = state.observing
            if prefix is None and complete:
                prefix, observing_label = (), state.typed
            if prefix is not None and token == marker[len(prefix)]:
                observing = (*prefix, token)
                if observing == marker:
                    if (
                        self._observe_no_loop
                        and state.segment is Segment.RELATION
                        and not state.selecting
                    ):
                        followed = frozenset(
                            r
                            for tokens, r, _ in self._candidates(state, observing_label, exact=True)
                            if tokens == observing_label
                        )
                        state = replace(
                            state, loop_history=(*state.loop_history, (state.subjects, followed))
                        )
                    if state.selecting:
                        named = {
                            e
                            for tokens, e, _ in self._candidates(state, observing_label, exact=True)
                            if tokens == observing_label
                        }
                        state = replace(
                            state,
                            selecting=False,
                            selected=True,
                            defer_pos=0,
                            selection_frontier=tuple(
                                (e, rank) for e, rank in state.selection_frontier if e in named
                            ),
                            segment=state.selection_segment,
                            typed=state.selection_label,
                        )
                        observing_label = state.selection_label
                    payload = self._observation(state, observing_label) if self._observe else ()
                    menu_payload = (
                        self._menu_display(state, observing_label)
                        if self._menu and not self._menu_close_next
                        else ()
                    )
                    return replace(
                        state,
                        typed=observing_label,
                        observed=True,
                        observation=payload,
                        observation_offset=0,
                        audit_phase=6 if self._menu and not payload else 0,
                        audit_payload=menu_payload,
                        audit_offset=0,
                        observing=None,
                        observing_label=(),
                        closing=None,
                        separating=None,
                        escaping=None,
                        operating=None,
                    )

        closes = (state.segment is not Segment.RELATION or self._paths) and complete
        if (
            observes
            or state.selecting
            or (required and (self._observe_defer or self._menu_close_next))
        ):
            closes = False
        if closes and state.segment is Segment.OBJECT and not self._repeats:
            closes = (*state.written, state.typed) not in state.facts
        if closes and state.operator:  # a comparison closes where it has a result
            closes = state.segment is Segment.RELATION and self._result(state, state.typed)
        closing, closing_label = self._branch(
            state.closing, state.closing_label, data, self._closer, closes, state.typed
        )
        operands = 2 if state.operator else 1  # the subject labels a path or comparison writes
        separates = (
            state.segment is Segment.SUBJECT
            and not self._answers_path(state)
            and any(e in self._graph(state).has_triples for e, _ in complete)
        ) or (
            state.segment is Segment.RELATION
            and complete
            and (
                not self._paths
                or (
                    len(state.written) - operands < MAX_PATH_RELATIONS - 1
                    and (
                        self._relations != "frontier"
                        or all(
                            any(
                                e in self._graph(state).has_triples
                                for e in self._frontier(state, state.typed, k)
                            )
                            for k in range(operands)
                        )
                    )
                    and (
                        self._relations != "successor" or self._successors_after(state, state.typed)
                    )
                )
            )
        )
        forced = not state.selecting and bool(complete) and self._residual_goal_stop(state)
        if self._observe_no_loop and state.observed and separates:
            separates = self._loop_goes_on(state, self._observed_frontier(state, state.typed))
        if forced or observes or state.selecting or required:
            separates = False
        # Operand A of a comparison is followed by the operand separator instead.
        operand_a = state.operator and state.segment is Segment.SUBJECT and not state.written
        separator = self._operand_separator if operand_a else self._separator
        separating, separating_label = self._branch(
            None if forced else state.separating,
            state.separating_label,
            data,
            separator,
            separates,
            state.typed,
        )
        operating = state.operating + data if state.operating is not None else None
        if operating is None and self._may_operate(state):
            operating = data
        if operating is not None and not any(op.startswith(operating) for op in self._operators):
            operating = None
        escaping, _ = self._branch(
            None if forced else state.escaping,
            (),
            data,
            self._escape,
            not required
            and not (forced and not observes)
            and self._escapes
            and state.typed is not None
            and self._may_escape(state),
            state.typed,
        )

        if closing == self._closer and state.operator:  # a comparison ends
            committed, result = self._result(state, closing_label)
            rows = self._rows_for(result)
            return OracleState(
                Phase.FREE,
                last_span=ClosedSpan(
                    (*state.written, closing_label),
                    tuple(sorted(result)),
                    (),
                    True,
                    COMPARISONS[state.operator - 1],
                    committed,
                ),
                facts=state.facts,
                reached=frozenset(rows[:MAX_REACHED]),
                version=state.version,
                stop=state.stop,
                audit_scores=state.audit_scores,
                audit_alias_ranks=state.audit_alias_ranks,
            )
        if closing == self._closer and (
            state.segment is Segment.RELATION or state.selected
        ):  # a path ends
            if self._relations != "frontier":
                return OracleState(
                    Phase.FREE,
                    last_span=ClosedSpan((*state.written, closing_label), (), (), False),
                    facts=state.facts,
                    version=state.version,
                    stop=state.stop,
                    audit_scores=state.audit_scores,
                    audit_alias_ranks=state.audit_alias_ranks,
                )
            frontier = self._observed_frontier(state, closing_label)
            if self._audit:
                plan = (*state.written, closing_label)
                state = replace(
                    state,
                    audit_phase=1,
                    audit_plan=plan,
                    audit_frontier=tuple(frontier.items()),
                    audit_penultimate=state.ranks,
                    typed=None,
                    closing=None,
                    separating=None,
                    escaping=None,
                )
                choices = [0, 1, 3]
                if len(plan) - 1 < MAX_PATH_RELATIONS and self._audit_available(state, frontier):
                    choices.append(2)
                if len(plan) > 2:
                    choices.append(4)
                return replace(
                    state,
                    audit_choices=tuple(choices),
                    audit_payload=self._audit_display(state, menus=True),
                )
            if self._committed_paths:  # what the best start with a non-empty path reaches
                best = min(frontier.values())
                frontier = {e: rank for e, rank in frontier.items() if rank == best}
            rows = self._rows_for(frontier)
            result = OracleState(
                Phase.FREE,
                last_span=ClosedSpan(
                    (*state.written, closing_label),
                    tuple(sorted(frontier)),
                    (),
                    True,
                    path=state.segment is Segment.SUBJECT,
                ),
                facts=state.facts,
                reached=frozenset(rows[:MAX_REACHED]),
                version=state.version,
                stop=state.stop,
                audit_scores=state.audit_scores,
                audit_alias_ranks=state.audit_alias_ranks,
                answer_pos=1 if self._observe_answer or self._observe_terminal else 0,
            )
            if self._observe_terminal:
                # The answer-row cap has already run; aliases count as one endpoint.
                deliverable = {self._entity_rows[r][1] for r in result.reached}
                if len(deliverable) > 1:
                    result = replace(
                        result,
                        audit_phase=5,
                        audit_frontier=tuple((e, 0) for e in sorted(deliverable)),
                    )
                    result = replace(result, audit_payload=self._audit_display(result, menus=False))
            return result
        if closing == self._closer:
            facts, reached = state.facts, state.reached
            if state.segment is Segment.OBJECT:
                if len(facts) < MAX_FACTS:
                    facts = (*facts, (*state.written, closing_label))
                if self._derivation:
                    reached = self._reached(state, closing_label)
            return OracleState(
                Phase.FREE,
                last_span=self._resolve(state, closing_label),
                answer_complete=self._end_after_answer and self._answers_path(state),
                facts=facts,
                reached=reached,
                version=state.version,
                stop=state.stop,
                audit_scores=state.audit_scores,
                audit_alias_ranks=state.audit_alias_ranks,
            )
        if separating == separator:
            return self._next_segment(state, separating_label)
        if escaping == self._escape:
            return OracleState(
                Phase.ESCAPED,
                last_span=state.last_span,
                facts=state.facts,
                reached=state.reached,
                version=state.version,
                stop=state.stop,
                audit_scores=state.audit_scores,
                audit_alias_ranks=state.audit_alias_ranks,
            )
        if operating in self._operators:  # operand A follows
            return OracleState(
                Phase.SPAN,
                last_span=state.last_span,
                facts=state.facts,
                reached=state.reached,
                version=state.version,
                stop=state.stop,
                audit_scores=state.audit_scores,
                audit_alias_ranks=state.audit_alias_ranks,
                operator=self._operators.index(operating) + 1,
            )
        alive = (label, closing, separating, escaping, operating, observing, selection)
        if not defer_alive and all(branch is None for branch in alive):
            return None
        return replace(
            state,
            typed=label,
            closing=closing,
            closing_label=closing_label,
            separating=separating,
            separating_label=separating_label,
            escaping=escaping,
            operating=operating,
            observing=observing,
            observing_label=observing_label if observing is not None else (),
            selection=selection,
            selection_label=selection_label,
        )

    def _after_observation(self, state):
        if not self._menu:
            return state
        if not self._menu_close_next:
            return replace(state, audit_phase=6)
        if self._requires_selection(state):
            return state
        if state.segment is Segment.SUBJECT:
            return replace(
                state,
                audit_phase=6,
                audit_payload=self._menu_display(state, state.typed),
                audit_offset=0,
            )
        frontier = self._observed_frontier(state, state.typed)
        can_next = (
            len(state.written) < MAX_PATH_RELATIONS
            and not self._residual_goal_stop(state)
            and self._loop_goes_on(state, frontier)
        )
        return replace(
            state, decision_phase=1, decision_pos=0, decision_choices=3 if can_next else 1
        )

    def _decision_step(self, state, token):
        phase, pos = state.decision_phase, state.decision_pos
        if phase == 2:
            choices = 0
            for command in range(2):
                if not state.decision_choices & (1 << command):
                    continue
                piece = self._decision_pieces[command + 1]
                if token != piece[pos]:
                    continue
                choices |= 1 << command
                if pos + 1 == len(piece):
                    result = replace(state, decision_phase=3 + command, decision_pos=0)
                    if command == 1:
                        result = replace(
                            result,
                            audit_phase=6,
                            audit_offset=0,
                            audit_payload=self._menu_display(state, state.typed),
                        )
                    return result
            return (
                replace(state, decision_choices=choices, decision_pos=pos + 1) if choices else None
            )
        piece = self._decision_pieces[0 if phase == 1 else phase]
        if token != piece[pos]:
            return None
        if phase == 1:
            return (
                replace(state, decision_pos=pos + 1)
                if pos + 1 < len(piece)
                else replace(state, decision_phase=2, decision_pos=0)
            )
        result = self._span_step(replace(state, decision_phase=0), token, self._bytes[token])
        if result is not None and pos + 1 < len(piece):
            result = replace(result, decision_phase=phase, decision_pos=pos + 1)
        return result

    def _audit_available(self, state, frontier):
        from collections import Counter

        return Counter(r for s, r, _ in self._graph(state).edges if s in frontier)

    def _menu_display(self, state, label):
        frontier = self._observed_frontier(state, label)
        counts = self._audit_available(state, frontier)
        candidates = self._relation_rows
        if self._observe_no_loop:
            next_state = replace(
                state,
                segment=Segment.RELATION,
                subjects=frozenset(frontier),
                observed=False,
                selecting=False,
            )
            candidates = [(tokens, owner) for tokens, owner, _ in self._candidates(next_state)]
        allowed = set(candidates)
        writable = {row for row, item in enumerate(self._relation_rows) if item in allowed}
        priorities = state.audit_alias_ranks or self._audit_terms.default_alias_ranks
        display = {}
        for relation in counts:
            row = self._audit_terms.display_row(relation, priorities, writable)
            if row >= 0:
                display[relation] = self._relation_rows[row][0]
        counts = {r: n for r, n in counts.items() if r in display}
        ranks = dict(state.audit_scores)
        relations = sorted(
            counts, key=lambda r: (ranks.get(r, 64), -counts[r], self._relation_numbers[r])
        )
        pieces, out = self._menu_pieces, []
        if self._observe:
            out.extend(pieces[0])
        for n, relation in enumerate(relations[: self._menu]):
            if n:
                out.extend(pieces[4])
            out.extend(display[relation])
        if not relations:
            out.extend(pieces[5])
        more = len(relations) - self._menu
        if more > 0:
            out.extend(pieces[6 + more])
        out.extend(pieces[6])
        return tuple(out)

    def _audit_display(self, state, *, menus):
        from xgraph.audit import (
            CLOSE,
            COUNT,
            DECISION,
            EMPTY,
            EXECUTED,
            JOIN,
            PENULTIMATE,
            TERMINAL,
        )

        pieces = self._audit_pieces
        ordered = sorted(
            dict(state.audit_frontier),
            key=lambda e: (
                -self._audit_sitelinks[e],
                -self._graph(state).degree.get(e, 0),
                self._first_row[e],
            ),
        )
        out = []

        def display(heading, rows, total):
            out.extend(pieces[heading])
            for n, row in enumerate(rows):
                if n:
                    out.extend(pieces[JOIN])
                out.extend(row)
            if not rows:
                out.extend(pieces[EMPTY])
            more = total - len(rows)
            if more:
                out.extend(pieces[COUNT + more - 1])
            out.extend(pieces[CLOSE])

        shown = ordered[: self._observe_terminal or self._audit_entities]
        if self._observe_terminal:
            from xgraph.audit import permute_terminal

            shown = permute_terminal(shown, self._terminal_order, self._terminal_seed)
        display(
            EXECUTED,
            [self._entity_rows[self._first_row[e]][0] for e in shown],
            len(ordered),
        )
        if menus and self._audit_relations:
            ranks = dict(state.audit_scores)
            for heading, frontier in (
                (TERMINAL, state.audit_frontier),
                (PENULTIMATE, state.audit_penultimate),
            ):
                counts = self._audit_available(state, dict(frontier))
                relations = sorted(
                    counts, key=lambda r: (ranks.get(r, 64), -counts[r], self._relation_numbers[r])
                )
                display(
                    heading,
                    [
                        self._relation_rows[
                            self._audit_terms.display_row(
                                r, state.audit_alias_ranks or self._audit_terms.default_alias_ranks
                            )
                        ][0]
                        for r in relations[: self._audit_relations]
                    ],
                    len(relations),
                )
        if menus:
            out.extend(pieces[DECISION])
        return tuple(out)

    def _audit_step(self, state, token):
        from xgraph.audit import DROP, KEEP, REPLACE, SELECT

        if state.audit_phase in (1, 5):
            if token != state.audit_payload[state.audit_offset]:
                return None
            offset = state.audit_offset + 1
            phase = (
                (2 if state.audit_phase == 1 else 0)
                if offset == len(state.audit_payload)
                else state.audit_phase
            )
            return replace(state, audit_offset=offset, audit_phase=phase)
        if state.audit_phase == 2:
            prefix = (*state.audit_command, token)
            choices = [
                c for c in state.audit_choices if self._audit_commands[c][: len(prefix)] == prefix
            ]
            if not choices:
                return None
            complete = next((c for c in choices if self._audit_commands[c] == prefix), None)
            if complete is None:
                return replace(state, audit_choices=tuple(choices), audit_command=prefix)
            if complete in (KEEP, DROP):
                return self._audit_finish(replace(state, audit_action=complete))
            return replace(
                state,
                audit_phase=3 if complete == SELECT else 4,
                audit_action=complete,
                typed=(),
                closing=None,
            )
        frontier = dict(
            state.audit_frontier if state.audit_action != REPLACE else state.audit_penultimate
        )
        if state.audit_action == SELECT:
            rows = [(tokens, owner) for tokens, owner in self._entity_rows if owner in frontier]
        else:
            available = self._audit_available(state, frontier)
            rows = [(tokens, owner) for tokens, owner in self._relation_rows if owner in available]
        typed = state.typed
        label = None
        if typed is not None and any(
            tokens[: len(typed) + 1] == (*typed, token) for tokens, _ in rows
        ):
            label = (*typed, token)
        complete = typed is not None and any(tokens == typed for tokens, _ in rows)
        data = self._bytes[token]
        if data is None:
            return None
        closing, closing_label = self._branch(
            state.closing, state.closing_label, data, self._closer, complete, typed
        )
        if closing == self._closer:
            return self._audit_finish(state, closing_label)
        if label is None and closing is None:
            return None
        return replace(state, typed=label, closing=closing, closing_label=closing_label)

    def _audit_finish(self, state, label=()):
        from xgraph.audit import APPEND, DROP, REPLACE, SELECT

        plan, action = state.audit_plan, state.audit_action
        frontier = dict(state.audit_frontier)
        if action == SELECT:
            named = {owner for tokens, owner in self._entity_rows if tokens == label}
            frontier = {e: rank for e, rank in frontier.items() if e in named}
        elif action == DROP:
            frontier, plan = dict(state.audit_penultimate), plan[:-1]
        elif action in (APPEND, REPLACE):
            source = frontier if action == APPEND else dict(state.audit_penultimate)
            walk = replace(state, subjects=frozenset(source), ranks=tuple(source.items()))
            frontier = self._frontier(walk, label)
            plan = (*plan, label) if action == APPEND else (*plan[:-1], label)
        if self._committed_paths:
            best = min(frontier.values())
            committed = {e for e, rank in frontier.items() if rank == best}
        else:
            committed = set(frontier)
        rows = [row for row, (_, owner) in enumerate(self._entity_rows) if owner in committed]
        result = OracleState(
            Phase.FREE,
            last_span=ClosedSpan(plan, tuple(sorted(committed)), (), True),
            reached=frozenset(rows[:MAX_REACHED]),
            facts=state.facts,
            version=state.version,
            stop=state.stop,
            audit_scores=state.audit_scores,
            audit_alias_ranks=state.audit_alias_ranks,
            answer_pos=1,
        )
        if action in (APPEND, REPLACE, DROP):
            result = replace(result, audit_phase=5, audit_frontier=tuple(frontier.items()))
            result = replace(result, audit_payload=self._audit_display(result, menus=False))
        return result

    def _requires_selection(self, state: OracleState) -> bool:
        return bool(
            self._observe_required
            and state.observed
            and not state.selected
            and not state.selecting
            and len(
                self._observed_frontier(
                    state, state.selection_label if state.selection is not None else state.typed
                )
            )
            > 1
        )

    def _loop_blocked(self, state: OracleState, frontier) -> set[str]:
        if not self._observe_no_loop:
            return set()
        key = frozenset(frontier)
        return {
            r for previous, relations in state.loop_history if previous == key for r in relations
        }

    def _loop_goes_on(self, state: OracleState, frontier) -> bool:
        have = {r for s, r, _ in self._graph(state).edges if s in frontier}
        blocked = self._loop_blocked(state, frontier) & have
        bad = {tokens for tokens, r in self._relation_rows if r in blocked}
        return any(r in have and tokens not in bad for tokens, r in self._relation_rows)

    def _observed_frontier(self, state: OracleState, label: tuple[int, ...]) -> dict[str, int]:
        if state.selected or state.selecting:
            return dict(state.selection_frontier)
        if state.segment is Segment.SUBJECT:
            candidates = [
                (e, r)
                for tokens, e, r in self._candidates(state, label, exact=True)
                if tokens == label and e in self._graph(state).has_triples
            ]
            starts = self._best_connected(state, candidates)
            return self._kept({e: rank for rank, e in enumerate(starts)})
        return self._frontier(state, label)

    def _observation(self, state: OracleState, label: tuple[int, ...]) -> tuple[int, ...]:
        """Independently rank the capped execution frontier; keep the execution state."""
        frontier = self._observed_frontier(state, label)
        ordered = self._best_connected(state, [(e, self._first_row[e]) for e in frontier])
        tokens: list[int] = []
        width = len(ordered) if len(ordered) <= self._observe_width else self._observe
        for n, entity in enumerate(ordered[:width]):
            if n:
                tokens.extend(self._observation_pieces[1])
            tokens.extend(self._entity_rows[self._first_row[entity]][0])
        if not ordered:
            tokens.extend(self._observation_pieces[2])
        more = len(ordered) - width
        if more > 0:
            tokens.extend(self._observation_pieces[more + 2])
        return tuple(tokens)

    def _successors_after(self, state: OracleState, label: tuple[int, ...]) -> set[str]:
        return set().union(
            *(
                self._successors[state.version].get(r, set())
                for tokens, r, _ in self._candidates(state, label, exact=True)
                if tokens == label
            )
        )

    def _residual_goal_stop(self, state: OracleState) -> bool:
        goal = state.stop
        if goal is None or not goal.head or state.operator or state.segment is not Segment.RELATION:
            return False
        followed = {
            self._relation_numbers[r] for tokens, r in self._relation_rows if tokens == state.typed
        }
        if not followed & goal.head:
            return False

        def text(tokens):
            return (
                b"".join(self._bytes[t] for t in tokens)
                .decode("utf-8", "replace")
                .removeprefix(self._label_prefix)
            )

        used = goal.used_by_start(text(state.written[0]))
        for label in (*state.written[1:], state.typed):
            used |= goal.used_by_relations(
                self._relation_numbers[r] for tokens, r in self._relation_rows if tokens == label
            )
        frontier = self._frontier(state, state.typed)
        return not any(
            goal.coverage.get(self._relation_numbers[r], 0) & ~used
            for subject, r, _ in self._graph(state).edges
            if subject in frontier
        )

    @staticmethod
    def _branch(
        typed: bytes | None,
        label: tuple[int, ...],
        data: bytes,
        delimiter: bytes,
        may_start: bool,
        current: tuple[int, ...] | None,
    ) -> tuple[bytes | None, tuple[int, ...]]:
        """A delimiter branch after this token: its bytes and the label it started after."""
        if typed is not None:
            typed += data
        elif may_start:
            typed, label = data, current or ()
        if typed is None or not delimiter.startswith(typed):
            return None, ()
        return typed, label

    def _resolve(self, state: OracleState, label: tuple[int, ...]) -> ClosedSpan:
        rows = [(item, row) for _, item, row in self._candidates(state, label, exact=True)]
        if state.segment is Segment.SUBJECT and self._derivation and state.reached is not None:
            return ClosedSpan((label,), (self._best_connected(state, rows)[0],), (), True)
        if state.segment is Segment.SUBJECT:
            return ClosedSpan((label,), tuple(sorted(e for e, _ in rows)), (), True)
        objects = {row: entity for entity, row in rows}
        triples = {
            (s, r, objects[o])
            for s, r, o in self._graph(state).edges
            if s in state.subjects and r in state.relations and o in objects
        }
        return ClosedSpan((*state.written, label), (), tuple(sorted(triples)), True)

    def _may_operate(self, state: OracleState) -> bool:
        """Expressions: an operator may start here, at the start of an unbound span (before
        any label token, which only the first token of its subject segment sees)."""
        return (
            self._expressions
            and state.segment is Segment.SUBJECT
            and not state.operator
            and state.typed == ()
            and not state.reached
        )

    def _result(
        self, state: OracleState, label: tuple[int, ...]
    ) -> tuple[tuple[str, str], frozenset[str]] | None:
        """A comparison closed after this relation label: the start entities its operands
        commit to and its result (the module docstring), or None where it cannot close."""
        if COMPARISONS[state.operator - 1] == "same":
            return self._same(state, label)
        fronts = [self._frontier(state, label, k) for k in (0, 1)]
        if not all(any(e in self._dates for e in front) for front in fronts):
            return None
        return self._compare(state, label)

    def _same(
        self, state: OracleState, label: tuple[int, ...]
    ) -> tuple[tuple[str, str], frozenset[str]] | None:
        """A value comparison: each operand's best start with any value, and the verdict."""
        reached = []
        for k in (0, 1):
            front = self._frontier(state, label, k)
            if not front:
                return None
            best = min(front.values())
            reached.append((state.starts[k][best], {e for e, r in front.items() if r == best}))
        (a, values_a), (b, values_b) = reached
        entities_a = {e for e in values_a if e not in self._dates}
        entities_b = {e for e in values_b if e not in self._dates}
        if bool(entities_a) != bool(entities_b):  # values of two kinds
            return None
        dates_a = {self._dates[e] for e in values_a if e in self._dates}
        dates_b = {self._dates[e] for e in values_b if e in self._dates}
        verdict = "yes" if entities_a & entities_b or dates_a & dates_b else "no"
        return (a, b), frozenset({self._verdicts[verdict]})

    def _compare(
        self, state: OracleState, label: tuple[int, ...]
    ) -> tuple[tuple[str, str], frozenset[str]]:
        """A comparison closed after this relation label: the start entity each operand
        commits to (its best start that reaches a date), and the result (the module
        docstring)."""
        picked: list[tuple[str, Date]] = []
        for k in (0, 1):
            dated = {e: r for e, r in self._frontier(state, label, k).items() if e in self._dates}
            best = min(dated.values())
            earliest = min(
                (self._dates[e] for e, r in dated.items() if r == best),
                key=lambda d: (d.lo, d.hi),
            )
            picked.append((state.starts[k][best], earliest))
        (a, date_a), (b, date_b) = picked
        if date_a.before(date_b):
            earlier, later = {a}, {b}
        elif date_b.before(date_a):
            earlier, later = {b}, {a}
        else:  # undetermined: equal or overlapping dates bind both
            earlier = later = {a, b}
        result = earlier if COMPARISONS[state.operator - 1] == "earlier" else later
        return (a, b), frozenset(result)

    def _may_escape(self, state: OracleState) -> bool:
        """Without bound escapes, a subject the binding restricts (an anchor, a path's
        frontier, a fact span's objects) cannot escape: it names what the KG reached."""
        if state.selecting or (
            (self._observe_answer or self._audit or self._observe_terminal)
            and self._answers_path(state)
        ):
            return False
        bound = state.segment is Segment.SUBJECT and self._derivation and bool(state.reached)
        return self._bound_escapes or not bound

    def _best_connected(self, state: OracleState, rows: list[tuple[str, int]]) -> list[str]:
        """The entities of these (entity, label index row) pairs, best-connected first: most
        adjacency entries (one per triple and object label row), then first row; with the
        sitelink rank, most sitelinks before either."""
        degree, sitelinks = self._graph(state).degree, self._sitelinks
        ordered = sorted(
            rows,
            key=lambda pair: (-sitelinks.get(pair[0], 0), -degree.get(pair[0], 0), pair[1]),
        )
        return list(dict.fromkeys(entity for entity, _ in ordered))

    def _kept(self, ranks: dict[str, int]) -> dict[str, int]:
        """Path mode: the first MAX_REACHED frontier entities, by first label index row, or
        with committed paths by start rank and then first label index row."""
        if self._committed_paths:
            order = sorted(ranks, key=lambda e: (ranks[e], self._first_row[e]))
        else:
            order = sorted(ranks, key=self._first_row.get)
        return {e: ranks[e] for e in order[:MAX_REACHED]}

    def _frontier(
        self, state: OracleState, label: tuple[int, ...], operand: int = 0
    ) -> dict[str, int]:
        """Path mode: the objects of the frontier (state.subjects, whose ranks state.ranks
        holds; with expressions operand B's, state.ranks_b) under every relation with the
        label, each with the least start rank of the entities it is reached from (the kept
        ones, see _kept)."""
        if state.selected:
            return dict(state.selection_frontier)
        relations = {r for tokens, r in self._relation_rows if tokens == label}
        start = dict(state.ranks_b if operand else state.ranks)
        subjects = frozenset(start) if operand else state.subjects
        ranks: dict[str, int] = {}
        for s, r, o in self._graph(state).from_subjects(subjects):
            if s in subjects and r in relations:
                entity = self._entity_rows[o][1]
                ranks[entity] = min(ranks.get(entity, start[s]), start[s])
        return self._kept(ranks)

    def _answers_path(self, state: OracleState) -> bool:
        """Path mode: the span right after a closed path (its answer), an entity span."""
        last = state.last_span
        return self._paths and last is not None and (len(last.labels) > 1 or last.path)

    def _reached(self, state: OracleState, label: tuple[int, ...]) -> frozenset[int]:
        """The label rows of the entities the span reaches (derivation): objects with
        the written label of the bound subject under the span's relations."""
        rows = {row for _, _, row in self._candidates(state, label, exact=True)}
        entities = {
            self._entity_rows[o][1]
            for s, r, o in self._graph(state).edges
            if s in state.subjects and r in state.relations and o in rows
        }
        reached = [row for row, (_, e) in enumerate(self._entity_rows) if e in entities]
        return frozenset(reached[:MAX_REACHED])

    def _next_segment(self, state: OracleState, label: tuple[int, ...]) -> OracleState:
        candidates = [(item, row) for _, item, row in self._candidates(state, label, exact=True)]
        rows = [item for item, _ in candidates]
        written = (*state.written, label)
        if self._relations != "frontier":
            return OracleState(
                Phase.SPAN,
                segment=Segment.RELATION,
                relations=frozenset(rows) if state.segment is Segment.RELATION else frozenset(),
                written=written,
                last_span=state.last_span,
                facts=state.facts,
                reached=state.reached,
                version=state.version,
                stop=state.stop,
                audit_scores=state.audit_scores,
                audit_alias_ranks=state.audit_alias_ranks,
            )
        if state.segment is Segment.SUBJECT:
            subjects = frozenset(e for e in rows if e in self._graph(state).has_triples)
            ranks: dict[str, int] = {}
            if self._paths:  # every sense: the relations written decide which go on
                starts = self._best_connected(
                    state, [(e, r) for e, r in candidates if e in subjects]
                )
                ranks = self._kept({e: rank for rank, e in enumerate(starts)})
                if state.selected:
                    ranks = dict(state.selection_frontier)
                subjects = frozenset(ranks)
            if state.operator:  # a comparison: rank this operand's starts
                ranked = tuple(sorted(ranks, key=ranks.get))
                if not state.written:  # operand A: operand B follows
                    return OracleState(
                        Phase.SPAN,
                        written=written,
                        last_span=state.last_span,
                        facts=state.facts,
                        reached=state.reached,
                        version=state.version,
                        stop=state.stop,
                        audit_scores=state.audit_scores,
                        audit_alias_ranks=state.audit_alias_ranks,
                        operator=state.operator,
                        starts=(ranked,),
                        ranks=tuple(sorted(ranks.items())),
                    )
                return OracleState(  # operand B: both frontiers go on in lock step
                    Phase.SPAN,
                    segment=Segment.RELATION,
                    subjects=frozenset(dict(state.ranks)),
                    written=written,
                    last_span=state.last_span,
                    facts=state.facts,
                    reached=state.reached,
                    version=state.version,
                    stop=state.stop,
                    audit_scores=state.audit_scores,
                    audit_alias_ranks=state.audit_alias_ranks,
                    ranks=state.ranks,
                    operator=state.operator,
                    starts=(*state.starts, ranked),
                    ranks_b=tuple(sorted(ranks.items())),
                )
            if not self._paths and self._derivation and subjects:
                best = self._best_connected(state, [(e, r) for e, r in candidates if e in subjects])
                subjects = frozenset(best[:1])
            return OracleState(
                Phase.SPAN,
                segment=Segment.RELATION,
                subjects=subjects,
                written=written,
                last_span=state.last_span,
                facts=state.facts,
                reached=state.reached,
                version=state.version,
                stop=state.stop,
                audit_scores=state.audit_scores,
                audit_alias_ranks=state.audit_alias_ranks,
                ranks=tuple(sorted(ranks.items())),
                loop_history=state.loop_history,
                defer_count=state.defer_count,
            )
        if self._paths:  # the next hop: the objects become the frontier
            frontier = self._frontier(state, label)
            other = self._frontier(state, label, 1) if state.operator else {}
            return OracleState(
                Phase.SPAN,
                segment=Segment.RELATION,
                subjects=frozenset(frontier),
                written=written,
                last_span=state.last_span,
                facts=state.facts,
                reached=state.reached,
                version=state.version,
                stop=state.stop,
                audit_scores=state.audit_scores,
                audit_alias_ranks=state.audit_alias_ranks,
                ranks=tuple(sorted(frontier.items())),
                operator=state.operator,
                starts=state.starts,
                ranks_b=tuple(sorted(other.items())),
                loop_history=state.loop_history,
                defer_count=state.defer_count,
            )
        return OracleState(
            Phase.SPAN,
            segment=Segment.OBJECT,
            subjects=state.subjects,
            relations=frozenset(rows),
            written=written,
            last_span=state.last_span,
            facts=state.facts,
            reached=state.reached,
            version=state.version,
            stop=state.stop,
            audit_scores=state.audit_scores,
            audit_alias_ranks=state.audit_alias_ranks,
        )


def _suffix(text: bytes, length: int) -> bytes:
    return text[len(text) - length :] if length > 0 else b""
