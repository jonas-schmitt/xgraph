"""Parse-then-execute: the model writes a relation path, the KG executes it.

The strongest baseline for answering through the KG (as GMeLLo and similar editors
do): asked for a path instead of an answer, the model names the question's starting
entity and the chain of relations to follow, `Path: ⟦ The Hobbit⟧ · author · place of
birth · country`; the KG store (xgraph.toolkg) starts from the best-connected entity
with that label or alias, follows each relation over sets of entities, and answers
with the best-connected entity of the final set. An edited KG thus edits the answer
whenever the model's path is right. A path span written without the engine (path mode's prompt,
`⟦ The Hobbit · author · place of birth⟧`) executes the same way: path mode with its
execution moved after decoding.

Observed spans add " ·→" and frontier labels to every segment, including the start.
parse_path and record_path ignore those observations and read exactly the same
start/relation plan, so MatchedExecutor and downstream plan scorers execute the same
path. They do not treat visible witnesses as selected intermediate entities.

    python -m xgraph.programs requests --questions target/mquake/questions.json \\
        --out target/mquake/requests-path.jsonl
    python -m xgraph.programs execute --outputs target/mquake/path.jsonl \\
        --db data/wikidata/mquake-kg.sqlite --out target/mquake/path-executed.jsonl

Matched execution (`--matched`, MatchedExecutor) executes a path as path mode with
committed paths does in the engine, over the KG index and served tokenizer path mode ran
with: labels read as label index rows, a relation label or alias followed as the union
of the relations it names, the start committed in best-connected order (`--start-rank`,
`--unquoted-starts` as the engine's options), frontiers capped at the engine's limit in
its order, and the answer the best connected of the entities the answer span may name.
Two offline repairs the engine never makes are options: `--repair` replaces a relation
the frontier lacks by its most similar one, and `--strip-quotes` reads a start that
names nothing as written without the quotation marks wrapping it (strip_quotes), as in
`⟦ "Go Tell Them" · performer⟧`. A matched record keeps its `plan`, the start and
relation labels the executor read (`{"start": ..., "relations": [...]}`, null where the
path reached nothing), which `python -m xgraph.repair deliver --executed` has the engine
replay so that the model chooses the answer inside the same frontier. A record executes
in its KG version (its own `kg_version`, or its request's from `--requests`) of the
version file `--kg-versions`, matched or not; `<out>.summary.json` counts the paths,
the empty ones, and with `--matched` the stripped starts and the plans:

    python -m xgraph.programs execute --outputs <outputs.jsonl> \\
        --db data/external/mintqa-full/kg.sqlite --matched \\
        --index data/external/mintqa-full/index-qwen2.5 --tokenizer <model>/tokenizer.json \\
        [--repair] [--strip-quotes] \\
        [--kg-versions <dir>/kg-versions --requests <requests.jsonl>] --out <out>

Multiplicity delivery (`deliver --rule multiplicity`) re-delivers path-mode answers: each
written path executes again as matched execution (with the run's `--start-rank` and
`--unquoted-starts`, in the record's KG version), its final frontier's entities are
counted by derivations (MatchedExecutor.derivations), and where the frontier has two or
more entities and the most derivations are more than one, the entity with the most
replaces the model's answer; a tie keeps the model's answer if it is among the tied, and
otherwise goes to the first tied entity in the sitelink rank's best-connected order (in
degree order for an index without sitelink counts), whatever the run's start rank. Each
record keeps its fields and gains `delivery`, `multiplicity` or `model`:

    python -m xgraph.programs deliver --rule multiplicity --outputs <paths.jsonl> \\
        --db data/external/mintqa-full/kg.sqlite \\
        --index data/external/mintqa-full/index-qwen2.5-ua --tokenizer <tokenizer.json> \\
        --start-rank sitelinks --unquoted-starts --out <out>

The reveal arm (`reveal`) continues path-mode answers from their path and the entities
the KG reaches by it, so the model sees the KG's value but writes its answer freely.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

PROGRAM_INSTRUCTIONS = """\
Do not answer the question. Instead, write the path through Wikidata that leads to its \
answer, on one line: "Path:", the entity the question starts from as a span with its \
English Wikidata label, like ⟦ Marie Curie⟧, then the Wikidata properties to follow \
from it in order, separated by " · ". Use the exact labels Wikidata uses."""

PROGRAM_EXAMPLES = [
    {
        "question": "what college did steve jobs attend?",
        "path": "Path: ⟦ Steve Jobs⟧ · educated at",
    },
    {
        "question": "On which continent is the country where Mount Kilimanjaro is?",
        "path": "Path: ⟦ Mount Kilimanjaro⟧ · country · continent",
    },
    {
        "question": "In which country was the author of The Hobbit born?",
        "path": "Path: ⟦ The Hobbit⟧ · author · place of birth · country",
    },
    {
        "question": (
            "What is the capital of the country whose citizen wrote "
            "Harry Potter and the Philosopher's Stone?"
        ),
        "path": (
            "Path: ⟦ Harry Potter and the Philosopher's Stone⟧ · author · "
            "country of citizenship · capital"
        ),
    },
]

_PATH = re.compile(r"^Path:\s*⟦\s*(.+?)\s*⟧(.*)$")
_PATH_SPAN = re.compile(r"⟦([^⟧]*)⟧")


def program_messages(question: str) -> list[dict]:
    messages = [{"role": "system", "content": PROGRAM_INSTRUCTIONS}]
    for example in PROGRAM_EXAMPLES:
        messages.append({"role": "user", "content": example["question"]})
        messages.append({"role": "assistant", "content": example["path"]})
    messages.append({"role": "user", "content": question})
    return messages


def parse_path(text: str) -> tuple[str, list[str]] | None:
    """The starting entity's label and the relations, from the last "Path:" line, or
    else from the first path span (path mode's ⟦ start · relation · relation⟧)."""
    from xgraph.menus import semantic_text
    from xgraph.observations import PathRelations, path_parts

    text = semantic_text(text, keep_decisions=True)

    for line in reversed(text.strip().splitlines()):
        match = _PATH.match(line.strip())
        if match and "\nExecuted: ⟦" not in text:
            start, rest = match.groups()
            segments, selections, defers, decisions = path_parts(start, choices=True)
            if len(segments) > 1 or selections or defers:
                return segments[0], PathRelations(
                    segments[1:], selections, defers=defers, decisions=decisions
                )
            relations = [r.strip() for r in rest.split("·") if r.strip()]
            return start, relations
    for span in _PATH_SPAN.finditer(text):
        segments, selections, defers, decisions = path_parts(span.group(1), choices=True)
        start, *relations = segments
        if not relations and not selections and not defers:
            continue  # an observed start by itself is still an entity span
        from xgraph.audit import read_audit

        relations = read_audit(
            text[span.end() :],
            PathRelations(
                [r for r in relations if r], selections, defers=defers, decisions=decisions
            ),
        )
        return (start, relations) if relations is not None else None
    return None


def record_path(record: dict) -> tuple[str, list[str]] | None:
    """The first complete generated path, or an agent's executed delivery plan.

    A declined or unconfirmed replay has no usable plan.
    A later Path: line cannot replace a closed path span. Generation failures
    remain on the record and are handled by the scorer, even after execution.
    Plain program prompts retain parse_path's last-Path-line convention.
    """
    from xgraph.score import replay_failure

    if replay_failure(record):
        return None
    if "executed_plans" in record or "plan" in record:
        from xgraph.observations import PathRelations

        plan = record.get("plan")
        return (
            (
                plan["start"],
                PathRelations(
                    plan["relations"],
                    plan.get("selections"),
                    plan.get("audit"),
                    defers=plan.get("defers", ()),
                    decisions=plan.get("decisions"),
                ),
            )
            if plan
            else None
        )
    if "actions" in record:
        from xgraph.planning import walked_paths

        path = walked_paths([record]).get(record["id"])
        return (path[0], path[1:]) if path and len(path) > 1 else None
    from xgraph.menus import semantic_text

    text = semantic_text(record.get("text") or "", keep_decisions=True)
    from xgraph.observations import PathRelations, path_parts

    for span in _PATH_SPAN.finditer(text):
        segments, selections, defers, decisions = path_parts(span.group(1), choices=True)
        start, *relations = segments
        if start and (relations or selections or defers) and all(relations):
            from xgraph.audit import read_audit

            relations = read_audit(
                text[span.end() :],
                PathRelations(relations, selections, defers=defers, decisions=decisions),
            )
            return (start, relations) if relations is not None else None
    return parse_path(record.get("text") or "")


def set_execution_fields(record: dict, request: dict, executor, read) -> dict:
    """Deliver the engine's answer support, retaining the singleton as a sensitivity."""
    from xgraph.oracle import MAX_REACHED
    from xgraph.prompts import parse_answer_set

    written = record_path(record)
    result = (
        executor.execute_set(*written, request.get("anchor"), request.get("kg_version"))
        if written
        else {"reached": [], "uncapped_reached": [], "cap_events": []}
    )
    labels = [read.label(int(e[1:])) for e in result["reached"]]
    original = record.get("answers", parse_answer_set(record.get("text", "")))
    ordered = list(dict.fromkeys([a for a in original if a in labels] + labels))
    return {
        "answer_mode": "set",
        "answers": ordered,
        "reached": result["reached"],
        "uncapped_answers": [read.label(int(e[1:])) for e in result["uncapped_reached"]],
        "delivery": {
            "rule": "matched_frontier_set",
            "engine_frontier_cap": MAX_REACHED,
            "cap_events": result["cap_events"],
            "cap_encountered": bool(result["cap_events"]),
            "endpoint_loss": len(set(result["uncapped_reached"]) - set(result["reached"])),
            "raw_answer": original,
            "no_path": written is None,
        },
    }


QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"), ("„", "“"), ("«", "»"))  # noqa: RUF001
"""The (opening, closing) quotation marks strip_quotes removes around a written start:
straight and typographic double and single quotation marks, and guillemets."""


def strip_quotes(text: str) -> str:
    """The text without the paired quotation marks (QUOTE_PAIRS) that wrap all of it,
    repeatedly, and the spaces inside them; unchanged where nothing would be left."""
    while len(text) > 2 and (text[0], text[-1]) in QUOTE_PAIRS and text[1:-1].strip():
        text = text[1:-1].strip()
    return text


REPAIR_SIMILARITY = 0.6
"""The least similarity for repairing an unknown relation: the larger of the labels'
character similarity (difflib ratio) and the overlap of their word sets (Jaccard)."""


def _similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    words_a, words_b = set(a.split()), set(b.split())
    overlap = len(words_a & words_b) / len(words_a | words_b) if words_a | words_b else 0.0
    return max(SequenceMatcher(None, a, b).ratio(), overlap)


def _repair(store, current: list[int], relation: str) -> list[int]:
    """The relation of the current entities whose label is most like the given one."""
    from xgraph.score import normalize

    best, score = None, REPAIR_SIMILARITY
    for prop, label in sorted(store.relations_of(current).items()):
        ratio = _similarity(normalize(label), normalize(relation))
        if ratio >= score:
            best, score = prop, ratio
    return [best] if best is not None else []


def execute(
    store,
    start: str,
    relations: list[str],
    limit: int = 1000,
    repair: bool = False,
    anchor: list[str] | None = None,
    names: dict[int, list[str]] | None = None,
    all_senses: bool = False,
) -> list[int]:
    """The entities the path reaches from the best-connected entity named start, by
    decreasing degree; empty if a step leads nowhere. With repair, a relation the
    current entities do not have is replaced by the one of theirs whose label is most
    alike (at least REPAIR_SIMILARITY). With an anchor (the question's entity ids), the
    path starts from the best-connected of them instead of the entity named start.

    Matched to path mode: names (property number to its labels and aliases, as
    xgraph.wikidata.relation_names keeps them) lets a relation be written by any of
    them, not only its canonical label; all_senses starts from every entity named start
    (or every anchor) rather than the best-connected one."""
    from xgraph.score import normalize

    items = [int(e[1:]) for e in anchor] if anchor else store.items_named(start)
    if not items:
        return []
    current = sorted(items) if all_senses else [max(items, key=lambda i: (store.degree(i), -i))]
    selections = getattr(relations, "selections", {})
    if 0 in selections:
        named_items = set(store.items_named(selections[0]))
        current = [e for e in current if e in named_items]
    named: dict[str, list[int]] = {}
    for prop, labels in (names or {}).items():
        for label in labels:
            named.setdefault(normalize(label), []).append(prop)
    for hop, relation in enumerate(relations, 1):
        reached: dict[int, str] = {}
        props = named.get(normalize(relation), []) if names else store.relation_named(relation)
        for prop in sorted(set(props)):
            reached.update(store.objects_of(current, prop))
        if not reached and repair:
            for prop in _repair(store, current, relation):
                reached.update(store.objects_of(current, prop))
        current = sorted(reached)[:limit]
        if hop in selections:
            named_items = set(store.items_named(selections[hop]))
            current = [e for e in current if e in named_items]
        if not current:
            return []
    return sorted(current, key=lambda i: (-store.degree(i), i))


@dataclass(frozen=True)
class MatchedPath:
    """A written path as path mode's committed paths resolve it (MatchedExecutor)."""

    reached: tuple[str, ...]
    """The entities the path resolves to, best-connected first; empty where the engine
    would not have let the path be written."""
    answer: str | None
    """The entity delivered as the answer: the best connected of those the answer span
    after the path may name."""
    plan: tuple[str, ...] | None = None
    """The path as path mode would write it to reach the same frontier: the start label
    as read and the relation labels followed (the written label or alias, a repaired
    relation's most similar one), each a label of the KG index, at most
    MAX_PATH_RELATIONS relations; None where the path is empty, names no relation, starts
    from anchors its start label does not name, or a repair has no label naming its
    relation alone among the frontier's."""
    stripped: bool = False
    """Whether the start was read without its wrapping quotation marks (strip_quotes)."""
    selections: dict[int, str] | None = None
    """Written selections, indexed by start (0) or completed relation (1..hops)."""
    audit: dict | None = None
    defers: set[int] | None = None
    decisions: dict[int, str] | None = None

    @property
    def recorded_plan(self) -> dict | None:
        if self.plan is None:
            return None
        result = {"start": self.plan[0], "relations": list(self.plan[1:])}
        if self.selections:
            result["selections"] = dict(self.selections)
        if self.audit is not None:
            result["audit"] = dict(self.audit)
        if self.defers:
            result["defers"] = sorted(self.defers)
        if self.decisions:
            result["decisions"] = dict(self.decisions)
        return result


class MatchedExecutor:
    """Parse-then-execute matched to path mode with committed paths (xgraph.oracle, its
    module docstring), over the KG index and served tokenizer the engine ran with, so that
    a path written without the engine executes as the engine would have executed it:

    - labels are read as the engine reads them, as the label index's rows: a start or
      relation is written exactly (after the label prefix), not up to case;
    - a relation label or alias follows every relation it names, the union;
    - the start is every entity the label names that has triples (or, anchored, the
      anchor's entities the label names, and every anchor entity if it names none), ranked
      in best-connected order (most sitelinks first under the sitelink rank, then most
      adjacency entries, then first label index row), the first MAX_REACHED kept; each
      frontier keeps its first MAX_REACHED entities by start rank and then first label
      index row, and the path resolves to the entities of the least start rank present,
      what the best start with a non-empty path reaches;
    - a path names at most MAX_PATH_RELATIONS relations; later ones are dropped, as the
      engine closes the path there;
    - with unquoted starts, an unanchored start that begins with a quotation mark starts
      nothing;
    - every read goes through the KG version the path names (xgraph.kgversions).

    The answer path mode delivers is the entity span the model writes after the path,
    bound to the first MAX_REACHED label rows of the entities reached and resolved to the
    best connected with the label written. Which label greedy decoding writes cannot be
    known offline, so the answer delivered is the best connected of every entity those
    rows name. With repair, a relation the frontier lacks is replaced by the frontier's
    relation with the most similar label or alias (at least REPAIR_SIMILARITY), which
    the engine never does: an offline repair. With strip_quotes, a start that names no
    entity with triples as written (or only labels unquoted starts forbid) is read without
    the quotation marks wrapping it (strip_quotes), a parsing repair the engine never does
    either.

    Each path also reports its plan (MatchedPath.plan), the labels path mode would write
    to reach the same frontier, so that the engine can replay it and let the model choose
    the answer inside that frontier (`python -m xgraph.repair deliver --executed`).
    With committed_paths=False, starts and frontiers use first-row order and delivery
    retains all surviving starts; the default keeps the committed-path contract above.
    An explicit anchor must have writable label rows; None selects unanchored execution.
    """

    def __init__(
        self,
        index,
        tokenizer,
        *,
        versions=None,
        start_rank: str = "degree",
        unquoted_starts: bool = False,
        strip_quotes: bool = False,
        observe_required: bool = False,
        observe_no_loop: bool = False,
        observe_defer: int = 0,
        menu_close_next: bool = False,
        committed_paths: bool = True,
    ) -> None:
        from xgraph.index import IndexMismatch, check_start_rank

        if index.tokenizer_fingerprint != tokenizer.fingerprint:
            raise IndexMismatch("the KG index was built for another tokenizer")
        if type(observe_defer) is not int or not 0 <= observe_defer <= 8:
            raise ValueError("observe_defer must be an integer in [0, 8]")
        if observe_defer and not observe_required:
            raise ValueError("observe_defer needs observe_required")
        self._index, self._tokenizer, self._versions = index, tokenizer, versions
        self._syntax = index.span_syntax
        ranked = check_start_rank(start_rank, index) == "sitelinks"
        self._sitelinks = index.sitelinks if ranked else None
        self._unquoted, self._strip = unquoted_starts, strip_quotes
        self._observe_required, self._observe_no_loop = observe_required, observe_no_loop
        self._observe_defer, self._menu_close_next = observe_defer, menu_close_next
        self._committed_paths = committed_paths
        self._first_rows: dict[int, int] | None = None
        self._names: dict[int, list[str]] | None = None
        self._menu_terms = None
        self._menu_question = None

    def execute(
        self,
        start: str,
        relations: list[str],
        anchor: list[str] | None = None,
        version: str | None = None,
        repair: bool = False,
    ) -> MatchedPath:
        """The path from the start label along the relation labels (see the class)."""
        from xgraph.oracle import MAX_REACHED

        index = self._index
        view = index if version is None else self._view(version)
        owners = index.labels.owners
        read = self._read_start(view, start, anchor)
        stripped = read != start
        walked = self._walk(view, read, relations, anchor, repair)
        if walked is None:  # an entity span: every entity the label names
            starts = self._order(view, [(int(owners[r]), r) for r in self._starts(read, anchor)])
            return self._path(starts, starts[0] if starts else None, None, stripped)
        ranks, rows_of, followed = walked
        if not ranks:
            return MatchedPath((), None, None, stripped)
        best = min(ranks.values())
        final = [e for e in ranks if ranks[e] == best]
        nameable = sorted({row for e in final for row in rows_of[e]})[:MAX_REACHED]
        answer = self._order(view, [(int(owners[r]), r) for r in nameable])[0]
        writable = None not in followed and self._names_start(view, read, anchor)
        plan = (read, *followed) if writable else None
        entities = self._order(view, [(e, min(rows_of[e])) for e in final])
        from dataclasses import replace

        result = self._path(entities, answer, plan, stripped)
        return replace(
            result,
            selections=getattr(relations, "selections", None) or None,
            audit=getattr(relations, "audit", None),
            defers=getattr(relations, "defers", None) or None,
            decisions=getattr(relations, "decisions", None) or None,
        )

    def execute_set(
        self,
        start: str,
        relations: list[str],
        anchor: list[str] | None = None,
        version: str | None = None,
        *,
        witnesses: bool = False,
    ) -> dict:
        """Complete answer support after commitment, with each actual cap encounter.

        The uncapped replay is a CPU sensitivity of this fixed plan. It removes
        start, live-frontier and answer-label-row caps, and may commit differently.
        It cannot restore a relation decision made under online truncation.
        With witnesses, each deliverable endpoint also has one start-to-endpoint
        entity chain through the capped walk. Chains are ranked lexicographically
        by descending sitelink count then numeric entity id at each step.
        """
        from xgraph.oracle import MAX_PATH_RELATIONS, MAX_REACHED

        view = self._index if version is None else self._view(version)
        if witnesses and self._index.sitelinks is None:
            raise ValueError("endpoint witnesses require an index with sitelink counts")
        start = self._read_start(view, start, anchor)
        events: list[dict] = []
        chains = {}
        if len(relations) > MAX_PATH_RELATIONS:
            events.append(
                {"stage": "path_relations", "before": len(relations), "kept": MAX_PATH_RELATIONS}
            )
        if anchor is not None:
            import numpy as np

            index = self._index
            entities = {index._entity_at[e] for e in anchor if e in index._entity_at}
            count = int(np.count_nonzero(np.isin(index.labels.owners, sorted(entities))))
            if not count:
                raise ValueError(f"anchor has no writable label rows: {anchor!r}")
            if count > MAX_REACHED:
                events.append({"stage": "anchor_label_rows", "before": count, "kept": MAX_REACHED})

        def endpoints(uncapped):
            trace = [] if witnesses and not uncapped else None
            walked = self._walk(
                view,
                start,
                relations,
                anchor,
                False,
                trace,
                cap_events=None if uncapped else events,
                uncapped=uncapped,
            )
            if not walked or not walked[0]:
                return []
            ranks, rows_of, _ = walked
            best = min(ranks.values())
            rows = sorted({r for e, rank in ranks.items() if rank == best for r in rows_of[e]})
            if not uncapped and len(rows) > MAX_REACHED:
                events.append(
                    {"stage": "answer_label_rows", "before": len(rows), "kept": MAX_REACHED}
                )
                rows = rows[:MAX_REACHED]
            owners = self._index.labels.owners
            ordered = self._order(view, [(int(owners[r]), r) for r in rows])
            if trace is not None:
                paths = self._endpoint_witnesses(view, start, relations, anchor, trace, best)
                ids = self._index.labels.owner_ids
                chains.update({ids[e]: [ids[p] for p in paths[e]] for e in ordered})
            return [self._index.labels.owner_ids[e] for e in ordered]

        reached = endpoints(False)
        result = {"reached": reached, "uncapped_reached": endpoints(True), "cap_events": events}
        if witnesses:
            result["witnesses"] = chains
        return result

    def _endpoint_witnesses(self, view, start, relations, anchor, trace, best):
        """One chain per endpoint, restricted to the exact capped execution trace."""
        import numpy as np

        from xgraph.oracle import MAX_PATH_RELATIONS

        index = self._index
        ids, owners = index.labels.owner_ids, index.labels.owners
        paths = {
            e: (e,)
            for rank, e in enumerate(self._start_entities(view, start, anchor))
            if not self._committed_paths or rank == best
        }
        selection = getattr(relations, "selections", {}).get(0)
        if selection is not None:
            named = self._rows_owned(index.labels, selection)
            paths = {e: p for e, p in paths.items() if e in named}

        def key(entity):
            return tuple((-int(index.sitelinks[e]), int(ids[e][1:])) for e in paths[entity])

        for written, kept in zip(relations[:MAX_PATH_RELATIONS], trace, strict=True):
            follow = sorted(self._rows_owned(index.relations, written))
            reached = {}
            # The first parent of a child supplies its lexicographically best
            # chain. Only nodes retained with the winning start rank may join it.
            for entity in sorted(paths, key=key):
                relation_of, object_of = view.edges_at(entity)
                for row in np.asarray(object_of)[np.isin(relation_of, follow)].tolist():
                    item = int(owners[row])
                    if kept.get(item) == best:
                        reached.setdefault(item, (*paths[entity], item))
            paths = reached
        return paths

    def starts(
        self, start: str, anchor: list[str] | None = None, version: str | None = None
    ) -> tuple[str, ...]:
        """The entities a path from this start starts from, in best-connected order (see
        the class): those the label names that have triples, the first MAX_REACHED."""
        view = self._index if version is None else self._view(version)
        ids = self._index.labels.owner_ids
        return tuple(ids[e] for e in self._start_entities(view, start, anchor))

    def derivations(
        self,
        start: str,
        relations: list[str],
        anchor: list[str] | None = None,
        version: str | None = None,
        rank: str | None = None,
    ) -> tuple[tuple[str, int], ...]:
        """The entities the path resolves to (MatchedPath.reached), each with its
        derivations: the distinct relation paths that reach it from the start it resolves
        from, one relation and one entity of each step's frontier per step. A step following
        a label that names two relations, both linking the same entities, doubles the count.
        Empty where the path is empty or names no relation. The entities come in
        best-connected order under rank (`degree`, or `sitelinks` for an index with sitelink
        counts), by default the executor's start rank."""
        import numpy as np

        from xgraph.index import check_start_rank
        from xgraph.oracle import MAX_PATH_RELATIONS

        index = self._index
        view = index if version is None else self._view(version)
        owners = index.labels.owners
        read = self._read_start(view, start, anchor)
        frontiers: list[dict[int, int]] = []
        walked = self._walk(view, read, relations, anchor, False, frontiers)
        if not walked or not walked[0]:
            if walked is None and anchor is not None:
                self._bound(anchor)
            return ()
        ranks, rows_of, _ = walked
        best = min(ranks.values())
        counts = {self._start_entities(view, read, anchor)[best]: 1}
        for written, kept in zip(relations[:MAX_PATH_RELATIONS], frontiers, strict=True):
            follow = sorted(self._rows_owned(index.relations, written))
            reached: dict[int, int] = {}
            for entity, n in counts.items():
                relation_of, object_of = view.edges_at(entity)
                chosen = np.isin(relation_of, follow)
                steps = zip(
                    np.asarray(relation_of)[chosen].tolist(),
                    np.asarray(object_of)[chosen].tolist(),
                    strict=True,
                )
                for _, item in {(r, int(owners[row])) for r, row in steps}:
                    if item in kept:
                        reached[item] = reached.get(item, 0) + n
            counts = reached
        final = [e for e in ranks if ranks[e] == best]
        ids = index.labels.owner_ids
        sitelinks = self._sitelinks
        if rank is not None:
            ranked = check_start_rank(rank, index) == "sitelinks"
            sitelinks = index.sitelinks if ranked else None
        ordered = self._order_by(view, [(e, min(rows_of[e])) for e in final], sitelinks)
        return tuple((ids[e], counts.get(e, 0)) for e in ordered)

    def named(self, label: str) -> set[str]:
        """The entities a label written exactly as this text names (label index rows)."""
        ids, owners = self._index.labels.owner_ids, self._index.labels.owners
        return {ids[int(owners[row])] for row in self._rows(self._index.labels, label)}

    def frontier_relations(
        self,
        start: str,
        relations: list[str],
        anchor: list[str] | None = None,
        version: str | None = None,
    ) -> list[tuple[str, int]]:
        """The relations path mode offers after the path written so far: every relation of
        its whole frontier (all start ranks, as the engine offers them), as (relation id,
        the frontier's edges with it), most edges first, then index order; empty where the
        path is empty or has as many relations as a path may name."""
        import numpy as np

        from xgraph.oracle import MAX_PATH_RELATIONS

        if len(relations) >= MAX_PATH_RELATIONS:
            if anchor is not None:
                self._bound(anchor)
            return []
        view = self._index if version is None else self._view(version)
        start = self._read_start(view, start, anchor)
        if relations or getattr(relations, "selections", {}):
            ranks = self._walk(view, start, relations, anchor, False)[0]
        else:
            ranks = dict.fromkeys(self._start_entities(view, start, anchor), 0)
        counts: dict[int, int] = {}
        for entity in ranks:
            found, times = np.unique(view.edges_at(entity)[0], return_counts=True)
            for relation, n in zip(found.tolist(), times.tolist(), strict=True):
                counts[relation] = counts.get(relation, 0) + n
        ids = self._index.relations.owner_ids
        return [(ids[r], counts[r]) for r in sorted(counts, key=lambda r: (-counts[r], r))]

    def frontier(
        self,
        start: str,
        relations: list[str],
        anchor: list[str] | None = None,
        version: str | None = None,
    ) -> tuple[str, ...]:
        """The capped live frontier before committing to one start at delivery.

        Starts have starts() order; later entities keep the engine's order by start
        rank and first label row. This retains every surviving start sense, including
        those that do not yet win the commitment in execute().reached.
        """
        view = self._index if version is None else self._view(version)
        start = self._read_start(view, start, anchor)
        if (
            not relations
            and not getattr(relations, "selections", {})
            and not getattr(relations, "defers", set())
        ):
            return self.starts(start, anchor, version)
        ranks = self._walk(view, start, relations, anchor, False)[0]
        ids = self._index.labels.owner_ids
        return tuple(ids[e] for e in ranks)

    def frontier_labels(
        self,
        start: str,
        relations: list[str],
        anchor: list[str] | None = None,
        version: str | None = None,
    ) -> list[str]:
        """The writable relation labels and aliases of the whole path frontier,
        sorted and deduplicated. Read the index, including its label exclusions,
        rather than the KG store's independent relation names."""
        have = {prop for prop, _ in self.frontier_relations(start, relations, anchor, version)}
        if not have:
            return []
        return self.relation_labels(have)

    def set_relation_labels(self, canonical: dict[str, str]):
        """Supply WT's canonical display names for legacy indexes, without changing them."""
        from dataclasses import replace

        from xgraph.audit import RelationScores
        from xgraph.index import primary_relation_rows

        index = self._index
        if index.relation_primary_rows is None:
            index = replace(
                index,
                relation_primary_rows=primary_relation_rows(
                    index.relations, self._tokenizer, self._syntax, canonical
                ),
            )
        self._menu_terms = RelationScores(index, self._tokenizer)
        self._menu_question = None

    def ranked_relation_labels(
        self, start, relations, question, anchor=None, version=None, *, width=5
    ):
        """WT's relation-id display rank and writable canonical/fallback labels.

        Only the staged variable-stride interface consumes this bounded menu;
        frontier_labels and ordinary matched execution retain their support.
        """
        from xgraph.audit import RelationScores

        if self._menu_terms is None:
            self._menu_terms = RelationScores(self._index, self._tokenizer)
        terms = self._menu_terms
        if self._menu_question != question:
            self._menu_question = question
            self._menu_ranks = dict(terms.for_question(question))
            self._menu_priorities = terms.alias_ranks(question)
        available = self.frontier_relations(start, relations, anchor, version)
        available.sort(
            key=lambda item: (self._menu_ranks.get(item[0], 64), -item[1], terms.owner_at[item[0]])
        )
        rows = [terms.display_row(r, self._menu_priorities) for r, _ in available]
        labels = [terms.names[row] for row in rows if row >= 0]
        return labels[:width], max(0, len(labels) - width)

    def frontier_observation(self, frontier, version=None, *, observe=3, width=8):
        """WT's adaptive-width first-row labels, in best-connected order."""
        import numpy as np

        index = self._index
        view = index if version is None else self._view(version)
        if self._first_rows is None:
            entities, first = np.unique(index.labels.owners, return_index=True)
            self._first_rows = dict(zip(entities.tolist(), first.tolist(), strict=True))
        ordered = self._order(
            view, [(index._entity_at[e], self._first_rows[index._entity_at[e]]) for e in frontier]
        )
        labels = []
        limit = len(ordered) if len(ordered) <= width else observe
        for entity in ordered[:limit]:
            tokens = index.labels.row_tokens(self._first_rows[entity])
            text = b"".join(self._tokenizer.token_bytes(t) for t in tokens).decode()
            labels.append(text.removeprefix(self._syntax.label_prefix))
        return labels, max(0, len(ordered) - limit)

    def entity_relation_labels(self, entities: set[str], version: str | None = None) -> list[str]:
        """Writable outgoing labels of reached entities, reading the indexed KG version."""
        view = self._index if version is None else self._view(version)
        at, ids = self._index._entity_at, self._index.relations.owner_ids
        have = {
            ids[int(r)] for entity in entities if entity in at for r in view.edges_at(at[entity])[0]
        }
        return self.relation_labels(have)

    def relation_labels(self, relations: set[str] | None = None) -> list[str]:
        """The writable relation vocabulary, or only the named relation ids' labels."""
        if self._names is None:
            indexed, prefix = self._index.relations, self._syntax.label_prefix
            self._names = {}
            for row in range(indexed.num_rows):
                text = b"".join(self._tokenizer.token_bytes(t) for t in indexed.row_tokens(row))
                name = text.decode("utf-8", errors="replace").removeprefix(prefix)
                self._names.setdefault(int(indexed.owners[row]), []).append(name)
        ids = self._index.relations.owner_ids
        return sorted(
            {
                name
                for r, labels in self._names.items()
                if relations is None or ids[r] in relations
                for name in labels
            }
        )

    def _starts(self, start: str, anchor: list[str] | None) -> list[int]:
        """The label rows the start may name (see the class)."""
        return self._written_starts(start, anchor) or (
            self._bound(anchor) if anchor is not None else []
        )

    def _written_starts(self, start: str, anchor: list[str] | None) -> list[int]:
        """The label rows the start label names that may start the path: the anchor's
        when anchored, none that unquoted starts forbid otherwise."""
        rows = self._rows(self._index.labels, start)
        if anchor is not None:
            bound = set(self._bound(anchor))
            return [r for r in rows if r in bound]
        if self._unquoted and self._syntax.quoted((self._syntax.label_prefix + start).encode()):
            return []
        return rows

    def _bound(self, anchor: list[str]) -> list[int]:
        """The first MAX_REACHED label rows of the anchor's entities."""
        import numpy as np

        from xgraph.oracle import MAX_REACHED

        at = self._index._entity_at
        entities = [at[e] for e in anchor if e in at]
        rows = np.flatnonzero(np.isin(self._index.labels.owners, entities))[:MAX_REACHED].tolist()
        if not rows:
            raise ValueError(f"anchor has no writable label rows: {anchor!r}")
        return rows

    def _names_start(self, view, start: str, anchor: list[str] | None) -> bool:
        """Whether the start label itself names an entity with triples the path may
        start from."""
        owners = self._index.labels.owners
        return any(
            len(view.edges_at(int(owners[r]))[0]) for r in self._written_starts(start, anchor)
        )

    def _read_start(self, view, start: str, anchor: list[str] | None) -> str:
        """The start as read: with strip_quotes, a start that names nothing to start from
        is read without its wrapping quotation marks where that names something."""
        if self._strip and not self._names_start(view, start, anchor):
            unquoted = strip_quotes(start)
            if unquoted != start and self._names_start(view, unquoted, anchor):
                return unquoted
        return start

    def _start_entities(
        self,
        view,
        start: str,
        anchor: list[str] | None,
        *,
        uncapped: bool = False,
        cap_events: list[dict] | None = None,
    ) -> list[int]:
        """A path's starts: the entities the start may name that have triples, in
        best-connected order, the first MAX_REACHED."""
        from xgraph.oracle import MAX_REACHED

        owners = self._index.labels.owners
        rows = self._starts(start, anchor)
        if uncapped and anchor is not None:
            import numpy as np

            entities = {self._index._entity_at[e] for e in anchor if e in self._index._entity_at}
            bound = np.flatnonzero(np.isin(owners, sorted(entities))).tolist()
            written = set(self._rows(self._index.labels, start))
            rows = [r for r in bound if r in written] or bound
        pairs = [(int(owners[r]), r) for r in rows]
        ordered = self._order(view, [(e, r) for e, r in pairs if len(view.edges_at(e)[0])])
        if not self._committed_paths:
            import numpy as np

            if self._first_rows is None:
                entities, first = np.unique(owners, return_index=True)
                self._first_rows = dict(zip(entities.tolist(), first.tolist(), strict=True))
            ordered.sort(key=self._first_rows.__getitem__)
        if cap_events is not None and len(ordered) > MAX_REACHED:
            cap_events.append({"stage": "starts", "before": len(ordered), "kept": MAX_REACHED})
        return ordered if uncapped else ordered[:MAX_REACHED]

    def _walk(
        self,
        view,
        start: str,
        relations: list[str],
        anchor: list[str] | None,
        repair: bool,
        trace: list[dict[int, int]] | None = None,
        *,
        cap_events: list[dict] | None = None,
        uncapped: bool = False,
    ) -> tuple[dict[int, int], dict[int, list[int]], list[str | None]] | None:
        """The path's last frontier, each entity with its start rank, the label rows that
        reached each, and the relation label each step followed (None for a repair without
        a label of its own, see _repair); None for an entity span (no relations), an empty
        frontier where a step leads nowhere. A trace list receives each step's frontier."""
        import numpy as np

        from xgraph.oracle import MAX_PATH_RELATIONS, MAX_REACHED

        if (audit := getattr(relations, "audit", None)) is not None:
            from xgraph.audit import edited_relations

            original = audit["original_relations"]
            expected = edited_relations(original, audit)
            if list(relations) != list(expected) or relations.selections != expected.selections:
                raise ValueError("audited plan does not match its recorded decision")
            blind = self._walk(view, start, original, anchor, False)
            if blind is None or not blind[0]:
                raise ValueError("audit's blind path does not execute")

        if (self._observe_required or self._observe_no_loop or self._menu_close_next) and len(
            relations
        ) > MAX_PATH_RELATIONS:
            raise ValueError(f"observed path exceeds the {MAX_PATH_RELATIONS}-relation cap")
        if (
            not relations
            and not getattr(relations, "selections", {})
            and not getattr(relations, "defers", set())
        ):
            if self._observe_required and len(self._start_entities(view, start, anchor)) > 1:
                raise ValueError("required frontier selection missing at boundary 0")
            return None
        index = self._index
        owners = index.labels.owners
        ranks = {
            e: rank if self._committed_paths else 0
            for rank, e in enumerate(
                self._start_entities(view, start, anchor, uncapped=uncapped, cap_events=cap_events)
            )
        }
        selections = getattr(relations, "selections", {})
        defers = getattr(relations, "defers", set())
        decisions = getattr(relations, "decisions", {})
        if defers & selections.keys() or any(
            h < 0 or h > len(relations) for h in defers | selections.keys()
        ):
            raise ValueError("invalid or conflicting frontier selection/DEFER boundary")
        if self._observe_required and len(defers) > self._observe_defer:
            raise ValueError("path exceeds observe_defer limit")
        for hop, command in decisions.items():
            if (
                not 1 <= hop <= len(relations)
                or command not in ("CLOSE", "NEXT")
                or (command == "CLOSE" and hop != len(relations))
            ):
                raise ValueError(f"invalid CLOSE/NEXT decision at boundary {hop}")
        if self._menu_close_next and (
            set(decisions) != set(range(1, len(relations) + 1))
            or (relations and decisions.get(len(relations)) != "CLOSE")
        ):
            raise ValueError("missing CLOSE/NEXT decision")

        def selected(frontier, hop):
            if hop in defers:
                if self._observe_required and len(frontier) <= 1:
                    raise ValueError(f"DEFER needs a required multi-entity boundary: {hop}")
                return frontier
            if hop not in selections:
                if self._observe_required and len(frontier) > 1:
                    raise ValueError(f"required frontier selection missing at boundary {hop}")
                return frontier
            named = self._rows_owned(index.labels, selections[hop])
            if (self._observe_required or self._observe_no_loop) and not named & frontier.keys():
                raise ValueError(
                    f"selection outside frontier at boundary {hop}: {selections[hop]!r}"
                )
            return {e: rank for e, rank in frontier.items() if e in named}

        ranks = selected(ranks, 0)
        loop_pairs: set[tuple[frozenset[int], int]] = set()
        rows_of: dict[int, list[int]] = {}
        if not relations:
            rows_of = {e: np.flatnonzero(owners == e).tolist() for e in ranks}
        followed: list[str | None] = []
        for written in relations[:MAX_PATH_RELATIONS]:
            follow, label = self._rows_owned(index.relations, written), written
            if repair and not follow & (have := self._have(view, ranks)):
                follow, label = self._repair(have, written)
            if self._observe_no_loop:
                key = frozenset(ranks)
                applied = follow & self._have(view, ranks)
                if any((key, r) in loop_pairs for r in applied):
                    raise ValueError(
                        f"loop guard: repeated relation at hop {len(followed) + 1}: {written!r}"
                    )
                loop_pairs.update((key, r) for r in applied)
            followed.append(label)
            reached: dict[int, int] = {}
            rows_of = {}
            for entity, rank in ranks.items():
                relation_of, object_of = view.edges_at(entity)
                chosen = np.isin(relation_of, sorted(follow))
                for row in np.asarray(object_of)[chosen].tolist():
                    item = int(owners[row])
                    reached[item] = min(reached.get(item, rank), rank)
                    rows_of.setdefault(item, []).append(row)
            kept = sorted(reached, key=lambda e: (reached[e], min(rows_of[e])))
            if not uncapped:
                if cap_events is not None and len(kept) > MAX_REACHED:
                    cap_events.append(
                        {
                            "stage": "frontier",
                            "hop": len(followed),
                            "before": len(kept),
                            "kept": MAX_REACHED,
                        }
                    )
                kept = kept[:MAX_REACHED]
            ranks = selected({e: reached[e] for e in kept}, len(followed))
            if trace is not None:
                trace.append(ranks)
            if not ranks:
                return {}, {}, followed
        return ranks, rows_of, followed

    def _path(
        self, entities: list[int], answer: int | None, plan: tuple[str, ...] | None, stripped: bool
    ) -> MatchedPath:
        ids = self._index.labels.owner_ids
        delivered = None if answer is None else ids[answer]
        return MatchedPath(tuple(ids[e] for e in entities), delivered, plan, stripped)

    def _view(self, version: str):
        if self._versions is None:
            raise ValueError(f"a path reads KG version {version!r}, but no versions were given")
        return self._versions.view(self._index, version)

    def _rows(self, labels, text: str) -> list[int]:
        """The label rows written exactly as text (after the label prefix)."""
        tokens = self._tokenizer.encode(self._syntax.label_prefix + text)
        if not text.strip() or not tokens:
            return []
        lo, hi = labels.prefix_range(tokens)
        rows = []
        for row in range(lo, hi):  # a row that ends sorts before the rows continuing it
            if int(labels.offsets[row + 1] - labels.offsets[row]) != len(tokens):
                break
            rows.append(row)
        return rows

    def _rows_owned(self, labels, text: str) -> set[int]:
        return {int(labels.owners[row]) for row in self._rows(labels, text)}

    def _order(self, view, pairs: list[tuple[int, int]]) -> list[int]:
        """The entities of these (entity, label row) pairs in best-connected order."""
        return self._order_by(view, pairs, self._sitelinks)

    @staticmethod
    def _order_by(view, pairs: list[tuple[int, int]], sitelinks) -> list[int]:
        """The same, most of these sitelink counts first (with None, degree first)."""

        def key(pair: tuple[int, int]) -> tuple[int, int, int]:
            entity, row = pair
            linked = int(sitelinks[entity]) if sitelinks is not None else 0
            return (-linked, -len(view.edges_at(entity)[0]), row)

        return list(dict.fromkeys(entity for entity, _ in sorted(pairs, key=key)))

    @staticmethod
    def _have(view, frontier: dict[int, int]) -> set[int]:
        import numpy as np

        found: set[int] = set()
        for entity in frontier:
            found.update(np.unique(view.edges_at(entity)[0]).tolist())
        return found

    def _repair(self, have: set[int], written: str) -> tuple[set[int], str | None]:
        """The relation of the frontier with the label or alias most like the one written,
        and its most similar label or alias that names no other relation of the frontier
        (so that path mode, which follows every relation a label names, follows it alone),
        None if it has none."""
        from xgraph.score import normalize

        if self._names is None:
            relations, prefix = self._index.relations, self._syntax.label_prefix
            self._names = {}
            for row in range(relations.num_rows):
                text = b"".join(self._tokenizer.token_bytes(t) for t in relations.row_tokens(row))
                name = text.decode("utf-8", errors="replace").removeprefix(prefix)
                self._names.setdefault(int(relations.owners[row]), []).append(name)
        best, score = None, REPAIR_SIMILARITY
        for relation in sorted(have):
            for name in self._names.get(relation, ()):
                ratio = _similarity(normalize(name), normalize(written))
                if ratio >= score:
                    best, score = relation, ratio
        if best is None:
            return set(), None
        names = sorted(
            self._names[best], key=lambda n: -_similarity(normalize(n), normalize(written))
        )
        alone = (n for n in names if self._rows_owned(self._index.relations, n) & have == {best})
        return {best}, next(alone, None)


def multiplicity_answer(frontier: tuple[tuple[str, int], ...], said: set[str]) -> str | None:
    """Multiplicity delivery's choice among a frontier of (entity, derivations) pairs in
    best-connected order (MatchedExecutor.derivations), given the entities the model's
    answer names: on a frontier of two or more entities whose most derivations are more
    than one, the entity with the most, the first of those tied; None, keeping the
    model's answer, otherwise, or where the model's answer is among them."""
    if len(frontier) < 2:
        return None
    top = max(n for _, n in frontier)
    tied = [entity for entity, n in frontier if n == top]
    if top <= 1 or said & set(tied):
        return None
    return tied[0]


def with_answer(text: str, label: str | None) -> str:
    """The text without its Answer lines, then this answer as an answer span, if any."""
    written = "\n".join(line for line in text.splitlines() if not line.startswith("Answer:"))
    return written + (f"\nAnswer: ⟦ {label}⟧" if label else "")


def multiplicity_records(
    records: list[dict],
    matched: MatchedExecutor,
    store,
    version_of: dict[str, str | None],
    rank: str = "sitelinks",
) -> list[dict]:
    """Each record, its path re-executed by matched execution in its KG version, with the
    answer multiplicity delivery chooses (multiplicity_answer) in place of the model's
    where it chooses one, `delivery: multiplicity`, and as it is otherwise, `delivery:
    model`. Ties go by best-connected order under rank, whatever the executor's start
    rank: the deliver command uses the sitelink rank wherever the index has sitelink
    counts, as the rule was defined."""
    from xgraph.prompts import final_answer

    rows = []
    for record in records:
        path, chosen = record_path(record), None
        if path and path[1]:
            version = version_of[record["id"]]
            frontier = matched.derivations(*path, version=version, rank=rank)
            said = final_answer(record["text"])
            chosen = multiplicity_answer(frontier, matched.named(said) if said else set())
        if chosen is None:
            rows.append({**record, "delivery": "model"})
            continue
        label = store.label(int(chosen[1:]))
        if label is None:
            raise ValueError(
                f"the KG store has no label for {chosen}, which multiplicity delivery "
                "answers: build the store from the KG directory the index was built from"
            )
        rows.append(
            {**record, "text": with_answer(record["text"], label), "delivery": "multiplicity"}
        )
    return rows


def _versions_of(records: list[dict], requests: Path | None) -> dict[str, str | None]:
    """Each record's KG version: its own kg_version, or else its request's."""
    asked = {}
    if requests:
        lines = requests.read_text().splitlines()
        asked = {r["id"]: r for r in (json.loads(line) for line in lines if line)}
    return {
        r["id"]: r.get("kg_version") or asked.get(r["id"], {}).get("kg_version") for r in records
    }


def _deliver(args: argparse.Namespace, store, parser: argparse.ArgumentParser, timer) -> None:
    """The deliver command: records and `<out>.summary.json`."""
    from xgraph.index import load_kg_index
    from xgraph.kgversions import load_kg_versions
    from xgraph.provenance import provenance
    from xgraph.tokenizer import ServedTokenizer

    records = [json.loads(line) for line in args.outputs.read_text().splitlines() if line]
    version_of = _versions_of(records, args.requests)
    named = sorted({v for v in version_of.values() if v})
    if named and not args.kg_versions:
        parser.error(f"records read KG versions {named[:5]}: deliver needs --kg-versions")
    index = load_kg_index(args.index)
    matched = MatchedExecutor(
        index,
        ServedTokenizer.from_file(args.tokenizer),
        versions=load_kg_versions(args.kg_versions, index) if named else None,
        start_rank=args.start_rank,
        unquoted_starts=args.unquoted_starts,
    )
    rank = "sitelinks" if index.sitelinks is not None else "degree"
    rows = multiplicity_records(records, matched, store, version_of, rank)
    with args.out.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "artifact_kind": "xgraph.programs.deliver",
        "version": 1,
        "rule": args.rule,
        "records": len(rows),
        "delivered": sum(row["delivery"] == args.rule for row in rows),
        "start_rank": args.start_rank,
        "unquoted_starts": args.unquoted_starts,
        "tie_rank": rank,
    }
    inputs = [args.outputs, args.index / "kg-index.json", args.tokenizer]
    summary["provenance"] = provenance(inputs + ([args.requests] if args.requests else []))
    summary.update(timer.summary())
    Path(f"{args.out}.summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}))


REVEALED = 8
"""The most reached entities a reveal shows, best connected first."""


def reveal_requests(outputs: list[dict], requests: list[dict], store) -> list[dict]:
    """The reveal arm: each request continued from the path its output wrote (path mode)
    and the entities the KG reaches by it (executed from the request's anchor), so the
    model sees the KG's value and writes its answer unconstrained."""
    by_id = {r["id"]: r for r in requests}
    revealed = []
    for output in outputs:
        request, text = by_id[output["id"]], output["text"]
        span = _PATH_SPAN.search(text)
        written = span.group(0) + "." if span else text.split("Answer:", 1)[0].strip()
        parsed = parse_path(text)
        reached = (
            execute(store, parsed[0], parsed[1], anchor=request.get("anchor") or None)
            if parsed
            else []
        )
        names = [f"⟦ {store.label(e)}⟧" for e in reached[:REVEALED] if store.label(e)]
        gives = "gives: " + ", ".join(names) + "." if names else "gives nothing."
        prefill = {"role": "assistant", "content": f"{written}\nThe knowledge graph {gives}\n"}
        revealed.append({**request, "messages": [*request["messages"], prefill]})
    return revealed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m xgraph.programs", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    requests = commands.add_parser("requests", help="chat requests asking for paths")
    requests.add_argument("--questions", type=Path, required=True, help="questions (JSON)")
    requests.add_argument("--out", type=Path, required=True, help="requests (JSONL)")
    run = commands.add_parser("execute", help="execute generated paths on a KG store")
    run.add_argument("--outputs", type=Path, required=True, help="xgraph.generate output")
    run.add_argument("--db", type=Path, required=True, help="KG store (xgraph.toolkg)")
    run.add_argument("--out", type=Path, required=True, help="answers (JSONL)")
    run.add_argument("--repair", action="store_true", help="repair unknown relations")
    run.add_argument(
        "--questions", type=Path, help="with --anchor: questions naming their entities (sparql)"
    )
    run.add_argument("--anchor", action="store_true", help="start from the question's entity")
    run.add_argument(
        "--relation-labels",
        type=Path,
        help="matched to path mode: relation labels and aliases (a KG's relation_labels.jsonl)",
    )
    run.add_argument(
        "--all-senses", action="store_true", help="matched to path mode: start from every sense"
    )
    run.add_argument(
        "--matched",
        action="store_true",
        help="execute as path mode's committed paths do, over --index (MatchedExecutor)",
    )
    run.add_argument("--index", type=Path, help="with --matched: the KG index path mode reads")
    run.add_argument(
        "--observe-required",
        action="store_true",
        help="with --matched: require frontier selections",
    )
    run.add_argument(
        "--observe-no-loop",
        action="store_true",
        help="with --matched: reject repeated frontier/relation pairs",
    )
    run.add_argument(
        "--observe-defer",
        type=int,
        default=0,
        help="with --matched and required selection: per-path DEFER limit",
    )
    run.add_argument(
        "--menu-close-next",
        action="store_true",
        help="with --matched: require recorded CLOSE/NEXT decisions",
    )
    run.add_argument("--tokenizer", type=Path, help="with --matched: its served tokenizer")
    run.add_argument(
        "--start-rank",
        choices=("degree", "sitelinks"),
        default="degree",
        help="with --matched: best-connected order, as XGRAPH_START_RANK",
    )
    run.add_argument(
        "--unquoted-starts",
        action="store_true",
        help="with --matched: unquoted starts, as XGRAPH_UNQUOTED_STARTS=1",
    )
    run.add_argument(
        "--strip-quotes",
        action="store_true",
        help="with --matched: read a start that names nothing without its wrapping quotes",
    )
    run.add_argument(
        "--kg-versions",
        type=Path,
        help="a KG version file: a record executes in its kg_version (or its request's)",
    )
    run.add_argument("--requests", type=Path, help="the outputs' requests, for their kg_version")
    give = commands.add_parser("deliver", help="re-deliver path-mode answers by a rule")
    give.add_argument("--rule", choices=("multiplicity",), required=True, help="the delivery rule")
    give.add_argument("--outputs", type=Path, required=True, help="path-mode records (JSONL)")
    give.add_argument("--db", type=Path, required=True, help="KG store (xgraph.toolkg)")
    give.add_argument("--index", type=Path, required=True, help="the KG index path mode read")
    give.add_argument("--tokenizer", type=Path, required=True, help="its served tokenizer")
    give.add_argument(
        "--start-rank",
        choices=("degree", "sitelinks"),
        default="degree",
        help="best-connected order, as XGRAPH_START_RANK",
    )
    give.add_argument("--unquoted-starts", action="store_true", help="as XGRAPH_UNQUOTED_STARTS=1")
    give.add_argument(
        "--kg-versions",
        type=Path,
        help="a KG version file: a record executes in its kg_version (or its request's)",
    )
    give.add_argument("--requests", type=Path, help="the outputs' requests, for their kg_version")
    give.add_argument("--out", type=Path, required=True, help="records (JSONL)")
    show = commands.add_parser("reveal", help="requests continuing a path with what the KG gives")
    show.add_argument("--outputs", type=Path, required=True, help="path-mode output (JSONL)")
    show.add_argument("--requests", type=Path, required=True, help="their requests (JSONL)")
    show.add_argument("--db", type=Path, required=True, help="KG store (xgraph.toolkg)")
    show.add_argument("--out", type=Path, required=True, help="reveal requests (JSONL)")
    args = parser.parse_args(argv)
    if args.command == "requests":
        questions = json.loads(args.questions.read_text())
        with args.out.open("w", encoding="utf-8") as out:
            for q in questions:
                record = {"id": q["id"], "messages": program_messages(q["utterance"])}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{len(questions)} requests")
        return
    from xgraph.cost import StageTimer
    from xgraph.toolkg import KGStore

    timer = StageTimer()
    store = KGStore(args.db)
    if args.command == "reveal":
        outputs = [json.loads(line) for line in args.outputs.read_text().splitlines()]
        asked = [json.loads(line) for line in args.requests.read_text().splitlines()]
        with args.out.open("w", encoding="utf-8") as out:
            for request in reveal_requests(outputs, asked, store):
                out.write(json.dumps(request, ensure_ascii=False) + "\n")
        print(f"{len(outputs)} requests")
        return
    if args.command == "deliver":
        _deliver(args, store, parser, timer)
        return
    anchors = {}
    if args.anchor:
        anchors = {
            q["id"]: re.findall(r"wd:(Q\d+)", q.get("sparql", ""))
            for q in json.loads(args.questions.read_text())
        }
    names = None
    if args.relation_labels:
        from xgraph.wikidata import relation_names

        rows = [json.loads(line) for line in args.relation_labels.read_text().splitlines()]
        names = {int(p[1:]): labels for p, labels in relation_names(map(tuple, rows)).items()}
    if args.matched and not (args.index and args.tokenizer):
        parser.error("--matched needs --index and --tokenizer")
    if args.matched and (names or args.all_senses):
        parser.error("--matched reads relation names from the index and starts as path mode")
    if args.strip_quotes and not args.matched:
        parser.error("--strip-quotes needs --matched")
    if (
        args.observe_required or args.observe_no_loop or args.observe_defer or args.menu_close_next
    ) and not args.matched:
        parser.error("observed selection and decision options need --matched")
    if not 0 <= args.observe_defer <= 8 or (args.observe_defer and not args.observe_required):
        parser.error("--observe-defer must be in [0, 8] and needs --observe-required")
    from xgraph.provenance import provenance

    records = [json.loads(line) for line in args.outputs.read_text().splitlines() if line]
    version_of = _versions_of(records, args.requests)
    named = sorted({v for v in version_of.values() if v})
    if named and not args.kg_versions:
        parser.error(f"records read KG versions {named[:5]}: execute needs --kg-versions")
    matched = stores = None
    if args.matched:
        from xgraph.index import load_kg_index
        from xgraph.kgversions import load_kg_versions
        from xgraph.tokenizer import ServedTokenizer

        index = load_kg_index(args.index)
        matched = MatchedExecutor(
            index,
            ServedTokenizer.from_file(args.tokenizer),
            versions=load_kg_versions(args.kg_versions, index) if named else None,
            start_rank=args.start_rank,
            unquoted_starts=args.unquoted_starts,
            strip_quotes=args.strip_quotes,
            observe_required=args.observe_required,
            observe_no_loop=args.observe_no_loop,
            observe_defer=args.observe_defer,
            menu_close_next=args.menu_close_next,
        )
    elif named:
        from xgraph.toolkg import version_stores

        stores = version_stores(store, args.kg_versions, named)
    executed = empty = stripped = planned = 0
    with args.out.open("w", encoding="utf-8") as out:
        for record in records:
            path = record_path(record)
            version = version_of[record["id"]]
            anchor = anchors.get(record["id"])
            answer = None
            if path and matched is not None:
                found = matched.execute(*path, anchor=anchor, version=version, repair=args.repair)
                reached = [int(e[1:]) for e in found.reached]
                answer = store.label(int(found.answer[1:])) if found.answer else None
                if found.answer and answer is None:
                    raise ValueError(
                        f"the KG store {args.db} has no label for {found.answer}, which the "
                        f"KG index {args.index} answers: build the store from the KG "
                        "directory the index was built from"
                    )
            elif path:
                read = stores[version] if version else store
                reached = execute(
                    read,
                    *path,
                    repair=args.repair,
                    anchor=anchor,
                    names=names,
                    all_senses=args.all_senses,
                )
                answer = store.label(reached[0]) if reached else None
            else:
                reached = []
            executed += path is not None
            empty += path is not None and not reached
            # the KG's answer only, not one the model wrote
            row = {**record, "text": with_answer(record["text"], answer), "reached": reached[:20]}
            if matched is not None:
                plan = found.plan if path else None
                stripped += bool(path and found.stripped)
                planned += plan is not None
                row["plan"] = found.recorded_plan if path else None
            out.write(json.dumps(row) + "\n")
    summary = {
        "artifact_kind": "xgraph.programs.execute",
        "version": 1,
        "records": len(records),
        "paths": executed,
        "empty": empty,
        "matched": args.matched,
        "repair": args.repair,
        "strip_quotes": args.strip_quotes,
        **({"observe_required": True} if args.observe_required else {}),
        **({"observe_no_loop": True} if args.observe_no_loop else {}),
        **({"observe_defer": args.observe_defer} if args.observe_defer else {}),
        **({"menu_close_next": True} if args.menu_close_next else {}),
        **({"stripped": stripped, "planned": planned} if args.matched else {}),
    }
    inputs = [args.outputs, *([args.index / "kg-index.json"] if args.matched else [])]
    summary["provenance"] = provenance(inputs + ([args.requests] if args.requests else []))
    summary.update(timer.summary())
    Path(f"{args.out}.summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}))


if __name__ == "__main__":
    main()
