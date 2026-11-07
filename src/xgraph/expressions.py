"""Expressions: paths, date comparisons, and value comparisons over the KG, run on the CPU.

Path mode answers entity chains only (`⟦ start · relation · relation⟧`), which covers
less than half of 2WikiMultihopQA's questions (xgraph.twowiki). Most of the rest ask for
a date at the end of a chain, compare two entities by a date their chains reach ("Which
film has the director born later, X or Y?" names the film), or ask whether two chains
reach a shared value ("Are X and Y of the same nationality?"). This module is the CPU
admission test of a typed language over path spans that also writes those, before any
engine work: its syntax and parser, an exact executor over the KG store, a profile of
the dataset's gold expressions, a few-shot prompt, and a scorer of free generations.

**The language.** Three expressions, each one span in the span syntax:

- a path, `⟦ The Hobbit · author · place of birth⟧`: a start label, then relations; it
  evaluates to the entities or the dates (literal values) the relations reach from the
  start (an entity set or a date set);
- a date comparison, `⟦ earlier: Jaws | Star Wars · director · date of birth⟧` (or
  `later:`): each operand, separated by ` |`, walks the shared relations (the via path)
  to dates, and the comparison returns the operand, the entity it started from, whose
  date is earlier (later);
- a value comparison, `⟦ same: Jaws | Alien · country of origin⟧`: yes when the two
  operands' via paths reach a shared entity or date, no when both reach values and none
  is shared.

A span with the escape marker is an escape (`Escaped`), as in path mode.

**Execution** (`Executor`) reads the KG store (xgraph.toolkg) for entity objects and a
date store (`build_date_store`, below) for date objects. A relation label or alias
names the properties of `Relations`: those with that label, else those with it as an
alias, else xgraph.twowiki's renamed relations, up to case (NFC, case folded), and a
step follows their union. An anchored start (the dataset's QIDs) starts from its
entities; a written start is committed as path mode commits a path: every entity the
label names (up to case, as the KG store's names) in best-connected order (most triples
in the store, then lowest QID), and the first from which the expression's own walk is
non-empty, so that a date comparison preserves the operand it started from. The result
(`Result`) has a status, so that an empty walk is told apart from a name the KG lacks
and from an escape: `value` (entities, dates, the yes or no of a value comparison, or
the operand a date comparison returns), `empty` (the names resolved, the walk reached
nothing), `unbound` (a start label names no entity, or a relation names no property),
`escape`, `undetermined` (a date comparison whose two dates overlap at their precision,
or are equal), and `type` (a relation after a date, or a date comparison of operands
that reach no dates). Store objects numbered as literal values (xgraph.literals) are
left to the date store.

**Date precision.** Truthy dumps keep no precision: a year is stored as its 1 January
and a month as its first day (xgraph.literals). A stored date therefore takes the
coarsest precision its value allows (`stored_date`): 1 January is a year, another first
of a month a month, any other day a day. Comparisons read a date as the interval its
precision covers, and a date is before another only when its interval ends before the
other's starts: 1979 is neither before nor after 15 June 1979. An operand with several
dates is compared by its earliest. A stored date matches a gold date (the dataset's
answer, `parse_date`: "6 June 1814", "June 6, 1814", "June 1814", "1814") when it
agrees on every field the gold states, so 1814 stored as 1 January matches "1814" and
not "6 June 1814".

**The date store** holds the date objects of a KG directory with literal values
(`python -m xgraph.literals`): its triples whose object is a date, with the date's
year, month, and day read from the value's ISO label. The gate KG with literals shares
its item triples with the gate KG streamed with `mul` labels, whose KG store the agent
reads, so the two stores together are the literal KG without a second copy of its
triples in SQLite:

    python -m xgraph.expressions dates --kg data/wikidata/gate-kg-lit \\
        --db data/wikidata/gate-kg-lit-dates.sqlite

**Structures and gold expressions** (`structure`, `gold_expression`). Each
2WikiMultihopQA dev question falls under one structure by its type, answer, and
evidence: `entity_path` (a compositional or inference question with an entity answer),
`date_path` (a compositional question with a date answer), `date_comparison` (a
comparison or bridge comparison whose answer is one of two entities and whose evidence
gives each one date), `yes_no` (a comparison answered yes or no), `date_difference` ("Who
lived longer", both dates of birth and death), and `count` (any other comparison: more
directors, a wider scope of profession). The first four get a gold expression from the
dataset alone, anchored to its QIDs: a path from the evidence chain's QIDs and relation
names; a date comparison of the two entities of `entity_ids` (a comparison's two, a
bridge comparison's first two), via the evidence's relation names, `earlier` when the
question says first, earlier, or older, `later` when it says later, more recently, or
younger; a value comparison of the same two entities via the evidence's relation names.
Count and date difference questions get none and stay in the denominator.

**The profile** (`python -m xgraph.expressions profile`) executes every gold expression
twice: anchored, and written, as its rendering with the store's canonical labels
reads (the starts committed by label), and counts per structure the questions, the
gold expressions built, and those that execute to the gold answer: the gold entity
among the entities a path reaches, a date matching the gold date, the gold operand, or
the gold yes or no. That executable-program coverage, over every dev question and over
Decoding on Graphs' subset, decides whether the language is worth engine work:

    python -m xgraph.expressions profile \\
        --dev data/external/2wikimultihopqa-ids/data_ids/dev.json \\
        --db data/wikidata/gate-kg-v3e.sqlite --dates data/wikidata/gate-kg-lit-dates.sqlite \\
        --relations data/wikidata/gate-kg-v3e/relation_labels.jsonl \\
        --dog data/external/dog-2wiki/release/data/2wiki/2wiki_4000.jsonl \\
        --out target/2wiki-expressions

It writes `gold.jsonl` (each question's structure, gold answer, gold expression or why
none was built, and both executions) and `profile.json` (the counts, with provenance).

**Free generation.** `requests` samples a panel, a number of questions of each of the
four structures with gold expressions, whether one was built or not (seeded, in dataset
order), and writes chat requests with the few-shot prompt (`expression_messages`), which
asks for the expression on a last line `Program: ⟦ ...⟧`; `score` reads each record's
expression (the last such line, else the last span), executes it written, and scores it
as the profile does, over every panel question, a missing or malformed expression
counted wrong, with each outcome counted per structure:

    python -m xgraph.expressions requests --gold target/2wiki-expressions/gold.jsonl \\
        --per-structure 200 --seed 0 --out target/2wiki-expressions/requests-panel.jsonl
    python -m xgraph.expressions score --outputs <records.jsonl> \\
        --gold target/2wiki-expressions/gold.jsonl \\
        --requests target/2wiki-expressions/requests-panel.jsonl \\
        --db data/wikidata/gate-kg-v3e.sqlite --dates data/wikidata/gate-kg-lit-dates.sqlite \\
        --relations data/wikidata/gate-kg-v3e/relation_labels.jsonl --out <records>.scores.json
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path as FilePath

from xgraph.score import record_text
from xgraph.syntax import SpanSyntax

MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
PRECISIONS = ("year", "month", "day")


class ExpressionError(ValueError):
    """A text that is no well-formed expression, with the reason."""


@dataclass(frozen=True)
class Date:
    """A date with the fields its precision states; a stored date (`stored_date`) keeps
    every field of its value and the coarsest precision the value allows."""

    year: int
    month: int | None = None
    day: int | None = None
    precision: str = ""

    def __post_init__(self) -> None:
        if not self.precision:
            given = "day" if self.day else "month" if self.month else "year"
            object.__setattr__(self, "precision", given)
        if self.precision not in PRECISIONS:
            raise ValueError(f"unknown date precision {self.precision!r}")

    @property
    def lo(self) -> tuple[int, int, int]:
        """The first day of the interval the date covers at its precision."""
        if self.precision == "year":
            return (self.year, 1, 1)
        return (self.year, self.month or 1, self.day if self.precision == "day" else 1)

    @property
    def hi(self) -> tuple[int, int, int]:
        """The last day (or a bound past it within the month) the date covers."""
        if self.precision == "year":
            return (self.year, 12, 31)
        return (self.year, self.month or 12, self.day if self.precision == "day" else 31)

    def before(self, other: Date) -> bool:
        """Whether the date's interval ends before the other's starts."""
        return self.hi < other.lo

    def matches(self, gold: Date) -> bool:
        """Whether the date agrees with the gold date on every field the gold states."""
        return (
            self.year == gold.year
            and (gold.month is None or self.month == gold.month)
            and (gold.day is None or self.day == gold.day)
        )

    def __str__(self) -> str:
        era = f"{-self.year} BC" if self.year < 0 else str(self.year)
        if self.precision == "year":
            return era
        if self.precision == "month":
            return f"{MONTHS[self.month - 1]} {era}"
        return f"{self.day} {MONTHS[self.month - 1]} {era}"


def stored_date(year: int, month: int, day: int) -> Date:
    """A date as the KG stores it, at the coarsest precision its value allows."""
    precision = "year" if (month, day) == (1, 1) else "month" if day == 1 else "day"
    return Date(year, month, day, precision)


_MONTH = "|".join(MONTHS)
_GOLD_DATES = (
    (re.compile(rf"(\d{{1,2}}) ({_MONTH}) (\d{{1,4}})"), ("day", "month", "year")),
    (re.compile(rf"({_MONTH}) (\d{{1,2}}), (\d{{1,4}})"), ("month", "day", "year")),
    (re.compile(rf"({_MONTH}) (\d{{1,4}})"), ("month", "year")),
    (re.compile(r"(\d{1,4})"), ("year",)),
)


def parse_date(text: str) -> Date | None:
    """A date as the dataset writes an answer, with the fields it states; None if the
    text is no date in one of its formats."""
    for pattern, fields in _GOLD_DATES:
        found = pattern.fullmatch(text.strip())
        if found:
            parts = dict(zip(fields, found.groups(), strict=True))
            month = MONTHS.index(parts["month"]) + 1 if "month" in parts else None
            day = int(parts["day"]) if "day" in parts else None
            if day is not None and not 1 <= day <= 31:
                return None
            return Date(int(parts["year"]), month, day)
    return None


@dataclass(frozen=True)
class Start:
    """Where a walk starts: a label, and for an anchored start the entities it names."""

    label: str
    items: tuple[int, ...] = ()


@dataclass(frozen=True)
class Path:
    start: Start
    relations: tuple[str, ...]


@dataclass(frozen=True)
class Compare:
    op: str
    """`earlier` or `later`."""
    starts: tuple[Start, Start]
    via: tuple[str, ...]


@dataclass(frozen=True)
class Same:
    starts: tuple[Start, Start]
    via: tuple[str, ...]


@dataclass(frozen=True)
class Escaped:
    text: str


Expression = Path | Compare | Same | Escaped
OPERATORS = ("earlier", "later", "same")
OPERAND_SEPARATOR = " |"
_SYNTAX = SpanSyntax()


def parse(text: str, syntax: SpanSyntax = _SYNTAX) -> Expression:
    """The expression one span writes (the module docstring); ExpressionError if the
    text is not one well-formed span."""
    text = text.strip()
    if not (text.startswith(syntax.opener) and text.endswith(syntax.closer)):
        raise ExpressionError(f"not a span: {text!r}")
    body = text[len(syntax.opener) : len(text) - len(syntax.closer)]
    if syntax.opener in body or syntax.closer in body:
        raise ExpressionError(f"more than one span: {text!r}")
    if syntax.escape.strip() in body:
        return Escaped(body.strip())
    op = None
    head = re.match(r"\s*(\w+):", body)
    if head and head.group(1).casefold() in OPERATORS:
        op = head.group(1).casefold()
        body = body[head.end() :]
    from xgraph.observations import path_segments

    segments = path_segments(body, syntax.separator, syntax.opener, syntax.escape)
    if not all(segments):
        raise ExpressionError(f"an empty segment: {text!r}")
    first, relations = segments[0], tuple(segments[1:])
    if op is None:
        return Path(Start(first), relations)
    operands = [o.strip() for o in first.split(OPERAND_SEPARATOR.strip())]
    if len(operands) != 2 or not all(operands):
        raise ExpressionError(f"{op} takes two operands: {text!r}")
    if not relations:
        raise ExpressionError(f"{op} needs relations to compare by: {text!r}")
    starts = (Start(operands[0]), Start(operands[1]))
    return Same(starts, relations) if op == "same" else Compare(op, starts, relations)


def render(expression: Expression, syntax: SpanSyntax = _SYNTAX) -> str:
    """The span that writes the expression, with its starts' labels."""
    lead = syntax.opener + syntax.label_prefix
    if isinstance(expression, Escaped):
        return f"{syntax.opener} {expression.text}{syntax.closer}"
    if isinstance(expression, Path):
        parts = [expression.start.label, *expression.relations]
    else:
        op = "same" if isinstance(expression, Same) else expression.op
        a, b = (s.label for s in expression.starts)
        parts = [f"{op}:{syntax.label_prefix}{a}{OPERAND_SEPARATOR} {b}", *expression.via]
    return lead + (syntax.separator + syntax.label_prefix).join(parts) + syntax.closer


def find_expression(text: str, syntax: SpanSyntax = _SYNTAX) -> str | None:
    """The span on the last line starting "Program:", or else the last span in the
    text; None if it has none."""
    from xgraph.audit import semantic_text

    text = semantic_text(text, syntax)
    span = (
        re.escape(syntax.opener)
        + r"[^"
        + re.escape(syntax.closer)
        + r"]*"
        + re.escape(syntax.closer)
    )
    lines = re.findall(r"^\s*Program:\s*(" + span + ")", text, re.MULTILINE)
    if lines:
        return lines[-1]
    spans = re.findall(span, text)
    return spans[-1] if spans else None


_ISO = re.compile(r"(-?)(\d{4,})-(\d{2})-(\d{2})")
CHUNK = 1 << 22


def date_values(kg: str | FilePath) -> dict[int, tuple[int, int, int]]:
    """The date values of a KG directory with literal values (xgraph.literals): each
    value's number (the n of its Qn) and its year, month, and day, read from its ISO
    label; quantities, which have none, are left out."""
    import json

    from xgraph.literals import FIRST_ID

    values: dict[int, tuple[int, int, int]] = {}
    with (FilePath(kg) / "entity_labels.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            end = line.index('"', 3)
            if end - 3 < len(str(FIRST_ID)) or int(line[3:end]) < FIRST_ID:
                continue
            item, label, _ = json.loads(line)
            found = _ISO.fullmatch(label)
            if found:
                sign, year, month, day = found.groups()
                values[int(item[1:])] = (-int(year) if sign else int(year), int(month), int(day))
    return values


def build_date_store(kg: str | FilePath, db: str | FilePath) -> dict:
    """The date store of a KG directory with literal values (the module docstring):
    every triple whose object is a date, as (subject, property, year, month, day)."""
    import sqlite3

    import numpy as np

    from xgraph.literals import FIRST_ID

    kg, db = FilePath(kg), FilePath(db)
    values = date_values(kg)
    db.unlink(missing_ok=True)
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE dates (s INTEGER, p INTEGER, year INTEGER, month INTEGER, day INTEGER)"
    )
    triples = np.load(kg / "triples.npy", mmap_mode="r")
    added = 0
    for at in range(0, len(triples), CHUNK):
        chunk = np.asarray(triples[at : at + CHUNK])
        rows = [
            (s, p, *values[o]) for s, p, o in chunk[chunk[:, 2] >= FIRST_ID].tolist() if o in values
        ]
        con.executemany("INSERT INTO dates VALUES (?, ?, ?, ?, ?)", rows)
        added += len(rows)
    con.execute("CREATE INDEX dates_sp ON dates (s, p)")
    con.commit()
    con.close()
    return {"date_values": len(values), "date_triples": added}


class DateStore:
    """The date objects of the KG (build_date_store), read from SQLite."""

    def __init__(self, db: str | FilePath) -> None:
        import sqlite3

        self._con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, check_same_thread=False)

    def dates_of(self, items: list[int], props: set[int]) -> dict[int, list[Date]]:
        """Each item's stored dates under the properties, earliest first; items without
        one are left out."""
        found: dict[int, list[Date]] = {}
        if not items or not props:
            return found
        wanted = sorted(props)
        for at in range(0, len(items), 500):
            batch = list(items[at : at + 500])
            rows = self._con.execute(
                f"SELECT DISTINCT s, year, month, day FROM dates WHERE s IN "
                f"({','.join('?' * len(batch))}) AND p IN ({','.join('?' * len(wanted))}) "
                "ORDER BY s, year, month, day",
                (*batch, *wanted),
            )
            for s, year, month, day in rows.fetchall():
                found.setdefault(s, []).append(stored_date(year, month, day))
        return found


class Relations:
    """Relation names to the properties they name (the module docstring)."""

    def __init__(self, labels: dict[str, set[int]], aliases: dict[str, set[int]]) -> None:
        self.labels, self.aliases = labels, aliases

    @classmethod
    def read(cls, path: str | FilePath) -> Relations:
        """From a KG directory's relation_labels.jsonl."""
        import json

        from xgraph.score import normalize

        labels: dict[str, set[int]] = {}
        aliases: dict[str, set[int]] = {}
        with FilePath(path).open(encoding="utf-8") as lines:
            for line in lines:
                prop, text, kind = json.loads(line)
                named = labels if kind == "label" else aliases
                named.setdefault(normalize(text), set()).add(int(prop[1:]))
        return cls(labels, aliases)

    def props(self, name: str) -> set[int]:
        from xgraph.score import normalize
        from xgraph.twowiki import RENAMED

        key = normalize(name)
        renamed = {normalize(k): v for k, v in RENAMED.items()}
        return set(self.labels.get(key) or self.aliases.get(key) or renamed.get(key, set()))


STATUSES = ("value", "empty", "unbound", "escape", "undetermined", "type")


@dataclass(frozen=True)
class Result:
    """What an expression evaluates to (the module docstring)."""

    status: str
    entities: frozenset[int] = frozenset()
    dates: tuple[Date, ...] = ()
    answer: int | bool | None = None
    """A date comparison's operand entity, or a value comparison's yes (True) or no."""
    starts: tuple[int, ...] = ()
    """The entities the walks started from: a path's one, a comparison's two."""


_FAILED = ("unbound", "type", "empty")
"""Failed statuses, the first of which a failed expression reports."""


class Executor:
    """Exact execution of expressions over a KG store and a date store (the module
    docstring)."""

    def __init__(self, store, dates: DateStore, relations: Relations) -> None:
        self.store, self.dates, self.relations = store, dates, relations
        self._degrees: dict[int, int] = {}

    def _candidates(self, start: Start) -> list[int]:
        if start.items:
            return list(start.items)
        items = self.store.items_named(start.label)
        for item in items:
            if item not in self._degrees:
                self._degrees[item] = self.store.degree(item)
        return sorted(items, key=lambda i: (-self._degrees[i], i))

    def _walk(self, item: int, steps: list[set[int]]) -> Result:
        from xgraph.literals import FIRST_ID

        entities: set[int] = {item}
        dates: list[Date] = []
        for props in steps:
            if not entities:
                return Result("type" if dates else "empty", starts=(item,))
            reached: set[int] = set()
            for prop in sorted(props):
                reached.update(o for o in self.store.objects_of(sorted(entities), prop))
            found = self.dates.dates_of(sorted(entities), props)
            dates = sorted({d for ds in found.values() for d in ds}, key=_date_key)
            entities = {o for o in reached if o < FIRST_ID}
        if not entities and not dates:
            return Result("empty", starts=(item,))
        return Result("value", frozenset(entities), tuple(dates), starts=(item,))

    def _steps(self, relations: tuple[str, ...]) -> list[set[int]] | None:
        steps = [self.relations.props(r) for r in relations]
        return None if not all(steps) else steps

    def _commit(self, start: Start, steps: list[set[int]], dated: bool = False) -> Result:
        """The walk from the start's first candidate whose walk is non-empty (reaches
        dates, if dated), or the first candidate's failure."""
        candidates = self._candidates(start)
        if not candidates:
            return Result("unbound")
        first = None
        for item in candidates:
            walked = self._walk(item, steps)
            if walked.status == "value" and (walked.dates or not dated):
                return walked
            if walked.status == "value":
                walked = Result("type", starts=(item,))
            first = first or walked
        return first

    def run(self, expression: Expression) -> Result:
        if isinstance(expression, Escaped):
            return Result("escape")
        relations = expression.relations if isinstance(expression, Path) else expression.via
        steps = self._steps(relations)
        if steps is None:
            return Result("unbound")
        if isinstance(expression, Path):
            if not steps:
                candidates = self._candidates(expression.start)
                if not candidates:
                    return Result("unbound")
                return Result("value", frozenset(candidates[:1]), starts=tuple(candidates[:1]))
            return self._commit(expression.start, steps)
        dated = isinstance(expression, Compare)
        a, b = (self._commit(s, steps, dated) for s in expression.starts)
        for status in _FAILED:
            if status in (a.status, b.status):
                return Result(status)
        starts = (a.starts[0], b.starts[0])
        if isinstance(expression, Same):
            if bool(a.entities) != bool(b.entities):
                return Result("type", starts=starts)
            shared = a.entities & b.entities or {_date_key(d) for d in a.dates} & {
                _date_key(d) for d in b.dates
            }
            return Result("value", answer=bool(shared), starts=starts)
        first_a, first_b = a.dates[0], b.dates[0]
        if first_a.before(first_b):
            earlier, later = starts
        elif first_b.before(first_a):
            later, earlier = starts
        else:
            return Result("undetermined", starts=starts)
        return Result(
            "value", answer=earlier if expression.op == "earlier" else later, starts=starts
        )


def _date_key(date: Date) -> tuple:
    return (*date.lo, *date.hi, date.year, date.month or 0, date.day or 0)


STRUCTURES = (
    "entity_path",
    "date_path",
    "date_comparison",
    "yes_no",
    "date_difference",
    "count",
)
"""The structures of 2WikiMultihopQA's questions (the module docstring); the first four
have gold expressions."""
CHAIN_TYPES = ("compositional", "inference")
EARLIER = re.compile(r"\b(first|earlier|older)\b", re.IGNORECASE)
LATER = re.compile(r"\b(later|more recently|younger)\b", re.IGNORECASE)


def structure(row: dict) -> str:
    """The structure of a 2WikiMultihopQA question (the module docstring)."""
    if row["type"] in CHAIN_TYPES:
        return "entity_path" if _qid(row["answer_id"]) is not None else "date_path"
    if row["answer"].strip().casefold() in ("yes", "no"):
        return "yes_no"
    relations = {r for _, r, _ in row["evidences"]}
    if {"date of birth", "date of death"} <= relations:
        return "date_difference"
    if row["evidences"] and parse_date(row["evidences"][-1][2]) is not None:
        return "date_comparison"
    return "count"


def _qid(text: str | None) -> int | None:
    found = re.fullmatch(r"Q(\d+)", text or "")
    return int(found.group(1)) if found else None


def gold_expression(row: dict, label) -> tuple[Expression | None, dict, str | None]:
    """A question's gold expression from the dataset alone (the module docstring), its
    gold answer ({"entity": QID}, {"date": text}, or {"yes_no": "yes" or "no"}), and why
    no expression was built (None if one was). label gives an entity's label, or None."""
    kind = structure(row)

    def start(item: int) -> Start:
        return Start(label(item) or f"Q{item}", (item,))

    if kind in ("entity_path", "date_path"):
        gold = {"entity": row["answer_id"]} if kind == "entity_path" else {"date": row["answer"]}
        hops = [(_qid(s), r, _qid(o)) for s, r, o in row["evidences_id"]]
        ends = hops[-1][2] if hops else None
        connected = (
            hops
            and all(s is not None for s, _, _ in hops)
            and all(o is not None and o == hops[k + 1][0] for k, (_, _, o) in enumerate(hops[:-1]))
            and (ends == _qid(row["answer_id"]) if kind == "entity_path" else ends is None)
        )
        if not connected:
            return None, gold, "disconnected evidence"
        if kind == "date_path" and parse_date(row["answer"]) is None:
            return None, gold, "answer is no date"
        return Path(start(hops[0][0]), tuple(r for _, r, _ in hops)), gold, None
    if kind == "yes_no":
        gold = {"yes_no": row["answer"].strip().casefold()}
    elif kind == "date_comparison":
        gold = {"entity": row["answer_id"]}
    else:
        return None, {"entity": row["answer_id"]}, "structure without an expression"
    operands = [_qid(e) for e in row["entity_ids"].split("_")][:2]
    evidences = row["evidences"]
    bridge = row["type"] == "bridge_comparison"
    if len(operands) < 2 or None in operands or len(evidences) < (3 if bridge else 2):
        return None, gold, "unexpected evidence"
    via = (evidences[0][1], evidences[-1][1]) if bridge else (evidences[0][1],)
    starts = (start(operands[0]), start(operands[1]))
    if kind == "yes_no":
        return Same(starts, via), gold, None
    if _qid(row["answer_id"]) not in operands:
        return None, gold, "answer not an operand"
    op = "earlier" if EARLIER.search(row["question"]) else None
    op = op or ("later" if LATER.search(row["question"]) else None)
    if op is None:
        return None, gold, "no comparison word"
    return Compare(op, starts, via), gold, None


def correct(result: Result, gold: dict) -> bool:
    """Whether a result is the gold answer: the gold entity among a path's entities or a
    date comparison's operand, a date matching the gold date, or the gold yes or no."""
    if result.status != "value":
        return False
    if "yes_no" in gold:
        return isinstance(result.answer, bool) and result.answer == (gold["yes_no"] == "yes")
    if "date" in gold:
        wanted = parse_date(gold["date"])
        return wanted is not None and any(d.matches(wanted) for d in result.dates)
    item = _qid(gold["entity"])
    if result.answer is not None and not isinstance(result.answer, bool):
        return result.answer == item
    return item in result.entities


def written(expression: Expression) -> Expression:
    """The expression as its rendering reads, its starts committed by label."""
    return parse(render(expression))


def _tally(records: list[dict]) -> dict:
    table: dict[str, dict] = {}
    for name in (*STRUCTURES, "total"):
        rows = [r for r in records if name in ("total", r["structure"])]
        built = [r for r in rows if r["expression"] is not None]
        table[name] = {
            "questions": len(rows),
            "built": len(built),
            "executes": sum(r["anchored"]["correct"] for r in built),
            "executes_written": sum(r["written"]["correct"] for r in built),
            "anchored_status": _statuses(r["anchored"]["status"] for r in built),
            "written_status": _statuses(r["written"]["status"] for r in built),
        }
    return table


def _statuses(found) -> dict[str, int]:
    from collections import Counter

    counts = Counter(found)
    return {s: counts[s] for s in STATUSES if counts[s]}


def _run(executor: Executor, expression: Expression, gold: dict) -> dict:
    result = executor.run(expression)
    return {"status": result.status, "correct": correct(result, gold), **_shown(result)}


def _shown(result: Result) -> dict:
    """A result's value for the records: its answer, or its first entities and dates."""
    if result.answer is not None:
        answer = result.answer if isinstance(result.answer, bool) else f"Q{result.answer}"
        return {"answer": answer}
    return {
        "entities": [f"Q{e}" for e in sorted(result.entities)[:10]],
        "dates": [str(d) for d in result.dates[:10]],
    }


def profile(
    rows: list[dict], executor: Executor, subsets: dict[str, set[str]] | None = None
) -> tuple[list[dict], dict]:
    """Every question's gold expression executed anchored and written (the module
    docstring), and the counts per structure over all questions and each subset (a set
    of the dataset's `_id`s)."""
    records = []
    for row in rows:
        expression, gold, why = gold_expression(row, executor.store.label)
        record = {
            "id": f"2wiki-{row['_id']}",
            "type": row["type"],
            "structure": structure(row),
            "question": row["question"],
            "gold": gold,
            "expression": None if expression is None else render(expression),
            "reason": why,
        }
        if expression is not None:
            record["anchored"] = _run(executor, expression, gold)
            record["written"] = _run(executor, written(expression), gold)
        records.append(record)
    counts = {"all": _tally(records)}
    for name, ids in (subsets or {}).items():
        counts[name] = _tally([r for r in records if r["id"].removeprefix("2wiki-") in ids])
    return records, counts


EXPRESSION_INSTRUCTIONS = """\
Do not answer the question. Instead, write the program that computes its answer over \
Wikidata, on a last line "Program: ⟦ ...⟧", as one of three kinds of span:

- A path: the entity the question starts from, with its English Wikidata label, then \
the Wikidata properties to follow from it, in order, separated by " · ", like \
⟦ The Hobbit · author · place of birth⟧. It gives the entities or the date the path \
reaches.
- A date comparison: "earlier:" or "later:", the two entities compared, separated by \
" |", then the properties that lead from each of them to the date compared, like \
⟦ earlier: Jaws | Star Wars · publication date⟧. It gives the entity whose date is \
earlier or later. Born first or older means an earlier date of birth, younger a later one.
- A value comparison: "same:", the two entities, then the properties that lead from \
each of them to the values compared, like ⟦ same: Jaws | Alien · country of origin⟧. It \
gives yes when the two share a value, and no otherwise.

Use the exact labels Wikidata uses."""

EXPRESSION_EXAMPLES = [
    {
        "question": "In which country was the author of The Hobbit born?",
        "response": "Program: ⟦ The Hobbit · author · place of birth · country⟧",
    },
    {
        "question": "When was the director of film Psycho born?",
        "response": "Program: ⟦ Psycho · director · date of birth⟧",
    },
    {
        "question": "Which film came out first, Jaws or Star Wars?",
        "response": "Program: ⟦ earlier: Jaws | Star Wars · publication date⟧",
    },
    {
        "question": "Which film has the director who was born later, Alien or Vertigo?",
        "response": "Program: ⟦ later: Alien | Vertigo · director · date of birth⟧",
    },
    {
        "question": "Who died first, Charles Darwin or Abraham Lincoln?",
        "response": "Program: ⟦ earlier: Charles Darwin | Abraham Lincoln · date of death⟧",
    },
    {
        "question": "Are Jaws and Alien from the same country?",
        "response": "Program: ⟦ same: Jaws | Alien · country of origin⟧",
    },
    {
        "question": "Do both directors of films Psycho and Alien have the same nationality?",
        "response": "Program: ⟦ same: Psycho | Alien · director · country of citizenship⟧",
    },
]
"""Worked examples, every structure with an expression; none is a 2WikiMultihopQA
dev question."""


def expression_messages(question: str) -> list[dict]:
    """Instructions, worked examples, and the question."""
    messages = [{"role": "system", "content": EXPRESSION_INSTRUCTIONS}]
    for example in EXPRESSION_EXAMPLES:
        messages.append({"role": "user", "content": example["question"]})
        messages.append({"role": "assistant", "content": example["response"]})
    messages.append({"role": "user", "content": question})
    return messages


PANEL_STRUCTURES = STRUCTURES[:4]


def panel(records: list[dict], per_structure: int, seed: int) -> list[dict]:
    """per_structure records of each structure with gold expressions (PANEL_STRUCTURES),
    drawn with one seeded generator, structure by structure, and kept in dataset order;
    ValueError if a structure has fewer."""
    import random

    rng = random.Random(seed)
    chosen: set[int] = set()
    for name in PANEL_STRUCTURES:
        pool = [k for k, r in enumerate(records) if r["structure"] == name]
        if len(pool) < per_structure:
            raise ValueError(f"{name} has {len(pool)} questions, fewer than {per_structure}")
        chosen.update(rng.sample(pool, per_structure))
    return [records[k] for k in sorted(chosen)]


OUTCOMES = ("correct", "wrong", "no record", "no expression", "malformed", *STATUSES[1:])


def outcome(text: str | None, record: dict, executor: Executor) -> str:
    """How a free generation for a question fares (OUTCOMES): no record, no expression,
    malformed, a failed status of its execution, or correct or wrong."""
    if text is None:
        return "no record"
    found = find_expression(text)
    if found is None:
        return "no expression"
    try:
        expression = parse(found)
    except ExpressionError:
        return "malformed"
    result = executor.run(expression)
    if result.status != "value":
        return result.status
    return "correct" if correct(result, record["gold"]) else "wrong"


def score(texts: dict[str, str], records: list[dict], executor: Executor) -> dict:
    """Accuracy and outcomes of free generations (id to text) per structure and in
    total, over every record given (gold.jsonl's), a missing one counted wrong."""
    from collections import Counter

    found = {r["id"]: outcome(texts.get(r["id"]), r, executor) for r in records}
    table = {}
    for name in [s for s in STRUCTURES if any(r["structure"] == s for r in records)] + ["total"]:
        ids = [r["id"] for r in records if name in ("total", r["structure"])]
        counts = Counter(found[q] for q in ids)
        table[name] = {
            "questions": len(ids),
            "correct": counts["correct"],
            "accuracy": counts["correct"] / len(ids) if ids else 0.0,
            "outcomes": {o: counts[o] for o in OUTCOMES if counts[o]},
        }
    return table


def _read_jsonl(path: FilePath) -> list[dict]:
    import json

    with path.open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def _executor(args) -> Executor:
    from xgraph.toolkg import KGStore

    return Executor(KGStore(args.db), DateStore(args.dates), Relations.read(args.relations))


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    from xgraph.provenance import provenance

    parser = argparse.ArgumentParser(prog="python -m xgraph.expressions", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dates = commands.add_parser("dates", help="the date store of a KG directory with literals")
    dates.add_argument("--kg", type=FilePath, required=True, help="KG directory with literals")
    dates.add_argument("--db", type=FilePath, required=True, help="date store (SQLite)")

    def stores(command) -> None:
        command.add_argument("--db", type=FilePath, required=True, help="KG store (SQLite)")
        command.add_argument("--dates", type=FilePath, required=True, help="date store")
        command.add_argument(
            "--relations", type=FilePath, required=True, help="a KG's relation_labels.jsonl"
        )

    profiled = commands.add_parser("profile", help="gold expressions executed per structure")
    profiled.add_argument("--dev", type=FilePath, required=True, help="2WikiMultihopQA dev.json")
    profiled.add_argument("--dog", type=FilePath, help="DoG's 2wiki_4000.jsonl: a subset")
    stores(profiled)
    profiled.add_argument("--out", type=FilePath, required=True, help="output directory")
    asked = commands.add_parser("requests", help="a panel's chat requests")
    asked.add_argument("--gold", type=FilePath, required=True, help="the profile's gold.jsonl")
    asked.add_argument("--per-structure", type=int, required=True, help="questions per structure")
    asked.add_argument("--seed", type=int, default=0, help="the panel's seed")
    asked.add_argument("--out", type=FilePath, required=True, help="requests (JSONL)")
    scored = commands.add_parser("score", help="free generations executed and scored")
    scored.add_argument("--outputs", type=FilePath, required=True, help="records (JSONL)")
    scored.add_argument("--gold", type=FilePath, required=True, help="the profile's gold.jsonl")
    scored.add_argument("--requests", type=FilePath, required=True, help="the panel's requests")
    stores(scored)
    scored.add_argument("--out", type=FilePath, required=True, help="scores (JSON)")
    args = parser.parse_args(argv)

    if args.command == "dates":
        counts = build_date_store(args.kg, args.db)
        manifest = {
            "artifact_kind": "xgraph.date_store",
            "version": 1,
            "counts": counts,
            "provenance": provenance([args.kg / "triples.npy", args.kg / "entity_labels.jsonl"]),
        }
        FilePath(f"{args.db}.json").write_text(json.dumps(manifest, indent=1) + "\n")
        print(json.dumps(counts))
    elif args.command == "profile":
        rows = json.loads(args.dev.read_text(encoding="utf-8"))
        subsets = None
        if args.dog is not None:
            subsets = {"dog": {r["id"] for r in _read_jsonl(args.dog)}}
        records, counts = profile(rows, _executor(args), subsets)
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / "gold.jsonl").open("w", encoding="utf-8") as written_records:
            for record in records:
                written_records.write(json.dumps(record, ensure_ascii=False) + "\n")
        inputs = [args.dev, args.db, args.dates, args.relations, *([args.dog] if args.dog else [])]
        report = {
            "artifact_kind": "xgraph.expression_profile",
            "version": 1,
            "counts": counts,
            "provenance": provenance(inputs),
        }
        (args.out / "profile.json").write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps({s: {k: v for k, v in c.items() if "status" not in k}
                          for s, c in counts["all"].items()}))  # fmt: skip
    elif args.command == "requests":
        chosen = panel(_read_jsonl(args.gold), args.per_structure, args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as out:
            for record in chosen:
                request = {
                    "id": record["id"],
                    "structure": record["structure"],
                    "messages": expression_messages(record["question"]),
                }
                out.write(json.dumps(request, ensure_ascii=False) + "\n")
        print(f"{len(chosen)} requests")
    else:
        wanted = {r["id"] for r in _read_jsonl(args.requests)}
        records = [r for r in _read_jsonl(args.gold) if r["id"] in wanted]
        if len(records) != len(wanted):
            raise SystemExit(f"{args.gold} lacks {len(wanted) - len(records)} panel questions")
        texts = {r["id"]: record_text(r) for r in _read_jsonl(args.outputs)}
        result = {
            "artifact_kind": "xgraph.expression_scores",
            "version": 1,
            "scores": score(texts, records, _executor(args)),
            "provenance": provenance([args.outputs, args.gold, args.requests, args.db, args.dates]),
        }
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({s: v["accuracy"] for s, v in result["scores"].items()}))


if __name__ == "__main__":
    main()
