"""Observed paths' wire format, compiled once for a served tokenizer.

Canonical tokenization concatenates independently encoded punctuation pieces and
unchanged label-index rows, including their label prefix. It need not equal BPE
encoding of the concatenated text. Every piece must round-trip to its exact bytes.
No tokenizer or host lookup participates in the GPU's per-token observation loop.
"""

from __future__ import annotations

from collections.abc import Callable

from xgraph.syntax import SpanSyntax
from xgraph.tokenizer import ServedTokenizer

JOIN = " ;"
MAX_OBSERVE = 64

PATH_OPTION_FIELDS = (
    "menu", "observe", "observe_select", "observe_answer", "observe_required",
    "observe_no_loop", "observe_width", "observe_defer", "menu_close_next",
    "observe_terminal",
)  # fmt: skip


def checked_path_options(
    value: dict, *, capabilities: dict | None = None, capability_union: bool = False
) -> dict:
    """A complete request view; omitted switches are off, demos describe its prompt.

    Demonstrations are rendered by the caller, never inserted into token history
    by the engine. Keeping their count here makes the stage contract auditable.
    """
    if not isinstance(value, dict) or value.keys() - {*PATH_OPTION_FIELDS, "path_demos"}:
        raise ValueError("xgraph_path_options needs supported path settings")
    options = dict.fromkeys(PATH_OPTION_FIELDS, 0)
    options.update(value)
    for key in ("menu", "observe", "observe_width", "observe_defer", "observe_terminal"):
        if type(options[key]) is not int:
            raise ValueError(f"xgraph_path_options {key} must be an integer")
    for key in set(PATH_OPTION_FIELDS) - {
        "menu",
        "observe",
        "observe_width",
        "observe_defer",
        "observe_terminal",
    }:
        if type(options[key]) not in (bool, int) or options[key] not in (0, 1):
            raise ValueError(f"xgraph_path_options {key} must be 0 or 1")
    if "path_demos" in options and (
        type(options["path_demos"]) is not int or options["path_demos"] not in (4, 5)
    ):
        raise ValueError("xgraph_path_options path_demos must be 4 or 5")
    from xgraph.menus import check_menu

    check_menu(options["menu"], paths=True, relations="frontier", expressions=False, audit=False)
    check_observe(options["observe"], paths=True, relations="frontier", expressions=False)
    options["observe_select"] |= options["observe_required"]
    options["observe_answer"] |= bool(options["observe_defer"] or options["menu_close_next"])
    check_observed_options(
        options["observe"], options["observe_select"], options["observe_answer"],
        options["observe_required"], options["observe_no_loop"],
        width=options["observe_width"], defer=options["observe_defer"],
        close_next=options["menu_close_next"], menu=options["menu"],
    )  # fmt: skip
    from xgraph.audit import check_terminal

    check_terminal(
        options["observe_terminal"],
        paths=True,
        relations="frontier",
        expressions=False,
        incompatible=()
        if capability_union
        else (options[k] for k in PATH_OPTION_FIELDS if k != "observe_terminal"),
    )
    if capabilities is not None and any(options[k] > capabilities[k] for k in PATH_OPTION_FIELDS):
        raise ValueError("request path options exceed engine capabilities")
    return options


def observation_marker(syntax: SpanSyntax) -> str:
    """Contains the existing separator, which every indexed label already excludes."""
    return syntax.separator + "→"


def selection_marker(syntax: SpanSyntax) -> str:
    """An excluded opener makes the marker unambiguous even after a label suffix."""
    return " " + syntax.opener + "="


def answer_prefix(syntax: SpanSyntax) -> str:
    return ".\nAnswer: " + syntax.opener


def check_observed_options(
    observe: int,
    select: bool,
    answer: bool,
    required: bool = False,
    no_loop: bool = False,
    *,
    width: int = 0,
    defer: int = 0,
    close_next: bool = False,
    menu: int = 0,
) -> None:
    if type(width) is not int or not 0 <= width <= MAX_OBSERVE:
        raise ValueError("observe_width must be an integer in [0, 64]")
    if width and not observe:
        raise ValueError("observe_width needs observe > 0")
    if type(defer) is not int or not 0 <= defer <= 8:
        raise ValueError("observe_defer must be an integer in [0, 8]")
    if defer and not (observe and required):
        raise ValueError("observe_defer needs observe > 0 and observe_required")
    if close_next and not (observe and menu):
        raise ValueError("menu_close_next needs observe > 0 and menu > 0")
    if (required or no_loop) and not observe:
        raise ValueError("observe_required and observe_no_loop need observe > 0")
    if (select or answer) and not observe:
        raise ValueError("observe_select and observe_answer need observe > 0")


def canonical_tokens(tokenizer: ServedTokenizer, texts: list[str]) -> tuple[int, ...]:
    tokens = []
    for text in texts:
        piece = tokenizer.encode(text)
        data = [tokenizer.token_bytes(t) for t in piece]
        if not piece or None in data or b"".join(data) != text.encode():
            raise ValueError(f"observation punctuation does not round-trip: {text!r}")
        tokens.extend(piece)
    return tuple(tokens)


def selection_tokens(tokenizer: ServedTokenizer, syntax: SpanSyntax) -> tuple[int, ...]:
    marker = selection_marker(syntax)
    for delimiter in syntax.delimiters:
        if marker in delimiter or (delimiter != syntax.opener and delimiter in marker):
            raise ValueError(f"selection punctuation conflicts with span delimiter {delimiter!r}")
        if marker.startswith(delimiter) or delimiter.startswith(marker):
            raise ValueError(f"selection marker shares a complete delimiter {delimiter!r}")
    return canonical_tokens(tokenizer, [marker])


def answer_tokens(tokenizer: ServedTokenizer, syntax: SpanSyntax) -> tuple[int, ...]:
    return canonical_tokens(tokenizer, [".\nAnswer: ", syntax.opener])


def decision_marker(syntax: SpanSyntax) -> str:
    return " " + syntax.opener + "!"


def decision_pieces(tokenizer: ServedTokenizer, syntax: SpanSyntax):
    """Forced opener, two model commands, forced path closer and NEXT separator."""
    selection_tokens(tokenizer, syntax)
    marker = decision_marker(syntax)
    for delimiter in syntax.delimiters:
        if marker in delimiter or (delimiter != syntax.opener and delimiter in marker):
            raise ValueError(f"decision punctuation conflicts with span delimiter {delimiter!r}")
        if any(delimiter in command for command in ("CLOSE", "NEXT", "DEFER")):
            raise ValueError(f"decision command conflicts with span delimiter {delimiter!r}")
    return [
        canonical_tokens(tokenizer, parts)
        for parts in (
            [marker],
            ["CLOSE", syntax.closer],
            ["NEXT", syntax.closer],
            [syntax.closer],
            [syntax.separator],
        )
    ]


def defer_tokens(tokenizer: ServedTokenizer, syntax: SpanSyntax):
    decision_pieces(tokenizer, syntax)
    return canonical_tokens(tokenizer, ["DEFER", syntax.closer])


def check_observe(observe: int, *, paths: bool, relations: str, expressions: bool) -> None:
    if type(observe) is not int or not 0 <= observe <= MAX_OBSERVE:
        raise ValueError(f"observe must be an integer in [0, {MAX_OBSERVE}], got {observe!r}")
    if observe and (not paths or relations != "frontier" or expressions):
        raise ValueError("observe needs path mode with frontier relations and expressions off")


def observation_pieces(tokenizer: ServedTokenizer, syntax: SpanSyntax) -> list[tuple[int, ...]]:
    """Marker, join, empty, then count markers for 1..64 omitted capped entities."""
    marker = observation_marker(syntax)
    texts = [marker, JOIN, syntax.label_prefix + "∅"]
    texts += [f" (+{n} more)" for n in range(1, MAX_OBSERVE + 1)]
    for delimiter in syntax.delimiters:
        if (
            delimiter in "".join(texts[1:])
            or marker in delimiter
            or (delimiter != syntax.separator and delimiter in marker)
        ):
            raise ValueError(f"observation punctuation conflicts with span delimiter {delimiter!r}")
    pieces = [tuple(tokenizer.encode(text)) for text in texts]
    for text, tokens in zip(texts, pieces, strict=True):
        data = [tokenizer.token_bytes(t) for t in tokens]
        if not tokens or None in data or b"".join(data) != text.encode():
            raise ValueError(f"observation punctuation does not round-trip: {text!r}")
    return pieces


class PathRelations(list[str]):
    """Relation labels with entity selections indexed by boundary (0 = start).

    Keeping the list interface preserves consumers that score only relations.
    Slices and copies retain selections on their corresponding relation boundaries.
    """

    def __init__(self, relations=(), selections=None, audit=None, *, defers=(), decisions=None):
        super().__init__(relations)
        self.selections = {int(hop): label for hop, label in (selections or {}).items()}
        self.audit = audit
        self.defers = {int(hop) for hop in defers}
        self.decisions = {int(hop): command for hop, command in (decisions or {}).items()}

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if not isinstance(key, slice):
            return result
        selections = {0: self.selections[0]} if 0 in self.selections else {}
        defers = {0} if 0 in self.defers else set()
        decisions = {}
        for hop, original in enumerate(range(len(self))[key], 1):
            if original + 1 in self.selections:
                selections[hop] = self.selections[original + 1]
            if original + 1 in self.defers:
                defers.add(hop)
            if original + 1 in self.decisions:
                decisions[hop] = self.decisions[original + 1]
        audit = self.audit if list(range(len(self)))[key] == list(range(len(self))) else None
        return type(self)(result, selections, audit, defers=defers, decisions=decisions)

    def copy(self):
        return self[:]


def path_parts(
    body: str,
    separator: str = " ·",
    opener: str = "⟦",
    escape: str = " ⁇",
    *,
    choices=False,
) -> tuple[list[str], dict[int, str]]:
    import re

    marker, select, decision = separator + "→", " " + opener + "=", " " + opener + "!"
    if (select not in body and decision not in body) or escape in body:
        parts = body.split(separator)
        if len(parts) % 2 == 0 and all(p.startswith("→") for p in parts[1::2]):
            result = ([part.strip() for part in parts[::2]], {})
        else:
            result = ([part.strip() for part in body.split(separator.strip())], {})
        return (*result, set(), {}) if choices else result
    parts = re.split(
        "(" + "|".join(re.escape(d) for d in (select, marker, decision, separator)) + ")", body
    )
    segments, selections, defers, decisions = [], {}, set(), {}
    kind = "written"
    for n in range(0, len(parts), 2):
        text = parts[n].strip()
        if kind == "written":
            segments.append(text)
        elif kind == "selection":
            selections[len(segments) - 1] = text
        elif kind == "decision":
            if text == "DEFER":
                defers.add(len(segments) - 1)
            elif text in ("CLOSE", "NEXT"):
                decisions[len(segments) - 1] = text
            else:
                raise ValueError(f"unknown path decision {text!r}")
        if n + 1 < len(parts):
            kind = {
                marker: "observation",
                select: "selection",
                decision: "decision",
                separator: "written",
            }[parts[n + 1]]
    return (segments, selections, defers, decisions) if choices else (segments, selections)


def path_segments(
    body: str, separator: str = " ·", opener: str = "⟦", escape: str = " ⁇"
) -> list[str]:
    """Read an observed span without reading its witnesses as part of the plan.

    An observed span alternates written segments and observations. The observation
    marker includes the ordinary separator; its arrow follows that separator
    directly, before the payload's label prefix. Arrows inside labels are ordinary
    text. Unobserved spans retain the old whitespace-tolerant splitting.
    """
    return path_parts(body, separator, opener, escape)[0]


def strip_observations(text: str, syntax: SpanSyntax) -> str:
    """Remove forced frontier readback, retaining selections and escaped text.

    Call after removing relation menus, whose closer belongs to the menu rather
    than the enclosing path. Selection labels remain semantic path decisions.
    """
    marker = observation_marker(syntax)
    pos, inside, escaped = 0, False, False
    ranges = []
    while pos < len(text):
        choices = (
            [syntax.opener]
            if not inside
            else [syntax.closer]
            if escaped
            else [marker, syntax.closer, syntax.escape]
        )
        found = [(text.find(d, pos), d) for d in choices]
        found = [(at, d) for at, d in found if at >= 0]
        if not found:
            break
        at, delimiter = min(found, key=lambda item: item[0])
        if delimiter == marker:
            ends = [
                text.find(d, at + len(marker))
                for d in (syntax.separator, selection_marker(syntax), syntax.closer, syntax.escape)
            ]
            end = min((at for at in ends if at >= 0), default=len(text))
            ranges.append((at, end))
            pos = end
            continue
        if delimiter == syntax.opener:
            inside, escaped = True, False
        elif delimiter == syntax.closer:
            inside, escaped = False, False
        else:
            escaped = True
        pos = at + len(delimiter)
    for lo, hi in reversed(ranges):
        text = text[:lo] + text[hi:]
    return text


def path_prefix(body: str, hops: int, syntax: SpanSyntax | None = None) -> str:
    """Retain a written prefix including its unchanged observations and selections."""
    import re

    syntax = syntax or SpanSyntax()
    delimiters = (selection_marker(syntax), observation_marker(syntax), syntax.separator)
    pattern = "|".join(re.escape(d) for d in delimiters)
    relations = 0
    for match in re.finditer(pattern, body):
        if match.group() == syntax.separator:
            relations += 1
            if relations > hops:
                return body[: match.start()]
    return body


def count_observation_tokens(
    tokens: list[int],
    tokenizer: ServedTokenizer,
    syntax: SpanSyntax,
    *,
    prefill: str = "",
    force_answer: bool = False,
    selection_required: Callable[[str], bool] | None = None,
) -> int:
    """Account once after generation, without replaying the KG or charging prompt tokens.

    Count generated tokens overlapping a committed observation marker or its payload,
    including a truncated payload. An unfinished marker has not committed an
    observation and belongs to the ordinary generated-token total. Escapes and text
    outside spans are never observations. Prefills establish the parsing state only.
    """
    prefix = prefill.encode()
    data = [tokenizer.token_bytes(t) or b"" for t in tokens]
    text = prefix + b"".join(data)
    opener, closer, separator, escape = (
        d.encode() for d in (syntax.opener, syntax.closer, syntax.separator, syntax.escape)
    )
    marker = observation_marker(syntax).encode()
    select = selection_marker(syntax).encode()
    from xgraph.menus import menu_marker

    menu = menu_marker(syntax).encode()
    decision = decision_marker(syntax).encode()
    ranges = []
    pos, inside, escaped, start = 0, False, False, None
    body_at = 0
    path, answering, answered_at = False, False, None
    next_separator = False
    while pos < len(text):
        choices = (
            [opener]
            if not inside
            else [closer]
            if escaped
            else [closer, marker, menu, decision, select, separator, escape]
        )
        found = [(text.find(d, pos), d) for d in choices]
        found = [(at, d) for at, d in found if at >= 0]
        if not found:
            break
        at, delimiter = min(found, key=lambda pair: pair[0])
        if start is not None and delimiter != marker:
            ranges.append((start, at))
            start = None
        if delimiter == menu:
            end = text.find(closer, at + len(menu))
            end = len(text) if end < 0 else end + len(closer)
            ranges.append((at, end))
            if next_separator:
                ranges.append((end, end + len(separator)))
            pos = end
            continue
        if delimiter == decision:
            ranges.append((at, at + len(decision)))
            end = text.find(closer, at + len(decision))
            if end < 0:
                break
            command = text[at + len(decision) : end]
            end += len(closer)
            if command == b"CLOSE":
                ranges.append((end, end + len(closer)))
            next_separator = command == b"NEXT"
            pos = end
            continue
        if delimiter == opener:
            inside, escaped = True, False
            path = False
            body_at = at + len(opener)
        elif delimiter == closer:
            if force_answer and path and not escaped:
                first = at + len(closer)
                ranges.append((first, first + len(answer_prefix(syntax).encode())))
                answering = True
            elif answering and not escaped:
                answered_at = at + len(closer)
            inside, escaped = False, False
        elif delimiter == escape:
            escaped = True
        elif delimiter == marker and start is None:
            start = at
        elif delimiter in (separator, select):
            path = True
            if delimiter == separator and next_separator:
                ranges.append((at, at + len(separator)))
                next_separator = False
            if (
                delimiter == select
                and selection_required is not None
                and selection_required(text[body_at:at].decode("utf-8"))
            ):
                ranges.append((at, at + len(select)))
            if delimiter == select and text.startswith(b"DEFER" + closer, at + len(select)):
                pos = at + len(select) + len(b"DEFER") + len(closer)
                continue
        pos = at + len(delimiter)
    if start is not None:
        ranges.append((start, len(text)))
    offset, count = len(prefix), 0
    for token, piece in zip(tokens, data, strict=True):
        end = offset + len(piece)
        count += bool(piece) and any(offset < hi and end > lo for lo, hi in ranges)
        if (
            force_answer
            and answered_at is not None
            and offset >= answered_at
            and not piece
            and token in tokenizer.special_ids
        ):
            count += 1
        offset = end
    return count
