"""Canonical in-path relation suggestions; execution never reads displayed labels."""

from __future__ import annotations

from xgraph.observations import (
    canonical_tokens,
    decision_marker,
    selection_marker,
    selection_tokens,
)
from xgraph.syntax import SpanSyntax

_SYNTAX = SpanSyntax()


def check_menu(menu, *, paths, relations, expressions, audit=False):
    if type(menu) is not int or not 0 <= menu <= 16:
        raise ValueError("menu must be an integer in [0, 16]")
    if menu and (not paths or relations != "frontier" or expressions):
        raise ValueError("menu needs frontier path mode with expressions off")
    if menu and audit:
        raise ValueError("menu cannot be combined with audit")


def menu_marker(syntax: SpanSyntax):
    return " " + syntax.opener + "?"


def menu_pieces(tokenizer, syntax: SpanSyntax, relation_count: int):
    """Audit's descriptor layout: heading 0, join 4, empty 5, close 6, counts 7+."""
    selection_tokens(tokenizer, syntax)
    marker = menu_marker(syntax)
    texts = [marker] * 4 + [" ;", syntax.label_prefix + "∅", syntax.closer]
    texts += [f" (+{n} more)" for n in range(1, max(64, relation_count) + 1)]
    for delimiter in syntax.delimiters:
        if (
            marker in delimiter
            or (delimiter != syntax.opener and delimiter in marker)
            or any(delimiter in text for text in texts[4:6] + texts[7:])
        ):
            raise ValueError(f"menu punctuation conflicts with span delimiter {delimiter!r}")
    return [canonical_tokens(tokenizer, [text]) for text in texts]


def _auxiliary(text: str, syntax: SpanSyntax):
    """Byte-independent text ranges, only within unescaped spans; includes truncation."""
    marker = menu_marker(syntax)
    decision = decision_marker(syntax)
    defer = selection_marker(syntax) + "DEFER" + syntax.closer
    pos, inside, escaped = 0, False, False
    while pos < len(text):
        choices = (
            [syntax.opener]
            if not inside
            else [syntax.closer]
            if escaped
            else [marker, decision, defer, syntax.closer, syntax.escape]
        )
        found = [(text.find(d, pos), d) for d in choices]
        found = [(at, d) for at, d in found if at >= 0]
        if not found:
            return
        at, delimiter = min(found, key=lambda item: item[0])
        if delimiter in (marker, decision, defer):
            end = (
                text.find(syntax.closer, at + len(delimiter))
                if delimiter != defer
                else at + len(defer) - len(syntax.closer)
            )
            complete = end >= 0
            command = (
                text[at + len(decision) : end]
                if delimiter == decision and complete
                else "DEFER"
                if delimiter == defer
                else None
            )
            end = len(text) if not complete else end + len(syntax.closer)
            yield at, end, command
            pos = end
            continue
        if delimiter == syntax.opener:
            inside, escaped = True, False
        elif delimiter == syntax.closer:
            inside, escaped = False, False
        else:
            escaped = True
        pos = at + len(delimiter)


def menu_ranges(text: str, syntax: SpanSyntax):
    for lo, hi, command in _auxiliary(text, syntax):
        if command is None:
            yield lo, hi


def auxiliary_ranges(text: str, syntax: SpanSyntax):
    for lo, hi, _ in _auxiliary(text, syntax):
        yield lo, hi


def semantic_text(text: str, syntax: SpanSyntax = _SYNTAX, *, keep_decisions=False):
    for lo, hi, command in reversed(list(_auxiliary(text, syntax))):
        replacement = decision_marker(syntax) + command if keep_decisions and command else ""
        text = text[:lo] + replacement + text[hi:]
    return text
