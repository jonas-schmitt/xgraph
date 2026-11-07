"""Span syntax: the delimiters that mark spans in the output."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpanSyntax:
    """The span opener, label prefix, separator, span closer, and escape marker, and the
    operand separator and operators that write expressions.

    The defaults follow GLOSSARY.md: an entity span is written `⟦ Marie Curie⟧`, a
    fact span `⟦ Marie Curie · spouse · Pierre Curie⟧`, an escape
    `⟦ ⁇ unverified text⟧` or `⟦ Marie ⁇ née Skłodowska⟧`, and a date comparison
    `⟦ earlier: Jaws | Star Wars · publication date⟧`, and a value comparison
    `⟦ same: Jaws | Alien · country of origin⟧`.
    """

    opener: str = "⟦"
    label_prefix: str = " "
    """Text written between the span opener (or a separator) and a label; labels are tokenized
    with it in front."""
    separator: str = " ·"
    """Text between the segments of a fact span; the next label's prefix follows it."""
    closer: str = "⟧"
    escape: str = " ⁇"
    operand_separator: str = " |"
    """Text between the two operands of an expression; the next label's prefix follows it."""
    operators: tuple[str, ...] = ("earlier:", "later:", "same:")
    """The operators an expression starts with, after the span opener and the label prefix,
    each followed by the label prefix and the first operand: the first compares by the
    earlier date, the second by the later, the third by a shared value (a value comparison,
    whose verdict is yes or no). Empty for a syntax without expressions."""

    def __post_init__(self) -> None:
        # An index header holds the syntax as JSON, which has no tuples.
        object.__setattr__(self, "operators", tuple(self.operators))
        for name in ("opener", "separator", "closer", "escape"):
            if not getattr(self, name):
                raise ValueError(f"the span {name} must not be empty")
        if self.operators and not self.operand_separator:
            raise ValueError("operators need an operand separator between their operands")
        if any(not op for op in self.operators):
            raise ValueError("an operator must not be empty")
        # Inside a span the closer, the separator, the escape marker, and the operand
        # separator are matched side by side, and at a span's start the operators (after the
        # label prefix) and the escape marker, so none may be a prefix of another.
        inside = {
            "span closer": self.closer,
            "separator": self.separator,
            "escape marker": self.escape,
        }
        if self.operand_separator:
            inside["operand separator"] = self.operand_separator
        starts = {"escape marker": self.escape}
        for k, op in enumerate(self.operators):
            starts[f"operator {k} ({op!r})"] = self.label_prefix + op
        for group in (inside, starts):
            for name, text in group.items():
                for other, other_text in group.items():
                    if name != other and other_text.encode("utf-8").startswith(
                        text.encode("utf-8")
                    ):
                        raise ValueError(
                            f"the {name} {text!r} must not be a prefix of the {other} "
                            f"{other_text!r}"
                        )

    @property
    def delimiters(self) -> tuple[str, ...]:
        """The span delimiters: opener, separator, closer, and escape marker."""
        return (self.opener, self.separator, self.closer, self.escape)

    @property
    def writes_expressions(self) -> bool:
        """Whether the syntax has operators (and so an operand separator)."""
        return bool(self.operators)

    def excludes(self, label: str) -> bool:
        """Whether a label's written form (the label prefix and the label) holds span syntax:
        a delimiter or the operand separator anywhere, or an operator at its start. Such a
        label is excluded from the label index."""
        written = self.label_prefix + label
        holds = [*self.delimiters, *([self.operand_separator] if self.operand_separator else [])]
        return any(d in written for d in holds) or any(
            written.startswith(self.label_prefix + op) for op in self.operators
        )

    def quoted(self, written: bytes) -> bool:
        """Whether a label written with its label prefix in front (as a label index row's
        bytes) starts with one of the QUOTATION_MARKS."""
        prefix = self.label_prefix.encode("utf-8")
        label = written[len(prefix) :] if written.startswith(prefix) else written
        return label.startswith(_QUOTATION_BYTES)


QUOTATION_MARKS = ('"', "“", "”", "„", "‟", "«", "»")
"""The marks that unquoted starts keep from opening an unbound subject: double quotation
marks, straight and typographic, and guillemets. Single quotes are left out, since the same
characters write the apostrophe that starts names such as 's-Hertogenbosch."""
_QUOTATION_BYTES = tuple(mark.encode("utf-8") for mark in QUOTATION_MARKS)


PRESETS: dict[str, SpanSyntax] = {
    # The default: rare characters, so delimiters never merge with other text in a token.
    "brackets": SpanSyntax(),
    # Formats a model has seen often in training: wiki links, tags, CJK brackets, and
    # JSON arrays. Labels follow the opener and separator directly.
    "wiki": SpanSyntax(opener="[[", label_prefix="", separator="|", closer="]]"),
    "tags": SpanSyntax(opener="<kg>", label_prefix="", separator="|", closer="</kg>"),
    "cjk": SpanSyntax(opener="【", label_prefix="", separator="|", closer="】"),
    "json": SpanSyntax(opener='["', label_prefix="", separator='", "', closer='"]'),
}
"""Named span syntaxes, for comparing how the output format affects generation."""


def preset(name: str) -> SpanSyntax:
    if name not in PRESETS:
        raise ValueError(f"unknown span syntax {name!r}; expected one of {sorted(PRESETS)}")
    return PRESETS[name]
