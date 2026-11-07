"""One blind path, a GPU readback, and exactly one terminal decision.

The host compiles punctuation and request relation ranks once. Payloads reference
unchanged entity/relation rows; no host work participates in forced decoding.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from xgraph.observations import canonical_tokens, selection_marker
from xgraph.syntax import SpanSyntax

MAX_RANKED_RELATIONS = 64


def checked_relation_ranks(ranked, ids=None):
    """Validate the bounded request payload before any native slot write."""
    context = "relation_ranks"
    if not isinstance(ranked, (list, tuple)) or len(ranked) > MAX_RANKED_RELATIONS:
        raise ValueError(f"{context} needs a list of at most 64 (relation id, rank) pairs")
    seen, result = set(), []
    for number, pair in enumerate(ranked):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"{context}[{number}] must be a (relation id, rank) pair")
        relation, rank = pair
        if not isinstance(relation, str) or not relation:
            raise ValueError(f"{context}[{number}] needs a nonempty relation id string")
        if relation in seen:
            raise ValueError(f"{context}: duplicate relation id {relation!r}")
        if ids is not None and relation not in ids:
            raise ValueError(f"{context}: unknown relation id {relation!r}")
        if type(rank) is not int or not 0 <= rank < MAX_RANKED_RELATIONS:
            raise ValueError(f"{context}[{number}]: rank must be an integer in [0, 64)")
        seen.add(relation)
        result.append([relation, rank])
    return result


_SYNTAX = SpanSyntax()
EXECUTED, TERMINAL, PENULTIMATE, DECISION, JOIN, EMPTY, CLOSE, COUNT = range(8)
KEEP, SELECT, APPEND, REPLACE, DROP = range(5)


def check_audit(audit, entities, menus, *, paths, relations, expressions, observed=()):
    if type(entities) is not int or not 1 <= entities <= 64:
        raise ValueError("audit_entities must be an integer in [1, 64]")
    if type(menus) is not int or not 0 <= menus <= 64:
        raise ValueError("audit_relations must be an integer in [0, 64]")
    if audit and (not paths or relations != "frontier" or expressions or any(observed)):
        raise ValueError("audit needs frontier path mode, expressions and observations off")


def audit_pieces(tokenizer, syntax: SpanSyntax, relation_count: int):
    plain = [
        "\nExecuted: ",
        "\nRelations (terminal): ",
        "\nRelations (penultimate): ",
        "\nDecision: ",
        "KEEP",
        "SELECT",
        "APPEND",
        "REPLACE",
        "DROP",
        " ;",
        syntax.label_prefix + "∅",
        *(f" (+{n} more)" for n in range(1, max(64, relation_count) + 1)),
    ]
    for delimiter in syntax.delimiters:
        if any(delimiter in text for text in plain):
            raise ValueError(f"audit punctuation conflicts with span delimiter {delimiter!r}")
    texts = [
        "\nExecuted: " + syntax.opener,
        "\nRelations (terminal): " + syntax.opener,
        "\nRelations (penultimate): " + syntax.opener,
        "\nDecision: " + syntax.opener,
        " ;",
        syntax.label_prefix + "∅",
        syntax.closer,
        *(f" (+{n} more)" for n in range(1, max(64, relation_count) + 1)),
    ]
    return [canonical_tokens(tokenizer, [text]) for text in texts]


def check_terminal(width, *, paths, relations, expressions, incompatible=()):
    if type(width) is not int or not 0 <= width <= 64:
        raise ValueError("observe_terminal must be an integer in [0, 64]")
    if width and (not paths or relations != "frontier" or expressions or any(incompatible)):
        raise ValueError(
            "observe_terminal needs frontier paths with other displays and expressions off"
        )


def terminal_pieces(tokenizer, syntax: SpanSyntax):
    """Use audit's endpoint-list descriptors, with a distinct read-only heading."""
    if any(d in "\nEndpoints: " for d in syntax.delimiters):
        raise ValueError("terminal punctuation conflicts with span delimiters")
    pieces = audit_pieces(tokenizer, syntax, 0)
    pieces[EXECUTED] = canonical_tokens(tokenizer, ["\nEndpoints: " + syntax.opener])
    return pieces


def check_terminal_order(order, seed, width):
    if order not in ("identity", "reversed", "random"):
        raise ValueError("terminal_order must be identity, reversed or random")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("terminal_seed must be an unsigned 32-bit integer")
    if not width and (order != "identity" or seed != 17):
        raise ValueError("terminal_order/terminal_seed need observe_terminal")


def permute_terminal(rows, order, seed):
    """Permute an already selected display, using CUDA's uint32 Fisher-Yates RNG."""
    rows = list(rows)
    if order == "reversed":
        rows.reverse()
    elif order == "random":
        for i in range(len(rows) - 1, 0, -1):
            seed = (1664525 * seed + 1013904223) & 0xFFFFFFFF
            j = seed % (i + 1)
            rows[i], rows[j] = rows[j], rows[i]
    return rows


def terminal_instruction(width):
    check_terminal(width, paths=True, relations="frontier", expressions=False)
    return (
        (
            " After the revised relation path closes, the KG will show Endpoints: "
            f"with up to {width} "
            "deliverable endpoint labels when more than one is available. Read them before "
            "writing the bound Answer span. Undisplayed endpoints remain selectable. "
            "No further relation edits are allowed after this display."
        )
        if width
        else ""
    )


def audit_commands(tokenizer, syntax: SpanSyntax):
    from xgraph.observations import selection_tokens

    selection_tokens(tokenizer, syntax)  # reject complete delimiter overlaps
    return [
        canonical_tokens(tokenizer, ["KEEP", syntax.closer]),
        canonical_tokens(tokenizer, ["SELECT", selection_marker(syntax)]),
        canonical_tokens(tokenizer, ["APPEND", syntax.separator]),
        canonical_tokens(tokenizer, ["REPLACE", syntax.separator]),
        canonical_tokens(tokenizer, ["DROP", syntax.closer]),
    ]


class RelationScores:
    """IDF-weighted question/alias stem overlap, as the round-18 display audit.

    Positive scores only, at most 64 relation ids with equal-score ranks. The GPU
    breaks rank ties by frontier adjacency frequency, then relation index id.
    Zero-score and omitted relations follow all listed ids, by the same tie-break.
    """

    def __init__(self, index, tokenizer, relation_labels=None):
        import json
        from pathlib import Path

        from xgraph.index import IndexMismatch, primary_relation_rows
        from xgraph.repair import content_stems

        words = [[] for _ in index.relations.owner_ids]
        self.names, self.rows = [], [[] for _ in words]
        for row, owner in enumerate(index.relations.owners):
            text = b"".join(tokenizer.token_bytes(t) for t in index.relations.row_tokens(row))
            name = text.decode().removeprefix(index.span_syntax.label_prefix)
            self.names.append(name)
            self.rows[int(owner)].append(row)
            words[int(owner)].append(content_stems(name))
        self.words = words
        self.ids = index.relations.owner_ids
        self.owner_at = {r: i for i, r in enumerate(self.ids)}
        self.primary = index.relation_primary_rows
        if relation_labels is not None:
            source = Path(relation_labels)
            canonical = {}
            for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
                context = f"{source}:{number}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{context}: invalid JSON: {error.msg}") from error
                if (
                    not isinstance(row, list)
                    or len(row) != 3
                    or not all(isinstance(v, str) for v in row)
                ):
                    raise ValueError(f"{context}: expected a [relation, label, kind] string row")
                relation, name, kind = row
                if kind not in ("label", "alias"):
                    raise ValueError(f"{context}: unknown relation label kind {kind!r}")
                if kind == "label":
                    if relation in canonical and canonical[relation] != name:
                        raise ValueError(f"{context}: conflicting canonical labels for {relation}")
                    canonical[relation] = name
            try:
                self.primary = primary_relation_rows(
                    index.relations, tokenizer, index.span_syntax, canonical
                )
            except IndexMismatch as error:
                raise IndexMismatch(f"{source}: {error}") from error
        if self.primary is None:
            raise ValueError(
                "relation display needs canonical label rows: rebuild the KG index or "
                "supply relation_labels (XGRAPH_RELATION_LABELS in serving)"
            )
        if len(self.primary) != len(self.ids) or any(
            row < -1
            or row >= index.relations.num_rows
            or (row >= 0 and int(index.relations.owners[row]) != owner)
            for owner, row in enumerate(self.primary)
        ):
            raise ValueError("canonical relation rows must name a row of their own relation or -1")
        frequency = Counter(w for aliases in words for w in set().union(*aliases))
        documents = sum(bool(aliases) for aliases in words)
        self.idf = {w: math.log((1 + documents) / (1 + n)) + 1 for w, n in frequency.items()}
        self.inverted: dict[str, set[int]] = {}
        for relation, aliases in enumerate(words):
            for word in set().union(*aliases):
                self.inverted.setdefault(word, set()).add(relation)
        self.default_alias_ranks = self.alias_ranks("")

    def for_question(self, question: str):
        from xgraph.repair import content_stems

        terms = content_stems(question)
        candidates = set().union(*(self.inverted.get(w, set()) for w in terms))
        scores = {
            r: max(
                sum(self.idf[w] for w in sorted(alias & terms)) / math.sqrt(max(1, len(alias)))
                for alias in self.words[r]
            )
            for r in candidates
        }
        ordered = sorted(scores, key=lambda r: (-scores[r], r))[:MAX_RANKED_RELATIONS]
        previous, rank = None, -1
        result = []
        for relation in ordered:
            score = scores[relation]
            if score != previous:
                rank += 1
            result.append((self.ids[relation], rank))
            previous = score
        return tuple(result)

    def alias_ranks(self, question: str):
        """Per-row fallback priority, scored once per request; ties use label text."""
        from xgraph.repair import content_stems

        terms = content_stems(question)
        priorities = np.empty(len(self.names), dtype=np.int32)
        for rows, aliases in zip(self.rows, self.words, strict=True):
            scored = (
                {
                    row: sum(self.idf[w] for w in sorted(alias & terms))
                    / math.sqrt(max(1, len(alias)))
                    for row, alias in zip(rows, aliases, strict=True)
                }
                if terms
                else dict.fromkeys(rows, 0)
            )
            for rank, row in enumerate(sorted(rows, key=lambda r: (-scored[r], self.names[r], r))):
                priorities[row] = rank
        return priorities

    def display_row(self, relation, priorities, writable=None):
        """The canonical row if writable, else the best remaining scored alias."""
        owner = self.owner_at[relation]
        rows = self.rows[owner]
        if writable is not None:
            rows = [row for row in rows if row in writable]
        primary = int(self.primary[owner])
        return primary if primary in rows else min(rows, key=lambda r: priorities[r], default=-1)


def edited_relations(original, audit):
    """Read the recorded decision; never infer execution from displayed labels."""
    from xgraph.observations import PathRelations

    original = list(original)
    if not 1 <= len(original) <= 8:
        raise ValueError("an audit needs a blind path with one through eight relations")
    decision = audit["decision"]
    selections, result = {}, list(original)
    if decision == "SELECT":
        selections[len(result)] = audit["label"]
    elif decision == "APPEND" and len(result) < 8:
        result.append(audit["relation"])
    elif decision == "REPLACE":
        result[-1] = audit["relation"]
    elif decision == "DROP" and len(result) > 1:
        result.pop()
    elif decision != "KEEP":
        raise ValueError(f"invalid audit decision {decision!r} at {len(original)} relations")
    return PathRelations(result, selections, {**audit, "original_relations": original})


def read_audit(suffix: str, original, syntax: SpanSyntax = _SYNTAX):
    """A completed decision after a forced block, or None for a truncated audit."""
    if not suffix.startswith("\nExecuted: " + syntax.opener):
        return original
    marker = "\nDecision: " + syntax.opener
    _, found, tail = suffix.partition(marker)
    if not found:
        return None
    body, closed, _ = tail.partition(syntax.closer)
    if not closed:
        return None
    audit = None
    if body in ("KEEP", "DROP"):
        audit = {"decision": body}
    for name, delimiter, key in (
        ("SELECT", selection_marker(syntax), "label"),
        ("APPEND", syntax.separator, "relation"),
        ("REPLACE", syntax.separator, "relation"),
    ):
        prefix = name + delimiter + syntax.label_prefix
        if body.startswith(prefix) and body[len(prefix) :]:
            audit = {"decision": name, key: body[len(prefix) :]}
    if audit is None:
        return None
    try:
        return edited_relations(original, audit)
    except (ValueError, KeyError):
        return None


def semantic_text(text: str, syntax: SpanSyntax = _SYNTAX):
    """Hide forced menus, frontier readback and audit displays; retain selections."""
    import re

    from xgraph.menus import semantic_text as without_menus
    from xgraph.observations import strip_observations

    text = strip_observations(without_menus(text, syntax), syntax)

    # Terminal observation has no decision and cannot change the closed program.
    marker = "\nEndpoints: " + syntax.opener
    at = text.find(marker)
    if at >= 0:
        end = text.find(syntax.closer, at + len(marker))
        text = text[:at] + (text[end + len(syntax.closer) :] if end >= 0 else "")

    at = text.find("\nExecuted: " + syntax.opener)
    if at < 0:
        return text
    prefix, suffix = text[:at], text[at:]
    marker = "\nDecision: " + syntax.opener
    _, found, tail = suffix.partition(marker)
    if not found or syntax.closer not in tail:
        return prefix
    _, _, rest = tail.partition(syntax.closer)
    if rest.startswith("\nExecuted: " + syntax.opener):
        _, _, rest = rest.partition(syntax.closer)
    pattern = re.escape(syntax.opener) + "(.*?)" + re.escape(syntax.closer)
    for match in re.finditer(pattern, prefix, re.DOTALL):
        from xgraph.observations import path_parts

        parts, _ = path_parts(match[1], syntax.separator, syntax.opener, syntax.escape)
        if len(parts) < 2:
            continue
        relations = read_audit(suffix, parts[1:], syntax)
        if relations is not None:
            path = syntax.opener + syntax.label_prefix + parts[0]
            path += "".join(syntax.separator + syntax.label_prefix + r for r in relations)
            prefix = prefix[: match.start()] + path + syntax.closer + prefix[match.end() :]
        break
    return prefix + rest


def count_terminal_tokens(tokens, tokenizer, syntax: SpanSyntax, *, prefill=""):
    """Charge the terminal display, forced answer prefix and EOS once after decoding."""
    import re

    from xgraph.observations import answer_prefix

    prefix = prefill.encode()
    data = [tokenizer.token_bytes(t) or b"" for t in tokens]
    text = prefix + b"".join(data)
    opener, closer, separator, escape = (
        d.encode() for d in (syntax.opener, syntax.closer, syntax.separator, syntax.escape)
    )
    spans = re.finditer(re.escape(opener) + b"(.*?)" + re.escape(closer), text, re.DOTALL)
    path = next((m for m in spans if separator in m[1] and escape not in m[1]), None)
    if path is None:
        return 0
    at = path.end()
    heading = ("\nEndpoints: " + syntax.opener).encode()
    tail = text[at:]
    if tail.startswith(heading) or heading.startswith(tail):
        end = text.find(closer, at + len(heading))
        at = len(text) if end < 0 else end + len(closer)
    forced_end = at + len(answer_prefix(syntax).encode())
    closed = text.find(closer, forced_end)
    answered = closed + len(closer) if closed >= 0 else None
    offset, count = len(prefix), 0
    for piece in data:
        end = offset + len(piece)
        if (answered is not None and offset >= answered) or (
            piece and offset < forced_end and end > path.end()
        ):
            count += 1
        offset = end
    return count


def count_audit_tokens(tokens, tokenizer, syntax: SpanSyntax, *, prefill=""):
    """Charge forced displays/prefixes and EOS, including truncated generated tails.

    Decision commands, their label, and their closer belong to the model's
    decision. A replayed prefill establishes state but contributes no charged tokens.
    """
    import re

    from xgraph.observations import answer_prefix

    prefix = prefill.encode()
    data = [tokenizer.token_bytes(t) or b"" for t in tokens]
    text = prefix + b"".join(data)
    opener, closer, separator, escape = (
        d.encode() for d in (syntax.opener, syntax.closer, syntax.separator, syntax.escape)
    )
    spans = re.finditer(re.escape(opener) + b"(.*?)" + re.escape(closer), text, re.DOTALL)
    path = next((m for m in spans if separator in m[1] and escape not in m[1]), None)
    if path is None:
        return 0
    at = path.end()
    decision = ("\nDecision: " + syntax.opener).encode()
    start = text.find(decision, at)
    forced_end = len(text) if start < 0 else start + len(decision)
    ranges = [(at, forced_end)]
    answered = None
    if start >= 0:
        end = text.find(closer, forced_end)
        if end >= 0:
            body, at = text[forced_end:end], end + len(closer)
            if body.startswith((b"APPEND", b"REPLACE", b"DROP")):
                end = text.find(closer, at)
                end = len(text) if end < 0 else end + len(closer)
                ranges.append((at, end))
                at = end
            end = at + len(answer_prefix(syntax).encode())
            ranges.append((at, end))
            closed = text.find(closer, end)
            if closed >= 0:
                answered = closed + len(closer)
    offset, count = len(prefix), 0
    for piece in data:
        end = offset + len(piece)
        if answered is not None and offset >= answered:
            # Served EOS ids are explicit metadata; they need not be special ids.
            # After the bound answer, the audit engine admits only those ids.
            count += 1
        elif piece:
            count += any(offset < hi and end > lo for lo, hi in ranges)
        offset = end
    return count
