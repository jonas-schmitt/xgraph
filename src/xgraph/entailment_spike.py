"""CPU-only bounded rule replay for the entailment feasibility study.

This module does not change the oracle, CUDA engine or decoding masks.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from xgraph.programs import MatchedExecutor


@dataclass(frozen=True)
class Rule:
    name: str
    head: str
    body: tuple[tuple[str, bool], ...]


class BoundedRules:
    def __init__(self, rules=(), *, depth=3):
        rules = tuple(rules)
        if type(depth) is not int or not 1 <= depth <= 8:
            raise ValueError("bounded rule depth must be in [1, 8]")
        if any(not r.head or not r.body for r in rules):
            raise ValueError("a rule needs a relation head and a nonempty body")
        self.rules = rules
        self.depth = depth
        self.programs = lru_cache(maxsize=256)(self._programs)

    def _programs(self, relation):
        """Finite rewritings into base edges; the bound counts base edges.

        Keep each original program too: derivable edges supplement stored edges.
        Inverse declarations are directional, and reversing a body reverses its
        order as well as every edge. A budget failure raises, never truncates.
        """
        seen = {((relation, False),)}
        pending = list(seen)
        while pending:
            program = pending.pop()
            for at, (head, reverse) in enumerate(program):
                for rule in self.rules:
                    if rule.head != head:
                        continue
                    body = rule.body
                    if reverse:
                        body = tuple((p, not rev) for p, rev in reversed(body))
                    expanded = (*program[:at], *body, *program[at + 1 :])
                    if len(expanded) <= self.depth and expanded not in seen:
                        seen.add(expanded)
                        pending.append(expanded)
                        if len(seen) > 10_000:
                            raise ValueError(f"rule rewriting exceeds budget for {relation}")
        return tuple(sorted(seen))

    def objects(self, subject, relation, neighbors):
        reached = set()
        for program in self.programs(relation):
            frontier = {subject}
            for prop, reverse in program:
                frontier = {o for s in frontier for o in neighbors(s, prop, reverse)}
                if not frontier:
                    break
            reached.update(frontier)
        return reached


def curated_rules():
    """Explicit experimental policy, not schema declarations found in the KG.

    P31 is membership: inherit along P279; never apply P31/P31 transitivity.
    This finite, positive path-rule subset is not a complete OWL RL/EL reasoner.
    """
    return (
        Rule("P279-transitivity", "P279", (("P279", False), ("P279", False))),
        Rule("P361-transitivity", "P361", (("P361", False), ("P361", False))),
        Rule("class-inheritance", "P31", (("P31", False), ("P279", False))),
    )


class EntailmentExecutor(MatchedExecutor):
    """Fixed-plan CPU sensitivity with stored start/answer ranks and caps.

    Only the walk's edge view changes. Stored degree breaks start/answer ties
    in every arm. Selections, relation aliases and label-row caps remain those
    of MatchedExecutor. No mask, inference interface or on-disk KG is modified.
    """

    def __init__(self, index, tokenizer, *, rules, depth=3, **kwargs):
        super().__init__(index, tokenizer, **kwargs)
        self.rules = BoundedRules(rules, depth=depth)
        self._row_order = np.argsort(index.labels.owners, kind="stable")
        self._row_offsets = np.concatenate(
            (
                [0],
                np.cumsum(np.bincount(index.labels.owners, minlength=len(index.labels.owner_ids))),
            )
        )

    def _start_entities(self, view, *args, **kwargs):
        return super()._start_entities(getattr(view, "base", view), *args, **kwargs)

    def _order(self, view, pairs):
        return super()._order(getattr(view, "base", view), pairs)

    def _walk(self, view, start, relations, *args, **kwargs):
        heads = {
            self._index.relations.owner_ids[r]
            for label in relations
            for r in self._rows_owned(self._index.relations, label)
        }
        heads &= {r.head for r in self.rules.rules}
        if heads:
            view = _EntailedView(self, view, heads)
        return super()._walk(view, start, relations, *args, **kwargs)


class _EntailedView:
    def __init__(self, executor, base, heads):
        self.executor, self.base, self.heads = executor, base, heads
        self.reverse = {}
        self.neighbors = lru_cache(maxsize=16_384)(self._neighbors)
        self.edges_at = lru_cache(maxsize=4096)(self._edges_at)

    def _neighbors(self, subject, prop, reverse):
        index = self.executor._index
        position = index._relation_at.get(prop)
        if position is None:
            return frozenset()
        if reverse:
            if prop not in self.reverse:
                if self.base is not index:
                    raise ValueError("inverse replay of KG versions needs a versioned reverse CSR")
                selected = np.flatnonzero(index.edge_relations == position)
                subjects = np.searchsorted(index.adjacency_offsets, selected, side="right") - 1
                reverse_edges = {}
                for s, row in zip(subjects, index.edge_objects[selected], strict=True):
                    owner = int(index.labels.owners[row])
                    reverse_edges.setdefault(owner, set()).add(int(s))
                self.reverse[prop] = reverse_edges
            return self.reverse[prop].get(subject, frozenset())
        relations, objects = self.base.edges_at(subject)
        first, last = (
            np.searchsorted(relations, position, side="left"),
            np.searchsorted(relations, position, side="right"),
        )
        return frozenset(map(int, index.labels.owners[objects[first:last]]))

    def _edges_at(self, subject):
        ex = self.executor
        relations, objects = self.base.edges_at(subject)
        extra = set()
        for head in self.heads:
            direct = self.neighbors(subject, head, False)
            inferred = ex.rules.objects(subject, head, self.neighbors) - direct
            for owner in inferred:
                first, last = ex._row_offsets[owner : owner + 2]
                extra.update(
                    (ex._index._relation_at[head], int(r)) for r in ex._row_order[first:last]
                )
        if not extra:
            return relations, objects
        edges = sorted(set(zip(map(int, relations), map(int, objects), strict=True)) | extra)
        return np.asarray([p for p, _ in edges]), np.asarray([o for _, o in edges])
