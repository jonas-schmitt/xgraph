"""Interventions: KG versions that change one fact on or off a question's gold chain.

Two questions share these inputs. Where does an answer's value come from: bound by
executing a plan (path mode), read from facts in the prompt, written by the model, or
recalled closed book? And is an answer faithful to the KG: does it change when a fact on
its chain changes, and hold when a fact off the chain changes? Each sampled question is
asked under several conditions, each a KG version (xgraph.kgversions) of the KG index
the engine serves, with the gold answer the version implies.

**The plan and its gold.** A question comes with its gold chain (subject, relation,
object) of QIDs: MINTQA's chains.json (python -m xgraph.mintqa_bench), or MQuAKE's edited
chains (python -m xgraph.planning chains). Its plan is the chain's relations, executed
from the chain's start over the KG store (xgraph.toolkg) as the version holds it: an
edit replaces every object of its (subject, relation) pair, and only entities with a
label row count, as in the KG index the version is read over. A question is used only
when its chain holds and its plan reaches exactly its gold endpoint, which must have a
label; every condition's gold is then the one entity its plan reaches in the condition's
version, computed, not assumed. For MQuAKE (--mquake), each case's own edits hold in
every condition (they are its base edits) and only each case's first question is asked.

**Conditions**, in order:

- `original`: no further edit (for MQuAKE, the case's edits), gold the chain's endpoint;
- `unedited` (MQuAKE only): no edit at all, gold what the plan reaches in the base KG,
  the real-world answer (stratum `real`);
- on-chain, `<position>-<stratum>`: the object of one hop replaced by a substitute, at the
  last hop (`last`) and at a non-last hop drawn with the question's seed (`inner`), whose
  new endpoint the remaining relations reach from the substitute;
- off-chain, `off-start` and `off-inner`: the object of a fact of the start entity, or of
  a chain subject after it drawn with the question's seed, under a relation that is
  neither on the chain nor a class relation (instance of, subclass of), whose pair has
  one object, replaced by a popular substitute of that object's class (`same-class`
  without sitelink counts): the gold must not change.

**Substitutes**, by a fixed rule. The replaced object's class is, of its classes under
instance of (P31), or else subclass of (P279), the one with the most members that are
objects of the replaced fact's relation (then the most members, then the lowest id): the
class the relation's values come from. Classes come from the KG directory, or from another
one (--classes) where its own are unreliable, as in MINTQA's release, whose instance-of
facts merge items that share a label. A substitute is an entity with a label (not the QID
placeholder MINTQA writes for an unlabelled item) other than the replaced object, named
apart from it, and no Wikimedia project page (PROJECT_PAGES), with which the plan reaches
one labelled entity other than the gold endpoint, neither of them naming the other (by
canonical label against labels and aliases). Strata, with sitelink counts (the KG
directory's sitelink_counts.npy, python -m xgraph.wikidata sitelink-counts, or
--sitelinks):

- `popular`: one of the --pool most linked such members of the class, drawn;
- `long-tail`: a member from the class's less linked half (by sitelink rank), drawn;
- `cross-class`: one of the --pool most linked such entities that have a class under the
  same class relation and share none with the replaced object, drawn.

Without sitelink counts the popularity strata give way to `same-class` (any member,
drawn) and `cross-class` (any such entity, drawn). For a non-last hop the substitute must
have the plan's next relation (in the KG directory's triples). Draws use a random
generator seeded with the seed and the question id, so a question's conditions do not
depend on which others were sampled.
A version adds no labels (docs/kg-versions-adr.md), which is why every substitute is an
existing labelled entity.

**Sampling.** The questions with a gold chain are grouped by their complexity field up
to the hop count (`ti 3-hop`, `2-hop`), each group shuffled with the seed, and taken
round robin, one usable question per group per round, until --size are taken: groups
are balanced until they run out. A question is usable when it has its original
condition, an on-chain one, and an off-chain one; `skipped` records why others were not.

**Files** written to --out: `conditions.json` (every question's conditions: version name,
edits, gold QIDs and labels, the hop and position changed, the stratum, the substitute
with its class relation, class, and sitelink count, and the facts a reader sees),
`kg-versions/` (a version file with every condition that has edits, named
`<question>.<condition>`), `labels.jsonl` (the gold answers' labels and aliases from the
KG directory and any --labels files), and requests (JSONL, ids `<question>@<condition>`,
each with its condition's `kg_version` where it has one):

- `requests-paths.jsonl`: path mode's prompt (xgraph.prompts, path spans), unanchored;
- `requests-paths-anchor.jsonl`: the same, anchored to the chain's start;
- `requests-read-answer.jsonl` and `requests-read-copy.jsonl`: the facts a reader sees in
  the prompt, as `subject · relation · object` lines (xgraph.rag): the facts the plan
  walks in the condition's version, after the facts of the off-chain conditions' pairs,
  so that every condition of a question lists facts of the same pairs; asked to answer
  (rag_messages) or to copy the answer from them (copy_messages);
- `requests-write.jsonl`: the chain prompt (each hop a fact span the model writes),
  anchored, for derivation mode (XG_DERIVATION=1), which binds each written fact to the
  version's KG;
- `requests-closed.jsonl`: the few-shot prompt, closed book, once per question (id
  `<question>`), since no version changes it.

    python -m xgraph.interventions build --questions target/mintqa/questions-representable.json \\
        --chains target/mintqa/chains.json --kg data/external/mintqa-full/kg \\
        --db data/external/mintqa-full/kg.sqlite --classes data/wikidata/gate-kg-v3e \\
        --labels target/mintqa/labels.jsonl --size 500 --seed 0 \\
        --out target/interventions/mintqa

**Selection** (`select`): a build's questions that have every condition --require lists
and at least --min-hops hops, in the build's order, each keeping only those conditions;
with --outside-inventory (MQuAKE JSON), only those whose chain uses a relation outside
MQuAKE's relation inventory (the relations of its requested rewrites, which are GWalk's
relation templates). The selection is written as a build is: `conditions.json` (with the
selection's settings, and its counts: questions, conditions, KG versions, relation
sequences, and with an inventory the questions whose on-chain conditions edit a relation
outside it), `kg-versions/` with only the kept versions, `labels.jsonl`, and every request
file of the build filtered to the kept conditions, so that it is scored like a build.

    python -m xgraph.interventions select --from target/interventions/mintqa \\
        --require original,inner-popular,last-popular,off-start --min-hops 2 \\
        --outside-inventory data/mquake/MQuAKE-Remastered-CF-3k.json \\
        --out target/interventions/mintqa/open-inventory

**Scores** (`score`): for each arm (--arm <name>=<records>; a record `<question>` stands
for every condition of its question when `<question>@<condition>` is missing, as closed
book's), each condition's answer (`Answer: ⟦ label⟧`) against its gold by label or alias
(xgraph.score): accuracy; `original`; `sensitivity`, the on-chain conditions answered with
their new gold, beside `old_answer`, those answered with the original gold, and
`sensitivity_given_original`, over the questions whose original condition is right;
`invariance`, the off-chain conditions answered with the original gold; `joint version
fidelity`, the questions all of whose conditions are right; on-chain scores by stratum,
by position, and by hop; and per condition. The first arm is compared with every other one,
paired over questions with a bootstrap interval that resamples whole clusters of
questions sharing a relation sequence (xgraph.score.paired_bootstrap,
paired_ratio_bootstrap): the reference minus the other.

    python -m xgraph.interventions score --conditions target/interventions/mintqa/conditions.json \\
        --labels target/interventions/mintqa/labels.jsonl \\
        --arm paths=<paths.jsonl> --arm read-answer=<read-answer.jsonl> \\
        --out target/interventions/mintqa/scores.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np

CLASS_RELATIONS = (31, 279)
"""instance of, then subclass of: the relations a replaced object's class is read from."""
PROJECT_PAGES = frozenset(
    (4167836, 4663903, 11266439, 13406463, 4167410, 17442446, 15184295, 14204246, 16695773)
)
"""Classes of Wikimedia project pages (category, portal, template, list article,
disambiguation page, internal item, module, project page, WikiProject). A project page,
an instance of one of them or an item whose label starts with a namespace prefix
(`Category:Asia`, `Wikipedia:Sandbox`) or with Wikimedia, Wikipedia or WikiProject, is no
world entity: never a substitute, nor the object an off-chain condition replaces."""
_PROJECT_LABEL = re.compile(r"[A-Za-z]+:\S|(Wikimedia|Wikipedia|WikiProject)\b")
POOL = 20
"""Popular and cross-class substitutes are drawn from this many most linked candidates."""
CHECKS = 200
"""The most candidates checked for one substitute."""
OFF_RELATIONS = 10
"""The most relations tried for one off-chain condition."""
FACTS = 30
"""The most facts a plan may walk in one condition (a reader's prompt)."""
FRONTIER = 1000
"""The most entities a plan may reach after one relation."""
TOP = 100_000
"""The most candidates walked or drawn for one substitute, before the checks."""
_PLACEHOLDER = re.compile(r"Q\d+")

Edits = dict[tuple[int, int], tuple[int, ...]]
Fact = tuple[int, int, int]


def _number(item: str) -> int:
    return int(str(item)[1:])


class InterventionKG:
    """A KG directory and its KG store, read as KG versions are: the store answers
    lookups, and the directory's triples give each relation's subjects and objects and,
    unless another KG directory gives them (classes), each entity's classes and each
    class's members, ordered by sitelink count when there is one."""

    def __init__(
        self,
        kg: str | Path,
        db: str | Path,
        sitelinks: str | Path | None = None,
        classes: str | Path | None = None,
    ):
        from xgraph.toolkg import KGStore
        from xgraph.wikidata import SITELINK_COUNTS

        kg = Path(kg)
        self.store = KGStore(db)
        self.relation_labels = dict(
            self.store._con.execute("SELECT prop, label FROM relations").fetchall()
        )
        path = Path(sitelinks) if sitelinks else kg / SITELINK_COUNTS
        self.sitelinks_file = path if path.is_file() else None
        if sitelinks and self.sitelinks_file is None:
            raise FileNotFoundError(f"no sitelink counts at {sitelinks}")
        if self.sitelinks_file is not None:
            rows = np.load(self.sitelinks_file).reshape(-1, 2).astype(np.int64)
            rows = rows[np.argsort(rows[:, 0], kind="stable")]
            self._linked, self._links = rows[:, 0], rows[:, 1]
        triples = np.load(kg / "triples.npy", mmap_mode="r")
        relations = np.asarray(triples[:, 1])
        order = np.argsort(relations, kind="stable")
        ordered = relations[order]
        self._relations, starts = np.unique(ordered, return_index=True)
        self._bounds = np.append(starts, len(ordered))
        del ordered, relations
        self._subjects_by_relation = np.asarray(triples[:, 0])[order]
        self._objects_by_relation = np.asarray(triples[:, 2])[order]
        del order
        if classes is not None:
            triples = np.load(Path(classes) / "triples.npy", mmap_mode="r")
        class_rows = _rows_of(triples, CLASS_RELATIONS)
        self._classes: dict[int, tuple[np.ndarray, ...]] = {}
        for relation in CLASS_RELATIONS:
            rows = class_rows[class_rows[:, 1] == relation]
            subjects, found = rows[:, 0], rows[:, 2]
            by_item = np.lexsort((found, subjects))
            links = self.sitelinks(subjects)
            by_class = np.lexsort((subjects, -links, found))
            typed = np.unique(subjects)
            typed = typed[np.lexsort((typed, -self.sitelinks(typed)))]
            self._classes[relation] = (
                subjects[by_item],
                found[by_item],
                found[by_class],
                subjects[by_class],
                typed,
            )
        self._sets: dict[tuple[str, int], np.ndarray] = {}
        self._overlaps: dict[tuple[int, int, int], int] = {}
        self._label_cache: dict[int, str | None] = {}

    @property
    def has_sitelinks(self) -> bool:
        return self.sitelinks_file is not None

    def _slice(self, relation: int) -> tuple[int, int]:
        at = int(np.searchsorted(self._relations, relation))
        if at == len(self._relations) or self._relations[at] != relation:
            return 0, 0
        return int(self._bounds[at]), int(self._bounds[at + 1])

    def sitelinks(self, items: np.ndarray) -> np.ndarray:
        """The items' sitelink counts (0 for an item without a row, or without counts)."""
        items = np.asarray(items, dtype=np.int64)
        if self.sitelinks_file is None or not len(self._linked):
            return np.zeros(len(items), dtype=np.int64)
        at = np.minimum(np.searchsorted(self._linked, items), len(self._linked) - 1)
        return np.where(self._linked[at] == items, self._links[at], 0)

    def sitelink(self, item: int) -> int | None:
        return int(self.sitelinks(np.array([item]))[0]) if self.has_sitelinks else None

    def raw_label(self, item: int) -> str | None:
        """The item's canonical label as the store holds it, a placeholder included."""
        if item not in self._label_cache:
            self._label_cache[item] = self.store.label(item)
        return self._label_cache[item]

    def label(self, item: int) -> str | None:
        """The item's canonical label, None for none or a QID placeholder."""
        text = self.raw_label(item)
        return None if text is None or _PLACEHOLDER.fullmatch(text) else text

    def relation_label(self, prop: int) -> str:
        return self.relation_labels.get(prop, f"P{prop}")

    def objects(self, item: int, prop: int, edits: Edits) -> list[int]:
        """The labelled objects of (item, prop) in the version with these edits."""
        if self.raw_label(item) is None:
            return []
        if (item, prop) in edits:
            return sorted(o for o in set(edits[item, prop]) if self.raw_label(o) is not None)
        return sorted(self.store.objects_of([item], prop))

    def execute(
        self, start: int, props: list[int], edits: Edits
    ) -> tuple[frozenset[int], list[Fact]] | None:
        """What the plan reaches from start in the version, and the facts it walks; None
        when it reaches more than FRONTIER entities after a relation."""
        frontier, walked = [start] if self.raw_label(start) is not None else [], []
        for prop in props:
            reached: set[int] = set()
            for item in frontier:
                objects = self.objects(item, prop, edits)
                walked.extend((item, prop, o) for o in objects)
                reached.update(objects)
            if len(reached) > FRONTIER:
                return None
            frontier = sorted(reached)
        return frozenset(frontier), walked

    def classes(self, item: int, relation: int | None = None) -> tuple[int, list[int]] | None:
        """The item's class relation and its classes under it: the given relation, or else
        instance of, or else subclass of; None without classes."""
        for under in (relation,) if relation is not None else CLASS_RELATIONS:
            subjects, classes, *_ = self._classes[under]
            lo, hi = np.searchsorted(subjects, [item, item + 1])
            if hi > lo:
                return under, [int(c) for c in classes[lo:hi]]
        return None

    def project_page(self, item: int) -> bool:
        """Whether the item is a Wikimedia project page (PROJECT_PAGES)."""
        if _PROJECT_LABEL.match(self.raw_label(item) or ""):
            return True
        found = self.classes(item, CLASS_RELATIONS[0])
        return found is not None and not PROJECT_PAGES.isdisjoint(found[1])

    def members(self, relation: int, cls: int) -> np.ndarray:
        """The class's members, most linked first (then by item)."""
        _, _, classes, subjects, _ = self._classes[relation]
        lo, hi = np.searchsorted(classes, [cls, cls + 1])
        return subjects[lo:hi]

    def typed(self, relation: int) -> np.ndarray:
        """Every entity with a class under the relation, most linked first."""
        return self._classes[relation][4]

    def relation_items(self, prop: int, side: str) -> np.ndarray:
        """The relation's subjects or objects in the KG directory's triples, sorted."""
        if (side, prop) not in self._sets:
            lo, hi = self._slice(prop)
            column = self._subjects_by_relation if side == "subjects" else self._objects_by_relation
            self._sets[side, prop] = np.unique(column[lo:hi])
        return self._sets[side, prop]

    def has_relation(self, item: int, prop: int) -> bool:
        """Whether the item is a subject of the relation in the KG directory's triples."""
        subjects = self.relation_items(prop, "subjects")
        at = int(np.searchsorted(subjects, item))
        return at < len(subjects) and int(subjects[at]) == item

    def range_share(self, relation: int, cls: int, prop: int) -> int:
        """How many members of the class are objects of the relation prop."""
        if (relation, cls, prop) not in self._overlaps:
            members = self.members(relation, cls)
            found = np.isin(members, self.relation_items(prop, "objects")).sum()
            self._overlaps[relation, cls, prop] = int(found)
        return self._overlaps[relation, cls, prop]


def _rows_of(triples: np.ndarray, relations: tuple[int, ...]) -> np.ndarray:
    """The triples (int64) whose relation is one of these, read in chunks."""
    found = [np.zeros((0, 3), dtype=np.int64)]
    for at in range(0, len(triples), 1 << 22):
        chunk = np.asarray(triples[at : at + (1 << 22)], dtype=np.int64)
        found.append(chunk[np.isin(chunk[:, 1], relations)])
    return np.concatenate(found)


def _merge(base: Edits, extra: Edits) -> Edits:
    return {**base, **extra}


def _rows(edits: Edits) -> list[list[int]]:
    return sorted([s, p, o] for (s, p), objects in edits.items() for o in objects)


def _same_names(kg: InterventionKG, a: int, b: int) -> bool:
    """Whether either item's canonical label names the other (label or alias)."""
    from xgraph.score import normalize

    label_a, label_b = kg.raw_label(a) or "", kg.raw_label(b) or ""
    if normalize(label_a) == normalize(label_b):
        return True
    return b in kg.store.items_named(label_a) or a in kg.store.items_named(label_b)


def _candidates(
    kg: InterventionKG,
    stratum: str,
    relation: int,
    cls: int,
    classes: list[int],
    prop: int | None,
    rng: random.Random,
) -> Iterator[int]:
    """Candidate substitutes in the order the stratum checks them: most linked first for a
    popular or (with sitelink counts) a cross-class stratum, else drawn at random; only
    subjects of prop when it is given, and for cross-class only entities with classes
    under the relation, none of them shared."""

    def fits(item: int) -> bool:
        if prop is not None and not kg.has_relation(item, prop):
            return False
        if stratum != "cross-class":
            return True
        found = kg.classes(item, relation)
        return found is not None and not set(classes) & set(found[1])

    if stratum == "cross-class":
        pool = kg.typed(relation)
        walk = (int(x) for x in pool[:TOP]) if kg.has_sitelinks else _draws(pool, rng)
    else:  # long-tail draws from the less linked half, same-class from every member
        members = kg.members(relation, cls)
        if prop is not None:
            members = members[np.isin(members, kg.relation_items(prop, "subjects"))]
        if stratum == "popular":
            walk = (int(x) for x in members[:TOP])
        else:
            walk = _draws(members[len(members) // 2 :] if stratum == "long-tail" else members, rng)
    yield from (item for item in walk if fits(item))


def _draws(items: np.ndarray, rng: random.Random) -> Iterator[int]:
    """Distinct items drawn at random, all of them in random order for a few, else up to
    TOP draws."""
    if len(items) <= CHECKS:
        order = [int(x) for x in items]
        rng.shuffle(order)
        yield from order
        return
    seen: set[int] = set()
    for _ in range(TOP):
        item = int(items[rng.randrange(len(items))])
        if item not in seen:
            seen.add(item)
            yield item


def _substitute(
    kg: InterventionKG,
    stratum: str,
    replaced: tuple[int, int, int],
    eligible,
    prop: int | None,
    rng: random.Random,
    pool: int,
) -> tuple[dict, object] | str:
    """A substitute for the replaced fact's object by the stratum's rule, and what
    eligible returned for it; or why there is none. The object's class is the one of its
    classes with the most members that are objects of the fact's relation (then the most
    members, then the lowest id): the class the relation's values are drawn from."""
    _, hop_relation, replaced = replaced
    found_classes = kg.classes(replaced)
    if found_classes is None:
        return "the replaced object has no class"
    relation, classes = found_classes
    cls = max(
        classes,
        key=lambda c: (kg.range_share(relation, c, hop_relation), len(kg.members(relation, c)), -c),
    )
    candidates = _candidates(kg, stratum, relation, cls, classes, prop, rng)
    drawn = kg.has_sitelinks and stratum in ("popular", "cross-class")
    chosen: list[tuple[int, object]] = []
    for checked, item in enumerate(candidates):
        if checked >= CHECKS or (drawn and len(chosen) >= pool):
            break
        if item == replaced or kg.label(item) is None or kg.project_page(item):
            continue
        if _same_names(kg, item, replaced):
            continue
        result = eligible(item)
        if result is not None:
            chosen.append((item, result))
            if not drawn:
                break
    if not chosen:
        return f"no {stratum} substitute"
    item, result = rng.choice(chosen) if drawn else chosen[0]
    return {
        "id": f"Q{item}",
        "label": kg.label(item),
        "replaces": f"Q{replaced}",
        "class_relation": f"P{relation}",
        "class": f"Q{cls}",
        "sitelinks": kg.sitelink(item),
    }, result


def interventions(
    kg: InterventionKG,
    qid: str,
    chain: list[list[str]],
    seed: int = 0,
    pool: int = POOL,
    base_edits: list[list[str]] | None = None,
) -> dict | str:
    """A question's conditions (module docstring), or why the question cannot be used."""
    hops = [(_number(s), _number(p), _number(o)) for s, p, o in chain]
    start, props, gold = hops[0][0], [p for _, p, _ in hops], hops[-1][2]
    base: Edits = {}
    for s, p, o in base_edits or []:
        base[_number(s), _number(p)] = (*base.get((_number(s), _number(p)), ()), _number(o))
    rng = random.Random(f"{seed}:{qid}")
    for k, (s, p, o) in enumerate(hops):
        if (k and s != hops[k - 1][2]) or o not in kg.objects(s, p, base):
            return f"the chain does not hold in the KG at hop {k + 1}"
    executed = kg.execute(start, props, base)
    if executed is None or executed[0] != {gold}:
        n = len(executed[0]) if executed else f"more than {FRONTIER}"
        return f"the plan reaches {n} entities, not only the gold endpoint"
    if kg.label(gold) is None:
        return "the gold endpoint has no label"
    if len(executed[1]) > FACTS:
        return f"the plan walks more than {FACTS} facts"

    def reaches(edits: Edits) -> tuple[int, list[Fact]] | None:
        """The one labelled entity the plan reaches in the version, other than the gold
        and named apart from it, with the facts walked; else None."""
        found = kg.execute(start, props, edits)
        if found is None or len(found[0]) != 1 or len(found[1]) > FACTS:
            return None
        (item,) = found[0]
        if item == gold or kg.label(item) is None or _same_names(kg, item, gold):
            return None
        return item, found[1]

    def keeps(edits: Edits) -> tuple[int, list[Fact]] | None:
        found = kg.execute(start, props, edits)
        if found is None or found[0] != {gold} or len(found[1]) > FACTS:
            return None
        return gold, found[1]

    conditions: list[dict] = []
    dropped: dict[str, str] = {}
    n = len(hops)

    def position(hop: int) -> str:
        return "last" if hop == n else "inner"

    def add(name, kind, stratum, where, hop, edits, reached, substitute=None, pair=None):
        conditions.append(
            {
                "name": name,
                "kind": kind,
                "stratum": stratum,
                "position": where,
                "hop": hop,
                "pair": [f"Q{pair[0]}", f"P{pair[1]}"] if pair else None,
                "edits": edits,
                "gold": [f"Q{reached[0]}"],
                "gold_labels": [kg.label(reached[0])],
                "substitute": substitute,
                "_walked": reached[1],
            }
        )

    add("original", "original", None, None, None, base, (gold, executed[1]))
    if base:
        found = reaches({})
        edited = [k + 1 for k, (s, p, _) in enumerate(hops) if (s, p) in base]
        if found is None or not edited:
            dropped["unedited"] = "the base KG's plan reaches no other single labelled entity"
        else:
            add("unedited", "on-chain", "real", position(edited[0]), edited[0], {}, found)
    strata = ("popular", "long-tail") if kg.has_sitelinks else ("same-class",)
    strata += ("cross-class",)
    chosen_hops = [n] + ([rng.randrange(1, n)] if n > 1 else [])
    for hop in chosen_hops:
        s, p, o = hops[hop - 1]
        following = props[hop] if hop < n else None
        for stratum in strata:
            name = f"{position(hop)}-{stratum.replace('-', '')}"

            def eligible(item: int, s=s, p=p) -> tuple | None:
                edits = _merge(base, {(s, p): (item,)})
                found = reaches(edits)
                return None if found is None else (edits, found)

            picked = _substitute(kg, stratum, (s, p, o), eligible, following, rng, pool)
            if isinstance(picked, str):
                dropped[name] = picked
                continue
            substitute, (edits, found) = picked
            add(name, "on-chain", stratum, position(hop), hop, edits, found, substitute, (s, p))
    off_stratum = "popular" if kg.has_sitelinks else "same-class"
    on_chain = set(props) | set(CLASS_RELATIONS)
    subjects = [("off-start", "start", start)]
    if n > 1:
        subjects.append(("off-inner", "inner", hops[rng.randrange(1, n)][0]))
    off_pairs: list[tuple[int, int]] = []
    for name, where, subject in subjects:
        relations = set(kg.store.relations_of([subject])) | {p for (s, p) in base if s == subject}
        relations = sorted(relations - on_chain)
        rng.shuffle(relations)
        picked: tuple | str = "no fact off the chain with one object"
        for relation in relations[:OFF_RELATIONS]:
            objects = kg.objects(subject, relation, base)
            if len(objects) != 1 or kg.project_page(objects[0]):
                continue

            def eligible(item: int, subject=subject, relation=relation) -> tuple | None:
                edits = _merge(base, {(subject, relation): (item,)})
                found = keeps(edits)
                return None if found is None else (edits, found)

            fact = (subject, relation, objects[0])
            picked = _substitute(kg, off_stratum, fact, eligible, None, rng, pool)
            if not isinstance(picked, str):
                substitute, (edits, found) = picked
                add(name, "off-chain", off_stratum, where, None, edits, found, substitute,
                    (subject, relation))  # fmt: skip
                off_pairs.append((subject, relation))
                break
        if isinstance(picked, str):
            dropped[name] = picked
    if not any(c["kind"] == "on-chain" for c in conditions):
        return "no on-chain condition"
    if not any(c["kind"] == "off-chain" for c in conditions):
        return "no off-chain condition"

    def labelled(facts: list[Fact]) -> list[list[str]]:
        return [
            [kg.raw_label(s) or f"Q{s}", kg.relation_label(p), kg.raw_label(o) or f"Q{o}"]
            for s, p, o in facts
        ]

    for condition in conditions:
        edits = condition["edits"]
        context = [
            (s, p, o) for s, p in off_pairs for o in kg.objects(s, p, edits)
        ]  # the off-chain pairs' facts in this version
        condition["facts"] = labelled(context + condition.pop("_walked"))
        condition["edits"] = _rows(edits)
        name = condition["name"]
        condition["version"] = f"{qid}.{name}" if condition["edits"] else None
    return {
        "id": qid,
        "start": f"Q{start}",
        "gold": [f"Q{gold}"],
        "gold_labels": [kg.label(gold)],
        "relation_sequence": ">".join(f"P{p}" for p in props),
        "chain": [list(hop) for hop in chain],
        "conditions": conditions,
        "dropped": dropped,
    }


def group_of(complexity: str) -> str:
    """A question's sampling group: its complexity up to the hop count."""
    words = complexity.split()
    for k, word in enumerate(words):
        if word.endswith("-hop"):
            return " ".join(words[: k + 1])
    return complexity or "all"


def sample(
    candidates: list[dict], size: int, seed: int, build
) -> tuple[list[dict], dict[str, str], dict[str, int]]:
    """Usable questions taken round robin over the groups (module docstring): built
    questions in candidate order, why the questions tried and not used were skipped, and
    how many each group gave."""
    groups: dict[str, list[dict]] = {}
    for q in candidates:
        groups.setdefault(group_of(q.get("complexity", "")), []).append(q)
    queues = {}
    for name in sorted(groups):
        members = list(groups[name])
        random.Random(f"{seed}:{name}").shuffle(members)
        queues[name] = iter(members)
    built: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    taken = dict.fromkeys(sorted(groups), 0)
    while len(built) < size and queues:
        for name in list(queues):
            if len(built) >= size:
                break
            for q in queues[name]:
                result = build(q)
                if isinstance(result, str):
                    skipped[q["id"]] = result
                    continue
                built[q["id"]] = result
                taken[name] += 1
                break
            else:
                del queues[name]
    order = [q["id"] for q in candidates if q["id"] in built]
    return [built[qid] for qid in order], skipped, taken


def requests(questions: list[dict]) -> dict[str, list[dict]]:
    """The request files of the module docstring, by file name."""
    from xgraph.prompts import wwq_messages
    from xgraph.rag import copy_messages, rag_messages

    files: dict[str, list[dict]] = {
        name: []
        for name in (
            "requests-paths.jsonl",
            "requests-paths-anchor.jsonl",
            "requests-read-answer.jsonl",
            "requests-read-copy.jsonl",
            "requests-write.jsonl",
            "requests-closed.jsonl",
        )
    }
    for q in questions:
        text, anchor = q["utterance"], [q["start"]]
        files["requests-closed.jsonl"].append({"id": q["id"], "messages": wwq_messages(text)})
        paths, chain = wwq_messages(text, path_spans=True), wwq_messages(text, chain=True)
        for c in q["conditions"]:
            head = {"id": f"{q['id']}@{c['name']}"}
            if c["version"]:
                head["kg_version"] = c["version"]
            facts = [tuple(f) for f in c["facts"]]
            files["requests-paths.jsonl"].append({**head, "messages": paths})
            files["requests-paths-anchor.jsonl"].append(
                {**head, "anchor": anchor, "messages": paths}
            )
            files["requests-read-answer.jsonl"].append(
                {**head, "messages": rag_messages(text, facts)}
            )
            files["requests-read-copy.jsonl"].append(
                {**head, "messages": copy_messages(text, facts)}
            )
            files["requests-write.jsonl"].append({**head, "anchor": anchor, "messages": chain})
    return files


def _label_rows(paths: list[Path], wanted: set[str]) -> list[list[str]]:
    rows: dict[tuple, None] = {}
    for path in paths:
        with path.open(encoding="utf-8") as lines:
            for line in lines:
                row = json.loads(line)
                if row[0] in wanted:
                    rows[tuple(row)] = None
    return [list(row) for row in rows]


def _write_jsonl(path: Path, rows: list) -> None:
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_command(args: argparse.Namespace) -> None:
    from xgraph.kgversions import write_kg_versions
    from xgraph.provenance import provenance

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    chains = json.loads(args.chains.read_text(encoding="utf-8"))
    base: dict[str, list] = {}
    candidates = [q for q in questions if q["id"] in chains]
    if args.mquake:
        from xgraph.mquake import case_id

        edits = {
            f"mquake-{c['case_id']}": c["orig"]["edit_triples"]
            for c in json.loads(args.mquake.read_text(encoding="utf-8"))
        }
        first: dict[str, dict] = {}
        for q in candidates:
            first.setdefault(case_id(q["id"]), q)
        missing = [case for case in first if case not in edits]
        if missing:
            raise SystemExit(f"{args.mquake} holds no cases {missing[:5]}")
        candidates = list(first.values())
        base = {q["id"]: edits[case_id(q["id"])] for q in candidates}
    kg = InterventionKG(args.kg, args.db, args.sitelinks, args.classes)

    def build(q: dict) -> dict | str:
        found = interventions(kg, q["id"], chains[q["id"]], args.seed, args.pool, base.get(q["id"]))
        if isinstance(found, dict):
            found = {"id": q["id"], "utterance": q["utterance"],
                     "complexity": q.get("complexity", ""), **found}  # fmt: skip
        return found

    built, skipped, taken = sample(candidates, args.size, args.seed, build)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    versions = {
        c["version"]: np.array(c["edits"], dtype=np.int32).reshape(-1, 3)
        for q in built
        for c in q["conditions"]
        if c["version"]
    }
    inputs = [args.questions, args.chains, args.kg / "triples.npy", args.kg / "entity_labels.jsonl"]
    inputs += [kg.sitelinks_file] if kg.sitelinks_file else []
    inputs += [args.mquake] if args.mquake else []
    inputs += [args.classes / "triples.npy"] if args.classes else []
    inputs += list(args.labels)
    record = provenance(inputs)
    write_kg_versions(out / "kg-versions", versions, provenance=record)
    for name, rows in requests(built).items():
        _write_jsonl(out / name, rows)
    wanted = {g for q in built for c in q["conditions"] for g in c["gold"]}
    labels = _label_rows([args.kg / "entity_labels.jsonl", *args.labels], wanted)
    _write_jsonl(out / "labels.jsonl", labels)
    conditions = [c for q in built for c in q["conditions"]]
    counts = {
        "questions": len(built),
        "conditions": len(conditions),
        "versions": len(versions),
        "by_condition": {},
        "skipped": len(skipped),
    }
    for c in conditions:
        counts["by_condition"][c["name"]] = counts["by_condition"].get(c["name"], 0) + 1
    result = {
        "artifact_kind": "xgraph.interventions",
        "version": 1,
        "settings": {
            "size": args.size,
            "seed": args.seed,
            "pool": args.pool,
            "kg": str(args.kg),
            "db": str(args.db),
            "sitelinks": str(kg.sitelinks_file) if kg.sitelinks_file else None,
            "classes": str(args.classes) if args.classes else None,
            "mquake": str(args.mquake) if args.mquake else None,
        },
        "counts": counts,
        "groups": {name: n for name, n in taken.items() if n},
        "skipped": skipped,
        "questions": built,
        "provenance": record,
    }
    (out / "conditions.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({**counts, "groups": result["groups"]}))


def inventory_relations(mquake: Path) -> list[str]:
    """MQuAKE's relation inventory: the relations of its cases' requested rewrites, the
    relation templates GWalk's bank holds (process_kg)."""
    cases = json.loads(mquake.read_text(encoding="utf-8"))
    found = {e["relation_id"] for case in cases for e in case["requested_rewrite"]}
    return sorted(found, key=_number)


def selected(
    questions: list[dict], require: list[str], min_hops: int, inventory: set[str] | None
) -> list[dict]:
    """The questions with every required condition and at least min_hops hops (and with
    a relation outside the inventory, when one is given), in order, each keeping only the
    required conditions in its own order."""
    kept = []
    for q in questions:
        names = {c["name"] for c in q["conditions"]}
        if not set(require) <= names or len(q["chain"]) < min_hops:
            continue
        if inventory is not None and {p for _, p, _ in q["chain"]} <= inventory:
            continue
        kept.append({**q, "conditions": [c for c in q["conditions"] if c["name"] in require]})
    return kept


def select_command(args: argparse.Namespace) -> None:
    from xgraph.kgversions import read_kg_versions, version_files, write_kg_versions
    from xgraph.provenance import provenance

    require = [name.strip() for name in args.require.split(",") if name.strip()]
    if not require or len(set(require)) != len(require):
        raise SystemExit(f"--require {args.require}: name each condition once")
    source = json.loads((args.source / "conditions.json").read_text(encoding="utf-8"))
    inventory = inventory_relations(args.outside_inventory) if args.outside_inventory else None
    kept = selected(
        source["questions"], require, args.min_hops, set(inventory) if inventory else None
    )
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    names = [c["version"] for q in kept for c in q["conditions"] if c["version"]]
    edits = read_kg_versions(args.source / "kg-versions")
    versions = {
        name: np.array([[_number(x) for x in edit] for edit in edits[name]], dtype=np.int32)
        .reshape(-1, 3)
        for name in names
    }  # fmt: skip
    inputs = [args.source / "conditions.json", *version_files(args.source / "kg-versions")]
    inputs += [args.source / "labels.jsonl"] + ([args.outside_inventory] if inventory else [])
    record = provenance(inputs)
    write_kg_versions(out / "kg-versions", versions, provenance=record)
    asked = {q["id"] for q in kept}
    pairs = {f"{q['id']}@{c['name']}" for q in kept for c in q["conditions"]}
    for path in sorted(args.source.glob("requests-*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        _write_jsonl(out / path.name, [r for r in rows if r["id"] in pairs | asked])
    wanted = {g for q in kept for c in q["conditions"] for g in c["gold"]}
    wanted |= {g for q in kept for g in q["gold"]}
    _write_jsonl(out / "labels.jsonl", _label_rows([args.source / "labels.jsonl"], wanted))
    by_condition = {
        name: sum(c["name"] == name for q in kept for c in q["conditions"]) for name in require
    }
    counts = {
        "questions": len(kept),
        "conditions": sum(len(q["conditions"]) for q in kept),
        "versions": len(versions),
        "relation_sequences": len({q["relation_sequence"] for q in kept}),
        "by_condition": by_condition,
    }
    if inventory is not None:
        counts["edited_outside_inventory"] = sum(
            any(c["kind"] == "on-chain" and c["pair"][1] not in inventory for c in q["conditions"])
            for q in kept
        )
    result = {
        "artifact_kind": "xgraph.interventions",
        "version": 1,
        "settings": source["settings"],
        "selection": {
            "from": str(args.source),
            "require": require,
            "min_hops": args.min_hops,
            "outside_inventory": str(args.outside_inventory) if inventory else None,
            "inventory": inventory,
            "source_questions": len(source["questions"]),
        },
        "counts": counts,
        "groups": {
            name: n
            for name in sorted({group_of(q.get("complexity", "")) for q in kept})
            if (n := sum(group_of(q.get("complexity", "")) == name for q in kept))
        },
        "questions": kept,
        "provenance": record,
    }
    (out / "conditions.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(counts))


def _share(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def score_arm(questions: list[dict], texts: dict[str, str], labels: dict[str, set[str]]) -> dict:
    """One arm's scores (module docstring), and each question's per-condition outcomes."""
    from xgraph.prompts import final_answer
    from xgraph.score import names_gold

    outcomes: dict[str, list[tuple[dict, bool, bool]]] = {}
    missing = 0
    for q in questions:
        rows = []
        for c in q["conditions"]:
            text = texts.get(f"{q['id']}@{c['name']}", texts.get(q["id"]))
            missing += text is None
            answer = final_answer(text or "")
            right = names_gold(answer, c["gold"], labels)
            old = c["kind"] == "on-chain" and names_gold(answer, q["gold"], labels)
            rows.append((c, right, old))
        outcomes[q["id"]] = rows
    every = [row for rows in outcomes.values() for row in rows]
    on = [row for row in every if row[0]["kind"] == "on-chain"]

    def on_chain(rows: list) -> dict:
        return {
            "conditions": len(rows),
            "sensitivity": _share([r for _, r, _ in rows]),
            "old_answer": _share([o for _, _, o in rows]),
        }

    def split(key) -> dict:
        found: dict[str, list] = {}
        for row in on:
            found.setdefault(str(key(row[0])), []).append(row)
        return {name: on_chain(rows) for name, rows in sorted(found.items())}

    originally = {
        qid
        for qid, rows in outcomes.items()
        if any(c["kind"] == "original" and r for c, r, _ in rows)
    }
    by_condition: dict[str, list[bool]] = {}
    for c, right, _ in every:
        by_condition.setdefault(c["name"], []).append(right)
    groups: dict[str, list[str]] = {}
    for q in questions:
        groups.setdefault(group_of(q.get("complexity", "")), []).append(q["id"])

    def joint(ids: list[str]) -> float | None:
        return _share([all(r for _, r, _ in outcomes[qid]) for qid in ids])

    return {
        "questions": len(questions),
        "conditions": len(every),
        "missing": missing,
        "accuracy": _share([r for _, r, _ in every]),
        "original": _share([r for c, r, _ in every if c["kind"] == "original"]),
        "sensitivity": _share([r for _, r, _ in on]),
        "old_answer": _share([o for _, _, o in on]),
        "sensitivity_given_original": _share(
            [r for qid in originally for c, r, _ in outcomes[qid] if c["kind"] == "on-chain"]
        ),
        "invariance": _share([r for c, r, _ in every if c["kind"] == "off-chain"]),
        "joint_version_fidelity": joint(list(outcomes)),
        "by_stratum": split(lambda c: c["stratum"]),
        "by_position": split(lambda c: c["position"]),
        "by_hop": split(lambda c: c["hop"]),
        "by_condition": {
            name: {"conditions": len(flags), "accuracy": _share(flags)}
            for name, flags in sorted(by_condition.items())
        },
        "by_group": {
            name: {"questions": len(ids), "joint_version_fidelity": joint(ids)}
            for name, ids in sorted(groups.items())
        },
    }, outcomes


def compare_arms(
    questions: list[dict],
    a: dict[str, list[tuple[dict, bool, bool]]],
    b: dict[str, list[tuple[dict, bool, bool]]],
    resamples: int,
) -> dict:
    """The reference arm a minus arm b, paired over questions, resampling whole relation
    sequences: joint version fidelity and the original condition's accuracy as means,
    sensitivity and invariance (overall, by stratum, by position) as pooled shares."""
    from xgraph.score import paired_bootstrap, paired_ratio_bootstrap

    ids = [q["id"] for q in questions]
    clusters = {q["id"]: q["relation_sequence"] for q in questions}

    def mean(value) -> dict:
        first = [float(value(a[qid])) for qid in ids]
        second = [float(value(b[qid])) for qid in ids]
        found = paired_bootstrap(
            first, second, resamples=resamples, clusters=[clusters[q] for q in ids]
        )
        return {"a": sum(first) / len(ids), "b": sum(second) / len(ids), **found}

    def pooled(chosen) -> dict | None:
        kept = [qid for qid in ids if any(chosen(c) for c, _, _ in a[qid])]
        if not kept:
            return None

        def counts(rows) -> tuple[float, float]:
            picked = [r for c, r, _ in rows if chosen(c)]
            return float(sum(picked)), float(len(picked))

        found = paired_ratio_bootstrap(
            [counts(a[qid]) for qid in kept],
            [counts(b[qid]) for qid in kept],
            resamples=resamples,
            clusters=[clusters[qid] for qid in kept],
        )
        return {
            "questions": len(kept),
            "a": found["treated"],
            "b": found["control"],
            "difference": found["difference"],
            "low": found["low"],
            "high": found["high"],
        }

    def on(key, value):
        return lambda c: c["kind"] == "on-chain" and c[key] == value

    strata = sorted({c["stratum"] for q in questions for c in q["conditions"]
                     if c["kind"] == "on-chain"})  # fmt: skip
    positions = sorted({c["position"] for q in questions for c in q["conditions"]
                        if c["kind"] == "on-chain"})  # fmt: skip
    return {
        "questions": len(ids),
        "clusters": len(set(clusters.values())),
        "joint_version_fidelity": mean(lambda rows: all(r for _, r, _ in rows)),
        "original": mean(lambda rows: any(c["kind"] == "original" and r for c, r, _ in rows)),
        "sensitivity": pooled(lambda c: c["kind"] == "on-chain"),
        "invariance": pooled(lambda c: c["kind"] == "off-chain"),
        "by_stratum": {s: pooled(on("stratum", s)) for s in strata},
        "by_position": {p: pooled(on("position", p)) for p in positions},
    }


def score_command(args: argparse.Namespace) -> None:
    from xgraph.provenance import provenance
    from xgraph.score import load_labels, record_outputs

    conditions = json.loads(args.conditions.read_text(encoding="utf-8"))
    questions = conditions["questions"]
    wanted = {g for q in questions for c in q["conditions"] for g in c["gold"]}
    wanted |= {g for q in questions for g in q["gold"]}
    labels = load_labels(args.labels, wanted)
    arms: dict[str, Path] = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        if not name or not path or name in arms:
            raise SystemExit(f"--arm takes a new <name>=<records.jsonl>, not {spec!r}")
        arms[name] = Path(path)
    scores, outcomes = {}, {}
    for name, path in arms.items():
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        texts = record_outputs((r for r in records if r), (r["id"] for r in records if r))
        scores[name], outcomes[name] = score_arm(questions, texts, labels)
    reference, *others = arms
    result = {
        "artifact_kind": "xgraph.interventions_scores",
        "version": 1,
        "arms": scores,
        "reference": reference,
        "compare": {
            name: compare_arms(questions, outcomes[reference], outcomes[name], args.resamples)
            for name in others
        },
        "provenance": provenance([args.conditions, *args.labels, *arms.values()]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    summary = ("original", "sensitivity", "invariance", "joint_version_fidelity")
    print(json.dumps({name: {k: s[k] for k in summary} for name, s in scores.items()}))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m xgraph.interventions", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    made = commands.add_parser("build", help="conditions, KG versions, labels, and requests")
    made.add_argument("--questions", type=Path, required=True, help="questions (WWQ format)")
    made.add_argument("--chains", type=Path, required=True, help="question id to gold chain")
    made.add_argument("--kg", type=Path, required=True, help="KG directory")
    made.add_argument("--db", type=Path, required=True, help="its KG store (xgraph.toolkg)")
    made.add_argument(
        "--sitelinks",
        type=Path,
        help="sitelink counts (a sitelink_counts.npy) instead of the KG directory's own",
    )
    made.add_argument(
        "--classes",
        type=Path,
        help="a KG directory whose instance of and subclass of facts give the classes, "
        "where the KG's own are unreliable",
    )
    made.add_argument(
        "--mquake", type=Path, help="MQuAKE JSON: each case's edits hold in all its conditions"
    )
    made.add_argument(
        "--labels", type=Path, action="append", default=[], help="more gold labels (JSONL)"
    )
    made.add_argument("--size", type=int, required=True, help="questions to sample")
    made.add_argument("--seed", type=int, default=0)
    made.add_argument("--pool", type=int, default=POOL, help="popular candidates drawn from")
    made.add_argument("--out", type=Path, required=True, help="output directory")
    chosen = commands.add_parser(
        "select", help="the questions of a build with every listed condition, as a build"
    )
    chosen.add_argument(
        "--from", dest="source", type=Path, required=True, help="a build's output directory"
    )
    chosen.add_argument(
        "--require",
        required=True,
        help="condition names, comma separated: a question is kept when it has all of them, "
        "and keeps only them",
    )
    chosen.add_argument("--min-hops", type=int, default=1, help="the fewest hops of a chain")
    chosen.add_argument(
        "--outside-inventory",
        type=Path,
        help="MQuAKE JSON: keep only chains with a relation outside its relation inventory "
        "(its requested rewrites' relations)",
    )
    chosen.add_argument("--out", type=Path, required=True, help="output directory")
    scored = commands.add_parser("score", help="arms against the conditions' gold")
    scored.add_argument("--conditions", type=Path, required=True, help="conditions.json")
    scored.add_argument("--labels", type=Path, action="append", required=True, help="JSONL")
    scored.add_argument(
        "--arm", action="append", required=True, help="<name>=<records.jsonl>; the first is "
        "compared with every other"
    )  # fmt: skip
    scored.add_argument("--resamples", type=int, default=10_000)
    scored.add_argument("--out", type=Path, required=True, help="scores (JSON)")
    args = parser.parse_args(argv)
    if args.command == "build":
        if args.size < 1 or args.pool < 1:
            parser.error("--size and --pool must be positive")
        build_command(args)
    elif args.command == "select":
        if args.min_hops < 1:
            parser.error("--min-hops must be positive")
        select_command(args)
    else:
        score_command(args)


if __name__ == "__main__":
    main()
