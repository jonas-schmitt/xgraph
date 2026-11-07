"""Path-writing examples: sampled paths of a KG directory as path-mode training data.

Path mode asks a model for the relations from a question's entity to its answer
(`⟦ The Hobbit · author · place of birth⟧`, then `Answer: ⟦ Bloemfontein⟧`). To train
that without benchmark data, paths are sampled from the gate KG and each becomes a
chat example in the format xgraph.train reads: the path-span prompt without worked
examples (PATH_INSTRUCTIONS) as the system message, a question about the path as the
user message, and as the reply the path span with the relations' labels and the
endpoint's label as the answer span. Each record also carries the sampled path's QIDs
and PIDs (`walk`), its start as `anchor`, and the hop count.

**Sampled paths.** A path starts at an entity drawn uniformly, with the seed, from the
eligible starts, and takes 1 to 4 hops (`--hops`, the paths spread evenly over them).
At each hop it takes a relation drawn uniformly from those of its current entity that
keep it valid, and an object of that relation drawn uniformly. It is kept when:

- every entity on it is an article: its English Wikipedia sitelink is in the main
  namespace, not a category, template, or other Wikipedia namespace (NAMESPACES), which
  keeps out Wikimedia's own meta relations (topic's main category, category's main
  topic, ...), whose objects are such pages;
- no entity on it but the endpoint is a class, an entity with a superclass (the subject
  of a subclass of, P279, triple, Wikidata's class hierarchy): a path may end at a class
  (`⟦ Ada Lovelace · occupation⟧`, then `⟦ mathematician⟧`) but neither starts at nor
  passes through one, so that it stays among specific entities instead of climbing the
  ontology (human · subclass of · different from · ...);
- every entity on it has a writable canonical label (`writable`: not blank, no
  surrounding whitespace, no span syntax, no text that looks like a special token
  such as `<|endoftext|>`), no entity appears twice, and no entity is held out;
- every relation on it has a writable canonical label that names no other relation
  (after `xgraph.wikidata.relation_names`, as the relation index names them);
- its start's canonical label names no other entity, as label or alias, so that the
  path span's first frontier is the start alone, anchored or not, committed or not;
  and the start has at least two relations, so that the first hop is a choice;
- the endpoint is unique: the relations, followed from the start through every object
  they reach (the path's frontier, as path mode walks it), reach exactly one entity,
  the path's last, and no frontier on the way holds more than the engine's MAX_REACHED
  entities; the endpoint has at most MAX_REACHED labels, so its canonical label is among
  the label rows the answer may name; a path whose relations reach several endpoints
  is skipped, not answered with one of them. With `--max-endpoints k`, the last
  frontier may hold up to k entities, the path's last among them, and the reply
  answers with that one (with several, its label can fall outside the first MAX_REACHED
  label rows the answer may name, a reply xgraph.train counts as off the constraint);
- it contains no held-out relation composition, and is no repeat of a path already kept.

These two structural rules (articles, and no class but at the end) are the only ones on
what a path may say: no relation is left out by name.

**Relation-balanced sampling** (`--balance`, `balanced_paths`). Drawn uniformly, paths
follow the relations their entities have most (instance of, located in), and a relation
whose triples mostly lead to held-out entities hardly appears. Balanced, every path is
drawn around a relation of the relation inventory: the writable relations with at least
`--min-triples` triples in the KG, so that the inventory and its weights come from the
KG alone, never from a benchmark. A path around a relation takes one of its triples
(drawn uniformly from those whose subject a path may go on from and whose object it may
visit) at a position drawn uniformly, walks backwards from the triple's subject through
uniformly drawn incoming triples to a start, and forwards from its object as above; it
is kept under the same rules. Sampling first goes in rounds over the relations with
fewer than `--min-occurrences` occurrences, one path per relation and round, then draws
relations by their triple count to the power `--balance-power`. No relation sequence gets
more than `--max-composition` paths, and a relation for which ATTEMPTS draws in a row find
no new path is exhausted: the KG offers no more under the rules and the holdout.

The frontier is followed over every triple of the relation, also to objects the KG
index leaves out for want of a writable label, so it can only be wider than path
mode's: an endpoint unique here is unique there. Labels are checked by rules on their
text, not by the served tokenizer, so a reply can still leave the constraint where a
tokenizer merges label text with a delimiter; xgraph.train counts such replies
(`off_policy_examples`).

**Questions.** `templated_question` names the relations from the start outwards (`What
is the place of birth of the author of The Hobbit?`; a relation label ending in a
preposition is asked as what its subject is: `What is Steve Jobs educated at?`), each
relation by its label or, at the rate `--alias-rate`, by one of its aliases, so that
the model learns to map words to relation labels. The reply always writes the labels.
`examples` takes another question writer (`ask`), the hook for model-written
paraphrases; each record names its writer in `question_source`. xgraph.naturalize rewrites
the questions of a written file with a model served by vLLM, and checks the rewrites.

**Holdout.** `holdout` lists the entities of benchmark questions, so that no sampled
path visits one: MQuAKE cases (`orig.triples`, with `new_triples` and `edit_triples`
for whole chains) and MINTQA questions (their `triple` labels, mapped to QIDs through
MINTQA's KG directory, `python -m xgraph.mintqa_bench kg`). By default the subjects the
questions start from; with `--chains`, every entity of their chains. With
`--patterns` it also lists the chains' relation compositions (two relations or more),
and no sampled path then contains one as consecutive relations; a single relation
listed as a pattern would leave that relation out altogether. `examples
--no-holdout-compositions` holds out a holdout file's entities alone, its compositions
not; the manifest's settings record which (`holdout_compositions`).

**Endpoint frontiers.** With `--index` and `--tokenizer` (the KG index path mode runs
with and its served tokenizer), each example records `endpoint_frontier`: how many
entities its path resolves to as path mode with committed paths resolves it from an
unanchored request, by matched execution (`endpoint_frontiers`, which runs
xgraph.programs.MatchedExecutor on the reply's path span), so that xgraph.train
`--answer-loss singleton` can leave out the answer of a path that reaches several. The
walk's own frontier, over every triple of the KG directory, is never narrower; at the
default `--max-endpoints 1` both hold the endpoint alone, unless the index lacks it.
`frontiers` adds the field to examples already written.

    python -m xgraph.pathdata holdout --mquake data/mquake/MQuAKE-Remastered-CF-3k.json \\
        --mintqa data/external/mintqa-full/questions/ti.jsonl \\
        --mintqa-kg data/external/mintqa-full/kg --out target/pathdata/holdout.json
    python -m xgraph.pathdata examples --kg data/wikidata/gate-kg --n 30000 --seed 0 \\
        --holdout target/pathdata/holdout.json --out target/pathdata/train.jsonl
    python -m xgraph.pathdata examples --kg data/wikidata/gate-kg --n 30000 --seed 0 \\
        --holdout target/pathdata/holdout.json --balance \\
        --out target/pathdata/train-balanced.jsonl
    python -m xgraph.pathdata examples --kg data/wikidata/gate-kg --n 30000 --seed 0 \\
        --holdout target/pathdata/holdout.json --no-holdout-compositions --balance \\
        --index data/wikidata/gate-index-qwen2.5 --tokenizer <model>/tokenizer.json \\
        --out target/pathdata/train-balanced-entities.jsonl
    python -m xgraph.pathdata frontiers --examples target/pathdata/train-balanced.jsonl \\
        --index data/wikidata/gate-index-qwen2.5 --tokenizer <model>/tokenizer.json \\
        --out target/pathdata/train-balanced-sized.jsonl

`examples` writes the JSONL and `<out>.manifest.json`: the counts per hop count, the
paths short of the request, each relation's occurrences, the relation sequence with the
most paths, the settings, and provenance; with an index, the examples per endpoint
frontier size (`endpoint_frontiers`, as `frontiers`' manifest also has them); balanced,
also the coverage: the inventory with
each relation's triple count, the fewest occurrences of an inventory relation, the
relations below `--min-occurrences` with their counts, and the exhausted ones. Memory
stays bounded: the label files are streamed, and the triples are held once as sorted
(subject, relation) keys and objects, balanced also sorted by object for the backward steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from array import array
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from xgraph.oracle import MAX_REACHED
from xgraph.syntax import SpanSyntax

NAMESPACES = frozenset(
    {
        "User", "Wikipedia", "File", "MediaWiki", "Template", "Help", "Category", "Portal",
        "Draft", "TimedText", "Module", "Special", "Media", "Book", "Education Program",
        "Gadget", "Gadget definition", "Topic", "Talk", "WP", "Image",
    }
)  # fmt: skip
"""English Wikipedia's namespaces other than articles; "<namespace> talk" too."""

SUBCLASS_OF = 279
"""Wikidata's subclass of: its subjects are classes, where a path may only end."""
MIN_START_RELATIONS = 2
"""A start offers at least this many relations, so that its first hop is a choice."""
ATTEMPTS = 200
"""Sampling attempts per requested path before a hop count is reported short."""
ALIAS_RATE = 0.3
FRAMES = ("What is {}?", "Which entity is {}?", "Name {}.")
PREPOSITIONS = frozenset(
    {"of", "in", "at", "by", "for", "from", "to", "with", "on", "as", "into", "about", "after"}
)
"""A relation label ending in one of these reads as a verb phrase (educated at)."""

_KEY_BITS = 20  # (subject << 20) | relation; property numbers stay below 2**20
_RELATION_MASK = (1 << _KEY_BITS) - 1
_SPECIAL = re.compile(r"<[^<>\s]+>")
_SYNTAX = SpanSyntax()


def writable(label: str, syntax: SpanSyntax = _SYNTAX) -> bool:
    """Whether a label can be written in a span, judged on its text and conservatively:
    not blank, no surrounding whitespace or line break, no span syntax in its written
    form, and nothing that looks like a special token's text."""
    return (
        bool(label.strip())
        and label == label.strip()
        and "\n" not in label
        and not syntax.excludes(label)
        and not _SPECIAL.search(label)
    )


def _article(title: str) -> bool:
    prefix, colon, _ = title.partition(":")
    if not colon:
        return True
    return prefix not in NAMESPACES and prefix.removesuffix(" talk") not in NAMESPACES


def _hash(label: str) -> int:
    return int.from_bytes(hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest())


def _jsonl(path: Path) -> Iterator[list]:
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            yield json.loads(line)


def _number(item: str) -> int:
    return int(item[1:])


@dataclass(frozen=True)
class Holdout:
    """Entities no sampled path visits, and relation compositions none contains."""

    entities: frozenset[int] = frozenset()
    patterns: frozenset[tuple[int, ...]] = frozenset()

    @classmethod
    def read(cls, path: str | Path) -> Holdout:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            frozenset(_number(e) for e in data.get("entities", [])),
            frozenset(tuple(_number(p) for p in pattern) for pattern in data.get("patterns", [])),
        )


@dataclass(frozen=True)
class SampledPath:
    entities: tuple[int, ...]
    """Item numbers from the start to the endpoint, one more than the relations."""
    relations: tuple[int, ...]


@dataclass
class Graph:
    """What sampling reads of a KG directory: its triples by (subject, relation), and per
    entity whether a path may visit it and whether one may start there."""

    kg: Path
    keys: np.ndarray
    """int64, sorted: (subject << 20) | relation of each triple with a writable relation."""
    objects: np.ndarray
    """int32: each key's object."""
    items: np.ndarray
    """int64, sorted: the item numbers with a canonical label."""
    node: np.ndarray
    """bool per item: a sampled path may visit it."""
    through: np.ndarray
    """bool per item: a sampled path may go on from it (a node and no class)."""
    start: np.ndarray
    """bool per item: a sampled path may start there."""
    relation_labels: dict[int, str]
    """Each writable relation's canonical label."""
    relation_aliases: dict[int, list[str]]
    patterns: frozenset[tuple[int, ...]] = field(default_factory=frozenset)
    max_endpoints: int = 1
    """The most entities a path's relations may reach from its start, its endpoint one."""

    def relations_of(self, subject: int) -> np.ndarray:
        """The distinct writable relations of a subject, ascending."""
        lo, hi = np.searchsorted(
            self.keys, [subject << _KEY_BITS, (subject + 1) << _KEY_BITS], side="left"
        )
        return np.unique(self.keys[lo:hi] & ((1 << _KEY_BITS) - 1))

    def objects_of(self, subjects: np.ndarray, relation: int) -> np.ndarray:
        """Every object the relation reaches from these subjects, distinct and ascending."""
        wanted = (np.asarray(subjects, dtype=np.int64) << _KEY_BITS) | relation
        lo = np.searchsorted(self.keys, wanted, side="left")
        hi = np.searchsorted(self.keys, wanted, side="right")
        if not (hi > lo).any():
            return np.zeros(0, dtype=np.int64)
        return np.unique(np.concatenate([self.objects[a:b] for a, b in zip(lo, hi, strict=True)]))

    def endpoints(self, start: int, relations: Sequence[int]) -> np.ndarray:
        """The frontier after following the relations from the start through every object."""
        frontier = np.array([start], dtype=np.int64)
        for relation in relations:
            frontier = self.objects_of(frontier, relation).astype(np.int64)
        return frontier

    def _flag(self, flags: np.ndarray, numbers: np.ndarray) -> np.ndarray:
        at = np.searchsorted(self.items, numbers)
        inside = at < len(self.items)
        found = np.zeros(len(numbers), dtype=bool)
        found[inside] = self.items[at[inside]] == numbers[inside]
        result = np.zeros(len(numbers), dtype=bool)
        result[found] = flags[at[found]]
        return result

    def may_visit(self, numbers: np.ndarray) -> np.ndarray:
        return self._flag(self.node, np.asarray(numbers, dtype=np.int64))

    def may_pass(self, numbers: np.ndarray) -> np.ndarray:
        return self._flag(self.through, np.asarray(numbers, dtype=np.int64))

    def may_start(self, numbers: np.ndarray) -> np.ndarray:
        return self._flag(self.start, np.asarray(numbers, dtype=np.int64))

    @cached_property
    def triple_counts(self) -> dict[int, int]:
        """Each writable relation's number of triples in the KG."""
        relations, counts = np.unique(self.keys & _RELATION_MASK, return_counts=True)
        return dict(zip(relations.tolist(), counts.tolist(), strict=True))

    @cached_property
    def _focus(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The triples a path may hold, whose subject a path may go on from and whose
        object it may visit, grouped by relation: (relations, offsets, triple positions)."""
        positions = np.flatnonzero(
            self.may_pass(self.keys >> _KEY_BITS) & self.may_visit(self.objects)
        )
        relations = self.keys[positions] & _RELATION_MASK
        order = np.argsort(relations, kind="stable")
        positions, relations = positions[order], relations[order]
        distinct, offsets = np.unique(relations, return_index=True)
        return distinct, np.append(offsets, len(positions)), positions

    def focus_triple(self, relation: int, rng: np.random.Generator) -> tuple[int, int] | None:
        """A (subject, object) of the relation drawn uniformly from the triples a path may
        hold, or None when it has none."""
        relations, offsets, positions = self._focus
        at = int(np.searchsorted(relations, relation))
        if at == len(relations) or relations[at] != relation:
            return None
        position = positions[offsets[at] + rng.integers(offsets[at + 1] - offsets[at])]
        return int(self.keys[position] >> _KEY_BITS), int(self.objects[position])

    @cached_property
    def _incoming(self) -> tuple[np.ndarray, np.ndarray]:
        """(object << 20) | relation of each triple, sorted, and each one's subject."""
        keys = (self.objects.astype(np.int64) << _KEY_BITS) | (self.keys & _RELATION_MASK)
        order = np.argsort(keys, kind="stable")
        return keys[order], self.keys[order] >> _KEY_BITS

    def predecessor(
        self, entity: int, avoid: Sequence[int], rng: np.random.Generator, tries: int = 8
    ) -> tuple[int, int] | None:
        """A (subject, relation) of a triple into the entity, drawn uniformly from its
        incoming triples, whose subject a path may go on from and is not to be avoided;
        None when tries draws find none."""
        keys, subjects = self._incoming
        lo, hi = np.searchsorted(keys, [entity << _KEY_BITS, (entity + 1) << _KEY_BITS])
        if hi == lo:
            return None
        for _ in range(tries):
            at = int(lo + rng.integers(hi - lo))
            subject = int(subjects[at])
            if subject not in avoid and self.may_pass(np.array([subject]))[0]:
                return subject, int(keys[at] & _RELATION_MASK)
        return None

    def labels(self, wanted: Iterable[int]) -> dict[int, str]:
        """The canonical labels of these items, read from the KG directory."""
        wanted = set(wanted)
        found: dict[int, str] = {}
        for item, label, kind in _jsonl(self.kg / "entity_labels.jsonl"):
            if kind == "label" and (number := _number(item)) in wanted:
                found.setdefault(number, label)
        return found


def _relations(kg: Path) -> tuple[dict[int, str], dict[int, list[str]]]:
    """The writable relations' canonical labels and their aliases, as the relation index
    names them: a canonical label shared with another relation's name is not writable."""
    from xgraph.wikidata import relation_names

    rows = list(_jsonl(kg / "relation_labels.jsonl"))
    names = relation_names(rows)
    owners: dict[str, set[str]] = {}
    for prop, labels in names.items():
        for label in labels:
            owners.setdefault(label, set()).add(prop)
    canonical: dict[int, str] = {}
    aliases: dict[int, list[str]] = {}
    for prop, label, kind in rows:
        if kind != "label" or _number(prop) in canonical:
            continue
        if writable(label) and owners.get(label) == {prop}:
            canonical[_number(prop)] = label
            aliases[_number(prop)] = [a for a in names[prop] if a != label and writable(a)]
    return canonical, aliases


def load_graph(
    kg: str | Path,
    holdout: Holdout = Holdout(),  # noqa: B008
    *,
    max_endpoints: int = 1,
) -> Graph:
    """Read a KG directory (python -m xgraph.wikidata stream) for sampling paths, whose
    relations may reach up to max_endpoints entities from their start."""
    if not 1 <= max_endpoints <= MAX_REACHED:
        raise ValueError(f"max_endpoints must be in [1, {MAX_REACHED}], got {max_endpoints}")
    kg = Path(kg)
    for name in ("entity_labels.jsonl", "relation_labels.jsonl", "sitelinks.jsonl", "triples.npy"):
        if not (kg / name).is_file():
            raise FileNotFoundError(f"the KG directory {kg} has no {name}")

    # Labels: per label row its item and text hash; per item its canonical label.
    row_items, row_hashes = array("q"), array("Q")
    canon_items, canon_hashes, canon_ok = array("q"), array("Q"), array("b")
    for item, label, kind in _jsonl(kg / "entity_labels.jsonl"):
        number, digest = _number(item), _hash(label)
        row_items.append(number)
        row_hashes.append(digest)
        if kind == "label":
            canon_items.append(number)
            canon_hashes.append(digest)
            canon_ok.append(writable(label))
    by_hash = np.frombuffer(row_hashes, np.uint64).view(np.int64)
    by_item = np.frombuffer(row_items, np.int64)
    order = np.lexsort((by_item, by_hash))
    by_hash, by_item = by_hash[order], by_item[order]
    del order, row_items, row_hashes
    distinct = np.ones(len(by_hash), dtype=bool)  # one row per (label, item)
    distinct[1:] = (by_hash[1:] != by_hash[:-1]) | (by_item[1:] != by_item[:-1])
    by_hash, by_item = by_hash[distinct], by_item[distinct]
    label_hashes, entities_per_label = np.unique(by_hash, return_counts=True)
    labelled, labels_per_item = np.unique(by_item, return_counts=True)
    del by_hash, by_item, distinct
    items, first = np.unique(np.frombuffer(canon_items, np.int64), return_index=True)
    hashes = np.frombuffer(canon_hashes, np.uint64).view(np.int64)[first]
    node = np.frombuffer(canon_ok, np.int8)[first].astype(bool)
    del canon_items, canon_hashes, canon_ok
    shared = entities_per_label[np.searchsorted(label_hashes, hashes)] > 1
    node &= labels_per_item[np.searchsorted(labelled, items)] <= MAX_REACHED

    others = array("q")
    for item, title in _jsonl(kg / "sitelinks.jsonl"):
        if not _article(title):
            others.append(_number(item))
    for excluded in (np.frombuffer(others, np.int64), np.array(sorted(holdout.entities))):
        node &= ~np.isin(items, excluded)

    relation_labels, relation_aliases = _relations(kg)
    triples = np.load(kg / "triples.npy", mmap_mode="r")
    if len(triples) and int(triples[:, 1].max()) >= 1 << _KEY_BITS:
        raise ValueError(f"a property number in {kg} needs more than {_KEY_BITS} bits")
    relation_column = np.asarray(triples[:, 1])
    classes = np.unique(np.asarray(triples[:, 0])[relation_column == SUBCLASS_OF])
    through = node & ~np.isin(items, classes)
    keep = np.isin(relation_column, np.array(sorted(relation_labels), dtype=np.int64))
    del relation_column, classes
    keys = (triples[:, 0][keep].astype(np.int64) << _KEY_BITS) | triples[:, 1][keep]
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    objects = np.asarray(triples[:, 2])[keep][order]
    del order, keep

    # Starts: unshared labels and at least MIN_START_RELATIONS relations.
    pairs = keys[np.concatenate([[True], keys[1:] != keys[:-1]])] if len(keys) else keys
    subjects, relation_counts = np.unique(pairs >> _KEY_BITS, return_counts=True)
    at = np.searchsorted(subjects, items)
    inside = at < len(subjects)
    has = np.zeros(len(items), dtype=bool)
    has[inside] = (subjects[at[inside]] == items[inside]) & (
        relation_counts[at[inside]] >= MIN_START_RELATIONS
    )
    start = through & ~shared & has
    return Graph(
        kg,
        keys,
        objects,
        items,
        node,
        through,
        start,
        relation_labels,
        relation_aliases,
        holdout.patterns,
        max_endpoints,
    )


def _held(graph: Graph, chain: Sequence[int]) -> bool:
    """Whether a relation chain ends in a held-out relation composition."""
    return any(tuple(chain[i:]) in graph.patterns for i in range(len(chain)))


def _prefix(
    graph: Graph, entities: list[int], relations: list[int], hops: int
) -> np.ndarray | None:
    """The frontier after a path's first relations, or None when they break a sampling
    rule: a start where no path may start, an entity twice or one a path may not visit
    or go on from, a held-out composition, a frontier over MAX_REACHED, or (when they are
    the whole path) a frontier without the last entity or over the graph's max_endpoints."""
    complete = len(relations) == hops
    if len(set(entities)) != len(entities) or not graph.may_start(np.array(entities[:1]))[0]:
        return None
    inner = np.array(entities[1:-1] if complete else entities[1:], dtype=np.int64)
    if not graph.may_pass(inner).all():
        return None
    if complete and not graph.may_visit(np.array(entities[-1:]))[0]:
        return None
    frontier = np.array(entities[:1], dtype=np.int64)
    for hop, relation in enumerate(relations):
        if _held(graph, relations[: hop + 1]):
            return None
        frontier = graph.objects_of(frontier, relation).astype(np.int64)
        if len(frontier) > MAX_REACHED:
            return None
    if complete and (entities[-1] not in frontier or len(frontier) > graph.max_endpoints):
        return None
    return frontier


def _walk(
    graph: Graph,
    start: int,
    hops: int,
    rng: np.random.Generator,
    prefix: tuple[list[int], list[int]] | None = None,
) -> SampledPath | None:
    """A path of hops relations from the start, drawn hop by hop; with a prefix, its
    entities (from the start) and relations come first."""
    entities, relations = [start], []
    frontier = np.array([start], dtype=np.int64)
    if prefix is not None:
        entities, relations = list(prefix[0]), list(prefix[1])
        if (reached := _prefix(graph, entities, relations, hops)) is None:
            return None
        frontier = reached
    for hop in range(len(relations), hops):
        last = hop == hops - 1
        here = entities[-1]
        for relation in rng.permutation(graph.relations_of(here)).tolist():
            if _held(graph, (*relations, relation)):
                continue
            reached = graph.objects_of(frontier, relation)
            if len(reached) > (graph.max_endpoints if last else MAX_REACHED):
                continue
            options = graph.objects_of(np.array([here]), relation)
            allowed = graph.may_visit(options) if last else graph.may_pass(options)
            options = options[allowed & ~np.isin(options, entities)]
            if len(options):
                entities.append(int(options[rng.integers(len(options))]))
                relations.append(relation)
                frontier = reached.astype(np.int64)
                break
        else:
            return None
    return SampledPath(tuple(entities), tuple(relations))


def _focused_walk(
    graph: Graph, relation: int, hops: int, rng: np.random.Generator
) -> SampledPath | None:
    """A path of hops relations that holds the relation at a position drawn uniformly:
    one of its triples, the relations before it walked backwards from the triple's
    subject, the ones after it forwards from its object."""
    triple = graph.focus_triple(relation, rng)
    if triple is None:
        return None
    entities, relations = list(triple), [relation]
    for _ in range(int(rng.integers(hops))):
        step = graph.predecessor(entities[0], entities, rng)
        if step is None:
            return None
        entities.insert(0, step[0])
        relations.insert(0, step[1])
    return _walk(graph, entities[0], hops, rng, prefix=(entities, relations))


def sample_paths(
    graph: Graph, n: int, seed: int, hops: Sequence[int] = (1, 2, 3, 4)
) -> list[SampledPath]:
    """n distinct sampled paths, spread evenly over the hop counts (the first ones take the
    remainder), in an order shuffled with the seed. A hop count whose paths run out
    within ATTEMPTS tries per path yields fewer; the caller sees the shortfall."""
    if n < 0 or not hops or any(k < 1 for k in hops):
        raise ValueError(f"need n >= 0 and hop counts of at least 1, got n={n}, hops={hops}")
    rng = np.random.default_rng(seed)
    starts = graph.items[graph.start]
    if not len(starts):
        raise ValueError(f"the KG directory {graph.kg} has no entity a path may start at")
    kept: list[SampledPath] = []
    seen: set[SampledPath] = set()
    for position, hop_count in enumerate(hops):
        quota = n // len(hops) + (position < n % len(hops))
        made = attempts = 0
        while made < quota and attempts < quota * ATTEMPTS:
            attempts += 1
            path = _walk(graph, int(starts[rng.integers(len(starts))]), hop_count, rng)
            if path is not None and path not in seen:
                seen.add(path)
                kept.append(path)
                made += 1
    return [kept[i] for i in rng.permutation(len(kept))]


@dataclass(frozen=True)
class Balance:
    """Relation-balanced sampling: the relation inventory (every writable relation with at
    least min_triples triples in the KG), the occurrences each should reach, the most
    paths one relation sequence may have, and the exponent of the weights (triples **
    power) by which relations are drawn once the inventory is covered."""

    min_triples: int = 1000
    min_occurrences: int = 20
    max_composition: int = 100
    power: float = 0.5

    def __post_init__(self) -> None:
        if self.min_triples < 1 or self.min_occurrences < 0 or self.max_composition < 1:
            raise ValueError(
                f"need min_triples >= 1, min_occurrences >= 0, and max_composition >= 1, got {self}"
            )


@dataclass(frozen=True)
class Coverage:
    """What relation-balanced sampling aimed at: the inventory with each relation's triple
    count, and the relations whose paths ran out (ATTEMPTS draws in a row found no new
    path the rules allow), so that a relation short of the minimum is one of them or
    short for want of examples."""

    inventory: dict[int, int]
    exhausted: frozenset[int]


def balanced_paths(
    graph: Graph, n: int, seed: int, hops: Sequence[int], balance: Balance
) -> tuple[list[SampledPath], Coverage]:
    """n distinct sampled paths, each drawn around a relation of the inventory
    (`_focused_walk`): first in rounds over the relations below min_occurrences, one path
    per relation and round, until each reaches it or runs out; then around relations
    drawn by their weights. A path's hop count is drawn by the hop counts' remaining
    shares of n (spread evenly, as in sample_paths), and no relation sequence gets more
    than max_composition paths. Seeded; a shortfall shows in the hop counts."""
    if n < 0 or not hops or any(k < 1 for k in hops):
        raise ValueError(f"need n >= 0 and hop counts of at least 1, got n={n}, hops={hops}")
    rng = np.random.default_rng(seed)
    inventory = {r: c for r, c in sorted(graph.triple_counts.items()) if c >= balance.min_triples}
    quota = np.array([n // len(hops) + (i < n % len(hops)) for i in range(len(hops))])
    kept: list[SampledPath] = []
    seen: set[SampledPath] = set()
    compositions: Counter[tuple[int, ...]] = Counter()
    occurrences: Counter[int] = Counter()
    exhausted: set[int] = set()

    def draw(relation: int) -> None:
        """Keep one new path around the relation, or mark it exhausted."""
        for _ in range(ATTEMPTS):
            open_ = np.flatnonzero(quota > 0)
            at = int(rng.choice(open_, p=quota[open_] / quota[open_].sum()))
            path = _focused_walk(graph, relation, hops[at], rng)
            if path is None or path in seen:
                continue
            if compositions[path.relations] >= balance.max_composition:
                continue
            seen.add(path)
            kept.append(path)
            compositions[path.relations] += 1
            occurrences.update(path.relations)
            quota[at] -= 1
            return
        exhausted.add(relation)

    order = rng.permutation(list(inventory)).tolist()
    while quota.sum() > 0:
        below = [r for r in order if occurrences[r] < balance.min_occurrences]
        below = [r for r in below if r not in exhausted]
        if not below:
            break
        for relation in below:
            if quota.sum() == 0:
                break
            if occurrences[relation] < balance.min_occurrences:
                draw(relation)
    while quota.sum() > 0:
        live = [r for r in inventory if r not in exhausted]
        if not live:
            break
        weights = np.array([inventory[r] for r in live], dtype=np.float64) ** balance.power
        draw(live[int(rng.choice(len(live), p=weights / weights.sum()))])
    shuffled = [kept[i] for i in rng.permutation(len(kept))]
    return shuffled, Coverage(inventory, frozenset(exhausted))


def _relation_phrase(label: str, inner: str) -> str:
    if label.split()[-1] in PREPOSITIONS:
        return f"what {inner} is {label}"
    return f"the {label} of {inner}"


def templated_question(start: str, relations: Sequence[str], rng: np.random.Generator) -> str:
    """A question that names the relations (labels or aliases) from the start outwards."""
    *inner_relations, outer = relations
    inner = start
    for relation in inner_relations:
        inner = _relation_phrase(relation, inner)
    if outer.split()[-1] in PREPOSITIONS:
        return f"What is {inner} {outer}?"
    return FRAMES[rng.integers(len(FRAMES))].format(f"the {outer} of {inner}")


QuestionWriter = Callable[[str, Sequence[str], np.random.Generator], str]


def examples(
    graph: Graph,
    paths: Sequence[SampledPath],
    seed: int,
    *,
    ask: QuestionWriter = templated_question,
    alias_rate: float = ALIAS_RATE,
) -> list[dict]:
    """One chat example per sampled path, in xgraph.train's format."""
    from xgraph.prompts import wwq_messages
    from xgraph.sft import _span

    if not 0.0 <= alias_rate <= 1.0:
        raise ValueError(f"alias_rate must be in [0, 1], got {alias_rate}")
    rng = np.random.default_rng(seed)
    names = graph.labels({e for p in paths for e in (p.entities[0], p.entities[-1])})
    built = []
    for number, path in enumerate(paths):
        start, end = names[path.entities[0]], names[path.entities[-1]]
        labels = [graph.relation_labels[r] for r in path.relations]
        asked = []
        for relation, label in zip(path.relations, labels, strict=True):
            aliases = graph.relation_aliases[relation]
            use_alias = aliases and rng.random() < alias_rate
            asked.append(aliases[rng.integers(len(aliases))] if use_alias else label)
        question = ask(start, asked, rng)
        reply = f"{_span(start, *labels)}.\nAnswer: {_span(end)}"
        built.append(
            {
                "id": f"path-{number}",
                "hops": len(path.relations),
                "question": question,
                "question_source": getattr(ask, "__name__", "custom"),
                "walk": {
                    "entities": [f"Q{e}" for e in path.entities],
                    "relations": [f"P{r}" for r in path.relations],
                },
                "anchor": [f"Q{path.entities[0]}"],
                "messages": [
                    *wwq_messages(question, path_spans=True, shots=False),
                    {"role": "assistant", "content": reply},
                ],
            }
        )
    return built


def endpoint_frontiers(records: Iterable[dict], index, tokenizer) -> list[int]:
    """The size of each example's endpoint frontier as path mode with committed paths
    reaches it, from an unanchored request: its path span (start and relation labels, read
    from the reply) executed by matched execution (xgraph.programs.MatchedExecutor) over
    the KG index and served tokenizer; more than one where the answer span may name
    several entities, and the reply's endpoint is one of them."""
    from xgraph.naturalize import path_labels
    from xgraph.programs import MatchedExecutor

    executor = MatchedExecutor(index, tokenizer)
    return [len(executor.execute(*path_labels(record)).reached) for record in records]


def _frontier_counts(records: Iterable[dict]) -> dict[str, int]:
    counts = Counter(record["endpoint_frontier"] for record in records)
    return {str(size): counts[size] for size in sorted(counts)}


def holdout(
    mquake: Sequence[str | Path],
    mintqa: Sequence[str | Path],
    mintqa_kg: str | Path | None,
    *,
    chains: bool,
) -> dict:
    """The entities (subjects, or with chains every entity of each chain) and the relation
    compositions of two or more hops of benchmark questions, as QIDs and PIDs. MINTQA's
    labels are mapped through its KG directory; labels it does not hold are counted in
    `unmapped`."""
    entities: set[str] = set()
    patterns: set[tuple[str, ...]] = set()
    for path in mquake:
        for case in json.loads(Path(path).read_text(encoding="utf-8")):
            orig = case["orig"]
            chain = orig["triples"]
            if chains:
                for s, _, o in chain + orig["new_triples"] + orig["edit_triples"]:
                    entities.update((s, o))
            else:
                entities.add(chain[0][0])
            if len(chain) > 1:
                patterns.add(tuple(p for _, p, _ in chain))
    asked = [row["triple"] for path in mintqa for row in _jsonl_dicts(Path(path))]
    unmapped = {"entities": 0, "relations": 0}
    if asked:
        if mintqa_kg is None:
            raise ValueError("MINTQA questions need --mintqa-kg to map their labels to QIDs")
        mintqa_kg = Path(mintqa_kg)
        wanted: set[str] = set()
        for chain in asked:
            if chains:
                wanted.update(label for s, _, o in chain for label in (s, o))
            else:
                wanted.add(chain[0][0])
        named: dict[str, set[str]] = {}
        for item, label, _ in _jsonl(mintqa_kg / "entity_labels.jsonl"):
            if label in wanted:
                named.setdefault(label, set()).add(item)
        unmapped["entities"] = len(wanted - set(named))
        for label in wanted:
            entities.update(named.get(label, ()))
        properties: dict[str, str] = {}
        for prop, label, _ in sorted(
            _jsonl(mintqa_kg / "relation_labels.jsonl"), key=lambda r: r[2] != "label"
        ):
            properties.setdefault(label, prop)
        missing: set[str] = set()
        for chain in asked:
            relations = [r for _, r, _ in chain]
            missing.update(r for r in relations if r not in properties)
            if len(chain) > 1 and all(r in properties for r in relations):
                patterns.add(tuple(properties[r] for r in relations))
        unmapped["relations"] = len(missing)
    return {
        "artifact_kind": "xgraph.path_holdout",
        "version": 1,
        "scope": "chains" if chains else "subjects",
        "entities": sorted(entities),
        "patterns": [list(p) for p in sorted(patterns)],
        "unmapped": unmapped,
    }


def _jsonl_dicts(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            if line.strip():
                yield json.loads(line)


def main(argv: list[str] | None = None) -> None:
    from xgraph.provenance import provenance

    parser = argparse.ArgumentParser(prog="python -m xgraph.pathdata", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    held = commands.add_parser("holdout", help="benchmark entities to hold out")
    held.add_argument("--mquake", type=Path, action="append", default=[], help="MQuAKE JSON")
    held.add_argument(
        "--mintqa", type=Path, action="append", default=[], help="a MINTQA split (JSONL)"
    )
    held.add_argument("--mintqa-kg", type=Path, help="MINTQA's KG directory, for its labels")
    held.add_argument("--chains", action="store_true", help="every entity of each chain")
    held.add_argument(
        "--patterns", action="store_true", help="also the chains' relation compositions"
    )
    held.add_argument("--out", type=Path, required=True, help="holdout file (JSON)")
    sized = commands.add_parser(
        "frontiers", help="add each example's endpoint frontier size to written examples"
    )
    sized.add_argument("--examples", type=Path, required=True, help="path-writing examples")
    sized.add_argument("--out", type=Path, required=True, help="the examples with sizes (JSONL)")
    made = commands.add_parser("examples", help="path-writing examples from sampled paths")
    made.add_argument("--kg", type=Path, required=True, help="KG directory")
    made.add_argument("--n", type=int, required=True, help="examples, spread over --hops")
    made.add_argument("--seed", type=int, default=0)
    made.add_argument("--hops", type=int, nargs="+", default=[1, 2, 3, 4])
    made.add_argument("--holdout", type=Path, help="holdout file (python -m xgraph.pathdata)")
    made.add_argument(
        "--holdout-compositions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="hold out the holdout file's relation compositions too, not its entities alone",
    )
    made.add_argument("--alias-rate", type=float, default=ALIAS_RATE)
    made.add_argument("--out", type=Path, required=True, help="examples (JSONL)")
    made.add_argument(
        "--max-endpoints",
        type=int,
        default=1,
        help="the most entities a path's relations may reach from its start",
    )
    made.add_argument(
        "--balance", action="store_true", help="relation-balanced sampling (balanced_paths)"
    )
    defaults = Balance()
    made.add_argument(
        "--min-triples",
        type=int,
        default=defaults.min_triples,
        help="with --balance: the inventory, relations with this many triples in the KG",
    )
    made.add_argument(
        "--min-occurrences",
        type=int,
        default=defaults.min_occurrences,
        help="with --balance: occurrences each inventory relation should reach",
    )
    made.add_argument(
        "--max-composition",
        type=int,
        default=defaults.max_composition,
        help="with --balance: the most paths one relation sequence may have",
    )
    made.add_argument(
        "--balance-power",
        type=float,
        default=defaults.power,
        help="with --balance: relations are then drawn by triples ** power",
    )
    for command, needed in ((made, False), (sized, True)):
        command.add_argument(
            "--index",
            type=Path,
            required=needed,
            help="the KG index path mode runs with, to record each endpoint_frontier",
        )
        command.add_argument(
            "--tokenizer", type=Path, required=needed, help="its served tokenizer (tokenizer.json)"
        )
    args = parser.parse_args(argv)
    if args.command == "examples" and (args.index is None) != (args.tokenizer is None):
        parser.error("--index and --tokenizer go together")
    frontier_inputs = (
        []
        if args.command == "holdout" or args.index is None
        else [args.index / "kg-index.json", args.tokenizer]
    )

    def sized_examples(records: list[dict]) -> list[dict]:
        from xgraph.index import load_kg_index
        from xgraph.tokenizer import ServedTokenizer

        index = load_kg_index(args.index)
        sizes = endpoint_frontiers(records, index, ServedTokenizer.from_file(args.tokenizer))
        return [{**r, "endpoint_frontier": n} for r, n in zip(records, sizes, strict=True)]

    def write_jsonl(path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as out:
            for record in records:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.command == "frontiers":
        records = sized_examples(list(_jsonl_dicts(args.examples)))
        write_jsonl(args.out, records)
        manifest = {
            "artifact_kind": "xgraph.path_frontiers",
            "version": 1,
            "examples": len(records),
            "endpoint_frontiers": _frontier_counts(records),
            "provenance": provenance([args.examples, *frontier_inputs]),
        }
        args.out.with_name(args.out.name + ".manifest.json").write_text(
            json.dumps(manifest, indent=1) + "\n"
        )
        print(f"{len(records)} examples, endpoint frontiers {manifest['endpoint_frontiers']}")
        return

    if args.command == "holdout":
        if not (args.mquake or args.mintqa):
            parser.error("holdout needs --mquake or --mintqa")
        result = holdout(args.mquake, args.mintqa, args.mintqa_kg, chains=args.chains)
        if not args.patterns:
            result["patterns"] = []
        inputs = [*args.mquake, *args.mintqa]
        if args.mintqa:
            inputs += [args.mintqa_kg / "entity_labels.jsonl"]
            inputs += [args.mintqa_kg / "relation_labels.jsonl"]
        result["provenance"] = provenance(inputs)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print(f"{len(result['entities'])} entities, {len(result['patterns'])} patterns held out")
        return

    held_out = Holdout.read(args.holdout) if args.holdout else Holdout()
    if not args.holdout_compositions:
        held_out = Holdout(held_out.entities)
    graph = load_graph(args.kg, held_out, max_endpoints=args.max_endpoints)
    balance = coverage = None
    if args.balance:
        balance = Balance(
            args.min_triples, args.min_occurrences, args.max_composition, args.balance_power
        )
        paths, coverage = balanced_paths(graph, args.n, args.seed, args.hops, balance)
    else:
        paths = sample_paths(graph, args.n, args.seed, args.hops)
    built = examples(graph, paths, args.seed, alias_rate=args.alias_rate)
    if args.index is not None:
        built = sized_examples(built)
    write_jsonl(args.out, built)
    by_hops = Counter(e["hops"] for e in built)
    share, rest = divmod(args.n, len(args.hops))
    requested = {k: share + (i < rest) for i, k in enumerate(args.hops)}
    relations = Counter(r for e in built for r in e["walk"]["relations"])
    sequence, most = (
        Counter(tuple(e["walk"]["relations"]) for e in built).most_common(1)[0]
        if built
        else ((), 0)
    )
    inputs = [args.kg / n for n in ("entity_labels.jsonl", "relation_labels.jsonl")]
    inputs += [args.kg / "sitelinks.jsonl", args.kg / "triples.npy"]
    inputs += [args.holdout] if args.holdout else []
    inputs += frontier_inputs
    manifest = {
        "artifact_kind": "xgraph.path_examples",
        "version": 1,
        "examples": len(built),
        "by_hops": {str(k): by_hops[k] for k in args.hops},
        "short": {
            str(k): requested[k] - by_hops[k] for k in args.hops if by_hops[k] < requested[k]
        },
        "starts": int(graph.start.sum()),
        "held_out": {"entities": len(held_out.entities), "patterns": len(held_out.patterns)},
        "relations": dict(relations.most_common()),
        "largest_composition": {"relations": list(sequence), "paths": most},
        "settings": {
            "n": args.n,
            "seed": args.seed,
            "hops": args.hops,
            "alias_rate": args.alias_rate,
            "holdout_compositions": args.holdout_compositions,
            "max_endpoints": args.max_endpoints,
            "min_start_relations": MIN_START_RELATIONS,
            "max_reached": MAX_REACHED,
            "balance": asdict(balance) if balance else None,
        },
        "provenance": provenance(inputs),
    }
    if args.index is not None:
        manifest["endpoint_frontiers"] = _frontier_counts(built)
    if coverage is not None:
        counts = {f"P{r}": relations[f"P{r}"] for r in coverage.inventory}
        manifest["coverage"] = {
            "inventory": {f"P{r}": c for r, c in coverage.inventory.items()},
            "minimum": min(counts.values(), default=0),
            "below_minimum": {
                r: c
                for r, c in sorted(counts.items(), key=lambda rc: rc[1])
                if c < balance.min_occurrences
            },
            "exhausted": [f"P{r}" for r in sorted(coverage.exhausted)],
            "largest_composition": manifest["largest_composition"],
        }
    args.out.with_name(args.out.name + ".manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n"
    )
    short = f", short of {sum(manifest['short'].values())}" if manifest["short"] else ""
    print(f"{len(built)} examples{short}")


if __name__ == "__main__":
    main()
