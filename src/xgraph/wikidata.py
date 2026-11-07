"""The quality gate's KG, streamed from the pinned Wikidata dumps.

`docs/quality-gate.md` fixes the recipe: items with an English Wikipedia sitelink,
their English labels and aliases, and the truthy statements between such items.
The truthy dump holds no sitelinks, so they come from the sitelink table
(`wb_items_per_site`) of the nearest database dump.

Labels in Wikidata's `mul` language (default names shared by all languages) are kept
too, for items and properties alike, since some items have their name only there. A
`mul` label is the label of an item without an English label and an alias otherwise;
`mul` aliases are aliases; a text the item already has is not repeated. Since an item's
lines come in no fixed order, the `mul` rows wait in a file until the stream ends, and
are then appended (`_add_mul`). The manifest records the label languages and how many
labels and aliases came from `mul`. KG directories streamed before `mul` was kept (the
gate's `data/wikidata/gate-kg`) have English rows only.

Nothing is decompressed to disk and memory stays bounded: the sitelinks fill a
bitmap over item numbers, and the dump streams through `lbzip2 -dc` and an `rg`
prefilter into `filter_truthy`, which writes its results as it goes:

- `sitelinks.jsonl`: `["Q31", "Belgium"]`, the item and its English Wikipedia title;
- `entity_labels.jsonl`: `["Q31", "Kingdom of Belgium", "alias"]`, for kept items;
- `relation_labels.jsonl`: `["P47", "shares border with", "label"]`, for every property;
- `triples.npy`: int32 rows (subject, relation, object) of item numbers and property
  numbers, for truthy statements whose subject and object are both kept items;
- `manifest.json`: counts and provenance.

    python -m xgraph.wikidata stream \\
        --truthy data/wikidata/wikidata-20261007-truthy-BETA.nt.bz2 \\
        --sitelinks data/wikidata/wikidatawiki-20261001-wb_items_per_site.sql.gz \\
        --out data/wikidata/gate-kg

`build_index` then compiles these files into a saved KG index for the served
tokenizer, with bounded memory (`python -m xgraph.wikidata index`).

`sitelinks.jsonl` keeps the English title only. Each item's sitelink count (its
sitelinks to every Wikimedia site, which the sitelink rank orders an ambiguous label's
entities by) is counted from the sitelink table into an optional fourth file:

- `sitelink_counts.npy`: int32 rows (item number, sitelink count), one per item of
  `entity_labels.jsonl`, in item order.

    python -m xgraph.wikidata sitelink-counts --kg data/wikidata/gate-kg \\
        --sitelinks data/wikidata/wikidatawiki-20261001-wb_items_per_site.sql.gz

`build_index` puts the counts in the index when the KG directory has them, and
`python -m xgraph.wikidata index-sitelinks --kg <KG directory> --index <its index>` gives
them to an index built before, without rebuilding it.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import re
import shutil
import subprocess
from array import array
from collections.abc import Container, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import numpy as np

from xgraph.provenance import provenance
from xgraph.syntax import SpanSyntax, preset
from xgraph.tokenizer import ServedTokenizer

ENTITY = "<http://www.wikidata.org/entity/"
LABEL = "<http://www.w3.org/2000/01/rdf-schema#label> "
ALIAS = "<http://www.w3.org/2004/02/skos/core#altLabel> "
DIRECT = "<http://www.wikidata.org/prop/direct/"
ENGLISH = '"@en .'
MUL = '"@mul .'  # Wikidata's default names for all languages

# Lines filter_truthy can use, for rg to pass through; everything else is dropped
# before Python sees it.
PREFILTER = (
    r"^<http://www\.wikidata\.org/entity/[QP][0-9]+> "
    r"<http://www\.w3\.org/(2000/01/rdf-schema#label|2004/02/skos/core#altLabel)> "
    r'".*"@(en|mul) \.$',
    r"^<http://www\.wikidata\.org/entity/Q[0-9]+> <http://www\.wikidata\.org/prop/direct/P[0-9]+> "
    r"<http://www\.wikidata\.org/entity/Q[0-9]+> \.$",
)

SITELINK = re.compile(r"\((\d+),(\d+),'((?:[^'\\]|\\.)*)','((?:[^'\\]|\\.)*)'\)")
MYSQL_ESCAPES = {"0": "\0", "n": "\n", "r": "\r", "t": "\t", "Z": "\x1a"}
NT_ESCAPE = re.compile(r"\\(?:u([0-9A-Fa-f]{4})|U([0-9A-Fa-f]{8})|(.))")
NT_ESCAPES = {"t": "\t", "b": "\b", "n": "\n", "r": "\r", "f": "\f"}
FLUSH = 1 << 20  # values per triple buffer flush


class ItemBitmap:
    """A set of item numbers (the n of Qn) as a growable bitmap: 1 bit per number."""

    def __init__(self) -> None:
        self._bits = bytearray()
        self.count = 0

    @classmethod
    def of(cls, numbers: Iterable[int]) -> ItemBitmap:
        bitmap = cls()
        for n in numbers:
            bitmap.add(n)
        return bitmap

    def add(self, n: int) -> None:
        byte = n >> 3
        if byte >= len(self._bits):
            self._bits.extend(bytes(max(byte + 1 - len(self._bits), len(self._bits))))
        mask = 1 << (n & 7)
        if not self._bits[byte] & mask:
            self._bits[byte] |= mask
            self.count += 1

    def __contains__(self, n: int) -> bool:
        byte = n >> 3
        return byte < len(self._bits) and bool(self._bits[byte] & (1 << (n & 7)))


def _mysql_unescape(text: str) -> str:
    return re.sub(r"\\(.)", lambda m: MYSQL_ESCAPES.get(m.group(1), m.group(1)), text)


def sitelinks(lines: Iterable[str], site: str = "enwiki") -> Iterator[tuple[int, str]]:
    """(item number, page title) for each sitelink to the site in a wb_items_per_site dump.

    Rows are read from each INSERT statement into the table up to its closing
    semicolon, whether the statement is one line or one row per line.
    """
    for match in _sitelink_rows(lines):
        if _mysql_unescape(match.group(3)) == site:
            yield int(match.group(2)), _mysql_unescape(match.group(4))


def _sitelink_rows(lines: Iterable[str]) -> Iterator[re.Match[str]]:
    """The rows of a wb_items_per_site dump: (row id, item number, site, page title)."""
    inside = False
    for line in lines:
        if line.startswith("INSERT INTO `wb_items_per_site`"):
            inside = True
        if not inside:
            continue
        yield from SITELINK.finditer(line)
        if line.rstrip().endswith(";"):
            inside = False


def count_sitelinks(table: Path, kg: Path) -> int:
    """Write a KG directory's sitelink_counts.npy: int32 rows (item number, sitelink count),
    one per item of its entity_labels.jsonl in item order, counting the item's sitelinks to
    every Wikimedia site in the sitelink table (wb_items_per_site, .sql.gz). Returns the
    rows written."""
    kg = Path(kg)
    items = _entity_items(kg)
    counts = np.zeros(len(items), dtype=np.int64)
    batch = array("q")

    def flush() -> None:
        at = _positions(items, np.array(batch, dtype=np.int64))
        np.add.at(counts, at[at >= 0], 1)
        del batch[:]

    with gzip.open(table, "rt", encoding="utf-8") as lines:
        for match in _sitelink_rows(lines):
            batch.append(int(match.group(2)))
            if len(batch) >= FLUSH:
                flush()
    flush()
    rows = np.stack([items, counts], axis=1).astype(np.int32).reshape(-1, 2)
    np.save(kg / SITELINK_COUNTS, rows)
    return len(rows)


SITELINK_COUNTS = "sitelink_counts.npy"
"""A KG directory's optional sitelink counts (count_sitelinks), which build_index and
index_sitelinks give the KG index for the sitelink rank."""


def _entity_items(kg: Path) -> np.ndarray:
    """The item numbers of a KG directory's entities (with a label row), sorted."""
    return np.unique(
        np.fromiter((int(item[1:]) for item, _, _ in _jsonl(kg / "entity_labels.jsonl")), np.int64)
    )


def _sitelinks_of(kg: Path, numbers: np.ndarray) -> np.ndarray | None:
    """The sitelink counts of these item numbers from the KG directory's sitelink counts (0
    for an item without a row), or None when it has none."""
    path = Path(kg) / SITELINK_COUNTS
    if not path.is_file():
        return None
    rows = np.load(path).reshape(-1, 2)
    known, counts = rows[:, 0].astype(np.int64), rows[:, 1]
    at = _positions(known, numbers)
    return np.where(at >= 0, counts[np.maximum(at, 0)] if len(known) else 0, 0).astype(np.int32)


def _positions(known: np.ndarray, numbers: np.ndarray) -> np.ndarray:
    """Where each number is in the sorted known numbers, or -1."""
    if not len(known):
        return np.full(len(numbers), -1, dtype=np.int64)
    at = np.searchsorted(known, numbers)
    clipped = np.minimum(at, len(known) - 1)
    return np.where(known[clipped] == numbers, clipped, -1)


def _dates_of(kg: Path, numbers: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """The date table (xgraph.index date_table) of the entities with these item numbers
    from the KG directory's merged dates (xgraph.literals DATE_VALUES), or None when it
    has none."""
    from xgraph.expressions import stored_date
    from xgraph.index import date_table
    from xgraph.literals import DATE_VALUES

    path = Path(kg) / DATE_VALUES
    if not path.is_file():
        return None
    rows = np.load(path).reshape(-1, 4)
    at = _positions(numbers, rows[:, 0].astype(np.int64))
    if len(at) and int(at.min()) < 0:
        missing = rows[at < 0, 0][:5].tolist()
        raise ValueError(f"{path} dates items the KG directory has no label for: {missing}")
    return date_table(
        {
            int(e): stored_date(int(y), int(m), int(d))
            for e, (_, y, m, d) in zip(at, rows, strict=True)
        }
    )


def index_sitelinks(kg: Path, index: Path) -> None:
    """Give a saved KG index of this KG directory the directory's sitelink counts, without
    rebuilding it (python -m xgraph.wikidata index-sitelinks)."""
    from xgraph.index import add_sitelinks, load_kg_index

    owners = load_kg_index(index).labels.owner_ids
    counts = _sitelinks_of(kg, np.array([int(q[1:]) for q in owners], dtype=np.int64))
    if counts is None:
        raise FileNotFoundError(
            f"{kg} has no {SITELINK_COUNTS}; write it with "
            f"python -m xgraph.wikidata sitelink-counts --kg {kg} --sitelinks <table>"
        )
    add_sitelinks(index, counts)


def unescape(text: str) -> str:
    """Undo the escapes of an N-Triples string literal."""

    def replace(m: re.Match[str]) -> str:
        code = m.group(1) or m.group(2)
        if code:
            return chr(int(code, 16))
        return NT_ESCAPES.get(m.group(3), m.group(3))

    return NT_ESCAPE.sub(replace, text)


def _label(subject: str, rest: str, language: str = ENGLISH) -> tuple[str, str] | None:
    """(text, label or alias) when the rest of a line is a label or alias in the language."""
    if rest.startswith((LABEL, ALIAS)) and rest.endswith(language):
        kind = "label" if rest.startswith(LABEL) else "alias"
        predicate = LABEL if kind == "label" else ALIAS
        return unescape(rest[len(predicate) + 1 : -len(language)]), kind
    return None


def _text_key(subject: str, text: str) -> int:
    """A 64-bit hash of an item's or property's text, for finding texts it already has."""
    digest = hashlib.blake2b(f"{subject}\t{text}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def item_labels(lines: Iterable[str], items: Container[int], out: Path) -> int:
    """Write the English labels and aliases of the items, any items, as JSON lines."""
    count = 0
    with Path(out).open("w", encoding="utf-8") as labels:
        for line in lines:
            line = line.rstrip("\n")
            if not line.startswith(ENTITY + "Q"):
                continue
            end = line.find(">", len(ENTITY))
            subject = line[len(ENTITY) : end]
            found = _label(subject, line[end + 2 :])
            if found and int(subject[1:]) in items:
                labels.write(json.dumps([subject, *found], ensure_ascii=False) + "\n")
                count += 1
    return count


def filter_truthy(lines: Iterable[str], keep: ItemBitmap, out: Path) -> dict[str, int]:
    """Write the labels and the item-to-item truthy triples the gate KG keeps.

    English labels and aliases are written as they come. "mul" ones (Wikidata's default
    names for all languages) are set aside and added by _add_mul once every English
    label is known, since an item's lines come in no fixed order.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {"entity_labels": 0, "relation_labels": 0, "triples": 0}
    raw = out / "triples.i32"
    buffer = array("i")
    english = _Texts()
    mul = _Texts(out / MUL_ROWS)
    with (
        (out / "entity_labels.jsonl").open("w", encoding="utf-8") as entities,
        (out / "relation_labels.jsonl").open("w", encoding="utf-8") as relations,
        raw.open("wb") as triples,
        mul,
    ):
        for line in lines:
            line = line.rstrip("\n")
            if not line.startswith(ENTITY):
                continue
            end = line.find(">", len(ENTITY))
            subject = line[len(ENTITY) : end]
            rest = line[end + 2 :]
            if rest.startswith((LABEL, ALIAS)):
                if subject[0] not in "QP" or (subject[0] == "Q" and int(subject[1:]) not in keep):
                    continue
                if found := _label(subject, rest):
                    english.add(subject, *found)
                    record = json.dumps([subject, *found], ensure_ascii=False) + "\n"
                    if subject[0] == "Q":
                        entities.write(record)
                        counts["entity_labels"] += 1
                    else:
                        relations.write(record)
                        counts["relation_labels"] += 1
                elif found := _label(subject, rest, MUL):
                    mul.add(subject, *found)
            elif rest.startswith(DIRECT) and subject[0] == "Q":
                stop = rest.find(">")
                obj = rest[stop + 2 :]
                if not obj.startswith(ENTITY + "Q"):
                    continue
                s, o = int(subject[1:]), int(obj[len(ENTITY) + 1 : obj.find(">")])
                if s in keep and o in keep:
                    buffer.extend((s, int(rest[len(DIRECT) + 1 : stop]), o))
                    counts["triples"] += 1
                    if len(buffer) >= FLUSH:
                        buffer.tofile(triples)
                        del buffer[:]
        buffer.tofile(triples)
    _raw_to_npy(raw, out / "triples.npy", counts["triples"])
    added = _add_mul(out, english, mul)
    counts["entity_labels"] += added["mul_entity_labels"] + added["mul_entity_aliases"]
    counts["relation_labels"] += added["mul_relation_labels"] + added["mul_relation_aliases"]
    return {**counts, **added}


PLURAL = {"label": "labels", "alias": "aliases"}
MUL_ROWS = "mul_labels.tmp.jsonl"
"""Where filter_truthy sets the mul label rows aside until _add_mul writes them."""


class _Texts:
    """The labels and aliases of one language seen so far, kept as 9 bytes per row: the
    hash of (item or property, text), whether the row is an alias, and a bitmap of the
    items and properties with a label. The rows themselves go to a file, if given."""

    def __init__(self, path: Path | None = None) -> None:
        self.keys = array("Q")
        self.alias = array("b")
        self.labelled = {"Q": ItemBitmap(), "P": ItemBitmap()}
        self.path = path
        self._file: IO[str] | None = None

    def __enter__(self) -> _Texts:
        if self.path is not None:
            self._file = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, *exc: object) -> None:
        if self._file is not None:
            self._file.close()

    def add(self, subject: str, text: str, kind: str) -> None:
        self.keys.append(_text_key(subject, text))
        self.alias.append(kind == "alias")
        if kind == "label":
            self.labelled[subject[0]].add(int(subject[1:]))
        if self._file is not None:
            self._file.write(json.dumps([subject, text, kind], ensure_ascii=False) + "\n")


def _add_mul(out: Path, english: _Texts, mul: _Texts) -> dict[str, int]:
    """Append the mul rows set aside to the label files and delete their file.

    A mul label becomes the label of an item or property without an English label, and
    an alias otherwise; mul aliases stay aliases. A text the item or property already
    has, in English or in an earlier mul row, is not written again, and of two equal mul
    texts the label is kept. Texts are compared case-sensitively through the 64-bit hash
    of (item or property, text), so a false match needs a hash collision. Memory stays at
    the hashes and bitmaps of _Texts.
    """
    counts = dict.fromkeys(
        ("mul_entity_labels", "mul_entity_aliases", "mul_relation_labels", "mul_relation_aliases"),
        0,
    )
    path = out / MUL_ROWS
    keys = np.frombuffer(mul.keys, dtype=np.uint64) if mul.keys else np.zeros(0, np.uint64)
    alias = np.frombuffer(mul.alias, dtype=np.int8) if mul.alias else np.zeros(0, np.int8)
    order = np.lexsort((alias, keys))  # by text, labels first, then in dump order
    first = np.ones(len(keys), dtype=bool)
    first[1:] = keys[order[1:]] != keys[order[:-1]]
    new = np.zeros(len(keys), dtype=bool)
    new[order[first]] = True
    del order, first
    if english.keys:
        known = np.sort(np.frombuffer(english.keys, dtype=np.uint64))
        at = np.minimum(np.searchsorted(known, keys), len(known) - 1)
        new &= known[at] != keys
        del known, at
    with (
        (out / "entity_labels.jsonl").open("a", encoding="utf-8") as entities,
        (out / "relation_labels.jsonl").open("a", encoding="utf-8") as relations,
        path.open(encoding="utf-8") as rows,
    ):
        for written, line in zip(new, rows, strict=True):
            if not written:
                continue
            subject, text, kind = json.loads(line)
            if kind == "label" and int(subject[1:]) in english.labelled[subject[0]]:
                kind = "alias"
            entity = subject[0] == "Q"
            (entities if entity else relations).write(
                json.dumps([subject, text, kind], ensure_ascii=False) + "\n"
            )
            counts[f"mul_{'entity' if entity else 'relation'}_{PLURAL[kind]}"] += 1
    path.unlink()
    return counts


def _raw_to_npy(raw: Path, target: Path, rows: int) -> None:
    """Copy raw int32 triples into a .npy file chunk by chunk, then delete the raw file."""
    triples = np.lib.format.open_memmap(target, mode="w+", dtype=np.int32, shape=(rows, 3))
    chunk_bytes = max(1, FLUSH // 3) * 3 * 4  # whole triples of int32
    with raw.open("rb") as source:
        start = 0
        while chunk := source.read(chunk_bytes):
            values = np.frombuffer(chunk, dtype=np.int32).reshape(-1, 3)
            triples[start : start + len(values)] = values
            start += len(values)
    triples.flush()
    del triples
    raw.unlink()


@dataclass
class Outputs:
    entity_labels: list[tuple[str, str, str]] = field(default_factory=list)
    relation_labels: list[tuple[str, str, str]] = field(default_factory=list)
    sitelinks: list[tuple[str, str]] = field(default_factory=list)
    triples: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int32))


def read_outputs(out: Path) -> Outputs:
    """Read a small output directory back (tests and inspection; the triples are memory-mapped)."""
    out = Path(out)

    def records(name: str) -> list[tuple]:
        path = out / name
        if not path.is_file():
            return []
        return [tuple(json.loads(line)) for line in path.open(encoding="utf-8")]

    return Outputs(
        entity_labels=records("entity_labels.jsonl"),
        relation_labels=records("relation_labels.jsonl"),
        sitelinks=records("sitelinks.jsonl"),
        triples=np.load(out / "triples.npy", mmap_mode="r"),
    )


@contextmanager
def _prefiltered(dump: Path, patterns: Iterable[str]) -> Iterator[IO[str]]:
    """The lines of a bz2 dump that match one of the patterns: lbzip2 piped into rg."""
    decompress = subprocess.Popen(["lbzip2", "-dc", str(dump)], stdout=subprocess.PIPE)
    prefilter = subprocess.Popen(
        ["rg", "--no-config", "--text", *(arg for p in patterns for arg in ("-e", p))],
        stdin=decompress.stdout,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert decompress.stdout is not None and prefilter.stdout is not None
    decompress.stdout.close()  # let lbzip2 see a broken pipe if rg exits
    yield prefilter.stdout
    if decompress.wait() != 0 or prefilter.wait() not in (0, 1):  # rg: 1 means no match
        raise RuntimeError(
            f"streaming {dump} failed: lbzip2 exited {decompress.returncode}, "
            f"rg exited {prefilter.returncode}"
        )


def build(dump: Path, table: Path, out: Path, site: str = "enwiki") -> dict:
    """Stream the sitelink table and the truthy dump into the gate KG files."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    keep = ItemBitmap()
    with (
        gzip.open(table, "rt", encoding="utf-8") as lines,
        (out / "sitelinks.jsonl").open("w", encoding="utf-8") as titles,
    ):
        for item, title in sitelinks(lines, site):
            keep.add(item)
            titles.write(json.dumps([f"Q{item}", title], ensure_ascii=False) + "\n")
    if keep.count == 0:
        raise ValueError(f"no sitelinks to {site} in {table}; is it a wb_items_per_site dump?")

    with _prefiltered(dump, PREFILTER) as lines:
        counts = filter_truthy(lines, keep, out)
    manifest = {
        "artifact_kind": "xgraph.wikidata_gate_kg",
        "version": 1,
        "site": site,
        "label_languages": ["en", "mul"],
        "counts": {"kept_items": keep.count, **counts},
        "provenance": provenance([Path(dump), Path(table)]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def add_titles(kg: Path, out: Path) -> int:
    """A variant gate KG whose entities also carry their English Wikipedia title as a label.

    Titles are unique per item and familiar to a model, so a model can name an entity
    whose label others share by its title ("Andrew Young (footballer)"). A title equal
    to one of the item's labels or aliases is not added again. The variant links the
    other files of the KG. Returns the number of titles added.
    """
    kg, out = Path(kg), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    titles = {item: title for item, title in _jsonl(kg / "sitelinks.jsonl")}
    covered = ItemBitmap()
    with (out / "entity_labels.jsonl").open("w", encoding="utf-8") as labels:
        with (kg / "entity_labels.jsonl").open(encoding="utf-8") as lines:
            for line in lines:
                labels.write(line)
                item, label, _ = json.loads(line)
                if titles.get(item) == label:
                    covered.add(int(item[1:]))
        added = 0
        for item, title in titles.items():
            if int(item[1:]) not in covered:
                labels.write(json.dumps([item, title, "title"], ensure_ascii=False) + "\n")
                added += 1
    for name in ("relation_labels.jsonl", "sitelinks.jsonl", "triples.npy"):
        target = out / name
        target.unlink(missing_ok=True)
        target.symlink_to((kg / name).resolve())
    return added


CHUNK = 1 << 18
"""Labels or triples handled per batch by build_index; bounds its working memory."""


def subset(kg: Path, entities: int, seed: int, out: Path) -> dict:
    """A smaller KG for measuring the engine against KG size: `entities` items, the
    labels of each, and the triples among them.

    Items are drawn by edge sampling, as for subgraphs of large graphs: triples between
    labelled items are visited in an order drawn with the seed, and their subjects and
    objects kept until there are `entities`. Drawing items uniformly instead would leave
    a small subset almost without triples (their number falls with the square of the
    share kept), unlike any real KG of that size.
    """
    kg, out = Path(kg), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    qids = np.unique(
        np.fromiter((int(item[1:]) for item, _, _ in _jsonl(kg / "entity_labels.jsonl")), np.int64)
    )
    labelled = np.zeros(int(qids[-1]) + 1, dtype=bool)
    labelled[qids] = True
    triples = np.load(kg / "triples.npy", mmap_mode="r")
    edges = np.asarray(triples)[:, [0, 2]].astype(np.int64)
    edges = edges[(edges < len(labelled)).all(axis=1)]
    edges = edges[labelled[edges[:, 0]] & labelled[edges[:, 1]]]
    order = np.random.default_rng(seed).permutation(len(edges))
    marked = np.zeros(len(labelled), dtype=bool)
    remaining = entities
    for at in range(0, len(order), CHUNK):
        sequence = edges[order[at : at + CHUNK]].ravel()  # subject, object, subject, ...
        new, first = np.unique(sequence, return_index=True)
        unseen = ~marked[new]
        new = new[unseen][np.argsort(first[unseen], kind="stable")][:remaining]
        marked[new] = True
        remaining -= len(new)
        if remaining == 0:
            break
    del edges, order
    kept = np.flatnonzero(marked).astype(np.int64)

    def member(numbers: np.ndarray) -> np.ndarray:
        at = np.minimum(np.searchsorted(kept, numbers), len(kept) - 1)
        return kept[at] == numbers

    chosen = set(kept.tolist())
    labels = 0
    with (
        (kg / "entity_labels.jsonl").open(encoding="utf-8") as lines,
        (out / "entity_labels.jsonl").open("w", encoding="utf-8") as written,
    ):
        for line in lines:
            item = line[2 : line.index('"', 2)]
            if int(item[1:]) in chosen:
                written.write(line)
                labels += 1
    shutil.copyfile(kg / "relation_labels.jsonl", out / "relation_labels.jsonl")
    raw = out / "triples.raw"
    rows = 0
    with raw.open("wb") as written:
        for at in range(0, len(triples), CHUNK):
            chunk = np.asarray(triples[at : at + CHUNK])
            chunk = chunk[
                member(chunk[:, 0].astype(np.int64)) & member(chunk[:, 2].astype(np.int64))
            ]
            chunk.astype(np.int32).tofile(written)
            rows += len(chunk)
    _raw_to_npy(raw, out / "triples.npy", rows)
    manifest = {
        "artifact_kind": "xgraph.wikidata_kg_subset",
        "version": 1,
        "seed": seed,
        "counts": {"entities": len(kept), "entity_labels": labels, "triples": rows},
        "provenance": provenance([kg / "entity_labels.jsonl", kg / "triples.npy"]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _jsonl(path: Path) -> Iterator[list]:
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            yield json.loads(line)


def _sort_key(tokens: np.ndarray, owner: int) -> bytes:
    """Bytes that sort like (token sequence, owner): 3-byte tokens + 1, a zero end, the owner."""
    codes = tokens.astype(np.int64) + 1
    parts = np.stack([(codes >> 16) & 255, (codes >> 8) & 255, codes & 255], axis=1)
    return parts.astype(np.uint8).tobytes() + b"\0\0\0" + owner.to_bytes(4, "big")


def _entity_rows(
    path: Path,
    qids: np.ndarray,
    tokenizer: ServedTokenizer,
    syntax: SpanSyntax,
    extra: Iterable[tuple[str, str]] = (),
):
    """The label index of the entities in qids, from entity_labels.jsonl, chunk by chunk,
    and the extra (item, label) rows."""
    from xgraph.index import ExcludedLabel, LabelIndex, _label_encoding_error

    if tokenizer.vocab_size >= (1 << 24) - 1:
        raise ValueError("build_index packs token ids into 3 bytes; the vocabulary is too large")
    flat, lengths, owners = array("i"), array("i"), array("i")
    excluded: set[tuple[str, str, str]] = set()

    def flush(batch: list[tuple[int, str, str]]) -> None:
        encoded = tokenizer.encode_batch([syntax.label_prefix + label for _, label, _ in batch])
        for (owner, label, item), tokens in zip(batch, encoded, strict=True):
            reason = _label_encoding_error(label, tokens, tokenizer, syntax)
            if reason:
                excluded.add((item, label, reason))
                continue
            flat.extend(tokens)
            lengths.append(len(tokens))
            owners.append(owner)

    batch: list[tuple[int, str, str]] = []
    for item, label, _ in itertools.chain(_jsonl(path), ((i, lab, "label") for i, lab in extra)):
        if not label.strip():
            excluded.add((item, label, "blank label"))
        elif syntax.excludes(label):
            excluded.add((item, label, "label contains span syntax"))
        else:
            batch.append((int(np.searchsorted(qids, int(item[1:]))), label, item))
            if len(batch) >= CHUNK:
                flush(batch)
                batch = []
    flush(batch)

    tokens = np.frombuffer(flat, dtype=np.int32)
    length = np.frombuffer(lengths, dtype=np.int32).astype(np.int64)
    owner = np.frombuffer(owners, dtype=np.int32)
    start = np.zeros(len(length) + 1, dtype=np.int64)
    np.cumsum(length, out=start[1:])
    keys = [_sort_key(tokens[start[i] : start[i + 1]], int(owner[i])) for i in range(len(length))]
    order = sorted(range(len(keys)), key=keys.__getitem__)
    order = [i for n, i in enumerate(order) if n == 0 or keys[i] != keys[order[n - 1]]]
    del keys
    order = np.array(order, dtype=np.int64)
    offsets = np.zeros(len(order) + 1, dtype=np.int64)
    np.cumsum(length[order], out=offsets[1:])
    sorted_tokens = np.empty(offsets[-1], dtype=np.int32)
    for at in range(0, len(order), CHUNK):  # gather the rows in order, a batch at a time
        rows = order[at : at + CHUNK]
        counts = length[rows]
        within = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        sorted_tokens[offsets[at] : offsets[at] + counts.sum()] = tokens[
            np.repeat(start[rows], counts) + within
        ]
    return LabelIndex(
        tokens=sorted_tokens,
        offsets=offsets,
        owners=owner[order].copy(),
        owner_ids=tuple(f"Q{q}" for q in qids),
        tokenizer_fingerprint=tokenizer.fingerprint,
        excluded=tuple(ExcludedLabel(*e) for e in sorted(excluded)),
    )


def _case_entity_rows(path, qids, tokenizer, syntax, extra, scratch):
    """Opt-in external row sort; keep large alias builds below the CPU memory cap.

    Exact surfaces are deduplicated before tokenization. SQLite's binary text
    comparison preserves owner collisions, even when a canonical row follows
    another owner's alias. Its temporary files stay in the destination index.
    """
    import sqlite3
    import tempfile
    from contextlib import closing

    from xgraph.index import ExcludedLabel, LabelIndex, _label_encoding_error, initial_case_alias

    if tokenizer.vocab_size >= (1 << 24) - 1:
        raise ValueError("build_index packs token ids into 3 bytes; the vocabulary is too large")
    with (
        tempfile.TemporaryDirectory(prefix="case-alias-", dir=scratch) as temporary,
        closing(sqlite3.connect(str(Path(temporary) / "rows.sqlite"))) as db,
    ):
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA cache_size=-16384")
        db.execute("PRAGMA temp_store=FILE")
        db.execute(
            "CREATE TABLE surfaces (owner INTEGER, label TEXT, added INTEGER, "
            "tokens BLOB, sort_key BLOB, PRIMARY KEY(owner, label)) WITHOUT ROWID"
        )

        def sources():
            return itertools.chain(_jsonl(path), ((i, lab, "label") for i, lab in extra))

        for generated in (False, True):
            batch = []
            for item, label, kind in sources():
                if generated:
                    label = initial_case_alias(label) if kind == "label" else None
                    if label is None:
                        continue
                owner = int(np.searchsorted(qids, int(item[1:])))
                batch.append((owner, label, int(generated)))
                if len(batch) >= CHUNK:
                    db.executemany("INSERT OR IGNORE INTO surfaces VALUES (?,?,?,NULL,NULL)", batch)
                    batch.clear()
            db.executemany("INSERT OR IGNORE INTO surfaces VALUES (?,?,?,NULL,NULL)", batch)
            db.commit()

        excluded = []
        cursor = db.execute("SELECT owner, label FROM surfaces")
        while batch := cursor.fetchmany(CHUNK):
            kept = []
            for owner, label in batch:
                reason = (
                    "blank label"
                    if not label.strip()
                    else ("label contains span syntax" if syntax.excludes(label) else None)
                )
                if reason:
                    excluded.append(ExcludedLabel(f"Q{qids[owner]}", label, reason))
                else:
                    kept.append((owner, label))
            encoded = tokenizer.encode_batch([syntax.label_prefix + label for _, label in kept])
            updates = []
            for (owner, label), tokens in zip(kept, encoded, strict=True):
                reason = _label_encoding_error(label, tokens, tokenizer, syntax)
                if reason:
                    excluded.append(ExcludedLabel(f"Q{qids[owner]}", label, reason))
                else:
                    tokens = np.asarray(tokens, dtype=np.int32)
                    updates.append((tokens.tobytes(), _sort_key(tokens, owner), owner, label))
            db.executemany(
                "UPDATE surfaces SET tokens=?, sort_key=? WHERE owner=? AND label=?", updates
            )
        db.commit()
        db.execute("CREATE INDEX surface_names ON surfaces(label) WHERE tokens IS NOT NULL")
        aliases = db.execute(
            "SELECT count(*) FROM surfaces WHERE added=1 AND tokens IS NOT NULL"
        ).fetchone()[0]
        collisions = db.execute(
            "SELECT count(*) FROM surfaces a WHERE a.added=1 AND a.tokens IS NOT NULL "
            "AND EXISTS (SELECT 1 FROM surfaces b WHERE b.label=a.label "
            "AND b.owner!=a.owner AND b.tokens IS NOT NULL)"
        ).fetchone()[0]
        db.execute("CREATE INDEX sorted_rows ON surfaces(sort_key) WHERE tokens IS NOT NULL")
        count, size = db.execute(
            "SELECT count(*), coalesce(sum(length(tokens)/4),0) FROM "
            "(SELECT tokens FROM surfaces WHERE tokens IS NOT NULL GROUP BY sort_key)"
        ).fetchone()
        tokens = np.empty(size, np.int32)
        offsets = np.zeros(count + 1, np.int64)
        owners = np.empty(count, np.int32)
        at = 0
        for row, (owner, blob) in enumerate(
            db.execute(
                "SELECT owner, tokens FROM surfaces WHERE tokens IS NOT NULL "
                "GROUP BY sort_key ORDER BY sort_key"
            )
        ):
            values = np.frombuffer(blob, np.int32)
            tokens[at : at + len(values)] = values
            at += len(values)
            offsets[row + 1] = at
            owners[row] = owner
        labels = LabelIndex(
            tokens,
            offsets,
            owners,
            tuple(f"Q{q}" for q in qids),
            tokenizer.fingerprint,
            tuple(sorted(excluded, key=lambda e: (e.owner_id, e.label, e.reason))),
        )
    return labels, {"rule": "initial-case-v1", "aliases": aliases, "collisions": collisions}


def relation_names(
    rows: Iterable[tuple[str, str, str]], unique_aliases: bool = False
) -> dict[str, list[str]]:
    """Each property's labels from relation label rows (property, label, kind), without an
    alias that is another property's label, ignoring case: written after a path, "head
    of state" names P35 (its label), not also P1906 (office held by head of state), whose
    alias it is. With unique_aliases, an alias several properties share is dropped too: a
    path naming it would otherwise follow every one of them."""
    rows = list(rows)
    labels = {label.casefold(): prop for prop, label, kind in rows if kind == "label"}
    owners: dict[str, set[str]] = {}
    for prop, label, kind in rows:
        if kind != "label":
            owners.setdefault(label.casefold(), set()).add(prop)
    names: dict[str, list[str]] = {}
    for prop, label, kind in sorted(rows, key=lambda row: row[2] != "label"):
        if kind != "label" and labels.get(label.casefold(), prop) != prop:
            continue
        if kind != "label" and unique_aliases and len(owners[label.casefold()]) > 1:
            continue
        names.setdefault(prop, []).append(label)
    return names


def build_index(
    kg: Path,
    tokenizer: ServedTokenizer,
    syntax: SpanSyntax,
    out: Path,
    unique_aliases: bool = False,
    verdicts: bool = False,
    *,
    initial_case_aliases: bool = False,
) -> None:
    """Build and save the KG index of a gate KG directory with bounded memory.

    The result equals build_kg_index on the same KG: entities are the items with a
    label or alias row, in item-number order; relations the properties with labels
    (relation_names: without an alias that is another property's label, and with
    unique_aliases without one several properties share).
    A triple is unwritable when its subject, relation, or object has no label row. With
    the KG directory's sitelink counts (count_sitelinks), the index holds them too, and
    with its merged dates (xgraph.literals merge_dates), its date table; with verdicts, the
    verdict entities (xgraph.index.VERDICTS), as build_kg_index adds them.
    Every entity, generated alias and relation row follows exact-v1: known token
    bytes must exactly reconstruct the UTF-8 label prefix plus source label.
    Empty, unknown-byte and non-round-tripping encodings are excluded/reported,
    never normalized into another label. The version is recorded in build_config;
    existing frozen indexes are not migrated by loading them.
    """
    from xgraph.index import (
        INDEX_FORMAT,
        VERDICT_NUMBERS,
        VERDICTS,
        KGIndex,
        _source_graph_hash,
        label_rows,
        primary_relation_rows,
    )
    from xgraph.kg import KGFormatError, Relation

    kg, out = Path(kg), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    qids = _entity_items(kg)
    extra: list[tuple[str, str]] = []
    if verdicts:
        numbers = np.array(sorted(VERDICT_NUMBERS.values()), dtype=qids.dtype)
        if np.isin(numbers, qids).any():
            raise KGFormatError(f"{kg} holds an item numbered as a verdict entity: {numbers}")
        qids = np.union1d(qids, numbers)
        extra = [(f"Q{n}", verdict) for verdict, n in VERDICT_NUMBERS.items()]
    aliases = None
    if initial_case_aliases:
        labels, aliases = _case_entity_rows(
            kg / "entity_labels.jsonl", qids, tokenizer, syntax, extra, out
        )
    else:
        labels = _entity_rows(kg / "entity_labels.jsonl", qids, tokenizer, syntax, extra)

    relation_labels = list(_jsonl(kg / "relation_labels.jsonl"))
    names = relation_names(relation_labels, unique_aliases)
    canonical = {r: label for r, label, kind in relation_labels if kind == "label"}
    properties = sorted(names, key=lambda p: int(p[1:]))
    relations = label_rows([Relation(p, tuple(names[p])) for p in properties], tokenizer, syntax)
    pids = np.array([int(p[1:]) for p in properties], dtype=np.int64)
    writable_relation = np.zeros(len(properties), dtype=bool)
    writable_relation[relations.owners] = True

    rows_per_entity = np.bincount(labels.owners, minlength=len(qids))
    row_start = np.zeros(len(qids) + 1, dtype=np.int64)
    np.cumsum(rows_per_entity, out=row_start[1:])
    rows_by_entity = np.argsort(labels.owners, kind="stable").astype(np.int32)

    def position(numbers: np.ndarray, known: np.ndarray) -> np.ndarray:
        at = np.searchsorted(known, numbers)
        found = (at < len(known)) & (known[np.minimum(at, len(known) - 1)] == numbers)
        return np.where(found, at, -1)

    triples = np.load(kg / "triples.npy", mmap_mode="r")
    raw = out / "edges.raw"
    runs: list[np.ndarray] = []  # (subject, first edge, edge count) per run of one subject
    unwritable = 0
    written = 0
    with raw.open("wb") as edges:
        for at in range(0, len(triples), CHUNK):
            chunk = np.asarray(triples[at : at + CHUNK], dtype=np.int64)
            s, r, o = (
                position(chunk[:, 0], qids),
                position(chunk[:, 1], pids),
                position(chunk[:, 2], qids),
            )
            ok = (s >= 0) & (o >= 0) & (r >= 0)
            ok[ok] &= (
                (rows_per_entity[s[ok]] > 0)
                & (rows_per_entity[o[ok]] > 0)
                & writable_relation[r[ok]]
            )
            unwritable += int((~ok).sum())
            s, r, o = s[ok], r[ok], o[ok]
            counts = rows_per_entity[o]
            within = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
            objects = rows_by_entity[np.repeat(row_start[o], counts) + within]
            subjects, relations_of = np.repeat(s, counts), np.repeat(r, counts)
            np.stack([subjects, relations_of, objects], axis=1).astype(np.int32).tofile(edges)
            if len(subjects):
                change = np.flatnonzero(np.diff(subjects)) + 1
                firsts = np.concatenate([[0], change])
                sizes = np.diff(np.concatenate([firsts, [len(subjects)]]))
                runs.append(np.stack([subjects[firsts], written + firsts, sizes], axis=1))
            written += len(subjects)

    # Order the edges by subject, then relation, then object row, merging the runs of
    # each subject (usually one) and dropping duplicates.
    spans = np.concatenate(runs) if runs else np.zeros((0, 3), dtype=np.int64)
    spans = spans[np.argsort(spans[:, 0], kind="stable")]
    stored = (
        np.memmap(raw, dtype=np.int32, mode="r").reshape(-1, 3)
        if written
        else np.zeros((0, 3), np.int32)
    )
    offsets = np.zeros(len(qids) + 1, dtype=np.int64)
    relations_raw, objects_raw = out / "edge_relations.raw", out / "edge_objects.raw"
    with relations_raw.open("wb") as relation_file, objects_raw.open("wb") as object_file:
        i = 0
        while i < len(spans):
            subject = spans[i, 0]
            j = i
            while j < len(spans) and spans[j, 0] == subject:
                j += 1
            block = np.concatenate([stored[a : a + n] for _, a, n in spans[i:j]])
            block = block[np.lexsort((block[:, 2], block[:, 1]))]
            keep = np.ones(len(block), dtype=bool)
            keep[1:] = np.any(block[1:, 1:] != block[:-1, 1:], axis=1)
            block = block[keep]
            np.ascontiguousarray(block[:, 1]).tofile(relation_file)
            np.ascontiguousarray(block[:, 2]).tofile(object_file)
            offsets[subject + 1] = len(block)
            i = j
    np.cumsum(offsets, out=offsets)
    del stored
    raw.unlink()

    def edge_array(path: Path) -> np.ndarray:
        if offsets[-1] == 0:
            return np.zeros(0, dtype=np.int32)
        return np.memmap(path, dtype=np.int32, mode="r")

    dates = _dates_of(kg, qids) or (None, None)
    index = KGIndex(
        labels=labels,
        relations=relations,
        relation_primary_rows=primary_relation_rows(relations, tokenizer, syntax, canonical),
        adjacency_offsets=offsets,
        edge_relations=edge_array(relations_raw),
        edge_objects=edge_array(objects_raw),
        unwritable_triples=unwritable,
        span_syntax=syntax,
        sitelinks=_sitelinks_of(kg, qids),
        date_entities=dates[0],
        date_keys=dates[1],
        verdicts=dict(VERDICTS) if verdicts else None,
        initial_case_aliases=aliases,
        source_graph_hash=_source_graph_hash(kg),
        build_config={
            "builder": "kg-directory",
            "label_encoding": "exact-v1",
            "unique_aliases": unique_aliases,
            "verdicts": verdicts,
            "initial_case_aliases": initial_case_aliases,
        },
        fingerprint_status="verified",
    )
    from xgraph.index import save_kg_index

    if (kg / "rdf-ids.json").is_file():
        from xgraph.rdf import restore_rdf_ids

        index = restore_rdf_ids(index, kg)
    assert INDEX_FORMAT == 1
    save_kg_index(index, out)  # copies the edge files into .npy files, page by page
    del index
    relations_raw.unlink()
    objects_raw.unlink()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m xgraph.wikidata", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stream = commands.add_parser("stream", help="stream the dumps into gate KG files")
    stream.add_argument("--truthy", type=Path, required=True, help="truthy N-Triples dump (.bz2)")
    stream.add_argument("--sitelinks", type=Path, required=True, help="wb_items_per_site (.sql.gz)")
    stream.add_argument("--out", type=Path, required=True, help="gate KG directory")
    stream.add_argument("--site", default="enwiki", help="sitelink site to keep items for")
    titled = commands.add_parser("titles", help="a variant KG with Wikipedia titles as labels")
    titled.add_argument("--kg", type=Path, required=True, help="gate KG directory")
    titled.add_argument("--out", type=Path, required=True, help="variant KG directory")
    labels = commands.add_parser(
        "labels", help="English labels and aliases of given items (no mul ones)"
    )
    labels.add_argument("--truthy", type=Path, required=True, help="truthy N-Triples dump (.bz2)")
    labels.add_argument("--items", type=Path, required=True, help="one QID per line")
    labels.add_argument("--out", type=Path, required=True, help="labels (JSON lines)")
    part = commands.add_parser("subset", help="a smaller KG of sampled items")
    part.add_argument("--kg", type=Path, required=True, help="gate KG directory")
    part.add_argument("--entities", type=int, required=True, help="items to keep")
    part.add_argument("--seed", type=int, default=0)
    part.add_argument("--out", type=Path, required=True, help="subset KG directory")
    index = commands.add_parser("index", help="build and save the KG index of gate KG files")
    index.add_argument("--kg", type=Path, required=True, help="gate KG directory")
    index.add_argument("--tokenizer", type=Path, required=True, help="served tokenizer.json")
    index.add_argument("--out", type=Path, required=True, help="KG index directory")
    index.add_argument(
        "--syntax", default="brackets", help="span syntax preset the index is built for"
    )
    index.add_argument(
        "--unique-aliases",
        action="store_true",
        help="drop relation aliases that several properties share (relation_names)",
    )
    index.add_argument(
        "--initial-case-aliases",
        action="store_true",
        help="add initial-case-v1 entity surfaces; preserve existing homonym resolution",
    )
    index.add_argument(
        "--verdicts",
        action="store_true",
        help="add the value comparison's verdict entities, yes and no (xgraph.index.VERDICTS)",
    )
    counted = commands.add_parser(
        "sitelink-counts", help="each item's sitelinks to every site, for the sitelink rank"
    )
    counted.add_argument("--kg", type=Path, required=True, help="KG directory to write it to")
    counted.add_argument(
        "--sitelinks", type=Path, required=True, help="wb_items_per_site (.sql.gz)"
    )
    linked = commands.add_parser(
        "index-sitelinks", help="give a saved KG index its KG's sitelink counts, no rebuild"
    )
    linked.add_argument("--kg", type=Path, required=True, help="KG directory with the counts")
    linked.add_argument("--index", type=Path, required=True, help="its saved KG index")
    args = parser.parse_args(argv)
    if args.command == "sitelink-counts":
        print(count_sitelinks(args.sitelinks, args.kg), "items counted")
        return
    if args.command == "index-sitelinks":
        index_sitelinks(args.kg, args.index)
        return
    if args.command == "subset":
        print(json.dumps(subset(args.kg, args.entities, args.seed, args.out)["counts"]))
    elif args.command == "titles":
        print(add_titles(args.kg, args.out), "titles added")
    elif args.command == "labels":
        wanted = {int(q.strip()[1:]) for q in args.items.read_text().split() if q.strip()}
        with _prefiltered(args.truthy, PREFILTER[:1]) as lines:
            print(item_labels(lines, wanted, args.out), "labels")
    elif args.command == "stream":
        manifest = build(args.truthy, args.sitelinks, args.out, args.site)
        print(json.dumps(manifest["counts"], indent=2))
    else:
        tokenizer = ServedTokenizer.from_file(args.tokenizer)
        build_index(
            args.kg,
            tokenizer,
            preset(args.syntax),
            args.out,
            args.unique_aliases,
            args.verdicts,
            initial_case_aliases=args.initial_case_aliases,
        )
        if args.initial_case_aliases:
            from xgraph.index import kg_index_fingerprint, load_kg_index

            print(
                json.dumps(
                    {
                        **load_kg_index(args.out).initial_case_aliases,
                        "index_fingerprint": kg_index_fingerprint(args.out),
                    }
                )
            )


if __name__ == "__main__":
    main()
