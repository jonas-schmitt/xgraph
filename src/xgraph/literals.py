"""Literal values (dates, quantities) as writable KG objects.

The gate KG holds item-to-item triples only, so a fact such as a date of birth cannot be
written in a fact span; on Mintaka the constraint loses where questions need such
values (docs/post-gate-findings.md). This module adds them: every distinct date
(xsd:dateTime) and quantity (xsd:decimal) value of a kept item's truthy statements
becomes an object with rendered labels, and each statement a triple (item, property,
value object). Strings, external identifiers among them, stay out.

Renderings, canonical first: a date as "11 March 1952" with aliases "1952-03-11",
"March 11, 1952" and "1952" (truthy dumps drop the precision and store a year as its
1 January, so for 1 January the year alone comes first); a quantity as "11825551"
with "11,825,551" (truthy dumps drop units, too).

    python -m xgraph.literals --truthy data/wikidata/wikidata-20261007-truthy-BETA.nt.bz2 \\
        --kg data/wikidata/gate-kg --out data/wikidata/gate-kg-lit

**Merged dates** (`merge_dates`). A literal KG is built on one entity KG; to give another
KG directory of the same items (such as the gate KG streamed with `mul` labels) its
dates without reading the dump again, `merge-dates` joins the literal KG's date triples
onto it by QID: a triple whose object is a date value and whose subject the entity KG
holds is added with the value's number, and the value becomes an entity labelled at its
stored precision (xgraph.expressions `stored_date`; `date_labels`), so that a path span
reaching it binds an answer written as the date is known: a day as "6 June 1814" with
the alias "June 6, 1814", a month as "June 1814", a year as "1814" (BC years as "44
BC"); no ISO form and no other precision. Quantities stay out. The entity KG's labels,
triples, relation labels, and sitelinks are kept; `manifest.json` counts the date
triples merged, the date entities, and the date triples (and their distinct subjects)
the entity KG has no item for. `date_values.npy` holds each date entity's year, month,
and day, which the KG index build (`python -m xgraph.wikidata index`) turns into the
index's date table (xgraph.index `date_table`):

    python -m xgraph.literals merge-dates --from data/wikidata/gate-kg-lit \\
        --kg data/wikidata/gate-kg-v3e --out data/wikidata/gate-kg-v3e-dates
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Container, Iterable, Iterator
from pathlib import Path

import numpy as np

MONTHS = [
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
]
FIRST_ID = 1_500_000_000  # above every item number, within int32
DATE_VALUES = "date_values.npy"
"""A KG directory's merged dates (merge_dates): int64 rows (value number, year, month,
day), sorted by number, from which the KG index build makes its date table."""
CHUNK = 1 << 22  # rows of the original triples copied at a time

_LINE = re.compile(
    r"^<http://www\.wikidata\.org/entity/Q(\d+)> <http://www\.wikidata\.org/prop/direct/P(\d+)> "
    r'"([^"]*)"\^\^<http://www\.w3\.org/2001/XMLSchema#(dateTime|decimal)> \.$'
)
_DATE = re.compile(r"^([+-]?)(\d{1,})-(\d{2})-(\d{2})T")
_DECIMAL = re.compile(r"^([+-]?)(\d+)(\.\d+)?$")
PREFILTER = (
    r"^<http://www\.wikidata\.org/entity/Q[0-9]+> <http://www\.wikidata\.org/prop/direct/P[0-9]+> "
    r'"[^"]*"\^\^<http://www\.w3\.org/2001/XMLSchema#(dateTime|decimal)> \.$',
)


def render_date(value: str) -> list[str]:
    match = _DATE.match(value)
    if not match:
        return []
    sign, year, month, day = match.groups()
    y, m, d = int(year), int(month), int(day)
    if not (1 <= m <= 12 and 1 <= d <= 31) or y == 0:
        return []
    era = f"{y} BC" if sign == "-" else str(y)
    text = f"{d} {MONTHS[m - 1]} {era}"
    iso = f"{'-' if sign == '-' else ''}{y:04d}-{m:02d}-{d:02d}"
    american = f"{MONTHS[m - 1]} {d}, {era}"
    labels = [text, iso, american, era]
    if m == 1 and d == 1:  # probably a year: truthy stores a year as its 1 January
        labels = [era, text, iso, american]
    return list(dict.fromkeys(labels))


def render_decimal(value: str) -> list[str]:
    match = _DECIMAL.match(value)
    if not match:
        return []
    sign, whole, fraction = match.groups()
    sign = "-" if sign == "-" else ""
    fraction = fraction or ""
    plain = f"{sign}{whole}{fraction}"
    grouped = f"{sign}{int(whole):,}{fraction}"
    return list(dict.fromkeys([plain, grouped]))


def literal_triples(
    lines: Iterable[str], keep: Container[int]
) -> Iterator[tuple[int, int, str, str]]:
    """(item, property, "date" or "decimal", value) for kept items' literal statements."""
    for line in lines:
        match = _LINE.match(line.rstrip("\n"))
        if not match:
            continue
        item, prop, value, datatype = match.groups()
        if int(item) in keep:
            yield int(item), int(prop), "date" if datatype == "dateTime" else "decimal", value


def build(dump: Path, kg: Path, out: Path) -> dict:
    """A KG directory with the gate KG's labels and triples plus literal objects."""
    from xgraph.provenance import provenance
    from xgraph.wikidata import ItemBitmap, _prefiltered, _raw_to_npy

    kg, out = Path(kg), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    keep = ItemBitmap()
    with (kg / "entity_labels.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            keep.add(int(line[3 : line.index('"', 3)]))
    values: dict[tuple[str, str], int] = {}
    raw = out / "literal-triples.raw"
    added = 0
    with _prefiltered(dump, PREFILTER) as lines, raw.open("wb") as written:
        batch = []
        for item, prop, kind, value in literal_triples(lines, keep):
            render = render_date if kind == "date" else render_decimal
            key = (kind, value)
            if key not in values:
                if not render(value):
                    continue
                values[key] = FIRST_ID + len(values)
            batch.append((item, prop, values[key]))
            if len(batch) >= 1 << 20:
                np.array(batch, dtype=np.int32).tofile(written)
                added += len(batch)
                batch = []
        np.array(batch, dtype=np.int32).reshape(-1, 3).tofile(written)
        added += len(batch)
    with (out / "entity_labels.jsonl").open("w", encoding="utf-8") as labels:
        with (kg / "entity_labels.jsonl").open(encoding="utf-8") as source:
            for line in source:
                labels.write(line)
        for (kind, value), number in values.items():
            rendered = (render_date if kind == "date" else render_decimal)(value)
            for n, label in enumerate(rendered):
                row = [f"Q{number}", label, "label" if n == 0 else "alias"]
                labels.write(json.dumps(row, ensure_ascii=False) + "\n")
    original = np.load(kg / "triples.npy", mmap_mode="r")
    combined = out / "triples.raw"
    with combined.open("wb") as written:
        for at in range(0, len(original), CHUNK):
            np.asarray(original[at : at + CHUNK]).tofile(written)
        with raw.open("rb") as source:
            while chunk := source.read(1 << 26):
                written.write(chunk)
    _raw_to_npy(combined, out / "triples.npy", len(original) + added)
    raw.unlink()
    for name in ("relation_labels.jsonl", "sitelinks.jsonl"):
        target = out / name
        target.unlink(missing_ok=True)
        target.symlink_to((kg / name).resolve())
    manifest = {
        "artifact_kind": "xgraph.wikidata_kg_literals",
        "version": 1,
        "counts": {
            "original_triples": len(original),
            "literal_triples": added,
            "literal_values": len(values),
        },
        "provenance": provenance([Path(dump), kg / "triples.npy"]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def date_labels(date) -> list[str]:
    """A date's labels at its precision (an xgraph.expressions Date), canonical first."""
    from xgraph.expressions import Date

    labels = [str(date)]
    if date.precision == "day":
        labels.append(f"{MONTHS[date.month - 1]} {date.day}, {Date(date.year)}")
    return labels


def merge_dates(literal: Path, kg: Path, out: Path) -> dict:
    """A KG directory with the entity KG's labels and triples plus the literal KG's dates
    of its items (the module docstring, merged dates)."""
    from xgraph.expressions import date_values, stored_date
    from xgraph.provenance import provenance
    from xgraph.wikidata import SITELINK_COUNTS, _entity_items, _positions, _raw_to_npy

    literal, kg, out = Path(literal), Path(kg), Path(out)
    items = _entity_items(kg)
    if len(items) and int(items[-1]) >= FIRST_ID:
        raise ValueError(
            f"{kg} has the item Q{int(items[-1])}, at or above Q{FIRST_ID}, from where the "
            f"literal KG numbers literal values: merged dates would collide with its items"
        )
    values = date_values(literal)
    dated = np.array(sorted(values), dtype=np.int64)
    used = np.zeros(len(dated), dtype=bool)
    out.mkdir(parents=True, exist_ok=True)
    original = np.load(kg / "triples.npy", mmap_mode="r")
    source = np.load(literal / "triples.npy", mmap_mode="r")
    raw = out / "triples.raw"
    added = unmatched = 0
    unmatched_subjects: set[int] = set()
    with raw.open("wb") as written:
        for at in range(0, len(original), CHUNK):
            np.asarray(original[at : at + CHUNK], dtype=np.int32).tofile(written)
        for at in range(0, len(source), CHUNK):
            chunk = np.asarray(source[at : at + CHUNK], dtype=np.int64)
            value = _positions(dated, chunk[:, 2])
            chunk, value = chunk[value >= 0], value[value >= 0]
            held = _positions(items, chunk[:, 0]) >= 0
            unmatched += int((~held).sum())
            unmatched_subjects.update(np.unique(chunk[~held, 0]).tolist())
            used[value[held]] = True
            chunk[held].astype(np.int32).tofile(written)
            added += int(held.sum())
    _raw_to_npy(raw, out / "triples.npy", len(original) + added)
    numbers = dated[used]
    with (out / "entity_labels.jsonl").open("w", encoding="utf-8") as labels:
        with (kg / "entity_labels.jsonl").open(encoding="utf-8") as entity_labels:
            for line in entity_labels:
                labels.write(line)
        for number in numbers.tolist():
            for n, label in enumerate(date_labels(stored_date(*values[number]))):
                row = [f"Q{number}", label, "label" if n == 0 else "alias"]
                labels.write(json.dumps(row, ensure_ascii=False) + "\n")
    rows = [(n, *values[n]) for n in numbers.tolist()]
    np.save(out / DATE_VALUES, np.array(rows, dtype=np.int64).reshape(-1, 4))
    for name in ("relation_labels.jsonl", "sitelinks.jsonl", SITELINK_COUNTS):
        target = out / name
        target.unlink(missing_ok=True)
        if (kg / name).exists():
            target.symlink_to((kg / name).resolve())
    manifest = {
        "artifact_kind": "xgraph.wikidata_kg_dates",
        "version": 1,
        "counts": {
            "entity_triples": len(original),
            "date_triples": added,
            "date_entities": int(used.sum()),
            "unmatched_date_triples": unmatched,
            "unmatched_subjects": len(unmatched_subjects),
        },
        "provenance": provenance(
            [literal / "entity_labels.jsonl", literal / "triples.npy", kg / "triples.npy"]
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["merge-dates"]:
        merging = argparse.ArgumentParser(
            prog="python -m xgraph.literals merge-dates",
            description="join a literal KG's dates onto a KG directory of the same items",
        )
        merging.add_argument(
            "--from", dest="literal", type=Path, required=True, help="KG directory with literals"
        )
        merging.add_argument("--kg", type=Path, required=True, help="entity KG directory")
        merging.add_argument("--out", type=Path, required=True, help="KG directory with dates")
        args = merging.parse_args(argv[1:])
        print(json.dumps(merge_dates(args.literal, args.kg, args.out)["counts"], indent=1))
        return
    parser = argparse.ArgumentParser(prog="python -m xgraph.literals", description=__doc__)
    parser.add_argument("--truthy", type=Path, required=True, help="truthy N-Triples dump (.bz2)")
    parser.add_argument("--kg", type=Path, required=True, help="gate KG directory")
    parser.add_argument("--out", type=Path, required=True, help="KG directory with literals")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.truthy, args.kg, args.out)["counts"], indent=1))


if __name__ == "__main__":
    main()
