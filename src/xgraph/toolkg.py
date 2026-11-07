"""The KG lookup tool baseline: the model queries the gate KG through function calls.

The first alternative to a decoding constraint is to let the model look facts up. Two
tools over the same KG, in Qwen's function-calling format:

- `search_entities(name)`: items whose English label or alias equals the name (NFC,
  case folded), the five with the most triples first;
- `get_facts(entity, relation?)`: the entity's triples as `subject · relation · object`
  lines with canonical labels, outgoing then incoming, optionally only one relation,
  at most 40.

The loop generates, runs the calls in the reply, appends their results as tool
messages, and generates again, up to `max_calls` calls per question; then it asks for
the final answer. Every round's prompt counts towards the cost (each call is another
prefill), so records keep prompt and generated tokens summed over rounds. A request
naming a KG version (`kg_version`) looks the KG up through it (VersionedStore, whose
tools answer as the store of the edited KG would), as the walk does.

The KG lives in an SQLite store on disk (`build_store`), so lookups need little memory.

    python -m xgraph.toolkg build --kg data/wikidata/gate-kg --db data/wikidata/gate-kg.sqlite
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections.abc import Iterator
from heapq import merge
from pathlib import Path
from typing import Any

import numpy as np

from xgraph.score import normalize

CHUNK = 1 << 20
SEARCH_LIMIT = 5
FACT_LIMIT = 40

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_entities",
            "description": "Find Wikidata items by their English name or alias.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "the name to look up"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_facts",
            "description": "List facts about a Wikidata item, optionally only one relation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "a Wikidata id such as Q42"},
                    "relation": {"type": "string", "description": "a property label, optional"},
                },
                "required": ["entity"],
            },
        },
    },
]

FINAL = 'Give your final answer now, ending with the line "Answer: ⟦ label⟧".'


def build_store(kg: str | Path, db: str | Path) -> None:
    kg, db = Path(kg), Path(db)
    db.unlink(missing_ok=True)
    con = sqlite3.connect(db)
    con.executescript(
        "CREATE TABLE names (norm TEXT, item INTEGER);"
        "CREATE TABLE labels (item INTEGER PRIMARY KEY, label TEXT);"
        "CREATE TABLE relations (prop INTEGER PRIMARY KEY, label TEXT);"
        "CREATE TABLE triples (s INTEGER, p INTEGER, o INTEGER);"
    )
    with (kg / "entity_labels.jsonl").open(encoding="utf-8") as lines:
        batch, canonical = [], []
        for line in lines:
            item, label, kind = json.loads(line)
            batch.append((normalize(label), int(item[1:])))
            if kind == "label":
                canonical.append((int(item[1:]), label))
            if len(batch) >= CHUNK:
                con.executemany("INSERT INTO names VALUES (?, ?)", batch)
                con.executemany("INSERT OR IGNORE INTO labels VALUES (?, ?)", canonical)
                batch, canonical = [], []
        con.executemany("INSERT INTO names VALUES (?, ?)", batch)
        con.executemany("INSERT OR IGNORE INTO labels VALUES (?, ?)", canonical)
    with (kg / "relation_labels.jsonl").open(encoding="utf-8") as lines:
        rows = [(int(p[1:]), label) for p, label, kind in map(json.loads, lines) if kind == "label"]
        con.executemany("INSERT OR IGNORE INTO relations VALUES (?, ?)", rows)
    triples = np.load(kg / "triples.npy", mmap_mode="r")
    for at in range(0, len(triples), CHUNK):
        chunk = np.asarray(triples[at : at + CHUNK]).tolist()
        con.executemany("INSERT INTO triples VALUES (?, ?, ?)", chunk)
    con.executescript(
        "CREATE INDEX names_norm ON names (norm);"
        "CREATE INDEX triples_spo ON triples (s, p, o);"
        "CREATE INDEX triples_ops ON triples (o, p, s);"
    )
    con.commit()
    con.close()


class KGStore:
    def __init__(self, db: str | Path) -> None:
        self.db = Path(db)
        self._con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, check_same_thread=False)

    def component(self, first: int, stop: int) -> KGStore:
        """A separate read connection restricted to entity numbers [first, stop).

        Temporary views also scope callers that read SQL directly (iterative retrieval).
        Both ends of every edge must belong to the component. The relation vocabulary
        stays shared, as it does in the tokenizer-bound index. No main table is changed.
        """
        if type(first) is not int or type(stop) is not int or not 0 <= first < stop:
            raise ValueError(f"invalid KG component range: {first!r}, {stop!r}")
        read = KGStore(self.db)
        read._con.execute("PRAGMA cache_size = -64")
        read._con.executescript(
            "CREATE TEMP VIEW names AS SELECT * FROM main.names "
            f"WHERE item >= {first} AND item < {stop};"
            "CREATE TEMP VIEW labels AS SELECT * FROM main.labels "
            f"WHERE item >= {first} AND item < {stop};"
            "CREATE TEMP VIEW triples AS SELECT * FROM main.triples "
            f"WHERE s >= {first} AND s < {stop} AND o >= {first} AND o < {stop};"
        )
        return read

    def label(self, item: int) -> str | None:
        row = self._con.execute("SELECT label FROM labels WHERE item = ?", (item,)).fetchone()
        return row[0] if row else None

    def degree(self, item: int) -> int:
        out = self._con.execute("SELECT count(*) FROM triples WHERE s = ?", (item,)).fetchone()
        into = self._con.execute("SELECT count(*) FROM triples WHERE o = ?", (item,)).fetchone()
        return out[0] + into[0]

    def search_entities(self, name: str) -> list[dict]:
        return _search_entities(self, name)

    def get_facts(self, entity: str, relation: str | None = None) -> list[str]:
        return _get_facts(self, entity, relation)

    def relation_labels(self) -> dict[int, str]:
        """Every property's canonical label."""
        return dict(self._con.execute("SELECT prop, label FROM relations").fetchall())

    def fact_rows(
        self, item: int, column: str, wanted: list[int] | None, limit: int
    ) -> list[tuple[int, int, int]]:
        """The first limit triples (subject, property, object) with the item as subject
        (column "s") or object ("o"), only the wanted properties unless None, ordered by
        property, subject, and object."""
        where = f"{column} = ?" + (f" AND p IN ({','.join('?' * len(wanted))})" if wanted else "")
        # Limited in SQL: a hub such as a country is the object of millions of triples.
        rows = self._con.execute(
            f"SELECT s, p, o FROM triples WHERE {where} ORDER BY p, s, o LIMIT ?",
            (item, *(wanted or ()), limit),
        )
        return [tuple(row) for row in rows.fetchall()]

    def items_named(self, name: str) -> list[int]:
        """Every item whose label or alias equals the name (NFC, case folded)."""
        rows = self._con.execute(
            "SELECT DISTINCT item FROM names WHERE norm = ?", (normalize(name),)
        )
        return sorted(r[0] for r in rows.fetchall())

    def relation_named(self, name: str) -> list[int]:
        """Every property whose canonical label equals the name (NFC, case folded)."""
        props = self._con.execute("SELECT prop, label FROM relations").fetchall()
        return sorted(p for p, label in props if normalize(label) == normalize(name))

    def relation_label(self, prop: int) -> str | None:
        row = self._con.execute("SELECT label FROM relations WHERE prop = ?", (prop,)).fetchone()
        return row[0] if row else None

    def outgoing_edges(self, item: int, *, by_object: bool = False) -> Iterator[tuple]:
        """Stream distinct labelled edges (relation, object, relation label, object label).

        Sorting stays in SQLite rather than materializing a hub's adjacency in Python.
        Object order also supports streaming parallel-relation collapse.
        """
        order = "t.o, t.p" if by_object else "t.p, t.o"
        yield from self._con.execute(
            "SELECT DISTINCT t.p, t.o, r.label, l.label FROM triples t "
            "JOIN relations r ON r.prop = t.p JOIN labels l ON l.item = t.o "
            f"WHERE t.s = ? ORDER BY {order}",
            (item,),
        )

    def relations_of(self, items: list[int], *, reverse: bool = False) -> dict[int, str]:
        """Labelled relations of outgoing triples, or incoming triples with reverse."""
        found: dict[int, str] = {}
        for item in items:
            rows = self._con.execute(
                "SELECT DISTINCT t.p, r.label FROM triples t JOIN relations r ON r.prop = t.p "
                f"WHERE t.{'o' if reverse else 's'} = ?",
                (item,),
            ).fetchall()
            found.update(rows)
        return found

    def neighbours_of(self, items: list[int], prop: int, *, reverse: bool = False) -> list[int]:
        """Distinct adjacent entity ids, including ids without labels, in QID order."""
        lookup = self.subjects_of if reverse else self.object_rows
        return sorted({other for item in items for other in lookup(item, prop)})

    def objects_of(self, items: list[int], prop: int) -> dict[int, str]:
        """The labelled objects of the items' triples with the property: an object
        segment's candidates."""
        found: dict[int, str] = {}
        for item in items:
            rows = self._con.execute(
                "SELECT DISTINCT t.o, l.label FROM triples t JOIN labels l ON l.item = t.o "
                "WHERE t.s = ? AND t.p = ?",
                (item, prop),
            ).fetchall()
            found.update(rows)
        return found

    def used_relations(self) -> set[int]:
        """Every property of a triple."""
        return {r[0] for r in self._con.execute("SELECT DISTINCT p FROM triples").fetchall()}

    def lowest_objects(self, prop: int) -> dict[int, int]:
        """Each subject's lowest labelled object under the property, over the whole KG."""
        rows = self._con.execute(
            "SELECT t.s, min(t.o) FROM triples t JOIN labels l ON l.item = t.o "
            "WHERE t.p = ? GROUP BY t.s",
            (prop,),
        )
        return dict(rows.fetchall())

    def object_counts(self) -> Iterator[tuple[int, int, int]]:
        """(property, object, triples) for every labelled object of every property: how
        many triples of the property reach the object, over the whole KG."""
        yield from self._con.execute(
            "SELECT t.p, t.o, count(*) FROM triples t JOIN labels l ON l.item = t.o "
            "GROUP BY t.o, t.p"
        )

    def object_rows(self, item: int, prop: int) -> list[int]:
        """The object of every triple of the item with the property, labelled or not, one
        per triple."""
        rows = self._con.execute("SELECT o FROM triples WHERE s = ? AND p = ?", (item, prop))
        return [r[0] for r in rows.fetchall()]

    def subjects_of(self, item: int, prop: int) -> list[int]:
        """The subjects of the triples with the property and the item as object."""
        rows = self._con.execute(
            "SELECT DISTINCT s FROM triples WHERE o = ? AND p = ?", (item, prop)
        ).fetchall()
        return sorted(r[0] for r in rows)

    def iter_fact_rows(self, item: int, column: str) -> Iterator[tuple[int, int, int]]:
        """Stream a complete adjacency in fact_rows order without a raw-row cap."""
        if column not in ("s", "o"):
            raise ValueError("fact column must be s or o")
        yield from self._con.execute(
            f"SELECT s, p, o FROM triples WHERE {column} = ? ORDER BY p, s, o", (item,)
        )

    def pairs(self, prop: int, limit: int) -> list[tuple[int, int]]:
        """Up to limit (subject, object) pairs of the property's triples."""
        rows = self._con.execute(
            "SELECT s, o FROM triples WHERE p = ? LIMIT ?", (prop, limit)
        ).fetchall()
        return [(s, o) for s, o in rows]

    def call(self, name: str, arguments: dict) -> Any:
        return _call(self, name, arguments)


def _search_entities(read, name: str) -> list[dict]:
    """The tool search_entities on a store, or a store read through a KG version."""
    ranked = sorted(read.items_named(name), key=lambda i: (-read.degree(i), i))[:SEARCH_LIMIT]
    return [{"id": f"Q{i}", "label": read.label(i) or ""} for i in ranked]


def _get_facts(read, entity: str, relation: str | None) -> list[str]:
    """The tool get_facts on a store, or a store read through a KG version."""
    if not re.fullmatch(r"Q\d+", entity.strip()):
        return []
    item = int(entity.strip()[1:])
    props = read.relation_labels()
    wanted = None
    if relation:
        wanted = [p for p, label in props.items() if normalize(label) == normalize(relation)]
        if not wanted:
            return []
    lines: list[str] = []
    for column in ("s", "o"):
        for s, p, o in read.fact_rows(item, column, wanted, FACT_LIMIT * 2):
            labels = (read.label(s), props.get(p), read.label(o))
            if all(labels) and len(lines) < FACT_LIMIT:
                lines.append(" · ".join(labels))
    return lines


def _call(read, name: str, arguments: dict) -> Any:
    if name == "search_entities":
        return read.search_entities(str(arguments.get("name", "")))
    if name == "get_facts":
        return read.get_facts(str(arguments.get("entity", "")), arguments.get("relation"))
    return {"error": f"no tool named {name}"}


class VersionedStore:
    """A KG store read through one KG version (xgraph.kgversions): every object of an
    edited (subject, property) pair is replaced by the version's objects, as
    xgraph.mquake.apply_edits replaces them in the triples, so that the reads below give
    what the store of the edited KG would. Versions add no labels: names and labels are
    the base store's, and an object without a label is no candidate, as in the store.
    Degrees count the edited triples, duplicates included, as the store's rows do. Only
    the reads gwalk.KGLookup, the walk (xgraph.walk), parse-then-execute (xgraph.programs),
    the lookup tools, and the KG agent make are offered: items_named, relation_named,
    relations_of, objects_of, degree, label, relation_label, relation_labels, fact_rows,
    neighbours_of (raw ids, either direction), and the tools themselves
    (search_entities, get_facts, call). relations_of also supports incoming triples.
    """

    def __init__(self, store: KGStore, edits: list[tuple[int, int, int]] | list[list[int]]):
        self.store = store
        self.edited: dict[tuple[int, int], list[int]] = {}
        for s, p, o in edits:
            self.edited.setdefault((int(s), int(p)), []).append(int(o))
        self._props: dict[int, list[int]] = {}
        for s, p in self.edited:
            self._props.setdefault(s, []).append(p)
        self._delta: dict[int, int] = {}
        for (s, p), objects in self.edited.items():
            old = store.object_rows(s, p)
            self._delta[s] = self._delta.get(s, 0) + len(objects) - len(old)
            for o in old:
                self._delta[o] = self._delta.get(o, 0) - 1
            for o in objects:
                self._delta[o] = self._delta.get(o, 0) + 1

    def label(self, item: int) -> str | None:
        return self.store.label(item)

    def items_named(self, name: str) -> list[int]:
        return self.store.items_named(name)

    def relation_named(self, name: str) -> list[int]:
        return self.store.relation_named(name)

    def relation_label(self, prop: int) -> str | None:
        return self.store.relation_label(prop)

    def relation_labels(self) -> dict[int, str]:
        return self.store.relation_labels()

    def outgoing_edges(self, item: int, *, by_object: bool = False) -> Iterator[tuple]:
        """Merge streamed base edges with replacement edits, without caching adjacency."""
        base = (
            edge for edge in self.store.outgoing_edges(item, by_object=by_object)
            if (item, edge[0]) not in self.edited
        )  # fmt: skip
        added = {
            (p, o, relation, label)
            for p in self._props.get(item, ())
            if (relation := self.relation_label(p)) is not None
            for o in self.edited[(item, p)]
            if (label := self.label(o)) is not None
        }
        key = (lambda edge: (edge[1], edge[0])) if by_object else (lambda edge: edge[:2])
        yield from merge(base, sorted(added, key=key), key=key)

    def degree(self, item: int) -> int:
        return self.store.degree(item) + self._delta.get(item, 0)

    def fact_rows(
        self, item: int, column: str, wanted: list[int] | None, limit: int
    ) -> list[tuple[int, int, int]]:
        """KGStore.fact_rows of the edited triples: the store's rows without those of the
        edited pairs, and the version's, in the same order."""
        if column == "s":
            pairs = [(item, p) for p in self._props.get(item, ())]
        else:
            pairs = [pair for pair, objects in self.edited.items() if item in objects]
            pairs += [(s, p) for s, p in self.edited if item in self.store.object_rows(s, p)]
        removed = sum(len(self.store.object_rows(s, p)) for s, p in set(pairs))
        rows = [
            row
            for row in self.store.fact_rows(item, column, wanted, limit + removed)
            if (row[0], row[1]) not in self.edited
        ]
        for s, p in dict.fromkeys(pairs):
            if wanted is None or p in wanted:
                rows += [(s, p, o) for o in self.edited[(s, p)] if column == "s" or o == item]
        return sorted(rows, key=lambda row: (row[1], row[0], row[2]))[:limit]

    def iter_fact_rows(self, item: int, column: str) -> Iterator[tuple[int, int, int]]:
        if column not in ("s", "o"):
            raise ValueError("fact column must be s or o")
        base = (
            row
            for row in self.store.iter_fact_rows(item, column)
            if (row[0], row[1]) not in self.edited
        )
        changed = [
            (s, p, o)
            for (s, p), objects in self.edited.items()
            for o in objects
            if (s if column == "s" else o) == item
        ]

        def key(row):
            return row[1], row[0], row[2]

        yield from merge(base, sorted(changed, key=key), key=key)

    def search_entities(self, name: str) -> list[dict]:
        return _search_entities(self, name)

    def get_facts(self, entity: str, relation: str | None = None) -> list[str]:
        return _get_facts(self, entity, relation)

    def call(self, name: str, arguments: dict) -> Any:
        return _call(self, name, arguments)

    def relations_of(self, items: list[int], *, reverse: bool = False) -> dict[int, str]:
        """The store's, and every labelled property the version gives one of the items
        (an edited pair keeps a triple, labelled object or not, as in the store)."""
        found = self.store.relations_of(items, reverse=reverse)
        if reverse:
            targets = set(items)
            for (_, prop), objects in self.edited.items():
                label = self.relation_label(prop)
                if targets.intersection(objects) and label is not None:
                    found[prop] = label
            return {
                p: label for p, label in found.items() if self.neighbours_of(items, p, reverse=True)
            }
        for item in items:
            for p in self._props.get(item, ()):
                label = self.store.relation_label(p)
                if p not in found and label is not None:
                    found[p] = label
        return found

    def neighbours_of(self, items: list[int], prop: int, *, reverse: bool = False) -> list[int]:
        """Raw neighbours after replacement edits, in either traversal direction."""
        if reverse:
            targets = set(items)
            found = {
                s
                for s in self.store.neighbours_of(items, prop, reverse=True)
                if (s, prop) not in self.edited
            }
            found.update(
                s
                for (s, p), objects in self.edited.items()
                if p == prop and targets.intersection(objects)
            )
        else:
            found = set()
            for item in items:
                found.update(
                    self.edited[(item, prop)]
                    if (item, prop) in self.edited
                    else self.store.neighbours_of([item], prop)
                )
        return sorted(found)

    def objects_of(self, items: list[int], prop: int) -> dict[int, str]:
        found: dict[int, str] = {}
        for item in items:
            if (item, prop) in self.edited:
                for obj in self.edited[(item, prop)]:
                    label = self.store.label(obj)
                    if label is not None:
                        found[obj] = label
            else:
                found.update(self.store.objects_of([item], prop))
        return found


def version_stores(
    store: KGStore, directory: str | Path, names: list[str] | set[str]
) -> dict[str, VersionedStore]:
    """The store read through each named KG version of a version file (xgraph.kgversions)."""
    from xgraph.kgversions import component_ranges, read_kg_versions

    versions = read_kg_versions(directory)
    components = component_ranges(directory)
    absent = sorted(set(names) - set(versions))
    if absent:
        raise ValueError(f"{directory} has no KG version {', '.join(absent)}")
    return {
        name: VersionedStore(
            store.component(*components[name]) if name in components else store,
            [(int(s[1:]), int(p[1:]), int(o[1:])) for s, p, o in versions[name]],
        )
        for name in sorted(set(names))
    }


_CALL = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAMETER = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_calls(text: str) -> tuple[str, list[tuple[str, dict]]]:
    """The text before the first call, and the calls (name, arguments) in the reply."""
    calls = [
        (name, {k: v.strip() for k, v in _PARAMETER.findall(body)})
        for name, body in _CALL.findall(text)
    ]
    before = text.split("<tool_call>", 1)[0].strip() if calls else text
    return before, calls


def store_for(request: dict, store: KGStore, versions: dict | None) -> KGStore | VersionedStore:
    """The store a request reads: the store through its kg_version, versions[name]
    (version_stores), or the store itself."""
    name = request.get("kg_version")
    if name is None:
        return store
    if name not in (versions or {}):
        raise ValueError(f"request {request['id']!r} reads KG version {name!r}, not given")
    return versions[name]


def run_with_tools(
    llm: Any,
    requests: list[dict],
    store: KGStore,
    sampling: Any,
    max_calls: int = 4,
    versions: dict | None = None,
) -> list[dict]:
    """Generate with the tools for every request, all requests a round at a time. A
    request with a kg_version looks the KG up through that KG version, versions[name]
    (version_stores)."""
    states = [
        {
            "id": r["id"],
            "read": store_for(r, store, versions),
            "kg_version": r.get("kg_version"),
            "messages": list(r["messages"]),
            "calls": 0,
            "rounds": 0,
            "prompt": 0,
            "generated": 0,
            "done": None,
        }
        for r in requests
    ]
    while pending := [s for s in states if s["done"] is None]:
        outputs = llm.chat(
            [s["messages"] for s in pending],
            sampling,
            tools=TOOLS,
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=False,
        )
        for state, output in zip(pending, outputs, strict=True):
            completion = output.outputs[0]
            state["rounds"] += 1
            state["prompt"] += len(output.prompt_token_ids)
            state["generated"] += len(completion.token_ids)
            before, calls = parse_calls(completion.text)
            if not calls:
                state["done"] = completion.text
                state["finish_reason"] = completion.finish_reason
                continue
            if state["calls"] + len(calls) > max_calls:
                state["messages"].append({"role": "assistant", "content": before})
                state["messages"].append({"role": "user", "content": FINAL})
                state["calls"] = max_calls + 1  # no more tools: the next reply is final
                continue
            state["messages"].append(
                {
                    "role": "assistant",
                    "content": before,
                    "tool_calls": [
                        {"type": "function", "function": {"name": n, "arguments": a}}
                        for n, a in calls
                    ],
                }
            )
            for name, arguments in calls:
                result = json.dumps(state["read"].call(name, arguments), ensure_ascii=False)
                state["messages"].append({"role": "tool", "content": result})
            state["calls"] += len(calls)
    return [
        {
            "id": s["id"],
            "arm": "tools",
            **({"kg_version": s["kg_version"]} if s["kg_version"] else {}),
            "text": s["done"],
            "finish_reason": s.get("finish_reason"),
            "tool_calls": min(s["calls"], max_calls),
            "rounds": s["rounds"],
            "prompt_tokens": s["prompt"],
            "generated_tokens": s["generated"],
        }
        for s in states
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m xgraph.toolkg", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="the SQLite store of a KG directory")
    build.add_argument("--kg", type=Path, required=True, help="gate KG directory")
    build.add_argument("--db", type=Path, required=True, help="store (SQLite)")
    args = parser.parse_args(argv)
    build_store(args.kg, args.db)
    print(f"wrote {args.db}")


if __name__ == "__main__":
    main()
