"""Optional RDFLib adapter; see docs/guides/rdflib.md.

RDF IRIs are KG ids. Literal value ids are JSON arrays [lexical form, language,
datatype], distinct from absolute IRIs and reversible without value normalization.
Importing this module does not import rdflib; install xgraph[rdf] to call the adapter.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from xgraph.kg import Entity, KGFormatError, KnowledgeGraph, Relation

if TYPE_CHECKING:
    from rdflib import Graph

LABEL_PREDICATES = (
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2004/02/skos/core#prefLabel",
    "http://www.w3.org/2004/02/skos/core#altLabel",
    "https://schema.org/name",
)
ALIAS_PREDICATES = (LABEL_PREDICATES[2],)
RDF_IDS = "rdf-ids.json"


def _rdflib():
    try:
        import rdflib
    except ModuleNotFoundError as exc:
        if exc.name != "rdflib":
            raise
        raise ImportError(
            "RDF support requires the optional dependency: install xgraph[rdf]"
        ) from exc
    return rdflib


def _iri(value: str) -> str:
    if not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value) or re.search(
        r'[\x00-\x20<>"{}|^`\\]', value
    ):
        raise KGFormatError(f"RDF requires an absolute IRI, got {value!r}")
    return value


def _literal_id(value) -> str:
    return json.dumps(
        [
            str(value),
            value.language.lower() if value.language else None,
            str(value.datatype) if value.datatype else None,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def from_rdflib(
    graph: Graph,
    *,
    label_predicates: Iterable[str] = LABEL_PREDICATES,
    languages: Iterable[str] | None = ("en",),
    relation_label_predicates: Iterable[str] | None = None,
    alias_predicates: Iterable[str] = ALIAS_PREDICATES,
    include_literals: bool = False,
    blank_nodes: str = "skip",
    missing_labels: str = "iri",
) -> KnowledgeGraph:
    """Project an RDF graph into entities, relations, labels and asserted triples.

    Predicate order, then language order, then lexical order determines labels;
    alias_predicates never supply the canonical label. Untagged labels follow
    the requested languages; None admits every language. Matching is exact and
    case insensitive. Label predicates are metadata, not edges. Blank-node
    triples are skipped (or rejected); owl:sameAs remains an ordinary edge.
    """
    rdf = _rdflib()
    if not isinstance(graph, rdf.Graph) or isinstance(graph, rdf.Dataset):
        raise TypeError("from_rdflib needs one rdflib.Graph; select a Dataset graph explicitly")
    if blank_nodes not in ("skip", "error"):
        raise ValueError(f"blank_nodes must be 'skip' or 'error', got {blank_nodes!r}")
    if missing_labels not in ("iri", "error"):
        raise ValueError(f"missing_labels must be 'iri' or 'error', got {missing_labels!r}")
    if isinstance(languages, str):
        raise ValueError("languages must be a sequence of language tags, not a string")
    langs = None if languages is None else tuple(str(lang).lower() for lang in languages)

    def predicates(values, option):
        if isinstance(values, str):
            raise ValueError(
                f"{option} must be an ordered iterable of predicate IRIs, not a string"
            )
        return tuple(_iri(str(p)) for p in values)

    entity_predicates = predicates(label_predicates, "label_predicates")
    relation_predicates = (
        entity_predicates
        if relation_label_predicates is None
        else predicates(relation_label_predicates, "relation_label_predicates")
    )
    aliases = predicates(alias_predicates, "alias_predicates")
    metadata = set((*entity_predicates, *relation_predicates, *aliases))
    labels: dict[str, list[tuple[str, str, str]]] = {}
    labelled: set[str] = set()
    entity_ids: set[str] = set()
    relation_ids: set[str] = set()
    triples: set[tuple[str, str, str]] = set()
    values = {}
    for subject, predicate, obj in graph:
        if isinstance(subject, rdf.BNode) or isinstance(obj, rdf.BNode):
            if blank_nodes == "error":
                raise KGFormatError(f"blank node in RDF triple {(subject, predicate, obj)!r}")
            continue
        if not isinstance(subject, rdf.URIRef) or not isinstance(predicate, rdf.URIRef):
            raise KGFormatError(
                f"RDF triple needs IRI subject and relation: {(subject, predicate)!r}"
            )
        s, p = _iri(str(subject)), _iri(str(predicate))
        if p in metadata:
            labelled.add(s)
            if not isinstance(obj, rdf.Literal):
                raise KGFormatError(f"label predicate {p} of {s} needs a literal, got {obj!r}")
            lang = (obj.language or "").lower()
            if str(obj).strip() and (langs is None or not lang or lang in langs):
                labels.setdefault(s, []).append((p, lang, str(obj)))
            continue
        if isinstance(obj, rdf.URIRef):
            o = _iri(str(obj))
        elif isinstance(obj, rdf.Literal):
            if not include_literals:
                continue
            o = _literal_id(obj)
            values[o] = obj
        else:
            raise KGFormatError(f"unsupported RDF object {obj!r} in {(subject, predicate)!r}")
        entity_ids.update((s, o))
        relation_ids.add(p)
        triples.add((s, p, o))
    # Label-only nodes are entities, except an IRI used exclusively as a relation.
    entity_ids.update(labelled - relation_ids)

    def names(item, predicates):
        primary = tuple(p for p in predicates if p not in aliases)
        priority = {p: n for n, p in enumerate((*primary, *aliases))}
        rows = [row for row in labels.get(item, ()) if row[0] in priority]
        rows.sort(
            key=lambda row: (
                priority[row[0]],
                (langs.index(row[1]) if row[1] in langs else len(langs))
                if langs is not None
                else row[1],
                row[2],
            )
        )
        canonical = next((text for p, _, text in rows if p in primary), None)
        if canonical is None:
            if missing_labels == "error":
                raise KGFormatError(f"no accepted RDF label for {item}")
            canonical = item
        return tuple(dict.fromkeys([canonical, *(text for _, _, text in rows)]))

    def value_labels(value):
        from xgraph.literals import render_date, render_decimal

        datatype = str(value.datatype)
        if datatype == "http://www.w3.org/2001/XMLSchema#dateTime":
            rendered = render_date(str(value))
        elif datatype == "http://www.w3.org/2001/XMLSchema#decimal":
            rendered = render_decimal(str(value))
        else:
            rendered = []
        return tuple(dict.fromkeys([*(rendered or [str(value)]), str(value)]))

    return KnowledgeGraph(
        entities=tuple(
            Entity(i, value_labels(values[i]) if i in values else names(i, entity_predicates))
            for i in sorted(entity_ids)
        ),
        relations=tuple(Relation(i, names(i, relation_predicates)) for i in sorted(relation_ids)),
        triples=tuple(sorted(triples)),
    )


def _term(identifier: str, role: str):
    rdf = _rdflib()
    if role == "object" and identifier.startswith("["):
        try:
            text, language, datatype = json.loads(identifier)
            if (
                not isinstance(text, str)
                or (language is not None and not isinstance(language, str))
                or (datatype is not None and not isinstance(datatype, str))
                or (language and datatype)
            ):
                raise ValueError("invalid literal fields")
            return rdf.Literal(text, lang=language, datatype=datatype, normalize=False)
        except (ValueError, TypeError) as exc:
            raise KGFormatError(f"invalid RDF literal value id {identifier!r}") from exc
    # The existing Turtle output uses Wikidata QIDs and direct-property PIDs.
    if re.fullmatch(r"P\d+" if role == "relation" else r"Q\d+", identifier):
        from xgraph.turtle import PREFIXES

        identifier = PREFIXES["wdt" if role == "relation" else "wd"] + identifier
    return rdf.URIRef(_iri(identifier))


def to_rdflib(
    triples: Iterable[tuple[str, str, str]] = (),
    *,
    spans=(),
    witnesses: Iterable[Iterable[tuple[str, str, str]]] = (),
) -> Graph:
    """Assert resolved triples and explicit path witness triples in a fresh RDF graph.

    As in xgraph.turtle, entity spans and escaped/unresolved spans emit no facts.
    Path endpoints alone assert no edge: supply their actual witness triples.
    Duplicate assertions collapse under RDF graph set semantics. Labels are
    import metadata and are not asserted by decoding results.
    """
    from itertools import chain

    from xgraph.turtle import span_triples

    graph = _rdflib().Graph()
    for subject, relation, obj in chain(
        triples, span_triples(spans), chain.from_iterable(witnesses)
    ):
        graph.add((_term(subject, "subject"), _term(relation, "relation"), _term(obj, "object")))
    return graph


def _materialize_options(options):
    """Freeze one-shot import iterables while preserving string-input errors."""
    config = dict(options)
    for key in ("label_predicates", "relation_label_predicates", "alias_predicates", "languages"):
        if (value := config.get(key)) is not None and not isinstance(value, str):
            config[key] = tuple(map(str, value))
    return config


def rdflib_edits(
    base: Graph,
    updated: Graph | None = None,
    *,
    added=(),
    removed=(),
    **options,
) -> tuple[tuple[str, str, str | None], ...]:
    """A projected RDF difference as kgversions pair replacements (None deletes).

    Supply either a complete updated graph or added/removed RDF triple sets.
    Nodes, relations and labels are frozen by the base index: additions that
    need new ids or changed labels require rebuilding it. Import options must
    match the base import. Iterable options are frozen once for both projections
    and metadata checks, so one-shot iterators work like tuples or lists.
    The supplied graphs are never mutated.
    """
    options = _materialize_options(options)
    added, removed = set(added), set(removed)
    if updated is not None and (added or removed):
        raise ValueError("supply either an updated graph or added/removed triples")
    before = from_rdflib(base, **options)
    if updated is None:
        if added & removed:
            raise KGFormatError("RDF edits both add and remove the same triple")
        missing = removed - set(base)
        if missing:
            raise KGFormatError(f"removed RDF triple is not in the base: {next(iter(missing))!r}")
        updated = _rdflib().Graph()
        for triple in base:
            if triple not in removed:
                updated.add(triple)
        for triple in added:
            updated.add(triple)
    after = from_rdflib(updated, **options)
    endpoints = {node for s, _, o in after.triples for node in (s, o)}
    old_relations = {relation.id for relation in before.relations}
    label_predicates = {
        str(p)
        for key, default in (
            ("label_predicates", LABEL_PREDICATES),
            ("relation_label_predicates", options.get("label_predicates", LABEL_PREDICATES)),
            ("alias_predicates", ALIAS_PREDICATES),
        )
        for p in (options.get(key) if options.get(key) is not None else default)
    }
    for kind in ("entities", "relations"):
        known = {e.id: e.labels for e in getattr(before, kind)}
        for item in getattr(after, kind):
            if kind == "entities" and item.id in old_relations and item.id not in endpoints:
                # With its last edge removed, a relation's label subject becomes
                # label-only in the new projection. It needs no entity row.
                subject = _rdflib().URIRef(item.id)
                annotations = [
                    {
                        (p, o)
                        for p, o in graph.predicate_objects(subject)
                        if str(p) in label_predicates
                    }
                    for graph in (base, updated)
                ]
                if annotations[0] != annotations[1]:
                    raise KGFormatError(f"rebuild the KG index for changed RDF labels: {item.id}")
                continue
            if item.id not in known:
                raise KGFormatError(f"rebuild the KG index for new RDF {kind}: {item.id}")
            if known[item.id] != item.labels:
                raise KGFormatError(f"rebuild the KG index for changed RDF labels: {item.id}")
    old, new = set(before.triples), set(after.triples)
    pairs = {(s, p) for s, p, _ in old ^ new}
    objects: dict[tuple[str, str], list[str]] = {}
    for s, p, o in sorted(new):
        if (s, p) in pairs:
            objects.setdefault((s, p), []).append(o)
    return tuple((s, p, o) for s, p in sorted(pairs) for o in objects.get((s, p), [None]))


def write_rdflib(graph: Graph, directory: str | Path, *, inputs=(), **options) -> KnowledgeGraph:
    """Write a new KG directory for xgraph.wikidata.build_index; return its RDF KG.

    Integer storage ids are local to this directory. rdf-ids.json restores RDF
    ids when the existing builder saves the index; no rdflib is needed to build
    or load that index. Refuse a nonempty destination to avoid stale arrays.
    """
    import hashlib
    from dataclasses import asdict

    import numpy as np

    from xgraph.provenance import provenance

    config = {
        "label_predicates": LABEL_PREDICATES,
        "languages": ("en",),
        "relation_label_predicates": None,
        "alias_predicates": ALIAS_PREDICATES,
        "include_literals": False,
        "blank_nodes": "skip",
        "missing_labels": "iri",
        **options,
    }
    config = _materialize_options(config)
    kg = from_rdflib(graph, **config)
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"RDF KG destination is not empty: {directory}")
    if max(len(kg.entities), len(kg.relations)) >= np.iinfo(np.int32).max:
        raise KGFormatError("RDF KG storage ids exceed int32")
    directory.mkdir(parents=True, exist_ok=True)
    entity_at = {e.id: n for n, e in enumerate(kg.entities, 1)}
    relation_at = {r.id: n for n, r in enumerate(kg.relations, 1)}
    for kind, records, positions, prefix in (
        ("entity", kg.entities, entity_at, "Q"),
        ("relation", kg.relations, relation_at, "P"),
    ):
        with (directory / f"{kind}_labels.jsonl").open("w", encoding="utf-8") as file:
            for item in records:
                for n, label in enumerate(item.labels):
                    row = [f"{prefix}{positions[item.id]}", label, "label" if n == 0 else "alias"]
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
    rows = [(entity_at[s], relation_at[p], entity_at[o]) for s, p, o in kg.triples]
    np.save(directory / "triples.npy", np.array(rows, dtype=np.int32).reshape(-1, 3))
    (directory / "sitelinks.jsonl").write_text("", encoding="utf-8")
    mapping = {
        "format": 1,
        "entities": {f"Q{n}": i for i, n in entity_at.items()},
        "relations": {f"P{n}": i for i, n in relation_at.items()},
    }
    (directory / RDF_IDS).write_text(
        json.dumps(mapping, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "artifact_kind": "xgraph.rdf_kg",
        "version": 1,
        "rdflib": _rdflib().__version__,
        "options": config,
        "counts": {
            "entities": len(kg.entities),
            "relations": len(kg.relations),
            "triples": len(rows),
        },
        "projection_sha256": hashlib.sha256(
            json.dumps(asdict(kg), sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "provenance": provenance(list(map(Path, inputs))),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return kg


def restore_rdf_ids(index, directory: str | Path):
    """Restore directory-local ids after the integer index build, without RDFLib."""
    from dataclasses import replace

    path = Path(directory) / RDF_IDS
    try:
        mapping = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise KGFormatError(f"{path}: invalid RDF id mapping JSON") from exc
    if not isinstance(mapping, dict) or mapping.get("format") != 1:
        raise KGFormatError(f"{path}: unsupported RDF id mapping format")
    parts = {}
    for part, key in (("labels", "entities"), ("relations", "relations")):
        ids = mapping.get(key)
        if (
            not isinstance(ids, dict)
            or any(not isinstance(i, str) or not i for i in ids.values())
            or len(set(ids.values())) != len(ids)
        ):
            raise KGFormatError(f"{path}: {key} must map storage ids to unique nonempty RDF ids")
        rows = getattr(index, part)
        # Verdict entities are added by the builder, outside the RDF projection.
        ids = {**{i: i for i in (index.verdicts or {}).values()}, **ids}
        missing = set(rows.owner_ids) - ids.keys()
        if missing:
            raise KGFormatError(f"{path}: missing RDF ids for {sorted(missing)[:5]}")
        parts[part] = replace(
            rows,
            owner_ids=tuple(ids[i] for i in rows.owner_ids),
            excluded=tuple(replace(e, owner_id=ids[e.owner_id]) for e in rows.excluded),
        )
    return replace(index, **parts)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m xgraph.rdf", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    importing = commands.add_parser("import", help="RDF files to a new KG directory")
    importing.add_argument("files", type=Path, nargs="+")
    importing.add_argument("--out", type=Path, required=True)
    importing.add_argument("--format", help="RDFLib parser format; otherwise inferred per file")
    importing.add_argument("--language", action="append", help="ordered tags; default en")
    importing.add_argument("--include-literals", action="store_true")
    importing.add_argument("--blank-nodes", choices=("skip", "error"), default="skip")
    importing.add_argument("--missing-labels", choices=("iri", "error"), default="iri")
    args = parser.parse_args(argv)
    graph = _rdflib().Graph()
    for path in args.files:
        if not path.is_file():
            parser.error(f"RDF input is not a local file: {path}")
        graph.parse(path, format=args.format)
    kg = write_rdflib(
        graph,
        args.out,
        inputs=args.files,
        languages=args.language or ("en",),
        include_literals=args.include_literals,
        blank_nodes=args.blank_nodes,
        missing_labels=args.missing_labels,
    )
    print(f"{len(kg.entities)} entities, {len(kg.relations)} relations, {len(kg.triples)} triples")


if __name__ == "__main__":
    main()
