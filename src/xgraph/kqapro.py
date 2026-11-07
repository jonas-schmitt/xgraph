"""KQA Pro's native KG and complete-program coverage over paths and expressions.

Read only the release's JSON. ``kg`` writes entity triples plus lossless attribute
and qualifier sidecars; ``profile`` translates complete KoPL dependency graphs,
executes supported shapes with the existing CPU executors, separately measures
chains with concept filters ignored, and retains unsupported questions;
``requests`` writes every question in the common path-request format;
``score`` applies native answer equality to generation records. See
docs/reproducing.md (KQA Pro) for commands and interpretation boundaries.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from xgraph.score import record_text

VALUE_TYPES = {"string", "quantity", "year", "date"}


def _json(path: Path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _questions(path: Path) -> list[dict]:
    rows = _json(path)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: questions must be a JSON array")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path} row {index}: question must be an object")
        _text(row.get("question"), f"{path} row {index} question")
        if "answer" in row and not isinstance(row["answer"], str):
            raise ValueError(f"{path} row {index}: answer must be text")
    return rows


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _line(file, value) -> None:
    file.write(json.dumps(value, ensure_ascii=False) + "\n")


def _number(item: str) -> int:
    from xgraph.literals import FIRST_ID

    if not isinstance(item, str) or not re.fullmatch(r"Q[1-9]\d*", item):
        raise ValueError(f"invalid native entity id: {item!r}")
    number = int(item[1:])
    if number >= FIRST_ID:
        raise ValueError(f"{item}: native id overlaps the literal-value id range")
    return number


def _time(value: dict) -> tuple[int, int | None, int | None]:
    if value["type"] == "year":
        return value["value"], None, None
    found = re.fullmatch(r"(\d+)/(\d{1,2})/(\d{1,2})", str(value["value"]))
    if found is None:
        raise ValueError(f"not a native KQA date: {value!r}")
    parsed = date(*map(int, found.groups()))
    return parsed.year, parsed.month, parsed.day


def _value(value: dict, context: str) -> None:
    if not isinstance(value, dict) or value.get("type") not in VALUE_TYPES or "value" not in value:
        raise ValueError(f"{context}: invalid typed value: {value!r}")
    kind, content = value["type"], value["value"]
    if kind == "string" and not isinstance(content, str):
        raise ValueError(f"{context}: string value is not text")
    if kind == "year" and type(content) is not int:
        raise ValueError(f"{context}: year value is not an integer")
    if kind == "quantity" and (
        type(content) not in (int, float)
        or (type(content) is float and not math.isfinite(content))
        or not isinstance(value.get("unit"), str)
    ):
        raise ValueError(f"{context}: quantity needs a finite number and a unit")
    if kind == "date":
        try:
            _time(value)
        except ValueError as error:
            raise ValueError(f"{context}: {error}") from error


def _text(value, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: expected nonempty text, got {value!r}")
    return value


def convert_kg(source: str | Path, out: str | Path) -> dict:
    """Read only kb.json; keep native QIDs, map relation names deterministically.

    Relation directions are distinct directed edge types. The only added reciprocal
    edges are those on concepts, as in the native rule executor. Literal values and
    qualifiers stay in sidecars, retaining their native types, units and fact ids.
    """
    from xgraph.provenance import provenance

    source, out = Path(source), Path(out)
    kb = _json(source)
    if not isinstance(kb, dict) or set(kb) != {"concepts", "entities"}:
        raise ValueError(f"{source}: expected concepts and entities maps")
    concepts, entities = kb["concepts"], kb["entities"]
    if (
        not isinstance(concepts, dict)
        or not isinstance(entities, dict)
        or concepts.keys() & entities.keys()
    ):
        raise ValueError(f"{source}: entity and concept maps must have disjoint ids")
    nodes = {**concepts, **entities}
    for item in nodes:
        _number(item)
    relation_keys = {("instance", "instance of", "forward"), ("subclass", "subclass of", "forward")}
    attribute_types: dict[str, Counter] = {}
    for item, node in nodes.items():
        _text(node.get("name"), f"{item} name")
        aliases = node.get("aliases", [])
        if not isinstance(aliases, list) or any(
            not isinstance(a, str) or not a.strip() for a in aliases
        ):
            raise ValueError(f"{item}: aliases must be nonempty strings")
        if not isinstance(node.get("instanceOf"), list):
            raise ValueError(f"{item}: missing instanceOf list")
        for parent in node["instanceOf"]:
            if parent not in concepts:
                raise ValueError(f"{item}: unknown concept {parent!r}")
        if item not in entities:
            continue
        for field in ("attributes", "relations"):
            if not isinstance(node.get(field), list):
                raise ValueError(f"{item}: missing {field} list")
            for position, fact in enumerate(node[field]):
                context = f"{item}/{field}/{position}"
                if field == "attributes":
                    name = _text(fact.get("key"), context)
                    _value(fact.get("value"), context)
                    attribute_types.setdefault(name, Counter())[fact["value"]["type"]] += 1
                    relation_keys.add(("attribute", name, "forward"))
                else:
                    name = _text(fact.get("predicate"), context)
                    if fact.get("direction") not in ("forward", "backward"):
                        raise ValueError(f"{context}: invalid direction {fact.get('direction')!r}")
                    if fact.get("object") not in nodes:
                        raise ValueError(f"{context}: unknown object {fact.get('object')!r}")
                    relation_keys.update(("relation", name, d) for d in ("forward", "backward"))
                if not isinstance(fact.get("qualifiers"), dict):
                    raise ValueError(f"{context}: missing qualifiers map")
                for name, values in fact["qualifiers"].items():
                    _text(name, context)
                    if not isinstance(values, list):
                        raise ValueError(f"{context}: qualifier {name!r} is not a list")
                    for value in values:
                        _value(value, f"{context} qualifier {name!r}")
    relations = []
    for number, (kind, name, direction) in enumerate(sorted(relation_keys), 1):
        label = f"attribute: {name}" if kind == "attribute" else name
        if direction == "backward":
            label = f"inverse of {name}"
        relations.append(
            {"id": f"P{number}", "kind": kind, "name": name, "direction": direction, "label": label}
        )
    # Executor unions relations with the same normalized label: never conflate kinds.
    from xgraph.score import normalize

    labels = [normalize(r["label"]) for r in relations]
    if len(set(labels)) != len(labels):
        raise ValueError("native relation names collide with the bridge's relation labels")
    props = {(r["kind"], r["name"], r["direction"]): int(r["id"][1:]) for r in relations}
    triples = []
    counts: Counter = Counter()
    attribute_values: Counter = Counter()
    qualifier_values: Counter = Counter()
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").unlink(missing_ok=True)
    with (
        (out / "entity_labels.jsonl").open("w", encoding="utf-8") as named,
        (out / "nodes.jsonl").open("w", encoding="utf-8") as saved_nodes,
        (out / "facts.jsonl").open("w", encoding="utf-8") as facts,
        (out / "attributes.jsonl").open("w", encoding="utf-8") as attributes,
        (out / "qualifiers.jsonl").open("w", encoding="utf-8") as qualifiers,
    ):
        for item in sorted(nodes, key=_number):
            node = nodes[item]
            canonical = " ".join(node["name"].split())
            names = list(dict.fromkeys([canonical, node["name"], *node.get("aliases", [])]))
            for index, label in enumerate(names):
                _line(named, [item, label, "alias" if index else "label"])
                counts["aliases"] += bool(index)
            kind = "entity" if item in entities else "concept"
            _line(
                saved_nodes,
                {"id": item, "kind": kind, "name": canonical, "instanceOf": node["instanceOf"]},
            )
            prop = (
                props[("instance", "instance of", "forward")]
                if kind == "entity"
                else props[("subclass", "subclass of", "forward")]
            )
            for parent in node["instanceOf"]:
                triples.append((_number(item), prop, _number(parent)))
                counts["class_edges"] += 1
            if kind == "concept":
                continue
            for field in ("attributes", "relations"):
                for position, fact in enumerate(node[field]):
                    fact_id = f"{item}/{field}/{position}"
                    common = {"fact_id": fact_id, "subject": item}
                    if field == "attributes":
                        prop = props[("attribute", fact["key"], "forward")]
                        _line(
                            attributes,
                            {
                                **common,
                                "key": fact["key"],
                                "relation": f"P{prop}",
                                "value": fact["value"],
                            },
                        )
                        attribute_values[fact["value"]["type"]] += 1
                    else:
                        prop = props[("relation", fact["predicate"], fact["direction"])]
                        _line(
                            facts,
                            {
                                **common,
                                "relation": f"P{prop}",
                                **{k: v for k, v in fact.items() if k != "qualifiers"},
                            },
                        )
                        triples.append((_number(item), prop, _number(fact["object"])))
                        counts["native_relation_facts"] += 1
                        if fact["object"] in concepts:
                            reverse = "backward" if fact["direction"] == "forward" else "forward"
                            inverse = props[("relation", fact["predicate"], reverse)]
                            triples.append((_number(fact["object"]), inverse, _number(item)))
                            counts["concept_reciprocal_edges"] += 1
                    for key, values in fact["qualifiers"].items():
                        for value in values:
                            _line(qualifiers, {"fact_id": fact_id, "key": key, "value": value})
                            qualifier_values[value["type"]] += 1
    array = np.asarray(triples, dtype=np.int32).reshape(-1, 3)
    unique = np.unique(array, axis=0)
    np.save(out / "triples.npy", unique)
    with (out / "relation_labels.jsonl").open("w", encoding="utf-8") as named:
        for relation in relations:
            _line(named, [relation["id"], relation["label"], "label"])
    (out / "sitelinks.jsonl").write_text("", encoding="utf-8")
    _write_json(
        out / "id-mappings.json",
        {
            "entities": {i: i for i in sorted(nodes, key=_number)},
            "relations": relations,
            "attribute_types": attribute_types,
        },
    )
    manifest = {
        "artifact_kind": "xgraph.kqapro_kg",
        "version": 1,
        "counts": {
            "entities": len(entities),
            "concepts": len(concepts),
            "relations": len(relations),
            "triples": len(unique),
            "duplicate_edges": len(array) - len(unique),
            **dict(counts),
        },
        "not_in_triples": {
            "attribute_values": dict(sorted(attribute_values.items())),
            "qualifier_values": dict(sorted(qualifier_values.items())),
            "fact_identity": "facts.jsonl and attributes.jsonl; qualifiers join by fact_id",
        },
        "provenance": provenance([source, Path(__file__)]),
    }
    _write_json(out / "manifest.json", manifest)
    return manifest


CLASSES = ("paths", "expressions", "neither", "unannotated")
RELAXED_CLASSES = ("entity_path", "date_path", "neither", "unannotated")
# Native KoPL's arities. QueryName is accepted as an alias for its released What.
SIGNATURES = {
    "FindAll": (0, 0),
    "Find": (0, 1),
    "FilterConcept": (1, 1),
    "FilterStr": (1, 2),
    "FilterNum": (1, 3),
    "FilterYear": (1, 3),
    "FilterDate": (1, 3),
    "QFilterStr": (1, 2),
    "QFilterNum": (1, 3),
    "QFilterYear": (1, 3),
    "QFilterDate": (1, 3),
    "Relate": (1, 2),
    "And": (2, 0),
    "Or": (2, 0),
    "What": (1, 0),
    "QueryName": (1, 0),
    "Count": (1, 0),
    "SelectBetween": (2, 2),
    "SelectAmong": (1, 2),
    "QueryAttr": (1, 1),
    "QueryAttrUnderCondition": (1, 3),
    "VerifyStr": (1, 1),
    "VerifyNum": (1, 2),
    "VerifyYear": (1, 2),
    "VerifyDate": (1, 2),
    "QueryRelation": (2, 0),
    "QueryAttrQualifier": (1, 3),
    "QueryRelationQualifier": (2, 2),
}
TRANSLATABLE = {"Find", "Relate", "What", "QueryName", "QueryAttr", "SelectBetween"}
MAX_HOPS = 8
SCORER_SOURCE = "https://github.com/shijx12/KQAPro_Baselines/blob/master/evaluate.py"
KOPL_SOURCE = "https://github.com/shijx12/KQAPro_Baselines/blob/master/Program/executor_rule.py"


def answers_equal(answer: str, prediction: str) -> bool:
    """KQA Pro's evaluate.py equality: numeric/unit normalization and year/date equality.

    Deliberately retains its asymmetric float-to-integer tolerance, including for
    negative numbers. String case and units remain significant. No substring match.
    """

    def numeric(text):
        fields = text.split()
        if not fields:
            return text
        try:
            number = float(fields[0])
            integer = int(number)
        except (ValueError, OverflowError):
            return text
        rendered = str(integer if number - integer < 1e-5 else number)
        return " ".join([rendered, *fields[1:]])

    def temporal(text):
        fields = text.split("-")
        return date(*map(int, fields)) if len(fields) == 3 else int(text)

    a, b = numeric(answer), numeric(prediction)
    if a == b:
        return True
    try:
        x, y = temporal(a), temporal(b)
    except (ValueError, OverflowError):
        return False
    if isinstance(x, date) and isinstance(y, date):
        return x == y
    return (x.year if isinstance(x, date) else x) == (y.year if isinstance(y, date) else y)


class NativeDates:
    """DateStore's lookup interface with KQA's explicit year/day precision intact."""

    def __init__(self, kg: Path, db: Path) -> None:
        db.unlink(missing_ok=True)
        self._con = sqlite3.connect(db)
        self._con.execute("CREATE TABLE dates (s INT, p INT, year INT, month INT, day INT)")
        with (kg / "attributes.jsonl").open(encoding="utf-8") as lines:
            for line in lines:
                row = json.loads(line)
                if row["value"]["type"] in ("date", "year"):
                    self._con.execute(
                        "INSERT INTO dates VALUES (?, ?, ?, ?, ?)",
                        (_number(row["subject"]), int(row["relation"][1:]), *_time(row["value"])),
                    )
        self._con.execute("CREATE INDEX dates_sp ON dates (s, p)")
        self._con.commit()

    def dates_of(self, items: list[int], props: set[int]) -> dict:
        from xgraph.expressions import Date

        found = {}
        for item in items:
            for prop in sorted(props):
                for year, month, day in self._con.execute(
                    "SELECT DISTINCT year, month, day FROM dates WHERE s = ? AND p = ? "
                    "ORDER BY year, month, day",
                    (item, prop),
                ):
                    found.setdefault(item, []).append(Date(year, month, day))
        return found

    def close(self) -> None:
        self._con.close()


def _program(row: dict, qid: str) -> list[dict] | None:
    if "program" not in row:
        return None
    program = row["program"]
    if not isinstance(program, list) or not program:
        raise ValueError(f"{qid}: program must be a nonempty list")
    for index, step in enumerate(program):
        context = f"{qid} step {index}"
        if not isinstance(step, dict) or not isinstance(step.get("function"), str):
            raise ValueError(f"{context}: missing function name")
        deps, inputs = step.get("dependencies"), step.get("inputs")
        if not isinstance(deps, list) or any(
            type(d) is not int or d < 0 or d >= index for d in deps
        ):
            raise ValueError(f"{context}: dependencies must name earlier steps")
        if not isinstance(inputs, list) or any(not isinstance(x, str) for x in inputs):
            raise ValueError(f"{context}: inputs must be strings")
        if (
            step["function"] in SIGNATURES
            and (len(deps), len(inputs)) != SIGNATURES[step["function"]]
        ):
            raise ValueError(f"{context}: wrong arity for {step['function']}")
        if step["function"] == "Relate" and inputs[1] not in ("forward", "backward"):
            raise ValueError(f"{context}: invalid direction {inputs[1]!r}")
        if step["function"] == "SelectBetween" and inputs[1] not in ("less", "greater"):
            raise ValueError(f"{context}: invalid comparison {inputs[1]!r}")
    return program


@dataclass
class Translation:
    category: str
    expression: object = None
    blockers: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


class Compiler:
    """Conservative whole-program translations, independent of answers and choices.

    Find/Relate*/What is a path; a date-valued QueryAttr adds a date endpoint.
    SelectBetween on two bare Find operands and a date key is earlier/later.
    same: cannot express KoPL's reusable And set or a Verify* literal constant.
    """

    def __init__(self, kg: Path) -> None:
        mappings = _json(kg / "id-mappings.json")
        self.types = {key: set(types) for key, types in mappings["attribute_types"].items()}
        self.relations = {
            (r["kind"], r["name"], r["direction"]): r["label"] for r in mappings["relations"]
        }
        self.names: dict[str, list[int]] = {}
        with (kg / "nodes.jsonl").open(encoding="utf-8") as lines:
            for line in lines:
                node = json.loads(line)
                self.names.setdefault(node["name"], []).append(_number(node["id"]))
        self.node_count = len(mappings["entities"])

    def compile(self, program: list[dict] | None) -> Translation:
        from xgraph.expressions import (
            Compare,
            ExpressionError,
            Start,
            parse,
            render,
        )
        from xgraph.expressions import (
            Path as ExprPath,
        )

        if program is None:
            return Translation("unannotated", reasons=("no_program",))
        blockers = {s["function"] for s in program if s["function"] not in TRANSLATABLE}
        reasons = set()
        used, pending = set(), [len(program) - 1]
        while pending:
            index = pending.pop()
            if index not in used:
                used.add(index)
                pending.extend(program[index]["dependencies"])
        if len(used) != len(program):
            reasons.add("unused_steps")
        for step in program:
            op, inputs = step["function"], step["inputs"]
            if op in ("QueryAttr", "SelectBetween"):
                types = self.types.get(inputs[0], set())
                if not types or not types <= {"date", "year"}:
                    blockers.add(op)
                    reasons.add(
                        "unknown_attribute" if not types else "literal_" + "+".join(sorted(types))
                    )
        if blockers or reasons:
            return Translation(
                "neither", blockers=tuple(sorted(blockers)), reasons=tuple(sorted(reasons))
            )

        def chain(index):
            step = program[index]
            op, inputs, deps = step["function"], step["inputs"], step["dependencies"]
            if op == "Find":
                return ExprPath(Start(inputs[0], tuple(self.names.get(inputs[0], []))), ())
            if op == "Relate":
                previous = chain(deps[0])
                label = self.relations.get(("relation", inputs[0], inputs[1]))
                if previous is not None and label is not None:
                    return ExprPath(previous.start, (*previous.relations, label))
            return None

        last = program[-1]
        op, deps, inputs = last["function"], last["dependencies"], last["inputs"]
        expression, category = None, "paths"
        if op in ("What", "QueryName"):
            expression = chain(deps[0])
        elif op == "QueryAttr":
            path = chain(deps[0])
            if path is not None:
                label = self.relations[("attribute", inputs[0], "forward")]
                expression = ExprPath(path.start, (*path.relations, label))
            category = "expressions"
        elif op == "SelectBetween":
            a, b = (chain(d) for d in deps)
            if a is not None and b is not None and not a.relations and not b.relations:
                label = self.relations[("attribute", inputs[0], "forward")]
                expression = Compare(
                    "earlier" if inputs[1] == "less" else "later", (a.start, b.start), (label,)
                )
            category = "expressions"
        if expression is None:
            return Translation("neither", reasons=("program_shape",))
        relations = expression.relations if isinstance(expression, ExprPath) else expression.via
        if len(relations) > MAX_HOPS:
            return Translation("neither", reasons=("hop_limit",))
        plain = (
            ExprPath(Start(expression.start.label), expression.relations)
            if isinstance(expression, ExprPath)
            else Compare(
                expression.op, tuple(Start(s.label) for s in expression.starts), expression.via
            )
        )
        try:
            spelled = parse(render(expression))
        except ExpressionError:
            spelled = None
        if spelled != plain:
            return Translation("neither", reasons=("unwritable_expression",))
        return Translation(category, expression)

    def compile_relaxed(self, program: list[dict] | None) -> tuple[Translation, list[dict]]:
        """Drop only FilterConcept on a single entity chain, then reuse strict checks.

        All original steps must belong to that chain. Filters on scalar values,
        unused branches, joins, other filters and comparison terminals do not qualify.
        Returned filter positions refer to the original program, before projection.
        """
        if program is None:
            return Translation("unannotated", reasons=("no_program",)), []
        last = program[-1]
        if last["function"] not in ("What", "QueryName", "QueryAttr"):
            return Translation("neither", reasons=("not_chain_answer",)), []
        used = {len(program) - 1}
        index = last["dependencies"][0]
        while True:
            used.add(index)
            step = program[index]
            op = step["function"]
            if op == "Find":
                break
            if op not in ("Relate", "FilterConcept"):
                return Translation("neither", blockers=(op,), reasons=("program_shape",)), []
            index = step["dependencies"][0]
        if len(used) != len(program):
            return Translation("neither", reasons=("unused_steps",)), []
        projected, dropped, positions = [], [], {}
        for index, step in enumerate(program):
            deps = [positions[d] for d in step["dependencies"]]
            if step["function"] == "FilterConcept":
                positions[index] = deps[0]
                dropped.append({"step": index, "concept": step["inputs"][0]})
            else:
                positions[index] = len(projected)
                projected.append({**step, "dependencies": deps})
        translated = self.compile(projected)
        category = {"paths": "entity_path", "expressions": "date_path"}.get(
            translated.category, translated.category
        )
        return Translation(
            category, translated.expression, translated.blockers, translated.reasons
        ), dropped


def _date_answer(value) -> str:
    if value.precision == "year":
        return str(value.year)
    return f"{value.year:04d}-{value.month:02d}-{value.day:02d}"


def _execution(
    executor, expression, gold: str | None, path_mode: bool, node_count: int, anchored: bool
) -> dict:
    from xgraph.expressions import Path as ExprPath
    from xgraph.expressions import Result, written
    from xgraph.programs import execute

    starts = (expression.start,) if isinstance(expression, ExprPath) else expression.starts
    if anchored and any(not s.items for s in starts):
        result = Result("unbound")
    elif path_mode:
        start = starts[0]
        items = execute(
            executor.store,
            start.label,
            list(expression.relations),
            limit=node_count,
            anchor=[f"Q{i}" for i in start.items] if anchored else None,
            all_senses=True,
        )
        result = Result("value" if items else "empty", entities=frozenset(items))
    else:
        result = executor.run(expression if anchored else written(expression))
    labels = sorted({executor.store.label(i) for i in result.entities})
    dates = [_date_answer(d) for d in result.dates]
    answer = None
    if isinstance(result.answer, bool):
        answer = "yes" if result.answer else "no"
    elif result.answer is not None:
        answer = executor.store.label(result.answer)
    candidates = [answer] if answer is not None else [*labels, *dates]
    return {
        "status": result.status,
        "entities": [f"Q{i}" for i in sorted(result.entities)],
        "dates": dates,
        "answer": answer,
        "correct": None
        if gold is None
        else result.status == "value" and any(answers_equal(gold, value) for value in candidates),
    }


def _tally(records: list[dict]) -> dict:
    supported = [r for r in records if r["expression"] is not None]
    scored = [r for r in supported if r["gold"] is not None]
    return {
        "questions": len(records),
        "labeled_questions": sum(r["gold"] is not None for r in records),
        "representable": len(supported),
        "scored": len(scored),
        "anchored_correct": sum(r["anchored"]["correct"] for r in scored),
        "written_correct": sum(r["written"]["correct"] for r in scored),
        "anchored_status": dict(Counter(r["anchored"]["status"] for r in supported)),
        "written_status": dict(Counter(r["written"]["status"] for r in supported)),
    }


def _relaxed_tally(records: list[dict]) -> dict:
    """Separate relaxed counts; strict record fields and their tallies stay intact."""
    views = [{"gold": r["gold"], **r["relaxed"]} for r in records]
    supported = [r for r in records if r["relaxed"]["expression"] is not None]
    return {
        **_tally(views),
        "additional_to_strict": sum(r["expression"] is None for r in supported),
        "with_dropped_filters": sum(bool(r["relaxed"]["dropped_filters"]) for r in supported),
    }


def profile(splits: dict[str, Path], kg: str | Path, out: str | Path) -> dict:
    """Execute strict translations and separate relaxed chains, retaining full denominators."""
    from xgraph.expressions import Executor, Relations, render
    from xgraph.provenance import provenance
    from xgraph.toolkg import KGStore, build_store

    kg, out = Path(kg), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    compiler = Compiler(kg)
    build_store(kg, out / "kg.sqlite")
    store = KGStore(out / "kg.sqlite")
    dates = NativeDates(kg, out / "dates.sqlite")
    executor = Executor(store, dates, Relations.read(kg / "relation_labels.jsonl"))
    summaries = {}
    try:
        for split, source in splits.items():
            _split_name(split)
            records = []
            for index, row in enumerate(_questions(Path(source))):
                qid = f"kqapro-{split}-{index}"
                program = _program(row, qid)
                translated = compiler.compile(program)
                expression = translated.expression
                record = {
                    "id": qid,
                    "question": row["question"],
                    "gold": row.get("answer"),
                    "program": program,
                    "operations": sorted({s["function"] for s in program or []}),
                    "terminal": program[-1]["function"] if program else None,
                    "class": translated.category,
                    "expression": render(expression) if expression is not None else None,
                    "blocking_operations": list(translated.blockers),
                    "reasons": list(translated.reasons),
                }
                if expression is not None:
                    for name, anchored in (("anchored", True), ("written", False)):
                        record[name] = _execution(
                            executor,
                            expression,
                            row.get("answer"),
                            translated.category == "paths",
                            compiler.node_count,
                            anchored,
                        )
                relaxed, dropped = compiler.compile_relaxed(program)
                record["relaxed"] = {
                    "class": relaxed.category,
                    "expression": render(relaxed.expression)
                    if relaxed.expression is not None
                    else None,
                    "blocking_operations": list(relaxed.blockers),
                    "reasons": list(relaxed.reasons),
                    "dropped_filters": dropped,
                }
                if relaxed.expression is not None:
                    for name, anchored in (("anchored", True), ("written", False)):
                        record["relaxed"][name] = _execution(
                            executor,
                            relaxed.expression,
                            row.get("answer"),
                            relaxed.category == "entity_path",
                            compiler.node_count,
                            anchored,
                        )
                records.append(record)
            with (out / f"{split}.jsonl").open("w", encoding="utf-8") as file:
                for record in records:
                    _line(file, record)
            operations = sorted({op for r in records for op in r["operations"]})
            terminals = sorted({r["terminal"] for r in records if r["terminal"] is not None})
            blockers = sorted({op for r in records for op in r["blocking_operations"]})
            summaries[split] = {
                **_tally(records),
                "by_class": {
                    name: _tally([r for r in records if r["class"] == name]) for name in CLASSES
                },
                "by_operation": {
                    op: _tally([r for r in records if op in r["operations"]]) for op in operations
                },
                "by_terminal": {
                    op: _tally([r for r in records if r["terminal"] == op]) for op in terminals
                },
                "blocking_operations": {
                    op: {
                        "questions": sum(op in r["blocking_operations"] for r in records),
                        "sole_blocker": sum(
                            r["blocking_operations"] == [op]
                            and all(
                                reason.startswith("literal_") or reason == "unknown_attribute"
                                for reason in r["reasons"]
                            )
                            for r in records
                        ),
                    }
                    for op in blockers
                },
                "reasons": dict(Counter(reason for r in records for reason in r["reasons"])),
                "operation_combinations": dict(Counter("+".join(r["operations"]) for r in records)),
                "relaxed": {
                    **_relaxed_tally(records),
                    "by_class": {
                        name: _relaxed_tally([r for r in records if r["relaxed"]["class"] == name])
                        for name in RELAXED_CLASSES
                    },
                },
            }
    finally:
        dates.close()
        store._con.close()
    inputs = (
        [Path(p) for p in splits.values()]
        + [
            kg / p
            for p in (
                "manifest.json",
                "id-mappings.json",
                "nodes.jsonl",
                "facts.jsonl",
                "attributes.jsonl",
                "qualifiers.jsonl",
                "entity_labels.jsonl",
                "relation_labels.jsonl",
                "triples.npy",
            )
        ]
        + [
            Path(__file__),
            Path(__file__).with_name("expressions.py"),
            Path(__file__).with_name("programs.py"),
        ]
    )
    report = {
        "artifact_kind": "xgraph.kqapro_profile",
        "version": 1,
        "splits": summaries,
        "configuration": {
            "max_hops": MAX_HOPS,
            "seed": None,
            "classes": "exclusive: expressions means additional to paths",
            "compilation": "complete programs; no elimination of filters, joins or unused branches",
            "paths": "all name senses, complete frontiers, no fuzzy relation repair",
            "expressions": "committed starts; native year/day precision via NativeDates",
            "correctness": "gold among reached answers using KQA equality; "
            "not endpoint selection accuracy",
            "same": "no KoPL translation: Verify takes a constant, And returns a reusable set",
            "native_execution": "no full KoPL or SPARQL executor is run",
            "relaxed": {
                "classes": "entity_path and date_path include their strictly admitted chains",
                "filters": "FilterConcept on chain entities ignored; no final-frontier type check",
                "scope": "one Find, Relate/FilterConcept steps, "
                "terminal What/QueryName or date QueryAttr",
                "execution": "same CPU path/expression rules as strict, "
                "without the dropped filters",
                "interpretation": "gold membership after relaxing semantics, "
                "not complete-program coverage",
            },
        },
        "sources": {"answer_equality": SCORER_SOURCE, "kopl_semantics": KOPL_SOURCE},
        "provenance": provenance(inputs),
    }
    _write_json(out / "profile.json", report)
    lines = [
        "# KQA Pro complete-program coverage",
        "",
        "Expressions are additional to paths. "
        "Gold-in-result is execution coverage, not model accuracy.",
        "",
        "| Split | Language | Questions | Anchored to gold | Written to gold |",
        "|---|---|---:|---:|---:|",
    ]
    for split, summary in summaries.items():
        for name, counts in summary["by_class"].items():
            n = counts["questions"]
            scored = counts["scored"]
            a, w = counts["anchored_correct"], counts["written_correct"]
            cells = (
                (f"{a}/{scored} ({a / scored:.2%})", f"{w}/{scored} ({w / scored:.2%})")
                if scored
                else ("—", "—")
            )
            lines.append(f"| {split} | {name} | {n} | {cells[0]} | {cells[1]} |")
    lines.extend(
        [
            "",
            "Unsupported and unannotated questions stay in the denominator.",
            "No full native KoPL or SPARQL execution parity is claimed.",
            "",
        ]
    )
    lines.extend(
        [
            "## Concept filters ignored",
            "",
            "Separate relaxed chains; the strict table above is unchanged. "
            "Entity and date paths include strictly admitted chains. No type filters are executed.",
            "",
            "| Split | Relaxed class | Questions | Additional to strict "
            "| Anchored to gold | Written to gold |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for split, summary in summaries.items():
        for name, counts in summary["relaxed"]["by_class"].items():
            scored = counts["scored"]
            a, w = counts["anchored_correct"], counts["written_correct"]
            cells = (
                (f"{a}/{scored} ({a / scored:.2%})", f"{w}/{scored} ({w / scored:.2%})")
                if scored
                else ("—", "—")
            )
            lines.append(
                f"| {split} | {name} | {counts['questions']} | {counts['additional_to_strict']} "
                f"| {cells[0]} | {cells[1]} |"
            )
    lines.extend(
        [
            "",
            "Gold membership under relaxed semantics does not establish native program parity.",
            "",
        ]
    )
    (out / "coverage.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def _split_name(name: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
        raise ValueError(f"invalid split name {name!r}")
    return name


def write_requests(source: str | Path, out: str | Path, split: str = "val") -> dict:
    """All questions as unanchored path requests; never consult programs or choices."""
    from xgraph.prompts import wwq_messages
    from xgraph.provenance import provenance

    source, out = Path(source), Path(out)
    _split_name(split)
    rows = _questions(source)
    out.mkdir(parents=True, exist_ok=True)
    questions = [
        {
            "id": f"kqapro-{split}-{i}",
            "utterance": row["question"],
            "sparql": "",
            "complexity": f"kqapro {split}",
        }
        for i, row in enumerate(rows)
    ]
    _write_json(out / "questions.json", questions)
    with (out / "requests-paths.jsonl").open("w", encoding="utf-8") as file:
        for question in questions:
            _line(
                file,
                {
                    "id": question["id"],
                    "messages": wwq_messages(question["utterance"], path_spans=True),
                },
            )
    _write_json(
        out / "gold.json",
        {
            "answers": {
                q["id"]: row["answer"]
                for q, row in zip(questions, rows, strict=True)
                if "answer" in row
            },
            "unanswered": [
                q["id"] for q, row in zip(questions, rows, strict=True) if "answer" not in row
            ],
        },
    )
    manifest = {
        "artifact_kind": "xgraph.kqapro_requests",
        "version": 1,
        "split": split,
        "questions": len(rows),
        "configuration": {
            "path_spans": True,
            "anchor": False,
            "programs_choices_answers_in_prompt": False,
        },
        "provenance": provenance([source, Path(__file__)]),
    }
    _write_json(out / "manifest.json", manifest)
    return manifest


def score(
    rows: list[dict], texts: dict[str, str], split: str = "val", whole_text: bool = False
) -> dict:
    """Native equality on the final answer, every labeled question in the denominator."""
    from xgraph.prompts import final_answer

    if any("answer" not in row for row in rows):
        raise ValueError(f"{split}: no answers for one or more questions; cannot score")
    ids = [f"kqapro-{split}-{i}" for i in range(len(rows))]
    unknown = texts.keys() - set(ids)
    if unknown:
        raise ValueError(f"{split}: output ids absent from the source: {sorted(unknown)[:5]}")
    records = []
    for qid, row in zip(ids, rows, strict=True):
        text = texts.get(qid)
        said = (
            text
            if whole_text
            else (final_answer(text) or final_answer(text, syntax=None))
            if text is not None
            else None
        )
        # A spanned escape is an abstention, including in a malformed/plain answer.
        if said and any(mark in said for mark in ("⟦", "⟧", "⁇")):
            said = None
        records.append(
            {
                "id": qid,
                "prediction": said,
                "gold": row["answer"],
                "correct": said is not None and answers_equal(row["answer"], said.strip()),
                "missing": text is None,
            }
        )
    correct = sum(r["correct"] for r in records)
    return {
        "questions": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "missing": sum(r["missing"] for r in records),
        "records": records,
    }


def main(argv: list[str] | None = None) -> None:
    """CPU-only import, complete-program coverage, requests and native scoring."""
    parser = argparse.ArgumentParser(prog="python -m xgraph.kqapro", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    kg = commands.add_parser("kg", help="native kb.json as a KG directory and lossless sidecars")
    kg.add_argument("--source", type=Path, required=True, help="kb.json (read as JSON only)")
    kg.add_argument("--out", type=Path, required=True)
    audit = commands.add_parser("profile", help="strict/relaxed program coverage and execution")
    audit.add_argument("--kg", type=Path, required=True)
    audit.add_argument(
        "--split", action="append", required=True, help="name=questions.json; repeat for train/val"
    )
    audit.add_argument("--out", type=Path, required=True)
    requests = commands.add_parser("requests", help="every question as a path-mode request")
    scoring = commands.add_parser("score", help="KQA equality over all labeled source rows")
    for command in (requests, scoring):
        command.add_argument("--source", type=Path, required=True, help="a native split JSON array")
        command.add_argument("--split", default="val", help="split name used in request ids")
        command.add_argument("--out", type=Path, required=True)
    scoring.add_argument("--outputs", type=Path, required=True, help="generation records (JSONL)")
    scoring.add_argument(
        "--whole-text", action="store_true", help="score text itself as the prediction"
    )
    args = parser.parse_args(argv)
    if args.command == "kg":
        report = convert_kg(args.source, args.out)
        print(json.dumps(report["counts"], sort_keys=True))
    elif args.command == "profile":
        splits = {}
        for spec in args.split:
            name, separator, path = spec.partition("=")
            if not separator or not path:
                raise ValueError(f"invalid --split {spec!r}; expected name=path.json")
            _split_name(name)
            if name in splits:
                raise ValueError(f"duplicate --split {name!r}")
            splits[name] = Path(path)
        report = profile(splits, args.kg, args.out)
        print(json.dumps({k: v["by_class"] for k, v in report["splits"].items()}, sort_keys=True))
        print(
            json.dumps(
                {"relaxed": {k: v["relaxed"]["by_class"] for k, v in report["splits"].items()}},
                sort_keys=True,
            )
        )
    elif args.command == "requests":
        report = write_requests(args.source, args.out, args.split)
        print(f"{report['questions']} requests")
    else:
        from xgraph.provenance import provenance

        _split_name(args.split)
        texts = {}
        with args.outputs.open(encoding="utf-8") as lines:
            for index, line in enumerate(lines):
                row = json.loads(line)
                if not isinstance(row.get("id"), str) or not isinstance(row.get("text"), str):
                    raise ValueError(
                        f"{args.outputs} line {index + 1}: expected id and text strings"
                    )
                if row["id"] in texts:
                    raise ValueError(
                        f"{args.outputs} line {index + 1}: duplicate output id {row['id']!r}"
                    )
                texts[row["id"]] = record_text(row)
        report = {
            "artifact_kind": "xgraph.kqapro_score",
            "version": 1,
            **score(_questions(args.source), texts, args.split, args.whole_text),
            "configuration": {"whole_text": args.whole_text, "equality": SCORER_SOURCE},
            "provenance": provenance([args.source, args.outputs, Path(__file__)]),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        _write_json(args.out, report)
        print(f"{report['correct']}/{report['questions']} correct")


if __name__ == "__main__":
    main()
