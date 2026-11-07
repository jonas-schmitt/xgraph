"""Independent STRIPS semantics for the opt-in PlanBench interface pilot.

Reads preconditions/effects from domain files, without PlanBench's executor or
xgraph's KG engine. Supports conjunctive STRIPS, optional typing and negative
preconditions. Quantifiers, conditional effects, numeric fluents, equality and
other PDDL constructs fail closed. Facts, not reachable states, are materialized.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

Fact = tuple[str, ...]
State = frozenset[Fact]


def sexpr(text: str) -> list:
    tokens = re.findall(r"\(|\)|[^\s()]+", re.sub(r";[^\n]*", "", text).lower())
    stack: list[list] = []
    roots = []
    for token in tokens:
        if token == "(":
            node = []
            (stack[-1] if stack else roots).append(node)
            stack.append(node)
        elif token == ")":
            if not stack:
                raise ValueError("PDDL: unmatched closing parenthesis")
            stack.pop()
        elif stack:
            stack[-1].append(token)
        else:
            raise ValueError(f"PDDL: text outside expression: {token!r}")
    if stack or len(roots) != 1 or not roots[0]:
        raise ValueError("PDDL: expected one complete expression")
    return roots[0]


def typed_names(items: list) -> tuple[tuple[str, str], ...]:
    result = []
    pending = []
    at = 0
    while at < len(items):
        item = items[at]
        if not isinstance(item, str):
            raise ValueError("PDDL: complex type expression is unsupported")
        if item == "-":
            if not pending or at + 1 == len(items) or not isinstance(items[at + 1], str):
                raise ValueError("PDDL: invalid typed name list")
            result.extend((name, items[at + 1]) for name in pending)
            pending = []
            at += 2
        else:
            pending.append(item)
            at += 1
    result.extend((name, "object") for name in pending)
    if len({name for name, _ in result}) != len(result):
        raise ValueError("PDDL: duplicate name")
    return tuple(result)


def literals(node: list) -> tuple[State, State]:
    if not isinstance(node, list) or not node or not isinstance(node[0], str):
        raise ValueError("PDDL: expected literal or conjunction")
    if node[0] == "and":
        positive, negative = set(), set()
        for child in node[1:]:
            pos, neg = literals(child)
            positive.update(pos)
            negative.update(neg)
        return frozenset(positive), frozenset(negative)
    if node[0] == "not":
        if len(node) != 2:
            raise ValueError("PDDL: invalid negation")
        pos, neg = literals(node[1])
        if neg or len(pos) != 1:
            raise ValueError("PDDL: negation must contain an atom")
        return frozenset(), pos
    if node[0] in {"or", "forall", "exists", "when", "=", "increase", "imply"} or any(
        not isinstance(x, str) for x in node
    ):
        raise ValueError(f"PDDL: unsupported construct {node[0]!r}")
    return frozenset({tuple(node)}), frozenset()


@dataclass(frozen=True)
class Action:
    name: str
    arguments: tuple[str, ...]
    positive: State
    negative: State
    add: State
    delete: State

    @property
    def text(self) -> str:
        return "(" + " ".join((self.name, *self.arguments)) + ")"

    def applicable(self, state: State) -> bool:
        return self.positive <= state and not self.negative & state

    def apply(self, state: State) -> State:
        if not self.applicable(state):
            raise ValueError(f"inapplicable action {self.text}")
        return (state - self.delete) | self.add


@dataclass(frozen=True)
class Schema:
    name: str
    parameters: tuple[tuple[str, str], ...]
    positive: State
    negative: State
    add: State
    delete: State


@dataclass(frozen=True)
class Problem:
    name: str
    domain: str
    objects: tuple[tuple[str, str], ...]
    init: State
    goal_positive: State
    goal_negative: State
    actions: tuple[Action, ...]
    domain_text: str
    problem_text: str

    def applicable(self, state: State) -> tuple[Action, ...]:
        return tuple(a for a in self.actions if a.applicable(state))

    def goal(self, state: State) -> bool:
        return self.goal_positive <= state and not self.goal_negative & state


def _sections(root: list, kind: str) -> tuple[str, list[list]]:
    if (
        len(root) < 2
        or root[0] != "define"
        or not isinstance(root[1], list)
        or not root[1]
        or root[1][0] != kind
    ):
        raise ValueError(f"PDDL: expected a {kind} definition")
    if len(root[1]) != 2 or any(not isinstance(x, list) or not x for x in root[2:]):
        raise ValueError(f"PDDL: malformed {kind} definition")
    return root[1][1], root[2:]


def load_problem(domain_text: str, problem_text: str) -> Problem:
    domain, sections = _sections(sexpr(domain_text), "domain")
    schemas = []
    predicates = {}
    types = {"object": None}
    for section in sections:
        key = section[0]
        if key == ":requirements":
            if set(section[1:]) - {":strips", ":typing", ":negative-preconditions"}:
                raise ValueError(f"PDDL: unsupported requirements {section[1:]}")
        elif key == ":types":
            types.update(typed_names(section[1:]))
        elif key == ":predicates":
            for declaration in section[1:]:
                if (
                    not isinstance(declaration, list)
                    or not declaration
                    or not isinstance(declaration[0], str)
                ):
                    raise ValueError("PDDL: malformed predicate declaration")
                if declaration[0] in predicates:
                    raise ValueError("PDDL: duplicate predicate")
                predicates[declaration[0]] = typed_names(declaration[1:])
        elif key == ":action":
            if len(section) != 8 or set(section[2::2]) != {
                ":parameters",
                ":precondition",
                ":effect",
            }:
                raise ValueError("PDDL: expected action parameters, precondition and effect")
            fields = dict(zip(section[2::2], section[3::2], strict=True))
            positive, negative = literals(fields[":precondition"])
            add, delete = literals(fields[":effect"])
            schemas.append(
                Schema(
                    section[1], typed_names(fields[":parameters"]), positive, negative, add, delete
                )
            )
        else:
            raise ValueError(f"PDDL: unsupported domain section {key}")
    if len({s.name for s in schemas}) != len(schemas):
        raise ValueError("PDDL: duplicate action")
    name, sections = _sections(sexpr(problem_text), "problem")
    fields = {}
    for section in sections:
        if section[0] not in {":domain", ":objects", ":init", ":goal"} or section[0] in fields:
            raise ValueError(f"PDDL: unsupported or duplicate problem section {section[0]}")
        fields[section[0]] = section[1:]
    if set(fields) != {":domain", ":objects", ":init", ":goal"} or fields[":domain"] != [domain]:
        raise ValueError("PDDL: missing problem sections or domain mismatch")
    objects = typed_names(fields[":objects"])
    init, init_negative = literals(["and", *fields[":init"]])
    if init_negative or len(fields[":goal"]) != 1:
        raise ValueError("PDDL: expected positive init and one goal")
    goal_positive, goal_negative = literals(fields[":goal"][0])
    object_types = dict(objects)

    def subtype(actual, expected):
        seen = set()
        while actual is not None:
            if actual in seen or actual not in types:
                raise ValueError(f"PDDL: invalid type hierarchy at {actual}")
            seen.add(actual)
            if actual == expected:
                return True
            actual = types[actual]
        return False

    def validate(facts, variables):
        for fact in facts:
            if fact[0] not in predicates or len(fact) - 1 != len(predicates[fact[0]]):
                raise ValueError(f"PDDL: unknown predicate or arity: {fact}")
            for value, (_, required) in zip(fact[1:], predicates[fact[0]], strict=True):
                actual = variables.get(value) if value.startswith("?") else object_types.get(value)
                if actual is None or not subtype(actual, required):
                    raise ValueError(f"PDDL: unknown object/variable or wrong type: {fact}")

    validate(init | goal_positive | goal_negative, {})
    for schema in schemas:
        variables = dict(schema.parameters)
        if any(not name.startswith("?") for name in variables):
            raise ValueError("PDDL: action parameters must be variables")
        validate(schema.positive | schema.negative | schema.add | schema.delete, variables)
    dynamic = {f[0] for s in schemas for f in s.add | s.delete}
    actions = []
    for schema in schemas:
        # Static unary facts sharply reduce grounding, especially in Logistics.
        choices = []
        static_pos = frozenset(f for f in schema.positive if f[0] not in dynamic)
        static_neg = frozenset(f for f in schema.negative if f[0] not in dynamic)
        for variable, kind in schema.parameters:
            unary = [f[0] for f in static_pos if f[1:] == (variable,)]
            choices.append(
                [o for o, t in objects if subtype(t, kind) and all((p, o) in init for p in unary)]
            )
        for arguments in itertools.product(*choices):
            bindings = dict(zip((v for v, _ in schema.parameters), arguments, strict=True))

            def ground(facts, bindings=bindings):
                return frozenset(tuple(bindings.get(x, x) for x in f) for f in facts)

            if not ground(static_pos) <= init or ground(static_neg) & init:
                continue
            actions.append(
                Action(
                    schema.name,
                    arguments,
                    ground(schema.positive),
                    ground(schema.negative),
                    ground(schema.add),
                    ground(schema.delete),
                )
            )
            if len(actions) > 200_000:
                raise ValueError("PDDL: grounding exceeds pilot limit of 200000 actions")
    return Problem(
        name,
        domain,
        objects,
        init,
        goal_positive,
        goal_negative,
        tuple(sorted(actions, key=lambda a: a.text)),
        domain_text,
        problem_text,
    )


class PlanSyntaxError(ValueError):
    def __init__(self, message: str, prefix: tuple[str, ...]):
        super().__init__(message)
        self.prefix = prefix


def parse_plan(text: str) -> tuple[str, ...]:
    """Read flat PDDL actions only. Never extract a convenient substring from prose."""
    text = re.sub(r";[^\n]*", "", text).strip().lower()
    actions = []
    while text:
        match = re.match(r"\(\s*([\w?-]+(?:\s+[\w?-]+)*)\s*\)", text)
        if not match:
            raise PlanSyntaxError(f"malformed plan near {text[:80]!r}", tuple(actions))
        actions.append("(" + " ".join(match[1].split()) + ")")
        text = text[match.end() :].lstrip()
    return tuple(actions)


@dataclass(frozen=True)
class CheckedPlan:
    valid: bool
    goal_reached: bool
    length: int
    prefix: tuple[str, ...]
    state: State
    failed_step: int | None = None
    error: str | None = None


def check_plan(problem: Problem, text: str, *, state: State | None = None) -> CheckedPlan:
    state = problem.init if state is None else state
    syntax_error = None
    try:
        plan = parse_plan(text)
    except PlanSyntaxError as exc:
        plan, syntax_error = exc.prefix, str(exc)
    lookup = {a.text: a for a in problem.actions}
    prefix = []
    for step, written in enumerate(plan):
        action = lookup.get(written)
        if action is None or not action.applicable(state):
            error = (
                f"unknown or statically impossible action: {written}"
                if action is None
                else (
                    f"inapplicable {written}; missing={sorted(action.positive - state)}; "
                    f"forbidden={sorted(action.negative & state)}"
                )
            )
            return CheckedPlan(False, False, len(plan), tuple(prefix), state, step, error)
        state = action.apply(state)
        prefix.append(written)
    if syntax_error:
        return CheckedPlan(False, False, len(plan), tuple(prefix), state, len(plan), syntax_error)
    return CheckedPlan(True, problem.goal(state), len(plan), tuple(prefix), state)
