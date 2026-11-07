"""The KG agent baseline: the model answers by calling tools over the KG store.

The strongest agent over the same KG as path mode: asked a question, the model calls
tools until it knows the answer, then writes it as `Answer: ⟦ label⟧`, the line the
scorers read. Four tools, over the KG store (xgraph.toolkg) read through the request's KG
version where it names one:

- `search(name)`: the entities whose label or alias is the name (NFC, case folded), the
  best connected first, at most toolkg.SEARCH_LIMIT;
- `relations(entity)`: every relation of the entity's outgoing triples, by label, none
  left out;
- `follow(entity, relation)`: the entities the relation leads to from the entity, the
  best connected first, at most SHOWN listed, with their count;
- `execute_path(start, relations)`: the entities a whole path reaches, executed as
  parse-then-execute executes it (xgraph.programs.execute): from the entity id, or the
  best-connected entity with the label, along each relation in turn; at most SHOWN listed,
  the best connected first, with their count, or the hop where the path reaches nothing.

A relation is named by its label, or with relation labels given (a KG's
relation_labels.jsonl, as path mode's relation index holds them) by any of its labels and
aliases, every relation the name names followed. Entities are named by id (`Q42`) or
label.

The action format leans on no chat template's tool calling, so any model can follow it:
each reply is one tool call, a line `Action: {"tool": "...", ...}` with the arguments as
JSON fields, or the final answer. parse_action also reads a call as a function
(`Action: follow("Q1", "author")`), Qwen 2.5's `<tool_call>` JSON, and Qwen 3's
`<tool_call><function=...>` form; only the first call of a reply runs. The result comes
back as a user message `Observation: <JSON>`. A reply without a call is the answer; a
plain `Answer: label` is delivered as `Answer: ⟦ label⟧`. Every call counts against the
call budget (`max_calls`, unreadable ones too); once it is spent, the model is asked for
its final answer (FINAL), and its next reply is the answer, and the record's
`budget_spent` is true. Every round's prompt counts towards the cost, so records sum
prompt and generated tokens over rounds. Raw replies and per-call usage are kept in
`turns` for controlled serving replay (`xgraph.serve_cost`).

Three options, each off by default, bring the agent level with path mode:

- matched demonstrations (`--matched-demos`): in place of the one worked example
  (EXAMPLE), the request's own demonstrations (`demonstrations`: path mode's few-shot
  pairs of question and path span, as `requests-paths.jsonl` carries them), each as a tool
  trajectory: the question, an `execute_path` call with the demonstration's start and
  relations, its result (the demonstration's answer: the best-connected entity with its
  label, or the label alone where the KG has none), and the answer line;
- matched paths (`--matched-paths`, MatchedPaths): `execute_path` executes as matched
  execution does (xgraph.programs.MatchedExecutor, over the KG index and served
  tokenizer path mode ran with): labels written exactly, a relation label or alias
  followed as the union of its relations, the start committed in best-connected order
  over every sense the label names (or, written as an id, that entity; with the request's
  anchor, the anchor's), frontiers capped at the engine's limit in its order, read through
  the request's KG version; the entities are listed in that order;
- pages (`--pages`): follow and execute_path take an `offset` and list the SHOWN entities
  after it, with `next_offset` while more remain; the record's `pages` counts the calls
  that listed a later page.

Failure-aware mode (`--agent-feedback`) enables all three options. Failed execute_path
calls add failed_hop (one-based relation, zero for no start, null for invalid input) and
a bounded frontier_relations page of the labels the indexed frontier offers. The same
offset pages these labels on failure. A final answer after a failed path is refused
until another valid tool call or budget exhaustion. After REFUSALS refused answers,
the model's own answer stands, with finish_reason=retry_refused.
Refusals are model turns, not tool calls. Records add failures, path retries (retries),
retry_changed_plan, answer_refusals, and each action's result. Path retries reference
their failed action by zero-based retry_of and report plan_changed; a lookup between
them is allowed. Repeating a page is an unchanged retry. Only this mode sets
budget_spent whenever the actual tool-call budget is exhausted.

Masked mode (--agent-mask) writes path calls as spans, for example
`Action: execute_path ⟦ The Hobbit · author⟧`; follow writes exactly one relation
in the same form. search and relations remain JSON. Each model round is a new engine
request with its own sampling parameters and KG version; prompt demonstrations and
earlier observations are never replayed. The mask uses path mode's relation support
(frontier, vocabulary or successor). It does not decide whether a path answers the
question. --agent-force-path continues `Action: execute_path ⟦` in round one only,
with xgraph_prefill when masked, or freely as the opener control. Both compose with
feedback. Masked/forced records keep masked, unmasked_calls (non-span path calls that
bypassed the mask), executed_plans, and the last executed plan/reached for separate
delivery. The primary text remains the agent's own answer. Without these options the
legacy protocol and record fields are unchanged.

The structured-output controls (--agent-schema static|turn) keep the matched JSON
prompt and use vLLM's native structural tags to constrain only Action: JSON calls,
leaving the agent's answer free. static uses the writable indexed relation vocabulary;
turn recomputes its enum each round from the outgoing relations of every entity
reached so far, in the request's KG version (anchors initialize that set). Before any
entity is known it uses the vocabulary; known leaves permit only lookups or an answer.
Full capped execution results, including entities beyond a displayed page, contribute
to this set. The enum is fixed throughout each call, without within-path execution.
Records add agent_schema and schema_bypass_calls for non-JSON path calls. These
controls exclude the span mask and forced span opener, and compose with feedback.

    scripts/generate.sh <model> target/mintqa/requests-paths.jsonl agent <out.jsonl>
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from xgraph.prompts import SET_FORMAT, SET_INSTRUCTION, set_answer_fields
from xgraph.score import normalize
from xgraph.toolkg import parse_calls, store_for

SHOWN = 20
"""The most entities follow and execute_path list; the count says how many there are."""
MAX_CALLS = 8
"""The default call budget: enough for a four-hop question relation by relation, or for a
search and a whole path several times over."""

SYSTEM = """\
Answer the question with the help of a knowledge graph (Wikidata), which you query with \
tools. Each reply is either one tool call, on a line of its own: "Action:" followed by a \
JSON object naming the tool and its arguments; or, once you know the answer, the final \
line "Answer: ⟦ label⟧" with the answer entity's label.

Tools:
- {{"tool": "search", "name": "<label>"}}: the entities with this label or alias, best \
connected first.
- {{"tool": "relations", "entity": "<id>"}}: every relation the entity has.
- {{"tool": "follow", "entity": "<id>", "relation": "<relation label>"}}: the entities \
the relation leads to.
- {{"tool": "execute_path", "start": "<id or label>", "relations": ["<relation label>", \
...]}}: follows the relations in order from the start and gives the entities reached.

Entities are named by their id (such as Q42) or label, relations by their Wikidata \
label. You can make at most {budget} tool calls."""

PAGES = f"""
follow and execute_path list at most {SHOWN} entities; when there are more, the result \
gives "next_offset": add "offset": <n> to the same call to list the entities after the \
first n."""
"""What the instructions add with pages."""

REFUSALS = 2
"""Feedback mode: answers refused after a failed path before the model's own answer stands."""

FEEDBACK = f"""
After an execute_path error or empty path, failed_hop gives the first empty hop \
(1 is the first relation; 0 means no start; null means invalid arguments). \
frontier_relations lists the labels and aliases available just before that hop, \
at most {SHOWN} per page. Its count and next_offset describe relation labels, not \
entities: repeat the failed call with offset to see its next page.
Use this feedback and the question to reconsider the plan. After a failed path \
you must make another tool call before answering, while calls remain. An answer \
before that retry is refused. A lookup can help; execute_path can test a revised plan."""

EXAMPLE = [
    {"role": "user", "content": "On which continent is the country where Mount Kilimanjaro is?"},
    {"role": "assistant", "content": 'Action: {"tool": "search", "name": "Mount Kilimanjaro"}'},
    {
        "role": "user",
        "content": 'Observation: {"entities": [{"id": "Q7296", "label": "Mount Kilimanjaro"}]}',
    },
    {
        "role": "assistant",
        "content": 'Action: {"tool": "execute_path", "start": "Q7296", '
        '"relations": ["country", "continent"]}',
    },
    {
        "role": "user",
        "content": 'Observation: {"count": 1, "entities": [{"id": "Q15", "label": "Africa"}]}',
    },
    {"role": "assistant", "content": "Answer: ⟦ Africa⟧"},
]

FINAL = 'You have no tool calls left. Give your final answer now, as the line "Answer: ⟦ label⟧".'
FORCED_PATH = "Action: execute_path ⟦"
SPAN_CALLS = """
For execute_path and follow, write the call as a path span instead of JSON:
Action: execute_path ⟦ Start label · relation label · relation label⟧
Action: follow ⟦ Start label · relation label⟧
Use an entity's exact label, not its id, inside a span. follow names exactly one
relation. search and relations keep their JSON format. To page a span call, append
offset=<n> after its closing delimiter. Give your own final answer after observing
the tool result, in a separate reply."""

TOOLS = ("search", "relations", "follow", "execute_path")
PAGED = ("follow", "execute_path")
"""The tools that take an offset with pages."""
_NAMES = {
    "search": "search",
    "search_entities": "search",
    "relations": "relations",
    "get_relations": "relations",
    "relations_of": "relations",
    "follow": "follow",
    "follow_relation": "follow",
    "execute_path": "execute_path",
    "path": "execute_path",
}
_POSITIONS = {
    "search": ("name",),
    "relations": ("entity",),
    "follow": ("entity", "relation"),
    "execute_path": ("start", "relations"),
}
_ACTION = re.compile(r"^\s*Action:\s*(.*)$", re.MULTILINE)
_TOOL_CALL = re.compile(r"<tool_call>\s*(\{)", re.S)
_FUNCTION = re.compile(r"^(\w+)\s*\((.*)\)\s*$", re.S)
_SPAN_ACTION = re.compile(r"^(execute_path|follow)\s+(⟦[^⟧]*⟧)(?:\s+offset=(\d+))?\s*$")
_ENTITY = re.compile(r"Q\d+")
_ANSWER = re.compile(r"^Answer:\s*(.+?)\s*$", re.MULTILINE)
INVALID = ("invalid", {})


def relation_names_of(rows: list[tuple[str, str, str]]) -> dict[int, list[str]]:
    """Each property's labels and aliases from relation label rows, as path mode's
    relation index holds them (xgraph.wikidata.relation_names)."""
    from xgraph.wikidata import relation_names

    return {int(p[1:]): labels for p, labels in relation_names(rows).items()}


def _from_object(found: Any) -> tuple[str, dict]:
    if not isinstance(found, dict):
        return INVALID
    found = dict(found)
    if "tool" in found:
        name = found.pop("tool")
        arguments = found.pop("arguments", None) or found
    elif "arguments" in found or "parameters" in found:
        name = found.get("name")
        arguments = found.get("arguments") or found.get("parameters")
    else:
        return INVALID
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return INVALID
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return INVALID
    return _NAMES.get(name, name), arguments


def _decode(text: str, at: int) -> Any:
    try:
        return json.JSONDecoder().raw_decode(text, at)[0]
    except json.JSONDecodeError:
        return None


def _from_function(text: str) -> tuple[str, dict]:
    match = _FUNCTION.match(text.strip())
    if not match:
        return INVALID
    name = _NAMES.get(match.group(1), match.group(1))
    try:
        call = ast.parse(f"f({match.group(2)})", mode="eval").body
        positional = [ast.literal_eval(a) for a in call.args]
        named = {k.arg: ast.literal_eval(k.value) for k in call.keywords if k.arg}
    except (SyntaxError, ValueError, TypeError):
        return INVALID
    return name, {**dict(zip(_POSITIONS.get(name, ()), positional, strict=False)), **named}


def parse_action(text: str, spans: bool = True) -> tuple[str, dict] | None:
    """The first tool call of a reply as (tool, arguments): None when the reply makes
    none (it is the answer), INVALID when it starts one that cannot be read."""
    _, calls = parse_calls(text)  # Qwen 3's <tool_call><function=...> form
    if calls:
        name, arguments = calls[0]
        name = _NAMES.get(name, name)
        if isinstance(arguments.get("relations"), str):
            listed = _decode(arguments["relations"], 0)
            arguments["relations"] = (
                listed if isinstance(listed, list) else arguments["relations"].split("·")
            )
        return name, arguments
    match = _TOOL_CALL.search(text)  # Qwen 2.5's <tool_call> JSON
    if match:
        return _from_object(_decode(text, match.start(1)))
    match = _ACTION.search(text)
    if not match:
        return None
    action = match.group(1).strip()
    span = _SPAN_ACTION.fullmatch(action) if spans else None
    if span:
        from xgraph.programs import parse_path

        path = parse_path(span[2])
        if (
            path is None
            or not path[1]
            or "⁇" in span[2]
            or (span[1] == "follow" and len(path[1]) != 1)
        ):
            return INVALID
        start, relations = path
        arguments = (
            {"start": start, "relations": relations}
            if span[1] == "execute_path"
            else {"entity": start, "relation": relations[0]}
        )
        if span[3] is not None:
            arguments["offset"] = int(span[3])
        return span[1], arguments
    if action.startswith("{"):
        return _from_object(_decode(action, 0))
    return _from_function(action)


def delivered(text: str, answer_mode: str | None = None) -> str:
    """The reply with its last "Answer:" line written as the span "Answer: ⟦ label⟧"
    the scorers read, if it is plain text."""
    from xgraph.prompts import final_answer

    if answer_mode == "set":
        return text
    if final_answer(text) is not None:
        return text
    found = list(_ANSWER.finditer(text))
    if not found or "⟦" in found[-1].group(1) or "⟧" in found[-1].group(1):
        return text
    last = found[-1]
    return text[: last.start()] + f"Answer: ⟦ {last.group(1)}⟧" + text[last.end() :]


class KGTools:
    """The agent's tools over a KG store, or a store read through a KG version (see the
    module docstring); names are the relations' labels and aliases (relation_names_of).
    With paths (MatchedPaths), execute_path executes as matched execution does; with
    pages, follow and execute_path take an offset, and `paged` counts the calls that
    listed a later page. Feedback needs matched paths and also pages the failed
    frontier's writable relation labels. named, where given, is named_relations(names),
    shared by all requests' tools."""

    def __init__(
        self,
        read,
        names: dict[int, list[str]] | None = None,
        paths: MatchedPaths | None = None,
        pages: bool = False,
        named: dict[str, list[int]] | None = None,
        feedback: bool = False,
        answer_mode: str | None = None,
    ) -> None:
        if feedback and paths is None:
            raise ValueError("agent feedback needs matched paths")
        self.read, self.names, self.paths, self.pages = read, names, paths, pages or feedback
        self.feedback = feedback
        self.answer_mode = answer_mode
        self.paged = 0
        self.execution: dict | None = None
        self.reached: list[str] = []
        self._named = named if named is not None else named_relations(names)

    def call(self, name: str, arguments: dict) -> dict:
        self.execution = None
        self.reached = []
        tools = {
            "search": self.search,
            "relations": self.relations,
            "follow": self.follow,
            "execute_path": self.execute_path,
        }
        if name not in tools:
            return {"error": f"no tool named {name!r}; the tools are {', '.join(TOOLS)}"}
        taken = _POSITIONS[name] + (("offset",) if self.pages and name in PAGED else ())
        try:
            return tools[name](**{k: v for k, v in arguments.items() if k in taken})
        except TypeError:
            wanted = ", ".join(_POSITIONS[name])
            error = f"{name} takes {wanted}"
            return (
                self._path_failure(error, None)
                if self.feedback and name == "execute_path"
                else {"error": error}
            )

    def search(self, name: Any) -> dict:
        return {"entities": self.read.search_entities(str(name))}

    def relations(self, entity: Any) -> dict:
        item = self._entity(entity)
        if item is None:
            return {"error": f"no entity {entity!r}"}
        labels = sorted(set(self.read.relations_of([item]).values()))
        return {"entity": f"Q{item}", "label": self.read.label(item), "relations": labels}

    def follow(self, entity: Any, relation: Any, offset: Any = 0) -> dict:
        start = _offset(offset)
        if start is None:
            return {"error": f"offset must be a whole number of entities, not {offset!r}"}
        item = self._entity(entity)
        if item is None:
            return {"error": f"no entity {entity!r}"}
        reached: dict[int, str] = {}
        for prop in self._props(str(relation)):
            reached.update(self.read.objects_of([item], prop))
        if not reached:
            return {"error": f"Q{item} has no relation {str(relation)!r}", "count": 0}
        return self._listed(list(reached), start)

    def execute_path(self, start: Any, relations: Any, offset: Any = 0) -> dict:
        from xgraph.programs import execute

        first = _offset(offset)
        if first is None:
            error = f"offset must be a whole number of entities, not {offset!r}"
            return self._path_failure(error, None) if self.feedback else {"error": error}
        if isinstance(relations, str):
            relations = [r for r in re.split(r"[·,]", relations) if r.strip()]
        relations = [str(r).strip() for r in relations or []]
        start = str(start).strip()
        self.execution = {"plan": None, "reached": []}
        anchor = [start] if _ENTITY.fullmatch(start) else None
        if anchor and self.read.label(int(start[1:])) is None:
            return self._path_failure(f"no entity {start!r}", 0)
        if self.paths is not None:
            if not self.paths.starts(start, anchor):
                return self._path_failure(f"no entity named {start!r}", 0)

            def run(hops: list[str]) -> list[int]:
                return self.paths.reached(start, hops, anchor)  # type: ignore[union-attr]

        else:
            if not anchor and not self.read.items_named(start):
                return self._path_failure(f"no entity named {start!r}", 0)

            def run(hops: list[str]) -> list[int]:
                return execute(self.read, start, hops, anchor=anchor, names=self.names)

        if self.paths is not None:
            found = self.paths.execute(start, relations, anchor)
            self.execution = {
                "plan": {"start": found.plan[0], "relations": list(found.plan[1:])}
                if found.plan
                else None,
                "reached": [int(e[1:]) for e in found.reached],
            }
            if self.answer_mode == "set":
                audit = self.paths.executor.execute_set(
                    start, relations, anchor or self.paths.anchor, self.paths.version
                )
                self.execution.update(
                    reached=[int(e[1:]) for e in audit["reached"]], cap_events=audit["cap_events"]
                )
            reached = self.execution["reached"]
        else:
            reached = run(relations)
            self.execution["reached"] = reached
        if reached:
            return self._listed(reached, first, ranked=self.paths is not None)
        hop = next(k for k in range(1, len(relations) + 1) if not run(relations[:k]))
        error = f"the path reaches nothing at its relation {hop}, {relations[hop - 1]!r}"
        labels = self.paths.relations(start, relations[: hop - 1], anchor) if self.feedback else []
        return self._path_failure(error, hop, labels, first)

    def _path_failure(
        self, error: str, hop: int | None, labels: list[str] | None = None, offset: int = 0
    ) -> dict:
        result: dict[str, Any] = {"count": 0, "entities": [], "error": error}
        if self.feedback:
            labels = labels or []
            page: dict[str, Any] = {
                "count": len(labels),
                "relations": labels[offset : offset + SHOWN],
            }
            if offset:
                page["offset"] = offset
                self.paged += 1
            if offset + SHOWN < len(labels):
                page["next_offset"] = offset + SHOWN
            result.update(failed_hop=hop, frontier_relations=page)
        return result

    def _listed(self, items: list[int], offset: int = 0, ranked: bool = False) -> dict:
        """The SHOWN entities after offset, the best connected first unless already
        ranked; with pages, the offset and the next one while more remain."""
        if not ranked:
            items = sorted(items, key=lambda i: (-self.read.degree(i), i))
        self.reached = [f"Q{i}" for i in items]
        page = items[offset : offset + SHOWN]
        shown = [{"id": f"Q{i}", "label": self.read.label(i) or ""} for i in page]
        listed: dict[str, Any] = {"count": len(items), "entities": shown}
        if offset:
            listed["offset"] = offset
            self.paged += 1
        if self.pages and offset + SHOWN < len(items):
            listed["next_offset"] = offset + SHOWN
        return listed

    def _entity(self, entity: Any) -> int | None:
        text = str(entity).strip()
        if _ENTITY.fullmatch(text):
            item = int(text[1:])
            return item if self.read.label(item) is not None else None
        items = self.read.items_named(text)
        return min(items, key=lambda i: (-self.read.degree(i), i)) if items else None

    def _props(self, relation: str) -> list[int]:
        if self.names is not None:
            return sorted(set(self._named.get(normalize(relation), [])))
        return self.read.relation_named(relation)


def named_relations(names: dict[int, list[str]] | None) -> dict[str, list[int]]:
    """Each normalized relation label or alias with the properties it names; run_agent
    builds it once for all requests' tools, which read it without changing it."""
    named: dict[str, list[int]] = {}
    for prop, labels in (names or {}).items():
        for label in labels:
            named.setdefault(normalize(label), []).append(prop)
    return named


def _offset(offset: Any) -> int | None:
    """An offset as a number of entities; None where it is not a whole number >= 0."""
    if isinstance(offset, bool):
        return None
    try:
        number = int(str(offset).strip())
    except ValueError:
        return None
    return number if number >= 0 else None


class MatchedPaths:
    """execute_path by matched execution (xgraph.programs.MatchedExecutor) in one
    request's KG version: a start written as an id starts from that entity, a label
    from the request's anchor where it has one, as path mode's anchored requests do."""

    def __init__(self, executor, version: str | None = None, anchor: list[str] | None = None):
        self.executor, self.version, self.anchor = executor, version, anchor

    def starts(self, start: str, anchor: list[str] | None) -> bool:
        """Whether the start names an entity a path can start from."""
        return bool(self.executor.starts(start, anchor or self.anchor, self.version))

    def reached(self, start: str, relations: list[str], anchor: list[str] | None) -> list[int]:
        """The entities the path resolves to, in the engine's order."""
        found = self.execute(start, relations, anchor)
        return [int(e[1:]) for e in found.reached]

    def execute(self, start: str, relations: list[str], anchor: list[str] | None):
        """The executed plan and its whole capped frontier, before result paging."""
        return self.executor.execute(start, relations, anchor or self.anchor, self.version)

    def relations(self, start: str, relations: list[str], anchor: list[str] | None) -> list[str]:
        """Writable relations of the whole frontier, including every start rank."""
        return self.executor.frontier_labels(start, relations, anchor or self.anchor, self.version)


def demonstrations(messages: list[dict]) -> list[dict]:
    """A path-mode request's demonstrations: each pair of a user question and an assistant
    reply before the request's own question, as question, the path span's start and
    relations, and the answer (prompts.PATH_EXAMPLES, as python -m xgraph.prompts writes
    them into requests)."""
    from xgraph.programs import parse_path
    from xgraph.prompts import final_answer

    turns = [m for m in messages if m["role"] in ("user", "assistant")]
    shown = []
    for asked, reply in zip(turns[:-1:2], turns[1::2], strict=False):
        if asked["role"] != "user" or reply["role"] != "assistant":
            raise ValueError(f"a demonstration is a user and an assistant turn: {turns!r}")
        path, answer = parse_path(reply["content"]), final_answer(reply["content"])
        if path is None or answer is None:
            raise ValueError(f"a demonstration without a path span and answer: {reply!r}")
        start, relations = path
        shown.append(
            {"question": asked["content"], "start": start, "relations": relations,
             "answer": answer}
        )  # fmt: skip
    return shown


def system_prompt(max_calls: int = MAX_CALLS, pages: bool = False, feedback: bool = False) -> str:
    """The agent's instructions, with pages explained where they are on."""
    return (
        SYSTEM.format(budget=max_calls)
        + (PAGES if pages or feedback else "")
        + (FEEDBACK if feedback else "")
    )


def trajectory(demo: dict, tools: KGTools, spans: bool = False) -> list[dict]:
    """A demonstration as a tool trajectory: its question, an execute_path call with its
    path, the result (its answer, the best-connected entity with that label, or the label
    alone), and its answer line."""
    call = {"tool": "execute_path", "start": demo["start"], "relations": demo["relations"]}
    action = "Action: " + json.dumps(call, ensure_ascii=False)
    if spans:
        from xgraph.repair import plan_text

        action = "Action: execute_path " + plan_text(demo["start"], demo["relations"])
    items = tools.read.items_named(demo["answer"])
    if items:
        result = tools._listed([min(items, key=lambda i: (-tools.read.degree(i), i))])
    else:
        result = {"count": 1, "entities": [{"label": demo["answer"]}]}
    return [
        {"role": "user", "content": demo["question"]},
        {"role": "assistant", "content": action},
        {"role": "user", "content": "Observation: " + json.dumps(result, ensure_ascii=False)},
        {"role": "assistant", "content": f"Answer: ⟦ {demo['answer']}⟧"},
    ]


def agent_messages(
    question: str,
    read,
    anchor: list[str] | None = None,
    max_calls: int = MAX_CALLS,
    demos: list[dict] | None = None,
    pages: bool = False,
    tools: KGTools | None = None,
    feedback: bool = False,
    spans: bool = False,
    answer_mode: str | None = None,
) -> list[dict]:
    """The conversation the agent starts from: the instructions, a worked example (or the
    demonstrations as tool trajectories, through tools), and the question, with an
    anchored question's entities named after it."""
    asked = question
    if anchor:
        named = ", ".join(f"{read.label(int(e[1:])) or e} ({e})" for e in anchor)
        asked += f"\nThe question's entities: {named}"
    shown = list(EXAMPLE)
    if spans:
        shown = [
            {**m, "content": "Action: execute_path ⟦ Mount Kilimanjaro · country · continent⟧"}
            if m["content"].startswith('Action: {"tool": "execute_path"')
            else m
            for m in shown
        ]
    if demos is not None:
        tools = tools if tools is not None else KGTools(read, pages=pages)
        shown = [m for demo in demos for m in trajectory(demo, tools, spans)]
    instructions = system_prompt(max_calls, pages, feedback) + (SPAN_CALLS if spans else "")
    if answer_mode == "set":
        instructions = (
            instructions.replace(
                'line "Answer: ⟦ label⟧" with the answer entity\'s label.',
                "answer set. " + SET_INSTRUCTION + "\n" + SET_FORMAT,
            )
            .replace("(Wikidata)", "(Freebase)")
            .replace("Wikidata", "KG")
        )
    return [
        {
            "role": "system",
            "content": instructions,
        },
        *shown,
        {"role": "user", "content": asked},
    ]


def run_agent(
    llm: Any,
    requests: list[dict],
    store,
    sampling: Any,
    max_calls: int = MAX_CALLS,
    names: dict[int, list[str]] | None = None,
    versions: dict | None = None,
    executor: Any = None,
    pages: bool = False,
    feedback: bool = False,
    *,
    mask: bool = False,
    force_path: bool = False,
    round_params: Any = None,
    schema: str | None = None,
    schema_params: Any = None,
    max_model_len: int | None = None,
) -> list[dict]:
    """One record per request (id, question, and optionally anchor and kg_version, as
    xgraph.generate.walk_requests gives them), all requests a round at a time. A request
    with a kg_version reads the store through it, versions[name] (toolkg.version_stores).
    A request with `demonstrations` (see demonstrations) is shown them as tool
    trajectories instead of the worked example; with an executor (MatchedExecutor),
    execute_path is matched execution; with pages, results take offsets. Feedback needs
    that executor, supplies failed-hop relation pages, and enforces another valid call
    before accepting an answer after a failed path, while budget remains."""
    if feedback and executor is None:
        raise ValueError("agent feedback needs matched paths")
    if mask and (executor is None or round_params is None):
        raise ValueError("agent mask needs matched paths and per-round sampling parameters")
    if schema not in (None, "static", "turn"):
        raise ValueError(f"unknown agent schema {schema!r}; expected static or turn")
    if schema and (
        mask or force_path or executor is None or schema_params is None or round_params is None
    ):
        raise ValueError(
            "agent schema needs matched paths and native structured outputs; "
            "it excludes the span mask and forced path opener"
        )
    spans = mask or force_path
    record_plans = spans or schema is not None
    vocabulary = executor.relation_labels() if schema else []
    states = []
    named = named_relations(names)
    if max_model_len is None:
        config = getattr(getattr(llm, "llm_engine", None), "model_config", None)
        max_model_len = getattr(config, "max_model_len", None)
    tokenizer = llm.get_tokenizer() if max_model_len is not None else None
    for request in requests:
        read = store_for(request, store, versions)
        paths = None
        if executor is not None:
            paths = MatchedPaths(executor, request.get("kg_version"), request.get("anchor"))
        tools = KGTools(read, names, paths, pages, named, feedback, request.get("answer_mode"))
        states.append(
            {
                "request": request,
                "tools": tools,
                "messages": agent_messages(
                    request["question"],
                    read,
                    request.get("anchor"),
                    max_calls,
                    request.get("demonstrations"),
                    pages,
                    tools,
                    feedback,
                    spans,
                    request.get("answer_mode"),
                ),
                "calls": 0,
                "cap_events": [],
                "rounds": 0,
                "prompt": 0,
                "generated": 0,
                "actions": [],
                "turns": [],
                "final": False,
                "done": None,
                "finish_reason": None,
                "retry_required": False,
                "failed_path": None,
                "failures": 0,
                "retries": 0,
                "retry_changed_plan": False,
                "answer_refusals": 0,
                "unmasked_calls": 0,
                "executed_plans": [],
                "known": set(request.get("anchor") or []),
                "schema_bypass_calls": 0,
            }
        )
    while pending := [s for s in states if s["done"] is None]:
        from xgraph.generate import CONTINUED

        prefill = FORCED_PATH if force_path and pending[0]["rounds"] == 0 else None
        conversations = [
            s["messages"] + ([{"role": "assistant", "content": prefill}] if prefill else [])
            for s in pending
        ]
        params = (
            [round_params(s["request"], prefill) for s in pending]
            if round_params is not None
            else sampling
        )
        if max_model_len is not None:
            admitted = []
            for i, messages in enumerate(conversations):
                encoded = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    enable_thinking=False,
                    **(CONTINUED if prefill else {"add_generation_prompt": True}),
                )
                ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
                param = params[i] if isinstance(params, list) else params
                max_tokens = param["max_tokens"] if isinstance(param, dict) else param.max_tokens
                if len(ids) + max_tokens > max_model_len:
                    pending[i]["done"] = ""
                    pending[i]["finish_reason"] = "context_limit"
                else:
                    admitted.append(i)
            conversations = [conversations[i] for i in admitted]
            pending = [pending[i] for i in admitted]
            if isinstance(params, list):
                params = [params[i] for i in admitted]
            if not pending:
                continue
        if schema:
            for state, param in zip(pending, params, strict=True):
                labels = vocabulary
                if schema == "turn" and state["known"]:
                    labels = executor.entity_relation_labels(
                        state["known"], state["request"].get("kg_version")
                    )
                tag = {
                    "type": "structural_tag",
                    "triggers": ["Action:"],
                    "structures": [
                        {"begin": "Action: ", "schema": tool_schema(labels), "end": "\n"}
                    ],
                }
                structured = schema_params(structural_tag=json.dumps(tag, ensure_ascii=False))
                if isinstance(param, dict):
                    param["structured_outputs"] = structured
                else:
                    param.structured_outputs = structured
        outputs = llm.chat(
            conversations,
            params,
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=False,
            **(CONTINUED if prefill else {}),
        )
        for state, output in zip(pending, outputs, strict=True):
            completion = output.outputs[0]
            text = (prefill or "") + completion.text
            state["rounds"] += 1
            state["prompt"] += len(output.prompt_token_ids)
            state["generated"] += len(completion.token_ids)
            state["turns"].append(
                {
                    "text": text,
                    "prompt_tokens": len(output.prompt_token_ids),
                    "generated_tokens": len(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                    **({"prefill": prefill} if prefill else {}),
                }
            )
            action = None if state["final"] else parse_action(text, spans=spans)
            if action is None:
                if feedback and state["retry_required"] and state["calls"] < max_calls:
                    if state["answer_refusals"] >= REFUSALS:
                        # A model that keeps answering instead of retrying keeps its own
                        # answer: the feedback adds retries but never takes answers away.
                        state["done"] = delivered(text, state["request"].get("answer_mode"))
                        state["finish_reason"] = "retry_refused"
                        continue
                    state["answer_refusals"] += 1
                    state["messages"].extend(
                        [
                            {"role": "assistant", "content": text.strip()},
                            {
                                "role": "user",
                                "content": "Answer refused: retry with a tool call after "
                                "the failed execute_path before answering. "
                                f"{max_calls - state['calls']} calls remain.",
                            },
                        ]
                    )
                    continue
                state["done"] = delivered(text, state["request"].get("answer_mode"))
                state["finish_reason"] = completion.finish_reason
                continue
            state["messages"].append({"role": "assistant", "content": text.strip()})
            if state["calls"] >= max_calls:
                final = (
                    SET_INSTRUCTION + "\n" + SET_FORMAT
                    if state["request"].get("answer_mode") == "set"
                    else FINAL
                )
                state["messages"].append({"role": "user", "content": final})
                state["final"] = True  # the next reply is the answer
                continue
            name, arguments = action
            line = _ACTION.search(text)
            span_call = bool(spans and line and _SPAN_ACTION.fullmatch(line[1].strip()))
            if mask and name in PAGED and not span_call:
                state["unmasked_calls"] += 1
            if schema and name in PAGED and not (line and line[1].lstrip().startswith("{")):
                state["schema_bypass_calls"] += 1
            if action == INVALID:
                result = {"error": 'write one call as Action: {"tool": ..., ...} on its own line'}
            else:
                if span_call and name == "follow":
                    path_args = {"start": arguments["entity"], "relations": [arguments["relation"]]}
                    if "offset" in arguments:
                        path_args["offset"] = arguments["offset"]
                    result = state["tools"].call("execute_path", path_args)
                else:
                    result = state["tools"].call(name, arguments)
            state["calls"] += 1
            if action != INVALID and state["tools"].execution is not None:
                state["cap_events"].extend(
                    {**e, "action": len(state["actions"])}
                    for e in state["tools"].execution.get("cap_events", [])
                )
            logged = {"tool": name, "arguments": arguments}
            if schema:
                state["known"].update(state["tools"].reached)
                state["known"].update(e["id"] for e in result.get("entities", []) if e.get("id"))
                if result.get("entity"):
                    state["known"].add(result["entity"])
            if record_plans:
                logged["result"] = result
                if (name == "execute_path" or (span_call and name == "follow")) and (
                    state["tools"].execution is not None
                ):
                    state["executed_plans"].append(
                        {
                            "action": len(state["actions"]),
                            **state["tools"].execution,
                        }
                    )
            if feedback:
                logged["result"] = result
                valid = name in TOOLS and all(k in arguments for k in _POSITIONS[name])
                if name in PAGED and _offset(arguments.get("offset", 0)) is None:
                    valid = False
                if (
                    name == "execute_path"
                    and "error" in result
                    and result.get("failed_hop") is None
                ):
                    valid = False
                if valid:
                    state["retry_required"] = False
                if name == "execute_path":
                    previous = state["failed_path"]
                    if valid and previous is not None:
                        changed = _plan(arguments) != _plan(state["actions"][previous]["arguments"])
                        logged.update(retry_of=previous, plan_changed=changed)
                        state["retries"] += 1
                        state["retry_changed_plan"] |= changed
                    failed = "error" in result or result.get("count") == 0
                    logged["failed"] = failed
                    if failed:
                        state["failures"] += 1
                        state["retry_required"] = True
                    if valid:
                        state["failed_path"] = len(state["actions"]) if failed else None
            state["actions"].append(logged)
            observation = "Observation: " + json.dumps(result, ensure_ascii=False)
            state["messages"].append({"role": "user", "content": observation})
    return [
        {
            "id": s["request"]["id"],
            "arm": "agent",
            **(
                {"kg_version": s["request"]["kg_version"]} if s["request"].get("kg_version") else {}
            ),
            "text": s["done"],
            **set_answer_fields(
                s["request"],
                "" if s["finish_reason"] == "context_limit" else s["turns"][-1]["text"],
            ),
            **({"cap_events": s["cap_events"]} if s["request"].get("answer_mode") == "set" else {}),
            "finish_reason": s["finish_reason"],
            **(
                {"error": "context_limit", "declined": True}
                if s["finish_reason"] == "context_limit"
                else {}
            ),
            "tool_calls": s["calls"],
            "budget_spent": s["final"] or (feedback and s["calls"] >= max_calls),
            **(
                {
                    "agent_feedback": True,
                    "failures": s["failures"],
                    "retries": s["retries"],
                    "retry_changed_plan": s["retry_changed_plan"],
                    "answer_refusals": s["answer_refusals"],
                }
                if feedback
                else {}
            ),
            "pages": s["tools"].paged,
            **(
                {
                    "masked": mask,
                    "unmasked_calls": s["unmasked_calls"],
                    "executed_plans": s["executed_plans"],
                    **(
                        {k: s["executed_plans"][-1][k] for k in ("plan", "reached")}
                        if s["executed_plans"]
                        else {"plan": None, "reached": []}
                    ),
                    **({"agent_force_path": True} if force_path else {}),
                }
                if record_plans
                else {}
            ),
            **(
                {"agent_schema": schema, "schema_bypass_calls": s["schema_bypass_calls"]}
                if schema
                else {}
            ),
            "rounds": s["rounds"],
            "actions": s["actions"],
            "turns": s["turns"],
            "token_ids": [],
            "prompt_tokens": s["prompt"],
            "generated_tokens": s["generated"],
        }
        for s in states
    ]


def tool_schema(labels: list[str]) -> dict:
    """A standard JSON schema for agent calls, with one enum fixed for the whole turn.

    A structural tag applies it only after Action:, preserving the ordinary agent
    prompt and its free final answer. Empty support leaves lookups available.
    """
    from xgraph.oracle import MAX_PATH_RELATIONS

    string = {"type": "string"}
    relation = {"type": "string", "enum": labels}
    fields = {"search": {"name": string}, "relations": {"entity": string}}
    if labels:
        fields.update(
            {
                "follow": {"entity": string, "relation": relation},
                "execute_path": {
                    "start": string,
                    "relations": {
                        "type": "array",
                        "items": relation,
                        "minItems": 1,
                        "maxItems": MAX_PATH_RELATIONS,
                    },
                },
            }
        )
    options = []
    for name, arguments in fields.items():
        properties = {"tool": {"type": "string", "enum": [name]}, **arguments}
        if name in PAGED:
            properties["offset"] = {"type": "integer", "minimum": 0}
        options.append(
            {
                "type": "object",
                "properties": properties,
                "required": ["tool", *arguments],
                "additionalProperties": False,
            }
        )
    return {"anyOf": options}


def _plan(arguments: dict) -> tuple[str, tuple[str, ...]]:
    """A written path's start and relations; offsets are pages, not plan changes."""
    relations = arguments.get("relations") or []
    if isinstance(relations, str):
        relations = [r for r in re.split(r"[·,]", relations) if r.strip()]
    return str(arguments.get("start", "")).strip(), tuple(str(r).strip() for r in relations)
