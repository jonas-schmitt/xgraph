"""Chain-of-Relations over KG stores, with externally loaded released prompts.

The default implements the released DFS and its separate reasoning decisions. Prompt
text is never distributed here; CorPrompts reads a local YAML file as data. The original
offline reconstruction remains available explicitly as mode="reconstructed". Backend,
delivery and resource-limit deviations are documented in docs/reproducing.md.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from xgraph.prompts import (
    SET_FORMAT,
    SET_INSTRUCTION,
    final_answer,
    parse_answer_set,
    set_answer_fields,
)
from xgraph.score import normalize
from xgraph.toolkg import KGStore, VersionedStore, store_for

WIDTH = 3
MAX_DEPTH = 3
MAX_CALLS = 8
EXPLORATION_TEMPERATURE = 0.3
REASONING_TEMPERATURE = 0.1
RELATION_THRESHOLD = 1000  # run.py overrides CoRAgent's constructor default (500).
QUERY_LIMIT = 1000
UPSTREAM_DEVIATIONS = (
    "local vLLM backbone/chat template replaces released API model; thinking disabled",
    "converted SQLite KG replaces SPARQL; missing RDF/qualifier facts remain unavailable",
    "first supplied linked topic only; no gold-chain start selection",
    "deterministic PID/QID query caps replace unordered SPARQL LIMIT",
    "release filter/direct comma-semicolon parsing and model-trusted answers retained",
    "context overflow falls back or returns Unknown; all issued calls and retries charged",
)
_PLACEHOLDER = re.compile(r"\{\{\s*([^{}]*?)\s*\}\}")


def temperature_profile(exploration: float, reasoning: float) -> str:
    if exploration == reasoning == 0.01:
        return "paper"
    if (exploration, reasoning) == (EXPLORATION_TEMPERATURE, REASONING_TEMPERATURE):
        return "released"
    return "custom"


@dataclass(frozen=True)
class CorPrompts:
    """Validated external prompt blocks and the SHA-256 of their source bytes."""

    path: Path
    sha256: str
    blocks: dict[str, dict[str, str]]

    @classmethod
    def load(cls, path: str | Path) -> CorPrompts:
        import yaml

        path = Path(path)
        try:
            source = path.read_bytes()
            content = yaml.safe_load(source.decode("utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise ValueError(f"CoR prompts {path}: {error}") from error
        if not isinstance(content, dict):
            raise ValueError(f"CoR prompts {path}: expected a YAML mapping")
        blocks = {}
        for stage, keys in {
            "rank": ("relation_pruning", "relation pruning", "extract_relation"),
            "reasoning": ("reasoning",),
            "filter": ("filter",),
            "fallback": ("generate_directly",),
        }.items():
            block = next((content[k] for k in keys if isinstance(content.get(k), dict)), None)
            if (
                not block
                or not isinstance(block.get("user_prompt"), str)
                or not block["user_prompt"].strip()
            ):
                raise ValueError(f"CoR prompts {path}: {keys[0]} needs a nonempty user_prompt")
            allowed = {"question"} | (
                {"relations"}
                if stage == "rank"
                else {"reasoning_path"}
                if stage in ("reasoning", "filter")
                else set()
            )
            blocks[stage] = {}
            for role in ("system", "user"):
                template = block.get(f"{role}_prompt")
                if template is None and role == "system":
                    continue
                if not isinstance(template, str):
                    raise ValueError(f"CoR prompts {path}: {keys[0]}.{role}_prompt must be text")
                unknown = set(_PLACEHOLDER.findall(template)) - allowed
                if unknown:
                    raise ValueError(
                        f"CoR prompts {path}: {keys[0]} has unknown placeholders {sorted(unknown)}"
                    )
                blocks[stage][role] = template
        return cls(path, hashlib.sha256(source).hexdigest(), blocks)

    def messages(self, stage: str, **values: str) -> list[dict]:
        return [
            {"role": role, "content": _PLACEHOLDER.sub(lambda m: values[m[1]], template)}
            for role, template in self.blocks[stage].items()
        ]


PROMPTS = {
    "start": """Identify the entity from which the question's relation chain starts.
Reply only with JSON: {"start": "English Wikidata label or QID"}. Do not answer the
question or name intermediate entities.""",
    "rank": """Plan a Chain-of-Relations to answer the question. The KG executes the chain;
intermediate entities remain implicit. Rank at most three of the offered outgoing
relations in the order to explore, most promising first. Reply only with JSON:
{"relations": ["P123", "P456"]}. Use the offered relation ids (labels and aliases are
also accepted). Use {"relations": []} if this branch cannot answer the question.
If the current chain reaches possible answers (including the start itself), reply
{"stop": true}; a separate filter will enforce the question's remaining constraints.
Do not invent relations or intermediate entities.""",
    "filter": """The KG executed the given relation chain and returned the candidates.
Use the question, chain, and your own knowledge to enforce any remaining constraints.
Keep only candidates that answer the question, most likely first. Reply only with JSON:
{"keep": ["Q123", "Q456"]}, using candidate ids. Use {"keep": []} if none answers it.
Do not introduce entities outside the candidate set.""",
    "fallback": """Answer the question directly using your own knowledge. Give one short
answer, ending with the line "Answer: ⟦ label⟧". If you cannot answer, use
"Answer: ⟦ Unknown⟧".""",
}


def _reply(text: str) -> tuple[Any, list[str]]:
    """Read a JSON value, a fenced value, an embedded object/list, or plain text."""
    source = text.strip()
    repairs = []
    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\s*```", source, re.DOTALL)
    if fenced:
        source = fenced[1].strip()
        repairs.append("markdown_fence")
    try:
        return json.loads(source), repairs
    except json.JSONDecodeError:
        pass
    # Do not mine strings or nested lists out of an invalid top-level JSON object.
    at = re.search(r"[\[{]", source)
    if at:
        try:
            value, _ = json.JSONDecoder().raw_decode(source, at.start())
            return value, [*repairs, "embedded_json"]
        except json.JSONDecodeError:
            repairs.append("invalid_json")
    return source, [*repairs, "plain_text"]


def _names(value: Any) -> list[str] | None:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else None
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return [v.strip() for v in value if v.strip()]
    return None


def _mentioned(text: str, offered: dict[str, list[str]]) -> list[str]:
    """Offered ids named in plain text, in reply order; prefer the longest label.

    Whole-word boundaries keep Q1 distinct from Q10. Overlapping labels select the
    longer name, so 'New York City' does not also select 'New York'.
    """
    hits = []
    for item, labels in offered.items():
        for label in [item, *labels]:
            if label.strip():
                for match in re.finditer(r"(?<!\w)" + re.escape(label) + r"(?!\w)", text, re.I):
                    hits.append((match.start(), match.end(), item))
    chosen, spans = [], []
    for start, end, item in sorted(hits, key=lambda h: (h[0], h[0] - h[1], h[2])):
        if any(start < b and a < end and (start, end) != (a, b) for a, b in spans):
            continue
        spans.append((start, end))
        if item not in chosen:
            chosen.append(item)
    return chosen


def parse_selection(stage: str, text: str) -> tuple[dict | None, list[str]]:
    """Normalize start/rank/filter formats without resolving names outside the KG."""
    value, repairs = _reply(text)
    key = {"start": "start", "rank": "relations", "filter": "keep"}[stage]
    if isinstance(value, dict):
        if stage == "rank":
            stop = value.get("stop", False)
            if not isinstance(stop, bool):
                return None, repairs
            if stop:
                return {"stop": True}, repairs
        raw = value.get(key)
    else:
        raw = value
        repairs.append("missing_object")
    names = _names(raw)
    if names is None:
        return None, repairs
    if stage == "rank" and len(names) == 1 and names[0].casefold().rstrip(".") == "stop":
        return {"stop": True}, [*repairs, "stop_text"]
    if stage == "start":
        cleaned = []
        for name in names:
            label = re.sub(r"^(?:Start|Answer):\s*", "", name, flags=re.I).strip()
            path = re.search(
                r"(?:<PATH>\s*|^#+\s*(?:Reasoning Path|Relation Chain):\s*(?:<PATH>\s*)?)"
                r"([^\r\n<>]+?)\s*->",
                label,
                re.I | re.M,
            )
            if path:
                label = path[1].strip()
                repairs.append("path_start")
            span = re.fullmatch(r"⟦\s*([^⟦⟧⁇]+?)\s*⟧", label)
            label = span[1].strip() if span else label
            ids = re.findall(r"\bQ\d+\b", label)
            if len(ids) > 1 and not re.sub(r"\bQ\d+\b", "", label).strip(",; \n\t"):
                cleaned.extend(ids)
            else:
                cleaned.append(label)
        if cleaned != names:
            repairs.append("start_text")
        names = cleaned
    if isinstance(raw, str) and stage != "start":
        repairs.append("scalar_selection")
    elif isinstance(raw, list) and stage == "start":
        repairs.append("start_list")
    return {key: names}, repairs


def _path(chain: tuple[int, ...]) -> list[str]:
    return [f"P{p}" for p in chain]


def _answer(label: str) -> str | None:
    """Render a label only when the scorer reads the same nonempty label back."""
    if not label.strip() or any(c in label for c in ("⟦", "⟧", "⁇", "\n", "\r")):
        return None
    text = f"Answer: ⟦ {label}⟧"
    return text if final_answer(text) == label else None


def parse_fallback(text: str) -> tuple[str | None, list[str]]:
    """Read the first answer, including inline Answer text and repeated delimiters.

    An explicit Answer takes precedence over spans in the rationale. A repeated
    opener ends its answer text, rather than making a later nested Unknown win.
    Return scorer-readable delivery and the repairs; the caller retains the raw reply.
    """
    value, repairs = _reply(text)
    if "embedded_json" in repairs and re.search(r"\bAnswer:\s*", text, re.IGNORECASE):
        value, repairs = text, ["plain_text"]
    if isinstance(value, dict):
        value = value.get("answer", value.get("label"))
        repairs.append("answer_object")
    if isinstance(value, list):
        values = value
        repairs.append("answer_list")
    else:
        values = [value]
    for value in values:
        if not isinstance(value, str):
            continue
        markers = list(re.finditer(r"\bAnswer:\s*", value, re.IGNORECASE))
        if markers:
            labels = []
            for marker in markers:
                source = value[marker.end() :].split("\n", 1)[0].strip()
                labels.append(
                    re.split(r"[⟦⟧]", source[1:], maxsplit=1)[0].strip()
                    if source.startswith("⟦")
                    else source
                )
        else:
            labels = [
                m[1].strip()
                for m in re.finditer(r"⟦[^\S\r\n]*([^⟦⟧\r\n]+?)(?=[⟦⟧\r\n]|$)", value)
                if " · " not in m[1]
            ] or [value.strip()]
        for label in labels:
            if answer := _answer(label):
                return answer, [*repairs, "answer_text"] if text.strip() != answer else []
    return None, repairs


@dataclass
class _Node:
    chain: tuple[int, ...]
    frontier: dict[int, str]
    ranked: list[int] | None = None


class _Search:
    def __init__(self, request: dict, read, names: dict[int, list[str]], max_depth: int):
        self.request, self.read, self.names, self.max_depth = request, read, names, max_depth
        self.stack: list[_Node] = []
        self.menu: dict[int, str] = {}
        self.trace: list[dict] = []
        self.cap_events: list[dict] = []
        self.turns: list[dict] = []
        self.start: list[str] = []
        self.path: list[str] = []
        self.candidates: list[str] = []
        self.selected: list[str] = []
        self.fallback = False
        self.fallback_reason: str | None = None
        self.budget_spent = False
        self.context_limited = False
        self.text: str | None = None
        self.finish_reason: str | None = None
        self.stage = "start"
        question = request.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"request {request['id']!r} needs a nonempty question")
        anchors = request.get("anchor") or request.get("topics") or []
        if not isinstance(anchors, list) or any(
            not isinstance(e, str) or not re.fullmatch(r"Q\d+", e) for e in anchors
        ):
            raise ValueError(f"request {request['id']!r} needs anchor/topics as a list of QIDs")
        if anchors:
            frontier = {}
            for entity in anchors:
                item = int(entity[1:])
                label = read.label(item)
                if label is None:
                    self.trace.append({"event": "start", "error": f"no label for {entity}"})
                else:
                    frontier[item] = label
            self.begin(frontier)

    def begin(self, frontier: dict[int, str]) -> None:
        self.start = [f"Q{i}" for i in sorted(frontier)]
        if frontier:
            self.stack = [_Node((), frontier)]
            self.stage = "rank"
        else:
            self.direct("no_start")

    def direct(self, reason: str) -> None:
        self.fallback, self.fallback_reason, self.stage = True, reason, "fallback"
        self.trace.append({"event": "fallback", "reason": reason})

    def backtrack(self, reason: str, error: str | None = None) -> None:
        node = self.stack.pop()
        event = {"event": "backtrack", "path": _path(node.chain), "reason": reason}
        if error:
            event["error"] = error
        self.trace.append(event)
        self.stage = "rank"

    def stop(self, reason: str) -> None:
        node = self.stack[-1]
        self.path = _path(node.chain)
        self.candidates = [f"Q{i}" for i in sorted(node.frontier)]
        self.trace.append({"event": "stop", "path": self.path, "reason": reason})
        self.stage = "filter"

    def prepare(self) -> tuple[str, list[dict]]:
        """Execute saved branches until the next rank, filter, start or fallback call."""
        while self.stage == "rank":
            if not self.stack:
                self.direct("search_exhausted")
                break
            node = self.stack[-1]
            if node.ranked is None:
                if len(node.chain) >= self.max_depth:
                    self.stop("depth")
                    break
                self.menu = self.read.relations_of(list(node.frontier))
                break
            if not node.ranked:
                self.backtrack("exhausted")
                continue
            prop = node.ranked.pop(0)
            chain = (*node.chain, prop)
            frontier = self.read.objects_of(list(node.frontier), prop)
            self.trace.append({"event": "expand", "path": _path(chain), "count": len(frontier)})
            if frontier:
                self.stack.append(_Node(chain, frontier))
            else:
                self.trace.append({"event": "backtrack", "path": _path(chain), "reason": "empty"})

        payload: dict = {"stage": self.stage, "question": self.request["question"]}
        if self.stage in ("rank", "filter"):
            node = self.stack[-1]
            payload["start"] = [
                {"id": entity, "label": self.read.label(int(entity[1:]))} for entity in self.start
            ]
            payload["chain"] = [
                {"id": f"P{p}", "label": self.read.relation_label(p)} for p in node.chain
            ]
            if self.stage == "rank":
                payload["relations"] = [
                    {
                        "id": f"P{p}",
                        "label": label,
                        "aliases": [n for n in self.names.get(p, []) if n != label],
                    }
                    for p, label in sorted(self.menu.items())
                ]
            else:
                payload["candidates"] = [
                    {"id": f"Q{i}", "label": label} for i, label in sorted(node.frontier.items())
                ]
        instructions = PROMPTS[self.stage]
        if self.request.get("answer_mode") == "set":
            instructions += "\n" + SET_INSTRUCTION
            if self.stage == "fallback":
                instructions = SET_INSTRUCTION + "\n" + SET_FORMAT
        return self.stage, [
            {"role": "system", "content": instructions},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    def accept(self, stage: str, text: str) -> None:
        if stage == "fallback":
            if self.request.get("answer_mode") == "set":
                self.text = "Answer: " + json.dumps(
                    parse_answer_set(text, bare=True), ensure_ascii=False
                )
                return
            answer, repairs = parse_fallback(text)
            if repairs:
                self.trace.append({
                    "event": "parse_repair", "stage": stage, "turn": len(self.turns) - 1,
                    "repairs": repairs, "parsed": {"answer": answer},
                })  # fmt: skip
            if answer is None:
                self.trace.append({"event": "fallback", "error": "no readable final answer"})
                answer = "Answer: ⟦ Unknown⟧"
            self.text = answer
            return
        data, repairs = parse_selection(stage, text)
        if data is not None and repairs:
            self.trace.append({
                "event": "parse_repair", "stage": stage, "turn": len(self.turns) - 1,
                "repairs": repairs, "parsed": data,
            })  # fmt: skip
        if stage == "start":
            names = data.get("start") if data else None
            if not names:
                self.trace.append({"event": "start", "error": "expected a start label or QID"})
                self.direct("invalid_start")
                return
            frontier = {}
            for name in names:
                items = (
                    [int(name[1:])] if re.fullmatch(r"Q\d+", name) else self.read.items_named(name)
                )
                items = [i for i in items if self.read.label(i) is not None]
                if not items:
                    self.trace.append({"event": "start", "error": f"no entity named {name!r}"})
                    continue
                item = min(items, key=lambda i: (-self.read.degree(i), i))
                frontier[item] = self.read.label(item)
            self.begin(frontier)
            return
        node = self.stack[-1]
        if stage == "rank":
            if not data or not isinstance(data.get("stop", False), bool):
                self.backtrack("invalid_rank", "expected a JSON relation list or boolean stop")
                return
            if data.get("stop"):
                self.stop("model")
                return
            ranked = data.get("relations")
            if not isinstance(ranked, list) or any(not isinstance(p, str) for p in ranked):
                self.backtrack("invalid_rank", "expected a JSON relation list")
                return
            if "plain_text" in repairs or "scalar_selection" in repairs:
                mentioned = _mentioned("\n".join(ranked), {
                    f"P{p}": [label, *self.names.get(p, [])] for p, label in self.menu.items()
                })  # fmt: skip
                if ranked and not mentioned:
                    self.backtrack("invalid_rank", "no offered relation named in reply")
                    return
                ranked = mentioned
            chosen = []
            for name in ranked:
                matches = [
                    p
                    for p, label in sorted(self.menu.items())
                    if normalize(name) in {
                        normalize(f"P{p}"), normalize(label),
                        *(normalize(n) for n in self.names.get(p, [])),
                    }
                ]  # fmt: skip
                if not matches:
                    self.trace.append({"event": "rank", "error": f"unavailable relation {name!r}"})
                for prop in matches:
                    if prop not in chosen:
                        chosen.append(prop)
            node.ranked = chosen[:WIDTH]
            self.trace.append(
                {"event": "rank", "path": _path(node.chain), "relations": _path(tuple(node.ranked))}
            )
            return
        kept = data.get("keep") if data else None
        if not isinstance(kept, list) or any(not isinstance(e, str) for e in kept):
            self.backtrack("invalid_filter", "expected a JSON list of candidate ids to keep")
            return
        if "plain_text" in repairs or "scalar_selection" in repairs:
            kept = _mentioned("\n".join(kept), {
                f"Q{i}": [label] for i, label in node.frontier.items()
            })  # fmt: skip
        valid = []
        for name in dict.fromkeys(kept):
            named = set(self.read.items_named(name)) if name not in self.candidates else set()
            matches = (
                [name]
                if name in self.candidates
                else [
                    f"Q{i}"
                    for i, label in sorted(node.frontier.items())
                    if normalize(name) == normalize(label) or i in named
                ]
            )
            if not matches:
                self.trace.append({"event": "filter", "error": f"not a candidate: {name!r}"})
            if matches and name not in self.candidates:
                self.trace.append({
                    "event": "parse_repair", "stage": stage, "turn": len(self.turns) - 1,
                    "repairs": ["candidate_label"], "parsed": {"keep": matches},
                })  # fmt: skip
            for entity in matches:
                if _answer(node.frontier[int(entity[1:])]) is None:
                    self.trace.append(
                        {"event": "filter", "error": f"unwritable answer label: {entity}"}
                    )
                elif entity not in valid:
                    valid.append(entity)
        if not valid and kept == [] and ("plain_text" in repairs or "scalar_selection" in repairs):
            self.trace.append({"event": "filter", "error": "no candidate named in reply"})
        self.trace.append({"event": "filter", "path": self.path, "keep": valid})
        if valid:
            self.selected = valid
            self.text = (
                "Answer: "
                + json.dumps([node.frontier[int(e[1:])] for e in valid], ensure_ascii=False)
                if self.request.get("answer_mode") == "set"
                else _answer(node.frontier[int(valid[0][1:])])
            )
        else:
            self.backtrack("filter_rejected")

    def record(self) -> dict:
        return {
            "id": self.request["id"],
            "arm": "cor",
            **(
                {"kg_version": self.request["kg_version"]} if self.request.get("kg_version") else {}
            ),
            "text": self.text,
            **set_answer_fields(self.request, self.text or ""),
            **({"cap_events": self.cap_events} if self.request.get("answer_mode") == "set" else {}),
            "finish_reason": self.finish_reason,
            "start": self.start,
            "path": self.path,
            "candidates": self.candidates,
            "selected": self.selected,
            "trace": self.trace,
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
            "budget_spent": self.budget_spent,
            "context_limited": self.context_limited,
            "calls": len(self.turns),
            "rounds": len(self.turns),
            "turns": self.turns,
            "replies": [t["text"] for t in self.turns],
            "token_ids": [],
            "prompt_tokens": sum(t["prompt_tokens"] for t in self.turns),
            "generated_tokens": sum(t["generated_tokens"] for t in self.turns),
        }


def _directed_path(chain: tuple[int, ...]) -> list[str]:
    return [f"{'^' if p < 0 else ''}P{abs(p)}" for p in chain]


class _ReleasedSearch(_Search):
    """Pending relation chains form a LIFO stack; reasoning controls expansion."""

    def __init__(self, request, read, names, max_depth, prompts):
        self.prompts = prompts
        self.current = _Node((), {})
        self.root = 0
        self.answers: list[str] = []
        self.action: str | None = None
        self.empty_replies = 0
        super().__init__(request, read, names, max_depth)
        if self.stage == "start":
            self.direct("no_start")

    def begin(self, frontier: dict[int, str]) -> None:
        if not frontier:
            self.direct("no_start")
            return
        # agent.answer uses topic_entities[0], not their union.
        starts = self.request.get("anchor") or self.request.get("topics")
        self.root = int(starts[0][1:])
        if self.root not in frontier:
            self.direct("no_start")
            return
        self.start = [f"Q{self.root}"]
        self.stack = [_Node((), {self.root: frontier[self.root]})]
        self.stage = "next"

    def _resume(self, reason: str) -> None:
        self.trace.append(
            {"event": "backtrack", "path": _directed_path(self.current.chain), "reason": reason}
        )
        self.stage = "next"

    def _entities(self) -> list[int]:
        items = [self.root]
        for prop in self.current.chain:
            items = self.read.neighbours_of(items, abs(prop), reverse=prop < 0)
        return items

    def _chain_text(self, candidates: bool = False) -> str:
        node = f"TopicEntity({self.read.label(self.root)})"
        lines = []
        for index, prop in enumerate(self.current.chain, 1):
            other = f"e{index}"
            label = self.read.relation_label(abs(prop)) or f"P{abs(prop)}"
            left, right = (other, node) if prop < 0 else (node, other)
            lines.append(f"{left} {label} {right} .")
            node = other
        if candidates and self.current.frontier:
            labels = ", ".join(self.current.frontier.values())
            lines.append(f"{node} candidate.target TargetEntity({labels}) .")
        return "\n".join(lines)

    def _relation_menu(self) -> dict[int, str]:
        items = self._entities()
        menu = {}
        # Mirror Wikidata's metadata-label filtering, retaining a direction's original
        # menu if the filter would remove everything. The store already excludes RDF
        # schema predicates: every relation is a PID.
        noise = (
            "id",
            "identifier",
            "code",
            "url",
            "username",
            "handle",
            "catalog",
            "index",
            "registry",
            "authority",
            "vocabulary",
            "thesaurus",
            "api endpoint",
            "image",
            "logo",
            "flag image",
            "coat of arms",
            "official website",
        )
        for reverse in (False, True):
            raw = self.read.relations_of(items, reverse=reverse)
            if len(raw) > QUERY_LIMIT:
                self.cap_events.append(
                    {
                        "stage": "cor_relation_query",
                        "before": len(raw),
                        "kept": QUERY_LIMIT,
                        "reverse": reverse,
                    }
                )
            props = sorted(raw, key=lambda p: f"P{p}")[:QUERY_LIMIT]
            props = [p for p in props if p not in (373, 910)]
            clean = [
                p
                for p in props
                if not any(
                    re.search(r"\b" + re.escape(word) + r"\b", raw[p], re.I) for word in noise
                )
            ]
            for prop in clean or props:
                signed = -prop if reverse else prop
                if self.current.chain and signed == -self.current.chain[-1]:
                    continue
                menu[signed] = raw[prop]
        self.available = set(menu)
        if len(menu) > RELATION_THRESHOLD:
            self.cap_events.append(
                {"stage": "cor_relation_menu", "before": len(menu), "kept": RELATION_THRESHOLD}
            )
        return dict(list(menu.items())[:RELATION_THRESHOLD])

    def prepare(self) -> tuple[str, list[dict]]:
        while self.stage in ("next", "rank"):
            if self.stage == "next":
                if not self.stack:
                    self.direct("search_exhausted")
                    break
                self.current = self.stack.pop()
                if self.current.chain:
                    # SPARQL caps query results, not intermediate join variables.
                    items = self._entities()
                    if len(items) > QUERY_LIMIT:
                        self.cap_events.append(
                            {
                                "stage": "cor_query_results",
                                "before": len(items),
                                "kept": QUERY_LIMIT,
                                "path": _directed_path(self.current.chain),
                            }
                        )
                    items = items[:QUERY_LIMIT]
                    self.current.frontier = {
                        i: label
                        for i in items
                        if (label := self.read.label(i)) and label != "UnName_Entity"
                    }
                    self.path = _directed_path(self.current.chain)
                    self.candidates = [f"Q{i}" for i in self.current.frontier]
                    self.trace.append(
                        {"event": "expand", "path": self.path, "count": len(self.candidates)}
                    )
                    if self.current.frontier:
                        self.stage = "reasoning"
                        break
                    if len(self.current.chain) >= self.max_depth:
                        self._resume("empty_at_depth")
                        continue
                self.stage = "rank"
            self.menu = self._relation_menu()
            if self.menu:
                break
            self._resume("no_relations")

        values = {"question": self.request["question"]}
        if self.stage == "rank":
            base = self._chain_text().replace("\n", " ")
            node = (
                f"e{len(self.current.chain)}"
                if self.current.chain
                else f"TopicEntity({self.read.label(self.root)})"
            )
            snippets = []
            for reverse in (False, True):
                labels = [label for p, label in self.menu.items() if (p < 0) == reverse]
                if labels:
                    triple = f"?x ?relation {node}" if reverse else f"{node} ?relation ?x"
                    snippets.append(
                        f"# {base + ' ' if base else ''}{triple}\n{'; '.join(labels)};\n"
                    )
            values["relations"] = "".join(snippets)
        elif self.stage in ("reasoning", "filter"):
            values["reasoning_path"] = self._chain_text(candidates=True)
        messages = self.prompts.messages(self.stage, **values)
        if self.request.get("answer_mode") == "set":
            messages[0] = {
                **messages[0],
                "content": messages[0]["content"] + "\n" + SET_INSTRUCTION,
            }
            if self.stage in ("filter", "fallback"):
                messages[0]["content"] += "\n" + SET_FORMAT
            elif self.stage == "reasoning":
                messages[0]["content"] += (
                    "\nKeep the decision object; put every answer in its answer array."
                )
        return self.stage, messages

    def _rank(self, text: str) -> list[int]:
        def order(value: dict) -> int:
            try:
                rank = int(value.get("rank", 0))
            except (ValueError, TypeError, OverflowError):
                rank = 0
            return rank if rank > 0 else 10**9

        data, _ = _reply(text)
        ranked = []
        if isinstance(data, dict):
            values = data.get("relation_by_relevance")
            values = [values] if isinstance(values, str) else values
            if not isinstance(values, list) or not values:
                values = list(data)
        elif isinstance(data, list):
            values = sorted((v for v in data if isinstance(v, dict)), key=order)
        else:
            values = []
        for value in values:
            if isinstance(value, dict):
                value = value.get(
                    "relation_id",
                    value.get(
                        "relation", value.get("id", next((k for k in value if k != "rank"), ""))
                    ),
                )
            elif isinstance(value, list):
                value = value[0] if value else ""
            if not isinstance(value, str):
                continue
            available = {abs(p) for p in self.menu}
            matches = [p for p in available if value == f"P{p}"] or [
                p for p in available if value == self.read.relation_label(p)
            ]
            if len(matches) == 1 and matches[0] not in ranked:
                ranked.append(matches[0])
        # Width counts relation ids; an id offered in both directions adds two states.
        return [signed for p in ranked[:WIDTH] for signed in (p, -p) if signed in self.available]

    @staticmethod
    def _answers(text: str) -> list[str]:
        # The released filter/direct parser accepts JSON lists and legacy braces.
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("answer", []), list):
            source = ", ".join(str(v).strip() for v in data.get("answer", []) if str(v).strip())
        else:
            # Match the first braced legacy answer, as the released tools do.
            match = re.search(r"\{([^}]*)\}", text)
            source = match[1] if match else ""
        return [v.strip() for v in re.split(r"[,;]", source) if v.strip()]

    def _deliver(self, answers: list[str], action: str) -> None:
        self.answers, self.action = answers, action
        self.selected = (
            [
                f"Q{i}"
                for name in answers
                for i, label in self.current.frontier.items()
                if name == label
            ]
            if action != "generate_directly"
            else []
        )
        self.text = (
            "Answer: " + json.dumps(answers, ensure_ascii=False)
            if self.request.get("answer_mode") == "set"
            else (_answer(answers[0]) if answers else None) or "Answer: ⟦ Unknown⟧"
        )

    def accept(self, stage: str, text: str) -> None:
        if not text and self.empty_replies < 2:
            self.empty_replies += 1
            self.trace.append({"event": "retry", "stage": stage, "reason": "empty_reply"})
            return
        self.empty_replies = 0
        if stage == "fallback":
            answers = (
                parse_answer_set(text, bare=True)
                if self.request.get("answer_mode") == "set"
                else self._answers(text)
            )
            self._deliver(answers, "generate_directly")
        elif stage == "rank":
            chosen = self._rank(text)
            self.trace.append(
                {
                    "event": "rank",
                    "path": _directed_path(self.current.chain),
                    "relations": _directed_path(tuple(chosen)),
                    **({"error": "no offered relation selected"} if not chosen else {}),
                }
            )
            self.stack.extend(_Node((*self.current.chain, p), {}) for p in reversed(chosen))
            self.stage = "next"
        elif stage == "filter":
            answers = (
                parse_answer_set(text, bare=True)
                if self.request.get("answer_mode") == "set"
                else self._answers(text)
            )
            self.trace.append({"event": "filter", "path": self.path, "answers": answers})
            if answers:
                self._deliver(answers, "Constraint")
            else:
                # Released agent breaks the entire DFS here, rather than resuming siblings.
                self.direct("filter_empty")
        else:
            data, _ = _reply(text)
            decision = (
                str(data.get("decision", "")).strip().lower() if isinstance(data, dict) else ""
            )
            decision = {"yes": "stop", "no": "forward", "constraint": "filter"}.get(
                decision, decision
            )
            self.trace.append({"event": "reasoning", "path": self.path, "decision": decision})
            if decision not in ("stop", "filter", "forward", "backtrack"):
                self.trace[-1]["error"] = "unrecognized reasoning decision"
            if not text:
                self.trace.append(
                    {
                        "event": "error",
                        "stage": stage,
                        "error": "empty reasoning response after retries",
                    }
                )
                self._deliver([], "ERROR")
            elif decision == "stop":
                raw = data.get("answer", [])
                answers = (
                    [str(v).strip() for v in raw if str(v).strip()] if isinstance(raw, list) else []
                )
                self._deliver(answers or list(self.current.frontier.values()), "Stop")
            elif decision == "filter":
                self.stage = "filter"
            elif decision == "forward" and len(self.current.chain) < self.max_depth:
                self.stage = "rank"
            else:
                self._resume("depth" if decision == "forward" else decision or "invalid_reasoning")

    def record(self) -> dict:
        return {
            **super().record(),
            "cor_mode": "upstream",
            "answers": self.answers,
            "action": self.action,
            "cor_prompts": {"path": str(self.prompts.path), "sha256": self.prompts.sha256},
        }


def run_cor(
    llm: Any,
    requests: list[dict],
    store: KGStore | VersionedStore,
    sampling: Any,
    max_calls: int | None = None,
    names: dict[int, list[str]] | None = None,
    versions: dict | None = None,
    max_depth: int = MAX_DEPTH,
    max_prompt_tokens: int | None = None,
    lora_request: Any = None,
    *,
    prompts: CorPrompts | str | Path | None = None,
    mode: str = "upstream",
    seed: int = 0,
    exploration_temperature: float = EXPLORATION_TEMPERATURE,
    reasoning_temperature: float = REASONING_TEMPERATURE,
) -> list[dict]:
    """One record per request (id, question, optional anchor/topics and kg_version).

    Upstream mode requires external prompts and uses the first topic entity. There is
    no total call cap unless max_calls is supplied, in which case one call is reserved
    for fallback. Reconstructed mode retains its historical start generation and cap.
    With max_prompt_tokens, render the chat template before each call; an oversized
    prompt goes directly to fallback. If even the direct question cannot fit, return
    an explicit Unknown answer with finish_reason length and no unissued call charged.
    """
    if mode not in ("upstream", "reconstructed"):
        raise ValueError(f"unknown CoR mode {mode!r}")
    for name, value in (
        ("exploration_temperature", exploration_temperature),
        ("reasoning_temperature", reasoning_temperature),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative, got {value!r}")
    if mode == "reconstructed" and max_calls is None:
        max_calls = MAX_CALLS
    for name, value in (("max_calls", max_calls), ("max_depth", max_depth)):
        if name == "max_calls" and value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if max_prompt_tokens is not None and max_prompt_tokens <= 0:
        raise ValueError(f"max_prompt_tokens must be positive, got {max_prompt_tokens!r}")
    if mode == "upstream":
        if prompts is None:
            raise ValueError("upstream CoR needs a local prompt file (--cor-prompts)")
        prompts = prompts if isinstance(prompts, CorPrompts) else CorPrompts.load(prompts)
        states = [
            _ReleasedSearch(r, store_for(r, store, versions), names or {}, max_depth, prompts)
            for r in requests
        ]
    else:
        if prompts is not None:
            raise ValueError("--cor-prompts needs upstream CoR mode")
        states = [
            _Search(r, store_for(r, store, versions), names or {}, max_depth) for r in requests
        ]
    tokenizer = llm.get_tokenizer() if max_prompt_tokens is not None else None
    while pending := [s for s in states if s.text is None]:
        jobs = []
        for state in pending:
            if max_calls is not None and len(state.turns) >= max_calls:
                state.budget_spent = True
                state.trace.append({"event": "call_limit", "stage": state.stage})
                state._deliver([], "generate_directly")
                continue
            stage, messages = state.prepare()
            if max_calls is not None and stage != "fallback" and len(state.turns) >= max_calls - 1:
                state.budget_spent = True
                state.direct("calls")
                stage, messages = state.prepare()
            if tokenizer is not None:
                while len(tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
                )) > max_prompt_tokens:  # fmt: skip
                    state.context_limited = True
                    state.trace.append({"event": "context_limit", "stage": stage})
                    if stage == "fallback":
                        state.text, state.finish_reason = "Answer: ⟦ Unknown⟧", "length"
                        break
                    state.direct("context_limit")
                    stage, messages = state.prepare()
            if state.text is None:
                jobs.append((state, stage, messages))
        if not jobs:
            continue
        params = sampling
        if mode == "upstream":
            params = []
            for state, stage, _ in jobs:
                one = copy(sampling) if sampling is not None else SimpleNamespace(max_tokens=512)
                one.temperature = (
                    exploration_temperature if stage == "rank" else reasoning_temperature
                )
                one.seed = state.request.get("seed", seed)
                params.append(one)
        outputs = llm.chat(
            [messages for _, _, messages in jobs],
            params,
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=False,
            **({"lora_request": lora_request} if lora_request is not None else {}),
        )
        for (state, stage, _), output in zip(jobs, outputs, strict=True):
            completion = output.outputs[0]
            state.finish_reason = completion.finish_reason
            state.turns.append(
                {
                    "stage": stage,
                    "text": completion.text,
                    "prompt_tokens": len(output.prompt_token_ids),
                    "generated_tokens": len(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                }
            )
            state.accept(stage, completion.text)
    return [
        dict(
            s.record(),
            cor_mode=mode,
            **(
                {
                    "deviations": [
                        *UPSTREAM_DEVIATIONS,
                        *(
                            [f"depth {max_depth} is a transfer from released depth {MAX_DEPTH}"]
                            if max_depth != MAX_DEPTH
                            else []
                        ),
                    ]
                }
                if mode == "upstream"
                else {}
            ),
            cor_config={
                "depth": max_depth,
                "temperature_exploration": exploration_temperature if mode == "upstream" else 0.0,
                "temperature_reasoning": reasoning_temperature if mode == "upstream" else 0.0,
                "temperature_profile": temperature_profile(
                    exploration_temperature, reasoning_temperature
                )
                if mode == "upstream"
                else "reconstructed",
            },
            **({"seed": s.request.get("seed", seed)} if mode == "upstream" else {}),
        )
        for s in states
    ]


def _question(request: dict) -> str:
    if isinstance(request.get("question"), str) and request["question"].strip():
        return request["question"]
    for message in reversed(request.get("messages", [])):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    raise ValueError(f"request {request['id']!r} needs a question or user message")


def linked_requests(panel: list[dict], linked: Iterable[dict]) -> list[dict]:
    """Copy an existing linker's topics by id, retaining the panel's order and prompts.

    The source is iterative requests built with --linked, which uses the same linker
    as retrieval. This join uses no annotations, gold chains or new model calls.
    """
    wanted = {r["id"] for r in panel}
    if len(wanted) != len(panel):
        raise ValueError("duplicate panel request id")
    topics, seen = {}, set()
    questions = {r["id"]: _question(r) for r in panel}
    for request in linked:
        qid = request["id"]
        if qid in seen:
            raise ValueError(f"duplicate linked request id {qid!r}")
        seen.add(qid)
        if qid not in wanted:
            continue
        entities = request.get("topics")
        if not isinstance(entities, list) or any(
            not isinstance(e, str) or not re.fullmatch(r"Q\d+", e) for e in entities
        ):
            raise ValueError(f"linked request {qid!r} needs topics as a list of QIDs")
        if _question(request).strip() != questions[qid].strip():
            raise ValueError(f"linked request {qid!r} has a different question from the panel")
        topics[qid] = list(entities)
    if missing := wanted - topics.keys():
        raise ValueError(f"missing linked request ids: {sorted(missing)}")
    for request in panel:
        if request.get("anchor"):
            raise ValueError(f"panel request {request['id']!r} already carries anchors")
    return [{**r, "topics": topics[r["id"]]} for r in panel]


def replay_fallbacks(records: list[dict]) -> tuple[list[dict], dict]:
    """Reparse saved fallback replies only; never invent missing search continuations.

    Historical model usage and raw turns are retained. The summary counts readable
    answers recovered from Unknown, not correctness or an online rerun's quality.
    """

    def unknown(text: str) -> bool:
        label = final_answer(text)
        return not label or normalize(label) == "unknown"

    result, seen = [], set()
    recovered: Counter[str] = Counter()
    available = unavailable = changed = 0
    for record in records:
        qid = record["id"]
        if qid in seen:
            raise ValueError(f"duplicate record id {qid!r}")
        seen.add(qid)
        if record.get("arm") != "cor":
            raise ValueError(f"record {qid!r} is not a cor record")
        updated = record
        if record.get("fallback"):
            turns = record.get("turns", [])
            if not turns or turns[-1].get("stage") != "fallback":
                unavailable += 1
            else:
                available += 1
                answer, repairs = parse_fallback(turns[-1]["text"])
                answer = answer or "Answer: ⟦ Unknown⟧"
                changed += answer != record["text"]
                if unknown(record["text"]) and not unknown(answer):
                    recovered[record.get("fallback_reason") or "unspecified"] += 1
                trace = list(record.get("trace", []))
                if repairs:
                    trace.append({
                        "event": "parse_repair", "stage": "fallback", "turn": len(turns) - 1,
                        "repairs": repairs, "parsed": {"answer": answer},
                    })  # fmt: skip
                updated = {
                    **record,
                    "text": answer,
                    "trace": trace,
                    "replay": {"kind": "fallback_parsing", "original_text": record["text"]},
                }
        result.append(updated)
    return result, {
        "requests": len(records),
        "unknown_before": sum(unknown(r["text"]) for r in records),
        "unknown_after": sum(unknown(r["text"]) for r in result),
        "recovered_from_unknown": sum(recovered.values()),
        "recovered_by_reason": dict(sorted(recovered.items())),
        "changed_answers": changed,
        "available_fallbacks": available,
        "unavailable_fallbacks": unavailable,
        "model_calls": 0,
        "scope": "saved fallback replies only; search trajectories were not rerun",
    }


def main(argv: list[str] | None = None) -> None:
    """CPU preparation and saved-reply replay; generation remains xgraph.generate."""
    import argparse

    from xgraph.cost import StageTimer
    from xgraph.provenance import provenance

    parser = argparse.ArgumentParser(prog="python -m xgraph.cor")
    commands = parser.add_subparsers(dest="command", required=True)
    requests = commands.add_parser("requests", help="copy existing linked topics onto a panel")
    requests.add_argument("--requests", type=Path, required=True, help="panel requests (JSONL)")
    requests.add_argument(
        "--linked-requests",
        type=Path,
        required=True,
        help="iterative requests built with --linked (JSONL)",
    )
    replay = commands.add_parser(
        "replay", help="reparse saved fallback replies without model calls"
    )
    replay.add_argument(
        "--records", type=Path, required=True, help="CoR generation records (JSONL)"
    )
    for command in (requests, replay):
        command.add_argument("--out", type=Path, required=True, help="output records (JSONL)")
    args = parser.parse_args(argv)
    timer = StageTimer()

    def read(path: Path):
        with path.open(encoding="utf-8") as lines:
            for line in lines:
                if line.strip():
                    yield json.loads(line)

    if args.command == "requests":
        rows = linked_requests(list(read(args.requests)), read(args.linked_requests))
        summary = {"requests": len(rows), "with_topics": sum(bool(r["topics"]) for r in rows)}
        inputs = [args.requests, args.linked_requests]
    else:
        rows, summary = replay_fallbacks(list(read(args.records)))
        inputs = [args.records]
        source = Path(str(args.records) + ".provenance.json")
        if source.exists():
            inputs.append(source)
            summary["source_generation"] = json.loads(source.read_text(encoding="utf-8"))
    if args.out.resolve() in {p.resolve() for p in inputs}:
        parser.error("--out must differ from the input paths")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as written:
        for row in rows:
            written.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary.update(
        {
            "artifact_kind": f"xgraph.cor_{args.command}",
            "version": 1,
            "seed": None,
            "provenance": provenance([*inputs, Path(__file__)]),
            **timer.summary(),
        }
    )
    Path(str(args.out) + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in {"provenance", "source_generation"}}
        )
    )


if __name__ == "__main__":
    main()
