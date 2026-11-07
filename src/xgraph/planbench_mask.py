"""Opt-in host-side planning masks, separate from the GPU-native KG engine.

A vocabulary token trie intersects the byte trie of applicable actions. A token
can cross an action boundary: effects commit at the closing parenthesis, then
its remaining bytes read the next state's support. Thus alternate tokenizations
and merged newlines do not silently lose legal continuations. EOS is available
at every action boundary, independent of the goal in the primary model-stop arm.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from xgraph.planbench_pddl import Action, Problem, State, check_plan, load_problem

try:
    from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor as _Base
except ImportError:
    _Base = object


@dataclass(eq=False)
class _Node:
    children: dict[int, _Node] = field(default_factory=dict)
    tokens: list[int] = field(default_factory=list)
    action: Action | None = None


def _insert(root: _Node, data: bytes) -> _Node:
    node = root
    for byte in data:
        if byte not in node.children:
            node.children[byte] = _Node()
        node = node.children[byte]
    return node


class Vocabulary:
    """Shared immutable trie of served token bytes; built once per worker/tokenizer."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.root = _Node()
        for token in range(tokenizer.vocab_size):
            data = tokenizer.token_bytes(token)
            if data:
                _insert(self.root, data).tokens.append(token)


@lru_cache(maxsize=2)
def vocabulary_for(tokenizer: Any) -> Vocabulary:
    return Vocabulary(tokenizer)


@dataclass(frozen=True)
class _Cursor:
    state: State
    node: _Node | None
    phase: str  # boundary, action, newline, ended
    count: int = 0


class PlanningMask:
    """Live applicable-action language with incremental replay and bounded caches.

    Output consists of lower-case PDDL action lines, separated by a single
    newline. EOS may follow the closing parenthesis or its following newline,
    or end the empty plan. The mask never exposes a goal test in model-stop mode.
    """

    def __init__(
        self, problem: Problem, tokenizer: Any, eos_token_ids, *, max_actions=64, stop="model"
    ):
        if stop not in ("model", "goal-reached-control"):
            raise ValueError(f"unknown planning stop rule: {stop}")
        if type(max_actions) is not int or max_actions < 1:
            raise ValueError("max_actions must be a positive integer")
        self.vocabulary = (
            tokenizer if isinstance(tokenizer, Vocabulary) else vocabulary_for(tokenizer)
        )
        self.tokenizer = self.vocabulary.tokenizer
        ids = tuple(eos_token_ids)
        if not ids or any(
            type(t) is not int or not 0 <= t < self.tokenizer.vocab_size for t in ids
        ):
            raise ValueError("planning mask needs explicit served-model EOS token ids")
        self.eos = frozenset(ids)
        self.problem, self.max_actions, self.stop = problem, max_actions, stop
        self._roots: OrderedDict[State, _Node] = OrderedDict()
        self._prefix: tuple[int, ...] = ()
        self._cursor = self._initial()
        self._sets: OrderedDict[_Cursor, frozenset[int]] = OrderedDict()

    def _root(self, state: State) -> _Node:
        if state not in self._roots:
            root = _Node()
            for action in self.problem.applicable(state):
                _insert(root, action.text.encode()).action = action
            self._roots[state] = root
            if len(self._roots) > 64:
                self._roots.popitem(last=False)
        self._roots.move_to_end(state)
        return self._roots[state]

    def _initial(self) -> _Cursor:
        return _Cursor(self.problem.init, self._root(self.problem.init), "boundary")

    def _byte(self, cursor: _Cursor, byte: int) -> _Cursor | None:
        if cursor.phase == "ended":
            return None
        if self.stop == "goal-reached-control" and self.problem.goal(cursor.state):
            return None
        if cursor.phase == "newline":
            return (
                _Cursor(cursor.state, self._root(cursor.state), "boundary", cursor.count)
                if (byte == 10)
                else None
            )
        if cursor.count >= self.max_actions:
            return None
        child = cursor.node.children.get(byte)
        if child is None:
            return None
        if child.action is not None:
            return _Cursor(child.action.apply(cursor.state), None, "newline", cursor.count + 1)
        return _Cursor(cursor.state, child, "action", cursor.count)

    def _token(self, cursor: _Cursor, token: int) -> _Cursor | None:
        if type(token) is not int or not 0 <= token < self.tokenizer.vocab_size:
            return None
        if token in self.eos:
            return (
                _Cursor(cursor.state, None, "ended", cursor.count)
                if cursor.phase in ("boundary", "newline", "ended")
                else None
            )
        data = self.tokenizer.token_bytes(token)
        if not data:
            return None
        for byte in data:
            cursor = self._byte(cursor, byte)
            if cursor is None:
                return None
        return cursor

    def allowed(self, prefix) -> frozenset[int]:
        prefix = tuple(prefix)
        if prefix[: len(self._prefix)] == self._prefix:
            cursor, tail = self._cursor, prefix[len(self._prefix) :]
        else:
            cursor, tail = self._initial(), prefix
        for token in tail:
            cursor = self._token(cursor, token)
            if cursor is None:
                raise ValueError(f"disallowed planning token {token} in prefix {prefix!r}")
        self._prefix, self._cursor = prefix, cursor
        if cursor not in self._sets:
            allowed = set(self.eos) if cursor.phase in ("boundary", "newline", "ended") else set()
            pending = [(self.vocabulary.root, cursor)]
            while pending:
                node, current = pending.pop()
                allowed.update(t for t in node.tokens if t not in self.eos)
                for byte, child in node.children.items():
                    advanced = self._byte(current, byte)
                    if advanced is not None:
                        pending.append((child, advanced))
            if not allowed:
                raise ValueError("planning token vocabulary cannot complete this legal prefix")
            self._sets[cursor] = frozenset(allowed)
            if len(self._sets) > 4096:
                self._sets.popitem(last=False)
        self._sets.move_to_end(cursor)
        return self._sets[cursor]


def planning_arguments(problem, tokenizer, eos_token_ids, max_actions=64, stop="model") -> dict:
    return {
        "planbench": {
            "domain_pddl": problem.domain_text,
            "problem_pddl": problem.problem_text,
            "tokenizer_fingerprint": tokenizer.fingerprint,
            "eos_token_ids": list(eos_token_ids),
            "max_actions": max_actions,
            "stop": stop,
        }
    }


def audit_records(cases, records, tokenizer, eos_token_ids):
    lookup = {c.id: c for c in cases}
    count = tokens_checked = 0
    for record in records:
        if record["arm"] not in ("mask", "mask-goal-reached-control"):
            continue
        case = lookup[record["id"]]
        turns = record["turns"]
        if len(turns) != 1 or turns[0].get("masked") is not True:
            raise ValueError(f"mask bypass or missing turn evidence: {case.id}")
        turn = turns[0]
        mask = PlanningMask(
            case.problem,
            tokenizer,
            eos_token_ids,
            max_actions=record["budget"]["max_actions"],
            stop=("model" if record["arm"] == "mask" else "goal-reached-control"),
        )
        tokens = turn["token_ids"]
        prefix = []
        for token in tokens:
            if token not in mask.allowed(prefix):
                raise ValueError(f"disallowed saved planning token {token} for {case.id}")
            prefix.append(token)
        if turn["finish_reason"] == "stop" and not set(eos_token_ids) & mask.allowed(tokens):
            raise ValueError(f"model stopped inside a partial action: {case.id}")
        text = b"".join(tokenizer.token_bytes(t) or b"" for t in tokens if t not in mask.eos)
        if text.decode() != turn["text"]:
            raise ValueError(f"token bytes and saved text disagree: {case.id}")
        if record["complete"] and not check_plan(case.problem, "\n".join(record["plan"])).valid:
            raise ValueError(f"complete mask plan is inapplicable: {case.id}")
        count += 1
        tokens_checked += len(tokens)
    if not count:
        raise ValueError("mask audit has no saved masked records")
    return {
        "status": "passed",
        "mask_records": count,
        "tokens_checked": tokens_checked,
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "scope": "saved token admission, termination and text fidelity; no GPU timing",
    }


def _payload(params):
    payload = (params.extra_args or {}).get("planbench")
    if payload is None:
        return None
    if not isinstance(payload, dict) or set(payload) != {
        "domain_pddl",
        "problem_pddl",
        "tokenizer_fingerprint",
        "eos_token_ids",
        "max_actions",
        "stop",
    }:
        raise ValueError("invalid planbench request fields")
    if any(
        not isinstance(payload[k], str) or not payload[k]
        for k in ("domain_pddl", "problem_pddl", "tokenizer_fingerprint")
    ) or any(len(payload[k]) > 100_000 for k in ("domain_pddl", "problem_pddl")):
        raise ValueError("planbench needs bounded PDDL text and a tokenizer fingerprint")
    eos = payload["eos_token_ids"]
    if not isinstance(eos, list) or not eos or any(type(t) is not int or t < 0 for t in eos):
        raise ValueError("planbench needs explicit EOS token ids")
    if type(payload["max_actions"]) is not int or not 1 <= payload["max_actions"] <= 512:
        raise ValueError("planbench max_actions must be an integer in [1, 512]")
    if payload["stop"] not in ("model", "goal-reached-control"):
        raise ValueError("unknown planbench stop rule")
    return payload


class PlanningLogitsProcessor(_Base):
    """vLLM V2 processor with host replay; opted-out requests remain untouched.

    Histories are copied to the CPU. This is deliberately an eager, synchronized
    first version, not a GPU-native planning engine. Speculation is unsupported.
    Slot reuse resets all planning state; shortened histories replay from init.
    """

    def __init__(self, vllm_config, req_states, *, tokenizer=None):
        if getattr(vllm_config, "speculative_config", None) is not None:
            raise ValueError("planning masks do not support speculative decoding")
        if tokenizer is None:
            from xgraph.tokenizer import ServedTokenizer

            path = os.environ.get("PLANBENCH_TOKENIZER")
            if not path:
                raise ValueError("PLANBENCH_TOKENIZER must name the served tokenizer.json")
            tokenizer = ServedTokenizer.from_file(path)
        self.vocabulary = vocabulary_for(tokenizer)
        self.states = req_states
        self.slots = {}

    @classmethod
    def validate_params(cls, sampling_params):
        payload = _payload(sampling_params)
        if payload is not None:
            load_problem(payload["domain_pddl"], payload["problem_pddl"])

    def add_request(self, slot, sampling_params):
        self.slots.pop(slot, None)
        payload = _payload(sampling_params)
        if payload is None:
            return False
        if payload["tokenizer_fingerprint"] != self.vocabulary.tokenizer.fingerprint:
            raise ValueError("planbench request tokenizer fingerprint differs from worker")
        problem = load_problem(payload["domain_pddl"], payload["problem_pddl"])
        self.slots[slot] = PlanningMask(
            problem,
            self.vocabulary,
            payload["eos_token_ids"],
            max_actions=payload["max_actions"],
            stop=payload["stop"],
        )
        return True

    def apply_staged_writes(self):
        pass  # All planning state is on the host.

    def apply(self, logits, ctx):
        if len(ctx.expanded_idx_mapping) != len(ctx.idx_mapping):
            raise ValueError("planning masks do not support speculative rows")
        for row, slot in enumerate(ctx.expanded_idx_mapping.tolist()):
            if slot not in self.slots:
                continue
            start = int(self.states.prompt_len.gpu[slot])
            end = int(self.states.total_len.gpu[slot])
            prefix = self.states.all_token_ids.gpu[slot, start:end].tolist()
            ids = sorted(self.slots[slot].allowed(prefix))
            if ids[-1] >= logits.shape[1]:
                raise ValueError("planbench allowed token exceeds logits width")
            values = logits[row, ids].clone()
            logits[row].fill_(float("-inf"))
            logits[row, ids] = values
        return logits
