"""Graph-constrained Reasoning: a bounded KG-Trie then general-model synthesis.

This is a reconstruction, not published-score reproduction. Sampling is the
historical default. Beam mode searches joint path-and-hypothesis sequences on the
host using raw vLLM logprobs, with default vLLM or opt-in ordinary/group HF semantics.
A second stage synthesizes them. Every issued scoring and generation request is charged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import resource
import sys
import time
import traceback
from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from heapq import nsmallest
from itertools import chain
from pathlib import Path
from typing import Any

from xgraph.cor import PROMPTS as COR_PROMPTS
from xgraph.cor import parse_fallback, parse_selection
from xgraph.method_prompts import PromptSource
from xgraph.path_trie import (
    CLOSER,
    MAX_PATHS,
    OPENER,
    KGPath,
    TokenTrie,
    collect_paths,
    serialize_path,
    trie_arguments,
)
from xgraph.prompts import (
    SET_FORMAT,
    SET_INSTRUCTION,
    final_answer,
    parse_answer_set,
    set_answer_fields,
)
from xgraph.score import normalize
from xgraph.toolkg import KGStore, VersionedStore, store_for

PROMPTS = {
    "start": COR_PROMPTS["start"],
    "paths": """Find a reasoning path in the knowledge graph that helps answer the question.
Start immediately with exactly one path in this format:
<PATH> start entity -> relation -> intermediate entity -> relation -> answer entity </PATH>
A path may be shorter. Include each entity and relation, using the KG's canonical labels.
Keep the spaces inside both path markers and around every arrow. The trie constrains
the entire path, including both markers. After </PATH>, give your hypothesis answer
under "# Answer:" in plain text. The hypothesis is unconstrained.
Do not write another path.""",
    "synthesis": """Answer the question using the generated reasoning paths and hypothesis
answers. A hypothesis is a model suggestion, not a verified KG answer. A generation marked
invalid or truncated may lack a complete grounded path. Give one short final answer ending
with the line "Answer: ⟦ label⟧". If you cannot answer, use "Answer: ⟦ Unknown⟧".""",
}
STAGES = ("start", "paths", "synthesis")


@dataclass(frozen=True)
class GCRPrompts:
    """Paper Figures 10/11, with literal instructions from the pinned MIT release.

    Upstream Python never executes. The paper's hash-prefixed synthesis headers
    differ from the release's headers. Evidence retains every joint hypothesis.
    """

    source: PromptSource

    @classmethod
    def load(cls, path: str | Path) -> GCRPrompts:
        return cls(
            PromptSource.load(
                path,
                (
                    "PathGenerationWithAnswerPromptBuilder.ZERO_SHOT_PROMPT",
                    "PromptBuilder.SAQ_RULE_INSTRUCTION",
                    "PromptBuilder.EACH_LINE",
                ),
            )
        )

    def metadata(self) -> dict:
        return {**self.source.metadata(), "profile": "paper_zero_shot"}

    def messages(self, stage, question, starts, samples):
        question += "" if question.endswith("?") else "?"
        strings = self.source.strings
        if stage == "paths":
            text = strings["PathGenerationWithAnswerPromptBuilder.ZERO_SHOT_PROMPT"].format(
                question=question, entities=",".join(starts)
            )
        elif stage == "synthesis":
            evidence = "\n".join((s["path"] or "") + " " + s["hypothesis"] for s in samples)
            text = (
                f"# Reasoning Paths:\n{evidence}\n\n# Question:\n{question}\n\n"
                + strings["PromptBuilder.SAQ_RULE_INSTRUCTION"]
                + strings["PromptBuilder.EACH_LINE"]
            )
        else:
            raise ValueError(f"paper GCR prompts require linked topics; unsupported stage {stage}")
        return [{"role": "user", "content": text}]


class GCRScreenBudget:
    """Abort a sequential screen using elapsed workload wall time per question.

    During a question the denominator includes that unfinished question. Checks
    run between model calls, so even the first question can exhaust its allowance.
    This is a scheduling projection, not a controlled GPU-time measurement.
    """

    def __init__(self, panel_questions: int, max_hours: float, startup_seconds: float = 0.0):
        if type(panel_questions) is not int or panel_questions <= 0:
            raise ValueError("GCR screen panel_questions must be a positive integer")
        if not math.isfinite(max_hours) or max_hours <= 0:
            raise ValueError("GCR screen max_hours must be finite and positive")
        if not math.isfinite(startup_seconds) or startup_seconds < 0:
            raise ValueError("GCR screen startup_seconds must be finite and nonnegative")
        self.panel_questions, self.max_hours = panel_questions, max_hours
        self.startup_seconds = startup_seconds
        self.started = time.perf_counter()
        self.completed_questions = 0
        self.elapsed = self.projected_hours = self.seconds_per_question = 0.0
        self.aborted = False

    def check(self, *, in_progress: bool) -> bool:
        self.elapsed = time.perf_counter() - self.started
        denominator = max(1, self.completed_questions + int(in_progress))
        self.seconds_per_question = self.elapsed / denominator
        self.projected_hours = (
            self.startup_seconds + self.seconds_per_question * self.panel_questions
        ) / 3600
        self.aborted |= self.projected_hours > self.max_hours
        return self.aborted

    def summary(self) -> dict:
        return {
            "status": "aborted_budget" if self.aborted else "completed",
            "seconds": self.elapsed,
            "startup_seconds": self.startup_seconds,
            "seconds_per_question": self.seconds_per_question,
            "completed_questions": self.completed_questions,
            "panel_questions": self.panel_questions,
            "max_full_gpu_hours": self.max_hours,
            "projected_full_gpu_hours": self.projected_hours,
            "timing_kind": "workload_wall_projection",
        }


@dataclass(frozen=True)
class GCRConfig:
    k: int = 10
    hops: int = 2
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    seed: int = 0
    empty: str = "open"
    max_paths: int = MAX_PATHS
    batch_size: int = 8
    parallel_relations: str = "preserve"
    path_overflow: str = "error"
    max_calls: int | None = None
    max_prompt_tokens: int | None = None
    repetition_penalty: float = 1.0
    decoding: str = "sampling"
    beam_semantics: str = "vllm"
    beam_early_stopping: bool | str = False
    beam_length_penalty: float = 1.0
    beam_return_sequences: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "k",
            "hops",
            "max_tokens",
            "max_paths",
            "batch_size",
            "max_calls",
            "max_prompt_tokens",
        ):
            value = getattr(self, name)
            if value is None and name in ("max_calls", "max_prompt_tokens"):
                continue
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.decoding not in ("sampling", "beam"):
            raise ValueError(f"unknown GCR decoding {self.decoding!r}")
        if self.beam_semantics not in ("vllm", "hf", "hf-group"):
            raise ValueError(f"unknown GCR beam semantics {self.beam_semantics!r}")
        if self.beam_semantics in ("hf", "hf-group") and (self.decoding != "beam" or self.k < 2):
            raise ValueError("GCR HF beam semantics needs decoding=beam and k >= 2")
        if type(self.beam_early_stopping) is not bool and self.beam_early_stopping != "never":
            raise ValueError("GCR beam_early_stopping must be False, True or 'never'")
        if not math.isfinite(self.beam_length_penalty):
            raise ValueError("GCR beam_length_penalty must be finite")
        if self.beam_return_sequences is not None and (
            type(self.beam_return_sequences) is not int
            or not 1 <= self.beam_return_sequences <= self.k
        ):
            raise ValueError("GCR beam_return_sequences must be between 1 and k")
        if self.beam_semantics == "hf-group" and (
            self.beam_early_stopping is not False
            or self.beam_length_penalty != 1.0
            or self.beam_return_sequences not in (None, self.k)
        ):
            raise ValueError(
                "GCR hf-group fixes early_stopping=False, length_penalty=1 and returns K"
            )
        if self.beam_semantics == "vllm" and (
            self.beam_early_stopping is not False
            or self.beam_length_penalty != 1.0
            or self.beam_return_sequences is not None
        ):
            raise ValueError("GCR HF beam options need beam_semantics=hf")
        if self.decoding == "beam" and self.repetition_penalty != 1.0:
            raise ValueError("GCR beam scoring uses raw logprobs and needs repetition_penalty=1")
        if (
            (self.decoding == "sampling" and not self.temperature > 0)
            or (self.decoding == "beam" and not self.temperature >= 0)
            or not 0 < self.top_p <= 1
        ):
            raise ValueError("GCR sampling needs temperature > 0 and 0 < top_p <= 1")
        if self.empty not in ("open", "closed"):
            raise ValueError(f"unknown empty trie policy {self.empty!r}")
        if self.parallel_relations not in ("preserve", "collapse"):
            raise ValueError(f"unknown parallel_relations {self.parallel_relations!r}")
        if self.path_overflow not in ("error", "truncate"):
            raise ValueError(f"unknown path_overflow {self.path_overflow!r}")

    def metadata(self) -> dict:
        values = asdict(self)
        if self.beam_semantics == "vllm":
            # Preserve historical default records, including their config object.
            for key in (
                "beam_semantics",
                "beam_early_stopping",
                "beam_length_penalty",
                "beam_return_sequences",
            ):
                values.pop(key)
        return values


@dataclass(frozen=True)
class PathBeam:
    token_ids: tuple[int, ...]
    logprob: float


@dataclass(frozen=True)
class BeamSearchResult:
    paths: tuple[PathBeam, ...]
    incomplete: tuple[PathBeam, ...]


def beam_search_paths(
    trie: TokenTrie,
    score_next: Callable[[list[tuple[tuple[int, ...], frozenset[int]]]], list[dict[int, float]]],
    *,
    k: int,
    max_tokens: int,
) -> BeamSearchResult:
    """Legacy path-only variant, retained for fixed-score comparison tests only.

    Scores are sums of raw LM logprobs (no trie renormalization or length penalty).
    Equal scores use lexicographic token-id order. All allowed tokens must be scored,
    even when absent from the vocabulary's top logprobs. Keep K candidates in each
    pool, and never return an incomplete prefix as a path. This is finite-width
    beam search, not an exhaustive global top-K guarantee.
    """
    search = beam_search_path_steps(trie, k=k, max_tokens=max_tokens)
    values = None
    while True:
        try:
            pending = search.send(values)
        except StopIteration as done:
            return done.value
        values = score_next(pending)


def beam_search_path_steps(trie: TokenTrie, *, k: int, max_tokens: int):
    """Yield path-only scoring waves, accepting their raw logprobs on resume.

    The synchronous helper and DoG's question scheduler share these transitions,
    including the completed pool, ties, token bound and nonpositive-score checks.
    """
    if type(k) is not int or k <= 0 or type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("beam k and max_tokens must be positive integers")
    if not trie.sequences:
        raise ValueError("beam search needs a nonempty KG-Trie")
    beams, completed = [PathBeam((), 0.0)], []
    for _ in range(max_tokens):
        active = [(beam, trie.allowed(beam.token_ids)) for beam in beams]
        if not active:
            break
        scores = yield [(b.token_ids, allowed) for b, allowed in active]
        if len(scores) != len(active):
            raise ValueError("GCR beam scoring returned the wrong number of prefixes")
        for (_, allowed), values in zip(active, scores, strict=True):
            for token in allowed:
                value = values.get(token)
                if value is None or math.isnan(value) or value > 0:
                    raise ValueError(f"GCR beam needs a nonpositive logprob for token {token}")

        def candidates(pending, logprobs):
            for (beam, allowed), values in zip(pending, logprobs, strict=True):
                for token in sorted(allowed):
                    if values[token] != -math.inf:
                        yield PathBeam((*beam.token_ids, token), beam.logprob + values[token])

        completed = nsmallest(
            k,
            chain(
                completed,
                (b for b in candidates(active, scores) if trie.allowed(b.token_ids) is None),
            ),
            key=lambda b: (-b.logprob, b.token_ids),
        )
        beams = nsmallest(
            k,
            (b for b in candidates(active, scores) if trie.allowed(b.token_ids) is not None),
            key=lambda b: (-b.logprob, b.token_ids),
        )
        # Future logprobs are nonpositive. Strict inequality also preserves ties.
        if len(completed) == k and beams and beams[0].logprob < completed[-1].logprob:
            beams = []
        if not beams:
            break
    return BeamSearchResult(tuple(completed), tuple(beams))


class _State:
    def __init__(self, request: dict, read: KGStore | VersionedStore):
        self.request, self.read = request, read
        if not isinstance(request.get("question"), str) or not request["question"].strip():
            raise ValueError(f"request {request['id']!r} needs a nonempty question")
        anchors = request.get("anchor") or request.get("topics") or []
        if not isinstance(anchors, list) or any(
            not isinstance(s, str) or not re.fullmatch(r"Q\d+", s) for s in anchors
        ):
            raise ValueError(f"request {request['id']!r} needs anchor/topics as a list of QIDs")
        self.start_source = (
            "anchor" if request.get("anchor") else "topics" if anchors else "generated"
        )
        self.start = sorted({s for s in anchors if read.label(int(s[1:])) is not None})
        self.turns: list[dict] = []
        self.samples: list[dict] = []
        self.trace: list[dict] = []
        self.text: str | None = None
        self.finish_reason: str | None = None
        self.status = "ok"
        self.trie: TokenTrie | None = None
        self.arguments: dict | None = None
        self.empty_trie: bool | None = None
        self.trie_info: dict = {}
        self.path_texts: set[str] = set()
        self.beam_info: dict = {}
        self.prompts: GCRPrompts | None = None
        self.path_hops: dict[str, int] = {}

    def messages(self, stage: str) -> list[dict]:
        if self.prompts is not None:
            return self.prompts.messages(
                stage,
                self.request["question"],
                [self.read.label(int(s[1:])) for s in self.start],
                self.samples,
            )
        payload: dict = {"stage": stage, "question": self.request["question"]}
        if stage == "paths":
            payload["starts"] = [
                {"id": s, "label": self.read.label(int(s[1:]))} for s in self.start
            ]
        if stage == "synthesis":
            payload.update(
                paths=[s["path"] for s in self.samples],
                hypotheses=[s["hypothesis"] for s in self.samples],
                # Beam token ids and scores are record metadata, not model input.
                generations=[
                    {
                        key: s[key]
                        for key in (
                            "path",
                            "hypothesis",
                            "text",
                            "valid_path",
                            "finish_reason",
                            "truncated",
                        )
                    }
                    for s in self.samples
                ],
            )
        instructions = PROMPTS[stage]
        if stage == "synthesis" and self.request.get("answer_mode") == "set":
            instructions = instructions[: instructions.index("Give one short final answer")]
            instructions += SET_INSTRUCTION + "\n" + SET_FORMAT
        return [
            {"role": "system", "content": instructions},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    def unanswered(self, reason: str, finish_reason: str) -> None:
        self.text, self.status, self.finish_reason = "Answer: ⟦ Unknown⟧", reason, finish_reason
        self.trace.append({"event": reason})

    def fits(self, stage: str, tokenizer: Any, config: GCRConfig) -> bool:
        limit = config.max_prompt_tokens
        if limit is not None and len(chat_token_ids(tokenizer, self.messages(stage))) > limit:
            self.unanswered("context_limit", "length")
            self.trace[-1]["stage"] = stage
            return False
        return True

    def charge(self, stage: str, output: Any, kind: str | None = None) -> dict:
        if len(output.outputs) != 1:
            raise ValueError("GCR expects one completion per charged trajectory")
        completion = output.outputs[0]
        self.finish_reason = completion.finish_reason
        turn = {
            "stage": stage,
            "text": completion.text,
            "token_ids": list(completion.token_ids),
            "prompt_tokens": len(output.prompt_token_ids),
            "generated_tokens": len(completion.token_ids),
            "finish_reason": completion.finish_reason,
            **({"kind": kind} if kind else {}),
        }
        self.turns.append(turn)
        return turn

    def accept(self, stage: str, output: Any) -> None:
        self.charge(stage, output)
        completion = output.outputs[0]
        if stage == "start":
            data, repairs = parse_selection(stage, completion.text)
            if data is not None and repairs:
                self.trace.append({
                    "event": "parse_repair", "stage": stage, "turn": len(self.turns) - 1,
                    "repairs": repairs, "parsed": data,
                })  # fmt: skip
            starts = set()
            for name in data.get("start", []) if data else []:
                items = (
                    [int(name[1:])] if re.fullmatch(r"Q\d+", name) else self.read.items_named(name)
                )
                items = [i for i in items if self.read.label(i) is not None]
                if items:
                    starts.add(min(items, key=lambda i: (-self.read.degree(i), i)))
                else:
                    self.trace.append({"event": "start", "error": f"no entity named {name!r}"})
            self.start = [f"Q{i}" for i in sorted(starts)]
            if not self.start:
                self.trace.append({"event": "invalid_start", "reply": completion.text})
        elif stage == "paths":
            self.add_sample(completion.text, completion.finish_reason)
        else:
            if self.request.get("answer_mode") == "set":
                self.text = "Answer: " + json.dumps(
                    parse_answer_set(completion.text, bare=True), ensure_ascii=False
                )
                return
            answer, repairs = parse_fallback(completion.text)
            if repairs:
                self.trace.append({
                    "event": "parse_repair", "stage": stage, "turn": len(self.turns) - 1,
                    "repairs": repairs, "parsed": {"answer": answer},
                })  # fmt: skip
            if answer is not None:
                self.text = answer
            else:
                self.unanswered("invalid_synthesis", completion.finish_reason)

    def add_sample(self, text: str, finish_reason: str, **metadata: Any) -> None:
        end = text.find(CLOSER)
        path = text[: end + len(CLOSER)] if text.startswith(OPENER) and end >= 0 else None
        hypothesis = text[end + len(CLOSER) :].strip() if path else text.strip()
        self.samples.append({
            "path": path, "hypothesis": hypothesis, "text": text,
            "valid_path": path in self.path_texts if path is not None else False,
            "finish_reason": finish_reason, "truncated": finish_reason == "length", **metadata,
        })  # fmt: skip
        if self.prompts is not None:
            self.samples[-1]["path_hops"] = self.path_hops.get(path)

    def record(self, config: GCRConfig) -> dict:
        stages = {}
        for stage in STAGES:
            turns = [t for t in self.turns if t["stage"] == stage]
            stages[stage] = {
                "calls": len(turns),
                "prompt_tokens": sum(t["prompt_tokens"] for t in turns),
                "generated_tokens": sum(t["generated_tokens"] for t in turns),
            }
        return {
            "id": self.request["id"],
            "arm": "gcr",
            "text": self.text,
            **set_answer_fields(self.request, self.text or ""),
            **(
                {"kg_version": self.request["kg_version"]} if self.request.get("kg_version") else {}
            ),
            "finish_reason": self.finish_reason,
            "status": self.status,
            "start": self.start,
            "start_source": self.start_source,
            "paths": [s["path"] for s in self.samples],
            "hypotheses": [s["hypothesis"] for s in self.samples],
            "samples": self.samples,
            "stages": stages,
            "turns": self.turns,
            "replies": [t["text"] for t in self.turns],
            "trace": self.trace,
            "calls": len(self.turns),
            "rounds": len(self.turns),
            "token_ids": [],
            "prompt_tokens": sum(t["prompt_tokens"] for t in self.turns),
            "generated_tokens": sum(t["generated_tokens"] for t in self.turns),
            "empty_trie": self.empty_trie,
            "empty_trie_policy": config.empty,
            "decoding": config.decoding,
            **({"beam_search": self.beam_info} if config.decoding == "beam" else {}),
            "config": config.metadata(),
            **({"prompts": self.prompts.metadata()} if self.prompts is not None else {}),
            "trie": self.trie_info,
            "context_limited": self.status == "context_limit",
            "budget_spent": self.status == "calls",
            "truncated": any(
                t["finish_reason"] == "length" and t.get("kind") != "beam_step" for t in self.turns
            )
            or any(s["truncated"] for s in self.samples)
            or (not self.samples and self.beam_info.get("token_limited", False)),
        }


@dataclass(frozen=True)
class _SequenceBeam(PathBeam):
    path_length: int = 0
    path_logprob: float = 0.0
    allowed_tokens: frozenset[int] | None = None


def chat_token_ids(tokenizer: Any, messages: list[dict]) -> list[int]:
    """The prompt's token ids under the chat template, thinking off. Hugging Face
    tokenizers return a BatchEncoding (a mapping) rather than the ids themselves."""
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    return [int(t) for t in ids]


class _JointSearch:
    def __init__(self, state: _State, tokenizer: Any, config: GCRConfig, eos_token_ids=None):
        self.state, self.config = state, config
        self.prompt = chat_token_ids(tokenizer, state.messages("paths"))
        self.eos = getattr(tokenizer, "eos_token_id", None)
        self.eos_ids = frozenset(
            [
                *(eos_token_ids or []),
                *([self.eos] if self.eos is not None else []),
            ]
        )
        if any(type(t) is not int or t < 0 for t in self.eos_ids):
            raise ValueError("GCR EOS ids must be nonnegative integers")
        self.live = [_SequenceBeam((), 0.0)]
        self.completed: list[_SequenceBeam] = []
        state.beam_info = {
            "backend": "host_joint_logprobs", "width": config.k,
            "score": "joint_logprob / (prompt_tokens + generated_tokens - final_eos)",
            "length_penalty": 1.0, "prompt_tokens": len(self.prompt),
            "eos_token_id": self.eos, "tie_break": "token_ids",
            "scoring_requests": 0, "expanded_prefixes": 0,
        }  # fmt: skip
        if eos_token_ids is not None:
            state.beam_info["eos_token_ids"] = sorted(self.eos_ids)

    def ended(self, beam: _SequenceBeam) -> bool:
        return bool(beam.path_length and beam.token_ids[-1] in self.eos_ids)

    def score(self, beam: _SequenceBeam) -> float:
        # vLLM v0.31.0 beam_search/utils.py includes the original prompt in
        # len(tokens), excludes final EOS, and defaults length_penalty to 1.
        length = len(self.prompt) + len(beam.token_ids) - self.ended(beam)
        return beam.logprob / max(length, 1)

    def key(self, beam: _SequenceBeam) -> tuple:
        return -self.score(beam), beam.token_ids

    def allowed(self, beam: _SequenceBeam) -> frozenset[int] | None:
        # Every retained path prefix was looked up when it became a candidate.
        # Keep only the live beams' sets, rather than caching the whole trie again.
        return self.state.trie.allowed(()) if not beam.token_ids else beam.allowed_tokens

    def advance(self, scores: list[dict[int, float]]) -> None:
        for row in scores:
            if not row or any(math.isnan(v) or v > 0 for v in row.values()):
                raise ValueError("GCR beam needs nonempty nonpositive raw token logprobs")

        completed = list(self.completed)

        def live_candidates():
            for beam, row in zip(self.live, scores, strict=True):
                for token, value in sorted(row.items()):
                    if value == -math.inf:
                        continue
                    tokens, logprob = (*beam.token_ids, token), beam.logprob + value
                    path_length, path_logprob = beam.path_length, beam.path_logprob
                    allowed = None
                    if not path_length:
                        allowed = self.state.trie.allowed(tokens)
                        if allowed is None:
                            path_length, path_logprob = len(tokens), logprob
                    candidate = _SequenceBeam(tokens, logprob, path_length, path_logprob, allowed)
                    if self.ended(candidate):
                        completed.append(candidate)
                    else:
                        yield candidate

        self.live = nsmallest(
            self.config.k,
            live_candidates(),
            key=self.key,
        )
        self.completed = nsmallest(self.config.k, completed, key=self.key)
        # No raw-score early stop: length-normalized live beams can improve.

    def deliver(self, tokenizer: Any) -> None:
        state = self.state
        if state.text is not None:
            return
        beams = nsmallest(
            self.config.k,
            (b for b in chain(self.completed, self.live) if b.path_length),
            key=self.key,
        )
        state.beam_info.update(
            returned_sequences=len(beams),
            returned_paths=len({b.token_ids[:b.path_length] for b in beams}),
            available_paths=len(state.trie.sequences), token_limited=bool(self.live),
        )  # fmt: skip
        if len(beams) < self.config.k:
            state.trace.append({
                "event": "beam_shortfall", "requested": self.config.k, "returned": len(beams),
                "available_paths": len(state.trie.sequences), "token_limited": bool(self.live),
            })  # fmt: skip
            state.status = "beam_shortfall"
        if not beams:
            state.unanswered("beam_no_path", "length" if self.live else "stop")
            return
        for beam in beams:
            ended = self.ended(beam)
            tokens = list(beam.token_ids[:-1] if ended else beam.token_ids)
            state.add_sample(
                tokenizer.decode(tokens, skip_special_tokens=False), "stop" if ended else "length",
                path_token_ids=list(beam.token_ids[:beam.path_length]),
                path_logprob=beam.path_logprob, sequence_token_ids=list(beam.token_ids),
                joint_logprob=beam.logprob, beam_score=self.score(beam),
            )  # fmt: skip


class _HFJointSearch(_JointSearch):
    """Ordinary prefix-constrained beam search from Transformers 4.44.0.

    The released GCR launcher instead selects diverse group beam search. This
    opt-in mode reproduces its ordinary ``beam`` option, given the same logits.
    CPU torch.topk preserves that oracle's tie order over the full K*vocab shape.
    """

    def __init__(self, state, tokenizer, config, eos_token_ids, vocab_size):
        import torch

        super().__init__(state, tokenizer, config, eos_token_ids)
        if eos_token_ids is not None:
            # GenerationConfig overrides the tokenizer's EOS, rather than unioning it.
            self.eos_ids = frozenset(eos_token_ids)
            state.beam_info["eos_token_ids"] = list(eos_token_ids)
        if vocab_size is None:
            vocab_size = len(tokenizer)
        if type(vocab_size) is not int or vocab_size <= max((0, *self.eos_ids)):
            raise ValueError("GCR HF beam needs the model's positive vocabulary size")
        self.torch, self.vocab_size = torch, vocab_size
        # _beam_search starts K identical prompts: first score 0, the rest -1e9.
        self.live = [_SequenceBeam((), 0.0 if i == 0 else -1e9) for i in range(config.k)]
        self.done = False
        self.steps = 0
        eos_count = len(eos_token_ids) if eos_token_ids is not None else len(self.eos_ids)
        self.candidate_count = max(2, 1 + eos_count) * config.k
        self.return_sequences = config.beam_return_sequences or config.k
        # HF keeps EOS order and finalizes with the first configured EOS id.
        self.final_eos = next(iter(eos_token_ids), None) if eos_token_ids is not None else self.eos
        state.beam_info.update(
            semantics="hf",
            transformers_version="4.44.0",
            eos_token_id=self.final_eos,
            score="joint_logprob / generated_tokens ** length_penalty (including EOS)",
            length_penalty=config.beam_length_penalty,
            early_stopping=config.beam_early_stopping,
            num_return_sequences=self.return_sequences,
            tie_break="torch.topk CPU; stable hypothesis insertion",
            torch_version=torch.__version__,
            vocab_size=vocab_size,
            candidates_per_step=self.candidate_count,
            steps=0,
            tie_fallback_requests=0,
        )

    def score(self, beam):
        return beam.logprob / len(beam.token_ids) ** self.config.beam_length_penalty

    def ended(self, beam):
        return bool(beam.token_ids and beam.token_ids[-1] in self.eos_ids)

    def add(self, beam):
        # BeamHypotheses.add: strictly better admission once full; oldest tied
        # minimum is removed. Final selection pops a stable ascending sort.
        worst = min((self.score(b) for b in self.completed), default=1e9)
        if len(self.completed) < self.config.k or self.score(beam) > worst:
            self.completed.append(beam)
            if len(self.completed) > self.config.k:
                index = min(
                    range(len(self.completed)), key=lambda i: (self.score(self.completed[i]), i)
                )
                del self.completed[index]

    def score_matrix(self, scores):
        torch = self.torch
        values = torch.full(
            (self.config.k, self.vocab_size), -math.inf, dtype=torch.float32, device="cpu"
        )
        for index, (beam, row) in enumerate(zip(self.live, scores, strict=True)):
            if not row or any(math.isnan(v) or v > 0 for v in row.values()):
                raise ValueError("GCR beam needs nonempty nonpositive raw token logprobs")
            if any(type(t) is not int or not 0 <= t < self.vocab_size for t in row):
                raise ValueError("GCR HF beam logprob token is outside the model vocabulary")
            tokens = list(row)
            values[index, tokens] = torch.tensor(
                list(row.values()), dtype=torch.float32, device="cpu"
            )
            values[index] += beam.logprob
        return values

    def tied_rows(self, scores):
        # torch.topk's equal-score order can depend even on lower-scored entries
        # during partitioning. Recover every sparse hypothesis row when selected
        # scores tie, or the extra guard token ties the selection boundary.
        requested = min(self.vocab_size, self.candidate_count + 1)
        sparse = [
            i
            for i, (beam, row) in enumerate(zip(self.live, scores, strict=True))
            if beam.path_length and requested <= len(row) < self.vocab_size
        ]
        if not sparse:
            return []
        values = self.score_matrix(scores)
        selected = values.reshape(-1).topk(self.candidate_count).values.tolist()
        if len(set(selected)) < len(selected) or any(
            min(values[index, list(scores[index])].tolist()) >= selected[-1] for index in sparse
        ):
            return sparse
        return []

    def advance(self, scores):
        values = self.score_matrix(scores)
        best_scores, indices = values.reshape(1, -1).topk(self.candidate_count, dim=1)
        live = []
        self.steps += 1
        self.state.beam_info["steps"] = self.steps
        for rank, (value, index) in enumerate(
            zip(best_scores[0].tolist(), indices[0].tolist(), strict=True)
        ):
            source, token = divmod(index, self.vocab_size)
            beam = self.live[source]
            tokens = (*beam.token_ids, token)
            path_length, path_logprob, allowed = beam.path_length, beam.path_logprob, None
            if not path_length:
                allowed = self.state.trie.allowed(tokens)
                if allowed is None:
                    path_length, path_logprob = len(tokens), value
            candidate = _SequenceBeam(tokens, value, path_length, path_logprob, allowed)
            if token in self.eos_ids:
                if rank < self.config.k:
                    # finalize() replaces the admitted EOS with eos_token_id[0].
                    candidate = _SequenceBeam(
                        (*tokens[:-1], self.final_eos), value, path_length, path_logprob
                    )
                    self.add(candidate)
            else:
                live.append(candidate)
            if len(live) == self.config.k:
                break
        if len(live) < self.config.k:
            raise ValueError("GCR HF beam could not fill K non-EOS candidates")
        if len(self.completed) == self.config.k:
            early = self.config.beam_early_stopping
            length = (
                self.config.max_tokens
                if early == "never" and self.config.beam_length_penalty > 0
                else self.steps
            )
            bound = best_scores.max().item() / length**self.config.beam_length_penalty
            self.done = early is True or min(self.score(b) for b in self.completed) >= bound
        self.live = [] if self.done else live

    def final_beams(self):
        token_limited = bool(self.live)
        if not self.done:
            for beam in self.live:
                self.add(beam)
        beams = list(reversed(sorted(self.completed, key=self.score)))[: self.return_sequences]
        return token_limited, beams

    def deliver(self, tokenizer):
        state = self.state
        if state.text is not None:
            return
        token_limited, beams = self.final_beams()
        state.beam_info.update(
            returned_sequences=len(beams),
            returned_paths=len({b.token_ids[: b.path_length] for b in beams if b.path_length}),
            available_paths=len(state.trie.sequences),
            token_limited=token_limited,
            is_done=self.done,
        )
        if len(beams) < self.return_sequences:
            state.trace.append({
                "event": "beam_shortfall", "requested": self.return_sequences,
                "returned": len(beams),
                "available_paths": len(state.trie.sequences), "token_limited": token_limited,
            })  # fmt: skip
            state.status = "beam_shortfall"
        if not beams:
            state.unanswered("beam_no_path", "length" if token_limited else "stop")
            return
        for beam in beams:
            ended = self.ended(beam)
            tokens = list(beam.token_ids[:-1] if ended else beam.token_ids)
            state.add_sample(
                tokenizer.decode(tokens, skip_special_tokens=False),
                "stop" if ended else "length",
                path_token_ids=list(
                    beam.token_ids[: beam.path_length] if beam.path_length else beam.token_ids
                ),
                path_logprob=beam.path_logprob if beam.path_length else beam.logprob,
                sequence_token_ids=list(beam.token_ids),
                joint_logprob=beam.logprob,
                beam_score=self.torch.tensor(
                    self.score(beam), dtype=self.torch.float32, device="cpu"
                ).item(),
            )
        if not any(beam.path_length for beam in beams):
            state.unanswered("beam_no_path", "length" if token_limited else "stop")


@dataclass(frozen=True)
class _GroupBeam(_SequenceBeam):
    search_logprob: float = 0.0
    diversity_cost: int = 0


class _HFGroupJointSearch(_HFJointSearch):
    """Transformers 4.44.0 group beam: K groups, size one, diversity 1.0.

    Model rows are unmasked full-vocabulary log-softmax scores. Apply Hamming
    diversity, then the prefix mask, then the accumulated processed score;
    never renormalize. Each group owns one BeamHypotheses slot independently.
    """

    def __init__(self, state, tokenizer, config, eos_token_ids, vocab_size, logits_dtype="float32"):
        super().__init__(state, tokenizer, config, eos_token_ids, vocab_size)
        dtype = str(logits_dtype).removeprefix("torch.")
        if dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(
                f"GCR HF group needs float32, float16 or bfloat16 logits, got {dtype!r}"
            )
        self.logits_dtype = getattr(self.torch, dtype)
        self.live = [_GroupBeam((), 0.0) for _ in range(config.k)]
        self.group_done = [False] * config.k
        self.hypotheses = [None] * config.k
        # The release explicitly supplies pad_token_id=tokenizer.eos_token_id.
        self.pad = self.eos
        self.candidate_count //= config.k
        # At most K-1 distinct ids are penalized. Fetch that many extra raw
        # candidates, plus one boundary guard; recover full rows on ties.
        self.request_count = min(vocab_size, self.candidate_count + config.k)
        state.beam_info.update(
            semantics="hf-group", num_beam_groups=config.k, group_size=1,
            diversity_penalty=1.0, scores_include_diversity_penalty=True,
            score="search_logprob / generated_tokens (including EOS and diversity penalty)",
            candidates_per_step=self.candidate_count, candidates_per_group=self.candidate_count,
            logprobs_per_hypothesis_request=self.request_count, pad_token_id=self.pad,
            processor_order=["full-vocabulary log_softmax", "HammingDiversityLogitsProcessor",
                             "PrefixConstrainedLogitsProcessor", "add beam score"],
            logits_dtype=dtype, accumulator_dtype="float32",
        )  # fmt: skip

    def score(self, beam):
        return beam.search_logprob / len(beam.token_ids)

    def group_scores(self, beam, row, current_tokens, *, count=None):
        torch = self.torch
        values = torch.full((self.vocab_size,), -math.inf, dtype=self.logits_dtype, device="cpu")
        if not row or any(math.isnan(v) or v > 0 for v in row.values()):
            raise ValueError("GCR beam needs nonempty nonpositive raw token logprobs")
        if any(type(t) is not int or not 0 <= t < self.vocab_size for t in row):
            raise ValueError("GCR HF beam logprob token is outside the model vocabulary")
        values[list(row)] = torch.tensor(list(row.values()), dtype=self.logits_dtype, device="cpu")
        if current_tokens:
            frequency = torch.bincount(
                torch.tensor(current_tokens, device="cpu"), minlength=self.vocab_size
            )
            values -= 1.0 * frequency
        # Constrained rows contain exactly the prefix function's allowed ids.
        # Missing ids are -inf; additive masking commutes with this subtraction.
        # HF keeps processor arithmetic in logits dtype, then its out-of-place
        # addition of the float32 beam-score column promotes the whole row.
        values = values.float() + beam.search_logprob
        return values.topk(count or self.candidate_count)

    def tied_rows(self, scores):
        sparse = [
            i
            for i, (beam, row) in enumerate(zip(self.live, scores, strict=True))
            if beam.path_length and len(row) < self.vocab_size and not self.group_done[i]
        ]
        if not sparse:
            return []
        current = []
        for i, (beam, row) in enumerate(zip(self.live, scores, strict=True)):
            if self.group_done[i]:
                current.append(self.pad)
                continue
            values, tokens = self.group_scores(
                beam,
                row,
                current,
                count=min(self.vocab_size, self.candidate_count + (i in sparse)),
            )
            # A tied earlier group can change every later group's penalty.
            # Recover all sparse active rows atomically before advancing any.
            if i in sparse and len(set(values.tolist())) < len(values):
                return sparse
            current.append(next(t for t in tokens.tolist() if t not in self.eos_ids))
        return []

    def candidate(self, beam, token, value, row, current):
        tokens = (*beam.token_ids, token)
        raw = (
            self.torch.tensor(beam.logprob, dtype=self.torch.float32, device="cpu") + row[token]
        ).item()
        path_length, path_logprob, allowed = beam.path_length, beam.path_logprob, None
        if not path_length:
            allowed = self.state.trie.allowed(tokens)
            if allowed is None:
                path_length, path_logprob = len(tokens), raw
        return _GroupBeam(
            tokens,
            raw,
            path_length,
            path_logprob,
            allowed,
            value,
            beam.diversity_cost + current.count(token),
        )

    def advance(self, scores):
        self.steps += 1
        self.state.beam_info["steps"] = self.steps
        current, live = [], []
        for i, (beam, row) in enumerate(zip(self.live, scores, strict=True)):
            if self.group_done[i]:
                # process() pads an already done group with score zero. Its pad
                # still participates in diversity for all later groups.
                live.append(_GroupBeam((*beam.token_ids, self.pad), 0.0, beam.path_length))
                current.append(self.pad)
                continue
            values, tokens = self.group_scores(beam, row, current)
            for rank, (value, token) in enumerate(
                zip(values.tolist(), tokens.tolist(), strict=True)
            ):
                # topk may select masked -inf entries, just as HF does.
                candidate = self.candidate(
                    beam, token, value, {token: row.get(token, -math.inf)}, current
                )
                if token in self.eos_ids:
                    if rank == 0:
                        candidate = replace(candidate, token_ids=(*beam.token_ids, self.final_eos))
                        old = self.hypotheses[i]
                        if old is None or self.score(candidate) > self.score(old):
                            self.hypotheses[i] = candidate
                else:
                    live.append(candidate)
                    current.append(token)
                    break
            else:
                raise ValueError("GCR HF group could not fill a non-EOS candidate")
            best = self.hypotheses[i]
            self.group_done[i] = (
                best is not None and self.score(best) >= values.max().item() / self.steps
            )
        self.done = all(self.group_done)
        self.live = [] if self.done else live

    def final_beams(self):
        # finalize() adds unfinished groups, retains their strictly best slot,
        # then stably sorts groups together and pops from the end (reverse ties).
        for i, beam in enumerate(self.live):
            old = self.hypotheses[i]
            if not self.group_done[i] and (old is None or self.score(beam) > self.score(old)):
                self.hypotheses[i] = beam
        self.selected = list(
            reversed(sorted((b for b in self.hypotheses if b is not None), key=self.score))
        )
        self.state.beam_info["groups_done"] = self.group_done
        return not self.done, self.selected

    def deliver(self, tokenizer):
        super().deliver(tokenizer)
        if self.state.samples:
            for sample, beam in zip(self.state.samples, self.selected, strict=True):
                sample.update(
                    search_logprob=beam.search_logprob, diversity_penalty=beam.diversity_cost
                )


def _beam_diagnostic(event, **fields):
    if os.environ.get("XGRAPH_GCR_DIAGNOSTICS") == "1":
        print(
            "[gcr-beam] " + json.dumps({
                "event": event, "monotonic_seconds": time.monotonic(),
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **fields,
            }),
            file=sys.stderr, flush=True,
        )  # fmt: skip


@contextmanager
def diagnostic_engine_shutdown(model):
    """Observe the pinned parent's SIGTERM caller without changing its cleanup."""
    if os.environ.get("XGRAPH_GCR_DIAGNOSTICS") != "1":
        yield
        return

    class ShutdownTrace(logging.Filter):
        def filter(self, record):
            if "Process manager: send sigterm" in record.getMessage():
                core = getattr(getattr(model, "llm_engine", None), "engine_core", None)
                manager = getattr(getattr(core, "resources", None), "engine_manager", None)
                _beam_diagnostic("parent_shutdown", processes=[
                    {"name": p.name, "pid": p.pid, "exitcode": p.exitcode}
                    for p in getattr(manager, "processes", ())
                ])  # fmt: skip
                traceback.print_stack(file=sys.stderr)
            return True

    logger = logging.getLogger("vllm.v1.utils")
    probe = ShutdownTrace()
    logger.addFilter(probe)
    try:
        yield
    finally:
        logger.removeFilter(probe)


def _score_beam_jobs(
    model, jobs, params_class, config, adapter_args, screen_budget, *, full=False, audits=None
):
    for at in range(0, len(jobs), 32):
        group = jobs[at : at + 32]
        params = []
        for search, _, _, tokens, _ in group:
            if tokens is not None:
                options = {"logprob_token_ids": tokens}
            else:
                count = 2 * config.k
                if config.beam_semantics == "hf-group":
                    count = -1 if full else search.request_count
                elif config.beam_semantics == "hf":
                    count = -1 if full else min(search.vocab_size, search.candidate_count + 1)
                options = {"logprobs": count}
            params.append(params_class(
                max_tokens=1, temperature=0.0, ignore_eos=True, detokenize=False, **options,
            ))  # fmt: skip
        step = max(len(job[2]) + 1 for job in group)
        context = {
            "questions": list(dict.fromkeys(job[0].state.request["id"] for job in group)),
            "step": step,
            "jobs": len(group),
            "full": full,
            "scoring_requests": {
                s.state.request["id"]: s.state.beam_info["scoring_requests"] for s, *_ in group
            },
        }
        if step <= 2 or step % 32 == 0 or full:
            _beam_diagnostic("score_start", **context)
        try:
            outputs = model.generate(
                [{"prompt_token_ids": [*s.prompt, *prefix]} for s, _, prefix, _, _ in group],
                params, use_tqdm=False, **adapter_args,
            )  # fmt: skip
        except BaseException:
            _beam_diagnostic("score_failed", **context)
            if os.environ.get("XGRAPH_GCR_DIAGNOSTICS") == "1":
                traceback.print_exc(file=sys.stderr)
            raise
        for (search, _, prefix, tokens, values), output in zip(group, outputs, strict=True):
            state = search.state
            turn = state.charge("paths", output, "beam_step")
            state.beam_info["scoring_requests"] += 1
            if full:
                state.beam_info["tie_fallback_requests"] += 1
                turn["tie_fallback"] = True
            completion = output.outputs[0]
            if len(completion.token_ids) != 1 or completion.finish_reason == "abort":
                raise ValueError(f"GCR beam request {state.request['id']!r}: scoring aborted")
            if completion.logprobs is None or len(completion.logprobs) != 1:
                raise ValueError("GCR beam requires vLLM raw token logprobs")
            row = completion.logprobs[0]
            if tokens is not None and set(tokens) - row.keys():
                raise ValueError("GCR beam missing requested token logprobs")
            if full and len(row) != search.vocab_size:
                raise ValueError("GCR HF boundary tie requires full-vocabulary logprobs")
            scored = {t: float(row[t].logprob) for t in (row if tokens is None else tokens)}
            values.update(scored)
            turn.update(
                prefix_sha256=hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),
                token_logprobs={} if full else scored,
                phase="hypothesis" if tokens is None else "path",
            )
            if full:
                # Full rows are selection scratch, not a vocabulary-sized audit
                # per beam per step. Keep raw witnesses after selection below.
                turn["full_logprob_count"] = len(scored)
                audits.append((search, prefix, turn, values))
        if screen_budget is not None and screen_budget.check(in_progress=True):
            return False
    return True


def _run_beams(
    model: Any,
    states: list[_State],
    tokenizer: Any,
    params_class: Any,
    config: GCRConfig,
    adapter: Any,
    screen_budget: GCRScreenBudget | None = None,
    eos_token_ids=None,
    beam_vocab_size=None,
    beam_logits_dtype=None,
) -> None:
    """Batch constrained explicit-id and unconstrained raw-logprob queries across questions."""
    searches = [
        _HFGroupJointSearch(
            s, tokenizer, config, eos_token_ids, beam_vocab_size, beam_logits_dtype or "float32"
        )
        if config.beam_semantics == "hf-group"
        else _HFJointSearch(s, tokenizer, config, eos_token_ids, beam_vocab_size)
        if config.beam_semantics == "hf"
        else _JointSearch(s, tokenizer, config, eos_token_ids)
        for s in states
    ]
    adapter_args = {"lora_request": adapter} if adapter is not None else {}
    for _ in range(config.max_tokens):
        jobs, pending, audits = [], [], []
        for search in searches:
            state = search.state
            if state.text is not None or not search.live:
                continue
            question_jobs = []
            values = [{} for _ in search.live]
            for index, beam in enumerate(search.live):
                if beam.path_length:
                    question_jobs.append((search, index, beam.token_ids, None, values[index]))
                else:
                    tokens = sorted(search.allowed(beam))
                    for at in range(0, len(tokens), 128):
                        question_jobs.append(
                            (
                                search,
                                index,
                                beam.token_ids,
                                tokens[at : at + 128],
                                values[index],
                            )
                        )
            # Reserve synthesis and never issue only part of a question's step.
            if (
                config.max_calls is not None
                and len(state.turns) + len(question_jobs) + 1 > config.max_calls
            ):
                state.unanswered("calls", "budget")
                continue
            state.beam_info["expanded_prefixes"] += len(search.live)
            jobs.extend(question_jobs)
            pending.append((search, values))
        if not jobs:
            break
        if not _score_beam_jobs(model, jobs, params_class, config, adapter_args, screen_budget):
            return
        if config.beam_semantics in ("hf", "hf-group"):
            tie_jobs = []
            for search, values in pending:
                indices = search.tied_rows(values)
                if config.max_calls is not None and (
                    len(search.state.turns) + len(indices) + 1 > config.max_calls
                ):
                    search.state.unanswered("calls", "budget")
                    continue
                tie_jobs.extend(
                    (search, i, search.live[i].token_ids, None, values[i]) for i in indices
                )
            if not _score_beam_jobs(
                model,
                tie_jobs,
                params_class,
                config,
                adapter_args,
                screen_budget,
                full=True,
                audits=audits,
            ):
                return
        for search, values in pending:
            if search.state.text is None:
                search.advance(values)
        for search, prefix, turn, row in audits:
            # EOS ids are kept before HF's configured-EOS normalization. Include
            # the provider's sampled token and every retained live/EOS extension;
            # these suffice to independently sum any eventually returned beam.
            beams = chain(search.live, search.completed, getattr(search, "hypotheses", ()))
            keep = set(search.eos_ids) | set(turn["token_ids"])
            keep.update(
                beam.token_ids[-1]
                for beam in beams
                if beam is not None and beam.token_ids[:-1] == prefix
            )
            turn["token_logprobs"] = {token: row[token] for token in sorted(keep) if token in row}
    for search in searches:
        search.deliver(tokenizer)
        _beam_diagnostic(
            "question_paths_complete",
            question=search.state.request["id"],
            scoring_requests=search.state.beam_info["scoring_requests"],
            steps=search.state.beam_info.get("steps"),
        )


def run_gcr(
    llm: Any,
    requests: list[dict],
    store: KGStore | VersionedStore,
    params_class: Any,
    *,
    config: GCRConfig | None = None,
    versions: dict | None = None,
    synthesis_llm: Any = None,
    synthesis_factory: Callable[[], Any] | None = None,
    prompts: GCRPrompts | None = None,
    eos_token_ids=None,
    beam_vocab_size: int | None = None,
    beam_logits_dtype=None,
    banned: list[int] | None = None,
    lora_request: Any = None,
    screen_budget: GCRScreenBudget | None = None,
    path_handoff: tuple | None = None,
) -> list[dict]:
    """Batched start, sampled trajectories or joint beams, then synthesis, in request order.

    Anchors take precedence over topics; absent both, use CoR's tolerant selection
    parser for one charged start call and each name's best-connected sense. max_calls
    includes every trajectory or one-token joint-beam scoring request and
    synthesis, not just Python batch invocations. Beam shortfalls are explicit.
    synthesis_factory may release/reload the first engine after all paths are collected;
    run_gcr drops its model reference before invoking it. Prompt overflow returns an
    explicit unanswered record without truncating inputs or charging unissued calls.
    """
    if synthesis_llm is not None and synthesis_factory is not None:
        raise ValueError("supply synthesis_llm or synthesis_factory, not both")
    config = config or GCRConfig()
    if screen_budget is not None and (
        config.decoding != "beam"
        or config.batch_size != 1
        or synthesis_llm is not None
        or synthesis_factory is not None
        or screen_budget.panel_questions < len(requests)
    ):
        raise ValueError("GCR screen needs sequential joint beams, one shared model and panel size")
    if config.decoding == "beam" and banned:
        raise ValueError("GCR beam scoring uses raw logprobs and does not support banned tokens")
    if config.decoding == "beam":
        engine_config = getattr(getattr(llm, "llm_engine", None), "vllm_config", None)
        model_config = getattr(engine_config, "model_config", None)
        if getattr(model_config, "logprobs_mode", "raw_logprobs") != "raw_logprobs":
            raise ValueError("GCR beam needs the vLLM model setting logprobs_mode=raw_logprobs")
        if config.beam_semantics == "hf-group":
            beam_logits_dtype = getattr(model_config, "dtype", None) or beam_logits_dtype
    states = [_State(r, store_for(r, store, versions)) for r in requests]
    for state in states:
        state.prompts = prompts
        if prompts is not None and (
            state.start_source == "generated" or state.request.get("answer_mode") == "set"
        ):
            raise ValueError("paper GCR profile requires linked topics and single-answer requests")
    tokenizer = llm.get_tokenizer()

    def sampling(stage: str, state: _State, seed: int = 0, shared: bool = True) -> Any:
        kwargs: dict = {
            "max_tokens": config.max_tokens,
            "temperature": config.temperature if stage == "paths" else 0.0,
            "extra_args": state.arguments if stage == "paths" else None,
            "repetition_penalty": config.repetition_penalty,
        }
        if stage == "paths":
            kwargs.update(top_p=config.top_p, seed=seed, skip_special_tokens=False)
        if shared and banned:
            kwargs["logit_bias"] = {t: -100.0 for t in banned}
        return params_class(**kwargs)

    def issue(model: Any, stage: str, jobs: list[tuple[_State, Any]], adapter: Any = None) -> None:
        if not jobs:
            return
        outputs = model.chat(
            [s.messages(stage) for s, _ in jobs],
            [p for _, p in jobs],
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=False,
            **({"lora_request": adapter} if adapter is not None else {}),
        )
        for (state, _), output in zip(jobs, outputs, strict=True):
            state.accept(stage, output)
        if screen_budget is not None:
            screen_budget.check(in_progress=True)

    def start(group):
        start_jobs = []
        for state in group:
            required = (config.k + 1 if config.decoding == "sampling" else 2) + (
                state.start_source == "generated"
            )
            if config.max_calls is not None and required > config.max_calls:
                state.unanswered("calls", "budget")
            elif state.start_source == "generated" and state.fits("start", tokenizer, config):
                start_jobs.append((state, sampling("start", state)))
        issue(llm, "start", start_jobs, lora_request)

    if screen_budget is None:
        start(states)

    # Bound resident preprocessing to a batch of questions, not the whole benchmark.
    for offset in range(0, len(states), config.batch_size):
        group = states[offset : offset + config.batch_size]
        if screen_budget is not None:
            start(group)
            if screen_budget.check(in_progress=True):
                break
        path_jobs, beam_states = [], []
        for index, state in enumerate(group, start=offset):
            if state.text is not None:
                continue
            try:
                enumeration = collect_paths(
                    state.read,
                    state.start,
                    config.hops,
                    config.max_paths,
                    config.parallel_relations,
                    config.path_overflow,
                )
            except ValueError as error:
                raise ValueError(
                    f"GCR request {state.request['id']!r}, KG version "
                    f"{state.request.get('kg_version')!r}: {error}"
                ) from error
            paths = enumeration.paths
            if prompts is not None:
                state.path_hops = {serialize_path(p): len(p.relations) for p in paths}
            state.path_texts = {serialize_path(p) for p in paths}
            state.trie = TokenTrie.from_paths(paths, tokenizer, config.empty)
            state.empty_trie = not paths
            state.trie_info = {
                "graph_paths": len(paths),
                "token_paths": len(state.trie.sequences),
                "nodes": len(state.trie.children),
                "truncated": enumeration.truncated,
                "max_paths": config.max_paths,
                "sha256": hashlib.sha256(json.dumps(state.trie.sequences).encode()).hexdigest(),
            }
            if prompts is not None:
                state.trie_info["paths_by_hop"] = dict(
                    Counter(str(len(p.relations)) for p in paths)
                )
            state.arguments = (
                trie_arguments(str(state.request["id"]), state.trie)
                if config.decoding == "sampling" or not paths
                else None
            )
            if not paths and config.empty == "closed":
                state.unanswered("empty_trie", "empty_trie")
                continue
            if state.fits("paths", tokenizer, config):
                if config.decoding == "beam" and paths:
                    beam_states.append(state)
                    continue
                if config.decoding == "beam":
                    state.beam_info = {"backend": "sampling_empty_trie", "returned_paths": 0}
                    if (
                        config.max_calls is not None
                        and len(state.turns) + config.k + 1 > config.max_calls
                    ):
                        state.unanswered("calls", "budget")
                        continue
                for sample in range(config.k):
                    path_jobs.append(
                        (state, sampling("paths", state, config.seed + index * config.k + sample))
                    )
        if screen_budget is not None and screen_budget.check(in_progress=True):
            break
        _run_beams(
            llm,
            beam_states,
            tokenizer,
            params_class,
            config,
            lora_request,
            screen_budget,
            eos_token_ids=eos_token_ids,
            beam_vocab_size=beam_vocab_size,
            beam_logits_dtype=beam_logits_dtype,
        )
        if screen_budget is not None and screen_budget.aborted:
            break
        issue(llm, "paths", path_jobs, lora_request)
        path_jobs.clear()
        for state in group:
            state.trie, state.arguments = None, None
            state.path_texts.clear()
            state.path_hops.clear()
        if screen_budget is not None:
            if screen_budget.aborted:
                break
            issue(llm, "synthesis", [
                (s, sampling("synthesis", s)) for s in group
                if s.text is None and s.fits("synthesis", tokenizer, config)
            ], lora_request)  # fmt: skip
            screen_budget.completed_questions += len(group)
            if screen_budget.check(in_progress=False):
                break

    if path_handoff is not None:
        from xgraph.gcr_stages import save_path_states

        save_path_states(states, config, *path_handoff)
        return [s.record(config) for s in states]
    pending = [s for s in states if s.text is None]
    if screen_budget is not None:
        for state in pending:
            state.unanswered("screen_infeasible", "abort")
    elif pending:
        shared = synthesis_llm is None and synthesis_factory is None
        if synthesis_factory is not None:
            llm = None  # The caller's factory may now shut down and release its engine.
            synthesis_llm = synthesis_factory()
        model = llm if synthesis_llm is None else synthesis_llm
        synth_tokenizer = model.get_tokenizer()
        jobs = [
            (s, sampling("synthesis", s, shared=shared))
            for s in pending
            if s.fits("synthesis", synth_tokenizer, config)
        ]
        issue(model, "synthesis", jobs, lora_request if shared else None)
    return [s.record(config) for s in states]


def replay_syntheses(records: list[dict]) -> tuple[list[dict], dict]:
    """Reparse saved synthesis only, retaining raw turns and historical model costs."""
    result, seen = [], set()
    available = readable = repaired = recovered_invalid = recovered_unknown = changed = 0

    def unknown(text: str) -> bool:
        if text.startswith("Answer: ["):
            return not parse_answer_set(text)
        label = final_answer(text)
        return not label or normalize(label) == "unknown"

    for record in records:
        qid = record["id"]
        if qid in seen:
            raise ValueError(f"duplicate record id {qid!r}")
        seen.add(qid)
        if record.get("arm") != "gcr":
            raise ValueError(f"record {qid!r} is not a gcr record")
        turns = record.get("turns", [])
        if not turns or turns[-1].get("stage") != "synthesis":
            result.append(record)
            continue
        available += 1
        if record.get("answer_mode") == "set":
            answer = "Answer: " + json.dumps(
                parse_answer_set(turns[-1]["text"], bare=True), ensure_ascii=False
            )
            repairs = []
        else:
            answer, repairs = parse_fallback(turns[-1]["text"])
        readable += answer is not None
        recovered_invalid += record.get("status") == "invalid_synthesis" and answer is not None
        repaired += bool(repairs) and answer is not None
        status = "ok" if answer is not None else "invalid_synthesis"
        answer = answer or "Answer: ⟦ Unknown⟧"
        recovered_unknown += unknown(record["text"]) and not unknown(answer)
        changed += answer != record["text"]
        trace = list(record.get("trace", []))
        if repairs:
            trace.append({
                "event": "parse_repair", "stage": "synthesis", "turn": len(turns) - 1,
                "repairs": repairs, "parsed": {"answer": answer if status == "ok" else None},
            })  # fmt: skip
        result.append({
            **record, "text": answer, "status": status, "trace": trace,
            **set_answer_fields(record, answer),
            "replay": {"kind": "synthesis_parsing", "original_text": record["text"],
                       "original_status": record.get("status")},
        })  # fmt: skip
    return result, {
        "requests": len(records),
        "available_syntheses": available,
        "unavailable_syntheses": len(records) - available,
        "readable_syntheses": readable,
        "repaired_syntheses": repaired,
        "invalid_synthesis_before": sum(r.get("status") == "invalid_synthesis" for r in records),
        "invalid_synthesis_after": sum(r.get("status") == "invalid_synthesis" for r in result),
        "unknown_before": sum(unknown(r["text"]) for r in records),
        "unknown_after": sum(unknown(r["text"]) for r in result),
        "recovered_invalid_syntheses": recovered_invalid,
        "recovered_from_unknown": recovered_unknown,
        "changed_answers": changed,
        "model_calls": 0,
        "scope": "saved synthesis replies only; starts and paths were not rerun",
    }


def prepare_pilot(
    panels, reference, requests, reference_requests, store, *, max_paths, depths=(2, 4)
):
    """Gold-free, exact walk counts before allocating any exponential path trie.

    Dynamic programming merges equal endpoints with their walk multiplicities.
    It counts the same directed labelled walks (including cycles and parallel
    relations) as collect_paths, without enumerating or truncating those walks.
    """
    from xgraph.mintqa_panels import indexed

    if type(max_paths) is not int or max_paths < 1:
        raise ValueError("GCR max_paths must be a positive integer")
    if not depths or any(type(d) is not int or not 1 <= d <= 4 for d in depths):
        raise ValueError("GCR admission depths must be integers in [1, 4]")
    depths = sorted(set(depths))
    if panels.get("pilot") != reference.get("pilot"):
        raise ValueError("GCR and basepilot2 pilot selections differ")
    members = panels["panels"]["development"]["members"]
    if members != reference["panels"]["development"]["members"]:
        raise ValueError("GCR and basepilot2 members or weights differ")
    rows, other = indexed(requests), indexed(reference_requests)
    ids = [r["id"] for r in members]
    if len(set(ids)) != len(ids) or set(rows) != set(ids) or set(other) != set(ids):
        raise ValueError("GCR and basepilot2 require identical complete pilot ids")
    result = {
        "model_calls": 0,
        "questions": len(ids),
        "ids": ids,
        "max_paths": max_paths,
        "admitted": True,
        "per_question": [],
        "depths": {},
    }
    for qid in ids:
        request = rows[qid]
        if any(
            request.get(k) != other[qid].get(k)
            for k in (
                "question",
                "messages",
                "anchor",
                "topics",
                "kg_version",
            )
        ):
            raise ValueError(f"GCR {qid}: starts or question differ from basepilot2")
        state = _State({**request, "question": request.get("question") or "audit"}, store)
        if state.start_source == "generated" or request.get("kg_version"):
            raise ValueError(f"GCR {qid}: admission requires unversioned linked requests")
        frontier = Counter({int(q[1:]): 1 for q in state.start})
        by_hop = {}
        for hop in range(1, max(depths) + 1):
            following = Counter()
            for item, multiplicity in frontier.items():
                subject = store.label(item)
                for relation, obj, relation_label, label in store.outgoing_edges(item):
                    serialize_path(
                        KGPath((item, obj), (relation,), (subject, relation_label, label))
                    )
                    following[obj] += multiplicity
            frontier = following
            by_hop[str(hop)] = sum(frontier.values())
        result["per_question"].append({"id": qid, "paths_by_hop": by_hop})
    for depth in depths:
        counts = [
            sum(r["paths_by_hop"][str(h)] for h in range(1, depth + 1))
            for r in result["per_question"]
        ]
        overflow = [q for q, n in zip(ids, counts, strict=True) if n > max_paths]
        result["depths"][str(depth)] = {
            "graph_paths": sum(counts),
            "max_graph_paths": max(counts, default=0),
            "overflow_questions": overflow,
            "empty_questions": counts.count(0),
        }
        result["admitted"] &= not overflow
    return result


def depth_behavior(records):
    """Describe actual returned depths; a four-hop bound is not four-hop behaviour."""
    counts, deep = Counter(), 0
    for row in records:
        depths = []
        for sample in row["samples"]:
            if not sample["valid_path"]:
                continue
            hops = sample["path"].count(" -> ") // 2
            if hops != sample.get("path_hops") or not 1 <= hops <= row["config"]["hops"]:
                raise ValueError(f"GCR {row['id']}: invalid returned path depth metadata")
            depths.append(hops)
            counts[str(hops)] += 1
        deep += any(h > 2 for h in depths)
    return {
        "questions": len(records),
        "returned_paths_by_hop": dict(sorted(counts.items())),
        "questions_with_depth_above_two": deep,
        "beam_shortfalls": sum(r["status"] == "beam_shortfall" for r in records),
        "empty_tries": sum(bool(r["empty_trie"]) for r in records),
        "truncated_tries": sum(bool(r["trie"].get("truncated")) for r in records),
        "searches_at_token_limit": sum(
            bool(r.get("beam_search", {}).get("token_limited")) for r in records
        ),
        "truncated_returned_sequences": sum(
            bool(s["truncated"]) for r in records for s in r["samples"]
        ),
        "status_counts": dict(Counter(r["status"] for r in records)),
    }


def main(argv: list[str] | None = None) -> None:
    """CPU synthesis replay; generation uses xgraph.generate and CoR builds linked requests."""
    import argparse

    from xgraph.cost import StageTimer
    from xgraph.provenance import provenance

    parser = argparse.ArgumentParser(prog="python -m xgraph.gcr")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="CPU pilot identity and exact path-count admission"
    )
    for name in ("panels", "reference-panels", "requests", "reference-requests", "db", "out"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    prepare.add_argument("--max-paths", type=int, default=MAX_PATHS)
    prepare.add_argument("--depth", type=int, action="append", choices=range(1, 5))
    depths = commands.add_parser("depth-report", help="CPU audit of completed GCR pilot outputs")
    for name in ("panels", "directory", "out"):
        depths.add_argument(f"--{name}", type=Path, required=True)
    arm_depths = {
        "gcr-shared-d2": 2,
        "gcr-native-d2": 2,
        "gcr-shared-d4": 4,
        "gcr-native-d4": 4,
    }
    depths.add_argument("--arm", action="append", choices=arm_depths)
    replay = commands.add_parser("replay", help="reparse saved synthesis without model calls")
    replay.add_argument(
        "--records", type=Path, required=True, help="GCR generation records (JSONL)"
    )
    replay.add_argument("--out", type=Path, required=True, help="replayed records (JSONL)")
    args = parser.parse_args(argv)
    if args.command == "depth-report":
        from xgraph.mintqa_panels import read_jsonl, write_json
        from xgraph.mintqa_rematch import audit_complete_usage

        members = json.loads(args.panels.read_text())["panels"]["development"]["members"]
        expected = {r["id"] for r in members}
        arms = args.arm or ("gcr-shared-d4", "gcr-native-d4", "gcr-native-d2")
        report, inputs = {}, [args.panels]
        for family in ("qwen38", "gemma31"):
            report[family] = {}
            for arm in arms:
                hops = arm_depths[arm]
                path = args.directory / family / "development" / f"{arm}.jsonl"
                rows = read_jsonl(path)
                audit_complete_usage(rows, expected)
                if any(
                    r["config"]["k"] != 10
                    or r["config"]["hops"] != hops
                    or r["config"]["decoding"] != "beam"
                    or r.get("prompts", {}).get("profile") != "paper_zero_shot"
                    for r in rows
                ):
                    raise ValueError(f"{family}/{arm}: incompatible GCR protocol")
                report[family][arm] = depth_behavior(rows)
                if all("hops" in m for m in members):
                    report[family][arm].update(
                        attempted=len(rows),
                        configured_trie_hops=hops,
                        unreachable_by_construction_ids=[
                            m["id"] for m in members if m["hops"] > hops
                        ],
                        by_gold_hop={
                            str(h): {
                                "attempted": sum(m["hops"] == h for m in members),
                                "unreachable_by_construction": sum(
                                    m["hops"] == h and h > hops for m in members
                                ),
                            }
                            for h in sorted({m["hops"] for m in members})
                        },
                    )
                inputs += [path, Path(str(path) + ".provenance.json")]
        if args.out.resolve() in {p.resolve() for p in inputs}:
            parser.error("--out must differ from the input paths")
        write_json(
            args.out,
            {
                "families": report,
                "native_reference_trie_hops": 2,
                "unreachable_by_construction_definition": (
                    "released gold-chain hops exceed the trie bound; all questions remain "
                    "attempted and scored. This does not rule out shorter KG routes or "
                    "a correct unconstrained hypothesis or reader answer."
                ),
                "provenance": provenance([*inputs, Path(__file__)]),
            },
        )
        return
    if args.command == "prepare":
        from xgraph.mintqa_panels import read_jsonl, write_json

        inputs = [
            args.panels,
            args.reference_panels,
            args.requests,
            args.reference_requests,
            args.db,
        ]
        if args.out.resolve() in {p.resolve() for p in inputs}:
            parser.error("--out must differ from the input paths")
        timer = StageTimer()
        report = prepare_pilot(
            json.loads(args.panels.read_text()),
            json.loads(args.reference_panels.read_text()),
            read_jsonl(args.requests),
            read_jsonl(args.reference_requests),
            KGStore(args.db),
            max_paths=args.max_paths,
            depths=args.depth or (2, 4),
        )
        report.update(
            artifact_kind="xgraph.gcr_pilot_admission",
            version=1,
            provenance=provenance([*inputs, Path(__file__)]),
            **timer.summary(),
        )
        write_json(args.out, report)
        print(
            json.dumps({k: v for k, v in report.items() if k not in ("provenance", "per_question")})
        )
        if not report["admitted"]:
            raise SystemExit(
                "GCR exact tries exceed max_paths; no generation admitted, no truncation"
            )
        return
    inputs = [args.records]
    source = Path(str(args.records) + ".provenance.json")
    if source.exists():
        inputs.append(source)
    if args.out.resolve() in {p.resolve() for p in inputs}:
        parser.error("--out must differ from the input paths")
    timer = StageTimer()
    with args.records.open(encoding="utf-8") as lines:
        rows, summary = replay_syntheses([json.loads(line) for line in lines if line.strip()])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as written:
        for row in rows:
            written.write(json.dumps(row, ensure_ascii=False) + "\n")
    if source.exists():
        summary["source_generation"] = json.loads(source.read_text(encoding="utf-8"))
    summary.update({
        "artifact_kind": "xgraph.gcr_replay", "version": 1, "seed": None,
        "provenance": provenance([*inputs, Path(__file__), Path(__file__).with_name("cor.py")]),
        **timer.summary(),
    })  # fmt: skip
    Path(str(args.out) + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        k: v for k, v in summary.items() if k not in {"provenance", "source_generation"}
    }))  # fmt: skip


if __name__ == "__main__":
    main()
