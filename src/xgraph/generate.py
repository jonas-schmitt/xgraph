"""Greedy generation for the quality gate, run inside the vLLM container.

    python -m xgraph.generate --model /model --requests requests.jsonl \\
        --arm constrained --out outputs.jsonl

Reads chat requests, one JSON object per line with an `id` and OpenAI-style
`messages`, generates greedily for all of them in one batch, and writes one record
per request, in request order: id, arm, text, token ids, finish reason, and prompt
length, plus a summary next to it (`<out>.summary.json`). Thinking is turned off by
default. --think-budget enables an unconstrained native thought stage for baseline
and constrained answers; --thoughts reuses its saved thoughts by request id.

--observe-paths K / XGRAPH_OBSERVE=K enables observed frontier paths for the constrained
arm (0/unset off). The maintained path demonstrations are converted automatically,
with questions and answers kept. generated_tokens includes the observation decode
steps; observation_tokens separately counts generated tokens in committed observation
blocks, including truncated payloads and excluding replayed prompt tokens. Accounting
runs once after generation; no host work participates in forcing a token. Custom
demonstrations and replayed prefills must already use the observed language and its
canonical piece tokenization (docs/observed-paths-adr.md).

The single-call arms differ only in the engine: the constrained arm loads the xgraph logits
processor (which needs XGRAPH_KG_INDEX and XGRAPH_TOKENIZER, see xgraph.vllm) and opts
every request in, passing on a request's `anchor` (entity ids) and `kg_version` (a KG
version's name, with XGRAPH_KG_VERSIONS set, see xgraph.kgversions); the baseline arm
does not load it. The tools, walk, iterative, agent and cor arms read their KG store
(--tools-db) through the KG version a request names, from the version file
--kg-versions; the iterative arm leaves out the facts that would make a prompt longer
than --max-model-len minus --max-tokens; the agent arm (xgraph.agent) knows relations by
their labels and aliases with --relation-labels, and spends at most --max-calls tool
calls (default xgraph.agent.MAX_CALLS); --matched-demos shows it each request's own path
demonstrations as tool trajectories, --matched-paths executes its execute_path as
matched execution over --kg-index and --tokenizer (with --start-rank and
--unquoted-starts), and --pages lets its results be paged by offset. --agent-feedback
enables these matched options, reports failed-hop relations and enforces a retry. With
--written-start the iterative arm first asks the model for the question's start entity,
whose facts open round 0. The cor arm (xgraph.cor) searches relation chains depth first
with separate reasoning, filtering and direct fallback. It defaults to the released
control flow and requires --cor-prompts (local YAML). --max-calls optionally bounds
all model calls and --cor-depth bounds chain length. --cor-mode reconstructed retains
the historical adaptation, including --relation-labels for its menu aliases.
The gcr arm builds its own bounded entity-relation trie over the KG store, generates
--gcr-k constrained paths with free hypotheses, then synthesizes an answer. By default
one model handles both stages; --gcr-synthesis-model reloads a second model after paths.
With --open (the constrained and fallback arms), every request
gets a final assistant message holding the span opener of the KG index's syntax, which
is continued, and starts inside a span in the engine (xgraph_open): a forced opener, so
that the answer begins with a span, in path mode a path; its record's text starts with
the opener. A request with `prefill_replay: true` (the constrained arm) ends with a
final assistant message, such as a path span, which is continued and which the engine
replays (xgraph_prefill, a replayed prefill), so that what the model writes next is
masked as if it had written the prefill itself, in path mode bound to the path's
frontier; its record's text is the prefill and the completion, and its `prefill` field
the prefill. Both ignore the model's own sampling defaults, use a fixed KV cache size
(which also skips vLLM's memory profiling, see DEVELOPMENT.md), and leave the vision
encoder of a multimodal model unloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xgraph.cost import distributions, request_cost
from xgraph.rag import CAP as RETRIEVAL_CAP
from xgraph.walk import CLASSLESS, OBSERVATIONS, SLOTTED

if TYPE_CHECKING:
    from xgraph.cor import CorPrompts
    from xgraph.dog import DoGConfig, DoGPrompts
    from xgraph.gcr import GCRConfig, GCRPrompts, GCRScreenBudget
    from xgraph.syntax import SpanSyntax
    from xgraph.tog import ToGConfig, ToGPrompts
    from xgraph.tokenizer import ServedTokenizer

ARMS: dict[str, dict | None] = {
    "baseline": None,
    "constrained": {"xgraph": 1},
    "tools": None,
    "fallback": {"xgraph": 1},
    "walk": None,
    "resume": None,
    "iterative": None,
    "agent": None,
    "cor": None,
    "tog": None,
    "dog": None,
    "dog-gold": None,
    "gcr": None,
}
"""tools: unconstrained, with KG lookups through function calls (xgraph.toolkg), read
through a request's KG version.
fallback: constrained, then unconstrained for the answers that escaped or hit the token
limit (see fallback).
walk: unconstrained, the model choosing KG relations round by round (xgraph.walk), told
what they reached as the walk observation --observe. With --matched-paths it reads
the index's whole capped frontier, writable labels and committed delivery.
resume: a free path, executed by MatchedExecutor; the first unavailable relation
gets prefix/frontier/relation-page feedback for at most --resume-max extra calls
(xgraph.resume). No model logits or engine mask are used during planning.
iterative: unconstrained, the topic entities' one-hop facts in the prompt, then those of
each entity the model asks for next, up to --max-calls lookups (xgraph.iterative).
agent: unconstrained, the KG agent calling tools over the KG store, a whole-path tool
among them (xgraph.agent).
cor: unconstrained, Chain-of-Relations search over the KG store (xgraph.cor).
gcr: its own per-request KG-Trie and two-stage synthesis (xgraph.gcr)."""
PROCESSOR = "xgraph.vllm:XGraphLogitsProcessor"
GCR_PROCESSOR = "xgraph.path_trie:GCRLogitsProcessor"


def reader_scheduling(args):
    """One scheduling definition shared by provenance and measured summaries."""
    if args.arm not in ("tog", "dog", "dog-gold", "cor", "gcr"):
        return {}
    if args.question_timings:
        scheduling, batch_size = "serial", 1
    elif args.arm == "cor":
        scheduling, batch_size = "native_batched", args.max_num_seqs
    elif args.arm == "gcr":
        scheduling, batch_size = "step_synchronous", args.gcr_batch_size or 8
    else:
        reader = "dog" if args.arm == "dog-gold" else args.arm
        scheduling, batch_size = "step_synchronous", getattr(args, f"{reader}_batch_size") or 32
    return {"question_scheduling": scheduling, "question_batch_size": batch_size}


def run_reader(reader, llm, requests, *args, question_timings=False, **kwargs):
    """Optionally measure whole questions serially, without apportioning batch time."""
    if not question_timings:
        return reader(llm, requests, *args, **kwargs)
    records = []
    for request in requests:
        started = time.perf_counter()
        rows = reader(llm, [request], *args, **kwargs)
        elapsed = time.perf_counter() - started
        if len(rows) != 1 or rows[0]["id"] != request["id"]:
            raise ValueError("timed reader must return exactly its requested question")
        records.append({**rows[0], "question_wall_seconds": elapsed})
    return records


def load_requests(path: str | Path) -> list[dict]:
    requests = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as lines:
        for line in lines:
            if not line.strip():
                continue
            request = json.loads(line)
            if request["id"] in seen:
                raise ValueError(f"request id {request['id']!r} appears twice in {path}")
            seen.add(request["id"])
            requests.append(request)
    return requests


def _arm(arm: str) -> dict | None:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
    return ARMS[arm]


def sampling_params(
    arm: str,
    max_tokens: int,
    params_class: Any,
    repetition_penalty: float = 1.0,
    banned: list[int] | None = None,
    anchor: list[str] | None = None,
    sampling: tuple[float, float] | None = None,
    seed: int | None = None,
    kg_version: str | None = None,
    opened: bool = False,
    prefill: str | None = None,
    question: str | None = None,
    replay_id: str | None = None,
    relation_ranks: list | None = None,
) -> Any:
    """Greedy decoding, or with sampling (temperature, top-p) and the request's seed;
    the constrained arm opts its requests in to the engine, with the request's anchor
    (entity ids) and KG version (a name) where it has them, with opened, starts them
    inside a span (their prompts end with the opener), and with a prefill (the text their
    prompts end with), has the engine replay it.

    banned token ids get a logit bias of -100: Gemma 4, for one, opens a thinking
    channel (`<|channel>`) though the chat template closes it when thinking is off.
    """
    extra: dict[str, Any] = {}
    if repetition_penalty != 1.0:
        extra["repetition_penalty"] = repetition_penalty
    if banned:
        extra["logit_bias"] = {token: -100.0 for token in banned}
    arguments = engine_arguments(arm, anchor, kg_version, opened, prefill, question, relation_ranks)
    if replay_id is not None:
        if arguments is None or not prefill:
            raise ValueError("a replay status id needs a constrained replayed prefill")
        arguments = {**arguments, "xgraph_replay_id": replay_id}
    if sampling is not None:
        extra.update(temperature=sampling[0], top_p=sampling[1], seed=seed)
    else:
        extra["temperature"] = 0.0
    return params_class(max_tokens=max_tokens, extra_args=arguments, **extra)


def engine_arguments(
    arm: str,
    anchor: list[str] | None = None,
    kg_version: str | None = None,
    opened: bool = False,
    prefill: str | None = None,
    question: str | None = None,
    relation_ranks: list | None = None,
) -> dict | None:
    """The per-request arguments the arm gives the engine (xgraph.vllm), None for an arm
    without it: the opt-in, with the anchor (entity ids) and KG version (a name) where
    given, xgraph_open for an opened request, and xgraph_prefill for a replayed prefill.
    An arm without the engine drops the anchor and KG version, and refuses the other two."""
    arguments = _arm(arm)
    if arguments is not None and anchor is not None:
        arguments = {**arguments, "xgraph_anchor": list(anchor)}
    if arguments is not None and kg_version:
        arguments = {**arguments, "xgraph_kg_version": str(kg_version)}
    if arguments is not None and question is not None:
        arguments = {**arguments, "xgraph_question": question}
    if arguments is not None and relation_ranks is not None:
        from xgraph.audit import checked_relation_ranks

        arguments = {**arguments, "xgraph_relation_ranks": checked_relation_ranks(relation_ranks)}
    if opened:
        if arguments is None:
            raise ValueError(f"an opened request needs the engine, which the {arm} arm lacks")
        arguments = {**arguments, "xgraph_open": 1}
    if prefill:
        if arguments is None:
            raise ValueError(f"a replayed prefill needs the engine, which the {arm} arm lacks")
        arguments = {**arguments, "xgraph_prefill": prefill}
    return arguments


def request_arguments(arm: str, request: dict, opened: bool = False) -> dict | None:
    """engine_arguments from a request's own fields: its anchor, its kg_version, and with
    prefill_replay, its final assistant message as the replayed prefill (replayed_prefill);
    opened when the request is to start inside a span. xgraph.serving sends these."""
    return engine_arguments(
        arm,
        request.get("anchor"),
        request.get("kg_version"),
        opened,
        replayed_prefill(request),
        _stop_question(request) if _arm(arm) is not None else None,
        request_relation_ranks(request),
    )


def request_relation_ranks(request: dict):
    """An absent field preserves legacy calls; explicit null is malformed input."""
    if "relation_ranks" not in request:
        return None
    from xgraph.audit import checked_relation_ranks

    try:
        return checked_relation_ranks(request["relation_ranks"])
    except ValueError as error:
        raise ValueError(f"request {request.get('id')!r}: {error}") from error


def _stop_question(request: dict) -> str | None:
    if (
        not os.environ.get("XGRAPH_STOP")
        and os.environ.get("XGRAPH_AUDIT") != "1"
        and not int(os.environ.get("XGRAPH_MENU") or "0")
    ):
        return None
    question = next(
        (m["content"] for m in reversed(request["messages"]) if m["role"] == "user"), None
    )
    if not isinstance(question, str) or not question.strip():
        rule = (
            "audit"
            if os.environ.get("XGRAPH_AUDIT") == "1"
            else "menu"
            if int(os.environ.get("XGRAPH_MENU") or "0")
            else "residual-goal STOP"
        )
        raise ValueError(f"request {request['id']!r}: {rule} needs a user question")
    return question


CONTINUED = {"continue_final_message": True, "add_generation_prompt": False}
"""Chat template options that continue a final assistant message instead of answering it."""


def replayed_prefill(request: dict) -> str | None:
    """The final assistant message a request with prefill_replay has the engine replay."""
    if not request.get("prefill_replay"):
        return None
    last = request["messages"][-1] if request["messages"] else {}
    if last.get("role") != "assistant" or not last.get("content"):
        raise ValueError(
            f"request {request['id']!r} has prefill_replay but no final assistant message "
            "holding the text to replay"
        )
    return last["content"]


def opened_requests(requests: list[dict], opener: str) -> list[dict]:
    """The requests with a final assistant message holding the span opener, to continue."""
    for r in requests:
        if r["messages"] and r["messages"][-1]["role"] == "assistant":
            raise ValueError(
                f"request {r['id']!r} already ends with an assistant turn; "
                "an opener is a final assistant message of its own"
            )
    return [
        {**r, "messages": [*r["messages"], {"role": "assistant", "content": opener}]}
        for r in requests
    ]


def sampled_requests(requests: list[dict], samples: int, seed: int = 0) -> list[dict]:
    """Each request repeated samples times, as id#k with seed seed + k (xgraph.vote
    keeps the answer most samples give)."""
    return [
        {**r, "id": f"{r['id']}#{k}", "seed": seed + k} for r in requests for k in range(samples)
    ]


def engine_args(
    arm: str,
    model: str,
    *,
    kv_cache_bytes: int,
    max_model_len: int = 8192,
    max_num_seqs: int = 64,
    prefix_caching: bool = True,
    seed: int = 0,
    memory_share: float = 0.45,
    speculative_tokens: int = 0,
    lora_rank: int = 0,
) -> dict:
    """Keyword arguments for vllm.LLM."""
    args: dict[str, Any] = {
        "model": model,
        "generation_config": "vllm",
        "kv_cache_memory_bytes": kv_cache_bytes,
        # Checked against free memory at startup, though the KV cache size is fixed.
        "gpu_memory_utilization": memory_share,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "enable_prefix_caching": prefix_caching,
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "seed": seed,
    }
    if _arm(arm) is not None:
        args["logits_processors"] = [PROCESSOR]
        args["worker_extension_cls"] = "xgraph.vllm.XGraphWorkerExtension"
    if arm in ("gcr", "dog", "dog-gold"):
        args["logits_processors"] = [GCR_PROCESSOR]
    if lora_rank:  # one fine-tuned adapter (xgraph.train) for every request
        args.update(enable_lora=True, max_lora_rank=lora_rank, max_loras=1)
    if speculative_tokens:  # drafts from the model's multi-token-prediction head
        args["speculative_config"] = {"method": "mtp", "num_speculative_tokens": speculative_tokens}
    return args


def generate(
    llm: Any,
    requests: list[dict],
    arm: str,
    max_tokens: int,
    params_class: Any,
    repetition_penalty: float = 1.0,
    lora_request: Any = None,
    banned: list[int] | None = None,
    sampling: tuple[float, float] | None = None,
    opener: str | None = None,
    *,
    thoughts: dict[str, dict] | None = None,
    reasoning_effort: str = "medium",
    replay_diagnostics: dict | None = None,
) -> list[dict]:
    """One record per request, in request order. A request's anchor (entity ids) goes to
    the engine as its derivation's starting entities, and its kg_version as the KG version
    it reads (xgraph.vllm); with sampling (temperature, top-p), each request samples with
    its own seed (sampled_requests). With an opener (the span opener), each request ends
    with it as a final assistant message and starts inside a span (opened_requests), and
    its record's text starts with it. A request with prefill_replay has the engine replay
    its final assistant message (replayed_prefill), and its record's text starts with it."""
    opened = opener is not None
    engine_opened = opened and (thoughts is None or _arm(arm) is not None)
    prefills = [replayed_prefill(r) for r in requests]
    from uuid import uuid4

    replay_ids = [uuid4().hex if prefill else None for prefill in prefills]
    params = sampling_params(
        arm, max_tokens, params_class, repetition_penalty, banned, opened=engine_opened
    )
    per_request = _arm(arm) is not None and (
        bool(os.environ.get("XGRAPH_STOP"))
        or os.environ.get("XGRAPH_AUDIT") == "1"
        or bool(int(os.environ.get("XGRAPH_MENU") or "0"))
        or any("anchor" in r or r.get("kg_version") or "relation_ranks" in r for r in requests)
    )
    if per_request or sampling is not None or any(prefills):
        params = [
            sampling_params(
                arm, max_tokens, params_class, repetition_penalty, banned,
                r.get("anchor") if per_request else None, sampling, r.get("seed"),
                r.get("kg_version") if per_request else None, engine_opened, prefill,
                _stop_question(r) if per_request else None,
                replay_id=replay_id,
                relation_ranks=request_relation_ranks(r) if per_request else None,
            )
            for r, prefill, replay_id in zip(requests, prefills, replay_ids, strict=True)
        ]  # fmt: skip
    if opened:
        requests = opened_requests(requests, opener)
    if thoughts is not None:
        requests = thought_requests(requests, thoughts)
    prefilled = {r["messages"][-1]["role"] == "assistant" for r in requests}
    if len(prefilled) > 1:
        raise ValueError("requests mix prefilled assistant turns with new ones")
    # A final assistant message (the reveal arm's prefill, an opener) is continued.
    continued = CONTINUED if prefilled == {True} else {}
    outputs = llm.chat(
        [r["messages"] for r in requests],
        params,
        chat_template_kwargs=(
            {
                "enable_thinking": False,
                "preserve_thinking": True,
                "reasoning_effort": reasoning_effort,
            }
            if thoughts is not None
            else {"enable_thinking": False}
        ),
        use_tqdm=False,
        lora_request=lora_request,
        **continued,
    )
    from xgraph.vllm import replay_statuses

    statuses = (
        replay_statuses(
            llm, [key for key in replay_ids if key is not None], diagnostics=replay_diagnostics
        )
        if any(prefills)
        else {}
    )
    records = []
    for request, output, prefill, replay_id in zip(
        requests, outputs, prefills, replay_ids, strict=True
    ):
        completion = output.outputs[0]
        records.append(
            {
                "id": request["id"],
                "arm": arm,
                "text": (opener or prefill or "") + completion.text,
                "token_ids": list(completion.token_ids),
                "finish_reason": completion.finish_reason,
                "prompt_tokens": len(output.prompt_token_ids),
                "model_calls": 1,
                **({"opener": opener} if opened else {}),
                **({"prefill": prefill} if prefill else {}),
                **(statuses[replay_id] if replay_id is not None else {}),
            }
        )
    return records


def thought_requests(requests: list[dict], thoughts: dict[str, dict]) -> list[dict]:
    """Continue answer content after a native, closed thought in the same assistant turn.

    Only content is an answer prefill; reasoning is ordinary prompt history
    and must never be supplied to xgraph_prefill.
    """
    result = []
    for request in requests:
        messages = request["messages"]
        if messages and messages[-1]["role"] == "assistant":
            last, before = messages[-1], messages[:-1]
        else:
            last, before = {"role": "assistant", "content": ""}, messages
        result.append(
            {
                **request,
                "messages": [
                    *before,
                    {
                        # Gemma omits a falsey thought entirely. A newline makes its native
                        # empty channel close even when a tiny budget produced no body.
                        **last,
                        # vLLM's chat parser reads reasoning and recreates both aliases;
                        # reasoning_content alone is discarded before template rendering.
                        "reasoning": thoughts[request["id"]]["thought"] or "\n",
                        "reasoning_content": thoughts[request["id"]]["thought"] or "\n",
                    },
                ],
            }
        )
    return result


def generate_thinking(
    llm: Any,
    requests: list[dict],
    arm: str,
    max_tokens: int,
    params_class: Any,
    *,
    think_budget: int,
    thoughts_out: Path,
    thoughts_file: Path | None = None,
    seed: int = 0,
    reasoning_effort: str = "medium",
    repetition_penalty: float = 1.0,
    lora_request: Any = None,
    banned: list[int] | None = None,
    opener: str | None = None,
    sampling: tuple[float, float] | None = None,
    source_provenance: dict | None = None,
) -> tuple[list[dict], dict]:
    """Unconstrained native thinking, then answer-only generation with full usage.

    Thought-stage tokens include the native channel delimiters when generated. Wall
    seconds belong to the whole batch, never apportioned into request latencies.
    """
    if think_budget <= 0 or arm not in ("baseline", "constrained"):
        raise ValueError("thinking needs a positive budget and baseline or constrained arm")
    tokenizer = llm.get_tokenizer()
    template = tokenizer.get_chat_template()
    if "reasoning_content" not in template:
        raise ValueError("thinking needs a chat template accepting reasoning_content")
    if "</think>" in template:
        end, start = "</think>", "<think>"
    elif "<channel|>" in template:
        end, start = "<channel|>", "<|channel>thought\n"
    else:
        raise ValueError("thinking needs a Qwen <think> or Gemma thought-channel template")
    template_kwargs = {
        "enable_thinking": True,
        "preserve_thinking": True,
        "reasoning_effort": reasoning_effort,
    }
    template_hash = hashlib.sha256(template.encode()).hexdigest()
    tokenizer_hash = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    # Answer prefills belong to stage 2, never to the thought prompt.
    conversations = [
        r["messages"][:-1]
        if r["messages"] and r["messages"][-1]["role"] == "assistant"
        else r["messages"]
        for r in requests
    ]
    reused = thoughts_file is not None
    if reused:
        saved = {r["id"]: r for r in load_requests(thoughts_file)}
        thoughts = {}
        for request, messages in zip(requests, conversations, strict=True):
            key = request["id"]
            if key not in saved:
                raise ValueError(f"request {key!r}: missing thought in {thoughts_file}")
            thought = saved[key]
            expected = {
                "artifact_kind": "xgraph.thought",
                "version": 1,
                "think_budget": think_budget,
                "reasoning_effort": reasoning_effort,
                "template_sha256": template_hash,
                "tokenizer_sha256": tokenizer_hash,
            }
            for field, value in expected.items():
                if thought.get(field) != value:
                    raise ValueError(f"request {key!r}: incompatible thought {field}")
            if _thought_question(thought.get("messages", [])) != _thought_question(messages):
                raise ValueError(f"request {key!r}: thought belongs to a different question")
            if not isinstance(thought.get("thought"), str):
                raise ValueError(f"request {key!r}: thought must be text")
            for field in ("prompt_tokens", "generated_tokens", "model_calls"):
                value = thought.get(field)
                if type(value) is not int or value < 0:
                    raise ValueError(f"request {key!r}: invalid thought cost {field}")
            tokens = thought.get("token_ids")
            if (
                not isinstance(tokens, list)
                or any(type(t) is not int or t < 0 for t in tokens)
                or len(tokens) != thought["generated_tokens"]
                or len(tokens) > think_budget
                or thought["model_calls"] != 1
            ):
                raise ValueError(f"request {key!r}: invalid thought token usage")
            batch_seconds = thought.get("batch_seconds")
            if (
                type(batch_seconds) not in (int, float)
                or not math.isfinite(batch_seconds)
                or batch_seconds < 0
            ):
                raise ValueError(f"request {key!r}: missing thought batch_seconds")
            if type(thought.get("truncated")) is not bool:
                raise ValueError(f"request {key!r}: missing thought truncation status")
            thoughts[key] = thought
        source_times = {t["batch_seconds"] for t in thoughts.values()}
        if len(source_times) > 1:
            raise ValueError("thoughts must come from one recorded batch")
        thought_seconds = next(iter(source_times), 0.0)
    else:
        from xgraph.provenance import provenance

        if source_provenance is None:
            source_provenance = provenance([])
        params = [
            params_class(
                max_tokens=think_budget,
                temperature=0.0,
                seed=r.get("seed", seed),
                extra_args=None,
                stop=[end],
                include_stop_str_in_output=True,
                skip_special_tokens=False,
            )
            for r in requests
        ]
        started = time.perf_counter()
        outputs = llm.chat(
            conversations,
            params,
            chat_template_kwargs=template_kwargs,
            use_tqdm=False,
            lora_request=lora_request,
        )
        thought_seconds = time.perf_counter() - started
        thoughts = {}
        for request, messages, output in zip(requests, conversations, outputs, strict=True):
            completion = output.outputs[0]
            raw = completion.text
            text = "" if raw and start.startswith(raw) else raw.removeprefix(start)
            text = text.split(end, 1)[0].strip()
            thoughts[request["id"]] = {
                "artifact_kind": "xgraph.thought",
                "version": 1,
                "id": request["id"],
                "thought": text,
                "raw_text": raw,
                "token_ids": list(completion.token_ids),
                "prompt_tokens": len(output.prompt_token_ids),
                "generated_tokens": len(completion.token_ids),
                "model_calls": 1,
                "finish_reason": completion.finish_reason,
                "truncated": end not in raw,
                "think_budget": think_budget,
                "seed": request.get("seed", seed),
                "reasoning_effort": reasoning_effort,
                "template_sha256": template_hash,
                "tokenizer_sha256": tokenizer_hash,
                "messages": messages,
                "batch_seconds": thought_seconds,
                "provenance": source_provenance,
            }
    if thoughts_file is None or thoughts_out.resolve() != thoughts_file.resolve():
        thoughts_out.parent.mkdir(parents=True, exist_ok=True)
        with thoughts_out.open("w", encoding="utf-8") as out:
            for thought in thoughts.values():
                out.write(json.dumps(thought, ensure_ascii=False) + "\n")
    started = time.perf_counter()
    records = generate(
        llm,
        requests,
        arm,
        max_tokens,
        params_class,
        repetition_penalty,
        lora_request,
        banned,
        sampling,
        opener,
        thoughts=thoughts,
        reasoning_effort=reasoning_effort,
    )
    answer_seconds = time.perf_counter() - started
    for record in records:
        thought = thoughts[record["id"]]
        answer_cost = {
            "prompt_tokens": record["prompt_tokens"],
            "generated_tokens": len(record["token_ids"]),
            "model_calls": 1,
        }
        thought_cost = {k: thought[k] for k in answer_cost}
        record.update(
            thought=thought["thought"],
            thought_tokens=thought["generated_tokens"],
            thought_token_ids=thought["token_ids"],
            thought_truncated=thought["truncated"],
            stages={"thought": thought_cost, "answer": answer_cost},
            calls=2,
            executed_calls=1 if reused else 2,
            prompt_tokens=thought_cost["prompt_tokens"] + answer_cost["prompt_tokens"],
            model_calls=thought_cost["model_calls"] + answer_cost["model_calls"],
            generated_tokens=thought_cost["generated_tokens"] + answer_cost["generated_tokens"],
        )
    return records, {
        "think_budget": think_budget,
        "reasoning_effort": reasoning_effort,
        "thoughts_file": str(thoughts_out),
        "source_thoughts_file": str(thoughts_file) if reused else None,
        "reused": reused,
        "executed_thought_seconds": 0.0 if reused else thought_seconds,
        "thought_tokens": sum(t["generated_tokens"] for t in thoughts.values()),
        "stage_prompt_tokens": {
            stage: sum(r["stages"][stage]["prompt_tokens"] for r in records)
            for stage in ("thought", "answer")
        },
        "stage_seconds": {"thought": thought_seconds, "answer": answer_seconds},
    }


def _thought_question(messages: list[dict]) -> Any:
    """Reuse keeps the question fixed, while planner instructions/demos may differ."""
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content")
    raise ValueError("thinking needs a user question")


def fallback(
    llm: Any,
    requests: list[dict],
    max_tokens: int,
    params_class: Any,
    repetition_penalty: float = 1.0,
    lora_request: Any = None,
    banned: list[int] | None = None,
    escape: str = "⁇",
    opener: str | None = None,
) -> list[dict]:
    """Constrained generation, then unconstrained generation for the requests whose
    constrained answer wrote the escape marker or hit the token limit: an escape says
    the KG cannot express what the model wants to write, so the answer falls back to
    the model alone. Both passes run on one vLLM instance (requests opt in to the
    constraint one by one). A record keeps the constrained pass's text, token ids, and
    finish reason of a fallback, and counts the prompt and generated tokens of both
    passes. With an opener, only the constrained pass starts inside a span; the answer
    again is the model alone, on the request as it came."""
    first = generate(
        llm, requests, "constrained", max_tokens, params_class, repetition_penalty, lora_request,
        banned, opener=opener,
    )  # fmt: skip
    from xgraph.score import replay_failure

    flagged = [
        i
        for i, r in enumerate(first)
        if not replay_failure(r) and (r["finish_reason"] == "length" or escape in r["text"])
    ]
    second = generate(
        llm, [requests[i] for i in flagged], "baseline", max_tokens, params_class,
        repetition_penalty, lora_request, banned,
    ) if flagged else []  # fmt: skip
    records = [dict(r, arm="fallback", fallback=False) for r in first]
    for i, redo in zip(flagged, second, strict=True):
        records[i] = dict(
            redo,
            arm="fallback",
            fallback=True,
            constrained_text=first[i]["text"],
            constrained_token_ids=first[i]["token_ids"],
            constrained_finish_reason=first[i]["finish_reason"],
            generated_tokens=len(first[i]["token_ids"]) + len(redo["token_ids"]),
            prompt_tokens=first[i]["prompt_tokens"] + redo["prompt_tokens"],
            model_calls=first[i]["model_calls"] + redo["model_calls"],
        )
    return records


def walk_requests(requests: list[dict]) -> list[dict]:
    """The walk arm's questions: each request's last user message, with its anchor and KG
    version."""
    return [
        {
            "id": r["id"],
            "question": next(m["content"] for m in reversed(r["messages"]) if m["role"] == "user"),
            **({"answer_mode": r["answer_mode"]} if "answer_mode" in r else {}),
            **({"anchor": r["anchor"]} if r.get("anchor") else {}),
            **({"kg_version": r["kg_version"]} if r.get("kg_version") else {}),
        }
        for r in requests
    ]


def model_revision(model: str | Path) -> str | None:
    """The commit a local model was downloaded at (`hf download --local-dir`), if recorded."""
    meta = Path(model) / ".cache" / "huggingface" / "download" / "config.json.metadata"
    if not meta.is_file():
        return None
    return meta.read_text(encoding="utf-8").split("\n", 1)[0].strip() or None


def _structured_params():
    """Native schema decoding in the pinned serving image, without a CPU vLLM import."""
    from vllm.sampling_params import StructuredOutputsParams

    return StructuredOutputsParams


@dataclass(frozen=True)
class ArmConfiguration:
    """Validated reader settings and prompts, without loading a model."""

    gcr_config: GCRConfig | None
    gcr_prompts: GCRPrompts | None
    cor_prompts: CorPrompts | None
    tog_config: ToGConfig | None
    tog_prompts: ToGPrompts | None
    dog_config: DoGConfig | None
    dog_prompts: DoGPrompts | None
    observe: str
    observe_seed: int


@dataclass(frozen=True)
class GenerationRun:
    """Execution outputs and the context required to write their receipts."""

    requests: list[dict]
    records: list[dict]
    # Keep the model resident through receipt writing, as in the original main.
    llm: Any
    vllm: Any
    llm_args: dict
    opener: str | None
    seconds: float
    llm_startup_seconds: float
    thinking: dict | None
    observation_syntax: SpanSyntax | None
    observation_tokenizer: ServedTokenizer | None
    screen_budget: GCRScreenBudget | None
    synthesis_loaded: bool


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m xgraph.generate", description=__doc__)
    parser.add_argument("--model", required=True, help="model directory or Hugging Face id")
    parser.add_argument("--requests", type=Path, required=True, help="chat requests (JSONL)")
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--out", type=Path, required=True, help="output records (JSONL)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--think-budget",
        type=int,
        default=0,
        help="unconstrained thought-stage token cap; 0 preserves thinking-off generation",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "xhigh"),
        default="medium",
        help="native thought template's effort (with --think-budget)",
    )
    parser.add_argument(
        "--thoughts",
        type=Path,
        help="reuse thoughts by request id instead of running stage 1 (needs --think-budget)",
    )
    parser.add_argument("--kv-cache-gib", type=float, default=8.0)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--no-prefix-caching", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--memory-share",
        type=float,
        default=0.45,
        help="share of all memory vLLM requires free at startup (gpu_memory_utilization)",
    )
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--samples", type=int, default=1, help="samples per request, as id#k (xgraph.vote)"
    )
    parser.add_argument("--sample", action="store_true", help="sample even one answer; keep its id")
    parser.add_argument("--temperature", type=float, default=0.7, help="with --samples")
    parser.add_argument("--top-p", type=float, default=0.95, help="with --samples")
    parser.add_argument(
        "--question-timings",
        action="store_true",
        help="time ToG/DoG/CoR questions serially for a pilot cost projection",
    )
    parser.add_argument("--lora", help="a LoRA adapter directory (xgraph.train)")
    parser.add_argument("--lora-rank", type=int, default=16, help="the adapter's rank")
    parser.add_argument(
        "--speculative-tokens", type=int, default=0, help="MTP draft tokens per step (0: off)"
    )
    parser.add_argument(
        "--ban-tokens", nargs="+", default=[], help="tokens never to generate, e.g. <|channel>"
    )
    parser.add_argument(
        "--tools-db",
        type=Path,
        help="the tools, walk, iterative, agent, cor, gcr and resume arms' KG store (SQLite)",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        help="the tools arm's calls (default 4); the walk arm's hops (4); the iterative "
        "arm's lookups (4); the agent's tool calls (8); cor's optional total model-call cap; "
        "gcr's optional total model-call cap (includes sampled trajectories); "
        "resume's path hop cap (default 8)",
    )
    parser.add_argument(
        "--walk-stride",
        type=int,
        choices=(1, 2),
        help="matched walk: 2 enables pair actions; 1 shows identical pairs as atomic lookahead",
    )
    parser.add_argument(
        "--resume-max",
        type=int,
        help="resume's maximum continuation calls after the free plan (default 3; pages count)",
    )
    parser.add_argument(
        "--cor-depth", type=int, help="cor's positive maximum chain length (default 3)"
    )
    parser.add_argument(
        "--cor-mode",
        choices=("upstream", "reconstructed"),
        help="CoR control flow (default upstream; reconstructed is the historical adaptation)",
    )
    parser.add_argument(
        "--cor-prompts", type=Path, help="local released CoR prompt.yml (never bundled)"
    )
    parser.add_argument("--cor-temperature-exploration", type=float)
    parser.add_argument("--cor-temperature-reasoning", type=float)
    parser.add_argument(
        "--tog-prompts", type=Path, help="official ToG prompt_list.py, read as data"
    )
    parser.add_argument("--tog-depth", type=int, help="ToG depth: native 3 or transfer 4")
    parser.add_argument("--tog-batch-size", type=int, help="concurrent ToG questions (default 32)")
    parser.add_argument("--tog-variant", choices=("tog", "tog-r"), help="ToG search variant")
    parser.add_argument("--dog-prompts", type=Path, help="released DoG templates.py, read as data")
    parser.add_argument("--dog-beam", type=int, help="DoG triple beam width (default 1)")
    parser.add_argument("--dog-steps", type=int, help="DoG maximum reasoning steps (default 5)")
    parser.add_argument("--dog-batch-size", type=int, help="concurrent DoG questions (default 32)")
    parser.add_argument("--gcr-stage", choices=("paths", "synthesis"))
    parser.add_argument("--gcr-handoff", type=Path)
    parser.add_argument("--gcr-k", type=int, help="GCR's sample count or beam width (default 10)")
    parser.add_argument(
        "--gcr-decoding",
        choices=("sampling", "beam"),
        help="GCR's decoder (default sampling; beam jointly ranks paths and hypotheses)",
    )
    parser.add_argument(
        "--gcr-beam-semantics",
        choices=("vllm", "hf", "hf-group"),
        help="beam semantics: vllm (default), hf ordinary beam, or released HF 4.44.0 group beam",
    )
    parser.add_argument(
        "--gcr-beam-early-stopping",
        choices=("false", "true", "never"),
        help="HF beam stopping rule (default false)",
    )
    parser.add_argument(
        "--gcr-beam-length-penalty", type=float, help="HF beam length penalty (default 1)"
    )
    parser.add_argument(
        "--gcr-beam-return-sequences", type=int, help="HF final sequence count (default K)"
    )
    parser.add_argument("--gcr-hops", type=int, help="GCR's maximum path hops (default 2)")
    parser.add_argument(
        "--gcr-prompts",
        type=Path,
        help="upstream qa_prompt_builder.py: paper zero-shot prompts, linked single-answer only",
    )
    parser.add_argument(
        "--gcr-empty-trie",
        choices=("open", "closed"),
        help="GCR's empty trie: unconstrained generation or unanswered (default open)",
    )
    parser.add_argument(
        "--gcr-max-paths", type=int, help="GCR's per-question path cap (default 100000)"
    )
    parser.add_argument(
        "--gcr-path-overflow",
        choices=("error", "truncate"),
        help="GCR path cap: error (default) or retain a bounded prefix and record truncation",
    )
    parser.add_argument(
        "--gcr-batch-size", type=int, help="questions whose tries are built together (default 8)"
    )
    parser.add_argument(
        "--gcr-parallel-relations",
        choices=("preserve", "collapse"),
        help="keep parallel relations or emulate DiGraph collapse (default preserve)",
    )
    parser.add_argument(
        "--gcr-synthesis-model",
        help="GCR's general model directory; default reuse --model",
    )
    parser.add_argument(
        "--gcr-screen-panel-questions",
        type=int,
        help="full-panel question count for the GCR time screen",
    )
    parser.add_argument(
        "--gcr-screen-max-hours",
        type=float,
        help="abort when projected full-panel workload GPU-hours exceed this cap",
    )
    parser.add_argument(
        "--relation-labels",
        type=Path,
        help="relation labels and aliases for agent/cor or audit/menu display "
        "(relation_labels.jsonl)",
    )
    parser.add_argument(
        "--matched-demos",
        action="store_true",
        help="the agent arm shows each request's own path demonstrations as execute_path "
        "trajectories instead of its worked example (xgraph.agent.demonstrations)",
    )
    parser.add_argument(
        "--matched-paths",
        action="store_true",
        help="the agent's execute_path or walk uses matched execution, over "
        "--kg-index and --tokenizer, with --start-rank and --unquoted-starts",
    )
    parser.add_argument(
        "--pages",
        action="store_true",
        help="the agent arm's follow and execute_path take an offset, to list later entities",
    )
    parser.add_argument(
        "--agent-answer-parser",
        choices=("agent-answer-v1", "agent-answer-v2"),
        help="explicit agent final-answer parser version; unset retains legacy records",
    )
    parser.add_argument(
        "--agent-final-controller",
        choices=("agent-final-retry-v1",),
        help="retry a tool reply after the final-answer instruction once, without executing it",
    )
    parser.add_argument(
        "--agent-feedback",
        action="store_true",
        help="the agent reports failed-path frontier relations and requires another tool call; "
        "enables --matched-demos, --matched-paths and --pages (needs --kg-index and --tokenizer)",
    )
    parser.add_argument(
        "--agent-mask",
        action="store_true",
        help="mask agent path spans with execution state; enables matched demos/paths/pages "
        "and allfix engine settings, without STOP or repair",
    )
    parser.add_argument(
        "--agent-force-path",
        action="store_true",
        help="prefill Action: execute_path ⟦ in round one (also usable without the mask)",
    )
    parser.add_argument(
        "--agent-schema",
        choices=("static", "turn"),
        help="native structured-output control on JSON actions: a vocabulary enum or "
        "a per-turn enum of reached entities' relations; implies matched demos/paths/pages",
    )
    parser.add_argument(
        "--written-start",
        action="store_true",
        help="the iterative arm starts from the facts of a start the model writes first",
    )
    parser.add_argument(
        "--observe",
        choices=OBSERVATIONS,
        help="the walk arm's observation of the entities reached (xgraph.walk; default labels)",
    )
    parser.add_argument(
        "--min-hops",
        type=int,
        help="the walk refuses stopping before this many relations (default zero)",
    )
    parser.add_argument(
        "--max-options",
        type=int,
        help="the walk arm's most relations offered a round, first by label (default: all)",
    )
    parser.add_argument(
        "--coarse-types",
        action="store_true",
        help="the walk arm shows coarse classes (xgraph.walk.COARSE_CLASSES) for classes",
    )
    parser.add_argument(
        "--observe-seed",
        type=int,
        help="the walk arm's seed of count-shuffled-previews' permutations (default 0)",
    )
    parser.add_argument("--kg-index", type=Path, help="the KG index, recorded in provenance")
    parser.add_argument("--tokenizer", type=Path, help="the served tokenizer, recorded too")
    parser.add_argument(
        "--kg-versions",
        type=Path,
        help="the KG version file (XGRAPH_KG_VERSIONS), recorded too; the tools, walk, "
        "iterative, agent, cor and gcr arms read it",
    )
    parser.add_argument(
        "--no-escapes", action="store_true", help="the engine ran without escapes, recorded too"
    )
    parser.add_argument(
        "--no-repeats", action="store_true", help="the engine ran without repeats, recorded too"
    )
    parser.add_argument(
        "--release-mass", type=float, default=0.0, help="the engine's release mass, recorded too"
    )
    parser.add_argument(
        "--release-top", action="store_true", help="the engine released on disallowed top tokens"
    )
    parser.add_argument(
        "--derivation", action="store_true", help="the engine ran in derivation mode, recorded"
    )
    parser.add_argument(
        "--escape-bias", type=float, default=0.0, help="the engine's escape bias, recorded"
    )
    parser.add_argument("--paths", action="store_true", help="the engine ran in path mode")
    parser.add_argument(
        "--audit", action="store_true", help="one blind path then one terminal audit (XGRAPH_AUDIT)"
    )
    parser.add_argument("--audit-entities", type=int, help="endpoint display cap (default 20)")
    parser.add_argument(
        "--observe-terminal",
        type=int,
        nargs="?",
        const=8,
        help="display up to W deliverable endpoints after path closure (default W=8; 0 off)",
    )
    parser.add_argument("--menu", type=int, help="per-step relation menu size, 1..16 (XGRAPH_MENU)")
    parser.add_argument(
        "--path-demos",
        type=int,
        choices=(4, 5),
        help="matched path demonstration set (XGRAPH_PATH_DEMOS; unset keeps legacy prompts)",
    )
    parser.add_argument(
        "--observe-width",
        type=int,
        nargs="?",
        const=8,
        help="show every entity in frontiers up to W (default 8 when enabled)",
    )
    parser.add_argument(
        "--observe-defer",
        type=int,
        nargs="?",
        const=2,
        help="allow up to N required selections to DEFER per path (default 2)",
    )
    parser.add_argument(
        "--menu-close-next", action="store_true", help="decide CLOSE/NEXT before post-start menus"
    )
    parser.add_argument(
        "--audit-relations", type=int, help="each relation menu cap (default 8; 0 hides menus)"
    )
    parser.add_argument(
        "--observe-paths",
        type=int,
        help="force up to K frontier labels after each path segment (XGRAPH_OBSERVE; 0 off)",
    )
    parser.add_argument(
        "--observe-select",
        action="store_true",
        help="allow entity selection after observations (XGRAPH_OBSERVE_SELECT)",
    )
    parser.add_argument(
        "--observe-answer",
        action="store_true",
        help="force the bound answer and EOS after a path (XGRAPH_OBSERVE_ANSWER)",
    )
    parser.add_argument(
        "--observe-required",
        action="store_true",
        help="require selection of multi-entity frontiers (XGRAPH_OBSERVE_REQUIRED)",
    )
    parser.add_argument(
        "--observe-no-loop",
        action="store_true",
        help="forbid repeated frontier/relation pairs (XGRAPH_OBSERVE_NO_LOOP)",
    )
    parser.add_argument(
        "--no-bound-escapes", action="store_true", help="bound subjects could not escape"
    )
    parser.add_argument(
        "--committed-paths",
        action="store_true",
        help="paths resolved to what their best start entity reaches",
    )
    parser.add_argument(
        "--start-rank",
        choices=("degree", "sitelinks"),
        default="degree",
        help="how the engine ranked a label's entities (XGRAPH_START_RANK), recorded",
    )
    parser.add_argument(
        "--unquoted-starts",
        action="store_true",
        help="unbound subjects could not start with a quotation mark (XGRAPH_UNQUOTED_STARTS)",
    )
    parser.add_argument(
        "--expressions",
        action="store_true",
        help="spans could be date comparisons (XGRAPH_EXPRESSIONS), recorded",
    )
    parser.add_argument(
        "--stop", choices=("residual-goal",), help="the engine's STOP rule (XGRAPH_STOP), recorded"
    )
    parser.add_argument(
        "--end-after-answer",
        action="store_true",
        help="end path-mode generation after its answer span (XGRAPH_END_AFTER_ANSWER), recorded",
    )
    parser.add_argument(
        "--relations",
        choices=("frontier", "vocabulary", "successor"),
        default="frontier",
        help="relation support in path mode (XGRAPH_RELATIONS), recorded",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="start every answer inside a span: the span opener as a prefilled assistant turn",
    )
    parser.add_argument(
        "--provenance-only",
        action="store_true",
        help="write <out>.provenance.json on the host, without generating",
    )
    parser.add_argument("--terminal-order", choices=("identity", "reversed", "random"))
    parser.add_argument("--terminal-seed", type=int)
    return parser


def _prepare_gcr(parser, args):
    gcr_prompts = None
    from xgraph.gcr import GCRConfig, GCRPrompts

    if args.tools_db is None:
        parser.error("the gcr arm needs --tools-db")
    if args.speculative_tokens:
        parser.error("GCR does not support --speculative-tokens")
    if args.gcr_decoding == "beam" and args.ban_tokens:
        parser.error("GCR beam scoring uses raw logprobs and does not support --ban-tokens")
    if args.max_tokens <= 0 or args.max_model_len <= args.max_tokens:
        parser.error("gcr needs 0 < --max-tokens < --max-model-len")
    try:
        if args.gcr_prompts is not None:
            gcr_prompts = GCRPrompts.load(args.gcr_prompts)
        gcr_config = GCRConfig(
            **{
                key: getattr(args, f"gcr_{key}")
                for key in (
                    "k",
                    "hops",
                    "max_paths",
                    "batch_size",
                    "parallel_relations",
                    "path_overflow",
                    "decoding",
                    "beam_semantics",
                    "beam_length_penalty",
                    "beam_return_sequences",
                )
                if getattr(args, f"gcr_{key}") is not None
            },
            empty=args.gcr_empty_trie or "open",
            max_calls=args.max_calls,
            max_tokens=args.max_tokens,
            max_prompt_tokens=args.max_model_len - args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            repetition_penalty=args.repetition_penalty,
            beam_early_stopping={"true": True, "false": False, "never": "never"}.get(
                args.gcr_beam_early_stopping, False
            ),
        )
    except ValueError as error:
        parser.error(str(error))
    if (args.gcr_screen_panel_questions is None) != (args.gcr_screen_max_hours is None):
        parser.error(
            "GCR screening needs both --gcr-screen-panel-questions and --gcr-screen-max-hours"
        )
    if args.gcr_screen_max_hours is not None and (
        args.gcr_screen_panel_questions <= 0
        or not math.isfinite(args.gcr_screen_max_hours)
        or args.gcr_screen_max_hours <= 0
        or gcr_config.decoding != "beam"
        or gcr_config.batch_size != 1
        or args.gcr_synthesis_model not in (None, args.model)
    ):
        parser.error(
            "GCR screening needs positive panel size/hours, joint beam, "
            "batch size one and the same model"
        )
    return gcr_config, gcr_prompts


def _prepare_dog(parser, args):
    from xgraph.dog import DoGConfig, DoGPrompts

    if args.dog_batch_size is not None and args.dog_batch_size < 1:
        parser.error("--dog-batch-size must be positive")
    if args.dog_prompts is None:
        parser.error("DoG needs --dog-prompts and pre-retrieved dog_graph requests")
    if args.speculative_tokens or args.ban_tokens or args.repetition_penalty != 1.0:
        parser.error(
            "DoG uses raw logits; speculative tokens, bans and repetition penalties are unsupported"
        )
    if args.max_model_len <= max(700, args.max_tokens):
        parser.error("DoG needs room for its prompts and triple/completion limits")
    try:
        dog_prompts = DoGPrompts.load(args.dog_prompts)
        dog_config = DoGConfig(
            beam_width=1 if args.dog_beam is None else args.dog_beam,
            max_steps=5 if args.dog_steps is None else args.dog_steps,
            max_tokens=args.max_tokens,
            max_prompt_tokens=args.max_model_len - max(700, args.max_tokens),
            max_calls=args.max_calls,
            seed=args.seed,
        )
    except ValueError as error:
        parser.error(str(error))
    return dog_config, dog_prompts


def _prepare_tog(parser, args):
    from xgraph.tog import ToGConfig, ToGPrompts

    if args.tog_batch_size is not None and args.tog_batch_size < 1:
        parser.error("--tog-batch-size must be positive")
    if args.tools_db is None or args.tog_prompts is None:
        parser.error("ToG needs --tools-db and --tog-prompts")
    if args.speculative_tokens or args.max_model_len <= args.max_tokens:
        parser.error("ToG needs room for prompts and does not support speculative tokens")
    try:
        tog_prompts = ToGPrompts.load(args.tog_prompts)
        tog_config = ToGConfig(
            variant=args.tog_variant or "tog",
            depth=3 if args.tog_depth is None else args.tog_depth,
            max_tokens=args.max_tokens,
            max_prompt_tokens=args.max_model_len - args.max_tokens,
            max_calls=args.max_calls,
            seed=args.seed,
        )
    except ValueError as error:
        parser.error(str(error))
    return tog_config, tog_prompts


def _prepare_cor(parser, args):
    cor_prompts = None
    from xgraph.cor import (
        EXPLORATION_TEMPERATURE,
        MAX_CALLS,
        MAX_DEPTH,
        REASONING_TEMPERATURE,
        CorPrompts,
    )

    args.cor_mode = args.cor_mode or "upstream"
    args.cor_depth = args.cor_depth if args.cor_depth is not None else MAX_DEPTH
    for name, default in (
        ("cor_temperature_exploration", EXPLORATION_TEMPERATURE),
        ("cor_temperature_reasoning", REASONING_TEMPERATURE),
    ):
        value = getattr(args, name)
        if value is not None and args.cor_mode != "upstream":
            parser.error(f"--{name.replace('_', '-')} needs upstream CoR")
        value = default if value is None else value
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
        setattr(args, name, value)
    if args.cor_depth <= 0:
        parser.error("cor's --cor-depth must be positive")
    if args.tools_db is None:
        parser.error("the cor arm needs --tools-db")
    if args.cor_mode == "upstream":
        if args.cor_prompts is None:
            parser.error("upstream CoR needs --cor-prompts <local prompt.yml>")
        try:
            cor_prompts = CorPrompts.load(args.cor_prompts)
        except ValueError as error:
            parser.error(str(error))
    elif args.cor_prompts is not None:
        parser.error("--cor-prompts needs --cor-mode upstream")
    elif args.max_calls is None:
        args.max_calls = MAX_CALLS
    if args.max_calls is not None and args.max_calls <= 0:
        parser.error("cor's --max-calls must be positive")
    if args.max_tokens <= 0 or args.max_model_len <= args.max_tokens:
        parser.error("cor needs 0 < --max-tokens < --max-model-len")
    return cor_prompts


def _parse_arguments(parser, argv) -> tuple[argparse.Namespace, ArmConfiguration]:
    args = parser.parse_args(argv)
    args.terminal_order = args.terminal_order or os.environ.get("XGRAPH_TERMINAL_ORDER", "identity")
    if args.terminal_seed is None:
        try:
            args.terminal_seed = int(os.environ.get("XGRAPH_TERMINAL_SEED") or "17")
        except ValueError:
            parser.error("XGRAPH_TERMINAL_SEED must be an integer")
    if args.question_timings and args.arm not in ("tog", "dog", "dog-gold", "cor"):
        parser.error("--question-timings needs the tog, dog, dog-gold or cor arm")
    from xgraph.observations import check_observe, check_observed_options

    args.observe_select |= os.environ.get("XGRAPH_OBSERVE_SELECT") == "1"
    args.observe_answer |= os.environ.get("XGRAPH_OBSERVE_ANSWER") == "1"
    args.observe_required |= os.environ.get("XGRAPH_OBSERVE_REQUIRED") == "1"
    args.observe_no_loop |= os.environ.get("XGRAPH_OBSERVE_NO_LOOP") == "1"
    args.observe_select |= args.observe_required
    args.menu_close_next |= os.environ.get("XGRAPH_MENU_CLOSE_NEXT") == "1"
    for option, variable in (
        ("observe_width", "XGRAPH_OBSERVE_WIDTH"),
        ("observe_defer", "XGRAPH_OBSERVE_DEFER"),
        ("observe_terminal", "XGRAPH_OBSERVE_TERMINAL"),
    ):
        if getattr(args, option) is None:
            try:
                setattr(args, option, int(os.environ.get(variable) or "0"))
            except ValueError:
                parser.error(f"{variable} must be an integer")
    args.observe_answer |= args.menu_close_next or bool(args.observe_defer)
    args.audit |= os.environ.get("XGRAPH_AUDIT") == "1"
    if args.path_demos is None and os.environ.get("XGRAPH_PATH_DEMOS"):
        try:
            args.path_demos = int(os.environ["XGRAPH_PATH_DEMOS"])
        except ValueError:
            parser.error("XGRAPH_PATH_DEMOS must be 4 or 5")
    if args.path_demos is not None:
        if args.path_demos not in (4, 5):
            parser.error("XGRAPH_PATH_DEMOS must be 4 or 5")
        if (
            args.arm != "constrained"
            or not (args.paths or os.environ.get("XGRAPH_PATHS") == "1")
            or args.audit
        ):
            parser.error("--path-demos needs constrained path mode with audit off")
    if args.menu is None:
        try:
            args.menu = int(os.environ.get("XGRAPH_MENU") or "0")
        except ValueError:
            parser.error("XGRAPH_MENU must be an integer")
    from xgraph.menus import check_menu

    try:
        check_menu(
            args.menu,
            paths=args.paths or os.environ.get("XGRAPH_PATHS") == "1",
            relations=os.environ.get("XGRAPH_RELATIONS", args.relations),
            expressions=args.expressions or os.environ.get("XGRAPH_EXPRESSIONS") == "1",
            audit=args.audit,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.menu and args.arm != "constrained":
        parser.error("--menu needs the constrained arm")
    for option, variable, default in (
        ("audit_entities", "XGRAPH_AUDIT_ENTITIES", 20),
        ("audit_relations", "XGRAPH_AUDIT_RELATIONS", 8),
    ):
        if getattr(args, option) is None:
            try:
                setattr(args, option, int(os.environ.get(variable) or default))
            except ValueError:
                parser.error(f"{variable} must be an integer")
    args.end_after_answer |= args.observe_answer or args.audit or bool(args.observe_terminal)

    if args.observe_paths is None:
        try:
            args.observe_paths = int(os.environ.get("XGRAPH_OBSERVE") or "0")
        except ValueError:
            parser.error("XGRAPH_OBSERVE must be an integer")
    try:
        check_observe(
            args.observe_paths,
            paths=args.paths or os.environ.get("XGRAPH_PATHS") == "1",
            relations=os.environ.get("XGRAPH_RELATIONS", args.relations),
            expressions=args.expressions or os.environ.get("XGRAPH_EXPRESSIONS") == "1",
        )
        check_observed_options(
            args.observe_paths,
            args.observe_select,
            args.observe_answer,
            args.observe_required,
            args.observe_no_loop,
            width=args.observe_width,
            defer=args.observe_defer,
            close_next=args.menu_close_next,
            menu=args.menu,
        )
        from xgraph.audit import check_audit, check_terminal, check_terminal_order

        check_terminal_order(args.terminal_order, args.terminal_seed, args.observe_terminal)

        check_terminal(
            args.observe_terminal,
            paths=args.paths or os.environ.get("XGRAPH_PATHS") == "1",
            relations=os.environ.get("XGRAPH_RELATIONS", args.relations),
            expressions=args.expressions or os.environ.get("XGRAPH_EXPRESSIONS") == "1",
            incompatible=(
                args.audit,
                args.menu,
                args.observe_paths,
                args.observe_select,
                args.observe_answer,
                args.observe_required,
                args.observe_no_loop,
            ),
        )

        check_audit(
            args.audit,
            args.audit_entities,
            args.audit_relations,
            paths=args.paths or os.environ.get("XGRAPH_PATHS") == "1",
            relations=os.environ.get("XGRAPH_RELATIONS", args.relations),
            expressions=args.expressions or os.environ.get("XGRAPH_EXPRESSIONS") == "1",
            observed=(
                args.observe_paths,
                args.observe_select,
                args.observe_answer,
                args.observe_required,
                args.observe_no_loop,
            ),
        )
    except ValueError as error:
        parser.error(str(error))
    if args.observe_paths and args.arm != "constrained":
        parser.error("--observe-paths needs the constrained arm")
    if args.audit and args.arm != "constrained":
        parser.error("--audit needs the constrained arm")
    if args.observe_terminal and args.arm != "constrained":
        parser.error("--observe-terminal needs the constrained arm")
    if args.agent_mask:
        args.paths = args.committed_paths = args.no_escapes = args.unquoted_starts = True
        args.start_rank = "sitelinks"
        args.stop = None
        args.expressions = args.end_after_answer = False
    if args.end_after_answer and not (args.paths or os.environ.get("XGRAPH_PATHS") == "1"):
        parser.error("--end-after-answer needs path mode (--paths)")
    if args.think_budget < 0:
        parser.error("--think-budget must be non-negative")
    if args.thoughts and not args.think_budget:
        parser.error("--thoughts needs a positive --think-budget")
    if args.think_budget and args.arm not in ("baseline", "constrained"):
        parser.error(
            "--think-budget supports only baseline and constrained arms; "
            f"the {args.arm} arm is unsupported"
        )
    from xgraph.relations import check_relations

    try:
        check_relations(
            args.relations, paths=args.paths, expressions=args.expressions, stop=args.stop
        )
    except ValueError as error:
        parser.error(str(error))
    if (args.samples > 1 or args.sample) and args.arm in (
        "tools",
        "walk",
        "resume",
        "agent",
        "cor",
        "tog",
        "dog",
        "dog-gold",
        "gcr",
        "fallback",
        "iterative",
    ):
        parser.error(f"--samples does not apply to the {args.arm} arm")
    if (
        args.open
        and args.arm not in ("constrained", "fallback")
        and not (args.arm == "baseline" and args.think_budget)
    ):
        parser.error(f"--open needs the engine, which the {args.arm} arm does not load")
    walk_flags = (
        (args.min_hops, "--min-hops"),
        (args.observe, "--observe"),
        (args.max_options, "--max-options"),
        (args.coarse_types or None, "--coarse-types"),
        (args.observe_seed, "--observe-seed"),
    )
    for given, flag in walk_flags:
        if given is not None and args.arm != "walk":
            parser.error(f"{flag} is the walk arm's, not the {args.arm} arm's")
    observe = args.observe or "labels"
    observe_seed = args.observe_seed or 0
    if args.coarse_types and observe in CLASSLESS:
        parser.error(f"--coarse-types needs an observation with a class, not {observe}")
    if (
        args.relation_labels
        and args.arm not in ("agent", "cor")
        and not (args.arm == "constrained" and (args.audit or args.menu))
    ):
        parser.error("--relation-labels needs agent, cor, or constrained audit/menu")
    for flag in (
        "cor_depth",
        "cor_mode",
        "cor_prompts",
        "cor_temperature_exploration",
        "cor_temperature_reasoning",
    ):
        if getattr(args, flag) is not None and args.arm != "cor":
            parser.error(f"--{flag.replace('_', '-')} needs the cor arm")
    for flag in (
        "gcr_k",
        "gcr_decoding",
        "gcr_beam_semantics",
        "gcr_beam_early_stopping",
        "gcr_beam_length_penalty",
        "gcr_beam_return_sequences",
        "gcr_hops",
        "gcr_empty_trie",
        "gcr_max_paths",
        "gcr_path_overflow",
        "gcr_batch_size",
        "gcr_parallel_relations",
        "gcr_synthesis_model",
        "gcr_prompts",
        "gcr_screen_panel_questions",
        "gcr_screen_max_hours",
    ):
        if getattr(args, flag) is not None and args.arm != "gcr":
            parser.error(f"--{flag.replace('_', '-')} needs the gcr arm")
    gcr_config = None
    gcr_prompts = None
    if args.arm == "gcr":
        gcr_config, gcr_prompts = _prepare_gcr(parser, args)
    agent_flags = (
        (args.agent_answer_parser, "--agent-answer-parser"),
        (args.agent_final_controller, "--agent-final-controller"),
        (args.matched_demos, "--matched-demos"),
        (args.pages, "--pages"),
        (args.agent_feedback, "--agent-feedback"),
        (args.agent_mask, "--agent-mask"),
        (args.agent_force_path, "--agent-force-path"),
        (args.agent_schema, "--agent-schema"),
    )
    for given, flag in agent_flags:
        if given and args.arm != "agent":
            parser.error(f"{flag} is the agent arm's, not the {args.arm} arm's")
    if args.matched_paths and args.arm not in ("agent", "walk", "resume"):
        parser.error("--matched-paths needs agent, walk or resume")
    if args.resume_max is not None and args.arm != "resume":
        parser.error("--resume-max needs the resume arm")
    if args.arm == "resume":
        args.matched_paths = True
        if args.resume_max is None:
            args.resume_max = 3
        if args.resume_max < 0:
            parser.error("--resume-max must be non-negative")
    if args.agent_schema and (args.agent_mask or args.agent_force_path):
        parser.error("--agent-schema excludes --agent-mask and --agent-force-path")
    if args.agent_feedback or args.agent_mask or args.agent_schema:
        args.matched_demos = args.matched_paths = args.pages = True
    if args.matched_paths and not (args.kg_index and args.tokenizer):
        parser.error("--matched-paths needs --kg-index and --tokenizer")
    if args.written_start and args.arm != "iterative":
        parser.error(f"--written-start is the iterative arm's, not the {args.arm} arm's")
    # Preserve provenance-only legacy walk commands, which may record no store.
    if (
        args.arm in ("walk", "resume")
        and args.tools_db is None
        and (args.arm == "resume" or not args.provenance_only)
    ):
        parser.error(f"the {args.arm} arm needs --tools-db")
    if args.arm == "walk" and args.matched_paths and observe != "labels":
        parser.error("matched walk needs --observe labels")
    if args.walk_stride is not None and (
        args.arm != "walk"
        or not args.matched_paths
        or args.max_options is not None
        or args.coarse_types
        or observe != "labels"
    ):
        parser.error("--walk-stride needs matched walk, labels and complete atomic menus")
    cor_prompts = None
    tog_prompts = tog_config = None
    dog_prompts = dog_config = None
    for flag in ("dog_prompts", "dog_beam", "dog_steps", "dog_batch_size"):
        if getattr(args, flag) is not None and args.arm not in ("dog", "dog-gold"):
            parser.error(f"--{flag.replace('_', '-')} needs the dog or dog-gold arm")
    if args.arm in ("dog", "dog-gold"):
        dog_config, dog_prompts = _prepare_dog(parser, args)
    for flag in ("tog_prompts", "tog_depth", "tog_batch_size", "tog_variant"):
        if getattr(args, flag) is not None and args.arm != "tog":
            parser.error(f"--{flag.replace('_', '-')} needs the tog arm")
    if args.arm == "tog":
        tog_config, tog_prompts = _prepare_tog(parser, args)
    if args.max_calls is None and args.arm not in ("cor", "gcr", "tog", "dog", "dog-gold"):
        from xgraph.agent import MAX_CALLS as AGENT_CALLS

        args.max_calls = AGENT_CALLS if args.arm in ("agent", "resume") else 4
    if args.arm == "walk" and not 0 <= (args.min_hops or 0) <= args.max_calls:
        parser.error("--min-hops must be between zero and --max-calls")
    if args.arm == "resume" or (args.arm == "walk" and args.matched_paths):
        from xgraph.oracle import MAX_PATH_RELATIONS

        if not 1 <= args.max_calls <= MAX_PATH_RELATIONS:
            parser.error(f"matched walk/resume need 1 <= --max-calls <= {MAX_PATH_RELATIONS}")
    if args.arm == "cor":
        cor_prompts = _prepare_cor(parser, args)
    if args.gcr_stage and (args.arm != "gcr" or not args.gcr_handoff):
        parser.error("--gcr-stage needs the gcr arm and --gcr-handoff")
    if args.gcr_handoff and args.arm != "gcr":
        parser.error("--gcr-handoff needs the gcr arm")
    return args, ArmConfiguration(
        gcr_config=gcr_config,
        gcr_prompts=gcr_prompts,
        cor_prompts=cor_prompts,
        tog_config=tog_config,
        tog_prompts=tog_prompts,
        dog_config=dog_config,
        dog_prompts=dog_prompts,
        observe=observe,
        observe_seed=observe_seed,
    )


def _write_provenance(args, config: ArmConfiguration) -> None:
    if args.arm == "cor":
        from xgraph.cor import RELATION_THRESHOLD, WIDTH, temperature_profile
    from xgraph.provenance import provenance

    inputs = [args.requests, args.tokenizer, args.kg_index and args.kg_index / "kg-index.json"]
    inputs.append(args.relation_labels)
    if args.agent_answer_parser or args.agent_final_controller:
        inputs.extend(
            Path(__file__).with_name(name)
            for name in (
                "agent_answers.py",
                "agent_protocol.py",
                "agent_final.py",
                "agent.py",
                "prompts.py",
            )
        )
    if args.think_budget:
        inputs.append(args.thoughts)
        inputs += [
            p
            for p in (
                Path(args.model) / "tokenizer_config.json",
                Path(args.model) / "chat_template.jinja",
            )
            if p.is_file()
        ]
    if config.cor_prompts is not None:
        inputs.append(config.cor_prompts.path)
    if config.tog_prompts is not None:
        inputs.append(config.tog_prompts.source.path)
    if config.dog_prompts is not None:
        inputs.append(config.dog_prompts.source.path)
    if config.gcr_prompts is not None:
        inputs.append(config.gcr_prompts.source.path)
    if args.arm in ("cor", "gcr", "tog", "resume") or (args.arm == "walk" and args.matched_paths):
        inputs.append(args.tools_db)
    if args.arm == "gcr":
        inputs += [Path(args.model) / "config.json", Path(args.model) / "tokenizer.json"]
        inputs += [
            p
            for p in (
                Path(args.model) / "generation_config.json",
                Path(args.model) / "tokenizer_config.json",
            )
            if p.is_file()
        ]
        if args.gcr_synthesis_model:
            inputs += [
                Path(args.gcr_synthesis_model) / "config.json",
                Path(args.gcr_synthesis_model) / "tokenizer.json",
            ]
    if args.kg_versions:
        from xgraph.kgversions import version_files

        inputs += version_files(args.kg_versions)
    if args.lora:
        inputs.append(Path(args.lora) / "adapter_model.safetensors")
    record = {
        "artifact_kind": "xgraph.generation_provenance",
        "version": 1,
        "arm": args.arm,
        "escapes": not args.no_escapes,
        "repeats": not args.no_repeats,
        "release_mass": args.release_mass,
        "release_top": args.release_top,
        "derivation": args.derivation,
        "escape_bias": args.escape_bias,
        "paths": args.paths,
        **(
            {"lexical_schema": os.environ["XGRAPH_LEXICAL_SCHEMA"]}
            if os.environ.get("XGRAPH_LEXICAL_SCHEMA")
            else {}
        ),
        **({"observe_paths": args.observe_paths} if args.observe_paths else {}),
        **({"observe_terminal": args.observe_terminal} if args.observe_terminal else {}),
        **(
            {"terminal_order": args.terminal_order, "terminal_seed": args.terminal_seed}
            if args.terminal_order != "identity" or args.terminal_seed != 17
            else {}
        ),
        **({"menu": args.menu} if args.menu else {}),
        **({"observe_width": args.observe_width} if args.observe_width else {}),
        **({"observe_defer": args.observe_defer} if args.observe_defer else {}),
        **({"menu_close_next": True} if args.menu_close_next else {}),
        **({"path_demos": args.path_demos} if args.path_demos is not None else {}),
        **({"observe_select": True} if args.observe_select else {}),
        **({"observe_answer": True} if args.observe_answer else {}),
        **({"observe_required": True} if args.observe_required else {}),
        **({"observe_no_loop": True} if args.observe_no_loop else {}),
        **(
            {
                "audit": True,
                "audit_entities": args.audit_entities,
                "audit_relations": args.audit_relations,
            }
            if args.audit
            else {}
        ),
        "bound_escapes": not args.no_bound_escapes,
        "committed_paths": args.committed_paths,
        "start_rank": args.start_rank,
        "unquoted_starts": args.unquoted_starts,
        "expressions": args.expressions,
        "stop": args.stop,
        "end_after_answer": args.end_after_answer,
        "relations": args.relations,
        "open": args.open,
        **(
            {
                "thinking": {
                    "think_budget": args.think_budget,
                    "reasoning_effort": args.reasoning_effort,
                    "thoughts": str(args.thoughts) if args.thoughts else None,
                    "seed": args.seed,
                    "temperature": 0.0,
                }
            }
            if args.think_budget
            else {}
        ),
        **reader_scheduling(args),
        "kg_versions": str(args.kg_versions) if args.kg_versions else None,
        "tools_db": str(args.tools_db) if args.tools_db else None,
        **({"question_timings": True} if args.question_timings else {}),
        "walk": (
            {
                "observe": config.observe,
                "hops": args.max_calls,
                "seed": args.seed,
                "max_options": args.max_options,
                "coarse_types": args.coarse_types or config.observe in SLOTTED,
                "observe_seed": config.observe_seed,
                **({"min_hops": args.min_hops} if args.min_hops else {}),
                **({"matched_paths": True} if args.matched_paths else {}),
                **({"stride": args.walk_stride} if args.walk_stride is not None else {}),
            }
            if args.arm == "walk"
            else None
        ),
        **(
            {
                "resume": {
                    "max": args.resume_max,
                    "hops": args.max_calls,
                    "delivery": "executor",
                    "repair": False,
                }
            }
            if args.arm == "resume"
            else {}
        ),
        "iterative": (
            {
                "lookups": args.max_calls,
                "cap": RETRIEVAL_CAP,
                "written_start": args.written_start,
            }
            if args.arm == "iterative"
            else None
        ),
        "agent": (
            {
                "calls": args.max_calls,
                "relation_labels": str(args.relation_labels) if args.relation_labels else None,
                "matched_demos": args.matched_demos,
                "matched_paths": args.matched_paths,
                "pages": args.pages,
                **(
                    {"answer_parser_version": args.agent_answer_parser}
                    if args.agent_answer_parser
                    else {}
                ),
                **(
                    {"final_answer_controller_version": args.agent_final_controller}
                    if args.agent_final_controller
                    else {}
                ),
                **({"feedback": True} if args.agent_feedback else {}),
                **({"mask": True} if args.agent_mask else {}),
                **({"force_path": True} if args.agent_force_path else {}),
                **(
                    {"schema": args.agent_schema, "schema_backend": "vllm-structural-tag"}
                    if args.agent_schema
                    else {}
                ),
            }
            if args.arm == "agent"
            else None
        ),
        "cor": (
            {
                "calls": args.max_calls,
                "depth": args.cor_depth,
                "width": WIDTH,
                "relation_labels": str(args.relation_labels) if args.relation_labels else None,
                "mode": args.cor_mode,
                "seed": args.seed,
                "prompts": {
                    "path": str(config.cor_prompts.path),
                    "sha256": config.cor_prompts.sha256,
                }
                if config.cor_prompts
                else None,
                "temperature_exploration": args.cor_temperature_exploration
                if config.cor_prompts
                else 0.0,
                "temperature_reasoning": args.cor_temperature_reasoning
                if config.cor_prompts
                else 0.0,
                "temperature_profile": temperature_profile(
                    args.cor_temperature_exploration, args.cor_temperature_reasoning
                )
                if config.cor_prompts
                else "reconstructed",
                "relation_threshold": RELATION_THRESHOLD if config.cor_prompts else None,
            }
            if args.arm == "cor"
            else None
        ),
        "gcr": None,
        "sampling": (
            {"samples": args.samples, "temperature": args.temperature, "top_p": args.top_p}
            if args.samples > 1 or args.sample
            else None
        ),
        "model": {"path": str(args.model), "revision": model_revision(args.model)},
        "provenance": provenance([Path(i) for i in inputs if i]),
    }
    if config.tog_config is not None:
        from dataclasses import asdict

        from xgraph.tog import SOURCE_COMMIT, method_deviations

        record["tog"] = {
            **asdict(config.tog_config),
            "batch_size": 1 if args.question_timings else (args.tog_batch_size or 32),
            "prompts": config.tog_prompts.source.metadata(),
            "source_commit": SOURCE_COMMIT,
            "deviations": method_deviations(config.tog_config),
        }
    if config.dog_config is not None:
        from dataclasses import asdict

        from xgraph.dog import ORACLE_INTERPRETATION, method_deviations

        record["dog"] = {
            **asdict(config.dog_config),
            "batch_size": 1 if args.question_timings else (args.dog_batch_size or 32),
            "prompts": config.dog_prompts.source.metadata(),
            "deviations": method_deviations(args.arm),
            **({"interpretation": ORACLE_INTERPRETATION} if args.arm == "dog-gold" else {}),
        }
    if config.gcr_config is not None:
        synthesis_path = args.gcr_synthesis_model or args.model
        record["gcr"] = {
            **config.gcr_config.metadata(),
            **(
                {"prompts": config.gcr_prompts.metadata()} if config.gcr_prompts is not None else {}
            ),
            **(
                {
                    "screening": {
                        "panel_questions": args.gcr_screen_panel_questions,
                        "max_full_gpu_hours": args.gcr_screen_max_hours,
                    }
                }
                if args.gcr_screen_max_hours is not None
                else {}
            ),
            "serialization": "<PATH> entity -> relation -> entity </PATH>",
            "synthesis_model": {
                "path": str(synthesis_path),
                "revision": model_revision(synthesis_path),
            },
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    destination = Path(str(args.out) + ".paths.jsonl") if args.gcr_stage == "paths" else args.out
    Path(f"{destination}.provenance.json").write_text(json.dumps(record, indent=1) + "\n")
    return


def _execute(args, config: ArmConfiguration, parser) -> GenerationRun:
    if args.observe_terminal:
        os.environ["XGRAPH_OBSERVE_TERMINAL"] = str(args.observe_terminal)
        os.environ["XGRAPH_TERMINAL_ORDER"] = args.terminal_order
        os.environ["XGRAPH_TERMINAL_SEED"] = str(args.terminal_seed)
    elif "XGRAPH_OBSERVE_TERMINAL" in os.environ:
        os.environ["XGRAPH_OBSERVE_TERMINAL"] = "0"
    if args.audit:
        os.environ["XGRAPH_AUDIT"] = "1"
        os.environ["XGRAPH_AUDIT_ENTITIES"] = str(args.audit_entities)
        os.environ["XGRAPH_AUDIT_RELATIONS"] = str(args.audit_relations)
    if args.menu:
        os.environ["XGRAPH_MENU"] = str(args.menu)
    if (args.audit or args.menu) and args.relation_labels:
        os.environ["XGRAPH_RELATION_LABELS"] = str(args.relation_labels)
    if args.audit or args.menu or args.observe_terminal:
        os.environ["XGRAPH_PATHS"] = "1"
        os.environ.setdefault("XGRAPH_START_RANK", args.start_rank)
        for enabled, name, value in (
            (args.committed_paths, "XGRAPH_COMMITTED_PATHS", "1"),
            (args.no_escapes, "XGRAPH_ESCAPES", "0"),
            (args.unquoted_starts, "XGRAPH_UNQUOTED_STARTS", "1"),
            (args.stop is not None, "XGRAPH_STOP", args.stop),
            (args.kg_versions is not None, "XGRAPH_KG_VERSIONS", str(args.kg_versions)),
        ):
            if enabled:
                os.environ[name] = value
        for name, path in (
            ("XGRAPH_KG_INDEX", args.kg_index),
            ("XGRAPH_TOKENIZER", args.tokenizer),
        ):
            if path is not None:
                os.environ[name] = str(path)
    if args.observe_paths:
        os.environ["XGRAPH_OBSERVE"] = str(args.observe_paths)
        if args.observe_width:
            os.environ["XGRAPH_OBSERVE_WIDTH"] = str(args.observe_width)
        if args.observe_defer:
            os.environ["XGRAPH_OBSERVE_DEFER"] = str(args.observe_defer)
        if args.menu_close_next:
            os.environ["XGRAPH_MENU_CLOSE_NEXT"] = "1"
        if args.observe_select:
            os.environ["XGRAPH_OBSERVE_SELECT"] = "1"
        if args.observe_answer:
            os.environ["XGRAPH_OBSERVE_ANSWER"] = "1"
        if args.observe_required:
            os.environ["XGRAPH_OBSERVE_REQUIRED"] = "1"
        if args.observe_no_loop:
            os.environ["XGRAPH_OBSERVE_NO_LOOP"] = "1"
        os.environ["XGRAPH_PATHS"] = "1"
        for name, path in (
            ("XGRAPH_KG_INDEX", args.kg_index),
            ("XGRAPH_TOKENIZER", args.tokenizer),
        ):
            if path is not None:
                os.environ[name] = str(path)
    elif os.environ.get("XGRAPH_OBSERVE"):
        os.environ["XGRAPH_OBSERVE"] = "0"

    requests = load_requests(args.requests)
    if (args.arm in ("constrained", "fallback") or args.agent_mask) and any(
        row.get("anchor") is not None for row in requests
    ):
        from xgraph.anchors import validate_anchors
        from xgraph.index import load_kg_index

        anchor_index = args.kg_index or os.environ.get("XGRAPH_KG_INDEX")
        if anchor_index is None:
            parser.error(
                "anchored generation needs --kg-index or XGRAPH_KG_INDEX for CPU pre-validation"
            )
        validation = validate_anchors(load_kg_index(anchor_index), requests)
        if not validation["valid"]:
            parser.error(
                "anchor pre-validation failed: " + json.dumps(validation, ensure_ascii=False)
            )

    import vllm

    if args.agent_mask:
        os.environ.update(
            {
                "XGRAPH_KG_INDEX": str(args.kg_index),
                "XGRAPH_TOKENIZER": str(args.tokenizer),
                "XGRAPH_PATHS": "1",
                "XGRAPH_COMMITTED_PATHS": "1",
                "XGRAPH_ESCAPES": "0",
                "XGRAPH_UNQUOTED_STARTS": "1",
                "XGRAPH_START_RANK": "sitelinks",
                "XGRAPH_STOP": "",
                "XGRAPH_EXPRESSIONS": "0",
                "XGRAPH_END_AFTER_ANSWER": "0",
                "XGRAPH_RELATIONS": args.relations,
            }
        )
        if args.kg_versions:
            os.environ["XGRAPH_KG_VERSIONS"] = str(args.kg_versions)

    if args.arm in ("dog", "dog-gold"):
        from xgraph.dog import DoGGraph, reader_metadata

        try:
            reader_metadata(requests, args.arm)
        except ValueError as error:
            parser.error(str(error))
        for request in requests:
            if not isinstance(request.get("dog_graph"), list):
                parser.error(f"DoG request {request['id']!r} needs pre-retrieved dog_graph")
            DoGGraph(request["dog_graph"])
    observation_syntax = observation_tokenizer = None
    if (
        args.observe_paths
        or args.audit
        or args.menu
        or args.observe_terminal
        or args.path_demos is not None
    ):
        from xgraph.index import saved_span_syntax
        from xgraph.prompts import observed_requests
        from xgraph.tokenizer import ServedTokenizer

        observation_syntax = saved_span_syntax(os.environ["XGRAPH_KG_INDEX"])
        observation_tokenizer = ServedTokenizer.from_file(os.environ["XGRAPH_TOKENIZER"])
        if args.path_demos is not None:
            from xgraph.prompts import path_demo_requests

            requests = path_demo_requests(requests, args.path_demos, observation_syntax)
        if args.audit:
            from xgraph.prompts import audit_requests

            requests = audit_requests(
                requests,
                observation_syntax,
                entities=args.audit_entities,
                menus=args.audit_relations,
            )
        elif args.observe_paths:
            requests = observed_requests(
                requests,
                args.observe_paths,
                observation_syntax,
                select=args.observe_select,
                width=args.observe_width,
                demos=args.path_demos,
            )
        if args.menu:
            from xgraph.prompts import menu_requests

            requests = menu_requests(
                requests,
                args.menu,
                observation_syntax,
                observe=args.observe_paths,
                select=args.observe_select,
                demos=args.path_demos,
                width=args.observe_width,
            )
        if args.menu_close_next or args.observe_defer:
            from xgraph.prompts import decision_requests

            requests = decision_requests(
                requests,
                observation_syntax,
                close_next=args.menu_close_next,
                defer=args.observe_defer,
            )
    if (
        args.arm in ("tools", "walk", "iterative", "agent", "cor", "tog", "gcr", "resume")
        and args.kg_versions is None
    ):
        named = sorted({r["kg_version"] for r in requests if r.get("kg_version")})
        if named:
            raise SystemExit(
                f"requests read KG versions {named[:5]}: the {args.arm} arm needs --kg-versions"
            )
    llm_args = engine_args(
        args.arm,
        args.model,
        kv_cache_bytes=int(args.kv_cache_gib * (1 << 30)),
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        prefix_caching=not args.no_prefix_caching,
        seed=args.seed,
        memory_share=args.memory_share,
        speculative_tokens=args.speculative_tokens,
        lora_rank=args.lora_rank if args.lora else 0,
    )
    if args.agent_mask:
        llm_args["logits_processors"] = [PROCESSOR]
        llm_args["worker_extension_cls"] = "xgraph.vllm.XGraphWorkerExtension"
    if config.gcr_config is not None and config.gcr_config.decoding == "beam":
        llm_args["logprobs_mode"] = "raw_logprobs"
        llm_args["max_logprobs"] = 2 * config.gcr_config.k
        if config.gcr_config.beam_semantics in ("hf", "hf-group"):
            generation_path = Path(args.model) / "generation_config.json"
            model_config = json.loads((Path(args.model) / "config.json").read_text())
            text_config = model_config.get("text_config", model_config)
            generation_values = (
                json.loads(generation_path.read_text())
                if generation_path.is_file()
                else text_config
            )
            configured_eos = generation_values.get("eos_token_id", text_config.get("eos_token_id"))
            configured_eos = (
                configured_eos
                if isinstance(configured_eos, list)
                else [configured_eos]
                if configured_eos is not None
                else []
            )
            # Full rows are requested only to resolve ties in selection.
            llm_args["max_logprobs"] = text_config.get("vocab_size")
            if type(llm_args["max_logprobs"]) is not int or llm_args["max_logprobs"] <= 0:
                raise ValueError("GCR HF beam needs vocab_size in the model's config.json")
    if config.dog_config is not None and config.dog_config.beam_width > 1:
        llm_args["logprobs_mode"] = "raw_logprobs"
    handoff = None
    if args.gcr_stage == "synthesis":
        from xgraph.gcr_stages import binding, load_handoff

        handoff = load_handoff(args.gcr_handoff, binding(args))
        llm_args = engine_args(
            "baseline",
            args.gcr_synthesis_model,
            kv_cache_bytes=int(args.kv_cache_gib * (1 << 30)),
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            prefix_caching=not args.no_prefix_caching,
            seed=args.seed,
            memory_share=args.memory_share,
        )
    llm_started = time.perf_counter()
    needs_model = handoff is None or any(r["state"]["text"] is None for r in handoff["states"])
    if needs_model and args.arm == "gcr":
        print(f"[xgraph-load] begin model={llm_args['model']}", flush=True)
    llm = vllm.LLM(**llm_args) if needs_model else None
    if needs_model and args.arm == "gcr":
        print("[xgraph-load] ready", flush=True)
    tokenizer = llm.get_tokenizer() if llm is not None else None
    llm_startup_seconds = time.perf_counter() - llm_started
    banned = []
    for token in args.ban_tokens:
        if args.gcr_stage == "synthesis":
            continue
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise SystemExit(f"--ban-tokens: {token!r} is not a token of this model")
        banned.append(int(token_id))
    opener = None
    if args.open:  # the span opener of the syntax the KG index was built for
        from xgraph.index import saved_span_syntax

        if args.arm == "baseline":
            from xgraph.syntax import SpanSyntax

            opener = (
                saved_span_syntax(args.kg_index).opener if args.kg_index else SpanSyntax().opener
            )
        else:
            opener = saved_span_syntax(os.environ["XGRAPH_KG_INDEX"]).opener
    started = time.perf_counter()
    lora_request = None
    if args.lora:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("span-format", 1, args.lora)
    interface_executor = None
    if args.arm == "resume" or (args.arm == "walk" and args.matched_paths):
        from xgraph.index import load_kg_index
        from xgraph.kgversions import load_kg_versions
        from xgraph.programs import MatchedExecutor
        from xgraph.tokenizer import ServedTokenizer

        index = load_kg_index(args.kg_index)
        interface_executor = MatchedExecutor(
            index,
            ServedTokenizer.from_file(args.tokenizer),
            versions=load_kg_versions(args.kg_versions, index) if args.kg_versions else None,
            start_rank=args.start_rank,
            unquoted_starts=args.unquoted_starts,
        )
    thinking = None
    if args.think_budget:
        if args.samples > 1:
            requests = sampled_requests(requests, args.samples, args.seed)
        # The launcher's host record has the real commit, input hashes and model
        # revision, while the wheel inside the container has no Git checkout.
        host_provenance = Path(f"{args.out}.provenance.json")
        source_provenance = (
            json.loads(host_provenance.read_text()) if host_provenance.is_file() else None
        )
        records, thinking = generate_thinking(
            llm,
            requests,
            args.arm,
            args.max_tokens,
            vllm.SamplingParams,
            think_budget=args.think_budget,
            thoughts_out=Path(f"{args.out}.thoughts.jsonl"),
            thoughts_file=args.thoughts,
            seed=args.seed,
            reasoning_effort=args.reasoning_effort,
            repetition_penalty=args.repetition_penalty,
            lora_request=lora_request,
            banned=banned,
            opener=opener,
            sampling=(args.temperature, args.top_p) if args.samples > 1 or args.sample else None,
            source_provenance=source_provenance,
        )
    elif args.arm == "tools":
        from xgraph.toolkg import KGStore, run_with_tools, version_stores

        store = KGStore(args.tools_db)
        named = {r["kg_version"] for r in requests if r.get("kg_version")}
        records = run_with_tools(
            llm,
            requests,
            store,
            sampling_params(args.arm, args.max_tokens, vllm.SamplingParams, banned=banned),
            args.max_calls,
            version_stores(store, args.kg_versions, named) if named else None,
        )
    elif args.arm == "walk":
        from xgraph.toolkg import KGStore, version_stores
        from xgraph.walk import walk

        asked = walk_requests(requests)
        store = KGStore(args.tools_db)
        named = {r["kg_version"] for r in asked if r.get("kg_version")}
        versions = version_stores(store, args.kg_versions, named) if named else None
        records = walk(
            llm,
            asked,
            store,
            sampling_params(args.arm, args.max_tokens, vllm.SamplingParams, banned=banned),
            args.max_calls,
            versions,
            config.observe,
            args.seed,
            args.max_options,
            args.coarse_types,
            config.observe_seed,
            min_hops=args.min_hops or 0,
            executor=interface_executor,
            **(
                {
                    "stride": args.walk_stride,
                    "max_prompt_tokens": args.max_model_len - args.max_tokens,
                }
                if args.walk_stride is not None
                else {}
            ),
        )
    elif args.arm == "resume":
        from xgraph.index import saved_span_syntax
        from xgraph.resume import resume
        from xgraph.toolkg import KGStore, version_stores

        store = KGStore(args.tools_db)
        named = {r["kg_version"] for r in requests if r.get("kg_version")}
        records = resume(
            llm,
            requests,
            store,
            interface_executor,
            sampling_params(args.arm, args.max_tokens, vllm.SamplingParams, banned=banned),
            resume_max=args.resume_max,
            max_hops=args.max_calls,
            versions=version_stores(store, args.kg_versions, named) if named else None,
            max_prompt_tokens=args.max_model_len - args.max_tokens,
            opener=saved_span_syntax(args.kg_index).opener,
        )
    elif args.arm == "iterative":
        from xgraph.iterative import START_TOKENS, iterate
        from xgraph.toolkg import KGStore, version_stores

        named = {r["kg_version"] for r in requests if r.get("kg_version")}
        versions = (
            version_stores(KGStore(args.tools_db), args.kg_versions, named) if named else None
        )
        records = iterate(
            llm,
            requests,
            args.tools_db,
            sampling_params(args.arm, args.max_tokens, vllm.SamplingParams, banned=banned),
            args.max_calls,
            versions,
            RETRIEVAL_CAP,
            max_prompt_tokens=args.max_model_len - args.max_tokens,
            written_start=args.written_start,
            start_params=sampling_params(
                args.arm, START_TOKENS, vllm.SamplingParams, banned=banned
            ),
        )
    elif args.arm in ("dog", "dog-gold"):
        from xgraph.dog import run_dog

        eos = []
        generation_config = Path(args.model) / "generation_config.json"
        if generation_config.is_file():
            value = json.loads(generation_config.read_text()).get("eos_token_id")
            eos = value if isinstance(value, list) else [value] if isinstance(value, int) else []
        records = run_reader(
            run_dog,
            llm,
            requests,
            vllm.SamplingParams,
            prompts=config.dog_prompts,
            config=config.dog_config,
            arm=args.arm,
            batch_size=1 if args.question_timings else (args.dog_batch_size or 32),
            lora_request=lora_request,
            eos_token_ids=eos,
            question_timings=args.question_timings,
        )
    elif args.arm == "tog":
        from xgraph.tog import run_tog
        from xgraph.toolkg import KGStore, version_stores

        asked = [
            {**a, "topics": r.get("topics", [])}
            for a, r in zip(walk_requests(requests), requests, strict=True)
        ]
        store = KGStore(args.tools_db)
        named = {r["kg_version"] for r in asked if r.get("kg_version")}
        records = run_reader(
            run_tog,
            llm,
            asked,
            store,
            sampling_params(
                args.arm,
                args.max_tokens,
                vllm.SamplingParams,
                repetition_penalty=args.repetition_penalty,
                banned=banned,
            ),
            prompts=config.tog_prompts,
            config=config.tog_config,
            batch_size=args.tog_batch_size or 32,
            question_timings=args.question_timings,
            versions=version_stores(store, args.kg_versions, named) if named else None,
            lora_request=lora_request,
        )
    elif args.arm == "cor":
        from xgraph.agent import relation_names_of
        from xgraph.cor import run_cor
        from xgraph.toolkg import KGStore, version_stores

        asked = [
            {**a, "topics": r.get("topics", [])}
            for a, r in zip(walk_requests(requests), requests, strict=True)
        ]
        store = KGStore(args.tools_db)
        named = {r["kg_version"] for r in asked if r.get("kg_version")}
        names = None
        if args.relation_labels:
            lines = args.relation_labels.read_text(encoding="utf-8").splitlines()
            names = relation_names_of([tuple(json.loads(line)) for line in lines if line])
        records = run_reader(
            run_cor,
            llm,
            asked,
            store,
            sampling_params(
                args.arm,
                args.max_tokens,
                vllm.SamplingParams,
                repetition_penalty=args.repetition_penalty,
                banned=banned,
            ),
            args.max_calls,
            names,
            version_stores(store, args.kg_versions, named) if named else None,
            args.cor_depth,
            max_prompt_tokens=args.max_model_len - args.max_tokens,
            lora_request=lora_request,
            prompts=config.cor_prompts,
            mode=args.cor_mode,
            seed=args.seed,
            exploration_temperature=args.cor_temperature_exploration,
            reasoning_temperature=args.cor_temperature_reasoning,
            question_timings=args.question_timings,
        )
    elif args.arm == "gcr":
        from xgraph.gcr import GCRScreenBudget, diagnostic_engine_shutdown, run_gcr
        from xgraph.toolkg import KGStore, version_stores

        asked = [
            {**a, "topics": r.get("topics", [])}
            for a, r in zip(walk_requests(requests), requests, strict=True)
        ]
        store = KGStore(args.tools_db)
        named = {r["kg_version"] for r in asked if r.get("kg_version")}
        synthesis_loaded = args.gcr_stage == "synthesis" and needs_model

        screen_budget = (
            GCRScreenBudget(
                args.gcr_screen_panel_questions,
                args.gcr_screen_max_hours,
                startup_seconds=llm_startup_seconds,
            )
            if args.gcr_screen_max_hours is not None
            else None
        )
        eos = []
        generation_config = Path(args.model) / "generation_config.json"
        if generation_config.is_file():
            value = json.loads(generation_config.read_text()).get("eos_token_id")
            eos = value if isinstance(value, list) else [value] if isinstance(value, int) else []
        beam_vocab_size = None
        beam_logits_dtype = None
        if config.gcr_config.beam_semantics in ("hf", "hf-group"):
            eos = configured_eos
            beam_vocab_size = text_config.get("vocab_size")
            if type(beam_vocab_size) is not int or beam_vocab_size <= 0:
                raise ValueError("GCR HF beam needs vocab_size in the model's config.json")
            if config.gcr_config.beam_semantics == "hf-group":
                beam_logits_dtype = (
                    text_config.get("dtype")
                    or text_config.get("torch_dtype")
                    or model_config.get("dtype")
                    or model_config.get("torch_dtype")
                    or "float32"
                )
        if args.gcr_stage == "synthesis":
            from xgraph.gcr_stages import synthesize

            records = synthesize(llm, handoff, config.gcr_config, vllm.SamplingParams)
        else:
            from xgraph.gcr_stages import binding

            with diagnostic_engine_shutdown(llm):
                records = run_gcr(
                    llm,
                    asked,
                    store,
                    vllm.SamplingParams,
                    config=config.gcr_config,
                    prompts=config.gcr_prompts,
                    eos_token_ids=eos,
                    beam_vocab_size=beam_vocab_size,
                    beam_logits_dtype=beam_logits_dtype,
                    versions=version_stores(store, args.kg_versions, named) if named else None,
                    banned=banned,
                    lora_request=lora_request,
                    screen_budget=screen_budget,
                    path_handoff=(args.gcr_handoff, binding(args))
                    if args.gcr_stage == "paths"
                    else None,
                )
    elif args.arm == "agent":
        from xgraph.agent import demonstrations, relation_names_of
        from xgraph.agent_protocol import run_agent
        from xgraph.toolkg import KGStore, version_stores

        asked = walk_requests(requests)
        if args.matched_demos:
            asked = [
                {**a, "demonstrations": demonstrations(r["messages"])}
                for a, r in zip(asked, requests, strict=True)
            ]
        store = KGStore(args.tools_db)
        named = {r["kg_version"] for r in asked if r.get("kg_version")}
        names = None
        if args.relation_labels:
            lines = args.relation_labels.read_text(encoding="utf-8").splitlines()
            names = relation_names_of([tuple(json.loads(line)) for line in lines if line])
        executor = None
        if args.matched_paths:
            from xgraph.index import load_kg_index
            from xgraph.kgversions import load_kg_versions
            from xgraph.programs import MatchedExecutor
            from xgraph.tokenizer import ServedTokenizer

            index = load_kg_index(args.kg_index)
            executor = MatchedExecutor(
                index,
                ServedTokenizer.from_file(args.tokenizer),
                versions=load_kg_versions(args.kg_versions, index) if named else None,
                start_rank=args.start_rank,
                unquoted_starts=args.unquoted_starts,
            )

        def round_params(request, prefill):
            return sampling_params(
                "constrained" if args.agent_mask else "agent",
                args.max_tokens,
                vllm.SamplingParams,
                args.repetition_penalty,
                banned,
                anchor=request.get("anchor"),
                kg_version=request.get("kg_version"),
                prefill=prefill if args.agent_mask else None,
            )

        records = run_agent(
            llm,
            asked,
            store,
            sampling_params(args.arm, args.max_tokens, vllm.SamplingParams, banned=banned),
            args.max_calls,
            names,
            version_stores(store, args.kg_versions, named) if named else None,
            executor,
            args.pages,
            args.agent_feedback,
            mask=args.agent_mask,
            force_path=args.agent_force_path,
            round_params=round_params
            if args.agent_mask or args.agent_force_path or args.agent_schema
            else None,
            schema=args.agent_schema,
            schema_params=_structured_params() if args.agent_schema else None,
            max_model_len=args.max_model_len,
            answer_parser_version=args.agent_answer_parser,
            final_answer_controller=args.agent_final_controller,
        )
    elif args.arm == "fallback":
        records = fallback(
            llm,
            requests,
            args.max_tokens,
            vllm.SamplingParams,
            args.repetition_penalty,
            lora_request,
            banned,
            opener=opener,
        )
    else:
        sampling = None
        if args.samples > 1:
            requests = sampled_requests(requests, args.samples, args.seed)
        if args.samples > 1 or args.sample:
            sampling = (args.temperature, args.top_p)
        records = generate(
            llm,
            requests,
            args.arm,
            args.max_tokens,
            vllm.SamplingParams,
            args.repetition_penalty,
            lora_request,
            banned,
            sampling,
            opener,
        )
    seconds = time.perf_counter() - started
    return GenerationRun(
        requests=requests,
        records=records,
        llm=llm,
        vllm=vllm,
        llm_args=llm_args,
        opener=opener,
        seconds=seconds,
        llm_startup_seconds=llm_startup_seconds,
        thinking=thinking,
        observation_syntax=observation_syntax,
        observation_tokenizer=observation_tokenizer,
        screen_budget=screen_budget if args.arm == "gcr" else None,
        synthesis_loaded=synthesis_loaded if args.arm == "gcr" else False,
    )


def _write_receipts(args, config: ArmConfiguration, run: GenerationRun) -> None:
    if config.dog_config is not None:
        from xgraph.dog import reader_metadata
    if args.observe_terminal:
        from xgraph.audit import count_terminal_tokens

        for record in run.records:
            record["generated_tokens"] = record.get("generated_tokens", len(record["token_ids"]))
            record["observation_tokens"] = count_terminal_tokens(
                record["token_ids"],
                run.observation_tokenizer,
                run.observation_syntax,
                prefill=record.get("opener") or record.get("prefill") or "",
            )
    if args.audit:
        from xgraph.audit import count_audit_tokens
        from xgraph.programs import record_path

        for record in run.records:
            record["generated_tokens"] = record.get("generated_tokens", len(record["token_ids"]))
            record["observation_tokens"] = count_audit_tokens(
                record["token_ids"],
                run.observation_tokenizer,
                run.observation_syntax,
                prefill=record.get("opener") or record.get("prefill") or "",
            )
            parsed = record_path(record)
            if parsed is not None and parsed[1].audit is not None:
                start, relations = parsed
                record["plan"] = {
                    "start": start,
                    "relations": list(relations),
                    "audit": relations.audit,
                }
                if relations.selections:
                    record["plan"]["selections"] = relations.selections
    if args.observe_paths or args.menu:
        from xgraph.observations import count_observation_tokens

        if args.observe_required:
            from xgraph.index import load_kg_index
            from xgraph.kgversions import load_kg_versions
            from xgraph.observations import PathRelations, path_parts
            from xgraph.programs import MatchedExecutor

            # Offline accounting only. Nothing runs on the host per decode step.
            index = load_kg_index(os.environ["XGRAPH_KG_INDEX"])
            versions_path = os.environ.get("XGRAPH_KG_VERSIONS")
            executor = MatchedExecutor(
                index,
                run.observation_tokenizer,
                versions=load_kg_versions(versions_path, index) if versions_path else None,
                start_rank=os.environ.get("XGRAPH_START_RANK", "degree"),
                unquoted_starts=os.environ.get("XGRAPH_UNQUOTED_STARTS") == "1",
                committed_paths=os.environ.get("XGRAPH_COMMITTED_PATHS") == "1",
            )
            request_of = {request["id"]: request for request in run.requests}
        for record in run.records:
            if args.menu_close_next or args.observe_defer:
                from xgraph.programs import record_path

                parsed = record_path(record)
                if parsed is not None:
                    start, relations = parsed
                    record["plan"] = {"start": start, "relations": list(relations)}
                    for name, value in (
                        ("selections", relations.selections),
                        ("defers", sorted(relations.defers)),
                        ("decisions", relations.decisions),
                    ):
                        if value:
                            record["plan"][name] = value
            selection_required = None
            if args.observe_required:
                request = request_of[record["id"]]

                def selection_required(body, request=request):
                    from xgraph.menus import semantic_text

                    body = semantic_text(
                        run.observation_syntax.opener + body,
                        run.observation_syntax,
                        keep_decisions=True,
                    )[len(run.observation_syntax.opener) :]
                    parts, selections, defers, decisions = path_parts(
                        body,
                        run.observation_syntax.separator,
                        run.observation_syntax.opener,
                        run.observation_syntax.escape,
                        choices=True,
                    )
                    return (
                        len(
                            executor.frontier(
                                parts[0],
                                PathRelations(
                                    parts[1:], selections, defers=defers, decisions=decisions
                                ),
                                request.get("anchor"),
                                request.get("kg_version"),
                            )
                        )
                        > 1
                    )

            record["generated_tokens"] = record.get("generated_tokens", len(record["token_ids"]))
            record["observation_tokens"] = count_observation_tokens(
                record["token_ids"],
                run.observation_tokenizer,
                run.observation_syntax,
                prefill=record.get("opener") or record.get("prefill") or "",
                force_answer=args.observe_answer,
                **({"selection_required": selection_required} if args.observe_required else {}),
            )
    output_path = Path(str(args.out) + ".paths.jsonl") if args.gcr_stage == "paths" else args.out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for record in run.records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        **(reader_metadata(run.requests, args.arm) if config.dog_config is not None else {}),
        "artifact_kind": "xgraph.generation",
        "version": 1,
        "arm": args.arm,
        **(
            {
                "agent_protocol": {
                    **(
                        {"answer_parser_version": args.agent_answer_parser}
                        if args.agent_answer_parser
                        else {}
                    ),
                    **(
                        {"final_answer_controller_version": args.agent_final_controller}
                        if args.agent_final_controller
                        else {}
                    ),
                }
            }
            if args.agent_answer_parser or args.agent_final_controller
            else {}
        ),
        "requests": len(run.records),
        "max_tokens": args.max_tokens,
        "repetition_penalty": args.repetition_penalty,
        "banned_tokens": args.ban_tokens,
        **(
            {"lexical_schema": os.environ["XGRAPH_LEXICAL_SCHEMA"]}
            if os.environ.get("XGRAPH_LEXICAL_SCHEMA")
            else {}
        ),
        "opener": run.opener,
        "engine_args": run.llm_args,
        "vllm_version": run.vllm.__version__,
        "seconds": run.seconds
        + (
            run.thinking["stage_seconds"]["thought"]
            if run.thinking and run.thinking["reused"]
            else 0.0
        ),
        "timing_kind": (
            "composed_workload_wall" if run.thinking and run.thinking["reused"] else "workload_wall"
        ),
        "cache_state": "uncontrolled",
        **(
            {**reader_scheduling(args), "startup_seconds": run.llm_startup_seconds}
            if args.arm in ("tog", "dog", "dog-gold", "cor", "gcr")
            else {}
        ),
        **(
            {
                "question_timings": True,
                "startup_seconds": run.llm_startup_seconds,
                "question_timing_kind": "serial wall time including CPU orchestration",
            }
            if args.question_timings
            else {}
        ),
        **(
            {"thinking": run.thinking, "executed_seconds": run.seconds}
            if run.thinking is not None
            else {}
        ),
        **(
            {
                "gcr": {
                    "decoding": config.gcr_config.decoding,
                    **(
                        {"prompts": config.gcr_prompts.metadata()}
                        if config.gcr_prompts is not None
                        else {}
                    ),
                    "k": config.gcr_config.k,
                    "hops": config.gcr_config.hops,
                    "truncated_tries": sum(r["trie"].get("truncated", False) for r in run.records),
                    "synthesis_model": args.gcr_synthesis_model or args.model,
                    "includes_synthesis_startup": run.synthesis_loaded,
                    "scoring_requests": sum(
                        r.get("beam_search", {}).get("scoring_requests", 0) for r in run.records
                    ),
                    "scoring_requests_per_second": (
                        sum(
                            r.get("beam_search", {}).get("scoring_requests", 0) for r in run.records
                        )
                        / run.seconds
                        if run.seconds > 0
                        else None
                    ),
                    **(
                        {"screening": run.screen_budget.summary()}
                        if run.screen_budget is not None
                        else {}
                    ),
                }
            }
            if args.arm == "gcr"
            else {}
        ),
        "prompt_tokens": sum(r["prompt_tokens"] for r in run.records),
        "generated_tokens": sum(
            r["generated_tokens"] if "generated_tokens" in r else len(r["token_ids"])
            for r in run.records
        ),
        **(
            {"observation_tokens": sum(r["observation_tokens"] for r in run.records)}
            if args.observe_paths or args.audit or args.menu or args.observe_terminal
            else {}
        ),
        "distributions": distributions(map(request_cost, run.records)),
        "fallbacks": sum(bool(r.get("fallback")) for r in run.records),
        "finish_reasons": {
            reason: sum(r["finish_reason"] == reason for r in run.records)
            for reason in sorted({str(r["finish_reason"]) for r in run.records})
        },
    }
    if args.gcr_stage:
        from xgraph.gcr_stages import finish_summary

        finish_summary(args, summary, run.llm_startup_seconds)
    Path(f"{output_path}.summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    if args.gcr_stage == "paths":
        from xgraph.gcr_stages import seal_handoff

        seal_handoff(args.gcr_handoff)
    from xgraph.score import replay_failure

    failed = [r["id"] for r in run.records if replay_failure(r)]
    if failed:
        raise SystemExit(
            f"replayed prefill failed for requests {failed!r}; outputs and costs retained"
        )


def main(argv: list[str] | None = None) -> None:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _argument_parser()
    args, config = _parse_arguments(parser, argv)
    if args.provenance_only:
        _write_provenance(args, config)
        return
    if args.agent_answer_parser or args.agent_final_controller:
        _write_provenance(args, config)
    # One isolated SQLite connection per question component can exceed Docker's
    # default soft limit. Do this before model construction so workers inherit it.
    import resource

    before = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = 65536 if before[1] == resource.RLIM_INFINITY else min(before[1], 65536)
    if before[0] != resource.RLIM_INFINITY and before[0] < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, before[1]))
    destination = Path(str(args.out) + ".paths.jsonl") if args.gcr_stage == "paths" else args.out
    provenance_path = Path(f"{destination}.provenance.json")
    limits = {"before": before, "after": resource.getrlimit(resource.RLIMIT_NOFILE)}
    if (
        args.arm == "gcr"
        and args.gcr_synthesis_model
        and args.gcr_synthesis_model != args.model
        and args.gcr_stage is None
    ):
        from xgraph.gcr_stages import coordinate

        coordinate(argv, args)
        return
    succeeded = False
    try:
        run = _execute(args, config, parser)
        succeeded = True
    finally:
        # Preserve the host provenance consumed by thinking during execution.
        # A failed launch updates an existing sidecar without creating output paths.
        if succeeded or provenance_path.is_file():
            provenance_path.parent.mkdir(parents=True, exist_ok=True)
            provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
            provenance["rlimit_nofile"] = limits
            provenance_path.write_text(json.dumps(provenance, indent=1) + "\n")
    _write_receipts(args, config, run)


if __name__ == "__main__":
    main()
