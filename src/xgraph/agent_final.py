"""Opt-in agent-final-retry-v1 controller, forked from the frozen agent controller.

Keep legacy protocol changes out of agent.py: sealed packets pin its bytes.
Only the forced-final transition and explicit retry fields differ here.
"""

from __future__ import annotations

import json
from typing import Any

from xgraph.agent import (
    _ACTION,
    _POSITIONS,
    _SPAN_ACTION,
    FINAL,
    FORCED_PATH,
    INVALID,
    MAX_CALLS,
    PAGED,
    REFUSALS,
    TOOLS,
    KGTools,
    MatchedPaths,
    _offset,
    _plan,
    agent_messages,
    delivered,
    named_relations,
    parse_action,
    tool_schema,
)
from xgraph.prompts import SET_FORMAT, SET_INSTRUCTION, set_answer_fields
from xgraph.toolkg import store_for

VERSION = "agent-final-retry-v1"
FINAL_ONLY = "Give a final answer only. Do not call tools. "


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
                "final_answer_retries": 0,
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
            if (
                state["final"]
                and not state["final_answer_retries"]
                and parse_action(text, spans=spans) is not None
            ):
                final_format = (
                    SET_INSTRUCTION + "\n" + SET_FORMAT
                    if state["request"].get("answer_mode") == "set"
                    else FINAL
                )
                state["messages"].extend(
                    [
                        {"role": "assistant", "content": text.strip()},
                        {"role": "user", "content": FINAL_ONLY + final_format},
                    ]
                )
                state["final_answer_retries"] = 1
                continue
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
            "final_answer_controller_version": VERSION,
            "final_answer_retries": s["final_answer_retries"],
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
