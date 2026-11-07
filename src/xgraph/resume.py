"""Black-box execute-and-resume over the same indexed KG state as path mode.

The first call continues the free planner's original path prompt and span opener.
MatchedExecutor executes the free path without repair. At its first unavailable
relation, the successful prefix is frozen. A user message shows that prefix, the
live frontier it reached and a page of its writable relations, in its KG version.
The model writes only a replacement suffix, or requests another relation page with
`Page: N`. Pages use agent.SHOWN and the failure-aware agent's count/offset/
next_offset format; every page and suffix call consumes one of resume_max extra
rounds (default three). No logits, masks, or gold enter planning.

Execution stops at the path-mode hop cap. Only a fully executable plan delivers an
answer, using matched executor delivery. plan/reached also support the existing
repair deliver/merge --executed pipeline for separate engine delivery. Raw turns
and summed prompt/generated tokens count all calls, including unsuccessful retries
and page requests. A context limit refuses the next call without clipping a menu.
"""

from __future__ import annotations

import json
import re

from xgraph.agent import SHOWN
from xgraph.oracle import MAX_PATH_RELATIONS
from xgraph.programs import parse_path
from xgraph.repair import plan_text
from xgraph.toolkg import store_for

RESUME_MAX = 3
INSTRUCTIONS = """Continue the path at its first unavailable relation. Keep the executed prefix \
fixed. Replace the failed relation and everything after it with a suffix: reply only \
⟦ relation · relation⟧, containing the remaining relations, without the start or prefix. \
Reply ⟦⟧ if the executed prefix already answers the question. \
Do not answer from memory. frontier_relations lists writable relation labels and aliases; \
when next_offset is present, reply Page: N to see that offset. A page request consumes \
one continuation round. At most {remaining} relations may be added.
Observation: """
_SUFFIX = re.compile(r"\s*⟦([^⟧]*)⟧\s*\.?\s*\Z")
_PAGE = re.compile(r"\s*Page:\s*(\d+)\s*\Z")


def resume(
    llm,
    requests: list[dict],
    store,
    executor,
    params,
    *,
    resume_max: int = RESUME_MAX,
    max_hops: int = MAX_PATH_RELATIONS,
    versions: dict | None = None,
    max_prompt_tokens: int | None = None,
    opener: str = "⟦",
) -> list[dict]:
    """One executed answer per question, with bounded text-only continuation calls."""
    if resume_max < 0:
        raise ValueError("resume_max must be non-negative")
    if not 1 <= max_hops <= MAX_PATH_RELATIONS:
        raise ValueError(f"max_hops must be between 1 and {MAX_PATH_RELATIONS}")
    states = []
    for request in requests:
        messages = [dict(m) for m in request["messages"]]
        if request.get("answer_mode") == "set":
            from xgraph.prompts import SET_INSTRUCTION

            instruction = (
                SET_INSTRUCTION + "\nChoose a path whose complete reached set answers the question."
            )
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] += "\n" + instruction
            else:
                messages.insert(0, {"role": "system", "content": instruction})
        if not messages or messages[-1]["role"] != "assistant":
            messages.append({"role": "assistant", "content": opener})
        if request.get("prefill_replay") or messages[-1]["content"] != opener:
            raise ValueError(f"resume request {request['id']!r} must prefill only the span opener")
        states.append({"request": request, "read": store_for(request, store, versions),
                       "messages": messages, "turns": [], "feedback": [], "replies": [],
                       "prefix": [], "relations": [], "start": None, "found": None,
                       "status": None, "offset": 0, "planner_text": ""})  # fmt: skip

    def inspect(state):
        request, start, relations = state["request"], state["start"], state["relations"]
        anchor, version = request.get("anchor"), request.get("kg_version")
        if not executor.starts(start, anchor, version):
            state["status"] = "no_start"
            return
        for hop in range(1, len(relations) + 1):
            found = executor.execute(start, relations[:hop], anchor, version)
            if not found.reached:
                state["prefix"] = relations[: hop - 1]
                state["failed_hop"] = hop
                state["offset"] = 0
                return
        if not relations or not found.plan:
            state["status"] = "malformed"
            return
        state["found"], state["status"] = found, "executed"

    def feedback(state):
        request, prefix = state["request"], state["prefix"]
        anchor, version = request.get("anchor"), request.get("kg_version")
        labels = executor.frontier_labels(state["start"], prefix, anchor, version)
        reached = executor.frontier(state["start"], prefix, anchor, version)
        offset = state["offset"]
        page = {"count": len(labels), "relations": labels[offset : offset + SHOWN]}
        if offset:
            page["offset"] = offset
        if offset + SHOWN < len(labels):
            page["next_offset"] = offset + SHOWN
        observation = {
            "executed_prefix": {"start": state["start"], "relations": prefix},
            "failed_hop": state["failed_hop"],
            "reached": {
                "count": len(reached),
                "entities": [
                    {"id": q, "label": state["read"].label(int(q[1:])) or ""}
                    for q in reached[:SHOWN]
                ],
            },
            "frontier_relations": page,
        }
        state["feedback"].append(observation)
        state["messages"].append(
            {
                "role": "user",
                "content": INSTRUCTIONS.format(remaining=max_hops - len(prefix))
                + json.dumps(observation, ensure_ascii=False),
            }
        )

    tokenizer = llm.get_tokenizer() if max_prompt_tokens is not None else None
    for step in range(resume_max + 1):
        active = []
        for state in states:
            if state["status"] is not None:
                continue
            if step:
                feedback(state)
            if tokenizer is not None:
                tokens = tokenizer.apply_chat_template(
                    state["messages"],
                    tokenize=True,
                    add_generation_prompt=step != 0,
                    continue_final_message=step == 0,
                    enable_thinking=False,
                )
                if len(tokens) > max_prompt_tokens:
                    state["status"] = "context_limit"
                    continue
            active.append(state)
        if not active:
            break
        outputs = llm.chat(
            [s["messages"] for s in active],
            params,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
            **(
                {"continue_final_message": True, "add_generation_prompt": False} if not step else {}
            ),
        )
        for state, output in zip(active, outputs, strict=True):
            completion = output.outputs[0]
            text = completion.text
            state["replies"].append(text)
            state["turns"].append(
                {
                    "stage": "planner" if not step else "resume",
                    "reply": text,
                    "prompt_tokens": len(output.prompt_token_ids),
                    "generated_tokens": len(completion.token_ids),
                    "token_ids": list(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                }
            )
            if not step:
                state["planner_text"] = opener + text
                state["messages"][-1]["content"] = opener + text
            else:
                state["messages"].append({"role": "assistant", "content": text})
            if completion.finish_reason not in ("stop", "eos"):
                state["status"] = "truncated" if completion.finish_reason == "length" else "error"
                continue
            if not step:
                path = parse_path(opener + text)
                if path is None:
                    state["status"] = "malformed"
                    continue
                state["start"], state["relations"] = path[0], path[1][:max_hops]
            elif page := _PAGE.fullmatch(text):
                state["offset"] = int(page[1])
                continue
            elif suffix := _SUFFIX.fullmatch(text):
                parts = [p.strip() for p in suffix[1].split("·")] if suffix[1].strip() else []
                if not all(parts):
                    state["status"] = "malformed"
                    continue
                state["relations"] = [*state["prefix"], *parts][:max_hops]
            else:
                state["status"] = "malformed"
                continue
            inspect(state)

    records = []
    for state in states:
        request, found = state["request"], state["found"]
        status = state["status"] or "resume_cap"
        answer = state["read"].label(int(found.answer[1:])) if found and found.answer else None
        records.append(
            {
                "id": request["id"],
                "arm": "resume",
                "text": plan_text(found.plan[0], list(found.plan[1:])) + f"\nAnswer: ⟦ {answer}⟧"
                if found and answer
                else "",
                "plan": {"start": found.plan[0], "relations": list(found.plan[1:])}
                if found
                else None,
                "reached": [int(q[1:]) for q in found.reached] if found else [],
                "planner_text": state["planner_text"],
                "feedback": state["feedback"],
                "replies": state["replies"],
                "turns": state["turns"],
                "rounds": len(state["turns"]),
                "resume_rounds": max(0, len(state["turns"]) - 1),
                "resume_status": status,
                "budget_spent": status == "resume_cap",
                "token_ids": [],
                "prompt_tokens": sum(t["prompt_tokens"] for t in state["turns"]),
                "generated_tokens": sum(t["generated_tokens"] for t in state["turns"]),
                "finish_reason": "stop" if status == "executed" else status,
                **({"kg_version": request["kg_version"]} if request.get("kg_version") else {}),
            }
        )
    from xgraph.programs import set_execution_fields

    for record, state in zip(records, states, strict=True):
        if state["request"].get("answer_mode") == "set":
            record.update(set_execution_fields(record, state["request"], executor, state["read"]))
    return records
