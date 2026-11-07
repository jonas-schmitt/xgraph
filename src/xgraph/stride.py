"""Variable-stride actions over matched execution, with an atomic lookahead control.

One ordinal model choice executes one or two ordinary relations. Candidate pairs
are derived from the capped live frontier, never a composed adjacency shortcut.
The ordinary path executor remains responsible for physical hops, versions,
start commitment and witnesses. No engine mask or slot state changes.
"""

from __future__ import annotations

import re

from xgraph.oracle import MAX_PATH_RELATIONS
from xgraph.repair import content_stems
from xgraph.walk import START_PROMPT, parse_start

MENU_WIDTH = 5
OBSERVE = 3
OBSERVE_WIDTH = 8


class ActionMenu(list):
    """Ordinal actions and display omissions, shared by both stride settings."""

    def __init__(self, atomic, pairs, omitted_atomic):
        super().__init__([[a] for a in atomic] + [list(pair) for pair in pairs[:MENU_WIDTH]])
        self.omitted_atomic = omitted_atomic
        self.omitted_pairs = max(0, len(pairs) - MENU_WIDTH)


def action_menu(executor, start, prefix, question, *, anchor=None, version=None, max_hops=8):
    """Five WT-ranked atomic relations and five question-ranked executable pairs.

    Both hops' relation-id menus use WT ranking, canonical/fallback labels and
    width five. Pairs keep the original content-overlap/label rank. Each first
    relation is expanded before discovering second relations, retaining ordinary
    execution caps and surviving start ranks. Both arms consume this same menu.
    """
    if not 1 <= max_hops <= MAX_PATH_RELATIONS or len(prefix) > max_hops:
        raise ValueError("max_hops must fit the physical path limit and prefix")
    if len(prefix) == max_hops:
        return []
    atomic, omitted = executor.ranked_relation_labels(
        start, prefix, question, anchor, version, width=MENU_WIDTH
    )
    pairs = []
    if len(prefix) + 2 <= max_hops:
        pairs = {
            (a, b)
            for a in atomic
            for b in executor.ranked_relation_labels(
                start, [*prefix, a], question, anchor, version, width=MENU_WIDTH
            )[0]
        }
    words = content_stems(question)
    pairs = sorted(pairs, key=lambda pair: (-len(words & content_stems(" ".join(pair))), pair))
    return ActionMenu(atomic, pairs, omitted)


def action_prompt(
    executor, start, prefix, question, *, stride, anchor=None, version=None, max_hops=8
):
    """The shared current-state display, also used for outcome-blind CPU censuses."""
    menu = action_menu(
        executor, start, prefix, question, anchor=anchor, version=version, max_hops=max_hops
    )
    if not menu:
        return None
    entries = [
        {
            "relations": action,
            "frontier": list(executor.frontier(start, [*prefix, *action], anchor, version)),
        }
        for action in menu
    ]
    caps = {
        "width": MENU_WIDTH,
        "observe": OBSERVE,
        "observe_width": OBSERVE_WIDTH,
        "omitted_atomic": menu.omitted_atomic,
        "omitted_pairs": menu.omitted_pairs,
    }

    def heading(name, more):
        return name + (f" (+{more} more)" if more else "")

    lines = [heading("Atomic relations", menu.omitted_atomic)]
    atomic = sum(len(e["relations"]) == 1 for e in entries)
    for n, entry in enumerate(entries, 1):
        if n == atomic + 1:
            lines.append(heading("Pairs", menu.omitted_pairs))
        labels, more = executor.frontier_observation(
            entry["frontier"], version, observe=OBSERVE, width=OBSERVE_WIDTH
        )
        entry["observation_labels"], entry["observation_omitted"] = labels, more
        display = " ; ".join(labels) if labels else "∅"
        if more:
            display += f" (+{more} more)"
        lines.append(
            f"{n}. {' > '.join(entry['relations'])} → {display} ({len(entry['frontier'])} entities)"
        )
    selectable = len(entries) if stride == 2 else atomic
    instruction = (
        "One choice executes its entire listed relation sequence."
        if stride == 2
        else "Pairs are lookahead only. Choose an atomic relation; "
        "the second hop needs a later choice."
    )
    prompt = (
        f"Question: {question}\nPath so far: "
        + " → ".join([start, *prefix])
        + f"\n{instruction}\nReply with one integer from 1 to {selectable}, "
        + "or 0 to stop when the current entities answer the question.\n"
        + "\n".join(lines)
    )
    return {"prompt": prompt, "entries": entries, "caps": caps, "selectable": selectable}


def walk(
    llm,
    requests,
    store,
    params,
    *,
    executor,
    stride,
    max_hops=8,
    versions=None,
    max_prompt_tokens=None,
    min_hops=0,
):
    """Batched start/ordinal calls; concrete plans support separate engine delivery.

    stride=1 shows pairs as lookahead, stride=2 additionally makes them actions.
    Invalid/truncated choices end planning, with all calls retained and charged.
    At least one physical relation is required before emitting an answer.
    """
    from xgraph.toolkg import store_for

    if type(stride) is not int or stride not in (1, 2) or executor is None:
        raise ValueError("stride needs a matched executor and stride=1 or stride=2")
    if not 0 <= min_hops <= max_hops <= MAX_PATH_RELATIONS or max_hops < 1:
        raise ValueError("physical hop limits must fit path mode")
    executor.set_relation_labels({f"P{p}": name for p, name in store.relation_labels().items()})
    states = [
        {
            "request": r,
            "read": store_for(r, store, versions),
            "start": None,
            "relations": [],
            "actions": [],
            "menus": [],
            "menu_caps": [],
            "turns": [],
            "prompt_checks": [],
            "status": None,
        }
        for r in requests
    ]
    tokenizer = llm.get_tokenizer() if max_prompt_tokens is not None else None

    def ask(active, prompts, kind):
        from xgraph.prompts import SET_INSTRUCTION

        admitted, conversations = [], []
        for state, prompt in zip(active, prompts, strict=True):
            if state["request"].get("answer_mode") == "set":
                prompt = (
                    SET_INSTRUCTION
                    + "\nChoose a path whose complete reached set answers the question.\n"
                    + prompt
                )
            messages = [{"role": "user", "content": prompt}]
            if tokenizer is not None:
                tokens = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=False,
                    truncation=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                state["prompt_checks"].append(
                    {
                        "kind": kind,
                        "prompt_tokens": len(tokens),
                        "max_prompt_tokens": max_prompt_tokens,
                        "excess_tokens": max(0, len(tokens) - max_prompt_tokens),
                        "admitted": len(tokens) <= max_prompt_tokens,
                    }
                )
                if len(tokens) > max_prompt_tokens:
                    state["status"] = "context_limit"
                    continue
            admitted.append(state)
            conversations.append(messages)
        if not admitted:
            return []
        outputs = llm.chat(
            conversations, params, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False
        )
        replies = []
        for state, output in zip(admitted, outputs, strict=True):
            completion = output.outputs[0]
            state["turns"].append(
                {
                    "kind": kind,
                    "text": completion.text,
                    "token_ids": list(completion.token_ids),
                    "prompt_tokens": len(output.prompt_token_ids),
                    "generated_tokens": len(completion.token_ids),
                    "forced_tokens": 0,
                    "finish_reason": completion.finish_reason,
                }
            )
            if completion.finish_reason == "length":
                state["status"] = "choice_limit"
            else:
                replies.append((state, completion.text))
        return replies

    def args(state):
        r = state["request"]
        return r.get("anchor"), r.get("kg_version")

    for state in states:
        anchor, _ = args(state)
        if anchor:
            state["start"] = state["read"].label(int(anchor[0][1:]))
    active = [s for s in states if not s["start"]]
    for state, reply in ask(
        active, [START_PROMPT.format(question=s["request"]["question"]) for s in active], "start"
    ):
        state["start"] = parse_start(reply)
    for state in states:
        anchor, version = args(state)
        if not state["start"] or not executor.starts(state["start"], anchor, version):
            state["status"] = state["status"] or "no_start"

    for _ in range(max_hops + 1):
        active, prompts = [], []
        for state in states:
            if state["status"] is not None:
                continue
            prefix, start = state["relations"], state["start"]
            if len(prefix) == max_hops:
                state["status"] = "hop_limit"
                continue
            anchor, version = args(state)
            display = action_prompt(
                executor,
                start,
                prefix,
                state["request"]["question"],
                stride=stride,
                anchor=anchor,
                version=version,
                max_hops=max_hops,
            )
            if display is None:
                state["status"] = "terminal"
                continue
            state["menus"].append(display["entries"])
            state["menu_caps"].append(display["caps"])
            prompts.append(display["prompt"])
            state["selectable"] = display["selectable"]
            active.append(state)
        if not active:
            break
        for state, reply in ask(active, prompts, "action"):
            match = re.fullmatch(r"\s*(\d+)\s*", reply)
            choice = int(match[1]) if match else None
            if choice == 0 and len(state["relations"]) < max(1, min_hops):
                continue
            if choice is None or choice > state["selectable"]:
                state["status"] = "invalid_choice"
            elif choice == 0:
                state["status"] = "stop"
            else:
                action = state["menus"][-1][choice - 1]["relations"]
                state["actions"].append(action)
                state["relations"].extend(action)

    records = []
    for state in states:
        start, prefix = state["start"], state["relations"]
        anchor, version = args(state)
        failed = state["status"] in {"context_limit", "choice_limit"}
        support = (
            executor.execute_set(start, prefix, anchor, version, witnesses=True)
            if not failed and start and len(prefix) >= max(1, min_hops)
            else {"reached": [], "witnesses": {}, "cap_events": []}
        )
        reached = support["reached"]
        answer = state["read"].label(int(reached[0][1:])) if reached else None
        # A store's canonical name can be excluded from the tokenizer-bound index.
        # Only emit it when it names the witnessed endpoint in that index.
        if answer and reached[0] not in executor.named(answer):
            answer = None
        turns = state["turns"]
        path = ([start] if start else []) + prefix
        records.append(
            {
                "id": state["request"]["id"],
                "arm": "walk",
                "stride": stride,
                "text": "Path: " + " → ".join(path) + (f"\nAnswer: ⟦ {answer}⟧" if answer else ""),
                "path": path,
                "plan": {"start": start, "relations": prefix} if reached else None,
                "actions": state["actions"],
                "action_menus": state["menus"],
                "menu_caps": state["menu_caps"],
                "witnesses": support["witnesses"],
                "cap_events": support["cap_events"],
                "reached": [int(e[1:]) for e in reached],
                "matched_paths": True,
                "turns": turns,
                "replies": [t["text"] for t in turns],
                "token_ids": [],
                "model_calls": len(turns),
                "prompt_tokens": sum(t["prompt_tokens"] for t in turns),
                "generated_tokens": sum(t["generated_tokens"] for t in turns),
                "forced_tokens": 0,
                "finish_reason": "stop",
                "planner_status": state["status"] or "decision_limit",
                **({"error": "context_limit"} if state["status"] == "context_limit" else {}),
                **({"truncated": True} if state["status"] == "choice_limit" else {}),
                **({"prompt_checks": state["prompt_checks"]} if tokenizer is not None else {}),
                **({"kg_version": version} if version else {}),
            }
        )
    from xgraph.programs import set_execution_fields

    for record, state in zip(records, states, strict=True):
        if state["request"].get("answer_mode") == "set":
            record.update(set_execution_fields(record, state["request"], executor, state["read"]))
    return records
