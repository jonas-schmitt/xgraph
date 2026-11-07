"""CPU policies and scoring for the matched MINTQA queue; never loads a model."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from xgraph.audit import semantic_text
from xgraph.cost import COST_DISTRIBUTIONS, request_cost
from xgraph.mintqa_bench import answer_contains, dataset_rows, paper_accuracy
from xgraph.mintqa_panels import indexed, read_jsonl, write_json, write_jsonl
from xgraph.programs import record_path
from xgraph.prompts import final_answer
from xgraph.provenance import provenance
from xgraph.score import record_failure, record_outputs

COST_FIELDS = ("prompt_tokens", "generated_tokens", "model_calls", "observation_tokens")


def _reader_record(record):
    """DoG delivers a parsed answer; raw trailing tokens can still hit their cap.

    Reparse the saved answer call before exempting only its length finish. Keep
    all explicit failures and the common path/agent failure policy unchanged.
    """
    if (
        record.get("arm") in {"dog", "dog-gold"}
        and record.get("status") == "answered"
        and record.get("finish_reason") in {"length", "max_tokens"}
        and final_answer(record.get("text", ""))
    ):
        from xgraph.dog import parse_answer

        for turn in record.get("turns", []):
            if turn.get("stage") != "answer":
                continue
            raw = turn.get("text", "")
            # A capped last line may contain an unfinished label or list. Only a
            # newline, sentence terminator or closed Answer span ends delivery.
            complete = (
                raw
                if raw.rstrip().endswith((".", "!", "?", "⟧", '."', ".”"))
                else raw.rpartition("\n")[0]
            )
            if parse_answer(raw) == record["text"] == parse_answer(complete):
                return {**record, "finish_reason": "stop"}
    return record


def usage(record: dict | None) -> dict[str, int]:
    """Policy overrides or shared native request costs, including every model call.

    Complete usage is required; observations are a subset of generated tokens.
    """
    if record is None:
        return dict.fromkeys(COST_FIELDS, 0)
    if "usage" in record:
        policy_usage = {"observation_tokens": 0, **record["usage"]}
        result = {k: policy_usage[k] for k in COST_FIELDS}
    else:
        for key in ("model_calls", "rounds", "calls"):
            if key in record and (type(record[key]) is not int or record[key] < 0):
                raise ValueError(f"{record['id']}: invalid usage {key}={record[key]!r}")
        result = {
            **request_cost(record),
            "observation_tokens": record.get("observation_tokens", 0),
        }
    if any(type(v) is not int or v < 0 for v in result.values()):
        raise ValueError(f"{record['id']}: invalid usage {result}")
    if result["observation_tokens"] > result["generated_tokens"]:
        raise ValueError(f"{record['id']}: observation_tokens exceeds generated_tokens")
    return result


def combined_usage(*records) -> dict[str, int]:
    costs = [usage(r) for r in records]
    return {k: sum(c[k] for c in costs) for k in COST_FIELDS}


def audit_complete_usage(records, expected):
    """Admit complete panels only with conserved per-turn and per-stage usage.

    Stages charge reused thoughts as well as executed answer calls.
    """
    rows = indexed(records)
    if set(rows) != set(expected):
        raise ValueError("published-method scoring needs complete outputs and agent reference")
    for row in records:
        if "usage" not in row and not any(k in row for k in ("model_calls", "rounds", "calls")):
            raise ValueError(f"{row['id']}: missing explicit model-call accounting")
        cost = usage(row)
        if "stages" in row and "usage" not in row:
            if not isinstance(row["stages"], dict) or any(
                not isinstance(stage, dict) for stage in row["stages"].values()
            ):
                raise ValueError(f"{row['id']}: invalid stage usage")
            stages = [usage({**stage, "id": row["id"]}) for stage in row["stages"].values()]
            if any(sum(stage[k] for stage in stages) != cost[k] for k in COST_DISTRIBUTIONS):
                raise ValueError(f"{row['id']}: per-stage usage does not conserve total accounting")
        if "turns" in row and "usage" not in row:
            turns, retrieval = row["turns"], row.get("retrieval", {})
            if any(
                type(t.get(k)) is not int or t[k] < 0
                for t in turns
                for k in ("prompt_tokens", "generated_tokens")
            ) or any(
                type(retrieval.get(k, 0)) is not int or retrieval.get(k, 0) < 0
                for k in ("model_calls", "prompt_tokens", "generated_tokens")
            ):
                raise ValueError(f"{row['id']}: invalid per-call usage")
            actual = {
                "model_calls": len(turns) + retrieval.get("model_calls", 0),
                "prompt_tokens": sum(t["prompt_tokens"] for t in turns)
                + retrieval.get("prompt_tokens", 0),
                "generated_tokens": sum(t["generated_tokens"] for t in turns)
                + retrieval.get("generated_tokens", 0),
            }
            if any(actual[k] != cost[k] for k in actual):
                raise ValueError(f"{row['id']}: per-call usage does not conserve total accounting")
    return {
        "complete": True,
        "usage_audited": True,
        "questions": len(records),
        "status_counts": dict(Counter(r.get("status", "unspecified") for r in records)),
        "context_limited": sum(bool(r.get("context_limited")) for r in records),
        "budget_spent": sum(bool(r.get("budget_spent")) for r in records),
        "retrieval_truncated": sum(bool(r.get("retrieval", {}).get("cap_events")) for r in records),
        "deviations": sorted({d for r in records for d in r.get("deviations", [])}),
        "published_score_reproduction": False,
    }


def _usable(record, support) -> bool:
    return bool(
        record
        and not record_failure(record)
        and record_path(record)
        and final_answer(record["text"])
        and support.get("reached")
        and support.get("answer_in_support", True)
    )


def escalation_ids(paths: list[dict], supports: list[dict], ids: list[str]) -> list[str]:
    first, support = indexed(paths), indexed(supports)
    if set(support) != set(ids):
        raise ValueError("cascade support must cover exactly the panel ids")
    return [
        qid
        for qid in ids
        if not _usable(first.get(qid), support[qid]) or len(support[qid]["reached"]) != 1
    ]


def cascade_records(paths, agents, supports, ids) -> list[dict]:
    first, agent = indexed(paths), indexed(agents)
    escalated = set(escalation_ids(paths, supports, ids))
    if set(ids) - set(first):
        raise ValueError("missing first-stage path records in cascade")
    if escalated - set(agent):
        raise ValueError("missing routed agent records in cascade")
    result = []
    for qid in ids:
        route = qid in escalated
        selected = agent.get(qid) if route else first.get(qid)
        result.append(
            {
                **(selected or {"id": qid, "text": "", "error": "missing policy output"}),
                "usage": combined_usage(first.get(qid), agent.get(qid) if route else None),
                "escalated": route,
                "cost_sources": ["paths-allfix", "cascade-agent"] if route else ["paths-allfix"],
            }
        )
    return result


def revision_records(
    paths, revisions, supports, ids, *, first_source="paths-allfix", second_source="revision"
) -> list[dict]:
    first, revised, support = indexed(paths), indexed(revisions), indexed(supports)
    if set(ids) - set(first):
        raise ValueError("missing first-stage path records in revision")
    if set(ids) - set(revised):
        raise ValueError("missing second-stage revision records")
    if set(support) != set(ids):
        raise ValueError("revision support must cover exactly the panel ids")
    result = []
    for qid in ids:
        accepted = _usable(revised.get(qid), support[qid])
        selected = revised.get(qid) if accepted else first.get(qid)
        result.append(
            {
                **(selected or {"id": qid, "text": "", "error": "missing first output"}),
                "usage": combined_usage(first.get(qid), revised.get(qid)),
                "revision_accepted": accepted,
                "cost_sources": [first_source, second_source],
            }
        )
    return result


def revision_requests(
    requests,
    paths,
    candidates,
    executor,
    label,
    shortlist=8,
    endpoints=20,
    *,
    observe_terminal=0,
    menus=None,
    tokenizer=None,
    visibility=None,
    menu_size=8,
    seed=17,
):
    """One fresh masked call per question, with up to eight alternatives and true endpoints.

    The original plan is shown separately. All information comes from the question,
    generated paths and KG execution, never benchmark chains or answer annotations.
    """
    from xgraph.audit import terminal_instruction
    from xgraph.repair import plan_text, shortlisted

    terminal = terminal_instruction(observe_terminal)

    if shortlist < 1 or endpoints < 1:
        raise ValueError("shortlist and endpoint display limits must be positive")
    if visibility is not None and (
        visibility not in {"R", "L", "S", "N"}
        or menus is None
        or tokenizer is None
        or menu_size < 1
    ):
        raise ValueError("repair visibility needs R/L/S/N, menus, tokenizer and positive menu size")
    first, alternatives = indexed(paths), indexed(candidates)
    if visibility is not None and ({r["id"] for r in requests} - (set(first) & set(alternatives))):
        raise ValueError("repair visibility needs every frozen first call and repair candidate")
    result = []
    for request in requests:
        qid = request["id"]
        record = first.get(qid, {"id": qid, "text": ""})
        candidate = alternatives.get(qid, {"candidates": []})
        shown = []

        def describe(start, relations, request=request):
            support = executor.execute_set(
                start, relations, request.get("anchor"), request.get("kg_version")
            )
            reached = support["reached"]
            return {
                "path": plan_text(start, relations),
                "endpoint_count": len(reached),
                "endpoints": [{"id": e, "label": label(int(e[1:]))} for e in reached[:endpoints]],
                "display_truncated": len(reached) > endpoints,
                "cap_events": support["cap_events"],
                **(
                    {"selections": dict(relations.selections)}
                    if getattr(relations, "selections", {})
                    else {}
                ),
            }

        original = record_path(record)
        executed = describe(*original) if original else {"path": None, "endpoints": []}
        for k in shortlisted(candidate, shortlist + 1)[1:]:
            shown.append(describe(candidate["start"], candidate["candidates"][k]["relations"]))
        question = next(m["content"] for m in reversed(request["messages"]) if m["role"] == "user")
        feedback = {
            "question": question,
            "executed_first_path": executed,
            "first_answer": final_answer(record.get("text", "")),
            "alternatives": shown,
        }
        instruction = (
            "Check every part of the original question using these executed paths and their "
            "endpoints. They are suggestions; you may replan completely. Write one complete "
            "path from the original starting entity and a final Answer span bound to its "
            "endpoint, in the same format as before." + terminal
        )
        visibility_metadata = {}
        if visibility is not None:
            # These frontiers precede the intervention and retain first-call selections.
            available = Counter()
            if original:
                for boundary in range(len(original[1]) + 1):
                    available.update(
                        dict(
                            executor.frontier_relations(
                                original[0],
                                original[1][:boundary],
                                request.get("anchor"),
                                request.get("kg_version"),
                            )
                        )
                    )
            size = min(menu_size, len(available), len(menus.names))
            ranked = {
                "R": menus.rank(question, [(p, 0) for p in menus.names], size) if size else [],
                "L": menus.rank(question, list(available.items()), size) if size else [],
                "S": menus.blind(qid, size, seed=seed),
                "N": [],
            }
            instruction += (
                " relation_suggestions are optional relation labels from the KG schema; "
                "they do not restrict replanning or executable support. The padding field "
                "only equalizes display length; ignore it."
            )
            displays = {
                arm: {
                    **feedback,
                    "relation_suggestions": [{"id": r["id"], "label": r["label"]} for r in rows],
                    "padding": "",
                }
                for arm, rows in ranked.items()
            }
            messages = _matched_revision_displays(instruction, displays, tokenizer)
            feedback = displays[visibility]
            instruction = messages[visibility]
            visibility_metadata = {
                "visibility": visibility,
                "seed": seed,
                "menu_size": menu_size,
                "rankings": ranked[visibility],
                "display": feedback,
                "instruction_tokens": len(tokenizer.encode(instruction)),
                "matched_instruction_tokens": len(tokenizer.encode(messages["R"])),
                "suggestion_count": size if visibility != "N" else 0,
            }
        else:
            instruction += "\n" + json.dumps(feedback, ensure_ascii=False)
        result.append(
            {
                **request,
                "messages": [
                    *request["messages"],
                    {
                        "role": "assistant",
                        "content": semantic_text(record.get("text") or "")
                        or "No complete first path.",
                    },
                    {"role": "user", "content": instruction},
                ],
                "revision": {
                    **visibility_metadata,
                    **({"observe_terminal": observe_terminal} if observe_terminal else {}),
                    "shortlist": shortlist,
                    "endpoint_limit": endpoints,
                    "alternatives_shown": len(shown),
                },
            }
        )
    return result


class AuditMenus:
    """Question/alias lexical ranking over the writable relation vocabulary.

    Each relation is one IDF document (the union of its aliases' content stems).
    A relation's score is its best alias's weighted overlap, divided by the square
    root of that alias's stem count. This is the round-18 display replay's ranking.
    """

    def __init__(self, index, tokenizer):
        from xgraph.repair import content_stems

        labels = index.relations
        self.names = defaultdict(list)
        for row in range(labels.num_rows):
            text = b"".join(tokenizer.token_bytes(t) for t in labels.row_tokens(row)).decode()
            prop = labels.owner_ids[int(labels.owners[row])]
            self.names[prop].append(text.removeprefix(index.span_syntax.label_prefix))
        self.words = {p: [content_stems(s) for s in names] for p, names in self.names.items()}
        counts = Counter(w for aliases in self.words.values() for w in set().union(*aliases))
        self.idf = {w: math.log((1 + len(self.words)) / (1 + n)) + 1 for w, n in counts.items()}

    def rank(self, question, available, size):
        """Top relations, with the best-scoring writable alias and ranking evidence."""
        from xgraph.repair import content_stems

        if size < 1:
            raise ValueError("audit menu size must be positive")
        asked = content_stems(question)
        scored = []
        for prop, frequency in available:
            aliases = [
                (
                    sum(self.idf[w] for w in sorted(words & asked)) / math.sqrt(max(1, len(words))),
                    name,
                )
                for name, words in zip(self.names[prop], self.words[prop], strict=True)
            ]
            score, name = min(aliases, key=lambda pair: (-pair[0], pair[1]))
            scored.append({"id": prop, "label": name, "score": score, "frequency": frequency})
        return sorted(scored, key=lambda r: (-r["score"], -r["frequency"], int(r["id"][1:])))[:size]

    def blind(self, qid, size, *, seed=17):
        """Question-blind hash order; all writable IDs eligible, including relevant IDs.

        Count alone is supplied by the common pre-intervention frontier. Alias choice
        is lexical, independent of both the question and any ranked menu's contents.
        """
        import hashlib

        if not 0 <= size <= len(self.names):
            raise ValueError("sham count outside writable relation vocabulary")
        ordered = sorted(
            self.names,
            key=lambda p: (
                hashlib.sha256(f"repair-visibility:{seed}:{qid}:{p}".encode()).hexdigest(),
                p,
            ),
        )
        return [{"id": p, "label": min(self.names[p])} for p in ordered[:size]]


def _matched_revision_displays(instruction, displays, tokenizer):
    """Exact R/L/S user-message token equality, with a bounded, recorded filler.

    Never truncate relation labels or alter counts. Search a small common envelope;
    an unsupported tokenizer fails closed rather than relaxing the token match.
    N retains an empty list and empty padding. Padding contains no KG information.
    """

    def render(display):
        return instruction + "\n" + json.dumps(display, ensure_ascii=False)

    maximum = max(len(tokenizer.encode(render(displays[a]))) for a in ("R", "L", "S"))
    options = {}
    for arm in ("R", "L", "S"):
        display = displays[arm]
        base = len(tokenizer.encode(render(display)))
        found = {}
        for prefix in ("", " ", ".", "x"):
            for n in range(maximum - base + 65):
                padding = prefix + " x" * n
                message = render({**display, "padding": padding})
                count = len(tokenizer.encode(message))
                if maximum <= count <= maximum + 64:
                    found.setdefault(count, (padding, message))
                if count > maximum + 64:
                    break
        options[arm] = found
    common = set.intersection(*(set(o) for o in options.values()))
    if not common:
        raise ValueError("cannot exactly match repair visibility token budgets")
    target = min(common)
    result = {"N": render(displays["N"])}
    for arm, found in options.items():
        padding, result[arm] = found[target]
        displays[arm]["padding"] = padding
    return result


def audit_requests(
    requests,
    paths,
    executor,
    label,
    menus,
    tokenizer,
    menu_size=8,
    endpoints=20,
    *,
    show_menus=True,
    observe_terminal=0,
    menu_source=None,
):
    """One fresh masked audit of each first plan, using generated text and KG reads.

    Endpoints use execute_set's committed support in the executor's sitelink order.
    Menus use frontier_relations: the capped live frontier, before commitment.
    Token counts encode the exact displayed JSON and new user message without
    chat wrappers; generation's native prompt usage remains the charged cost.
    Explicit menu_source selects the matched local/global prompt control. Global
    ranking uses the complete writable schema, zero frequencies and numeric-id
    ties; only its displayed size comes from the local frontier. Neither setting
    changes the decoding constraint: frontier KG execution still applies.
    """
    from xgraph.audit import terminal_instruction
    from xgraph.oracle import MAX_PATH_RELATIONS
    from xgraph.repair import plan_text

    terminal = terminal_instruction(observe_terminal)

    if menu_size < 1 or not 1 <= endpoints <= 20:
        raise ValueError("audit menu size must be positive and endpoint display limit in [1, 20]")
    if menu_source not in (None, "local", "global") or (menu_source and not show_menus):
        raise ValueError("audit menu_source needs local or global with menus enabled")
    first = indexed(paths)
    if {r["id"] for r in requests} - set(first):
        raise ValueError("missing first-stage path records in audit")
    result = []
    for request in requests:
        record = first[request["id"]]
        question = next(m["content"] for m in reversed(request["messages"]) if m["role"] == "user")
        original = record_path(record)
        executed = {
            "path": None, "endpoint_count": 0, "endpoints": [],
            "display_truncated": False, "cap_events": [],
        }  # fmt: skip
        available = {"terminal": [], "penultimate": []}
        if original:
            start, relations = original
            support = executor.execute_set(
                start, relations, request.get("anchor"), request.get("kg_version")
            )
            relations = relations[:MAX_PATH_RELATIONS]
            reached = support["reached"]
            executed = {
                "path": plan_text(start, relations),
                "endpoint_count": len(reached),
                "endpoints": [{"id": e, "label": label(int(e[1:]))} for e in reached[:endpoints]],
                "display_truncated": len(reached) > endpoints,
                "cap_events": support["cap_events"],
                **(
                    {"selections": dict(relations.selections)}
                    if getattr(relations, "selections", {})
                    else {}
                ),
            }
            if show_menus:
                for boundary, prefix in (("terminal", relations), ("penultimate", relations[:-1])):
                    available[boundary] = executor.frontier_relations(
                        start, prefix, request.get("anchor"), request.get("kg_version")
                    )
        feedback = {
            "question": question,
            "executed_first_path": executed,
            "first_answer": final_answer(record.get("text", "")),
        }
        rankings = {}
        if show_menus:
            for boundary, found in available.items():
                pool, size = found, menu_size
                if menu_source == "global":
                    pool, size = [(p, 0) for p in menus.names], min(menu_size, len(found))
                ranked = menus.rank(question, pool, size) if size else []
                rankings[boundary] = ranked
                feedback[f"{boundary}_menu"] = {
                    "count": len(found),
                    "relations": [{"id": r["id"], "label": r["label"]} for r in ranked],
                    "display_truncated": len(found) > len(ranked),
                }
        display = json.dumps(feedback, ensure_ascii=False)
        instruction = (
            "Check the original question against the first plan and its executed endpoints. "
            "Either KEEP the plan and answer, SELECT one of its endpoints, or make exactly "
            "one terminal edit: APPEND one relation, REPLACE_LAST with one relation, or "
            "DROP_LAST (only if at least one relation remains). Do not change earlier "
            "relations or the original start. Write your choice as one complete path "
            "from that original start, followed by the bound Answer span, in the same "
            "format as before. Selection means writing the unchanged complete path and "
            "the selected endpoint in Answer. Write relation labels, not relation ids. "
            "Counts and truncation describe the display, not correctness. "
            "Undisplayed available relations and endpoints remain selectable."
        )
        if menu_source:
            instruction += (
                " terminal_menu suggests relations for APPEND; penultimate_menu suggests "
                "relations for REPLACE_LAST. Menus never change executable support. Counts "
                "and truncation report the matched local frontier sizes; suggestions may "
                "come from the whole KG schema."
            )
        elif show_menus:
            instruction += (
                " terminal_menu lists relations available after the whole path (APPEND); "
                "penultimate_menu lists relations at the frontier the last relation was "
                "applied to (REPLACE_LAST). Menus are suggestions."
            )
        instruction += terminal + "\n" + display
        messages = [
            *request["messages"],
            {
                "role": "assistant",
                "content": semantic_text(record.get("text") or "") or "No complete first path.",
            },
            {"role": "user", "content": instruction},
        ]
        result.append(
            {
                **request,
                "messages": messages,
                "audit": {
                    **({"menu_source": menu_source} if menu_source else {}),
                    **({"observe_terminal": observe_terminal} if observe_terminal else {}),
                    "menu_size": menu_size if show_menus else 0,
                    "endpoint_limit": endpoints,
                    "display": feedback,
                    "rankings": rankings,
                    "display_tokens": len(tokenizer.encode(display)),
                    "instruction_tokens": len(tokenizer.encode(instruction)),
                    "message_content_tokens": sum(
                        len(tokenizer.encode(m["content"])) for m in messages
                    ),
                },
            }
        )
    return result


def _witness_display(feedback):
    """Compact text for the witness audit; structured evidence stays in metadata."""
    lines = [f"Question: {feedback['question']}", f"First answer: {feedback['first_answer']}"]

    def path_lines(title, path):
        shown = path["endpoints"]
        count = path["endpoint_count"]
        lines.append(
            f"{title}: {path['path'] or 'No complete path'} "
            f"[{count} endpoints; {count - len(shown)} omitted]"
        )
        if path.get("selections"):
            lines.append(
                "Frontier selections: " + json.dumps(path["selections"], ensure_ascii=False)
            )
        lines.extend(" <- ".join(reversed(e["witness"])) for e in shown)
        if path["cap_events"]:
            lines.append("Execution caps: " + json.dumps(path["cap_events"], separators=(",", ":")))

    path_lines("First path", feedback["executed_first_path"])
    lines.append(f"Alternatives: {len(feedback['alternatives'])}/{feedback['alternative_count']}")
    for k, path in enumerate(feedback["alternatives"], 1):
        path_lines(f"Alternative {k}", path)
    for boundary in ("terminal", "penultimate"):
        menu = feedback[f"{boundary}_menu"]
        names = "; ".join(f"{r['id']}: {r['label']}" for r in menu["relations"])
        lines.append(f"{boundary}_menu [{len(menu['relations'])}/{menu['count']}]: {names}")
    return "\n".join(lines)


def audit_witness_requests(
    requests,
    paths,
    candidates,
    executor,
    label,
    menus,
    tokenizer,
    menu_size=8,
    endpoints=12,
    alternatives=4,
):
    """Opt-in plan audit with endpoint witnesses and ranked executed alternatives.

    Repair candidates supply only relation sequences and gold-free shortlist
    features. Endpoint identities and witnesses are re-executed in the request's
    KG version. Existing audit builders and their rendered messages stay unchanged.
    """
    from xgraph.oracle import MAX_PATH_RELATIONS
    from xgraph.repair import plan_text, shortlisted

    if menu_size < 1 or not 1 <= endpoints <= 20 or not 0 <= alternatives <= 8:
        raise ValueError(
            "witness audit needs positive menus, endpoints in [1, 20], alternatives in [0, 8]"
        )
    cached = indexed(candidates)
    if {r["id"] for r in requests} - set(cached):
        raise ValueError("missing repair candidates in witness audit")
    first = indexed(paths)
    base = audit_requests(requests, paths, executor, label, menus, tokenizer, menu_size, endpoints)
    result = []
    for made in base:
        qid = made["id"]
        record, candidate = first[qid], cached[qid]
        original = record_path(record)
        feedback = made["audit"]["display"]

        def describe(start, relations, limit, *, made=made, qid=qid):
            support = executor.execute_set(
                start,
                relations,
                made.get("anchor"),
                made.get("kg_version"),
                witnesses=True,
            )
            shown = []
            for entity in support["reached"][:limit]:
                chain = support["witnesses"][entity]
                names = [label(int(e[1:])) for e in chain]
                if any(name is None for name in names):
                    raise ValueError(
                        f"{qid}: endpoint witness has an entity without a store label: {chain}"
                    )
                shown.append(
                    {"id": entity, "label": names[-1], "witness_ids": chain, "witness": names}
                )
            return {
                "path": plan_text(start, relations[:MAX_PATH_RELATIONS]),
                "endpoint_count": len(support["reached"]),
                "endpoints": shown,
                "display_truncated": len(support["reached"]) > limit,
                "cap_events": support["cap_events"],
                **(
                    {"selections": dict(relations.selections)}
                    if getattr(relations, "selections", {})
                    else {}
                ),
            }

        shown = []
        available = max(0, len(candidate["candidates"]) - 1)
        if original:
            start, relations = original
            if candidate.get("start") != start or candidate.get("kg_version") != made.get(
                "kg_version"
            ):
                raise ValueError(
                    f"{qid}: witness audit repair candidates have a different start or KG version"
                )
            feedback["executed_first_path"] = describe(start, relations, endpoints)
            if alternatives and available:
                for k in shortlisted(candidate, alternatives + 1)[1:]:
                    shown.append(describe(start, candidate["candidates"][k]["relations"], 5))
        elif candidate["candidates"]:
            raise ValueError(f"{qid}: witness audit has candidates without a readable first path")
        feedback.update(
            alternatives=shown,
            alternative_count=available,
            alternatives_truncated=available > len(shown),
        )
        display = _witness_display(feedback)
        instruction = (
            "Check every part of the original question against these executed paths. "
            "KEEP the first plan and answer, SELECT one of its endpoints, make one terminal "
            "edit (APPEND one relation, REPLACE_LAST with one relation, or DROP_LAST if a "
            "relation remains), or ADOPT one displayed alternative path. Keep the original "
            "start; terminal edits keep all earlier relations. Write one complete path "
            "from that start followed by its bound Answer span, in the same format as before. "
            "SELECT writes the unchanged path and the selected endpoint in Answer. "
            "Write relation labels, not ids. Each endpoint's witness runs backwards to the "
            "start and shows one surviving chain. terminal_menu suggests APPEND; "
            "penultimate_menu suggests REPLACE_LAST. Counts, omissions and execution caps "
            "describe support, not correctness. Undisplayed available endpoints and "
            "relations remain selectable.\n" + display
        )
        messages = [*made["messages"][:-1], {"role": "user", "content": instruction}]
        result.append(
            {
                **made,
                "messages": messages,
                "audit": {
                    **made["audit"],
                    "variant": "witness",
                    "alternative_limit": alternatives,
                    "alternative_endpoint_limit": 5,
                    "alternatives_shown": len(shown),
                    "display_text": display,
                    "display_tokens": len(tokenizer.encode(display)),
                    "instruction_tokens": len(tokenizer.encode(instruction)),
                    "message_content_tokens": sum(
                        len(tokenizer.encode(m["content"])) for m in messages
                    ),
                },
            }
        )
    return result


def _paired_bootstrap(a, b, members, resamples, seed):
    if not members or len(a) != len(members) or len(b) != len(members) or resamples < 1:
        raise ValueError("bootstrap needs nonempty aligned pairs and positive resamples")
    if any(not math.isfinite(r["weight"]) or r["weight"] <= 0 for r in members):
        raise ValueError("bootstrap weights must be finite and positive")
    rng = np.random.default_rng(seed)
    groups = defaultdict(dict)
    mass = sum(r["weight"] for r in members)
    point = np.array(
        [
            sum(v * r["weight"] for v, r in zip(values, members, strict=True)) / mass
            for values in (a, b)
        ]
    )
    for x, y, row in zip(a, b, members, strict=True):
        pair = groups[row["hops"]].setdefault(row["cluster"], [0.0, 0.0, 0.0])
        pair[0] += x * row["weight"]
        pair[1] += y * row["weight"]
        pair[2] += row["weight"]
    draws = np.zeros((resamples, 2))
    for clusters in groups.values():
        values = np.array([clusters[c] for c in sorted(clusters)])
        hop_mass = values[:, 2].sum()
        for start in range(0, resamples, 128):
            stop = min(start + 128, resamples)
            chosen = rng.integers(len(values), size=(stop - start, len(values)))
            sampled = values[chosen].sum(axis=1)
            draws[start:stop] += sampled[:, :2] / sampled[:, 2, None] * hop_mass / mass
    return point, draws, sum(map(len, groups.values()))


def clustered_difference(a, b, members, resamples=10_000, seed=17, *, hypothesis=None) -> dict:
    """Paired percentile bootstrap, retaining weights and recalibrating hop mass.

    An optional one-sided test uses a centered bootstrap under the boundary null,
    with an add-one Monte Carlo correction. Passing also requires the percentile
    lower bound to exceed that null strictly. Intervals condition on the panel;
    they do not undo earlier MINTQA exposure.
    """
    means, samples, clusters = _paired_bootstrap(a, b, members, resamples, seed)
    point = float(means[0] - means[1])
    draws = samples[:, 0] - samples[:, 1]
    low, high = np.quantile(draws, [0.025, 0.975])
    result = {
        "difference": point,
        "low": float(low),
        "high": float(high),
        "clusters": clusters,
        "resamples": resamples,
        "seed": seed,
        "method": "paired relation-sequence clusters within hops, calibrated weights",
    }
    if hypothesis is not None:
        null, alpha = hypothesis["null_difference"], hypothesis["alpha"]
        if not math.isfinite(null) or not 0 < alpha < 1:
            raise ValueError("hypothesis needs a finite null difference and alpha in (0, 1)")
        lower = float(np.quantile(draws, alpha))
        p = float((1 + np.count_nonzero(draws - point >= point - null)) / (resamples + 1))
        result.update(
            null_difference=null,
            alpha=alpha,
            confidence=1 - alpha,
            one_sided_lower=lower,
            p_value=p,
            passes_unadjusted=lower > null and p <= alpha,
            test="centered paired cluster bootstrap upper tail, add-one correction",
        )
    return result


def holm_adjust(p_values):
    """Holm step-down adjusted p-values in their original comparison order."""
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in p_values):
        raise ValueError("Holm needs finite p-values in [0, 1]")
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, i in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def registered_comparisons(tables, registration, *, family=None):
    """Apply only frozen comparisons to complete paired score-panel tables."""
    result = {}
    for comparison in registration["comparisons"]:
        if "reference_by_family" in comparison:
            if family not in comparison["reference_by_family"]:
                raise ValueError("frozen comparison needs an explicitly registered family")
            comparison = {**comparison, "reference": comparison["reference_by_family"][family]}
        first = tables[comparison["arm"]]["per_question"]
        reference = indexed(tables[comparison["reference"]]["per_question"])
        if set(indexed(first)) != set(reference):
            raise ValueError("registered comparisons need aligned panel ids")
        second = [reference[r["id"]] for r in first]
        if any(
            any(a[k] != b[k] for k in ("hops", "cluster", "weight"))
            for a, b in zip(first, second, strict=True)
        ):
            raise ValueError("registered comparisons need aligned panel weights and clusters")
        resamples, seed = registration["resamples"], registration["seed"]
        item = clustered_difference(
            [float(r["correct"]) for r in first],
            [float(r["correct"]) for r in second],
            first,
            resamples,
            seed,
            hypothesis={**comparison, "alpha": registration["alpha"]},
        )
        point, draws, _ = _paired_bootstrap(
            [r["total_tokens"] for r in first],
            [r["total_tokens"] for r in second],
            first,
            resamples,
            seed,
        )
        if point[1] <= 0 or np.any(draws[:, 1] <= 0):
            raise ValueError("registered token ratio needs positive reference token usage")
        low, high = np.quantile(draws[:, 0] / draws[:, 1], [0.025, 0.975])
        item.update(
            comparison=comparison,
            token_ratio={
                "ratio": float(point[0] / point[1]),
                "low": float(low),
                "high": float(high),
                "definition": "ratio of weighted total tokens",
            },
        )
        result[comparison["id"]] = item
    adjusted = holm_adjust([r["p_value"] for r in result.values()])
    for item, p in zip(result.values(), adjusted, strict=True):
        item.update(p_holm=p, passes=item["passes_unadjusted"] and p <= registration["alpha"])
    return result


def score_panel(records, agent, members, questions, answers, resamples=10_000, seed=17):
    """Native MINTQA correctness with population weights and complete policy usage."""
    rows, controls, asked = indexed(records), indexed(agent), indexed(questions)
    ids = [r["id"] for r in members]
    if len(set(ids)) != len(ids) or set(ids) != set(asked):
        raise ValueError("panel questions and unique member ids must agree")
    if (set(rows) | set(controls)) - set(ids):
        raise ValueError("output ids outside the frozen panel")
    if any(not math.isfinite(r["weight"]) or r["weight"] <= 0 for r in members):
        raise ValueError("panel weights must be finite and positive")
    texts = record_outputs((_reader_record(r) for r in records), ids)
    other = record_outputs(agent, ids)
    correct = {q: float(answer_contains(final_answer(texts[q]), answers[q])) for q in ids}
    reference = {q: float(answer_contains(final_answer(other[q]), answers[q])) for q in ids}
    costs = {q: usage(rows.get(q)) for q in ids}
    reference_costs = {q: usage(controls.get(q)) for q in ids}
    for cost in [*costs.values(), *reference_costs.values()]:
        cost["total_tokens"] = cost["prompt_tokens"] + cost["generated_tokens"]

    def table(subset):
        mass = sum(r["weight"] for r in subset)
        qids = [r["id"] for r in subset]
        return {
            "questions": len(subset),
            "weight_sum": mass,
            "missing_outputs": sum(q not in rows for q in qids),
            "accuracy": sum(correct[r["id"]] * r["weight"] for r in subset) / mass,
            "agent_accuracy": sum(reference[r["id"]] * r["weight"] for r in subset) / mass,
            "vs_agent": clustered_difference(
                [correct[q] for q in qids], [reference[q] for q in qids], subset, resamples, seed
            ),
            "cost": {
                "totals": {
                    k: sum(costs[q][k] for q in qids) for k in (*COST_FIELDS, "total_tokens")
                },
                "per_question": {
                    k: sum(costs[q][k] for q in qids) / len(qids)
                    for k in (*COST_FIELDS, "total_tokens")
                },
                "weighted_per_question": {
                    k: sum(costs[r["id"]][k] * r["weight"] for r in subset) / mass
                    for k in (*COST_FIELDS, "total_tokens")
                },
            },
            "agent_cost": {
                "totals": {
                    k: sum(reference_costs[q][k] for q in qids)
                    for k in (*COST_FIELDS, "total_tokens")
                },
                "per_question": {
                    k: sum(reference_costs[q][k] for q in qids) / len(qids)
                    for k in (*COST_FIELDS, "total_tokens")
                },
                "weighted_per_question": {
                    k: sum(reference_costs[r["id"]][k] * r["weight"] for r in subset) / mass
                    for k in (*COST_FIELDS, "total_tokens")
                },
            },
        }

    result = table(members)
    result["native_unweighted"] = paper_accuracy(texts, answers, questions)
    for field, key in (
        ("by_split", lambda r: r["id"].split("-")[1]),
        ("by_hop", lambda r: str(r["hops"])),
        ("by_group", lambda r: asked[r["id"]]["complexity"]),
    ):
        groups = defaultdict(list)
        for row in members:
            groups[key(row)].append(row)
        result[field] = {name: table(group) for name, group in sorted(groups.items())}
    result["per_question"] = [
        {
            **r,
            "correct": bool(correct[r["id"]]),
            "agent_correct": bool(reference[r["id"]]),
            **costs[r["id"]],
            "escalated": rows.get(r["id"], {}).get("escalated", False),
        }
        for r in members
    ]
    if any("escalated" in row for row in records):
        escalated = [r for r in members if rows.get(r["id"], {}).get("escalated")]
        result["escalation"] = {
            "questions": len(escalated),
            "rate": len(escalated) / len(ids),
            "weighted_rate": sum(r["weight"] for r in escalated) / result["weight_sum"],
        }
    return result


def pilot_timing_report(manifest, directory, families, arms):
    """Project a workload using the actual serial or batched pilot scheduling."""
    from xgraph.dog import reader_metadata

    if "pilot" not in manifest:
        raise ValueError("timings projection requires a frozen development pilot")
    members = manifest["panels"]["development"]["members"]
    full_counts = manifest["pilot"]["full_panel_by_hop"]
    if (
        set(full_counts) != {str(r["hops"]) for r in members}
        or any(type(n) is not int or n < 1 for n in full_counts.values())
        or sum(full_counts.values()) != manifest["pilot"]["full_panel_questions"]
    ):
        raise ValueError("pilot timing projection needs aligned full-panel hop counts")
    result = {
        "artifact_kind": "xgraph.mintqa_pilot_timings",
        "version": 1,
        "pilot": manifest["pilot"],
        "families": {},
        "timing_kind": "per-arm workload wall time including CPU orchestration with model loaded",
        "limitations": [
            "uncontrolled cache and shared GPU contention",
            "excludes CPU preparation",
            "projections require the same serving and batching; cache/occupancy can change",
            "small clustered pilot may not represent full-panel question costs",
        ],
    }

    def seconds(value, name):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid or missing {name} time")
        return value

    for family in families:
        result["families"][family] = {}
        for arm in arms:
            path = directory / family / "development" / f"{arm}.jsonl"
            records = read_jsonl(path)
            qualification = audit_complete_usage(records, {r["id"] for r in members})
            rows = indexed(records)
            summary = json.loads(Path(f"{path}.summary.json").read_text())
            if summary.get("requests") != len(members):
                raise ValueError(f"{family}/{arm}: incomplete timing summary")
            wall = seconds(summary.get("seconds"), "arm wall")
            startup = seconds(summary.get("startup_seconds"), "model startup")
            by_hop = defaultdict(list)
            measured = []
            serial = summary.get("question_timings") is True
            scheduling = "serial" if serial else summary.get("question_scheduling")
            if serial:
                for member in members:
                    row = rows[member["id"]]
                    elapsed = seconds(row.get("question_wall_seconds"), f"question {member['id']}")
                    by_hop[str(member["hops"])].append(elapsed)
                    measured.append(
                        {"id": member["id"], "hops": member["hops"], "seconds": elapsed}
                    )
                question_total = sum(r["seconds"] for r in measured)
                if wall + 1e-6 < question_total:
                    raise ValueError("arm wall time is shorter than its serial question times")
                workload = sum(
                    sum(times) / len(times) * full_counts[hop] for hop, times in by_hop.items()
                )
            else:
                if scheduling not in ("step_synchronous", "native_batched"):
                    raise ValueError(f"{family}/{arm}: missing serial or batched timing summary")
                batch = summary.get("question_batch_size")
                if type(batch) is not int or batch < 1:
                    raise ValueError("batched timing needs a positive question batch size")
                counts = Counter(str(m["hops"]) for m in members)
                ratios = {full_counts[h] / n for h, n in counts.items()}
                if len(ratios) != 1:
                    raise ValueError("aggregate batched timing cannot reweight a different hop mix")
                workload = wall * ratios.pop()
                question_total = None
            costs = combined_usage(*records)
            statuses = Counter(r.get("status", "unspecified") for r in records)
            failures = {
                "failed_questions": sum(
                    bool(
                        record_failure(_reader_record(r))
                        or r.get("status", "answered")
                        not in (
                            ("ok", "beam_shortfall") if r.get("arm") == "gcr" else ("answered",)
                        )
                        or final_answer(r.get("text", "")) in (None, "", "Unknown")
                    )
                    for r in records
                ),
                "context_limited": sum(
                    bool(r.get("context_limited") or r.get("status") == "context_limit")
                    for r in records
                ),
                "budget_spent": sum(
                    bool(r.get("budget_spent") or r.get("status") == "call_limit") for r in records
                ),
                "retrieval_truncated": qualification["retrieval_truncated"],
                "status_counts": dict(statuses),
            }
            result["families"][family][arm] = {
                **reader_metadata(records, "dog-gold" if arm == "dog-gold" else "dog"),
                "questions": len(members),
                "question_scheduling": scheduling,
                "question_batch_size": summary.get("question_batch_size", 1),
                "wall_seconds": wall,
                "startup_seconds": startup,
                "question_wall_seconds": question_total,
                **costs,
                "total_tokens": costs["prompt_tokens"] + costs["generated_tokens"],
                "failure_counts": failures,
                "per_question": measured,
                "by_hop": {
                    h: {
                        "questions": len(ts),
                        "mean_question_seconds": sum(ts) / len(ts),
                        "full_panel_questions": full_counts[h],
                    }
                    for h, ts in sorted(by_hop.items())
                },
                "projection": {
                    "workload_seconds": workload,
                    "startup_seconds": startup,
                    "full_panel_gpu_hours": (workload + startup) / 3600,
                },
            }
    result["projected_full_panel_gpu_hours"] = sum(
        a["projection"]["full_panel_gpu_hours"]
        for arms in result["families"].values()
        for a in arms.values()
    )
    return result


def terminal_replay(records, requests, members, answers, executor, *, width=8):
    """Fixed saved paths and gold-free displays; exposure is not model accuracy."""
    from xgraph.audit import CLOSE, COUNT, EXECUTED, JOIN, terminal_instruction, terminal_pieces

    terminal_instruction(width)
    if not width:
        raise ValueError("terminal replay needs a positive display width")
    index, tokenizer = executor._index, executor._tokenizer
    if index.sitelinks is None:
        raise ValueError("terminal replay needs sitelink counts")
    asked, saved, panel = indexed(requests), indexed(records), indexed(members)
    if set(panel) != set(asked) or set(saved) - set(panel) or set(panel) - set(answers):
        raise ValueError("terminal replay requests, members, outputs and answers disagree")
    if any(not math.isfinite(m["weight"]) or m["weight"] <= 0 for m in members):
        raise ValueError("terminal replay weights must be finite and positive")
    total = sum(m["weight"] for m in members)
    if not total:
        raise ValueError("terminal replay panel is empty")
    owners, first = np.unique(index.labels.owners, return_index=True)
    first_row = dict(zip(owners.tolist(), first.tolist(), strict=True))
    pieces = terminal_pieces(tokenizer, index.span_syntax)
    texts = record_outputs(records, list(panel))
    rows = []
    for qid in panel:
        record, request = saved.get(qid, {}), asked[qid]
        path = record_path(record)
        support = (
            executor.execute_set(
                *path, request.get("anchor"), record.get("kg_version", request.get("kg_version"))
            )
            if path
            else {"reached": [], "cap_events": []}
        )
        reached = support["reached"]
        # execute_set has already committed and capped answer rows. Re-sort only
        # the display, independently of the executor's start-rank configuration.
        version = record.get("kg_version", request.get("kg_version"))
        view = index if version is None else executor._view(version)
        ordered = sorted(
            reached,
            key=lambda e: (
                -int(index.sitelinks[index._entity_at[e]]),
                -len(view.edges_at(index._entity_at[e])[0]),
                first_row[index._entity_at[e]],
            ),
        )
        shown = ordered[:width] if len(ordered) > 1 else []
        tokens, labels = [], []
        if shown:
            tokens.extend(pieces[EXECUTED])
            for n, entity in enumerate(shown):
                row = index.labels.row_tokens(first_row[index._entity_at[entity]])
                if n:
                    tokens.extend(pieces[JOIN])
                tokens.extend(row)
                labels.append(
                    b"".join(tokenizer.token_bytes(t) for t in row)
                    .decode()
                    .removeprefix(index.span_syntax.label_prefix)
                )
            more = len(ordered) - len(shown)
            if more:
                tokens.extend(pieces[COUNT + more - 1])
            tokens.extend(pieces[CLOSE])
        wrong = not answer_contains(final_answer(texts[qid]), answers[qid])
        rows.append(
            {
                "id": qid,
                "weight": panel[qid]["weight"],
                "saved_failure": wrong,
                "deliverable": reached,
                "displayed": shown,
                "labels": labels,
                "display_tokens": len(tokens),
                "cap_events": support["cap_events"],
                "gold_supported_failure": wrong
                and bool(executor.named(answers[qid]) & set(reached)),
                "literal_label_exposed_failure": wrong
                and any(answer_contains(s, answers[qid]) for s in labels),
            }
        )

    def exposure(field):
        selected = [r for r in rows if r[field]]
        return {
            "n": len(selected),
            "population_points": 100 * sum(r["weight"] for r in selected) / total,
        }

    return {
        "kind": "display_oracle",
        "questions": len(panel),
        "missing_outputs": len(set(panel) - set(saved)),
        "width": width,
        "singleton_display": False,
        "ranking": (
            "sitelinks descending, versioned adjacency degree descending, "
            "first label-index row ascending"
        ),
        "interpretation": (
            "Gold scores exposure only. Saved paths stay fixed; "
            "no new answers or accuracy are predicted."
        ),
        "token_scope": (
            "Canonical additional display tokens only; all are generated tokens. "
            "Excludes changes to answers, prefixes, EOS, budgets or model decisions."
        ),
        "gold_supported_failures": exposure("gold_supported_failure"),
        "literal_label_exposed_failures": exposure("literal_label_exposed_failure"),
        "weighted_display_tokens_per_question": sum(r["weight"] * r["display_tokens"] for r in rows)
        / total,
        "display_tokens_per_question": sum(r["display_tokens"] for r in rows) / len(panel),
        "rows": rows,
    }


def _executor(args):
    from xgraph.index import load_kg_index
    from xgraph.kgversions import load_kg_versions
    from xgraph.programs import MatchedExecutor
    from xgraph.tokenizer import ServedTokenizer

    index = load_kg_index(args.index)
    versions = getattr(args, "kg_versions", None)
    return MatchedExecutor(
        index,
        ServedTokenizer.from_file(args.tokenizer),
        **({"versions": load_kg_versions(versions, index)} if versions else {}),
        start_rank="sitelinks",
        unquoted_starts=True,
    )


def support_records(records, requests, executor):
    asked = indexed(requests)
    result = []
    for row in records:
        path = record_path(row)
        request = asked[row["id"]]
        support = (
            executor.execute_set(*path, request.get("anchor"), request.get("kg_version"))
            if path
            else {"reached": [], "cap_events": [], "uncapped_reached": []}
        )
        answer = final_answer(row.get("text", ""))
        support["answer_in_support"] = bool(
            answer and executor.named(answer) & set(support["reached"])
        )
        result.append({"id": row["id"], **support})
    return result


def main(argv=None):
    from xgraph.dog import reader_metadata

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    timings = commands.add_parser("timings")
    timings.add_argument("--panels", type=Path, required=True)
    timings.add_argument("--directory", type=Path, required=True)
    timings.add_argument("--family", action="append", required=True)
    timings.add_argument("--arm", action="append", required=True)
    timings.add_argument("--out", type=Path, required=True)
    for name in (
        "source",
        "open",
        "restore",
        "support",
        "route",
        "revision",
        "terminal-replay",
        "audit",
        "audit-witness",
        "cascade",
        "revise",
        "account",
    ):
        cmd = commands.add_parser(name)
        cmd.add_argument("--out", type=Path, required=True)
        cmd.add_argument("--requests", type=Path)
        cmd.add_argument("--outputs", type=Path)
        cmd.add_argument("--second", type=Path)
        cmd.add_argument("--support", type=Path)
        cmd.add_argument("--candidates", type=Path)
        cmd.add_argument("--index", type=Path)
        cmd.add_argument("--tokenizer", type=Path)
        cmd.add_argument("--db", type=Path)
        cmd.add_argument("--source", type=Path, action="append", default=[])
        if name in {"audit", "audit-witness"}:
            cmd.add_argument("--menu-size", type=int, default=8)
        if name == "audit":
            cmd.add_argument("--no-menus", action="store_true")
            cmd.add_argument("--menu-source", choices=("local", "global"))
        if name in {"audit", "revision"}:
            cmd.add_argument("--observe-terminal", type=int, nargs="?", const=8, default=0)
        if name == "terminal-replay":
            cmd.add_argument("--width", type=int, default=8)
            cmd.add_argument("--panels", type=Path, required=True)
            cmd.add_argument("--data", type=Path, required=True)
            cmd.add_argument("--kg-versions", type=Path)
        if name == "audit-witness":
            cmd.add_argument("--endpoints", type=int, default=12)
            cmd.add_argument("--alternatives", type=int, default=4)
        if name == "revise":
            cmd.add_argument("--first-source", default="paths-allfix")
            cmd.add_argument("--second-source", default="revision")
    score = commands.add_parser("score")
    score.add_argument("--panels", type=Path, required=True)
    score.add_argument("--family", required=True)
    score.add_argument("--directory", type=Path, required=True)
    score.add_argument(
        "--agent-directory",
        type=Path,
        help="cached agent family directory, with panel subdirectories; defaults to --directory",
    )
    score.add_argument("--data", type=Path, required=True)
    score.add_argument("--arm", action="append", required=True)
    score.add_argument(
        "--panel", choices=("development", "validation"), action="append", required=True
    )
    score.add_argument("--resamples", type=int, default=10_000)
    score.add_argument(
        "--require-complete",
        action="store_true",
        help="require complete panel/reference and audit all per-call usage",
    )
    score.add_argument("--seed", type=int, default=17)
    score.add_argument("--out", type=Path, required=True)
    score.add_argument(
        "--freeze-manifest",
        type=Path,
        help="enforce the frozen validation registration and provenance",
    )
    args = parser.parse_args(argv)
    if args.command == "timings":
        manifest = json.loads(args.panels.read_text())
        result = pilot_timing_report(manifest, args.directory, args.family, args.arm)
        inputs = [args.panels]
        for family in args.family:
            for arm in args.arm:
                path = args.directory / family / "development" / f"{arm}.jsonl"
                inputs.extend(
                    p
                    for p in (path, Path(f"{path}.summary.json"), Path(f"{path}.provenance.json"))
                    if p.is_file()
                )
        result["provenance"] = provenance(inputs)
        write_json(args.out, result)
        return
    required = {
        "source": ("requests", "outputs"),
        "open": ("requests",),
        "restore": ("outputs",),
        "support": ("requests", "outputs", "index", "tokenizer"),
        "route": ("requests", "outputs", "support"),
        "revision": ("requests", "outputs", "candidates", "index", "tokenizer", "db"),
        "terminal-replay": ("requests", "outputs", "index", "tokenizer"),
        "audit": ("requests", "outputs", "index", "tokenizer", "db"),
        "audit-witness": ("requests", "outputs", "candidates", "index", "tokenizer", "db"),
        "cascade": ("requests", "outputs", "second", "support"),
        "revise": ("requests", "outputs", "second", "support"),
        "account": ("outputs", "source"),
    }
    for name in required.get(args.command, ()):
        if not getattr(args, name):
            parser.error(f"{args.command} requires --{name}")
    if args.command == "audit-witness" and (
        args.menu_size < 1 or not 1 <= args.endpoints <= 20 or not 0 <= args.alternatives <= 8
    ):
        parser.error(
            "witness audit needs positive menus, endpoints in [1, 20], alternatives in [0, 8]"
        )
    inputs = []
    if args.command == "score":
        frozen = None
        if args.freeze_manifest is not None:
            from xgraph.mintqa_validation import validate_score_args

            frozen = validate_score_args(args)
        manifest = json.loads(args.panels.read_text())
        splits = [args.data / f"questions/{name}.jsonl" for name in ("ti", "pop")]
        data = dataset_rows(
            [f"{name}={path}" for name, path in zip(("ti", "pop"), splits, strict=True)]
        )
        answers = {q: row["answer"] for q, row in data.items()}
        result = {
            "artifact_kind": "xgraph.mintqa_rematch",
            "version": 1,
            "family": args.family,
            "panels": {},
            "missing_population_groups": {},
            "population_by_group": manifest["population_by_group"],
        }
        if "pilot" in manifest:
            if args.panel != ["development"] or args.freeze_manifest is not None:
                parser.error("pilot scores are restricted to development")
            result["pilot"] = manifest["pilot"]
        inputs = [args.panels, *splits]
        for panel in args.panel:
            directory = args.directory / panel
            question_path = args.panels.parent / panel / "questions.json"
            questions = json.loads(question_path.read_text())
            agent_path = (args.agent_directory or args.directory) / panel / "agent-strong-8.jsonl"
            agent = read_jsonl(agent_path)
            if "pilot" in manifest:
                agent_rows = indexed(agent)  # Still reject duplicate ids in the cached reference.
                agent = [
                    agent_rows[r["id"]]
                    for r in manifest["panels"][panel]["members"]
                    if r["id"] in agent_rows
                ]
            if args.require_complete:
                expected = {r["id"] for r in manifest["panels"][panel]["members"]}
                audit_complete_usage(agent, expected)
            inputs.extend([question_path, agent_path])
            result["panels"][panel] = {}
            missing = sorted(
                set(manifest["population_by_group"]) - {q["complexity"] for q in questions}
            )
            result["missing_population_groups"][panel] = missing
            for arm in args.arm:
                path = directory / f"{arm}.jsonl"
                inputs.append(path)
                rows = read_jsonl(path)
                qualification = (
                    audit_complete_usage(rows, expected) if args.require_complete else None
                )
                result["panels"][panel][arm] = score_panel(
                    rows,
                    agent,
                    manifest["panels"][panel]["members"],
                    questions,
                    answers,
                    args.resamples,
                    args.seed,
                )
                result["panels"][panel][arm].update(
                    reader_metadata(rows, "dog-gold" if arm == "dog-gold" else "dog")
                )
                if qualification is not None:
                    result["panels"][panel][arm]["qualification"] = qualification
                    for source in (path, agent_path):
                        inputs.extend(
                            p
                            for p in (
                                Path(f"{source}.provenance.json"),
                                Path(f"{source}.summary.json"),
                            )
                            if p.is_file()
                        )
                table = result["panels"][panel][arm]["by_group"]
                for group, count in manifest["population_by_group"].items():
                    table.setdefault(
                        group, {"questions": 0, "accuracy": None, "vs_agent": None, "weight_sum": 0}
                    )
                    table[group]["population_questions"] = count
        if frozen is not None:
            from xgraph.mintqa_validation import load_frozen
            from xgraph.provenance import sha256_file

            result["registration"] = frozen["registration"]
            result["registered_comparisons"] = registered_comparisons(
                result["panels"]["validation"], frozen["registration"]
            )
            if "amendment" in frozen:
                result["amended_comparisons"] = registered_comparisons(
                    result["panels"]["validation"],
                    {**frozen["registration"], "comparisons": frozen["amendment"]["comparisons"]},
                    family=args.family,
                )
            if "secondary_cost_points" in frozen:
                result["secondary_cost_points"] = frozen["secondary_cost_points"]
                result["secondary_comparisons"] = registered_comparisons(
                    result["panels"]["validation"],
                    {
                        **frozen["registration"],
                        "comparisons": frozen["secondary_cost_points"]["comparisons"],
                    },
                    family=args.family,
                )
            result["freeze_sha256"] = sha256_file(args.freeze_manifest)
            inputs.append(args.freeze_manifest)
            load_frozen(args.freeze_manifest)
        result["provenance"] = provenance(list(dict.fromkeys(inputs)))
        write_json(args.out, result)
        return
    inputs = [
        p
        for p in (
            args.requests,
            args.outputs,
            args.second,
            args.support,
            args.candidates,
            args.tokenizer,
            args.db,
        )
        if p
    ] + args.source
    requests = read_jsonl(args.requests) if args.requests else []
    outputs = read_jsonl(args.outputs) if args.outputs else []
    second = read_jsonl(args.second) if args.second else []
    support = read_jsonl(args.support) if args.support else []
    ids = [r["id"] for r in requests]
    if args.command == "terminal-replay":
        from xgraph.cost import StageTimer

        timer = StageTimer()
        executor = _executor(args)
        manifest = json.loads(args.panels.read_text())
        splits = [args.data / "questions" / f"{s}.jsonl" for s in ("ti", "pop")]
        data = dataset_rows([f"{s}={p}" for s, p in zip(("ti", "pop"), splits, strict=True)])
        result = terminal_replay(
            outputs,
            requests,
            manifest["panels"]["development"]["members"],
            {q: r["answer"] for q, r in data.items()},
            executor,
            width=args.width,
        )
        inputs += [
            args.panels,
            *splits,
            *sorted(args.index.glob("*.npy")),
            args.index / "kg-index.json",
        ]
        result["artifact_kind"] = "xgraph.mintqa_terminal_replay"
        result["panel_seed"] = manifest.get("settings", {}).get("seed", manifest.get("seed"))
        if args.kg_versions:
            from xgraph.kgversions import version_files

            inputs += version_files(args.kg_versions)
        result.update(timer.summary())
        result["provenance"] = provenance(inputs)
        write_json(args.out, result)
        return
    if args.command == "source":
        if set(indexed(outputs)) != set(indexed(requests)):
            raise ValueError("first-stage source must cover exactly the request panel ids")
        outputs = [dict(row) for row in outputs]
        for row in outputs:
            if (
                row.get("arm") in ("baseline", "constrained")
                and "usage" not in row
                and "model_calls" not in row
                and not any(k in row for k in ("rounds", "calls", "turns", "thought", "stages"))
                and isinstance(row.get("token_ids"), list)
                and row.get("generated_tokens", len(row["token_ids"])) == len(row["token_ids"])
            ):
                # generate() writes one native completion per record. Thought,
                # fallback and multicall records must supply their own counts.
                row["model_calls"] = 1
            if row.get("arm") == "agent" and "usage" not in row:
                if "model_calls" not in row:
                    if "rounds" in row:
                        row["model_calls"] = row["rounds"]
                    elif isinstance(row.get("turns"), list):
                        row["model_calls"] = len(row["turns"])
                    else:
                        raise ValueError(f"{row['id']}: missing explicit model-call accounting")
                if "rounds" in row and row["rounds"] != row["model_calls"]:
                    raise ValueError(f"{row['id']}: inconsistent model-call accounting")
            if "usage" not in row and not (
                "prompt_tokens" in row and ("generated_tokens" in row or "token_ids" in row)
            ):
                raise ValueError(f"{row['id']}: missing first-call token usage")
            try:
                usage(row)  # Reject incomplete costs before scheduling a second call.
            except KeyError as exc:
                raise ValueError(
                    f"{row['id']}: missing first-call usage field {exc.args[0]}"
                ) from exc
            if row.get("arm") == "agent" and "usage" not in row:
                audit_complete_usage([row], {row["id"]})
        rows = outputs
    elif args.command == "open":
        from xgraph.generate import opened_requests

        rows = opened_requests(requests, "⟦")
    elif args.command == "restore":
        rows = [{**r, "text": "⟦" + r["text"]} for r in outputs]
    elif args.command == "support":
        rows = support_records(outputs, requests, _executor(args))
    elif args.command == "route":
        selected = set(escalation_ids(outputs, support, ids))
        rows = [r for r in requests if r["id"] in selected]
    elif args.command == "revision":
        from xgraph.toolkg import KGStore

        store = KGStore(args.db)
        rows = revision_requests(
            requests,
            outputs,
            read_jsonl(args.candidates),
            _executor(args),
            store.label,
            observe_terminal=args.observe_terminal,
        )
    elif args.command == "audit":
        from xgraph.toolkg import KGStore

        if args.menu_size < 1:
            parser.error("audit --menu-size must be positive")
        store = KGStore(args.db)
        executor = _executor(args)
        rows = audit_requests(
            requests,
            outputs,
            executor,
            store.label,
            AuditMenus(executor._index, executor._tokenizer),
            executor._tokenizer,
            menu_size=args.menu_size,
            show_menus=not args.no_menus,
            observe_terminal=args.observe_terminal,
            menu_source=args.menu_source,
        )
    elif args.command == "audit-witness":
        from xgraph.toolkg import KGStore

        store = KGStore(args.db)
        executor = _executor(args)
        rows = audit_witness_requests(
            requests,
            outputs,
            read_jsonl(args.candidates),
            executor,
            store.label,
            AuditMenus(executor._index, executor._tokenizer),
            executor._tokenizer,
            menu_size=args.menu_size,
            endpoints=args.endpoints,
            alternatives=args.alternatives,
        )
    elif args.command == "cascade":
        rows = cascade_records(outputs, second, support, ids)
    elif args.command == "revise":
        rows = revision_records(
            outputs,
            second,
            support,
            ids,
            first_source=args.first_source,
            second_source=args.second_source,
        )
    else:
        sources = [indexed(read_jsonl(p)) for p in args.source]
        if {r["id"] for r in outputs} - set(sources[0]):
            raise ValueError("account needs the first-stage source for every selected output")
        rows = [
            {
                **r,
                "usage": combined_usage(*(s.get(r["id"]) for s in sources)),
                "cost_sources": [str(p) for p in args.source],
            }
            for r in outputs
        ]
    if args.index:
        inputs.extend(sorted(args.index.glob("*")))
    write_jsonl(args.out, rows)
    write_json(
        Path(f"{args.out}.summary.json"),
        {
            "artifact_kind": f"xgraph.mintqa_rematch.{args.command}",
            "version": 1,
            "records": len(rows),
            "provenance": provenance(inputs),
        },
    )


if __name__ == "__main__":
    main()
