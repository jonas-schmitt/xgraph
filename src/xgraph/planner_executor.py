"""Unconstrained complete-path planning with matched CPU execution and revision."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_PLAN = re.compile(
    r"(?:Path:\s*)?⟦([^⟦⟧\n]+)⟧\.?\s*(?:Answer:\s*⟦[^⟦⟧]*⟧\.?)?",
    re.DOTALL,
)


def parse_plan(text: str):
    """Read one complete relation path, without live observations or selections."""
    match = _PLAN.fullmatch(text.strip())
    if not match or "→" in match[1]:
        return None
    parts = [part.strip() for part in match[1].split("·")]
    return (parts[0], parts[1:]) if len(parts) > 1 and all(parts) else None


def _bound_answer(answer, path, request, support, executor):
    if not answer or not executor.named(answer) & set(support["reached"]):
        return False
    if not any(e["stage"] == "answer_label_rows" for e in support["cap_events"]):
        return True
    from xgraph.oracle import MAX_REACHED

    # Reuse the frozen walk's row provenance. Entity membership alone would admit
    # an alias whose own label row was removed by the answer-row cap.
    version = request.get("kg_version")
    view = executor._index if version is None else executor._view(version)
    ranks, rows_of, _ = executor._walk(view, *path, request.get("anchor"), False)
    best = min(ranks.values())
    rows = sorted({row for e in ranks if ranks[e] == best for row in rows_of[e]})[:MAX_REACHED]
    return bool(set(executor._rows(executor._index.labels, answer)) & set(rows))


def execute_records(records, requests, executor, store, stores=None, *, stage="first"):
    """Execute a complete generated batch; no model or live feedback is involved."""
    from xgraph.mintqa_panels import indexed
    from xgraph.prompts import final_answer
    from xgraph.repair import plan_text
    from xgraph.score import record_failure
    from xgraph.toolkg import store_for
    from xgraph.webqsp import deliver_records

    if stage not in {"first", "second"}:
        raise ValueError(f"unknown planner stage: {stage!r}")
    raw = indexed(records)
    if set(raw) != {r["id"] for r in requests}:
        raise ValueError("planner outputs must cover exactly the requested panel")
    result = []
    for request in requests:
        source = raw[request["id"]]
        path = parse_plan(source.get("text", ""))
        error = None
        if record_failure(source):
            error = "generation_failure"
        elif path is None:
            error = "invalid_path"
        elif request.get("anchor") and not executor.named(path[0]) & set(request["anchor"]):
            error = "start_outside_anchor"
        execution = None
        support = None
        if not error:
            support = executor.execute_set(*path, request.get("anchor"), request.get("kg_version"))
            execution = executor.execute(*path, request.get("anchor"), request.get("kg_version"))
            if not execution.reached:
                error = "empty_path_support"
            elif execution.plan is None:
                error = "unwritable_path"
        answer = final_answer(source["text"])
        if execution and stage == "first" and execution.answer:
            answer = store_for(request, store, stores).label(int(execution.answer[1:]))
        if (
            not error
            and stage == "second"
            and not _bound_answer(answer, path, request, support, executor)
        ):
            error = "answer_outside_support"
        row = {
            **source,
            "arm": "planner-executor",
            "planner_executor_version": "planner-executor-v1",
            "planner_stage": stage,
            "raw_planner_reply": source.get("text", ""),
            "cap_events": support["cap_events"] if support else [],
            "plan": execution.recorded_plan if execution and not error else None,
            "text": (
                plan_text(execution.plan[0], list(execution.plan[1:])) + f".\nAnswer: ⟦ {answer}⟧"
                if execution and not error
                else ""
            ),
            **({"error": error} if error else {}),
        }
        delivered = deliver_records(
            [row], [request], executor, store, stores, complete_path_set=True
        )[0]
        result.append(delivered)
    return result


def revision_requests(
    requests, first, executor, store, tokenizer, *, relation_rows=(), stores=None
):
    """WT's endpoint feedback and executed alternative shortlist without terminal readback."""
    from xgraph.path_policy import second_call_requests

    # The terminal variant adds only a live-readback instruction and metadata.
    # The endpoint feedback and alternative construction are identical here.
    clean = []
    for request in requests:
        messages = [dict(m) for m in request["messages"]]
        if "planner_question_text" in request:
            messages[-1]["content"] = request["planner_question_text"]
        elif "planner_start_menus" in request:
            messages[-1]["content"] = messages[-1]["content"].rsplit(
                "\nStarting relation suggestions: ", 1
            )[0]
        clean.append({**request, "messages": messages})
    result = second_call_requests(
        "two-pass",
        clean,
        first,
        executor,
        store,
        tokenizer,
        relation_rows=relation_rows,
        stores=stores,
    )
    for request, revision in zip(requests, result, strict=True):
        # Preserve P's actual prompt history; only the feedback question uses
        # WT's original question text, without duplicating the static menu.
        revision["messages"] = [*request["messages"], *revision["messages"][-2:]]
    return result


def select_records(first, second, requests, executor):
    """Accept a usable bound revision, otherwise retain the first; charge both calls."""
    from xgraph.path_policy import select_second_call

    return select_second_call(
        first, second, requests, executor, first_source="P-first", second_source="P-second"
    )


def first_requests(requests, executor, store):
    """Render static starting menus from WT-D ranks, retaining demonstration questions."""
    from xgraph.audit import RelationScores, checked_relation_ranks
    from xgraph.programs import record_path
    from xgraph.repair import plan_text

    terms = RelationScores(executor._index, executor._tokenizer)
    result = []
    instruction = (
        "Answer using the supplied Freebase question entities and exact relation ids. "
        "Write just one complete path as ⟦ start entity · relation.id · next.relation.id⟧. "
        "Use at least one and at most eight directed physical hops. CVT/mediator nodes "
        "require both relations and count as two hops. No inverse edges have been added. "
        "The starting relation suggestions use the same saved dense ranks as WT-D; "
        "any writable relation may be used, including relations absent from the suggestions. "
        "Write the entire path in one reply. There are no observations or selections during "
        "the path. The executor delivers its complete capped answer set, ranking its "
        "best-connected endpoint first. On revision, write one complete path and "
        "Answer: ⟦ label⟧; that label must belong to the executed endpoint support. "
        "All other endpoints are delivered too. Keep names intact, including commas."
    )
    for request in requests:
        anchor = request.get("anchor")
        if not anchor or request.get("answer_mode") != "set":
            raise ValueError(f"{request['id']}: planner needs supplied starts and set delivery")
        ranks = dict(checked_relation_ranks(request.get("relation_ranks"), terms.owner_at))
        question = next(m["content"] for m in reversed(request["messages"]) if m["role"] == "user")
        priorities = terms.alias_ranks(question)
        menus = []
        for start in dict.fromkeys(store.label(int(e[1:])) for e in anchor):
            available = executor.frontier_relations(start, [], anchor, request.get("kg_version"))
            available.sort(key=lambda r: (ranks.get(r[0], 64), -r[1], terms.owner_at[r[0]]))
            labels = [terms.names[terms.display_row(prop, priorities)] for prop, _ in available]
            menus.append(
                {
                    "start": start,
                    "anchor_ids": [e for e in anchor if e in executor.named(start)],
                    "relations": labels[:5],
                    "omitted_relations": max(0, len(labels) - 5),
                }
            )
        messages = []
        for message in request["messages"]:
            if message["role"] == "system":
                messages.append({"role": "system", "content": instruction})
            elif message["role"] == "assistant":
                path = record_path({"text": message["content"]})
                if path is None:
                    raise ValueError(f"{request['id']}: incomplete planner demonstration")
                messages.append({"role": "assistant", "content": plan_text(*path)})
            else:
                messages.append(dict(message))
        messages[-1]["content"] += "\nStarting relation suggestions: " + json.dumps(
            menus, ensure_ascii=False
        )
        result.append({**request, "messages": messages, "planner_start_menus": menus})
    return result


def read_jsonl(path):
    """Records of a JSONL file, split only at LF: str.splitlines would also split at
    U+2028, U+0085 and other separators that are legal inside JSON strings."""
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def main(argv=None):
    """CPU stages around two ordinary baseline generation writer invocations."""
    from xgraph.index import load_kg_index
    from xgraph.kgversions import load_kg_versions
    from xgraph.programs import MatchedExecutor
    from xgraph.tokenizer import ServedTokenizer
    from xgraph.toolkg import KGStore, version_stores

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute", "revision", "select"))
    for flag in ("requests", "data", "index", "tokenizer", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--outputs", type=Path)
    parser.add_argument("--second", type=Path)
    parser.add_argument("--stage", choices=("first", "second"), default="first")
    args = parser.parse_args(argv)

    if args.out.exists():
        raise ValueError(f"refusing to replace planner artifact: {args.out}")
    requests = read_jsonl(args.requests)
    saved, tok = load_kg_index(args.index), ServedTokenizer.from_file(args.tokenizer)
    executor = MatchedExecutor(
        saved,
        tok,
        versions=load_kg_versions(args.data / "versions", saved),
        start_rank="sitelinks",
        unquoted_starts=True,
    )
    store = KGStore(args.data / "kg.sqlite")
    stores = version_stores(store, args.data / "versions", {r["kg_version"] for r in requests})
    if args.action == "prepare":
        rows = first_requests(requests, executor, store)
    elif args.outputs is None:
        parser.error("--outputs is required for this action")
    elif args.action == "execute":
        rows = execute_records(
            read_jsonl(args.outputs), requests, executor, store, stores, stage=args.stage
        )
    elif args.action == "revision":
        rows = revision_requests(
            requests,
            read_jsonl(args.outputs),
            executor,
            store,
            tok,
            relation_rows=read_jsonl(args.data / "kg/relation_labels.jsonl"),
            stores=stores,
        )
    elif args.second is None:
        parser.error("--second is required for selection")
    else:
        rows = select_records(read_jsonl(args.outputs), read_jsonl(args.second), requests, executor)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
