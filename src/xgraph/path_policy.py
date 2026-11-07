"""Benchmark-neutral access to the frozen rematch second-call policies.

Only requests, generated plans and KG execution enter these builders. Benchmark
gold, population membership and scoring stay with the caller. MINTQA's existing
entry points and their byte rendering remain unchanged.
"""

from __future__ import annotations

from xgraph.mintqa_rematch import (
    AuditMenus,
    audit_requests,
    revision_records,
    revision_requests,
    support_records,
)
from xgraph.programs import record_path


def second_call_requests(
    kind,
    requests,
    records,
    executor,
    store,
    tokenizer,
    *,
    relation_rows=(),
    stores=None,
    terminal_width=8,
):
    """Frozen audit (menu eight) or unrestricted two-pass revision (eight candidates)."""
    terminal = kind.endswith("-terminal")
    kind = kind.removesuffix("-terminal")
    extra = {"observe_terminal": terminal_width} if terminal else {}
    if kind == "audit":
        return audit_requests(
            requests,
            records,
            executor,
            store.label,
            AuditMenus(executor._index, tokenizer),
            tokenizer,
            menu_size=8,
            **extra,
        )
    if kind != "two-pass":
        raise ValueError(f"unknown second-call policy {kind!r}")
    from xgraph.repair import Repairer

    first = {r["id"]: r for r in records}
    repairer = Repairer(executor, store, relation_rows, edits=4, stores=stores)
    candidates = []
    for request in requests:
        record = first[request["id"]]
        path = record_path(record)
        row = {
            "id": request["id"],
            "start": path[0] if path else None,
            "question": request.get("question", ""),
            "kg_version": request.get("kg_version"),
            "candidates": [],
        }
        if path:
            row["candidates"], _ = repairer.candidates(
                row["question"], *path, request.get("anchor"), request.get("kg_version")
            )
        candidates.append(row)
    return revision_requests(requests, records, candidates, executor, store.label, **extra)


def select_second_call(first, second, requests, executor, *, first_source, second_source):
    """Original frozen acceptance: usable bound revision, otherwise first; charge both."""
    return revision_records(
        first,
        second,
        support_records(second, requests, executor),
        [r["id"] for r in requests],
        first_source=first_source,
        second_source=second_source,
    )
