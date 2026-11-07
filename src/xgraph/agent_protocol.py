"""Opt-in agent protocols beside the byte-pinned legacy controller."""

from __future__ import annotations

from xgraph.agent_answers import VERSIONS, answer_fields


def run_agent(
    *args,
    answer_parser_version: str | None = None,
    final_answer_controller: str | None = None,
    **kwargs,
) -> list[dict]:
    """Select parser and controller independently, preserving the default writer."""
    from xgraph.agent import run_agent as legacy_agent
    from xgraph.agent_final import VERSION
    from xgraph.agent_final import run_agent as retry_agent

    if answer_parser_version is not None and answer_parser_version not in VERSIONS:
        raise ValueError(f"unknown agent answer parser version: {answer_parser_version!r}")
    if final_answer_controller not in (None, VERSION):
        raise ValueError(f"unknown agent final-answer controller: {final_answer_controller!r}")
    controller = retry_agent if final_answer_controller else legacy_agent
    records = controller(*args, **kwargs)
    if answer_parser_version is not None:
        for record in records:
            record.update(answer_fields(record, answer_parser_version))
    return records
