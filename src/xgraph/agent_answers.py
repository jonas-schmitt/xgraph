"""Versioned agent final-answer parsing; the frozen parser remains in prompts.py."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

V1 = "agent-answer-v1"
V2 = "agent-answer-v2"
VERSIONS = (V1, V2)


@dataclass(frozen=True)
class ParsedAnswer:
    """An ordered answer set and its syntactic acceptance or rejection status."""

    answers: list[str]
    status: str


def parse_agent_answer(text: str, version: str = V1, *, bare: bool = True) -> ParsedAnswer:
    """Parse only a final physical reply, retaining a status for empty/rejected sets.

    V2 accepts JSON string arrays, an object with an ``answers`` array, and the
    documented one-label-per-line format. It flattens one array level explicitly;
    bracketed unquoted comma lists are rejected because labels can contain commas.
    """
    from xgraph.agent import parse_action
    from xgraph.prompts import parse_answer_set

    if version not in VERSIONS:
        raise ValueError(f"unknown agent answer parser version: {version!r}")
    if version == V1:
        return ParsedAnswer(parse_answer_set(text, bare=bare), "legacy")
    if parse_action(text) is not None:
        return ParsedAnswer([], "terminal_tool_call")
    text = text.strip()
    markers = list(re.finditer(r"^\s*(?:#\s*)?Answers?:[ \t]*", text, re.M | re.I))
    if markers:
        text = text[markers[-1].end() :].strip()
    elif not bare and not text.startswith(("[", "{", "```")):
        return ParsedAnswer([], "missing_answer_marker")
    if text.startswith("```"):
        fence = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\s*```", text, re.S)
        if fence is None:
            return ParsedAnswer([], "invalid_json")
        text = fence[1].strip()
    status = "ok"
    if text.startswith(("[", "{")):
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            inner = text[1:-1]
            unquoted = (
                text.startswith("[")
                and text.endswith("]")
                and "," in inner
                and not any(c in inner for c in "[]{}\"'\n")
            )
            return ParsedAnswer([], "rejected_unquoted_list" if unquoted else "invalid_json")
        if isinstance(value, dict):
            value = value.get("answers")
        if not isinstance(value, list):
            return ParsedAnswer([], "invalid_answer_type")
        if not value:
            return ParsedAnswer([], "explicit_empty")
        values = []
        for member in value:
            if isinstance(member, list):
                values.extend(member)
                status = "recovered_nested_array"
            else:
                values.append(member)
        if any(not isinstance(v, str) for v in values):
            return ParsedAnswer([], "invalid_answer_type")
    else:
        # Remove a single outer span before splitting the documented line format.
        if text.startswith("⟦") and text.endswith("⟧") and text.count("⟦") == text.count("⟧") == 1:
            text = text[1:-1]
        values = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith(("Action:", "Next:", "Path:", "Rationale:")):
                break
            spans = re.findall(r"⟦\s*([^⟧]*)⟧", line)
            values.extend(spans or [re.sub(r"^(?:[-*]\s+|\d+[.)]\s+)", "", line)])
    answers = []
    for value in values:
        value = value.replace("⟦", "").replace("⟧", "").strip()
        if (
            value
            and value.casefold() != "unknown"
            and not value.startswith("⁇")
            and value not in answers
        ):
            answers.append(value)
    return ParsedAnswer(answers, status if answers else "empty")


def record_parse(record: dict, version: str) -> ParsedAnswer:
    """Keep generation failures and truncation separate from syntactic recovery."""
    if version not in VERSIONS:
        raise ValueError(f"unknown agent answer parser version: {version!r}")
    raw = record.get("raw_final_reply")
    if not isinstance(raw, str):
        raise ValueError("versioned agent record needs a string raw_final_reply")
    if version == V2:
        if record.get("finish_reason") == "context_limit":
            return ParsedAnswer([], "context_limit")
        if (
            record.get("error")
            or record.get("declined")
            or record.get("finish_reason")
            in {
                "error",
                "abort",
                "declined",
            }
        ):
            return ParsedAnswer([], "generation_failure")
        if record.get("truncated") or record.get("finish_reason") in {"length", "max_tokens"}:
            return ParsedAnswer([], "truncated_final")
    return parse_agent_answer(raw, version)


def answer_fields(record: dict, version: str) -> dict:
    """Fields added only by an explicitly selected versioned agent writer."""
    from xgraph.prompts import parse_answer_set

    turns = record.get("turns") or []
    raw = turns[-1]["text"] if turns else ""
    # Explicit v1 names the legacy writer; it must not add a set to singleton
    # records or reparse an earlier reply after a context-limit refusal.
    if version == V1:
        return {
            "answer_parser_version": version,
            "raw_final_reply": raw,
            "answer_parse_status": "legacy",
        }
    parsed = record_parse({**record, "raw_final_reply": raw}, version)
    return {
        "answer_parser_version": version,
        "raw_final_reply": raw,
        "answer_parse_status": parsed.status,
        "answers": parsed.answers,
        "legacy_answers": record.get("answers", parse_answer_set(raw, bare=True)),
    }


def recorded_answers(record: dict) -> list[str] | None:
    """Dispatch recorded versions; None leaves all legacy scoring rules intact."""
    version = record.get("answer_parser_version", V1)
    if version not in VERSIONS:
        raise ValueError(f"unknown agent answer parser version: {version!r}")
    return record_parse(record, version).answers if version == V2 else None
