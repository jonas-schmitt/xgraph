"""Reproducible descriptive MINTQA errors from completed, read-only outputs."""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
import random
import re
import resource
import shlex
import sqlite3
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
FAMILIES = ("qwen38", "gemma31")
ARMS = ("WT", "W", "B2", "agent")
AGENTS = {"qwen38": "agent-strong-8", "gemma31": "agent-feedback-8"}
DEV_AGENTS = {"qwen38": "agent8", "gemma31": "feedback8"}
SEED = 17
RESAMPLES = 10_000


class Inputs:
    """Hash streamed inputs, refuse concurrent changes, and never copy a KG."""

    def __init__(self):
        self.files = {}

    def digest(self, path):
        path = Path(path).resolve()
        if str(path) not in self.files:
            before = path.stat()
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"input changed during hashing: {path}")
            self.files[str(path)] = {
                "path": str(path),
                "bytes": after.st_size,
                "sha256": hasher.hexdigest(),
            }
        return self.files[str(path)]

    def read_json(self, path):
        self.digest(path)
        return json.loads(Path(path).read_text())

    def records(self, path, ids):
        self.digest(path)
        rows = {}
        with Path(path).open() as stream:
            for number, line in enumerate(stream, 1):
                row = json.loads(line)
                if row["id"] in rows or row["id"] not in ids:
                    raise ValueError(f"unexpected/duplicate id: {path}:{number}")
                rows[row["id"]] = {**row, "_source": f"{path}:{number}"}
        if rows.keys() != set(ids):
            raise ValueError(f"incomplete records: {path}")
        return rows


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def git(*arguments, cwd=WORKTREE):
    return subprocess.check_output(
        ["git", "--no-optional-locks", *arguments], cwd=cwd, text=True
    ).strip()


def source_gold(root, members, inputs):
    gold = {}
    for subset in ("pop", "ti"):
        wanted = {
            int(m["id"].rsplit("-", 1)[1]): m["id"]
            for m in members
            if m["id"].startswith(f"mintqa-{subset}-")
        }
        path = root / f"data/external/mintqa-full/questions/{subset}.jsonl"
        inputs.digest(path)
        with path.open() as stream:
            for number, line in enumerate(stream):
                if number in wanted:
                    gold[wanted[number]] = {**json.loads(line), "_source": f"{path}:{number + 1}"}
    if gold.keys() != {m["id"] for m in members}:
        raise ValueError("released metadata does not cover panel ids")
    for m in members:
        if len(gold[m["id"]]["triple"]) != m["hops"] or gold[m["id"]]["num_hops"] != m["hops"]:
            raise ValueError(f"{m['id']}: released hop count differs from panel")
    return gold


def source_paths(args, panel, family):
    if panel == "validation":
        base = args.sealed_root / family / "validation"
        return {
            arm: base / f"{name}.jsonl"
            for arm, name in (
                ("WT", "finalist"),
                ("agent", AGENTS[family]),
                ("W", "W"),
                ("B2", "B2"),
            )
        }
    root = args.source_root
    wide = (
        root
        / ".claude/worktrees/codex-integrate/target/mintqa-width8/terminal/menuobs-width8"
        / family
        / "development"
    )
    blind = (
        root / f".claude/worktrees/codex-menu-audit/target/mintqa-blind5-audit/{family}/development"
    )
    agent = (
        root / f"target/mintqa-rematch/{family}/development/agent-strong-8.jsonl"
        if family == "qwen38"
        else root / f"target/agent-qual/{family}/development/agent-strong-8-feedback.jsonl"
    )
    return {
        "WT": wide / "menuobs-width8-two-pass-terminal.jsonl",
        "agent": agent,
        "W": wide / "menuobs-width8.jsonl",
        "B2": blind / "blind5-two-pass.jsonl",
    }


def cost_stages(panel, paths, arm):
    if arm in {"agent", "W"}:
        return [paths[arm]]
    base = paths[arm].parent
    if panel == "validation":
        return (
            [paths["W"], base / "W-revision-terminal-raw.jsonl"]
            if arm == "WT"
            else [base / "blind5.jsonl", base / "blind5-revision-raw.jsonl"]
        )
    return (
        [paths["W"], base / "menuobs-width8-revision-terminal-raw.jsonl"]
        if arm == "WT"
        else [base / "blind5.jsonl", base / "blind5-revision-raw.jsonl"]
    )


def scorer_worker(args):
    """Run actual producer scorer code under an isolated, source-pinned interpreter."""
    from xgraph.mintqa_rematch import (
        _paired_bootstrap,
        clustered_difference,
        combined_usage,
        score_panel,
        usage,
    )
    from xgraph.programs import record_path
    from xgraph.prompts import final_answer
    from xgraph.score import record_failure

    inputs = Inputs()
    freeze = inputs.read_json(args.sealed_root / "freeze-manifest.json")
    manifest = inputs.read_json(Path(freeze["panels"]))
    panel_name = "development" if args.panel == "analysis-dev" else "validation"
    members = manifest["panels"][panel_name]["members"]
    ids = {m["id"] for m in members}
    gold = source_gold(args.source_root, members, inputs)
    questions = [
        {
            "id": m["id"],
            "complexity": question_strata(m["id"], gold[m["id"]], m["hops"])["release_group"],
        }
        for m in members
    ]
    paths = source_paths(args, args.panel, args.family)
    records = {arm: inputs.records(path, ids) for arm, path in paths.items()}
    result = {"panel": args.panel, "family": args.family, "arms": {}}
    for arm in ARMS:
        table = score_panel(
            list(records[arm].values()),
            list(records["agent"].values()),
            members,
            questions,
            {qid: g["answer"] for qid, g in gold.items()},
            RESAMPLES,
            SEED,
        )
        # Re-derive complete composite costs from both original model calls.
        stages = [inputs.records(p, ids) for p in cost_stages(args.panel, paths, arm)]
        for qid, record in records[arm].items():
            if usage(record) != combined_usage(*(stage[qid] for stage in stages)):
                raise ValueError(f"{args.panel}/{args.family}/{arm}/{qid}: stage costs differ")
        evidence = {}
        for qid, record in records[arm].items():
            path = record_path(record)
            text = record.get("text", "")
            evidence[qid] = {
                "answer": final_answer(text),
                "record_failure": bool(record_failure(record)),
                "path": (
                    {
                        "start": path[0],
                        "relations": list(path[1]),
                        "selections": getattr(path[1], "selections", {}),
                    }
                    if path
                    else None
                ),
                "invalid_path": invalid_generated_path(text, path),
            }
        result["arms"][arm] = {
            "table": table,
            "evidence": evidence,
            "complete_stage_costs_reproduced": True,
        }
        print(
            f"{args.panel}/{args.family}/{arm}: sealed scorer and stage usage complete", flush=True
        )
    scorer_root = args.sealed_root.parents[1]
    if git("rev-parse", "HEAD", cwd=scorer_root) != freeze["code_commit"]:
        raise ValueError("sealed scorer checkout moved from its frozen producer commit")
    sources = []
    for name, module in list(sys.modules.items()):
        if name.startswith("xgraph") and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(scorer_root / "src"):
                raise ValueError(f"scorer imported another checkout: {path}")
            committed = subprocess.check_output(
                ["git", "show", f"{freeze['code_commit']}:{path.relative_to(scorer_root)}"],
                cwd=scorer_root,
            )
            identity = inputs.digest(path)
            if hashlib.sha256(committed).hexdigest() != identity["sha256"]:
                raise ValueError(f"sealed scorer source changed: {path}")
            sources.append(identity)
    result["provenance"] = {
        "producer_commit": freeze["code_commit"],
        "scorer_sources": sources,
        "inputs": list(inputs.files.values()),
        "python": sys.version,
        "numpy": importlib.metadata.version("numpy"),
        "bootstrap_function_hashes": {
            function.__name__: hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
            for function in (_paired_bootstrap, clustered_difference)
        },
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    write_json(args.out / f"rescore-{args.panel}-{args.family}.json", result)


RULES = {
    "extraction_or_format_failure": (
        "No parsed answer after an explicit Answer: marker, or the "
        "scorer rejects a gold-containing answer record, or a wrong native "
        "score has parsed-answer/gold name-ID overlap (an alias/containment mismatch)."
    ),
    "no_final_answer": "The sealed final_answer parser returns None or an empty string.",
    "invalid_or_closed_off_path": (
        "An explicit escaped/malformed path, or no complete parsed path "
        "and an unclosed opener after menu projection, "
        "or the selected execution action has a saved error/empty "
        "result."
    ),
    "agent_budget_exhaustion": (
        "An incorrect agent record explicitly sets budget_spent=true; "
        "reaching eight calls alone is insufficient."
    ),
    "agent_lookup_loop": (
        "An incorrect agent repeats an identical tool+arguments lookup "
        "at least twice; arguments are JSON-key-sorted, with strings "
        "whitespace-normalized and casefolded. Different pages remain "
        "distinct."
    ),
    "wrong_start_entity": (
        "An attributable full-origin path's start differs from the "
        "gold start, with no normalized-name or KG-name-ID overlap."
    ),
    "wrong_relation_at_hop_k": (
        "The first 1-based relation mismatch between an attributable "
        "gold-start path and the released chain, using "
        "relation-label/alias ID overlap. Length differences alone are "
        "separate."
    ),
    "stopped_early": (
        "An attributable gold-start path is a strict prefix of the released relation sequence."
    ),
    "stopped_late": (
        "An attributable gold-start path strictly extends the released relation sequence."
    ),
    "explicit_old_answer": (
        "A wrong answer equals an explicitly recorded old_answer on an "
        "otherwise exact gold-start/relation path. TI N/O tags alone "
        "never trigger this rule."
    ),
    "correct_path_wrong_final_entity": (
        "A non-rejected, parsed wrong final answer follows the exact "
        "released start/relation path, with no answer/gold name-ID overlap. "
        "This is plan agreement, not "
        "proof of gold endpoint reachability or a temporal cause."
    ),
    "agent_tool_error": (
        "At least one saved action result has an explicit error, an "
        "action is marked failed, or the tool name is invalid."
    ),
    "unattributed": (
        "No other rule is supported by the observed record, including "
        "missing full-origin agent paths."
    ),
}


def normalized(value):
    return unicodedata.normalize("NFC", value).casefold().strip()


def invalid_generated_path(text, path):
    before_answer = re.split(r"(?m)^Answer:", text, maxsplit=1)[0]
    if "⁇" in before_answer:
        return True
    if path is not None:
        return False
    from xgraph.menus import semantic_text

    projected = semantic_text(before_answer)
    return projected.count("⟦") > projected.count("⟧")


def _same(first, second, ids):
    return normalized(first) == normalized(second) or bool(
        ids and set(ids(first)) & set(ids(second))
    )


def _canonical_arguments(value):
    if isinstance(value, dict):
        return {k: _canonical_arguments(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_canonical_arguments(v) for v in value]
    return " ".join(normalized(value).split()) if isinstance(value, str) else value


def repeated_lookups(actions):
    """Count redundant recorded requests; this does not establish causation."""
    lookups = {"search", "relations", "follow", "execute_path", "lookup", "get_facts"}
    calls = [
        json.dumps([a["tool"], _canonical_arguments(a.get("arguments", {}))], sort_keys=True)
        for a in actions
        if a.get("tool") in lookups
    ]
    return sum(n - 1 for n in Counter(calls).values())


def diagnose(record, gold, evidence, *, entity_ids=None, relation_ids=None):
    """Assign observable error labels without changing scorer correctness."""
    labels = []
    actions = record.get("actions", [])
    duplicates = repeated_lookups(actions)
    tool_errors = sum(
        bool(a.get("result", {}).get("error") or a.get("failed") or a.get("tool") == "invalid")
        for a in actions
    )
    path = evidence.get("path")
    answer = evidence.get("answer")
    answer_gold_identity_overlap = bool(answer and _same(answer, gold["answer"], entity_ids))
    full_origin = evidence.get("full_origin", True)
    exact = False
    first_mismatch = None
    same_start = None
    if path and full_origin:
        same_start = _same(path["start"], gold["triple"][0][0], entity_ids)
        relations = path["relations"]
        gold_relations = [t[1] for t in gold["triple"]]
        first_mismatch = next(
            (
                k
                for k, (a, b) in enumerate(zip(relations, gold_relations, strict=False), 1)
                if not _same(a, b, relation_ids)
            ),
            None,
        )
        exact = same_start and first_mismatch is None and len(relations) == len(gold_relations)
    if not evidence["correct"]:
        text = record.get("text", "")
        if (not answer and re.search(r"\bAnswer\s*:", text, re.IGNORECASE)) or (
            answer
            and (
                answer_gold_identity_overlap
                or (evidence["record_failure"] and normalized(gold["answer"]) in normalized(answer))
            )
        ):
            labels.append("extraction_or_format_failure")
        if not answer:
            labels.append("no_final_answer")
        if record.get("escaped") or record.get("malformed") or evidence.get("invalid_path"):
            labels.append("invalid_or_closed_off_path")
        if record.get("arm") == "agent" and record.get("budget_spent"):
            labels.append("agent_budget_exhaustion")
        if record.get("arm") == "agent" and duplicates:
            labels.append("agent_lookup_loop")
        if path and full_origin:
            if not same_start:
                labels.append("wrong_start_entity")
            elif first_mismatch is not None:
                labels.append(f"wrong_relation_at_hop_{first_mismatch}")
            elif len(path["relations"]) < len(gold["triple"]):
                labels.append("stopped_early")
            elif len(path["relations"]) > len(gold["triple"]):
                labels.append("stopped_late")
            if (
                exact
                and answer
                and not evidence["record_failure"]
                and not answer_gold_identity_overlap
            ):
                if gold.get("old_answer") and normalized(answer) == normalized(gold["old_answer"]):
                    labels.append("explicit_old_answer")
                labels.append("correct_path_wrong_final_entity")
        if record.get("arm") == "agent" and tool_errors:
            labels.append("agent_tool_error")
        if not labels:
            labels.append("unattributed")
    return {
        "correct": evidence["correct"],
        "primary": labels[0] if labels else "correct",
        "labels": labels,
        "first_wrong_relation_hop": first_mismatch,
        "same_start": same_start,
        "gold_plan_agreement": bool(exact),
        "answer_gold_identity_overlap": answer_gold_identity_overlap,
        "full_origin": full_origin,
        "path": path,
        "path_attribution": evidence.get("path_attribution", "generated path"),
        "redundant_lookup_calls": duplicates,
        "saved_tool_errors": tool_errors,
        "actions_with_saved_result": sum("result" in a for a in actions),
        "actions": len(actions),
    }


def agent_evidence(record, gold, evidence, *, entity_ids=None):
    """Attribute saved execution actions, without inventing missing results."""
    entity_ids = entity_ids or (lambda name: {normalized(name)})
    roots, candidates = {}, []
    for index, action in enumerate(record.get("actions", [])):
        arguments = action.get("arguments", {})
        if not isinstance(arguments, dict):
            continue
        tool = action.get("tool")
        if tool == "execute_path" and "start" in arguments and "relations" in arguments:
            start = str(arguments["start"]).strip()
            relations = arguments["relations"] or []
            if isinstance(relations, str):
                relations = re.split(r"[·,]", relations)
            if not isinstance(relations, list):
                continue
            relations = [str(r).strip() for r in relations if str(r).strip()]
        elif tool == "follow" and "entity" in arguments and "relation" in arguments:
            start = str(arguments["entity"]).strip()
            relations = [str(arguments["relation"]).strip()]
        else:
            continue
        prior = next((roots[q] for q in sorted(entity_ids(start)) if q in roots), None)
        if prior:
            path = {
                "start": prior["path"]["start"],
                "relations": [*prior["path"]["relations"], *relations],
            }
            full_origin = prior["full_origin"]
        else:
            path = {"start": start, "relations": relations}
            gold_start = gold["triple"][0][0]
            full_origin = _same(start, gold_start, entity_ids) or (
                not candidates
                and not any(_same(start, t[0], entity_ids) for t in gold["triple"][1:])
            )
        result = action.get("result", {})
        entities = result.get("entities", [])
        supports_answer = any(
            _same(evidence.get("answer") or "", entity.get("label", ""), entity_ids)
            or bool(entity.get("id") in entity_ids(evidence.get("answer") or ""))
            for entity in entities
        )
        invalid = bool(
            result.get("error")
            or action.get("failed")
            or result.get("count") == 0
            or ("entities" in result and not entities)
        )
        candidate = {
            "path": path,
            "full_origin": full_origin,
            "selected_action": index,
            "selected_result_observed": "result" in action,
            "invalid_path": invalid,
            "supports_answer": supports_answer,
            "has_saved_endpoints": bool(entities),
        }
        candidates.append(candidate)
        if not invalid:
            for entity in entities:
                if entity.get("id"):
                    roots[entity["id"]] = candidate
    matched = [r for r in candidates if r["supports_answer"]]
    successful = [r for r in candidates if r["has_saved_endpoints"] and not r["invalid_path"]]
    selected = (matched or successful or candidates or [{}])[-1]
    attribution = (
        "last saved action supporting answer"
        if matched
        else "last action with saved endpoints"
        if successful
        else "last attempted action; result may be unobserved"
        if candidates
        else "no execution action"
    )
    return {
        **evidence,
        "path": None,
        "full_origin": False,
        **selected,
        "path_attribution": attribution,
    }


def question_strata(qid, gold, hops):
    """Release annotations stay distinct from answer-surface proxies."""
    from xgraph.mintqa_bench import stratum

    subset = qid.split("-")[1]
    fraction = stratum(subset, gold["type"], hops)
    mismatch = len(gold["type"]) != hops
    result = {
        "subset": subset,
        "hops": str(hops),
        "subset_hops": f"{subset} {hops}-hop",
        "release_group": f"{subset} {hops}-hop {fraction}",
        "knowledge_status": "not_annotated",
        "popularity_bucket": "not_annotated",
        "update_status": str(gold.get("update_status", "not_recorded")),
        "answer_type": str(gold.get("answer_type", "not_recorded")),
    }
    if subset == "ti":
        old = gold["type"].count("O")
        result["knowledge_status"] = (
            "types_mismatch"
            if mismatch
            else "all_new"
            if old == 0
            else "all_old"
            if old == hops
            else "mixed_old_new"
        )
        result["old_hop_fraction"] = fraction
    else:
        popular = gold["type"].count("F")
        result["popularity_bucket"] = (
            "types_mismatch"
            if mismatch
            else "all_rare"
            if popular == 0
            else "all_popular"
            if popular == hops
            else "mixed_popular_rare"
        )
        result["popular_hop_fraction"] = fraction
    answer = gold["answer"].strip()
    month = (
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    )
    if re.fullmatch(r"[+-]?\d{1,6}-\d{2}(?:-\d{2})?", answer) or re.fullmatch(
        rf"(?:\d{{1,2}} {month} \d{{1,6}}|{month} \d{{1,2}},? \d{{1,6}}|{month} \d{{4}})",
        answer,
        re.IGNORECASE,
    ):
        surface = "date_like"
    elif re.fullmatch(r"[+-]?[\d,]+(?:\.\d+)?%?", answer):
        surface = "number_like"
    elif normalized(answer) in {"yes", "no", "true", "false"}:
        surface = "boolean_like"
    else:
        surface = "other_label"
    return {**result, "answer_surface_type": surface}


def paired_outcomes(first, second):
    from xgraph.validation_analysis import transition_matrix

    transition = transition_matrix(second, first)
    names = {
        "right_to_right": "both_right",
        "wrong_to_right": "first_only",
        "right_to_wrong": "second_only",
        "wrong_to_wrong": "both_wrong",
    }
    return {
        **{
            k: v
            for k, v in transition.items()
            if k not in {"counts", "weights", "weighted_shares", "unweighted_shares"}
        },
        **{
            field: {names[k]: v for k, v in transition[field].items()}
            for field in ("counts", "weights", "weighted_shares", "unweighted_shares")
        },
    }


def verify_rescore(actual, expected, context):
    """Refuse even a single changed flag, metadata value or accounting margin."""
    from xgraph.validation_analysis import COST_FIELDS, aligned_rows, close_enough

    pairs = aligned_rows(actual["per_question"], expected["per_question"])
    for a, b in pairs:
        for field in ("correct", "hops", "cluster", *COST_FIELDS):
            if a[field] != b[field]:
                raise ValueError(f"{context}/{a['id']}: {field} differs from sealed report")
    for field in ("accuracy", "questions", "weight_sum", "missing_outputs"):
        close_enough(actual[field], expected[field], f"{context}/{field}")
    for dimension in ("cost",):
        for kind in ("totals", "per_question", "weighted_per_question"):
            for field, value in expected[dimension][kind].items():
                close_enough(
                    actual[dimension][kind][field], value, f"{context}/{dimension}/{kind}/{field}"
                )
    for dimension in ("by_hop", "by_split", "by_group"):
        if actual[dimension].keys() != expected[dimension].keys():
            raise ValueError(f"{context}: {dimension} groups differ")
        for name, group in expected[dimension].items():
            for field in ("accuracy", "questions", "weight_sum"):
                close_enough(
                    actual[dimension][name][field],
                    group[field],
                    f"{context}/{dimension}/{name}/{field}",
                )
    if actual["native_unweighted"] != expected["native_unweighted"]:
        raise ValueError(f"{context}: native unweighted totals differ")
    return {
        "questions": len(pairs),
        "exact_correctness_vector": True,
        "exact_row_costs_and_metadata": True,
        "all_report_totals_reproduced": True,
    }


def sample_discordant(first, second, *, seed=17, size=5):
    from xgraph.validation_analysis import aligned_rows

    pairs = aligned_rows(first, second)
    result = {}
    for name, flags in (("first_only", (True, False)), ("second_only", (False, True))):
        ids = sorted(a["id"] for a, b in pairs if (a["correct"], b["correct"]) == flags)
        result[name] = sorted(random.Random(seed).sample(ids, min(size, len(ids))))
    return result


def taxonomy_tables(diagnostics, members):
    ids = {m["id"] for m in members}
    if diagnostics.keys() != ids:
        raise ValueError("taxonomy ids differ from scored members")
    mass = sum(m["weight"] for m in members)
    errors = [m for m in members if not diagnostics[m["id"]]["correct"]]
    error_mass = sum(m["weight"] for m in errors)
    inventory = (
        {k for k in RULES if k != "wrong_relation_at_hop_k"}
        | {f"wrong_relation_at_hop_{k}" for k in range(1, 9)}
        | {label for d in diagnostics.values() for label in d["labels"]}
    )
    result = {"questions": len(members), "errors": len(errors), "error_weight": error_mass}
    for field in ("primary", "overlapping"):
        table = {}
        for label in sorted(inventory):
            chosen = [
                m
                for m in errors
                if (
                    diagnostics[m["id"]]["primary"] == label
                    if field == "primary"
                    else label in diagnostics[m["id"]]["labels"]
                )
            ]
            weight = sum(m["weight"] for m in chosen)
            table[label] = {
                "count": len(chosen),
                "weight": weight,
                "weighted_share_all": weight / mass,
                "weighted_share_errors": weight / error_mass if error_mass else 0,
            }
        result[field] = table
    if sum(r["count"] for r in result["primary"].values()) != len(errors):
        raise ValueError("primary labels do not partition errors")
    result["top_three_primary"] = [
        {"category": label, **value}
        for label, value in sorted(
            result["primary"].items(), key=lambda item: (-item[1]["count"], item[0])
        )[:3]
        if value["count"]
    ]
    return result


def error_transitions(first, second, members):
    cells = defaultdict(lambda: {"count": 0, "weight": 0})
    mass = sum(m["weight"] for m in members)
    for m in members:
        a, b = first[m["id"]], second[m["id"]]
        if a["correct"] == b["correct"]:
            continue
        cell = cells[a["primary"], b["primary"]]
        cell["count"] += 1
        cell["weight"] += m["weight"]
    return [
        {"first": a, "second": b, **cell, "weighted_share_all": cell["weight"] / mass}
        for (a, b), cell in sorted(cells.items())
    ]


class EntityNames:
    """Read only indexed name aliases; never load or execute a KG index."""

    def __init__(self, root, inputs):
        path = root / "data/external/mintqa-full/kg.sqlite"
        inputs.digest(path)
        self.connection = sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)
        self.connection.executescript(
            "PRAGMA query_only=ON; PRAGMA mmap_size=0; PRAGMA cache_size=-1024;"
        )
        relation_path = root / "data/external/mintqa-full/kg/relation_labels.jsonl"
        inputs.digest(relation_path)
        self.relations = defaultdict(set)
        with relation_path.open() as stream:
            for line in stream:
                qid, label, *_ = json.loads(line)
                self.relations[normalized(label)].add(qid)
        self.named = functools.lru_cache(maxsize=50_000)(self._named)

    def _named(self, label):
        if not label:
            return frozenset()
        if re.fullmatch(r"Q\d+", label):
            return frozenset([label])
        return frozenset(
            f"Q{r[0]}"
            for r in self.connection.execute(
                "SELECT item FROM names WHERE norm = ?", (normalized(label),)
            )
        )

    def relation_ids(self, label):
        return (
            {label}
            if re.fullmatch(r"P\d+", label)
            else self.relations.get(normalized(label), set())
        )


def stratum_tables(rows, strata):
    from xgraph.validation_analysis import cost_per_correct, paired_table

    result = defaultdict(dict)
    indexes = {arm: {r["id"]: r for r in records} for arm, records in rows.items()}
    mass = sum(r["weight"] for r in rows["WT"])
    groups = defaultdict(lambda: defaultdict(list))
    # Keep the scorer's member order: it determines the bootstrap RNG traversal.
    for member in rows["WT"]:
        for dimension, group in strata[member["id"]].items():
            groups[dimension][group].append(member["id"])
    for dimension, grouped in groups.items():
        for group, ids in sorted(grouped.items()):
            selected = {arm: [indexes[arm][qid] for qid in ids] for arm in ARMS}
            comparisons = {
                f"{a} - {b}": paired_table(
                    selected[a], selected[b], resamples=RESAMPLES, seed=SEED, total_mass=mass
                )
                for a, b in combinations(ARMS, 2)
            }
            result[dimension][group] = {
                "questions": len(ids),
                "arms": {arm: cost_per_correct(records) for arm, records in selected.items()},
                "comparisons": comparisons,
            }
    return dict(result)


def compact_examples(samples, records, evidence, diagnostics, gold):
    result = {}
    for direction, ids in samples.items():
        examples = []
        for qid in ids:
            actions = records["agent"][qid].get("actions", [])
            keep = (
                list(range(len(actions)))
                if len(actions) <= 6
                else [0, 1, 2, *range(len(actions) - 3, len(actions))]
            )
            calls = []
            for index in keep:
                text = json.dumps(actions[index], sort_keys=True, ensure_ascii=False)
                calls.append(
                    {
                        "index": index,
                        "excerpt": text[:240],
                        "characters_omitted": max(0, len(text) - 240),
                    }
                )
            examples.append(
                {
                    "id": qid,
                    "question": gold[qid]["question"],
                    "gold": gold[qid]["answer"],
                    "answers": {
                        arm: {
                            "parsed": evidence[arm][qid]["answer"],
                            "scorer_rejected_record": evidence[arm][qid]["record_failure"],
                            "correct": diagnostics[arm][qid]["correct"],
                        }
                        for arm in ("WT", "agent")
                    },
                    "WT_path": evidence["WT"][qid]["path"],
                    "agent_tool_calls": calls,
                    "tool_calls_omitted": len(actions) - len(keep),
                    "failures": {arm: diagnostics[arm][qid]["primary"] for arm in ("WT", "agent")},
                    "sources": {
                        **{arm: records[arm][qid]["_source"] for arm in ("WT", "agent")},
                        "gold": gold[qid]["_source"],
                    },
                }
            )
        result["WT_only" if direction == "first_only" else "agent_only"] = examples
    return result


def _sidecars(inputs, paths, panel):
    for arm, path in paths.items():
        for source in {path, *cost_stages(panel, paths, arm)}:
            for suffix in (".summary.json", ".provenance.json", ".done.json"):
                sidecar = source.with_name(source.name + suffix)
                if sidecar.exists():
                    data = inputs.read_json(sidecar)
                    if suffix == ".done.json":
                        for item in data["outputs"]:
                            if (
                                Path(item["path"]).resolve() == source.resolve()
                                and inputs.digest(source)["sha256"] != item["sha256"]
                            ):
                                raise ValueError(f"receipt output hash mismatch: {source}")


def build_report(args):
    from xgraph.validation_analysis import close_enough, cost_per_correct

    started, inputs = time.monotonic(), Inputs()
    driver_hash = inputs.digest(Path(__file__))["sha256"]
    freeze_path = args.sealed_root / "freeze-manifest.json"
    freeze = inputs.read_json(freeze_path)
    conjunction = inputs.read_json(args.sealed_root / "conjunction.json")
    if conjunction["freeze_sha256"] != inputs.digest(freeze_path)["sha256"]:
        raise ValueError("conjunction and freeze differ")
    manifest = inputs.read_json(Path(freeze["panels"]))
    frozen_inputs = {str(Path(item["path"]).resolve()): item for item in freeze["inputs"]}
    for path in (
        Path(freeze["panels"]),
        *(
            args.source_root / f"data/external/mintqa-full/questions/{s}.jsonl"
            for s in ("pop", "ti")
        ),
    ):
        if inputs.digest(path)["sha256"] != frozen_inputs[str(path.resolve())]["sha256"]:
            raise ValueError(f"frozen metadata input changed: {path}")
    dev = inputs.read_json(args.source_root / "target/analysis-dev/scores.json")
    vectors = inputs.read_json(args.source_root / "target/analysis-dev/correctness-vectors.json")
    old_costs = inputs.read_json(args.source_root / "target/analysis-dev/costs.json")
    frontier = inputs.read_json(args.budget_root / "frontier.json")
    inputs.digest(args.budget_root / "provenance.json")
    names = EntityNames(args.source_root, inputs)
    result = {
        "scope": "post hoc descriptive; outside registered multiplicity control",
        "config": {
            "seed": SEED,
            "resamples": RESAMPLES,
            "sample_per_direction": 5,
            "cluster_unit": "relation sequence within hops, hop-mass calibrated survey weights",
            "comparators": AGENTS,
        },
        "panels": {},
    }
    commands, examples = [], {}
    worker_command = [
        "nice",
        "-n",
        "19",
        "taskset",
        "-c",
        "1-4",
        "prlimit",
        "--as=4000000000",
        sys.executable,
        "-I",
        "-S",
        str(WORKTREE / "scripts/mintqa/error-analysis.py"),
        "--source-root",
        str(args.source_root),
        "--sealed-root",
        str(args.sealed_root),
        "--out",
        str(args.out),
        "--scorer-worker",
    ]
    with (args.out / "per-question.jsonl").open("w") as per_question:
        for panel in ("validation", "analysis-dev"):
            result["panels"][panel] = {}
            examples[panel] = {}
            members = manifest["panels"]["validation" if panel == "validation" else "development"][
                "members"
            ]
            ids = {m["id"] for m in members}
            gold = source_gold(args.source_root, members, inputs)
            strata = {m["id"]: question_strata(m["id"], gold[m["id"]], m["hops"]) for m in members}
            for family in FAMILIES:
                command = [*worker_command, "--panel", panel, "--family", family]
                commands.append(command)
                subprocess.run(command, cwd=WORKTREE, check=True)
                rescored = inputs.read_json(args.out / f"rescore-{panel}-{family}.json")
                from xgraph import mintqa_rematch

                for name, expected in rescored["provenance"]["bootstrap_function_hashes"].items():
                    function = getattr(mintqa_rematch, name)
                    if hashlib.sha256(inspect.getsource(function).encode()).hexdigest() != expected:
                        raise ValueError(f"analysis bootstrap differs from sealed scorer: {name}")
                for item in rescored["provenance"]["inputs"]:
                    if inputs.digest(item["path"])["sha256"] != item["sha256"]:
                        raise ValueError(f"worker input changed: {item['path']}")
                paths = source_paths(args, panel, family)
                records = {arm: inputs.records(path, ids) for arm, path in paths.items()}
                _sidecars(inputs, paths, panel)
                rows = {arm: rescored["arms"][arm]["table"]["per_question"] for arm in ARMS}
                evidence = {arm: rescored["arms"][arm]["evidence"] for arm in ARMS}
                identity = {}
                report = inputs.read_json(args.sealed_root / family / "report.json")
                if report["freeze_sha256"] != inputs.digest(freeze_path)["sha256"]:
                    raise ValueError("sealed report and freeze differ")
                if (
                    freeze["strongest_agents"][family] != AGENTS[family]
                    or freeze["families"][family]["selected_policy"]["policy"] != "WT"
                ):
                    raise ValueError("sealed finalist/comparator differs")
                for arm in ARMS:
                    table = rescored["arms"][arm]["table"]
                    if any(
                        any(row[k] != m[k] for k in m)
                        for row, m in zip(rows[arm], members, strict=True)
                    ):
                        raise ValueError("rescored member metadata/order differs")
                    if panel == "validation":
                        label = {"WT": "finalist", "agent": AGENTS[family]}.get(arm, arm)
                        identity[arm] = verify_rescore(
                            table, report["panels"]["validation"][label], f"{family}/{arm}"
                        )
                    else:
                        label = DEV_AGENTS[family] if arm == "agent" else arm
                        vector = {r["id"]: r["correct"] for r in rows[arm]}
                        if vector != vectors[family][label]:
                            raise ValueError(
                                f"{family}/{arm}: analysis-dev correctness vector differs"
                            )
                        for expected in (
                            dev[family][label],
                            frontier["families"][family]["policies"][label]["score"],
                        ):
                            close_enough(
                                table["accuracy"],
                                expected["accuracy"],
                                f"{family}/{arm} dev accuracy",
                            )
                            if sum(vector.values()) != expected["correct_n"]:
                                raise ValueError("analysis-dev correct count differs")
                        for kind in ("totals", "per_question", "weighted_per_question"):
                            for field, value in table["cost"][kind].items():
                                for expected in (
                                    old_costs[family][label]["overall"],
                                    frontier["families"][family]["policies"][label]["cost"],
                                ):
                                    close_enough(
                                        value,
                                        expected[kind][field],
                                        f"{family}/{arm}/dev/{kind}/{field}",
                                    )
                        identity[arm] = {
                            "questions": len(ids),
                            "exact_correctness_vector": True,
                            "analysis_dev_and_budget_frontier_totals_reproduced": True,
                        }
                comparisons = {
                    f"{a} - {b}": {
                        **paired_outcomes(rows[a], rows[b]),
                        "interval": rescored["arms"][a]["table"]["vs_agent"]
                        if b == "agent"
                        else _interval(rows[a], rows[b]),
                    }
                    for a, b in combinations(ARMS, 2)
                }
                if panel == "validation":
                    for key in ("difference", "low", "high"):
                        close_enough(
                            comparisons["WT - agent"]["interval"][key],
                            report["amended_comparisons"]["F"][key],
                            f"{family}/F/{key}",
                        )
                    identity["F_point_and_interval_reproduced"] = True
                else:
                    for key in ("difference", "low", "high"):
                        close_enough(
                            comparisons["WT - agent"]["interval"][key],
                            dev[family]["WT"]["vs_strongest_cluster_bootstrap"][key],
                            f"{family}/dev/WT-agent/{key}",
                        )
                diagnostics = {}
                for arm in ARMS:
                    diagnostics[arm] = {}
                    for row in rows[arm]:
                        qid, record = row["id"], records[arm][row["id"]]
                        e = {**evidence[arm][qid], "correct": row["correct"]}
                        if arm == "agent":
                            e = agent_evidence(record, gold[qid], e, entity_ids=names.named)
                        d = diagnose(
                            record,
                            gold[qid],
                            e,
                            entity_ids=names.named,
                            relation_ids=names.relation_ids,
                        )
                        diagnostics[arm][qid] = d
                        per_question.write(
                            json.dumps(
                                {
                                    "panel": panel,
                                    "family": family,
                                    "arm": arm,
                                    **row,
                                    "strata": strata[qid],
                                    "diagnostic": d,
                                    "parsed_answer": e["answer"],
                                    "scorer_rejected_record": e["record_failure"],
                                    "source": record["_source"],
                                    "gold_source": gold[qid]["_source"],
                                },
                                ensure_ascii=False,
                                allow_nan=False,
                            )
                            + "\n"
                        )
                transitions = error_transitions(diagnostics["WT"], diagnostics["agent"], members)
                paired = comparisons["WT - agent"]["counts"]
                if (
                    sum(r["count"] for r in transitions if r["first"] == "correct")
                    != paired["first_only"]
                    or sum(r["count"] for r in transitions if r["second"] == "correct")
                    != paired["second_only"]
                ):
                    raise ValueError("discordant error transitions do not conserve paired counts")
                examples[panel][family] = compact_examples(
                    sample_discordant(rows["WT"], rows["agent"]),
                    records,
                    evidence,
                    diagnostics,
                    gold,
                )
                result["panels"][panel][family] = {
                    "questions": len(ids),
                    "agent": AGENTS[family],
                    "identity": identity,
                    "arms": {arm: cost_per_correct(rows[arm]) for arm in ARMS},
                    "comparisons": comparisons,
                    "strata": stratum_tables(rows, strata),
                    "taxonomy": {arm: taxonomy_tables(diagnostics[arm], members) for arm in ARMS},
                    "discordant_error_transitions": transitions,
                    "observability": {
                        arm: {
                            "actions": sum(d["actions"] for d in diagnostics[arm].values()),
                            "actions_with_saved_result": sum(
                                d["actions_with_saved_result"] for d in diagnostics[arm].values()
                            ),
                            "unreconstructed_origins": sum(
                                not d["full_origin"] for d in diagnostics[arm].values()
                            ),
                            "version_or_old_answer_metadata": sum(
                                bool(g.get("old_answer")) for g in gold.values()
                            ),
                        }
                        for arm in ARMS
                    },
                    "scorer_provenance": rescored["provenance"],
                }
                print(
                    f"{panel}/{family}: report identity, paired tables, strata, "
                    "taxonomy and samples complete",
                    flush=True,
                )
    names.connection.close()
    names.named.cache_clear()
    if Inputs().digest(Path(__file__))["sha256"] != driver_hash:
        raise ValueError("analysis code changed while running; regenerate with fixed source")
    for name, module in list(sys.modules.items()):
        if name.startswith("xgraph") and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(WORKTREE / "src"):
                raise ValueError(f"analysis imported another checkout: {path}")
            inputs.digest(path)
    for path in (
        Path(__file__),
        WORKTREE / "scripts/mintqa/error-analysis.py",
        WORKTREE / "tests/test_mintqa_rematch.py",
        WORKTREE / "uv.lock",
    ):
        inputs.digest(path)
    write_json(args.out / "tables.json", result)
    write_json(
        args.out / "examples.json",
        {"seed": SEED, "sample_size_per_direction": 5, "panels": examples},
    )
    write_json(
        args.out / "taxonomy-rules.json",
        {
            "rules_in_precedence_order": RULES,
            "scope": (
                "Errors only; primary label partitions errors, all supported labels also retained."
            ),
            "attribution": (
                "Saved tool results only; plain-agent missing results remain "
                "unobserved. SQL supplies name aliases, not trajectory replay."
            ),
        },
    )
    provenance = {
        "artifact_kind": "xgraph.error_analysis",
        "status": "COMPLETE",
        "created_at": datetime.now(UTC).isoformat(),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(git("status", "--porcelain")),
        "sealed_producer_commit": freeze["code_commit"],
        "freeze_sha256": inputs.digest(freeze_path)["sha256"],
        "model_inventory": {family: freeze["families"][family]["model"] for family in FAMILIES},
        "config": result["config"],
        "command": _command(args),
        "worker_commands": commands,
        "inputs": list(inputs.files.values()),
        "python": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "tokenizers")},
        "sqlite": sqlite3.sqlite_version,
        "elapsed_seconds": time.monotonic() - started,
        "parent_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "affinity": sorted(os.sched_getaffinity(0)),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "address_space_limit_bytes": resource.getrlimit(resource.RLIMIT_AS)[0],
        "gpu_hours": 0,
        "gpu_queue": None,
        "gpu_hours_basis": (
            "No model calls; existing records, CPU resampling and indexed SQLite name lookups only."
        ),
    }
    render_artifacts(args.out, result, examples, provenance)
    provenance["outputs"] = [
        Inputs().digest(path)
        for path in sorted(args.out.iterdir())
        if path.is_file() and path.name != "provenance.json"
    ]
    write_json(args.out / "provenance.json", provenance)
    print(f"COMPLETE: {args.out / 'report.md'}", flush=True)


def _interval(first, second):
    from xgraph.mintqa_rematch import clustered_difference

    return clustered_difference(
        [r["correct"] for r in first], [r["correct"] for r in second], first, RESAMPLES, SEED
    )


def _command(args):
    return [
        "nice",
        "-n",
        "19",
        "taskset",
        "-c",
        "1-4",
        "prlimit",
        "--as=4000000000",
        str(WORKTREE / ".venv/bin/python"),
        "-I",
        "-S",
        str(WORKTREE / "scripts/mintqa/error-analysis.py"),
        "--source-root",
        str(args.source_root),
        "--sealed-root",
        str(args.sealed_root),
        "--budget-root",
        str(args.budget_root),
        "--out",
        str(args.out),
    ]


def _taxonomy_cell(item):
    return f"{item['count']} ({100 * item['weighted_share_all']:.2f})"


def _summary(result, provenance):
    aliases = {
        "correct_path_wrong_final_entity": "endpoint",
        "agent_budget_exhaustion": "budget",
        "wrong_start_entity": "start",
        "unattributed": "unknown",
        "invalid_or_closed_off_path": "path",
        "agent_lookup_loop": "loop",
        "stopped_early": "early",
        "stopped_late": "late",
        "no_final_answer": "no-answer",
        "extraction_or_format_failure": "format",
        "agent_tool_error": "tool-error",
    }
    aliases.update({f"wrong_relation_at_hop_{k}": f"R{k}" for k in range(1, 9)})
    pairs, top = [], []
    for panel, families in result["panels"].items():
        for family, data in families.items():
            counts = data["comparisons"]["WT - agent"]["counts"]
            pairs.append([panel, family, counts["first_only"], counts["second_only"]])
            if panel == "validation":
                for arm in ARMS:
                    top.append(
                        [
                            family,
                            arm,
                            "; ".join(
                                f"{aliases.get(r['category'], r['category'])} {r['count']}"
                                for r in data["taxonomy"][arm]["top_three_primary"]
                            ),
                        ]
                    )
    val = result["panels"]["validation"]

    def gaps(dimension, group):
        points = [
            val[f]["strata"][dimension][group]["comparisons"]["WT - agent"]["difference"]
            for f in FAMILIES
        ]
        return "/".join(f"{100 * point:+.2f}" for point in points)

    return "\n".join(
        [
            "# Descriptive MINTQA errors",
            "",
            "Post hoc: weighted estimates, raw counts; seed 17, 10,000 clustered resamples.",
            "",
            _table(["Panel", "Family", "WT only", "Agent only"], pairs),
            "",
            "Validation primary errors:",
            "",
            _table(["Family", "Arm", "Top three, counts"], top),
            "",
            "Rk=wrong relation at hop k; endpoint=gold plan/wrong final entity; "
            "budget=spent budget; unknown=unattributed; early=stopped early.",
            "",
            f"Validation WT-agent, Qwen/Gemma pp: 1-hop {gaps('hops', '1')}; "
            f"4-hop {gaps('hops', '4')}; "
            f"mixed old/new {gaps('knowledge_status', 'mixed_old_new')}; "
            f"all-new {gaps('knowledge_status', 'all_new')}. Strata overlap; no interaction claim.",
            "",
            "Missing semantic types, update versions and Qwen tool results limit attribution. "
            "Tables/samples: adjacent MD/JSON.",
            "",
            f"Commit: `{provenance['commit']}`. "
            "Regenerate: `./target/error-analysis/regenerate.sh`. "
            "Sealed totals and dev vectors/costs match. GPU-hours: 0; no model calls.",
            "",
        ]
    )


def _table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        [
            "| " + " | ".join(map(cell, headers)) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(map(cell, row)) + " |" for row in rows),
        ]
    )


def _ci(item):
    return f"{100 * item['difference']:+.2f} [{100 * item['low']:+.2f}, {100 * item['high']:+.2f}]"


def render_artifacts(out, result, examples, provenance):
    paired_lines, strata_lines, taxonomy_lines, example_lines = [], [], [], []
    top = []
    for panel, families in result["panels"].items():
        paired_lines += [f"# {panel}", ""]
        taxonomy_lines += [f"# {panel}", ""]
        example_lines += [
            f"# {panel}",
            "",
            (
                "Uniform question sample without replacement within each "
                "direction/family, sorted IDs, seed 17 reset per direction. No "
                "manual curation. Each call excerpt is at most 240 characters; "
                "traces longer than six calls show the first and last three."
            ),
            "",
        ]
        for family, data in families.items():
            paired_lines += [
                f"## {family}; comparator {data['agent']}",
                "",
                _table(
                    [
                        "Arm",
                        "Correct n",
                        "Accuracy %",
                        "Prompt/q",
                        "Generated/q",
                        "Observation/q (included in generated)",
                        "Total tokens/q",
                        "Calls/q",
                    ],
                    [
                        [
                            arm,
                            data["arms"][arm]["correct_count"],
                            f"{100 * data['arms'][arm]['accuracy']:.2f}",
                            *(
                                f"{data['arms'][arm]['weighted_per_question'][field]:.2f}"
                                for field in (
                                    "prompt_tokens",
                                    "generated_tokens",
                                    "observation_tokens",
                                    "total_tokens",
                                    "model_calls",
                                )
                            ),
                        ]
                        for arm in ARMS
                    ],
                ),
                "",
                _table(
                    [
                        "First - second",
                        "Both right",
                        "First only",
                        "Second only",
                        "Both wrong",
                        "Weighted difference pp [CI]",
                    ],
                    [
                        [
                            name,
                            *(
                                item["counts"][k]
                                for k in ("both_right", "first_only", "second_only", "both_wrong")
                            ),
                            _ci(item["interval"]),
                        ]
                        for name, item in data["comparisons"].items()
                    ],
                ),
                "",
                (
                    "JSON retains the weighted mass and share of every paired "
                    "cell. Off-diagonal mass difference reproduces the accuracy "
                    "difference."
                ),
                "",
            ]
            for arm in ARMS:
                top.append(
                    [
                        panel,
                        family,
                        arm,
                        "; ".join(
                            f"{r['category']} {r['count']}"
                            for r in data["taxonomy"][arm]["top_three_primary"]
                        ),
                    ]
                )
            for dimension, groups in data["strata"].items():
                strata_lines += [
                    f"## {panel} / {family} / {dimension}",
                    "",
                    _table(
                        [
                            "Group",
                            "n",
                            "Clusters",
                            "WT %",
                            "W %",
                            "B2 %",
                            "Agent %",
                            "WT-agent pp [CI]",
                            "WT-W pp [CI]",
                            "WT-B2 pp [CI]",
                            "Total tokens/q WT / W / B2 / agent",
                        ],
                        [
                            [
                                group,
                                item["questions"],
                                item["comparisons"]["WT - agent"]["clusters"],
                                *(f"{100 * item['arms'][arm]['accuracy']:.2f}" for arm in ARMS),
                                *(
                                    _ci(item["comparisons"][f"WT - {arm}"])
                                    for arm in ("agent", "W", "B2")
                                ),
                                " / ".join(
                                    f"{item['arms'][arm]['weighted_per_question']['total_tokens']:.1f}"
                                    for arm in ARMS
                                ),
                            ]
                            for group, item in groups.items()
                        ],
                    ),
                    "",
                ]
            taxonomy_lines += [
                f"## {family}",
                "",
                (
                    "Primary labels partition scorer errors. Overlapping labels "
                    "count every supported rule. Correct records never acquire an "
                    "error label, even if a tool failed or the budget was spent."
                ),
                "",
            ]
            labels = sorted(data["taxonomy"]["WT"]["primary"])
            for kind in ("primary", "overlapping"):
                taxonomy_lines += [
                    f"### {kind}",
                    "",
                    _table(
                        [
                            "Category",
                            *(f"{arm} count (weighted % of all questions)" for arm in ARMS),
                        ],
                        [
                            [
                                label,
                                *(
                                    _taxonomy_cell(data["taxonomy"][arm][kind][label])
                                    for arm in ARMS
                                ),
                            ]
                            for label in labels
                        ],
                    ),
                    "",
                ]
            taxonomy_lines += [
                "### Discordant failure transition matrix",
                "",
                _table(
                    [
                        "WT primary label",
                        "Agent primary label",
                        "Count",
                        "Weight",
                        "Weighted % of all",
                    ],
                    [
                        [
                            r["first"],
                            r["second"],
                            r["count"],
                            f"{r['weight']:.4f}",
                            f"{100 * r['weighted_share_all']:.2f}",
                        ]
                        for r in data["discordant_error_transitions"]
                    ],
                ),
                "",
                _table(
                    [
                        "Arm",
                        "Tool actions",
                        "Saved results",
                        "Unreconstructed origins",
                        "Explicit old-answer metadata",
                    ],
                    [
                        [
                            arm,
                            *(
                                data["observability"][arm][k]
                                for k in (
                                    "actions",
                                    "actions_with_saved_result",
                                    "unreconstructed_origins",
                                    "version_or_old_answer_metadata",
                                )
                            ),
                        ]
                        for arm in ARMS
                    ],
                ),
                "",
            ]
            for direction, rows in examples[panel][family].items():
                example_lines += [f"## {family} / {direction} ({len(rows)} examples)", ""]
                for item in rows:
                    example_lines += [
                        f"### {item['id']}",
                        "",
                        f"Question: {item['question']}",
                        "",
                        f"Gold: {item['gold']}",
                        "",
                        *(
                            f"{arm}: {item['answers'][arm]['parsed']!r}; "
                            f"scorer rejected={item['answers'][arm]['scorer_rejected_record']}; "
                            f"primary={item['failures'][arm]}"
                            for arm in ("WT", "agent")
                        ),
                        "",
                        "WT path: `"
                        + json.dumps(item["WT_path"], ensure_ascii=False).replace("`", "\\`")
                        + "`",
                        "",
                        "Agent tool calls:",
                        "",
                        "```text",
                        *(
                            f"{call['index']}: {call['excerpt']}"
                            + (
                                f" ... [{call['characters_omitted']} characters omitted]"
                                if call["characters_omitted"]
                                else ""
                            )
                            for call in item["agent_tool_calls"]
                        ),
                        "```",
                        "",
                        f"Omitted calls: {item['tool_calls_omitted']}. Sources: "
                        + "; ".join(f"{k}={v}" for k, v in item["sources"].items()),
                        "",
                    ]
    (out / "report.md").write_text(_summary(result, provenance))
    (out / "paired.md").write_text("\n".join(paired_lines))
    (out / "strata.md").write_text(
        (
            "# Descriptive strata\n\nSparse means fewer than ten clusters; "
            "their intervals can be degenerate. All intervals are "
            "pointwise and unadjusted. Answer-surface types are lexical "
            "patterns, not semantic classes. Update status and semantic "
            "answer type remain not_recorded unless explicitly annotated. "
            "POP F/R are released per-hop popularity tags; TI O/N are "
            "old/new fact tags. Types-length mismatches remain separate. "
            "No sitelink bucket or temporal-update label is inferred. Full "
            "per-arm usage and all pairwise CIs are in tables.json.\n\n"
        )
        + "\n".join(strata_lines)
    )
    (out / "taxonomy.md").write_text(
        (
            "# Deterministic taxonomy\n\nRules and precedence are in "
            "taxonomy-rules.json and src/xgraph/error_analysis.py, with "
            "synthetic tests in tests/test_mintqa_rematch.py. The first "
            "relation mismatch is compared with the released gold chain; "
            "an alternative valid route can be correct. These are "
            "observable signatures, not causal or human annotations. Agent "
            "paths use the last saved action supporting the answer, else "
            "the last action with saved endpoints, else the last attempted "
            "action. Only saved results connect intermediate entities into "
            "a longer path. Entity aliases come from indexed read-only "
            "SQLite name queries; relation aliases come from released "
            "relation labels. No KG trajectory replay is performed.\n\n"
        )
        + _table(["Panel", "Family", "Arm", "Top three primary errors"], top)
        + "\n\n"
        + "\n".join(taxonomy_lines)
    )
    (out / "examples.md").write_text("\n".join(example_lines))
    launcher = out / "regenerate.sh"
    launcher.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd "
        + shlex.quote(str(WORKTREE))
        + "\nexec "
        + shlex.join(provenance["command"])
        + "\n"
    )
    launcher.chmod(0o755)
    write_json(
        out / "argv.json",
        {
            "command": provenance["command"],
            "worker_commands": provenance["worker_commands"],
            "gpu_hours": 0,
            "gpu_queue": None,
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/home/jonas/git/xgraph"))
    parser.add_argument("--sealed-root", type=Path)
    parser.add_argument("--budget-root", type=Path)
    parser.add_argument("--out", type=Path, default=WORKTREE / "target/error-analysis")
    parser.add_argument("--scorer-worker", action="store_true")
    parser.add_argument("--panel", choices=("validation", "analysis-dev"))
    parser.add_argument("--family", choices=FAMILIES)
    args = parser.parse_args(argv)
    args.source_root, args.out = args.source_root.resolve(), args.out.resolve()
    args.sealed_root = (
        args.sealed_root
        or args.source_root / ".claude/worktrees/codex-final/target/final-validation"
    ).resolve()
    args.budget_root = (
        args.budget_root
        or args.source_root / ".claude/worktrees/codex-budget-frontier/target/budget-frontier"
    ).resolve()
    if not args.out.is_relative_to(WORKTREE / "target"):
        parser.error("--out must be inside this worktree's target directory")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.scorer_worker:
        if not args.panel or not args.family:
            parser.error("scorer worker requires --panel and --family")
        scorer_worker(args)
    else:
        build_report(args)


if __name__ == "__main__":
    main()
