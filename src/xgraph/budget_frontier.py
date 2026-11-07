"""CPU accounting and interpolation for a measured accuracy-cost frontier."""

from __future__ import annotations

import math
from collections import Counter
from copy import deepcopy
from itertools import pairwise


def audit_agent_budget(baseline, candidate, baseline_summary, candidate_summary, budget):
    """Require matched agent runs, distinguishing argv order from option values.

    Only the tool budget, output path and creation time may change. The native
    summaries' runtime totals vary; their generation settings must agree.
    """
    configurations, provenance, commands, ordered = [], [], [], []
    for run, expected in ((baseline, 8), (candidate, budget)):
        config = deepcopy(run)
        source = config.pop("provenance")
        argv = source.pop("command")
        source.pop("created_at")
        if config["agent"].pop("calls") != expected:
            raise ValueError("agent budget differs from declared budget")
        options, normalized = {}, list(argv[:2])
        i = 2
        while i < len(argv):
            flag = argv[i]
            if not flag.startswith("--") or flag in options:
                raise ValueError(f"unexpected or duplicate argv option: {flag}")
            i += 1
            values = []
            while i < len(argv) and not argv[i].startswith("--"):
                values.append(argv[i])
                i += 1
            options[flag] = values
            normalized.extend(
                [flag, *(["<allowed>"] if flag in {"--out", "--max-calls"} else values)]
            )
        if options.pop("--max-calls", None) != [str(expected)]:
            raise ValueError("argv budget differs from declared budget")
        if len(options.pop("--out", [])) != 1:
            raise ValueError("argv needs exactly one output path")
        configurations.append(config)
        provenance.append(source)
        commands.append((argv[:2], options))
        ordered.append(normalized)
    if commands[0] != commands[1]:
        raise ValueError("agent argv differs beyond budget, output and option order")
    if configurations[0] != configurations[1]:
        raise ValueError("agent configuration differs beyond budget")
    if provenance[0] != provenance[1]:
        raise ValueError("agent provenance inputs, versions or generation commit differ")
    for key in (
        "artifact_kind",
        "version",
        "arm",
        "requests",
        "max_tokens",
        "repetition_penalty",
        "banned_tokens",
        "opener",
        "engine_args",
        "vllm_version",
        "timing_kind",
        "cache_state",
    ):
        if baseline_summary.get(key) != candidate_summary.get(key):
            raise ValueError(f"generation summary {key} differs")
    return {
        "status": "passed",
        "reference_budget": 8,
        "candidate_budget": budget,
        "argv_identical_after_budget_and_output_substitution": ordered[0] == ordered[1],
        "parsed_argv_identical_except_budget_and_output": True,
        "same_configuration_inputs_versions_and_commit": True,
    }


def agent_budget_diagnostics(records, members, budget, *, failed, final_answer):
    """Audit plain-agent usage, retaining failures under the supplied scorer.

    A tool cap is separate from model rounds and the recorded budget-spent flag.
    Plain-agent records omit tool execution results, so execution failures stay
    unknown unless every action carries its result. Percentiles use linear ranks.
    """
    ids = {m["id"] for m in members}
    if not ids or len(ids) != len(members) or set(records) != ids:
        raise ValueError("agent record ids must equal unique panel ids")
    if type(budget) is not int or budget <= 0:
        raise ValueError("agent budget must be a positive integer")
    if any(not math.isfinite(m["weight"]) or m["weight"] <= 0 for m in members):
        raise ValueError("members need positive finite weights")
    mass = sum(m["weight"] for m in members)
    by_question = {}
    length_reasons = {"length", "max_tokens"}
    for qid, row in records.items():
        calls, rounds, turns = row["tool_calls"], row["rounds"], row["turns"]
        if type(calls) is not int or not 0 <= calls <= budget or calls != len(row["actions"]):
            raise ValueError(f"{qid}: invalid tool_calls")
        if type(rounds) is not int or rounds < 1 or rounds != len(turns):
            raise ValueError(f"{qid}: rounds differ from actual turns")
        for field in ("prompt_tokens", "generated_tokens"):
            if row[field] != sum(t[field] for t in turns):
                raise ValueError(f"{qid}: {field} differs from turn usage")
        lengths = sum(t.get("finish_reason") in length_reasons for t in turns)
        by_question[qid] = {
            "tool_calls": calls,
            "model_calls": rounds,
            "at_tool_cap": calls == budget,
            "budget_spent_flag": bool(row["budget_spent"]),
            "scorer_failures": bool(failed(row)),
            "missing_parsed_answer": not bool(final_answer(row["text"])),
            "length_limited_questions": lengths > 0,
            "length_limited_turns": lengths,
            "final_length_limited": row.get("finish_reason") in length_reasons,
            "invalid_actions": sum(a["tool"] == "invalid" for a in row["actions"]),
            "prompt_tokens": row["prompt_tokens"],
            "generated_tokens": row["generated_tokens"],
            "total_tokens": row["prompt_tokens"] + row["generated_tokens"],
        }

    def distribution(field):
        values = sorted(r[field] for r in by_question.values())

        def percentile(p):
            rank = (len(values) - 1) * p
            lo, hi = math.floor(rank), math.ceil(rank)
            return values[lo] + (values[hi] - values[lo]) * (rank - lo)

        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "weighted_mean": sum(m["weight"] * by_question[m["id"]][field] for m in members) / mass,
            "p50": percentile(0.5),
            "p95": percentile(0.95),
            "min": values[0],
            "max": values[-1],
            "histogram": {str(k): v for k, v in sorted(Counter(values).items())},
        }

    def tally(field):
        count = sum(r[field] for r in by_question.values())
        return {
            "n": count,
            "rate": count / len(members),
            "weighted_rate": sum(m["weight"] * by_question[m["id"]][field] for m in members) / mass,
        }

    actions = [a for row in records.values() for a in row["actions"]]
    tool_failures = (
        sum(
            bool(a.get("failed") or a["result"].get("error") or a["result"].get("count") == 0)
            for a in actions
        )
        if all("result" in a for a in actions)
        else None
    )
    return {
        "n": len(members),
        "budget": budget,
        "tool_calls": distribution("tool_calls"),
        "model_calls": distribution("model_calls"),
        "tokens": {
            field: distribution(field)
            for field in ("prompt_tokens", "generated_tokens", "total_tokens")
        },
        **{
            field: tally(field)
            for field in (
                "at_tool_cap",
                "budget_spent_flag",
                "scorer_failures",
                "missing_parsed_answer",
                "length_limited_questions",
                "final_length_limited",
            )
        },
        "length_limited_turns": sum(r["length_limited_turns"] for r in by_question.values()),
        "invalid_actions": sum(r["invalid_actions"] for r in by_question.values()),
        "tool_execution_failures": tool_failures,
        "tool_execution_failures_note": "unknown when plain-agent actions omit tool results",
        "finish_reasons": dict(
            sorted(Counter(str(r.get("finish_reason")) for r in records.values()).items())
        ),
        "turn_finish_reasons": dict(
            sorted(
                Counter(
                    str(t.get("finish_reason")) for r in records.values() for t in r["turns"]
                ).items()
            )
        ),
        "by_question": by_question,
    }


def log_budget_interpolation(points, budget):
    """Interpolate a cumulative-maximum envelope, without lowering raw accuracy.

    Tied costs retain their best accuracy. Below the cheapest observation there is
    no estimate; above the most expensive one the best observed policy remains
    affordable, without predicting any unmeasured improvement. A zero-only axis
    is a constant; interpolation between zero and a positive cost is undefined.
    """
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("cost budget must be finite and nonnegative")
    best = {}
    for cost, accuracy in points:
        if (
            not math.isfinite(cost)
            or cost < 0
            or not math.isfinite(accuracy)
            or not 0 <= accuracy <= 1
        ):
            raise ValueError("points need finite nonnegative costs and accuracies in [0, 1]")
        best[cost] = max(best.get(cost, 0), accuracy)
    if not best:
        raise ValueError("interpolation needs measured points")
    knots, running = [], 0
    for cost, accuracy in sorted(best.items()):
        running = max(running, accuracy)
        knots.append([cost, running])
    result = {"knots": knots, "accuracy": None, "bracket": None, "fraction": None}
    if budget < knots[0][0]:
        return {**result, "status": "below_minimum"}
    for cost, accuracy in knots:
        if budget == cost:
            return {**result, "accuracy": accuracy, "bracket": [cost, cost], "status": "exact"}
    if budget > knots[-1][0]:
        return {**result, "accuracy": knots[-1][1], "status": "best_observed_plateau"}
    for (lo, a), (hi, b) in pairwise(knots):
        if lo < budget < hi:
            if lo == 0:
                return {**result, "bracket": [lo, hi], "status": "zero_log_gap"}
            fraction = math.log(budget / lo) / math.log(hi / lo)
            return {
                **result,
                "accuracy": a + fraction * (b - a),
                "fraction": fraction,
                "bracket": [lo, hi],
                "status": "interpolated",
            }
    raise AssertionError("validated budget must have a bracket")


def aggregate_costs(stages, summaries, members):
    """Sum native usage stages once per question; wall time stays unweighted."""
    ids = {m["id"] for m in members}
    if not ids or len(ids) != len(members):
        raise ValueError("members need unique nonempty ids")
    if not stages or len(stages) != len(summaries):
        raise ValueError("each generation stage needs one summary")
    if any(not math.isfinite(m["weight"]) or m["weight"] <= 0 for m in members):
        raise ValueError("members need positive finite weights")
    fields = ("model_calls", "prompt_tokens", "generated_tokens", "observation_tokens")
    by_question = {m["id"]: dict.fromkeys(fields, 0) for m in members}
    seconds = 0
    for stage, summary in zip(stages, summaries, strict=True):
        if set(stage) != ids:
            raise ValueError("generation stage ids differ from panel ids")
        for qid, row in stage.items():
            for field in fields:
                value = row.get(field)
                if type(value) is not int or value < 0:
                    raise ValueError(f"{qid}: invalid {field}: {value}")
                by_question[qid][field] += value
            if row["observation_tokens"] > row["generated_tokens"]:
                raise ValueError(f"{qid}: observation_tokens exceeds generated_tokens")
        wall = summary.get("seconds")
        if not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall < 0:
            raise ValueError("generation summary needs finite nonnegative seconds")
        if summary.get("requests", len(ids)) != len(ids):
            raise ValueError("generation summary requests differ from panel ids")
        for field in fields:
            if field in summary and summary[field] != sum(row[field] for row in stage.values()):
                raise ValueError(f"generation summary {field} differs from record total")
        seconds += wall
    for row in by_question.values():
        row["total_tokens"] = row["prompt_tokens"] + row["generated_tokens"]
        row["non_observation_generated_tokens"] = (
            row["generated_tokens"] - row["observation_tokens"]
        )
    all_fields = (*fields, "non_observation_generated_tokens", "total_tokens")
    totals = {k: sum(row[k] for row in by_question.values()) for k in all_fields}
    mass = sum(m["weight"] for m in members)
    weighted = {
        k: sum(m["weight"] * by_question[m["id"]][k] for m in members) / mass for k in all_fields
    }
    for row in by_question.values():
        row["workload_wall_seconds"] = None
    return {
        "totals": {**totals, "workload_wall_seconds": seconds},
        "per_question": {
            **{k: v / len(ids) for k, v in totals.items()},
            "workload_wall_seconds": seconds / len(ids),
        },
        "weighted_per_question": {**weighted, "workload_wall_seconds": None},
        "by_question": by_question,
        "timing_label": "uncontrolled-cache workload wall",
        "timing_allocation": "sum of stage summary seconds / panel size; question timings unknown",
    }


def compare_measured_agents(points, cost, accuracy):
    """Distinguish an interpolated advantage from dominance below measured support."""
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError("policy accuracy must be finite and in [0, 1]")
    points = list(points)
    estimate = log_budget_interpolation(points, cost)
    delta = None if estimate["accuracy"] is None else 100 * (accuracy - estimate["accuracy"])
    dominates = all(cost < c and accuracy > a for c, a in points)
    return {
        "above_measured_envelope": delta > 0 if delta is not None else True if dominates else None,
        "interpolated_advantage_pp": delta,
        "basis": (
            "measured cost envelope"
            if delta is not None
            else "higher accuracy and cheaper than every measured agent"
            if dominates
            else "outside measured interpolation support"
        ),
    }
