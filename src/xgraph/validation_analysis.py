"""Descriptive analysis of completed MINTQA scoring; never generates or rescores."""

from __future__ import annotations

import argparse
import ast
import functools
import gc
import hashlib
import importlib.metadata
import json
import math
import re
import resource
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

COST_FIELDS = (
    "prompt_tokens",
    "generated_tokens",
    "observation_tokens",
    "model_calls",
    "total_tokens",
)


def aligned_rows(first, second):
    a, b = ({r["id"]: r for r in rows} for rows in (first, second))
    if len(a) != len(first) or len(b) != len(second):
        raise ValueError("duplicate question ids")
    if not a or a.keys() != b.keys():
        raise ValueError("paired questions must be nonempty and aligned")
    for qid, row in a.items():
        if row["weight"] != b[qid]["weight"]:
            raise ValueError(f"{qid}: paired weights differ")
        if not math.isfinite(row["weight"]) or row["weight"] <= 0:
            raise ValueError(f"{qid}: weights must be finite and positive")
        if any(type(r["correct"]) is not bool for r in (row, b[qid])):
            raise ValueError(f"{qid}: correctness must be a scorer boolean")
    return [(row, b[qid]) for qid, row in a.items()]


def transition_matrix(first, second):
    """Summarize paired scorer-of-record correctness flags and survey weights."""
    pairs = aligned_rows(first, second)
    counts, weights = {}, {}
    for before in (False, True):
        for after in (False, True):
            name = f"{'right' if before else 'wrong'}_to_{'right' if after else 'wrong'}"
            selected = [a for a, b in pairs if (a["correct"], b["correct"]) == (before, after)]
            counts[name] = len(selected)
            weights[name] = sum(r["weight"] for r in selected)
    mass = sum(weights.values())
    return {
        "questions": len(pairs),
        "weight_sum": mass,
        "counts": counts,
        "weights": weights,
        "weighted_shares": {k: v / mass for k, v in weights.items()},
        "unweighted_shares": {k: v / len(pairs) for k, v in counts.items()},
        "difference": (weights["wrong_to_right"] - weights["right_to_wrong"]) / mass,
    }


def cost_per_correct(rows):
    """Charge all questions, dividing total cost by correct answers."""
    aligned_rows(rows, rows)
    for row in rows:
        if any(type(row[k]) is not int or row[k] < 0 for k in COST_FIELDS):
            raise ValueError(f"{row['id']}: invalid cost")
        if row["observation_tokens"] > row["generated_tokens"]:
            raise ValueError(f"{row['id']}: observation tokens exceed generated tokens")
        if row["total_tokens"] != row["prompt_tokens"] + row["generated_tokens"]:
            raise ValueError(f"{row['id']}: total tokens do not conserve usage")
    mass = sum(r["weight"] for r in rows)
    correct_mass = sum(r["weight"] for r in rows if r["correct"])
    correct_count = sum(r["correct"] for r in rows)
    weighted = {k: sum(r["weight"] * r[k] for r in rows) for k in COST_FIELDS}
    totals = {k: sum(r[k] for r in rows) for k in COST_FIELDS}
    return {
        "questions": len(rows),
        "correct_count": correct_count,
        "correct_weight": correct_mass,
        "accuracy": correct_mass / mass,
        "weighted_totals": weighted,
        "totals": totals,
        "weighted_per_question": {k: v / mass for k, v in weighted.items()},
        "unweighted_per_question": {k: v / len(rows) for k, v in totals.items()},
        "weighted_per_correct": {
            k: v / correct_mass if correct_mass else None for k, v in weighted.items()
        },
        "unweighted_per_correct": {
            k: v / correct_count if correct_count else None for k, v in totals.items()
        },
    }


def shared_shrinkage(development, validation):
    """Every pair among the same policy recipes; no validation-based selection."""
    result = {}
    shared = [key for key in development if key in validation]
    for first, second in combinations(shared, 2):
        dev, val = (scores[first] - scores[second] for scores in (development, validation))
        result[f"{first} - {second}"] = {
            "first": first,
            "second": second,
            "development_difference": dev,
            "validation_difference": val,
            "shrinkage": dev - val,
            "validation_minus_development": val - dev,
        }
    return result


FAMILIES = ("qwen38", "gemma31")
POLICIES = {
    "WT": "finalist",
    "W": "W",
    "B2": "B2",
    "M2": "menuobs-two-pass",
    "incumbent": "menuobs-audit",
    "agent8": "agent-strong-8",
    "feedback8": "agent-feedback-8",
}
WORKTREE = Path(__file__).resolve().parents[2]
TAXONOMY_PRIORITY = (
    "answer_format_or_containment",
    "multi_valued_endpoint_choice",
    "wrong_start_entity",
    "stopped_early",
    "stopped_late",
    "wrong_or_missing_relation",
    "agent_budget_exhaustion",
    "agent_tool_error",
    "unattributed_or_no_plan",
)


class Inputs:
    """Stream hashes and retain exact file identities without copying data."""

    def __init__(self):
        self.files = {}

    def digest(self, path):
        path = Path(path).resolve()
        if str(path) not in self.files:
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
            self.files[str(path)] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": hasher.hexdigest(),
            }
        return self.files[str(path)]

    def read_json(self, path):
        self.digest(path)
        return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def close_enough(actual, expected, context):
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"identity failure: {context}: {actual} != {expected}")


def source_gold(source_root, members, inputs):
    """Parse only selected original release rows, never a sealed gold or panel file."""
    result = {}
    for split in ("pop", "ti"):
        wanted = {
            int(r["id"].rsplit("-", 1)[1]): r["id"]
            for r in members
            if r["id"].startswith(f"mintqa-{split}-")
        }
        path = source_root / f"data/external/mintqa-full/questions/{split}.jsonl"
        inputs.digest(path)
        with path.open() as stream:
            for index, line in enumerate(stream):
                if index in wanted:
                    row = json.loads(line)
                    result[wanted[index]] = {
                        k: row[k] for k in ("question", "answer", "triple", "type")
                    } | {"_source": f"{path}:{index + 1}"}
    if result.keys() != {r["id"] for r in members}:
        raise ValueError("original release does not cover scorer ids")
    return result


def read_records(path, members, inputs):
    inputs.digest(path)
    rows = {}
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            if row["id"] in rows:
                raise ValueError(f"duplicate id in {path}:{number}")
            rows[row["id"]] = {**row, "_source": f"{path}:{number}"}
    if rows.keys() != {r["id"] for r in members}:
        raise ValueError(f"incomplete or unexpected ids: {path}")
    return rows


def member_groups(members, gold):
    from xgraph.mintqa_bench import stratum

    groups = defaultdict(lambda: defaultdict(list))
    for row in members:
        qid, hops = row["id"], row["hops"]
        split = qid.split("-")[1]
        if len(gold[qid]["triple"]) != hops:
            raise ValueError(f"{qid}: released chain length disagrees with scorer")
        group = f"{split} {hops}-hop {stratum(split, gold[qid]['type'], hops)}"
        for dimension, name in (
            ("overall", "all"),
            ("by_hop", str(hops)),
            ("by_split", split),
            ("by_group", group),
            ("by_split_hop", f"{split} {hops}-hop"),
            ("by_hop_band", "1-hop" if hops == 1 else "multi-hop"),
        ):
            groups[dimension][name].append(qid)
    return {key: dict(value) for key, value in groups.items()}


def paired_table(first, second, *, resamples, seed, total_mass):
    from xgraph.mintqa_rematch import clustered_difference

    pairs = aligned_rows(first, second)
    if any(any(a[k] != b[k] for k in ("hops", "cluster")) for a, b in pairs):
        raise ValueError("paired clusters and hop strata differ")
    interval = clustered_difference(
        [a["correct"] for a, _ in pairs],
        [b["correct"] for _, b in pairs],
        first,
        resamples,
        seed,
    )
    mass = sum(r["weight"] for r in first)
    return {
        **interval,
        "questions": len(first),
        "weight_sum": mass,
        "population_share": mass / total_mass,
        "first_accuracy": sum(a["weight"] * a["correct"] for a, _ in pairs) / mass,
        "second_accuracy": sum(b["weight"] * b["correct"] for _, b in pairs) / mass,
        "contribution": interval["difference"] * mass / total_mass,
        "sparse_cluster_caveat": interval["clusters"] < 10,
    }


def validate_report(report, groups):
    """Reproduce saved aggregates using only their own per-question flags/costs."""
    tables = report["panels"]["validation"]
    master = tables["finalist"]["per_question"]
    for arm, table in tables.items():
        rows = table["per_question"]
        pairs = aligned_rows(master, rows)
        if any(any(a[k] != b[k] for k in ("hops", "cluster")) for a, b in pairs):
            raise ValueError(f"{arm}: scorer metadata differs")
        costs = cost_per_correct(rows)
        close_enough(costs["accuracy"], table["accuracy"], f"{arm} accuracy")
        for key, value in costs["weighted_per_question"].items():
            close_enough(value, table["cost"]["weighted_per_question"][key], f"{arm} {key}")
        indexed = {r["id"]: r for r in rows}
        for dimension in ("by_hop", "by_split", "by_group"):
            if groups[dimension].keys() != table[dimension].keys():
                raise ValueError(f"{arm}: group membership mismatch for {dimension}")
            for name, ids in groups[dimension].items():
                saved = table[dimension][name]
                chosen = [indexed[qid] for qid in ids]
                if saved["questions"] != len(chosen):
                    raise ValueError(f"{arm}: group count mismatch")
                mass = sum(r["weight"] for r in chosen)
                close_enough(mass, saved["weight_sum"], f"{arm}/{name} mass")
                close_enough(
                    sum(r["weight"] * r["correct"] for r in chosen) / mass,
                    saved["accuracy"],
                    f"{arm}/{name} accuracy",
                )
    return {"all_saved_accuracy_group_and_cost_aggregates_reproduced": True}


def second_call_analysis(first, final, records, raw, *, first_source, raw_source):
    from xgraph.mintqa_rematch import combined_usage
    from xgraph.programs import record_path
    from xgraph.prompts import final_answer

    indexed = {r["id"]: r for r in first} if first is not None else {}
    mass = sum(r["weight"] for r in final)
    flags = defaultdict(list)
    stage_rows = {"first": [], "second": []}
    per_question = []
    for row in final:
        qid = row["id"]
        before, after, second = first_source[qid], records[qid], raw[qid]
        actual = combined_usage(before, second)
        if any(actual[k] != row[k] for k in actual):
            raise ValueError(f"{qid}: multistage costs differ from sealed report")
        if row["model_calls"] != 2 or type(after.get("revision_accepted")) is not bool:
            raise ValueError(f"{qid}: unconditional revision ledger missing")
        accepted = after["revision_accepted"]
        selected = second if accepted else before
        if after["text"] != selected["text"]:
            raise ValueError(f"{qid}: selected answer does not match saved acceptance")
        item = {
            "id": qid,
            "weight": row["weight"],
            "second_call_executed": True,
            "revision_accepted": accepted,
            "answer_changed": final_answer(after["text"]) != final_answer(before["text"]),
            "start_or_relation_labels_changed": record_path(after) != record_path(before),
            "before_correct": indexed[qid]["correct"] if indexed else None,
            "after_correct": row["correct"],
            "scorer_escalated_field": row["escalated"],
        }
        for key in (
            "second_call_executed",
            "revision_accepted",
            "answer_changed",
            "start_or_relation_labels_changed",
        ):
            if item[key]:
                flags[key].append(row)
        for stage, native in (("first", before), ("second", second)):
            from xgraph.mintqa_rematch import usage

            cost = usage(native)
            cost["total_tokens"] = cost["prompt_tokens"] + cost["generated_tokens"]
            stage_rows[stage].append({**row, **cost})
        per_question.append(item)
    return {
        "definition": (
            "unconditional second call; acceptance means usable supported output, not correctness"
        ),
        "raw_second_source": str(raw_source),
        "rates": {
            key: {
                "count": len(flags[key]),
                "rate": len(flags[key]) / len(final),
                "weighted_rate": sum(r["weight"] for r in flags[key]) / mass,
            }
            for key in (
                "second_call_executed",
                "revision_accepted",
                "answer_changed",
                "start_or_relation_labels_changed",
            )
        },
        "conditional_escalation": {"applicable": False, "reason": "two calls on every question"},
        "stages": {
            key: {
                "weighted_per_question": cost_per_correct(rows)["weighted_per_question"],
                "per_question": [{k: r[k] for k in ("id", "weight", *COST_FIELDS)} for r in rows],
            }
            for key, rows in stage_rows.items()
        },
        "per_question": per_question,
    }


def legacy_taxonomy(source_root, inputs):
    """Reuse exact diagnostic definitions without importing their old audit hook or main."""
    from xgraph.agent import KGTools, MatchedPaths, relation_names_of
    from xgraph.index import load_kg_index
    from xgraph.mintqa_bench import answer_contains
    from xgraph.observations import PathRelations
    from xgraph.programs import MatchedExecutor, record_path
    from xgraph.prompts import final_answer
    from xgraph.score import normalize, record_failure
    from xgraph.tokenizer import ServedTokenizer
    from xgraph.toolkg import KGStore

    namespace = {
        "ROOT": source_root,
        "Path": Path,
        "digest": inputs.digest,
        "functools": functools,
        "re": re,
        "KGTools": KGTools,
        "MatchedPaths": MatchedPaths,
        "relation_names_of": relation_names_of,
        "load_kg_index": load_kg_index,
        "answer_contains": answer_contains,
        "PathRelations": PathRelations,
        "MatchedExecutor": MatchedExecutor,
        "record_path": record_path,
        "final_answer": final_answer,
        "normalize": normalize,
        "record_failure": record_failure,
        "ServedTokenizer": ServedTokenizer,
        "KGStore": KGStore,
    }
    selected = {}
    for filename, names in (
        ("analyze.py", {"TraceReader", "mean", "tally", "taxonomy_tables", "LABELS"}),
        ("refine_starts.py", {"gold_roots", "committed_root"}),
    ):
        path = source_root / "target/analysis-dev" / filename
        inputs.digest(path)
        tree = ast.parse(path.read_text(), filename=str(path))
        nodes = [
            node
            for node in tree.body
            if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names)
            or (
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
            )
        ]
        selected[filename] = hashlib.sha256(
            ast.dump(ast.Module(body=nodes, type_ignores=[])).encode()
        ).hexdigest()
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, selected


def verify_taxonomy_cache(cached, inputs, *, driver_path=Path(__file__)):
    """Reject diagnostics made with changed inputs or a different analysis driver."""
    if cached["provenance"]["driver_sha256"] != inputs.digest(driver_path)["sha256"]:
        raise ValueError("taxonomy driver changed; recompute the automated diagnostics")
    for item in cached["provenance"]["inputs"]:
        if inputs.digest(item["path"])["sha256"] != item["sha256"]:
            raise ValueError(f"taxonomy input changed: {item['path']}")


def taxonomy_worker(args, inputs):
    started = time.monotonic()
    driver_identity = inputs.digest(Path(__file__))
    inputs.digest(WORKTREE / "scripts/mintqa/validation-analysis.py")
    inputs.digest(WORKTREE / "uv.lock")
    report = inputs.read_json(args.sealed_root / args.taxonomy_worker / "report.json")
    tables = report["panels"]["validation"]
    members = tables["finalist"]["per_question"]
    gold = source_gold(args.source_root, members, inputs)
    namespace, definitions = legacy_taxonomy(args.source_root, inputs)
    published = inputs.read_json(args.source_root / "target/analysis-dev/provenance.json")
    # All originally recorded trace-reader dependencies must still have their published hashes.
    source_checks = []
    for item in published["inputs"]:
        if "/src/xgraph/" in item["path"] or item["path"].endswith("analysis-dev/analyze.py"):
            actual = inputs.digest(args.source_root / item["path"])
            if actual["sha256"] != item["sha256"]:
                raise ValueError(f"development taxonomy source changed: {item['path']}")
            source_checks.append(actual)
    for name, module in list(sys.modules.items()):
        if name.startswith("xgraph") and getattr(module, "__file__", None):
            expected = args.source_root / ".claude/worktrees/codex-integrate/src"
            if not Path(module.__file__).resolve().is_relative_to(expected):
                raise ValueError(f"taxonomy imported a different checkout: {module.__file__}")
            inputs.digest(module.__file__)
    relation_path = args.source_root / "data/external/mintqa-full/kg/relation_labels.jsonl"
    inputs.digest(relation_path)
    with relation_path.open() as stream:
        relation_rows = [json.loads(line) for line in stream]
    inputs.digest(args.source_root / "data/external/mintqa-full/kg.sqlite")
    reader = namespace["TraceReader"](args.taxonomy_worker, relation_rows)
    base = args.sealed_root / args.taxonomy_worker / "validation"
    strongest = "agent-strong-8" if args.taxonomy_worker == "qwen38" else "agent-feedback-8"
    result = {"initial": {}, "refined": {}, "per_question": {}, "support_checks": []}
    saved_support = read_records(base / "W-revision-terminal-support.jsonl", members, inputs)
    raw = read_records(base / "W-revision-terminal-raw.jsonl", members, inputs)
    raw_paths = {qid: namespace["record_path"](row) for qid, row in raw.items()}
    for policy, arm in (("WT", "finalist"), ("strongest", strongest)):
        records = read_records(base / f"{arm}.jsonl", members, inputs)
        correctness = {r["id"]: r["correct"] for r in tables[arm]["per_question"]}
        diagnosed = {}
        for i, row in enumerate(members):
            qid = row["id"]
            diagnosed[qid] = reader.diagnose(
                records[qid], gold[qid], correctness[qid], agent=policy == "strongest"
            )
            if policy == "WT" and records[qid]["revision_accepted"]:
                replay = reader.support(raw_paths[qid])
                if any(
                    replay[k] != saved_support[qid][k]
                    for k in ("reached", "uncapped_reached", "cap_events")
                ):
                    result["support_checks"].append(qid)
            if (i + 1) % 512 == 0:
                print(f"{args.taxonomy_worker}: automated {policy} taxonomy {i + 1}", flush=True)
        result["initial"][policy] = namespace["taxonomy_tables"](diagnosed, members)
        changes = []
        for qid, item in diagnosed.items():
            if item["correct"]:
                continue
            roots, _ = namespace["gold_roots"](reader, gold[qid])
            committed = namespace["committed_root"](reader, item["plan"])
            before = list(item["automatic_labels"])
            item["flags"].update(
                compatible_gold_start_ids=roots,
                committed_start_ids=committed,
                gold_start_identity_resolved=bool(roots),
                committed_start_excludes_gold=bool(
                    roots and committed and not set(roots) & set(committed)
                ),
            )
            if item["flags"]["committed_start_excludes_gold"]:
                item["automatic_labels"] = sorted(set(before) | {"wrong_start_entity"})
                if "unattributed_or_no_plan" in item["automatic_labels"]:
                    item["automatic_labels"].remove("unattributed_or_no_plan")
            item["primary_label"] = next(
                k for k in TAXONOMY_PRIORITY if k in item["automatic_labels"]
            )
            if before != item["automatic_labels"]:
                changes.append(qid)
        result["refined"][policy] = {
            **namespace["taxonomy_tables"](diagnosed, members),
            "refinement_changed_ids": changes,
        }
        result["per_question"][policy] = {
            qid: {
                k: item[k]
                for k in (
                    "correct",
                    "automatic_labels",
                    "primary_label",
                    "flags",
                    "plan",
                    "source",
                    "gold_source",
                )
            }
            for qid, item in diagnosed.items()
        }
        if policy == "strongest":
            actions = [a for row in records.values() for a in row["actions"]]
            result["agent_result_audit"] = {
                "actions_replayed": len(actions),
                "actions_with_saved_result": sum("result" in a for a in actions),
                "limit": "only action entries containing result permit a saved-result comparison",
            }
            result["agent_replay_mismatches"] = {
                qid: d["trace"]["recorded_result_mismatches"]
                for qid, d in diagnosed.items()
                if d["trace"]["recorded_result_mismatches"]
            }
    if result["support_checks"] or result["agent_replay_mismatches"]:
        raise ValueError("taxonomy execution replay mismatch; refusing descriptive attribution")
    reader.store._con.close()
    for method in (reader.named, reader.label, reader.relation_ids):
        method.cache_clear()
    del reader
    gc.collect()
    result["provenance"] = {
        "family": args.taxonomy_worker,
        "driver_sha256": driver_identity["sha256"],
        "definition_ast_sha256": definitions,
        "source_checks": source_checks,
        "inputs": list(inputs.files.values()),
        "seconds": time.monotonic() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "correctness": "sealed report flags, passed to original diagnose; never recomputed",
        "manual_review": False,
    }
    write_json(args.out / f"taxonomy-{args.taxonomy_worker}.json", result)


def analysis_family(report, freeze, gold, inputs, args, development, frontier):
    family = report["family"]
    tables = report["panels"]["validation"]
    members = tables["finalist"]["per_question"]
    groups = member_groups(members, gold)
    identity = validate_report(report, groups)
    if (
        report["freeze_sha256"]
        != inputs.digest(args.sealed_root / "freeze-manifest.json")["sha256"]
    ):
        raise ValueError("sealed report and freeze identity differ")
    if freeze["families"][family]["selected_policy"]["policy"] != "WT":
        raise ValueError("unexpected frozen finalist")
    resamples, seed = report["registration"]["resamples"], report["registration"]["seed"]
    strongest = "agent8" if family == "qwen38" else "feedback8"
    if freeze["strongest_agents"][family] != POLICIES[strongest]:
        raise ValueError("strongest comparator differs from frozen choice")
    policy_rows = {key: tables[arm]["per_question"] for key, arm in POLICIES.items()}
    indexed = {key: {r["id"]: r for r in rows} for key, rows in policy_rows.items()}
    mass = sum(r["weight"] for r in members)
    comparisons = {}
    for reference in (strongest, "W", "B2"):
        name = "WT - " + ("strongest" if reference == strongest else reference)
        comparisons[name] = {
            dimension: {
                group: paired_table(
                    [indexed["WT"][qid] for qid in ids],
                    [indexed[reference][qid] for qid in ids],
                    resamples=resamples,
                    seed=seed,
                    total_mass=mass,
                )
                for group, ids in strata.items()
            }
            for dimension, strata in groups.items()
        }
    f = comparisons["WT - strongest"]["overall"]["all"]
    for key in ("difference", "low", "high"):
        close_enough(f[key], report["amended_comparisons"]["F"][key], f"{family}/F/{key}")
    identity["frozen_F_point_and_CI_reproduced"] = True
    if family == "qwen38":
        for dimension in ("by_hop", "by_split", "by_group"):
            for name, item in comparisons["WT - strongest"][dimension].items():
                for key in ("difference", "low", "high"):
                    close_enough(
                        item[key],
                        tables["finalist"][dimension][name]["vs_agent"][key],
                        f"{family}/{dimension}/{name}/{key}",
                    )
        identity["saved_stratum_CIs_reproduced"] = True
    costs = {}
    for key in ("WT", "W", "B2", "agent8", "feedback8"):
        costs[key] = {
            **cost_per_correct(policy_rows[key]),
            "per_question": policy_rows[key],
            "by_hop": {
                name: cost_per_correct([indexed[key][qid] for qid in ids])
                for name, ids in groups["by_hop"].items()
            },
        }
    base = args.sealed_root / family / "validation"
    records = {
        key: read_records(base / f"{POLICIES[key]}.jsonl", members, inputs)
        for key in ("WT", "W", "B2", "agent8", "feedback8")
    }
    from xgraph.mintqa_rematch import audit_complete_usage, usage

    for key, arm in records.items():
        for qid, row in arm.items():
            if any(value != indexed[key][qid][field] for field, value in usage(row).items()):
                raise ValueError(f"{family}/{key}/{qid}: raw and report usage differ")
        if key in ("agent8", "feedback8"):
            audit_complete_usage(list(arm.values()), list(arm))
    identity["raw_multicall_and_agent_usage_matches_sealed_report"] = True
    second = {}
    for key, before, raw_name in (
        ("WT", "W", "W-revision-terminal-raw"),
        ("B2", "blind5", "blind5-revision-raw"),
    ):
        first = (
            records["W"]
            if before == "W"
            else read_records(base / f"{before}.jsonl", members, inputs)
        )
        raw_path = base / f"{raw_name}.jsonl"
        second[key] = second_call_analysis(
            policy_rows[before] if before == "W" else None,
            policy_rows[key],
            records[key],
            read_records(raw_path, members, inputs),
            first_source=first,
            raw_source=raw_path,
        )
    transitions = {
        dimension: {
            name: transition_matrix(
                [indexed["W"][qid] for qid in ids], [indexed["WT"][qid] for qid in ids]
            )
            for name, ids in strata.items()
        }
        for dimension, strata in groups.items()
    }
    dev_scores = {
        key: development[family][key]["accuracy"] for key in POLICIES if key in development[family]
    }
    for key in POLICIES:
        if key in frontier["families"][family]["policies"]:
            point = frontier["families"][family]["policies"][key]["score"]["accuracy"]
            if key in dev_scores:
                close_enough(point, dev_scores[key], f"{family}/{key} development sources")
            dev_scores[key] = point
    val_scores = {key: tables[arm]["accuracy"] for key, arm in POLICIES.items()}
    return {
        "strongest": strongest,
        "questions": len(members),
        "seed": seed,
        "resamples": resamples,
        "identity": identity,
        "accuracy": val_scores,
        "development_accuracy": dev_scores,
        "comparisons": comparisons,
        "costs": costs,
        "second_call": second,
        "W_to_WT": transitions,
        "shrinkage": shared_shrinkage(dev_scores, val_scores),
        "validation_absent_development_policies": sorted(
            (set(development[family]) | set(frontier["families"][family]["policies"]))
            - set(POLICIES)
            - {"strongest", "strongest_agent"}
        ),
    }


def interval_text(item):
    return f"{100 * item['difference']:+.2f} [{100 * item['low']:+.2f}, {100 * item['high']:+.2f}]"


def markdown_table(headers, rows):
    return "\n".join(
        [
            "|" + "|".join(headers) + "|",
            "|" + "|".join("---" for _ in headers) + "|",
            *("|" + "|".join(str(cell) for cell in row) + "|" for row in rows),
        ]
    )


def render_report(result, provenance):
    families = result["families"]
    lines = [
        "# MINTQA validation: descriptive, post hoc analysis",
        "",
        "Every analysis here is outside the registered Holm family. Accuracy and survey weights "
        "come only from the completed sealed report's per-question rows. No headline was rescored. "
        "MINTQA was exposed during development; this is internal validation, "
        "not a fresh benchmark.",
        "",
        "Pointwise 95% percentile intervals reuse the scorer's paired relation-sequence bootstrap "
        "within hop strata, retaining each stratum's weight mass (10,000 resamples; seed 17). "
        "Subgroup intervals condition on subgroup membership, without multiplicity correction. "
        "Sparse groups can have degenerate intervals; cluster counts are retained in tables.json.",
        "",
        "## Accuracy and source of the advantage",
        "",
    ]
    summary = []
    for family, data in families.items():
        scores, costs = data["accuracy"], data["costs"]
        token_ratio = (
            costs["WT"]["weighted_per_question"]["total_tokens"]
            / costs[data["strongest"]]["weighted_per_question"]["total_tokens"]
        )
        summary.append(
            [
                family,
                *(f"{100 * scores[k]:.2f}" for k in ("WT", data["strongest"], "W", "B2")),
                interval_text(data["comparisons"]["WT - strongest"]["overall"]["all"]),
                f"{token_ratio:.3f}",
            ]
        )
    lines += [
        markdown_table(
            [
                "Family",
                "WT %",
                "Strongest %",
                "W %",
                "B2 %",
                "WT - strongest pp [CI]",
                "Token ratio",
            ],
            summary,
        ),
        "",
        "The frozen strongest agents are Qwen plain-8 and Gemma feedback-8; no comparator "
        "was selected again on validation. Token ratios describe workload, not measured GPU cost.",
        "",
    ]
    for family, data in families.items():
        comparisons = data["comparisons"]["WT - strongest"]
        one, multi = (comparisons["by_hop_band"][name] for name in ("1-hop", "multi-hop"))
        pop, ti = (comparisons["by_split"][name] for name in ("pop", "ti"))
        lines += [
            f"{family}: WT - strongest is {100 * one['difference']:+.2f} pp on 1-hop "
            f"and {100 * multi['difference']:+.2f} pp on multi-hop. Multi-hop contributes "
            f"{100 * multi['contribution']:+.2f} pp to the overall gap, offset by "
            f"{100 * one['contribution']:+.2f} pp on 1-hop. The split gaps are "
            f"{100 * pop['difference']:+.2f} pp (pop), {100 * ti['difference']:+.2f} pp (ti).",
            "",
        ]
    lines += [
        "Qwen's largest hop-specific contribution comes from 4-hop; its ti point gap is larger "
        "than pop. Gemma gains across 2-4 hops and similarly in both splits. These descriptions "
        "do not test an arm-by-split interaction.",
        "",
        markdown_table(
            ["Family", "Overall comparison", "Difference pp [95% CI]"],
            [
                [family, name, interval_text(data["comparisons"][name]["overall"]["all"])]
                for family, data in families.items()
                for name in ("WT - W", "WT - B2")
            ],
        ),
        "",
    ]
    for dimension, title in (
        ("by_hop", "Hop"),
        ("by_split", "Split"),
        ("by_group", "Popularity/age group"),
    ):
        rows = []
        for family, data in families.items():
            for group, item in data["comparisons"]["WT - strongest"][dimension].items():
                rows.append(
                    [
                        family,
                        group,
                        item["questions"],
                        item["clusters"],
                        f"{100 * item['first_accuracy']:.2f}",
                        f"{100 * item['second_accuracy']:.2f}",
                        *(
                            interval_text(data["comparisons"][name][dimension][group])
                            for name in ("WT - strongest", "WT - W", "WT - B2")
                        ),
                    ]
                )
        lines += [
            f"### {title}",
            "",
            markdown_table(
                [
                    "Family",
                    title,
                    "n",
                    "Clusters",
                    "WT %",
                    "Strongest %",
                    "WT - strongest pp [CI]",
                    "WT - W pp [CI]",
                    "WT - B2 pp [CI]",
                ],
                rows,
            ),
            "",
        ]
    contributions = []
    for family, data in families.items():
        for dimension in ("by_hop_band", "by_split"):
            for group, item in data["comparisons"]["WT - strongest"][dimension].items():
                contributions.append(
                    [
                        family,
                        group,
                        f"{100 * item['population_share']:.2f}",
                        interval_text(item),
                        f"{100 * item['contribution']:+.2f}",
                    ]
                )
    lines += [
        "### Weighted contributions",
        "",
        markdown_table(
            [
                "Family",
                "Subset",
                "Weight mass %",
                "WT - strongest pp [CI]",
                "Contribution to total pp",
            ],
            contributions,
        ),
        "",
        "Contributions sum to the overall difference within each partition. Hop and split "
        "partitions overlap, so do not add their contributions. Joint split/hop "
        "tables are in tables.json. These are descriptive allocations, not causal effects.",
        "",
        "## Complete token and call costs",
        "",
        "All means below are survey weighted. Observation tokens are a subset of generated tokens; "
        "total = prompt + generated. Tokens per correct answer = all questions' weighted tokens / "
        "weighted correct answers, including costs of failures. Per-question ledgers, unweighted "
        "means, costs per correct answer, stage costs and hop breakdowns are in tables.json.",
        "",
    ]
    rows = []
    for family, data in families.items():
        for policy, cost in data["costs"].items():
            mean = cost["weighted_per_question"]
            rows.append(
                [
                    family,
                    policy,
                    *(f"{mean[k]:.2f}" for k in COST_FIELDS),
                    f"{cost['weighted_per_correct']['total_tokens']:.2f}",
                ]
            )
    lines += [
        markdown_table(
            [
                "Family",
                "Policy",
                "Prompt/q",
                "Generated/q",
                "Observation/q",
                "Model calls/q",
                "Total/q",
                "Tokens/correct",
            ],
            rows,
        ),
        "",
        "### Second calls and W → WT transitions",
        "",
        "WT and B2 call twice on every question. Conditional escalation is inapplicable. "
        "The scorer's default escalated=false does not mean a second call was skipped. "
        "Acceptance means a usable supported revision; an answer need not change or become "
        "correct. Rejected attempts are charged. W is one call.",
        "",
    ]
    rows = []
    for family, data in families.items():
        for policy, item in data["second_call"].items():
            accepted = item["rates"]["revision_accepted"]
            rows.append(
                [
                    family,
                    policy,
                    "100%",
                    f"{accepted['count']}/{data['questions']}",
                    f"{100 * accepted['weighted_rate']:.2f}%",
                    f"{100 * item['rates']['answer_changed']['weighted_rate']:.2f}%",
                ]
            )
    lines += [
        markdown_table(
            [
                "Family",
                "Policy",
                "Executed",
                "Accepted n",
                "Accepted weighted",
                "Answer changed weighted",
            ],
            rows,
        ),
        "",
    ]
    rows = []
    for family, data in families.items():
        item = data["W_to_WT"]["overall"]["all"]
        cells = ("wrong_to_wrong", "wrong_to_right", "right_to_wrong", "right_to_right")
        rows.append(
            [
                family,
                *(f"{item['counts'][k]} ({100 * item['weighted_shares'][k]:.2f}%)" for k in cells),
                f"{100 * item['difference']:+.2f}",
            ]
        )
    lines += [
        markdown_table(
            [
                "Family",
                "Wrong → wrong",
                "Wrong → right",
                "Right → wrong",
                "Right → right",
                "Net pp",
            ],
            rows,
        ),
        "",
        "Transition cells show raw counts and weighted shares of all questions. Their signed "
        "off-diagonal difference exactly reproduces WT - W. No before-call correctness is invented "
        "for B2: blind5 has no scorer-of-record table in the sealed report.",
        "",
        "## Development → validation calibration",
        "",
        "Every unordered contrast among shared recipes appears below, for both families in one "
        "table. Development estimates come from analysis-dev/scores.json and budget-frontier. "
        "Shared published accuracies were checked for identity. Positive shrinkage = development "
        "difference minus validation difference; negative values mean expansion. The contrasts "
        "share arms and are correlated. Names: M2 = menuobs-two-pass; incumbent = menuobs-audit; "
        "agent8 = plain-8; feedback8 = failure-aware-8.",
        "",
    ]
    rows = []
    for contrast in families["qwen38"]["shrinkage"]:
        values = []
        for data in families.values():
            item = data["shrinkage"][contrast]
            values.extend(
                f"{100 * item[k]:+.2f}"
                for k in ("development_difference", "validation_difference", "shrinkage")
            )
        rows.append([contrast, *values])
    lines += [
        markdown_table(
            [
                "Contrast",
                "Q dev pp",
                "Q val pp",
                "Q shrink pp",
                "G dev pp",
                "G val pp",
                "G shrink pp",
            ],
            rows,
        ),
        "",
    ]
    for family, data in families.items():
        m2 = data["shrinkage"]["WT - M2"]
        agent = data["shrinkage"][f"WT - {data['strongest']}"]
        lines += [
            f"{family}: WT - M2 changes from {100 * m2['development_difference']:+.2f} pp "
            f"on development to {100 * m2['validation_difference']:+.2f} pp on validation. "
            f"WT - strongest changes from {100 * agent['development_difference']:+.2f} to "
            f"{100 * agent['validation_difference']:+.2f} pp.",
            "",
        ]
    lines += [
        "Development optimism is comparison-specific: shared-policy gains can contract or "
        "expand. Agent-1/2/4, agent-16 and mechanism interventions have no matching sealed "
        "validation "
        "arms here; this table cannot validate those development-only findings or extrapolate a "
        "numerical correction to them. Keep them explicitly development-only in the paper.",
        "",
        "## Automated error taxonomy",
        "",
        "The exact analysis-dev TraceReader.diagnose and taxonomy_tables definitions are reused "
        "with their published reader dependencies, plus the existing committed-start refinement. "
        "Correctness comes from the sealed scorer. No validation example was manually reviewed. "
        "Accepted WT supports were checked against saved supports by CPU replay. Agent actions "
        "were replayed; actions_with_saved_result records how many support a saved-result check. "
        "Initial and refined labels and per-question automatic diagnostics are retained.",
        "",
    ]
    rows = []
    for label in TAXONOMY_PRIORITY:
        values = []
        for data in families.values():
            for policy in ("WT", "strongest"):
                item = data["taxonomy"]["refined"][policy]["primary"][label]
                values.append(f"{item['count']} ({100 * item['weighted_rate']:.2f}%)")
        rows.append([label, *values])
    lines += [
        markdown_table(
            ["Primary error", "Q WT", "Q strongest", "G WT", "G strongest"],
            rows,
        ),
        "",
        "Cells are counts and weighted percentages of all questions. Primary labels partition "
        "errors; multiple labels can overlap and are available in tables.json. Diagnostic plan "
        "attribution is heuristic, especially for branched agents and homonymous labels.",
        "",
        "WT has fewer primary relation and early-stop errors in both families. Gemma also "
        "has fewer format errors. WT has more primary endpoint-choice errors in both families; "
        "that limitation remains. These labels are descriptive attributions, not causal effects.",
        "",
        "Development agreement caveat: the earlier assistant review used 40 enriched questions per "
        "family, 80 policy decisions, not independent human annotation. Exact-set agreement was "
        "91.25% → 98.75% (Qwen), 98.75% → 100% (Gemma), initial → refined. Refinement used that "
        "same sample; the refined agreement is an in-sample audit, not validation agreement.",
        "",
        "## Reproduction and limits",
        "",
        f"Analysis code commit at execution: `{provenance['repo_commit']}`; dirty state: "
        f"`{provenance['repo_dirty']}`. Working-source hashes, producer commit/freeze, "
        "model identities, input hashes, Python/packages, command, seed and resource use are in "
        "provenance.json. The CLI writes argv-preview.json in dry-run mode.",
        "",
        "CPU only. GPU queue script: none. GPU dry-run: not applicable; the saved argv preview is "
        "CPU only. Estimated GPU-hours: 0, because all model outputs already exist and every new "
        "operation is parsing, bootstrap resampling or CPU KG replay. No GPU, vLLM or Docker run "
        "was launched. Tokens establish no latency, throughput or controlled GPU-cost claim.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/home/jonas/git/xgraph"))
    parser.add_argument("--sealed-root", type=Path)
    parser.add_argument("--budget-root", type=Path)
    parser.add_argument("--out", type=Path, default=WORKTREE / "target/validation-analysis")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-taxonomy", action="store_true")
    parser.add_argument("--taxonomy-worker", choices=FAMILIES)
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
        parser.error("--out must be within this worktree's target directory")
    args.out.mkdir(parents=True, exist_ok=True)
    command = [
        "env",
        "OPENBLAS_NUM_THREADS=1",
        "OMP_NUM_THREADS=1",
        "TOKENIZERS_PARALLELISM=false",
        "PYTHONDONTWRITEBYTECODE=1",
        "CUDA_VISIBLE_DEVICES=",
        "nice",
        "-n",
        "10",
        "prlimit",
        "--as=4000000000",
        sys.executable,
        "-I",
        "-S",
        str(WORKTREE / "scripts/mintqa/validation-analysis.py"),
        "--source-root",
        str(args.source_root),
        "--sealed-root",
        str(args.sealed_root),
        "--budget-root",
        str(args.budget_root),
        "--out",
        str(args.out),
    ]
    workers = {family: [*command, "--taxonomy-worker", family] for family in FAMILIES}
    actual_command = [*command, *(["--reuse-taxonomy"] if args.reuse_taxonomy else [])]
    if not args.taxonomy_worker:
        write_json(
            args.out / "argv-preview.json",
            {
                "cwd": str(WORKTREE),
                "argv": actual_command,
                "taxonomy_worker_argv": workers,
                "gpu_queue": None,
                "gpu_hours": 0,
                "basis": "cached sealed outputs plus CPU bootstrap and KG replay",
            },
        )
    if args.dry_run:
        print(json.dumps(command))
        return
    inputs = Inputs()
    if args.taxonomy_worker:
        taxonomy_worker(args, inputs)
        return
    started = time.monotonic()
    freeze = inputs.read_json(args.sealed_root / "freeze-manifest.json")
    conjunction = inputs.read_json(args.sealed_root / "conjunction.json")
    if (
        conjunction["freeze_sha256"]
        != inputs.digest(args.sealed_root / "freeze-manifest.json")["sha256"]
    ):
        raise ValueError("conjunction and freeze identity differ")
    development = inputs.read_json(args.source_root / "target/analysis-dev/scores.json")
    inputs.digest(args.source_root / "target/analysis-dev/findings.md")
    inputs.digest(args.source_root / "target/analysis-dev/taxonomy.json")
    inputs.digest(args.source_root / "target/analysis-dev/manual-review.json")
    frontier = inputs.read_json(args.budget_root / "frontier.json")
    inputs.digest(args.budget_root / "provenance.json")
    result = {
        "scope": "descriptive/post hoc; outside the registered Holm family",
        "scorer_of_record": "completed sealed report per_question correctness, weights and costs",
        "gpu_queue": None,
        "gpu_hours": 0,
        "families": {},
    }
    for family in FAMILIES:
        report = inputs.read_json(args.sealed_root / family / "report.json")
        gold = source_gold(
            args.source_root, report["panels"]["validation"]["finalist"]["per_question"], inputs
        )
        result["families"][family] = analysis_family(
            report, freeze, gold, inputs, args, development, frontier
        )
        print(
            f"{family}: score flags, groups, CIs, costs, transitions and shrinkage complete",
            flush=True,
        )
        path = args.out / f"taxonomy-{family}.json"
        if not args.reuse_taxonomy:
            subprocess.run(workers[family], cwd=WORKTREE, check=True)
        taxonomy = inputs.read_json(path)
        verify_taxonomy_cache(taxonomy, inputs)
        result["families"][family]["taxonomy"] = taxonomy
        print(f"{family}: automated taxonomy and source identity complete", flush=True)
        del report, gold
        gc.collect()

    def git(*arguments):
        return subprocess.check_output(
            ["git", "--no-optional-locks", *arguments], cwd=WORKTREE, text=True
        ).strip()

    for path in (
        Path(__file__),
        WORKTREE / "scripts/mintqa/validation-analysis.py",
        WORKTREE / "tests/test_mintqa_rematch.py",
        WORKTREE / "uv.lock",
    ):
        inputs.digest(path)
    for name, module in list(sys.modules.items()):
        if name.startswith("xgraph") and getattr(module, "__file__", None):
            inputs.digest(module.__file__)
    provenance = {
        "artifact_kind": "xgraph.validation_analysis",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": result["scope"],
        "command": actual_command,
        "taxonomy_worker_commands": workers,
        "repo_commit": git("rev-parse", "HEAD"),
        "repo_dirty": bool(git("status", "--porcelain")),
        "repo_branch": git("branch", "--show-current"),
        "sealed_producer_commit": freeze["code_commit"],
        "freeze_sha256": inputs.digest(args.sealed_root / "freeze-manifest.json")["sha256"],
        "sealed_model_inventory": {
            family: freeze["families"][family]["model"] for family in FAMILIES
        },
        "config": {
            family: {k: result["families"][family][k] for k in ("seed", "resamples", "strongest")}
            for family in FAMILIES
        },
        "inputs": list(inputs.files.values()),
        "python": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "tokenizers")},
        "seconds": time.monotonic() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "gpu_queue": None,
        "gpu_hours": 0,
        "gpu_hours_basis": "zero model calls; cached outputs and CPU resampling/replay only",
        "no_manual_validation_review": True,
        "identity": {family: result["families"][family]["identity"] for family in FAMILIES},
    }
    write_json(args.out / "tables.json", result)
    (args.out / "report.md").write_text(render_report(result, provenance))
    output_identity = Inputs()
    provenance["outputs"] = [
        output_identity.digest(args.out / name) for name in ("tables.json", "report.md")
    ]
    write_json(args.out / "provenance.json", provenance)
    print(f"Descriptive report: {args.out / 'report.md'}", flush=True)
