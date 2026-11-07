"""CPU-only secondary diagnostics, independent of the frozen WebQSP producer."""

import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("qwen38", "gemma31")
PARENT_SHA256 = "7731cd31160615e1fa895fe3eb2a1188b91431fbc287d9a502d92b39c640e414"


def _path(arguments):
    if not isinstance(arguments, dict):
        return None
    start, relations = arguments.get("start"), arguments.get("relations")
    if isinstance(relations, str):
        relations = [r for r in re.split(r"[·,]", relations) if r.strip()]
    if (
        not isinstance(start, str)
        or not start.strip()
        or not isinstance(relations, list)
        or any(not isinstance(r, str) or not r.strip() for r in relations)
    ):
        return None
    return start.strip(), [r.strip() for r in relations]


def prefix_counts(record):
    """Count later-hop retries and requested successful-prefix positions."""
    counts = dict.fromkeys(
        (
            "retries",
            "retained",
            "known_positions",
            "requested_positions",
            "unassessable",
            "retry_of_mismatches",
            "failures_without_retry",
        ),
        0,
    )
    if record.get("agent_feedback") is not True or not isinstance(record.get("actions"), list):
        return {**counts, "status": "UNAVAILABLE"}
    paths = [(i, a) for i, a in enumerate(record["actions"]) if a.get("tool") == "execute_path"]
    for position, (index, action) in enumerate(paths):
        result = action.get("result")
        if not isinstance(result, dict):
            counts["unassessable"] += 1
            continue
        hop = result.get("failed_hop")
        if action.get("failed") is True and (type(hop) is not int or hop < 0):
            counts["unassessable"] += 1
            continue
        if type(hop) is not int or hop < 2:
            continue
        if position + 1 == len(paths):
            counts["failures_without_retry"] += 1
            continue
        retry = paths[position + 1][1]
        counts["retries"] += 1
        if "retry_of" in retry and retry["retry_of"] != index:
            counts["retry_of_mismatches"] += 1
        before, after = _path(action.get("arguments", {})), _path(retry.get("arguments", {}))
        if before is None or after is None or hop > len(before[1]):
            counts["unassessable"] += 1
            continue
        prefix = before[1][: hop - 1]
        if before[0] == after[0] and prefix == after[1][: hop - 1]:
            counts["retained"] += 1
            counts["known_positions"] += hop - 1
            counts["requested_positions"] += len(after[1])
    return {**counts, "status": "COMPLETE" if not counts["unassessable"] else "INCOMPLETE"}


def free_cap_pairs(wt, agents, ids, *, max_calls=8, budget_flag=False):
    """Keep every WT token; zero only the capped agent question's cost."""
    wt, agents = _indexed(wt, ids), _indexed(agents, ids)
    if type(max_calls) is not int or max_calls < 1:
        raise ValueError("invalid registered tool-call cap")
    numerator, denominator = [], []
    for q in ids:
        calls = agents[q].get("tool_calls")
        if type(calls) is not int or calls < 0:
            raise ValueError(f"{q}: missing/invalid tool_calls")
        cap = calls >= max_calls
        if budget_flag:
            cap = agents[q].get("budget_spent")
            if type(cap) is not bool:
                raise ValueError(f"{q}: missing/invalid budget_spent")
        numerator.append(_tokens(wt[q]))
        agent_tokens = _tokens(agents[q])  # validate even costed-free questions
        denominator.append(0 if cap else agent_tokens)
    return numerator, denominator


def _indexed(records, ids):
    rows = {}
    for row in records:
        if row["id"] in rows:
            raise ValueError(f"duplicate diagnostic record: {row['id']}")
        rows[row["id"]] = row
    if set(rows) != set(ids):
        raise ValueError("diagnostics require complete records on exactly the frozen cohort")
    return rows


def _tokens(row):
    usage = row.get("usage", row)
    prompt, generated = usage.get("prompt_tokens"), usage.get("generated_tokens")
    if generated is None and isinstance(usage.get("token_ids"), list):
        generated = len(usage["token_ids"])
    if any(type(v) is not int or v < 0 for v in (prompt, generated)):
        raise ValueError(f"{row['id']}: missing/invalid complete token usage")
    return prompt + generated


def cluster_ratio(
    numerator,
    denominator,
    members,
    *,
    resamples=10000,
    seed=17,
    minimum_clusters=20,
    alpha=0.025 / 8,
):
    """Paired whole-question cluster bootstrap with registered survey weights."""
    import numpy as np

    from xgraph.mintqa_rematch import _paired_bootstrap

    if (
        not members
        or len(numerator) != len(members)
        or len(denominator) != len(members)
        or not 0 < alpha < 0.5
        or type(resamples) is not int
        or resamples < 1
        or any(not math.isfinite(v) for v in [*numerator, *denominator])
        or any(v < 0 for v in denominator)
    ):
        raise ValueError(
            "cluster ratio requires finite aligned values and nonnegative denominators"
        )
    ids = [m["id"] for m in members]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate bootstrap question")
    point, draws, clusters = _paired_bootstrap(numerator, denominator, members, resamples, seed)
    contributing = len({m["cluster"] for m, d in zip(members, denominator, strict=True) if d > 0})
    base = {
        "questions": len(members),
        "clusters": clusters,
        "contributing_clusters": contributing,
        "resamples": resamples,
        "seed": seed,
        "alpha_per_component": alpha,
        "multiplicity": "Bonferroni .025 / (4 diagnostics x 2 model families), directional bounds",
        "minimum_clusters": minimum_clusters,
    }
    if point[1] <= 0:
        return {**base, "status": "UNAVAILABLE", "reason": "zero denominator", "estimate": None}
    valid = draws[:, 1] > 0
    samples = draws[valid, 0] / draws[valid, 1]
    enough = contributing >= minimum_clusters and len(samples) >= 0.95 * resamples
    return {
        **base,
        "status": "COMPLETE" if enough else "DESCRIPTIVE",
        "estimate": float(point[0] / point[1]),
        "valid_resamples": len(samples),
        "lower": float(np.quantile(samples, alpha)) if len(samples) else None,
        "upper": float(np.quantile(samples, 1 - alpha)) if len(samples) else None,
        "enough_clusters": enough,
    }


def selection_replay(record, request, executor, gold_ids):
    """Replay the fixed delivered program with actual, terminal-free and no selections."""
    from xgraph.observations import PathRelations
    from xgraph.programs import record_path
    from xgraph.score import record_failure

    zero = dict.fromkeys(("actual", "terminal_removed", "all_removed", "delta", "selections"), 0)
    if record_failure(record, require_text=False):
        return {**zero, "status": "FAILED_ZERO", "reached": {}}
    written = record_path(record)
    if written is None:
        return {**zero, "status": "NO_PATH_ZERO", "reached": {}}
    start, relations = written
    if getattr(relations, "audit", None) is not None:
        raise ValueError(f"{record['id']}: audited program unsupported for selection removal")
    selections = dict(getattr(relations, "selections", {}))
    terminal_free = {h: v for h, v in selections.items() if h != len(relations)}
    reached, caps = {}, {}
    for name, selected in (
        ("actual", selections),
        ("terminal_removed", terminal_free),
        ("all_removed", {}),
    ):
        program = PathRelations(
            relations,
            selected,
            defers=getattr(relations, "defers", ()),
            decisions=getattr(relations, "decisions", {}),
        )
        result = executor.execute_set(
            start, program, request.get("anchor"), request.get("kg_version")
        )
        reached[name], caps[name] = result["reached"], result["cap_events"]
    if record.get("delivery", {}).get("rule") == "committed_path_set" and (
        "reached" not in record or set(record["reached"]) != set(reached["terminal_removed"])
    ):
        raise ValueError(f"{record['id']}: replay disagrees with saved full-set support")
    support = {k: int(bool(set(v) & gold_ids)) for k, v in reached.items()}
    return {
        **support,
        "delta": support["all_removed"] - support["actual"],
        "selections": len(selections),
        "status": "COMPLETE",
        "reached": reached,
        "cap_events": caps,
    }


def source_gold_ids(identities, questions):
    """Join frozen question-component source identities to exact released answers."""
    wanted = {r["id"]: set(r["answers"]) for r in questions}
    if len(wanted) != len(questions):
        raise ValueError("duplicate gold question")
    gold = {q: set() for q in wanted}
    seen = set()
    for row in identities:
        if len(row) != 4 or any(not isinstance(v, str) for v in row):
            raise ValueError("invalid frozen source identity row")
        entity, qid, source, _ = row
        if entity in seen:
            raise ValueError(f"duplicate frozen source identity: {entity}")
        seen.add(entity)
        if qid in wanted and source in wanted[qid]:
            gold[qid].add(entity)
    return gold


def mintqa_check(source, out):
    """Read only the explicitly permitted sealed MINTQA agent files, never WebQSP."""
    from xgraph import webqsp_test as study
    from xgraph.provenance import provenance

    source = Path(source).resolve()
    if source.name != "final-validation" or source.parent.name != "target":
        raise ValueError("MINTQA check requires the permitted target/final-validation directory")
    expected = {"qwen38": (323, 365), "gemma31": (205, 274)}
    paths, families = [], {}
    for family in FAMILIES:
        counts = []
        for name in ("agent-feedback-8", "agent-strong-8"):
            path = source / family / "validation" / f"{name}.jsonl"
            paths.append(path)
            totals, questions, unavailable = {}, 0, 0
            for row in _lines(path):
                questions += 1
                result = prefix_counts(row)
                unavailable += result["status"] == "UNAVAILABLE"
                for key, value in result.items():
                    if key != "status":
                        totals[key] = totals.get(key, 0) + value
            counts.append(
                {
                    "arm": name,
                    "questions": questions,
                    "unavailable_questions": unavailable,
                    **totals,
                }
            )
        retained = sum(c["retained"] for c in counts)
        retries = sum(c["retries"] for c in counts)
        if (retained, retries) != expected[family] or any(c["unassessable"] for c in counts):
            raise ValueError(f"{family}: MINTQA reference mismatch: {retained}/{retries}")
        families[family] = {"retained": retained, "retries": retries, "arms": counts}
    report = {
        "artifact_kind": "xgraph.webqsp_diagnostics_mintqa_check",
        "post_hoc": True,
        "gold_read": False,
        "webqsp_read": False,
        "bootstrap": "none; raw-count check",
        "seed": 17,
        "families": families,
        "inputs": [study.identity(p) for p in paths],
        "provenance": provenance(paths),
    }
    study.write_sealed(out, report)
    return report


def _lines(path):
    import json

    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def amendment_spec(parent, parent_identity, protocol_identity):
    """An additive record: never transform or repin the generation registration."""
    import copy

    if parent.get("amendment", {}).get("amendment") != 6:
        raise ValueError("diagnostic amendment requires amendment 6")
    return {
        "artifact_kind": "xgraph.webqsp_secondary_registration",
        "amendment": 7,
        "definition": "webqsp-diagnostics-v1",
        "core_registration": parent_identity,
        "protocol": protocol_identity,
        "seed": 17,
        "resamples": 10000,
        "cluster_method": parent["analysis"]["cluster_method"],
        "strongest_agents": copy.deepcopy(parent["strongest_agents"]),
        "minimum_clusters": 20,
        "minimum_valid_draw_share": 0.95,
        "alpha": 0.025,
        "alpha_per_component": 0.025 / 8,
        "multiplicity": "separate Bonferroni family, 4 diagnostics x 2 model families",
        "outside_primary": True,
        "extra_gpu_hours": 0,
        "extra_generation_jobs": [],
        "score_not_before": parent.get("score_not_before", "2026-11-23T23:00:00+00:00"),
        "registration_deadline": "2026-11-01T22:59:59+00:00",
        "depth_interaction_unchanged": copy.deepcopy(parent["analysis"]["depth_interaction"]),
        "diagnostics": {
            "prefix_retention": {
                "prediction_at_least": 0.70,
                "lower_exceeds": 0.50,
                "numerator": "retained",
                "denominator": "retries",
            },
            "known_prefix_share": {
                "lower_exceeds": 0.50,
                "numerator": "known_positions",
                "denominator": "requested_positions",
            },
            "free_cap_tokens": {"upper_below": 1, "tool_call_caps": {f: 8 for f in FAMILIES}},
            "selection_effect": {
                "prediction_at_least": 0.01,
                "lower_exceeds": 0,
                "program": "delivered WT program before set delivery",
                "minimum_changed_clusters": 20,
            },
        },
        "missing_qwen_feedback": "UNAVAILABLE under amendment-4 core; no new arm authorized",
        "mintqa_reference": {"post_hoc": True, "qwen38": [323, 365], "gemma31": [205, 274]},
    }


def register(parent_path, out):
    """Freeze source/metadata only, before the early seal; never activate a producer."""
    from datetime import UTC, datetime

    from xgraph import webqsp_test as study
    from xgraph.provenance import provenance

    parent_path, out = Path(parent_path), Path(out)
    parent = study.read_sealed(parent_path)
    parent_id = study.identity(parent_path)
    if parent_id["sha256"] != PARENT_SHA256:
        raise ValueError("reviewed amendment-6 parent identity changed")
    protocol = ROOT / "eval/webqsp-test-amendment-7-2026-10-28/protocol.md"
    spec = amendment_spec(parent, parent_id, study.identity(protocol))
    if datetime.now(UTC) > datetime.fromisoformat(spec["registration_deadline"]):
        raise ValueError("amendment-7 prospective deadline has passed")
    if (Path(parent["config"]["out"]) / "execution").exists():
        raise ValueError("early execution seal exists; prospective amendment refused")
    spec["implementation_commit"] = study.git_commit()
    sources = [
        ROOT / p
        for p in (
            "src/xgraph/webqsp_diagnostics.py",
            "tests/test_webqsp.py",
            "scripts/webqsp-diagnostics-queue.sh",
            "uv.lock",
        )
    ]
    spec["sources"] = [study.identity(p) for p in sources]
    spec["created_at"] = datetime.now(UTC).isoformat()
    spec["provenance"] = provenance([parent_path, protocol, *sources])
    spec["preserved_core_sha256"] = parent_id["sha256"]
    study.verify_identity(parent_id)
    study.write_sealed(out, spec)
    return spec


def score(registration_path, execution_dir, out):
    """Open diagnostics only after the embargo and the parent's sealed scoring."""
    from xgraph import webqsp_test as study

    spec = study.read_sealed(registration_path)
    # No execution metadata, receipt, source, output or gold access above this check.
    study.enforce_embargo(spec, execution_dir)
    return _score_released(spec, registration_path, Path(execution_dir), Path(out))


def reduce_family(members, rows, reference, selection, *, resamples=10000):
    """Reduce the frozen cohort without dropping zero-event or failed-program questions."""
    from collections import Counter

    ids = [m["id"] for m in members]
    for records in rows.values():
        _indexed(records, ids)
    selection = _indexed(selection, ids)
    if any(
        r["status"] not in ("COMPLETE", "FAILED_ZERO", "NO_PATH_ZERO") for r in selection.values()
    ):
        raise ValueError("incomplete selection replay")
    feedback = [
        records
        for name, records in rows.items()
        if name.startswith("agent-") and any(r.get("agent_feedback") is True for r in records)
    ]
    prefix = [
        dict.fromkeys(("retries", "retained", "known_positions", "requested_positions"), 0)
        for _ in ids
    ]
    unassessable = mismatches = missing = 0
    for records in feedback:
        by_id = _indexed(records, ids)
        for p, q in zip(prefix, ids, strict=True):
            counts = prefix_counts(by_id[q])
            missing += counts["status"] == "UNAVAILABLE"
            unassessable += counts["unassessable"]
            mismatches += counts["retry_of_mismatches"]
            for field in p:
                p[field] += counts[field]
    result = {}
    for name, numerator, denominator in (
        ("prefix_retention", "retained", "retries"),
        ("known_prefix_share", "known_positions", "requested_positions"),
    ):
        stat = cluster_ratio(
            [p[numerator] for p in prefix],
            [p[denominator] for p in prefix],
            members,
            resamples=resamples,
        )
        stat.update(
            raw_numerator=sum(p[numerator] for p in prefix),
            raw_denominator=sum(p[denominator] for p in prefix),
            unassessable=unassessable,
            missing_feedback_questions=missing,
            retry_of_mismatches=mismatches,
        )
        if not feedback:
            stat.update(status="UNAVAILABLE", reason="no registered failure-aware agent records")
        elif unassessable or missing:
            stat.update(
                status="INCOMPLETE", reason="missing/unassessable saved feedback or arguments"
            )
        result[name] = stat
    a, b = free_cap_pairs(rows["WT"], rows[reference], ids)
    result["free_cap_tokens"] = cluster_ratio(a, b, members, resamples=resamples)
    agents = _indexed(rows[reference], ids)
    result["free_cap_tokens"]["raw_capped_questions"] = sum(
        agents[q]["tool_calls"] >= 8 for q in ids
    )
    try:
        a, b = free_cap_pairs(rows["WT"], rows[reference], ids, budget_flag=True)
        sensitivity = cluster_ratio(a, b, members, resamples=resamples)
        sensitivity["descriptive_only"] = True
    except ValueError as error:
        sensitivity = {"status": "UNAVAILABLE", "reason": str(error), "descriptive_only": True}
    result["free_cap_tokens"]["budget_spent_sensitivity"] = sensitivity
    effect = cluster_ratio(
        [selection[q]["delta"] for q in ids], [1] * len(ids), members, resamples=resamples
    )
    changed = len({m["cluster"] for m in members if selection[m["id"]]["delta"] != 0})
    if changed < 20:
        effect.update(status="DESCRIPTIVE", reason="fewer than 20 clusters with a support change")
    effect.update(
        changed_clusters=changed,
        gross_gains=sum(selection[q]["delta"] == 1 for q in ids),
        gross_losses=sum(selection[q]["delta"] == -1 for q in ids),
        selected_programs=sum(selection[q]["selections"] > 0 for q in ids),
        replay_states=dict(Counter(selection[q]["status"] for q in ids)),
        terminal_only=cluster_ratio(
            [selection[q]["terminal_removed"] - selection[q]["actual"] for q in ids],
            [1] * len(ids),
            members,
            resamples=resamples,
        ),
        per_question=[selection[q] for q in ids],
        units="fraction; multiply by 100 for percentage points",
    )
    effect["terminal_only"]["descriptive_only"] = True
    result["selection_effect"] = effect
    for name, stat in result.items():
        complete = stat["status"] == "COMPLETE"
        if name == "free_cap_tokens":
            passed = complete and stat["upper"] < 1
        else:
            threshold = 0 if name == "selection_effect" else 0.5
            point_floor = (
                0.01 if name == "selection_effect" else 0.70 if name == "prefix_retention" else 0.5
            )
            passed = complete and stat["lower"] > threshold and stat["estimate"] >= point_floor
        stat["confirmed"] = bool(passed)
    return result


def _score_released(spec, registration_path, directory, out):

    from xgraph import webqsp_test as study
    from xgraph.index import load_kg_index
    from xgraph.kgversions import load_kg_versions
    from xgraph.programs import MatchedExecutor
    from xgraph.provenance import provenance
    from xgraph.tokenizer import ServedTokenizer
    from xgraph.webqsp import verify_catalog

    if spec.get("definition") != "webqsp-diagnostics-v1" or spec.get("amendment") != 7:
        raise ValueError("unsupported diagnostic registration")
    if study.git_commit() != spec["implementation_commit"]:
        raise ValueError("diagnostic implementation commit changed")
    for item in [spec["core_registration"], spec["protocol"], *spec["sources"]]:
        study.verify_identity(item)
    parent = study.read_sealed(spec["core_registration"]["path"])
    if parent["analysis"]["depth_interaction"] != spec["depth_interaction_unchanged"]:
        raise ValueError("registered depth prediction changed")
    execution_path, score_path = directory / "execution.json", directory / "score.json"
    execution, original_score = study.read_sealed(execution_path), study.read_sealed(score_path)
    if (
        execution["registration"] != spec["core_registration"]
        or execution["commit"] != parent["commit"]
        or original_score["registration"] != spec["core_registration"]
        or original_score["execution"] != study.identity(execution_path)
        or original_score["selected_ids"] != execution["ids"]
    ):
        raise ValueError("original scoring/registration/execution identities disagree")
    for item in execution["inputs"]:
        study.verify_identity(item)
    study.verify_receipts(
        execution, directory, incomplete_gcr=original_score["gcr_comparisons"]["failures"]
    )
    study.verify_gpu_ledger(parent, execution, directory)
    data = directory / "inputs"
    verify_catalog(data)
    ids = execution["ids"]
    expected_clusters = parent["question_cluster_audit"]["methods"][spec["cluster_method"]][
        "assignments"
    ]
    if execution["cluster"] != {q: expected_clusters[q] for q in ids}:
        raise ValueError("registered question dependence clusters changed")
    if ids != study.select_ids(parent["population"]["intersection"], len(ids)):
        raise ValueError("registered cohort changed")
    members = [{"id": q, "cluster": execution["cluster"][q], "hops": 1, "weight": 1.0} for q in ids]
    questions = study.load_gold(parent, ids)
    identity_path = data / "kg/identities.jsonl"
    gold_ids = source_gold_ids(_lines(identity_path), questions)
    local_ids = {q: set() for q in ids}
    for entity, q, _, _ in _lines(identity_path):
        if q in local_ids:
            local_ids[q].add(entity)
    requests_path = data / "requests-menuobs.jsonl"
    requests = _indexed(list(_lines(requests_path)), ids)
    input_paths = [
        Path(registration_path),
        execution_path,
        score_path,
        identity_path,
        requests_path,
    ]
    results = {}
    for family in FAMILIES:
        arm_paths = {
            a["arm"]: Path(a["output"])
            for a in execution["arms"]
            if a["scope"] == "primary"
            and a["family"] == family
            and (a["arm"] == "WT" or a["arm"].startswith("agent-"))
        }
        rows = {name: list(_lines(path)) for name, path in arm_paths.items()}
        input_paths.extend(arm_paths.values())
        for records in rows.values():
            _indexed(records, ids)
        index = load_kg_index(data / f"index-{family}")
        tokenizer_path = Path(parent["config"]["models"][family]) / "tokenizer.json"
        tokenizer = ServedTokenizer.from_file(tokenizer_path)
        input_paths.append(tokenizer_path)
        executor = MatchedExecutor(
            index,
            tokenizer,
            versions=load_kg_versions(data / "versions", index),
            start_rank="sitelinks",
            unquoted_starts=True,
        )
        wt = _indexed(rows["WT"], ids)
        selection = []
        for q in ids:
            result = selection_replay(wt[q], requests[q], executor, gold_ids[q])
            if any(set(v) - local_ids[q] for v in result["reached"].values()):
                raise ValueError(f"{q}: replay endpoint has no frozen component source identity")
            selection.append({"id": q, **result})
        results[family] = reduce_family(
            members, rows, parent["strongest_agents"][family], selection
        )
    report = {
        "artifact_kind": "xgraph.webqsp_secondary_diagnostics",
        "registration": study.identity(registration_path),
        "core_registration": spec["core_registration"],
        "execution": study.identity(execution_path),
        "original_score": study.identity(score_path),
        "selected_ids": ids,
        "results": results,
        "depth_interaction_unchanged": spec["depth_interaction_unchanged"],
        "extra_gpu_hours": 0,
        "seed": 17,
        "inputs": [study.identity(p) for p in input_paths],
        "provenance": provenance(input_paths),
        "joint_predictions": {
            name: all(results[f][name]["confirmed"] for f in FAMILIES)
            for name in spec["diagnostics"]
        },
    }
    if out.exists():
        saved = study.read_sealed(out)
        # Re-derive outputs and input identities before accepting an existing result.
        if any(saved.get(k) != report[k] for k in report if k != "provenance"):
            raise ValueError("existing diagnostics differ from frozen inputs/results")
        return saved
    study.write_sealed(out, report)
    return report


def main(argv=None):
    import argparse

    from xgraph import webqsp_test as study

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("register", help="seal additive analysis without accessing test data")
    freeze.add_argument("--parent", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    check = sub.add_parser("mintqa", help="post hoc read-only raw-count reproduction")
    check.add_argument("--source", type=Path, required=True)
    check.add_argument("--out", type=Path, required=True)
    for name in ("check-embargo", "score"):
        command = sub.add_parser(name)
        command.add_argument("--registration", type=Path, required=True)
        command.add_argument("--execution", type=Path, required=True)
        if name == "score":
            command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "register":
        register(args.parent, args.out)
    elif args.command == "mintqa":
        report = mintqa_check(args.source, args.out)
        for family, row in report["families"].items():
            print(f"{family}: {row['retained']}/{row['retries']} (post hoc)")
    elif args.command == "check-embargo":
        study.enforce_embargo(study.read_sealed(args.registration), args.execution)
    else:
        score(args.registration, args.execution, args.out)


if __name__ == "__main__":
    main()
