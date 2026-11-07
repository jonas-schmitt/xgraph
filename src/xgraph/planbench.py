"""Bounded PlanBench pilot: frozen panels, CPU references, interfaces and scoring.

GPU execution is opt-in via ``run``. Preparation, validation, scoring and queue
dry runs import no model and never initialize CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

from xgraph.cost import StageTimer
from xgraph.planbench_data import Case, load_cases, verify_renaming
from xgraph.planbench_pddl import check_plan
from xgraph.provenance import provenance, sha256_file


def _budget_args(parser):
    parser.add_argument("--max-actions", type=int, default=64)
    parser.add_argument("--max-calls", type=int, default=65)
    parser.add_argument("--generated-tokens", type=int, default=2048)
    parser.add_argument("--prompt-tokens", type=int, default=262144)
    parser.add_argument("--context-tokens", type=int, default=16384)


def _panel_cases(cases, panel):
    lookup = {c.id: c for c in cases}
    result = []
    for row in panel["cases"]:
        if row["id"] not in lookup:
            raise ValueError(f"panel contains unknown id: {row['id']}")
        case = lookup[row["id"]]
        if (row["domain"], row["cluster"], row["size"]) != (
            case.domain,
            case.cluster,
            len(case.problem.objects),
        ):
            raise ValueError(f"panel metadata disagrees with PDDL: {row['id']}")
        result.append(case)
    return result


def _episodes(args, cases, inputs):
    from xgraph.planbench_interfaces import Budget, run_episode

    budget = Budget(
        args.max_actions,
        args.max_calls,
        args.generated_tokens,
        args.prompt_tokens,
        args.context_tokens,
    )
    panel = read_panel(args.panel, args.data)
    cases = _panel_cases(cases, panel)
    inputs = [*inputs, args.panel]
    timer = StageTimer()
    backend, served = None, None
    planner_fn = None
    if args.command == "run" and args.fast_downward:
        from xgraph.planbench_reference import optimal_plan

        planner_fn = partial(optimal_plan, fast_downward=args.fast_downward)
        inputs.extend([args.fast_downward, args.fast_downward.parent / "install.json"])
    if args.out.exists() or args.out.with_suffix(".run.json").exists():
        raise ValueError("run output already exists; choose a fresh output path")
    arms = ["random", "random-goal-reached-control"] if args.command == "random" else args.arms
    if args.command == "run":
        from xgraph.planbench_backend import VLLMBackend, model_metadata_paths
        from xgraph.planbench_interfaces import ARMS
        from xgraph.tokenizer import ServedTokenizer

        if len(arms) != len(set(arms)) or any(a not in ARMS for a in arms):
            raise ValueError("run needs distinct known interface arms")
        if args.mask_stop != "model" and arms != ["mask"]:
            raise ValueError("goal-reached control must run separately with --arms mask")
        reference = json.loads(args.references.read_text())
        if reference["panel_sha256"] != sha256_file(args.panel):
            raise ValueError("optimal references belong to a different panel")
        inputs.extend([args.references, *model_metadata_paths(args.model)])
        served = ServedTokenizer.from_file(args.model / "tokenizer.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "running",
        "configuration": vars(args),
        "arms": arms,
        "provenance": provenance(inputs),
        "panel_sha256": sha256_file(args.panel),
    }
    if args.command == "run" and args.run_manifest:
        metadata["host_manifest"] = json.loads(args.run_manifest.read_text())
    metadata_path = args.out.with_suffix(".run.json")

    def save_metadata():
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n")

    save_metadata()
    written = 0
    try:
        if args.command == "run":
            begin = time.perf_counter()
            backend = VLLMBackend(
                args.model, budget, args.kv_cache_gib, args.seed, args.memory_share
            )
            served.check_eos_token_ids(backend.eos)
            metadata["model_load_seconds"] = time.perf_counter() - begin
            metadata["tokenizer_fingerprint"] = served.fingerprint
            metadata["eos_token_ids"] = backend.eos
            save_metadata()
        with args.out.open("x") as handle:
            for case in cases:
                case_seed = args.seed + int.from_bytes(
                    hashlib.sha256(case.cluster.encode()).digest()[:4]
                )
                order = list(arms)
                random.Random(case_seed).shuffle(order)
                for arm in order:
                    arguments = None
                    if arm == "mask":
                        from xgraph.planbench_mask import planning_arguments

                        arguments = planning_arguments(
                            case.problem, served, backend.eos, args.max_actions, args.mask_stop
                        )
                    result = run_episode(
                        case.problem,
                        arm,
                        backend,
                        budget,
                        seed=case_seed,
                        mask_arguments=arguments,
                        planner_fn=planner_fn,
                    )
                    if arm == "mask" and args.mask_stop != "model":
                        result["arm"] = "mask-goal-reached-control"
                    result.update(
                        id=case.id,
                        domain=case.domain,
                        cluster=case.cluster,
                        size=len(case.problem.objects),
                        panel_sha256=metadata["panel_sha256"],
                    )
                    handle.write(json.dumps(result) + "\n")
                    handle.flush()
                    written += 1
        metadata["status"] = "completed"
    except BaseException as exc:
        metadata["status"], metadata["error"] = "aborted", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata.update(timer.summary())
        metadata["records_written"] = written
        save_metadata()
    print(json.dumps({"out": str(args.out), "records": written, "status": metadata["status"]}))


def select_panel(cases: list[Case], count: int, seed: int) -> list[Case]:
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    rng = random.Random(seed)
    selected = []
    for domain in ("blocksworld", "logistics"):
        strata = defaultdict(list)
        for case in sorted(cases, key=lambda c: c.id):
            if case.domain == domain:
                strata[len(case.problem.objects)].append(case)
        if sum(map(len, strata.values())) < count:
            raise ValueError(f"not enough {domain} cases for count={count}")
        for group in strata.values():
            rng.shuffle(group)
        chosen = []
        while len(chosen) < count:
            for size in sorted(strata):
                if strata[size] and len(chosen) < count:
                    chosen.append(strata[size].pop())
        selected.extend(chosen)
    mystery = {c.cluster: c for c in cases if c.domain == "mystery_blocksworld"}
    for case in list(selected):
        if case.domain == "blocksworld":
            if case.cluster not in mystery:
                raise ValueError(f"missing Mystery renaming for {case.id}")
            selected.append(mystery[case.cluster])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="cross-check every released gold plan against VAL")
    validate.add_argument("--data", type=Path, default=Path("data/external/planbench"))
    validate.add_argument("--val", type=Path, default=Path("target/planbench/tools/val/Validate"))
    validate.add_argument("--out", type=Path, required=True)
    prepare = sub.add_parser("prepare", help="freeze a size-stratified panel, pairing renamings")
    prepare.add_argument("--data", type=Path, default=Path("data/external/planbench"))
    prepare.add_argument("--count", type=int, default=100)
    prepare.add_argument("--seed", type=int, default=16)
    prepare.add_argument("--out", type=Path, required=True)
    references = sub.add_parser("references", help="bounded independent optimal planning")
    references.add_argument("--data", type=Path, default=Path("data/external/planbench"))
    references.add_argument("--panel", type=Path, required=True)
    references.add_argument("--planner-seconds", type=float, default=10)
    references.add_argument("--fast-downward", type=Path)
    references.add_argument("--out", type=Path, required=True)
    for name in ("run", "random"):
        command = sub.add_parser(
            name,
            help="opt-in GPU interfaces"
            if name == "run"
            else "CPU random policy and separately named goal-stop control",
        )
        command.add_argument("--data", type=Path, default=Path("data/external/planbench"))
        command.add_argument("--panel", type=Path, required=True)
        command.add_argument("--seed", type=int, default=16)
        command.add_argument("--out", type=Path, required=True)
        _budget_args(command)
        if name == "run":
            command.add_argument("--model", type=Path, required=True)
            command.add_argument("--references", type=Path, required=True)
            command.add_argument("--kv-cache-gib", type=float, default=16)
            command.add_argument("--memory-share", type=float, default=0.45)
            command.add_argument(
                "--arms",
                nargs="+",
                default=["free", "resume", "feedback", "menu", "mask", "llm_planner"],
            )
            command.add_argument(
                "--mask-stop", choices=("model", "goal-reached-control"), default="model"
            )
            command.add_argument("--run-manifest", type=Path)
            command.add_argument("--fast-downward", type=Path)
    scoring = sub.add_parser("score", help="independent final-goal scoring and cost accounting")
    scoring.add_argument("--data", type=Path, default=Path("data/external/planbench"))
    scoring.add_argument("--panel", type=Path, required=True)
    scoring.add_argument("--references", type=Path, required=True)
    scoring.add_argument("--records", nargs="+", type=Path, required=True)
    scoring.add_argument("--arms", nargs="+", required=True)
    scoring.add_argument("--seed", type=int, default=16)
    scoring.add_argument("--resamples", type=int, default=2000)
    scoring.add_argument("--out", type=Path, required=True)
    audit = sub.add_parser("audit-mask", help="CPU replay of saved masked vLLM tokens")
    audit.add_argument("--data", type=Path, default=Path("data/external/planbench"))
    audit.add_argument("--panel", type=Path, required=True)
    audit.add_argument("--records", type=Path, required=True)
    audit.add_argument("--run", type=Path, required=True)
    audit.add_argument("--tokenizer", type=Path, required=True)
    audit.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    timer = StageTimer()
    cases = load_cases(args.data)
    inputs = [args.data / "manifest.json"]
    if args.command in ("run", "random"):
        _episodes(args, cases, inputs)
        return
    if args.command == "audit-mask":
        from xgraph.planbench_mask import audit_records
        from xgraph.tokenizer import ServedTokenizer

        panel = read_panel(args.panel, args.data)
        selected = _panel_cases(cases, panel)
        tokenizer = ServedTokenizer.from_file(args.tokenizer)
        metadata = json.loads(args.run.read_text())
        if metadata["panel_sha256"] != sha256_file(args.panel) or (
            metadata.get("tokenizer_fingerprint") != tokenizer.fingerprint
        ):
            raise ValueError("mask audit inputs do not match the saved run")
        records = [
            json.loads(line) for line in args.records.read_text().splitlines() if line.strip()
        ]
        report = audit_records(selected, records, tokenizer, metadata["eos_token_ids"])
        inputs.extend([args.panel, args.records, args.run, args.tokenizer])
    elif args.command == "score":
        from xgraph.planbench_score import score

        panel = read_panel(args.panel, args.data)
        cases = _panel_cases(cases, panel)
        references = json.loads(args.references.read_text())
        if references["panel_sha256"] != sha256_file(args.panel):
            raise ValueError("references belong to a different panel")
        records = []
        for path in args.records:
            for line in path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    if record["panel_sha256"] != sha256_file(args.panel):
                        raise ValueError(f"record belongs to a different panel: {record['id']}")
                    records.append(record)
        report = score(
            cases,
            records,
            references["references"],
            args.arms,
            seed=args.seed,
            resamples=args.resamples,
        )
        inputs.extend([args.panel, args.references, *args.records])
    elif args.command == "validate":
        from xgraph.planbench_reference import validate_with_val

        mysteries = {c.cluster: c for c in cases if c.domain == "mystery_blocksworld"}
        pairs = 0
        for case in cases:
            if case.domain == "blocksworld":
                verify_renaming(case.problem, mysteries[case.cluster].problem)
                pairs += 1
        results = []
        for case in cases:
            local = check_plan(case.problem, case.gold_plan)
            external = validate_with_val(
                args.val, case.domain_path, case.problem_path, case.gold_plan
            )
            if not local.goal_reached or not external:
                raise ValueError(f"released gold disagreement: {case.id}, {local}, VAL={external}")
            # Empty/wrong-goal and invalid action plans must agree too.
            for bad in ("", "(not-a-domain-action)"):
                checked = check_plan(case.problem, bad)
                val = validate_with_val(args.val, case.domain_path, case.problem_path, bad)
                if checked.goal_reached != val:
                    raise ValueError(f"negative control disagreement: {case.id}, {bad!r}")
            results.append(
                {
                    "id": case.id,
                    "domain": case.domain,
                    "gold_length": local.length,
                    "checker": True,
                    "val": external,
                }
            )
        inputs.append(args.val)
        report = {
            "status": "passed",
            "gold": results,
            "counts": dict(Counter(c.domain for c in cases)),
            "verified_renaming_pairs": pairs,
        }
    elif args.command == "prepare":
        selected = select_panel(cases, args.count, args.seed)
        report = {
            "seed": args.seed,
            "count_per_domain": args.count,
            "data_manifest_sha256": sha256_file(inputs[0]),
            "cases": [
                {
                    "id": c.id,
                    "domain": c.domain,
                    "cluster": c.cluster,
                    "size": len(c.problem.objects),
                }
                for c in selected
            ],
        }
    else:
        from xgraph.planbench_reference import optimal_plan

        panel = read_panel(args.panel, args.data)
        inputs.append(args.panel)
        cases = _panel_cases(cases, panel)
        if args.fast_downward:
            inputs.extend([args.fast_downward, args.fast_downward.parent / "install.json"])
        results = []
        for case in cases:
            result = optimal_plan(
                case.problem, seconds=args.planner_seconds, fast_downward=args.fast_downward
            )
            if (
                result["status"] == "optimal"
                and not check_plan(case.problem, "\n".join(result["plan"])).goal_reached
            ):
                raise ValueError(f"independent planner/checker disagreement: {case.id}")
            results.append(
                {
                    "id": case.id,
                    "domain": case.domain,
                    "size": len(case.problem.objects),
                    "cluster": case.cluster,
                    **result,
                }
            )
        report = {
            "references": results,
            "panel_sha256": sha256_file(args.panel),
            "status_counts": dict(Counter(r["status"] for r in results)),
        }
    report.update(timer.summary())
    report["provenance"] = provenance(inputs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "command": args.command,
                "out": str(args.out),
                "seconds": report["seconds"],
                "peak_rss_mib": report["peak_rss_mib"],
            }
        )
    )


def read_panel(path: Path, data: Path) -> dict:
    panel = json.loads(path.read_text())
    if panel["data_manifest_sha256"] != sha256_file(data / "manifest.json"):
        raise ValueError("panel belongs to a different data manifest")
    ids = [r["id"] for r in panel["cases"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate panel ids")
    return panel


if __name__ == "__main__":
    main()
