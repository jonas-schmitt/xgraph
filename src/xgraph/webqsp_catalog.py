"""Opt-in frozen policy catalog for WebQSP; command planning never loads a model."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

from xgraph.mintqa_queue import FAMILIES
from xgraph.policy_catalog import recipe as policy_recipe
from xgraph.webqsp import CATALOG_DEMOS

ARMS = (
    "S",
    "W",
    "M2",
    "MT",
    "WT",
    "B2",
    "BT",
    "blind5",
    "agent-strong-8",
    "agent-strong-1",
    "agent-strong-4",
)
ROOT = Path(__file__).resolve().parents[2]
PUBLISHED_ARMS = ("gcr-native", "dog")
FEEDBACK_ARMS = ("agent-feedback-8",)
DEFAULT_GCR_MODEL = Path("/home/jonas/models/GCR-Meta-Llama-3.1-8B-Instruct")
DEFAULT_DOG_PROMPTS = Path("/home/jonas/git/xgraph/data/external/dog-2wiki/release/templates.py")


def timing_assumption(root, family, name):
    """MINTQA workload proxies; GCR has no qualified timing and keeps a wide forecast."""
    root = Path(root)
    base = root / "target/mintqa-rematch" / family / "development"
    menu = (
        root
        / ".claude/worktrees/codex-menu/target"
        / ("menu-smoke/development" if family == "qwen38" else f"menu-dev/{family}/development")
        / "paths-menu-5-observed-3.jsonl.summary.json"
    )
    second = (
        root
        / ".claude/worktrees/codex-menu-audit/target/mintqa-menu-audit"
        / family
        / "development"
    )
    source = None
    if name in ("gcr-shared", "gcr-native"):
        fallback = [8, 90]
    elif name == "dog":
        fallback = [4, 30]
    elif name in ("menuobs", "W"):
        source, fallback = menu, [1, 6]
    elif name.startswith(("agent-strong", "agent-feedback")):
        source, fallback = base / "agent-strong-8.jsonl.summary.json", [2, 14]
    elif name == "x-plan":
        source, fallback = base / "free-opened-raw.jsonl.summary.json", [0.4, 2]
    elif name == "x-delivery":
        source, fallback = base / "x-delivery.jsonl.summary.json", [0.4, 2]
    elif name.endswith("-raw"):
        suffix = "audit" if "audit" in name else "revision"
        source = second / f"menuobs-{suffix}-raw.jsonl.summary.json"
        fallback = [1, 6]
    else:
        source, fallback = base / "paths-allfix.jsonl.summary.json", [0.4, 2]
    if source is None or not source.is_file():
        return {
            "seconds_per_question": fallback,
            "source": None,
            "proxy": True,
            "basis": "unmeasured forecast; no qualified MINTQA timing available",
        }
    from xgraph.provenance import sha256_file

    saved = json.loads(source.read_text())
    seconds, count = saved["seconds"], saved["requests"]
    if not math.isfinite(seconds) or seconds <= 0 or count <= 0:
        raise ValueError(f"invalid MINTQA timing: {source}")
    rate = seconds / count
    return {
        "seconds_per_question": [rate, 3 * rate],
        "source": str(source),
        "sha256": sha256_file(source),
        "requests": count,
        "seconds": seconds,
        "proxy": True,
        "basis": "MINTQA workload per question to three times that rate; "
        "agent budgets 1/4 use agent8; blind5 uses S; blind second calls use menuobs",
    }


def price_from_pilot(jobs, timing, signature):
    """Require matching timings; charge every unique stage and discarded revision."""
    if not timing.get("complete"):
        raise ValueError("pilot timings must be complete, including CPU stages")
    if timing.get("catalog_signature") != signature:
        raise ValueError("pilot timings catalog signature changed")
    gpu = [j for j in jobs if j["kind"] == "gpu"]
    measured = {}
    for row in timing["per_arm"]:
        key = row["family"], row["name"]
        seconds = row.get("gpu_seconds", row["wall_seconds"])
        if key in measured:
            raise ValueError("duplicate pilot stage timing")
        if (
            row["returncode"] != 0
            or row["questions"] <= 0
            or not math.isfinite(row["wall_seconds"])
            or row["wall_seconds"] <= 0
            or not math.isfinite(seconds)
            or seconds <= 0
            or ("gpu_seconds" in row and row.get("accounting") != "measured")
        ):
            raise ValueError("failed or invalid pilot stage timing")
        measured[key] = row
    if not {(j["family"], j["name"]) for j in gpu} <= set(measured):
        raise ValueError("pilot timings do not cover the complete requested stage matrix")
    for job in gpu:
        row = measured[job["family"], job["name"]]
        job["projected_wall_seconds"] = row["wall_seconds"] * job["questions"] / row["questions"]
        job["projected_gpu_seconds"] = (
            row.get("gpu_seconds", row["wall_seconds"]) * job["questions"] / row["questions"]
        )
    return {
        "measured": False,
        "pilot_measured": True,
        "gpu_hours": sum(j["projected_gpu_seconds"] for j in gpu) / 3600,
        "rule": "linear per-stage pilot GPU charge/question including startup; "
        "container occupancy when recorded, legacy launcher wall otherwise; "
        "unique first calls charged once; CPU stages separate; no speed claim",
    }


def catalog_plan(
    data,
    out,
    models,
    *,
    pilot=False,
    pilot_questions=100,
    pilot_seed=0,
    pilot_budget_hours=8,
    project_questions=None,
    arms=None,
    pilot_timings=None,
    policy_freeze=None,
    timing_root=Path("/home/jonas/git/xgraph"),
    gcr_model=DEFAULT_GCR_MODEL,
    dog_prompts=DEFAULT_DOG_PROMPTS,
    secondary_pilot=False,
):
    from xgraph.webqsp_queue import pilot_stratum, pilot_subset

    data, out = Path(data).absolute(), Path(out).absolute()
    metadata_path = data / "manifest.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    secondary_population = metadata.get("secondary_population", False)
    if secondary_pilot and not (
        pilot
        and secondary_population
        and metadata.get("population") == "dog_top120_answer_present"
        and metadata.get("split") == "test"
        and metadata.get("top_triples") == 120
    ):
        raise ValueError(
            "secondary cost pilot requires the released DoG test population and --pilot"
        )
    defaults = [*ARMS, "gcr-native", *(["dog"] if secondary_population or pilot else [])]
    selected = list(defaults if arms is None else arms)
    allowed = {*ARMS, *FEEDBACK_ARMS, *PUBLISHED_ARMS, *(["finalist"] if policy_freeze else [])}
    if not selected or len(set(selected)) != len(selected) or set(selected) - allowed:
        raise ValueError(f"catalog arms must be distinct members of {(*ARMS, *PUBLISHED_ARMS)}")
    if set(models) - set(FAMILIES):
        raise ValueError("unknown catalog model family")
    questions = metadata.get("counts", {}).get("questions", 0)
    if "dog" in selected and not secondary_population and not pilot:
        raise ValueError(
            "DoG test reader requires the separately declared released top-120 population"
        )
    primary_population = (
        metadata["population"] if secondary_population else "rog_released_subgraphs"
    )
    declared_population = (
        metadata["population"]
        if secondary_population
        else "rog_declared_including_native_omissions"
    )
    pilot_info = None
    if pilot:
        if metadata.get("split") != "validation" and not secondary_pilot:
            raise ValueError("catalog pilot requires validation, never test questions")
        if set(models) != {"qwen38", "gemma31"}:
            raise ValueError("catalog pilot requires both large families")
        if not math.isfinite(pilot_budget_hours) or pilot_budget_hours <= 0:
            raise ValueError("pilot budget must be finite and positive")
        panel = json.loads((data / "questions.json").read_text())
        cohort = pilot_subset(panel, pilot_questions, pilot_seed)
        questions = len(cohort)
        counts = Counter(pilot_stratum(q) for q in panel)
        sampled = Counter(pilot_stratum(q) for q in cohort)
        pilot_info = {
            "scope": "released_dog_cost_only" if secondary_pilot else "validation",
            "seed": pilot_seed,
            "cohort_ids": [q["id"] for q in cohort],
            "per_arm_ids": [q["id"] for q in cohort],
            "questions_per_arm": questions,
            "budget_gpu_hours": pilot_budget_hours,
            "rule": "same complete seeded cohort for all catalog arms and families",
            "strata": {
                k: {"available": counts[k], "cohort": sampled[k], "per_arm": sampled[k]}
                for k in sorted(counts)
            },
        }
    inputs = out / "pilot" if pilot else data
    jobs, results = [], []
    declared = selected
    frozen = {}
    for family, model in models.items():
        selected = declared
        if policy_freeze:
            from xgraph.policy_catalog import frozen_recipe

            frozen[family] = frozen_recipe(policy_freeze, family, model=model)
            policy = frozen[family]["recipe"]["policy"]
            if policy not in ARMS:
                raise ValueError(f"{family}: frozen policy {policy} is outside the WebQSP catalog")
            selected = [policy if a == "finalist" else a for a in declared]
            if len(set(selected)) != len(selected):
                raise ValueError("duplicate resolved WebQSP policies")
        if "agent-feedback-8" in selected and (
            family not in frozen or "agent-feedback-8" not in frozen[family].get("comparators", [])
        ):
            raise ValueError("feedback serving requires its sealed comparator qualification")
        needed = set(selected)
        if {"M2", "MT"} & needed:
            needed.add("menuobs")
        if "WT" in needed:
            needed.add("W")
        if {"B2", "BT"} & needed:
            needed.add("blind5")
        model = Path(model).absolute()
        folder, index = out / family, data / f"index-{family}"
        tokenizer = model / "tokenizer.json"
        common = [
            "--max-model-len",
            "16384",
            "--max-num-seqs",
            "64",
            "--kv-cache-gib",
            "16" if family == "gemma31" else "8",
        ]
        base = {
            "XG_NEED_GIB": str(FAMILIES[family][2]),
            "XG_GPU_IDLE": "1",
            "XG_TOOLS_DB": str(data / "kg.sqlite"),
            "XG_KG_INDEX": str(index),
            "XG_TOKENIZER": str(tokenizer),
            "XG_KG_VERSIONS": str(data / "versions"),
            "XG_RELATION_LABELS": str(data / "kg/relation_labels.jsonl"),
            "XG_PATHS": "1",
            "XG_COMMITTED_PATHS": "1",
            "XG_ESCAPES": "0",
            "XG_START_RANK": "sitelinks",
            "XG_UNQUOTED_STARTS": "1",
            "XG_END_AFTER_ANSWER": "1",
            "XG_OPEN": "1",
            "XG_RELATIONS": "frontier",
            "XG_STOP": "",
            "XG_EXPRESSIONS": "0",
            "XG_DERIVATION": "0",
            "XG_REPEATS": "1",
            "XG_BOUND_ESCAPES": "1",
            "XG_LORA": "",
            "XG_RELEASE_MASS": "",
            "XG_RELEASE_TOP": "0",
            "XG_ESCAPE_BIAS": "",
            "XG_OBSERVE": "0",
            "XG_OBSERVE_SELECT": "0",
            "XG_OBSERVE_ANSWER": "0",
            "XG_OBSERVE_REQUIRED": "0",
            "XG_OBSERVE_NO_LOOP": "0",
            "XG_MENU": "0",
            "XG_AUDIT": "0",
            "XG_PATH_DEMOS": "",
            "XG_OBSERVE_WIDTH": "0",
            "XG_MENU_CLOSE_NEXT": "0",
            "XG_OBSERVE_DEFER": "0",
            "XG_OBSERVE_TERMINAL": "0",
            "XG_GCR_SYNTHESIS_MODEL": "",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "XG_LEXICAL_SCHEMA": "freebase",
        }

        def cpu(name, module, *args, family=family):
            jobs.append(
                {
                    "kind": "cpu",
                    "family": family,
                    "name": name,
                    "env": {
                        "CUDA_VISIBLE_DEVICES": "",
                        "OPENBLAS_NUM_THREADS": "1",
                        "OMP_NUM_THREADS": "1",
                        "TOKENIZERS_PARALLELISM": "false",
                        "XGRAPH_LEXICAL_SCHEMA": "freebase",
                    },
                    "command": [
                        "nice",
                        "-n",
                        "10",
                        "prlimit",
                        f"--as={7516192768 if pilot else 4294967296}",
                        sys.executable,
                        "-m",
                        module,
                        *map(str, args),
                    ],
                }
            )

        def gpu(
            name,
            request,
            *,
            arm="constrained",
            settings=None,
            extra=(),
            tokens=256,
            seed=17,
            generation=None,
            first_model=None,
            base=base,
            family=family,
            model=model,
            common=common,
            folder=folder,
        ):
            dest = folder / f"{name}.jsonl"
            assumption = timing_assumption(timing_root, family, name)
            jobs.append(
                {
                    "kind": "gpu",
                    "family": family,
                    "name": name,
                    "questions": questions,
                    "env": {**base, **(settings or {})},
                    "assumed_seconds_per_question": assumption["seconds_per_question"],
                    "timing_assumption": assumption,
                    "command": [
                        "nice",
                        "-n",
                        "10",
                        str(ROOT / "scripts/generate.sh"),
                        str(first_model or model),
                        str(request),
                        arm,
                        str(dest),
                        *(
                            generation["argv"]
                            if generation
                            else [
                                *common,
                                "--seed",
                                str(seed),
                                "--max-tokens",
                                str(tokens),
                                *map(str, extra),
                                *(
                                    ["--ban-tokens", "<|channel>"]
                                    if family == "gemma31" and arm not in ("gcr", "dog")
                                    else []
                                ),
                            ]
                        ),
                    ],
                }
            )
            return dest

        neutral = {
            "XG_PATHS": "0",
            "XG_COMMITTED_PATHS": "0",
            "XG_OPEN": "0",
            "XG_END_AFTER_ANSWER": "0",
        }

        def finish(
            name,
            raw,
            request,
            *,
            path_set=True,
            family=family,
            folder=folder,
            cpu=cpu,
            index=index,
            tokenizer=tokenizer,
            selected=selected,
        ):
            final = raw
            if path_set:
                final = folder / f"{name}-sets.jsonl"
                cpu(
                    name + "-sets",
                    "xgraph.webqsp",
                    "deliver",
                    "--outputs",
                    raw,
                    "--requests",
                    request,
                    "--db",
                    data / "kg.sqlite",
                    "--index",
                    index,
                    "--tokenizer",
                    tokenizer,
                    "--kg-versions",
                    data / "versions",
                    "--start-rank",
                    "sitelinks",
                    "--complete-path-set",
                    "--out",
                    final,
                )
            score = folder / (f"{name}-sets.score.json" if path_set else f"{name}.score.json")
            cpu(
                name + "-score",
                "xgraph.webqsp",
                "score",
                "--outputs",
                final,
                "--questions",
                inputs / "questions.json",
                "--failed-zero",
                "--released-only",
                "--population",
                primary_population,
                *(["--secondary-population"] if secondary_population else []),
                "--out",
                score,
            )
            declared_score = score.with_name(
                score.name.removesuffix(".score.json") + ".declared.score.json"
            )
            cpu(
                name + "-declared-score",
                "xgraph.webqsp",
                "score",
                "--outputs",
                final,
                "--questions",
                inputs / "questions.json",
                "--failed-zero",
                "--population",
                declared_population,
                "--secondary-population",
                "--out",
                declared_score,
            )
            if name in selected:
                results.append(
                    {
                        "family": family,
                        "arm": name,
                        "score_file": str(score),
                        "declared_score_file": str(declared_score),
                    }
                )

        for name in ("S", "blind5", "menuobs", "W"):
            if name not in needed:
                continue
            spec = (
                policy_recipe("M2" if name == "menuobs" else name, family)
                if name != "S"
                else {
                    "menu": 0,
                    "observe": 0,
                    "observe_width": 0,
                    "path_demos": 4,
                    "max_tokens": 256,
                    "seed": 17,
                }
            )
            request = inputs / f"requests-{'menuobs' if name == 'W' else name}.jsonl"
            settings = {
                "XG_MENU": str(spec["menu"]),
                "XG_OBSERVE": str(spec["observe"]),
                "XG_OBSERVE_WIDTH": str(spec["observe_width"]),
                "XG_PATH_DEMOS": str(spec["generation"]["first"]["path_demos"] or "")
                if "generation" in spec
                else str(spec["path_demos"]),
                **{
                    f"XG_OBSERVE_{k}": str(int(bool(spec["menu"])))
                    for k in ("SELECT", "ANSWER", "REQUIRED", "NO_LOOP")
                },
            }
            raw = gpu(
                name,
                request,
                settings=settings,
                tokens=spec["max_tokens"],
                seed=spec["seed"],
                generation=spec.get("generation", {}).get("first"),
            )
            finish(name, raw, request)
        for name in ("M2", "MT", "WT", "B2", "BT"):
            if name not in selected:
                continue
            spec = policy_recipe(name, family)
            first = "blind5" if name in ("B2", "BT") else "W" if name == "WT" else "menuobs"
            request = folder / f"{name}-requests.jsonl"
            base_request = inputs / "requests-S.jsonl"
            cpu(
                name + "-requests",
                "xgraph.webqsp",
                "second-requests",
                "--mode",
                spec["second"],
                "--outputs",
                folder / f"{first}.jsonl",
                "--requests",
                base_request,
                "--data",
                data,
                "--index",
                index,
                "--tokenizer",
                tokenizer,
                "--terminal-width",
                spec["terminal_width"],
                "--out",
                request,
            )
            raw = gpu(
                name + "-raw",
                request,
                settings={
                    "XG_OBSERVE_TERMINAL": str(spec["terminal_width"]),
                    "XG_PATH_DEMOS": str(spec["generation"]["second"]["path_demos"] or ""),
                },
                tokens=spec["second_max_tokens"],
                seed=spec["second_seed"],
                generation=spec["generation"]["second"],
            )
            merged = folder / f"{name}.jsonl"
            cpu(
                name + "-select",
                "xgraph.webqsp",
                "select-second",
                "--outputs",
                folder / f"{first}.jsonl",
                "--second",
                raw,
                "--requests",
                base_request,
                "--data",
                data,
                "--index",
                index,
                "--tokenizer",
                tokenizer,
                "--first-source",
                first,
                "--second-source",
                name + "-raw",
                "--out",
                merged,
            )
            finish(name, merged, base_request)
        for name in ("agent-strong-1", "agent-strong-4", "agent-strong-8", *FEEDBACK_ARMS):
            if name in selected:
                budget = policy_recipe(name, family)["agent_budget"]
                request = inputs / "requests-agent.jsonl"
                raw = gpu(
                    name,
                    request,
                    arm="agent",
                    settings=neutral,
                    generation=policy_recipe(name, family)["generation"]["first"],
                    extra=[
                        "--max-calls",
                        budget,
                        "--matched-demos",
                        "--matched-paths",
                        "--pages",
                        "--start-rank",
                        "sitelinks",
                        "--unquoted-starts",
                    ],
                )
                finish(name, raw, request)
        if "gcr-native" in selected:
            request = inputs / "requests-gcr.jsonl"
            raw = gpu(
                "gcr-native",
                request,
                arm="gcr",
                first_model=Path(gcr_model).absolute(),
                settings={
                    **neutral,
                    "XG_KG_INDEX": "",
                    "XG_TOKENIZER": "",
                    "XG_TOOLS_DB": str(data / "kg-gcr.sqlite"),
                    "XG_GCR_SYNTHESIS_MODEL": str(model),
                },
                tokens=512,
                seed=0,
                extra=[
                    "--gcr-k",
                    "10",
                    "--gcr-hops",
                    "2",
                    "--gcr-decoding",
                    "beam",
                    "--gcr-empty-trie",
                    "open",
                    "--gcr-max-paths",
                    "100000",
                    "--gcr-path-overflow",
                    "error",
                    "--gcr-batch-size",
                    "1",
                    "--gcr-parallel-relations",
                    "preserve",
                ],
            )
            jobs[-1]["startup_seconds"] = [180, 360]
            finish("gcr-native", raw, request, path_set=False)
        if "dog" in selected:
            request = folder / "dog-requests.jsonl"
            cpu(
                "dog-requests",
                "xgraph.webqsp",
                "dog-requests",
                "--data",
                data,
                "--requests",
                inputs / "requests-gcr.jsonl",
                "--out",
                request,
                *(["--validation-proxy"] if pilot and not secondary_population else []),
            )
            raw = gpu(
                "dog",
                request,
                arm="dog",
                settings={**neutral, "XG_KG_INDEX": "", "XG_TOKENIZER": ""},
                tokens=512,
                extra=[
                    "--dog-prompts",
                    str(Path(dog_prompts).absolute()),
                    "--dog-beam",
                    "1",
                    "--dog-steps",
                    "5",
                    "--dog-batch-size",
                    "32",
                ],
            )
            finish("dog", raw, request, path_set=False)
    signature = hashlib.sha256(
        json.dumps(
            {
                "families": sorted(models),
                "dog_graph_source": "released_top120"
                if secondary_population
                else "validation_bm25_proxy",
                "published_inputs": [
                    (str(p), hashlib.sha256(p.read_bytes()).hexdigest())
                    for p in [Path(dog_prompts), *sorted(Path(gcr_model).glob("*.json"))]
                    if p.is_file()
                ],
                "model_inventory": {
                    family: [
                        (p.name, hashlib.sha256(p.read_bytes()).hexdigest())
                        for p in sorted(Path(model).glob("*.json"))
                    ]
                    + [
                        (p.name, p.stat().st_size, p.stat().st_mtime_ns)
                        for p in sorted(Path(model).glob("*.safetensors"))
                    ]
                    for family, model in models.items()
                },
                "demos": CATALOG_DEMOS,
                "code": [
                    (p.name, hashlib.sha256(p.read_bytes()).hexdigest())
                    for p in [
                        *sorted(Path(__file__).parent.glob("*.py")),
                        ROOT / "scripts/generate.sh",
                        ROOT / "scripts/webqsp-queue.sh",
                    ]
                ],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    gpu_jobs = [j for j in jobs if j["kind"] == "gpu"]
    estimate = {
        "measured": False,
        "pilot_measured": False,
        "questions_per_arm": questions,
        "low_gpu_hours": sum(
            questions * j["assumed_seconds_per_question"][0]
            + j.get("startup_seconds", [90, 180])[0]
            for j in gpu_jobs
        )
        / 3600,
        "high_gpu_hours": sum(
            questions * j["assumed_seconds_per_question"][1]
            + j.get("startup_seconds", [90, 180])[1]
            for j in gpu_jobs
        )
        / 3600,
        "startup_seconds_per_job": [90, 180],
        "rule": "MINTQA workload proxies and unmeasured reader forecasts, plus startups; "
        "shared first calls counted once; CPU stages separate; no speed claim",
    }
    if pilot_timings is not None:
        estimate = {
            **estimate,
            **price_from_pilot(jobs, json.loads(Path(pilot_timings).read_text()), signature),
        }
    return {
        "catalog": True,
        "catalog_signature": signature,
        "arms": declared,
        "frozen_policies": frozen,
        "policy_recipes": {
            f: {a: policy_recipe(a, f) for a in (*ARMS, *FEEDBACK_ARMS) if a != "S"} for f in models
        },
        "jobs": jobs,
        "primary_results": results,
        "questions": questions,
        "primary_questions": questions
        if pilot
        else questions - metadata.get("counts", {}).get("missing_subgraphs", 0),
        "pilot": pilot_info,
        "project_questions": project_questions,
        "request_kinds": ["S", "blind5", "menuobs", "agent", "gcr"],
        "protocol": "RoG/GCR label protocol on released subgraphs; "
        "full declared-panel sensitivity; failed outputs count zero",
        "population": primary_population,
        "declared_population": declared_population,
        "secondary_population": secondary_population,
        "primary_metrics": ["gcr_hit", "gcr_f1"],
        "secondary_metrics": ["hits_at_1", "f1"],
        "delivery": "all endpoints of committed path, intermediate selections retained; "
        "terminal label ranks set; capped and uncapped support recorded",
        "estimate": estimate,
        "gcr_published_reproduction": False,
        "published_comparators": {
            "gcr-native": {
                "status": "declared",
                "planner": str(Path(gcr_model).absolute()),
                "paper": "https://arxiv.org/abs/2410.13080",
                "hops": 2,
                "beams": 10,
                "deviations": [
                    "host joint beam instead of released HF group-beam",
                    "reconstructed prompts; bounded trie errors on overflow",
                    "local family reader instead of paper ChatGPT/GPT-4o-mini; complete label sets",
                    "released DiGraph last-edge collapse; question-local label identity",
                ],
            },
            "dog": {
                "status": "declared" if "dog" in declared else "separate_test_population",
                "graph_source": "released_top120"
                if secondary_population
                else "validation_bm25_proxy"
                if pilot
                else "separate_released_top120_panel",
                "deviations": [
                    "local family backbone and complete-set answer parsing",
                    "endpoint expansion, whole-triple trie and complete string stops "
                    "from baselines4",
                    "validation BM25 timing proxy is not released-graph quality "
                    "or qualified test timing",
                ],
            },
            "tog-r": {
                "status": "blocked_freebase_adapter",
                "missing": "baselines4 ToG-R uses Wikidata prompts/parser; "
                "no Freebase relation/CVT adapter or parity tests",
                "engineering_days_estimate": [3, 5],
            },
            "cor": {
                "status": "blocked_freebase_adapter",
                "missing": "upstream Freebase SPARQL backend exists; local reader still "
                "hardcodes Wikidata PID exclusions and noise filters; "
                "needs Freebase SQLite/CVT parity tests",
                "engineering_days_estimate": [2, 4],
            },
        },
        "limitations": [
            "No WebQSP generation or controlled GPU timing measured.",
            "Gold presence is used only to define the labelled DoG secondary population.",
            "Removing terminal selection for set delivery is an explicit WebQSP variant; "
            "intermediate selection and cap losses remain.",
        ],
    }
