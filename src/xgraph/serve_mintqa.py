"""Live MINTQA policies on the owned controlled-cost server; no model imports.

Reuse the native rematch prompts, agent, matched execution and second-call
builders. Every dependent request is constructed inside the reserved interval.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from xgraph.policy_catalog import ALIASES, CATALOG, FINALISTS, fingerprint
from xgraph.policy_catalog import recipe as catalog_recipe

SERVING_CAP_SECONDS = 24 * 3600
SERVING_CLEANUP_SECONDS = 180
REDUCED_DESIGN = {
    "primary": {"questions": 128, "repeats": 5, "arms": ["WT", "strongest-agent"]},
    "descriptive": {"questions": 32, "repeats": 3, "arms": ["W", "B2"]},
    "families": ["qwen38", "gemma31"],
    "concurrency": [1, 8],
    "cache": ["warm"],
    "seed": 17,
    "selection": "id-hash",
    "lifetime_cap_gpu_seconds": SERVING_CAP_SECONDS,
    "overhead": "locked WT replay",
    "screen": False,
    "cold": False,
    "top_up": False,
}


class ServingBudget:
    """Seal-bound lifetime reservation, shared by serial cells and engine replay.

    Each interval encloses the worker and confirmed cleanup, conservatively
    including worker CPU preparation. Headroom waits and wheel builds precede
    these intervals. An unknown lifetime retains its entire reservation.
    """

    def __init__(self, path: Path, seal_sha256: str):
        self.path = Path(path)
        self.data = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {
                "seal_sha256": seal_sha256,
                "lifetime_cap_gpu_seconds": SERVING_CAP_SECONDS,
                "charged_gpu_seconds": 0,
                "status": "active",
                "attempts": [],
                "boundary": "worker start through confirmed cleanup; conservative GPU reservation",
            }
        )
        if (
            self.data["seal_sha256"] != seal_sha256
            or self.data["lifetime_cap_gpu_seconds"] != SERVING_CAP_SECONDS
        ):
            raise ValueError("serving budget identity or lifetime cap changed")
        charged = 0
        for attempt in self.data["attempts"]:
            value = attempt.get("charged_gpu_seconds", attempt["reserved_gpu_seconds"])
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid serving budget charge")
            charged += value
        if charged != self.data["charged_gpu_seconds"]:
            raise ValueError("serving budget charge total changed")

    def write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=1) + "\n")
        temporary.replace(self.path)

    def begin(self, label: str, wall_seconds: float) -> dict:
        if any(a["status"] == "running" for a in self.data["attempts"]):
            raise RuntimeError("unclosed serving budget attempt; INCOMPLETE, no relaunch")
        if self.data["status"] == "INCOMPLETE":
            raise RuntimeError("serving budget INCOMPLETE; no top-up")
        remaining = SERVING_CAP_SECONDS - self.data["charged_gpu_seconds"]
        if remaining <= SERVING_CLEANUP_SECONDS:
            self.data["status"] = "INCOMPLETE"
            self.write()
            raise RuntimeError("serving lifetime cap: INCOMPLETE; no top-up")
        bound = min(wall_seconds, remaining - SERVING_CLEANUP_SECONDS)
        attempt = {
            "label": label,
            "status": "running",
            "started_monotonic": time.monotonic(),
            "started_unix": time.time(),
            "wall_seconds": bound,
            "capped": bound < wall_seconds,
            "reserved_gpu_seconds": bound + SERVING_CLEANUP_SECONDS,
        }
        self.data["attempts"].append(attempt)
        self.data["charged_gpu_seconds"] += attempt["reserved_gpu_seconds"]
        self.write()  # durable reservation before starting any GPU worker
        return attempt

    def finish(self, attempt: dict, *, cleanup_confirmed: bool, failed: bool):
        elapsed = max(0, time.monotonic() - attempt["started_monotonic"])
        charge = elapsed if cleanup_confirmed else attempt["reserved_gpu_seconds"]
        attempt.update(
            status="failed" if failed else "completed",
            elapsed_seconds=elapsed,
            cleanup_confirmed=cleanup_confirmed,
            charged_gpu_seconds=charge,
        )
        self.data["charged_gpu_seconds"] = sum(
            a.get("charged_gpu_seconds", a["reserved_gpu_seconds"]) for a in self.data["attempts"]
        )
        if (
            not cleanup_confirmed
            or (failed and attempt["capped"])
            or self.data["charged_gpu_seconds"] >= SERVING_CAP_SECONDS
        ):
            self.data["status"] = "INCOMPLETE"
        self.write()


DEFAULT_POLICIES = (
    "paths-allfix",
    "paths-allfix-stop",
    "paths-menu-5-observed-3",
    "two-pass",
    "audit",
    "agent-strong-1",
    "agent-strong-2",
    "agent-strong-4",
    "agent-strong-8",
    "cascade",
    "cascade-menu",
    "cascade-audit",
)

POLICIES = (
    *DEFAULT_POLICIES,
    "blind5",
    "blind5-audit",
    "blind5-two-pass",
    "menuobs-audit",
    "menuobs-two-pass",
    *FINALISTS,
    "menuobs-audit-terminal",
)

FRONTIER_POLICIES = {
    "blind5",
    "blind5-audit",
    "blind5-two-pass",
    "menuobs-audit",
    "menuobs-two-pass",
    *FINALISTS,
    "menuobs-audit-terminal",
}


def policy_recipe(name: str, family=None, *, qualified_feedback=False) -> dict:
    """Explicit rematch contracts; budget labels count external agent tool calls."""
    from xgraph.serve_cost import calibration_settings

    if name not in (*POLICIES, *CATALOG, *ALIASES):
        raise ValueError(f"unknown live MINTQA policy {name!r}")
    spec = catalog_recipe(name, family) if name in (*CATALOG, *ALIASES) else None
    if spec and spec["agent_feedback"] and not qualified_feedback:
        raise ValueError("feedback serving requires a qualified frozen selection")
    agent = bool(spec["agent_budget"]) if spec else name.startswith("agent-strong-")
    menu = (
        bool(spec["menu"])
        if spec
        else (name in ("paths-menu-5-observed-3", "cascade-menu") or name.startswith("menuobs-"))
    )
    settings = calibration_settings("paths" if name == "paths-allfix-stop" else "paths-no-stop")[
        "engine_settings"
    ]
    settings.update(
        XGRAPH_MENU="5" if menu else "0",
        XGRAPH_OBSERVE="3" if menu else "0",
        XGRAPH_AUDIT="0",
        XGRAPH_REPEATS="1",
        XGRAPH_BOUND_ESCAPES="1",
        **{
            f"XGRAPH_OBSERVE_{k}": "1" if menu else "0"
            for k in ("SELECT", "ANSWER", "REQUIRED", "NO_LOOP")
        },
    )
    recipe = {
        "policy": name,
        "max_tokens": 1024 if menu or name.startswith("blind5") else 256,
        "agent_max_tokens": 256,
        "agent_budget": spec["agent_budget"]
        if agent and spec
        else int(name.rsplit("-", 1)[1])
        if agent
        else 8,
        "opt_in": not agent,
        "opener": None if agent else "⟦",
        "engine_settings": settings,
        "options": {"revision": name == "two-pass" or name.endswith("-two-pass")},
    }
    if name in FRONTIER_POLICIES or (spec and not agent):
        settings["XGRAPH_REQUEST_PATH_OPTIONS"] = "1"
        recipe.update(
            first_path_options={
                "menu": 5 if menu else 0,
                "observe": 3 if menu else 0,
                "path_demos": 5,
                **{f"observe_{k}": int(menu) for k in ("select", "answer", "required", "no_loop")},
            },
            second_path_options={"menu": 0, "observe": 0, "path_demos": 4},
            second_max_tokens=256,
            stage_seeds={"first": 0, "second": 17},
        )
    if spec:
        recipe.update(
            catalog_policy=spec,
            policy_sha256=fingerprint(spec),
            max_tokens=spec["max_tokens"],
            second_kind=spec["second"],
            agent_feedback=spec["agent_feedback"],
        )
        recipe["options"]["revision"] = bool(
            spec["second"] and spec["second"].startswith("two-pass")
        )
        if agent:
            recipe["stage_seeds"] = {"agent": spec["seed"]}
        if not agent:
            recipe["first_path_options"]["observe_width"] = spec["observe_width"]
            settings["XGRAPH_OBSERVE_WIDTH"] = str(spec["observe_width"])
            if spec["terminal_width"]:
                settings["XGRAPH_OBSERVE_TERMINAL"] = str(spec["terminal_width"])
                recipe["second_path_options"]["observe_terminal"] = spec["terminal_width"]
    return recipe


def run_policy(url: str, request: dict, arm: dict, *, context: dict, timeout=600) -> dict:
    """Execute fresh outputs, all CPU stages and only actually selected branches.

    HTTP failures retain their attempt and elapsed interval with unknown usage.
    They fail the cell; no retry, missing cost or successful-only denominator is
    introduced. Native agent policy retries remain in its own charged trajectory.
    """
    from xgraph.agent import demonstrations, run_agent
    from xgraph.generate import walk_requests
    from xgraph.mintqa_rematch import (
        audit_requests,
        escalation_ids,
        revision_records,
        revision_requests,
        support_records,
    )
    from xgraph.programs import record_path
    from xgraph.prompts import final_answer
    from xgraph.serve_cost import _TimedStore
    from xgraph.serving import _one

    started = time.perf_counter()
    calls, stages = [], []
    policy = arm["policy"]
    qid = request["id"]
    base = {
        k: v for k, v in request.items() if k not in ("record", "agent_request", "blind_request")
    }
    blind_base = request.get("blind_request", base)
    agent_base = request.get("agent_request", base)
    original_question = next(
        m["content"] for m in reversed(base["messages"]) if m["role"] == "user"
    )
    read = _TimedStore(context["store"])
    executor = context["executor"]
    executor.calls, executor.seconds = 0, 0.0
    if context.get("sql") is not None:
        context["sql"].statements = 0
    native_agent = None
    flags = {}

    def stage(name, action):
        wall, cpu = time.perf_counter(), time.thread_time()
        try:
            return action()
        finally:
            stages.append(
                {
                    "stage": name,
                    "wall_seconds": time.perf_counter() - wall,
                    "cpu_seconds": time.thread_time() - cpu,
                }
            )

    def generate(name, asked, opt_in=True, maximum=None):
        asked = {
            **asked,
            "seed": arm.get("stage_seeds", {}).get(name, base.get("seed", 17)),
            "sampling": base.get("sampling", {}),
        }
        path_options = arm.get(f"{name}_path_options") if opt_in else None
        extra = {"xgraph_question": original_question} if opt_in else None
        if path_options is not None:
            extra["xgraph_path_options"] = path_options
        entry = {
            "stage": name,
            "request": asked,
            "started_seconds": time.perf_counter() - started,
            "status": "running",
            "max_tokens": maximum or arm["max_tokens"],
        }
        calls.append(entry)
        if path_options is not None:
            entry["engine_arguments"] = extra
        wall = time.perf_counter()
        try:
            measured = _one(
                url,
                "/model",
                asked,
                entry["max_tokens"],
                opt_in,
                arm.get("opener")
                if opt_in and asked["messages"][-1]["role"] != "assistant"
                else None,
                timeout=timeout,
                extra_args=extra,
                return_token_ids=True,
            )
            entry.update(measured, status="completed")
            if arm.get("require_request_metrics"):
                metrics = measured.get("engine_metrics") or {}
                if (
                    any(
                        type(metrics.get(k)) not in (int, float)
                        or not math.isfinite(metrics[k])
                        or metrics[k] < 0
                        for k in ("time_to_first_token_ms", "generation_time_ms", "queue_time_ms")
                    )
                    or type(measured.get("cached_prompt_tokens")) is not int
                    or not (0 <= measured["cached_prompt_tokens"] <= measured["prompt_tokens"])
                ):
                    raise ValueError(f"{qid}: native engine timings or prefix-cache usage missing")
            entry["observation_tokens"] = 0
            menu = (
                path_options["menu"]
                if path_options is not None
                else int(arm["engine_settings"].get("XGRAPH_MENU", "0"))
            )
            if opt_in and menu:
                from xgraph.menus import semantic_text
                from xgraph.observations import PathRelations, count_observation_tokens, path_parts

                served = context["served_tokenizer"]
                syntax = executor._index.span_syntax

                def selection_required(body):
                    body = semantic_text(syntax.opener + body, syntax)[len(syntax.opener) :]
                    parts, selections = path_parts(
                        body, syntax.separator, syntax.opener, syntax.escape
                    )
                    return (
                        len(
                            executor.frontier(
                                parts[0],
                                PathRelations(parts[1:], selections),
                                base.get("anchor"),
                                base.get("kg_version"),
                            )
                        )
                        > 1
                    )

                entry["observation_tokens"] = count_observation_tokens(
                    measured["token_ids"],
                    served,
                    syntax,
                    prefill=measured.get("opener", ""),
                    force_answer=True,
                    selection_required=selection_required,
                )
            terminal = (
                path_options.get("observe_terminal", 0)
                if path_options is not None
                else int(arm["engine_settings"].get("XGRAPH_OBSERVE_TERMINAL", "0"))
            )
            if opt_in and terminal:
                from xgraph.audit import count_terminal_tokens

                entry["observation_tokens"] = count_terminal_tokens(
                    measured["token_ids"],
                    context["served_tokenizer"],
                    executor._index.span_syntax,
                    prefill=measured.get("opener", ""),
                )
            return {**measured, "generated_tokens": measured["completion_tokens"]}
        except Exception as error:
            entry.update(
                status="failed",
                error=str(error),
                latency=time.perf_counter() - wall,
                prompt_tokens=entry.get("prompt_tokens"),
                completion_tokens=entry.get("completion_tokens"),
            )
            raise

    def agent():
        nonlocal native_agent

        class Model:
            def chat(self, conversations, sampling, **kwargs):
                result = []
                for messages in conversations:
                    row = generate(
                        "agent", {"id": qid, "messages": messages}, False, arm["agent_max_tokens"]
                    )
                    result.append(
                        SimpleNamespace(
                            prompt_token_ids=[0] * row["prompt_tokens"],
                            outputs=[
                                SimpleNamespace(
                                    text=row["text"],
                                    token_ids=[0] * row["completion_tokens"],
                                    finish_reason=row["finish_reason"],
                                )
                            ],
                        )
                    )
                return result

        asked = {**agent_base, **walk_requests([agent_base])[0]}
        asked["demonstrations"] = demonstrations(agent_base["messages"])
        (native_agent,) = run_agent(
            Model(),
            [asked],
            read,
            SimpleNamespace(max_tokens=arm["agent_max_tokens"]),
            max_calls=arm["agent_budget"],
            names=context["names"],
            executor=executor,
            pages=True,
            **({"feedback": True} if arm.get("agent_feedback") else {}),
        )
        return native_agent

    try:
        if arm.get("agent_feedback") or policy.startswith("agent-strong-"):
            row = stage("agent", agent)
        else:
            row = stage("first", lambda: generate("first", base))
            first = row
            if (
                policy in ("audit", "two-pass", "cascade-audit")
                or (policy in FRONTIER_POLICIES and policy not in ("blind5", "W"))
                or arm.get("second_kind")
            ):

                def build_second():
                    terminal = arm.get("catalog_policy", {}).get("terminal_width", 0)
                    if arm["options"].get("revision"):
                        path = record_path(first)
                        found, empty = (
                            context["repairer"].candidates(
                                original_question, *path, base.get("anchor"), base.get("kg_version")
                            )
                            if path
                            else ([], 0)
                        )
                        candidate = {
                            "id": qid,
                            "question": original_question,
                            "start": path[0] if path else None,
                            "text": first["text"],
                            "answer": final_answer(first["text"]),
                            "candidates": found,
                            "empty": empty,
                        }
                        return revision_requests(
                            [blind_base],
                            [first],
                            [candidate],
                            executor,
                            read.label,
                            **({"observe_terminal": terminal} if terminal else {}),
                        )[0]
                    return audit_requests(
                        [blind_base],
                        [first],
                        executor,
                        read.label,
                        context["audit_menus"],
                        context["served_tokenizer"],
                        **({"observe_terminal": terminal} if terminal else {}),
                    )[0]

                second_request = stage("build_second", build_second)
                second = stage(
                    "second",
                    lambda: generate(
                        "second", second_request, maximum=arm.get("second_max_tokens")
                    ),
                )
                support = stage(
                    "second_support", lambda: support_records([second], [blind_base], executor)
                )
                row = stage(
                    "revision_select",
                    lambda: revision_records(
                        [first], [second], support, [qid], second_source=policy
                    ),
                )[0]
                flags["revision_accepted"] = row["revision_accepted"]
            if policy in ("cascade", "cascade-menu", "cascade-audit"):
                support = stage("route_support", lambda: support_records([row], [base], executor))
                route = bool(escalation_ids([row], support, [qid]))
                if policy == "cascade-audit":
                    # Two independently generated answers agree only after normalizing
                    # harmless formatting. No answer gold or cached agent is read.
                    from xgraph.score import normalize

                    route |= not flags["revision_accepted"] or normalize(
                        final_answer(first["text"]) or ""
                    ) != normalize(final_answer(row["text"]) or "")
                flags["escalated"] = route
                if route:
                    row = stage("agent", agent)
        latency = time.perf_counter() - started
        return {
            "id": qid,
            "text": row["text"],
            "trajectory_kind": "live_mintqa_policy",
            "calls": calls,
            "stages": stages,
            "model_calls": len(calls),
            "tool_calls": native_agent.get("tool_calls", len(native_agent.get("actions", [])))
            if native_agent
            else 0,
            "retries": native_agent.get("retries", 0) if native_agent else 0,
            "kg_store_calls": read.calls,
            "kg_store_seconds": read.seconds,
            "kg_sql_statements": context["sql"].statements if context.get("sql") else None,
            "executor_calls": executor.calls,
            "executor_seconds": executor.seconds,
            "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
            "completion_tokens": sum(c["completion_tokens"] for c in calls),
            "observation_tokens": sum(c["observation_tokens"] for c in calls),
            "cached_prompt_tokens": sum(c["cached_prompt_tokens"] for c in calls)
            if all(c.get("cached_prompt_tokens") is not None for c in calls)
            else None,
            "uncached_prompt_tokens": sum(
                c["prompt_tokens"] - c["cached_prompt_tokens"] for c in calls
            )
            if all(c.get("cached_prompt_tokens") is not None for c in calls)
            else None,
            "engine_prefill_first_token_seconds": sum(
                (c["engine_metrics"] or {})["time_to_first_token_ms"] for c in calls
            )
            / 1000
            if all(
                (c.get("engine_metrics") or {}).get("time_to_first_token_ms") is not None
                for c in calls
            )
            else None,
            "engine_decode_seconds": sum(c["engine_metrics"]["generation_time_ms"] for c in calls)
            / 1000
            if all(
                (c.get("engine_metrics") or {}).get("generation_time_ms") is not None for c in calls
            )
            else None,
            "engine_queue_seconds": sum(c["engine_metrics"]["queue_time_ms"] for c in calls) / 1000
            if all((c.get("engine_metrics") or {}).get("queue_time_ms") is not None for c in calls)
            else None,
            "client_thread_cpu_seconds": sum(s["cpu_seconds"] for s in stages),
            "ttft": calls[0]["started_seconds"] + calls[0]["ttft"]
            if calls[0].get("ttft") is not None
            else None,
            "finish_reason": row.get("finish_reason"),
            "truncated": any(c.get("finish_reason") == "length" for c in calls),
            "reader_cpu_and_tool_wall_seconds": max(0, latency - sum(c["latency"] for c in calls)),
            "native_agent": native_agent,
            **flags,
        }
    except Exception as error:
        error.result = {
            "calls": calls,
            "stages": stages,
            "model_calls": len(calls),
            "prompt_tokens": None,
            "completion_tokens": None,
        }
        raise


def timing_panel(
    members: list[dict], questions: int = 128, seed: int = 17, *, selection="population-matched"
) -> dict:
    """Uniform draws within original equal-weight/hop strata, with survey correction.

    Integer quotas approximate the population mixture without replacement. Each
    stratum retains its original total weight exactly in the weighted estimand.
    This makes batched physical throughput close to, but not exactly, the target
    population; the weighted batched allocation is reported only as a sensitivity.
    """
    if len({r["id"] for r in members}) != len(members):
        raise ValueError("duplicate development panel id")
    if selection == "id-hash":
        if type(questions) is not int or not 0 < questions <= len(members):
            raise ValueError("timing subset capacity must fit the development panel")
        if any(not math.isfinite(r["weight"]) or r["weight"] <= 0 for r in members):
            raise ValueError("development population weights must be finite and positive")
        probability = questions / len(members)
        selected = sorted(
            members,
            key=lambda r: (hashlib.sha256(f"{seed}:{r['id']}".encode()).digest(), r["id"]),
        )[:questions]
        selected = sorted(
            (
                {
                    **r,
                    "source_weight": r["weight"],
                    "subset_probability": probability,
                    "weight": r["weight"] / probability,
                }
                for r in selected
            ),
            key=lambda r: r["id"],
        )
        return {
            "ids": [r["id"] for r in selected],
            "members": selected,
            "seed": seed,
            "sampling": "SHA256(seed:id) prefix, ties by id; uniform without replacement",
            "development_questions": len(members),
            "population_weight": sum(r["weight"] for r in members),
        }
    if selection != "population-matched":
        raise ValueError(f"unknown timing selection: {selection}")
    groups = defaultdict(list)
    for member in members:
        w = member["weight"]
        if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0:
            raise ValueError("development population weights must be finite and positive")
        groups[(member["hops"], float(w))].append(member)
    if type(questions) is not int or not len(groups) <= questions <= len(members):
        raise ValueError(
            "timing subset capacity must cover every weight stratum without replacement"
        )
    mass = {key: len(rows) * key[1] for key, rows in groups.items()}
    total = sum(mass.values())
    quotas = dict.fromkeys(sorted(groups), 1)
    for _ in range(questions - len(groups)):
        key = max(
            (key for key in quotas if quotas[key] < len(groups[key])),
            key=lambda key: (questions * mass[key] / total - quotas[key], key),
        )
        quotas[key] += 1
    rng = random.Random(seed)
    selected = []
    for key, count in quotas.items():
        rows = sorted(groups[key], key=lambda r: r["id"])
        probability = count / len(rows)
        selected.extend(
            {
                **row,
                "source_weight": row["weight"],
                "subset_probability": probability,
                "weight": row["weight"] / probability,
            }
            for row in rng.sample(rows, count)
        )
    selected.sort(key=lambda r: r["id"])
    return {
        "ids": [r["id"] for r in selected],
        "members": selected,
        "seed": seed,
        "sampling": "uniform within equal survey-weight/hop strata; population-proportional quotas",
        "development_questions": len(members),
        "population_weight": total,
    }


def freeze_arms(args, panel: dict, tokenizer: Path, header: dict, inputs: list[Path]) -> dict:
    """Freeze online recipes, never reading historical completions or validation requests."""
    from xgraph.prompts import menu_requests, observed_requests, path_demo_requests
    from xgraph.serve_cost import panel_records
    from xgraph.syntax import SpanSyntax
    from xgraph.tokenizer import ServedTokenizer

    if (
        args.arm_config
        or args.reader
        or args.reader_config
        or args.engine_config
        or args.kg_versions
    ):
        raise ValueError("live MINTQA cannot mix replay recipes, engine overrides or KG versions")
    if not args.tools_db or not args.relation_labels:
        raise ValueError("live MINTQA needs --tools-db and --relation-labels")
    if not header.get("sitelinks") or not (args.kg_index / "sitelinks.npy").is_file():
        raise ValueError("live MINTQA needs the rematch index with sitelink counts")
    if ServedTokenizer.from_file(tokenizer).fingerprint != header["tokenizer_fingerprint"]:
        raise ValueError("MINTQA served tokenizer differs from the KG index")
    if args.repeats < 3 or not args.cache or set(args.cache) - {"cold", "warm"}:
        raise ValueError(
            "live MINTQA needs at least three repetitions and a cold/warm cache subset"
        )
    if not args.concurrency or set(args.concurrency) - {1, 8}:
        raise ValueError("live MINTQA concurrency must be a declared subset of 1 and 8")
    if args.cost_resamples <= 0:
        raise ValueError("live MINTQA needs positive --cost-resamples")
    if args.max_tokens != 256 or args.max_model_len <= 1024:
        raise ValueError("MINTQA rematch budgets are 256, with 1024 for menu/observed")
    if args.cost_ratio or args.cost_clusters:
        raise ValueError("MINTQA comparisons and relation clusters come from the frozen panel")
    inputs.extend([args.tools_db, args.relation_labels])
    inputs.extend([p for p in (args.kg_index.with_suffix(".manifest.json"),) if p.is_file()])
    requests = panel_records(args.paths_requests, [*panel["ids"], *panel.get("warmup_ids", [])])
    if any(r.get("anchor") or r.get("kg_version") for r in requests):
        raise ValueError("MINTQA rematch must use unanchored base-KG requests")
    syntax = SpanSyntax(**header.get("span_syntax", {}))
    frozen = None
    family = args.policy_family
    if args.policy_freeze:
        from xgraph.policy_catalog import frozen_recipe

        frozen = frozen_recipe(args.policy_freeze, family, model=args.model)
        family = frozen["family"]
        inputs.extend([args.policy_freeze, Path(f"{args.policy_freeze}.sha256")])
    arms = {}
    for name in args.arm or DEFAULT_POLICIES:
        if name == "finalist" and frozen is None:
            raise ValueError("--arm finalist requires --policy-freeze and --policy-family")
        recipe = policy_recipe(
            frozen["recipe"]["policy"] if name == "finalist" else name,
            family,
            qualified_feedback=bool(
                frozen
                and (
                    frozen["recipe"]["agent_feedback"]
                    or "agent-feedback-8" in frozen.get("comparators", [])
                )
            ),
        )
        if (
            frozen
            and "finalist" in (args.arm or ())
            and name != "finalist"
            and recipe.get("catalog_policy") == frozen["recipe"]
        ):
            continue
        asked = requests
        if (
            recipe.get("first_path_options", {}).get("path_demos") == 5
            and recipe["engine_settings"]["XGRAPH_MENU"] == "0"
        ):
            asked = path_demo_requests(requests, 5, syntax)
        if recipe["engine_settings"]["XGRAPH_MENU"] == "5":
            profile = recipe.get("first_path_options", {})
            demos = profile.get("path_demos")
            width = profile.get("observe_width", 0)
            asked = menu_requests(
                observed_requests(requests, 3, syntax, select=True, width=width, demos=demos),
                5,
                syntax,
                observe=3,
                select=True,
                demos=demos,
                width=width,
            )
            if name == "cascade-menu":
                asked = [
                    {**menu, "agent_request": original}
                    for menu, original in zip(asked, requests, strict=True)
                ]
        if recipe.get("first_path_options"):
            asked = [
                {**first, "blind_request": original}
                for first, original in zip(asked, requests, strict=True)
            ]
        arms[name] = {
            **recipe,
            "kind": "mintqa",
            "live_mintqa": True,
            "require_request_metrics": True,
            "requests": asked[: len(panel["ids"])],
            "warmup_requests": asked[len(panel["ids"]) :],
            "opener": syntax.opener if recipe["opt_in"] else None,
            "options": {
                "start_rank": "sitelinks",
                "unquoted_starts": True,
                "matched_paths": True,
                "relation_labels": str(args.relation_labels.resolve()),
                **recipe["options"],
            },
            "tools_db": str(args.tools_db.resolve()),
            "kg_index": str(args.kg_index.resolve()),
            "tokenizer": str(tokenizer.resolve()),
            "model": str(args.model.resolve()),
            "cost_reference": (frozen or {}).get("strongest_agent") or "agent-strong-8",
            "policy_freeze": frozen,
        }
    reference = (frozen or {}).get("strongest_agent") or "agent-strong-8"
    if args.descriptive and set(arms) != {"W", "B2"}:
        raise ValueError("descriptive MINTQA timing is W/B2 only")
    if reference not in arms and not args.descriptive:
        raise ValueError(f"MINTQA timing needs the paired {reference} reference")
    return arms


def reservation_estimate(study: dict, trials: list[dict]) -> dict:
    """Explicit planning assumptions, never a token-to-time conversion or a measurement."""
    serial = {
        "W": 3,
        "M2": 7,
        "MT": 7,
        "WT": 7,
        "B2": 4,
        "BT": 4,
        "menuobs-audit-terminal": 6,
        "blind5": 1,
        "blind5-audit": 3,
        "blind5-two-pass": 4,
        "menuobs-audit": 6,
        "menuobs-two-pass": 7,
        "paths-allfix": 1,
        "paths-allfix-stop": 0.6,
        "paths-menu-5-observed-3": 3,
        "two-pass": 4,
        "audit": 3,
        "agent-strong-1": 2,
        "agent-strong-2": 3,
        "agent-strong-4": 6,
        "agent-strong-8": 12,
        "agent-strong-8-feedback": 12,
        "agent-feedback-8": 12,
        "cascade": 7,
        "cascade-menu": 9,
        "cascade-audit": 9,
    }
    startup, teardown = 90, 15
    service = warmup = 0.0
    for trial in trials:
        factor = 1 if trial["concurrency"] == 1 else 2.5
        arm = study.get("arms", {}).get(trial["arm"], {})
        policy = arm.get("catalog_policy", {}).get("policy", arm.get("policy", trial["arm"]))
        seconds = serial[policy] / factor
        service += len(study["panel_ids"]) * seconds
        if trial["cache"] == "warm":
            warmup += min(study["warmup_questions"], len(study["panel_ids"])) * seconds
    fixed = len(trials) * (startup + teardown)
    return {
        "measured": False,
        "cells": len(trials),
        "questions": len(study["panel_ids"]),
        "gpu_hours": (fixed + service + warmup) / 3600,
        "double_service_gpu_hours": (fixed + 2 * (service + warmup)) / 3600,
        "service_gpu_hours": service / 3600,
        "warmup_gpu_hours": warmup / 3600,
        "startup_teardown_gpu_hours": fixed / 3600,
        "assumptions": {
            "serial_seconds_per_question": serial,
            "concurrency_8_speedup": 2.5,
            "startup_seconds_per_cell": startup,
            "teardown_seconds_per_cell": teardown,
            "cascade_agent_fraction": 0.5,
        },
        "excluded": "index construction, qualification, preparation, external reservation waits",
    }


def score_live_trial(report: dict, answers: dict[str, str]) -> None:
    """Score retained live outputs after timing; every admitted question stays charged."""
    from xgraph.mintqa_bench import answer_contains
    from xgraph.prompts import final_answer
    from xgraph.score import record_failure

    rows = {r["id"]: r for r in report["results"]}
    if len(rows) != len(report["results"]) or set(rows) - answers.keys():
        raise ValueError("live quality has duplicate or unregistered question ids")
    correct = {
        q: (
            q in rows
            and rows[q].get("status") == "completed"
            and not record_failure(rows[q])
            and answer_contains(final_answer(rows[q]["text"]), answer)
        )
        for q, answer in answers.items()
    }
    report["quality"] = {
        "correct": correct,
        "scorer": "MINTQA normalized final-answer containment; failures wrong",
    }
    report["summary"].update(
        correct_questions=sum(correct.values()),
        accuracy=sum(correct.values()) / len(answers),
        correct_questions_per_second=sum(correct.values()) / report["summary"]["wall_seconds"],
        reserved_gpu_seconds=report["summary"]["gpu_seconds"],
    )


def amortized_cost(report: dict, questions: list[int], *, gpu_count: int) -> list[dict]:
    """Show loading/startup separately, without inventing historic index build cost."""
    service = report["summary"]["gpu_seconds"] / report["summary"]["requests"]
    setup = gpu_count * (report["startup_seconds"] + report["reader_preparation_seconds"])
    warmup = gpu_count * report["warmup"]["seconds"]
    teardown = gpu_count * report["teardown_seconds"]
    protocol = gpu_count * report.get("protocol_overhead_seconds", 0)
    return [
        {
            "amortization_questions": n,
            "physical_gpu_seconds_per_question": service
            + (setup + warmup + teardown + protocol) / n,
            "startup_and_index_load_gpu_seconds_per_question": setup / n,
            "warmup_gpu_seconds_per_question": warmup / n,
            "teardown_gpu_seconds_per_question": teardown / n,
            "protocol_overhead_gpu_seconds_per_question": protocol / n,
            "index_construction_gpu_seconds_per_question": None,
        }
        for n in questions
    ]
