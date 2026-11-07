"""Frozen confirmation arms: real CPU stages between measured model calls.

This module reuses the public executor, STOP, repair and delivery algorithms. It
does not infer composite cost from the generation fields copied by a CPU merge.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path


def freeze_arms(
    args, ids: list[str], tokenizer: Path, header: dict, inputs: list[Path]
) -> tuple[dict, dict]:
    """Freeze named calibration recipes and audit the support before GPU admission.

    Relative filenames are relative to the CLI working directory. Missing source
    provenance is visible in CPU plans and forbidden with --run. A known mismatch
    is always an error, even during planning.
    """
    import numpy as np

    from xgraph.frontier import read_records
    from xgraph.generate import model_revision, walk_requests
    from xgraph.index import label_rows
    from xgraph.kg import Relation
    from xgraph.provenance import sha256_file
    from xgraph.serve_cost import calibration_settings, panel_records
    from xgraph.syntax import SpanSyntax
    from xgraph.tokenizer import ServedTokenizer
    from xgraph.wikidata import relation_names

    if not args.tools_db or not args.relation_labels:
        raise ValueError("confirmation arms need --tools-db and --relation-labels")
    if not header.get("sitelinks") or not (args.kg_index / "sitelinks.npy").is_file():
        raise ValueError("confirmation arms need the calibration index with sitelink counts")
    rows = [
        tuple(json.loads(line))
        for line in args.relation_labels.read_text().splitlines()
        if line.strip()
    ]
    names = relation_names(rows, unique_aliases=True)
    owners = header["relations"]["owner_ids"]
    served = ServedTokenizer.from_file(tokenizer)
    if served.fingerprint != header["tokenizer_fingerprint"]:
        raise ValueError("confirmation tokenizer differs from the KG index fingerprint")
    if set(owners) != set(names):
        raise ValueError("confirmation relation inventory differs from the calibration KG")
    syntax = header.get("span_syntax", {})
    legacy_syntax = "operand_separator" not in syntax and not header.get("expressions", False)
    expected = label_rows(
        tuple(Relation(p, tuple(names[p])) for p in owners),
        served,
        SpanSyntax(
            **({"operand_separator": "", "operators": (), **syntax} if legacy_syntax else syntax)
        ),
    )
    if set(owners) != set(names) or any(
        not np.array_equal(
            getattr(expected, key), np.load(args.kg_index / f"relations.{key}.npy", mmap_mode="r")
        )
        for key in ("tokens", "owners", "offsets")
    ):
        raise ValueError("confirmation relation index differs from the unique-alias policy")
    raw = json.loads(args.arm_config.read_text())
    if not isinstance(raw, dict) or not raw:
        raise ValueError("--arm-config needs a nonempty object of named arm recipes")
    inputs.extend([args.arm_config, args.relation_labels])
    arms, audits = {}, []

    def source(filename, *, complete=False):
        path = Path(filename).resolve()
        inputs.append(path)
        for suffix in (".summary.json", ".provenance.json"):
            if Path(str(path) + suffix).is_file():
                inputs.append(Path(str(path) + suffix))
        loaded = (
            panel_records(path, ids)
            if complete
            else [r for r in read_records(path) if r["id"] in ids]
        )
        return path, {r["id"]: r for r in loaded}

    def audit_source(
        path, profile, request_path, maximum, *, engine=False, native=None, options=None
    ):
        sidecar = Path(str(path) + ".provenance.json")
        audit = {
            "records": str(path),
            "profile": profile if engine else native or "free",
            "engine_opt_in": engine,
            "verified": sidecar.is_file(),
        }
        audits.append(audit)
        if not sidecar.is_file():
            if args.run:
                raise ValueError(f"{path}: confirmation execution needs generation provenance")
            return {}
        data = json.loads(sidecar.read_text())
        if not engine and native is None and data.get("arm") in ("constrained", "fallback"):
            raise ValueError(
                f"{path}: {data.get('relations', 'frontier')} planner serving accounting "
                "is unavailable for an unmasked replay; planner and delivery support "
                "need separate serving phases"
            )
        if data.get("model", {}).get("path") != str(args.model.resolve()):
            raise ValueError(f"{path}: model differs from calibration generation provenance")
        if data.get("model", {}).get("revision") != model_revision(args.model):
            raise ValueError(f"{path}: model revision differs from calibration")
        command = data.get("provenance", {}).get("command", [])
        if (
            "--max-tokens" not in command
            or int(command[command.index("--max-tokens") + 1]) != maximum
        ):
            raise ValueError(f"{path}: max_tokens differs from calibration generation provenance")
        if native and data.get("arm") != native:
            raise ValueError(f"{path}: native reader arm differs from calibration")
        if native == "gcr" and (data.get("gcr") or {}).get("synthesis_model", {}).get(
            "path"
        ) != str(args.model.resolve()):
            raise ValueError(
                f"{path}: GCR timing requires the same frozen planner and synthesis model"
            )
        options = options or {}
        if native in ("agent", "walk", "resume"):
            for key in ("start_rank", "unquoted_starts"):
                if data.get(key) != options[key]:
                    raise ValueError(f"{path}: matched reader {key} differs from calibration")
        expected_options = {}
        if native == "agent":
            expected_options = {
                "calls": options.get("max_calls", 8),
                "matched_demos": True,
                "matched_paths": True,
                "pages": True,
                **{k: options.get(k, False) for k in ("feedback", "mask", "force_path")},
                "schema": options.get("schema"),
            }
        if native == "iterative":
            from xgraph.rag import CAP

            expected_options = {
                "lookups": options.get("max_lookups", 8),
                "cap": options.get("cap", CAP),
                "written_start": options.get("written_start", True),
            }
        if native == "walk":
            expected_options = {
                "hops": options.get("max_calls", 8),
                "max_options": None,
                "observe": "labels",
                "matched_paths": True,
                "min_hops": options.get("min_hops", 0),
            }
        if native == "resume":
            expected_options = {
                "hops": options.get("max_calls", 8),
                "max": options.get("resume_max", 3),
            }
        if native in ("iterative", "resume"):
            limit = (
                int(command[command.index("--max-model-len") + 1])
                if "--max-model-len" in command
                else 8192
            )
            if options.get("max_prompt_tokens") != limit - maximum:
                raise ValueError(f"{path}: context clipping differs from calibration")
        for key, value in expected_options.items():
            actual = (data.get(native) or {}).get(
                key, False if isinstance(value, bool) else 0 if key == "min_hops" else None
            )
            if actual != value:
                raise ValueError(
                    f"{path}: calibration {native}.{key} differs: {actual!r} versus {value!r}"
                )
        if engine:
            for key, value in calibration_settings(profile)["engine_settings"].items():
                key = key.removeprefix("XGRAPH_").lower()
                expected_value = (value == "1") if value in ("0", "1") else (value or None)
                if data.get(key) != expected_value:
                    raise ValueError(
                        f"{path}: calibration support {key}={data.get(key)!r}, "
                        f"expected {expected_value!r}"
                    )
            if data.get("open") != calibration_settings(profile)["opened"]:
                raise ValueError(f"{path}: calibration opener differs")
        frozen = {r.get("sha256") for r in data.get("provenance", {}).get("inputs", [])}
        must_match = [request_path]
        if native:
            store = data.get("tools_db")
            store_inputs = [
                r
                for r in data.get("provenance", {}).get("inputs", [])
                if store and Path(r["path"]).resolve() == Path(store).resolve()
            ]
            if store_inputs:
                must_match.append(args.tools_db)
            elif store and Path(store).resolve() == args.tools_db.resolve():
                audit["implicit_store_path_only"] = True
            else:
                raise ValueError(f"{path}: KG store differs from calibration provenance")
            if (data.get("agent") or {}).get("relation_labels"):
                must_match.append(args.relation_labels)
        if engine or "--tokenizer" in command:
            must_match.append(tokenizer)
        else:
            audit["implicit_tokenizer_revision_only"] = True
        if engine or native in ("agent", "walk", "resume"):
            must_match.append(args.kg_index / "kg-index.json")
        if args.kg_versions and (engine or native):
            must_match.extend(p for p in args.kg_versions.rglob("*") if p.is_file())
        for file in must_match:
            if sha256_file(file) not in frozen:
                raise ValueError(f"{path}: calibration input hash differs or is absent: {file}")
        banned = []
        if "--ban-tokens" in command:
            for token in command[command.index("--ban-tokens") + 1 :]:
                if token.startswith("--"):
                    break
                encoded = served.encode(token)
                if len(encoded) != 1:
                    raise ValueError(f"{path}: banned token does not map to one served token")
                banned.append(encoded[0])
        return {"logit_bias": {str(t): -100.0 for t in banned}} if banned else {}

    for name, spec in raw.items():
        if args.arm and name not in args.arm:
            continue
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"invalid confirmation arm name {name!r}")
        if not isinstance(spec, dict) or set(spec) - {
            "kind",
            "requests",
            "records",
            "options",
            "max_tokens",
            "delivery",
            "delivery_requests",
            "delivery_generation_requests",
            "verifier_requests",
            "delivery_records",
            "verifier_records",
            "final_records",
        }:
            raise ValueError(f"{name}: invalid confirmation arm recipe")
        kind = spec.get("kind")
        if kind == "gwalk":
            raise ValueError(
                "GWalk is ineligible: existing records omit replayable raw prompts/replies "
                "and shared-case accounting"
            )
        if kind not in (
            "paths",
            "repair",
            "execute",
            "agent",
            "iterative",
            "walk",
            "resume",
            "gcr",
        ):
            raise ValueError(f"{name}: unknown confirmation arm kind {kind!r}")
        if kind == "execute" and any(
            spec.get(key) for key in ("verifier_records", "verifier_requests")
        ):
            raise ValueError(
                f"{name}: symmetric X repair serving accounting is unavailable; "
                "execute recipes cannot time verification and a second delivery"
            )
        if not isinstance(spec.get("records"), str) or not spec["records"]:
            raise ValueError(f"{name}: records must name the frozen generation file")
        if not isinstance(spec.get("options", {}), dict):
            raise ValueError(f"{name}: options must be an object")
        options = {"start_rank": "sitelinks", "unquoted_starts": True, **spec.get("options", {})}
        allowed = {"start_rank", "unquoted_starts"} | {
            "paths": {"stop"},
            "repair": {"stop", "matched_paths"},
            "execute": {"planner_opened", "cpu_stop", "repair", "strip_quotes", "matched_paths"},
            "agent": {
                "max_calls",
                "feedback",
                "mask",
                "force_path",
                "schema",
                "matched_demos",
                "matched_paths",
                "pages",
                "relations",
            },
            "iterative": {
                "max_lookups",
                "cap",
                "max_prompt_tokens",
                "written_start",
                "start_max_tokens",
            },
            "walk": {"max_calls", "min_hops", "matched_paths"},
            "resume": {"max_calls", "resume_max", "max_prompt_tokens", "matched_paths"},
            "gcr": set(),
        }[kind]
        if set(options) - allowed:
            raise ValueError(f"{name}: unknown options {sorted(set(options) - allowed)}")
        for key in (
            "unquoted_starts",
            "stop",
            "planner_opened",
            "cpu_stop",
            "repair",
            "strip_quotes",
            "feedback",
            "mask",
            "force_path",
            "written_start",
            "matched_demos",
            "matched_paths",
            "pages",
        ):
            if key in options and type(options[key]) is not bool:
                raise ValueError(f"{name}: {key} must be boolean")
        for key in (
            "max_calls",
            "min_hops",
            "resume_max",
            "max_lookups",
            "cap",
            "start_max_tokens",
            "max_prompt_tokens",
        ):
            if key in options and (
                type(options[key]) is not int
                or options[key] < (0 if key in ("min_hops", "resume_max") else 1)
            ):
                raise ValueError(f"{name}: invalid {key}")
        if options["start_rank"] != "sitelinks" or options["unquoted_starts"] is not True:
            raise ValueError(f"{name}: confirmation needs calibration's start fixes")
        maximum = spec.get("max_tokens", 512 if kind == "iterative" else args.max_tokens)
        options["relation_labels"] = str(args.relation_labels.resolve())
        if kind in ("agent", "walk", "resume", "execute", "repair"):
            options.setdefault("matched_paths", True)
            if options["matched_paths"] is not True:
                raise ValueError(f"{name}: confirmation requires matched paths")
        if kind == "agent":
            for flag in ("matched_demos", "pages"):
                options.setdefault(flag, True)
                if options[flag] is not True:
                    raise ValueError(f"{name}: calibration requires {flag}")
            options.setdefault("max_calls", 8)
            if options.get("schema") not in (None, "static", "turn") or (
                options.get("schema") and (options.get("mask") or options.get("force_path"))
            ):
                raise ValueError(f"{name}: invalid native schema/mask/forced-path combination")
        if kind in ("walk", "resume"):
            options.setdefault("max_calls", 8)
        if kind == "execute":
            options.setdefault("strip_quotes", True)
            options.setdefault("cpu_stop", True)
        if kind == "iterative":
            options.setdefault("max_lookups", 8)
            options.setdefault("written_start", True)
        if kind in ("iterative", "resume"):
            options.setdefault("max_prompt_tokens", args.max_model_len - maximum)
        profile = (
            ("paths" if options.get("stop", True) else "paths-no-stop")
            if kind in ("paths", "repair")
            else ("masked-agent" if kind == "agent" else "delivery")
        )
        if kind == "agent" and options.get("relations", "frontier") != "frontier":
            profile = "masked-agent-" + options["relations"]
        support = calibration_settings(profile)
        request_path = Path(spec.get("requests", args.paths_requests)).resolve()
        asked = panel_records(request_path, ids)
        if any(r.get("kg_version") for r in asked) and not args.kg_versions:
            raise ValueError(f"{name}: versioned calibration requests need --kg-versions")
        inputs.append(request_path)
        path, records = source(spec["records"], complete=True)
        main_sampling = audit_source(
            path,
            profile,
            request_path,
            maximum,
            engine=kind in ("paths", "repair") or bool(options.get("mask")),
            native=kind if kind in ("agent", "iterative", "walk", "resume", "gcr") else None,
            options=options,
        )
        extras = {}
        for key in ("delivery_records", "verifier_records", "final_records"):
            if spec.get(key):
                extra_path, extras[key] = source(spec[key], complete=key == "final_records")
                if key != "final_records":
                    # Delivery request hashes are checked by runtime reconstruction;
                    # the source support and tokenizer/version hashes are checked here.
                    req = Path(
                        spec.get(
                            "delivery_generation_requests",
                            str(extra_path).replace("delivered.jsonl", "deliver.jsonl"),
                        )
                    ).resolve()
                    if key == "verifier_records":
                        req = Path(
                            spec.get("verifier_requests", extra_path.parent / "requests.jsonl")
                        ).resolve()
                    extra_sampling = audit_source(
                        extra_path,
                        "delivery" if key == "delivery_records" else "verifier",
                        req,
                        8 if key == "verifier_records" else maximum,
                        engine=key == "delivery_records",
                    )
                    if extra_sampling != main_sampling:
                        raise ValueError(
                            f"{name}: stages use different sampling/banned-token settings"
                        )
                    if req.is_file():
                        (
                            _,
                            extras[
                                "delivery_trace_requests"
                                if key == "delivery_records"
                                else "verifier_trace_requests"
                            ],
                        ) = source(req)
        base = panel_records(Path(spec.get("delivery_requests", args.paths_requests)), ids)
        inputs.append(Path(spec.get("delivery_requests", args.paths_requests)))
        if kind in ("agent", "walk", "gcr"):
            converted = walk_requests(asked)
            asked = [{**a, **b} for a, b in zip(asked, converted, strict=True)]
        packed = []
        for a, b in zip(asked, base, strict=True):
            qid = a["id"]
            record = records[qid]
            if a.get("kg_version") != b.get("kg_version"):
                raise ValueError(
                    f"{name} {qid}: planner and delivery requests select different KG versions"
                )
            if kind == "execute" and options.get("planner_opened"):
                opener = SpanSyntax(**header.get("span_syntax", {})).opener
                if not record["text"].startswith(opener):
                    record = {**record, "text": opener + record["text"]}
            if record.get("kg_version") is not None and record["kg_version"] != a.get("kg_version"):
                raise ValueError(f"{name} {qid}: record and request KG versions differ")
            if kind in ("agent", "iterative", "walk", "resume", "gcr"):
                turns = record.get(
                    "turns", record.get("replies") if kind in ("walk", "iterative") else None
                )
                if not isinstance(turns, list) or record.get("prompt_tokens") is None:
                    raise ValueError(
                        f"{name} {qid}: complete raw turns/replies and prompt usage are required"
                    )
                if (
                    kind == "iterative"
                    and options.get("max_prompt_tokens")
                    and "turns" not in record
                ):
                    raise ValueError(f"{name} {qid}: iterative clipping requires per-turn usage")
                if kind == "gcr" and (
                    record.get("decoding") != "beam"
                    or any(
                        t.get("kind") == "beam_step"
                        and (not t.get("token_logprobs") or not t.get("prefix_sha256"))
                        for t in turns
                    )
                ):
                    raise ValueError(f"{name} {qid}: GCR beam needs its complete raw scoring trace")
                if kind == "gcr" and record["config"]["max_tokens"] != maximum:
                    raise ValueError(f"{name} {qid}: GCR call token budget differs from its trace")
            question = a.get("question") or next(
                m["content"] for m in reversed(a["messages"]) if m["role"] == "user"
            )
            packed.append(
                {
                    **a,
                    "question": question,
                    "record": record,
                    "sampling": main_sampling,
                    "delivery_request": {**b, "question": question},
                    **{
                        key.removesuffix("s"): values[qid]
                        for key, values in extras.items()
                        if qid in values
                    },
                }
            )
        delivery = spec.get(
            "delivery", "engine" if kind in ("paths", "execute", "repair") else "reader"
        )
        if delivery not in ("engine", "executor", "reader"):
            raise ValueError(f"{name}: unknown delivery {delivery!r}")
        if kind in ("agent", "iterative", "gcr") and delivery != "reader":
            raise ValueError(f"{name}: reader must retain its own final answer")
        arms[name] = {
            "kind": kind,
            "confirmation": True,
            "requests": packed,
            "max_tokens": maximum,
            "options": options,
            "delivery": delivery,
            "opt_in": kind in ("paths", "repair")
            or delivery == "engine"
            or bool(options.get("mask")),
            "opener": SpanSyntax(**header.get("span_syntax", {})).opener
            if support["opened"]
            else None,
            "engine_settings": support["engine_settings"],
            "support_profile": profile,
            "expected_sampling": main_sampling,
            "tools_db": str(args.tools_db.resolve()),
            "kg_index": str(args.kg_index.resolve()),
            "tokenizer": str(tokenizer.resolve()),
            "model": str(args.model.resolve()),
            "kg_versions": str(args.kg_versions.resolve()) if args.kg_versions else None,
        }
    return arms, {
        "unique_alias_index": True,
        "legacy_span_syntax": legacy_syntax,
        "sitelinks": True,
        "generation_sources": audits,
        "verified_generation_sources": all(a["verified"] for a in audits),
    }


def replay_arm(url: str, request: dict, arm: dict, *, timeout=600, context=None) -> dict:
    """Time a complete question; frozen replies determine its policy and delivery."""
    from xgraph.programs import parse_path, record_path, with_answer
    from xgraph.prompts import final_answer
    from xgraph.repair import (
        delivery_requests,
        executed_delivery_requests,
        merged_executed,
        merged_records,
        select,
        verifier_request,
    )
    from xgraph.serve_cost import _TimedStore, prepare_reader, replay_reader
    from xgraph.serving import _one
    from xgraph.stopping import rewrite_record

    started = time.perf_counter()
    calls, stages = [], []
    attempts = 0
    owns_context = context is None
    if owns_context:
        context = prepare_reader({**arm, "requests": [request]})
    if context.get("sql") is not None:
        context["sql"].statements = 0
    read = _TimedStore(context["store"])
    executor = context.get("executor_timing")
    if executor is not None:
        executor.calls, executor.seconds = 0, 0.0
    elif context["executor"] is not None:
        executor = _TimedStore(context["executor"])
    options, qid, kind = arm.get("options", {}), request["id"], arm["kind"]
    original = request.get("delivery_request") or request
    asked = {
        k: v
        for k, v in original.items()
        if k
        not in ("record", "delivery_record", "verifier_record", "final_record", "delivery_request")
    }

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

    def generate(name, messages, saved, opt_in=False, opener=None, stop=False, max_tokens=None):
        nonlocal attempts
        if saved is None:
            raise ValueError(f"{qid}: missing frozen {name} generation record")
        attempts += 1
        extra = None
        if opt_in:
            extra = {"xgraph_question": asked["question"]} if stop else {"xgraph_stop": 0}
        call_started = time.perf_counter() - started
        measured = stage(
            name,
            lambda: _one(
                url,
                "/model",
                {
                    **messages,
                    "seed": messages.get("seed", request.get("seed", 0)),
                    "sampling": request.get("sampling", {}),
                },
                max_tokens or arm["max_tokens"],
                opt_in,
                opener,
                timeout=timeout,
                extra_args=extra,
            ),
        )
        calls.append({**measured, "stage": name, "started_seconds": call_started})
        if saved.get("prompt_tokens") != measured["prompt_tokens"]:
            raise ValueError(f"{qid}: reconstructed {name} prompt token count differs from trace")
        return saved

    try:
        if kind in ("paths", "repair", "execute"):
            opt_in = kind != "execute"
            opened = None
            if opt_in or options.get("planner_opened"):
                opened = arm.get("opener") or "⟦"
            if request["messages"][-1]["role"] == "assistant":
                opened = None
            row = generate(
                "planner",
                request,
                request["record"],
                opt_in,
                opened,
                stop=opt_in and options.get("stop", True),
            )
        else:
            measured = stage(
                "reader", lambda: replay_reader(url, request, arm, timeout=timeout, context=context)
            )
            calls.extend(measured["calls"])
            attempts += measured["model_calls"]
            row = measured.get("executed_record", {"id": qid, "text": measured["text"]})

        if kind == "execute":

            def execute():
                path = record_path(row)
                found = (
                    executor.execute(
                        *path, version=request.get("kg_version"), repair=options.get("repair", True)
                    )
                    if path
                    else None
                )
                label = read.label(int(found.answer[1:])) if found and found.answer else None
                if found and found.answer and label is None:
                    raise ValueError(f"{qid}: store and index disagree on the delivered label")
                return {
                    **row,
                    "text": with_answer(row["text"], label),
                    "plan": found.recorded_plan if found else None,
                    "reached": [int(q[1:]) for q in found.reached[:20]] if found else [],
                }

            row = stage("execute", execute)
            if options.get("cpu_stop", True):
                row = stage(
                    "cpu_stop",
                    lambda: rewrite_record(
                        row,
                        asked["question"],
                        executor,
                        read,
                        version=request.get("kg_version"),
                        terms=context.get("stop_terms"),
                    ),
                )

        candidates = selected = None
        if kind == "repair":

            def build_candidates():
                path = parse_path(row["text"])
                found, empty = (
                    context["repairer"].candidates(
                        asked["question"], *path, version=request.get("kg_version")
                    )
                    if path
                    else ([], 0)
                )
                return {
                    "id": qid,
                    "question": asked["question"],
                    "start": path[0] if path else None,
                    "kg_version": request.get("kg_version"),
                    "text": row["text"],
                    "answer": final_answer(row["text"]),
                    "candidates": found,
                    "empty": empty,
                }

            candidates = stage("repair_candidates", build_candidates)
            verify = stage("verifier_request", lambda: verifier_request(candidates, seed=0))
            if (
                request.get("verifier_trace_request")
                and verify != request["verifier_trace_request"]
            ):
                raise ValueError(f"{qid}: reconstructed verifier request differs from calibration")
            verdicts = {}
            if verify is not None:
                saved = generate("verifier", verify, request.get("verifier_record"), max_tokens=8)
                verdicts[qid] = saved["text"]
            elif request.get("verifier_record") is not None:
                raise ValueError(f"{qid}: frozen verifier was not required by reconstructed repair")
            (selected,) = stage(
                "repair_select",
                lambda: select([candidates], {qid: verify} if verify else {}, verdicts),
            )
            row = selected

        if arm.get("delivery", "engine") == "engine" and kind != "paths":
            delivered = stage(
                "delivery_requests",
                lambda: (
                    delivery_requests([candidates], [selected], {qid: asked})
                    if kind == "repair"
                    else executed_delivery_requests([row], {qid: asked})
                ),
            )
            generated = {}
            if delivered:
                saved_request = request.get("delivery_trace_request")
                if saved_request is not None and any(
                    delivered[0].get(k) != saved_request.get(k)
                    for k in ("messages", "anchor", "kg_version", "prefill_replay")
                ):
                    raise ValueError(
                        f"{qid}: reconstructed delivery request differs from calibration"
                    )
                generated[qid] = generate(
                    "delivery", delivered[0], request.get("delivery_record"), opt_in=True
                )
            elif request.get("delivery_record") is not None:
                raise ValueError(f"{qid}: frozen delivery was not required by reconstructed plan")
            (row,) = stage(
                "merge",
                lambda: (
                    merged_records([selected], {r["id"]: r for r in delivered}, generated)
                    if kind == "repair"
                    else merged_executed([row], {r["id"]: r for r in delivered}, generated)
                ),
            )
        expected = request.get("final_record")
        if expected is not None and row["text"] != expected["text"]:
            raise ValueError(f"{qid}: reconstructed composite final delivery differs from trace")
        return {
            "id": qid,
            "text": row["text"],
            "trajectory_kind": "frozen_composite_replay",
            "calls": calls,
            "stages": stages,
            "model_calls": attempts,
            "tool_calls": measured.get("tool_calls", 0)
            if kind not in ("paths", "execute", "repair")
            else 0,
            "kg_store_calls": read.calls,
            "kg_store_seconds": read.seconds,
            "kg_sql_statements": context["sql"].statements if context.get("sql") else None,
            "executor_calls": executor.calls if executor else 0,
            "executor_seconds": executor.seconds if executor else 0,
            "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
            "completion_tokens": sum(c["completion_tokens"] for c in calls),
            "ttft": calls[0]["ttft"] + calls[0]["started_seconds"]
            if calls and calls[0]["ttft"] is not None
            else None,
            "finish_reason": row.get("finish_reason"),
            "truncated": any(
                c["finish_reason"] == "length" and c.get("kind") != "beam_step" for c in calls
            ),
            "retries": measured.get("retries", 0)
            if kind not in ("paths", "execute", "repair")
            else 0,
            "latency": time.perf_counter() - started,
        }
    except Exception as error:
        error.result = {
            "calls": calls,
            "stages": stages,
            "model_calls": attempts,
            "prompt_tokens": None,
            "completion_tokens": None,
        }
        raise
    finally:
        if owns_context:
            context["store"]._con.close()
            if context.get("scoring_pool") is not None:
                context["scoring_pool"].shutdown(wait=True)
