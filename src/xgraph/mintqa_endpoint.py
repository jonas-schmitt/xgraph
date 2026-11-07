"""Development-only terminal-order controls and the declared WT-PC policy.

No model is loaded here. The generated queue is launched separately by the GPU owner.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shlex
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from xgraph.mintqa_panels import indexed, read_jsonl, write_json, write_jsonl
from xgraph.programs import record_path

FAMILIES = ("qwen38", "gemma31")


def cardinality_bin(count):
    return "0" if count == 0 else "1" if count == 1 else "2-8" if count <= 8 else "9+"


def endpoint_members(members, cardinalities, *, size=160, seed=17):
    """SRS relation strata within cardinality bins, then one SRS member per stratum.

    Every question has positive inclusion probability; no model outcome is an input.
    Survey weights divide the frozen panel weight by this second-stage probability.
    """
    groups = defaultdict(lambda: defaultdict(list))
    for member in sorted(members, key=lambda m: m["id"]):
        if not math.isfinite(member["weight"]) or member["weight"] <= 0:
            raise ValueError("endpoint panel weights must be finite and positive")
        count = cardinalities[member["id"]]
        if type(count) is not int or count < 0:
            raise ValueError("endpoint cardinalities must be nonnegative integers")
        groups[cardinality_bin(count)][member["cluster"]].append(member)
    if not len(groups) <= size <= sum(len(g) for g in groups.values()):
        raise ValueError("endpoint panel size must fit the nonempty relation/cardinality strata")
    quota = dict.fromkeys(sorted(groups), 1)
    mass = {b: sum(len(s) for s in g.values()) for b, g in groups.items()}
    while sum(quota.values()) < size:
        b = max(
            (b for b in quota if quota[b] < len(groups[b])),
            key=lambda b: (mass[b] / (quota[b] + 1), b),
        )
        quota[b] += 1
    rng, selected = random.Random(seed), []
    for b in sorted(groups):
        for cluster in rng.sample(sorted(groups[b]), quota[b]):
            candidates = groups[b][cluster]
            member = rng.choice(candidates)
            probability = quota[b] / len(groups[b]) / len(candidates)
            selected.append(
                {
                    **member,
                    "panel_weight": member["weight"],
                    "weight": member["weight"] / probability,
                    "endpoint_cardinality": cardinalities[member["id"]],
                    "endpoint_bin": b,
                    "endpoint_inclusion_probability": probability,
                }
            )
    return sorted(selected, key=lambda m: m["id"])


def endpoint_choice(endpoints, choices, *, prior=None):
    """Mean categorical choices; optional log prior cancellation with add-one smoothing.

    One constrained sample per permutation is an unbiased Monte Carlo draw from its
    endpoint distribution. Smoothing is fixed before outputs; ties use entity id.
    """
    support = sorted(set(endpoints))
    if not support or not choices or any(e not in support for e in choices):
        raise ValueError("endpoint choices must name nonempty canonical support")
    counts = Counter(choices)
    scores = {e: (counts[e] + 1) / (len(choices) + len(support)) for e in support}
    if prior is not None:
        if len(prior) != len(choices) or any(e not in support for e in prior):
            raise ValueError("endpoint prior must pair every view within the same support")
        counts0 = Counter(prior)
        scores = {
            e: math.log(scores[e]) - math.log((counts0[e] + 1) / (len(prior) + len(support)))
            for e in support
        }
    return max(support, key=lambda e: (scores[e], -support.index(e)))


def endpoint_request(request, revision, *, prior=False, seed=17):
    """Replay the same revised path; the prior removes all question-bearing history."""
    parsed = record_path(revision)
    if parsed is None:
        raise ValueError(f"{request['id']}: no complete revised relation path")
    start, relations = parsed
    if getattr(relations, "selections", {}) or getattr(relations, "audit", None):
        raise ValueError("WT-PC needs the plain revised path, without selections or edits")
    path = "⟦ " + " · ".join([start, *relations]) + "⟧"
    result = copy.deepcopy(request)
    messages = (
        [
            *copy.deepcopy(request["messages"][:-3]),
            {
                "role": "user",
                "content": "Continue with one bound Answer span after the KG display.",
            },
        ]
        if prior
        else copy.deepcopy(request["messages"])
    )
    result.update(
        messages=[*messages, {"role": "assistant", "content": path}], prefill_replay=True, seed=seed
    )
    return result


def endpoint_policy_record(wt, samples, endpoints, *, prior=None, policy="WT-PC"):
    from xgraph.mintqa_rematch import combined_usage

    consumed = [*samples, *(prior or [])]
    result = {
        **wt,
        "policy": policy,
        "usage": combined_usage(wt, *consumed),
        "endpoint_pc": {
            "K": len(samples),
            "prior": prior is not None,
            "estimator": "categorical Monte Carlo; add-one prior smoothing",
        },
        "cost_sources": [
            "cached WT first call",
            "cached WT revision",
            *[r["id"] for r in consumed],
        ],
    }
    if not samples or any(r.get("endpoint_entity_id") is None for r in consumed):
        result["endpoint_pc"]["fallback"] = "unresolved or truncated bound sample"
        return result
    entity = endpoint_choice(
        endpoints,
        [r["endpoint_entity_id"] for r in samples],
        prior=None if prior is None else [r["endpoint_entity_id"] for r in prior],
    )
    path = endpoint_request({"id": wt["id"], "messages": []}, wt)["messages"][-1]["content"]
    result.update(text=path + f".\nAnswer: ⟦ {endpoints[entity]}⟧", answer_entity_id=entity)
    # A deterministic policy answer has no model-generated token stream of its own.
    result.pop("token_ids", None)
    result.pop("opener", None)
    result.pop("prefill", None)
    return result


def paired_summary(members, left, right, *, seed=17):
    """Paired population-weighted delta, relation-cluster bootstrap and LOSO deletion."""
    grouped = defaultdict(lambda: [0.0, 0.0])
    for m in members:
        delta = left[m["id"]] - right[m["id"]]
        grouped[m["cluster"]][0] += m["weight"] * delta
        grouped[m["cluster"]][1] += m["weight"]
    values = list(grouped.values())
    numerator, denominator = map(sum, zip(*values, strict=True))
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(2000):
        drawn = rng.choices(values, k=len(values))
        bootstrap.append(sum(v[0] for v in drawn) / sum(v[1] for v in drawn))
    bootstrap.sort()
    loso = [(numerator - n) / (denominator - d) for n, d in values if d < denominator]
    return {
        "delta": numerator / denominator,
        "cluster_interval": [bootstrap[49], bootstrap[1949]],
        "loso_min": min(loso) if loso else None,
    }


def endpoint_crossfit(members, scores, candidates, *, seed=17):
    """Select one shared configuration across families using other relation clusters only."""
    clusters = sorted({m["cluster"] for m in members})
    random.Random(seed).shuffle(clusters)
    if len(clusters) < 2:
        raise ValueError("endpoint cross-fitting needs at least two relation sequences")
    assignments = {c: i % 2 for i, c in enumerate(clusters)}
    result, predictions = {"folds": [], "families": {}}, {f: {} for f in scores}

    def select(train):
        total = sum(m["weight"] for m in train)
        return max(
            candidates,
            key=lambda arm: (
                sum(m["weight"] * scores[f][arm][m["id"]] for f in scores for m in train)
                / total
                / len(scores),
                -candidates.index(arm),
            ),
        )

    for fold in range(2):
        train = [m for m in members if assignments[m["cluster"]] != fold]
        test = [m for m in members if assignments[m["cluster"]] == fold]
        selected = select(train)
        result["folds"].append(
            {
                "train": [m["id"] for m in train],
                "test": [m["id"] for m in test],
                "selected": selected,
            }
        )
        for f in scores:
            predictions[f].update({m["id"]: scores[f][selected][m["id"]] for m in test})
    loo = {f: {} for f in scores}
    for cluster in clusters:
        chosen = select([m for m in members if m["cluster"] != cluster])
        for f in scores:
            loo[f].update(
                {m["id"]: scores[f][chosen][m["id"]] for m in members if m["cluster"] == cluster}
            )
    for f in scores:
        result["families"][f] = {
            reference: paired_summary(members, predictions[f], scores[f][reference], seed=seed)
            for reference in ("WT", "agent")
        }
        result["families"][f]["leave_sequence_out_refit"] = {
            reference: paired_summary(members, loo[f], scores[f][reference], seed=seed)
            for reference in ("WT", "agent")
        }
        controls = {
            "pc2": "identity2",
            "pc4": "identity4",
            "pc2-prior": "identity4",
            "pc4-prior": "identity8",
        }
        if all(a in scores[f] for a in controls.values()):
            control = {}
            for fold in result["folds"]:
                reference = controls[fold["selected"]]
                control.update({q: scores[f][reference][q] for q in fold["test"]})
            result["families"][f]["matched_identity_control"] = paired_summary(
                members, predictions[f], control, seed=seed
            )
    return result


def endpoint_commands(checkout, out, specs):
    commands = []
    for family in FAMILIES:
        spec = specs[family]
        for arm in ("reversed", "random", "view0", "view1", "view2", "view3", "identity8"):
            view = int(arm[-1]) if arm.startswith("view") else None
            environment = {
                "XG_NEED_GIB": "60" if family == "qwen38" else "75",
                "XG_AVAILABLE_WAIT": "43200",
                "XG_KG_INDEX": spec["index"],
                "XG_TOKENIZER": spec["tokenizer"],
                "XG_ESCAPES": "0",
                "XG_PATHS": "1",
                "XG_COMMITTED_PATHS": "1",
                "XG_UNQUOTED_STARTS": "1",
                "XG_START_RANK": "sitelinks",
                "XG_OBSERVE_TERMINAL": "8",
                "XG_END_AFTER_ANSWER": "1",
                "XG_RELATIONS": "frontier",
                "XG_TERMINAL_ORDER": "random"
                if view is not None or arm == "random"
                else "reversed"
                if arm == "reversed"
                else "identity",
                "XG_TERMINAL_SEED": str(17 + (view or 0)),
            }
            requests = "second-requests" if arm in ("reversed", "random") else arm + "-requests"
            argv = [
                "bash",
                str(checkout / "scripts/generate.sh"),
                spec["model"],
                str(out / family / (requests + ".jsonl")),
                "constrained",
                str(out / family / (arm + "-raw.jsonl")),
                "--max-tokens",
                "256",
                "--max-model-len",
                "16384",
                "--max-num-seqs",
                "64",
                "--kv-cache-gib",
                "8",
                "--seed",
                "17",
            ]
            if arm in ("reversed", "random"):
                argv += ["--open"]
            else:
                argv += ["--sample", "--temperature", "1", "--top-p", "1"]
                if arm == "identity8":
                    argv += ["--samples", "8"]
            commands.append(
                {"family": family, "arm": arm, "environment": environment, "argv": argv}
            )
    return commands


def write_endpoint_queue(path, commands, checkout):
    """Write an inert-by-default, serial GPU-owner handoff with exact exit logging."""
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "mode=${1:---dry-run}",
        '[[ "$mode" == --dry-run || "$mode" == --run ]] || exit 2',
        "cd " + shlex.quote(str(checkout)),
        "log=/home/jonas/git/xgraph/target/queue/endpoint.log",
        "run() {",
        "  local fam=$1 arm=$2; shift 2",
        '  if [[ "$mode" == --dry-run ]]; then',
        "    printf '%q ' \"$@\"; printf '\\n'",
        "    return",
        "  fi",
        "  local status=0 variable; local -a clean=()",
        '  while IFS= read -r variable; do clean+=(-u "$variable"); done < <(',
        "    compgen -v XG_ || true; compgen -v XGRAPH_ || true)",
        '  env "${clean[@]}" "$@" || status=$?',
        '  mkdir -p "$(dirname "$log")"',
        '  printf "endpoint %s %s exit %s %s\\n" "$fam" "$arm" "$status" "$(date -Is)" >> "$log"',
        '  return "$status"',
        "}",
        # Queue-specific GPU checks run before generation; dry-run only prints them.
        "run both integrity uv run --no-sync python -m xgraph.mintqa_endpoint verify --out "
        + shlex.quote(str(path.parent)),
        "run both agreement uv run --no-sync pytest -q tests/test_engine.py "
        "tests/test_vllm_adapter.py -k terminal",
        "started=$SECONDS",
    ]
    for c in commands:
        env = [f"{k}={v}" for k, v in c["environment"].items()]
        lines += [
            'if [[ "$mode" == --run && $((SECONDS - started)) -ge 28800 ]]; then exit 124; fi',
            "run " + shlex.join([c["family"], c["arm"], "env", *env, *c["argv"]]),
        ]
    lines += [
        "run both scoring uv run --no-sync python -m xgraph.mintqa_endpoint score --out "
        + shlex.quote(str(path.parent))
    ]
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o755)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cached_directory(source, family):
    return source / (
        ".claude/worktrees/codex-integrate/target/mintqa-width8/terminal/"
        f"menuobs-width8/{family}/development"
    )


def load_executor(spec):
    from xgraph.index import load_kg_index
    from xgraph.programs import MatchedExecutor
    from xgraph.tokenizer import ServedTokenizer

    index = load_kg_index(spec["index"])
    tokenizer = ServedTokenizer.from_file(spec["tokenizer"])
    return MatchedExecutor(index, tokenizer, start_rank="sitelinks", unquoted_starts=True)


def prepare(source, out):
    """Consume the already frozen panel; write gold-free requests and a priced queue."""
    checkout = Path(__file__).resolve().parents[2]
    panel = json.loads((out / "panel.json").read_text())
    members, specs, inputs, costs = panel["members"], {}, [out / "panel.json"], {}
    ids = [m["id"] for m in members]
    if len(ids) != 160 or panel["seed"] != 17 or len(set(ids)) != len(ids):
        raise ValueError("endpoint development panel must be preselected with seed 17")
    for family in FAMILIES:
        directory = cached_directory(source, family)
        requests_path = directory / "menuobs-width8-revision-terminal-requests.jsonl"
        raw_path = directory / "menuobs-width8-revision-terminal-raw.jsonl"
        provenance_path = Path(str(raw_path) + ".provenance.json")
        original = json.loads(provenance_path.read_text())
        model = original["model"]["path"]
        specs[family] = {
            "model": model,
            "index": str(source / f"target/mintqa-rematch/indexes/{family}"),
            "tokenizer": model + "/tokenizer.json",
        }
        requests, revisions = indexed(read_jsonl(requests_path)), indexed(read_jsonl(raw_path))
        wt_path = directory / "menuobs-width8-two-pass-terminal.jsonl"
        wts = indexed(read_jsonl(wt_path))
        support_path = directory / "menuobs-width8-revision-terminal-support.jsonl"
        support = indexed(read_jsonl(support_path))
        target = out / family
        target.mkdir(parents=True, exist_ok=True)
        # Copy literal selected lines: formatting, key order, messages and metadata stay unchanged.
        lines = requests_path.read_text().splitlines(keepends=True)
        identity = "".join(line for line in lines if json.loads(line)["id"] in set(ids))
        (target / "second-requests.jsonl").write_text(identity)
        fixed, fallback = [], []
        for qid in ids:
            if (
                not wts[qid].get("revision_accepted")
                or not support[qid]["reached"]
                or record_path(revisions[qid]) is None
            ):
                fallback.append(qid)
                continue
            fixed.append(endpoint_request(requests[qid], revisions[qid], seed=17))
        write_json(target / "fallback.json", fallback)
        write_jsonl(target / "identity8-requests.jsonl", fixed)
        for view in range(4):
            batch = []
            for request in fixed:
                for prior in (False, True):
                    r = endpoint_request(
                        requests[request["id"]],
                        revisions[request["id"]],
                        prior=prior,
                        seed=17 + view,
                    )
                    r["id"] += "@prior" if prior else "@question"
                    batch.append(r)
            write_jsonl(target / f"view{view}-requests.jsonl", batch)
        summary_path = Path(str(raw_path) + ".summary.json")
        summary = json.loads(summary_path.read_text())
        # 2 full second calls + 4*(question+prior) + 8 identity draws per question.
        workload = summary["seconds"] / summary["requests"] * len(ids) * 18 / 3600
        costs[family] = {
            "historical_second_call_seconds": summary["seconds"],
            "historical_requests": summary["requests"],
            "scaled_workload_hours": workload,
            "startup_allowance_hours": 0.7,
            "priced_hours": 2 * workload + 0.7,
        }
        inputs += [
            requests_path,
            raw_path,
            provenance_path,
            summary_path,
            wt_path,
            support_path,
            directory / "menuobs-width8.jsonl",
            Path(specs[family]["tokenizer"]),
        ]
        inputs += sorted(p for p in Path(specs[family]["index"]).iterdir() if p.is_file())
        agent_path = (
            directory / "agent-strong-8.jsonl"
            if family == "qwen38"
            else source / "target/agent-qual/gemma31/development/agent-strong-8-feedback.jsonl"
        )
        specs[family]["agent"] = str(agent_path)
        inputs.append(agent_path)
    total = sum(c["priced_hours"] for c in costs.values()) + 0.5
    if total > 8:
        raise ValueError(
            f"endpoint price {total:.3f} GPU-hours exceeds eight; shrink from cost only"
        )
    commands = endpoint_commands(checkout, out, specs)
    write_endpoint_queue(out / "endpoint.sh", commands, checkout)
    files = [p for p in (checkout / "src/xgraph").glob("*.py")]
    files += [checkout / "csrc/engine.cu", checkout / "scripts/generate.sh"]
    inputs += files
    inputs += [p for family in FAMILIES for p in (out / family).glob("*-requests.jsonl")]
    write_json(
        out / "plan.json",
        {
            "policy": "WT-PC",
            "development_only": True,
            "specs": specs,
            "seed": 17,
            "K": [2, 4],
            "prior": [False, True],
            "identity_resamples": [2, 4, 8],
            "permutation_seeds": [17, 18, 19, 20],
            "estimator": "one temperature=1 top_p=1 constrained categorical draw per permutation; "
            "pool by canonical entity id; add-one smoothing; optional log mean prior subtraction",
            "estimator_reason": "one short continuation per view, independent of endpoint count; "
            "engine answer-span logprobs for all bound answers need extra scoring support",
            "cached_calls": "WT width-eight first call and revised path, each charged once",
            "control_matching": "K identity draws for K queries; 2K for K queries plus K priors",
            "prior_context": "remove target question; keep demos, replayed path, KG, display",
            "tie_rule": "canonical entity-id lexical order",
            "sample_fallback": "retain WT; charge every call",
            "cost_price": {
                "families": costs,
                "smoke_allowance_hours": 0.5,
                "total_gpu_hours": total,
                "limit_gpu_hours": 8,
                "scope": "forecast: twofold workload allowance and startup; not measured latency",
            },
            "commands": commands,
            "source_root": str(source),
            "checkout": str(checkout),
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "inputs": [{"path": str(p), "sha256": digest(p)} for p in dict.fromkeys(inputs)],
        },
    )
    return {"questions": len(ids), "generation_launches": len(commands), "priced_gpu_hours": total}


def verify(out):
    plan = json.loads((out / "plan.json").read_text())
    for item in plan["inputs"]:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError(f"endpoint frozen input/code changed: {item['path']}")
    if plan["cost_price"]["total_gpu_hours"] > 8:
        raise ValueError("endpoint queue exceeds its GPU-hour budget")
    return plan


def identity_audit(out):
    """Compare old/new exact terminal payloads on independently executed dev paths.

    The large immutable row tables are read lazily: this audit needs neither the
    oracle's full Python graph nor a vocabulary scan for every forced token.
    """
    import inspect
    import sys
    import types

    import numpy as np

    from xgraph.audit import terminal_pieces
    from xgraph.oracle import Oracle

    plan = json.loads((out / "plan.json").read_text())
    baseline = types.ModuleType("xgraph._endpoint_wt_oracle")
    sys.modules[baseline.__name__] = baseline
    old_source = subprocess.check_output(["git", "show", "7b45622:src/xgraph/oracle.py"], text=True)
    exec(compile(old_source, "7b45622:src/xgraph/oracle.py", "exec"), baseline.__dict__)
    defaults = inspect.signature(Oracle).parameters
    assert defaults["terminal_order"].default == "identity"
    assert defaults["terminal_seed"].default == 17
    report = {
        "baseline_commit": "7b45622",
        "families": {},
        "scope": "request bytes and terminal token payloads on independently re-executed paths; "
        "full mask/state agreement is the separate CUDA test layer",
    }
    for family, spec in plan["specs"].items():
        executor = load_executor(spec)
        index, tokenizer = executor._index, executor._tokenizer
        labels = index.labels
        owners, first = np.unique(labels.owners, return_index=True)
        first_row = {labels.owner_ids[int(o)]: int(r) for o, r in zip(owners, first, strict=True)}

        class Rows:
            def __init__(self, table):
                self.table = table

            def __getitem__(self, row):
                return (tuple(map(int, self.table.row_tokens(row))), self.table.owner_id(row))

        class Degrees:
            def __init__(self, table):
                self.table = table

            def get(self, entity, default=0):
                at = self.table._entity_at.get(entity)
                return (
                    default
                    if at is None
                    else int(
                        self.table.adjacency_offsets[at + 1] - self.table.adjacency_offsets[at]
                    )
                )

        fields = {
            "_first_row": first_row,
            "_entity_rows": Rows(labels),
            "_audit_sitelinks": dict(zip(labels.owner_ids, index.sitelinks, strict=True)),
            "_audit_pieces": terminal_pieces(tokenizer, index.span_syntax),
            "_graphs": {None: types.SimpleNamespace(degree=Degrees(index))},
            "_observe_terminal": 8,
            "_audit_entities": 20,
            "_audit_relations": 0,
        }
        oracles = []
        for cls in (baseline.Oracle, Oracle, Oracle):
            instance = cls.__new__(cls)
            instance.__dict__.update(fields)
            instance._terminal_order = defaults["terminal_order"].default
            instance._terminal_seed = defaults["terminal_seed"].default
            oracles.append(instance)
        directory = cached_directory(Path(plan["source_root"]), family)
        raw = read_jsonl(directory / "menuobs-width8-revision-terminal-raw.jsonl")
        requests_path = directory / "menuobs-width8-revision-terminal-requests.jsonl"
        requests = indexed(read_jsonl(requests_path))
        forced, displays, incomplete = hashlib.sha256(), 0, 0
        for row in raw:
            parsed = record_path(row)
            request = requests[row["id"]]
            reached = (
                executor.execute_set(*parsed, request.get("anchor"))["reached"] if parsed else []
            )
            incomplete += int(parsed is None)
            payloads = [()] * 3
            if len(reached) > 1:
                state = types.SimpleNamespace(
                    audit_frontier=tuple((e, 0) for e in reached), version=None
                )
                payloads = [o._audit_display(state, menus=False) for o in oracles]
            if not payloads[0] == payloads[1] == payloads[2]:
                raise ValueError(f"{row['id']}: identity terminal token payload changed")
            text = b"".join(tokenizer.token_bytes(t) for t in payloads[0]).decode()
            begin = row["text"].find("\nEndpoints: ")
            actual = "" if begin < 0 else row["text"][begin : row["text"].find("⟧", begin) + 1]
            if text != actual and not (
                row.get("finish_reason") == "length" and text.startswith(actual)
            ):
                raise ValueError(f"{row['id']}: independently executed display differs from WT")
            forced.update(text.encode())
            displays += int(bool(text))
        selected = {m["id"] for m in json.loads((out / "panel.json").read_text())["members"]}
        original_bytes = b"".join(
            line
            for line in requests_path.read_bytes().splitlines(keepends=True)
            if json.loads(line)["id"] in selected
        )
        if original_bytes != (out / family / "second-requests.jsonl").read_bytes():
            raise ValueError("identity development request bytes changed")
        report["families"][family] = {
            "requests": len(requests),
            "executed_records": len(raw),
            "displays": displays,
            "incomplete_or_unparsed": incomplete,
            "forced_bytes_sha256": forced.hexdigest(),
            "full_requests_sha256": digest(requests_path),
            "selected_request_bytes_sha256": hashlib.sha256(original_bytes).hexdigest(),
            "byte_identical": True,
        }
    write_json(out / "identity-audit.json", report)
    return report


def _sample_oracle(executor, *, order, seed):
    from xgraph.oracle import Oracle

    index, tok = executor._index, executor._tokenizer
    return Oracle(
        index,
        tok,
        index.span_syntax,
        paths=True,
        escapes=False,
        committed_paths=True,
        start_rank="sitelinks",
        unquoted_starts=True,
        observe_terminal=8,
        terminal_order=order,
        terminal_seed=seed,
        eos_token_ids=sorted(tok.special_ids),
    )


def annotate_endpoint_samples(records, request_map, executor, *, order, seed, oracle=None):
    """Resolve the actual engine answer, including alias ownership, from saved token ids."""
    from xgraph.audit import check_terminal_order
    from xgraph.oracle import TokenNotAllowed
    from xgraph.score import record_failure

    check_terminal_order(order, seed, 8)
    if not records:
        return {}
    # Only the display permutation differs between batches. Keep their states
    # separate while sharing the immutable reference tables built for this KG.
    oracle = (
        copy.copy(oracle)
        if oracle is not None
        else _sample_oracle(executor, order=order, seed=seed)
    )
    oracle._terminal_order, oracle._terminal_seed = order, seed
    tok = executor._tokenizer
    annotated = {}
    for row in records:
        base_id = row["id"].split("#", 1)[0]
        request = request_map[base_id]
        state = (
            oracle.anchored_state(request["anchor"])
            if request.get("anchor")
            else oracle.initial_state()
        )
        try:
            tokens = tok.encode(row.get("prefill") or row.get("opener") or "") + row["token_ids"]
            for token in tokens:
                state = oracle.advance(state, token)
        except TokenNotAllowed as exc:
            raise ValueError(f"{row['id']}: engine/oracle sample replay disagrees") from exc
        entity = None
        if not record_failure(row) and state.answer_complete and state.last_span:
            entities = state.last_span.entity_ids
            if len(entities) == 1:
                entity = entities[0]
        annotated[row["id"]] = {**row, "endpoint_entity_id": entity}
    return annotated


def score_endpoint(out):
    from xgraph.mintqa_bench import answer_contains
    from xgraph.mintqa_rematch import revision_records, support_records, usage
    from xgraph.prompts import final_answer
    from xgraph.score import record_failure

    plan = verify(out)
    source = Path(plan["source_root"])
    members = json.loads((out / "panel.json").read_text())["members"]
    ids = [m["id"] for m in members]
    selected = set(ids)
    gold = {}
    for split in ("pop", "ti"):
        for i, line in enumerate(
            (source / f"data/external/mintqa-full/questions/{split}.jsonl").open()
        ):
            qid = f"mintqa-{split}-{i}"
            if qid in selected:
                gold[qid] = json.loads(line)
    all_scores, arms, input_paths, diagnostics, stability = {}, {}, [], {}, {}
    for family in FAMILIES:
        target, directory = out / family, cached_directory(source, family)
        executor = load_executor(plan["specs"][family])
        labels = executor._index.labels
        supports = indexed(read_jsonl(directory / "menuobs-width8-revision-terminal-support.jsonl"))
        needed = {entity for q in ids for entity in supports[q]["reached"]}
        canonical = {}
        for row, owner in enumerate(labels.owners):
            entity = labels.owner_ids[int(owner)]
            if entity in needed and entity not in canonical:
                canonical[entity] = (
                    b"".join(executor._tokenizer.token_bytes(t) for t in labels.row_tokens(row))
                    .decode()
                    .removeprefix(" ")
                )
        first = indexed(read_jsonl(directory / "menuobs-width8.jsonl"))
        wt = indexed(read_jsonl(directory / "menuobs-width8-two-pass-terminal.jsonl"))
        requests = read_jsonl(target / "second-requests.jsonl")
        agent = indexed(read_jsonl(Path(plan["specs"][family]["agent"])))
        refs = {
            "WT": [wt[q] for q in ids],
            "agent": [agent[q] for q in ids],
        }
        for order in ("reversed", "random"):
            path = target / f"{order}-raw.jsonl"
            records = read_jsonl(path)
            if set(indexed(records)) != set(ids):
                raise ValueError(f"{family}/{order}: incomplete endpoint control")
            support = support_records(records, requests, executor)
            refs[order] = revision_records(
                [first[q] for q in ids],
                records,
                support,
                ids,
                first_source="cached WT first call",
                second_source=order,
            )
            input_paths.append(path)
        views = []
        sample_oracle = None
        for view in range(4):
            path = target / f"view{view}-raw.jsonl"
            asked = indexed(read_jsonl(target / f"view{view}-requests.jsonl"))
            records = read_jsonl(path)
            if set(indexed(records)) != set(asked):
                raise ValueError(f"{family}/view{view}: incomplete endpoint samples")
            if records and sample_oracle is None:
                sample_oracle = _sample_oracle(executor, order="random", seed=17 + view)
            views.append(
                annotate_endpoint_samples(
                    records, asked, executor, order="random", seed=17 + view, oracle=sample_oracle
                )
            )
            input_paths.append(path)
        identity_path = target / "identity8-raw.jsonl"
        identity_asked = indexed(read_jsonl(target / "identity8-requests.jsonl"))
        identity_raw = read_jsonl(identity_path)
        expected = {f"{q}#{j}" for q in identity_asked for j in range(8)}
        if set(indexed(identity_raw)) != expected:
            raise ValueError(f"{family}: incomplete identity resamples")
        if identity_raw and sample_oracle is None:
            sample_oracle = _sample_oracle(executor, order="identity", seed=17)
        identity = annotate_endpoint_samples(
            identity_raw, identity_asked, executor, order="identity", seed=17, oracle=sample_oracle
        )
        del sample_oracle
        input_paths.append(identity_path)

        def pair_disagreement(choices):
            pairs = [(a, b) for i, a in enumerate(choices) for b in choices[i + 1 :]]
            return sum(a != b for a, b in pairs) / len(pairs)

        comparable = [m for m in members if m["id"] in identity_asked]
        mass = sum(m["weight"] for m in comparable)
        stability[family] = {
            "question_view_disagreement": sum(
                m["weight"]
                * pair_disagreement(
                    [views[j][m["id"] + "@question"]["endpoint_entity_id"] for j in range(4)]
                )
                for m in comparable
            )
            / mass
            if mass
            else None,
            "identity_draw_disagreement": sum(
                m["weight"]
                * pair_disagreement(
                    [identity[f"{m['id']}#{j}"]["endpoint_entity_id"] for j in range(4)]
                )
                for m in comparable
            )
            / mass
            if mass
            else None,
            "scope": "categorical disagreement includes Monte Carlo noise; not causal robustness",
        }
        for k in (2, 4):
            for prior in (False, True):
                name = f"pc{k}" + ("-prior" if prior else "")
                records = []
                for q in ids:
                    if q not in identity_asked:
                        records.append(
                            {**wt[q], "endpoint_pc": {"fallback": "WT revision unusable"}}
                        )
                        continue
                    endpoints = {e: canonical[e] for e in set(supports[q]["reached"])}
                    samples = [views[j][q + "@question"] for j in range(k)]
                    priors = [views[j][q + "@prior"] for j in range(k)] if prior else None
                    records.append(endpoint_policy_record(wt[q], samples, endpoints, prior=priors))
                refs[name] = records
        for k in (2, 4, 8):
            refs[f"identity{k}"] = [
                endpoint_policy_record(
                    wt[q],
                    [identity[f"{q}#{j}"] for j in range(k)],
                    {e: canonical[e] for e in set(supports[q]["reached"])},
                    policy="WT identity resamples",
                )
                if q in identity_asked
                else wt[q]
                for q in ids
            ]
        own_scores, diagnostics[family] = {}, {}
        weights = {m["id"]: m["weight"] for m in members}
        for name, records in refs.items():
            rows = indexed(records)
            own_scores[name] = {
                q: int(
                    not record_failure(rows[q])
                    and answer_contains(final_answer(rows[q]["text"]), gold[q]["answer"])
                )
                for q in ids
            }
            exact, single = {}, {}
            for q in ids:
                answer = final_answer(rows[q]["text"])
                predicted = (
                    {rows[q]["answer_entity_id"]}
                    if rows[q].get("answer_entity_id")
                    else executor.named(answer or "")
                )
                expected_gold = executor.named(gold[q]["answer"])
                exact[q] = int(len(predicted) == 1 and bool(predicted & expected_gold))
                single[q] = int(len(predicted) == 1)
            mass = sum(weights.values())
            diagnostics[family][name] = {
                "accuracy": sum(weights[q] * own_scores[name][q] for q in ids) / mass,
                "exact_entity_accuracy": sum(weights[q] * exact[q] for q in ids) / mass,
                "single_entity_answer_share": sum(weights[q] * single[q] for q in ids) / mass,
                "usage": {
                    field: sum(usage(r)[field] for r in records) for field in usage(records[0])
                },
                "paired": {
                    reference: paired_summary(members, own_scores[name], own_scores[reference])
                    for reference in ("WT", "agent")
                    if reference in own_scores
                },
            }
            if name not in ("WT", "agent"):
                write_jsonl(target / f"{name}.jsonl", records)
        # Compute all paired comparisons after every fixed arm exists.
        for name in own_scores:
            diagnostics[family][name]["paired"] = {
                ref: paired_summary(members, own_scores[name], own_scores[ref])
                for ref in ("WT", "agent")
            }
        all_scores[family], arms[family] = own_scores, refs
    candidates = ["pc2", "pc2-prior", "pc4", "pc4-prior"]
    report = {
        "development_only": True,
        "diagnostics": diagnostics,
        "crossfit": endpoint_crossfit(members, all_scores, candidates),
        "without_sport": endpoint_crossfit(
            [m for m in members if "sport" not in m["cluster"].lower()], all_scores, candidates
        ),
        "equal_compute": {
            f: {
                a: paired_summary(members, all_scores[f][a], all_scores[f][control])
                for a, control in (
                    ("pc2", "identity2"),
                    ("pc4", "identity4"),
                    ("pc2-prior", "identity4"),
                    ("pc4-prior", "identity8"),
                )
            }
            for f in FAMILIES
        },
        "order_sensitivity": {
            f: {
                order: sum(
                    m["weight"]
                    * int(
                        final_answer(indexed(arms[f][order])[m["id"]]["text"])
                        != final_answer(indexed(arms[f]["WT"])[m["id"]]["text"])
                    )
                    for m in members
                )
                / sum(m["weight"] for m in members)
                for order in ("reversed", "random")
            }
            for f in FAMILIES
        },
        "sample_stability": stability,
        "inputs": [{"path": str(p), "sha256": digest(p)} for p in input_paths],
        "plan_sha256": digest(out / "plan.json"),
        "claim_boundary": "exposed development; cached revised paths; no serving efficiency claim",
    }
    write_json(out / "scores.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "identity", "score"))
    parser.add_argument("--out", type=Path, default=Path("target/queue-endpoint"))
    parser.add_argument("--source-root", type=Path, default=Path("/home/jonas/git/xgraph"))
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if args.command == "prepare":
        result = prepare(args.source_root.resolve(), out)
    elif args.command == "verify":
        plan = verify(out)
        result = {"verified_inputs": len(plan["inputs"])}
    elif args.command == "identity":
        result = identity_audit(out)
    else:
        result = score_endpoint(out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
