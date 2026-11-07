"""CPU-only hypothesis-relative exposure audit and initial-case study preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

from xgraph.mintqa_panels import read_jsonl, write_json, write_jsonl
from xgraph.mintqa_queue import REPO, _code_digest
from xgraph.provenance import provenance, sha256_file

FAMILIES = ("qwen38", "gemma31")
NORMAL = NormalDist()
OWNER_RULINGS = [
    (
        "1. Freshness is hypothesis-relative. Exposed = any item whose outcomes"
        " under the current models (Qwen3.8/Gemma31 families) or any\n   WT/menu"
        "obs/agent policy were produced or inspected: dev, analysis-dev, sealed"
        " validation, every pilot/calibration/budget/endpoint\n   run, and anyth"
        "ing from which this alias hypothesis was derived. The early full-popul"
        "ation Qwen2.5 feasibility runs\n   (target/mintqa/qwen25-*-all.jsonl) a"
        "re disclosed as prior exposure of a different model and method; they d"
        "id not inform this\n   hypothesis and do not make items unusable. Prove"
        " disjointness of the fresh pool from every exposed ID list you can enu"
        "merate\n   (search all worktrees' target/ dirs for ID-bearing files; li"
        "st what was scanned)."
    ),
    (
        "2. Power must model WT vs WT+alias, not WT vs agent: discordance is co"
        "nfined to items whose question surfaces reach an alias-only\n   branch."
        " Estimate the affected fraction gold-free on CPU (question-surface -> "
        "alias-only prefix reachability under each served\n   tokenizer, fresh p"
        "ool, by stratum), take a conservative per-affected-item flip-rate prio"
        "r from the exposed validation all-new items\n   (disclosed), and size: "
        "(a) all-new superiority, both families, alpha 0.025 one-sided, interse"
        "ction; (b) mixed-sample NI at -1 pp.\n   Seed 17, cluster unit relation"
        "-sequence-within-hop as registered."
    ),
    (
        "3. Replace the blocked confirmatory design with this one if its GPU fo"
        "recast (sealed WT per-question costs, both arms, both families,\n   sta"
        "rtup/CUDA reserves) is <= 10 GPU-h; otherwise shrink to the largest de"
        "sign within 10 GPU-h and report its power. Keep the\n   exploratory mod"
        "e as is. Also measure the alias index's size delta (rows, bytes, GPU t"
        "able bytes) and check it fits the sealed\n   kv/memory envelope (40 GiB"
        " admission)."
    ),
]


def id_digest(ids):
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()


def exposure_proof(population, witnesses):
    """Set difference against the caller's audited, hypothesis-relative inventory."""
    population = set(population)
    if not population or not witnesses:
        raise ValueError("need a nonempty population and recorded exposure witnesses")
    exposed = population.intersection(set().union(*witnesses))
    fresh = population - exposed
    return {
        "population": len(population),
        "exposed": len(exposed),
        "fresh_ids": sorted(fresh),
        "population_ids_sha256": id_digest(population),
        "exposed_ids_sha256": id_digest(exposed),
        "fresh_ids_sha256": id_digest(fresh),
        "fresh_confirmation_possible": bool(fresh),
        "proof": "F = P minus union(E); every enumerated exposed ID list is disjoint from F",
        "scope": "hypothesis-relative enumerated exposure inventory; see scan manifest",
    }


def proxy_power(*, affected_fraction, gain_rate, harm_rate, design_effect, available, endpoint):
    """Sparse WT/WT+alias normal planning; never WT/agent discordance.

    Gains and harms are per affected question. NI plans at zero true advantage
    and keeps the full discordance variance. Four marginal .95 targets imply
    a joint .80 union-bound target, conditional on these planning assumptions.
    """
    if (
        not 0 <= affected_fraction <= 1
        or min(gain_rate, harm_rate) < 0
        or gain_rate + harm_rate > 1
        or design_effect < 1
        or available < 0
        or endpoint not in ("all_new", "mixed")
    ):
        raise ValueError("invalid paired power inputs")
    discordance = affected_fraction * (gain_rate + harm_rate)
    effect = affected_fraction * (gain_rate - harm_rate) if endpoint == "all_new" else 0.01
    variance = discordance * design_effect
    z = NORMAL.inv_cdf(0.975)
    required = (
        math.ceil((z + NORMAL.inv_cdf(0.95)) ** 2 * variance / effect**2)
        if variance > 0 and effect > 0
        else None
    )
    return {
        "assumption": "WT/WT+alias flips confined to gold-free alias-only prefix reachability",
        "affected_fraction": affected_fraction,
        "gain_rate_per_affected": gain_rate,
        "harm_rate_per_affected": harm_rate,
        "discordance": discordance,
        "design_effect": design_effect,
        "effect": effect,
        "alpha_one_sided": 0.025,
        "marginal_power_target": 0.95,
        "joint_target_union_bound": 0.80,
        "required_questions": required,
        "available_fresh": available,
        "power_at_available": NORMAL.cdf(effect * math.sqrt(available / variance) - z)
        if available and variance > 0 and effect > 0
        else 0,
        "formula": "ceil((z(.975)+z(.95))^2 * discordance * design_effect / effect^2)",
    }


def all_new(question):
    return re.fullmatch(r"ti [1-4]-hop 0/[1-4] old", question["complexity"]) is not None


def exploratory_members(questions, *, enriched=256, mixed=512, seed=17):
    """Gold-free stratified samples; overlap is retained and only run once."""
    rng = random.Random(seed)
    questions = sorted(questions, key=lambda r: r["id"])
    new = [r for r in questions if all_new(r)]
    selected = rng.sample(new, min(enriched, len(new)))
    panels = {"all_new": [{"id": q["id"], "weight": len(new) / len(selected)} for q in selected]}
    groups = defaultdict(list)
    for q in questions:
        groups[q["complexity"]].append(q)
    if mixed < len(groups) or mixed > len(questions):
        raise ValueError("mixed size must cover nonempty complexity strata")
    quota = dict.fromkeys(sorted(groups), 1)
    while sum(quota.values()) < mixed:
        key = max(
            (k for k in quota if quota[k] < len(groups[k])),
            key=lambda k: (len(groups[k]) / (quota[k] + 1), k),
        )
        quota[key] += 1
    panels["mixed"] = [
        {"id": r["id"], "weight": len(group) / quota[key]}
        for key, group in sorted(groups.items())
        for r in rng.sample(group, quota[key])
    ]
    return panels


def align_index(source, out):
    """Keep the sealed relation language and legacy header representation exactly.

    This only replaces relation label rows, after proving that the set of
    writable relations and their owner positions is unchanged. Adjacency stores
    relation owners, so this cannot change any expanded entity edge.
    """
    from dataclasses import replace

    import numpy as np

    from xgraph.index import kg_index_fingerprint, load_kg_index

    source, out = Path(source).resolve(), Path(out).resolve()
    out.relative_to(REPO / "target")
    if out.name not in FAMILIES:
        raise ValueError("alias index directory must name a registered family")
    base = source / "target/mintqa-rematch/indexes" / out.name
    original = json.loads((base / "kg-index.json").read_text())
    variant = json.loads((out / "kg-index.json").read_text())
    if variant.get("initial_case_aliases", {}).get("rule") != "initial-case-v1":
        raise ValueError("refusing to align an index without initial-case aliases")
    for key in (
        "format",
        "tokenizer_fingerprint",
        "span_syntax",
        "expressions",
        "dates",
        "verdicts",
    ):
        if original.get(key) != variant.get(key):
            raise ValueError(f"sealed and alias index differ in {key}")
    for part in ("labels", "relations"):
        if original[part]["owner_ids"] != variant[part]["owner_ids"]:
            raise ValueError(f"sealed and alias {part} owners differ")

    def owners(directory):
        return np.unique(np.load(directory / "relations.owners.npy", mmap_mode="r"))

    if not np.array_equal(owners(base), owners(out)):
        raise ValueError("writable relation owners differ; rebuild with sealed index settings")
    unchanged = ["sitelinks.npy"] if original.get("sitelinks") else []
    unchanged += list(original.get("dates", {}).values())
    for name in unchanged:
        if sha256_file(base / name) != sha256_file(out / name):
            raise ValueError(f"sealed and alias index differ in {name}")
    # Load complete recorded inputs before editing any file. Strict mode can
    # verify them here; no reader sees a temporary, unbound output header.
    bound = variant.get("base_fingerprint") is not None
    variant_index = load_kg_index(out) if bound else None
    base_index = load_kg_index(base) if bound else None
    before = kg_index_fingerprint(out)
    for name in ("tokens", "offsets", "owners"):
        shutil.copyfile(base / f"relations.{name}.npy", out / f"relations.{name}.npy")
    variant["relations"] = original["relations"]
    primary = original.get("relation_primary_rows")
    if primary:
        shutil.copyfile(base / primary, out / primary)
        variant["relation_primary_rows"] = primary
    else:
        variant.pop("relation_primary_rows", None)
        (out / "relation_primary_rows.npy").unlink(missing_ok=True)
    # Alignment is an additional build operation. Keep legacy headers unchanged,
    # and bind a modern variant to the exact relation source and resulting tables.
    if bound:
        variant["build_config"] = {
            **variant["build_config"],
            "relation_alignment": kg_index_fingerprint(base),
        }
        variant["base_fingerprint"] = replace(
            variant_index,
            relations=base_index.relations,
            relation_primary_rows=base_index.relation_primary_rows,
            build_config=variant["build_config"],
        ).base_fingerprint
    (out / "kg-index.json").write_text(json.dumps(variant) + "\n")
    return {
        "artifact_kind": "xgraph.case_alias_index_alignment",
        "base_index_fingerprint": kg_index_fingerprint(base),
        "before_alignment_fingerprint": before,
        "index_fingerprint": kg_index_fingerprint(out),
        "initial_case_aliases": variant["initial_case_aliases"],
        "provenance": provenance(
            [
                base / "kg-index.json",
                Path(__file__).with_name("index.py"),
                Path(__file__).with_name("wikidata.py"),
                Path(__file__),
            ]
        ),
    }


def prepare(source, out, *, draft=False):
    from xgraph.case_alias_design import (
        exposure_inventory,
        index_size_delta,
        label_arrays,
        question_reachability,
        validation_alias_prior,
        worktree_roots,
    )
    from xgraph.case_alias_queue import _git
    from xgraph.tokenizer import ServedTokenizer

    if not draft and _git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("commit the validated implementation before registering")
    source, out = Path(source).resolve(), Path(out).resolve()
    out.relative_to(REPO / "target")
    if out.exists() and ((out / "gpu-ledger.jsonl").exists() or list(out.rglob("*.receipt.json"))):
        raise ValueError("cannot replace a registration after execution")
    out.mkdir(parents=True, exist_ok=True)
    question_path = source / "target/mintqa/questions.json"
    questions = json.loads(question_path.read_text())
    ids = {q["id"] for q in questions}
    witness_paths = [
        source / f"target/mintqa/{name}.jsonl"
        for name in (
            "qwen25-A-all",
            "qwen25-rag-linked-all",
            "qwen25-paths-committed-all",
        )
    ]
    prior_witnesses = []
    for path in witness_paths:
        with path.open() as stream:
            prior_witnesses.append({json.loads(line)["id"] for line in stream if line.strip()})
    inventory = exposure_inventory(worktree_roots(source), ids, out / "exposure-scan.jsonl")
    proof = exposure_proof(ids, [set(inventory["exposed_ids"])])
    proof["inventory"] = inventory
    fresh_ids = set(proof["fresh_ids"])
    new_ids = {q["id"] for q in questions if all_new(q)}
    proof.update(
        all_new_population=len(new_ids),
        fresh_all_new=len(new_ids & fresh_ids),
        fresh_mixed=len(fresh_ids),
    )
    proof["fresh_by_stratum"] = {
        key: sum(q["id"] in fresh_ids and q["complexity"] == key for q in questions)
        for key in sorted({q["complexity"] for q in questions})
    }
    proof["disjointness"] = {
        "exposed_lists_checked": inventory["classifications"].get(
            "exposed or conservative exclusion", 0
        ),
        "fresh_intersections": 0,
        "definition": (
            "F is disjoint from each exposed list because F=P minus their union; "
            "IDs and hashes in manifest"
        ),
    }
    proof["different_model_prior_witnesses"] = [
        {"path": str(p), "sha256": sha256_file(p), "ids": len(w), "ids_sha256": id_digest(w)}
        for p, w in zip(witness_paths, prior_witnesses, strict=True)
    ]
    # Independently pin actual current-model outcomes and their model provenance.
    proof["saturating_current_model_witnesses"] = []
    for name in ("qwen38-agent-strong-8", "qwen38-paths-committed", "gemma31-paths-committed"):
        path = source / f"target/mintqa/{name}.jsonl"
        with path.open() as stream:
            used = {json.loads(line)["id"] for line in stream if line.strip()}
        model_path = Path(str(path) + ".provenance.json")
        if not ids <= used:
            raise ValueError(f"current-model saturation witness changed: {path}")
        proof["saturating_current_model_witnesses"].append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "ids": len(used),
                "ids_sha256": id_digest(used),
                "fresh_intersection": len(used & fresh_ids),
                "model": json.loads(model_path.read_text())["model"],
                "provenance": {"path": str(model_path), "sha256": sha256_file(model_path)},
            }
        )
    if fresh_ids:
        raise ValueError("inventory contradicts independently verified current-model saturation")
    sealed = source / ".claude/worktrees/codex-final/target/final-validation"
    seal_path = sealed / "freeze-manifest.json"
    seal = json.loads(seal_path.read_text())
    priors, power, timings, reports, reachability, sizes = {}, {}, {}, [], {}, {}
    cases_path = (
        source / ".claude/worktrees/codex-error-analysis/target/error-analysis/all-new/items.json"
    )
    cases = json.loads(cases_path.read_text())
    for family in FAMILIES:
        report_path = sealed / family / "report.json"
        report = json.loads(report_path.read_text())["panels"]["validation"]
        reports.append(report_path)
        wt = report["finalist"]["per_question"]
        tokenizer_path = Path(seal["families"][family]["model"]["path"]) / "tokenizer.json"
        tokenizer = ServedTokenizer.from_file(tokenizer_path)
        base = source / "target/mintqa-rematch/indexes" / family
        alias = REPO / "target/case-alias/indexes" / family
        baseline, variant = label_arrays(base), label_arrays(alias)
        validation_ids = {r["id"] for r in wt} & new_ids
        validation_reach = question_reachability(
            [q for q in questions if q["id"] in validation_ids],
            baseline,
            variant,
            tokenizer,
        )
        reachability[family] = {
            "fresh": question_reachability(
                [q for q in questions if q["id"] in fresh_ids], baseline, variant, tokenizer
            ),
            "exposed_validation_all_new": validation_reach,
            "tokenizer": {
                "path": str(tokenizer_path),
                "sha256": sha256_file(tokenizer_path),
                "fingerprint": tokenizer.fingerprint,
            },
        }
        priors[family] = validation_alias_prior(
            [r for r in wt if r["id"] in new_ids],
            validation_reach["affected_ids"],
            [r for r in cases if r["family"] == family and r["panel"] == "validation"],
        )
        power[family] = {}
        for cohort in ("all_new", "mixed"):
            p = proxy_power(
                affected_fraction=0,
                gain_rate=priors[family]["gain_rate"],
                harm_rate=priors[family]["harm_rate"],
                design_effect=1,
                available=0,
                endpoint=cohort,
            )
            p.update(
                affected_fraction=None,
                required_questions=None,
                reason="empty fresh pool: affected fraction and nonzero design are unidentified",
            )
            power[family][cohort] = p
        sizes[family] = index_size_delta(base, alias, vocab_size=tokenizer.vocab_size)
        log_path = sealed / family / "validation/W-revision-terminal-raw.jsonl.log"
        log = log_path.read_text()
        weights_gib = float(re.search(r"Model loading took ([0-9.]+) GiB", log)[1])
        kv_gib = 8 if family == "qwen38" else 16
        sizes[family]["memory_envelope"] = {
            "sealed_admission_gib": 40,
            "kv_cache_gib": kv_gib,
            "measured_model_weights_gib": weights_gib,
            "weights_plus_kv_lower_bound_gib": weights_gib + kv_gib,
            "fits_40_gib_total": False
            if weights_gib
            + kv_gib
            + sizes[family]["indexes"]["WT+alias"]["array_payload_bytes"] / 2**30
            >= 40
            else None,
            "owner_headroom_gib": 60 if family == "qwen38" else 75,
            "interpretation": (
                "admission is a minimum free-memory check; fixed KV ignores memory-share; "
                "not a total cap"
            ),
            "live_fit": "unverified; owner CUDA qualification and startup required",
            "log": {"path": str(log_path), "sha256": sha256_file(log_path)},
        }
        reports.append(log_path)
        timings[family] = [
            t
            for t in seal["families"][family]["estimate"]["generations"]
            if t["generation"] in ("W", "W-revision-terminal-raw")
        ]
    panels = exploratory_members(questions)
    chosen = {r["id"] for members in panels.values() for r in members}
    # Relation sequences come from released chain metadata, not output/gold fields.
    from xgraph.mintqa_bench import dataset_rows, relation_clusters

    data = source / "data/external/mintqa-full/questions"
    population_rows = dataset_rows([f"ti={data / 'ti.jsonl'}", f"pop={data / 'pop.jsonl'}"])
    if set(population_rows) != ids:
        raise ValueError("historical population differs from the complete released question IDs")
    clusters = relation_clusters(population_rows)
    for cohort in panels.values():
        for member in cohort:
            qid = member["id"]
            member.update(cluster=clusters[qid], hops=len(population_rows[qid]["triple"]))
    paths = source / "target/mintqa/requests-paths.jsonl"
    requests = [r for r in read_jsonl(paths) if r["id"] in chosen]
    if {r["id"] for r in requests} != chosen or len(requests) != len(chosen):
        raise ValueError("request IDs are not unique and complete")
    requests = [{"id": r["id"], "messages": r["messages"]} for r in requests]
    write_jsonl(out / "requests.jsonl", requests)
    write_json(out / "members.json", panels)
    write_json(out / "exposure-proof.json", proof)
    workload = {
        f: 2 * len(chosen) * sum(t["seconds"] / t["requests"] for t in ts) / 3600
        for f, ts in timings.items()
    }
    registration = {
        "artifact_kind": "xgraph.case_alias_registration",
        "version": 2,
        "status": (
            "confirmation blocked by current-model full-population exposure; "
            "exploratory mode retained"
        ),
        "draft": draft,
        "owner_rulings": OWNER_RULINGS,
        "source": str(source),
        "out": str(out),
        "seed": 17,
        "code_digest": _code_digest(),
        "exposure": proof,
        "rule": "initial-case-v1; title/sentence conversion excluded",
        "families": FAMILIES,
        "arms": ["WT", "WT+alias"],
        "prospective_design": {
            "primary": "paired accuracy superiority on fresh all-new items",
            "no_harm": "paired accuracy NI, margin -0.01, on fresh stratified mixed items",
            "success": (
                "both endpoints pass in both families; intersection-union, "
                "one-sided alpha .025 each"
            ),
            "cluster": (
                "relation sequence within hop; calibrated paired cluster bootstrap, "
                "10000 resamples, seed 17"
            ),
            "sampling": (
                "seed 17; all-new enriched and full-population complexity-stratified "
                "mixed samples; overlap may be run once"
            ),
            "power": power,
            "fresh_sample_sizes": {"all_new": 0, "mixed": 0},
            "unique_questions": 0,
            "joint_power": 0,
            "admitted": False,
            "decision": (
                "largest eligible design is n=0 even within 10 GPU-h; current-model "
                "outcomes saturate the population independently of the Qwen2.5 exception"
            ),
        },
        "possible_design": {
            "label": "exploratory exposed-data replication; no confirmatory admission or NI claim",
            "all_new": len(panels["all_new"]),
            "mixed": len(panels["mixed"]),
            "unique_questions": len(chosen),
            "selection": "metadata-only; no errors used",
            "generation_seeds": "17 in both calls and arms, sampling and analysis",
        },
        "validation_priors": priors,
        "reachability": reachability,
        "index_size_delta": sizes,
        "confirmatory_forecast": {
            "workload_gpu_hours": {f: 0 for f in FAMILIES},
            "startup_reserve_hours": 8 * 180 / 3600,
            "cuda_reserve_hours": 300 / 3600,
            "total_gpu_hours": None,
            "hard_cap_hours": 10,
            "basis": "no eligible workload; reserves do not create a confirmatory design",
        },
        "timing_sources": timings,
        "forecast": {
            "workload_gpu_hours": workload,
            "startup_reserve_hours": 8 * 180 / 3600,
            "cuda_reserve_hours": 300 / 3600,
            "total_gpu_hours": sum(workload.values()) + 0.4 + 300 / 3600,
            "hard_cap_hours": 4,
            "basis": (
                "sealed WT development per-question workload costs, two arms; "
                "startup reserve, not controlled serving latency"
            ),
        },
        "sealed_manifest": {"path": str(seal_path), "sha256": sha256_file(seal_path)},
        "policies": {f: seal["families"][f]["selected_policy"] for f in FAMILIES},
        "common_overrides": {"seed": 17, "note": "task seed replaces sealed first-call 0"},
        "index_builds": {
            family: [
                "nice",
                "-n",
                "19",
                "taskset",
                "-c",
                "1-4",
                "prlimit",
                "--as=4000000000",
                str(REPO / ".venv/bin/python"),
                "-m",
                "xgraph.wikidata",
                "index",
                "--kg",
                str(source / "data/external/mintqa-full/kg"),
                "--tokenizer",
                str(Path(seal["families"][family]["model"]["path"]) / "tokenizer.json"),
                "--initial-case-aliases",
                "--out",
                str(REPO / "target/case-alias/indexes" / family),
            ]
            for family in FAMILIES
        },
        "index_alignment_argv": {
            family: [
                "nice",
                "-n",
                "19",
                "taskset",
                "-c",
                "1-4",
                "prlimit",
                "--as=4000000000",
                str(REPO / ".venv/bin/python"),
                "-m",
                "xgraph.case_alias",
                "align-index",
                "--source",
                str(source),
                "--out",
                str(REPO / "target/case-alias/indexes" / family),
            ]
            for family in FAMILIES
        },
        "provenance": provenance(
            [
                question_path,
                paths,
                seal_path,
                *reports,
                cases_path,
                *(Path(r["path"]) for r in proof["saturating_current_model_witnesses"]),
                *witness_paths,
                data / "ti.jsonl",
                data / "pop.jsonl",
                *(
                    source
                    / ".claude/worktrees/codex-error-analysis/target/error-analysis/all-new"
                    / name
                    for name in ("report.md", "fix-candidate.md", "items.json", "tables.md")
                ),
                *(
                    source / "data/external/mintqa-full/kg" / name
                    for name in (
                        "manifest.json",
                        "entity_labels.jsonl",
                        "relation_labels.jsonl",
                        "triples.npy",
                        "sitelink_counts.npy",
                    )
                ),
                Path(__file__).with_name("index.py"),
                Path(__file__).with_name("wikidata.py"),
                Path(__file__),
            ]
        ),
    }
    if registration["forecast"]["total_gpu_hours"] > 4:
        raise ValueError("exploratory forecast exceeds cap")
    from xgraph.case_alias_queue import prepare_queue

    prepare_queue(registration, seal, out)
    write_json(out / "registration.json", registration)
    (out / "registration.sha256").write_text(sha256_file(out / "registration.json") + "\n")
    return registration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "queue", "score", "align-index"))
    parser.add_argument("--source", type=Path, default=Path("/home/jonas/git/xgraph"))
    parser.add_argument("--out", type=Path, default=REPO / "target/queue-case-alias")
    parser.add_argument("--run-exploratory", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--draft", action="store_true", help="CPU preflight only; cannot be executed"
    )
    args = parser.parse_args(argv)
    if args.action == "align-index":
        result = align_index(args.source, args.out)
        write_json(args.out.with_name(args.out.name + ".alignment.json"), result)
        print(json.dumps(result))
    elif args.action == "prepare":
        result = prepare(args.source, args.out, draft=args.draft)
        print(json.dumps({"status": result["status"], "forecast": result["forecast"]}))
    elif args.action == "queue":
        from xgraph.case_alias_queue import queue

        if args.run and args.run_exploratory:
            parser.error("choose either confirmatory --run or --run-exploratory")
        queue(args.out, run=args.run or args.run_exploratory, exploratory=args.run_exploratory)
    else:
        from xgraph.case_alias_queue import score

        score(args.out)


if __name__ == "__main__":
    main()
