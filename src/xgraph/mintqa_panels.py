"""Seeded, relation-disjoint MINTQA development and internal validation panels.

Within each hop count, order relation-sequence clusters by size and form blocks of
two (a final odd block has three). Randomly put one cluster in development and
the others in validation. Sample questions uniformly without replacement inside
each panel's hop pool. Record both sampling stages, then calibrate their inverse
probability weights to the population hop totals. These are two-stage survey
weights; calibration gives a bounded, hop-poststratified Hajek accuracy estimate.
The split uses no answers or model outcomes. MINTQA remains exposed development
data: internal validation is not fresh confirmatory evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from xgraph.mintqa_bench import dataset_rows, relation_clusters
from xgraph.provenance import provenance, sha256_file

PANEL_NAMES = ("development", "validation")


def relation_cardinality_members(members, cardinalities, *, size, seed):
    """Sample relation strata within endpoint-cardinality bins, then one member.

    Cardinalities must come from a gold-free, predeclared execution. Neither
    answers nor correctness enter selection. Divide the parent survey weight by
    the conditional inclusion probability. This follows the endpoint pilot's
    sampling method with an independently declared seed and cardinality source.
    """
    ids = [m["id"] for m in members]
    if len(set(ids)) != len(ids) or set(ids) != set(cardinalities):
        raise ValueError("cardinalities must match unique member ids")
    groups = defaultdict(lambda: defaultdict(list))
    for member in sorted(members, key=lambda m: m["id"]):
        count, weight = cardinalities[member["id"]], member["weight"]
        if type(count) is not int or count < 0:
            raise ValueError("endpoint cardinalities must be nonnegative integers")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("member weights must be finite and positive")
        bin_name = "0" if count == 0 else "1" if count == 1 else "2-8" if count <= 8 else "9+"
        groups[bin_name][member["cluster"]].append(member)
    if type(size) is not int or not len(groups) <= size <= sum(map(len, groups.values())):
        raise ValueError("sample size must fit the nonempty relation/cardinality strata")
    quota = dict.fromkeys(sorted(groups), 1)
    mass = {b: sum(map(len, g.values())) for b, g in groups.items()}
    while sum(quota.values()) < size:
        bin_name = max(
            (b for b in quota if quota[b] < len(groups[b])),
            key=lambda b: (mass[b] / (quota[b] + 1), b),
        )
        quota[bin_name] += 1
    rng, selected = random.Random(seed), []
    for bin_name, count in quota.items():
        strata = groups[bin_name]
        for cluster in rng.sample(sorted(strata), count):
            candidates = strata[cluster]
            member = rng.choice(candidates)
            probability = count / len(strata) / len(candidates)
            selected.append(
                {
                    **member,
                    "panel_weight": member["weight"],
                    "weight": member["weight"] / probability,
                    "endpoint_cardinality": cardinalities[member["id"]],
                    "endpoint_bin": bin_name,
                    "inclusion_probability": probability,
                }
            )
    return sorted(selected, key=lambda m: m["id"])


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def indexed(records: list[dict]) -> dict[str, dict]:
    result = {}
    for row in records:
        if row["id"] in result:
            raise ValueError(f"duplicate record id {row['id']!r}")
        result[row["id"]] = row
    return result


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def construct_panels(
    rows: dict[str, dict], development: int = 1024, validation: int = 2048, seed: int = 17
) -> dict[str, list[dict]]:
    """Draw both panels, with exact hop balance and recorded population weights."""
    if not rows or any(not r.get("triple") for r in rows.values()):
        raise ValueError("every population question needs a nonempty chain")
    clusters = relation_clusters(rows)
    grouped = defaultdict(lambda: defaultdict(list))
    for qid in sorted(rows):
        chain = rows[qid]["triple"]
        if any(len(t) != 3 or not t[1] for t in chain):
            raise ValueError(f"{qid}: invalid chain")
        grouped[len(chain)][clusters[qid]].append(qid)
    sizes = dict(zip(PANEL_NAMES, (development, validation), strict=True))
    if any(n <= 0 or n % len(grouped) for n in sizes.values()):
        raise ValueError("panel sizes must be positive multiples of the number of hop strata")
    rng = random.Random(seed)
    panels = {name: [] for name in PANEL_NAMES}
    for hops, groups in sorted(grouped.items()):
        ordered = sorted(groups, key=lambda c: (len(groups[c]), c))
        if len(ordered) < 2:
            raise ValueError(f"hop {hops}: capacity requires at least two relation sequences")
        blocks, at = [], 0
        while at < len(ordered):
            count = 3 if len(ordered) - at == 3 else 2
            blocks.append(ordered[at : at + count])
            at += count
        # No redraws conditioned on pool capacity: those would change inclusion probabilities.
        minimum = {
            "development": sum(min(len(groups[c]) for c in b) for b in blocks),
            "validation": sum(sum(sorted(len(groups[c]) for c in b)[: len(b) - 1]) for b in blocks),
        }
        if any(minimum[name] < sizes[name] // len(grouped) for name in PANEL_NAMES):
            raise ValueError(f"hop {hops}: guaranteed pool capacity {minimum} is too small")
        pools = {name: [] for name in PANEL_NAMES}
        probability = {}
        for block in blocks:
            rng.shuffle(block)
            for k, cluster in enumerate(block):
                name = "development" if k == 0 else "validation"
                probability[cluster] = (1 if k == 0 else len(block) - 1) / len(block)
                pools[name].extend(groups[cluster])
        population_hop = sum(map(len, groups.values()))
        for name, pool in pools.items():
            n = sizes[name] // len(grouped)
            chosen = sorted(rng.sample(sorted(pool), n))
            members = []
            for qid in chosen:
                first = probability[clusters[qid]]
                inclusion = first * n / len(pool)
                members.append(
                    {
                        "id": qid,
                        "hops": hops,
                        "cluster": clusters[qid],
                        "cluster_probability": first,
                        "within_pool_probability": n / len(pool),
                        "inclusion_probability": inclusion,
                        "design_weight": 1 / inclusion,
                        "population_hop": population_hop,
                        "pool_questions": len(pool),
                    }
                )
            scale = population_hop / sum(r["design_weight"] for r in members)
            for row in members:
                row["weight"] = row["design_weight"] * scale
            panels[name].extend(members)
    return panels


def build_panels(
    source: Path,
    data: Path,
    out: Path,
    development: int = 1024,
    validation: int = 2048,
    seed: int = 17,
) -> dict:
    """Freeze inputs, requests, members, weights and provenance; refuse partial/stale artifacts."""
    inputs = [
        source / "questions.json",
        source / "requests-paths.jsonl",
        data / "questions/ti.jsonl",
        data / "questions/pop.jsonl",
    ]
    settings = {"development": development, "validation": validation, "seed": seed}
    manifest_path = out / "panels.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["settings"] != settings:
            raise ValueError("frozen panel settings differ; choose a new output directory")
        for item in saved["provenance"]["inputs"] + saved["outputs"]:
            if sha256_file(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"frozen panel input/output changed: {item['path']}")
        if {str(p.resolve()) for p in inputs} != {
            str(Path(i["path"]).resolve()) for i in saved["provenance"]["inputs"]
        }:
            raise ValueError("frozen panel source paths differ")
        return saved
    if any((out / name).exists() for name in PANEL_NAMES):
        raise ValueError("partial frozen panels; preserve artifacts and choose a new directory")
    rows = dataset_rows([f"ti={inputs[2]}", f"pop={inputs[3]}"])
    questions = indexed(json.loads(inputs[0].read_text()))
    requests = indexed(read_jsonl(inputs[1]))
    if set(rows) != set(questions) or set(rows) != set(requests):
        raise ValueError("population question/request/source ids differ")
    panels = construct_panels(rows, development, validation, seed)
    result = {
        "artifact_kind": "xgraph.mintqa_panels",
        "version": 1,
        "settings": settings,
        "population": len(rows),
        "population_by_group": dict(
            sorted(Counter(q["complexity"] for q in questions.values()).items())
        ),
        "panels": {},
        "outputs": [],
        "provenance": provenance(inputs),
    }
    for name, members in panels.items():
        selected_requests, selected_questions = [], []
        for member in members:
            qid = member["id"]
            request = requests[qid]
            if request.get("anchor") or request.get("kg_version"):
                raise ValueError(f"{qid}: rematch inputs must be unanchored on the base KG")
            if request["messages"][-1] != {"role": "user", "content": rows[qid]["question"]}:
                raise ValueError(f"{qid}: request question differs from the released question")
            selected_requests.append({"id": qid, "messages": request["messages"]})
            selected_questions.append(
                {k: questions[qid][k] for k in ("id", "utterance", "complexity")}
            )
        files = [out / name / "questions.json", out / name / "requests.jsonl"]
        write_json(files[0], selected_questions)
        write_jsonl(files[1], selected_requests)
        result["outputs"].extend({"path": str(p), "sha256": sha256_file(p)} for p in files)
        result["panels"][name] = {
            "members": members,
            "questions": len(members),
            "clusters": len({r["cluster"] for r in members}),
            "by_hop": {str(k): v for k, v in sorted(Counter(r["hops"] for r in members).items())},
            "by_group": dict(sorted(Counter(q["complexity"] for q in selected_questions).items())),
            "weight_sum": sum(r["weight"] for r in members),
        }
    write_json(manifest_path, result)
    return result


def pilot_members(members: list[dict], size: int, seed: int = 17) -> list[dict]:
    """Balance hops, maximise whole-cluster coverage, split at most one per hop.

    Seeded subset-sum selection keeps every cluster whole whenever an exact fit
    exists. Otherwise the largest whole-cluster fit is completed from one cluster.
    Parent weights retain their relative values within each hop and are calibrated
    back to its full-panel mass. This is a conditional development diagnostic,
    without a new probability-sample claim about the population.
    """
    indexed(members)
    groups = defaultdict(lambda: defaultdict(list))
    cluster_hops = {}
    for row in members:
        hop, cluster = row["hops"], row["cluster"]
        if not math.isfinite(row["weight"]) or row["weight"] <= 0:
            raise ValueError("pilot needs finite positive parent weights")
        if cluster_hops.setdefault(cluster, hop) != hop:
            raise ValueError(f"pilot cluster {cluster!r} crosses hop strata")
        groups[hop][cluster].append(row)
    if not groups or type(size) is not int or size <= 0 or size % len(groups):
        raise ValueError("pilot size must be a positive multiple of the number of hop strata")
    quota = size // len(groups)
    rng = random.Random(seed)
    selected = []
    for hop, clusters in sorted(groups.items()):
        if sum(map(len, clusters.values())) < quota:
            raise ValueError(f"pilot hop {hop}: insufficient frozen development capacity")
        order = sorted(clusters)
        rng.shuffle(order)
        reachable = {0: ()}
        for cluster in order:
            for mass, chosen in list(reachable.items()):
                total = mass + len(clusters[cluster])
                if total <= quota:
                    reachable.setdefault(total, (*chosen, cluster))
        whole = reachable[max(reachable)]
        chosen = [r for c in whole for r in clusters[c]]
        remainder = quota - len(chosen)
        if remainder:
            cluster = next(c for c in order if c not in whole)
            chosen += rng.sample(sorted(clusters[cluster], key=lambda r: r["id"]), remainder)
        mass = sum(r["weight"] for rows in clusters.values() for r in rows)
        scale = mass / sum(r["weight"] for r in chosen)
        selected.extend(
            {**r, "full_panel_weight": r["weight"], "weight": r["weight"] * scale}
            for r in sorted(chosen, key=lambda r: r["id"])
        )
    return selected


def build_pilot(frozen: Path, out: Path, size: int, seed: int = 17) -> dict:
    """Freeze a pilot beside, never inside, its read-only parent panel."""
    frozen, out = frozen.resolve(), out.resolve()
    if out.resolve().is_relative_to(
        frozen.parent.resolve()
    ) or frozen.parent.resolve().is_relative_to(out.resolve()):
        raise ValueError("pilot outputs must be separate from the frozen panel")
    inputs = [
        frozen,
        frozen.parent / "development/questions.json",
        frozen.parent / "development/requests.jsonl",
    ]
    parent = json.loads(frozen.read_text())
    members = parent["panels"]["development"]["members"]
    selected = pilot_members(members, size, seed)
    questions = indexed(json.loads(inputs[1].read_text()))
    requests = indexed(read_jsonl(inputs[2]))
    if set(questions) != set(requests) or set(questions) != set(indexed(members)):
        raise ValueError("frozen development question/request/member ids differ")
    settings = {"pilot": size, "pilot_seed": seed}
    manifest_path = out / "panels.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["settings"] != settings or [str(p) for p in inputs] != [
            i["path"] for i in saved["provenance"]["inputs"]
        ]:
            raise ValueError("pilot settings or source differ; choose a new directory")
        for item in saved["provenance"]["inputs"] + saved["outputs"]:
            if sha256_file(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"frozen pilot input/output changed: {item['path']}")
        return saved
    if (out / "development").exists():
        raise ValueError("partial pilot; preserve artifacts and choose a new directory")
    counts = Counter(r["cluster"] for r in selected)
    full_counts = Counter(r["cluster"] for r in members)
    result = {
        "artifact_kind": "xgraph.mintqa_panels",
        "version": 1,
        "settings": settings,
        "population_by_group": parent["population_by_group"],
        "pilot": {
            "size": size,
            "seed": seed,
            "full_panel_questions": len(members),
            "full_panel_by_hop": dict(sorted(Counter(str(r["hops"]) for r in members).items())),
            "split_clusters": sum(n < full_counts[c] for c, n in counts.items()),
            "method": "seeded whole-cluster subset sum within hops; at most one split per hop",
            "weights": "parent relative weights renormalised to full-panel mass within each hop",
            "interpretation": "conditional development pilot; not confirmatory population evidence",
        },
        "panels": {
            "development": {
                "members": selected,
                "questions": size,
                "clusters": len(counts),
                "by_hop": dict(sorted(Counter(str(r["hops"]) for r in selected).items())),
                "weight_sum": sum(r["weight"] for r in selected),
            }
        },
        "outputs": [],
        "provenance": provenance(inputs),
    }
    question_file, request_file = (
        out / "development/questions.json",
        out / "development/requests.jsonl",
    )
    write_json(question_file, [questions[r["id"]] for r in selected])
    write_jsonl(request_file, [requests[r["id"]] for r in selected])
    result["outputs"] = [
        {"path": str(p), "sha256": sha256_file(p)} for p in (question_file, request_file)
    ]
    write_json(manifest_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--frozen", type=Path, help="parent panels.json for a development pilot")
    parser.add_argument("--pilot", type=int)
    parser.add_argument("--pilot-seed", type=int, default=17)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--development", type=int, default=1024)
    parser.add_argument("--validation", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    if args.frozen is not None and args.pilot is not None:
        if args.source is not None or args.data is not None:
            parser.error("--frozen/--pilot cannot be combined with --source/--data")
        result = build_pilot(args.frozen, args.out, args.pilot, args.pilot_seed)
    else:
        if (
            args.frozen is not None
            or args.pilot is not None
            or args.source is None
            or args.data is None
        ):
            parser.error("supply --source and --data, or --frozen and --pilot")
        result = build_panels(
            args.source, args.data, args.out, args.development, args.validation, args.seed
        )
    print(
        json.dumps(
            {
                name: {k: v for k, v in r.items() if k != "members"}
                for name, r in result["panels"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
