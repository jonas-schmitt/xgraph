"""Hash-checked PlanBench gold cases; inference uses PDDL, never cached model replies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from xgraph.planbench_pddl import Problem, load_problem

COMMIT = "fc638a1aff7df3fe7a1a1d289fa2c04cc24dc284"
DOMAINS = ("blocksworld", "logistics", "mystery_blocksworld")


def verify_renaming(base: Problem, mystery: Problem) -> None:
    predicates = {
        "clear": "province",
        "ontable": "planet",
        "handempty": "harmony",
        "holding": "pain",
        "on": "craves",
    }
    names = {"pick-up": "attack", "put-down": "succumb", "stack": "overcome", "unstack": "feast"}

    def facts(state):
        return frozenset((predicates[f[0]], *f[1:]) for f in state)

    def action_key(action, rename):
        return (
            (names[action.name] if rename else action.name),
            action.arguments,
            *(
                facts(getattr(action, k)) if rename else getattr(action, k)
                for k in ("positive", "negative", "add", "delete")
            ),
        )

    if (
        base.objects != mystery.objects
        or any(
            facts(getattr(base, k)) != getattr(mystery, k)
            for k in ("init", "goal_positive", "goal_negative")
        )
        or {action_key(a, True) for a in base.actions}
        != {action_key(a, False) for a in mystery.actions}
    ):
        raise ValueError("Mystery control is not an exact Blocksworld renaming")


@dataclass(frozen=True)
class Case:
    id: str
    domain: str
    cluster: str
    problem: Problem
    gold_plan: str
    domain_path: Path
    problem_path: Path


def load_cases(root: Path, domain: str | None = None) -> list[Case]:
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["commit"] != COMMIT or (domain is not None and domain not in DOMAINS):
        raise ValueError("unsupported PlanBench commit or domain")
    sources = {}
    for entry in manifest["files"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root) or entry["path"] in sources:
            raise ValueError(f"unsafe or duplicate manifest path: {path}")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"PlanBench checksum mismatch: {path}")
        sources[entry["path"]] = data.decode()
    gold_sources = {}
    result = []
    seen = set()
    for row in manifest["cases"]:
        if row["domain"] not in DOMAINS:
            raise ValueError(f"unsupported PlanBench domain: {row['domain']}")
        if domain is not None and row["domain"] != domain:
            continue
        key = f"{row['domain']}:{row['id']}"
        if key in seen:
            raise ValueError(f"duplicate PlanBench case: {key}")
        seen.add(key)
        source = row["gold_source"]
        if source not in gold_sources:
            records = json.loads(sources[source])["instances"]
            gold_sources[source] = {int(r["instance_id"]): r["ground_truth_plan"] for r in records}
            if len(gold_sources[source]) != len(records):
                raise ValueError(f"duplicate gold id in {source}")
        problem = load_problem(sources[row["domain_path"]], sources[row["problem_path"]])
        cluster = "logistics" if row["domain"] == "logistics" else "blocks"
        result.append(
            Case(
                key,
                row["domain"],
                f"{cluster}:{row['id']}",
                problem,
                gold_sources[source][int(row["id"])],
                root / row["domain_path"],
                root / row["problem_path"],
            )
        )
    return sorted(result, key=lambda c: (c.domain, int(c.id.split(":")[1])))
