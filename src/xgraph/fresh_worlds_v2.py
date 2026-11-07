"""Prospectively fixed difficulty interventions; no model or policy imports."""

from dataclasses import asdict, dataclass

from xgraph.fresh_worlds import CONDITIONS, _rng, digest
from xgraph.fresh_worlds_gold import execute

DESIGN = "controlled-v2"
TYPES = {"person": 8, "team": 52, "project": 20, "organization": 16, "city": 24, "country": 8}
CELLS = (
    (1, "distinct", 1),
    (1, "confusable", 1),
    (4, "distinct", 1),
    (4, "confusable", 1),
    (4, "distinct", 16),
    (4, "confusable", 16),
)
ROLES = (
    (0, 0, 0, 0),
    (1, 1, 1, 1),
    (2, 2, 2, 2),
    (0, 1, 2, 0),
    (1, 2, 0, 1),
    (2, 0, 1, 2),
    (0, 2, 1, 0),
    (2, 1, 0, 2),
)
LABELS = (
    (("member of", "left", "joining"), ("current team", "former team", "future team")),
    (
        ("works on", "reserve assignment", "bid for"),
        ("main project", "backup project", "proposed project"),
    ),
    (
        ("sponsored by", "managed by", "checked by"),
        ("funding organization", "operating organization", "auditing organization"),
    ),
    (
        ("registered in", "operates in", "departed from"),
        ("legal headquarters", "operational headquarters", "former headquarters"),
    ),
)
CLAUSES = (
    (
        "the team to which {x} presently belongs",
        "the team that {x} previously left",
        "the team that {x} intends to join",
    ),
    (
        "the project on which {x} primarily works",
        "the project held in reserve by {x}",
        "the project for which {x} has submitted a bid",
    ),
    (
        "the organization that supplies money for {x}",
        "the organization that runs {x}",
        "the organization that examines the accounts of {x}",
    ),
    (
        "the city that is the statutory seat of {x}",
        "the city where {x} conducts daily operations",
        "the city from which {x} has moved away",
    ),
)
CHAIN_TYPES = ("person", "team", "project", "organization", "city")
SCHEMA = (
    *(
        (label, CHAIN_TYPES[g], CHAIN_TYPES[g + 1])
        for g, modes in enumerate(LABELS)
        for labels in modes
        for label in labels
    ),
    ("reference record", "record", "record"),
)


@dataclass(frozen=True)
class V2Config:
    seed: int
    worlds: int = 96
    split: str = "development"
    pairs: int = 368
    design: str = DESIGN


def relation(group, role, confusable):
    return group * 6 + int(confusable) * 3 + role + 1


def question(roles, depth):
    text = "{start}"
    for group in range(4 - depth, 4):
        text = CLAUSES[group][roles[group]].format(x=text)
    return "Name " + text + "."


def world(config, number, template, cell, *, demo=False):
    depth, mode, width = cell
    block = number // len(CELLS)
    rng = _rng("topology-v2", 871 if demo else config.seed, "demo" if demo else config.split, block)
    offset = (number if demo else number + 5) * 128 + 1
    ids = list(range(offset, offset + 128))
    rng.shuffle(ids)
    pools, entities, at = {}, {}, 0
    for kind, count in TYPES.items():
        pools[kind] = ids[at : at + count]
        entities.update((item, kind) for item in pools[kind])
        at += count
    roles = ROLES[template]
    rels = [relation(g, roles[g], mode == "confusable") for g in range(4)]
    roots = pools["person"][:2]
    teams, projects, orgs, cities = (pools[k] for k in CHAIN_TYPES[1:])
    # All six cells share this entire graph. Width changes only the query root.
    # Each sibling has matched arity, so degree cannot identify the right relation.
    edges = {(s, 25, o) for s, o in zip(ids, ids[1:] + ids[:1], strict=True)}
    role_order = [roles[0], *(r for r in range(3) if r != roles[0])]
    for rank, role in enumerate(role_order):
        p = relation(0, role, mode == "confusable")
        edges.add((roots[0], p, teams[48 + rank]))
        edges.update((roots[1], p, teams[rank * 16 + i]) for i in range(16))
    for i in range(16):
        edges.add((teams[i], rels[1], projects[i]))
        for rank, role in enumerate(r for r in range(3) if r != roles[1]):
            edges.add((teams[i], relation(1, role, mode == "confusable"), projects[16 + rank]))
    for i in range(16, 51):
        for role in range(3):
            selected = 16 + i % 2
            if i == 48:
                selected = (
                    0
                    if role == roles[1]
                    else 16 + [r for r in range(3) if r != roles[1]].index(role)
                )
            edges.add((teams[i], relation(1, role, mode == "confusable"), projects[selected]))
    for i in range(16):
        for rank, role in enumerate(r for r in range(3) if r != roles[2]):
            edges.add((projects[i], relation(2, role, mode == "confusable"), orgs[1 + rank]))
    edges.add((projects[0], rels[2], orgs[0]))
    for i in (16, 17):
        for role in range(3):
            edges.add((projects[i], relation(2, role, mode == "confusable"), orgs[1 + i % 2]))
    for i in range(4):
        edges.add((orgs[i], rels[3], cities[i]))
        for rank, role in enumerate(r for r in range(3) if r != roles[3]):
            edges.add((orgs[i], relation(3, role, mode == "confusable"), cities[4 + rank]))
    # One-hop worlds necessarily edit the answer edge; four-hop worlds edit hop 3.
    start, plan = (roots[int(width == 16)], rels) if depth == 4 else (orgs[0], rels[3:])
    inner = [projects[0], rels[2], orgs[3]] if depth == 4 else [orgs[0], rels[3], cities[3]]
    off = [projects[18], rels[2], orgs[5]] if depth == 4 else [orgs[6], rels[3], cities[7]]
    edges.add((projects[18], rels[2], orgs[4]))
    edges.add((orgs[6], rels[3], cities[6]))
    # Model-free background topology varies across blocks, while the protected
    # query motifs and every cell's global graph size remain identical.
    protected = {0: set(roots), 1: set(teams[:51]), 2: set(projects[:19]), 3: set(orgs[:7])}
    candidates = [
        (s, g, role)
        for g in range(4)
        for s in pools[CHAIN_TYPES[g]]
        if s not in protected[g]
        for role in range(3)
    ]
    rng.shuffle(candidates)
    needed = config.pairs - len({(s, p) for s, p, _ in edges})
    if not 0 <= needed <= len(candidates):
        raise ValueError("v2 fixed pair quota cannot fit protected query motifs")
    for s, g, role in candidates[:needed]:
        edges.add(
            (s, relation(g, role, mode == "confusable"), rng.choice(pools[CHAIN_TYPES[g + 1]]))
        )
    triples = sorted(edges)
    conditions = []
    for name, edits in zip(CONDITIONS, ([], [inner], [off]), strict=True):
        gold = execute(triples, start, plan, edits)
        if len(gold["answers"]) != 1 or max(map(len, gold["frontiers"])) != width:
            raise ValueError("v2 singleton/frontier invariant failed")
        conditions.append(
            {"name": name, "edits": edits, "gold": gold["answers"], "max_frontier": width}
        )
    if (
        conditions[0]["gold"] == conditions[1]["gold"]
        or conditions[0]["gold"] != conditions[2]["gold"]
    ):
        raise ValueError("v2 version invariant failed")
    return {
        "id": f"demo-{number}" if demo else f"{config.split}-{number:04d}",
        "block": f"{config.split}-b{block:04d}",
        "template": f"v2-t{template}",
        "cell": list(cell),
        "question_template": question(roles, depth),
        "relations": plan,
        "start": start,
        "entities": dict(sorted(entities.items())),
        "bounds": [offset, offset + 128],
        "triples": triples,
        "conditions": conditions,
        "edit_hop": 2 if depth == 4 else 0,
        "topology_sha256": digest([(s - offset, p, o - offset) for s, p, o in triples]),
    }


def generate(config, *, confirmation_seal=None):
    if (
        type(config.seed) is not int
        or type(config.worlds) is not int
        or config.worlds < 6
        or config.worlds % 6
        or config.split not in ("development", "confirmation")
        or config.pairs != 368
    ):
        raise ValueError("v2 requires complete six-cell blocks and fixed construction parameters")
    if config.split == "confirmation":
        from xgraph.fresh_worlds import _confirm_admission

        _confirm_admission(confirmation_seal, config)
    worlds = [world(config, n, (n // 6) % len(ROLES), CELLS[n % 6]) for n in range(config.worlds)]
    return {
        "format": 2,
        "config": asdict(config),
        "schema": SCHEMA,
        "templates": [{"id": f"v2-t{i}", "roles": roles} for i, roles in enumerate(ROLES)],
        "worlds": worlds,
        "demonstrations": [world(config, n, n, CELLS[n], demo=True) for n in range(5)],
    }


def register(study_dir, audit_path, v1_archive):
    """Bind the prospective design before the owner may prepare a model queue."""
    import json
    from datetime import UTC, datetime
    from pathlib import Path

    from xgraph.fresh_worlds import generator_sources, write_json
    from xgraph.provenance import provenance, sha256_file

    study_dir, audit_path, v1_archive = map(Path, (study_dir, audit_path, v1_archive))
    out = study_dir / "registration.json"
    if out.exists():
        raise ValueError("v2 registration already exists; no output-guided revision is allowed")
    study = json.loads((study_dir / "study.json").read_text())
    review = json.loads(audit_path.read_text())
    repo = Path(__file__).resolve().parents[2]
    if study["config"] != asdict(V2Config(seed=26028)):
        raise ValueError("v2 registration requires the fixed development configuration")
    for path in (study_dir / "study.json", repo / "scripts/fresh-worlds-v2-audit.py"):
        if review["provenance"]["inputs"].get(str(path.resolve())) != sha256_file(path):
            raise ValueError("v2 independent validity audit is stale")
    if review.get("status") != "passed":
        raise ValueError("v2 independent validity audit failed")
    if study["source_sha256"] != {
        name: sha256_file(Path(__file__).with_name(name)) for name in generator_sources(study)
    }:
        raise ValueError("v2 registration generator source changed; rebuild before registration")
    paths = [
        *(p for p in study_dir.rglob("*") if p.is_file()),
        audit_path,
        repo / "docs/fresh-worlds-v2.md",
        repo / "scripts/fresh-worlds-v2-audit.py",
        *(Path(__file__).with_name(n) for n in generator_sources(study)),
        *(
            Path(__file__).with_name(n)
            for n in (
                "fresh_worlds_analysis.py",
                "fresh_worlds_v2_analysis.py",
                "fresh_worlds_queue.py",
            )
        ),
    ]
    disclosure = [
        v1_archive / n
        for n in ("analysis.json", "assurance.json", "budget.json", "registration-draft.md")
    ]
    record = {
        "format": 2,
        "design": DESIGN,
        "created_at": datetime.now(UTC).isoformat(),
        "files": {str(p.resolve()): sha256_file(p) for p in paths},
        "v1_seen_before_registration": {str(p.resolve()): sha256_file(p) for p in disclosure},
        "config": study["config"],
        "confirmation_seed": None,
        "primary_contrast": "D(4,confusable,1)-D(1,distinct,1)",
        "predictions": {
            f: {"interaction": ">0; useful threshold .03", "easy": "within +/-.03"}
            for f in ("qwen38", "gemma31")
        },
        "use_of_v1": "diagnosis only; retain and report the entire rejected exploratory v1",
        "provenance": provenance(paths),
    }
    record["sha256"] = digest(record)
    write_json(out, record)
    return record


def validate_registration(study_dir):
    import json
    from pathlib import Path

    from xgraph.fresh_worlds_queue import _commit
    from xgraph.provenance import sha256_file

    path = Path(study_dir) / "registration.json"
    if not path.is_file():
        raise ValueError("v2 requires a prospective registration and independent validity audit")
    record = json.loads(path.read_text())
    expected = record.pop("sha256")
    if digest(record) != expected:
        raise ValueError("v2 registration hash mismatch")
    repo = Path(__file__).resolve().parents[2]
    if _commit(repo) != record["provenance"]["repo_commit"]:
        raise ValueError("v2 registration code commit changed; prompts and policies are frozen")
    for name, value in {**record["files"], **record["v1_seen_before_registration"]}.items():
        if sha256_file(Path(name)) != value:
            raise ValueError(f"v2 registration frozen input changed: {name}")
    return expected
