"""Fresh graph instances, with independent gold and opt-in policy evaluation.

Only development worlds may be built without a prospective confirmation seal.
The graph RNG and nonce-label RNG have separate namespaces; neither reads model
outputs. All changed endpoints exist in the original label vocabulary.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from xgraph.fresh_worlds_gold import adjacency, execute

# Ordinary relation labels with explicit source/target types. Numeric P/Q ids are
# compatibility ids for the KG directory, not assertions about Wikidata.
SCHEMA = (
    ("member of", "person", "team"),
    ("led by", "team", "person"),
    ("works on", "team", "project"),
    ("managed by", "project", "person"),
    ("sponsored by", "project", "organization"),
    ("employer", "person", "organization"),
    ("headquarters", "organization", "city"),
    ("country", "city", "country"),
    ("capital", "country", "city"),
    ("partner", "organization", "organization"),
    ("collaborates with", "team", "team"),
    ("parent organization", "organization", "organization"),
)
TYPES = {"person": 24, "team": 20, "project": 20, "organization": 24, "city": 24, "country": 16}
# Two phrasings per semantic sequence. Shared in development and confirmation.
_TEMPLATES = (
    (
        (1, 2),
        "Who leads the team that {start} is a member of?",
        "Who is the leader of {start}'s team?",
    ),
    (
        (1, 3),
        "Which project does the team that {start} belongs to work on?",
        "What project is worked on by {start}'s team?",
    ),
    (
        (1, 11),
        "Which team does the team that {start} belongs to collaborate with?",
        "What is the collaborating team of {start}'s team?",
    ),
    (
        (6, 7),
        "In which city is {start}'s employer headquartered?",
        "What is the headquarters city of the organization employing {start}?",
    ),
    (
        (3, 5),
        "Which organization sponsors the project that {start} works on?",
        "Who is the sponsor of {start}'s project?",
    ),
    (
        (2, 6),
        "Which organization employs the leader of {start}?",
        "Who is the employer of the person leading {start}?",
    ),
    (
        (7, 8),
        "Which country contains the headquarters of {start}?",
        "In what country is {start}'s headquarters city?",
    ),
    (
        (4, 6),
        "Which organization employs the manager of {start}?",
        "Who is the employer of the person managing {start}?",
    ),
    (
        (1, 3, 5),
        "Which organization sponsors the project of {start}'s team?",
        "Who sponsors the project worked on by the team that {start} belongs to?",
    ),
    (
        (1, 2, 6),
        "Which organization employs the leader of {start}'s team?",
        "Who employs the person leading the team that {start} is a member of?",
    ),
    (
        (6, 7, 8),
        "Which country contains the headquarters of {start}'s employer?",
        "In what country is the headquarters city of the organization employing {start}?",
    ),
    (
        (3, 5, 7),
        "In which city is the sponsor of {start}'s project headquartered?",
        "What is the headquarters city of the organization sponsoring the project of {start}?",
    ),
    (
        (2, 6, 7),
        "In which city is the employer of {start}'s leader headquartered?",
        "What is the headquarters city of the organization employing the person leading {start}?",
    ),
    (
        (1, 3, 5, 7),
        "In which city is the sponsor of {start}'s team's project headquartered?",
        "What is the headquarters city of the organization sponsoring "
        "the project of {start}'s team?",
    ),
    (
        (1, 2, 6, 7),
        "In which city is the employer of the leader of {start}'s team headquartered?",
        "What is the headquarters city of the organization employing the leader of {start}'s team?",
    ),
    (
        (3, 5, 7, 8),
        "Which country contains the headquarters of the sponsor of {start}'s project?",
        "In what country is the headquarters city of the organization "
        "sponsoring {start}'s project?",
    ),
)
TEMPLATES = tuple(
    {"id": f"t{i:02d}-{j}", "relations": list(relations), "question": text}
    for i, (relations, *texts) in enumerate(_TEMPLATES)
    for j, text in enumerate(texts)
)
CONDITIONS = ("original", "inner-hop-edit", "off-chain-edit")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _rng(*parts):
    return random.Random(int(digest(parts), 16))


@dataclass(frozen=True)
class WorldConfig:
    seed: int
    worlds: int = 64
    split: str = "development"
    pairs: int = 200
    multivalued_rate: float = 0.25


def _world(config, number, template, *, demo=False):
    name = f"demo-{number}" if demo else f"{config.split}-{number:04d}"
    rng = _rng(
        "topology-v1", 871 if demo else config.seed, "demo" if demo else config.split, number
    )
    offset = (number if demo else number + 5) * 128 + 1
    ids = list(range(offset, offset + 128))
    rng.shuffle(ids)
    entities, pools, at = {}, {}, 0
    for kind, count in TYPES.items():
        pools[kind] = ids[at : at + count]
        entities.update((item, kind) for item in pools[kind])
        at += count
    used = set()

    def node(kind):
        candidates = [item for item in pools[kind] if item not in used]
        item = rng.choice(candidates)
        used.add(item)
        return item

    relations = template["relations"]
    chain_types = [SCHEMA[relations[0] - 1][1], *(SCHEMA[p - 1][2] for p in relations)]
    assert all(SCHEMA[p - 1][1] == chain_types[i] for i, p in enumerate(relations))
    chain = [node(kind) for kind in chain_types]
    # Change an intermediate reached node, never just the final answer edge.
    edit_hop = rng.randrange(len(relations) - 1)
    alternate = chain[: edit_hop + 1] + [node(kind) for kind in chain_types[edit_hop + 1 :]]
    edges = set(zip(chain[:-1], relations, chain[1:], strict=True))
    edges.update(
        zip(
            alternate[edit_hop + 1 : -1],
            relations[edit_hop + 1 :],
            alternate[edit_hop + 2 :],
            strict=True,
        )
    )
    protected = {(s, p) for s, p, _ in edges} | {(alternate[edit_hop], relations[edit_hop])}
    # Some reachable internal pairs branch and reconverge. The final denotation
    # stays singleton even though an intermediate relation has multiple values.
    if rng.random() < config.multivalued_rate:
        hop = rng.choice([h for h in range(len(relations) - 1) if h != edit_hop - 1])
        branch = node(chain_types[hop + 1])
        edges.add((chain[hop], relations[hop], branch))
        edges.add((branch, relations[hop + 1], chain[hop + 2]))
        protected.add((branch, relations[hop + 1]))
    inner = [chain[edit_hop], relations[edit_hop], alternate[edit_hop + 1]]
    # Same relation, source type, target type and singleton replacement size.
    off_subject = node(chain_types[edit_hop])
    off_old, off_new = node(chain_types[edit_hop + 1]), node(chain_types[edit_hop + 1])
    edges.add((off_subject, relations[edit_hop], off_old))
    if sum(s == chain[edit_hop] and p == relations[edit_hop] for s, p, _ in edges) > 1:
        edges.add((off_subject, relations[edit_hop], node(chain_types[edit_hop + 1])))
    protected.add((off_subject, relations[edit_hop]))
    off = [off_subject, relations[edit_hop], off_new]
    # Connect every node into the bounded component, using unprotected pairs.
    # Reserving all gold pairs prevents filler from changing either denotation.
    pairs = {(s, p): {o} for s, p, o in edges}
    for s, p, o in edges:
        pairs[s, p].add(o)
    candidates = [
        (s, p)
        for p, (_, source, _) in enumerate(SCHEMA, 1)
        for s in pools[source]
        if (s, p) not in protected
    ]
    rng.shuffle(candidates)
    # First join weak components, including isolated labelled nodes. Edges still
    # respect the ordinary relation schema and never add to a protected pair.
    parent = {item: item for item in ids}

    def root(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for s, _, o in edges:
        parent[root(s)] = root(o)
    while len(pairs) < config.pairs:
        connecting = len({root(item) for item in ids}) > 1
        for at in range(len(candidates) - 1, -1, -1):
            s, p = candidates[at]
            choices = [
                o
                for o in pools[SCHEMA[p - 1][2]]
                if o != s and (not connecting or root(o) != root(s))
            ]
            if choices:
                candidates.pop(at)
                break
        else:
            raise ValueError("cannot connect typed component with unprotected pairs")
        o = rng.choice(choices)
        pairs[s, p] = {o}
        parent[root(s)] = root(o)
    wanted = round(config.pairs * config.multivalued_rate)
    many = sum(len(objects) > 1 for objects in pairs.values())
    eligible = [
        pair for pair, objects in pairs.items() if pair not in protected and len(objects) == 1
    ]
    rng.shuffle(eligible)
    if wanted < many or wanted - many > len(eligible):
        raise ValueError("multivalued rate cannot be met with the protected gold pairs")
    for s, p in eligible[: wanted - many]:
        choices = [o for o in pools[SCHEMA[p - 1][2]] if o not in pairs[s, p] and o != s]
        pairs[s, p].add(rng.choice(choices))
    triples = sorted((s, p, o) for (s, p), objects in pairs.items() for o in objects)
    conditions = []
    for condition, edits in zip(CONDITIONS, ([], [inner], [off]), strict=True):
        gold = execute(triples, chain[0], relations, edits)
        if len(gold["answers"]) != 1 or max(map(len, gold["frontiers"])) > 64:
            raise ValueError("generator invariant: singleton answer or frontier admission")
        conditions.append(
            {
                "name": condition,
                "edits": edits,
                "gold": gold["answers"],
                "max_frontier": max(map(len, gold["frontiers"])),
            }
        )
    if (
        conditions[0]["gold"] == conditions[1]["gold"]
        or conditions[0]["gold"] != conditions[2]["gold"]
    ):
        raise ValueError("generator invariant: edit sensitivity/invariance")
    return {
        "id": name,
        "template": template["id"],
        "question_template": template["question"],
        "relations": relations,
        "start": chain[0],
        "entities": dict(sorted(entities.items())),
        "bounds": [offset, offset + 128],
        "triples": triples,
        "conditions": conditions,
        "edit_hop": edit_hop,
        "multivalued_pairs": wanted,
        "topology_sha256": digest([(s - offset, p, o - offset) for s, p, o in triples]),
    }


def generate_worlds(config, *, confirmation_seal=None):
    if getattr(config, "design", None) == "mechanism-v2-hard":
        from xgraph.mechanism import generate_fresh_worlds

        return generate_fresh_worlds(config, confirmation_seal=confirmation_seal)
    if getattr(config, "design", None) == "controlled-v2":
        from xgraph.fresh_worlds_v2 import generate

        return generate(config, confirmation_seal=confirmation_seal)
    if (
        type(config.seed) is not int
        or type(config.worlds) is not int
        or config.worlds < 1
        or config.split not in ("development", "confirmation")
        or not 160 <= config.pairs <= 240
        or not 0 <= config.multivalued_rate <= 0.5
    ):
        raise ValueError("invalid world count, seed, split, pair count or multivalued rate")
    if config.split == "confirmation":
        _confirm_admission(confirmation_seal, config)
    order = list(range(len(TEMPLATES)))
    _rng("templates-v1", config.seed, config.split).shuffle(order)
    worlds = [_world(config, n, TEMPLATES[order[n % len(order)]]) for n in range(config.worlds)]
    if len({w["topology_sha256"] for w in worlds}) != len(worlds):
        raise ValueError("duplicate topology")
    demos = [_world(config, n, TEMPLATES[k], demo=True) for n, k in enumerate((0, 6, 16, 20, 26))]
    return {
        "format": 1,
        "config": asdict(config),
        "schema": SCHEMA,
        "templates": TEMPLATES,
        "worlds": worlds,
        "demonstrations": demos,
    }


def _normal(text):
    return unicodedata.normalize("NFKC", text).strip().casefold()


def nonce_labels(count, seed, namespace, tokenizers, collision_files, common_files):
    """Screen a bounded candidate pool while streaming the external KG label set.

    Web-common screening is corpus-relative, not proof of absence on the web.
    Labels must be at least four tokens in BOTH served vocabularies, round-trip
    byte-exactly in the label context, and equal no complete vocabulary piece.
    """
    from xgraph.tokenizer import ServedTokenizer

    served = {name: ServedTokenizer.from_file(path) for name, path in tokenizers.items()}
    rng = _rng("labels-v1", seed, namespace)
    candidates = []
    seen = set()
    for _ in range(count * 3 + 1024):
        label = "".join(
            rng.choice("bcdfghjklmnprstvz") + rng.choice("aeiou") for _ in range(7)
        ).capitalize()
        if _normal(label) not in seen:
            candidates.append(label)
            seen.add(_normal(label))
    banned = set()
    source_counts = {}
    for path in [*collision_files, *common_files]:
        n = 0
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                text = json.loads(line)[1] if Path(path).suffix == ".jsonl" else line.strip()
                if not isinstance(text, str):
                    raise ValueError(f"non-string collision label in {path}")
                key = _normal(text)
                if key in seen:
                    banned.add(key)
                n += 1
        source_counts[str(path)] = n
    pieces = set()
    for tokenizer in served.values():
        for token in range(tokenizer.vocab_size):
            raw = tokenizer.token_bytes(token)
            if raw:
                pieces.add(_normal(raw.decode("utf-8", errors="replace")))
    accepted, lengths = [], {name: [] for name in served}
    rejected = Counter()
    for label in candidates:
        if _normal(label) in banned or _normal(label) in pieces:
            rejected["corpus_or_vocabulary_collision"] += 1
            continue
        encoded = {name: tokenizer.encode(" " + label) for name, tokenizer in served.items()}
        if any(
            not 4 <= len(ids) <= 20
            or any(i in served[name].special_ids for i in ids)
            or b"".join(served[name].token_bytes(i) or b"" for i in ids) != (" " + label).encode()
            for name, ids in encoded.items()
        ):
            rejected["tokenization"] += 1
            continue
        accepted.append(label)
        for name, ids in encoded.items():
            lengths[name].append(len(ids))
        if len(accepted) == count:
            break
    if len(accepted) != count:
        raise ValueError("nonce candidate pool exhausted; freeze a new generator before retrying")
    audit = {
        "screen": "normalized exact corpus and whole vocabulary piece; >=4 context tokens",
        "web_absence_proven": False,
        "sources_rows": source_counts,
        "rejected": dict(rejected),
        "tokenizers": {
            name: {
                "fingerprint": tok.fingerprint,
                "min_tokens": min(lengths[name]),
                "max_tokens": max(lengths[name]),
            }
            for name, tok in served.items()
        },
    }
    return accepted, audit


def schema_text(schema=SCHEMA):
    return (
        "KG relation schema: directed triples from subject to object; follow all values; "
        "no implied inverse relations. "
    ) + "; ".join(f"{label}: {source} to {target}" for label, source, target in schema)


def _demo_response(world, labels, *, observed=False, observations_only=False, schema=SCHEMA):
    trace = execute(world["triples"], world["start"], world["relations"])["frontiers"]
    graph = adjacency(world["triples"])
    text = "⟦ " + labels[str(world["start"])]
    for hop, frontier in enumerate(trace):
        if observed:
            displayed = (
                frontier[:3]
                if observations_only or ("cell" in world and len(frontier) > 8)
                else frontier
            )
            text += " ·→ " + " ; ".join(labels[str(item)] for item in displayed)
            if len(displayed) < len(frontier):
                text += f" (+{len(frontier) - len(displayed)} more)"
        if observed and not observations_only:
            menu = sorted(
                {schema[p - 1][0] for (s, p), objects in graph.items() if s in frontier and objects}
            )
            text += " ⟦? " + (" ; ".join(menu[:5]) or "∅")
            if len(menu) > 5:
                text += f" (+{len(menu) - 5} more)"
            text += "⟧"
            if len(frontier) > 1:
                selected_id = frontier[0]
                if "cell" in world:
                    remaining = world["relations"][hop:]
                    selected_id = next(
                        item
                        for item in frontier
                        if execute(world["triples"], item, remaining)["answers"] == [trace[-1][0]]
                    )
                    trace[hop + 1 :] = execute(world["triples"], selected_id, remaining)[
                        "frontiers"
                    ][1:]
                selected = labels[str(selected_id)]
                text += " ⟦= " + selected + " ·→ " + selected
                selected_menu = sorted(
                    {
                        schema[p - 1][0]
                        for (s, p), objects in graph.items()
                        if s == selected_id and objects
                    }
                )
                text += " ⟦? " + (" ; ".join(selected_menu[:5]) or "∅")
                if len(selected_menu) > 5:
                    text += f" (+{len(selected_menu) - 5} more)"
                text += "⟧"
        if hop < len(world["relations"]):
            text += " · " + schema[world["relations"][hop] - 1][0]
    return text + "⟧\nAnswer: ⟦ " + labels[str(trace[-1][0])] + "⟧"


def requests_for(study, kind="paths"):
    """Gold-free policy-arm hook: anchors, schema, demos and component evidence.

    New policies consume these requests or provide the same record IDs to score.
    The complete reader receives every versioned triple in its numeric bounds.
    """
    if kind not in ("paths", "wt", "agent", "plan", "reader", "factorial"):
        raise ValueError(f"unknown request kind {kind}")
    labels = study["labels"]
    schema = study["schema"]
    instructions = (
        "Answer using the supplied KG, whose entity labels are unique. Write a relation-only "
        "path as ⟦ start label · relation · relation⟧, without intermediate entity names. "
        "End with Answer: ⟦ label⟧. " + schema_text(schema)
    )
    if kind == "plan":
        instructions += "\nWrite only the relation path. An exact executor will deliver its answer."
    if kind == "reader":
        instructions += (
            "\nEvery triple of the bounded question component is supplied; use all of it as needed."
        )
    demos = []
    for demo in study["demonstrations"]:
        question = demo["question_template"].format(start=labels[str(demo["start"])])
        response = _demo_response(
            demo,
            labels,
            observed=kind in ("wt", "factorial"),
            observations_only=kind == "factorial",
            schema=schema,
        )
        if kind == "plan":
            response = response.split("\nAnswer:")[0]
        demos += [{"role": "user", "content": question}, {"role": "assistant", "content": response}]
    rows = []
    for world in study["worlds"]:
        question = world["question_template"].format(start=labels[str(world["start"])])
        inventory = ""
        if study["config"].get("design") in ("controlled-v2", "mechanism-v2-hard"):
            available = sorted({p for _, p, _ in world["triples"]})
            inventory = (
                "\nCurrent component relation inventory: "
                + "; ".join(schema[p - 1][0] for p in available)
                + ". Use only this inventory for the current question; "
                "demonstration components have their own inventories."
            )
        # Agents rebuild their system messages. Preserve their schema in the
        # user turn; other interfaces keep it in the system turn so it cannot
        # contaminate the question's lexical relation-menu scores.
        asked = question + ("\n" + schema_text(schema) + inventory if kind == "agent" else "")
        for condition in world["conditions"]:
            evidence = ""
            if kind == "reader":
                graph = adjacency(world["triples"], condition["edits"])
                evidence = "\nComponent triples (complete):\n" + "\n".join(
                    f"({labels[str(s)]}, {schema[p - 1][0]}, {labels[str(o)]})"
                    for (s, p), objects in sorted(graph.items())
                    for o in sorted(objects)
                )
            rows.append(
                {
                    "id": condition["record_id"],
                    "world_id": world["id"],
                    "question": question,
                    "anchor": [f"Q{world['start']}"],
                    "kg_version": condition["version"],
                    "messages": [
                        {"role": "system", "content": instructions + inventory},
                        *demos,
                        {"role": "user", "content": asked + evidence},
                    ],
                }
            )
    return rows


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def generator_sources(study):
    names = ("fresh_worlds.py", "fresh_worlds_gold.py")
    if study["config"].get("design") == "mechanism-v2-hard":
        return (*names, "fresh_worlds_v2.py", "mechanism.py")
    return (
        (*names, "fresh_worlds_v2.py")
        if study["config"].get("design") == "controlled-v2"
        else names
    )


def write_study(out, config, tokenizers, collision_files, common_files, *, confirmation_seal=None):
    import numpy as np

    from xgraph.kgversions import write_kg_versions
    from xgraph.provenance import provenance, sha256_file
    from xgraph.syntax import SpanSyntax
    from xgraph.tokenizer import ServedTokenizer
    from xgraph.wikidata import build_index

    out = Path(out)
    if out.exists():
        raise ValueError(f"study already exists: {out}")
    if set(tokenizers) != {"qwen38", "gemma31"} or not collision_files or not common_files:
        raise ValueError(
            "both served tokenizers, KG label collision files "
            "and a common-string corpus are required"
        )
    study = generate_worlds(config, confirmation_seal=confirmation_seal)
    all_worlds = [*study["demonstrations"], *study["worlds"]]
    items = sorted(item for world in all_worlds for item in world["entities"])
    demo_names, demo_audit = nonce_labels(
        640, 871, "demonstrations", tokenizers, collision_files, common_files
    )
    prior_labels = []
    if confirmation_seal:
        prior_labels = [
            Path(json.loads(Path(confirmation_seal).read_text())["development"])
            / "kg/entity_labels.jsonl"
        ]
    world_names, audit = nonce_labels(
        len(items) - 640,
        config.seed,
        config.split,
        tokenizers,
        [*collision_files, *prior_labels],
        common_files,
    )
    names = demo_names + world_names
    if len(set(names)) != len(names):
        raise ValueError("nonce labels overlap the fixed demonstration vocabulary")
    study["labels"] = {str(item): name for item, name in zip(items, names, strict=True)}
    study["label_audit"] = audit
    study["demonstration_label_audit"] = demo_audit
    study["tokenizer_files"] = {
        family: {"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path))}
        for family, path in tokenizers.items()
    }
    files = [
        *map(Path, tokenizers.values()),
        *map(Path, collision_files),
        *map(Path, common_files),
        *prior_labels,
    ]
    study["provenance"] = provenance(files)
    study["collision_files"] = [str(Path(p).resolve()) for p in collision_files]
    study["common_files"] = [str(Path(p).resolve()) for p in common_files]
    study["source_sha256"] = {
        name: sha256_file(Path(__file__).with_name(name)) for name in generator_sources(study)
    }
    out.mkdir(parents=True)
    write_json(out / "tokenizers.json", study["tokenizer_files"])
    kg = out / "kg"
    kg.mkdir()
    write_jsonl(
        kg / "entity_labels.jsonl",
        [(f"Q{item}", study["labels"][str(item)], "label") for item in items],
    )
    write_jsonl(
        kg / "relation_labels.jsonl",
        [(f"P{p}", row[0], "label") for p, row in enumerate(study["schema"], 1)],
    )
    write_jsonl(kg / "sitelinks.jsonl", [])
    np.save(
        kg / "triples.npy",
        np.asarray([edge for world in all_worlds for edge in world["triples"]], dtype=np.int32),
    )
    np.save(kg / "sitelink_counts.npy", np.asarray([(item, 0) for item in items], dtype=np.int64))
    versions, components = {}, {}
    for world in study["worlds"]:
        for condition in world["conditions"]:
            version = world["id"] + "-" + condition["name"]
            condition.update(version=version, record_id=version)
            versions[version] = np.asarray(condition["edits"], dtype=np.int32).reshape(-1, 3)
            components[version] = world["bounds"]
    write_kg_versions(out / "kg-versions", versions, components=components)
    for family, path in tokenizers.items():
        build_index(kg, ServedTokenizer.from_file(path), SpanSyntax(), out / "indexes" / family)
    from xgraph.toolkg import build_store

    build_store(kg, out / "kg.sqlite")
    for kind in ("paths", "wt", "agent", "plan", "reader"):
        write_jsonl(out / f"requests-{kind}.jsonl", requests_for(study, kind))
    write_json(
        out / "policy-arm.json",
        {
            "format": 1,
            "requests": str((out / "requests-paths.jsonl").resolve()),
            "request_variants": {
                kind: str((out / f"requests-{kind}.jsonl").resolve())
                for kind in ("paths", "wt", "agent", "plan", "reader")
            },
            "kg": str(kg.resolve()),
            "store": str((out / "kg.sqlite").resolve()),
            "kg_versions": str((out / "kg-versions").resolve()),
            "indexes": {f: str((out / "indexes" / f).resolve()) for f in tokenizers},
            "tokenizers": study["tokenizer_files"],
            "output_contract": "one record per request id; final Answer span "
            "or explicit answer_id Q-number",
            "score_command": "python -I -m xgraph.fresh_worlds score "
            "--study STUDY --records OUTPUT --out SCORE",
        },
    )
    write_json(out / "study.json", study)
    # Canonical JSON types at the public disk boundary, including numeric map keys.
    return json.loads((out / "study.json").read_text())


def score_records(study, records):
    from xgraph.score import record_failure, spans

    expected = {c["record_id"] for w in study["worlds"] for c in w["conditions"]}
    by_id = {}
    for row in records:
        if row["id"] in by_id or row["id"] not in expected:
            raise ValueError(f"duplicate or unknown record id: {row['id']}")
        by_id[row["id"]] = row
    names = {_normal(label): int(item) for item, label in study["labels"].items()}
    if len(names) != len(study["labels"]):
        raise ValueError("ambiguous labels")
    per_world, per_condition, failures = [], Counter(), Counter()
    edit_changes = off_invariant = 0
    for world in study["worlds"]:
        answers = []
        for condition in world["conditions"]:
            row = by_id.get(condition["record_id"], {})
            answer = None
            status = "missing" if not row else "unsupported_answer"
            if record_failure(row, require_text=False):
                if row.get("truncated") or row.get("finish_reason") in ("length", "max_tokens"):
                    status = "truncated"
                elif row.get("escaped") or (
                    isinstance(row.get("text"), str)
                    and any(kind == "escaped" for kind, _ in spans(row["text"]))
                ):
                    status = "escaped_answer"
                else:
                    status = "generation_error"
            elif row:
                if "answer_id" in row:
                    value = row["answer_id"]
                    if isinstance(value, str) and re.fullmatch(r"Q[0-9]+", value):
                        answer = int(value[1:])
                else:
                    match = re.search(
                        r"(?:^|\n)Answer:[ \t]*⟦[ \t]*([^⟦⟧\n]+?)\s*⟧\s*$",
                        row.get("text", ""),
                        re.I,
                    )
                    if match:
                        answer = names.get(_normal(match[1]))
            if answer is not None and answer not in map(int, world["entities"]):
                status = "out_of_component"
                answer = None
            correct = answer is not None and [answer] == condition["gold"]
            if answer is not None:
                status = "correct" if correct else "wrong_id"
            if not correct:
                failures[status] += 1
            per_condition[condition["name"]] += correct
            answers.append(
                {
                    "condition": condition["name"],
                    "id": condition["record_id"],
                    "answer_id": f"Q{answer}" if answer is not None else None,
                    "correct": correct,
                    "status": status,
                }
            )
        original, inner, off = (a["answer_id"] for a in answers)
        # Missing or invalid predictions stay in the world denominator.
        edit_changes += original is not None and inner is not None and original != inner
        off_invariant += original is not None and off is not None and original == off
        per_world.append(
            {
                "id": world["id"],
                "template": world["template"],
                "joint": all(a["correct"] for a in answers),
                "conditions": answers,
            }
        )
    n = len(per_world)
    if not n:
        raise ValueError("scoring requires at least one world")
    return {
        "worlds": n,
        "joint_version_fidelity": sum(w["joint"] for w in per_world) / n,
        "condition_accuracy": {c: per_condition[c] / n for c in CONDITIONS},
        "edit_sensitivity": edit_changes / n,
        "off_chain_invariance": off_invariant / n,
        "failure_counts": dict(failures),
        "missing": len(expected - set(by_id)),
        "per_world": per_world,
    }


def _confirm_admission(path, config):
    from xgraph.provenance import sha256_file

    if path is None:
        raise ValueError("confirmation generation requires a post-freeze seal")
    seal = json.loads(Path(path).read_text())
    now = datetime.now(UTC)
    created = datetime.fromisoformat(seal["created_at"])
    if (
        not datetime(2026, 11, 22, 23, tzinfo=UTC)
        <= created
        < datetime(2026, 11, 23, 23, tzinfo=UTC)
        or seal["not_before"] != "2026-11-24T00:00:00+00:00"
        or not seal["files"]
    ):
        raise ValueError("invalid Nov 23 confirmation freeze")
    if now < datetime.fromisoformat(seal["not_before"]):
        raise ValueError("confirmation worlds may be generated only AFTER the Nov 23 freeze")
    if (
        config.seed != seal["seed"]
        or config.worlds != (384 if getattr(config, "design", None) == "controlled-v2" else 256)
        or config.split != "confirmation"
        or asdict(config) != seal["config"]
    ):
        raise ValueError("confirmation seed, split or size differs from freeze")
    for name, expected in seal["files"].items():
        if sha256_file(Path(name)) != expected:
            raise ValueError(f"confirmation frozen input changed: {name}")
    return seal


def freeze_confirmation(development, assurance_path, models, seed, out):
    from xgraph.fresh_worlds_queue import FAMILIES, _commit
    from xgraph.provenance import provenance, sha256_file

    now = datetime.now(UTC)
    # Berlin's Nov 23 ends at 23:00 UTC. Use the following UTC day as a
    # conservative, timezone-independent generation boundary.
    if not datetime(2026, 11, 22, 23, tzinfo=UTC) <= now < datetime(2026, 11, 23, 23, tzinfo=UTC):
        raise ValueError("fix the confirmation seed at the Nov 23 freeze, not before or after")
    if Path(out).exists():
        raise ValueError("confirmation seal already exists")
    assurance_report = json.loads(Path(assurance_path).read_text())
    if not assurance_report.get("admitted"):
        raise ValueError("confirmation requires an admitted measured pilot assurance report")
    development, models = Path(development).resolve(), Path(models).resolve()
    study = json.loads((development / "study.json").read_text())
    v2 = study.get("config", {}).get("design") == "controlled-v2"
    if v2:
        from xgraph.fresh_worlds_v2 import validate_registration

        validate_registration(development)
        admitted_at = datetime.fromisoformat(assurance_report["provenance"]["created_at"])
        if (
            admitted_at >= datetime(2026, 11, 14, tzinfo=UTC)
            or assurance_report.get("confirmation_worlds") != 384
            or assurance_report.get("trials", 0) < 2000
            or assurance_report.get("nested_resamples", 0) < 2000
            or not assurance_report.get("observed_leave_one_template_out_positive")
            or not 0 < assurance_report.get("pilot_gpu_seconds", 0) <= 7200
            or assurance_report.get("projected_gpu_seconds", 36001) > 36000
            or {s["prior_mass_per_template"] for s in assurance_report.get("scenarios", [])}
            != {0.5, 2.0}
            or min(s["mc_lower_95"] for s in assurance_report["scenarios"]) < 0.8
        ):
            raise ValueError("v2 admission must pass joint assurance and cost by Nov 13")
    pilot_inputs = assurance_report.get("provenance", {}).get("inputs", [])
    pilot_hashes = {str(Path(p["path"]).resolve()): p["sha256"] for p in pilot_inputs}
    if assurance_report.get("status") != "pilot_projection" or pilot_hashes.get(
        str(development / "study.json")
    ) != sha256_file(development / "study.json"):
        raise ValueError(
            "pilot assurance must bind this development study and its measured outputs"
        )
    for name, expected in pilot_hashes.items():
        if sha256_file(Path(name)) != expected:
            raise ValueError(f"pilot assurance input changed: {name}")
    repo = Path(__file__).resolve().parents[2]
    _commit(repo, clean=True)
    paths = [Path(p) for p in subprocess_files(repo)]
    paths += [p for p in development.rglob("*") if p.is_file()]
    paths += [Path(p["path"]) for p in study["provenance"]["inputs"]]
    paths.append(Path(assurance_path).resolve())
    paths += list(map(Path, pilot_hashes))
    for model in FAMILIES.values():
        paths += [
            p for p in (models / model).iterdir() if p.suffix in (".json", ".jinja", ".safetensors")
        ]
    files = {str(p.resolve()): sha256_file(p) for p in paths}
    seal = {
        "format": 1,
        "seed": seed,
        "created_at": now.isoformat(),
        "not_before": "2026-11-24T00:00:00+00:00",
        "development": str(development),
        "files": files,
        "config": {
            **study["config"],
            "seed": seed,
            "worlds": 384 if v2 else 256,
            "split": "confirmation",
        },
        "tokenizers": study["tokenizer_files"],
        "collision_files": study["collision_files"],
        "common_files": study["common_files"],
        "provenance": provenance([Path(assurance_path)]),
    }
    write_json(out, seal)
    return seal


def subprocess_files(repo):
    import subprocess

    return [
        str(repo / path)
        for path in subprocess.check_output(
            ["git", "ls-files", "src", "scripts", "docs", "GLOSSARY.md", "uv.lock"],
            cwd=repo,
            text=True,
        ).splitlines()
    ]


def analyze(study_dir, outputs_dir, out, budget=None, *, trials=2000):
    from xgraph.fresh_worlds_analysis import assurance, comparisons
    from xgraph.provenance import provenance

    study = json.loads((study_dir / "study.json").read_text())
    v2 = study["config"].get("design") == "controlled-v2"
    if v2:
        from xgraph.fresh_worlds_v2_analysis import assurance, comparisons

    scores, files = {}, [study_dir / "study.json"]
    for family in ("qwen38", "gemma31"):
        for arm in ("WT", "agent", "plan-exact") if v2 else ("WT", "agent", "plan-exact", "reader"):
            path = outputs_dir / family / f"{arm}.jsonl"
            if path.exists():
                scores[f"{family}:{arm}"] = score_records(study, read_jsonl(path))
                files.append(path)
    complete = len(scores) == (6 if v2 else 8) and all(s["missing"] == 0 for s in scores.values())
    gpu_seconds = None
    if budget and Path(budget).exists():
        ledger = json.loads(Path(budget).read_text())
        if any(a["status"] == "running" for a in ledger["attempts"]):
            complete = False
        gpu_seconds = ledger["used_seconds"]
        files.append(Path(budget))
    if v2 and (study_dir / "registration.json").exists():
        files.append(study_dir / "registration.json")
    result = {
        "status": "completed" if complete else "INCOMPLETE",
        "scores": scores,
        "comparisons": (comparisons(study, scores) if v2 else comparisons(scores))
        if complete
        else None,
        "assurance": (
            {"status": "confirmation_analysis", "admitted": False}
            if v2 and study["config"]["split"] == "confirmation"
            else assurance(
                *([study, scores if complete else {}] if v2 else [scores if complete else {}]),
                pilot_gpu_seconds=gpu_seconds,
                trials=trials,
            )
        ),
        "provenance": provenance(files),
    }
    result["assurance"]["provenance"] = result["provenance"]
    write_json(out, result)
    write_json(Path(out).with_name("assurance.json"), result["assurance"])
    return result


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--seed", type=int, required=True)
    build.add_argument("--worlds", type=int)
    build.add_argument(
        "--design", choices=("v1", "controlled-v2", "mechanism-v2-hard"), default="v1"
    )
    build.add_argument("--pairs", type=int)
    build.add_argument("--multivalued-rate", type=float)
    build.add_argument("--qwen-tokenizer", type=Path, required=True)
    build.add_argument("--gemma-tokenizer", type=Path, required=True)
    build.add_argument("--collision-labels", type=Path, action="append", required=True)
    build.add_argument("--common-strings", type=Path, action="append", required=True)
    score = sub.add_parser("score")
    score.add_argument("--study", type=Path, required=True)
    score.add_argument("--records", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--study", type=Path, required=True)
    analysis.add_argument("--outputs-dir", type=Path, required=True)
    analysis.add_argument("--budget", type=Path)
    analysis.add_argument("--out", type=Path, required=True)
    analysis.add_argument("--trials", type=int, default=2000)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--development", type=Path, required=True)
    freeze.add_argument("--assurance", type=Path, required=True)
    freeze.add_argument("--models", type=Path, required=True)
    freeze.add_argument("--seed", type=int, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    confirmation = sub.add_parser("confirmation")
    confirmation.add_argument("--seal", type=Path, required=True)
    confirmation.add_argument("--out", type=Path, required=True)
    registration = sub.add_parser("register-v2")
    registration.add_argument("--study", type=Path, required=True)
    registration.add_argument("--audit", type=Path, required=True)
    registration.add_argument("--v1-archive", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        config_type = WorldConfig
        if args.design == "controlled-v2":
            from xgraph.fresh_worlds_v2 import V2Config

            config_type = V2Config
        elif args.design == "mechanism-v2-hard":
            from xgraph.mechanism import FreshConfig

            config_type = FreshConfig
        parameters = {
            name: value
            for name, value in (("worlds", args.worlds), ("pairs", args.pairs))
            if value is not None
        }
        if args.multivalued_rate is not None:
            if args.design != "v1":
                parser.error(
                    "v2 fixes reachable widths and has no global multivalued-rate parameter"
                )
            parameters["multivalued_rate"] = args.multivalued_rate
        study = write_study(
            args.out,
            config_type(args.seed, **parameters),
            {"qwen38": args.qwen_tokenizer, "gemma31": args.gemma_tokenizer},
            args.collision_labels,
            args.common_strings,
        )
        print(json.dumps({"worlds": len(study["worlds"]), "label_audit": study["label_audit"]}))
    elif args.command == "score":
        from xgraph.provenance import provenance

        study = json.loads((args.study / "study.json").read_text())
        write_json(
            args.out,
            {
                **score_records(study, read_jsonl(args.records)),
                "provenance": provenance([args.study / "study.json", args.records]),
            },
        )
    elif args.command == "analyze":
        analyze(args.study, args.outputs_dir, args.out, args.budget, trials=args.trials)
    elif args.command == "freeze":
        freeze_confirmation(args.development, args.assurance, args.models, args.seed, args.out)
    elif args.command == "register-v2":
        from xgraph.fresh_worlds_v2 import register

        print(json.dumps(register(args.study, args.audit, args.v1_archive)))
    else:
        seal = json.loads(args.seal.read_text())
        config_type = WorldConfig
        if seal["config"].get("design") == "controlled-v2":
            from xgraph.fresh_worlds_v2 import V2Config

            config_type = V2Config
        config = config_type(**seal["config"])
        _confirm_admission(args.seal, config)
        write_study(
            args.out,
            config,
            {f: Path(p["path"]) for f, p in seal["tokenizers"].items()},
            list(map(Path, seal["collision_files"])),
            list(map(Path, seal["common_files"])),
            confirmation_seal=args.seal,
        )


if __name__ == "__main__":
    main()
