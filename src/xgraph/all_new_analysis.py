"""Read-only evidence for MINTQA start-entity discordances; never edits a policy."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import resource
import shlex
import sqlite3
import sys
import unicodedata
from collections import defaultdict
from functools import cache
from pathlib import Path


def normalized(text):
    return unicodedata.normalize("NFC", text).casefold().strip()


def gold_identity(connection, gold, relations):
    """Recover QIDs only when every released hop exists in the read-only KG."""
    triples = gold["triple"]
    subjects = [named(connection, t[0]) for t in triples]
    objects = [named(connection, t[2]) for t in triples]

    @cache
    def walk(entity, hop):
        if hop == len(triples):
            return (entity,)
        if entity not in subjects[hop]:
            return ()
        props = sorted(relations.get(normalized(triples[hop][1]), ()))
        if not props:
            return ()
        rows = connection.execute(
            "SELECT DISTINCT o FROM triples WHERE s=? "
            f"AND p IN ({','.join('?' * len(props))}) ORDER BY o",
            (entity, *props),
        )
        for (other,) in rows:
            if other in objects[hop] and (tail := walk(other, hop + 1)):
                return (entity, *tail)
        return ()

    witnesses = {
        f"Q{item}": [f"Q{e}" for e in path]
        for item in sorted(subjects[0])
        if (path := walk(item, 0))
    }
    return {
        "named": [f"Q{e}" for e in sorted(subjects[0])],
        "ids": list(witnesses),
        "witnesses": witnesses,
    }


def named(connection, label):
    if re.fullmatch(r"Q\d+", label):
        return {int(label[1:])}
    return {
        row[0]
        for row in connection.execute("SELECT item FROM names WHERE norm=?", (normalized(label),))
    }


def resolve_wt_start(executor, path):
    """Read the saved path's surviving start rank, including frontier selections.

    The matched executor is the existing CPU path contract. Its private walk is
    read here to expose the start rank without replaying uncapped hub frontiers.
    No execution or masking rule is changed.
    """
    if not path:
        return {"candidates": [], "chosen": [], "endpoints": []}
    start, relations = path
    candidates = executor.starts(start)
    walked = executor._walk(executor._index, start, relations, None, False)
    if walked is None:
        # An entity-only span resolves all matching label rows by start rank.
        result = executor.execute(start, relations)
        return {
            "candidates": list(candidates),
            "chosen": list(result.reached[:1]),
            "endpoints": list(result.reached),
        }
    ranks = walked[0]
    best = min(ranks.values()) if ranks else None
    return {
        "candidates": list(candidates),
        "chosen": [candidates[best]] if best is not None else [],
        "endpoints": [
            executor._index.labels.owner_ids[e] for e, rank in ranks.items() if rank == best
        ],
    }


def written_start(record):
    from xgraph.menus import semantic_text
    from xgraph.observations import strip_observations
    from xgraph.programs import record_path
    from xgraph.syntax import SpanSyntax

    path = record_path(record)
    if path:
        return path, "complete_relation_path"
    text = strip_observations(semantic_text(record.get("text", "")), SpanSyntax())
    first = re.match(r"\s*⟦\s*([^⟦⟧·]+)⟧", text)
    if first and "⁇" not in first[1]:
        return (first[1].strip(), []), "entity_span_without_relation_path"
    return None, "unresolved"


def gold_plan_probe(executor, labels_by_id, gold_ids, relations):
    """A fixed-gold-plan sensitivity; this is not a new inference policy."""
    rows = []
    for label in sorted({label for qid in gold_ids for label in labels_by_id.get(qid, [])}):
        resolved = resolve_wt_start(executor, (label, relations))
        rows.append({"label": label, "owners": sorted(executor.named(label)), **resolved})
    return {
        "rows": rows,
        "any_alias_selects_gold": any(set(r["chosen"]) & set(gold_ids) for r in rows),
    }


def explain_item(item):
    e = item["evidence"]
    mechanism = item["mechanism"]
    descriptions = {
        "label_variant_missing": (
            "The canonical gold label is writable, but the question's case variant is absent. "
            f"The written/question token prefix {e['variant_prefix']['text']!r} excludes every "
            "gold QID; the generated label therefore resolves to a different entity."
        ),
        "homonym_different_qid": (
            "The written name has multiple QID owners. Capped path execution commits to a "
            "different surviving start from every released-chain-compatible gold QID. "
            "This identity difference is hidden by name-only agreement."
        ),
        "template_artifact": (
            "The released start is a United States patent with a generic noun-phrase title. "
            "WT substitutes another entity. No shared unavailable question-spelling branch "
            "was established, so missing aliases alone do not explain the substitution."
        ),
        "other_start_substitution": (
            "WT substitutes another entity despite a writable canonical gold start. "
            "The missing-spelling and shared-label rules do not apply; this remains a "
            "model/entity-selection signature, not an established label-index defect."
        ),
        "same_start": (
            "WT commits to a released-chain-compatible gold QID. Its scorer outcome differs "
            "from the agent after start resolution; inspect the saved relation plan, final "
            "answer and agent attempts rather than attributing the discordance to WT linking."
        ),
    }
    text = descriptions.get(mechanism, f"Evidence is insufficient to resolve {mechanism}.")
    if item["agent_origin"]:
        origin = item["agent_origin"]
        if origin["status"] == "lookup_without_executed_path":
            text += (
                " The agent selected its reported entity for a relation lookup "
                "but executed no path."
            )
        elif not origin["supports_final_answer"]:
            text += (
                " The agent root is its last resolved attempt; no saved/replayed navigation "
                "supports its final answer."
            )
        elif not set(origin["ids"]) & set(e["gold_ids"]):
            text += (
                " The agent's final answer is supported by navigation from another start; "
                "no link to the released gold entity is established."
            )
    return text


def classify_start(evidence):
    """Classify an observed start using explicit KG and label-row evidence."""
    gold = set(evidence["gold_ids"])
    if not evidence["gold_named_ids"]:
        return "absent_from_kg"
    if not gold:
        return "unresolved_gold_identity"
    if not gold.intersection(evidence["gold_index_ids"]):
        return "gold_missing_from_label_index"
    if not evidence["wt_ids"]:
        return "unresolved_wt_start"
    if gold.intersection(evidence["wt_ids"]):
        return "same_start"
    if gold.intersection(evidence["wt_candidate_ids"]):
        return "homonym_different_qid"
    if evidence.get("variant_branch_excludes_gold"):
        return "label_variant_missing"
    if evidence["gold_label"] == evidence["wt_label"]:
        return "homonym_different_qid"
    if evidence.get("patent"):
        return "template_artifact"
    return "other_start_substitution"


def pair_mechanism(item):
    """Account for the losing arm in either discordance direction."""
    if not item["discordant"]:
        return item["mechanism"]
    if item["direction"] == "agent_only":
        return (
            item["mechanism"]
            if item["mechanism"] != "same_start"
            else "other_downstream_or_delivery"
        )
    gold = set(item["evidence"]["gold_ids"])
    agent = item["agent_start"]
    if not agent:
        return "unresolved_agent_start"
    if gold.intersection(e["id"] for e in agent):
        return "other_downstream_or_delivery"
    if any(
        normalized(e["label"] or "") == normalized(item["evidence"]["gold_label"]) for e in agent
    ):
        return "homonym_different_qid"
    return "other_start_substitution"


def cohort_summary(rows, mechanism_key="mechanism"):
    """Weighted, signed decomposition over the complete stratum denominator."""
    total = sum(r["weight"] for r in rows)
    if not rows or total <= 0:
        raise ValueError("cohort must have positive weight")

    def count(predicate):
        selected = [r for r in rows if predicate(r)]
        return {
            "n": len(selected),
            "weight": sum(r["weight"] for r in selected),
            "rate_pct": 100 * sum(r["weight"] for r in selected) / total,
        }

    losses = count(lambda r: not r["wt_correct"] and r["agent_correct"])
    wins = count(lambda r: r["wt_correct"] and not r["agent_correct"])
    decomposition = {}
    for mechanism in sorted({r[mechanism_key] for r in rows}):
        wrong = count(
            lambda r, m=mechanism: (
                r[mechanism_key] == m and not r["wt_correct"] and r["agent_correct"]
            )
        )
        right = count(
            lambda r, m=mechanism: (
                r[mechanism_key] == m and r["wt_correct"] and not r["agent_correct"]
            )
        )
        decomposition[mechanism] = {
            "agent_only_n": wrong["n"],
            "WT_only_n": right["n"],
            "agent_only_weight": wrong["weight"],
            "WT_only_weight": right["weight"],
            "agent_only_share_pct": 100 * wrong["n"] / losses["n"] if losses["n"] else 0,
            "agent_only_weight_share_pct": 100 * wrong["weight"] / losses["weight"]
            if losses["weight"]
            else 0,
            "net_pp": right["rate_pct"] - wrong["rate_pct"],
        }
    return {
        "n": len(rows),
        "clusters": len({r["cluster"] for r in rows}),
        "weight": total,
        "gap_pp": wins["rate_pct"] - losses["rate_pct"],
        "agent_only": losses,
        "WT_only": wins,
        "decomposition": decomposition,
        "old_wrong_start_error": count(lambda r: r["old_wrong_start"]),
        "qid_wrong_start_error": count(
            lambda r: (
                not r["wt_correct"]
                and bool(r["evidence"]["gold_ids"])
                and bool(r["evidence"]["wt_ids"])
                and not set(r["evidence"]["gold_ids"]).intersection(r["evidence"]["wt_ids"])
            )
        ),
        "missing_gold_rows": count(lambda r: not r["evidence"]["gold_index_ids"]),
        "missing_question_variant": count(
            lambda r: (
                r["evidence"]["question_surface"] is not None
                and not r["evidence"]["question_surface_gold_ids"]
            )
        ),
        "variant_branch_error": count(
            lambda r: not r["wt_correct"] and r["mechanism"] == "label_variant_missing"
        ),
        "patent": count(lambda r: r["evidence"]["patent"]),
        "one_hop": count(lambda r: r.get("hops") == 1),
    }


def question_surface(question, label):
    match = re.search(r"(?<!\w)" + re.escape(label) + r"(?!\w)", question, re.IGNORECASE)
    return match[0] if match else None


def variant_branch(executor, surface, wt_label, gold_ids):
    """Whether a shared, non-space token prefix excludes every gold label row."""
    if not surface or set(executor.named(surface)).intersection(gold_ids):
        return {"excludes_gold": False, "tokens": [], "text": ""}
    prefix = executor._syntax.label_prefix
    a = executor._tokenizer.encode(prefix + surface)
    b = executor._tokenizer.encode(prefix + wt_label)
    shared = []
    for first, second in zip(a, b, strict=False):
        if first != second:
            break
        shared.append(first)
    text = b"".join(executor._tokenizer.token_bytes(t) or b"" for t in shared).decode(
        "utf-8", errors="replace"
    )
    if not any(c.isalnum() for c in text):
        return {"excludes_gold": False, "tokens": shared, "text": text}
    labels = executor._index.labels
    lo, hi = labels.prefix_range(shared)
    owned = {labels.owner_ids[int(p)] for p in set(labels.owners[lo:hi].tolist())}
    return {
        "excludes_gold": not owned.intersection(gold_ids),
        "tokens": shared,
        "text": text,
        "row_range": [lo, hi],
    }


def agent_origin(connection, record, executor, tools, answer):
    """Connect explicit navigation starts using saved results or a base-KG replay.

    Qwen did not save tool results. Replayed roots are labelled, never described
    as saved. Every attempt and explicit input remains in the item evidence.
    """
    origins, attempts, supported, searches, lookups = {}, [], [], [], []
    answer_ids = {f"Q{e}" for e in named(connection, answer)} if answer else set()
    for at, action in enumerate(record.get("actions", [])):
        tool, args = action.get("tool"), action.get("arguments", {})
        replayed = tools.call(tool, args)
        if tool == "search":
            searches.append(
                {
                    "action": at,
                    "name": args.get("name"),
                    "result": action.get("result", replayed),
                    "evidence": "saved_tool_result"
                    if "result" in action
                    else "base_kg_replay_unsaved_result",
                }
            )
        if tool == "relations":
            result = action.get("result", replayed)
            if result.get("entity"):
                lookups.append(
                    {
                        "action": at,
                        "id": result["entity"],
                        "label": result["label"],
                        "evidence": "saved_tool_result"
                        if "result" in action
                        else "base_kg_replay_unsaved_result",
                    }
                )
        if tool not in {"follow", "execute_path"}:
            continue
        start = str(args.get("entity" if tool == "follow" else "start", ""))
        props = [args.get("relation", "")] if tool == "follow" else args.get("relations", [])
        explicit = bool(re.fullmatch(r"Q\d+", start))
        if explicit:
            starts = {start}
        elif tool == "follow":
            resolved_entity = tools._entity(start)
            starts = {f"Q{resolved_entity}"} if resolved_entity is not None else set()
        else:
            starts = set(resolve_wt_start(executor, (start, props))["chosen"])
        roots = set().union(*(origins.get(e, {e}) for e in starts))
        saved = action.get("result")
        if saved is not None:
            endpoints = {e["id"] for e in saved.get("entities", [])}
            mode = "saved_tool_result"
        else:
            endpoints = {e["id"] for e in replayed.get("entities", [])}
            mode = "base_kg_replay_unsaved_result"
        for entity in endpoints:
            origins[entity] = origins.get(entity, set()) | roots
        matches = bool(endpoints & answer_ids)
        if matches:
            supported = sorted(roots)
        attempts.append(
            {
                "action": at,
                "tool": tool,
                "input": start,
                "relations": props,
                "resolved_input_ids": sorted(starts),
                "root_ids": sorted(roots),
                "result_ids": sorted(endpoints),
                "saved_matches_replay": saved == replayed if saved is not None else None,
                "saved_result": saved,
                "replayed_result": replayed,
                "evidence": mode,
                "supports_final_answer": matches,
            }
        )
    chosen = supported or next((a["root_ids"] for a in reversed(attempts) if a["root_ids"]), [])
    status = "saved_or_replayed_navigation" if chosen else "no_resolvable_navigation"
    if not chosen and lookups:
        chosen = [lookups[-1]["id"]]
        status = "lookup_without_executed_path"
    return {
        "ids": chosen,
        "supports_final_answer": bool(supported),
        "attempts": attempts,
        "searches": searches,
        "lookups": lookups,
        "status": status,
    }


def run(args):
    from xgraph.agent import KGTools, MatchedPaths
    from xgraph.error_analysis import Inputs, git, source_gold
    from xgraph.index import load_kg_index
    from xgraph.mintqa_bench import answer_contains
    from xgraph.programs import MatchedExecutor
    from xgraph.prompts import final_answer
    from xgraph.score import record_outputs
    from xgraph.tokenizer import ServedTokenizer
    from xgraph.toolkg import KGStore

    inputs = Inputs()
    for relative in (
        "src/xgraph/all_new_analysis.py",
        "scripts/mintqa/all-new-analysis.py",
        "src/xgraph/programs.py",
        "src/xgraph/agent.py",
        "src/xgraph/toolkg.py",
        "src/xgraph/index.py",
        "src/xgraph/tokenizer.py",
        "src/xgraph/observations.py",
        "src/xgraph/menus.py",
        "src/xgraph/error_analysis.py",
        "src/xgraph/mintqa_bench.py",
        "src/xgraph/prompts.py",
        "src/xgraph/score.py",
    ):
        inputs.digest(Path(__file__).resolve().parents[2] / relative)
    prior_provenance = inputs.read_json(args.analysis / "provenance.json")
    rows_path = args.analysis / "per-question.jsonl"
    inputs.digest(rows_path)
    with rows_path.open() as stream:
        rows = [json.loads(line) for line in stream]
    rows = [
        r
        for r in rows
        if r["arm"] in {"WT", "agent"}
        and r["strata"]["knowledge_status"] in {"all_new", "all_old", "mixed_old_new"}
    ]
    members = {r["id"]: {"id": r["id"], "hops": r["hops"]} for r in rows}
    golds = source_gold(args.source_root, list(members.values()), inputs)
    freeze = inputs.read_json(args.sealed_root / "freeze-manifest.json")
    core_blobs = {}
    for relative in (
        "src/xgraph/programs.py",
        "src/xgraph/agent.py",
        "src/xgraph/toolkg.py",
        "src/xgraph/index.py",
        "src/xgraph/tokenizer.py",
        "src/xgraph/observations.py",
        "src/xgraph/menus.py",
        "src/xgraph/mintqa_bench.py",
        "src/xgraph/syntax.py",
        "src/xgraph/score.py",
    ):
        blob = git("rev-parse", f"{freeze['code_commit']}:{relative}")
        if blob != git("hash-object", relative):
            raise ValueError(f"replay code differs from sealed producer: {relative}")
        core_blobs[relative] = blob
    # The prompt builder gained a demonstration option after the seal. Only
    # final_answer is called here; verify that function, not the unrelated builder.
    prompt_path = "src/xgraph/prompts.py"
    inputs.digest(Path(__file__).resolve().parents[2] / prompt_path)
    parser_sources = [
        git("show", f"{freeze['code_commit']}:{prompt_path}"),
        (Path(__file__).resolve().parents[2] / prompt_path).read_text(),
    ]
    parser_trees = [
        next(
            n
            for n in ast.parse(s).body
            if isinstance(n, ast.FunctionDef) and n.name == "final_answer"
        )
        for s in parser_sources
    ]
    parser_dumps = [ast.dump(tree, include_attributes=False) for tree in parser_trees]
    if parser_dumps[0] != parser_dumps[1]:
        raise ValueError("native final-answer parser differs from sealed producer")
    core_blobs[prompt_path + ":final_answer_ast_sha256"] = hashlib.sha256(
        parser_dumps[0].encode()
    ).hexdigest()
    freeze = {**freeze, "replay_core_blobs": core_blobs}
    expected = {e["path"]: e["sha256"] for e in freeze["inputs"]}
    for entry in prior_provenance["inputs"]:
        if entry["path"] in expected and expected[entry["path"]] != entry["sha256"]:
            raise ValueError(f"prior analysis and seal disagree: {entry['path']}")
        expected[entry["path"]] = entry["sha256"]
    requests = Path(freeze["panels"]).parent / "validation/requests.jsonl"
    inputs.digest(requests)
    for line in requests.open():
        request = json.loads(line)
        if request["id"] in members and any(
            request.get(key)
            for key in ("kg_version", "xgraph_kg_version", "anchor", "xgraph_anchor")
        ):
            raise ValueError(f"versioned/anchored request needs separate replay: {request['id']}")
    database = args.source_root / "data/external/mintqa-full/kg.sqlite"
    inputs.digest(database)
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro&immutable=1", uri=True)
    connection.executescript("PRAGMA query_only=ON; PRAGMA mmap_size=0; PRAGMA cache_size=-2048;")
    relations = defaultdict(set)
    relation_file = args.source_root / "data/external/mintqa-full/kg/relation_labels.jsonl"
    inputs.digest(relation_file)
    for qid, label, _kind in map(json.loads, relation_file.open()):
        relations[normalized(label)].add(int(qid[1:]))
    relation_names = defaultdict(list)
    for qid, label, _kind in map(json.loads, relation_file.open()):
        relation_names[int(qid[1:])].append(label)
    store = KGStore(database)
    store._con.executescript("PRAGMA query_only=ON; PRAGMA mmap_size=0; PRAGMA cache_size=-2048;")
    identities = {qid: gold_identity(connection, g, relations) for qid, g in golds.items()}
    patent_ids = named(connection, "United States patent")
    instance = relations["instance of"]

    @cache
    def label(qid):
        row = connection.execute(
            "SELECT label FROM labels WHERE item=?", (int(qid[1:]),)
        ).fetchone()
        return row[0] if row else None

    def entities(ids):
        return [{"id": qid, "label": label(qid)} for qid in ids]

    raw = {}
    for path in sorted({r["source"].rsplit(":", 1)[0] for r in rows}):
        inputs.digest(path)
        wanted = {r["id"] for r in rows if r["source"].rsplit(":", 1)[0] == path}
        with Path(path).open() as stream:
            for n, line in enumerate(stream, 1):
                record = json.loads(line)
                if record["id"] in wanted:
                    raw[(path, record["id"])] = {**record, "_source": f"{path}:{n}"}
    all_items = []
    for family in ("qwen38", "gemma31"):
        directory = args.source_root / "target/mintqa-rematch/indexes" / family
        manifest = inputs.read_json(directory.with_suffix(".manifest.json"))
        for entry in manifest["files"]:
            got = inputs.digest(directory / entry["name"])
            if got["sha256"] != entry["sha256"]:
                raise ValueError(f"index manifest mismatch: {got['path']}")
        tokenizer_path = Path(freeze["families"][family]["model"]["path"]) / "tokenizer.json"
        inputs.digest(tokenizer_path)
        index = load_kg_index(directory)
        tokenizer = ServedTokenizer.from_file(tokenizer_path)
        executor = MatchedExecutor(index, tokenizer, start_rank="sitelinks", unquoted_starts=True)
        # A bounded scan of the label-owner array proves any-row coverage without
        # constructing a multi-million-entry Python inverse map.
        import numpy as np

        wanted_ids = {qid for identity in identities.values() for qid in identity["ids"]}
        positions = {e: p for p, e in enumerate(index.labels.owner_ids) if e in wanted_ids}
        gold_rows = np.flatnonzero(np.isin(index.labels.owners, list(positions.values())))
        present_positions = set(np.unique(index.labels.owners[gold_rows]).tolist())
        present_ids = {e for e, p in positions.items() if p in present_positions}
        labels_by_id = defaultdict(list)
        for row in gold_rows:
            text = b"".join(
                tokenizer.token_bytes(t) or b"" for t in index.labels.row_tokens(row)
            ).decode("utf-8")
            if not text.startswith(index.span_syntax.label_prefix):
                raise ValueError(f"label row lost its tokenization context: {row}")
            labels_by_id[index.labels.owner_id(row)].append(
                text.removeprefix(index.span_syntax.label_prefix)
            )
        agents = {
            (r["panel"], r["id"]): r for r in rows if r["family"] == family and r["arm"] == "agent"
        }
        for wt in (r for r in rows if r["family"] == family and r["arm"] == "WT"):
            qid, panel = wt["id"], wt["panel"]
            agent = agents[(panel, qid)]
            gold = golds[qid]
            identity = identities[qid]
            wt_raw = raw[(wt["source"].rsplit(":", 1)[0], qid)]
            agent_raw = raw[(agent["source"].rsplit(":", 1)[0], qid)]
            if any(record.get("kg_version") for record in (wt_raw, agent_raw)):
                raise ValueError(f"versioned record needs a separate replay: {qid}")
            for inherited, original in ((wt, wt_raw), (agent, agent_raw)):
                scored_text = record_outputs([original], [qid])[qid]
                correct = bool(answer_contains(final_answer(scored_text), gold["answer"]))
                if correct != inherited["correct"]:
                    raise ValueError(f"native correctness differs: {panel}/{family}/{qid}")
            if wt["agent_correct"] != agent["correct"] or wt["weight"] != agent["weight"]:
                raise ValueError(f"unpaired inherited rows: {panel}/{family}/{qid}")
            path, path_attribution = written_start(wt_raw)
            resolved = resolve_wt_start(executor, path)
            gold_label = gold["triple"][0][0]
            wt_label = path[0] if path else ""
            surface = question_surface(gold["question"], gold_label)
            variant = variant_branch(executor, surface, wt_label, identity["ids"])
            patent = any(
                row[0] in patent_ids
                for e in identity["ids"]
                for p in instance
                for row in connection.execute(
                    "SELECT o FROM triples WHERE s=? AND p=?", (int(e[1:]), p)
                )
            )
            evidence = {
                "gold_named_ids": identity["named"],
                "gold_ids": identity["ids"],
                "gold_index_ids": sorted(set(identity["ids"]) & present_ids),
                "gold_canonical_row_ids": sorted(executor.named(gold_label)),
                "gold_label": gold_label,
                "wt_label": wt_label,
                "wt_ids": resolved["chosen"],
                "wt_candidate_ids": resolved["candidates"],
                "question_surface": surface,
                "question_surface_gold_ids": sorted(executor.named(surface) & set(identity["ids"]))
                if surface
                else [],
                "variant_branch_excludes_gold": variant["excludes_gold"],
                "variant_prefix": variant,
                "patent": patent,
                "initial_case_alias_covers_surface": bool(
                    surface
                    and surface != gold_label
                    and surface[1:] == gold_label[1:]
                    and surface[0].casefold() == gold_label[0].casefold()
                ),
            }
            mechanism = classify_start(evidence)
            discordant = wt["correct"] != agent["correct"]
            direction = "WT_only" if wt["correct"] else "agent_only"
            tools = KGTools(
                store,
                relation_names,
                MatchedPaths(executor),
                pages=True,
                feedback=bool(agent_raw.get("agent_feedback")),
            )
            origin = (
                agent_origin(connection, agent_raw, executor, tools, agent["parsed_answer"])
                if discordant or mechanism != "same_start"
                else None
            )
            item = {
                "panel": panel,
                "family": family,
                "id": qid,
                "question": gold["question"],
                "knowledge_status": wt["strata"]["knowledge_status"],
                "hops": wt["hops"],
                "cluster": wt["cluster"],
                "weight": wt["weight"],
                "wt_correct": wt["correct"],
                "agent_correct": agent["correct"],
                "discordant": discordant,
                "direction": direction if discordant else "concordant",
                "mechanism": mechanism,
                "evidence": evidence,
                "gold_start": entities(identity["ids"]),
                "gold_named_candidates": entities(identity["named"]),
                "gold_witnesses": identity["witnesses"],
                "gold_plan": gold["triple"],
                "wt_start": entities(resolved["chosen"]),
                "wt_candidate_starts": entities(resolved["candidates"]),
                "wt_path": wt["diagnostic"]["path"],
                "wt_text": wt_raw["text"],
                "wt_start_attribution": path_attribution,
                "wt_primary": wt["diagnostic"]["primary"],
                "old_wrong_start": "wrong_start_entity" in wt["diagnostic"]["labels"],
                "agent_primary": agent["diagnostic"]["primary"],
                "agent_origin": origin,
                "agent_start": entities(origin["ids"]) if origin else [],
                "agent_text": agent_raw["text"] if discordant else None,
                "gold_plan_probe": gold_plan_probe(
                    executor, labels_by_id, identity["ids"], [t[1] for t in gold["triple"]]
                )
                if discordant or mechanism != "same_start"
                else None,
                "sources": {
                    "WT": wt["source"],
                    "agent": agent["source"],
                    "gold": wt["gold_source"],
                },
            }
            item["explanation"] = explain_item(item)
            item["pair_mechanism"] = pair_mechanism(item)
            all_items.append(item)
        del executor, index, tokenizer
        print(f"completed {family}", flush=True)
    for path, digest in inputs.files.items():
        if path in expected and digest["sha256"] != expected[path]:
            raise ValueError(f"sealed input differs: {path}")
    args.out.mkdir(parents=True, exist_ok=True)
    return all_items, inputs, freeze, git


def render(args, items, inputs, freeze, git):
    from xgraph.error_analysis import Inputs, write_json

    summary = {}
    for panel in ("validation", "analysis-dev"):
        for family in ("qwen38", "gemma31"):
            for status in ("all_new", "all_old", "mixed_old_new"):
                selected = [
                    r
                    for r in items
                    if (r["panel"], r["family"], r["knowledge_status"]) == (panel, family, status)
                ]
                summary[f"{panel}/{family}/{status}"] = cohort_summary(selected)
                summary[f"{panel}/{family}/{status}"]["pair_decomposition"] = cohort_summary(
                    selected, "pair_mechanism"
                )["decomposition"]
    spot_checks = args.out / "manual-checks.json"
    if spot_checks.exists():
        cases = inputs.read_json(spot_checks)["cases"]
        by_key = {(r["panel"], r["family"], r["id"]): r for r in items}
        verified = []
        for case in cases:
            for family in case["families"]:
                item = by_key[(case["panel"], family, case["id"])]
                if (
                    item["mechanism"] != case["mechanism"]
                    or set(item["evidence"]["gold_ids"]) != set(case["gold_ids"])
                    or set(item["evidence"]["wt_ids"]) != set(case["wt_ids"][family])
                ):
                    raise ValueError(f"manual check differs from replay: {case['id']}/{family}")
                verified.append({"panel": case["panel"], "family": family, "id": case["id"]})
        write_json(args.out / "manual-validation.json", {"status": "passed", "checks": verified})
    detailed = [
        r
        for r in items
        if (r["knowledge_status"] == "all_new" and r["discordant"])
        or r["mechanism"] != "same_start"
    ]
    write_json(args.out / "items.json", detailed)
    write_json(
        args.out / "cohort.json",
        [
            {
                key: r[key]
                for key in (
                    "panel",
                    "family",
                    "id",
                    "knowledge_status",
                    "hops",
                    "cluster",
                    "weight",
                    "wt_correct",
                    "agent_correct",
                    "discordant",
                    "direction",
                    "mechanism",
                    "pair_mechanism",
                    "old_wrong_start",
                    "evidence",
                )
            }
            for r in items
        ],
    )
    write_json(args.out / "summary.json", summary)
    write_json(
        args.out / "homonym-spot-checks.json",
        [
            {key: r[key] for key in ("panel", "family", "id", "gold_start", "gold_plan_probe")}
            for r in detailed
            if r["panel"] == "validation"
            and r["knowledge_status"] == "all_new"
            and r["discordant"]
            and r["mechanism"] == "homonym_different_qid"
        ],
    )

    def entity_text(entities):
        return (
            "; ".join(f"{e['label']} ({e['id']})" for e in entities)
            or "unresolved; see saved searches and attempts"
        )

    lines = [
        "# Every all-new discordance",
        "",
        "Both directions; source references, full candidate sets, tool attempts and gold-chain "
        "witnesses are in items.json. Gold QIDs are inferred from the complete released chain, "
        "not provided QID annotations. Multiple matching QIDs remain a set.",
        "",
    ]
    for r in detailed:
        if r["knowledge_status"] != "all_new" or not r["discordant"]:
            continue
        e = r["evidence"]
        lines += [
            f"## {r['panel']} / {r['family']} / {r['id']} / {r['direction']}",
            "",
            r["question"],
            "",
            f"- Gold start: {entity_text(r['gold_start'])}.",
            f"- WT start: {entity_text(r['wt_start'])}; written `{e['wt_label']}` "
            f"({r['wt_start_attribution']}).",
            f"- Agent resolved root: {entity_text(r['agent_start'])}; supports its final answer "
            f"in saved/replayed navigation: {r['agent_origin']['supports_final_answer']}.",
            f"- Signature: **{r['mechanism']}**; WT taxonomy `{r['wt_primary']}`; "
            f"agent taxonomy `{r['agent_primary']}`.",
            f"- Losing arm's paired signature: **{r['pair_mechanism']}**.",
            f"- Canonical gold row QIDs: {e['gold_canonical_row_ids']}; "
            f"question spelling {e['question_surface']!r}, "
            f"gold row QIDs {e['question_surface_gold_ids']}.",
            f"- Shared token prefix {e['variant_prefix']['text']!r}; "
            f"excludes gold: {e['variant_branch_excludes_gold']}; patent class: {e['patent']}.",
            f"- Sources: {r['sources']}",
            f"- Explanation: {r['explanation']}",
            "",
            "```text",
            r["wt_text"],
            "```",
            "",
        ]
    (args.out / "discordances.md").write_text("\n".join(lines).rstrip() + "\n")

    lines = [
        "# Base rates and gap decomposition",
        "",
        "All percentages use inherited survey weights. Counts are unweighted. A QID wrong-start "
        "error requires a scored WT error and a reconstructed start outside every full-chain "
        "compatible gold start. The original taxonomy uses normalized names and misses some "
        "homonyms and entity-only spans. This is descriptive; strata differ in hop composition.",
        "",
        "| Panel/family/status | n | 1-hop n | Original wrong-start n (%) | "
        "QID wrong-start n (%) | Variant branch errors n (%) | Missing question spelling n (%) | "
        "WT-agent pp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, s in summary.items():
        values = [
            f"{s[k]['n']} ({s[k]['rate_pct']:.2f})"
            for k in (
                "old_wrong_start_error",
                "qid_wrong_start_error",
                "variant_branch_error",
                "missing_question_variant",
            )
        ]
        lines.append(
            f"| {key} | {s['n']} | {s['one_hop']['n']} | "
            + " | ".join(values)
            + f" | {s['gap_pp']:+.2f} |"
        )
    for panel in ("validation", "analysis-dev"):
        for family in ("qwen38", "gemma31"):
            key = f"{panel}/{family}/all_new"
            s = summary[key]
            lines += [
                "",
                f"## {key}",
                "",
                "| WT start signature | Agent-only n | WT-only n | "
                "Share of agent-only count / weight (%) | Signed contribution pp | "
                "Share of net gap (%) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for mechanism, cell in s["decomposition"].items():
                if not (cell["agent_only_n"] or cell["WT_only_n"]):
                    continue
                lines.append(
                    f"| {mechanism} | {cell['agent_only_n']} | {cell['WT_only_n']} | "
                    f"{cell['agent_only_share_pct']:.2f} / "
                    f"{cell['agent_only_weight_share_pct']:.2f} | {cell['net_pp']:+.4f} | "
                    f"{100 * cell['net_pp'] / s['gap_pp']:.2f} |"
                )
            lines += [
                "",
                f"Contributions sum to {s['gap_pp']:+.6f} pp. Negative contributions favor the "
                "agent. Shares can exceed 100% when WT-only cases offset losses. These are "
                "accounting shares of observed signatures, not estimated causal mediation.",
            ]
            lines += [
                "",
                "Both directions, classifying the losing arm's start:",
                "",
                "| Signature | Agent-only n | WT-only n | Net contribution pp |",
                "|---|---:|---:|---:|",
            ]
            for mechanism, cell in s["pair_decomposition"].items():
                if cell["agent_only_n"] or cell["WT_only_n"]:
                    lines.append(
                        f"| {mechanism} | {cell['agent_only_n']} | {cell['WT_only_n']} | "
                        f"{cell['net_pp']:+.4f} |"
                    )
    (args.out / "tables.md").write_text("\n".join(lines) + "\n")

    methods = """# Rules and limits

This is a post hoc, read-only diagnosis. Neither a registered policy nor a scorer is
changed. All computations are CPU only, nice 19, cores 1-4, with a four-billion-byte
address-space ceiling. No model calls, GPU probes, containers or downloads are required.

Gold start recovery uses NFC/casefold exact name lookup in the frozen SQLite KG store, then
requires every released subject-relation-object hop in a connected chain. Relation aliases
are read from the KG directory. One witness per compatible start is saved; ambiguous starts
are never arbitrarily collapsed. All recovered gold starts have label-index rows. N means a
released new-fact annotation; it does not itself prove a missing entity or a KG-version
update.

WT starts are replayed with the existing MatchedExecutor, sitelink start ranking, unquoted
starts, production caps and saved selections. The surviving start rank is read from its
capped walk. An initial closed entity-only span is also resolved, even when the old path
parser reports no path. Original correctness and original taxonomy stay unchanged. QID
mismatches are only counted as errors when the sealed scorer says WT is wrong; a same-answer
alternate start is not promoted to a new scoring failure.

Agent navigation uses the existing KGTools and MatchedPaths over a read-only KGStore. Saved
Gemma results are compared against the replay. Qwen results were not saved: replayed results
are explicitly labelled. Follow resolves labels by the tool's degree ranking; execute_path
uses committed path execution. Subsequent navigation through a returned entity inherits its
root. The last navigation supporting the agent's parsed answer supplies the reported root;
otherwise the last resolved root is reported. Without navigation, the last resolved
entity lookup is reported with status lookup_without_executed_path. Failed attempts do not
erase an earlier resolved root. Searches and all attempts remain available. Matching a
wrong final answer only identifies the agent's trajectory, not gold correctness or a causal
explanation.

Classifier precedence: no KG name -> absent_from_kg; names but no full gold chain ->
unresolved_gold_identity; no row for any compatible gold QID ->
gold_missing_from_label_index; unresolved WT -> unresolved_wt_start; overlapping
committed/gold QIDs -> same_start; shared writable start candidates with wrong commitment ->
homonym_different_qid; absent question spelling with a shared non-space WT/question token
prefix whose row range excludes all gold QIDs -> label_variant_missing; identical label with
incompatible QID -> homonym_different_qid; patent-class gold and other start ->
template_artifact; otherwise other_start_substitution. Every predicate and prefix row range
is saved. Patent/template and missing-spelling flags may overlap other signatures.
Template_artifact means a patent title was substituted by another start; it does not
establish that the released question is invalid.

Missing spelling alone is not labelled a cause: the correct canonical label is available. A
variant branch is a stricter observed signature: the model selected a spelling whose
accepted token-prefix rows cannot name gold. Adding an alias would reopen that branch; it
would not force correct entity, relation, stopping or answer choices. No pre-mask logits or
counterfactual generation are available. Thus the evidence implicates case-sensitive
span/index coverage interacting with model choices, rather than proving an intrinsic
model-knowledge deficit or a causal improvement from aliases. The fixed gold-plan probe
tries every indexed gold label with the released relations and no additional selections.
Failure can expose frozen start ranking even with correct relation choices; it does not
prove that every alternative path or selected first-pass trajectory is impossible. Gold
names absent from the question cannot be declared missing aliases merely from string
mismatch. Same-start failures additionally involve later model planning/selection.

The paired signature classifies WT start failures on agent-only items and agent start
failures on WT-only items. An agent lookup selecting another QID with the same normalized
name is a homonym choice even without a successful executed path; its lookup-only status
stays explicit. Overlapping starts and remaining non-start failures are grouped as
other_downstream_or_delivery. Both this signed decomposition and the WT-only signature
decomposition are reported; neither is causal mediation.

Gap accounting uses the complete all-new stratum's inherited survey-weight denominator and
sums weight*(WT_correct-agent_correct) within each exclusive WT start signature. It includes
both discordance directions and preserves every negative result. Base-rate controls include
every well-formed all-old and mixed TI item and analysis-dev all-new item; these are not
matched random interventions. Mixed has no one-hop cases. No new significance claim is made;
inherited all-new intervals are pointwise, post hoc and unadjusted. The classifier
regression tests live in tests/test_mintqa_rematch.py.
"""
    (args.out / "methods.md").write_text(methods)
    q = summary["validation/qwen38/all_new"]
    g = summary["validation/gemma31/all_new"]
    qv, gv = (
        q["decomposition"]["label_variant_missing"],
        g["decomposition"]["label_variant_missing"],
    )
    qh = q["pair_decomposition"]["homonym_different_qid"]
    gh = g["pair_decomposition"]["homonym_different_qid"]
    report = f"""Case-sensitive spelling gaps cover {qv["agent_only_n"]}/17 Qwen and
{gv["agent_only_n"]}/13 Gemma agent-only losses
({qv["agent_only_weight_share_pct"]:.1f}%/{gv["agent_only_weight_share_pct"]:.1f}% weighted),
contributing {qv["net_pp"]:.2f}/{gv["net_pp"]:.2f} pp to the -8.32/-4.22 pp gaps.
Homonym losses/wins are 2/1 Qwen and 2/4 Gemma: their net contributions are
{qh["net_pp"]:+.2f}/{gh["net_pp"]:+.2f} pp. Gemma's patent-title substitution contributes
-0.72 pp; remaining downstream/delivery differences contribute -1.35/+1.82 pp.

Every gold QID has an indexed label: no missing KG entity or stale label snapshot.
Missing case variants interact with model start choices. Gold relations still choose
wrong homonyms for Waterhouse, Boundary Rock and Myrtle Creek; their gold QIDs have
only shared labels, exposing frozen start ranking beyond relation prediction.

Analysis-dev has one spelling-branch error per family (both agents also wrong);
validation all-old has two, mixed none. These strata differ in hop composition.

Nov 13 hypothesis only: initial-case aliases, capitalization normalization and separately
QID-based starts. Accuracy effects are unmeasured; full-variant optimistic rescue ceilings
are +6.18/+5.76 pp. No registered policy changed.

Regenerate: `bash target/error-analysis/all-new/regenerate.sh`. Commit/provenance: see
provenance.json and the containing commit. GPU-hours: 0; no model execution.
"""
    if len(report.split()) > 200:
        raise ValueError("short report exceeds the requested word limit")
    (args.out / "report.md").write_text(report.lstrip())
    command = [
        "nice",
        "-n",
        "19",
        "taskset",
        "-c",
        "1-4",
        "prlimit",
        "--as=4000000000",
        str(Path(__file__).resolve().parents[2] / ".venv/bin/python"),
        "-I",
        "-S",
        str(Path(__file__).resolve().parents[2] / "scripts/mintqa/all-new-analysis.py"),
        "--source-root",
        str(args.source_root),
        "--sealed-root",
        str(args.sealed_root),
        "--analysis",
        str(args.analysis),
        "--out",
        str(args.out),
    ]
    write_json(args.out / "argv.json", {"command": command, "gpu_hours": 0, "gpu_queue": None})
    (args.out / "regenerate.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd "
        + shlex.quote(str(Path(__file__).resolve().parents[2]))
        + "\nexec "
        + shlex.join(command)
        + "\n"
    )
    after = Inputs()
    for path, entry in inputs.files.items():
        if after.digest(path)["sha256"] != entry["sha256"]:
            raise ValueError(f"input changed during analysis: {path}")
    write_json(
        args.out / "provenance.json",
        {
            "artifact_kind": "xgraph.mintqa_all_new_diagnosis",
            "status": "post_hoc_descriptive",
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "dirty": bool(git("status", "--porcelain")),
            "sealed_producer_commit": freeze["code_commit"],
            "sealed_replay_core_blobs": freeze["replay_core_blobs"],
            "command": command,
            "config": {
                "seed": None,
                "deterministic": True,
                "policy_changed": False,
                "start_rank": "sitelinks",
                "unquoted_starts": True,
            },
            "inputs": list(inputs.files.values()),
            "input_rehash_unchanged": True,
            "native_correctness_reproduced": True,
            "nice": os.getpriority(os.PRIO_PROCESS, 0),
            "affinity": sorted(os.sched_getaffinity(0)),
            "address_space_limit_bytes": resource.getrlimit(resource.RLIMIT_AS)[0],
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "python": sys.version,
            "sqlite": sqlite3.sqlite_version,
            "gpu_hours": 0,
            "gpu_hours_basis": "No model calls, GPU probes or engine execution.",
            "outputs": [
                after.digest(path)
                for path in sorted(args.out.iterdir())
                if path.is_file() and path.name != "provenance.json"
            ],
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sealed-root", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in ("source_root", "sealed_root", "analysis", "out"):
        setattr(args, name, getattr(args, name).resolve())
    render(args, *run(args))


if __name__ == "__main__":
    main()
