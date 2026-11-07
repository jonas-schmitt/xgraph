"""2WikiMultihopQA as an external benchmark for path mode, over a Wikidata KG directory.

2WikiMultihopQA (Ho et al., COLING 2020; Apache-2.0) asks multi-hop questions about
Wikipedia articles, with each question's evidence as Wikidata triples. Its release
with ids (`data_ids.zip`, linked from the authors' README at
github.com/Alab-NII/2wikimultihop) gives the evidence as QIDs too (`evidences_id`:
subject QID, relation name, object QID) and the answer's QID (`answer_id`). Decoding
on Graphs (DoG, Li et al., ACL 2025) evaluates on its dev set, each question over a
graph a model extracted from the question's passages; its released file
(`data/2wiki/2wiki_4000.jsonl` at github.com/leekum2018/Decoding-on-Graph) holds a
subset of the dev questions, keyed by 2WikiMultihopQA's `_id`.

Path mode asks every question over one whole KG, here a KG directory streamed from
Wikidata (`python -m xgraph.wikidata stream`, the gate KG), whose KG index the engine
already has. This module turns the dev set into xgraph's inputs over it.

**The questions** (`questions`). Each dev question becomes a WWQ-format question with
id `2wiki-<_id>` and its type (`compositional`, `inference`, `comparison`,
`bridge_comparison`) as `complexity`; with `--dog`, the questions of DoG's release
form a subset of their own, in its order.

**Representable questions**, decided before any model run: a question is
representable when

1. it is a chain question, compositional or inference (a comparison or bridge
   comparison compares two entities, which a relation path cannot express);
2. its answer is an entity (`answer_id` a QID; a date's or a number's is not);
3. its evidence is a chain to that answer: each hop starts where the one before ended,
   and the last ends at the answer;
4. every hop holds in the KG under exactly one property its relation name names:
   the properties with that English label, else those with it as an alias, else
   `RENAMED` (labels Wikidata has changed since the dataset was made).

Its gold answer is `answer_id`, its QID chain the hops' triples, and its first subject
the question entity (for anchored requests). Every other question is dropped with the
first reason found: not a chain question, literal answer, disconnected evidence, or
for a hop, unknown relation, no KG triple, or several KG triples. It stays in
`questions.json` and in `gold.json` under `failed`, so xgraph.editing's scores over all
questions count it as wrong. It is still asked, and Hits@1 scores it.

Written to `--out`: `questions.json` (every dev question),
`questions-representable.json`, `gold.json` (`answers` and `failed`), `chains.json`
(each representable question's QID chain), `labels.jsonl` (each gold answer as the
dataset writes it, then its labels in the KG), `requests-paths.jsonl` (the path-span
prompt, unanchored: path mode) and `requests-fewshot.jsonl` (the gate's few-shot
prompt: the prompted arm) for every question; with `--dog`, `questions-dog.json`,
`questions-dog-representable.json`, `requests-paths-dog.jsonl`, and
`requests-fewshot-dog.jsonl` for DoG's subset. `manifest.json` counts, per subset
(`all`, `dog`), the questions, the representable ones, and the dropped ones by type
and reason; beside them `evidence_chains`, the questions representable over a KG of
the evidence triples alone (rules 1 to 3), and, among the representable ones,
`single_endpoint` (the chain's relations reach the answer alone from the start) and
`linkable_start` (a label or alias of the start, casefolded, is in the question's text);
for DoG's subset `dog_graph`, how many of its graphs name the answer (`answer_label`)
and hold the gold chain's label triples (`gold_chain`). Anchored or linked requests
come from `python -m xgraph.prompts wwq --questions <out>/questions-representable.json`
with the options of the MQuAKE arms.

    python -m xgraph.twowiki questions \\
        --dev data/external/2wikimultihopqa-ids/data_ids/dev.json \\
        --kg data/wikidata/gate-kg-v3e \\
        --dog data/external/dog-2wiki/release/data/2wiki/2wiki_4000.jsonl \\
        --out target/2wiki

`requests --expressions` writes `requests-paths-expr.jsonl` (and `-dog`) beside them: the
path-span prompt with date paths, date comparisons, and value comparisons, for path mode
with expressions (XGRAPH_EXPRESSIONS=1), whose answers name a date, a compared entity, or
a verdict (`--name` writes them under another name):

    python -m xgraph.twowiki requests --dir target/2wiki --expressions

Scores come from `python -m xgraph.editing` with `gold.json` as `--gold-new` and no
`--gold-old`, over the representable questions and over all of them. DoG reports
Hits@1, "the proportion of instances whose top-1 prediction is the ground truth
answer"; its release holds no scorer, and its predictions follow the format that
Reasoning on Graphs' `evaluate_results.py` reads, whose `eval_hit` counts a prediction
when it contains a gold answer string, both normalized (lowercased, ASCII punctuation
and the articles a, an, the removed, whitespace collapsed). `score` computes that with
the dataset's answer as the gold string and the final answer's label as the top-1
prediction, or with `--whole-text` the whole output, as `eval_hit` over DoG's
generated sequence would; a question without an output or an answer line is wrong:

    python -m xgraph.twowiki score --outputs <arm.jsonl> \\
        --questions target/2wiki/questions-dog.json \\
        --dev data/external/2wikimultihopqa-ids/data_ids/dev.json --out <arm>.hits.json
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np

from xgraph.score import record_text

CHAIN_TYPES = ("compositional", "inference")
RENAMED = {"founded by": {112}}
"""Relation names the dataset uses that are no longer a property's label or alias:
P112, "founded by" in 2020, is "founder" today."""
QID = re.compile(r"Q(\d+)")

Facts = dict[tuple[int, int], set[int]]


def _relation_names(kg: Path) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    labels: dict[str, set[int]] = {}
    aliases: dict[str, set[int]] = {}
    with (kg / "relation_labels.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            prop, text, kind = json.loads(line)
            (labels if kind == "label" else aliases).setdefault(text, set()).add(int(prop[1:]))
    return labels, aliases


def properties(name: str, labels: dict, aliases: dict) -> set[int]:
    """The properties a relation name names (rule 4 of the module docstring)."""
    return labels.get(name) or aliases.get(name) or RENAMED.get(name, set())


class _Facts:
    """The KG's triples of chosen subjects and properties, loaded on demand."""

    def __init__(self, kg: Path, props: set[int]) -> None:
        self.triples = np.load(kg / "triples.npy", mmap_mode="r")
        self.props = np.fromiter(props, dtype=np.int32)
        self.loaded: set[int] = set()
        self.facts: Facts = {}

    def load(self, subjects: set[int]) -> None:
        new = subjects - self.loaded
        if not new:
            return
        self.loaded |= new
        rows = self.triples[np.isin(self.triples[:, 0], np.fromiter(new, dtype=np.int32))]
        rows = rows[np.isin(rows[:, 1], self.props)]
        for s, p, o in rows.tolist():
            self.facts.setdefault((s, p), set()).add(o)

    def objects(self, subject: int, prop: int) -> set[int]:
        return self.facts.get((subject, prop), set())


def _number(qid: str | None) -> int | None:
    found = QID.fullmatch(qid or "")
    return int(found.group(1)) if found else None


def _evidence_chain(row: dict) -> tuple[list[tuple[int, str, int]] | None, str | None]:
    """Rules 1 to 3: the evidence as (subject, relation name, object) numbers, or why not."""
    if row["type"] not in CHAIN_TYPES:
        return None, "not a chain question"
    answer = _number(row["answer_id"])
    if answer is None:
        return None, "literal answer"
    hops = [(_number(s), r, _number(o)) for s, r, o in row["evidences_id"]]
    connected = (
        hops
        and all(s is not None and o is not None for s, _, o in hops)
        and all(hops[k][2] == hops[k + 1][0] for k in range(len(hops) - 1))
        and hops[-1][2] == answer
    )
    if not connected:
        return None, "disconnected evidence"
    return hops, None


def _start_labels(kg: Path, items: set[int]) -> dict[int, list[tuple[str, str]]]:
    """Each wanted item's labels and aliases, in the KG's order, read in one pass."""
    wanted = {f"Q{n}" for n in items}
    found: dict[int, list[tuple[str, str]]] = {}
    with (kg / "entity_labels.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            end = line.find('"', 2)
            if line[2:end] in wanted:
                qid, text, kind = json.loads(line)
                found.setdefault(int(qid[1:]), []).append((text, kind))
    return found


def build_questions(rows: list[dict], kg: str | Path) -> dict:
    """Questions, gold answers, QID chains, why each dropped question was dropped, and
    the diagnostics of the module docstring; `answers` holds each gold answer as the
    dataset writes it."""
    kg = Path(kg)
    labels, aliases = _relation_names(kg)
    evidence: dict[str, list[tuple[int, str, int]]] = {}
    dropped: dict[str, tuple[int, str]] = {}
    for row in rows:
        hops, why = _evidence_chain(row)
        if hops is None:
            dropped[f"2wiki-{row['_id']}"] = (0, why)
        else:
            evidence[f"2wiki-{row['_id']}"] = hops
    props = {
        p for hops in evidence.values() for _, r, _ in hops for p in properties(r, labels, aliases)
    }
    facts = _Facts(kg, props)
    facts.load({s for hops in evidence.values() for s, _, _ in hops})
    resolved: dict[str, list[tuple[int, int, int]]] = {}
    for qid, hops in evidence.items():
        chain = []
        for k, (s, r, o) in enumerate(hops, start=1):
            named = properties(r, labels, aliases)
            holding = sorted(p for p in named if o in facts.objects(s, p))
            if len(holding) != 1:
                why = (
                    "unknown relation"
                    if not named
                    else "no KG triple"
                    if not holding
                    else "several KG triples"
                )
                dropped[qid] = (k, why)
                break
            chain.append((s, holding[0], o))
        else:
            resolved[qid] = chain
    frontiers = {qid: {chain[0][0]} for qid, chain in resolved.items()}
    for k in range(max((len(c) for c in resolved.values()), default=0)):
        facts.load({e for qid, f in frontiers.items() if k < len(resolved[qid]) for e in f})
        for qid, chain in resolved.items():
            if k < len(chain):
                frontiers[qid] = {o for e in frontiers[qid] for o in facts.objects(e, chain[k][1])}
    names = _start_labels(
        kg, {c[0][0] for c in resolved.values()} | {c[-1][2] for c in resolved.values()}
    )
    questions, gold, answers, chains, linkable = [], {}, {}, {}, set()
    for row in rows:
        qid = f"2wiki-{row['_id']}"
        chain = resolved.get(qid)
        start = f" wd:Q{chain[0][0]}" if chain else ""
        questions.append(
            {
                "id": qid,
                "utterance": row["question"],
                "sparql": f"# question entities:{start}",
                "complexity": row["type"],
            }
        )
        if chain is None:
            continue
        gold[qid] = {"entities": [f"Q{chain[-1][2]}"], "literals": []}
        answers[qid] = row["answer"]
        chains[qid] = [[f"Q{s}", f"P{p}", f"Q{o}"] for s, p, o in chain]
        text = row["question"].casefold()
        if any(label.casefold() in text for label, _ in names.get(chain[0][0], [])):
            linkable.add(qid)
    failed = {
        q: f"not representable: hop {k}, {why}" if k else f"not representable: {why}"
        for q, (k, why) in dropped.items()
    }
    return {
        "questions": questions,
        "gold": {"answers": gold, "failed": dict(sorted(failed.items(), key=_order(questions)))},
        "answers": answers,
        "chains": chains,
        "dropped": dropped,
        "endpoints": {q: len(frontiers[q]) for q in resolved},
        "linkable": linkable,
        "evidence_chains": set(evidence),
    }


def _order(questions: list[dict]):
    position = {q["id"]: k for k, q in enumerate(questions)}
    return lambda item: position[item[0]]


def _clean(text: str) -> str:
    """A graph label as DoG's Graph_Processor reads it."""
    return text.replace(" : ", ": ").replace(" ./. ", "./. ").strip()


def dog_graph_counts(dog: list[dict], rows: dict[str, dict]) -> dict:
    """How many of DoG's graphs name their question's answer as an entity, and how many
    hold every label triple of a chain question's evidence."""
    counts = Counter(answer_label=0, gold_chain=0)
    for record in dog:
        graph = {tuple(_clean(x) for x in triple) for triple in record["graph"]}
        names = {s for s, _, _ in graph} | {o for _, _, o in graph}
        counts["answer_label"] += _clean(record["answer"]) in names
        row = rows[record["id"]]
        if row["type"] in CHAIN_TYPES and row["evidences"]:
            counts["gold_chain"] += all(
                tuple(_clean(x) for x in h) in graph for h in row["evidences"]
            )
    return dict(counts)


def tally(questions: list[dict], built: dict) -> dict:
    """Questions, representable ones, and dropped ones by reason, over the questions
    given and per type, with the diagnostics of the module docstring."""
    dropped = built["dropped"]
    table: dict[str, dict] = {}
    for q in questions:
        row = table.setdefault(q["complexity"], {"questions": 0, "representable": 0, "dropped": {}})
        row["questions"] += 1
        if q["id"] in dropped:
            why = dropped[q["id"]][1]
            row["dropped"][why] = row["dropped"].get(why, 0) + 1
        else:
            row["representable"] += 1
    kept = [q["id"] for q in questions if q["id"] not in dropped]
    return {
        "questions": len(questions),
        "representable": len(kept),
        "evidence_chains": sum(q["id"] in built["evidence_chains"] for q in questions),
        "single_endpoint": sum(built["endpoints"][q] == 1 for q in kept),
        "linkable_start": sum(q in built["linkable"] for q in kept),
        "by_type": dict(sorted(table.items())),
    }


def _write_requests(questions: Path, out: Path, suffix: str) -> None:
    from xgraph.prompts import wwq_requests

    for name, path_spans in (("requests-paths", True), ("requests-fewshot", False)):
        with (out / f"{name}{suffix}.jsonl").open("w", encoding="utf-8") as written:
            for request in wwq_requests(questions, path_spans=path_spans):
                written.write(json.dumps(request, ensure_ascii=False) + "\n")


def write_expression_requests(directory: Path, name: str = "requests-paths-expr") -> dict[str, int]:
    """requests-paths-expr.jsonl and, if the directory has DoG's subset,
    requests-paths-expr-dog.jsonl: the path-span prompt with date paths and date
    comparisons (xgraph.prompts EXPRESSION_PATH_INSTRUCTIONS), for path mode with
    expressions; returns how many requests each file holds."""
    from xgraph.prompts import wwq_requests

    counts = {}
    for suffix in ("", "-dog"):
        questions = directory / f"questions{suffix}.json"
        if not questions.exists():
            continue
        out = directory / f"{name}{suffix}.jsonl"
        requests = wwq_requests(questions, path_spans=True, expressions=True)
        with out.open("w", encoding="utf-8") as written:
            for request in requests:
                written.write(json.dumps(request, ensure_ascii=False) + "\n")
        counts[out.name] = len(requests)
    return counts


def write_questions(built: dict, kg: Path, out: Path, dog: list[dict] | None = None) -> None:
    """The question, gold, label, and request files of the module docstring."""
    out.mkdir(parents=True, exist_ok=True)
    questions, dropped = built["questions"], built["dropped"]
    kept = [q for q in questions if q["id"] not in dropped]
    (out / "questions.json").write_text(json.dumps(questions, indent=1, ensure_ascii=False) + "\n")
    (out / "questions-representable.json").write_text(
        json.dumps(kept, indent=1, ensure_ascii=False) + "\n"
    )
    (out / "gold.json").write_text(json.dumps(built["gold"]) + "\n")
    (out / "chains.json").write_text(json.dumps(built["chains"]) + "\n")
    golden = {q: a["entities"][0] for q, a in built["gold"]["answers"].items()}
    rows = dict.fromkeys((golden[q], built["answers"][q]) for q in golden)
    items = set(golden.values())
    with (out / "labels.jsonl").open("w", encoding="utf-8") as written:
        for item, answer in rows:
            written.write(json.dumps([item, answer, "label"], ensure_ascii=False) + "\n")
        with (kg / "entity_labels.jsonl").open(encoding="utf-8") as lines:
            for line in lines:
                end = line.find('"', 2)
                if line[2:end] in items:
                    written.write(line)
    _write_requests(out / "questions.json", out, "")
    if dog is not None:
        by_id = {q["id"]: q for q in questions}
        subset = [by_id[f"2wiki-{r['id']}"] for r in dog]
        (out / "questions-dog.json").write_text(
            json.dumps(subset, indent=1, ensure_ascii=False) + "\n"
        )
        (out / "questions-dog-representable.json").write_text(
            json.dumps([q for q in subset if q["id"] not in dropped], indent=1, ensure_ascii=False)
            + "\n"
        )
        _write_requests(out / "questions-dog.json", out, "-dog")


def rog_normalize(text: str) -> str:
    """Reasoning on Graphs' normalize: lowercase, no ASCII punctuation, no articles,
    whitespace collapsed."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _hit(text: str | None, answer: str, whole_text: bool) -> bool:
    from xgraph.prompts import final_answer

    if text is None:
        return False
    said = text if whole_text else final_answer(text)
    return bool(said) and rog_normalize(answer) in rog_normalize(said)


def hits_at_1(
    texts: dict[str, str], answers: dict[str, str], questions: list[dict], whole_text: bool = False
) -> dict:
    """DoG's Hits@1 as the module docstring defines it, over the questions given and
    per type (complexity)."""

    def share(ids: list[str]) -> dict:
        n = sum(_hit(texts.get(q), answers[q], whole_text) for q in ids)
        return {"questions": len(ids), "hits@1": n / len(ids) if ids else 0.0}

    types: dict[str, list[str]] = {}
    for q in questions:
        types.setdefault(q["complexity"], []).append(q["id"])
    return {
        "all": share([q["id"] for q in questions]),
        "by_type": {name: share(ids) for name, ids in sorted(types.items())},
    }


def main(argv: list[str] | None = None) -> None:
    from xgraph.provenance import provenance

    parser = argparse.ArgumentParser(prog="python -m xgraph.twowiki", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    asked = commands.add_parser("questions", help="questions, gold, labels, and requests")
    asked.add_argument(
        "--dev", type=Path, required=True, help="2WikiMultihopQA's dev.json with ids"
    )
    asked.add_argument("--kg", type=Path, required=True, help="a Wikidata KG directory")
    asked.add_argument("--dog", type=Path, help="DoG's released 2wiki_4000.jsonl: a subset")
    asked.add_argument("--out", type=Path, required=True, help="output directory")
    requested = commands.add_parser(
        "requests", help="further requests for the questions of a questions directory"
    )
    requested.add_argument("--dir", type=Path, required=True, help="the questions' --out")
    requested.add_argument(
        "--expressions",
        action="store_true",
        required=True,
        help="the path-span prompt with date paths and date and value comparisons",
    )
    requested.add_argument(
        "--name",
        default="requests-paths-expr",
        help="the request files' name, before -dog and .jsonl (default: requests-paths-expr)",
    )
    score = commands.add_parser("score", help="DoG's Hits@1 over the questions given")
    score.add_argument("--outputs", type=Path, required=True, help="an arm's records (JSONL)")
    score.add_argument("--questions", type=Path, required=True, help="a questions file")
    score.add_argument("--dev", type=Path, required=True, help="2WikiMultihopQA's dev.json")
    score.add_argument("--whole-text", action="store_true", help="the whole output as prediction")
    score.add_argument("--out", type=Path, required=True, help="scores (JSON)")
    args = parser.parse_args(argv)
    if args.command == "requests":
        print(json.dumps(write_expression_requests(args.dir, args.name)))
        return
    rows = json.loads(args.dev.read_text(encoding="utf-8"))
    if args.command == "score":
        answers = {f"2wiki-{r['_id']}": r["answer"] for r in rows}
        records = (json.loads(line) for line in args.outputs.read_text().splitlines() if line)
        texts = {r["id"]: record_text(r) for r in records}
        questions = json.loads(args.questions.read_text(encoding="utf-8"))
        result = {
            "artifact_kind": "xgraph.twowiki_hits",
            "version": 1,
            "whole_text": args.whole_text,
            **hits_at_1(texts, answers, questions, args.whole_text),
            "provenance": provenance([args.outputs, args.questions, args.dev]),
        }
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result["all"]))
        return
    dog = None
    if args.dog is not None:
        with args.dog.open(encoding="utf-8") as lines:
            dog = [json.loads(line) for line in lines if line.strip()]
    built = build_questions(rows, args.kg)
    write_questions(built, args.kg, args.out, dog)
    subsets = {"all": tally(built["questions"], built)}
    if dog is not None:
        by_id = {q["id"]: q for q in built["questions"]}
        subset = [by_id[f"2wiki-{r['id']}"] for r in dog]
        subsets["dog"] = {
            **tally(subset, built),
            "dog_graph": dog_graph_counts(dog, {r["_id"]: r for r in rows}),
        }
    inputs = [args.dev, args.kg / "triples.npy", args.kg / "relation_labels.jsonl"]
    manifest = {
        "artifact_kind": "xgraph.twowiki_questions",
        "version": 1,
        "renamed_relations": {name: sorted(f"P{p}" for p in ps) for name, ps in RENAMED.items()},
        "subsets": subsets,
        "provenance": provenance(inputs + ([args.dog] if args.dog else [])),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(
        json.dumps(
            {name: {k: v for k, v in s.items() if k != "by_type"} for name, s in subsets.items()}
        )
    )


if __name__ == "__main__":
    main()
