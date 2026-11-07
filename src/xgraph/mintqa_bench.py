"""MINTQA as an external benchmark for path mode, over the KG its retrieval baselines read.

MINTQA (He et al., ACL 2026; questions at Hugging Face `probejie/MINTQA`, KG at
`Sp1der/MintQA-KG`) asks 1- to 4-hop questions over Wikidata facts that are new (TI:
in the 2024 dump but not the 2021 one) or unpopular (POP: rare in Wikipedia). Its
retrieval baselines read the KG it releases, so path mode over that same KG compares
methods, not knowledge sources. This module turns the release into xgraph's inputs.

**The KG** (`kg`). The release holds its triples twice, line by line:
`mintqa_wikidata_w_id.txt` (`Q… <tab> P… <tab> Q…`) and `mintqa_wikidata_w_name.txt`
(the same triples as labels), with `relID2name_dict.json` naming the properties. They
become a KG directory, as `python -m xgraph.wikidata stream` writes one:

- `triples.npy`: int32 rows (subject, relation, object) of item and property numbers,
  without duplicates;
- `entity_labels.jsonl`: each item's label as the name file writes it, `["Q90",
  "Paris", "label"]`. The name file writes an item without a label as a QID (an object
  without one as its subject's QID); such a text is a placeholder, not a label, and an
  item with placeholders only is labelled with its own QID;
- `relation_labels.jsonl`: each property's label from `relID2name_dict.json`, or its
  PID where the dictionary has none; with `--relation-aliases`, a Wikidata
  `relation_labels.jsonl` (the gate KG's), each property's label and aliases there
  follow as aliases (the release names some properties as Wikidata did earlier,
  "founded by" for today's "founder"), so a path may name a relation either way;
- `manifest.json`: counts and provenance.

The conversion stops where the files disagree: different lengths, a line whose
relation label is not its property's, or an item named by two labels. A line of the id
file that is not three ids is skipped and counted. `python -m xgraph.wikidata index`
and `python -m xgraph.toolkg build` then build its KG index and KG store.

**The questions** (`questions`). Each split (`--split ti=<file>`, `--split pop=<file>`:
JSON lines with the dataset's fields, its parquet files converted in order) becomes
WWQ-format questions with ids `mintqa-<split>-<row>`. `complexity` names the split, the
hop count, and the stratum, how many hops are old (TI: type `O`, not `N`) or popular
(POP: `F`, not `R`), as in `ti 3-hop 1/3 old`; it is `types mismatch` where the dataset
gives types for another number of hops than the chain has. Gold answers are QIDs: each
hop's label triple is joined with the name file's lines, and the id file's triple on
the same line is the hop's.

**Representable questions**, decided before any model run: a question is
representable when every hop of its chain, matched by (subject label, relation label,
object label) exactly as the dataset writes them, maps to exactly one distinct KG
triple, and those triples connect (each hop's subject is the object of the hop
before). Its gold answer is the last hop's object, and its first subject is the
question entity (for anchored requests). Every other question is dropped, with the
first reason found: a hop with no KG triple, a hop with several, or a hop that does
not start where the one before ended. It stays in `questions.json` and in `gold.json`
under `failed`: it has no QID gold, so xgraph.editing's scores over all questions
count it as wrong. It is still asked, and MINTQA's own measure scores it.

Written to `--out`: `questions.json` (every question), `questions-representable.json`,
`gold.json` (`answers` and `failed`), `chains.json` (each representable question's QID
chain), `labels.jsonl` (each gold answer as the dataset writes it, then its KG label),
`requests-paths.jsonl` (the path-span prompt, unanchored: path mode) and
`requests-fewshot.jsonl` (the gate's few-shot prompt: the prompted arm), both for every
question (a request is the question's text alone; whether its chain maps to the KG is
not consulted), and `manifest.json` with the kept and dropped questions per
split and per group.

    python -m xgraph.mintqa_bench kg --source data/external/mintqa-full/kg-source \\
        --relation-aliases data/wikidata/gate-kg/relation_labels.jsonl \\
        --out data/external/mintqa-full/kg
    python -m xgraph.mintqa_bench questions --source data/external/mintqa-full/kg-source \\
        --kg data/external/mintqa-full/kg \\
        --split ti=data/external/mintqa-full/questions/ti.jsonl \\
        --split pop=data/external/mintqa-full/questions/pop.jsonl --out target/mintqa

Scores come from `python -m xgraph.editing` with `gold.json` as `--gold-new` and no
`--gold-old`: over `questions-representable.json`, and over `questions.json`, where
the dropped questions count as wrong. MINTQA's own measure (`score`, paper_accuracy:
the dataset's answer contained in the final answer) uses the dataset's answer strings,
so it scores every question, a question without an output as wrong:

    python -m xgraph.mintqa_bench score --outputs target/mintqa/qwen25-A.jsonl \\
        --questions target/mintqa/questions.json \\
        --split ti=data/external/mintqa-full/questions/ti.jsonl \\
        --split pop=data/external/mintqa-full/questions/pop.jsonl \\
        --out target/mintqa/qwen25-A.mintqa.json
"""

from __future__ import annotations

import argparse
import json
import re
from array import array
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from xgraph.score import record_text

ID_FILE = "mintqa_wikidata_w_id.txt"
NAME_FILE = "mintqa_wikidata_w_name.txt"
RELATION_FILE = "relID2name_dict.json"
ID_LINE = re.compile(r"Q(\d+)\tP(\d+)\tQ(\d+)")
PLACEHOLDER = re.compile(r"Q\d+")
STRATA = {"ti": ("O", "N", "old"), "pop": ("F", "R", "popular")}
"""Per split: the hop type counted in the stratum, the other type, and the count's name."""

Labels = tuple[str, str, str]
Triple = tuple[str, str, str]


def _relation_names(source: Path) -> dict[int, str]:
    names = json.loads((source / RELATION_FILE).read_text(encoding="utf-8"))
    return {int(p[1:]): label for p, label in names.items()}


def _lines(source: Path) -> Iterator[tuple[int, tuple[int, int, int] | None, list[str]]]:
    """Both files together: line number, the id line's numbers (None unless it holds
    three ids), and the name line's fields, its relation checked against the id line's."""
    relations = _relation_names(source)
    with (
        (source / ID_FILE).open(encoding="utf-8") as ids,
        (source / NAME_FILE).open(encoding="utf-8") as names,
    ):
        n = 0
        try:
            for n, (id_line, name_line) in enumerate(zip(ids, names, strict=True), start=1):
                found = ID_LINE.fullmatch(id_line.rstrip("\n"))
                fields = name_line.rstrip("\n").split("\t")
                if found is None:
                    yield n, None, fields
                    continue
                s, p, o = (int(x) for x in found.groups())
                label = relations.get(p, f"P{p}")
                if len(fields) != 3 or fields[1] != label:
                    raise ValueError(
                        f"line {n}: {NAME_FILE} has {name_line.strip()!r} where {ID_FILE} "
                        f"has {id_line.strip()!r} (P{p} is {label!r}); the files are out of step"
                    )
                yield n, (s, p, o), fields
        except ValueError as error:
            if "zip()" not in str(error):
                raise
            raise ValueError(f"{ID_FILE} and {NAME_FILE} differ in length after line {n}") from None


def convert_kg(
    source: str | Path, out: str | Path, relation_aliases: str | Path | None = None
) -> dict:
    """The released KG as a KG directory (see the module docstring). relation_aliases, a
    Wikidata relation_labels.jsonl, adds each property's label and aliases there as
    aliases of the KG's properties, after the release's label."""
    from xgraph.provenance import provenance

    source, out = Path(source), Path(out)
    relations = _relation_names(source)
    labels: dict[int, str | None] = {}
    flat = array("i")
    counts: Counter[str] = Counter()
    for n, numbers, fields in _lines(source):
        counts["lines"] += 1
        if numbers is None:
            counts["skipped_lines"] += 1
            continue
        s, _, o = numbers
        flat.extend(numbers)
        for item, text in ((s, fields[0]), (o, fields[2])):
            known = labels.get(item)
            if PLACEHOLDER.fullmatch(text):
                labels[item] = known
            elif known is None:
                labels[item] = text
            elif known != text:
                raise ValueError(f"line {n}: Q{item} is named both {known!r} and {text!r}")
    triples = np.frombuffer(flat, dtype=np.int32).reshape(-1, 3)
    unique = np.unique(triples, axis=0)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "triples.npy", unique)
    with (out / "entity_labels.jsonl").open("w", encoding="utf-8") as written:
        for item in sorted(labels):
            text = labels[item]
            if text is None:
                counts["entities_labelled_by_qid"] += 1
                text = f"Q{item}"
            written.write(json.dumps([f"Q{item}", text, "label"], ensure_ascii=False) + "\n")
    used = set(np.unique(unique[:, 1]).tolist())
    others: dict[int, list[str]] = {}
    if relation_aliases is not None:
        with Path(relation_aliases).open(encoding="utf-8") as lines:
            for line in lines:
                prop, text, _ = json.loads(line)
                others.setdefault(int(prop[1:]), []).append(text)
    with (out / "relation_labels.jsonl").open("w", encoding="utf-8") as written:
        for pid in sorted(set(relations) | used):
            if pid not in relations:
                counts["relations_labelled_by_pid"] += 1
            label = relations.get(pid, f"P{pid}")
            written.write(json.dumps([f"P{pid}", label, "label"], ensure_ascii=False) + "\n")
            for text in dict.fromkeys(others.get(pid, [])):
                if text != label:
                    counts["relation_aliases"] += 1
                    row = [f"P{pid}", text, "alias"]
                    written.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "artifact_kind": "xgraph.mintqa_kg",
        "version": 1,
        "counts": {
            "lines": counts["lines"],
            "skipped_lines": counts["skipped_lines"],
            "triples": len(unique),
            "duplicate_triples": len(triples) - len(unique),
            "entities": len(labels),
            "entities_labelled_by_qid": counts["entities_labelled_by_qid"],
            "relations": len(set(relations) | used),
            "relations_in_triples": len(used),
            "relations_labelled_by_pid": counts["relations_labelled_by_pid"],
            **({"relation_aliases": counts["relation_aliases"]} if relation_aliases else {}),
        },
        "provenance": provenance(
            [source / ID_FILE, source / NAME_FILE, source / RELATION_FILE]
            + ([Path(relation_aliases)] if relation_aliases else [])
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def match_triples(source: str | Path, wanted: Iterable[Labels]) -> dict[Labels, set[Triple]]:
    """For each wanted label triple, the distinct QID triples of the lines that write it."""
    found: dict[Labels, set[Triple]] = {tuple(t): set() for t in wanted}
    for _, numbers, fields in _lines(Path(source)):
        if numbers is not None and (key := tuple(fields)) in found:
            s, p, o = numbers
            found[key].add((f"Q{s}", f"P{p}", f"Q{o}"))
    return found


def stratum(split: str, types: list[str], hops: int) -> str:
    """How many of the hops are old (TI) or popular (POP), as `<k>/<hops> old`."""
    counted, other, name = STRATA[split]
    unknown = set(types) - {counted, other}
    if unknown:
        raise ValueError(f"{split}: unknown hop types {sorted(unknown)} in {types}")
    if len(types) != hops:
        return "types mismatch"
    return f"{types.count(counted)}/{hops} {name}"


def resolve(
    chain: list[list[str]], matches: dict[Labels, set[Triple]]
) -> tuple[list[Triple], None] | tuple[None, tuple[int, str]]:
    """A label chain's QID chain if it is representable; else None and why not: the
    first failing hop and "no KG triple", "several KG triples", or "disconnected"."""
    resolved: list[Triple] = []
    for k, hop in enumerate(chain, start=1):
        found = matches.get(tuple(hop), set())
        if len(found) != 1:
            return None, (k, "no KG triple" if not found else "several KG triples")
        (triple,) = found
        if resolved and resolved[-1][2] != triple[0]:
            return None, (k, "disconnected")
        resolved.append(triple)
    return resolved, None


def build_questions(splits: dict[str, list[dict]], matches: dict[Labels, set[Triple]]) -> dict:
    """Questions, gold answers, QID chains, and why each dropped question was dropped
    (see the module docstring); `answers` holds each gold answer as the dataset writes it."""
    questions, gold, answers, chains, dropped = [], {}, {}, {}, {}
    for split, rows in splits.items():
        if split not in STRATA:
            raise ValueError(f"unknown split {split!r}; expected one of {sorted(STRATA)}")
        for row_number, row in enumerate(rows):
            qid = f"mintqa-{split}-{row_number}"
            chain = row["triple"]
            if not chain or any(len(hop) != 3 for hop in chain):
                raise ValueError(f"{qid}: not a chain of (subject, relation, object): {chain}")
            resolved, reason = resolve(chain, matches)
            start = f" wd:{resolved[0][0]}" if resolved else ""
            questions.append(
                {
                    "id": qid,
                    "utterance": row["question"],
                    "sparql": f"# question entities:{start}",
                    "complexity": f"{split} {len(chain)}-hop "
                    + stratum(split, row["type"], len(chain)),
                }
            )
            if resolved is None:
                dropped[qid] = reason
                continue
            gold[qid] = {"entities": [resolved[-1][2]], "literals": []}
            answers[qid] = row["answer"]
            chains[qid] = [list(t) for t in resolved]
    failed = {q: f"not representable: hop {k}, {why}" for q, (k, why) in dropped.items()}
    return {
        "questions": questions,
        "gold": {"answers": gold, "failed": failed},
        "answers": answers,
        "chains": chains,
        "dropped": dropped,
    }


def tally(questions: list[dict], dropped: dict[str, tuple[int, str]]) -> dict:
    """Questions, representable ones, and dropped ones by reason, per split and per group."""

    def by(key) -> dict[str, dict]:
        table: dict[str, dict] = {}
        for q in questions:
            row = table.setdefault(key(q), {"questions": 0, "representable": 0, "dropped": {}})
            row["questions"] += 1
            if q["id"] in dropped:
                why = dropped[q["id"]][1]
                row["dropped"][why] = row["dropped"].get(why, 0) + 1
            else:
                row["representable"] += 1
        return dict(sorted(table.items()))

    return {
        "by_split": by(lambda q: q["id"].split("-")[1]),
        "by_group": by(lambda q: q["complexity"]),
    }


def write_questions(built: dict, kg: Path, out: Path) -> None:
    """The question, gold, label, and request files of the module docstring."""
    from xgraph.prompts import wwq_requests

    out.mkdir(parents=True, exist_ok=True)
    questions, dropped = built["questions"], built["dropped"]
    kept = [q for q in questions if q["id"] not in dropped]
    (out / "questions.json").write_text(json.dumps(questions, indent=1) + "\n")
    (out / "questions-representable.json").write_text(json.dumps(kept, indent=1) + "\n")
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
                if json.loads(line)[0] in items:
                    written.write(line)
    for name, path_spans in (("requests-paths.jsonl", True), ("requests-fewshot.jsonl", False)):
        with (out / name).open("w", encoding="utf-8") as written:
            for request in wwq_requests(out / "questions.json", path_spans=path_spans):
                written.write(json.dumps(request, ensure_ascii=False) + "\n")


def answer_contains(said: str | None, answer: str) -> bool:
    """MINTQA's measure for one answer label: the dataset's answer, normalized
    (xgraph.score.normalize), contained in it, normalized; no answer is wrong."""
    from xgraph.score import normalize

    return bool(said) and normalize(answer) in normalize(said)


def _right(texts: dict[str, str], answers: dict[str, str], qid: str) -> bool:
    """MINTQA's measure for one question (see paper_accuracy)."""
    from xgraph.prompts import final_answer

    return answer_contains(final_answer(texts.get(qid, "")), answers[qid])


def dataset_rows(specs: list[str]) -> dict[str, dict]:
    """The dataset's rows by question id, from `<split>=<file>` specs (JSON lines)."""
    rows: dict[str, dict] = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        with Path(path).open(encoding="utf-8") as lines:
            found = [json.loads(line) for line in lines if line.strip()]
        rows.update({f"mintqa-{name}-{k}": row for k, row in enumerate(found)})
    return rows


def relation_clusters(rows: dict[str, dict]) -> dict[str, str]:
    """Each question's cluster: the relation sequence of its chain as the dataset writes
    it (questions sharing one are correlated), whatever the KG mapping."""
    return {qid: " · ".join(hop[1] for hop in row["triple"]) for qid, row in rows.items()}


def compare_arms(
    a: dict[str, str],
    b: dict[str, str],
    answers: dict[str, str],
    questions: list[dict],
    resamples: int = 10_000,
    clusters: dict[str, str] | None = None,
) -> dict:
    """Per split and over all questions, two arms' accuracy under MINTQA's measure and
    a minus b with a paired percentile bootstrap interval over questions, or, given
    clusters (relation_clusters), over whole clusters of questions."""
    from xgraph.score import paired_bootstrap

    splits: dict[str, list[str]] = {"all": [q["id"] for q in questions]}
    for q in questions:
        splits.setdefault(q["id"].split("-")[1], []).append(q["id"])
    result = {}
    for name, ids in splits.items():
        first = [float(_right(a, answers, qid)) for qid in ids]
        second = [float(_right(b, answers, qid)) for qid in ids]
        groups = [clusters[qid] for qid in ids] if clusters is not None else None
        paired = paired_bootstrap(first, second, resamples=resamples, clusters=groups)
        result[name] = {"a": sum(first) / len(ids), "b": sum(second) / len(ids), **paired}
        if groups is not None:
            result[name]["clusters"] = len(set(groups))
    return result


def paper_accuracy(texts: dict[str, str], answers: dict[str, str], questions: list[dict]) -> dict:
    """MINTQA's measure: a question is right when its dataset answer, normalized
    (xgraph.score.normalize), is contained in the final answer's label (the last
    "Answer: ⟦ label⟧" line), normalized; a question without an output or an answer line
    is wrong. Over all the questions given, per split and per group (complexity)."""

    def share(ids: list[str]) -> dict:
        n = sum(_right(texts, answers, qid) for qid in ids)
        return {"questions": len(ids), "accuracy": n / len(ids) if ids else 0.0}

    splits: dict[str, list[str]] = {}
    groups: dict[str, list[str]] = {}
    for q in questions:
        splits.setdefault(q["id"].split("-")[1], []).append(q["id"])
        groups.setdefault(q["complexity"], []).append(q["id"])
    return {
        "all": share([q["id"] for q in questions]),
        "by_split": {name: share(ids) for name, ids in splits.items()},
        "by_group": {name: share(ids) for name, ids in sorted(groups.items())},
    }


def main(argv: list[str] | None = None) -> None:
    from xgraph.provenance import provenance

    parser = argparse.ArgumentParser(prog="python -m xgraph.mintqa_bench", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    kg = commands.add_parser("kg", help="the released KG as a KG directory")
    kg.add_argument("--source", type=Path, required=True, help="directory of the MintQA-KG files")
    kg.add_argument("--out", type=Path, required=True, help="KG directory")
    kg.add_argument(
        "--relation-aliases",
        type=Path,
        help="a Wikidata relation_labels.jsonl whose labels and aliases join as aliases",
    )
    asked = commands.add_parser("questions", help="questions, gold, labels, and requests")
    asked.add_argument(
        "--source", type=Path, required=True, help="directory of the MintQA-KG files"
    )
    asked.add_argument("--kg", type=Path, required=True, help="the converted KG directory")
    asked.add_argument(
        "--split", action="append", required=True, help="<ti|pop>=<questions (JSON lines)>"
    )
    asked.add_argument("--out", type=Path, required=True, help="output directory")
    score = commands.add_parser("score", help="MINTQA's measure over every question")
    score.add_argument("--outputs", type=Path, required=True, help="an arm's records (JSONL)")
    score.add_argument("--questions", type=Path, required=True, help="questions.json (all)")
    score.add_argument(
        "--split", action="append", required=True, help="<ti|pop>=<questions (JSON lines)>"
    )
    score.add_argument("--out", type=Path, required=True, help="scores (JSON)")
    score.add_argument(
        "--compare", type=Path, help="another arm's records: a paired difference per split"
    )
    score.add_argument(
        "--cluster",
        action="store_true",
        help="with --compare: resample clusters of questions sharing a relation sequence",
    )
    args = parser.parse_args(argv)
    if args.command == "kg":
        manifest = convert_kg(args.source, args.out, args.relation_aliases)
        print(json.dumps(manifest["counts"], indent=2))
        return
    if args.command == "score":
        source = dataset_rows(args.split)
        answers = {qid: row["answer"] for qid, row in source.items()}

        def texts_of(path: Path) -> dict[str, str]:
            records = (json.loads(line) for line in path.read_text().splitlines() if line)
            return {r["id"]: record_text(r) for r in records}

        texts = texts_of(args.outputs)
        questions = json.loads(args.questions.read_text(encoding="utf-8"))
        result = {
            "artifact_kind": "xgraph.mintqa_score",
            "version": 1,
            **paper_accuracy(texts, answers, questions),
        }
        if args.compare:
            result["compare"] = {
                "against": str(args.compare),
                **compare_arms(
                    texts,
                    texts_of(args.compare),
                    answers,
                    questions,
                    clusters=relation_clusters(source) if args.cluster else None,
                ),
            }
        inputs = [args.outputs, args.questions] + ([args.compare] if args.compare else [])
        result["provenance"] = provenance(inputs)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({k: result[k] for k in ("all", "by_split")}))
        return
    splits, files = {}, []
    for spec in args.split:
        name, _, path = spec.partition("=")
        if not name or not path:
            parser.error(f"--split takes <name>=<file>, not {spec!r}")
        files.append(Path(path))
        with Path(path).open(encoding="utf-8") as lines:
            splits[name] = [json.loads(line) for line in lines if line.strip()]
    wanted = {tuple(hop) for rows in splits.values() for row in rows for hop in row["triple"]}
    built = build_questions(splits, match_triples(args.source, wanted))
    write_questions(built, args.kg, args.out)
    kept = len(built["questions"]) - len(built["dropped"])
    inputs = [*files, args.source / ID_FILE, args.source / NAME_FILE, args.source / RELATION_FILE]
    manifest = {
        "artifact_kind": "xgraph.mintqa_questions",
        "version": 1,
        "questions": len(built["questions"]),
        "representable": kept,
        **tally(built["questions"], built["dropped"]),
        "provenance": provenance([*inputs, args.kg / "entity_labels.jsonl"]),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("questions", "representable", "by_split")}))


if __name__ == "__main__":
    main()
