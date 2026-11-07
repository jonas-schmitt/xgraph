"""Residual-goal STOP: CPU replay of written path-mode records.

The question's answer head must be a word of the just-followed relation's labels
or aliases. No question content stem unused by the consumed relations may match a
relation offered by the new, whole frontier. This is the lookahead brainstorm's
STOP rule, using repair.answer_head/content_stems/stem and plural stripping for
head-word matching. It neither requires a singleton nor consults entity types.

    python -m xgraph.stopping --outputs <paths.jsonl> --questions <questions.json> \\
        --index <index> --tokenizer <tokenizer.json> --db <kg.sqlite> \\
        [--start-rank sitelinks] [--unquoted-starts] \\
        [--kg-versions <versions> --requests <requests.jsonl>] --out <stopped.jsonl>

Only truncated paths get new delivery: keep the model's answer when its label names
an entity in the committed prefix frontier; otherwise use programs.multiplicity_answer,
then MatchedExecutor's best-connected answer if multiplicity has no preference. This
offline simulation cannot know which answer the model would generate after STOP.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xgraph.index import KGIndex
    from xgraph.tokenizer import ServedTokenizer

from xgraph.oracle import MAX_PATH_RELATIONS
from xgraph.programs import MatchedExecutor, multiplicity_answer, parse_path
from xgraph.prompts import final_answer
from xgraph.repair import _WH_HEADS, answer_head, content_stems, stem


def _words(text: str) -> set[str]:
    return {word.rstrip("s") for word in re.findall(r"\w+", text.casefold())}


MAX_STOP_WORDS = 64
MAX_STOP_BYTES = 4096
"""Bounds on a request's relevant raw question words and their UTF-8 bytes."""


@dataclass(frozen=True)
class StopRelations:
    """Host-compiled STOP relations, using one bit per relevant raw question word.

    Raw words, rather than stems, keep start-word removal exact when two question
    words share a stem. A consumed relation covers all words sharing its stems.
    Words no indexed relation could cover cannot block STOP and need no bit.
    Coverage is sparse: an absent relation covers no question words.
    """

    head: frozenset[int]
    residual: frozenset[int]
    words: tuple[str, ...]
    coverage: dict[int, int]

    def used_by_start(self, start: str) -> int:
        found = set(re.findall(r"\w+", start.casefold()))
        return sum(1 << i for i, word in enumerate(self.words) if word in found)

    def used_by_relations(self, relations) -> int:
        mask = 0
        for relation in relations:
            mask |= self.coverage.get(relation, 0)
        return mask


@dataclass(frozen=True)
class RelationTerms:
    """Words and content stems of indexed labels and aliases, indexed once for lookup."""

    words: dict[int, set[str]]
    stems: dict[int, set[str]]
    _by_word: dict[str, set[int]] = field(init=False, repr=False, compare=False)
    _by_stem: dict[str, set[int]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for source, name in ((self.words, "_by_word"), (self.stems, "_by_stem")):
            indexed: dict[str, set[int]] = {}
            for relation, terms in source.items():
                for term in terms:
                    indexed.setdefault(term, set()).add(relation)
            object.__setattr__(self, name, indexed)

    @classmethod
    def from_executor(cls, matched: MatchedExecutor) -> RelationTerms:
        return cls.from_index(matched._index, matched._tokenizer)

    @classmethod
    def from_index(cls, index: KGIndex, tokenizer: ServedTokenizer) -> RelationTerms:
        words: dict[int, set[str]] = {}
        stems: dict[int, set[str]] = {}
        for row in range(index.relations.num_rows):
            tokens = index.relations.row_tokens(row)
            raw = b"".join(tokenizer.token_bytes(t) or b"" for t in tokens)
            name = raw.decode("utf-8", "replace").removeprefix(index.span_syntax.label_prefix)
            relation = int(index.relations.owners[row])
            words.setdefault(relation, set()).update(_words(name))
            stems.setdefault(relation, set()).update(content_stems(name))
        return cls(words, stems)

    def for_question(self, question: str) -> StopRelations:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("residual-goal STOP needs the question text")
        head = answer_head(question)
        heads = frozenset(self._by_word.get(head.rstrip("s"), ())) if head else frozenset()
        if head is None or head in _WH_HEADS or not heads:
            return StopRelations(frozenset(), frozenset(), (), {})
        head_stem = stem(head)
        question_stems = {
            word: normalized
            for word in set(re.findall(r"\w+", question.casefold()))
            for normalized in content_stems(word)
            if normalized in self._by_stem and normalized != head_stem
        }
        words = tuple(sorted(question_stems))
        if (
            len(words) > MAX_STOP_WORDS
            or sum(len(w.encode("utf-8")) for w in words) > MAX_STOP_BYTES
        ):
            raise ValueError(
                f"residual-goal STOP supports at most {MAX_STOP_WORDS} relevant question "
                f"words and {MAX_STOP_BYTES} UTF-8 bytes"
            )
        coverage: dict[int, int] = {}
        for i, word in enumerate(words):
            for relation in self._by_stem[question_stems[word]]:
                coverage[relation] = coverage.get(relation, 0) | (1 << i)
        return StopRelations(heads, frozenset(coverage), words, coverage)


def residual_goal_stop(
    question: str,
    start: str,
    relations: list[str],
    matched: MatchedExecutor,
    *,
    version: str | None = None,
    anchor: list[str] | None = None,
    terms: RelationTerms | None = None,
) -> int | None:
    """The first qualifying proper prefix's hop count, or None.

    Remove the written start's words from question content, accumulate stems from
    all indexed names of consumed relations, and remove the head's stem too. A
    residual stem blocks STOP only if a relation of the whole frontier offers it.
    Head matching is by words, not stems. Execution, alias unions, start ranks,
    frontier limits and KG versions are MatchedExecutor's, without repair. The
    final hop needs no truncation and is not a STOP. Later invalid hops cannot
    veto an earlier STOP. Pass terms to reuse the indexed vocabulary across records.
    """
    head = answer_head(question)
    relations = relations[:MAX_PATH_RELATIONS]
    if head is None or head in _WH_HEADS or len(relations) < 2:
        return None
    terms = terms if terms is not None else RelationTerms.from_executor(matched)
    view = matched._index if version is None else matched._view(version)
    trace: list[dict[int, int]] = []
    matched._walk(view, start, relations, anchor, False, trace)
    content = content_stems(question, set(re.findall(r"\w+", start.casefold())))
    used: set[str] = set()
    for hop, frontier in enumerate(trace, 1):
        if hop == len(relations) or not frontier:
            break
        written = relations[hop - 1]
        followed = matched._rows_owned(matched._index.relations, written)
        last_words = _words(written)
        for relation in followed:
            used.update(terms.stems.get(relation, ()))
            last_words.update(terms.words.get(relation, ()))
        if head.rstrip("s") not in last_words:
            continue
        left = content - used - {stem(head)}
        if not left or not any(
            terms.stems.get(relation, set()) & left for relation in matched._have(view, frontier)
        ):
            return hop
    return None


def rewrite_record(
    record: dict,
    question: str,
    matched: MatchedExecutor,
    store,
    *,
    version: str | None = None,
    anchor: list[str] | None = None,
    terms: RelationTerms | None = None,
) -> dict:
    """Replay one record, recording whether/where it stopped without mutating it.

    A record's own KG version takes precedence over the supplied request version.
    Unchanged paths keep their text and model answer. Truncations use model delivery
    if that answer is still in the committed prefix frontier, else multiplicity delivery
    (ties in sitelink rank when available), else matched executor delivery. The latter
    uses the executor's configured rank, exactly as programs.execute does.
    """
    version = record.get("kg_version") or version
    anchor = record.get("anchor") or anchor
    path = parse_path(record["text"])
    hop = (
        residual_goal_stop(question, *path, matched, version=version, anchor=anchor, terms=terms)
        if path
        else None
    )
    stop = {
        "stopped": hop is not None,
        "hop": hop,
        "original_hops": len(path[1]) if path else None,
        "head": answer_head(question),
        "kg_version": version,
    }
    result = {**record, "stop": stop, "delivery": "model"}
    if hop is None:
        return result
    start, relations = path[0], path[1][:hop]
    reached = matched.execute(start, relations, anchor=anchor, version=version)
    stop["frontier"] = list(reached.reached)
    said = final_answer(record["text"])
    if said and matched.named(said) & set(reached.reached):
        label = said
    else:
        rank = "sitelinks" if matched._index.sitelinks is not None else "degree"
        frontier = matched.derivations(start, relations, anchor=anchor, version=version, rank=rank)
        chosen = multiplicity_answer(frontier, set())
        result["delivery"] = "multiplicity" if chosen else "executor"
        chosen = chosen or reached.answer
        label = store.label(int(chosen[1:])) if chosen else None
        if label is None:
            raise ValueError(
                f"record {record['id']!r}: the KG store has no label for {chosen}, "
                "which STOP delivers; use the store built from the index's KG directory"
            )
    result["text"] = "⟦ " + " · ".join([start, *relations]) + f"⟧.\nAnswer: ⟦ {label}⟧"
    selections = getattr(relations, "selections", {})
    if selections:
        from xgraph.observations import path_prefix

        span = next(
            m
            for m in re.finditer(r"⟦([^⟧]*·[^⟧]*)⟧", record["text"])
            if (parsed := parse_path(m.group())) == path
            and getattr(parsed[1], "selections", {}) == selections
        )
        result["text"] = "⟦" + path_prefix(span.group(1), hop) + f"⟧.\nAnswer: ⟦ {label}⟧"
    # Replace an existing executed plan too, so it cannot describe the discarded suffix.
    if "plan" in result or selections:
        result["plan"] = {"start": start, "relations": relations}
        if selections:
            result["plan"]["selections"] = selections
    return result


def main(argv: list[str] | None = None) -> None:
    from xgraph.index import load_kg_index
    from xgraph.kgversions import load_kg_versions
    from xgraph.provenance import provenance
    from xgraph.tokenizer import ServedTokenizer
    from xgraph.toolkg import KGStore

    parser = argparse.ArgumentParser(prog="python -m xgraph.stopping", description=__doc__)
    parser.add_argument("--outputs", type=Path, required=True, help="path-mode records (JSONL)")
    parser.add_argument(
        "--questions", type=Path, required=True, help="id/utterance questions (JSON)"
    )
    parser.add_argument("--index", type=Path, required=True, help="the run's KG index")
    parser.add_argument("--tokenizer", type=Path, required=True, help="its served tokenizer")
    parser.add_argument("--db", type=Path, required=True, help="KG store for delivered labels")
    parser.add_argument("--start-rank", choices=("degree", "sitelinks"), default="degree")
    parser.add_argument(
        "--unquoted-starts", action="store_true", help="as XGRAPH_UNQUOTED_STARTS=1"
    )
    parser.add_argument("--kg-versions", type=Path, help="KG version directory")
    parser.add_argument(
        "--requests", type=Path, help="requests for KG versions and anchors (JSONL)"
    )
    parser.add_argument("--out", type=Path, required=True, help="rewritten records (JSONL)")
    args = parser.parse_args(argv)
    if args.outputs.resolve() == args.out.resolve():
        parser.error("--out must differ from --outputs")
    started = time.perf_counter()
    questions = {q["id"]: q["utterance"] for q in json.loads(args.questions.read_text())}
    asked = {}
    if args.requests:
        with args.requests.open() as source:
            asked = {r["id"]: r for line in source if line.strip() for r in [json.loads(line)]}
    index = load_kg_index(args.index)
    matched = MatchedExecutor(
        index,
        ServedTokenizer.from_file(args.tokenizer),
        versions=load_kg_versions(args.kg_versions, index) if args.kg_versions else None,
        start_rank=args.start_rank,
        unquoted_starts=args.unquoted_starts,
    )
    terms, store = RelationTerms.from_executor(matched), KGStore(args.db)
    counts, hops, heads, delivery = Counter(), Counter(), Counter(), Counter()
    with args.outputs.open() as source, args.out.open("w", encoding="utf-8") as out:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            qid = record["id"]
            if qid not in questions:
                raise ValueError(f"record {qid!r}: --questions has no question with that id")
            request = asked.get(qid, {})
            version = record.get("kg_version") or request.get("kg_version")
            if version and not args.kg_versions:
                parser.error(f"record {qid!r} reads KG version {version!r}: need --kg-versions")
            row = rewrite_record(
                record,
                questions[qid],
                matched,
                store,
                version=version,
                anchor=request.get("anchor"),
                terms=terms,
            )
            counts["records"] += 1
            if row["stop"]["stopped"]:
                counts["stopped"] += 1
                hops[str(row["stop"]["hop"])] += 1
                heads[row["stop"]["head"]] += 1
            delivery[row["delivery"]] += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "artifact_kind": "xgraph.stopping",
        "version": 1,
        "rule": "residual-goal-stop",
        "records": counts["records"],
        "stopped": counts["stopped"],
        "by_hop": dict(hops),
        "by_head": dict(heads),
        "delivery": dict(delivery),
        "start_rank": args.start_rank,
        "unquoted_starts": args.unquoted_starts,
        "delivery_policy": "model-in-frontier-else-multiplicity-else-executor",
        "multiplicity_tie_rank": "sitelinks" if index.sitelinks is not None else "degree",
        "seed": None,
        "replay_seconds": time.perf_counter() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
    }
    inputs = [args.outputs, args.questions, args.index / "kg-index.json", args.tokenizer]
    if args.requests:
        inputs.append(args.requests)
    if args.kg_versions:
        inputs.extend(sorted(args.kg_versions.glob("*.json")))
        inputs.extend(sorted(args.kg_versions.glob("*.npy")))
    inputs.extend(
        [
            Path(__file__),
            Path(__file__).with_name("programs.py"),
            Path(__file__).with_name("repair.py"),
        ]
    )
    summary["provenance"] = provenance(inputs)
    # Large, read-only KG inputs: record identity without hashing the entire store.
    summary["kg_files"] = [
        {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in [args.db, *sorted(args.index.glob("*.npy"))]
    ]
    Path(f"{args.out}.summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"provenance", "kg_files"}}))


if __name__ == "__main__":
    main()
