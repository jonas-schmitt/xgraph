"""Decoding on Graphs over a retrieved subgraph, with triple/explanation alternation.

Reimplemented from the paper and inspected release; downloaded DoG code is never imported.
The existing per-request token-trie adapter masks complete triples. Optional host
beams use raw LM logprobs and charge every scoring request, including discarded work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

from xgraph.cor import linked_requests, parse_fallback
from xgraph.gcr import beam_search_path_steps, beam_search_paths
from xgraph.method_prompts import PromptSource
from xgraph.path_trie import TokenTrie, encode, trie_arguments
from xgraph.prompts import SET_INSTRUCTION, parse_answer_set
from xgraph.toolkg import KGStore, store_for, version_stores

DEVIATIONS = (
    "MINTQA SQLite/linked starts and local backbone replace native datasets/checkpoints",
    "retrieval records Stella or the explicit BM25 fallback; no gold filtering",
    "four-hop bidirectional candidates, recorded cap, and connector-preserving top-120",
    "paper endpoint adjacency replaces active release outgoing-head-only expansion",
    "released raw direct_icl_template uses the served tokenizer without a chat wrapper",
    "complete string stops replace Llama first-token stops and answer token 334",
    "explanations consume and charge a double newline or the next numbered opener",
    "numbered triple prefix is generated under the trie rather than prefilled",
    "exact whole-triple trie replaces release suffix trie; repeated triples excluded by identity",
    "raw-logprob host beams use paper sum scores; release HF beams use normalized sequence scores",
    "all questions retained; final answer marker parsed without an extraction LLM",
)
ORACLE_INTERPRETATION = "oracle-retrieval upper bound; not DoG's score"


def method_deviations(arm="dog"):
    return [
        ORACLE_INTERPRETATION + "; mapped gold triples augment the frozen retrieved graph"
        if arm == "dog-gold" and deviation == DEVIATIONS[1]
        else deviation
        for deviation in DEVIATIONS
    ]


def reader_metadata(requests, arm):
    """Prevent oracle graphs from being reported as the gold-free DoG arm."""
    if arm not in ("dog", "dog-gold"):
        raise ValueError(f"unknown DoG reader arm {arm!r}")
    for request in requests:
        oracle = request.get("oracle_retrieval")
        if arm == "dog":
            valid = oracle is None and request.get("arm") != "dog-gold"
        else:
            valid = (
                request.get("arm") == "dog-gold"
                and request.get("interpretation") == ORACLE_INTERPRETATION
                and isinstance(oracle, dict)
                and type(oracle.get("mapped")) is bool
                and type(oracle.get("added_triples")) is int
                and oracle["added_triples"] >= 0
            )
        if not valid:
            raise ValueError(
                f"DoG question {request['id']!r}: {arm} requires matching oracle-retrieval labels"
            )
    if arm == "dog":
        return {}
    mapped = sum(r["oracle_retrieval"]["mapped"] for r in requests)
    return {
        "interpretation": ORACLE_INTERPRETATION,
        "oracle_retrieval": {
            "mapped_questions": mapped,
            "unmapped_questions": len(requests) - mapped,
            "added_triples": sum(r["oracle_retrieval"]["added_triples"] for r in requests),
        },
    }


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 120
    hops: int = 4
    candidate_limit: int = 100_000
    candidate_overflow: str = "error"
    bm25_prefilter: int = 0

    def __post_init__(self):
        for name in ("top_k", "hops"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"DoG retrieval {name} must be a positive integer")
        if type(self.candidate_limit) is not int or self.candidate_limit < 0:
            raise ValueError("DoG candidate_limit must be nonnegative; zero is uncapped")
        if self.candidate_overflow not in ("error", "truncate"):
            raise ValueError("DoG candidate_overflow must be error or truncate")
        if type(self.bm25_prefilter) is not int or self.bm25_prefilter < 0:
            raise ValueError("DoG bm25_prefilter must be nonnegative; zero disables it")


@dataclass(frozen=True)
class DoGConfig:
    beam_width: int = 1
    min_steps: int = 1
    max_steps: int = 5
    max_tokens: int = 512
    triple_tokens: int = 700
    seed: int = 17
    max_prompt_tokens: int | None = None
    max_calls: int | None = None

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "seed" or (value is None and name in ("max_prompt_tokens", "max_calls")):
                continue
            if type(value) is not int or value <= 0:
                raise ValueError(f"DoG {name} must be a positive integer")
        if self.min_steps > self.max_steps:
            raise ValueError("DoG min_steps must not exceed max_steps")


@dataclass(frozen=True)
class DoGPrompts:
    source: PromptSource

    @classmethod
    def load(cls, path: str | Path) -> DoGPrompts:
        source = PromptSource.load(path, ("direct_icl_template",))
        try:
            source.strings["direct_icl_template"].format(graph="G", question="Q")
        except (KeyError, ValueError, IndexError) as error:
            raise ValueError(f"DoG template placeholders {path}: {error}") from error
        return cls(source)

    def render(self, question: str, rows: list[dict]) -> str:
        graph = " | ".join(" -> ".join(row["labels"]) for row in rows)
        return self.source.strings["direct_icl_template"].format(
            graph=graph or "No information provided", question=question
        )


class DoGGraph:
    """Available triples touch any visited endpoint, retaining original orientation."""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        entities = {}
        for row in rows:
            ids, labels = row.get("ids"), row.get("labels")
            if (
                not isinstance(ids, list)
                or len(ids) != 3
                or any(type(i) is not int or i < 0 for i in ids)
            ):
                raise ValueError("DoG triples need three nonnegative integer ids")
            if (
                not isinstance(labels, list)
                or len(labels) != 3
                or any(
                    not isinstance(s, str)
                    or not s.strip()
                    or any(c in s for c in (" -> ", "<", ">", "\n", "\r"))
                    for s in labels
                )
            ):
                raise ValueError(f"unwritable DoG triple labels: {labels!r}")
            for item, label in ((ids[0], labels[0]), (ids[2], labels[2])):
                if item in entities and entities[item] != label:
                    raise ValueError(f"DoG entity {item} has inconsistent canonical labels")
                entities[item] = label
        self.entities = entities

    def available(self, visited: set[int], used: set[int]) -> list[int]:
        return [
            i
            for i, row in enumerate(self.rows)
            if i not in used and (row["ids"][0] in visited or row["ids"][2] in visited)
        ]

    def advance(self, visited: set[int], index: int) -> set[int]:
        s, _, o = self.rows[index]["ids"]
        return visited | {s, o}

    def text(self, index: int, step: int) -> str:
        return f"{step}. < " + " -> ".join(self.rows[index]["labels"]) + " >"

    def trie(self, tokenizer, visited, used, *, step):
        return TokenTrie(
            [encode(tokenizer, self.text(i, step)) for i in self.available(visited, used)], "closed"
        )


def retrieve_subgraph(
    read, topics: list[str], question: str, *, config=None, ranker=None, candidate_index=None
) -> dict:
    """Gold-free BFS candidates and relevance ranking, adding shortest connectors.

    Hubs stream from SQL, so unlabelled rows cannot hide later valid edges. A probed
    extra labelled edge distinguishes an exact fit from truncated extraction. The
    ranker sees only question text and labelled candidate triples, never annotations.
    """
    config = config or RetrievalConfig()
    if not isinstance(topics, list) or any(
        not isinstance(s, str) or not re.fullmatch(r"Q\d+", s) for s in topics
    ):
        raise ValueError("DoG retrieval topics must be QID lists")
    if candidate_index is not None:
        from xgraph.dog_retrieval import rank_prefiltered

        if candidate_index.read is not read:
            raise ValueError("DoG candidate index belongs to a different KG store")
        return rank_prefiltered(candidate_index.prepare(topics, question, config), question, ranker)
    started, started_cpu = time.perf_counter(), time.process_time()
    roots = sorted({int(s[1:]) for s in topics if read.label(int(s[1:])) is not None})
    parents = dict.fromkeys(roots)
    pending = deque((item, 0) for item in roots)
    triples, caps, hop_by_key = {}, [], {}
    while pending:
        item, depth = pending.popleft()
        if depth >= config.hops:
            continue
        overflow = False
        for column in ("s", "o"):
            for triple in read.iter_fact_rows(item, column):
                triple = tuple(triple)
                if triple in triples:
                    continue
                labels = [
                    read.label(triple[0]),
                    read.relation_label(triple[1]),
                    read.label(triple[2]),
                ]
                if not all(labels):
                    continue
                if config.candidate_limit and len(triples) >= config.candidate_limit:
                    overflow = True
                    break
                triples[triple] = {"ids": list(triple), "labels": labels}
                hop_by_key[triple] = depth + 1
                other = triple[2] if column == "s" else triple[0]
                if other not in parents:
                    parents[other] = (item, triple)
                    pending.append((other, depth + 1))
            if overflow:
                break
        if overflow:
            caps.append(
                {
                    "stage": "retrieval_candidates",
                    "kept": len(triples),
                    "at_least": len(triples) + 1,
                }
            )
            break
    if caps and config.candidate_overflow == "error":
        raise ValueError(
            f"DoG candidate limit {config.candidate_limit} truncates the graph; "
            "raise the limit, use zero for uncapped extraction, or explicitly select "
            "candidate_overflow=truncate for an unqualified diagnostic"
        )
    keys = sorted(triples)
    rows = [triples[key] for key in keys]
    prefilter_counts = []
    if config.bm25_prefilter:
        from xgraph.dog_retrieval import bm25_scores

        lexical = [0.0] * len(rows)
        shortlist = []
        for hop in range(1, config.hops + 1):
            indices = [i for i, key in enumerate(keys) if hop_by_key[key] == hop]
            values = bm25_scores(question, [rows[i] for i in indices])
            for i, value in zip(indices, values, strict=True):
                lexical[i] = value
            chosen = sorted(indices, key=lambda i: (-lexical[i], keys[i]))[: config.bm25_prefilter]
            shortlist.extend(chosen)
            prefilter_counts.append(len(chosen))
        if ranker is not None:
            keys = [keys[i] for i in sorted(shortlist)]
            rows = [triples[key] for key in keys]
            scores = list(ranker(question, rows)) if rows else []
            method = "bm25+" + getattr(ranker, "name", "supplied_ranker")
        else:
            scores, method = lexical, "bm25"
    elif ranker is None:
        from rank_bm25 import BM25Okapi

        def words(text):
            return re.findall(r"\w+", text.casefold()) or ["__empty__"]

        scores = (
            list(
                BM25Okapi([words(" ".join(r["labels"])) for r in rows]).get_scores(words(question))
            )
            if rows
            else []
        )
        method = "bm25"
    else:
        scores, method = list(ranker(question, rows)), getattr(ranker, "name", "supplied_ranker")
    if len(scores) != len(rows) or any(not math.isfinite(float(s)) for s in scores):
        raise ValueError("DoG ranker must return one finite score per candidate triple")

    def connector(item):
        path = []
        while parents[item] is not None:
            item, edge = parents[item]
            path.append(edge)
        return list(reversed(path))

    selected, seen = [], set()
    for at in sorted(range(len(rows)), key=lambda i: (-scores[i], keys[i])):
        s, _, o = key = keys[at]
        paths = [connector(item) for item in (s, o) if item in parents]
        path = min(paths, key=lambda p: (sum(e not in seen for e in p), p))
        addition = list(dict.fromkeys([e for e in [*path, key] if e not in seen]))
        if len(seen) + len(addition) > config.top_k:
            continue
        selected.extend(triples[e] for e in addition)
        seen.update(addition)
        if len(seen) == config.top_k:
            break
    # Reject ambiguous serialization before any model is loaded.
    DoGGraph(selected)
    return {
        "graph": selected,
        "config": asdict(config),
        "ranker": method,
        "ranker_metadata": getattr(ranker, "metadata", {}),
        "candidate_triples": len(triples),
        "candidate_complete": not caps,
        "candidate_coverage_fraction": None if caps else 1.0,
        "candidate_triples_lower_bound": len(triples) + bool(caps),
        "prefilter_triples_by_hop": prefilter_counts,
        "embedded_triples": len(rows) if ranker is not None else 0,
        "retained_triples": len(selected),
        "cap_events": caps,
        "graph_sha256": hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
        "model_calls": 0,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "cpu_seconds": time.process_time() - started_cpu,
        "wall_seconds": time.perf_counter() - started,
        "deviation": "BM25 prefilter per BFS hop before dense reranking; inspect ranker provenance"
        if config.bm25_prefilter and ranker is not None
        else "BM25 instead of paper stella_en_400M_v5"
        if ranker is None
        else "Stella query instruction follows the model card; paper leaves it unspecified"
        if method == "stella_en_400M_v5"
        else "supplied ranker; inspect provenance",
    }


STELLA_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "
)


class StellaRanker:
    """Paper triplet text and cosine similarity, with bounded embedding batches."""

    name = "stella_en_400M_v5"

    def __init__(self, encoder, *, batch_size=32, bucket_size=None, metadata=None):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Stella batch_size must be positive")
        bucket_size = batch_size * 16 if bucket_size is None else bucket_size
        if type(bucket_size) is not int or bucket_size < batch_size:
            raise ValueError("Stella bucket_size must be an integer >= batch_size")
        self.encoder, self.batch_size = encoder, batch_size
        self.bucket_size = bucket_size
        self.metadata = metadata or {}

    def __call__(self, question, rows):
        from itertools import islice

        import numpy as np

        def normalized(texts):
            vectors = np.asarray(self.encoder.encode(texts), dtype=np.float32)
            if (
                vectors.ndim != 2
                or vectors.shape[0] != len(texts)
                or not np.isfinite(vectors).all()
            ):
                raise ValueError("Stella encoder must return finite vectors for every text")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if (norms == 0).any():
                raise ValueError("Stella encoder returned a zero vector")
            return vectors / norms

        query = normalized([STELLA_QUERY_PREFIX + question])[0]
        scores = []
        candidates = iter(rows)
        while window := list(islice(candidates, self.bucket_size)):
            texts = ["(" + ", ".join(r["labels"]) + ")" for r in window]
            order = list(range(len(texts)))
            if hasattr(self.encoder, "token_lengths"):
                lengths = self.encoder.token_lengths(texts)
                order.sort(key=lambda i: lengths[i])
            offset = len(scores)
            scores.extend([0.0] * len(texts))
            for at in range(0, len(texts), self.batch_size):
                indices = order[at : at + self.batch_size]
                vectors = normalized([texts[i] for i in indices])
                if vectors.shape[1] != len(query):
                    raise ValueError("Stella query and document dimensions differ")
                for index, score in zip(indices, (vectors @ query).tolist(), strict=True):
                    scores[offset + index] = score
        return scores


def load_stella(path, *, batch_size=32, bucket_size=None, device="cpu", dtype="float32"):
    """Load local tensors with the native Stella implementation; never download code."""
    from xgraph.provenance import provenance
    from xgraph.stella import NativeStella

    path = Path(path).resolve()
    if not (path / "model.safetensors").is_file():
        raise ValueError(
            "Stella needs a local NovaSearch/stella_en_400M_v5 checkpoint "
            "(paper: dunzhang/stella_en_400M_v5), about 1.75 GB with the 1024-dense head"
        )
    encoder = NativeStella(path, device=device, dtype=dtype)
    bucket_size = batch_size * 16 if bucket_size is None else bucket_size
    files = [
        path / name
        for name in (
            "config.json",
            "model.safetensors",
            "1_Pooling/config.json",
            "2_Dense_1024/config.json",
            "2_Dense_1024/model.safetensors",
        )
    ]
    files += sorted(
        p
        for p in path.glob("*")
        if p.is_file() and p.suffix in (".json", ".txt") and p not in files
    )
    return StellaRanker(
        encoder,
        batch_size=batch_size,
        bucket_size=bucket_size,
        metadata={
            "model": "NovaSearch/stella_en_400M_v5",
            "device": str(encoder.device),
            "implementation": "xgraph.stella.NativeStella",
            "dtype": encoder.dtype,
            "pooling_dtype": "float32",
            "normalization_dtype": "float32",
            "batch_size": batch_size,
            "bucket_size": bucket_size,
            "query_prompt": STELLA_QUERY_PREFIX,
            "document_format": "(head, relation, tail)",
            "max_length": 512,
            "dimension": encoder.dimension,
            "provenance": provenance(files),
        },
    )


def retrieval_coverage(requests, chains):
    """Post-retrieval diagnostic only; gold never enters candidate construction."""
    from xgraph.mintqa_panels import indexed

    if not isinstance(chains, dict):
        raise ValueError("DoG gold-chain audit needs a mapping from question ids to chains")
    indexed(requests)  # Refuse duplicate question ids, including unmapped ones.
    rows = []
    for request in requests:
        chain = chains.get(request["id"])
        if chain is not None and (
            not isinstance(chain, (list, tuple))
            or not chain
            or any(
                not isinstance(t, (list, tuple))
                or len(t) != 3
                or any(
                    not isinstance(value, str) or not re.fullmatch(pattern, value)
                    for value, pattern in zip(t, (r"Q\d+", r"P\d+", r"Q\d+"), strict=True)
                )
                for t in chain
            )
        ):
            raise ValueError(
                f"DoG gold-chain audit question {request['id']!r}: "
                "expected nonempty QID/PID/QID triples"
            )
        graph = request["dog_graph"]
        triples = {(f"Q{r['ids'][0]}", f"P{r['ids'][1]}", f"Q{r['ids'][2]}") for r in graph}
        retrieval = request.get("retrieval", {})
        complete = retrieval.get(
            "candidate_complete", False if retrieval.get("cap_events") else None
        )
        rows.append(
            {
                "id": request["id"],
                "mapped": chain is not None,
                "gold_start_linked": bool(chain and chain[0][0] in request.get("topics", [])),
                "gold_chain_retained": all(tuple(t) in triples for t in chain) if chain else None,
                "candidate_truncated": complete is False,
                "candidate_coverage_fraction": 1.0 if complete is True else None,
                "candidate_triples": retrieval.get("candidate_triples"),
                "retained_triples": len(graph),
            }
        )
    return {
        "questions": len(rows),
        "mapped_questions": sum(r["mapped"] for r in rows),
        "complete_gold_chains": sum(r["gold_chain_retained"] is True for r in rows),
        "gold_start_linked": sum(r["gold_start_linked"] for r in rows),
        "candidate_truncated": sum(r["candidate_truncated"] for r in rows),
        "per_question": rows,
        "interpretation": (
            "post-retrieval gold diagnostic; unknown coverage stays null; no accuracy ceiling"
        ),
    }


def gold_requests(requests, chains, store, versions=None):
    """Union frozen retrieved graphs with complete mapped chains for dog-gold.

    Retrieval, its ordering and linked starts stay fixed. Gold additions may exceed
    retrieval's top-k; the reader's beams, steps and token/call caps stay unchanged.
    A chain absent from the mapping or unwritable in this KG adds no triples.
    """
    retrieval_coverage(requests, chains)  # Validate chain syntax and unique ids first.
    result = []
    for request in requests:
        if request.get("oracle_retrieval") is not None:
            raise ValueError(f"dog-gold question {request['id']!r} is already oracle-augmented")
        graph = list(request["dog_graph"])
        DoGGraph(graph)
        chain = chains.get(request["id"])
        read = store_for(request, store, versions)
        additions = []
        reason = None if chain else "gold chain has no KG mapping"
        for triple in chain or []:
            s, p, o = (int(value[1:]) for value in triple)
            labels = [read.label(s), read.relation_label(p), read.label(o)]
            if any(label is None for label in labels) or o not in read.object_rows(s, p):
                reason = "gold chain has a missing triple or label in the supplied KG"
                break
            row = {"ids": [s, p, o], "labels": labels}
            try:
                DoGGraph([row])
            except ValueError:
                reason = "gold chain has unwritable DoG labels"
                break
            additions.append(row)
        mapped = reason is None
        seen = {tuple(row["ids"]) for row in graph}
        if mapped:
            for row in additions:
                key = tuple(row["ids"])
                if key not in seen:
                    graph.append(row)
                    seen.add(key)
        DoGGraph(graph)
        result.append(
            {
                **request,
                "arm": "dog-gold",
                "interpretation": ORACLE_INTERPRETATION,
                "dog_graph": graph,
                "oracle_retrieval": {
                    "mapped": mapped,
                    "unmapped_reason": reason,
                    "added_triples": len(graph) - len(request["dog_graph"]),
                    "retrieved_triples": len(request["dog_graph"]),
                    "augmented_triples": len(graph),
                    "source_graph_sha256": hashlib.sha256(
                        json.dumps(request["dog_graph"], sort_keys=True).encode()
                    ).hexdigest(),
                    "graph_sha256": hashlib.sha256(
                        json.dumps(graph, sort_keys=True).encode()
                    ).hexdigest(),
                },
            }
        )
    return result


def triple_beam(trie, score_next, *, width, max_tokens):
    return beam_search_paths(trie, score_next, k=width, max_tokens=max_tokens).paths


def parse_answer(text: str, *, answer_set=False):
    matches = list(re.finditer(r"(?:the\s+)?answer\s+is\s*\*\s*([^\r\n]+)", text, re.I))
    if matches:
        label = matches[-1][1].strip().removesuffix(".").strip()
        if answer_set:
            return parse_answer_set(label, bare=True)
        return parse_fallback(label)[0] if label else None
    if re.search(r"\bAnswer:\s*", text, re.I):
        if answer_set:
            return parse_answer_set(text)
        return parse_fallback(text)[0]
    return None


def _prompt_tokens(tokenizer, text):
    # utils_icl.py encodes raw completion prompts with add_special_tokens=True.
    # Triples still use path_trie's no-BOS encoding as continuation tokens.
    result = tokenizer.encode(text, add_special_tokens=True)
    return list(result.ids if hasattr(result, "ids") else result)


def run_dog(
    llm,
    requests,
    params_class,
    *,
    prompts: DoGPrompts,
    config=None,
    lora_request=None,
    eos_token_ids=None,
    batch_size=32,
    arm="dog",
) -> list[dict]:
    """Submit one pending call per question per wave through the same engine.

    Question episodes own all beams, stops, budgets and seeds. A batch size of
    one retains serial scheduling, including per-prefix beam-scoring calls.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("DoG batch_size must be a positive integer")
    reader_metadata(requests, arm)
    config = config or DoGConfig()
    tokenizer = llm.get_tokenizer()
    eos = sorted(
        set(
            [
                *(eos_token_ids or []),
                *([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []),
            ]
        )
    )
    records = [None] * len(requests)
    pending = []
    at = 0

    def advance(index, episode, output=None):
        try:
            prompt, params = episode.send(output)
        except StopIteration as done:
            records[index] = done.value
        else:
            pending.append((index, episode, prompt, params))

    while at < len(requests) or pending:
        while at < len(requests) and len(pending) < batch_size:
            advance(at, _episode(requests[at], tokenizer, params_class, prompts, config, eos, arm))
            at += 1
        if not pending:
            continue
        wave, pending = pending, []
        outputs = llm.generate(
            [prompt for _, _, prompt, _ in wave],
            [params for _, _, _, params in wave],
            use_tqdm=False,
            **({"lora_request": lora_request} if lora_request is not None else {}),
        )
        if len(outputs) != len(wave) or any(len(output.outputs) != 1 for output in outputs):
            raise ValueError("DoG expects one completion per charged call")
        for (index, episode, _, _), output in zip(wave, outputs, strict=True):
            advance(index, episode, output)
    return records


def _triple_beam_episode(trie, score_next, *, width, max_tokens):
    search = beam_search_path_steps(trie, k=width, max_tokens=max_tokens)
    values = None
    while True:
        try:
            pending = search.send(values)
        except StopIteration as done:
            return done.value.paths
        values = yield from score_next(pending)


def _episode(request, tokenizer, params_class, prompts, config, eos, arm):
    if not isinstance(request.get("question"), str) or not request["question"].strip():
        raise ValueError(f"DoG request {request['id']!r} needs a nonempty question")
    if not isinstance(request.get("dog_graph"), list):
        raise ValueError(f"DoG request {request['id']!r} needs an explicit dog_graph list")
    graph = DoGGraph(request["dog_graph"])
    topics = request.get("anchor") or request.get("topics") or []
    if not isinstance(topics, list) or any(
        not isinstance(s, str) or not re.fullmatch(r"Q\d+", s) for s in topics
    ):
        raise ValueError("DoG anchor/topics must be QID lists")
    roots = {int(s[1:]) for s in topics if int(s[1:]) in graph.entities}
    question = (
        request["question"] if request["question"].endswith("?") else request["question"] + "?"
    )
    base = prompts.render(question, graph.rows)
    answer_set = request.get("answer_mode") == "set"
    if answer_set:
        base = (
            SET_INSTRUCTION
            + "\nReturn the complete set as a JSON array after the answer marker.\n"
            + base
        )
    turns, trace, completed = [], [], []
    status = "searching"

    def issue(stage, text, request=request, turns=turns, **kwargs):
        nonlocal status
        ids = text if isinstance(text, list) else _prompt_tokens(tokenizer, text)
        if config.max_calls is not None and len(turns) >= config.max_calls:
            status = "call_limit"
            return None
        if config.max_prompt_tokens is not None and len(ids) > config.max_prompt_tokens:
            status = "context_limit"
            return None
        params = params_class(
            **{
                "temperature": 0.0,
                "seed": request.get("seed", config.seed),
                "max_tokens": config.max_tokens,
                **kwargs,
            }
        )
        output = yield {"prompt_token_ids": ids}, params
        completion = output.outputs[0]
        turns.append(
            {
                "stage": stage,
                "prompt_tokens": len(output.prompt_token_ids),
                "generated_tokens": len(completion.token_ids),
                "token_ids": list(completion.token_ids),
                "text": completion.text,
                "finish_reason": completion.finish_reason,
                "prompt_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            }
        )
        if completion.finish_reason == "abort":
            status = "aborted"
            return None
        return completion

    initial = {
        "text": base,
        "visited": roots,
        "used": set(),
        "score": 0.0,
        "selected": [],
        "triple_scores": [],
    }
    beams = [initial]
    for step in range(1, config.max_steps + 1):
        if not beams or (step > config.min_steps and completed) or status != "searching":
            break
        expanded = []
        for beam in beams:
            available = graph.available(beam["visited"], beam["used"])
            if not available:
                continue
            trie = graph.trie(tokenizer, beam["visited"], beam["used"], step=step)
            if config.beam_width == 1:
                result = yield from issue(
                    "triple",
                    beam["text"],
                    # V2 retains finished slots until reuse; each triple step
                    # needs its own immutable trie identity, not the question id.
                    extra_args=trie_arguments(json.dumps(["dog", str(request["id"]), step]), trie),
                    max_tokens=config.triple_tokens,
                    stop=[" >"],
                    include_stop_str_in_output=True,
                    skip_special_tokens=False,
                )
                if result is None:
                    break
                generated = tokenizer.decode(result.token_ids, skip_special_tokens=False)
                choices = [(generated, 0.0)]
            else:

                class DecodeLimit(Exception):
                    pass

                def score_next(rows, beam=beam, turns=turns):
                    values = []
                    for prefix, allowed in rows:
                        row = {}
                        for at in range(0, len(allowed), 128):
                            tokens = sorted(allowed)[at : at + 128]
                            result = yield from issue(
                                "triple_score",
                                [*_prompt_tokens(tokenizer, beam["text"]), *prefix],
                                max_tokens=1,
                                logprob_token_ids=tokens,
                                ignore_eos=True,
                                detokenize=False,
                            )
                            if result is None:
                                raise DecodeLimit
                            if (
                                len(result.token_ids) != 1
                                or not result.logprobs
                                or set(tokens) - result.logprobs[0].keys()
                            ):
                                raise ValueError("DoG beam needs complete one-token raw logprobs")
                            row.update({t: float(result.logprobs[0][t].logprob) for t in tokens})
                            turns[-1]["token_logprobs"] = dict(row)
                        values.append(row)
                    return values

                try:
                    paths = yield from _triple_beam_episode(
                        trie,
                        score_next,
                        width=config.beam_width,
                        max_tokens=config.triple_tokens,
                    )
                except DecodeLimit:
                    break
                if not paths:
                    status = "truncated_triple"
                    break
                choices = [
                    (tokenizer.decode(p.token_ids, skip_special_tokens=False), p.logprob)
                    for p in paths
                ]
            for text, score in choices:
                selected = [i for i in available if graph.text(i, step) == text]
                if not selected:
                    trace.append({"event": "invalid_triple", "text": text, "step": step})
                    status = "invalid_triple"
                    continue
                visited = set(beam["visited"])
                for i in selected:
                    visited = graph.advance(visited, i)
                expanded.append(
                    {
                        "text": beam["text"] + text,
                        "visited": visited,
                        "used": beam["used"] | set(selected),
                        "score": beam["score"] + score,
                        "selected": [*beam["selected"], selected[0]],
                        "triple_scores": [*beam["triple_scores"], score],
                    }
                )
        beams = []
        if status != "searching":
            break
        for beam in sorted(expanded, key=lambda b: (-b["score"], b["selected"]))[
            : config.beam_width
        ]:
            marker = f"\n{step + 1}. <"
            result = yield from issue(
                "explanation",
                beam["text"] + " This tells us",
                stop=["\n\n", marker],
                include_stop_str_in_output=True,
                stop_token_ids=eos,
                skip_special_tokens=False,
            )
            if result is None:
                break
            text = result.text
            terminal = (
                text.endswith("\n\n")
                or parse_answer(text) is not None
                or getattr(result, "stop_reason", None) in eos
            )
            if text.endswith(marker):
                text = text[: -len(marker)] + "\n"
            beam["text"] += " This tells us" + text
            if terminal:
                completed.append(beam)
            else:
                beams.append(beam)
    final, selected = None, initial
    if status == "searching":
        if not completed:
            trace.append({"event": "no_completed_chain", "fallback": "released_direct"})
        candidates = sorted(completed, key=lambda b: (-b["score"], b["selected"])) or [initial]
        answers = []
        for beam in candidates:
            result = yield from issue(
                "answer", beam["text"], stop_token_ids=eos, skip_special_tokens=False
            )
            if result is None:
                break
            parsed = parse_answer(result.text, answer_set=answer_set)
            if parsed is None:
                parsed = parse_answer(beam["text"][len(base) :], answer_set=answer_set)
            answers.append((beam, parsed))
        if answers:
            selected, final = answers[0]
        if status == "searching":
            status = "answered" if final is not None else "invalid_answer"
    retrieval = request.get("retrieval", {})
    return {
        "id": request["id"],
        "arm": arm,
        **(
            {
                "interpretation": ORACLE_INTERPRETATION,
                "oracle_retrieval": request["oracle_retrieval"],
            }
            if arm == "dog-gold"
            else {}
        ),
        "text": ("Answer: " + json.dumps(final or [], ensure_ascii=False))
        if answer_set
        else final or "Answer: ⟦ Unknown⟧",
        **({"answer_mode": "set", "answers": final or []} if answer_set else {}),
        "status": status,
        "config": asdict(config),
        "prompts": prompts.source.metadata(),
        **({"kg_version": request["kg_version"]} if request.get("kg_version") else {}),
        "start": [f"Q{i}" for i in sorted(roots)],
        "selected_triples": selected["selected"],
        "chain_score": selected["score"],
        "triple_scores": selected["triple_scores"],
        "trace": trace,
        "retrieval": retrieval,
        "deviations": method_deviations(arm),
        "stopping": {
            "triple": " >",
            "triple_tokens": encode(tokenizer, " >"),
            "eos_token_ids": eos,
            "description": "double newline or next numbered opener",
        },
        "model_calls": len(turns) + retrieval.get("model_calls", 0),
        "rounds": len(turns),
        "turns": turns,
        "token_ids": [],
        "prompt_tokens": sum(t["prompt_tokens"] for t in turns) + retrieval.get("prompt_tokens", 0),
        "generated_tokens": sum(t["generated_tokens"] for t in turns)
        + retrieval.get("generated_tokens", 0),
        "finish_reason": turns[-1]["finish_reason"] if turns else "length",
        "context_limited": status == "context_limit",
        "budget_spent": status == "call_limit",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="DoG retrieval, audit, or declared dog-gold oracle-retrieval upper bound"
    )
    parser.add_argument("command", choices=("requests", "audit", "gold"))
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--linked-requests", type=Path)
    parser.add_argument("--db", type=Path)
    parser.add_argument(
        "--gold-chains", type=Path, help="audit or dog-gold oracle-retrieval upper bound"
    )
    parser.add_argument("--kg-versions", type=Path)
    parser.add_argument("--top-k", type=int, default=120)
    parser.add_argument("--hops", type=int, default=4)
    parser.add_argument("--candidate-limit", type=int, default=100_000)
    parser.add_argument("--candidate-overflow", choices=("error", "truncate"), default="error")
    parser.add_argument("--retriever", choices=("bm25", "stella", "bm25-stella"), default="bm25")
    parser.add_argument("--bm25-prefilter", type=int, default=2000, help="top N per BFS hop")
    parser.add_argument("--embedding-model", type=Path, help="local Stella checkpoint; no download")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-bucket-size", type=int)
    parser.add_argument("--embedding-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--embedding-dtype", choices=("float32", "bfloat16", "float16"), default="float32"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    from xgraph.cost import StageTimer
    from xgraph.mintqa_panels import read_jsonl
    from xgraph.provenance import provenance

    if args.command == "gold":
        if args.gold_chains is None or args.db is None:
            parser.error("gold needs --gold-chains and --db for the oracle-retrieval upper bound")
        timer = StageTimer()
        inputs = [args.requests, args.gold_chains, args.db, Path(__file__)]
        source_summary = Path(f"{args.requests}.summary.json")
        if source_summary.is_file():
            inputs.append(source_summary)
        if args.out.resolve() in {p.resolve() for p in inputs}:
            parser.error("--out must differ from input paths")
        requests = read_jsonl(args.requests)
        store = KGStore(args.db)
        named = {r["kg_version"] for r in requests if r.get("kg_version")}
        if named and args.kg_versions is None:
            parser.error("versioned dog-gold augmentation needs --kg-versions")
        versions = version_stores(store, args.kg_versions, named) if named else None
        if args.kg_versions:
            from xgraph.kgversions import version_files

            inputs.extend(version_files(args.kg_versions))
        rows = gold_requests(requests, json.loads(args.gold_chains.read_text()), store, versions)
        mapped = sum(r["oracle_retrieval"]["mapped"] for r in rows)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        Path(f"{args.out}.summary.json").write_text(
            json.dumps(
                {
                    "artifact_kind": "xgraph.dog_gold_requests",
                    "arm": "dog-gold",
                    "interpretation": ORACLE_INTERPRETATION,
                    "records": len(rows),
                    "mapped_questions": mapped,
                    "unmapped_questions": len(rows) - mapped,
                    "added_triples": sum(r["oracle_retrieval"]["added_triples"] for r in rows),
                    "model_calls": 0,
                    "provenance": provenance(inputs),
                    **timer.summary(),
                },
                indent=2,
            )
            + "\n"
        )
        return
    if args.command == "audit":
        if args.gold_chains is None:
            parser.error("audit needs --gold-chains; gold is never used for retrieval")
        inputs = [args.requests, args.gold_chains, Path(__file__)]
        if args.out.resolve() in {p.resolve() for p in inputs}:
            parser.error("--out must differ from input paths")
        result = retrieval_coverage(
            read_jsonl(args.requests), json.loads(args.gold_chains.read_text())
        )
        result["provenance"] = provenance(inputs)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as out:
            out.write(json.dumps(result, indent=2) + "\n")
        return
    if args.gold_chains is not None:
        parser.error("--gold-chains requires audit or gold (oracle-retrieval upper bound)")
    if args.linked_requests is None or args.db is None:
        parser.error("requests needs --linked-requests and --db")
    if args.retriever != "bm25" and args.embedding_model is None:
        parser.error(f"--retriever {args.retriever} needs --embedding-model")
    if args.retriever == "bm25" and args.embedding_model is not None:
        parser.error("--embedding-model requires a Stella retriever")
    if args.retriever == "bm25-stella" and args.embedding_dtype != "float32":
        parser.error("--retriever bm25-stella requires --embedding-dtype float32")
    if args.bm25_prefilter <= 0:
        parser.error("--bm25-prefilter must be positive")
    config = RetrievalConfig(
        args.top_k,
        args.hops,
        args.candidate_limit,
        args.candidate_overflow,
        args.bm25_prefilter if args.retriever == "bm25-stella" else 0,
    )
    timer = StageTimer()

    def embedding_ranker():
        return load_stella(
            args.embedding_model,
            batch_size=args.embedding_batch_size,
            bucket_size=args.embedding_bucket_size,
            device=args.embedding_device,
            dtype=args.embedding_dtype,
        )

    ranker = embedding_ranker() if args.retriever == "stella" else None
    inputs = [
        args.requests,
        args.linked_requests,
        args.db,
        Path(__file__),
        Path(__file__).with_name("dog_retrieval.py"),
    ]
    if args.out.resolve() in {p.resolve() for p in inputs}:
        parser.error("--out must differ from input paths")
    requests = linked_requests(read_jsonl(args.requests), read_jsonl(args.linked_requests))
    store = KGStore(args.db)
    named = {r["kg_version"] for r in requests if r.get("kg_version")}
    if named and args.kg_versions is None:
        parser.error("versioned DoG retrieval needs --kg-versions")
    versions = version_stores(store, args.kg_versions, named) if named else None
    candidate_index = None
    if args.retriever == "bm25-stella":
        from xgraph.dog_retrieval import BM25CandidateIndex

        if versions or args.kg_versions:
            parser.error("indexed BM25-Stella currently requires an unversioned KG")
        candidate_index = BM25CandidateIndex(store, [_question(r) for r in requests])
    if args.kg_versions:
        from xgraph.kgversions import version_files

        inputs.extend(version_files(args.kg_versions))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    index_cost = candidate_index.cost if candidate_index is not None else None
    work = (
        _prepared_requests(requests, candidate_index, config, args.out.parent)
        if candidate_index is not None
        else ((request, None) for request in requests)
    )
    candidate_index = None
    count = truncated = 0
    with args.out.open("x", encoding="utf-8") as written:
        for request, prepared in work:
            question = _question(request)
            if prepared is not None:
                from xgraph.dog_retrieval import rank_prefiltered

                if ranker is None:
                    ranker = embedding_ranker()
                retrieved = rank_prefiltered(prepared, question, ranker)
            else:
                retrieved = retrieve_subgraph(
                    store_for(request, store, versions),
                    request["topics"],
                    question,
                    config=config,
                    ranker=ranker,
                )
            graph = retrieved.pop("graph")
            row = {
                "id": request["id"],
                "question": question,
                "topics": request["topics"],
                "dog_graph": graph,
                "retrieval": retrieved,
                **({"kg_version": request["kg_version"]} if request.get("kg_version") else {}),
            }
            written.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            truncated += bool(retrieved["cap_events"])
    Path(str(args.out) + ".summary.json").write_text(
        json.dumps(
            {
                "artifact_kind": "xgraph.dog_requests",
                "records": count,
                "candidate_truncated": truncated,
                "config": asdict(config),
                "retriever": args.retriever,
                "ranker_metadata": getattr(ranker, "metadata", {}),
                "index_cost": index_cost,
                "model_calls": 0,
                "seed": None,
                "provenance": provenance(inputs),
                **timer.summary(),
            },
            indent=2,
        )
        + "\n"
    )


def _question(request):
    return request.get("question") or next(
        m["content"] for m in reversed(request["messages"]) if m["role"] == "user"
    )


def _prepared_requests(requests, index, config, directory):
    """Spool the entire CPU stage before loading CUDA; bound resident Python rows."""
    from tempfile import TemporaryFile

    with TemporaryFile(mode="w+t", encoding="utf-8", dir=directory) as spool:
        for request in requests:
            prepared = index.prepare(request["topics"], _question(request), config)
            spool.write(json.dumps([request, prepared], ensure_ascii=False) + "\n")
        del index
        spool.seek(0)
        for line in spool:
            yield json.loads(line)


if __name__ == "__main__":
    main()
