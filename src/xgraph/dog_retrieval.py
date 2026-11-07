"""Gold-free BM25 candidate prefilter for the MINTQA DoG transfer."""

import math
import time
from array import array
from collections import Counter
from dataclasses import asdict
from itertools import chain

import numpy as np


def words(text, *, empty=True):
    import re

    return re.findall(r"\w+", text.casefold()) or (["__empty__"] if empty else [])


def bm25_scores(question, rows):
    """BM25 with k1=1.5, b=.75 and positive Robertson IDF, within one BFS hop."""
    documents = [Counter(words(" ".join(row["labels"]))) for row in rows]
    if not documents:
        return []
    lengths = np.array([sum(doc.values()) for doc in documents], dtype=float)
    scores = np.zeros(len(documents))
    for term in words(question):
        tf = np.array([doc[term] for doc in documents], dtype=float)
        df = np.count_nonzero(tf)
        idf = np.log1p((len(documents) - df + 0.5) / (df + 0.5))
        scores += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * lengths / lengths.mean()))
    return scores.tolist()


class BM25CandidateIndex:
    """SQL snapshot with label lengths and postings for the supplied query vocabulary.

    Traversal uses every labelled edge. Query vocabulary only limits stored term
    frequencies; it does not change candidates or corpus statistics. No gold enters
    this index. Only shortlisted edges and their connectors need rendered labels.
    """

    def __init__(self, read, questions):
        from xgraph.toolkg import KGStore

        if type(read) is not KGStore:
            raise ValueError("indexed BM25 requires an unversioned KGStore")
        started, cpu = time.perf_counter(), time.process_time()
        self.read = read
        self.questions = set(questions)
        terms = {term for question in self.questions for term in words(question)}
        postings = {term: (array("I"), array("f")) for term in terms}
        items, lengths = array("q"), array("I")
        for item, label in read._con.execute("SELECT item, label FROM labels ORDER BY item"):
            if not label:
                continue
            counts = Counter(words(label, empty=False))
            for term in counts.keys() & terms:
                postings[term][0].append(len(items))
                postings[term][1].append(counts[term])
            items.append(item)
            lengths.append(sum(counts.values()))
        self.items = np.asarray(items)
        self.postings = {
            term: (np.asarray(at), np.asarray(tf)) for term, (at, tf) in postings.items()
        }
        relations = {p: label for p, label in read.relation_labels().items() if label}
        self.props = np.array(sorted(relations), dtype=np.int64)
        prop_counts = [Counter(words(relations[p], empty=False)) for p in self.props]
        self.prop_tf = {term: np.array([c[term] for c in prop_counts]) for term in terms}
        prop_lengths = np.array([sum(c.values()) for c in prop_counts], dtype=np.uint32)
        raw = np.fromiter(
            chain.from_iterable(
                read._con.execute("SELECT DISTINCT s, p, o FROM triples ORDER BY s, p, o")
            ),
            dtype=np.int64,
        ).reshape(-1, 3)
        if len(self.items) and len(self.props):
            ends = np.searchsorted(self.items, raw[:, (0, 2)])
            props = np.searchsorted(self.props, raw[:, 1])
            valid = (
                (ends < len(self.items)).all(axis=1)
                & (self.items[np.minimum(ends, len(self.items) - 1)] == raw[:, (0, 2)]).all(axis=1)
                & (props < len(self.props))
                & (self.props[np.minimum(props, len(self.props) - 1)] == raw[:, 1])
            )
            self.edges, self.ends, self.prop_at = (
                raw[valid],
                ends[valid].astype(np.int32),
                props[valid],
            )
        else:
            self.edges = np.empty((0, 3), dtype=np.int64)
            self.ends = np.empty((0, 2), dtype=np.int32)
            self.prop_at = np.empty(0, dtype=np.int32)
        self.lengths = np.asarray(lengths)[self.ends].sum(axis=1) + prop_lengths[self.prop_at]
        self.tokenless = self.lengths == 0
        self.lengths[self.tokenless] = 1
        self.cost = {
            "wall_seconds": time.perf_counter() - started,
            "cpu_seconds": time.process_time() - cpu,
        }

    def prepare(self, topics, question, config):
        """Extract the full radius, then save top-N per hop plus shortest connectors."""
        if question not in self.questions:
            raise ValueError("question was not indexed in the BM25 query vocabulary")
        if not config.bm25_prefilter:
            raise ValueError("indexed BM25 needs a positive bm25_prefilter")
        started, cpu = time.perf_counter(), time.process_time()
        roots = np.searchsorted(self.items, [int(q[1:]) for q in topics])
        roots = [
            int(i)
            for i, q in zip(roots, topics, strict=True)
            if i < len(self.items) and self.items[i] == int(q[1:])
        ]
        distance = np.full(len(self.items), config.hops + 1, dtype=np.int32)
        distance[roots] = 0
        parent = np.full(len(self.items), len(self.edges), dtype=np.int64)
        parent[roots] = -1
        s, o = self.ends.T
        for depth in range(config.hops):
            hit = np.flatnonzero((distance[s] == depth) | (distance[o] == depth))
            if not len(hit):
                break
            for column in (0, 1):
                nodes = self.ends[hit, column]
                new = distance[nodes] == config.hops + 1
                np.minimum.at(parent, nodes[new], hit[new])
            distance[(distance == config.hops + 1) & (parent < len(self.edges))] = depth + 1
        edge_hops = np.minimum(distance[s], distance[o]) + 1
        candidates = np.flatnonzero(edge_hops <= config.hops)
        if config.candidate_limit and len(candidates) > config.candidate_limit:
            raise ValueError(
                f"DoG candidate limit {config.candidate_limit} truncates the indexed graph; "
                "use zero for uncapped extraction"
            )
        scores = np.zeros(len(self.edges))
        groups = [np.flatnonzero(edge_hops == hop) for hop in range(1, config.hops + 1)]
        averages = [self.lengths[g].mean() if len(g) else 1 for g in groups]
        for term, multiplicity in Counter(words(question)).items():
            node_tf = np.zeros(len(self.items), dtype=np.float32)
            at, tf = self.postings[term]
            node_tf[at] = tf
            freq = node_tf[s] + node_tf[o] + self.prop_tf[term][self.prop_at]
            if term == "__empty__":
                freq = freq + self.tokenless
            for group, avg in zip(groups, averages, strict=True):
                matches = group[freq[group] > 0]
                df = len(matches)
                idf = np.log1p((len(group) - df + 0.5) / (df + 0.5))
                scores[matches] += (
                    multiplicity
                    * idf
                    * freq[matches]
                    * 2.5
                    / (freq[matches] + 1.5 * (0.25 + 0.75 * self.lengths[matches] / avg))
                )

        def top(group, limit):
            if len(group) > limit:
                threshold = np.partition(scores[group], len(group) - limit)[len(group) - limit]
                better = group[scores[group] > threshold]
                ties = group[scores[group] == threshold][: limit - len(better)]
                group = np.concatenate((better, ties))
            return group[np.lexsort((group, -scores[group]))]

        rendered, parents = {}, {}

        def row(at):
            at = int(at)
            if at not in rendered:
                head, prop, tail = map(int, self.edges[at])
                rendered[at] = {
                    "ids": [head, prop, tail],
                    "labels": [
                        self.read.label(head),
                        self.read.relation_label(prop),
                        self.read.label(tail),
                    ],
                }
            return rendered[at]

        def connector(node):
            while parent[node] >= 0 and parent[node] < len(self.edges):
                edge = int(parent[node])
                left, right = self.ends[edge]
                previous = int(right if left == node else left)
                parents[str(int(self.items[node]))] = [int(self.items[previous]), row(edge)]
                node = previous

        shortlist = []
        for hop, group in enumerate(groups, 1):
            for rank, at in enumerate(top(group, config.bm25_prefilter)):
                shortlist.append({"row": row(at), "hop": hop, "bm25_rank": rank})
                for node in self.ends[at]:
                    connector(int(node))
        # Full BM25-only control, including low-scoring edges if connector bundles skip.
        order = top(candidates, len(candidates))
        selected, seen = [], set()
        for at in order:
            paths = []
            for node in self.ends[at]:
                connector(int(node))
                paths.append(connector_path(int(self.items[node]), parents))
            path = min(
                paths,
                key=lambda p: (sum(tuple(e["ids"]) not in seen for e in p), [e["ids"] for e in p]),
            )
            addition = {tuple(e["ids"]): e for e in [*path, row(at)] if tuple(e["ids"]) not in seen}
            if len(seen) + len(addition) > config.top_k:
                continue
            selected.extend(addition.values())
            seen.update(addition)
            if len(seen) == config.top_k:
                break
        return {
            "question": question,
            "config": asdict(config),
            "candidate_triples": len(candidates),
            "candidate_triples_by_hop": [len(g) for g in groups],
            "candidates": sorted(shortlist, key=lambda c: c["row"]["ids"]),
            "parents": parents,
            "bm25_graph": selected,
            "bm25": {
                "k1": 1.5,
                "b": 0.75,
                "idf": "log1p((n-df+.5)/(df+.5))",
                "corpus": "labelled triples at each minimum BFS hop",
                "tokens": "casefold Unicode word regex; repeated query terms counted",
            },
            "preparation_wall_seconds": time.perf_counter() - started,
            "preparation_cpu_seconds": time.process_time() - cpu,
        }


def connector_path(item, parents):
    path = []
    while str(item) in parents:
        item, edge = parents[str(item)]
        path.append(edge)
    return path[::-1]


def rank_prefiltered(prepared, question, ranker=None, *, limit=None):
    """Rank saved, gold-free candidates; smaller N reuses the nested BM25 prefix."""
    import hashlib
    import json

    from xgraph.dog import DoGGraph

    started, cpu = time.perf_counter(), time.process_time()
    if prepared["question"] != question:
        raise ValueError("prepared BM25 candidates belong to a different question")
    config = dict(prepared["config"])
    limit = config["bm25_prefilter"] if limit is None else limit
    if type(limit) is not int or not 1 <= limit <= config["bm25_prefilter"]:
        raise ValueError("reranking limit must fit the prepared BM25 prefix")
    config["bm25_prefilter"] = limit
    rows = [c["row"] for c in prepared["candidates"] if c["bm25_rank"] < limit]
    if ranker is None:
        selected, method = prepared["bm25_graph"], "bm25"
    else:
        scores = list(ranker(question, rows)) if rows else []
        if len(scores) != len(rows) or any(not math.isfinite(float(s)) for s in scores):
            raise ValueError("DoG ranker must return one finite score per candidate triple")
        selected, seen = [], set()
        for at in sorted(range(len(rows)), key=lambda i: (-scores[i], rows[i]["ids"])):
            paths = [connector_path(rows[at]["ids"][i], prepared["parents"]) for i in (0, 2)]
            path = min(
                paths,
                key=lambda p: (sum(tuple(e["ids"]) not in seen for e in p), [e["ids"] for e in p]),
            )
            addition = {
                tuple(e["ids"]): e for e in [*path, rows[at]] if tuple(e["ids"]) not in seen
            }
            if len(seen) + len(addition) > config["top_k"]:
                continue
            selected.extend(addition.values())
            seen.update(addition)
            if len(seen) == config["top_k"]:
                break
        method = "bm25+" + getattr(ranker, "name", "supplied_ranker")
    DoGGraph(selected)
    return {
        "graph": selected,
        "config": config,
        "ranker": method,
        "ranker_metadata": {"bm25": prepared["bm25"], **getattr(ranker, "metadata", {})},
        "candidate_triples": prepared["candidate_triples"],
        "candidate_complete": True,
        "candidate_coverage_fraction": 1.0,
        "candidate_triples_lower_bound": prepared["candidate_triples"],
        "candidate_triples_by_hop": prepared["candidate_triples_by_hop"],
        "prefilter_triples_by_hop": [min(n, limit) for n in prepared["candidate_triples_by_hop"]],
        "embedded_triples": len(rows) if ranker is not None else 0,
        "retained_triples": len(selected),
        "cap_events": [],
        "graph_sha256": hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
        "model_calls": 0,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "cpu_seconds": prepared["preparation_cpu_seconds"] + time.process_time() - cpu,
        "wall_seconds": prepared["preparation_wall_seconds"] + time.perf_counter() - started,
        "deviation": "BM25 prefilter per BFS hop before Stella; positive Robertson IDF; "
        "lexicographically first shortest parent; connector-preserving hard top-k"
        if ranker is not None
        else "BM25-only control; positive Robertson IDF",
    }
