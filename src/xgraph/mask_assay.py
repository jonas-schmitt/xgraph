"""Bounded, synthetic, canonical-label mask comparison; never a serving benchmark.

The entry point is ``python -m xgraph.bench assay``. Production masks are untouched.
Text-grammar backends retain their tokenizer bytes and use a charged canonical
filter; their unfiltered extra tokenizations remain visible in the agreement report.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import resource
import subprocess
import time
from dataclasses import dataclass, field
from functools import cache
from importlib import metadata
from pathlib import Path

import numpy as np

from xgraph.index import KGIndex, build_kg_index
from xgraph.kg import Entity, KnowledgeGraph, Relation
from xgraph.oracle import MAX_REACHED, Oracle, TokenNotAllowed
from xgraph.provenance import provenance, sha256_file
from xgraph.syntax import SpanSyntax
from xgraph.tokenizer import ServedTokenizer

SETTINGS = {"paths": True, "committed_paths": True, "escapes": False}
RELATION_LABELS = {"route-a": "route a", "route-b": "route b", "next": "next", "noise": "noise"}
ROOT = Path(os.environ.get("XGRAPH_ASSAY_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_MANIFEST = ROOT / "scripts/systems-assay/manifest.json"


def load_manifest(path: Path) -> dict:
    config = json.loads(path.read_text())
    if (
        config.get("schema") != 1
        or config.get("batches") != [1, 8, 32]
        or config.get("repetitions") != 5
        or config.get("frontier_cap") != MAX_REACHED
        or config.get("answer_row_cap") != MAX_REACHED
        or len(config.get("fixtures", [])) != 24
        or len({f["id"] for f in config["fixtures"]}) != 24
        or config.get("steps", 0) <= config.get("warmup_steps", 0)
    ):
        raise ValueError("not the bounded systems-assay workload manifest")
    return config


def identity(manifest: Path, tokenizer: ServedTokenizer) -> dict:
    sources = [
        ROOT / "src/xgraph/mask_assay.py",
        ROOT / "src/xgraph/bench.py",
        ROOT / "src/xgraph/index.py",
        ROOT / "src/xgraph/oracle.py",
        ROOT / "src/xgraph/engine.py",
        ROOT / "src/xgraph/kgversions.py",
        ROOT / "src/xgraph/tokenizer.py",
        ROOT / "csrc/engine.cu",
        ROOT / "uv.lock",
    ]
    return {
        "manifest_sha256": sha256_file(manifest),
        "repo_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "tokenizer_sha256": tokenizer.fingerprint,
        "sources": {str(p.relative_to(ROOT)): sha256_file(p) for p in sources},
        "packages": {p: metadata.version(p) for p in ("xgrammar", "llguidance", "outlines-core")},
    }


def write_report(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "screen", "run", "native-screen", "reference-screen")
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tokenizer", type=Path, help="optional local served tokenizer JSON")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--screen", type=Path, help="required agreement receipt for timing")
    parser.add_argument(
        "--native-screen", type=Path, help="required CUDA agreement receipt for timing"
    )
    parser.add_argument("--backend", choices=("xgraph", "xgrammar", "llguidance", "outlines_core"))
    args = parser.parse_args(argv)
    config = load_manifest(args.manifest)
    tokenizer = (
        ServedTokenizer.from_file(args.tokenizer) if args.tokenizer else synthetic_tokenizer()
    )
    pin = identity(args.manifest, tokenizer)
    if pin["packages"] != config["packages"]:
        raise ValueError("installed comparator packages differ from the frozen pins")
    inputs = [args.manifest, Path(__file__), ROOT / "src/xgraph/bench.py", ROOT / "uv.lock"]
    if args.tokenizer:
        inputs.append(args.tokenizer)
    result = {
        "artifact_kind": "xgraph.systems_assay",
        "schema": 1,
        "identity": pin,
        "configuration": config,
        "engine_settings": SETTINGS
        | {
            "start_rank": "degree",
            "unquoted_starts": False,
            "relations": "frontier",
            "repeats": True,
            "expressions": False,
            "stop": None,
            "observe": 0,
            "menu": 0,
            "audit": False,
            "end_after_answer": False,
        },
        "admission": {
            "anchor": ["start"],
            "opened": True,
            "prefix": "canonical two-hop path and bound answer opener",
        },
        "tokenizer": {
            "source": str(args.tokenizer) if args.tokenizer else config["tokenizer"],
            "fingerprint": tokenizer.fingerprint,
            "vocab_size": tokenizer.vocab_size,
        },
        "provenance": provenance(inputs),
        "scope": "synthetic pinned-KG bound-label continuations; no model or serving evidence",
    }
    if args.command == "prepare":
        result["status"] = "PREPARED"
    elif args.command == "reference-screen":
        if args.tokenizer:
            raise ValueError("the scalar-oracle CUDA regression uses the hermetic tokenizer")
        spec = next(
            f
            for f in config["fixtures"]
            if f["width"] == 65 and f["aliases"] and f["conflicting_versions"]
        )
        result.update(native_screen({"fixtures": [spec]}, tokenizer))
        result["reference_fixture_ids"] = [spec["id"]]
    elif args.command == "screen":
        results = []
        started = time.perf_counter()
        for name in config["backends"][1:]:
            backend = GrammarBackend(name, tokenizer)
            alternative_mode = "token_ids" if name == "xgrammar" else "text"
            alternative = (
                GrammarBackend(name, tokenizer, grammar_mode=alternative_mode)
                if name != "outlines_core"
                else None
            )
            for spec in config["fixtures"]:
                fixture = make_fixture(spec, tokenizer)
                row = screen_fixture(fixture, backend)
                row.update(
                    screen_batch_fixture(
                        fixture, backend, config["batches"], config["steps"], config["seed"]
                    )
                )
                if row["batch_mismatches"]:
                    row["status"] = "MISMATCH"
                if alternative is not None:
                    row[f"{alternative_mode}_grammar_diagnostic"] = screen_fixture(
                        fixture, alternative
                    )
                row["graph_sha256"] = fixture.index.base_fingerprint
                results.append(row)
                print(
                    f"screen {name} {spec['id']} {row['status']} {row['masks_checked']}", flush=True
                )
        result.update(
            results=results,
            wall_seconds=time.perf_counter() - started,
            status="AGREEMENT" if all(r["status"] == "AGREEMENT" for r in results) else "MISMATCH",
        )
    else:
        if args.screen is None:
            raise ValueError("timing and native checks require an agreement receipt")
        screen = json.loads(args.screen.read_text())
        validate_screen(screen, config, pin)
        if args.command == "native-screen":
            result.update(native_screen(config, tokenizer))
        else:
            if not args.backend:
                parser.error("run requires --backend")
            if args.native_screen is None:
                raise ValueError("timing requires a native agreement receipt")
            native = json.loads(args.native_screen.read_text())
            expected = {f["id"] for f in config["fixtures"]}
            if (
                native.get("status") != "AGREEMENT"
                or native.get("identity") != pin
                or {r["fixture"] for r in native.get("results", [])} != expected
                or any(r["mismatches"] or not r["masks_checked"] for r in native["results"])
            ):
                raise ValueError("timing requires a successful, current native agreement receipt")
            result.update(run_backend(config, tokenizer, screen, args.backend))
    result["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    write_report(args.out, result)
    if result["status"] in {"MISMATCH", "INCOMPLETE"}:
        raise SystemExit(1)


def synthetic_tokenizer() -> ServedTokenizer:
    """Hermetic byte BPE with explicit merges, no training or network access."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {text: i for i, text in enumerate(["<eos>", *alphabet])}
    merges = []
    for word in ("node", "mid", "start", "route", "next", "noise"):
        left = word[0]
        for right in word[1:]:
            combined = left + right
            if combined not in vocab:
                vocab[combined] = len(vocab)
                merges.append((left, right))
            left = combined
        combined = "Ġ" + word
        vocab[combined] = len(vocab)
        merges.append(("Ġ", word))
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(["<eos>"])
    return ServedTokenizer(tokenizer)


@dataclass(frozen=True)
class Support:
    frontier: tuple[str, ...]
    rows: tuple[int, ...]
    sequences: tuple[tuple[int, ...], ...]


@dataclass
class Fixture:
    spec: dict
    tokenizer: ServedTokenizer
    graph: KnowledgeGraph
    index: KGIndex
    edits: dict
    adjacency: dict
    first_rows: dict[str, int]
    entity_rows: dict[str, list[int]]

    def support(self, version: str | None, path: tuple[str, ...]) -> Support:
        """Independent adjacency-list execution: cap before every subsequent hop.

        All starts are unambiguous and anchored, so every endpoint has start rank
        zero. Order is the first label row, followed by the separate answer-row cap.
        No oracle transition or engine query supplies this support.
        """
        if version not in self.adjacency:
            raise ValueError(f"unknown pinned KG version {version!r}")
        frontier = ("start",)
        for relation in path:
            objects = {
                obj
                for subject in frontier
                for obj in self.adjacency[version].get((subject, relation), ())
            }
            frontier = tuple(sorted(objects, key=self.first_rows.__getitem__)[:MAX_REACHED])
        rows = tuple(sorted(row for e in frontier for row in self.entity_rows[e])[:MAX_REACHED])
        return Support(frontier, rows, tuple(self.index.labels.row_tokens(row) for row in rows))

    def prefix(self, path: tuple[str, ...]) -> list[int]:
        """A forced-open path, then a bound answer opener; boundaries stay separate."""
        syntax = self.index.span_syntax
        pieces = [syntax.label_prefix + "start"]
        for relation in path:
            pieces.extend([syntax.separator, syntax.label_prefix + RELATION_LABELS[relation]])
        pieces.extend([syntax.closer, syntax.opener])
        return [token for piece in pieces for token in self.tokenizer.encode(piece)]

    def oracle_state(self, version: str | None, path: tuple[str, ...]):
        oracle = self.oracle()
        state = oracle.opened_state(oracle.anchored_state(["start"], version))
        for token in self.prefix(path):
            state = oracle.advance(state, token)
        return state

    def oracle(self) -> Oracle:
        return Oracle(
            self.index, self.tokenizer, self.index.span_syntax, versions=self.edits, **SETTINGS
        )

    def lanes(self) -> list[tuple[str | None, tuple[str, ...]]]:
        paths = [("route-a", "next")]
        if self.spec["churn"]:
            paths.append(("route-b", "next"))
        versions = [None, "other"] if self.spec["conflicting_versions"] else [None]
        return [(version, path) for version in versions for path in paths]


def make_fixture(spec: dict, tokenizer: ServedTokenizer) -> Fixture:
    """Fixed topology; graph size includes disconnected labelled background nodes."""
    width, count, degree = spec["width"], spec["entities"], spec["degree"]
    if not 1 <= width <= 65 or count < 512 or not 1 <= degree <= 128:
        raise ValueError("fixture needs width 1..65, entities >=512 and degree 1..128")
    entities = [Entity("start", ("start",)), Entity("trap", ("trap",))]
    for i in range(130):
        entities.append(Entity(f"mid:{i}", (f"mid {i:06d}",)))
        labels = (f"node {i:06d}",)
        if spec["aliases"] and i % 2 == 0:
            labels += (f"node {i:06d} longer",)
        entities.append(Entity(f"end:{i}", labels))
    entities.extend(
        Entity(f"bg:{i}", (f"background {i:06d}",)) for i in range(count - len(entities))
    )
    triples = [("start", "route-a", f"mid:{i}") for i in range(width)]
    triples += [("start", "route-b", f"mid:{65 + i}") for i in range(width)]
    for i in range(130):
        triples.append((f"mid:{i}", "next", f"end:{i}"))
        triples.extend((f"mid:{i}", "noise", f"bg:{j}") for j in range(degree))
    graph = KnowledgeGraph(
        entities=tuple(entities),
        relations=tuple(Relation(r, (label,)) for r, label in RELATION_LABELS.items()),
        triples=tuple(sorted(triples)),
    )
    edits = {
        "other": [("start", "route-a", f"mid:{65 + i}") for i in range(width)]
        + [("mid:0", "next", "trap")]
    }
    index = build_kg_index(graph, tokenizer, SpanSyntax())
    rows: dict[str, list[int]] = {}
    for row in range(index.labels.num_rows):
        rows.setdefault(index.labels.owner_id(row), []).append(row)
    adjacency = {}
    for version, changes in [(None, []), *edits.items()]:
        touched = {edge[:2] for edge in changes}
        edges = [edge for edge in graph.triples if edge[:2] not in touched] + changes
        pairs: dict[tuple[str, str], set[str]] = {}
        for s, r, o in edges:
            pairs.setdefault((s, r), set()).add(o)
        adjacency[version] = pairs
    return Fixture(
        spec,
        tokenizer,
        graph,
        index,
        edits,
        adjacency,
        {entity: rr[0] for entity, rr in rows.items()},
        rows,
    )


class CanonicalLanguage:
    """Canonical label trie plus byte-exact, freely tokenized span closer.

    Intersect this allowed set with the text grammar's mask. Never fill missing
    grammar tokens from the trie: an under-admission must fail qualification.
    """

    def __init__(self, sequences, tokenizer: ServedTokenizer, closer: str):
        self.children: list[dict[int, int]] = [{}]
        self.terminals: set[int] = set()
        self.sequences = tuple(sorted(set(sequences)))
        for sequence in self.sequences:
            node = 0
            for token in sequence:
                if token not in self.children[node]:
                    self.children[node][token] = len(self.children)
                    self.children.append({})
                node = self.children[node][token]
            self.terminals.add(node)
        self.closer = closer.encode()
        self.delimiters = delimiter_transitions(tokenizer, self.closer)
        self.vocab = tokenizer.vocab_size
        self.packed_masks = {}

    def initial(self):
        return 0, None

    def allowed(self, state) -> frozenset[int]:
        node, closing = state
        label = self.children[node] if node is not None else {}
        delimiter = self.delimiters.get(closing, {})
        if node in self.terminals:
            delimiter = delimiter | self.delimiters[b""]
        return frozenset(label.keys() | delimiter.keys())

    def advance(self, state, token: int):
        node, closing = state
        new_node = self.children[node].get(token) if node is not None else None
        progress = self.delimiters.get(closing, {}).get(token)
        if node in self.terminals:
            progress = self.delimiters[b""].get(token, progress)
        if new_node is None and progress is None:
            raise ValueError(f"token {token} violates canonical label continuation")
        return new_node, progress

    def finished(self, state) -> bool:
        return state[1] == self.closer

    def packed(self, state) -> np.ndarray:
        if state not in self.packed_masks:
            words = words_from_tokens(self.allowed(state), self.vocab)
            words.flags.writeable = False
            self.packed_masks[state] = words
        return self.packed_masks[state]


@cache
def delimiter_transitions(tokenizer: ServedTokenizer, closer: bytes) -> dict:
    """Tokenizer-wide substructure, shared across support sets and charged at setup."""
    return {
        closer[:pos]: {
            token: closer[:pos] + data
            for token in range(tokenizer.vocab_size)
            if (data := tokenizer.token_bytes(token)) and closer.startswith(closer[:pos] + data)
        }
        for pos in range(len(closer))
    }


def tokens_from_words(words, vocab: int) -> frozenset[int]:
    data = np.asarray(words, dtype="<u4").view(np.uint8)
    return frozenset(np.flatnonzero(np.unpackbits(data, bitorder="little")[:vocab]).tolist())


def words_from_tokens(tokens, vocab: int) -> np.ndarray:
    words = np.zeros((vocab + 31) // 32, dtype=np.uint32)
    for token in tokens:
        words[token >> 5] |= np.uint32(1 << (token & 31))
    return words


@dataclass
class CompiledSupport:
    language: CanonicalLanguage
    grammar: object
    source_bytes: int
    native_bytes: int | None
    canonical_bytes: int
    compile_ms: float
    word_masks: dict = field(default_factory=dict)


class GrammarBackend:
    """A shared tokenizer, native grammar caches, and exact-language support reuse.

    Caches are keyed by canonical support content, allowing reuse across requests
    and KG versions with identical languages. KG-query caches are separately
    keyed by version and path in the replay driver. Native cache counters, where
    unavailable, are reported as unknown; wrapper hits are measured explicitly.
    """

    def __init__(
        self, name: str, tokenizer: ServedTokenizer, *, canonical_filter=True, grammar_mode=None
    ):
        self.name, self.tokenizer = name, tokenizer
        self.canonical_filter = canonical_filter
        self.grammar_mode = grammar_mode or ("token_ids" if name == "llguidance" else "text")
        if self.grammar_mode not in {"text", "token_ids"} or (
            name == "outlines_core" and self.grammar_mode != "text"
        ):
            raise ValueError(f"unsupported grammar mode {self.grammar_mode} for {name}")
        self.module = importlib.import_module(name)
        self.cache: dict[tuple, CompiledSupport] = {}
        self.hits = self.misses = 0
        self.packed_hits = self.packed_misses = 0
        eos = min(tokenizer.special_ids)
        if name == "xgrammar":
            info = self.module.TokenizerInfo(
                [tokenizer.token_bytes(t) or b"" for t in range(tokenizer.vocab_size)],
                stop_token_ids=[eos],
            )
            self.shared = self.module.GrammarCompiler(info, max_threads=1, cache_enabled=True)
            self.executor = self.module.BatchGrammarMatcher(max_threads=1)
        elif name == "llguidance":
            self.shared = self.module.LLTokenizer(tokenizer._tokenizer.to_str(), eos_token=eos)
            self.executor = self.module.LLExecutor(num_threads=1)
            self.numpy_api = importlib.import_module("llguidance.numpy")
        elif name == "outlines_core":
            mapping: dict[bytes, list[int]] = {}
            for token in range(tokenizer.vocab_size):
                if data := tokenizer.token_bytes(token):
                    mapping.setdefault(data, []).append(token)
            self.shared = self.module.Vocabulary(eos, mapping)
        else:
            raise ValueError(f"unknown grammar backend {name!r}")
        delimiter_transitions(tokenizer, "⟧".encode())

    def compile(self, support: Support) -> CompiledSupport:
        key = support.sequences
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        started = time.perf_counter()
        language = (
            CanonicalLanguage(key, self.tokenizer, "⟧")
            if self.canonical_filter or self.grammar_mode == "token_ids"
            else None
        )
        if self.grammar_mode == "token_ids":
            source = token_lark(language)
        else:
            texts = [
                b"".join(self.tokenizer.token_bytes(t) for t in seq).decode() + "⟧" for seq in key
            ]
            # Escape regex metacharacters only, preserving literal spaces.
            source = (
                "(?:"
                + "|".join(re.sub(r"([\\.^$|?*+()\[\]{}])", r"\\\1", text) for text in texts)
                + ")"
            )
        native_bytes = None
        if self.name == "xgrammar":
            grammar = (
                self.shared.compile_lark(source)
                if self.grammar_mode == "token_ids"
                else self.shared.compile_regex(source)
            )
            native_bytes = grammar.memory_size_bytes
        elif self.name == "llguidance":
            definition = (
                self.module.LLMatcher.grammar_from_lark(source)
                if self.grammar_mode == "token_ids"
                else self.module.LLMatcher.grammar_from_regex(source)
            )
            grammar = self.module.LLMatcher(self.shared, definition)
            if grammar.is_error():
                raise ValueError(grammar.get_error())
        else:
            grammar = self.module.Index(source, self.shared)
        # Python allocation of the canonical filter, excluding shared tokenizer tables.
        language = language if self.canonical_filter else None
        canonical_bytes = (
            deep_size((language.children, language.terminals, language.sequences))
            if language is not None
            else 0
        )
        compiled = CompiledSupport(
            language,
            grammar,
            len(source.encode()),
            native_bytes,
            canonical_bytes,
            (time.perf_counter() - started) * 1000,
        )
        self.cache[key] = compiled
        return compiled

    def matcher(self, compiled: CompiledSupport, prefix=()):
        if self.name == "xgrammar":
            native = self.module.GrammarMatcher(compiled.grammar)
        elif self.name == "llguidance":
            native = compiled.grammar.deep_copy()
        else:
            native = self.module.Guide(compiled.grammar)
        matcher = GrammarMatcher(self, compiled, native)
        for token in prefix:
            matcher.advance(token)
        return matcher

    def fill_batch(self, matchers, buffer, *, project=True):
        """Fill pinned or ordinary CPU storage through the released batch APIs."""
        buffer.fill(-1)
        native = [matcher.native for matcher in matchers]
        if self.name == "xgrammar":
            self.executor.batch_fill_next_token_bitmask(native, buffer)
        elif self.name == "llguidance":
            self.numpy_api.fill_next_token_bitmask_par(
                self.executor,
                [(m, i) for i, m in enumerate(native)],
                buffer,
            )
        else:
            for row, matcher in enumerate(matchers):
                buffer[row] = matcher.raw_words().view(np.int32)
        if project and self.canonical_filter:
            for row, matcher in enumerate(matchers):
                buffer[row].view(np.uint32)[:] &= matcher.compiled.language.packed(matcher.state)

    def advance_batch(self, matchers, tokens):
        native = [matcher.native for matcher in matchers]
        if self.name == "xgrammar":
            accepted = self.module.BatchGrammarMatcher.batch_accept_token(native, tokens)
        elif self.name == "llguidance":
            accepted = self.executor.consume_token_par(list(zip(native, tokens, strict=True)))
        else:
            for matcher, token in zip(matchers, tokens, strict=True):
                matcher.native.advance(token)
            accepted = [True] * len(tokens)
        if not all(accepted):
            raise ValueError(f"{self.name} rejected a batched canonical continuation")
        for matcher, token in zip(matchers, tokens, strict=True):
            if matcher.compiled.language is not None:
                matcher.state = matcher.compiled.language.advance(matcher.state, token)


@dataclass
class GrammarMatcher:
    backend: GrammarBackend
    compiled: CompiledSupport
    native: object
    state: tuple = (0, None)
    buffer: object = None

    def raw_words(self) -> np.ndarray:
        backend = self.backend
        vocab = backend.tokenizer.vocab_size
        if backend.name == "xgrammar":
            if self.buffer is None:
                self.buffer = backend.module.allocate_token_bitmask(1, vocab)
            needed = self.native.fill_next_token_bitmask(self.buffer)
            return (
                self.buffer.numpy()[0].view(np.uint32)
                if needed
                else words_from_tokens(range(vocab), vocab)
            )
        if backend.name == "llguidance":
            return np.frombuffer(self.native.compute_bitmask(), dtype="<u4")
        state = self.native.get_state()
        if state not in self.compiled.word_masks:
            backend.packed_misses += 1
            words = words_from_tokens(self.native.get_tokens(), vocab)
            words.flags.writeable = False
            self.compiled.word_masks[state] = words
        else:
            backend.packed_hits += 1
        return self.compiled.word_masks[state]

    def projected_words(self) -> np.ndarray:
        raw = self.raw_words()
        return raw & self.compiled.language.packed(self.state)

    def advance(self, token: int) -> None:
        if self.backend.name == "xgrammar":
            accepted = self.native.accept_token(token)
        elif self.backend.name == "llguidance":
            accepted = self.native.consume_token(token)
        else:
            self.native.advance(token)
            accepted = True
        if not accepted:
            raise ValueError(f"{self.backend.name} rejected canonical token {token}")
        if self.compiled.language is not None:
            self.state = self.compiled.language.advance(self.state, token)


def deep_size(value, seen=None) -> int:
    """Resident Python object bytes; native grammar heaps are reported separately."""
    import sys

    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(deep_size(k, seen) + deep_size(v, seen) for k, v in value.items())
    items = value if isinstance(value, (tuple, list, set, frozenset)) else ()
    return size + sum(deep_size(item, seen) for item in items)


def token_lark(language: CanonicalLanguage) -> str:
    """Native token-ID grammar with shared trie nodes and a shared closer DAG.

    The served vocabulary is unchanged. Token terminals constrain original IDs,
    avoiding both text retokenization and an external canonical filter at runtime.
    Both selected engines support these terminals in their released Lark API.
    """
    lines = ["start: n0"]
    for node, children in enumerate(language.children):
        branches = [f"<[{token}]> n{child}" for token, child in sorted(children.items())]
        if node in language.terminals:
            branches.append("d0")
        lines.append(f"n{node}: " + " | ".join(branches))
    for prefix, transitions in language.delimiters.items():
        branches = []
        for token, following in sorted(transitions.items()):
            suffix = f" d{len(following)}" if following != language.closer else ""
            branches.append(f"<[{token}]>" + suffix)
        lines.append(f"d{len(prefix)}: " + " | ".join(branches))
    return "\n".join(lines)


def screen_fixture(fixture: Fixture, backend: GrammarBackend, *, oracle_vocab_limit=4096) -> dict:
    """Compare complete masks at every canonical label/closer prefix.

    The hermetic panel exhaustively scans the scalar oracle's vocabulary. For a
    served vocabulary, independently executed support defines the exact label
    contract; check oracle transitions for every expected or raw-admitted token.
    The native owner compares all vocabulary bits against this same contract.
    This avoids using Python vocabulary scanning as a performance baseline.
    """
    oracle = fixture.oracle()
    result = {
        "fixture": fixture.spec["id"],
        "backend": backend.name,
        "grammar_mode": backend.grammar_mode,
        "masks_checked": 0,
        "canonical_mismatches": 0,
        "oracle_mismatches": 0,
        "raw_extra_tokens": 0,
        "raw_mismatching_masks": 0,
        "oracle_mode": "full_vocabulary"
        if fixture.tokenizer.vocab_size <= oracle_vocab_limit
        else "transition_contract",
        "versions_checked": list(dict.fromkeys(v for v, _ in fixture.lanes())),
    }
    for version, path in fixture.lanes():
        support = fixture.support(version, path)
        state = oracle.opened_state(oracle.anchored_state(["start"], version))
        for token in fixture.prefix(path):
            state = oracle.advance(state, token)
        if support.rows != tuple(sorted(state.reached)):
            raise ValueError("independent support disagrees with the pinned oracle")
        compiled = backend.compile(support)
        backend.compile(support)  # explicit warm-cache qualification
        language = compiled.language
        pending = [((), language.initial(), state)]
        seen = set()
        while pending:
            prefix, canonical, state = pending.pop()
            identity = (canonical, state)
            if identity in seen:
                continue
            seen.add(identity)
            expected = language.allowed(canonical)
            matcher = backend.matcher(compiled, prefix)
            raw = tokens_from_words(matcher.raw_words(), fixture.tokenizer.vocab_size)
            if result["oracle_mode"] == "full_vocabulary":
                reference = oracle.allowed(state)
            else:
                reference = set()
                for token in expected | raw:
                    try:
                        oracle.advance(state, token)
                        reference.add(token)
                    except TokenNotAllowed:
                        pass
            projected = raw & expected
            result["masks_checked"] += 1
            result["canonical_mismatches"] += projected != reference
            result["oracle_mismatches"] += expected != reference
            result["raw_extra_tokens"] += len(raw - reference)
            result["raw_mismatching_masks"] += raw != reference
            if projected != reference and "first_mismatch" not in result:
                result["first_mismatch"] = {
                    "version": version,
                    "prefix": prefix,
                    "missing": sorted(reference - projected),
                    "extra": sorted(projected - reference),
                }
            for token in expected:
                following = language.advance(canonical, token)
                advanced = oracle.advance(state, token)
                if not language.finished(following):
                    pending.append(((*prefix, token), following, advanced))
    result["cache_hits"] = backend.hits
    result["status"] = (
        "AGREEMENT"
        if not (result["canonical_mismatches"] or result["oracle_mismatches"])
        else "MISMATCH"
    )
    return result


def validate_screen(screen: dict, config: dict, pin: dict) -> None:
    expected = {(name, f["id"]) for name in config["backends"][1:] for f in config["fixtures"]}
    rows = screen.get("results", [])
    if (
        screen.get("status") != "AGREEMENT"
        or screen.get("identity") != pin
        or len(rows) != len(expected)
        or {(r["backend"], r["fixture"]) for r in rows} != expected
        or any(
            r["status"] != "AGREEMENT"
            or r["canonical_mismatches"]
            or r["oracle_mismatches"]
            or r["masks_checked"] <= 0
            or r.get("batch_mismatches") != 0
            or r.get("batches_checked") != config["batches"]
            for r in rows
        )
    ):
        raise ValueError("missing, stale or unsuccessful canonical-token agreement receipt")


def screen_batch_fixture(fixture, backend, batches, steps, seed):
    """Qualify native batch fills and advances on the actual replenished workload."""
    from xgraph.bench import replenished_schedule

    requests = replay_requests(fixture, seed)
    mismatches = raw_mismatches = masks_checked = raw_extra = 0
    references = {}
    for batch in batches:
        matchers, states = {}, {}
        words = np.empty((batch, (fixture.tokenizer.vocab_size + 31) // 32), dtype=np.int32)
        schedule = replenished_schedule(
            [len(r["tokens"]) for r in requests], batch=batch, steps=steps
        )
        for entries in schedule:
            for slot, number, offset in entries:
                request = requests[number % len(requests)]
                if offset == 0:
                    support = fixture.support(request["version"], request["path"])
                    compiled = backend.compile(support)
                    matchers[slot] = backend.matcher(compiled)
                    key = support.sequences
                    if key not in references:
                        references[key] = CanonicalLanguage(key, fixture.tokenizer, "⟧")
                    language = references[key]
                    states[slot] = language, language.initial()
            ordered = [matchers[i] for i in range(batch)]
            backend.fill_batch(ordered, words, project=False)
            for slot, matcher in enumerate(ordered):
                language, state = states[slot]
                expected = language.allowed(state)
                raw = tokens_from_words(words[slot], fixture.tokenizer.vocab_size)
                raw_mismatches += raw != expected
                raw_extra += len(raw - expected)
                if backend.canonical_filter:
                    projected = raw & matcher.compiled.language.allowed(matcher.state)
                else:
                    projected = raw
                mismatches += projected != expected
                masks_checked += 1
            tokens = [requests[n % len(requests)]["tokens"][off] for _, n, off in entries]
            backend.advance_batch(ordered, tokens)
            for slot, token in enumerate(tokens):
                language, state = states[slot]
                states[slot] = language, language.advance(state, token)
    return {
        "batch_mismatches": mismatches,
        "batch_masks_checked": masks_checked,
        "batch_raw_mismatches": raw_mismatches,
        "batch_raw_extra_tokens": raw_extra,
        "batches_checked": batches,
    }


def measure_replenished(backend, requests, *, batch, steps, repeats, warmup) -> dict:
    """One matched replay driver, including replacement admission on the critical path."""
    from xgraph.bench import percentile, replenished_schedule

    raw, all_steps, all_admissions, first_masks = [], [], [], []
    for repetition in range(repeats):
        # Rotate the fixed streams prospectively: batch one also visits each KG version.
        rotation = repetition % len(requests)
        current = requests[rotation:] + requests[:rotation]
        schedule = replenished_schedule(
            [len(r["tokens"]) for r in current], batch=batch, steps=steps
        )
        backend.reset()
        admissions, timings, completed = [], [], 0
        previous_supports, support_ids, replacements, changes = {}, set(), 0, 0
        repetition_started = time.perf_counter()
        for step, entries in enumerate(schedule):
            for slot, number, offset in entries:
                if offset == 0:
                    request = current[number % len(current)]
                    support_id = request.get("support_sha256")
                    if support_id is not None:
                        support_ids.add(support_id)
                    if slot in previous_supports:
                        replacements += 1
                        changes += previous_supports[slot] != support_id
                    previous_supports[slot] = support_id
                    started = time.perf_counter()
                    metrics = backend.admit(slot, request)
                    elapsed = (time.perf_counter() - started) * 1000
                    admissions.append(
                        {
                            "step": step,
                            "slot": slot,
                            "request": number,
                            "variant": (number + rotation) % len(requests),
                            "version": request["version"],
                            "support_sha256": support_id,
                            "wall_ms": elapsed,
                            **metrics,
                        }
                    )
                    all_admissions.append(elapsed)
                    first_masks.append(metrics["first_mask_ms"])
            started = time.perf_counter()
            metrics = backend.step(entries, current)
            elapsed = (time.perf_counter() - started) * 1000
            timings.append({"step": step, "wall_ms": elapsed, "warmup": step < warmup, **metrics})
            if step >= warmup:
                all_steps.append(elapsed)
            completed += sum(
                offset + 1 == len(current[number % len(current)]["tokens"])
                for _, number, offset in entries
            )
        ended = sum(
            offset + 1 == len(current[number % len(current)]["tokens"])
            for _, number, offset in schedule[-1]
        )
        raw.append(
            {
                "repetition": repetition,
                "admissions": admissions,
                "steps": timings,
                "wall_ms": (time.perf_counter() - repetition_started) * 1000,
                "completed_requests": completed,
                "censored_requests": batch - ended,
                "unique_support_sets": len(support_ids),
                "support_set_changes": changes,
                "replacements": replacements,
                "support_set_churn": changes / replacements if replacements else 0.0,
                "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                **backend.metrics(),
            }
        )
    return {
        "batch": batch,
        "repetitions": repeats,
        "steps": steps,
        "active_slot_share": 1.0,
        "raw_repetitions": raw,
        "admission_ms_p50": percentile(all_admissions, 0.5),
        "admission_ms_p95": percentile(all_admissions, 0.95),
        "first_mask_ms_p50": percentile(first_masks, 0.5),
        "first_mask_ms_p95": percentile(first_masks, 0.95),
        "step_ms_p50": percentile(all_steps, 0.5),
        "step_ms_p95": percentile(all_steps, 0.95),
    }


def replay_requests(fixture: Fixture, seed: int) -> list[dict]:
    """Prospectively fixed replay tokens, common to every backend, with variable lengths."""
    requests = []
    lanes = fixture.lanes()
    for number in range(max(8, 2 * len(lanes))):
        version, path = lanes[number % len(lanes)]
        support = fixture.support(version, path)
        tokens = support.sequences[(seed + number) % len(support.sequences)]
        requests.append(
            {
                "variant": number,
                "version": version,
                "path": path,
                "prefix": fixture.prefix(path),
                "support_sha256": support_identity(support.sequences),
                "tokens": [*tokens, *fixture.tokenizer.encode("⟧")],
            }
        )
    return requests


def support_identity(sequences) -> str:
    return hashlib.sha256(json.dumps(sequences, separators=(",", ":")).encode()).hexdigest()


class GPUReplay:
    """Owner-only GPU boundary: preallocated logits and packed-mask transport.

    Replay targets originate on device. Host grammars pay their device-to-host
    readback and packed-mask upload. Timing is synchronous; it cannot price
    overlap with inference. Driver allocation samples are boundary observations,
    while torch's allocation peak is its actual allocator high-water mark.
    """

    def __init__(self, fixture: Fixture, requests: list[dict], *, native=False, grammar=None):
        import torch

        from xgraph.engine import Engine
        from xgraph.kgversions import build_kg_versions

        self.torch, self.fixture, self.native, self.grammar = torch, fixture, native, grammar
        self.batch = 1
        self.free_at_start = torch.cuda.mem_get_info()[0]
        self.device_peak_delta = 0
        self.engine = (
            Engine(
                fixture.index,
                fixture.tokenizer,
                fixture.index.span_syntax,
                32,
                versions=build_kg_versions(fixture.index, fixture.edits),
                **SETTINGS,
            )
            if native
            else None
        )
        torch.cuda.synchronize()
        self.native_resident_bytes = self.free_at_start - torch.cuda.mem_get_info()[0]
        vocab = fixture.tokenizer.vocab_size
        self.logits = torch.zeros((32, vocab), dtype=torch.float32, device="cuda")
        self.slots = torch.arange(32, dtype=torch.int32, device="cuda")
        self.lengths = torch.zeros(32, dtype=torch.int32, device="cuda")
        self.prompts = torch.zeros_like(self.lengths)
        self.history = torch.zeros(
            (32, max(len(r["prefix"]) + len(r["tokens"]) for r in requests)),
            dtype=torch.int32,
            device="cuda",
        )
        targets = np.zeros((len(requests), max(len(r["tokens"]) for r in requests)), np.int32)
        for i, request in enumerate(requests):
            targets[i, : len(request["tokens"])] = request["tokens"]
        self.targets = torch.tensor(targets, device="cuda")
        self.host_masks = torch.empty((32, (vocab + 31) // 32), dtype=torch.int32, pin_memory=True)
        self.device_masks = torch.empty(self.host_masks.shape, dtype=torch.int32, device="cuda")
        self.host_samples = torch.empty(32, dtype=torch.int32, pin_memory=True)
        self.query_cache, self.matchers = {}, {}
        self.query_hits = self.query_misses = 0
        self.state_readback_bytes = 0
        # The same optimized packed-mask application is used for every host engine.
        self.mask_apply = (
            importlib.import_module("xgrammar").apply_token_bitmask_inplace if not native else None
        )
        torch.cuda.synchronize()
        self.observe_memory()

    def observe_memory(self):
        self.device_peak_delta = max(
            self.device_peak_delta, self.free_at_start - self.torch.cuda.mem_get_info()[0]
        )

    def event_ms(self, operation):
        torch = self.torch
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        operation()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    def reset(self):
        self.matchers.clear()
        self.lengths.zero_()
        if self.engine is not None:
            self.engine.reset()
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()

    def admit(self, slot, request):
        torch = self.torch
        started = time.perf_counter()
        if self.native:
            from xgraph.engine import PHASE_ERROR

            # Read the native state matrix once so diagnostic transfer bytes are
            # measured, including fields filtered out by Engine.states_host.
            states = self.engine._native.states_host(self.engine.max_sequences)
            self.state_readback_bytes = states.nbytes
            if np.any(states[:, 0] == PHASE_ERROR):
                raise ValueError("a prior request failed before slot reuse; assay is incomplete")
            slots = self.slots[slot : slot + 1]
            self.engine.end(slots)
            self.engine.begin(slots)
            self.engine.set_version(slot, request["version"])
            self.engine.anchor(slot, ["start"])
            self.engine.open_span(slot)
            tokens = torch.tensor(request["prefix"], dtype=torch.int32, device="cuda")
            self.history[slot, : len(tokens)] = tokens
            self.lengths[slot] = len(tokens)
            self.engine.sync(slots, self.history, self.prompts, self.lengths)
            self.engine.apply_mask(self.logits[:1], slots)
            torch.cuda.synchronize()
            details = {
                "compile_ms": 0.0,
                "support_query_ms": None,
                "h2d_bytes": 4 * (len(tokens) + 2),
                "d2h_bytes": self.state_readback_bytes,
                "support_cache_hit": None,
                "grammar_cache_hit": None,
            }
        else:
            query_started = time.perf_counter()
            key = (request["version"], request["path"])
            query_hit = key in self.query_cache
            if query_hit:
                self.query_hits += 1
            else:
                self.query_misses += 1
                self.query_cache[key] = self.fixture.support(*key)
            query_ms = (time.perf_counter() - query_started) * 1000
            misses = self.grammar.misses
            compiled = self.grammar.compile(self.query_cache[key])
            grammar_hit = self.grammar.misses == misses
            matcher = self.grammar.matcher(compiled)
            self.matchers[slot] = matcher
            self.grammar.fill_batch([matcher], self.host_masks[:1].numpy())
            self.device_masks[:1].copy_(self.host_masks[:1], non_blocking=True)
            self.logits[:1].zero_()
            self.mask_apply(self.logits[:1], self.device_masks[:1])
            torch.cuda.synchronize()
            details = {
                "compile_ms": 0.0 if grammar_hit else compiled.compile_ms,
                "support_query_ms": query_ms,
                "h2d_bytes": self.host_masks.shape[1] * 4,
                "d2h_bytes": 0,
                "support_cache_hit": query_hit,
                "grammar_cache_hit": grammar_hit,
            }
        self.observe_memory()
        return {"first_mask_ms": (time.perf_counter() - started) * 1000, **details}

    def step(self, entries, requests):
        torch, batch = self.torch, len(entries)
        host_started = time.perf_counter()
        indices = torch.tensor(
            [requests[n % len(requests)]["variant"] for _, n, _ in entries], dtype=torch.long
        )
        positions = torch.tensor([offset for _, _, offset in entries], dtype=torch.long)
        metadata_started = time.perf_counter()
        indices_gpu = indices.to("cuda")
        positions_gpu = positions.to("cuda")
        torch.cuda.synchronize()
        metadata_transfer_ms = (time.perf_counter() - metadata_started) * 1000
        sampled = self.targets[indices_gpu, positions_gpu]
        self.logits[:batch].zero_()
        metadata_bytes = indices.numel() * indices.element_size() * 2
        transfer_ms, d2h_bytes = metadata_transfer_ms, 0
        if self.native:
            device_ms = self.event_ms(
                lambda: self.engine.apply_mask(self.logits[:batch], self.slots[:batch])
            )
            row_numbers = self.slots[:batch].long()
            self.history[row_numbers, self.lengths[:batch].long()] = sampled
            self.lengths[:batch] += 1
            device_ms += self.event_ms(
                lambda: self.engine.sync(
                    self.slots[:batch], self.history, self.prompts, self.lengths
                )
            )
            h2d_bytes = metadata_bytes
            host_ms = max(0.0, (time.perf_counter() - host_started) * 1000 - device_ms)
        else:
            ordered = [self.matchers[slot] for slot in range(batch)]
            self.grammar.fill_batch(ordered, self.host_masks[:batch].numpy())
            host_ms = (time.perf_counter() - host_started) * 1000
            transfer_ms += self.event_ms(
                lambda: self.device_masks[:batch].copy_(self.host_masks[:batch], non_blocking=True)
            )
            device_ms = self.event_ms(
                lambda: self.mask_apply(self.logits[:batch], self.device_masks[:batch])
            )
            transfer_ms += self.event_ms(
                lambda: self.host_samples[:batch].copy_(sampled, non_blocking=True)
            )
            d2h_bytes = batch * 4
            advance_started = time.perf_counter()
            self.grammar.advance_batch(ordered, self.host_samples[:batch].tolist())
            host_ms += (time.perf_counter() - advance_started) * 1000
            h2d_bytes = batch * self.host_masks.shape[1] * 4 + metadata_bytes
        self.observe_memory()
        return {
            "host_ms": host_ms,
            "device_ms": device_ms,
            "transfer_ms": transfer_ms,
            "h2d_bytes": h2d_bytes,
            "d2h_bytes": d2h_bytes,
            "replay_metadata_h2d_bytes": metadata_bytes,
            "replay_metadata_h2d_wall_ms": metadata_transfer_ms,
        }

    def metrics(self):
        return {
            "errors": self.engine.errors() if self.engine is not None else 0,
            "device_peak_torch_allocated_bytes": self.torch.cuda.max_memory_allocated(),
            "device_peak_torch_reserved_bytes": self.torch.cuda.max_memory_reserved(),
            "device_peak_observed_delta_bytes": self.device_peak_delta,
            "native_resident_device_delta_bytes": self.native_resident_bytes,
            "query_cache_hits": self.query_hits,
            "query_cache_misses": self.query_misses,
            "grammar_cache_hits": self.grammar.hits if self.grammar else None,
            "grammar_cache_misses": self.grammar.misses if self.grammar else None,
            "native_internal_cache_hits": None,
            "packed_mask_cache_hits": self.grammar.packed_hits if self.grammar else None,
            "packed_mask_cache_misses": self.grammar.packed_misses if self.grammar else None,
            "end_error_check_d2h_bytes": self.state_readback_bytes if self.native else 0,
        }


def run_backend(config, tokenizer, screen, name) -> dict:
    """Owner measurement; callers must already hold an idle GPU reservation."""
    import gc

    import torch

    if not torch.cuda.is_available():
        raise ValueError("owner timing requires CUDA; no CPU stand-in is permitted")
    use_filter = any(
        r["raw_mismatching_masks"] or r["batch_raw_mismatches"]
        for r in screen["results"]
        if r["backend"] == name
    )
    started = time.perf_counter()
    grammar = (
        GrammarBackend(name, tokenizer, canonical_filter=use_filter) if name != "xgraph" else None
    )
    shared_setup_ms = (time.perf_counter() - started) * 1000
    results, setups, support_bytes = [], [], []
    for spec in config["fixtures"]:
        started = time.perf_counter()
        fixture = make_fixture(spec, tokenizer)
        requests = replay_requests(fixture, config["seed"])
        backend = GPUReplay(fixture, requests, native=name == "xgraph", grammar=grammar)
        torch.cuda.synchronize()
        setups.append(
            {
                "fixture": spec["id"],
                "wall_ms": (time.perf_counter() - started) * 1000,
                "graph_sha256": fixture.index.base_fingerprint,
                "graph_entities": len(fixture.graph.entities),
                "graph_edges": len(fixture.graph.triples),
                "setup_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            }
        )
        for batch in config["batches"]:
            row = measure_replenished(
                backend,
                requests,
                batch=batch,
                steps=config["steps"],
                repeats=config["repetitions"],
                warmup=config["warmup_steps"],
            )
            row.update(fixture=spec["id"], backend=name)
            if any(r["errors"] for r in row["raw_repetitions"]):
                raise ValueError("native replay rejected an agreed token stream")
            results.append(row)
            if name == "xgraph":
                support_bytes.append(
                    {
                        "fixture": spec["id"],
                        "active_support_sets": batch,
                        "resident_device_bytes_per_active_support_set": (
                            backend.native_resident_bytes / batch
                        ),
                        "includes_shared_index_and_all_slot_capacity": True,
                        "standalone_support_allocation_bytes": None,
                    }
                )
        if grammar:
            support_bytes = [
                {
                    "support_sha256": support_identity(key),
                    "canonical_rows": len(key),
                    "source_bytes": value.source_bytes,
                    "native_bytes": value.native_bytes,
                    "canonical_python_bytes": value.canonical_bytes,
                    "packed_word_cache_bytes": sum(w.nbytes for w in value.word_masks.values()),
                    "canonical_packed_cache_bytes": sum(
                        w.nbytes for w in value.language.packed_masks.values()
                    )
                    if value.language
                    else 0,
                }
                for key, value in grammar.cache.items()
            ]
        print(f"timing {name} {spec['id']} completed", flush=True)
        del backend, fixture
        gc.collect()
    return {
        "status": "COMPLETED",
        "backend": name,
        "canonical_filter": use_filter,
        "grammar_mode": grammar.grammar_mode if grammar else "GPU indexed execution",
        "shared_setup_ms": shared_setup_ms,
        "fixture_setups": setups,
        "results": results,
        "bytes_per_support_set": support_bytes,
        "caps": {
            "max_sequences": 32,
            "frontier": MAX_REACHED,
            "answer_rows": MAX_REACHED,
            "compiler_threads": 1,
            "rayon_threads": 1,
        },
        "memory_scope": (
            "RSS process high-water; torch allocator peaks; driver samples at boundaries; "
            "native heaps with no size API are null"
        ),
        "transfer_scope": (
            "explicit replay tensors, packed masks, sampled-token readback and native "
            "anchor/prefix copies; metadata identified separately"
        ),
    }


def native_screen(config, tokenizer) -> dict:
    """Owner-only oracle/CUDA agreement, retaining a peer on an incompatible version.

    Uses sparse slots zero and 31, replaces their requests and rechecks the peer
    after every replacement. Every canonical prefix and delimiter tokenization
    is checked, including both caps. A missing extension raises, never skips.
    """
    import gc

    import torch

    from xgraph.engine import Engine
    from xgraph.kgversions import build_kg_versions

    rows = []
    for spec in config["fixtures"]:
        fixture = make_fixture(spec, tokenizer)
        oracle = fixture.oracle()
        engine = Engine(
            fixture.index,
            tokenizer,
            fixture.index.span_syntax,
            32,
            versions=build_kg_versions(fixture.index, fixture.edits),
            **SETTINGS,
        )
        history = torch.zeros((32, 256), dtype=torch.int32, device="cuda")
        lengths = torch.zeros(32, dtype=torch.int32, device="cuda")
        prompts = torch.zeros_like(lengths)
        peers, checked, mismatches = {}, 0, 0
        for lane, (version, path) in enumerate(fixture.lanes() * 2):
            slot = 31 if lane % 2 else 0
            selected = torch.tensor([slot], dtype=torch.int32, device="cuda")
            support = fixture.support(version, path)
            language = CanonicalLanguage(support.sequences, tokenizer, "⟧")
            state = oracle.opened_state(oracle.anchored_state(["start"], version))
            prefix = fixture.prefix(path)
            for token in prefix:
                state = oracle.advance(state, token)
            pending = [((), language.initial(), state)]
            seen = set()
            while pending:
                suffix, canonical, state = pending.pop()
                if (canonical, state) in seen:
                    continue
                seen.add((canonical, state))
                engine.end(selected)
                engine.begin(selected)
                engine.set_version(slot, version)
                engine.anchor(slot, ["start"])
                engine.open_span(slot)
                tokens = [*prefix, *suffix]
                history[slot, : len(tokens)] = torch.tensor(
                    tokens, dtype=torch.int32, device="cuda"
                )
                lengths[slot] = len(tokens)
                engine.sync(selected, history, prompts, lengths)
                peers[slot] = (state, language, canonical)
                slots = sorted(peers)
                logits = torch.zeros((len(slots), tokenizer.vocab_size), device="cuda")
                engine.apply_mask(logits, torch.tensor(slots, dtype=torch.int32, device="cuda"))
                masks = torch.isfinite(logits).cpu().numpy()
                for row, peer in enumerate(slots):
                    peer_state, peer_language, peer_canonical = peers[peer]
                    reference = (
                        oracle.allowed(peer_state)
                        if tokenizer.vocab_size <= 4096
                        else peer_language.allowed(peer_canonical)
                    )
                    checked += 1
                    mismatches += frozenset(np.flatnonzero(masks[row]).tolist()) != reference
                if engine.errors():
                    raise ValueError("native agreement history was rejected")
                for token in language.allowed(canonical):
                    following = language.advance(canonical, token)
                    if not language.finished(following):
                        pending.append(((*suffix, token), following, oracle.advance(state, token)))
        rows.append(
            {
                "fixture": spec["id"],
                "masks_checked": checked,
                "mismatches": mismatches,
                "graph_sha256": fixture.index.base_fingerprint,
                "version_isolation": mismatches == 0,
                "sparse_slots": [0, 31],
            }
        )
        print(f"native-screen {spec['id']} {checked} masks {mismatches} mismatches", flush=True)
        del engine, fixture
        gc.collect()
    return {
        "status": "AGREEMENT" if all(r["mismatches"] == 0 for r in rows) else "MISMATCH",
        "backend": "xgraph",
        "results": rows,
    }
