"""Drafts the KG could propose (`docs/quality-gate.md`, run setup), measured on outputs.

Since the escape marker is allowed at every token boundary inside a span, no token
there is the only one allowed. What KG-proposed speculative drafts could use instead
is a position where the label branch has exactly one continuation: the KG would
propose that token and the model verify it. This counts, over a generation output,
the span tokens, the positions with a single label continuation, and how often the
constrained arm's greedy choice there was that continuation (the acceptance rate such
drafts would have).

The label branch's continuations are the mask's tokens without the tokens that
continue the span closer, separator, or escape marker; a label token that is itself
a piece of one of those (a lone space, say) is therefore not counted.

    python -m xgraph.drafts --index data/wikidata/gate-index \\
        --tokenizer data/tokenizers/qwen3.8/tokenizer.json \\
        --outputs target/gate/wwq-constrained.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _delimiter_words(tokenizer, syntax, words: int) -> np.ndarray:
    from xgraph.engine import delimiter_tokens

    token_bytes = [tokenizer.token_bytes(t) for t in range(tokenizer.vocab_size)]
    mask = np.zeros(words, dtype=np.uint32)
    for delimiter in (syntax.closer, syntax.separator, syntax.escape):
        _, tokens, _ = delimiter_tokens(delimiter.encode("utf-8"), token_bytes)
        tokens = tokens.astype(np.int64)
        np.bitwise_or.at(mask, tokens >> 5, (1 << (tokens & 31)).astype(np.uint32))
    return mask


def draft_stats(engine, tokenizer, syntax, streams: list[list[int]]) -> dict[str, int]:
    """Span tokens, single-continuation positions, and accepted ones, over the streams."""
    from xgraph.engine import PHASES

    span = PHASES.index("span")
    batch = engine.max_sequences
    counts = {"span_tokens": 0, "single": 0, "accepted": 0}
    delimiters = None
    for start in range(0, len(streams), batch):
        group = streams[start : start + batch]
        engine.reset()
        for step in range(max(len(s) for s in group)):
            engine.fill_mask(len(group))
            mask = engine.mask_host(len(group)).view(np.uint32)
            if delimiters is None:
                delimiters = _delimiter_words(tokenizer, syntax, mask.shape[1])
            phases = engine.states_host(len(group))["phase"]
            label = mask & ~delimiters
            for row, tokens in enumerate(group):
                if step >= len(tokens) or phases[row] != span:
                    continue
                counts["span_tokens"] += 1
                bits = np.unpackbits(label[row].view(np.uint8), bitorder="little")
                found = np.flatnonzero(bits)
                if len(found) == 1:
                    counts["single"] += 1
                    counts["accepted"] += int(found[0]) == tokens[step]
            # Finished streams repeat their last token; their counts are no longer read.
            engine.advance(np.array([t[min(step, len(t) - 1)] for t in group], dtype=np.int32))
    return counts


def main(argv: list[str] | None = None) -> None:
    from xgraph.engine import Engine
    from xgraph.index import load_kg_index
    from xgraph.syntax import SpanSyntax
    from xgraph.tokenizer import ServedTokenizer

    parser = argparse.ArgumentParser(prog="python -m xgraph.drafts", description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="KG index directory")
    parser.add_argument("--tokenizer", type=Path, required=True, help="served tokenizer.json")
    parser.add_argument("--outputs", type=Path, required=True, help="constrained outputs (JSONL)")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args(argv)
    tokenizer = ServedTokenizer.from_file(args.tokenizer)
    engine = Engine(load_kg_index(args.index), tokenizer, SpanSyntax(), args.batch)
    streams = [r["token_ids"] for r in map(json.loads, args.outputs.read_text().splitlines()) if r]
    counts = draft_stats(engine, tokenizer, SpanSyntax(), streams)
    counts["single_share"] = counts["single"] / max(counts["span_tokens"], 1)
    counts["acceptance"] = counts["accepted"] / max(counts["single"], 1)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
