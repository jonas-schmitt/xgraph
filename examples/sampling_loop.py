"""CPU: mask synthetic logits, sample greedily, then advance one token."""

import json

import numpy as np
from toy_kg import graph, served_tokenizer

from xgraph import Oracle, SpanSyntax, build_kg_index

tokenizer = served_tokenizer()
syntax = SpanSyntax()
index = build_kg_index(graph(), tokenizer, syntax)
oracle = Oracle(index, tokenizer, syntax)
state = oracle.initial_state()
blocked = 0
for piece in (
    syntax.opener,
    " Atlas",
    syntax.separator,
    " creator",
    syntax.separator,
    " Ada",
    syntax.closer,
):
    for desired in tokenizer.encode(piece):
        allowed = oracle.allowed(state)
        logits = np.full(tokenizer.vocab_size, -10.0)
        logits[desired] = 1.0  # A tiny, deterministic stand-in for a model.
        forbidden = next((t for t in range(tokenizer.vocab_size) if t not in allowed), None)
        if forbidden is not None:
            logits[forbidden] = 100.0
            blocked += 1
        mask = np.zeros(tokenizer.vocab_size, dtype=bool)
        mask[list(allowed)] = True
        logits[~mask] = -np.inf
        if not np.isfinite(logits).any():
            raise RuntimeError("no finite admitted logit")
        sampled = int(logits.argmax())
        state = oracle.advance(state, sampled)
assert state.last_span.triples == (("item:atlas", "rel:creator", "person:ada"),)
assert blocked > 0
print(json.dumps({"triples": state.last_span.triples, "blocked_steps": blocked}))
