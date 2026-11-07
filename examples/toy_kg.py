"""CPU: build a tokenizer-bound index, resolve spans, and bind a path's answer.

Run from the repository: .venv/bin/python examples/toy_kg.py
Pass --tokenizer /path/to/tokenizer.json to use a model's real tokenizer, without weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from xgraph.index import build_kg_index, load_kg_index, save_kg_index
from xgraph.kg import Entity, KnowledgeGraph, Relation
from xgraph.oracle import Oracle, TokenNotAllowed
from xgraph.syntax import SpanSyntax
from xgraph.tokenizer import ServedTokenizer


def graph(*, numeric=False):
    """These labels and triples belong to our application, independently of Wikidata."""
    entities = (
        ("Q1", "Q2", "Q3", "Q4")
        if numeric
        else ("item:atlas", "person:ada", "place:cedar", "place:delta")
    )
    relations = ("P1", "P2") if numeric else ("rel:creator", "rel:born-in")
    a, b, c, _ = entities
    creator, born = relations
    return KnowledgeGraph(
        entities=tuple(
            Entity(i, (label,))
            for i, label in zip(entities, ("Atlas", "Ada", "Cedar", "Delta"), strict=True)
        ),
        relations=(Relation(creator, ("creator",)), Relation(born, ("born in",))),
        triples=((a, creator, b), (b, born, c)),
    )


def served_tokenizer(path=None):
    if path is not None:
        return ServedTokenizer.from_file(path)
    # A hermetic demonstration vocabulary, not a tokenizer to use with model weights.
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        [" Atlas", " Ada", " Cedar", " Delta", " creator", " born in"],
        trainers.BpeTrainer(
            vocab_size=320,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<eos>"],
            show_progress=False,
        ),
    )
    return ServedTokenizer(tokenizer)


def walk(oracle, tokenizer, state, *pieces):
    # Keep delimiter boundaries and each label's indexed tokenization context.
    # advance() validates every token and returns a new immutable state.
    for piece in pieces:
        for token in tokenizer.encode(piece):
            state = oracle.advance(state, token)
    return state


def rejected(oracle, tokenizer, state, *pieces):
    try:
        walk(oracle, tokenizer, state, *pieces)
    except TokenNotAllowed:
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--out", type=Path, default=Path("target/examples/toy-index"))
    args = parser.parse_args()
    tokenizer, syntax = served_tokenizer(args.tokenizer), SpanSyntax()
    save_kg_index(build_kg_index(graph(), tokenizer, syntax), args.out)
    index = load_kg_index(args.out)

    oracle = Oracle(index, tokenizer, syntax)
    initial = oracle.initial_state()
    entity = walk(oracle, tokenizer, initial, syntax.opener, " Atlas", syntax.closer)
    fact = walk(
        oracle,
        tokenizer,
        initial,
        syntax.opener,
        " Atlas",
        syntax.separator,
        " creator",
        syntax.separator,
        " Ada",
        syntax.closer,
    )
    escaped = walk(
        oracle,
        tokenizer,
        initial,
        syntax.opener,
        syntax.escape,
        " outside this graph",
        syntax.closer,
    )

    paths = Oracle(index, tokenizer, syntax, paths=True, committed_paths=True)
    state = walk(
        paths,
        tokenizer,
        paths.initial_state(),
        syntax.opener,
        " Atlas",
        syntax.separator,
        " creator",
        syntax.separator,
        " born in",
        syntax.closer,
    )
    answer = walk(paths, tokenizer, state, "\nAnswer: ", syntax.opener, " Cedar", syntax.closer)
    anchored = walk(
        paths,
        tokenizer,
        paths.anchored_state(["item:atlas", "item:atlas"]),
        syntax.opener,
        " Atlas",
        syntax.separator,
        " creator",
        syntax.separator,
        " born in",
        syntax.closer,
    )
    rejected_anchors = []
    for anchor in ([], ["unknown:id"]):
        try:
            paths.anchored_state(anchor)
        except ValueError:
            rejected_anchors.append(anchor)
    print(
        json.dumps(
            {
                "entity": entity.last_span.entity_ids,
                "fact": fact.last_span.triples,
                "path": state.last_span.entity_ids,
                "answer": answer.last_span.entity_ids,
                "anchored_path": anchored.last_span.entity_ids,
                "rejected_anchors": rejected_anchors,
                "escape_resolved": escaped.last_span.resolved,
                "rejected_wrong_answer": rejected(
                    paths, tokenizer, state, syntax.opener, " Delta", syntax.closer
                ),
                "allowed_start_tokens": len(oracle.allowed(oracle.opened_state(initial))),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
