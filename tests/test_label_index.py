"""KG index build seam: the label index, observed through its prefix queries."""

from __future__ import annotations

import json

import pytest
from tokenizers import Tokenizer

from tests.conftest import train_byte_level_bpe
from xgraph.index import build_label_index
from xgraph.kg import Entity, KnowledgeGraph
from xgraph.syntax import SpanSyntax
from xgraph.tokenizer import ServedTokenizer

# Hand-listed from TOY_ENTITIES: every (entity, label) pair the index must keep,
# once each, and the pairs it must exclude.
KEPT = {
    ("Q1", "Marie Curie"),
    ("Q1", "Maria Skłodowska"),
    ("Q2", "Pierre Curie"),
    ("Q3", "Paris"),
    ("Q4", "Paris"),
    ("Q5", "Paris Hilton"),
    ("Q6", "Curie"),
    ("Q7", "University of Paris"),
    ("Q8", "Nobel Prize in Physics"),
    ("Q9", "Nobel Prize"),
    ("Q12", "Rhône–Alpes"),
    ("Q17", "Curie"),
}
# "⁇ Unknown" is excluded because the label prefix in front makes " ⁇", the escape marker.
EXCLUDED = {("Q10", "A⟧B"), ("Q11", "   "), ("Q15", "Paris<|endoftext|>"), ("Q16", "⁇ Unknown")}


def encode(tokenizer: Tokenizer, label: str) -> tuple[int, ...]:
    """The tokens of a label as a span writes it: after a space, the default label prefix."""
    return tuple(tokenizer.encode(" " + label, add_special_tokens=False).ids)


def build(kg: KnowledgeGraph, tokenizer: Tokenizer):
    return build_label_index(kg, ServedTokenizer(tokenizer), SpanSyntax())


def rows_in(index, lo: int, hi: int) -> list[tuple[tuple[int, ...], str]]:
    return sorted((index.row_tokens(r), index.owner_id(r)) for r in range(lo, hi))


def test_prefix_range_selects_exactly_the_rows_with_that_prefix(toy_kg, toy_tokenizer):
    index = build(toy_kg, toy_tokenizer)
    expected = [(encode(toy_tokenizer, label), entity) for entity, label in KEPT]
    prefixes = {tokens[:k] for tokens, _ in expected for k in range(len(tokens) + 1)}
    prefixes.add((toy_tokenizer.get_vocab_size() - 1,))  # a prefix no label has
    for prefix in sorted(prefixes):
        lo, hi = index.prefix_range(prefix)
        want = sorted(row for row in expected if row[0][: len(prefix)] == prefix)
        assert rows_in(index, lo, hi) == want, prefix


def test_rows_are_in_lexicographic_token_order(toy_kg, toy_tokenizer):
    index = build(toy_kg, toy_tokenizer)
    rows = [index.row_tokens(r) for r in range(index.num_rows)]
    assert rows == sorted(rows)
    assert index.num_rows == len(KEPT)


def test_ambiguous_label_keeps_every_entity(toy_kg, toy_tokenizer):
    index = build(toy_kg, toy_tokenizer)
    paris = encode(toy_tokenizer, "Paris")
    lo, hi = index.prefix_range(paris)
    complete = {index.owner_id(r) for r in range(lo, hi) if index.row_tokens(r) == paris}
    assert complete == {"Q3", "Q4"}


def test_excluded_labels_are_reported(toy_kg, toy_tokenizer):
    index = build(toy_kg, toy_tokenizer)
    assert {(e.owner_id, e.label) for e in index.excluded} == EXCLUDED


def test_index_records_its_served_tokenizer(toy_kg, toy_tokenizer):
    index = build(toy_kg, toy_tokenizer)
    assert index.tokenizer_fingerprint == ServedTokenizer(toy_tokenizer).fingerprint
    other = train_byte_level_bpe(["something entirely different"], vocab_size=300)
    assert ServedTokenizer(other).fingerprint != index.tokenizer_fingerprint


def label_byte_tokenizer(kind):
    """Small real encoders for label-byte loss, unknown bytes and empty encodings."""
    from tokenizers import decoders, models, normalizers, pre_tokenizers

    from tests.conftest import byte_level_tokenizer
    from tests.test_tokenizer import sentencepiece_bpe

    raw = byte_level_tokenizer([])
    if kind == "missing-byte":
        alphabet = sorted(set(pre_tokenizers.ByteLevel.alphabet()) - {"z"})
        raw = Tokenizer(models.BPE({c: i for i, c in enumerate(alphabet)}, []))
        raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        raw.decoder = decoders.ByteLevel()
    elif kind == "unknown-fallback":
        spec = json.loads(sentencepiece_bpe().to_str())
        del spec["model"]["vocab"]["<0x7A>"]
        spec["model"]["vocab"] = {piece: i for i, piece in enumerate(spec["model"]["vocab"])}
        raw = Tokenizer.from_str(json.dumps(spec))
    elif kind == "normalizer":
        raw.normalizer = normalizers.NFC()
    elif kind == "empty":
        raw.normalizer = normalizers.Replace(" Az", "")
    return ServedTokenizer(raw)


@pytest.mark.parametrize(
    "kind", ["complete", "missing-byte", "unknown-fallback", "normalizer", "empty"]
)
def test_label_admission_requires_exact_nonempty_known_bytes(kind):
    served, syntax = label_byte_tokenizer(kind), SpanSyntax()
    kg = KnowledgeGraph(
        (Entity("a", ("A", "Az", "é", "e\u0301", "日本")), Entity("shared", ("A",)))
    )
    index = build_label_index(kg, served, syntax)
    expected = {
        ("a", "A"),
        ("a", "Az"),
        ("a", "é"),
        ("a", "e\u0301"),
        ("a", "日本"),
        ("shared", "A"),
    }
    lost = {
        "complete": set(),
        "missing-byte": {("a", "Az")},
        "unknown-fallback": {("a", "Az")},
        "normalizer": {("a", "e\u0301")},
        "empty": {("a", "Az")},
    }[kind]
    assert {(e.owner_id, e.label) for e in index.excluded} == lost
    actual = {
        (
            index.owner_id(row),
            b"".join(served.token_bytes(t) for t in index.row_tokens(row)).decode()[1:],
        )
        for row in range(index.num_rows)
    }
    assert actual == expected - lost
    if kind == "empty":
        assert index.excluded[0].reason == "empty label encoding"
    if kind == "unknown-fallback":
        assert index.excluded[0].reason == "label contains a token with unknown bytes"
