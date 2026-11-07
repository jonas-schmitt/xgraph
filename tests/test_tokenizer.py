"""Served tokenizer: the bytes each token stands for."""

from __future__ import annotations

import pytest
from tokenizers import Tokenizer, decoders, models

from xgraph.tokenizer import ServedTokenizer, UnsupportedTokenizer

TEXTS = ["Marie Curie", " Rhône–Alpes", "⟦Paris⟧ ⁇", "Skłodowska\nnew line", "日本"]


@pytest.mark.parametrize("text", TEXTS)
def test_token_bytes_concatenate_to_the_utf8_text(toy_tokenizer, text):
    served = ServedTokenizer(toy_tokenizer)
    ids = served.encode(text)
    assert b"".join(served.token_bytes(i) for i in ids) == text.encode("utf-8")


def test_special_tokens_have_no_bytes(toy_tokenizer):
    served = ServedTokenizer(toy_tokenizer)
    eos = toy_tokenizer.token_to_id("<|endoftext|>")
    assert served.token_bytes(eos) is None
    assert served.special_ids == frozenset({eos})


def test_non_byte_level_tokenizers_are_rejected():
    tokenizer = Tokenizer(models.WordLevel({"a": 0, "[UNK]": 1}, unk_token="[UNK]"))
    tokenizer.decoder = decoders.WordPiece()
    with pytest.raises(UnsupportedTokenizer, match="byte-level"):
        ServedTokenizer(tokenizer)


@pytest.mark.parametrize("sparse", ["above-count", "hole", "collision", "empty"])
@pytest.mark.parametrize("route", ["constructor", "file"])
def test_sparse_token_ids_are_rejected_at_admission(tmp_path, sparse, route):
    from tokenizers import pre_tokenizers

    vocab = {c: i for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    if sparse == "above-count":
        vocab["A"] = len(vocab) + 10
    elif sparse == "hole":
        del vocab["A"]
    elif sparse == "collision":
        vocab["A"] = vocab["B"]
    else:
        vocab = {}
    raw = Tokenizer(models.BPE(vocab, []))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    with pytest.raises(UnsupportedTokenizer, match=r"dense.*token.*[Ii][Dd]"):
        if route == "file":
            path = tmp_path / "tokenizer.json"
            raw.save(str(path))
            ServedTokenizer.from_file(path)
        else:
            ServedTokenizer(raw)


def test_dense_added_tokens_and_explicit_eos_domain_remain_supported():
    from tests.conftest import byte_level_tokenizer

    raw = byte_level_tokenizer([])
    raw.add_tokens([" Atlas"])
    raw.add_special_tokens(["<eos>"])
    served = ServedTokenizer(raw)
    assert set(raw.get_vocab().values()) == set(range(served.vocab_size))
    assert b"".join(served.token_bytes(t) for t in served.encode(" Atlas")) == b" Atlas"
    eos = raw.token_to_id("<eos>")
    assert served.check_eos_token_ids([eos, eos]) == frozenset({eos})
    for invalid in ([], [-1], [served.vocab_size], [True]):
        with pytest.raises(ValueError, match="EOS token ids"):
            served.check_eos_token_ids(invalid)


@pytest.mark.parametrize(
    "tree", ["none", "empty", "nested-empty", "byte", "nested-byte", "deep-byte"]
)
def test_nested_decoder_admission_preserves_byte_family_detection(tree):
    import json

    from tests.conftest import byte_level_tokenizer

    spec = json.loads(byte_level_tokenizer([]).to_str())
    byte = spec["decoder"]
    empty = {"type": "Sequence", "decoders": []}
    nested = {"type": "Sequence", "decoders": [empty, byte]}
    spec["decoder"] = {
        "none": None,
        "empty": empty,
        "nested-empty": {"type": "Sequence", "decoders": [empty]},
        "byte": byte,
        "nested-byte": nested,
        "deep-byte": {
            "type": "Sequence",
            "decoders": [empty, {"type": "Sequence", "decoders": [nested]}],
        },
    }[tree]
    raw = Tokenizer.from_str(json.dumps(spec))
    if tree in ("none", "empty", "nested-empty"):
        with pytest.raises(UnsupportedTokenizer, match="byte-level"):
            ServedTokenizer(raw)
    else:
        served = ServedTokenizer(raw)
        for text in TEXTS:
            assert b"".join(served.token_bytes(t) for t in served.encode(text)) == text.encode(
                "utf-8"
            )


def sentencepiece_bpe() -> Tokenizer:
    """A toy SentencePiece-style BPE tokenizer, as Gemma's: '▁' for spaces, byte fallback."""
    from tokenizers import normalizers

    vocab = {"<unk>": 0, "<eos>": 1}
    vocab.update({f"<0x{b:02X}>": 2 + b for b in range(256)})
    for piece in ["▁", "M", "a", "r", "i", "e", "C", "u", "▁M", "⟦", "⟧"]:
        vocab[piece] = len(vocab)
    tokenizer = Tokenizer(
        models.BPE(vocab, [("▁", "M")], unk_token="<unk>", byte_fallback=True, fuse_unk=True)
    )
    tokenizer.normalizer = normalizers.Replace(" ", "▁")
    tokenizer.decoder = decoders.Sequence(
        [decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()]
    )
    tokenizer.add_special_tokens(["<eos>"])
    return tokenizer


@pytest.mark.parametrize("text", TEXTS)
def test_sentencepiece_token_bytes_concatenate_to_the_utf8_text(text):
    served = ServedTokenizer(sentencepiece_bpe())
    ids = served.encode(text)
    assert b"".join(served.token_bytes(i) for i in ids) == text.encode("utf-8")


def test_sentencepiece_control_tokens_have_no_bytes():
    tokenizer = sentencepiece_bpe()
    served = ServedTokenizer(tokenizer)
    assert served.token_bytes(tokenizer.token_to_id("<unk>")) is None
    assert served.token_bytes(tokenizer.token_to_id("<eos>")) is None
    assert served.token_bytes(tokenizer.token_to_id("▁M")) == b" M"
    assert served.token_bytes(tokenizer.token_to_id("<0x0A>")) == b"\n"
