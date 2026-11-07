"""The served tokenizer: the tokenizer of the model whose output is constrained."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from functools import cache
from pathlib import Path

from tokenizers import Tokenizer


class UnsupportedTokenizer(ValueError):
    """The tokenizer cannot meet the byte or token-ID domain contract."""


class ServedTokenizer:
    """The served model's tokenizer, with a byte view of its vocabulary.

    Args:
        tokenizer: A ``tokenizers.Tokenizer`` using byte-level BPE or
            SentencePiece-style BPE with byte fallback. Other vocabulary
            families or a non-dense token-ID domain raise
            :class:`UnsupportedTokenizer`.

    Encoding never inserts special tokens. ``fingerprint`` is the SHA-256 of
    the serialized tokenizer, binding a KG index to this exact vocabulary and
    configuration. Loading a tokenizer does not load model weights or run code
    from a model repository. The wrapper owns a serialized snapshot; subsequent
    changes to the caller's tokenizer cannot change encoding or vocabulary bytes.
    IDs must form the nonempty dense domain ``0 .. vocab_size - 1``, including
    added tokens.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        serialized = tokenizer.to_str()
        self._tokenizer = Tokenizer.from_str(serialized)
        ids = set(self._tokenizer.get_vocab(with_added_tokens=True).values())
        size = self._tokenizer.get_vocab_size(with_added_tokens=True)
        if not ids or ids != set(range(size)):
            raise UnsupportedTokenizer(
                "a nonempty dense token ID domain 0 .. vocab_size - 1 is required"
            )
        self.fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        """SHA-256 of the tokenizer's serialized form; a KG index is valid only for this value."""
        self._token_bytes, self._special_ids = _vocabulary_bytes(json.loads(serialized))

    @classmethod
    def from_file(cls, path: str | Path) -> ServedTokenizer:
        """Read a local ``tokenizer.json``; no network access or weights required."""
        return cls(Tokenizer.from_file(str(path)))

    @property
    def vocab_size(self) -> int:
        """Number of vocabulary entries, including added and special tokens."""
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    def encode(self, text: str) -> list[int]:
        """Encode text without special tokens; preserve the indexed label prefix."""
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode each text independently, without inserting special tokens."""
        return [e.ids for e in self._tokenizer.encode_batch(texts, add_special_tokens=False)]

    @property
    def special_ids(self) -> frozenset[int]:
        """Special tokens (such as end of text): control tokens that stand for no text."""
        return self._special_ids

    def token_bytes(self, token: int) -> bytes | None:
        """The bytes a token stands for; None for special tokens and unused ids."""
        return self._token_bytes[token] if 0 <= token < len(self._token_bytes) else None

    def check_eos_token_ids(self, tokens: Iterable[int]) -> frozenset[int]:
        """Validate explicit served-model EOS ids, without inferring them from token text."""
        ids = tuple(tokens)
        if not ids or any(type(t) is not int or not 0 <= t < self.vocab_size for t in ids):
            raise ValueError(
                "end_after_answer needs nonempty EOS token ids in the served vocabulary"
            )
        return frozenset(ids)


@cache
def _byte_level_alphabet() -> dict[str, int]:
    """GPT-2's printable stand-in character for each byte, inverted: character to byte."""
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1)]
    printable += range(ord("®"), ord("ÿ") + 1)
    chars = {b: chr(b) for b in printable}
    shifted = (b for b in range(256) if b not in chars)
    for n, b in enumerate(shifted):
        chars[b] = chr(256 + n)
    return {char: b for b, char in chars.items()}


def _decoders(decoder: dict | None) -> list[dict]:
    if decoder is None:
        return []
    if decoder.get("type") == "Sequence":
        return [d for inner in decoder.get("decoders", []) for d in _decoders(inner)]
    return [decoder]


def _is_sentencepiece(spec: dict) -> bool:
    """SentencePiece-style BPE (as Gemma's): '▁' stands for a space, '<0xNN>' for a byte."""
    steps = _decoders(spec.get("decoder"))
    spaces = {"type": "Replace", "pattern": {"String": SPACE}, "content": " "}
    return (
        spec["model"].get("byte_fallback") is True
        and spaces in steps
        and any(d.get("type") == "ByteFallback" for d in steps)
    )


SPACE = "\u2581"
BYTE_PIECE = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def _piece_bytes(piece: str, token: int, byte_level: bool) -> bytes:
    if not byte_level:
        byte = BYTE_PIECE.fullmatch(piece)
        return bytes([int(byte[1], 16)]) if byte else piece.replace(SPACE, " ").encode("utf-8")
    alphabet = _byte_level_alphabet()
    try:
        return bytes(alphabet[char] for char in piece)
    except KeyError as error:
        raise UnsupportedTokenizer(
            f"token {token} ({piece!r}) has a character outside the byte-level alphabet"
        ) from error


def _vocabulary_bytes(spec: dict) -> tuple[list[bytes | None], frozenset[int]]:
    """Map every token id of a serialized tokenizer to its bytes.

    Two vocabularies are supported: byte-level BPE (as GPT-2's and Qwen's), whose pieces
    spell bytes in a printable alphabet, and SentencePiece-style BPE with byte fallback
    (as Gemma's), whose pieces are text with '▁' for a space, or '<0xNN>' for one byte.
    The unknown token stands for no text.
    """
    byte_level = any(d.get("type") == "ByteLevel" for d in _decoders(spec.get("decoder")))
    if spec["model"].get("type") != "BPE" or not (byte_level or _is_sentencepiece(spec)):
        raise UnsupportedTokenizer(
            "only byte-level or SentencePiece-style byte-fallback BPE tokenizers are "
            f"supported; got model {spec['model'].get('type')!r} with decoder "
            f"{spec.get('decoder')!r}"
        )
    added = spec.get("added_tokens", [])
    size = 1 + max([*spec["model"]["vocab"].values(), *(a["id"] for a in added)])
    token_bytes: list[bytes | None] = [None] * size
    unknown = spec["model"].get("unk_token")
    for piece, token in spec["model"]["vocab"].items():
        if piece != unknown or byte_level:
            token_bytes[token] = _piece_bytes(piece, token, byte_level)
    special = set()
    for token in added:
        if token["special"]:
            token_bytes[token["id"]] = None
            special.add(token["id"])
        else:
            token_bytes[token["id"]] = token["content"].encode("utf-8")
    return token_bytes, frozenset(special)
