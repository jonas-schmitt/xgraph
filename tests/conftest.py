"""Shared fixtures: a toy KG and a hermetic byte-level BPE tokenizer trained on it."""

from __future__ import annotations

import os
import shutil

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from xgraph.kg import Entity, KnowledgeGraph, Relation


@pytest.fixture(params=["cpu", "cuda"])
def engine_device(request, monkeypatch):
    """Run engine consumers on CPU too; only CUDA supplies native agreement."""
    if request.param == "cpu":
        monkeypatch.setattr("xgraph.engine.Engine", OracleHostEngine)
    elif os.environ.get("CUDA_VISIBLE_DEVICES") == "" or shutil.which("nvidia-smi") is None:
        pytest.skip("no visible NVIDIA GPU for engine consumers")
    return request.param


class OracleHostEngine:
    """Test-only host/tensor interface backed by the oracle, never native evidence.

    Host span records still go through engine.closed_span, so Python consumers
    exercise the production record conversion and export contract on CPU.
    """

    def __init__(self, index, tokenizer, syntax, max_sequences=1, **options):
        from xgraph.oracle import Oracle

        self.index, self.max_sequences = index, max_sequences
        self.vocab_size = tokenizer.vocab_size
        self.oracle = Oracle(index, tokenizer, syntax, **options)
        self.reset()

    def reset(self):
        self.states = [self.oracle.initial_state() for _ in range(self.max_sequences)]
        self.bad = set()

    def begin(self, slots):
        for slot in slots.tolist():
            self.states[slot] = self.oracle.initial_state()
            self.bad.discard(slot)

    def anchor(self, slot, entities):
        self.states[slot] = self.oracle.anchored_state(entities)

    def _advance(self, slot, token):
        from xgraph.oracle import TokenNotAllowed

        if slot not in self.bad:
            try:
                self.states[slot] = self.oracle.advance(self.states[slot], int(token))
            except TokenNotAllowed:
                self.bad.add(slot)

    def advance(self, tokens):
        for slot, token in enumerate(tokens):
            self._advance(slot, token)

    def fill_mask(self, sequences=None):
        import numpy as np

        from xgraph.engine import mask_words

        count = self.max_sequences if sequences is None else sequences
        self.mask = np.zeros((count, mask_words(self.vocab_size)), np.uint32)
        for slot in range(count):
            for token in self.oracle.allowed(self.states[slot]):
                self.mask[slot, token >> 5] |= np.uint32(1 << (token & 31))

    def mask_host(self, sequences=None):
        return self.mask if sequences is None else self.mask[:sequences]

    def states_host(self, sequences=None):
        import numpy as np

        from xgraph.engine import CLOSED
        from xgraph.oracle import Phase

        count = self.max_sequences if sequences is None else sequences
        fields = (
            "phase",
            "last_kind",
            "last_lo",
            "last_hi",
            "last_subj_lo",
            "last_subj_hi",
            "last_rel_lo",
            "last_rel_hi",
        )
        result = {name: np.zeros(count, np.int32) for name in fields}

        def exact_rows(table, tokens):
            lo, hi = table.prefix_range(tokens)
            rows = [row for row in range(lo, hi) if table.row_tokens(row) == tokens]
            return rows[0], rows[-1] + 1

        for slot, state in enumerate(self.states[:count]):
            result["phase"][slot] = (
                3
                if slot in self.bad
                else {Phase.FREE: 0, Phase.SPAN: 1, Phase.ESCAPED: 2}[state.phase]
            )
            span = state.last_span
            if span is None:
                continue
            kind = (
                "unresolved"
                if not span.resolved
                else "fact span"
                if span.triples
                else "entity span"
            )
            result["last_kind"][slot] = CLOSED.index(kind)
            if kind == "fact span":
                for prefix, table, label in (
                    ("last_subj", self.index.labels, span.labels[0]),
                    ("last_rel", self.index.relations, span.labels[1]),
                    ("last", self.index.labels, span.labels[2]),
                ):
                    result[prefix + "_lo"][slot], result[prefix + "_hi"][slot] = exact_rows(
                        table, label
                    )
            elif kind == "entity span":
                result["last_lo"][slot], result["last_hi"][slot] = exact_rows(
                    self.index.labels, span.labels[0]
                )
        return result

    def apply_mask(self, logits, slots, *, local_pos, input_ids):
        import torch

        cursors = {}
        for row, (slot, position) in enumerate(
            zip(slots.tolist(), local_pos.tolist(), strict=True)
        ):
            for at in range(cursors.get(slot, 0), position):
                self._advance(slot, input_ids[at + 1])
            cursors[slot] = position
            if slot in self.bad:
                continue
            allowed = torch.zeros(logits.shape[1], dtype=torch.bool, device=logits.device)
            allowed[list(self.oracle.allowed(self.states[slot]))] = True
            logits[row, ~allowed] = -float("inf")


@pytest.fixture(autouse=True)
def unchanged_xgraph_environment():
    """Check after fixture cleanup, including monkeypatch's environment restoration."""
    before = {k: v for k, v in os.environ.items() if k.startswith(("XGRAPH_", "XG_"))}
    yield
    after = {k: v for k, v in os.environ.items() if k.startswith(("XGRAPH_", "XG_"))}
    assert after == before, "test leaked XGRAPH_*/XG_* environment changes; use monkeypatch"


# (entity id, labels). Chosen to hit the hard cases: an ambiguous label (Q3, Q4),
# labels that are prefixes of others (Paris / Paris Hilton, Nobel Prize / Nobel
# Prize in Physics), multi-byte UTF-8 (ł, the en dash), a duplicate alias (Q1),
# and labels the index must exclude (span syntax inside a label or formed by the
# label prefix in front of it, a blank label, the text of a special token, which
# the tokenizer turns into that token).
TOY_ENTITIES = (
    ("Q1", ("Marie Curie", "Maria Skłodowska", "Marie Curie")),
    ("Q2", ("Pierre Curie",)),
    ("Q3", ("Paris",)),
    ("Q4", ("Paris",)),
    ("Q5", ("Paris Hilton",)),
    ("Q6", ("Curie",)),
    ("Q7", ("University of Paris",)),
    ("Q8", ("Nobel Prize in Physics",)),
    ("Q9", ("Nobel Prize",)),
    ("Q10", ("A⟧B",)),
    ("Q11", ("   ",)),
    ("Q12", ("Rhône–Alpes",)),
    ("Q15", ("Paris<|endoftext|>",)),
    ("Q16", ("⁇ Unknown",)),
    ("Q17", ("Curie",)),
)

# (relation id, labels). "located in" is P131's label and also an alias of P276.
# "award" (P2121, Pierre Curie only) is a prefix of "award received" (Marie Curie's),
# so after Marie Curie a complete relation label can belong to a relation she lacks.
TOY_RELATIONS = (
    ("P26", ("spouse",)),
    ("P108", ("employer",)),
    ("P166", ("award received",)),
    ("P19", ("place of birth",)),
    ("P131", ("located in",)),
    ("P276", ("location", "located in")),
    ("P2121", ("award",)),
)

# Chosen for the hard cases: two objects of one (subject, relation) pair whose labels
# are prefixes of each other (Q1 award received), an ambiguous relation label (Q7
# located in), a subject label shared with an entity without triples (Paris: Q3, Q4),
# a subject label shared by two entities with triples (Curie: Q6, Q17), and a triple
# whose object label is excluded (Q1 spouse Q10), so it cannot be written.
TOY_TRIPLES = (
    ("Q1", "P26", "Q2"),
    ("Q2", "P26", "Q1"),
    ("Q1", "P108", "Q7"),
    ("Q2", "P108", "Q7"),
    ("Q1", "P166", "Q8"),
    ("Q1", "P166", "Q9"),
    ("Q7", "P131", "Q3"),
    ("Q7", "P276", "Q4"),
    ("Q3", "P131", "Q12"),
    ("Q5", "P19", "Q4"),
    ("Q1", "P26", "Q10"),
    ("Q2", "P2121", "Q8"),
    ("Q6", "P19", "Q3"),
    ("Q17", "P19", "Q4"),
    ("Q17", "P108", "Q7"),
)

TOY_TEXT = (
    "She met ⟦ Marie Curie⟧ in ⟦ Paris⟧. ⟦ Pierre Curie⟧ won the ⟦ Nobel Prize in Physics⟧.",
    "The ⟦ University of Paris⟧ is old. ⟦ Rhône–Alpes⟧ is a region. ⟦ Maria ⁇ maybe⟧.",
    "Unverified spans look like ⟦ ⁇ something else⟧ and plain text has no spans at all.",
    "As facts: ⟦ Marie Curie · spouse · Pierre Curie⟧ and ⟦ Pierre Curie · employer · "
    "University of Paris⟧ and ⟦ University of Paris · located in · Paris⟧.",
)


@pytest.fixture(scope="session")
def toy_kg() -> KnowledgeGraph:
    return KnowledgeGraph(
        entities=tuple(Entity(id=i, labels=ls) for i, ls in TOY_ENTITIES),
        relations=tuple(Relation(id=i, labels=ls) for i, ls in TOY_RELATIONS),
        triples=TOY_TRIPLES,
    )


def train_byte_level_bpe(corpus: list[str], vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    return tokenizer


@pytest.fixture(scope="session")
def toy_tokenizer() -> Tokenizer:
    labels = [label for _, labels in TOY_ENTITIES + TOY_RELATIONS for label in labels]
    return train_byte_level_bpe(labels + list(TOY_TEXT), vocab_size=400)


def byte_level_tokenizer(merges: list[tuple[str, str]]) -> Tokenizer:
    """A byte-level BPE tokenizer with exactly these merges on top of the 256 byte tokens.

    Merge parts are written in the byte-level alphabet (for example "âŁ" is E2 9F).
    """
    vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    for left, right in merges:
        vocab[left + right] = len(vocab)
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(["<|endoftext|>"])
    return tokenizer


# A KG directory with literal values (python -m xgraph.literals) and the entity KG
# directory of the same items without them, for merging dates (xgraph.literals
# merge_dates). Values are numbered from FIRST_ID as the literal KG numbers them: a day
# (Spielberg's birth), a month (Jaws's release, stored as its first day), a quantity
# (left out of a merge), a day (Lucas's birth), a year (Star Wars's, stored as 1
# January), and the birth date of Q5, an item the entity KG lacks.
DATED_ITEMS = {1: "Jaws", 2: "Steven Spielberg", 3: "Star Wars", 4: "George Lucas"}
DATED_RELATIONS = {57: "director", 569: "date of birth", 577: "publication date", 2047: "duration"}
DATED_VALUES = (
    ("date", "1946-12-18T00:00:00Z"),
    ("date", "1975-06-01T00:00:00Z"),
    ("decimal", "+124"),
    ("date", "1944-05-14T00:00:00Z"),
    ("date", "1977-01-01T00:00:00Z"),
    ("date", "1900-03-02T00:00:00Z"),
)
DATED_ITEM_TRIPLES = ((1, 57, 2), (3, 57, 4))
DATED_VALUE_TRIPLES = (
    (2, 569, 0),
    (1, 577, 1),
    (1, 2047, 2),
    (4, 569, 3),
    (3, 577, 4),
    (5, 569, 5),
)
"""(subject, property, value): the value as a position in DATED_VALUES."""


def write_dated_kgs(root) -> tuple:
    """The literal KG directory and the entity KG directory under root: (literal, entity)."""
    import json
    from pathlib import Path

    import numpy as np

    from xgraph.literals import FIRST_ID, render_date, render_decimal

    def lines(rows) -> str:
        return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)

    root = Path(root)
    literal, entity = root / "literal-kg", root / "entity-kg"
    for directory in (literal, entity):
        directory.mkdir(parents=True)
        relations = [[f"P{p}", label, "label"] for p, label in DATED_RELATIONS.items()]
        (directory / "relation_labels.jsonl").write_text(lines(relations))
        (directory / "sitelinks.jsonl").write_text("")
    entity_rows = [[f"Q{q}", label, "label"] for q, label in DATED_ITEMS.items()]
    (entity / "entity_labels.jsonl").write_text(lines(entity_rows))
    np.save(entity / "triples.npy", np.array(DATED_ITEM_TRIPLES, dtype=np.int32))
    literal_rows = [*entity_rows, ["Q5", "Unmatched Person", "label"]]
    for n, (kind, value) in enumerate(DATED_VALUES):
        rendered = (render_date if kind == "date" else render_decimal)(value)
        for k, label in enumerate(rendered):
            literal_rows.append([f"Q{FIRST_ID + n}", label, "label" if k == 0 else "alias"])
    (literal / "entity_labels.jsonl").write_text(lines(literal_rows))
    triples = [*DATED_ITEM_TRIPLES, *((s, p, FIRST_ID + v) for s, p, v in DATED_VALUE_TRIPLES)]
    np.save(literal / "triples.npy", np.array(triples, dtype=np.int32))
    return literal, entity
