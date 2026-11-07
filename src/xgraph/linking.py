"""Entity linking by label match: the retrieval baseline's topic entities without an oracle.

The gate's retrieval arm takes a question's topic entities from its gold query, which a
deployed system does not have. This linker finds them in the question text, as simple
KGQA baselines do: every word n-gram of the question (up to six words) that equals an
English label or alias of a KG item, after NFC normalization and case folding, is a
mention, unless it is only function words ("Who" and "The First" are items too); a
single word must be capitalized, after the question's first word, or a number (common
nouns such as "author" are items as well). Mentions are taken longest first, without
overlapping a taken one, and each links to its item with the most triples (subject or
object); at most three items per question. The rules were set on Mintaka's dev split.

    python -m xgraph.rag --test target/mintaka/questions.json --kg data/wikidata/gate-kg \\
        --linked --out target/mintaka/requests-rag-linked.jsonl
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np

LONGEST = 6
MAX_ENTITIES = 3
CHUNK = 1 << 22
WORD = re.compile(r"[\w][\w'\u2019.\-&]*[\w]|[\w]")


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def _words(question: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group()) for m in WORD.finditer(question)]


def mentions(question: str, longest: int = LONGEST) -> list[str]:
    """The question's word n-grams, up to `longest` words, as they appear in it."""
    words = _words(question)
    return [
        question[words[i][0] : words[j - 1][1]]
        for i in range(len(words))
        for j in range(i + 1, min(i + longest, len(words)) + 1)
    ]


FUNCTION_WORDS = frozenset(
    [
        "a",
        "about",
        "after",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "before",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "during",
        "first",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "last",
        "least",
        "less",
        "many",
        "more",
        "most",
        "much",
        "not",
        "of",
        "on",
        "one",
        "or",
        "other",
        "part",
        "same",
        "she",
        "so",
        "some",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "under",
        "up",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "would",
    ]
)


def _candidates(text: str, longest: int) -> list[tuple[int, int, str]]:
    """(first word, end word, n-gram) of the question's linkable n-grams: not only function
    words, and a single word only when capitalized (or a number) after the first word."""
    words = _words(text)
    found = []
    for i in range(len(words)):
        for j in range(i + 1, min(i + longest, len(words)) + 1):
            gram = text[words[i][0] : words[j - 1][1]]
            if all(w.casefold() in FUNCTION_WORDS for _, _, w in words[i:j]):
                continue
            if j - i == 1 and (i == 0 or not (gram[:1].isupper() or gram[:1].isdigit())):
                continue
            found.append((i, j, gram))
    return found


def link(
    questions: dict[str, str], kg: str | Path, longest: int = LONGEST, cap: int = MAX_ENTITIES
) -> dict[str, list[int]]:
    """Per question id, the item numbers its text mentions, in the order taken."""
    kg = Path(kg)
    grams = {qid: _candidates(text, longest) for qid, text in questions.items()}
    wanted = {_norm(g) for found in grams.values() for _, _, g in found}
    items: dict[str, set[int]] = {}
    with (kg / "entity_labels.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            item, label, _ = json.loads(line)
            key = _norm(label)
            if key in wanted:
                items.setdefault(key, set()).add(int(item[1:]))
    known = np.array(sorted(set().union(*items.values())), dtype=np.int64)
    degree = np.zeros(len(known), dtype=np.int64)
    if len(known):
        triples = np.load(kg / "triples.npy", mmap_mode="r")
        for at in range(0, len(triples), CHUNK):
            chunk = np.asarray(triples[at : at + CHUNK], dtype=np.int64)
            for column in (chunk[:, 0], chunk[:, 2]):
                at_known = np.searchsorted(known, column)
                hit = (at_known < len(known)) & (
                    known[np.minimum(at_known, len(known) - 1)] == column
                )
                np.add.at(degree, at_known[hit], 1)
    rank = dict(zip(known.tolist(), degree.tolist(), strict=True))

    linked: dict[str, list[int]] = {}
    for qid, found in grams.items():
        found = sorted(
            (g for g in found if _norm(g[2]) in items), key=lambda g: (g[0] - g[1], g[0])
        )
        taken: list[tuple[int, int]] = []
        chosen: list[int] = []
        for i, j, gram in found:
            if any(i < b and a < j for a, b in taken) or len(chosen) == cap:
                continue
            best = max(sorted(items[_norm(gram)]), key=lambda n: rank[n])
            taken.append((i, j))
            if best not in chosen:
                chosen.append(best)
        linked[qid] = chosen
    return linked
