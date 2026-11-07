"""The judge's Wikipedia: passages from the CirrusSearch dump, stored by title, and BM25.

As FActScore does (`docs/quality-gate.md`): each article's text is split into
consecutive passages of at most 256 roberta-large tokens, stored in SQLite by exact
title, and the passages BM25 ranks highest for a query are retrieved with
rank_bm25's BM25Okapi over whitespace-split words. Titles that redirect to an
article (listed with it in the dump) find its passages too.

    python -m xgraph.wikipedia build \\
        --dump data/wikipedia/enwiki-cirrus-20261004 \\
        --tokenizer data/tokenizers/roberta-large/tokenizer.json \\
        --out data/wikipedia/enwiki-20261004.db
"""

from __future__ import annotations

import argparse
import itertools
import json
import sqlite3
import subprocess
import threading
from collections.abc import Iterable, Iterator
from pathlib import Path

MAX_LENGTH = 256
SEPARATOR = "\x1e"  # between passages in a stored article (ASCII record separator)
BATCH = 2000


def articles(lines: Iterable[str]) -> Iterator[tuple[str, str, list[str]]]:
    """(title, text, redirect titles) for each main-namespace article with text."""
    for line in lines:
        doc = json.loads(line)
        if "index" in doc or str(doc.get("namespace")) != "0":
            continue
        text = doc.get("text") or ""
        if not text.strip():
            continue  # redirect pages and empty pages
        redirects = [r["title"] for r in doc.get("redirect") or [] if str(r["namespace"]) == "0"]
        yield doc["title"], text, redirects


def _chunks(text: str, offsets: list[tuple[int, int]], max_length: int) -> list[str]:
    """The text cut before every max_length-th token, so the chunks join back to it."""
    cuts = [0, *(offsets[i][0] for i in range(max_length, len(offsets), max_length))]
    return [text[a:b] for a, b in itertools.pairwise([*cuts, len(text)])]


def split_passages(text: str, tokenizer, max_length: int = MAX_LENGTH) -> list[str]:
    """Consecutive chunks of at most max_length tokens."""
    offsets = tokenizer.encode(text, add_special_tokens=False).offsets
    return _chunks(text, offsets, max_length)


def build_db(
    parts: Iterable[Iterable[str]], tokenizer, out: str | Path, max_length: int = MAX_LENGTH
) -> dict[str, int]:
    """Write the passages of every article, and the redirect titles, into SQLite."""
    out = Path(out)
    out.unlink(missing_ok=True)
    db = sqlite3.connect(out)
    db.execute("CREATE TABLE documents (title TEXT PRIMARY KEY, text TEXT)")
    db.execute("CREATE TABLE redirects (title TEXT PRIMARY KEY, target TEXT)")
    stats = {"articles": 0, "redirects": 0}

    def flush(batch: list[tuple[str, str, list[str]]]) -> None:
        texts = [text for _, text, _ in batch]
        encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
        rows, redirect_rows = [], []
        for (title, text, redirects), encoding in zip(batch, encodings, strict=True):
            rows.append((title, SEPARATOR.join(_chunks(text, encoding.offsets, max_length))))
            redirect_rows.extend((r, title) for r in redirects)
        stats["articles"] += db.executemany(
            "INSERT OR IGNORE INTO documents VALUES (?, ?)", rows
        ).rowcount
        stats["redirects"] += db.executemany(
            "INSERT OR IGNORE INTO redirects VALUES (?, ?)", redirect_rows
        ).rowcount

    for lines in parts:
        batch: list[tuple[str, str, list[str]]] = []
        for article in articles(lines):
            batch.append(article)
            if len(batch) >= BATCH:
                flush(batch)
                batch = []
        if batch:
            flush(batch)
        db.commit()
    db.close()
    return stats


class PassageDB:
    """Read-only lookups; each thread gets its own SQLite connection."""

    def __init__(self, path: str | Path) -> None:
        self._path = path
        self._local = threading.local()

    @property
    def _db(self) -> sqlite3.Connection:
        if not hasattr(self._local, "db"):
            self._local.db = sqlite3.connect(f"file:{self._path}?mode=ro", uri=True)
        return self._local.db

    def passages(self, title: str) -> list[str]:
        """The passages of the article with exactly this title, or of its redirect target."""
        for query in (
            "SELECT text FROM documents WHERE title = ?",
            "SELECT d.text FROM redirects r JOIN documents d ON d.title = r.target"
            " WHERE r.title = ?",
        ):
            row = self._db.execute(query, (title,)).fetchone()
            if row:
                return row[0].split(SEPARATOR)
        return []


def bm25_top(query: str, passages: list[str], k: int = 5) -> list[str]:
    """The k passages BM25Okapi ranks highest for the query, as FActScore ranks them."""
    from rank_bm25 import BM25Okapi

    if not passages:
        return []
    scores = BM25Okapi([p.split() for p in passages]).get_scores(query.split())
    order = sorted(range(len(passages)), key=lambda i: -scores[i])
    return [passages[i] for i in order[:k]]


def _dump_lines(path: Path) -> Iterator[str]:
    with subprocess.Popen(
        ["lbzip2", "-dc", str(path)], stdout=subprocess.PIPE, text=True, encoding="utf-8"
    ) as process:
        assert process.stdout is not None
        yield from process.stdout
    if process.returncode:
        raise RuntimeError(f"lbzip2 failed on {path} with exit status {process.returncode}")


def main(argv: list[str] | None = None) -> None:
    from tokenizers import Tokenizer

    parser = argparse.ArgumentParser(prog="python -m xgraph.wikipedia", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build the passage database from the dump")
    build.add_argument("--dump", type=Path, required=True, help="directory of .json.bz2 parts")
    build.add_argument("--tokenizer", type=Path, required=True, help="roberta-large tokenizer.json")
    build.add_argument("--out", type=Path, required=True, help="SQLite database")
    args = parser.parse_args(argv)
    parts = sorted(args.dump.glob("*.json.bz2"))
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    print(json.dumps(build_db((_dump_lines(p) for p in parts), tokenizer, args.out)))


if __name__ == "__main__":
    main()
