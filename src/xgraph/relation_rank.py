"""Offline, CPU-only question ranks for relation menus; no answers or outputs are read.

The frozen encoder is contriever-msmarco, last two relation-name segments, masked
mean pooling and fp32 dot product. Every example has the same inference shape,
so request order and callers' batching cannot change its embedding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sqlite3
from itertools import batched
from pathlib import Path

import numpy as np

from xgraph.audit import MAX_RANKED_RELATIONS
from xgraph.provenance import provenance, sha256_file

MODEL = "facebook/contriever-msmarco"
REVISION = "abe8c1493371369031bcb1e02acb754cf4e162fa"
FIELD = "relation_ranks"
MAX_REQUEST_BYTES = 16 * 1024**2
BATCH = 32


def relation_text(name: str) -> str:
    """Freebase source relation IDs are canonical labels in numeric KG directories."""
    return " ".join(name.split(".")[-2:]).replace("_", " ")


class DenseEncoder:
    """Fixed CPU fp32 inference, one thread, fixed 32 x 64 token shape."""

    def __init__(self, snapshot: Path):
        import torch
        from transformers import AutoModel, AutoTokenizer

        snapshot = Path(snapshot).resolve()
        if snapshot.name != REVISION:
            raise ValueError(f"expected {MODEL} snapshot {REVISION}, got {snapshot}")
        weights = snapshot / "pytorch_model.bin"
        if not weights.is_file():
            raise ValueError(f"missing frozen offline weights: {weights}")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        self.model = (
            AutoModel.from_pretrained(snapshot, local_files_only=True)
            .to(device="cpu", dtype=torch.float32)
            .eval()
        )
        self.metadata = {
            "model": MODEL,
            "snapshot_revision": REVISION,
            "weight_sha256": sha256_file(weights),
            "snapshot": str(snapshot),
            "device": "cpu",
            "dtype": "float32",
            "threads": 1,
            "inference_batch_size": BATCH,
            "max_length": 64,
            "pooling": "masked mean",
            "score": "dot product",
            "relation_text": "last two dot-separated segments; _ to space",
            "files": {p.name: sha256_file(p) for p in sorted(snapshot.iterdir()) if p.is_file()},
            "torch": torch.__version__,
        }

    def encode(self, texts) -> np.ndarray:
        import torch

        result = []
        with torch.inference_mode():
            for batch in batched(texts, BATCH):
                tokens = self.tokenizer(
                    [*batch, *([""] * (BATCH - len(batch)))],
                    padding="max_length",
                    truncation=True,
                    max_length=64,
                    return_tensors="pt",
                )
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(dtype=torch.float32)
                result.append(((hidden * mask).sum(1) / mask.sum(1)).numpy()[: len(batch)].copy())
        return np.concatenate(result).astype(np.float32, copy=False)


class RelationPool:
    """Stream one version/component's relation vocabulary from a read-only KG store.

    Temporary SQL tables hold only the selected version's edits. Replacement
    pairs suppress base edges before relation presence is tested, including
    explicit deletions. Component bounds restrict both ends of an edge.
    """

    def __init__(self, data: Path):
        from xgraph.kgversions import component_ranges

        self.data = Path(data).resolve()
        self.db = self.data / "kg.sqlite"
        self.con = sqlite3.connect(self.db.as_uri() + "?mode=ro", uri=True)
        self.con.execute("PRAGMA cache_size=-4096")
        self.con.execute("PRAGMA temp_store=FILE")
        self.con.execute("CREATE TEMP TABLE rank_edits(s INTEGER, p INTEGER, o INTEGER)")
        self.con.execute("CREATE INDEX temp.rank_edits_sp ON rank_edits(s,p)")
        self.names = {
            f"P{p}": name for p, name in self.con.execute("SELECT prop,label FROM relations")
        }
        header = self.data / "versions/kg-versions.json"
        self.header = json.loads(header.read_text()) if header.exists() else None
        self.entries = {v["name"]: v for v in self.header["versions"]} if self.header else {}
        self.components = component_ranges(header.parent) if self.header else {}
        self.inputs = {self.db, *([header] if self.header else [])}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.con.close()

    def _edits(self, version):
        if version not in self.entries:
            raise ValueError(f"unknown KG version {version!r} in {self.data}")
        entry = self.entries[version]
        relative = Path(entry["edits"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"KG version edit path must be within versions: {relative}")
        # Frozen inputs may symlink to the original version array. Reads follow
        # those links; the manifest and backing bytes are never changed.
        path = self.data / "versions" / relative
        self.inputs.add(path)
        if self.header["format"] == 1:
            edits = np.load(path, mmap_mode="r", allow_pickle=False)
            if edits.ndim != 2 or edits.shape[1] != 3 or edits.dtype.kind not in "iu":
                raise ValueError(f"{path}: expected integer (n, 3) edits")
            for s, p, o in edits:
                yield int(s), int(p), int(o)
        else:
            if sha256_file(path) != entry.get("sha256"):
                raise ValueError(f"{path}: KG edits SHA-256 mismatch")
            with path.open() as file:
                for line in file:
                    s, p, o = json.loads(line)
                    if (
                        not s.startswith("Q")
                        or not p.startswith("P")
                        or (o is not None and not o.startswith("Q"))
                    ):
                        raise ValueError(f"{path}: the SQLite rank builder needs numeric Q/P IDs")
                    yield int(s[1:]), int(p[1:]), int(o[1:]) if o is not None else None

    def relations(self, version: str | None) -> tuple[str, ...]:
        self.con.execute("DELETE FROM rank_edits")
        if version is not None:
            self.con.executemany("INSERT INTO rank_edits VALUES(?,?,?)", self._edits(version))
        bounds = self.components.get(version)
        where, arguments = "1", ()
        if bounds is not None:
            first, stop = bounds
            where, arguments = "s>=? AND s<? AND o>=? AND o<?", (first, stop, first, stop)
        base = (
            "SELECT DISTINCT p FROM triples t WHERE "
            + where
            + " AND NOT EXISTS(SELECT 1 FROM rank_edits e WHERE e.s=t.s AND e.p=t.p)"
        )
        added = (
            "SELECT DISTINCT p FROM rank_edits WHERE "
            + where
            + " AND EXISTS(SELECT 1 FROM labels WHERE item=s)"
            + " AND EXISTS(SELECT 1 FROM labels WHERE item=o)"
            + " AND EXISTS(SELECT 1 FROM relations WHERE prop=p)"
        )
        return tuple(
            sorted(
                f"P{p}"
                for (p,) in self.con.execute(base + " UNION " + added, arguments + arguments)
            )
        )


def rank_relations(relations, vectors, question) -> list[list]:
    """Each fp32 dot product has the same reduction shape; ID ties are explicit."""
    scored = [(relation, float(vectors[relation] @ question)) for relation in relations]
    if any(not np.isfinite(score) for _, score in scored):
        raise ValueError("relation rank scores must be finite")
    ordered = sorted(scored, key=lambda pair: (-pair[1], pair[0]))[:MAX_RANKED_RELATIONS]
    return [[relation, rank] for rank, (relation, _) in enumerate(ordered)]


def cached_embeddings(pool: RelationPool, encoder: DenseEncoder, cache: Path):
    """Disk cache per KG/name table and encoder identity; read with mmap on reuse."""
    ids = sorted(pool.names)
    identity = {
        "kg": str(pool.data),
        "names": [[p, pool.names[p]] for p in ids],
        "encoder": encoder.metadata,
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    array, receipt = cache / f"{key}.npy", cache / f"{key}.json"
    if array.exists() and receipt.exists():
        if sha256_file(array) != json.loads(receipt.read_text())["sha256"]:
            raise ValueError(f"relation embedding cache changed: {array}")
    else:
        # Bound inference and transient embeddings independently of schema size.
        mapped = None
        partial = array.with_suffix(".partial.npy")
        for at in range(0, len(ids), 32):
            batch = encoder.encode([relation_text(pool.names[p]) for p in ids[at : at + 32]])
            if mapped is None:
                mapped = np.lib.format.open_memmap(
                    partial, mode="w+", dtype=np.float32, shape=(len(ids), batch.shape[1])
                )
            mapped[at : at + len(batch)] = batch
        if mapped is None:
            raise ValueError("KG has no relation names to embed")
        mapped.flush()
        del mapped
        partial.replace(array)
        receipt.write_text(
            json.dumps({"identity": identity, "sha256": sha256_file(array)}, indent=2) + "\n"
        )
    embeddings = np.load(array, mmap_mode="r", allow_pickle=False)
    return dict(zip(ids, embeddings, strict=True)), [array, receipt]


def request_lines(source):
    """At most one bounded request per read; reject invalid copies before publication."""
    with source.open("rb") as incoming:
        number = 0
        while line := incoming.readline(MAX_REQUEST_BYTES + 1):
            number += 1
            context = f"{source}:{number}"
            if len(line) > MAX_REQUEST_BYTES:
                raise ValueError(f"{context}: request exceeds {MAX_REQUEST_BYTES} bytes")
            try:
                request = json.loads(line)
                if not isinstance(request, dict) or FIELD in request:
                    raise ValueError(f"expected a request object without {FIELD}")
                question = request.get("question")
                if not isinstance(question, str) or not question.strip():
                    raise ValueError("expected nonempty question text in 'question'")
            except (ValueError, TypeError) as error:
                raise ValueError(f"{context}: {error}") from error
            yield line, request, context


def rank_requests(source: Path, out: Path, pool: RelationPool, encoder, vectors) -> int:
    """Insert one field before each closing brace; retain all original bytes."""
    source, out = Path(source), Path(out)
    if source.resolve() == out.resolve() or out.exists():
        raise ValueError(f"ranked requests need a new output file: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    count = 0
    try:
        with partial.open("xb") as written:
            for batch in batched(request_lines(source), BATCH):
                embeddings = encoder.encode([request["question"] for _, request, _ in batch])
                for (line, request, context), question in zip(batch, embeddings, strict=True):
                    try:
                        ranks = rank_relations(
                            pool.relations(request.get("kg_version")), vectors, question
                        )
                    except (ValueError, KeyError, TypeError) as error:
                        raise ValueError(f"{context}: {error}") from error
                    at = line.rfind(b"}")
                    addition = ", " + json.dumps(FIELD) + ": " + json.dumps(ranks)
                    written.write(line[:at] + addition.encode() + line[at:])
                    count += 1
        partial.replace(out)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "requests", "out", "cache"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=(
            Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
            / "models--facebook--contriever-msmarco/snapshots"
            / REVISION
        ),
    )
    args = parser.parse_args(argv)
    encoder = DenseEncoder(args.snapshot)
    with RelationPool(args.data) as pool:
        vectors, cache_inputs = cached_embeddings(pool, encoder, args.cache)
        count = rank_requests(args.requests, args.out, pool, encoder, vectors)
        report = {
            "requests": count,
            "field": FIELD,
            "encoder": encoder.metadata,
            "output_sha256": sha256_file(args.out),
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "provenance": provenance(
                [args.requests, *sorted(pool.inputs), *cache_inputs, Path(__file__)]
            ),
        }
    Path(str(args.out) + ".ranks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("requests", "output_sha256", "peak_rss_mib")}))


if __name__ == "__main__":
    main()
