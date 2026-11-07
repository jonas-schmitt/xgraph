"""Build unique-alias MINTQA KG indexes with exact reuse and bounded-process provenance.

Legacy indexes do not record their alias policy or source hashes. To establish
identity, build once with the calibration builder/options and compare every file.
An identical candidate is reused via a worktree-local symlink; only the freshly
generated duplicate is removed. Subsequent invocations verify the frozen manifest.
"""

from __future__ import annotations

import argparse
import json
import resource
import shutil
from pathlib import Path

from xgraph.mintqa_panels import write_json
from xgraph.provenance import provenance, sha256_file


def file_inventory(directory: Path) -> list[dict]:
    return [
        {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in sorted(directory.iterdir())
        if p.is_file()
    ]


def build_unique_index(kg: Path, tokenizer: Path, out: Path, reuse=()) -> dict:
    """Use wikidata.build_index, unique aliases, default syntax, without verdicts."""
    from xgraph.syntax import SpanSyntax
    from xgraph.tokenizer import ServedTokenizer
    from xgraph.wikidata import build_index

    manifest = out.with_name(out.name + ".manifest.json")
    if manifest.exists():
        result = json.loads(manifest.read_text())
        expected = {
            str(p.resolve())
            for p in (
                *sorted(kg.glob("*")),
                tokenizer,
                Path(__file__),
                Path(__file__).with_name("wikidata.py"),
                Path(__file__).with_name("index.py"),
            )
            if p.is_file()
        }
        if expected != {str(Path(i["path"]).resolve()) for i in result["provenance"]["inputs"]}:
            raise ValueError("frozen index source paths changed")
        if any(sha256_file(Path(i["path"])) != i["sha256"] for i in result["provenance"]["inputs"]):
            raise ValueError("frozen index inputs changed")
        if file_inventory(out) != result["files"]:
            raise ValueError(f"frozen index changed: {out}")
        return result
    if out.exists() or out.is_symlink():
        raise ValueError(f"partial index {out}; preserve artifacts and choose a new directory")
    if not (kg / "sitelink_counts.npy").is_file():
        raise ValueError(f"{kg}: sitelink counts are required for the allfix rank")
    out.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out.parent).free
    if free < 2 * 1024**3:
        raise ValueError(f"not enough disk for index construction: {free} bytes free")
    inputs = [p for p in sorted(kg.iterdir()) if p.is_file()] + [
        tokenizer,
        Path(__file__),
        Path(__file__).with_name("wikidata.py"),
        Path(__file__).with_name("index.py"),
    ]
    before = provenance(inputs)
    served = ServedTokenizer.from_file(tokenizer)
    build_index(kg, served, SpanSyntax(), out, unique_aliases=True, verdicts=False)
    files = file_inventory(out)
    reused = None
    for candidate in reuse:
        candidate = Path(candidate).resolve()
        if candidate != out.resolve() and candidate.is_dir() and file_inventory(candidate) == files:
            shutil.rmtree(out)  # This invocation's verified duplicate, never the candidate.
            out.symlink_to(candidate, target_is_directory=True)
            reused = str(candidate)
            break
    if any(sha256_file(Path(i["path"])) != i["sha256"] for i in before["inputs"]):
        raise ValueError("index source changed during construction; output is incomplete")
    result = {
        "artifact_kind": "xgraph.mintqa_index",
        "version": 1,
        "settings": {"unique_aliases": True, "verdicts": False, "syntax": "default"},
        "tokenizer_fingerprint": served.fingerprint,
        "reused_from": reused,
        "index": str(out.resolve()),
        "files": files,
        "bytes": sum(f["bytes"] for f in files),
        "disk_free_before": free,
        "disk_free_after": shutil.disk_usage(out.parent).free,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "provenance": before,
    }
    write_json(manifest, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kg", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reuse", type=Path, action="append", default=[])
    result = build_unique_index(**vars(parser.parse_args(argv)))
    print(json.dumps({k: v for k, v in result.items() if k not in ("files", "provenance")}))


if __name__ == "__main__":
    main()
