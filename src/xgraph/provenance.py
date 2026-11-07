"""Provenance for results that leave the repository: inputs, command, code, and environment."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    repo = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def provenance(inputs: list[Path]) -> dict:
    """Input files with their sizes and SHA-256, the command, the commit, and the versions."""
    status = _git("status", "--porcelain")
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": [Path(sys.executable).name, *sys.argv],
        "inputs": [
            {"path": str(p), "bytes": Path(p).stat().st_size, "sha256": sha256_file(p)}
            for p in inputs
        ],
        "repo_commit": _git("rev-parse", "HEAD"),
        "repo_dirty": bool(status) if status is not None else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: metadata.version(name) for name in ("xgraph", "numpy", "tokenizers")},
    }
