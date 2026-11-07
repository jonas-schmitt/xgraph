"""Drop the page cache of large files before a model loads.

On unified memory (the GB10) the GPU and the page cache share one pool, and CUDA
allocations fail when the page cache fills memory, even while MemAvailable counts that
cache as reclaimable. scripts/gpu-headroom.sh calls this when MemFree is below what a run
needs; the files are re-read from disk on demand.

    python -m xgraph.pagecache data /home/jonas/models/gemma-4-31B-it
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from pathlib import Path

MINIMUM = 1 << 28  # files of at least 256 MiB: model shards, KG indexes and stores


def large_files(roots: Iterable[str | Path], minimum: int = MINIMUM):
    """Regular files of at least minimum bytes under the roots, symlinks not followed."""
    for root in roots:
        for base, _, names in os.walk(root, followlinks=False):
            for name in names:
                path = os.path.join(base, name)
                try:
                    if not os.path.islink(path) and os.path.getsize(path) >= minimum:
                        yield path
                except OSError:
                    pass


def drop(roots: Iterable[str | Path], minimum: int = MINIMUM) -> int:
    """Advise the kernel to drop the cached pages of every large file under the roots;
    the number of files advised."""
    count = 0
    for path in large_files(roots, minimum):
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            count += 1
        except OSError:
            pass
        finally:
            os.close(fd)
    return count


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("roots", nargs="+", help="directories whose large files to drop")
    parser.add_argument("--minimum", type=int, default=MINIMUM, help="smallest file (bytes)")
    args = parser.parse_args(argv)
    print(f"page cache dropped for {drop(args.roots, args.minimum)} files")


if __name__ == "__main__":
    main()
