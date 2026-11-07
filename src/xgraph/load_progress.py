"""Bound engine startup silence independently of the lifetime GPU allowance."""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path


class LoadProgressTimeout(subprocess.TimeoutExpired):
    pass


class StartupTimeout(subprocess.TimeoutExpired):
    pass


class LoadProgress:
    """Every new byte counts during startup; readiness disarms the silence timer."""

    markers = re.compile(
        r"\[xgraph-load\] begin|Initializing a V1 LLM engine|Using V2 Model Runner|"
        r"\[xgraph-load\] ready|init engine \("
    )

    def __init__(self):
        self.last = None
        self.started = None
        self.tail = ""

    def feed(self, text, now):
        if not text:
            return
        if isinstance(text, bytes):
            text = text.decode("latin1")  # Markers are ASCII; every byte is progress.
        if self.last is not None:
            self.last = now
        joined = self.tail + text
        for match in self.markers.finditer(joined):
            if match.end() <= len(self.tail):
                continue
            if match[0] in ("[xgraph-load] ready", "init engine ("):
                self.last = self.started = None
            else:
                if self.started is None:
                    self.started = now
                self.last = now
        # Retain only enough bytes to recognize a marker split across reads.
        self.tail = joined[-32:]


def wait_for_load(process, log_path, silence_seconds, wall_seconds, *, startup_seconds=1800):
    progress = LoadProgress()
    deadline = time.monotonic() + wall_seconds
    with Path(log_path).open("rb") as stream:
        while True:
            now = time.monotonic()
            progress.feed(stream.read(), now)
            if progress.last is not None and now - progress.last >= silence_seconds:
                raise LoadProgressTimeout(process.args, silence_seconds)
            if progress.started is not None and now - progress.started >= startup_seconds:
                raise StartupTimeout(process.args, startup_seconds)
            remaining = deadline - now
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, wall_seconds)
            try:
                return process.wait(timeout=min(0.1, silence_seconds / 2, remaining))
            except subprocess.TimeoutExpired:
                pass
