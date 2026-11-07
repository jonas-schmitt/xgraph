"""Confirm removal of one queue-owned container before closing its occupancy."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from contextlib import suppress
from pathlib import Path


def cleanup_argvs(container):
    """Expose every conditional cleanup command to CPU owner previews."""
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", container):
        raise ValueError("invalid owned container name")
    literal = container.replace(".", r"\.")
    query = ["docker", "ps", "-aq", "--filter", f"name=^/{literal}$"]
    return [
        query,
        ["docker", "stop", "--time", "3", container],
        ["docker", "rm", "--force", container],
        query.copy(),
    ]


def stop_container(container, events):
    """Call only after the launcher has exited: it must not launch again after cleanup."""
    query, stop, remove, confirm = cleanup_argvs(container)

    def run(argv, timeout=5):
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)

    try:
        before = run(query)
        if before.returncode:
            raise ValueError("owned container cleanup unconfirmed")
        if before.stdout.strip():
            # Removal is still attempted if graceful stop fails or times out.
            with suppress(subprocess.TimeoutExpired):
                run(stop, timeout=8)
            run(remove)
        after = run(confirm)
        if after.returncode or after.stdout.strip():
            raise ValueError("owned container cleanup unconfirmed")
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("owned container cleanup unconfirmed") from error

    # Append exactly one end, only after successful absence verification. An
    # interrupted/invalid event stream remains uncertain rather than inventing a start.
    events = Path(events)
    data = events.read_bytes() if events.exists() else b""
    if data and not data.endswith(b"\n"):
        raise ValueError("incomplete owned container events")
    active, previous = False, 0.0
    for line in data.splitlines():
        row = json.loads(line)
        stamp = row["monotonic"]
        if (
            row.get("container") != container
            or type(stamp) not in (int, float)
            or not math.isfinite(stamp)
            or stamp < previous
            or (row["event"] == "container_start" and active)
            or (row["event"] == "container_end" and not active)
            or row["event"] not in ("container_start", "container_end")
        ):
            raise ValueError("invalid owned container events")
        active, previous = row["event"] == "container_start", stamp
    ended = time.monotonic()
    if previous > ended:
        raise ValueError("owned container event is in the future")
    if active:
        with events.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "event": "container_end",
                        "container": container,
                        "monotonic": ended,
                        "cleanup_confirmed": True,
                    }
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
    return ended
