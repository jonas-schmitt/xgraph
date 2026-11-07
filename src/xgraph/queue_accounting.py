"""Durable container occupancy and watchdog accounting shared by owner queues."""

from __future__ import annotations

import json
import math
import os
import subprocess
from contextlib import suppress


def gpu_occupancy(path, offset, started, *, require_end=False):
    """Menu-control's durable launch boundaries, including failed launch intent."""
    import time

    with path.open("rb") as stream:
        if os.fstat(stream.fileno()).st_size < offset:
            raise ValueError("GPU launch events were truncated")
        stream.seek(offset)
        data = stream.read()
    # The writer can append after the caller's clock sample but before this read.
    now = time.monotonic()
    if data and not data.endswith(b"\n"):
        raise ValueError("incomplete GPU launch event")
    active, seconds, launched, previous = None, 0.0, False, started
    for line in data.splitlines():
        event = json.loads(line)
        stamp = event["monotonic"]
        if (
            type(stamp) not in (int, float)
            or not math.isfinite(stamp)
            or not previous <= stamp <= now
        ):
            raise ValueError("invalid GPU launch event timestamp")
        previous = stamp
        if event["event"] == "container_start" and active is None:
            active, launched = stamp, True
        elif event["event"] == "container_end" and active is not None:
            seconds += stamp - active
            active = None
        else:
            raise ValueError("invalid GPU launch event sequence")
    if require_end and active is not None:
        raise ValueError("stage succeeded without a container end event")
    return seconds + (now - active if active is not None else 0), launched


def wait_stage(process, stage, started, wall_started, events, offset):
    """Poll GPU occupancy independently of the preparation/headroom wall bound."""
    import time

    code = None
    while True:
        now = time.monotonic()
        seconds, launched = (
            gpu_occupancy(events, offset, started) if events else (now - started, True)
        )
        left = stage.get("wall_timeout_seconds", 24 * 3600) - (now - wall_started)
        gpu = stage.get("gpu_seconds", 0)
        if left <= 0 or (gpu and launched and seconds >= gpu):
            raise subprocess.TimeoutExpired(stage["argv"], gpu if left > 0 else 0)
        if code is not None:
            return code
        with suppress(subprocess.TimeoutExpired):
            code = process.wait(
                timeout=min(1, left, gpu - seconds) if gpu and launched else min(1, left)
            )
