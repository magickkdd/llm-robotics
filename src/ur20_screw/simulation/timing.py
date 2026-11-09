"""Wall-clock pacing helpers for UR20 simulation control intervals."""

from __future__ import annotations

import time


def wait_for_simulated_time(started: float, simulated_seconds: float) -> None:
    remaining = started + simulated_seconds - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)
