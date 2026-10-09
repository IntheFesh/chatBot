"""Polling helpers for tests that wait on real threads, tasks or processes."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable


async def wait_until(predicate: Callable[[], bool], timeout: float = 5.0, interval: float = 0.02) -> None:
    """Await until ``predicate()`` is true (real time); fail the test on timeout."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"condition not met within {timeout}s")
        await asyncio.sleep(interval)


def wait_until_sync(predicate: Callable[[], bool], timeout: float = 10.0, interval: float = 0.05) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"condition not met within {timeout}s")
        time.sleep(interval)
