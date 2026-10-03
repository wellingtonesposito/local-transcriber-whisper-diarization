"""Thread-safe fan-out of job events to SSE subscribers."""
from __future__ import annotations

import asyncio
import threading


class EventBus:
    def __init__(self) -> None:
        self._subs: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
        self._lock = threading.Lock()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        with self._lock:
            self._subs.append((asyncio.get_running_loop(), q))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subs = [(l, x) for l, x in self._subs if x is not q]

    def publish(self, event: dict) -> None:
        with self._lock:
            subs = list(self._subs)
        for loop, q in subs:
            def put(q=q):
                if q.full():  # slow client: drop the oldest event rather than block the worker thread
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait(event)
            try:
                loop.call_soon_threadsafe(put)
            except RuntimeError:  # loop closed
                pass
