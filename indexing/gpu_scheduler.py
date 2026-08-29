from __future__ import annotations

import heapq
import itertools
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


class GpuTaskCancelled(RuntimeError):
    pass


@dataclass(order=True)
class _Ticket:
    priority: int
    sequence: int
    name: str = field(compare=False)


class GpuScheduler:
    """Cooperative, priority-aware scheduler for the single local GPU.

    Indexers acquire the device for one bounded batch at a time. Interactive
    inference uses a higher priority and can therefore run between batches.
    """

    INTERACTIVE = 0
    INDEXING = 10

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._queue: list[_Ticket] = []
        self._sequence = itertools.count()
        self._active: _Ticket | None = None

    @contextmanager
    def acquire(
        self,
        name: str,
        *,
        priority: int = INDEXING,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[None]:
        ticket = _Ticket(priority, next(self._sequence), name)
        with self._condition:
            heapq.heappush(self._queue, ticket)
            while self._active is not None or self._queue[0] is not ticket:
                if should_stop is not None and should_stop():
                    self._queue.remove(ticket)
                    heapq.heapify(self._queue)
                    self._condition.notify_all()
                    raise GpuTaskCancelled(f"GPU task cancelled while waiting: {name}")
                self._condition.wait(timeout=0.1)
            heapq.heappop(self._queue)
            self._active = ticket

        try:
            if should_stop is not None and should_stop():
                raise GpuTaskCancelled(f"GPU task cancelled before start: {name}")
            yield
        finally:
            with self._condition:
                if self._active is ticket:
                    self._active = None
                self._condition.notify_all()

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "active": self._active.name if self._active is not None else None,
                "waiting": [ticket.name for ticket in sorted(self._queue)],
            }


_gpu_scheduler = GpuScheduler()


def get_gpu_scheduler() -> GpuScheduler:
    return _gpu_scheduler
