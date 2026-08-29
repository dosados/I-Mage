from __future__ import annotations

import threading
import time

from indexing.gpu_scheduler import GpuScheduler, GpuTaskCancelled


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert predicate()


def test_interactive_work_preempts_waiting_index_batches() -> None:
    scheduler = GpuScheduler()
    release = threading.Event()
    order: list[str] = []

    def blocker() -> None:
        with scheduler.acquire("active-index"):
            release.wait(2)

    def run(name: str, priority: int) -> None:
        with scheduler.acquire(name, priority=priority):
            order.append(name)

    active = threading.Thread(target=blocker)
    low = threading.Thread(target=run, args=("next-index", GpuScheduler.INDEXING))
    high = threading.Thread(target=run, args=("query", GpuScheduler.INTERACTIVE))
    active.start()
    _wait_until(lambda: scheduler.snapshot()["active"] == "active-index")
    low.start()
    high.start()
    _wait_until(lambda: len(scheduler.snapshot()["waiting"]) == 2)
    release.set()
    active.join(2)
    low.join(2)
    high.join(2)

    assert order == ["query", "next-index"]


def test_waiting_task_can_be_cancelled() -> None:
    scheduler = GpuScheduler()
    release = threading.Event()
    stop = threading.Event()
    cancelled: list[BaseException] = []

    def blocker() -> None:
        with scheduler.acquire("active-index"):
            release.wait(2)

    def waiter() -> None:
        try:
            with scheduler.acquire("cancel-me", should_stop=stop.is_set):
                pass
        except BaseException as exc:  # noqa: BLE001
            cancelled.append(exc)

    active = threading.Thread(target=blocker)
    waiting = threading.Thread(target=waiter)
    active.start()
    _wait_until(lambda: scheduler.snapshot()["active"] == "active-index")
    waiting.start()
    _wait_until(lambda: scheduler.snapshot()["waiting"] == ["cancel-me"])
    stop.set()
    waiting.join(2)
    release.set()
    active.join(2)

    assert len(cancelled) == 1
    assert isinstance(cancelled[0], GpuTaskCancelled)
