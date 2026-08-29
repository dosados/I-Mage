from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from db.database import Database
from db.types import MODULE_CLIP, MODULE_YOLO
from fakes import FakeClip, FakeFaces, FakeVectorStore, FakeYolo
from helpers import configure_scan, make_image_file
from indexing.executor import IndexExecutor, IndexRunConflictError
from indexing.runner import IndexModels, reconcile_catalog


class SlowClip(FakeClip):
    def __init__(self, delay: float = 0.02) -> None:
        super().__init__()
        self.delay = delay
        self.started = threading.Event()

    def encode_image(self, image):
        self.started.set()
        time.sleep(self.delay)
        return super().encode_image(image)


def _models(*, clip: FakeClip | None = None) -> IndexModels:
    return IndexModels(
        clip=clip or FakeClip(),
        yolo=FakeYolo(),
        faces=FakeFaces(),
    )


def _reconcile(db: Database) -> None:
    with db:
        config = db.get_scan_config()
    reconcile_catalog(db, config)


async def _await_run(executor: IndexExecutor, run_id: str) -> None:
    task = executor._tasks.get(run_id)
    assert task is not None
    await asyncio.wait_for(task, timeout=15)


def test_full_run_on_empty_database_finishes_cleanly(
    db: Database, image_dir: Path
) -> None:
    configure_scan(db, [image_dir])
    executor = IndexExecutor(db, FakeVectorStore(), _models())

    async def scenario() -> str:
        run_id = await executor.start_full_run(MODULE_YOLO)
        await _await_run(executor, run_id)
        return run_id

    run_id = asyncio.run(scenario())
    with db:
        run = db.index_runs.get(run_id)
        assert run is not None
        assert run.status == "done"
        assert run.progress_total == 0
        assert db.images.count_all() == 0
    assert not executor.has_manual_run_in_progress()


def test_full_then_partial_run_only_indexes_the_gap(
    db: Database, image_dir: Path
) -> None:
    for index in range(12):
        make_image_file(image_dir, f"{index}.jpg")
    configure_scan(db, [image_dir])
    _reconcile(db)
    store = FakeVectorStore()
    first = FakeClip()
    executor = IndexExecutor(db, store, _models(clip=first))

    async def first_run() -> None:
        run_id = await executor.start_full_run(MODULE_CLIP)
        await _await_run(executor, run_id)

    asyncio.run(first_run())
    assert len(first.encode_image_calls) == 12

    second = FakeClip()
    executor = IndexExecutor(db, store, _models(clip=second))

    async def second_run() -> str:
        run_id = await executor.start_full_run(MODULE_CLIP)
        await _await_run(executor, run_id)
        return run_id

    run_id = asyncio.run(second_run())
    assert second.encode_image_calls == []
    with db:
        run = db.index_runs.get(run_id)
        assert run is not None and run.status == "done"
        assert db.images.count_module_done(MODULE_CLIP) == 12


def test_same_module_conflicts_and_start_does_not_wait_for_io(
    db: Database, image_dir: Path
) -> None:
    make_image_file(image_dir, "blocked.jpg")
    configure_scan(db, [image_dir])
    _reconcile(db)
    clip = SlowClip(delay=0.3)
    executor = IndexExecutor(db, FakeVectorStore(), _models(clip=clip))

    async def scenario() -> None:
        started_at = time.perf_counter()
        run_id = await executor.start_full_run(MODULE_CLIP)
        assert time.perf_counter() - started_at < 1.0
        assert await asyncio.to_thread(clip.started.wait, 3)
        with pytest.raises(IndexRunConflictError):
            await executor.start_full_run(MODULE_CLIP)

        # The event loop and an independent status read remain responsive while
        # model inference is blocked in the executor thread.
        await asyncio.wait_for(asyncio.sleep(0), timeout=0.2)
        active = await asyncio.wait_for(
            asyncio.to_thread(_active_run_id, db, MODULE_CLIP), timeout=1.0
        )
        assert active == run_id
        await _await_run(executor, run_id)

    asyncio.run(scenario())
    assert not executor.has_manual_run_in_progress()


def _active_run_id(db: Database, module: str) -> str | None:
    with db:
        run = db.index_runs.get_active(module=module)
        return run.id if run is not None else None


def test_interrupted_run_is_recoverable_and_does_not_reindex_done_rows(
    db: Database, image_dir: Path
) -> None:
    total = 96
    for index in range(total):
        make_image_file(image_dir, f"{index:03d}.jpg")
    configure_scan(db, [image_dir])
    _reconcile(db)
    store = FakeVectorStore()
    slow = SlowClip(delay=0.03)
    executor = IndexExecutor(db, store, _models(clip=slow))

    async def interrupt() -> str:
        run_id = await executor.start_full_run(MODULE_CLIP)
        assert await asyncio.to_thread(slow.started.wait, 3)
        deadline = time.monotonic() + 3
        while len(slow.encode_image_calls) < 3 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        await executor.shutdown(timeout=5)
        return run_id

    interrupted_id = asyncio.run(interrupt())
    with db:
        interrupted = db.index_runs.get(interrupted_id)
        done_before_resume = db.images.count_module_done(MODULE_CLIP)
    assert interrupted is not None
    assert interrupted.status == "failed"
    assert "Остановлено" in (interrupted.last_error or "")
    assert 0 < done_before_resume < total
    assert not executor.has_manual_run_in_progress()

    resumed_model = FakeClip()
    resumed = IndexExecutor(db, store, _models(clip=resumed_model))

    async def resume() -> str:
        run_id = await resumed.start_full_run(MODULE_CLIP)
        await _await_run(resumed, run_id)
        return run_id

    resumed_id = asyncio.run(resume())
    with db:
        completed = db.index_runs.get(resumed_id)
        assert completed is not None and completed.status == "done"
        assert db.images.count_module_done(MODULE_CLIP) == total
    assert len(resumed_model.encode_image_calls) == total - done_before_resume


def test_distinct_module_runs_complete_without_deadlock(
    db: Database, image_dir: Path
) -> None:
    for index in range(6):
        make_image_file(image_dir, f"{index}.jpg")
    configure_scan(db, [image_dir])
    _reconcile(db)
    executor = IndexExecutor(db, FakeVectorStore(), _models(clip=SlowClip(0.01)))

    async def scenario() -> tuple[str, str]:
        clip_id = await executor.start_full_run(MODULE_CLIP)
        yolo_id = await executor.start_full_run(MODULE_YOLO)
        await asyncio.wait_for(
            asyncio.gather(
                executor._tasks[clip_id],
                executor._tasks[yolo_id],
            ),
            timeout=10,
        )
        return clip_id, yolo_id

    run_ids = asyncio.run(scenario())
    with db:
        assert all(db.index_runs.get(run_id).status == "done" for run_id in run_ids)  # type: ignore[union-attr]
    assert not executor.has_manual_run_in_progress()
