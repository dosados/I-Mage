from __future__ import annotations

import asyncio
from pathlib import Path

from db.database import Database
from db.scan_config import ScanConfig
from db.types import MODULE_CLIP
from fakes import FakeClip, FakeFaces, FakeVectorStore, FakeYolo
from helpers import configure_scan, make_image_file
from indexing.background_indexer import background_indexer_loop
from indexing.executor import IndexExecutor, IndexRunConflictError
from indexing.runner import IndexModels, reconcile_catalog


def _models(*, clip: FakeClip | None = None) -> IndexModels:
    return IndexModels(
        clip=clip or FakeClip(),
        yolo=FakeYolo(),
        faces=FakeFaces(),
    )


def test_background_gap_indexes_only_missing_rows_without_manual_run_record(
    db: Database, image_dir: Path
) -> None:
    paths = [make_image_file(image_dir, f"{index}.jpg") for index in range(8)]
    config = configure_scan(db, [image_dir])
    reconcile_catalog(db, config)
    store = FakeVectorStore()
    initial = FakeClip()
    manual = IndexExecutor(db, store, _models(clip=initial))

    async def seed_index() -> None:
        run_id = await manual.start_full_run(MODULE_CLIP)
        await manual._tasks[run_id]

    asyncio.run(seed_index())
    assert len(initial.encode_image_calls) == len(paths)

    # A manual catalog reconcile invalidates the changed row. Background gap
    # processes it without creating a manual module run record.
    paths[3].write_bytes(b"changed-content")
    reconcile_catalog(db, config)
    background_clip = FakeClip()
    background = IndexExecutor(db, store, _models(clip=background_clip))
    with db:
        config = db.get_scan_config()
        db.scan_config.save(
            ScanConfig(
                include_directories=config.include_directories,
                ignore_globs=config.ignore_globs,
                background_indexer_enabled=True,
                schedule_interval_days=1,
                background_modules=[MODULE_CLIP],
            )
        )
        latest_before = db.index_runs.get_latest(module=MODULE_CLIP)

    async def run_gap() -> None:
        await background.start_background_gap()
        task = background._tasks.get("background-gap")
        assert task is not None
        await asyncio.wait_for(task, timeout=10)

    asyncio.run(run_gap())
    assert [path.name for path in background_clip.encode_image_calls] == [paths[3].name]
    with db:
        latest_after = db.index_runs.get_latest(module=MODULE_CLIP)
        config_after = db.get_scan_config()
    assert latest_before is not None and latest_after is not None
    assert latest_after.id == latest_before.id
    assert config_after.last_background_run_at is not None


def test_background_loop_triggers_when_due_and_stops_cleanly(db: Database) -> None:
    with db:
        db.scan_config.save(
            ScanConfig(
                background_indexer_enabled=True,
                schedule_interval_days=1,
                background_modules=[MODULE_CLIP],
                last_background_run_at=None,
            )
        )

    class RecordingExecutor:
        def __init__(self) -> None:
            self.called = asyncio.Event()
            self.calls = 0

        async def start_background_gap(self) -> None:
            self.calls += 1
            self.called.set()

    async def scenario() -> int:
        stop = asyncio.Event()
        executor = RecordingExecutor()
        task = asyncio.create_task(
            background_indexer_loop(
                db=db,
                vector_store=FakeVectorStore(),
                models=_models(),
                stop_event=stop,
                executor=executor,  # type: ignore[arg-type]
            )
        )
        await asyncio.wait_for(executor.called.wait(), timeout=2)
        stop.set()
        await asyncio.wait_for(task, timeout=2)
        return executor.calls

    assert asyncio.run(scenario()) == 1


def test_background_loop_tolerates_manual_run_conflict(db: Database) -> None:
    with db:
        db.scan_config.save(
            ScanConfig(
                background_indexer_enabled=True,
                last_background_run_at=None,
            )
        )

    class ConflictingExecutor:
        def __init__(self) -> None:
            self.called = asyncio.Event()

        async def start_background_gap(self) -> None:
            self.called.set()
            raise IndexRunConflictError("manual run")

    async def scenario() -> None:
        stop = asyncio.Event()
        executor = ConflictingExecutor()
        task = asyncio.create_task(
            background_indexer_loop(
                db=db,
                vector_store=FakeVectorStore(),
                models=_models(),
                stop_event=stop,
                executor=executor,  # type: ignore[arg-type]
            )
        )
        await asyncio.wait_for(executor.called.wait(), timeout=2)
        stop.set()
        await asyncio.wait_for(task, timeout=2)

    asyncio.run(scenario())
