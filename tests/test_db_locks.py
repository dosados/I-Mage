"""Targeted SQLite lock / concurrency checks (lightweight, no stress catalog).

Mirrors real app patterns: WAL writers + parallel status readers, thread-local
sessions on a shared Database, short overlapping writes. Catalog size stays
tiny so CI and a laptop without power stay responsive.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest
from sqlalchemy.exc import OperationalError

from db.database import Database
from db.types import MODULE_CLIP, MODULE_FACES, MODULE_YOLO
from helpers import make_image_file, register_file, seed_catalog

LOCK_N = 40  # enough rows to exercise IN/scan paths, cheap to seed


@pytest.fixture
def lock_db(tmp_path: Path) -> tuple[Database, list[Path]]:
    image_dir = tmp_path / "photos"
    image_dir.mkdir()
    db = Database(path=tmp_path / "locks.db")
    paths = seed_catalog(
        db,
        image_dir,
        LOCK_N,
        mark_clip_done=LOCK_N // 2,
        mark_yolo_done=LOCK_N // 3,
        mark_faces_done=LOCK_N // 4,
    )
    yield db, paths
    db.close()


def _is_lock_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "database is locked" in text or "database table is locked" in text


class TestPragmas:
    def test_wal_and_busy_timeout(self, lock_db) -> None:
        db, _ = lock_db
        with db.engine.connect() as conn:
            raw = conn.connection.dbapi_connection
            cursor = raw.cursor()
            cursor.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            cursor.execute("PRAGMA busy_timeout")
            timeout = cursor.fetchone()[0]
            cursor.close()
        assert str(mode).lower() == "wal"
        assert int(timeout) >= 30_000


class TestConcurrentAccess:
    def test_parallel_status_reads_no_lock_error(self, lock_db) -> None:
        """UI polls /index/status from several threads while catalog is idle."""
        db_path = lock_db[0].path
        errors: list[BaseException] = []

        def status_once() -> None:
            local = Database(path=db_path)
            try:
                with local:
                    local.images.catalog_module_stats()
                    for mod in (MODULE_YOLO, MODULE_CLIP, MODULE_FACES):
                        local.index_runs.get_active(module=mod)
                        local.index_runs.get_latest(module=mod)
                    local.images.count_all()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                local.close()

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: status_once(), range(12)))

        lock_errors = [e for e in errors if _is_lock_error(e)]
        assert errors == [], f"unexpected errors: {errors!r}"
        assert lock_errors == []

    def test_writer_plus_status_readers(self, lock_db) -> None:
        """One writer reconciles while others poll status — must not deadlock."""
        db_path = lock_db[0].path
        paths = lock_db[1]
        barrier = threading.Barrier(4)
        errors: list[BaseException] = []

        def writer() -> None:
            local = Database(path=db_path)
            try:
                barrier.wait(timeout=5)
                with local:
                    local.reconcile_paths(set(paths), remove_missing=False)
                    run = local.index_runs.create(
                        module="clip", mode="full", progress_total=LOCK_N
                    )
                    for done in (10, 20, LOCK_N):
                        local.index_runs.update_progress(run.id, progress_done=done)
                        local.commit()
                    local.index_runs.mark_done(run.id)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                local.close()

        def reader() -> None:
            local = Database(path=db_path)
            try:
                barrier.wait(timeout=5)
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline:
                    with local:
                        local.images.catalog_module_stats()
                        local.index_runs.get_active(module="clip")
                    time.sleep(0.02)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                local.close()

        threads = [threading.Thread(target=writer)] + [
            threading.Thread(target=reader) for _ in range(3)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
            assert not thread.is_alive(), "thread hung — possible lock stall"

        assert errors == [], f"errors under writer+readers: {errors!r}"

    def test_shared_database_thread_local_writes(
        self, lock_db, tmp_path: Path
    ) -> None:
        """Same Database facade, different threads → no lock / no session clash."""
        shared, _ = lock_db
        extra_dir = tmp_path / "extras"
        extra_dir.mkdir()
        extras = [make_image_file(extra_dir, f"extra_{i}.jpg") for i in range(6)]
        errors: list[BaseException] = []

        def worker(path: Path) -> None:
            try:
                with shared:
                    register_file(shared, path)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(worker, extras))

        assert errors == []
        with shared:
            # Original LOCK_N + 6 extras.
            assert shared.images.count_all() == LOCK_N + 6

    def test_two_writers_short_transactions(self, lock_db) -> None:
        """Two Database instances writing index_runs concurrently."""
        db_path = lock_db[0].path
        errors: list[BaseException] = []
        created: list[str] = []
        lock = threading.Lock()

        def write_runs(module: str) -> None:
            local = Database(path=db_path)
            try:
                for i in range(5):
                    with local:
                        run = local.index_runs.create(
                            module=module,
                            mode="gap",
                            progress_total=10,
                        )
                        local.index_runs.update_progress(run.id, progress_done=i + 1)
                        local.index_runs.mark_done(run.id)
                        with lock:
                            created.append(run.id)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                local.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(write_runs, "clip"),
                pool.submit(write_runs, "yolo"),
            ]
            for fut in as_completed(futures):
                fut.result(timeout=10)

        assert errors == []
        assert len(created) == 10
        assert len(set(created)) == 10


class TestBusyTimeoutHonored:
    def test_busy_timeout_waits_then_write_succeeds(self, tmp_path: Path) -> None:
        """Second writer waits under busy_timeout, then commits after lock release.

        WAL allows concurrent readers, so the contender must *write*. Raw sqlite3
        holds BEGIN IMMEDIATE; the app connection uses busy_timeout=30s.
        """
        db_file = tmp_path / "busy.db"
        Database(path=db_file).close()

        holder = sqlite3.connect(str(db_file), timeout=0.05)
        holder.execute("PRAGMA journal_mode=WAL")
        holder.execute("BEGIN IMMEDIATE")
        holder.execute("CREATE TABLE IF NOT EXISTS lock_probe (id INTEGER)")

        ready = threading.Event()
        finished = threading.Event()
        result: dict[str, float | BaseException | None] = {"error": None, "waited": 0.0}

        def contender() -> None:
            ready.set()
            t0 = time.perf_counter()
            local = Database(path=db_file)
            try:
                with local:
                    # Write path — blocked until holder releases IMMEDIATE lock.
                    local.index_runs.create(
                        module="clip", mode="gap", progress_total=1
                    )
            except BaseException as exc:  # noqa: BLE001
                result["error"] = exc
            finally:
                local.close()
                result["waited"] = time.perf_counter() - t0
                finished.set()

        thread = threading.Thread(target=contender)
        thread.start()
        assert ready.wait(timeout=2)
        time.sleep(0.3)
        holder.rollback()
        holder.close()
        assert finished.wait(timeout=10)
        thread.join(timeout=2)

        assert result["error"] is None, result["error"]
        assert float(result["waited"]) >= 0.25

    def test_immediate_lock_failure_without_waiting_forever(
        self, tmp_path: Path
    ) -> None:
        """Sanity: a connection with timeout=0 fails quickly while locked."""
        db_file = tmp_path / "busy0.db"
        Database(path=db_file).close()

        holder = sqlite3.connect(str(db_file), timeout=0)
        holder.execute("PRAGMA journal_mode=WAL")
        holder.execute("BEGIN IMMEDIATE")
        holder.execute("CREATE TABLE IF NOT EXISTS lock_probe (id INTEGER)")

        contender = sqlite3.connect(str(db_file), timeout=0)
        t0 = time.perf_counter()
        with pytest.raises(sqlite3.OperationalError) as exc_info:
            contender.execute("BEGIN IMMEDIATE")
        elapsed = time.perf_counter() - t0
        contender.close()
        holder.rollback()
        holder.close()

        assert _is_lock_error(exc_info.value)
        assert elapsed < 1.0


class TestNoHangingSessions:
    def test_failed_context_releases_for_next_writer(
        self, db_path: Path, image_dir: Path
    ) -> None:
        path = make_image_file(image_dir, "a.jpg")
        db = Database(path=db_path)
        try:
            with db:
                register_file(db, path)
                raise RuntimeError("abort")
        except RuntimeError:
            pass

        # Second context must not see a leftover write lock / dirty session.
        with db:
            assert db.images.count_all() == 0
            register_file(db, path)
            assert db.images.count_all() == 1
        db.close()

    def test_operational_error_on_closed_engine_is_not_lock(
        self, lock_db
    ) -> None:
        db, _ = lock_db
        db.close()
        # Re-open session after close on same facade should still work (new session).
        with db:
            n = db.images.count_all()
        assert n == LOCK_N
        # Ensure we didn't accidentally wrap a lock error.
        try:
            with db:
                db.images.catalog_module_stats()
        except OperationalError as exc:
            assert not _is_lock_error(exc)
            raise
