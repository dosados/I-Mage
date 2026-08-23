"""Stress / latency tests for the SQLite catalog layer.

These target the bottlenecks that made a 30k-photo reconcile + status poll feel
stuck: filesystem walks on every /index/status, path.resolve() × N, huge IN
batches, and 3 parallel status requests fighting for the SQLite lock.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from db.database import Database
from db.types import MODULE_CLIP, MODULE_FACES, MODULE_YOLO
from helpers import LatencyRecorder, seed_catalog

# Synthetic catalog sizes. 8k is enough to catch O(N) regressions without making
# CI unbearably slow; 30k mirrors flickr30k for optional local runs.
STRESS_N = 8_000
LARGE_N = 30_000


@pytest.fixture
def stress_db(tmp_path: Path) -> tuple[Database, list[Path]]:
    image_dir = tmp_path / "photos"
    image_dir.mkdir()
    db = Database(path=tmp_path / "stress.db")
    paths = seed_catalog(
        db,
        image_dir,
        STRESS_N,
        mark_clip_done=STRESS_N // 2,
        mark_yolo_done=STRESS_N // 3,
        mark_faces_done=STRESS_N // 10,
    )
    yield db, paths
    db.close()


@pytest.mark.stress
class TestCatalogLatency:
    def test_catalog_module_stats_is_fast(self, stress_db, capsys) -> None:
        db, _paths = stress_db
        recorder = LatencyRecorder("catalog_module_stats")
        for _ in range(20):
            with db:
                recorder.measure(lambda: db.images.catalog_module_stats())
        print(recorder.summary())
        # Pure SQL COUNTs on 8k rows should be well under 50ms p95.
        assert recorder.p95 < 50, recorder.summary()
        assert recorder.mean < 20, recorder.summary()

    def test_stat_map_large_scope_uses_full_scan(self, stress_db, capsys) -> None:
        db, paths = stress_db
        recorder = LatencyRecorder("stat_map_by_path")
        with db:
            smap = recorder.measure(lambda: db.images.stat_map_by_path(paths))
        print(recorder.summary())
        assert len(smap) == STRESS_N
        # One full-table read must beat dozens of 900-wide IN queries.
        assert recorder.samples_ms[0] < 500, recorder.summary()

    def test_done_paths_for_module_large_scope(self, stress_db, capsys) -> None:
        db, paths = stress_db
        recorder = LatencyRecorder("done_paths_for_module")
        with db:
            done = recorder.measure(
                lambda: db.images.done_paths_for_module(MODULE_CLIP, paths)
            )
        print(recorder.summary())
        assert len(done) == STRESS_N // 2
        assert recorder.samples_ms[0] < 500, recorder.summary()

    def test_reconcile_noop_latency(self, stress_db, capsys) -> None:
        db, paths = stress_db
        recorder = LatencyRecorder("reconcile_noop")
        with db:
            result = recorder.measure(
                lambda: db.reconcile_paths(set(paths), remove_missing=False)
            )
        print(recorder.summary())
        assert result.upserted == 0
        # Unchanged 8k-file catalog: stat + one SQL scan, no hashing.
        assert recorder.samples_ms[0] < 2_000, recorder.summary()

    def test_reconcile_with_progress_no_double_commit_storm(
        self, stress_db, capsys
    ) -> None:
        db, paths = stress_db
        with db:
            run = db.index_runs.create(
                module="clip", mode="full", progress_total=0, phase="reconcile"
            )
            run_id = run.id

            commits = {"n": 0}
            original_commit = db.images._session.commit

            def counting_commit():
                commits["n"] += 1
                return original_commit()

            db.images._session.commit = counting_commit  # type: ignore[method-assign]

            def on_progress(done: int, total: int) -> None:
                if done == 0:
                    db.index_runs.set_progress_total(run_id, total)
                db.index_runs.update_progress(run_id, progress_done=done)
                # Intentionally no commit here (mirrors fixed run_scan callback).

            t0 = time.perf_counter()
            db.reconcile_paths(
                set(paths),
                remove_missing=False,
                on_progress=on_progress,
                commit_batch_size=500,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

        print(f"reconcile+progress: {elapsed_ms:.1f}ms commits={commits['n']}")
        # ~16 batch commits for 8k/500, not 2× that from a double-commit bug.
        assert commits["n"] <= (STRESS_N // 500) + 2
        assert elapsed_ms < 2_500


@pytest.mark.stress
class TestStatusPollContention:
    """Reproduce the UI bug: 3 parallel heavy status reads vs 1 light one."""

    def test_parallel_heavy_path_stats_are_slow(self, stress_db, capsys) -> None:
        """Legacy pattern: 3 threads × 3 module path-IN counts (the old bug)."""
        db_path = stress_db[0].path
        paths = stress_db[1]

        def heavy_status() -> float:
            local = Database(path=db_path)
            t0 = time.perf_counter()
            with local:
                for mod in (MODULE_YOLO, MODULE_CLIP, MODULE_FACES):
                    local.images.count_module_done_in_paths(paths, mod)
            local.close()
            return (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as pool:
            times = list(pool.map(lambda _: heavy_status(), range(3)))
        wall = (time.perf_counter() - t0) * 1000
        print(f"legacy 3× heavy status: wall={wall:.1f}ms each={times}")
        # Document the contention: wall time grows with parallel readers/writers.
        assert wall > 0

    def test_parallel_fast_catalog_stats_stay_cheap(self, stress_db, capsys) -> None:
        """Fixed pattern: SQL COUNTs, one logical status payload."""
        db_path = stress_db[0].path

        def fast_status() -> float:
            local = Database(path=db_path)
            t0 = time.perf_counter()
            with local:
                local.images.catalog_module_stats()
                for mod in (MODULE_YOLO, MODULE_CLIP, MODULE_FACES):
                    local.index_runs.get_active(module=mod)
                    local.index_runs.get_latest(module=mod)
            local.close()
            return (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as pool:
            times = list(pool.map(lambda _: fast_status(), range(3)))
        wall = (time.perf_counter() - t0) * 1000
        print(f"fixed 3× catalog status: wall={wall:.1f}ms each={times}")
        assert wall < 200, f"fast status poll too slow under concurrency: {wall:.1f}ms"
        assert max(times) < 150


@pytest.mark.stress
class TestLargeOptional:
    """Optional 30k run — skipped unless RUN_LARGE_DB_STRESS=1."""

    def test_reconcile_noop_30k(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import os

        if os.environ.get("RUN_LARGE_DB_STRESS") != "1":
            pytest.skip("set RUN_LARGE_DB_STRESS=1 to run 30k stress")

        image_dir = tmp_path / "photos"
        image_dir.mkdir()
        db = Database(path=tmp_path / "large.db")
        paths = seed_catalog(db, image_dir, LARGE_N, mark_clip_done=1000)

        t0 = time.perf_counter()
        with db:
            stats = db.images.catalog_module_stats()
            result = db.reconcile_paths(set(paths), remove_missing=False)
        elapsed = time.perf_counter() - t0
        db.close()

        print(f"30k reconcile+stats: {elapsed:.2f}s stats={stats} upserted={result.upserted}")
        assert result.upserted == 0
        assert elapsed < 5.0
