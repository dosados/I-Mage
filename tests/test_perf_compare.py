"""Performance comparisons: legacy slow paths vs current fast paths.

These document the regressions we fixed (status poll storms, N+1 id maps,
huge MatchAny filters, GPU bruteforce fallback) and assert the fixed path
stays within a clear budget relative to the legacy one.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from db.database import Database
from db.hash import resolved_path_key
from db.types import MODULE_CLIP, MODULE_FACES, MODULE_YOLO
from fakes import FakeClip, FakeFaces, FakeVectorStore, FakeYolo, unit_vec
from helpers import LatencyRecorder, configure_scan, make_image_file, register_file, seed_catalog
from indexing.faces import index_faces_image
from indexing.runner import module_stats_in_scope
from ml.faces.base import Face

PERF_N = 3_000


@pytest.fixture
def perf_catalog(tmp_path: Path):
    image_dir = tmp_path / "photos"
    image_dir.mkdir()
    db = Database(path=tmp_path / "perf.db")
    paths = seed_catalog(
        db,
        image_dir,
        PERF_N,
        mark_clip_done=PERF_N // 2,
        mark_yolo_done=PERF_N // 3,
        mark_faces_done=PERF_N // 4,
    )
    configure_scan(db, [image_dir])
    yield db, paths, image_dir
    db.close()


@pytest.mark.stress
class TestCatalogStatsPerf:
    def test_catalog_counts_beat_path_scoped_counts(self, perf_catalog, capsys) -> None:
        db, paths, _ = perf_catalog
        legacy = LatencyRecorder("legacy_path_IN_x3")
        fixed = LatencyRecorder("catalog_module_stats")

        def legacy_once():
            with db:
                for mod in (MODULE_YOLO, MODULE_CLIP, MODULE_FACES):
                    db.images.count_module_done_in_paths(paths, mod)

        def fixed_once():
            with db:
                db.images.catalog_module_stats()

        for _ in range(8):
            legacy.measure(legacy_once)
            fixed.measure(fixed_once)

        print(legacy.summary())
        print(fixed.summary())
        # Fixed path should be clearly faster (typically 10–100×).
        assert fixed.mean < legacy.mean
        assert fixed.p95 < 30
        assert fixed.mean * 5 < legacy.mean or fixed.mean < 5


@pytest.mark.stress
class TestIdMapPerf:
    def test_batch_id_map_beats_n_plus_one(self, perf_catalog, capsys) -> None:
        db, paths, _ = perf_catalog
        sample = paths  # full scope

        def n_plus_one():
            with db:
                mapping = {}
                for path in sample:
                    record = db.images.get_by_path(path)
                    if record is not None:
                        mapping[record.id] = path
                return mapping

        def batched():
            with db:
                return db.images.id_to_path_for_scope(sample)

        t0 = time.perf_counter()
        legacy_map = n_plus_one()
        legacy_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        fast_map = batched()
        fast_ms = (time.perf_counter() - t0) * 1000

        print(f"id_map N+1: {legacy_ms:.1f}ms  batch: {fast_ms:.1f}ms  n={PERF_N}")
        assert set(legacy_map) == set(fast_map)
        assert fast_ms < legacy_ms
        assert fast_ms < 400
        # Expect at least a 5× win on a few thousand rows.
        assert fast_ms * 5 < legacy_ms or fast_ms < 50


@pytest.mark.stress
class TestStatusPollPerf:
    def test_parallel_catalog_stats_vs_path_stats(self, perf_catalog, capsys) -> None:
        db_path = perf_catalog[0].path
        paths = perf_catalog[1]

        def heavy():
            local = Database(path=db_path)
            t0 = time.perf_counter()
            with local:
                for mod in (MODULE_YOLO, MODULE_CLIP, MODULE_FACES):
                    module_stats_in_scope(local, paths, mod)
            local.close()
            return (time.perf_counter() - t0) * 1000

        def light():
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
        with ThreadPoolExecutor(3) as pool:
            heavy_times = list(pool.map(lambda _: heavy(), range(3)))
        heavy_wall = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        with ThreadPoolExecutor(3) as pool:
            light_times = list(pool.map(lambda _: light(), range(3)))
        light_wall = (time.perf_counter() - t0) * 1000

        print(f"legacy 3× status wall={heavy_wall:.1f}ms each={heavy_times}")
        print(f"fixed  3× status wall={light_wall:.1f}ms each={light_times}")
        assert light_wall < heavy_wall
        assert light_wall < 150


@pytest.mark.stress
class TestFaceSearchPerf:
    def test_indexed_search_faster_than_bruteforce_analyze(
        self, tmp_path: Path, capsys
    ) -> None:
        image_dir = tmp_path / "photos"
        image_dir.mkdir()
        n = 120
        db = Database(path=tmp_path / "faces.db")
        store = FakeVectorStore()
        indexer = FakeFaces()
        query = unit_vec(11)

        paths = []
        with db:
            for i in range(n):
                path = make_image_file(image_dir, f"{i:04d}.jpg", f"x{i}".encode())
                paths.append(path)
                register_file(db, path)
                emb = query if i == 17 else unit_vec(i + 20)
                indexer.set_faces(
                    path.name,
                    [Face(bbox=(0, 0, 1, 1), detection_score=0.9, embedding=emb)],
                )
                index_faces_image(
                    db, store, path, indexer, model_version=indexer.model_name
                )
        configure_scan(db, [image_dir])

        # Separate recognizer for search with simulated GPU latency.
        searcher = FakeFaces()
        for path in paths:
            searcher.set_faces(path.name, indexer._by_key[path.name])
        _orig = searcher.analyze

        def slow_analyze(image):
            time.sleep(0.002)
            return _orig(image)

        searcher.analyze = slow_analyze  # type: ignore[method-assign]

        from api.search import run_search_by_face
        from ml.faces.service import search_by_face

        searcher.analyze_calls.clear()
        t0 = time.perf_counter()
        indexed = run_search_by_face(query, searcher, db, store, k=5, threshold=0.2)
        indexed_ms = (time.perf_counter() - t0) * 1000
        indexed_analyzes = len(searcher.analyze_calls)

        searcher.analyze_calls.clear()
        t0 = time.perf_counter()
        brute = search_by_face(query, paths, searcher, k=5, threshold=0.2)
        brute_ms = (time.perf_counter() - t0) * 1000
        brute_analyzes = len(searcher.analyze_calls)

        print(
            f"indexed search: {indexed_ms:.1f}ms analyzes={indexed_analyzes} "
            f"| bruteforce: {brute_ms:.1f}ms analyzes={brute_analyzes}"
        )
        assert indexed.matches
        assert brute.matches
        assert indexed_analyzes == 0
        assert brute_analyzes == n
        assert indexed_ms < brute_ms
        db.close()


@pytest.mark.stress
class TestReconcilePerf:
    def test_noop_reconcile_scales(self, perf_catalog, capsys) -> None:
        db, paths, _ = perf_catalog
        recorder = LatencyRecorder("reconcile_noop")
        with db:
            result = recorder.measure(
                lambda: db.reconcile_paths(set(paths), remove_missing=False)
            )
        print(recorder.summary())
        assert result.upserted == 0
        assert recorder.samples_ms[0] < 1_500

    def test_done_paths_beats_map_records_for_gap(
        self, perf_catalog, capsys
    ) -> None:
        db, paths, _ = perf_catalog

        t0 = time.perf_counter()
        with db:
            done = db.images.done_paths_for_module(MODULE_CLIP, paths)
            gap_fast = [p for p in paths if resolved_path_key(p) not in done]
        fast_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        with db:
            records = db.images.map_records_by_path(paths)
            gap_slow = []
            for path in paths:
                record = records.get(resolved_path_key(path))
                if record is None or MODULE_CLIP not in record.modules:
                    gap_slow.append(path)
                    continue
                if record.modules[MODULE_CLIP].status.value != "done":
                    gap_slow.append(path)
        slow_ms = (time.perf_counter() - t0) * 1000

        print(f"gap via done_paths: {fast_ms:.1f}ms | via map_records: {slow_ms:.1f}ms")
        assert set(gap_fast) == set(gap_slow)
        assert fast_ms < slow_ms
        assert fast_ms < 300
