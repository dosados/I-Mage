from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from db.database import Database
from db.types import MODULE_FACES
from fakes import FakeClip, FakeFaces, FakeVectorStore, FakeYolo
from helpers import configure_scan, make_image_file
from indexing.runner import IndexModels, ScanStopped, reconcile_catalog, run_scan


class ConcurrentFaces(FakeFaces):
    def __init__(self, delay: float = 0.04) -> None:
        super().__init__()
        self.delay = delay
        self._guard = threading.Lock()
        self.active = 0
        self.max_active = 0

    def analyze(self, image):
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            return super().analyze(image)
        finally:
            with self._guard:
                self.active -= 1

    def analyze_batch(self, images, *, should_stop=None):
        paths = list(images)
        self.analyze_batch_calls.append(paths)
        results = [None] * len(paths)
        threads = []
        errors: list[BaseException] = []

        def worker(index, image) -> None:
            try:
                if should_stop is not None and should_stop():
                    raise InterruptedError("face analysis stopped")
                results[index] = self.analyze(image)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        for index, image in enumerate(paths):
            thread = threading.Thread(target=worker, args=(index, image))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]
        return results


def test_face_analysis_overlaps_io_and_gpu_submission(
    db: Database, image_dir: Path
) -> None:
    for index in range(12):
        make_image_file(image_dir, f"{index}.jpg")
    config = configure_scan(db, [image_dir])
    reconcile_catalog(db, config)
    faces = ConcurrentFaces()
    started = time.perf_counter()
    results = run_scan(
        db,
        FakeVectorStore(),
        IndexModels(clip=FakeClip(), yolo=FakeYolo(), faces=faces),
        config,
        modules=[MODULE_FACES],
        mode="full",
    )
    elapsed = time.perf_counter() - started

    assert results[0].indexed == 12
    assert faces.max_active > 1, "face analysis unexpectedly ran serially"
    assert elapsed < 12 * faces.delay * 0.8


def test_parallel_faces_honors_stop_without_deadlock(
    db: Database, image_dir: Path
) -> None:
    for index in range(40):
        make_image_file(image_dir, f"{index}.jpg")
    config = configure_scan(db, [image_dir])
    reconcile_catalog(db, config)
    faces = ConcurrentFaces(delay=0.05)
    stop = threading.Event()
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            run_scan(
                db,
                FakeVectorStore(),
                IndexModels(clip=FakeClip(), yolo=FakeYolo(), faces=faces),
                config,
                modules=[MODULE_FACES],
                mode="full",
                should_stop=stop.is_set,
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    deadline = time.monotonic() + 3
    while faces.max_active < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    stop.set()
    thread.join(timeout=5)

    assert not thread.is_alive(), "parallel face workers did not stop"
    assert len(errors) == 1
    assert isinstance(errors[0], ScanStopped)
    assert faces.active == 0


def test_full_run_requires_exactly_one_module(
    db: Database, image_dir: Path
) -> None:
    config = configure_scan(db, [image_dir])
    with pytest.raises(ValueError, match="exactly one module"):
        run_scan(
            db,
            FakeVectorStore(),
            IndexModels(clip=FakeClip(), yolo=FakeYolo(), faces=FakeFaces()),
            config,
            modules=[],
            mode="full",
        )
