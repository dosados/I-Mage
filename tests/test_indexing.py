from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from db.database import Database
from db.types import MODULE_CLIP, MODULE_FACES, MODULE_YOLO, ModuleStatus
from fakes import FakeClip, FakeFaces, FakeVectorStore, FakeYolo, unit_vec
from helpers import configure_scan, make_image_file, register_file, seed_catalog
from indexing.clip import index_clip_gap, index_clip_image
from indexing.faces import index_faces_gap, index_faces_image, store_faces
from indexing.gap import clip_gap_paths, faces_gap_paths, module_gap_paths, yolo_gap_paths
from indexing.runner import IndexModels, collect_scope_paths, module_stats_in_scope, run_scan
from indexing.yolo import index_yolo_gap, index_yolo_image
from ml.faces.base import Face
from ml.objects.base import Detection


class TestGapDetection:
    def test_gap_includes_missing_and_not_done(self, db: Database, image_dir: Path) -> None:
        paths = seed_catalog(db, image_dir, 4, mark_clip_done=2)
        with db:
            gap = clip_gap_paths(db, paths)
        assert len(gap) == 2
        assert {p.name for p in gap} == {"img_00002.jpg", "img_00003.jpg"}

    def test_gap_with_preloaded_records(self, db: Database, image_dir: Path) -> None:
        paths = seed_catalog(db, image_dir, 3, mark_yolo_done=1)
        with db:
            records = db.images.map_records_by_path(paths)
            gap = yolo_gap_paths(None, paths, records_by_path=records)
        assert len(gap) == 2

    def test_gap_requires_db_or_records(self, image_dir: Path) -> None:
        path = make_image_file(image_dir, "a.jpg")
        with pytest.raises(ValueError, match="db is required"):
            module_gap_paths(None, [path], MODULE_FACES)

    def test_failed_module_counts_as_gap(self, db: Database, image_dir: Path) -> None:
        path = make_image_file(image_dir, "a.jpg")
        with db:
            record = register_file(db, path)
            db.image_faces.mark_failed(record.id, "boom")
            gap = faces_gap_paths(db, [path])
        assert gap == [path]


class TestIndexClip:
    def test_index_clip_image_writes_vector_and_done(
        self, db: Database, image_dir: Path
    ) -> None:
        path = make_image_file(image_dir, "a.jpg")
        store = FakeVectorStore()
        model = FakeClip()
        with db:
            record = register_file(db, path)
            index_clip_image(db, store, path, model, model_version=model.model_name)
            status = db.image_clip.get(record.id)
        assert status is not None and status.status == ModuleStatus.DONE
        assert record.id in store.context
        assert len(model.encode_image_calls) == 1

    def test_index_clip_marks_failed_on_error(self, db: Database, image_dir: Path) -> None:
        path = make_image_file(image_dir, "a.jpg")
        store = FakeVectorStore()

        class Boom(FakeClip):
            def encode_image(self, image):
                raise RuntimeError("gpu down")

        with db:
            record = register_file(db, path)
            with pytest.raises(RuntimeError, match="gpu down"):
                index_clip_image(db, store, path, Boom(), model_version="x")
            status = db.image_clip.get(record.id)
        assert status is not None and status.status == ModuleStatus.FAILED

    def test_index_clip_gap_skips_failures(self, db: Database, image_dir: Path) -> None:
        good = make_image_file(image_dir, "good.jpg")
        bad = make_image_file(image_dir, "bad.jpg")
        store = FakeVectorStore()
        model = FakeClip()

        class Selective(FakeClip):
            def encode_image(self, image):
                if Path(image).name == "bad.jpg":
                    raise RuntimeError("nope")
                return super().encode_image(image)

        with db:
            register_file(db, good)
            register_file(db, bad)
            index_clip_gap(db, store, [good, bad], Selective(), model_version="x")
            assert db.image_clip.is_indexed(
                db.images.get_by_path(good).id  # type: ignore[union-attr]
            )
            assert not db.image_clip.is_indexed(
                db.images.get_by_path(bad).id  # type: ignore[union-attr]
            )


class TestIndexYolo:
    def test_index_yolo_stores_detections(self, db: Database, image_dir: Path) -> None:
        path = make_image_file(image_dir, "dog.jpg")
        yolo = FakeYolo()
        yolo.set_detections(
            path,
            [Detection(label="dog", confidence=0.88, bbox=(1, 2, 3, 4))],
        )
        with db:
            record = register_file(db, path)
            index_yolo_image(db, path, yolo, model_version=yolo.model_name)
            dets = db.detections.list_for_image(record.id)
            assert len(dets) == 1
            assert dets[0].label == "dog"
            assert db.image_yolo.is_indexed(record.id)

    def test_index_yolo_gap_continues_after_error(
        self, db: Database, image_dir: Path
    ) -> None:
        a = make_image_file(image_dir, "a.jpg")
        b = make_image_file(image_dir, "b.jpg")

        class Boom(FakeYolo):
            def detect(self, image):
                if Path(image).name == "a.jpg":
                    raise RuntimeError("fail")
                return super().detect(image)

        with db:
            register_file(db, a)
            register_file(db, b)
            index_yolo_gap(db, [a, b], Boom(), model_version="x")
            assert not db.image_yolo.is_indexed(db.images.get_by_path(a).id)  # type: ignore[union-attr]
            assert db.image_yolo.is_indexed(db.images.get_by_path(b).id)  # type: ignore[union-attr]


class TestIndexFaces:
    def test_index_faces_image_upserts_vectors(
        self, db: Database, image_dir: Path
    ) -> None:
        path = make_image_file(image_dir, "person.jpg")
        store = FakeVectorStore()
        faces = FakeFaces()
        with db:
            record = register_file(db, path)
            index_faces_image(db, store, path, faces, model_version=faces.model_name)
            assert db.image_faces.is_indexed(record.id)
            assert store.faces
            assert len(faces.analyze_calls) == 1

    def test_store_faces_requires_registered_image(
        self, db: Database, image_dir: Path
    ) -> None:
        path = make_image_file(image_dir, "ghost.jpg")
        with db:
            with pytest.raises(ValueError, match="not registered"):
                store_faces(
                    db,
                    FakeVectorStore(),
                    path,
                    [Face(bbox=(0, 0, 1, 1), detection_score=1.0, embedding=unit_vec(1))],
                    model_version="x",
                )

    def test_index_faces_gap(self, db: Database, image_dir: Path) -> None:
        paths = [make_image_file(image_dir, f"{i}.jpg") for i in range(3)]
        store = FakeVectorStore()
        with db:
            for path in paths:
                register_file(db, path)
            index_faces_gap(db, store, paths, FakeFaces(), model_version="fake")
            assert all(
                db.image_faces.is_indexed(db.images.get_by_path(p).id)  # type: ignore[union-attr]
                for p in paths
            )


class TestRunScan:
    def test_full_scan_reconcile_and_index_clip(
        self, db: Database, image_dir: Path
    ) -> None:
        paths = [make_image_file(image_dir, f"{i}.jpg", f"p{i}".encode()) for i in range(5)]
        configure_scan(db, [image_dir])
        store = FakeVectorStore()
        models = IndexModels(
            clip=FakeClip(),
            yolo=FakeYolo(),
            faces=FakeFaces(),
        )
        with db:
            run = db.index_runs.create(module="clip", mode="full", progress_total=0)
            run_id = run.id
        results = run_scan(
            db,
            store,
            models,
            db.get_scan_config(),
            modules=[MODULE_CLIP],
            mode="full",
            remove_missing=False,
            run_id=run_id,
        )
        assert len(results) == 1
        assert results[0].indexed == 5
        assert results[0].failed == 0
        with db:
            assert db.images.count_all() == 5
            assert db.images.count_module_done(MODULE_CLIP) == 5
        assert store.upsert_context_calls == 5

    def test_second_scan_has_empty_gap(self, db: Database, image_dir: Path) -> None:
        for i in range(4):
            make_image_file(image_dir, f"{i}.jpg")
        configure_scan(db, [image_dir])
        store = FakeVectorStore()
        models = IndexModels(
            clip=FakeClip(),
            yolo=FakeYolo(),
            faces=FakeFaces(),
        )
        config = db.get_scan_config()
        run_scan(db, store, models, config, modules=[MODULE_CLIP], mode="full")
        clip = FakeClip()
        models2 = IndexModels(
            clip=clip,
            yolo=FakeYolo(),
            faces=FakeFaces(),
        )
        results = run_scan(db, store, models2, config, modules=[MODULE_CLIP], mode="full")
        assert results[0].total == 0
        assert results[0].indexed == 0
        assert clip.encode_image_calls == []

    def test_collect_scope_and_module_stats(self, db: Database, image_dir: Path) -> None:
        seed_catalog(db, image_dir, 6, mark_faces_done=2)
        configure_scan(db, [image_dir])
        paths = collect_scope_paths(db.get_scan_config())
        assert len(paths) == 6
        done, total = module_stats_in_scope(db, paths, MODULE_FACES)
        assert (done, total) == (2, 6)
