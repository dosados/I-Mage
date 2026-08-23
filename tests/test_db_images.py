from __future__ import annotations

from pathlib import Path

import pytest

from db.database import Database
from db.hash import compute_content_hash, read_file_stat, resolved_path_key
from db.types import MODULE_CLIP, MODULE_FACES, MODULE_YOLO, ModuleStatus
from helpers import make_image_file, register_file, seed_catalog


class TestResolvedPathKey:
    def test_absolute_kept_as_is(self, tmp_path: Path) -> None:
        path = (tmp_path / "a.jpg").resolve()
        path.write_bytes(b"x")
        assert resolved_path_key(path) == str(path)

    def test_relative_is_resolved(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        path = Path("rel.jpg")
        path.write_bytes(b"x")
        assert resolved_path_key(path) == str(path.resolve())


class TestImageUpsert:
    def test_insert_new_image(self, db: Database, image_dir: Path) -> None:
        path = make_image_file(image_dir, "a.jpg", b"hello")
        with db:
            record = register_file(db, path)
        assert record.id
        assert record.path == path
        assert record.content_hash == compute_content_hash(path)
        assert record.size == len(b"hello")
        assert record.modules == {}

    def test_upsert_same_content_updates_mtime_only(self, db: Database, image_dir: Path) -> None:
        path = make_image_file(image_dir, "a.jpg", b"same")
        with db:
            first = register_file(db, path)
            image_id = first.id

        # Touch mtime without changing bytes.
        path.write_bytes(b"same")
        with db:
            second = register_file(db, path)
            assert second.id == image_id
            assert second.content_hash == first.content_hash
            assert db.image_clip.get(image_id) is None

    def test_content_change_invalidates_module_indexes(
        self, db: Database, image_dir: Path
    ) -> None:
        path = make_image_file(image_dir, "a.jpg", b"v1")
        with db:
            record = register_file(db, path)
            db.image_yolo.mark_done(record.id, model_version="y")
            db.image_clip.mark_done(record.id, model_version="c")
            db.image_faces.mark_done(record.id, model_version="f")

        path.write_bytes(b"v2-changed")
        with db:
            updated = register_file(db, path)
            assert updated.id == record.id
            assert updated.content_hash != record.content_hash
            assert db.image_yolo.get(record.id) is None
            assert db.image_clip.get(record.id) is None
            assert db.image_faces.get(record.id) is None

    def test_get_by_path_and_id(self, db: Database, image_dir: Path) -> None:
        path = make_image_file(image_dir, "a.jpg")
        with db:
            record = register_file(db, path)
            by_id = db.images.get_by_id(record.id)
            by_path = db.images.get_by_path(path)
        assert by_id is not None and by_id.id == record.id
        assert by_path is not None and by_path.id == record.id

    def test_get_missing_returns_none(self, db: Database) -> None:
        with db:
            assert db.images.get_by_id("missing") is None
            assert db.images.get_by_path("/no/such/file.jpg") is None

    def test_list_all_ordered_and_limited(self, db: Database, image_dir: Path) -> None:
        paths = seed_catalog(db, image_dir, 5)
        with db:
            all_rows = db.images.list_all()
            limited = db.images.list_all(limit=2)
        assert [r.path for r in all_rows] == sorted(paths)
        assert len(limited) == 2

    def test_delete_by_id(self, db: Database, image_dir: Path) -> None:
        path = make_image_file(image_dir, "a.jpg")
        with db:
            record = register_file(db, path)
            db.images.delete_by_id(record.id)
            assert db.images.get_by_id(record.id) is None


class TestReconcile:
    def test_reconcile_inserts_new_files(self, db: Database, image_dir: Path) -> None:
        paths = {make_image_file(image_dir, f"{i}.jpg", f"{i}".encode()) for i in range(3)}
        with db:
            result = db.reconcile_paths(paths, remove_missing=False)
            assert result.upserted == 3
            assert result.removed == 0
            assert db.images.count_all() == 3

    def test_reconcile_noop_when_unchanged(self, db: Database, image_dir: Path) -> None:
        paths = seed_catalog(db, image_dir, 4)
        with db:
            result = db.reconcile_paths(set(paths), remove_missing=False)
            assert result.upserted == 0
            assert db.images.count_all() == 4

    def test_reconcile_detects_mtime_size_change(self, db: Database, image_dir: Path) -> None:
        path = make_image_file(image_dir, "a.jpg", b"old")
        with db:
            register_file(db, path)

        path.write_bytes(b"brand-new-bytes")
        with db:
            result = db.reconcile_paths({path}, remove_missing=False)
            assert result.upserted == 1
            record = db.images.get_by_path(path)
            assert record is not None
            assert record.content_hash == compute_content_hash(path)

    def test_reconcile_remove_missing(self, db: Database, image_dir: Path) -> None:
        keep = make_image_file(image_dir, "keep.jpg")
        drop = make_image_file(image_dir, "drop.jpg")
        with db:
            register_file(db, keep)
            register_file(db, drop)
            result = db.reconcile_paths({keep}, remove_missing=True)
            assert result.removed == 1
            assert db.images.count_all() == 1
            assert db.images.get_by_path(drop) is None

    def test_reconcile_progress_callback(self, db: Database, image_dir: Path) -> None:
        paths = {make_image_file(image_dir, f"{i}.jpg") for i in range(5)}
        events: list[tuple[int, int]] = []
        with db:
            db.reconcile_paths(
                paths,
                on_progress=lambda done, total: events.append((done, total)),
                commit_batch_size=2,
            )
        assert events[0] == (0, 5)
        assert events[-1] == (5, 5)
        assert any(done == 2 for done, _ in events)


class TestModuleStats:
    def test_catalog_module_stats(self, db: Database, image_dir: Path) -> None:
        seed_catalog(
            db,
            image_dir,
            10,
            mark_yolo_done=3,
            mark_clip_done=7,
            mark_faces_done=1,
        )
        with db:
            stats = db.images.catalog_module_stats()
        assert stats[MODULE_YOLO] == {"done": 3, "total": 10}
        assert stats[MODULE_CLIP] == {"done": 7, "total": 10}
        assert stats[MODULE_FACES] == {"done": 1, "total": 10}

    def test_count_module_done_empty(self, db: Database) -> None:
        with db:
            assert db.images.count_module_done(MODULE_CLIP) == 0
            assert db.images.count_all() == 0

    def test_done_paths_for_module(self, db: Database, image_dir: Path) -> None:
        paths = seed_catalog(db, image_dir, 6, mark_clip_done=2)
        with db:
            done = db.images.done_paths_for_module(MODULE_CLIP, paths)
        assert len(done) == 2
        assert all(isinstance(item, str) for item in done)

    def test_count_module_done_in_paths_subset(self, db: Database, image_dir: Path) -> None:
        paths = seed_catalog(db, image_dir, 8, mark_clip_done=5)
        subset = paths[:3]
        with db:
            # First 3 were marked done (indices 0..4), so subset of 3 → 3 done.
            assert db.images.count_module_done_in_paths(subset, MODULE_CLIP) == 3
            assert db.images.count_module_done_in_paths(paths[5:], MODULE_CLIP) == 0

    def test_stat_map_by_path(self, db: Database, image_dir: Path) -> None:
        paths = seed_catalog(db, image_dir, 3)
        with db:
            smap = db.images.stat_map_by_path(paths)
        assert set(smap) == {str(p) for p in paths}
        for path in paths:
            mtime, size, content_hash = smap[str(path)]
            fs_mtime, fs_size = read_file_stat(path)
            assert mtime == fs_mtime
            assert size == fs_size
            assert content_hash == compute_content_hash(path)

    def test_map_records_includes_module_status(self, db: Database, image_dir: Path) -> None:
        paths = seed_catalog(db, image_dir, 2, mark_yolo_done=1)
        with db:
            records = db.images.map_records_by_path(paths)
        keyed = records[str(paths[0])]
        assert MODULE_YOLO in keyed.modules
        assert keyed.modules[MODULE_YOLO].status == ModuleStatus.DONE
        assert MODULE_YOLO not in records[str(paths[1])].modules
