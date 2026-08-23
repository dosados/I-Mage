from __future__ import annotations

from pathlib import Path

from db.database import Database
from db.types import ModuleStatus
from helpers import make_image_file, register_file


def _new_image(db: Database, image_dir: Path, name: str = "a.jpg"):
    path = make_image_file(image_dir, name)
    return register_file(db, path)


class TestImageYoloService:
    def test_mark_running_done_failed_cycle(self, db: Database, image_dir: Path) -> None:
        with db:
            record = _new_image(db, image_dir)
            running = db.image_yolo.mark_running(record.id, model_version="yolov8")
            assert running.status == ModuleStatus.RUNNING
            assert db.image_yolo.is_running(record.id)
            assert db.image_yolo.needs_reindex(record.id) is False

            done = db.image_yolo.mark_done(record.id, model_version="yolov8")
            assert done.status == ModuleStatus.DONE
            assert done.indexed_at is not None
            assert db.image_yolo.is_indexed(record.id)

            failed = db.image_yolo.mark_failed(record.id, "boom", model_version="yolov8")
            assert failed.status == ModuleStatus.FAILED
            assert failed.last_error == "boom"
            assert db.image_yolo.needs_reindex(record.id)

    def test_missing_image_needs_reindex(self, db: Database) -> None:
        with db:
            assert db.image_yolo.needs_reindex("nope") is True
            assert db.image_yolo.is_indexed("nope") is False
            assert db.image_yolo.get("nope") is None


class TestImageClipService:
    def test_status_transitions(self, db: Database, image_dir: Path) -> None:
        with db:
            record = _new_image(db, image_dir)
            db.image_clip.mark_running(record.id, model_version="clip")
            assert db.image_clip.is_indexed(record.id) is False
            done = db.image_clip.mark_done(record.id, model_version="clip")
            assert done.status == ModuleStatus.DONE
            assert db.image_clip.is_indexed(record.id)
            failed = db.image_clip.mark_failed(record.id, "gpu oom")
            assert failed.status == ModuleStatus.FAILED
            assert failed.last_error == "gpu oom"


class TestImageFacesService:
    def test_status_helpers(self, db: Database, image_dir: Path) -> None:
        with db:
            record = _new_image(db, image_dir)
            assert db.image_faces.needs_reindex(record.id) is True
            db.image_faces.mark_running(record.id, model_version="arcface")
            assert db.image_faces.is_running(record.id)
            db.image_faces.mark_done(record.id, model_version="arcface")
            assert db.image_faces.is_indexed(record.id)
            assert db.image_faces.needs_reindex(record.id) is False
            db.image_faces.mark_failed(record.id, "no face")
            assert db.image_faces.needs_reindex(record.id)
