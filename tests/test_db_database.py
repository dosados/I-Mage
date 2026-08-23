from __future__ import annotations

from pathlib import Path

import pytest

from db.database import Database
from helpers import make_image_file, register_file


class TestDatabaseFacade:
    def test_context_commits_on_success(self, db_path: Path, image_dir: Path) -> None:
        path = make_image_file(image_dir, "a.jpg")
        with Database(path=db_path) as db:
            register_file(db, path)

        with Database(path=db_path) as db:
            assert db.images.count_all() == 1

    def test_context_rolls_back_on_error(self, db_path: Path, image_dir: Path) -> None:
        path = make_image_file(image_dir, "a.jpg")
        db = Database(path=db_path)
        try:
            with db:
                register_file(db, path)
                raise RuntimeError("boom")
        except RuntimeError:
            pass

        with Database(path=db_path) as db2:
            # Rolled-back insert must not be visible.
            assert db2.images.count_all() == 0

    def test_register_image_file_helper(self, db: Database, image_dir: Path) -> None:
        path = make_image_file(image_dir, "a.jpg", b"abc")
        with db:
            record = db.register_image_file(path)
            assert record.size == 3
            assert db.images.get_by_path(path) is not None

    def test_thread_local_sessions(self, db_path: Path, image_dir: Path) -> None:
        import threading

        errors: list[BaseException] = []
        paths = [make_image_file(image_dir, f"t{i}.jpg") for i in range(4)]
        shared = Database(path=db_path)

        def worker(path: Path) -> None:
            try:
                with shared:
                    register_file(shared, path)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(path,)) for path in paths
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        shared.close()

        assert errors == []
        with Database(path=db_path) as db:
            assert db.images.count_all() == 4
