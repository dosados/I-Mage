from __future__ import annotations

from pathlib import Path

import pytest

from db.database import Database


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def db(db_path: Path) -> Database:
    database = Database(path=db_path)
    yield database
    database.close()


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "photos"
    directory.mkdir()
    return directory
