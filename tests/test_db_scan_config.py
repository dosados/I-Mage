from __future__ import annotations

from pathlib import Path

from db.database import Database
from db.scan_config import ScanConfig, default_scan_config
from db.types import MODULE_CLIP, MODULE_FACES, MODULE_YOLO


class TestScanConfig:
    def test_defaults_when_unset(self, db: Database, tmp_path: Path) -> None:
        search = tmp_path / "search"
        faces = tmp_path / "faces"
        search.mkdir()
        faces.mkdir()
        defaults = default_scan_config(search_dir=search, face_search_dir=faces)
        db.set_default_scan_config(defaults)
        with db:
            config = db.get_scan_config()
        assert str(search.resolve()) in config.include_directories
        assert str(faces.resolve()) in config.include_directories
        assert config.background_indexer_enabled is False
        assert set(config.background_modules) == {MODULE_YOLO, MODULE_CLIP, MODULE_FACES}

    def test_save_and_reload(self, db: Database, tmp_path: Path) -> None:
        config = ScanConfig(
            include_directories=[str(tmp_path / "a")],
            ignore_globs=["*.tmp", "**/cache/**"],
            background_indexer_enabled=True,
            schedule_interval_days=3,
            background_modules=[MODULE_CLIP],
        )
        with db:
            db.scan_config.save(config)
            loaded = db.scan_config.get(defaults=ScanConfig())
        assert loaded.include_directories == config.include_directories
        assert loaded.ignore_globs == config.ignore_globs
        assert loaded.background_indexer_enabled is True
        assert loaded.schedule_interval_days == 3
        assert loaded.background_modules == [MODULE_CLIP]

    def test_update_last_background_run(self, db: Database) -> None:
        with db:
            db.scan_config.save(ScanConfig(include_directories=["/tmp"]))
            updated = db.scan_config.update_last_background_run("2026-01-01T00:00:00+00:00")
        assert updated.last_background_run_at == "2026-01-01T00:00:00+00:00"

    def test_include_paths_resolves(self, tmp_path: Path) -> None:
        directory = tmp_path / "photos"
        directory.mkdir()
        config = ScanConfig(include_directories=[str(directory)])
        assert config.include_paths() == [directory.resolve()]

    def test_corrupt_json_falls_back_to_defaults(self, db: Database) -> None:
        from db.models import AppSetting
        from db.scan_config import SCAN_CONFIG_KEY

        defaults = ScanConfig(include_directories=["/defaults"])
        with db:
            db.session.add(AppSetting(key=SCAN_CONFIG_KEY, value="{not-json"))
            db.session.flush()
            loaded = db.scan_config.get(defaults=defaults)
        assert loaded.include_directories == ["/defaults"]
